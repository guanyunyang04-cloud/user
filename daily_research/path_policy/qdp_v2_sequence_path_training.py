from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import random
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
    _write_json,
    path_summary_columns,
    path_value_column,
)


DEFAULT_OUTPUT_ROOT = Path("daily_research/output/path_policy/sequence_path_training")
DEFAULT_TOP_K = (5, 10, 20, 50, 100)
DEFAULT_SEED = 7
PATH_VALUE_V2_WAITING_PENALTY = 0.04
PATH_VALUE_V2_DRAWDOWN_PENALTY = 0.60
PATH_VALUE_V2_TRANSACTION_COST = 0.002
PATH_VALUE_V2_TEMPERATURE = 0.03
SUMMARY_LOSS_PROFILE_BASE = "base"
SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC = "multi_horizon_ohlc"
SUMMARY_LOSS_PROFILES = (SUMMARY_LOSS_PROFILE_BASE, SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC)
MULTI_HORIZON_OHLC_WINDOWS = (5, 10, 20, 40, 60)
INPUT_CHANNEL_PROFILE_ALL = "all"
INPUT_CHANNEL_PROFILE_DAILY_ONLY = "daily_only"
INPUT_CHANNEL_PROFILES = (INPUT_CHANNEL_PROFILE_ALL, INPUT_CHANNEL_PROFILE_DAILY_ONLY)
INPUT_CHANNEL_PROFILE_ORDERS = {
    INPUT_CHANNEL_PROFILE_ALL: ("daily_raw", "daily_state", "intraday_summary", "limit_structure"),
    INPUT_CHANNEL_PROFILE_DAILY_ONLY: ("daily_raw", "daily_state"),
}
UNIFIED_VALUE_WAIT_PENALTY = 0.015
UNIFIED_VALUE_HOLD_PENALTY = 0.025
UNIFIED_VALUE_DRAWDOWN_PENALTY = 0.60
UNIFIED_VALUE_LIQUIDITY_PENALTY = 0.02
UNIFIED_VALUE_TRANSACTION_COST = 0.002
UNIFIED_VALUE_TEMPERATURE = 0.03
RICHER_MODEL_TYPES = {"gru_richer_path_value", "gru_richer_path_value_symbol"}
OHLCVA_MODEL_TYPES = {"gru_ohlcva_path_value"}
PATH_VALUE_MODEL_TYPES = {
    "gru_path_value",
    "gru_path_value_symbol",
    "gru_path_value_residual",
    *OHLCVA_MODEL_TYPES,
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


class SequencePathPackDataset(Dataset):
    def __init__(
        self,
        manifest: Mapping[str, Any],
        *,
        split: str,
        max_samples: int = 0,
        input_channel_profile: str = INPUT_CHANNEL_PROFILE_ALL,
    ) -> None:
        self.manifest = dict(manifest)
        self.input_channel_profile = str(input_channel_profile or INPUT_CHANNEL_PROFILE_ALL).strip().lower()
        self.lookback_days = int(self.manifest.get("lookback_days", DEFAULT_LOOKBACK_DAYS) or DEFAULT_LOOKBACK_DAYS)
        self.forward_days = int(self.manifest.get("forward_days", DEFAULT_FORWARD_DAYS) or DEFAULT_FORWARD_DAYS)
        sample_index = pd.read_parquet(str(self.manifest["sample_index_path"]))
        self.sample_index = sample_index[sample_index["split"].astype(str).eq(str(split))].reset_index(drop=True)
        if int(max_samples) > 0 and len(self.sample_index) > int(max_samples):
            self.sample_index = self.sample_index.head(int(max_samples)).reset_index(drop=True)
        self.date_idx_values = self.sample_index["date_idx"].astype(np.int32).to_numpy(copy=True)
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
        labels = dict(self.manifest.get("label_arrays", {}) or {})
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
        self.input_dim = int(sum(len(self.feature_columns[name]) for name in self.channel_order))
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

    def __len__(self) -> int:
        return int(len(self.sample_index))

    def _normalize(self, name: str, values: np.ndarray) -> np.ndarray:
        stats = dict(self.normalization.get(name, {}) or {})
        mean = np.asarray(stats.get("mean", [0.0] * values.shape[-1]), dtype=np.float32)
        std = np.asarray(stats.get("std", [1.0] * values.shape[-1]), dtype=np.float32)
        out = (values.astype(np.float32, copy=False) - mean.reshape(1, -1)) / np.maximum(std.reshape(1, -1), 1.0e-6)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def _normalize_batch(self, name: str, values: np.ndarray) -> np.ndarray:
        stats = dict(self.normalization.get(name, {}) or {})
        mean = np.asarray(stats.get("mean", [0.0] * values.shape[-1]), dtype=np.float32)
        std = np.asarray(stats.get("std", [1.0] * values.shape[-1]), dtype=np.float32)
        out = (values.astype(np.float32, copy=False) - mean.reshape(1, 1, -1)) / np.maximum(std.reshape(1, 1, -1), 1.0e-6)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def _normalize_selected_future_batch(self, name: str, values: np.ndarray, indices: list[int]) -> np.ndarray:
        stats = dict(self.normalization.get(name, {}) or {})
        mean_all = np.asarray(stats.get("mean", [0.0] * len(self.feature_columns[name])), dtype=np.float32)
        std_all = np.asarray(stats.get("std", [1.0] * len(self.feature_columns[name])), dtype=np.float32)
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

    def get_batch(self, indices: list[int] | np.ndarray) -> dict[str, Any]:
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
        x = np.concatenate(channel_parts, axis=2).astype(np.float32, copy=False)
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
        y_richer_path = self._future_richer_path_batch(y_path, date_idx, symbol_idx)
        y_summary = (
            np.asarray(self.path_summary[date_idx, label_symbol_idx, :], dtype=np.float32).copy()
            if self.path_summary is not None
            else _derive_path_summary_numpy(y_ohlcva_path, price_anchor=self.price_anchor)
        )
        return {
            "x": torch.from_numpy(x),
            "y_path": torch.from_numpy(y_path),
            "y_ohlcva_path": torch.from_numpy(y_ohlcva_path),
            "y_richer_path": torch.from_numpy(y_richer_path),
            "y_summary": torch.from_numpy(y_summary),
            "date_idx": torch.from_numpy(date_idx.astype(np.int64, copy=False)),
            "symbol_idx": torch.from_numpy(symbol_idx.astype(np.int64, copy=False)),
            "trade_date": [str(item) for item in self.trade_date_values[idx]],
            "symbol": [str(item) for item in self.symbol_values[idx]],
        }


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
        self._length = sum((len(items) + self.batch_size - 1) // self.batch_size for items in self.groups.values())

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        dates = list(self.date_indices)
        if self.shuffle:
            rng.shuffle(dates)
        for date_idx in dates:
            items = list(self.groups[date_idx])
            if self.shuffle:
                rng.shuffle(items)
            for start in range(0, len(items), self.batch_size):
                yield items[start : start + self.batch_size]

    def __len__(self) -> int:
        return int(self._length)


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
            "gru_richer_path_value",
            "gru_richer_path_value_symbol",
        }
        if normalized_model_type not in allowed_model_types:
            raise ValueError(
                "model_type must be gru_last, gru_attention, gru_path_value, gru_path_value_symbol, "
                "gru_path_value_residual, gru_ohlcva_path_value, gru_richer_path_value, or gru_richer_path_value_symbol"
            )
        self.model_type = normalized_model_type
        self.uses_derived_path_value = normalized_model_type in PATH_VALUE_MODEL_TYPES
        self.uses_symbol_embedding = normalized_model_type in {"gru_path_value_symbol", "gru_richer_path_value_symbol"}
        self.uses_residual_score = normalized_model_type in RESIDUAL_MODEL_TYPES
        self.uses_richer_path = normalized_model_type in RICHER_MODEL_TYPES
        self.uses_ohlcva_path = normalized_model_type in OHLCVA_MODEL_TYPES
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
        self.richer_path_dim = int(max(4, richer_path_dim)) if self.uses_richer_path else self.path_dim
        self.path_head = nn.Linear(head_dim, int(forward_days) * self.richer_path_dim)
        if self.uses_derived_path_value:
            self.summary_head = None
            self.score_head = None
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
        path_output = self.path_head(pooled).view(-1, self.forward_days, self.richer_path_dim)
        future_path = path_output[:, :, : self.path_dim] if self.uses_richer_path else path_output
        if self.uses_derived_path_value:
            outputs = {"future_path": future_path}
            if self.uses_richer_path:
                outputs["future_richer_path"] = path_output
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
    if not bool(mask.any()):
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[mask], target[mask], reduction="mean")


def _finite_smooth_l1_columns(pred: torch.Tensor, target: torch.Tensor, columns: list[int]) -> torch.Tensor:
    if not columns:
        return pred.sum() * 0.0
    index = torch.as_tensor(columns, device=pred.device, dtype=torch.long)
    return _finite_smooth_l1(pred.index_select(1, index), target.index_select(1, index))


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


def _multi_horizon_ohlc_summary_features(path: torch.Tensor, *, price_anchor: str) -> torch.Tensor:
    path = _legacy_entry_relative_path_torch(path[:, :, :4], price_anchor=price_anchor)
    batch_size = int(path.shape[0])
    forward_days = int(path.shape[1])
    windows = _summary_loss_windows(forward_days)
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
    candidate = torch.where(valid_by_horizon, candidate, torch.full_like(candidate, float("-inf")))
    best_idx = torch.argmax(candidate, dim=1)
    best_exit_close = torch.gather(close_ret, 1, best_idx)
    best_pre_exit_drawdown = torch.gather(pre_exit_drawdown, 1, best_idx)

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
) -> torch.Tensor:
    target_features = _multi_horizon_ohlc_summary_features(target_path, price_anchor=price_anchor).detach()
    pred_features = _multi_horizon_ohlc_summary_features(pred_path, price_anchor=price_anchor)
    if int(pred_features.shape[1]) == 0:
        return pred_path.sum() * 0.0
    finite_target = torch.isfinite(target_features)
    raw_loss = F.smooth_l1_loss(pred_features, target_features, reduction="none")
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
        )
        target_summary = _derive_path_summary_torch(target_path, smooth_value=False, price_anchor=price_anchor).detach()
        pred_summary = _derive_path_summary_torch(pred_path, smooth_value=True, price_anchor=price_anchor)
        return summary_loss, target_summary, pred_summary

    target_summary = _derive_path_summary_torch(target_path, smooth_value=False, price_anchor=price_anchor).detach()
    pred_summary = _derive_path_summary_torch(pred_path, smooth_value=True, price_anchor=price_anchor)
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
    y_richer_path: torch.Tensor | None = None,
    value_index: int,
    path_weight: float = 0.35,
    summary_weight: float = 0.35,
    value_weight: float = 0.15,
    rank_weight: float = 0.15,
    richer_weight: float = 0.0,
    rank_max_per_side: int = 64,
    residual_weight: float = 0.25,
    residual_penalty_weight: float = 0.01,
    price_anchor: str = "next_open",
    summary_loss_profile: str = SUMMARY_LOSS_PROFILE_BASE,
) -> tuple[torch.Tensor, dict[str, float]]:
    path_loss = _finite_smooth_l1(outputs["future_path"], y_path)
    richer_loss = outputs["future_path"].sum() * 0.0
    if y_richer_path is not None and "future_richer_path" in outputs:
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
    rank_loss = _rank_loss_by_date(score, value_target, date_idx, max_per_side=int(rank_max_per_side))
    total = (
        float(path_weight) * path_loss
        + float(summary_weight) * summary_loss
        + float(value_weight) * value_loss
        + float(rank_weight) * rank_loss
        + float(richer_weight) * richer_loss
        + float(residual_penalty_weight) * residual_penalty
    )
    return total, {
        "loss": float(total.detach().cpu().item()),
        "path_loss": float(path_loss.detach().cpu().item()),
        "summary_loss": float(summary_loss.detach().cpu().item()),
        "richer_loss": float(richer_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "rank_loss": float(rank_loss.detach().cpu().item()),
        "residual_penalty": float(residual_penalty.detach().cpu().item()),
    }


def _batch_to_device(
    batch: Mapping[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"].to(device, non_blocking=device.type == "cuda")
    y_path = batch["y_path"].to(device, non_blocking=device.type == "cuda")
    y_ohlcva_path = batch["y_ohlcva_path"].to(device, non_blocking=device.type == "cuda")
    y_richer_path = batch["y_richer_path"].to(device, non_blocking=device.type == "cuda")
    y_summary = batch["y_summary"].to(device, non_blocking=device.type == "cuda")
    date_idx = batch["date_idx"].to(device, non_blocking=device.type == "cuda")
    symbol_idx = batch["symbol_idx"].to(device, non_blocking=device.type == "cuda")
    return x, y_path, y_ohlcva_path, y_richer_path, y_summary, date_idx, symbol_idx


def _iter_index_batches(dataset: SequencePathPackDataset, *, batch_size: int, shuffle: bool, seed: int) -> Iterator[list[int]]:
    sampler = DateGroupedBatchSampler(dataset.sample_index, batch_size=int(batch_size), shuffle=bool(shuffle), seed=int(seed))
    yield from sampler


def _daily_spearman(frame: pd.DataFrame, *, score_col: str, target_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        if len(group) < 5:
            continue
        corr = group[[score_col, target_col]].corr(method="spearman").iloc[0, 1]
        rows.append({"trade_date": str(trade_date), "rank_ic": float(corr) if pd.notna(corr) else np.nan, "count": int(len(group))})
    return pd.DataFrame(rows)


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
    ]
    for top_k in top_k_values:
        for trade_date, group in frame.groupby("trade_date", sort=True):
            group = group.dropna(subset=["score", value_column])
            if group.empty:
                continue
            top = group.sort_values("score", ascending=False, kind="mergesort").head(int(top_k))
            row: dict[str, Any] = {"trade_date": str(trade_date), "top_k": int(top_k)}
            for col in [*metric_cols, *[col for col in optional_metric_cols if col in group.columns]]:
                universe_mean = pd.to_numeric(group[col], errors="coerce").mean()
                selected_mean = pd.to_numeric(top[col], errors="coerce").mean()
                row[f"selected_{col}"] = float(selected_mean)
                row[f"universe_{col}"] = float(universe_mean)
                row[f"alpha_{col}"] = float(selected_mean - universe_mean)
            row["selected_hit_5pct_rate"] = float((pd.to_numeric(top[f"future_max_return_{suffix}"], errors="coerce") >= 0.05).mean())
            row["selected_hit_10pct_rate"] = float((pd.to_numeric(top[f"future_max_return_{suffix}"], errors="coerce") >= 0.10).mean())
            row["selected_hit_20pct_rate"] = float((pd.to_numeric(top[f"future_max_return_{suffix}"], errors="coerce") >= 0.20).mean())
            row["selected_loss_3pct_rate"] = float((pd.to_numeric(top[f"future_min_return_{suffix}"], errors="coerce") <= -0.03).mean())
            row["selected_loss_5pct_rate"] = float((pd.to_numeric(top[f"future_min_return_{suffix}"], errors="coerce") <= -0.05).mean())
            row["selected_loss_10pct_rate"] = float((pd.to_numeric(top[f"future_min_return_{suffix}"], errors="coerce") <= -0.10).mean())
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
            if "missed_opportunity" in top.columns:
                row["selected_missed_opportunity_rate"] = float(pd.to_numeric(top["missed_opportunity"], errors="coerce").mean())
            rows.append(row)
    return rows


def _aggregate_topk_daily_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    daily = pd.DataFrame(rows)
    if daily.empty:
        return pd.DataFrame()
    out_rows: list[dict[str, Any]] = []
    for top_k, group in daily.groupby("top_k", sort=True):
        out: dict[str, Any] = {"top_k": int(top_k), "day_count": int(len(group))}
        for col in [c for c in group.columns if c not in {"trade_date", "top_k"}]:
            out[col] = float(pd.to_numeric(group[col], errors="coerce").mean())
        out_rows.append(out)
    return pd.DataFrame(out_rows)


def _topk_metrics(frame: pd.DataFrame, *, top_k_values: tuple[int, ...], forward_days: int, value_column: str) -> pd.DataFrame:
    rows = _topk_daily_rows(frame, top_k_values=top_k_values, forward_days=forward_days, value_column=value_column)
    return _aggregate_topk_daily_rows(rows)


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
    return pd.DataFrame(rows)


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


def _summary_loss_windows(forward_days: int) -> tuple[int, ...]:
    horizon = int(forward_days)
    windows = [int(window) for window in MULTI_HORIZON_OHLC_WINDOWS if int(window) <= horizon]
    if horizon not in windows:
        windows.append(horizon)
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


def _candidate_path_values_torch(path: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def _derive_unified_path_summary_torch(path: torch.Tensor, *, smooth_value: bool) -> torch.Tensor:
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
        temperature = max(float(UNIFIED_VALUE_TEMPERATURE), 1.0e-6)
        best_value = temperature * torch.logsumexp(flat / temperature, dim=1)
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


def _derive_path_summary_torch(path: torch.Tensor, *, smooth_value: bool, price_anchor: str = "next_open") -> torch.Tensor:
    if int(path.shape[2]) >= 6:
        return _derive_unified_path_summary_torch(path, smooth_value=smooth_value)
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
    candidate, pre_exit_drawdown = _candidate_path_values_torch(path)
    best_value_hard, best_idx = torch.max(candidate, dim=1)
    if smooth_value:
        temperature = max(float(PATH_VALUE_V2_TEMPERATURE), 1.0e-6)
        best_value = temperature * torch.logsumexp(candidate / temperature, dim=1)
    else:
        best_value = best_value_hard
    best_exit_close = close_ret.gather(1, best_idx.view(-1, 1)).squeeze(1)
    best_pre_exit_drawdown = pre_exit_drawdown.gather(1, best_idx.view(-1, 1)).squeeze(1)
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
            best_idx.to(path.dtype) + 1.0,
            best_exit_close,
            best_pre_exit_drawdown,
            best_value,
        ],
        dim=1,
    )


def _derive_path_summary_numpy(path: np.ndarray, *, price_anchor: str = "next_open") -> np.ndarray:
    values = np.asarray(path, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] not in {4, 6}:
        raise ValueError("path must have shape [batch, forward_days, 4 or 6]")
    if values.shape[2] >= 6:
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
    return summary.astype(np.float32, copy=False)


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
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pred_dir = output_dir / "predictions"
    if write_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{split}_predictions.csv"
    if write_predictions and pred_path.exists():
        pred_path.unlink()
    uses_derived_path_value = bool(getattr(model, "uses_derived_path_value", False))
    output_path_dim = 6 if bool(getattr(model, "uses_ohlcva_path", False)) else 4
    output_path_fields = PATH_OHLCVA_FIELDS if output_path_dim >= 6 else PATH_OHLC_FIELDS
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
    row_count = 0
    date_values: set[str] = set()
    target_sum = 0.0
    target_count = 0
    prediction_sums: dict[str, float] = {"score": 0.0}
    prediction_counts: dict[str, int] = {"score": 0}
    first_write = True
    model.eval()

    def flush_pending_date() -> None:
        nonlocal pending_date, pending_frames, ic_rows_by_score, topk_daily_rows_by_score
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
            topk_daily_rows_by_score.setdefault(score_col, []).extend(
                _topk_daily_rows(
                    score_frame,
                    top_k_values=top_k,
                    forward_days=dataset.forward_days,
                    value_column=value_column,
                )
            )
        pending_frames = []
        pending_date = None

    for predict_batch_count, batch_indices in enumerate(
        _iter_index_batches(dataset, batch_size=int(batch_size), shuffle=False, seed=0),
        start=1,
    ):
        batch = dataset.get_batch(batch_indices)
        x, y_path, y_ohlcva_path, _y_richer_path, y_summary, _date_idx, symbol_idx = _batch_to_device(batch, device)
        target_path = y_ohlcva_path if output_path_dim >= 6 else y_path
        trade_dates = list(batch["trade_date"])
        symbols = list(batch["symbol"])
        del batch
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            out = model(x, symbol_idx=symbol_idx)
        pred_path_np = out["future_path"].detach().float().cpu().numpy()
        true_path_np = target_path.detach().float().cpu().numpy()
        if "score" in out:
            pred_summary_np = out["path_summary"].detach().float().cpu().numpy()
            score_np = out["score"].detach().float().cpu().numpy()
            true_summary_np = y_summary.detach().float().cpu().numpy()
        else:
            pred_summary_np = _derive_path_summary_numpy(pred_path_np, price_anchor=dataset.price_anchor)
            true_summary_np = _derive_path_summary_numpy(true_path_np, price_anchor=dataset.price_anchor)
            path_value_score_np = pred_summary_np[:, summary_columns.index(value_column)]
            if "residual_score" in out:
                residual_np = out["residual_score"].detach().float().cpu().numpy()
                score_np = path_value_score_np + float(getattr(model, "residual_weight", 0.25)) * residual_np
            else:
                residual_np = None
                score_np = path_value_score_np
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
        chunk = pd.DataFrame(rows)
        if write_predictions:
            if bool(write_path_predictions):
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
        del x, y_path, y_ohlcva_path, _y_richer_path, y_summary, target_path, symbol_idx, out, pred_path_np, pred_summary_np, score_np, true_path_np, true_summary_np, chunk, metric_frame
        del trade_dates, symbols
        if predict_batch_count % 100 == 0:
            gc.collect()
            _trim_working_set()
    flush_pending_date()
    ic = pd.DataFrame(ic_rows_by_score.get("score", []))
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
    return ic, topk, metrics


def _write_report(
    output_dir: Path,
    *,
    summary: Mapping[str, Any],
    split_metrics: pd.DataFrame,
    topk: pd.DataFrame,
    baseline_summary: Mapping[str, Any],
) -> str:
    forward_days = int(summary.get("forward_days", DEFAULT_FORWARD_DAYS) or DEFAULT_FORWARD_DAYS)
    suffix = f"{forward_days}d"
    value_col = str(summary.get("value_column", "") or path_value_column(forward_days))
    max_col = f"future_max_return_{suffix}"
    final_col = f"future_final_return_{suffix}"
    best_exit_col = f"best_exit_close_return_{suffix}"
    lines: list[str] = []
    lines.append("# QDP v2 Sequence Path Model")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        f"This model reads past 100-day multi-channel sequences from QDP v2 and predicts future {forward_days}-day OHLC paths. "
        "Path summaries and ranking values are derived from the predicted path when using a path-value model."
    )
    if str(summary.get("loss_weights", {}).get("summary_profile", "")) == SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC:
        lines.append("Summary loss uses multi-horizon OHLC-derived constraints while the model output remains the future OHLC path.")
    early = dict(summary.get("early_stopping", {}) or {})
    if early:
        lines.append(
            f"Early stopping monitors {early.get('metric', 'validation_rank_ic_mean')} with "
            f"patience={int(early.get('patience', 0))}, min_delta={float(early.get('min_delta', 0.0)):.4g}."
        )
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
    summary_loss_weight: float
    richer_loss_weight: float
    value_loss_weight: float
    rank_loss_weight: float
    residual_score_weight: float
    residual_penalty_weight: float
    summary_loss_profile: str
    input_channel_profile: str
    rank_max_per_side: int
    device: str
    amp: bool
    seed: int
    top_k: tuple[int, ...]
    max_samples_per_split: int
    prediction_mode: str
    early_stopping_patience: int
    early_stopping_min_delta: float


def train_sequence_path_model(config: TrainConfig) -> dict[str, Any]:
    _set_seed(config.seed)
    manifest = json.loads(Path(config.pack_manifest).read_text(encoding="utf-8"))
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
    val_ds = SequencePathPackDataset(
        manifest,
        split="validation",
        max_samples=int(config.max_samples_per_split),
        input_channel_profile=str(config.input_channel_profile),
    )
    test_ds = SequencePathPackDataset(
        manifest,
        split="test",
        max_samples=int(config.max_samples_per_split),
        input_channel_profile=str(config.input_channel_profile),
    )
    if str(config.model_type) in OHLCVA_MODEL_TYPES and not bool(train_ds.has_ohlcva_path):
        raise ValueError("model_type=gru_ohlcva_path_value requires a pack with label_arrays.future_ohlcva_path")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_val_ic = -1e9
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = output_dir / "best_model.pt"
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config.epochs) + 1):
        _write_json(progress_path, {"status": "training", "epoch": epoch, "updated_at": _now()})
        model.train()
        loss_totals: dict[str, float] = {
            "loss": 0.0,
            "path_loss": 0.0,
            "summary_loss": 0.0,
            "richer_loss": 0.0,
            "value_loss": 0.0,
            "rank_loss": 0.0,
            "residual_penalty": 0.0,
        }
        batch_count = 0
        sample_count = 0
        train_batches = DateGroupedBatchSampler(
            train_ds.sample_index,
            batch_size=int(config.batch_size),
            shuffle=True,
            seed=int(config.seed) + int(epoch) * 1009,
        )
        total_batches = len(train_batches)
        for batch_indices in train_batches:
            batch = train_ds.get_batch(batch_indices)
            x, y_path, y_ohlcva_path, y_richer_path, y_summary, date_idx, symbol_idx = _batch_to_device(batch, device)
            target_path = y_ohlcva_path if bool(getattr(model, "uses_ohlcva_path", False)) else y_path
            del batch
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                out = model(x, symbol_idx=symbol_idx)
                loss, parts = _compute_loss(
                    out,
                    target_path,
                    y_summary,
                    date_idx,
                    y_richer_path=y_richer_path,
                    value_index=train_ds.value_index,
                    path_weight=float(config.path_loss_weight),
                    summary_weight=float(config.summary_loss_weight),
                    value_weight=float(config.value_loss_weight),
                    rank_weight=float(config.rank_loss_weight),
                    richer_weight=float(config.richer_loss_weight),
                    rank_max_per_side=int(config.rank_max_per_side),
                    residual_weight=float(config.residual_score_weight),
                    residual_penalty_weight=float(config.residual_penalty_weight),
                    price_anchor=train_ds.price_anchor,
                    summary_loss_profile=str(config.summary_loss_profile),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            for key, value in parts.items():
                loss_totals[key] += float(value)
            batch_count += 1
            sample_count += int(x.shape[0])
            if batch_count == 1 or batch_count % 200 == 0:
                _write_json(
                    progress_path,
                    {
                        "status": "training",
                        "epoch": int(epoch),
                        "batch": int(batch_count),
                        "total_batches": int(total_batches),
                        "sample_count": int(sample_count),
                        "updated_at": _now(),
                    },
                )
            del x, y_path, y_ohlcva_path, y_richer_path, target_path, y_summary, date_idx, symbol_idx, out, loss, parts
            if batch_count % 200 == 0:
                gc.collect()
                _trim_working_set()
        train_row = {
            "epoch": int(epoch),
            "train_sample_count": int(sample_count),
            **{key: float(value / max(batch_count, 1)) for key, value in loss_totals.items()},
        }
        _write_json(progress_path, {"status": "validating_epoch", "epoch": int(epoch), "updated_at": _now()})
        val_ic, val_topk, val_metrics = _predict_split(
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
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
        train_row["early_stopping_wait"] = int(epochs_without_improvement)
        history.append(train_row)
        pd.DataFrame(history).to_csv(output_dir / "training_history_partial.csv", index=False, encoding="utf-8-sig")
        _write_json(
            progress_path,
            {
                "status": "epoch_completed",
                "epoch": int(epoch),
                "validation_rank_ic_mean": float(val_metrics["rank_ic_mean"]),
                "best_validation_rank_ic_mean": float(best_val_ic),
                "best_epoch": int(best_epoch),
                "early_stopping_wait": int(epochs_without_improvement),
                "early_stopping_patience": int(config.early_stopping_patience),
                "updated_at": _now(),
            },
        )
        del val_ic, val_topk
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        _trim_working_set()
        if int(config.early_stopping_patience) > 0 and epochs_without_improvement >= int(config.early_stopping_patience):
            _write_json(
                progress_path,
                {
                    "status": "early_stopped",
                    "epoch": int(epoch),
                    "best_epoch": int(best_epoch),
                    "best_validation_rank_ic_mean": float(best_val_ic),
                    "early_stopping_wait": int(epochs_without_improvement),
                    "updated_at": _now(),
                },
            )
            break
    if best_path.exists():
        payload = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
    _write_json(progress_path, {"status": "final_evaluation", "updated_at": _now()})
    val_ic, val_topk, val_metrics = _predict_split(
        model=model,
        dataset=val_ds,
        device=device,
        output_dir=output_dir,
        split="validation",
        batch_size=int(config.batch_size),
        amp_enabled=amp_enabled,
        top_k=config.top_k,
        write_predictions=str(config.prediction_mode) != "none",
        write_path_predictions=str(config.prediction_mode) == "full",
    )
    test_ic, test_topk, test_metrics = _predict_split(
        model=model,
        dataset=test_ds,
        device=device,
        output_dir=output_dir,
        split="test",
        batch_size=int(config.batch_size),
        amp_enabled=amp_enabled,
        top_k=config.top_k,
        write_predictions=str(config.prediction_mode) != "none",
        write_path_predictions=str(config.prediction_mode) == "full",
    )
    split_metrics = pd.DataFrame([val_metrics, test_metrics])
    topk = pd.concat([val_topk.assign(split="validation"), test_topk.assign(split="test")], ignore_index=True)
    daily_ic = pd.concat([val_ic.assign(split="validation"), test_ic.assign(split="test")], ignore_index=True)
    split_metrics_path = output_dir / "split_metrics.csv"
    topk_path = output_dir / "topk_metrics.csv"
    daily_ic_path = output_dir / "daily_rank_ic.csv"
    history_path = output_dir / "training_history.csv"
    split_metrics.to_csv(split_metrics_path, index=False, encoding="utf-8-sig")
    topk.to_csv(topk_path, index=False, encoding="utf-8-sig")
    daily_ic.to_csv(daily_ic_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")
    baseline_summary = _find_latest_baseline_summary(train_ds.forward_days)
    active_path_dim = 6 if bool(getattr(model, "uses_ohlcva_path", False)) else 4
    active_value_column = (
        value_column_for_path(train_ds.forward_days, path_dim=active_path_dim)
        if bool(getattr(model, "uses_derived_path_value", False))
        else train_ds.value_column
    )
    active_summary_columns = (
        derived_path_summary_columns(train_ds.forward_days, path_dim=active_path_dim)
        if bool(getattr(model, "uses_derived_path_value", False))
        else list(train_ds.path_summary_columns)
    )
    summary = {
        "artifact_type": "qdp_v2_sequence_path_training",
        "generated_at": _now(),
        "pack_manifest": str(Path(config.pack_manifest).resolve()),
        "output_dir": str(output_dir.resolve()),
        "lookback_days": int(train_ds.lookback_days),
        "forward_days": int(train_ds.forward_days),
        "price_anchor": str(train_ds.price_anchor),
        "value_column": str(active_value_column),
        "path_summary_columns": list(active_summary_columns),
        "device": str(device),
        "amp_enabled": bool(amp_enabled),
        "prediction_mode": str(config.prediction_mode),
        "epochs": int(config.epochs),
        "completed_epochs": int(len(history)),
        "best_epoch": int(best_epoch),
        "batch_size": int(config.batch_size),
        "max_samples_per_split": int(config.max_samples_per_split),
        "prediction_mode": str(config.prediction_mode),
        "input_channel_profile": str(train_ds.input_channel_profile),
        "input_channels": list(train_ds.channel_order),
        "model": {
            "type": f"SequencePathModel_{str(config.model_type)}",
            "input_dim": int(train_ds.input_dim),
            "hidden_dim": int(config.hidden_dim),
            "layers": int(config.layers),
            "dropout": float(config.dropout),
            "uses_derived_path_value": bool(getattr(model, "uses_derived_path_value", False)),
            "uses_symbol_embedding": bool(getattr(model, "uses_symbol_embedding", False)),
            "uses_residual_score": bool(getattr(model, "uses_residual_score", False)),
            "uses_richer_path": bool(getattr(model, "uses_richer_path", False)),
            "uses_ohlcva_path": bool(getattr(model, "uses_ohlcva_path", False)),
            "path_dim": int(getattr(model, "path_dim", 4)),
            "richer_path_dim": int(getattr(model, "richer_path_dim", 4)),
            "richer_path_fields": list(train_ds.richer_path_fields) if bool(getattr(model, "uses_richer_path", False)) else [],
            "symbol_embedding_dim": int(config.symbol_embedding_dim) if bool(getattr(model, "uses_symbol_embedding", False)) else 0,
            "residual_score_weight": float(config.residual_score_weight) if bool(getattr(model, "uses_residual_score", False)) else 0.0,
        },
        "loss_weights": {
            "path": float(config.path_loss_weight),
            "summary": float(config.summary_loss_weight),
            "richer": float(config.richer_loss_weight) if bool(getattr(model, "uses_richer_path", False)) else 0.0,
            "value": float(config.value_loss_weight),
            "rank": float(config.rank_loss_weight),
            "rank_max_per_side": int(config.rank_max_per_side),
            "summary_profile": str(config.summary_loss_profile),
        },
        "early_stopping": {
            "metric": "validation_rank_ic_mean",
            "patience": int(config.early_stopping_patience),
            "min_delta": float(config.early_stopping_min_delta),
            "stopped_early": bool(int(config.early_stopping_patience) > 0 and len(history) < int(config.epochs)),
        },
        "best_checkpoint": str(best_path.resolve()),
        "history": history,
        "split_metrics": split_metrics.to_dict("records"),
        "outputs": {
            "split_metrics_csv": str(split_metrics_path.resolve()),
            "topk_metrics_csv": str(topk_path.resolve()),
            "daily_rank_ic_csv": str(daily_ic_path.resolve()),
            "training_history_csv": str(history_path.resolve()),
            "validation_predictions_csv": str((output_dir / "predictions" / "validation_predictions.csv").resolve())
            if str(config.prediction_mode) != "none"
            else "",
            "test_predictions_csv": str((output_dir / "predictions" / "test_predictions.csv").resolve())
            if str(config.prediction_mode) != "none"
            else "",
        },
        "baseline_feature_summary": baseline_summary.get("output_dir", ""),
    }
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
            "gru_richer_path_value",
            "gru_richer_path_value_symbol",
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
    train.add_argument("--summary-loss-weight", type=float, default=0.20)
    train.add_argument("--richer-loss-weight", type=float, default=0.10)
    train.add_argument("--value-loss-weight", type=float, default=0.25)
    train.add_argument("--rank-loss-weight", type=float, default=0.15)
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
    train.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    train.add_argument("--amp", dest="amp", action="store_true", default=True)
    train.add_argument("--no-amp", dest="amp", action="store_false")
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument("--top-k", default="5,10,20,50,100")
    train.add_argument("--max-samples-per-split", type=int, default=0)
    train.add_argument("--prediction-mode", default="compact", choices=("full", "compact", "none"))
    train.add_argument(
        "--allow-large-predictions",
        action="store_true",
        help="Allow prediction-mode=full to write wide per-day path prediction CSVs.",
    )
    train.add_argument("--early-stopping-patience", type=int, default=0)
    train.add_argument("--early-stopping-min-delta", type=float, default=0.0)
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
        summary_loss_weight=float(args.summary_loss_weight),
        richer_loss_weight=float(args.richer_loss_weight),
        value_loss_weight=float(args.value_loss_weight),
        rank_loss_weight=float(args.rank_loss_weight),
        residual_score_weight=float(args.residual_score_weight),
        residual_penalty_weight=float(args.residual_penalty_weight),
        summary_loss_profile=str(args.summary_loss_profile),
        input_channel_profile=str(args.input_channel_profile),
        rank_max_per_side=int(args.rank_max_per_side),
        device=str(args.device),
        amp=bool(args.amp),
        seed=int(args.seed),
        top_k=_parse_int_list(args.top_k, default=DEFAULT_TOP_K),
        max_samples_per_split=int(args.max_samples_per_split),
        prediction_mode=str(args.prediction_mode),
        early_stopping_patience=int(args.early_stopping_patience),
        early_stopping_min_delta=float(args.early_stopping_min_delta),
    )
    result = train_sequence_path_model(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default) if bool(args.json) else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
