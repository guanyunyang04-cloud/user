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
RICHER_MODEL_TYPES = {"gru_richer_path_value", "gru_richer_path_value_symbol"}
PATH_VALUE_MODEL_TYPES = {
    "gru_path_value",
    "gru_path_value_symbol",
    "gru_path_value_residual",
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


class SequencePathPackDataset(Dataset):
    def __init__(self, manifest: Mapping[str, Any], *, split: str, max_samples: int = 0) -> None:
        self.manifest = dict(manifest)
        self.lookback_days = int(self.manifest.get("lookback_days", DEFAULT_LOOKBACK_DAYS) or DEFAULT_LOOKBACK_DAYS)
        self.forward_days = int(self.manifest.get("forward_days", DEFAULT_FORWARD_DAYS) or DEFAULT_FORWARD_DAYS)
        sample_index = pd.read_parquet(str(self.manifest["sample_index_path"]))
        self.sample_index = sample_index[sample_index["split"].astype(str).eq(str(split))].reset_index(drop=True)
        if int(max_samples) > 0 and len(self.sample_index) > int(max_samples):
            self.sample_index = self.sample_index.head(int(max_samples)).reset_index(drop=True)
        self.date_idx_values = self.sample_index["date_idx"].astype(np.int32).to_numpy(copy=True)
        self.symbol_idx_values = self.sample_index["symbol_idx"].astype(np.int32).to_numpy(copy=True)
        self.trade_date_values = self.sample_index["trade_date"].astype(str).to_numpy(copy=True)
        self.symbol_values = self.sample_index["symbol"].astype(str).to_numpy(copy=True)
        self.split = str(split)
        channels = dict(self.manifest.get("feature_channels", {}) or {})
        self.channel_order = ["daily_raw", "daily_state", "intraday_summary", "limit_structure"]
        self.feature_arrays = {name: _open_memmap(channels[name], dtype="float32") for name in self.channel_order}
        self.feature_columns = {name: list(channels[name].get("columns", []) or []) for name in self.channel_order}
        self.normalization = dict(self.manifest.get("normalization", {}) or {})
        labels = dict(self.manifest.get("label_arrays", {}) or {})
        self.future_path = _open_memmap(labels["future_ohlc_path"], dtype="float32")
        self.path_summary = _open_memmap(labels["path_summary"], dtype="float32")
        self.path_summary_columns = list(labels["path_summary"].get("columns", []) or path_summary_columns(self.forward_days))
        self.value_column = path_value_column(self.forward_days)
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
        start = date_idx - self.lookback_days + 1
        end = date_idx + 1
        parts = [
            self._normalize(name, np.asarray(self.feature_arrays[name][start:end, symbol_idx, :], dtype=np.float32))
            for name in self.channel_order
        ]
        x = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        y_path = np.asarray(self.future_path[date_idx, symbol_idx, :, :], dtype=np.float32).copy()
        y_summary = np.asarray(self.path_summary[date_idx, symbol_idx, :], dtype=np.float32).copy()
        y_richer_path = self._future_richer_path_batch(
            y_path.reshape(1, self.forward_days, 4),
            np.asarray([date_idx], dtype=np.int64),
            np.asarray([symbol_idx], dtype=np.int64),
        )[0]
        return {
            "x": torch.from_numpy(x),
            "y_path": torch.from_numpy(y_path),
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
        y_path = np.asarray(self.future_path[date_idx, symbol_idx, :, :], dtype=np.float32).copy()
        y_richer_path = self._future_richer_path_batch(y_path, date_idx, symbol_idx)
        y_summary = np.asarray(self.path_summary[date_idx, symbol_idx, :], dtype=np.float32).copy()
        return {
            "x": torch.from_numpy(x),
            "y_path": torch.from_numpy(y_path),
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
            "gru_richer_path_value",
            "gru_richer_path_value_symbol",
        }
        if normalized_model_type not in allowed_model_types:
            raise ValueError(
                "model_type must be gru_last, gru_attention, gru_path_value, gru_path_value_symbol, "
                "gru_path_value_residual, gru_richer_path_value, or gru_richer_path_value_symbol"
            )
        self.model_type = normalized_model_type
        self.uses_derived_path_value = normalized_model_type in PATH_VALUE_MODEL_TYPES
        self.uses_symbol_embedding = normalized_model_type in {"gru_path_value_symbol", "gru_richer_path_value_symbol"}
        self.uses_residual_score = normalized_model_type in RESIDUAL_MODEL_TYPES
        self.uses_richer_path = normalized_model_type in RICHER_MODEL_TYPES
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
        self.richer_path_dim = int(max(4, richer_path_dim)) if self.uses_richer_path else 4
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
        future_path = path_output[:, :, :4] if self.uses_richer_path else path_output
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
        target_summary = _derive_path_summary_torch(y_path, smooth_value=False).detach()
        pred_summary = _derive_path_summary_torch(outputs["future_path"], smooth_value=True)
        summary_loss = _finite_smooth_l1_columns(
            pred_summary,
            target_summary,
            _derived_summary_loss_indices(int(outputs["future_path"].shape[1])),
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"].to(device, non_blocking=device.type == "cuda")
    y_path = batch["y_path"].to(device, non_blocking=device.type == "cuda")
    y_richer_path = batch["y_richer_path"].to(device, non_blocking=device.type == "cuda")
    y_summary = batch["y_summary"].to(device, non_blocking=device.type == "cuda")
    date_idx = batch["date_idx"].to(device, non_blocking=device.type == "cuda")
    symbol_idx = batch["symbol_idx"].to(device, non_blocking=device.type == "cuda")
    return x, y_path, y_richer_path, y_summary, date_idx, symbol_idx


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


def derived_path_summary_columns(forward_days: int) -> list[str]:
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


def _derived_summary_loss_indices(forward_days: int) -> list[int]:
    columns = derived_path_summary_columns(forward_days)
    keep = {
        f"future_max_return_{int(forward_days)}d",
        f"future_min_return_{int(forward_days)}d",
        f"future_final_return_{int(forward_days)}d",
        f"drawdown_after_peak_{int(forward_days)}d",
        f"best_exit_close_return_{int(forward_days)}d",
        f"pre_exit_max_drawdown_{int(forward_days)}d",
    }
    return [idx for idx, col in enumerate(columns) if col in keep]


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


def _derive_path_summary_torch(path: torch.Tensor, *, smooth_value: bool) -> torch.Tensor:
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


def _derive_path_summary_numpy(path: np.ndarray) -> np.ndarray:
    values = np.asarray(path, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 4:
        raise ValueError("path must have shape [batch, forward_days, 4]")
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
    summary_columns = derived_path_summary_columns(dataset.forward_days) if uses_derived_path_value else list(dataset.path_summary_columns)
    value_column = path_value_v2_column(dataset.forward_days) if uses_derived_path_value else str(dataset.value_column)
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
        x, y_path, _y_richer_path, y_summary, _date_idx, symbol_idx = _batch_to_device(batch, device)
        trade_dates = list(batch["trade_date"])
        symbols = list(batch["symbol"])
        del batch
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            out = model(x, symbol_idx=symbol_idx)
        pred_path_np = out["future_path"].detach().float().cpu().numpy()
        true_path_np = y_path.detach().float().cpu().numpy()
        if "score" in out:
            pred_summary_np = out["path_summary"].detach().float().cpu().numpy()
            score_np = out["score"].detach().float().cpu().numpy()
            true_summary_np = y_summary.detach().float().cpu().numpy()
        else:
            pred_summary_np = _derive_path_summary_numpy(pred_path_np)
            true_summary_np = _derive_path_summary_numpy(true_path_np)
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
        chunk = pd.DataFrame(rows)
        if write_predictions:
            if bool(write_path_predictions):
                path_rows: dict[str, Any] = {}
                for day in range(dataset.forward_days):
                    for field_idx, field in enumerate(PATH_OHLC_FIELDS):
                        path_rows[f"true_{field}_ret_d{day + 1}"] = true_path_np[:, day, field_idx]
                        path_rows[f"pred_{field}_ret_d{day + 1}"] = pred_path_np[:, day, field_idx]
                chunk = pd.concat([chunk, pd.DataFrame(path_rows)], axis=1, copy=False)
            chunk.to_csv(pred_path, index=False, mode="w" if first_write else "a", header=first_write, encoding="utf-8-sig")
            first_write = False
        metric_frame = chunk[["trade_date", "symbol", "score", *[f"true_{c}" for c in summary_columns]]].copy()
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
        del x, y_path, _y_richer_path, y_summary, symbol_idx, out, pred_path_np, pred_summary_np, score_np, true_path_np, true_summary_np, chunk, metric_frame
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
    train_ds = SequencePathPackDataset(manifest, split="train", max_samples=int(config.max_samples_per_split))
    val_ds = SequencePathPackDataset(manifest, split="validation", max_samples=int(config.max_samples_per_split))
    test_ds = SequencePathPackDataset(manifest, split="test", max_samples=int(config.max_samples_per_split))
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
            x, y_path, y_richer_path, y_summary, date_idx, symbol_idx = _batch_to_device(batch, device)
            del batch
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                out = model(x, symbol_idx=symbol_idx)
                loss, parts = _compute_loss(
                    out,
                    y_path,
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
            del x, y_path, y_richer_path, y_summary, date_idx, symbol_idx, out, loss, parts
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
    active_value_column = path_value_v2_column(train_ds.forward_days) if bool(getattr(model, "uses_derived_path_value", False)) else train_ds.value_column
    active_summary_columns = (
        derived_path_summary_columns(train_ds.forward_days)
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
        "value_column": str(active_value_column),
        "path_summary_columns": list(active_summary_columns),
        "device": str(device),
        "amp_enabled": bool(amp_enabled),
        "epochs": int(config.epochs),
        "completed_epochs": int(len(history)),
        "best_epoch": int(best_epoch),
        "batch_size": int(config.batch_size),
        "max_samples_per_split": int(config.max_samples_per_split),
        "prediction_mode": str(config.prediction_mode),
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
            "validation_predictions_csv": str((output_dir / "predictions" / "validation_predictions.csv").resolve()),
            "test_predictions_csv": str((output_dir / "predictions" / "test_predictions.csv").resolve()),
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
    train.add_argument("--pack-manifest", type=Path, required=True)
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
    train.add_argument("--path-loss-weight", type=float, default=0.45)
    train.add_argument("--summary-loss-weight", type=float, default=0.20)
    train.add_argument("--richer-loss-weight", type=float, default=0.10)
    train.add_argument("--value-loss-weight", type=float, default=0.20)
    train.add_argument("--rank-loss-weight", type=float, default=0.15)
    train.add_argument("--residual-score-weight", type=float, default=0.25, help=argparse.SUPPRESS)
    train.add_argument("--residual-penalty-weight", type=float, default=0.01, help=argparse.SUPPRESS)
    train.add_argument("--rank-max-per-side", type=int, default=64)
    train.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    train.add_argument("--amp", dest="amp", action="store_true", default=True)
    train.add_argument("--no-amp", dest="amp", action="store_false")
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument("--top-k", default="5,10,20,50,100")
    train.add_argument("--max-samples-per-split", type=int, default=0)
    train.add_argument("--prediction-mode", default="full", choices=("full", "compact", "none"))
    train.add_argument("--early-stopping-patience", type=int, default=0)
    train.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    train.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
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
        pack_manifest=Path(args.pack_manifest),
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
