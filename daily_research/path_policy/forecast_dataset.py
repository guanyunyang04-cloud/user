from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from daily_research.continuous_policy.runtime import write_json
from daily_research.continuous_policy.state_builder import PreparedPolicyInputs
from daily_research.path_policy.forecast_features import (
    DEFAULT_FORECAST_FEATURE_PROFILE,
    DEFAULT_FORECAST_MAX_FEATURE_COLUMNS,
    build_forecast_feature_panels,
    build_forecast_feature_store,
)
from daily_research.path_policy.labels import (
    PATH20_CUMULATIVE_HORIZONS,
    PATH20_HORIZON,
    build_path20_labels,
    normalize_cumulative_horizons,
)


@dataclass(frozen=True)
class ForecastSequenceDataset:
    x: np.ndarray
    y_daily_excess: np.ndarray
    y_cum_excess: np.ndarray
    y_rank_by_horizon: np.ndarray
    y_rank_20d: np.ndarray
    y_drawdown_by_horizon: np.ndarray
    y_worst_by_horizon: np.ndarray
    y_upside_by_horizon: np.ndarray
    y_max_drawdown_20d: np.ndarray
    y_worst_1d_20d: np.ndarray
    y_upside_20d: np.ndarray
    date: np.ndarray
    stock: np.ndarray
    role: np.ndarray
    sequence_start_dates: np.ndarray
    label_end_dates: np.ndarray
    feature_columns: list[str]
    normalization_manifest: dict[str, Any]
    manifest: dict[str, Any]
    static_context_ids: np.ndarray | None = None

    @property
    def dates_by_role(self) -> dict[str, list[pd.Timestamp]]:
        result: dict[str, list[pd.Timestamp]] = {}
        for role_name in sorted({str(item) for item in self.role.tolist()}):
            mask = self.role == role_name
            result[role_name] = sorted({pd.Timestamp(item).normalize() for item in self.date[mask].tolist()})
        return result


class ForecastMemmapTorchDataset(Dataset):
    def __init__(self, dataset: "ForecastMemmapDataset", indices: np.ndarray, *, target_scale: float = 100.0) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.target_scale = float(target_scale)
        self._feature_store: np.memmap | None = None

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        row_idx = int(self.indices[int(item)])
        if self._feature_store is None:
            self._feature_store = self.dataset.open_feature_store()
        x = self.dataset.input_window(row_idx, store=self._feature_store)
        y_daily = self.dataset.y_daily_excess[row_idx] * self.target_scale
        y_cum = self.dataset.y_cum_excess[row_idx] * self.target_scale
        y_risk = self.dataset.risk_by_horizon(row_idx) * self.target_scale
        items: tuple[torch.Tensor, ...] = (
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y_daily, dtype=torch.float32),
            torch.as_tensor(y_cum, dtype=torch.float32),
            torch.as_tensor(y_risk, dtype=torch.float32),
            torch.as_tensor(row_idx, dtype=torch.long),
        )
        if self.dataset.static_context_ids is not None:
            static_ids = np.asarray(self.dataset.static_context_ids[row_idx], dtype=np.int64).copy()
            items = (*items, torch.as_tensor(static_ids, dtype=torch.long))
        return items

    def __getitems__(self, items: list[int]) -> list[tuple[torch.Tensor, ...]]:
        item_positions = np.asarray(items, dtype=np.int64)
        row_indices = self.indices[item_positions]
        input_windows = getattr(self.dataset, "input_windows", None)
        if input_windows is None:
            return [self.__getitem__(int(item)) for item in item_positions.tolist()]
        x = np.asarray(input_windows(row_indices), dtype=np.float32)
        y_daily = np.asarray(self.dataset.y_daily_excess[row_indices], dtype=np.float32).copy() * self.target_scale
        y_cum = np.asarray(self.dataset.y_cum_excess[row_indices], dtype=np.float32).copy() * self.target_scale
        y_risk = self.dataset.risk_by_horizon(row_indices).copy() * self.target_scale
        static_ids = (
            np.asarray(self.dataset.static_context_ids[row_indices], dtype=np.int64).copy()
            if self.dataset.static_context_ids is not None
            else None
        )
        result: list[tuple[torch.Tensor, ...]] = []
        for pos, row_idx in enumerate(row_indices.tolist()):
            row_items: tuple[torch.Tensor, ...] = (
                torch.as_tensor(x[pos], dtype=torch.float32),
                torch.as_tensor(y_daily[pos], dtype=torch.float32),
                torch.as_tensor(y_cum[pos], dtype=torch.float32),
                torch.as_tensor(y_risk[pos], dtype=torch.float32),
                torch.as_tensor(int(row_idx), dtype=torch.long),
            )
            if static_ids is not None:
                row_items = (*row_items, torch.as_tensor(static_ids[pos], dtype=torch.long))
            result.append(row_items)
        return result


class ForecastMemmapBatchTorchDataset(Dataset):
    def __init__(self, dataset: "ForecastMemmapDataset", indices: np.ndarray, *, batch_size: int, target_scale: float = 100.0) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = max(int(batch_size), 1)
        self.target_scale = float(target_scale)
        self._feature_store: np.memmap | None = None

    def __len__(self) -> int:
        return int((len(self.indices) + self.batch_size - 1) // self.batch_size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        start = int(item) * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        row_indices = self.indices[start:end]
        input_windows = getattr(self.dataset, "input_windows", None)
        if input_windows is not None:
            x = np.asarray(input_windows(row_indices), dtype=np.float32)
        else:
            if self._feature_store is None:
                self._feature_store = self.dataset.open_feature_store()
            x = np.stack(
                [self.dataset.input_window(int(row_idx), store=self._feature_store) for row_idx in row_indices],
                axis=0,
            ).astype(np.float32)
        y_daily = np.asarray(self.dataset.y_daily_excess[row_indices], dtype=np.float32).copy() * self.target_scale
        y_cum = np.asarray(self.dataset.y_cum_excess[row_indices], dtype=np.float32).copy() * self.target_scale
        y_risk = self.dataset.risk_by_horizon(row_indices).copy() * self.target_scale
        items: tuple[torch.Tensor, ...] = (
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y_daily, dtype=torch.float32),
            torch.as_tensor(y_cum, dtype=torch.float32),
            torch.as_tensor(y_risk, dtype=torch.float32),
            torch.as_tensor(row_indices, dtype=torch.long),
        )
        if self.dataset.static_context_ids is not None:
            static_ids = np.asarray(self.dataset.static_context_ids[row_indices], dtype=np.int64).copy()
            items = (*items, torch.as_tensor(static_ids, dtype=torch.long))
        return items


class ForecastDateBatchTorchDataset(Dataset):
    """Date-level view for cross-sectional research models."""

    def __init__(self, dataset: "ForecastMemmapDataset", indices: np.ndarray, *, target_scale: float = 100.0) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.target_scale = float(target_scale)
        if len(self.indices):
            rows = dataset.sample_index.iloc[self.indices]
            self._groups = [
                np.asarray(group.index.to_numpy(dtype=np.int64), dtype=np.int64)
                for _, group in rows.groupby("date", sort=True)
            ]
        else:
            self._groups = []
        self._feature_store: np.memmap | None = None

    def __len__(self) -> int:
        return int(len(self._groups))

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        row_indices = np.asarray(self._groups[int(item)], dtype=np.int64)
        date_input_windows = getattr(self.dataset, "date_input_windows", None)
        if callable(date_input_windows):
            x = np.asarray(date_input_windows(row_indices), dtype=np.float32)
        else:
            if self._feature_store is None:
                self._feature_store = self.dataset.open_feature_store()
            x = np.stack(
                [self.dataset.input_window(int(row_idx), store=self._feature_store) for row_idx in row_indices],
                axis=0,
            ).astype(np.float32)
        y_daily = np.asarray(self.dataset.y_daily_excess[row_indices], dtype=np.float32).copy() * self.target_scale
        y_cum = np.asarray(self.dataset.y_cum_excess[row_indices], dtype=np.float32).copy() * self.target_scale
        y_risk = self.dataset.risk_by_horizon(row_indices).copy() * self.target_scale
        mask = np.ones((len(row_indices),), dtype=bool)
        items: tuple[torch.Tensor, ...] = (
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(mask, dtype=torch.bool),
            torch.as_tensor(y_daily, dtype=torch.float32),
            torch.as_tensor(y_cum, dtype=torch.float32),
            torch.as_tensor(y_risk, dtype=torch.float32),
            torch.as_tensor(row_indices, dtype=torch.long),
        )
        if self.dataset.static_context_ids is not None:
            static_ids = np.asarray(self.dataset.static_context_ids[row_indices], dtype=np.int64).copy()
            items = (*items, torch.as_tensor(static_ids, dtype=torch.long))
        return items


@dataclass
class ForecastMemmapDataset:
    root: Path
    feature_store_path: Path
    feature_store_shape: tuple[int, int, int]
    sample_index: pd.DataFrame
    y_daily_excess: np.memmap
    y_cum_excess: np.memmap
    y_rank_by_horizon: np.memmap
    y_rank_20d: np.memmap
    y_drawdown_by_horizon: np.memmap
    y_worst_by_horizon: np.memmap
    y_upside_by_horizon: np.memmap
    y_max_drawdown_20d: np.memmap
    y_worst_1d_20d: np.memmap
    y_upside_20d: np.memmap
    feature_columns: list[str]
    normalization_manifest: dict[str, Any]
    manifest: dict[str, Any]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    date_values: np.ndarray
    stock_values: np.ndarray
    static_context_ids: np.memmap | np.ndarray | None = None

    @property
    def row_count(self) -> int:
        return int(len(self.sample_index))

    @property
    def input_dim(self) -> int:
        return int(len(self.feature_columns))

    @property
    def lookback_days(self) -> int:
        return int(self.manifest.get("lookback_days", self.sample_index.get("lookback_days", pd.Series([0])).iloc[0] if len(self.sample_index) else 0))

    @property
    def role(self) -> np.ndarray:
        return self.sample_index["role"].astype(str).to_numpy(dtype=object)

    @property
    def date(self) -> np.ndarray:
        return pd.to_datetime(self.sample_index["date"]).to_numpy(dtype=object)

    @property
    def stock(self) -> np.ndarray:
        return self.sample_index["stock"].astype(str).to_numpy(dtype=object)

    @property
    def sequence_start_dates(self) -> np.ndarray:
        return pd.to_datetime(self.sample_index["sequence_start_date"]).to_numpy(dtype=object)

    @property
    def label_end_dates(self) -> np.ndarray:
        return pd.to_datetime(self.sample_index["label_end_date"]).to_numpy(dtype=object)

    def open_feature_store(self) -> np.memmap:
        return np.memmap(self.feature_store_path, dtype="float32", mode="r", shape=self.feature_store_shape)

    def input_window(self, row_idx: int, *, store: np.memmap | None = None) -> np.ndarray:
        row = self.sample_index.iloc[int(row_idx)]
        start = int(row["sequence_start_pos"])
        end = int(row["date_pos"]) + 1
        stock_pos = int(row["stock_pos"])
        store = store if store is not None else self.open_feature_store()
        window = np.asarray(store[start:end, stock_pos, :], dtype=np.float32)
        normalized = ((window - self.feature_mean.reshape(1, -1)) / self.feature_std.reshape(1, -1)).astype(np.float32)
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    @property
    def cumulative_horizons(self) -> tuple[int, ...]:
        return normalize_cumulative_horizons(
            self.manifest.get("cumulative_horizons", PATH20_CUMULATIVE_HORIZONS),
            horizon=int(self.manifest.get("horizon", PATH20_HORIZON) or PATH20_HORIZON),
        )

    def risk_by_horizon(self, row_idx: int | np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.asarray(self.y_drawdown_by_horizon[row_idx], dtype=np.float32),
                np.asarray(self.y_worst_by_horizon[row_idx], dtype=np.float32),
                np.asarray(self.y_upside_by_horizon[row_idx], dtype=np.float32),
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

    def role_indices(self, role: str) -> np.ndarray:
        return np.flatnonzero(self.sample_index["role"].astype(str).to_numpy() == str(role))

    def cache_friendly_indices(self, indices: np.ndarray) -> np.ndarray:
        row_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if row_indices.size <= 1:
            return row_indices
        order_frame = self.sample_index.iloc[row_indices][["stock", "date"]].copy()
        order_frame["_row_idx"] = row_indices
        order_frame["stock"] = order_frame["stock"].astype(str)
        order_frame["date"] = pd.to_datetime(order_frame["date"])
        return order_frame.sort_values(["stock", "date", "_row_idx"], kind="mergesort")["_row_idx"].to_numpy(dtype=np.int64)

    def torch_dataset(self, indices: np.ndarray, *, target_scale: float = 100.0) -> ForecastMemmapTorchDataset:
        return ForecastMemmapTorchDataset(self, indices, target_scale=target_scale)

    def batch_torch_dataset(self, indices: np.ndarray, *, batch_size: int, target_scale: float = 100.0) -> ForecastMemmapBatchTorchDataset:
        return ForecastMemmapBatchTorchDataset(self, indices, batch_size=batch_size, target_scale=target_scale)

    def date_batch_torch_dataset(self, indices: np.ndarray, *, target_scale: float = 100.0) -> ForecastDateBatchTorchDataset:
        return ForecastDateBatchTorchDataset(self, indices, target_scale=target_scale)


def _row_selector_to_indices(selector: Any, *, row_count: int) -> tuple[np.ndarray, bool]:
    if isinstance(selector, slice):
        return np.arange(int(row_count), dtype=np.int64)[selector], False
    if isinstance(selector, (int, np.integer)):
        return np.asarray([int(selector)], dtype=np.int64), True
    arr = np.asarray(selector)
    if arr.dtype == bool:
        arr = np.flatnonzero(arr)
    return arr.astype(np.int64, copy=False).reshape(-1), False


class _QdpShardedLabelArray:
    def __init__(self, dataset: "ForecastShardedMemmapDataset", label_name: str, tail_shape: tuple[int, ...]) -> None:
        self.dataset = dataset
        self.label_name = str(label_name)
        self.tail_shape = tuple(int(item) for item in tail_shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return (int(self.dataset.row_count), *self.tail_shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __len__(self) -> int:
        return int(self.dataset.row_count)

    def __getitem__(self, key: Any) -> np.ndarray:
        row_selector = key
        tail_selector: tuple[Any, ...] = ()
        if isinstance(key, tuple):
            if not key:
                row_selector = slice(None)
            else:
                row_selector = key[0]
                tail_selector = tuple(key[1:])
        rows, scalar = _row_selector_to_indices(row_selector, row_count=self.dataset.row_count)
        values = self.dataset._label_values(self.label_name, rows)
        if scalar:
            values = values[0]
            if tail_selector:
                values = values[tail_selector]
            return np.asarray(values, dtype=np.float32)
        if tail_selector:
            values = values[(slice(None), *tail_selector)]
        return np.asarray(values, dtype=np.float32)


class ForecastShardedMemmapDataset:
    """Training view over QDP sharded memmap manifests.

    QDP shards store yearly feature panels. This wrapper presents the same lazy
    interface as ForecastMemmapDataset while assembling lookback windows across
    year boundaries by stock and global trading date.
    """

    def __init__(
        self,
        *,
        root: Path,
        manifest_path: Path,
        manifest: dict[str, Any],
        shards: list[dict[str, Any]],
        sample_index: pd.DataFrame,
        feature_columns: list[str],
        normalization_manifest: dict[str, Any],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        date_values: list[str],
        static_context_ids: np.ndarray | None = None,
        stock_feature_cache_size: int = 64,
    ) -> None:
        self.root = Path(root)
        self.manifest_path = Path(manifest_path)
        self.manifest = dict(manifest)
        self.shards = [dict(item) for item in shards]
        self.sample_index = sample_index.reset_index(drop=True).copy()
        self.feature_columns = list(feature_columns)
        self.normalization_manifest = dict(normalization_manifest)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)
        self.date_values = np.asarray([str(item) for item in date_values], dtype=object)
        self.stock_values = np.asarray(sorted({str(item) for item in self.sample_index.get("stock", pd.Series(dtype=object)).astype(str)}), dtype=object)
        self.static_context_ids = static_context_ids
        self._stock_feature_cache_size = max(int(stock_feature_cache_size), 1)
        self._feature_store_cache: OrderedDict[int, np.memmap] = OrderedDict()
        self._label_store_cache: OrderedDict[tuple[int, str], np.memmap] = OrderedDict()
        self._stock_feature_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._global_date_to_pos = {str(value): pos for pos, value in enumerate(self.date_values.tolist())}
        self._shard_date_values: list[list[str]] = []
        self._shard_global_positions: list[np.ndarray] = []
        self._shard_stock_positions: list[dict[str, int]] = []
        self._stock_year_to_shard: dict[tuple[int, str], int] = {}
        for shard_idx, shard in enumerate(self.shards):
            label_manifest = dict(shard.get("label_manifest", {}) or {})
            date_values_local = [str(item) for item in label_manifest.get("date_values", [])]
            stock_values_local = [str(item).strip().upper() for item in label_manifest.get("stock_values", [])]
            self._shard_date_values.append(date_values_local)
            self._shard_global_positions.append(
                np.asarray([self._global_date_to_pos[str(item)] for item in date_values_local if str(item) in self._global_date_to_pos], dtype=np.int64)
            )
            stock_positions = {stock: pos for pos, stock in enumerate(stock_values_local)}
            self._shard_stock_positions.append(stock_positions)
            year = int(shard.get("year", 0) or 0)
            if year:
                for stock in stock_positions:
                    self._stock_year_to_shard[(year, stock)] = shard_idx
        horizon = int(self.manifest.get("horizon", PATH20_HORIZON) or PATH20_HORIZON)
        cumulative_horizons = normalize_cumulative_horizons(
            self.manifest.get("cumulative_horizons", PATH20_CUMULATIVE_HORIZONS),
            horizon=horizon,
        )
        cumulative_count = len(cumulative_horizons)
        all_horizon = int(horizon)
        self.y_daily_excess = _QdpShardedLabelArray(self, "daily_excess_return", (horizon,))
        self.y_cum_excess = _QdpShardedLabelArray(self, "cumulative_excess_return", (cumulative_count,))
        self.y_rank_by_horizon = _QdpShardedLabelArray(self, "rank_by_horizon", (cumulative_count,))
        self.y_rank_20d = _QdpShardedLabelArray(self, "rank_20d", ())
        self.y_drawdown_by_horizon = _QdpShardedLabelArray(self, "drawdown_by_horizon", (cumulative_count,))
        self.y_worst_by_horizon = _QdpShardedLabelArray(self, "worst_by_horizon", (cumulative_count,))
        self.y_upside_by_horizon = _QdpShardedLabelArray(self, "upside_by_horizon", (cumulative_count,))
        self.y_max_drawdown_20d = _QdpShardedLabelArray(self, "max_drawdown_20d", ())
        self.y_worst_1d_20d = _QdpShardedLabelArray(self, "worst_1d_20d", ())
        self.y_upside_20d = _QdpShardedLabelArray(self, "upside_20d", ())
        self.y_daily_return = _QdpShardedLabelArray(self, "daily_return", (all_horizon,))
        self.y_benchmark_daily_return = _QdpShardedLabelArray(self, "benchmark_daily_return", (all_horizon,))
        self.y_cum_return = _QdpShardedLabelArray(self, "cumulative_return", (cumulative_count,))
        self.y_benchmark_cum_return = _QdpShardedLabelArray(self, "benchmark_cumulative_return", (cumulative_count,))
        self.y_cum_return_1to20 = _QdpShardedLabelArray(self, "cumulative_return_1to20", (all_horizon,))
        self.y_benchmark_cum_return_1to20 = _QdpShardedLabelArray(self, "benchmark_cumulative_return_1to20", (all_horizon,))
        self.y_cum_excess_1to20 = _QdpShardedLabelArray(self, "cumulative_excess_return_1to20", (all_horizon,))
        self.y_rank_1to20 = _QdpShardedLabelArray(self, "rank_1to20", (all_horizon,))
        self.y_industry_rank_by_horizon = _QdpShardedLabelArray(self, "industry_rank_by_horizon", (cumulative_count,))
        self.y_entry_tradeable = _QdpShardedLabelArray(self, "entry_tradeable", ())
        self.y_entry_limit_up_buy_blocked = _QdpShardedLabelArray(self, "entry_limit_up_buy_blocked", ())
        self.y_entry_suspended_or_no_open = _QdpShardedLabelArray(self, "entry_suspended_or_no_open", ())
        self.y_forward_tradeable_ratio_by_horizon = _QdpShardedLabelArray(self, "forward_tradeable_ratio_by_horizon", (cumulative_count,))

    @property
    def row_count(self) -> int:
        return int(len(self.sample_index))

    @property
    def input_dim(self) -> int:
        return int(len(self.feature_columns))

    @property
    def feature_count(self) -> int:
        return int(len(self.feature_columns))

    @property
    def lookback_days(self) -> int:
        return int(self.manifest.get("lookback_days", 0) or 0)

    @property
    def role(self) -> np.ndarray:
        return self.sample_index["role"].astype(str).to_numpy(dtype=object)

    @property
    def date(self) -> np.ndarray:
        return pd.to_datetime(self.sample_index["date"]).to_numpy(dtype=object)

    @property
    def stock(self) -> np.ndarray:
        return self.sample_index["stock"].astype(str).to_numpy(dtype=object)

    @property
    def sequence_start_dates(self) -> np.ndarray:
        if "sequence_start_date" in self.sample_index.columns:
            return pd.to_datetime(self.sample_index["sequence_start_date"]).to_numpy(dtype=object)
        return self.date

    @property
    def label_end_dates(self) -> np.ndarray:
        if "label_end_date" in self.sample_index.columns:
            return pd.to_datetime(self.sample_index["label_end_date"]).to_numpy(dtype=object)
        return self.date

    @property
    def cumulative_horizons(self) -> tuple[int, ...]:
        return normalize_cumulative_horizons(
            self.manifest.get("cumulative_horizons", PATH20_CUMULATIVE_HORIZONS),
            horizon=int(self.manifest.get("horizon", PATH20_HORIZON) or PATH20_HORIZON),
        )

    @property
    def feature_store_shape(self) -> tuple[int, int, int]:
        return (
            int(len(self.date_values)),
            int(len(self.stock_values)),
            int(len(self.feature_columns)),
        )

    @property
    def feature_store_path(self) -> Path:
        return self.manifest_path

    def open_feature_store(self) -> "ForecastShardedMemmapDataset":
        return self

    def _open_feature_store_for_shard(self, shard_idx: int) -> np.memmap:
        idx = int(shard_idx)
        cached = self._feature_store_cache.get(idx)
        if cached is not None:
            self._feature_store_cache.move_to_end(idx)
            return cached
        shard = self.shards[idx]
        shape = tuple(int(item) for item in list(shard.get("feature_store_shape", []) or []))
        if len(shape) != 3:
            raise ValueError(f"qdp sharded memmap shard has invalid feature_store_shape: {shard.get('shard_key', idx)}")
        path = Path(str(shard.get("feature_store_path", "") or ""))
        store = np.memmap(path, dtype="float32", mode="r", shape=shape)
        self._feature_store_cache[idx] = store
        if len(self._feature_store_cache) > 8:
            self._feature_store_cache.popitem(last=False)
        return store

    def _open_label_store_for_shard(self, shard_idx: int, label_name: str) -> np.memmap:
        key = (int(shard_idx), str(label_name))
        cached = self._label_store_cache.get(key)
        if cached is not None:
            self._label_store_cache.move_to_end(key)
            return cached
        shard = self.shards[int(shard_idx)]
        arrays = dict(dict(shard.get("label_manifest", {}) or {}).get("arrays", {}) or {})
        meta = dict(arrays.get(str(label_name), {}) or {})
        path = Path(str(meta.get("path", "") or ""))
        shape = tuple(int(item) for item in list(meta.get("shape", []) or []))
        if not path.exists() or not shape:
            raise ValueError(f"qdp sharded memmap missing label array {label_name}: {path}")
        store = np.memmap(path, dtype=str(meta.get("dtype", "float32") or "float32"), mode="r", shape=shape)
        self._label_store_cache[key] = store
        if len(self._label_store_cache) > 32:
            self._label_store_cache.popitem(last=False)
        return store

    def _stock_feature_matrix(self, stock: str) -> np.ndarray:
        symbol = str(stock).strip().upper()
        cached = self._stock_feature_cache.get(symbol)
        if cached is not None:
            self._stock_feature_cache.move_to_end(symbol)
            return cached
        matrix = np.full((len(self.date_values), self.input_dim), np.nan, dtype=np.float32)
        years = sorted({pd.Timestamp(item).year for item in self.date_values.tolist()})
        for year in years:
            shard_idx = self._stock_year_to_shard.get((int(year), symbol))
            if shard_idx is None:
                continue
            stock_pos = self._shard_stock_positions[int(shard_idx)].get(symbol)
            if stock_pos is None:
                continue
            global_positions = self._shard_global_positions[int(shard_idx)]
            if global_positions.size == 0:
                continue
            store = self._open_feature_store_for_shard(int(shard_idx))
            local_count = min(int(store.shape[0]), int(global_positions.size))
            matrix[global_positions[:local_count], :] = np.asarray(store[:local_count, int(stock_pos), :], dtype=np.float32)
        self._stock_feature_cache[symbol] = matrix
        if len(self._stock_feature_cache) > self._stock_feature_cache_size:
            self._stock_feature_cache.popitem(last=False)
        return matrix

    def raw_input_window(self, row_idx: int) -> np.ndarray:
        row = self.sample_index.iloc[int(row_idx)]
        stock = str(row["stock"]).strip().upper()
        date_key = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        end_pos = int(self._global_date_to_pos.get(date_key, -1)) + 1
        if end_pos <= 0:
            return np.full((self.lookback_days, self.input_dim), np.nan, dtype=np.float32)
        start_pos = end_pos - int(self.lookback_days)
        source_start = max(start_pos, 0)
        source_end = end_pos
        target_start = source_start - start_pos
        out = np.full((self.lookback_days, self.input_dim), np.nan, dtype=np.float32)
        matrix = self._stock_feature_matrix(stock)
        if source_end > source_start:
            out[target_start : target_start + (source_end - source_start), :] = matrix[source_start:source_end, :]
        return out

    def raw_input_windows(self, row_indices: np.ndarray) -> np.ndarray:
        rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        lookback = int(self.lookback_days)
        out = np.full((len(rows), lookback, self.input_dim), np.nan, dtype=np.float32)
        if len(rows) == 0:
            return out
        request = self.sample_index.iloc[rows][["stock", "date"]].copy()
        request["_request_pos"] = np.arange(len(rows), dtype=np.int64)
        request["stock"] = request["stock"].astype(str).str.strip().str.upper()
        request["date_key"] = pd.to_datetime(request["date"]).dt.strftime("%Y-%m-%d")
        for stock, group in request.groupby("stock", sort=False):
            matrix = self._stock_feature_matrix(str(stock))
            if matrix.shape[0] < lookback:
                for _, row in group.iterrows():
                    out[int(row["_request_pos"])] = self.raw_input_window(int(rows[int(row["_request_pos"])]))
                continue
            windows = np.moveaxis(
                np.lib.stride_tricks.sliding_window_view(matrix, window_shape=lookback, axis=0),
                -1,
                1,
            )
            end_positions = np.asarray(
                [int(self._global_date_to_pos.get(str(date_key), -1)) + 1 for date_key in group["date_key"].tolist()],
                dtype=np.int64,
            )
            starts = end_positions - lookback
            request_positions = group["_request_pos"].to_numpy(dtype=np.int64, copy=False)
            valid = (starts >= 0) & (starts < int(windows.shape[0]))
            if np.any(valid):
                out[request_positions[valid]] = windows[starts[valid]]
            if np.any(~valid):
                for request_pos in request_positions[~valid]:
                    out[int(request_pos)] = self.raw_input_window(int(rows[int(request_pos)]))
        return out

    def input_window(self, row_idx: int, *, store: Any | None = None) -> np.ndarray:
        del store
        window = self.raw_input_window(int(row_idx))
        normalized = ((window - self.feature_mean.reshape(1, -1)) / self.feature_std.reshape(1, -1)).astype(np.float32)
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    def input_windows(self, row_indices: np.ndarray) -> np.ndarray:
        windows = self.raw_input_windows(row_indices)
        normalized = ((windows - self.feature_mean.reshape(1, 1, -1)) / self.feature_std.reshape(1, 1, -1)).astype(np.float32)
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    def _label_values(self, label_name: str, row_indices: np.ndarray) -> np.ndarray:
        rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        tail_shape = tuple(dict(self._label_tail_shapes()).get(str(label_name), ()))
        out = np.empty((len(rows), *tail_shape), dtype=np.float32)
        if len(rows) == 0:
            return out
        request = self.sample_index.iloc[rows].reset_index(drop=True).copy()
        request["_request_pos"] = np.arange(len(rows), dtype=np.int64)
        for shard_idx, group in request.groupby("_shard_idx", sort=False):
            store = self._open_label_store_for_shard(int(shard_idx), str(label_name))
            date_pos = group["date_pos"].to_numpy(dtype=np.int64, copy=False)
            stock_pos = group["stock_pos"].to_numpy(dtype=np.int64, copy=False)
            values = np.asarray(store[date_pos, stock_pos], dtype=np.float32)
            out[group["_request_pos"].to_numpy(dtype=np.int64, copy=False)] = values.reshape((len(group), *tail_shape))
        return out

    def _label_tail_shapes(self) -> dict[str, tuple[int, ...]]:
        horizon = int(self.manifest.get("horizon", PATH20_HORIZON) or PATH20_HORIZON)
        cumulative_count = len(self.cumulative_horizons)
        return _label_array_specs(horizon, self.cumulative_horizons)

    def risk_by_horizon(self, row_idx: int | np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.asarray(self.y_drawdown_by_horizon[row_idx], dtype=np.float32),
                np.asarray(self.y_worst_by_horizon[row_idx], dtype=np.float32),
                np.asarray(self.y_upside_by_horizon[row_idx], dtype=np.float32),
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

    def role_indices(self, role: str) -> np.ndarray:
        return np.flatnonzero(self.sample_index["role"].astype(str).to_numpy() == str(role))

    def cache_friendly_indices(self, indices: np.ndarray) -> np.ndarray:
        row_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if row_indices.size <= 1:
            return row_indices
        order_frame = self.sample_index.iloc[row_indices][["stock", "date"]].copy()
        order_frame["_row_idx"] = row_indices
        order_frame["stock"] = order_frame["stock"].astype(str)
        order_frame["date"] = pd.to_datetime(order_frame["date"])
        return order_frame.sort_values(["stock", "date", "_row_idx"], kind="mergesort")["_row_idx"].to_numpy(dtype=np.int64)

    def torch_dataset(self, indices: np.ndarray, *, target_scale: float = 100.0) -> ForecastMemmapTorchDataset:
        return ForecastMemmapTorchDataset(self, indices, target_scale=target_scale)

    def batch_torch_dataset(self, indices: np.ndarray, *, batch_size: int, target_scale: float = 100.0) -> ForecastMemmapBatchTorchDataset:
        return ForecastMemmapBatchTorchDataset(self, indices, batch_size=batch_size, target_scale=target_scale)

    def date_batch_torch_dataset(self, indices: np.ndarray, *, target_scale: float = 100.0) -> ForecastDateBatchTorchDataset:
        return ForecastDateBatchTorchDataset(self, indices, target_scale=target_scale)


class ForecastTrainingPackDataset:
    """Training-optimized QDP pack with stock-major feature panels and sample-major labels."""

    def __init__(
        self,
        *,
        root: Path,
        manifest_path: Path,
        manifest: dict[str, Any],
        sample_index: pd.DataFrame,
        feature_columns: list[str],
        feature_panel_path: Path,
        feature_panel_shape: tuple[int, int, int],
        feature_dtype: str,
        date_major_feature_panel_path: Path | None = None,
        date_major_feature_panel_shape: tuple[int, int, int] | None = None,
        date_major_feature_dtype: str = "",
        label_arrays: dict[str, np.memmap | np.ndarray],
        normalization_manifest: dict[str, Any],
        static_context_ids: np.memmap | np.ndarray | None,
        date_values: list[str],
        stock_values: list[str],
    ) -> None:
        self.root = Path(root)
        self.manifest_path = Path(manifest_path)
        self.manifest = dict(manifest)
        self.sample_index = sample_index.reset_index(drop=True).copy()
        self.feature_columns = list(feature_columns)
        self.feature_panel_path = Path(feature_panel_path)
        self._feature_panel_shape = tuple(int(item) for item in feature_panel_shape)
        self._feature_dtype = str(feature_dtype or "float16")
        self._feature_panel: np.memmap | None = None
        self.date_major_feature_panel_path = Path(date_major_feature_panel_path) if date_major_feature_panel_path else None
        self._date_major_feature_panel_shape = (
            tuple(int(item) for item in date_major_feature_panel_shape)
            if date_major_feature_panel_shape is not None
            else ()
        )
        self._date_major_feature_dtype = str(date_major_feature_dtype or self._feature_dtype or "float16")
        self._date_major_feature_panel: np.memmap | None = None
        self.normalization_manifest = dict(normalization_manifest)
        self.feature_mean = np.asarray(self.normalization_manifest.get("feature_mean", []), dtype=np.float32)
        self.feature_std = np.asarray(self.normalization_manifest.get("feature_std", []), dtype=np.float32)
        if self.feature_mean.size != len(self.feature_columns):
            self.feature_mean = np.zeros((len(self.feature_columns),), dtype=np.float32)
        if self.feature_std.size != len(self.feature_columns):
            self.feature_std = np.ones((len(self.feature_columns),), dtype=np.float32)
        self.date_values = np.asarray([str(item) for item in date_values], dtype=object)
        self.stock_values = np.asarray([str(item).strip().upper() for item in stock_values], dtype=object)
        self.static_context_ids = static_context_ids
        self.label_arrays = dict(label_arrays)
        self.y_daily_excess = label_arrays["daily_excess_return"]
        self.y_cum_excess = label_arrays["cumulative_excess_return"]
        self.y_rank_by_horizon = label_arrays["rank_by_horizon"]
        self.y_rank_20d = label_arrays["rank_20d"]
        self.y_drawdown_by_horizon = label_arrays["drawdown_by_horizon"]
        self.y_worst_by_horizon = label_arrays["worst_by_horizon"]
        self.y_upside_by_horizon = label_arrays["upside_by_horizon"]
        self.y_max_drawdown_20d = label_arrays["max_drawdown_20d"]
        self.y_worst_1d_20d = label_arrays["worst_1d_20d"]
        self.y_upside_20d = label_arrays["upside_20d"]
        self.y_daily_return = label_arrays.get("daily_return")
        self.y_benchmark_daily_return = label_arrays.get("benchmark_daily_return")
        self.y_cum_return = label_arrays.get("cumulative_return")
        self.y_benchmark_cum_return = label_arrays.get("benchmark_cumulative_return")
        self.y_cum_return_1to20 = label_arrays.get("cumulative_return_1to20")
        self.y_benchmark_cum_return_1to20 = label_arrays.get("benchmark_cumulative_return_1to20")
        self.y_cum_excess_1to20 = label_arrays.get("cumulative_excess_return_1to20")
        self.y_rank_1to20 = label_arrays.get("rank_1to20")
        self.y_industry_rank_by_horizon = label_arrays.get("industry_rank_by_horizon")
        self.y_entry_tradeable = label_arrays.get("entry_tradeable")
        self.y_entry_limit_up_buy_blocked = label_arrays.get("entry_limit_up_buy_blocked")
        self.y_entry_suspended_or_no_open = label_arrays.get("entry_suspended_or_no_open")
        self.y_forward_tradeable_ratio_by_horizon = label_arrays.get("forward_tradeable_ratio_by_horizon")

    @property
    def row_count(self) -> int:
        return int(len(self.sample_index))

    @property
    def input_dim(self) -> int:
        return int(len(self.feature_columns))

    @property
    def feature_count(self) -> int:
        return int(len(self.feature_columns))

    @property
    def lookback_days(self) -> int:
        return int(self.manifest.get("lookback_days", 0) or 0)

    @property
    def feature_store_shape(self) -> tuple[int, int, int]:
        return (
            int(len(self.date_values)),
            int(len(self.stock_values)),
            int(len(self.feature_columns)),
        )

    @property
    def feature_store_path(self) -> Path:
        return self.feature_panel_path

    @property
    def role(self) -> np.ndarray:
        return self.sample_index["role"].astype(str).to_numpy(dtype=object)

    @property
    def date(self) -> np.ndarray:
        return pd.to_datetime(self.sample_index["date"]).to_numpy(dtype=object)

    @property
    def stock(self) -> np.ndarray:
        return self.sample_index["stock"].astype(str).to_numpy(dtype=object)

    @property
    def sequence_start_dates(self) -> np.ndarray:
        return pd.to_datetime(self.sample_index["sequence_start_date"]).to_numpy(dtype=object)

    @property
    def label_end_dates(self) -> np.ndarray:
        return pd.to_datetime(self.sample_index["label_end_date"]).to_numpy(dtype=object)

    @property
    def cumulative_horizons(self) -> tuple[int, ...]:
        return normalize_cumulative_horizons(
            self.manifest.get("cumulative_horizons", PATH20_CUMULATIVE_HORIZONS),
            horizon=int(self.manifest.get("horizon", PATH20_HORIZON) or PATH20_HORIZON),
        )

    def open_feature_store(self) -> np.memmap:
        if self._feature_panel is None:
            self._feature_panel = np.memmap(
                self.feature_panel_path,
                dtype=self._feature_dtype,
                mode="r",
                shape=self._feature_panel_shape,
            )
        return self._feature_panel

    @property
    def has_date_major_feature_panel(self) -> bool:
        return (
            self.date_major_feature_panel_path is not None
            and self.date_major_feature_panel_path.exists()
            and len(self._date_major_feature_panel_shape) == 3
        )

    def open_date_major_feature_store(self) -> np.memmap:
        if not self.has_date_major_feature_panel:
            raise ValueError("qdp training pack does not have a date-major feature panel.")
        if self._date_major_feature_panel is None:
            self._date_major_feature_panel = np.memmap(
                self.date_major_feature_panel_path,
                dtype=self._date_major_feature_dtype,
                mode="r",
                shape=self._date_major_feature_panel_shape,
            )
        return self._date_major_feature_panel

    def raw_input_window(self, row_idx: int) -> np.ndarray:
        return self.raw_input_windows(np.asarray([int(row_idx)], dtype=np.int64))[0]

    def raw_input_windows(self, row_indices: np.ndarray) -> np.ndarray:
        rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        lookback = int(self.lookback_days)
        out = np.zeros((len(rows), lookback, self.input_dim), dtype=np.float32)
        if len(rows) == 0:
            return out
        panel = self.open_feature_store()
        request = self.sample_index.iloc[rows][["global_stock_pos", "global_date_pos"]].copy()
        request["_request_pos"] = np.arange(len(rows), dtype=np.int64)
        request["global_stock_pos"] = pd.to_numeric(request["global_stock_pos"], errors="coerce").fillna(-1).astype("int64")
        request["global_date_pos"] = pd.to_numeric(request["global_date_pos"], errors="coerce").fillna(-1).astype("int64")
        for stock_pos, group in request.groupby("global_stock_pos", sort=False):
            stock_idx = int(stock_pos)
            if stock_idx < 0 or stock_idx >= int(panel.shape[0]):
                continue
            matrix = panel[stock_idx, :, :]
            if matrix.shape[0] < lookback:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(matrix, window_shape=lookback, axis=0)
            windows = np.moveaxis(windows, -1, 1)
            end_positions = group["global_date_pos"].to_numpy(dtype=np.int64, copy=False) + 1
            starts = end_positions - lookback
            request_positions = group["_request_pos"].to_numpy(dtype=np.int64, copy=False)
            valid = (starts >= 0) & (starts < int(windows.shape[0]))
            if np.any(valid):
                out[request_positions[valid]] = windows[starts[valid]]
            if np.any(~valid):
                for request_pos, end_pos in zip(request_positions[~valid], end_positions[~valid], strict=False):
                    source_start = max(int(end_pos) - lookback, 0)
                    source_end = max(int(end_pos), 0)
                    target_start = source_start - (int(end_pos) - lookback)
                    if source_end > source_start:
                        out[int(request_pos), target_start : target_start + (source_end - source_start), :] = matrix[source_start:source_end, :]
        return out

    def raw_date_input_windows(self, row_indices: np.ndarray) -> np.ndarray:
        rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        lookback = int(self.lookback_days)
        out = np.zeros((len(rows), lookback, self.input_dim), dtype=np.float32)
        if len(rows) == 0:
            return out

        use_date_major = self.has_date_major_feature_panel
        if use_date_major:
            panel = self.open_date_major_feature_store()
            date_axis = 0
            stock_axis = 1
        else:
            panel = self.open_feature_store()
            date_axis = 1
            stock_axis = 0

        request = self.sample_index.iloc[rows][["global_stock_pos", "global_date_pos"]].copy()
        request["_request_pos"] = np.arange(len(rows), dtype=np.int64)
        request["global_stock_pos"] = pd.to_numeric(request["global_stock_pos"], errors="coerce").fillna(-1).astype("int64")
        request["global_date_pos"] = pd.to_numeric(request["global_date_pos"], errors="coerce").fillna(-1).astype("int64")
        date_count = int(panel.shape[date_axis])
        stock_count = int(panel.shape[stock_axis])
        for date_pos, group in request.groupby("global_date_pos", sort=False):
            end_pos = int(date_pos) + 1
            if end_pos <= 0:
                continue
            source_start = max(end_pos - lookback, 0)
            source_end = min(end_pos, date_count)
            if source_end <= source_start:
                continue
            target_start = source_start - (end_pos - lookback)
            stock_idx = group["global_stock_pos"].to_numpy(dtype=np.int64, copy=False)
            request_positions = group["_request_pos"].to_numpy(dtype=np.int64, copy=False)
            valid = (stock_idx >= 0) & (stock_idx < stock_count)
            if not np.any(valid):
                continue
            valid_stock_idx = stock_idx[valid]
            valid_request_positions = request_positions[valid]
            if use_date_major:
                values = np.asarray(panel[source_start:source_end, valid_stock_idx, :], dtype=np.float32)
                if values.ndim == 3 and values.shape[0] == source_end - source_start:
                    values = np.transpose(values, (1, 0, 2))
            else:
                values = np.asarray(panel[valid_stock_idx, source_start:source_end, :], dtype=np.float32)
            out[
                valid_request_positions,
                target_start : target_start + (source_end - source_start),
                :,
            ] = values
        return out

    def input_window(self, row_idx: int, *, store: Any | None = None) -> np.ndarray:
        del store
        return self.raw_input_window(int(row_idx))

    def input_windows(self, row_indices: np.ndarray) -> np.ndarray:
        return self.raw_input_windows(row_indices)

    def date_input_windows(self, row_indices: np.ndarray) -> np.ndarray:
        return self.raw_date_input_windows(row_indices)

    def risk_by_horizon(self, row_idx: int | np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.asarray(self.y_drawdown_by_horizon[row_idx], dtype=np.float32),
                np.asarray(self.y_worst_by_horizon[row_idx], dtype=np.float32),
                np.asarray(self.y_upside_by_horizon[row_idx], dtype=np.float32),
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

    def role_indices(self, role: str) -> np.ndarray:
        return np.flatnonzero(self.sample_index["role"].astype(str).to_numpy() == str(role))

    def cache_friendly_indices(self, indices: np.ndarray) -> np.ndarray:
        row_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if row_indices.size <= 1:
            return row_indices
        order_frame = self.sample_index.iloc[row_indices][["global_stock_pos", "global_date_pos"]].copy()
        order_frame["_row_idx"] = row_indices
        return order_frame.sort_values(["global_stock_pos", "global_date_pos", "_row_idx"], kind="mergesort")[
            "_row_idx"
        ].to_numpy(dtype=np.int64)

    def torch_dataset(self, indices: np.ndarray, *, target_scale: float = 100.0) -> ForecastMemmapTorchDataset:
        return ForecastMemmapTorchDataset(self, indices, target_scale=target_scale)

    def batch_torch_dataset(self, indices: np.ndarray, *, batch_size: int, target_scale: float = 100.0) -> ForecastMemmapBatchTorchDataset:
        return ForecastMemmapBatchTorchDataset(self, indices, batch_size=batch_size, target_scale=target_scale)

    def date_batch_torch_dataset(self, indices: np.ndarray, *, target_scale: float = 100.0) -> ForecastDateBatchTorchDataset:
        return ForecastDateBatchTorchDataset(self, indices, target_scale=target_scale)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


DEFAULT_STATIC_CONTEXT_FIELDS: tuple[str, ...] = (
    "symbol",
    "exchange",
    "industry",
    "liquidity_bucket",
    "price_bucket",
)
AVAILABLE_STATIC_CONTEXT_FIELDS: tuple[str, ...] = (
    "symbol",
    "exchange",
    "industry",
    "board",
    "liquidity_bucket",
    "price_bucket",
)
STATIC_CONTEXT_ID_COLUMNS: dict[str, str] = {
    "symbol": "symbol_id",
    "exchange": "exchange_id",
    "industry": "industry_id",
    "board": "board_id",
    "liquidity_bucket": "liquidity_bucket_id",
    "price_bucket": "price_bucket_id",
}
STATIC_CONTEXT_FIELDS: tuple[str, ...] = tuple(STATIC_CONTEXT_ID_COLUMNS[field] for field in DEFAULT_STATIC_CONTEXT_FIELDS)


def normalize_static_context_fields(fields: tuple[str, ...] | list[str] | str | None = None) -> tuple[str, ...]:
    if fields is None:
        values = list(DEFAULT_STATIC_CONTEXT_FIELDS)
    elif isinstance(fields, str):
        values = [item.strip() for item in fields.split(",") if item.strip()]
    else:
        values = [str(item).strip() for item in fields if str(item).strip()]
    if not values:
        values = list(DEFAULT_STATIC_CONTEXT_FIELDS)
    normalized: list[str] = []
    for item in values:
        value = item[:-3] if item.endswith("_id") else item
        if value not in AVAILABLE_STATIC_CONTEXT_FIELDS:
            raise ValueError(f"Unsupported static context field: {item}")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def static_context_id_columns(fields: tuple[str, ...] | list[str] | str | None = None) -> tuple[str, ...]:
    return tuple(STATIC_CONTEXT_ID_COLUMNS[field] for field in normalize_static_context_fields(fields))


def _fingerprint_mapping(mapping: dict[str, int]) -> str:
    payload = json.dumps({str(key): int(value) for key, value in sorted(mapping.items())}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _exchange_for_symbol(symbol: str) -> str:
    value = str(symbol).strip().upper()
    if value.endswith(".SH"):
        return "SH"
    if value.endswith(".SZ"):
        return "SZ"
    if value.endswith(".BJ"):
        return "BJ"
    return "UNKNOWN"


def _metadata_series(prepared: PreparedPolicyInputs, frame_name: str, value_column: str, universe: list[str]) -> pd.Series:
    frame = dict(getattr(prepared, "metadata_frames", {}) or {}).get(frame_name)
    if frame is None or frame.empty or not {"symbol", value_column}.issubset(frame.columns):
        return pd.Series("", index=universe, dtype=object)
    work = frame.copy()
    work["symbol"] = work["symbol"].astype(str).str.strip().str.upper()
    work[value_column] = work[value_column].astype(str).str.strip()
    return work.drop_duplicates(subset=["symbol"]).set_index("symbol")[value_column].reindex(universe).fillna("")


def _primary_board_series(prepared: PreparedPolicyInputs, universe: list[str]) -> pd.Series:
    board_frame = dict(getattr(prepared, "metadata_frames", {}) or {}).get("board_membership")
    if board_frame is None or board_frame.empty or not {"symbol", "board_kind", "board_name"}.issubset(board_frame.columns):
        return pd.Series("", index=universe, dtype=object)
    board_work = board_frame.copy()
    board_work["symbol"] = board_work["symbol"].astype(str).str.strip().str.upper()
    board_work = board_work[board_work["symbol"].isin(universe)]
    if board_work.empty:
        return pd.Series("", index=universe, dtype=object)
    board_work["board_key"] = (
        board_work["board_kind"].astype(str).str.strip()
        + ":"
        + board_work["board_name"].astype(str).str.strip()
    )
    board_work = board_work[board_work["board_key"].astype(str).str.strip() != ":"]
    if board_work.empty:
        return pd.Series("", index=universe, dtype=object)
    primary = (
        board_work.sort_values(["symbol", "board_key"], kind="mergesort")
        .drop_duplicates(subset=["symbol"], keep="first")
        .set_index("symbol")["board_key"]
    )
    return primary.reindex(universe).fillna("")


def build_static_context_vocab(
    prepared: PreparedPolicyInputs,
    *,
    static_context_fields: tuple[str, ...] | list[str] | str | None = None,
) -> dict[str, Any]:
    fields = normalize_static_context_fields(static_context_fields)
    universe = sorted({str(stock).strip().upper() for stock in prepared.universe})
    industry = _metadata_series(prepared, "industry_map", "industry", universe)
    board = _primary_board_series(prepared, universe)
    board_values = sorted({str(item) for item in board.dropna().tolist() if str(item).strip()})
    symbol_vocab = {"<UNK>": 0, **{symbol: idx + 1 for idx, symbol in enumerate(universe)}}
    exchange_values = sorted({_exchange_for_symbol(symbol) for symbol in universe if _exchange_for_symbol(symbol) != "UNKNOWN"})
    exchange_vocab = {"<UNK>": 0, **{value: idx + 1 for idx, value in enumerate(exchange_values)}}
    industry_values = sorted({str(item) for item in industry.dropna().tolist() if str(item).strip()})
    industry_vocab = {"<UNK>": 0, **{value: idx + 1 for idx, value in enumerate(industry_values)}}
    board_vocab = {"<UNK>": 0, **{value: idx + 1 for idx, value in enumerate(board_values)}}
    bucket_vocab = {"<UNK>": 0, **{str(idx): idx for idx in range(1, 6)}}
    return {
        "enabled": True,
        "fields": list(fields),
        "id_columns": list(static_context_id_columns(fields)),
        "symbol_vocab": symbol_vocab,
        "exchange_vocab": exchange_vocab,
        "industry_vocab": industry_vocab,
        "board_vocab": board_vocab,
        "liquidity_bucket_vocab": dict(bucket_vocab),
        "price_bucket_vocab": dict(bucket_vocab),
        "symbol_vocab_fingerprint": _fingerprint_mapping(symbol_vocab),
        "industry_vocab_fingerprint": _fingerprint_mapping(industry_vocab),
        "board_vocab_fingerprint": _fingerprint_mapping(board_vocab),
        "exchange_vocab_fingerprint": _fingerprint_mapping(exchange_vocab),
        "liquidity_bucket_vocab_fingerprint": _fingerprint_mapping(bucket_vocab),
        "price_bucket_vocab_fingerprint": _fingerprint_mapping(bucket_vocab),
        "vocab_sizes": {
            "symbol": int(len(symbol_vocab)),
            "exchange": int(len(exchange_vocab)),
            "industry": int(len(industry_vocab)),
            "board": int(len(board_vocab)),
            "liquidity_bucket": int(len(bucket_vocab)),
            "price_bucket": int(len(bucket_vocab)),
        },
    }


def _static_context_for_universe(
    prepared: PreparedPolicyInputs,
    *,
    universe: list[str],
    vocab: dict[str, Any],
) -> pd.DataFrame:
    industry = _metadata_series(prepared, "industry_map", "industry", universe)
    board = _primary_board_series(prepared, universe)
    symbol_vocab = dict(vocab.get("symbol_vocab", {}) or {})
    exchange_vocab = dict(vocab.get("exchange_vocab", {}) or {})
    industry_vocab = dict(vocab.get("industry_vocab", {}) or {})
    board_vocab = dict(vocab.get("board_vocab", {}) or {})
    rows: list[dict[str, Any]] = []
    for symbol in universe:
        exchange = _exchange_for_symbol(symbol)
        rows.append(
            {
                "stock": symbol,
                "symbol_id": int(symbol_vocab.get(symbol, 0)),
                "exchange_id": int(exchange_vocab.get(exchange, 0)),
                "industry_id": int(industry_vocab.get(str(industry.get(symbol, "") or ""), 0)),
                "board_id": int(board_vocab.get(str(board.get(symbol, "") or ""), 0)),
                "liquidity_bucket_id": 0,
                "price_bucket_id": 0,
            }
        )
    return pd.DataFrame(rows).set_index("stock")


def _cross_section_bucket(
    source: pd.DataFrame,
    *,
    window: int = 20,
    buckets: int = 5,
) -> dict[tuple[pd.Timestamp, str], int]:
    rolling = source.rolling(max(int(window), 1), min_periods=1).mean()
    out: dict[tuple[pd.Timestamp, str], int] = {}
    for dt, row in rolling.iterrows():
        values = pd.to_numeric(row, errors="coerce")
        valid = values.replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            continue
        ranks = valid.rank(method="first", pct=True)
        for stock, pct in ranks.items():
            bucket = int(np.ceil(float(pct) * int(buckets)))
            out[(pd.Timestamp(dt), str(stock))] = max(1, min(int(buckets), bucket))
    return out


def _static_context_schema_disabled(
    static_context_fields: tuple[str, ...] | list[str] | str | None = None,
) -> dict[str, Any]:
    fields = normalize_static_context_fields(static_context_fields)
    return {
        "enabled": False,
        "fields": list(fields),
        "id_columns": list(static_context_id_columns(fields)),
        "vocab_sizes": {},
    }


def _role_for_year(
    year: int,
    *,
    train_start_year: int,
    train_end_year: int,
    validation_year: int,
    test_year: int,
) -> str | None:
    if int(train_start_year) <= int(year) <= int(train_end_year):
        return "train"
    if int(year) == int(validation_year):
        return "validation"
    if int(year) == int(test_year):
        return "test"
    return None


def _date_role_boundaries(
    dates: list[pd.Timestamp],
    *,
    train_start_year: int,
    train_end_year: int,
    validation_year: int,
    test_year: int,
    purge_trading_days: int,
) -> dict[str, list[pd.Timestamp]]:
    by_role: dict[str, list[pd.Timestamp]] = {"train": [], "validation": [], "test": []}
    for dt in dates:
        role = _role_for_year(
            int(dt.year),
            train_start_year=train_start_year,
            train_end_year=train_end_year,
            validation_year=validation_year,
            test_year=test_year,
        )
        if role:
            by_role[role].append(dt)
    eligible: dict[str, list[pd.Timestamp]] = {}
    for role, role_dates in by_role.items():
        if len(role_dates) > int(purge_trading_days):
            eligible[role] = role_dates[: -int(purge_trading_days)]
        else:
            eligible[role] = []
    return eligible


def _rotated_items(items: list[str] | tuple[str, ...], offset: int) -> list[str]:
    values = list(items)
    if not values:
        return []
    pivot = int(offset) % len(values)
    return [*values[pivot:], *values[:pivot]]


def _safe_label_value(frame: pd.DataFrame, date: pd.Timestamp, stock: str) -> float:
    try:
        value = frame.loc[date, stock]
    except KeyError:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _feature_nanmean_nanstd(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    sums = np.where(finite, values, 0.0).sum(axis=0)
    mean = np.divide(
        sums,
        counts,
        out=np.zeros((values.shape[-1],), dtype=np.float64),
        where=counts > 0,
    )
    centered = np.where(finite, values - mean.reshape(1, -1), 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=0),
        counts,
        out=np.zeros((values.shape[-1],), dtype=np.float64),
        where=counts > 0,
    )
    std = np.sqrt(variance)
    std = np.where(counts > 0, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def _empty_dataset(
    *,
    feature_columns: list[str],
    lookback_days: int,
    horizon: int,
    cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    manifest: dict[str, Any],
    normalization_manifest: dict[str, Any],
) -> ForecastSequenceDataset:
    resolved_horizons = normalize_cumulative_horizons(cumulative_horizons, horizon=horizon)
    x = np.empty((0, int(lookback_days), int(len(feature_columns))), dtype=np.float32)
    return ForecastSequenceDataset(
        x=x,
        y_daily_excess=np.empty((0, int(horizon)), dtype=np.float32),
        y_cum_excess=np.empty((0, len(resolved_horizons)), dtype=np.float32),
        y_rank_by_horizon=np.empty((0, len(resolved_horizons)), dtype=np.float32),
        y_rank_20d=np.empty((0,), dtype=np.float32),
        y_drawdown_by_horizon=np.empty((0, len(resolved_horizons)), dtype=np.float32),
        y_worst_by_horizon=np.empty((0, len(resolved_horizons)), dtype=np.float32),
        y_upside_by_horizon=np.empty((0, len(resolved_horizons)), dtype=np.float32),
        y_max_drawdown_20d=np.empty((0,), dtype=np.float32),
        y_worst_1d_20d=np.empty((0,), dtype=np.float32),
        y_upside_20d=np.empty((0,), dtype=np.float32),
        date=np.array([], dtype=object),
        stock=np.array([], dtype=object),
        role=np.array([], dtype=object),
        sequence_start_dates=np.array([], dtype=object),
        label_end_dates=np.array([], dtype=object),
        feature_columns=list(feature_columns),
        normalization_manifest=dict(normalization_manifest),
        manifest=dict(manifest),
        static_context_ids=None,
    )


def build_forecast_sequence_dataset(
    prepared: PreparedPolicyInputs,
    *,
    train_start_year: int = 2019,
    train_end_year: int = 2022,
    validation_year: int = 2023,
    test_year: int = 2024,
    lookback_days: int = 252,
    horizon: int = PATH20_HORIZON,
    cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    execution_mode: str = "next_open",
    max_samples_per_role: int = 0,
    max_samples_per_date_per_role: int = 0,
    feature_profile: str = DEFAULT_FORECAST_FEATURE_PROFILE,
    max_feature_columns: int = DEFAULT_FORECAST_MAX_FEATURE_COLUMNS,
) -> ForecastSequenceDataset:
    lookback_days = int(lookback_days)
    horizon = int(horizon)
    resolved_horizons = normalize_cumulative_horizons(cumulative_horizons, horizon=horizon)
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")
    if int(train_start_year) > int(train_end_year):
        raise ValueError("train_start_year must be <= train_end_year.")

    dates = [pd.Timestamp(dt).normalize() for dt in prepared.close.index]
    date_to_pos = {dt: idx for idx, dt in enumerate(dates)}
    next_open_extra_day = 1 if str(execution_mode or "next_open").strip().lower() == "next_open" else 0
    label_forward_offset = horizon + next_open_extra_day
    eligible_dates_by_role = _date_role_boundaries(
        dates,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        validation_year=validation_year,
        test_year=test_year,
        purge_trading_days=label_forward_offset,
    )
    panels, feature_columns, feature_manifest = build_forecast_feature_panels(
        prepared,
        dates,
        feature_profile=feature_profile,
        max_feature_columns=max_feature_columns,
    )
    labels = build_path20_labels(prepared, execution_mode=execution_mode, horizon=horizon, cumulative_horizons=resolved_horizons)

    x_rows: list[np.ndarray] = []
    y_daily_rows: list[list[float]] = []
    y_cum_rows: list[list[float]] = []
    y_rank_by_horizon_rows: list[list[float]] = []
    y_rank_rows: list[float] = []
    y_drawdown_by_horizon_rows: list[list[float]] = []
    y_worst_by_horizon_rows: list[list[float]] = []
    y_upside_by_horizon_rows: list[list[float]] = []
    y_drawdown_rows: list[float] = []
    y_worst_rows: list[float] = []
    y_upside_rows: list[float] = []
    date_rows: list[pd.Timestamp] = []
    stock_rows: list[str] = []
    role_rows: list[str] = []
    sequence_start_rows: list[pd.Timestamp] = []
    label_end_rows: list[pd.Timestamp] = []
    dropped_target_nan = 0
    dropped_missing_lookback = 0
    capped = int(max_samples_per_role) > 0
    per_date_cap = max(int(max_samples_per_date_per_role), 0)
    capped_per_date = per_date_cap > 0
    sample_count_by_role = {"train": 0, "validation": 0, "test": 0}

    for role in ("train", "validation", "test"):
        for date_idx, signal_dt in enumerate(eligible_dates_by_role.get(role, [])):
            if capped and sample_count_by_role[role] >= int(max_samples_per_role):
                break
            signal_pos = date_to_pos.get(signal_dt)
            if signal_pos is None or signal_pos < lookback_days - 1:
                dropped_missing_lookback += len(prepared.universe)
                continue
            label_end_pos = signal_pos + label_forward_offset
            if label_end_pos >= len(dates):
                dropped_missing_lookback += len(prepared.universe)
                continue
            sequence_dates = dates[signal_pos - lookback_days + 1 : signal_pos + 1]
            membership = prepared.membership_frame.reindex(index=[signal_dt], columns=list(prepared.universe))
            membership_row = membership.iloc[0].fillna(False) if not membership.empty else pd.Series(False, index=prepared.universe)
            sample_count_this_date = 0
            stock_iter = _rotated_items(list(prepared.universe), date_idx * max(per_date_cap, 1)) if capped_per_date else list(prepared.universe)
            for stock in stock_iter:
                if capped and sample_count_by_role[role] >= int(max_samples_per_role):
                    break
                if capped_per_date and sample_count_this_date >= per_date_cap:
                    break
                if not bool(membership_row.get(stock, False)):
                    continue
                daily_target = [
                    _safe_label_value(labels.daily_excess_return[step], signal_dt, str(stock))
                    for step in range(1, horizon + 1)
                ]
                cum_target = [
                    _safe_label_value(labels.cumulative_excess_return[step], signal_dt, str(stock))
                    for step in resolved_horizons
                ]
                rank_by_horizon_target = [
                    _safe_label_value(labels.forward_rank[step], signal_dt, str(stock))
                    for step in resolved_horizons
                ]
                rank_horizon = int(horizon) if int(horizon) in labels.forward_rank else int(resolved_horizons[-1])
                rank_target = _safe_label_value(labels.forward_rank[rank_horizon], signal_dt, str(stock))
                drawdown_by_horizon_target = [
                    _safe_label_value(labels.path_max_drawdown_by_horizon[step], signal_dt, str(stock))
                    for step in resolved_horizons
                ]
                worst_by_horizon_target = [
                    _safe_label_value(labels.path_worst_1d_by_horizon[step], signal_dt, str(stock))
                    for step in resolved_horizons
                ]
                upside_by_horizon_target = [
                    _safe_label_value(labels.path_upside_capture_by_horizon[step], signal_dt, str(stock))
                    for step in resolved_horizons
                ]
                drawdown_target = _safe_label_value(labels.path_max_drawdown_20d, signal_dt, str(stock))
                worst_target = _safe_label_value(labels.path_worst_1d_20d, signal_dt, str(stock))
                upside_target = _safe_label_value(labels.path_upside_capture_20d, signal_dt, str(stock))
                all_targets = [
                    *daily_target,
                    *cum_target,
                    *rank_by_horizon_target,
                    rank_target,
                    *drawdown_by_horizon_target,
                    *worst_by_horizon_target,
                    *upside_by_horizon_target,
                    drawdown_target,
                    worst_target,
                    upside_target,
                ]
                if not np.isfinite(np.asarray(all_targets, dtype=float)).all():
                    dropped_target_nan += 1
                    continue
                try:
                    sequence = np.stack(
                        [
                            panels[dt].reindex(index=list(prepared.universe), columns=feature_columns).loc[str(stock)].to_numpy(
                                dtype=np.float32,
                                copy=False,
                            )
                            for dt in sequence_dates
                        ],
                        axis=0,
                    )
                except KeyError:
                    dropped_missing_lookback += 1
                    continue
                x_rows.append(sequence)
                y_daily_rows.append(daily_target)
                y_cum_rows.append(cum_target)
                y_rank_by_horizon_rows.append(rank_by_horizon_target)
                y_rank_rows.append(rank_target)
                y_drawdown_by_horizon_rows.append(drawdown_by_horizon_target)
                y_worst_by_horizon_rows.append(worst_by_horizon_target)
                y_upside_by_horizon_rows.append(upside_by_horizon_target)
                y_drawdown_rows.append(drawdown_target)
                y_worst_rows.append(worst_target)
                y_upside_rows.append(upside_target)
                date_rows.append(signal_dt)
                stock_rows.append(str(stock))
                role_rows.append(role)
                sequence_start_rows.append(sequence_dates[0])
                label_end_rows.append(dates[label_end_pos])
                sample_count_by_role[role] += 1
                sample_count_this_date += 1

    normalization_manifest: dict[str, Any] = {
        "fit_role": "train_only",
        "method": "zscore",
        "feature_count": int(len(feature_columns)),
    }
    base_manifest: dict[str, Any] = {
        "status": "completed",
        "stage": "forecast_sequence_dataset",
        "lookback_days": int(lookback_days),
        "horizon": int(horizon),
        "forecast_horizon": int(horizon),
        "execution_mode": str(execution_mode),
        "label_semantics": dict(labels.metadata),
        "role_years": {
            "train_start_year": int(train_start_year),
            "train_end_year": int(train_end_year),
            "validation_year": int(validation_year),
            "test_year": int(test_year),
        },
        "role_purge_trading_days": int(label_forward_offset),
        "next_open_label_extra_trading_day": int(next_open_extra_day),
        "feature_profile": str(feature_manifest.get("feature_profile", feature_profile)),
        "feature_manifest": dict(feature_manifest),
        "feature_columns": list(feature_columns),
        "feature_group_counts": dict(feature_manifest.get("feature_group_counts", {})),
        "feature_count_before_cap": int(feature_manifest.get("feature_count_before_cap", len(feature_columns))),
        "feature_count_after_cap": int(feature_manifest.get("feature_count_after_cap", len(feature_columns))),
        "raw_kline_feature_count": int(feature_manifest.get("raw_kline_feature_count", 0)),
        "market_context_feature_count": int(feature_manifest.get("market_context_feature_count", 0)),
        "peer_context_feature_count": int(feature_manifest.get("peer_context_feature_count", 0)),
        "sector_context_feature_count": int(feature_manifest.get("sector_context_feature_count", 0)),
        "sector_relative_context_feature_count": int(feature_manifest.get("sector_relative_context_feature_count", 0)),
        "regime_context_feature_count": int(feature_manifest.get("regime_context_feature_count", 0)),
        "source_sector_board_view_id": str(feature_manifest.get("source_sector_board_view_id", "")),
        "alpha_prior_feature_count": int(feature_manifest.get("alpha_prior_feature_count", 0)),
        "history_quality_feature_count": int(feature_manifest.get("history_quality_feature_count", 0)),
        "feature_profile_audit": dict(feature_manifest.get("feature_profile_audit", {})),
        "cumulative_horizons": [int(item) for item in resolved_horizons],
        "rank_horizons": [int(item) for item in resolved_horizons],
        "risk_horizons": [int(item) for item in resolved_horizons],
        "sample_count_by_role": {key: int(value) for key, value in sample_count_by_role.items()},
        "dropped_target_nan": int(dropped_target_nan),
        "dropped_missing_lookback": int(dropped_missing_lookback),
        "max_samples_per_role": int(max_samples_per_role),
        "max_samples_per_date_per_role": int(max_samples_per_date_per_role),
    }

    if not x_rows:
        base_manifest["status"] = "insufficient_or_incomplete"
        base_manifest["reason"] = "no_forecast_samples"
        return _empty_dataset(
            feature_columns=feature_columns,
            lookback_days=lookback_days,
            horizon=horizon,
            cumulative_horizons=resolved_horizons,
            manifest=base_manifest,
            normalization_manifest=normalization_manifest,
        )

    x_raw = np.stack(x_rows, axis=0).astype(np.float32, copy=False)
    feature_nan_ratio = float(np.isnan(x_raw).mean()) if x_raw.size else 0.0
    roles_np = np.array(role_rows, dtype=object)
    train_mask = roles_np == "train"
    if bool(train_mask.any()):
        train_values = x_raw[train_mask].reshape(-1, x_raw.shape[-1])
        feature_mean, feature_std = _feature_nanmean_nanstd(train_values)
    else:
        feature_mean = np.zeros((x_raw.shape[-1],), dtype=np.float32)
        feature_std = np.ones((x_raw.shape[-1],), dtype=np.float32)
        base_manifest["status"] = "insufficient_or_incomplete"
        base_manifest["reason"] = "no_train_samples"
    feature_mean = np.where(np.isfinite(feature_mean), feature_mean, 0.0).astype(np.float32)
    feature_std = np.where(np.isfinite(feature_std) & (np.abs(feature_std) > 1.0e-8), feature_std, 1.0).astype(np.float32)
    x = ((x_raw - feature_mean.reshape(1, 1, -1)) / feature_std.reshape(1, 1, -1)).astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    normalization_manifest.update(
        {
            "feature_mean": feature_mean.astype(float).tolist(),
            "feature_std": feature_std.astype(float).tolist(),
            "raw_feature_nan_ratio": feature_nan_ratio,
        }
    )
    base_manifest.update(
        {
            "sample_count": int(x.shape[0]),
            "feature_count": int(x.shape[-1]),
            "sequence_shape": [int(x.shape[0]), int(x.shape[1]), int(x.shape[2])],
            "target_shape": [int(x.shape[0]), int(horizon)],
            "normalization": normalization_manifest,
        }
    )

    return ForecastSequenceDataset(
        x=x,
        y_daily_excess=np.asarray(y_daily_rows, dtype=np.float32),
        y_cum_excess=np.asarray(y_cum_rows, dtype=np.float32),
        y_rank_by_horizon=np.asarray(y_rank_by_horizon_rows, dtype=np.float32),
        y_rank_20d=np.asarray(y_rank_rows, dtype=np.float32),
        y_drawdown_by_horizon=np.asarray(y_drawdown_by_horizon_rows, dtype=np.float32),
        y_worst_by_horizon=np.asarray(y_worst_by_horizon_rows, dtype=np.float32),
        y_upside_by_horizon=np.asarray(y_upside_by_horizon_rows, dtype=np.float32),
        y_max_drawdown_20d=np.asarray(y_drawdown_rows, dtype=np.float32),
        y_worst_1d_20d=np.asarray(y_worst_rows, dtype=np.float32),
        y_upside_20d=np.asarray(y_upside_rows, dtype=np.float32),
        date=np.array(date_rows, dtype=object),
        stock=np.array(stock_rows, dtype=object),
        role=roles_np,
        sequence_start_dates=np.array(sequence_start_rows, dtype=object),
        label_end_dates=np.array(label_end_rows, dtype=object),
        feature_columns=list(feature_columns),
        normalization_manifest=normalization_manifest,
        manifest=base_manifest,
    )


def _history_bucket(value: float) -> str:
    if not np.isfinite(float(value)):
        return "low"
    if float(value) >= 0.95:
        return "high"
    if float(value) >= 0.80:
        return "medium"
    return "low"


def _write_array_memmap(path: Path, values: np.ndarray, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.memmap(path, dtype="float32", mode="w+", shape=shape)
    if int(np.prod(shape, dtype=np.int64)) > 0:
        array[...] = values.astype(np.float32, copy=False).reshape(shape)
    array.flush()
    return np.memmap(path, dtype="float32", mode="r", shape=shape)


def _resolve_manifest_path(value: Any, *, manifest_path: Path) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute():
        path = manifest_path.parent / path
    elif not path.exists():
        sibling = manifest_path.parent / path.name
        if sibling.exists():
            path = sibling
    return path


def _validate_memmap_file(path: Path, *, shape: tuple[int, ...], label: str, dtype: str | np.dtype = "float32") -> None:
    if not path.exists():
        raise ValueError(f"forecast memmap manifest missing {label}: {path}")
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    actual_bytes = int(path.stat().st_size)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"forecast memmap manifest has invalid {label} size: "
            f"expected {expected_bytes} bytes for shape {shape}, got {actual_bytes} bytes at {path}"
        )


def _cap_samples_by_role(sample_index: pd.DataFrame, *, max_samples_per_role: int, max_samples_per_date_per_role: int) -> pd.DataFrame:
    out = sample_index.copy()
    if out.empty:
        return out.reset_index(drop=True)
    out = out.sort_values(["role", "date", "_shard_idx", "stock"], kind="mergesort").reset_index(drop=True)
    per_date_cap = max(int(max_samples_per_date_per_role), 0)
    if per_date_cap > 0:
        out = out.groupby(["role", "date"], sort=False, group_keys=False).head(per_date_cap).reset_index(drop=True)
    role_cap = max(int(max_samples_per_role), 0)
    if role_cap <= 0:
        return out.reset_index(drop=True)

    capped: list[pd.DataFrame] = []
    for _, group in out.groupby("role", sort=False):
        if len(group) <= role_cap:
            capped.append(group)
            continue
        positions = np.linspace(0, len(group) - 1, role_cap, dtype=np.int64)
        capped.append(group.iloc[np.unique(positions)])
    return pd.concat(capped, ignore_index=True) if capped else out.iloc[0:0].copy()


def _feature_stats_from_sums(sums: np.ndarray, sq_sums: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0).astype(np.float32)
    variance = np.divide(sq_sums, counts, out=np.zeros_like(sq_sums), where=counts > 0) - np.square(mean.astype(np.float64))
    std = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
    std = np.where(np.isfinite(std) & (np.abs(std) > 1.0e-8), std, 1.0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    return mean, std


def _fit_qdp_sharded_panel_normalization(
    *,
    shards: list[dict[str, Any]],
    feature_count: int,
    train_start_year: int,
    train_end_year: int,
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((int(feature_count),), dtype=np.float64)
    sq_sums = np.zeros((int(feature_count),), dtype=np.float64)
    counts = np.zeros((int(feature_count),), dtype=np.float64)
    for shard in shards:
        year = int(shard.get("year", 0) or 0)
        if year < int(train_start_year) or year > int(train_end_year):
            continue
        shape = tuple(int(item) for item in list(shard.get("feature_store_shape", []) or []))
        if len(shape) != 3 or int(shape[2]) != int(feature_count):
            continue
        path = Path(str(shard.get("feature_store_path", "") or ""))
        if not path.exists():
            continue
        values = np.asarray(np.memmap(path, dtype="float32", mode="r", shape=shape), dtype=np.float64).reshape(-1, int(feature_count))
        finite = np.isfinite(values)
        clean = np.where(finite, values, 0.0)
        sums += clean.sum(axis=0)
        sq_sums += np.square(clean).sum(axis=0)
        counts += finite.sum(axis=0)
    return _feature_stats_from_sums(sums, sq_sums, counts)


def _fit_qdp_sharded_sample_window_normalization(
    dataset: ForecastShardedMemmapDataset,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feature_count = int(dataset.input_dim)
    sums = np.zeros((feature_count,), dtype=np.float64)
    sq_sums = np.zeros((feature_count,), dtype=np.float64)
    counts = np.zeros((feature_count,), dtype=np.float64)
    for row_idx in np.asarray(train_indices, dtype=np.int64).reshape(-1):
        values = np.asarray(dataset.raw_input_window(int(row_idx)), dtype=np.float64).reshape(-1, feature_count)
        finite = np.isfinite(values)
        clean = np.where(finite, values, 0.0)
        sums += clean.sum(axis=0)
        sq_sums += np.square(clean).sum(axis=0)
        counts += finite.sum(axis=0)
    return _feature_stats_from_sums(sums, sq_sums, counts)


def _qdp_sharded_normalization_cache_path(
    manifest_path: Path,
    *,
    feature_columns: list[str],
    train_start_year: int,
    train_end_year: int,
) -> Path:
    payload = json.dumps(
        {
            "feature_columns": list(feature_columns),
            "train_start_year": int(train_start_year),
            "train_end_year": int(train_end_year),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return manifest_path.with_name(f"{manifest_path.stem}.train_norm_{int(train_start_year)}_{int(train_end_year)}_{digest}.json")


def _qdp_training_pack_progress(path: Path, stage: str, **payload: Any) -> None:
    write_json(
        path,
        _json_ready(
            {
                "artifact_type": "qdp_training_pack_build_progress",
                "stage": str(stage),
                "updated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
                **payload,
            }
        ),
    )


_REQUIRED_QDP_LABEL_ARRAYS: tuple[str, ...] = (
    "daily_excess_return",
    "cumulative_excess_return",
    "rank_by_horizon",
    "drawdown_by_horizon",
    "worst_by_horizon",
    "upside_by_horizon",
    "rank_20d",
    "max_drawdown_20d",
    "worst_1d_20d",
    "upside_20d",
)


def _label_array_specs(horizon: int, cumulative_horizons: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
    cumulative_count = len(cumulative_horizons)
    return {
        "daily_return": (int(horizon),),
        "benchmark_daily_return": (int(horizon),),
        "daily_excess_return": (int(horizon),),
        "cumulative_return": (cumulative_count,),
        "benchmark_cumulative_return": (cumulative_count,),
        "cumulative_excess_return": (cumulative_count,),
        "cumulative_return_1to20": (int(horizon),),
        "benchmark_cumulative_return_1to20": (int(horizon),),
        "cumulative_excess_return_1to20": (int(horizon),),
        "rank_1to20": (int(horizon),),
        "rank_by_horizon": (cumulative_count,),
        "industry_rank_by_horizon": (cumulative_count,),
        "drawdown_by_horizon": (cumulative_count,),
        "worst_by_horizon": (cumulative_count,),
        "upside_by_horizon": (cumulative_count,),
        "rank_20d": (),
        "max_drawdown_20d": (),
        "worst_1d_20d": (),
        "upside_20d": (),
        "entry_tradeable": (),
        "entry_limit_up_buy_blocked": (),
        "entry_suspended_or_no_open": (),
        "forward_tradeable_ratio_by_horizon": (cumulative_count,),
    }


def _common_sharded_label_arrays(shards: list[dict[str, Any]]) -> set[str]:
    names: set[str] | None = None
    for shard in shards:
        arrays = set(dict(dict(shard.get("label_manifest", {}) or {}).get("arrays", {}) or {}).keys())
        names = arrays if names is None else names & arrays
    return set() if names is None else set(names)


def _open_pack_label_arrays(manifest: dict[str, Any]) -> dict[str, np.memmap]:
    arrays: dict[str, np.memmap] = {}
    root = Path(str(manifest.get("manifest_json", "") or ".")).parent
    for name, meta_raw in dict(manifest.get("label_arrays", {}) or {}).items():
        meta = dict(meta_raw or {})
        path = Path(str(meta.get("path", "") or ""))
        if not path.is_absolute():
            path = root / path
        shape = tuple(int(item) for item in list(meta.get("shape", []) or []))
        dtype = str(meta.get("dtype", "float32") or "float32")
        _validate_memmap_file(path, shape=shape, label=f"training_pack_label_{name}", dtype=dtype)
        arrays[str(name)] = np.memmap(path, dtype=dtype, mode="r", shape=shape)
    return arrays


def _slice_pack_label_arrays(
    label_arrays: dict[str, np.memmap | np.ndarray],
    row_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    return {
        name: np.asarray(values[rows], dtype=np.float32).copy()
        for name, values in label_arrays.items()
    }


def _sample_count_by_role(sample_index: pd.DataFrame) -> dict[str, int]:
    roles = sample_index["role"].astype(str) if "role" in sample_index.columns else pd.Series(dtype=str)
    ordered_roles = [role for role in ("train", "validation", "test") if role in set(roles.tolist())]
    ordered_roles.extend(sorted(set(roles.tolist()) - set(ordered_roles)))
    return {role: int((roles == role).sum()) for role in ordered_roles}


def _load_training_pack_forecast_memmap_dataset(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    max_samples_per_role: int = 0,
    max_samples_per_date_per_role: int = 0,
) -> ForecastTrainingPackDataset:
    root = manifest_path.parent
    sample_index_path = Path(str(manifest.get("sample_index_path", "") or ""))
    if not sample_index_path.is_absolute():
        sample_index_path = root / sample_index_path
    if not sample_index_path.exists():
        raise ValueError(f"qdp training pack missing sample_index_path: {sample_index_path}")
    sample_index = pd.read_parquet(sample_index_path)
    source_sample_count_by_role = _sample_count_by_role(sample_index)
    feature_panel_path = Path(str(manifest.get("feature_panel_path", "") or ""))
    if not feature_panel_path.is_absolute():
        feature_panel_path = root / feature_panel_path
    feature_panel_shape = tuple(int(item) for item in list(manifest.get("feature_panel_shape", []) or []))
    feature_dtype = str(manifest.get("feature_dtype", "float16") or "float16")
    _validate_memmap_file(feature_panel_path, shape=feature_panel_shape, label="qdp_training_pack_feature_panel", dtype=feature_dtype)
    date_major_feature_panel_path: Path | None = None
    date_major_feature_panel_shape: tuple[int, int, int] | None = None
    date_major_feature_dtype = str(manifest.get("date_major_feature_dtype", feature_dtype) or feature_dtype)
    date_major_meta = dict(manifest.get("date_major_feature_panel", {}) or {})
    date_major_path_raw = str(
        date_major_meta.get("path", "")
        or manifest.get("date_major_feature_panel_path", "")
        or ""
    )
    if date_major_path_raw.strip():
        candidate_path = Path(date_major_path_raw)
        if not candidate_path.is_absolute():
            candidate_path = root / candidate_path
        candidate_shape_raw = (
            date_major_meta.get("shape")
            or manifest.get("date_major_feature_panel_shape")
            or []
        )
        candidate_shape = tuple(int(item) for item in list(candidate_shape_raw or []))
        candidate_dtype = str(
            date_major_meta.get("dtype")
            or manifest.get("date_major_feature_dtype")
            or feature_dtype
        )
        if len(candidate_shape) == 3:
            _validate_memmap_file(
                candidate_path,
                shape=candidate_shape,
                label="qdp_training_pack_date_major_feature_panel",
                dtype=candidate_dtype,
            )
            date_major_feature_panel_path = candidate_path
            date_major_feature_panel_shape = candidate_shape
            date_major_feature_dtype = candidate_dtype
    static_context_ids: np.memmap | np.ndarray | None = None
    static_meta = dict(manifest.get("static_context_ids", {}) or {})
    if static_meta:
        static_path = Path(str(static_meta.get("path", "") or ""))
        if not static_path.is_absolute():
            static_path = root / static_path
        static_shape = tuple(int(item) for item in list(static_meta.get("shape", []) or []))
        static_dtype = str(static_meta.get("dtype", "int32") or "int32")
        _validate_memmap_file(static_path, shape=static_shape, label="qdp_training_pack_static_context_ids", dtype=static_dtype)
        static_context_ids = np.memmap(static_path, dtype=static_dtype, mode="r", shape=static_shape)
    label_arrays = _open_pack_label_arrays(manifest)
    role_cap = max(int(max_samples_per_role), 0)
    per_date_cap = max(int(max_samples_per_date_per_role), 0)
    if role_cap > 0 or per_date_cap > 0:
        indexed_sample_index = sample_index.copy()
        indexed_sample_index["_pack_row_idx"] = np.arange(len(indexed_sample_index), dtype=np.int64)
        sample_index = _cap_samples_by_role(
            indexed_sample_index,
            max_samples_per_role=role_cap,
            max_samples_per_date_per_role=per_date_cap,
        ).reset_index(drop=True)
        pack_row_indices = sample_index["_pack_row_idx"].to_numpy(dtype=np.int64, copy=False)
        label_arrays = _slice_pack_label_arrays(label_arrays, pack_row_indices)
        if static_context_ids is not None:
            static_context_ids = np.asarray(static_context_ids[pack_row_indices], dtype=np.int64).copy()
        capped_manifest = dict(manifest)
        capped_manifest["sample_count"] = int(len(sample_index))
        capped_manifest["sample_count_by_role"] = _sample_count_by_role(sample_index)
        capped_manifest["qdp_training_pack_sample_cap_applied"] = True
        capped_manifest["qdp_training_pack_source_sample_count"] = int(sum(source_sample_count_by_role.values()))
        capped_manifest["qdp_training_pack_source_sample_count_by_role"] = source_sample_count_by_role
        capped_manifest["max_samples_per_role"] = int(role_cap)
        capped_manifest["max_samples_per_date_per_role"] = int(per_date_cap)
        manifest = capped_manifest
    return ForecastTrainingPackDataset(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        sample_index=sample_index,
        feature_columns=[str(item) for item in list(manifest.get("feature_columns", []) or [])],
        feature_panel_path=feature_panel_path,
        feature_panel_shape=feature_panel_shape,
        feature_dtype=feature_dtype,
        date_major_feature_panel_path=date_major_feature_panel_path,
        date_major_feature_panel_shape=date_major_feature_panel_shape,
        date_major_feature_dtype=date_major_feature_dtype,
        label_arrays=label_arrays,
        normalization_manifest=dict(manifest.get("normalization", {}) or {}),
        static_context_ids=static_context_ids,
        date_values=[str(item) for item in list(manifest.get("date_values", []) or [])],
        stock_values=[str(item) for item in list(manifest.get("stock_values", []) or [])],
    )


def _load_qdp_sharded_forecast_memmap_dataset(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    train_start_year: int = 2019,
    train_end_year: int = 2022,
    validation_year: int = 2023,
    test_year: int = 2024,
    max_samples_per_role: int = 0,
    max_samples_per_date_per_role: int = 0,
) -> ForecastShardedMemmapDataset:
    root = manifest_path.parent
    horizon = int(manifest.get("horizon", PATH20_HORIZON) or PATH20_HORIZON)
    cumulative_horizons = normalize_cumulative_horizons(
        manifest.get("cumulative_horizons", PATH20_CUMULATIVE_HORIZONS),
        horizon=horizon,
    )
    lookback_days = int(manifest.get("lookback_days", 0) or 0)
    feature_columns = [str(item) for item in manifest.get("feature_columns", [])]
    if lookback_days <= 0:
        raise ValueError(f"qdp sharded memmap manifest has invalid lookback_days: {manifest_path}")
    if not feature_columns:
        raise ValueError(f"qdp sharded memmap manifest missing feature_columns: {manifest_path}")

    loaded_shards: list[dict[str, Any]] = []
    sample_frames: list[pd.DataFrame] = []
    all_dates: set[str] = set()
    for raw_shard in list(manifest.get("shards", []) or []):
        shard = dict(raw_shard)
        if str(shard.get("status", "")) != "completed" or int(shard.get("sample_count", 0) or 0) <= 0:
            continue
        feature_shape = tuple(int(item) for item in list(shard.get("feature_store_shape", []) or []))
        if len(feature_shape) != 3 or int(feature_shape[2]) != len(feature_columns):
            raise ValueError(f"qdp sharded memmap shard has invalid feature_store_shape: {shard.get('shard_key', '')}")
        feature_path = Path(str(shard.get("feature_store_path", "") or ""))
        _validate_memmap_file(feature_path, shape=feature_shape, label=f"qdp shard feature_store_path {shard.get('shard_key', '')}")
        label_manifest_path = Path(str(shard.get("label_manifest_json", "") or ""))
        if not label_manifest_path.exists():
            raise ValueError(f"qdp sharded memmap missing labels_manifest: {label_manifest_path}")
        label_manifest = json.loads(label_manifest_path.read_text(encoding="utf-8"))
        shard["label_manifest"] = label_manifest
        for date_value in list(label_manifest.get("date_values", []) or []):
            all_dates.add(pd.Timestamp(date_value).strftime("%Y-%m-%d"))
        sample_index_path = Path(str(shard.get("sample_index_path", "") or ""))
        if not sample_index_path.exists():
            raise ValueError(f"qdp sharded memmap missing sample_index_path: {sample_index_path}")
        shard_idx = len(loaded_shards)
        loaded_shards.append(shard)
        frame = pd.read_parquet(sample_index_path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["_shard_idx"] = int(shard_idx)
        frame["_shard_key"] = str(shard.get("shard_key", ""))
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        frame["stock"] = frame["stock"].astype(str).str.strip().str.upper()
        frame["date_pos"] = pd.to_numeric(frame["date_pos"], errors="coerce").fillna(-1).astype("int32")
        frame["stock_pos"] = pd.to_numeric(frame["stock_pos"], errors="coerce").fillna(-1).astype("int32")
        sample_frames.append(frame)
    if not loaded_shards:
        raise ValueError(f"qdp sharded memmap has no completed shards: {manifest_path}")

    global_dates = sorted(all_dates)
    global_timestamps = [pd.Timestamp(item).normalize() for item in global_dates]
    next_open_extra_day = 1 if str(manifest.get("execution_mode", "next_open") or "next_open").strip().lower() == "next_open" else 0
    label_forward_offset = int(horizon) + int(next_open_extra_day)
    eligible_dates = _date_role_boundaries(
        global_timestamps,
        train_start_year=int(train_start_year),
        train_end_year=int(train_end_year),
        validation_year=int(validation_year),
        test_year=int(test_year),
        purge_trading_days=int(label_forward_offset),
    )
    role_by_date = {
        pd.Timestamp(dt).strftime("%Y-%m-%d"): role
        for role, dates in eligible_dates.items()
        for dt in dates
    }
    sample_index = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    if not sample_index.empty:
        sample_index["role"] = sample_index["date"].map(role_by_date).fillna("")
        sample_index = sample_index.loc[sample_index["role"].astype(str).ne("")].copy()
    if sample_index.empty:
        raise ValueError(f"qdp sharded memmap has no samples for requested role years: {manifest_path}")

    global_date_pos = {date: pos for pos, date in enumerate(global_dates)}
    date_positions = sample_index["date"].map(global_date_pos).fillna(-1).astype("int64").to_numpy()
    sequence_start_pos = np.maximum(date_positions - int(lookback_days) + 1, 0)
    label_end_pos = np.minimum(date_positions + int(label_forward_offset), len(global_dates) - 1)
    sample_index["sequence_start_pos"] = sequence_start_pos.astype("int32")
    sample_index["label_end_pos"] = label_end_pos.astype("int32")
    sample_index["sequence_start_date"] = [global_dates[int(pos)] if int(pos) >= 0 else "" for pos in sequence_start_pos]
    sample_index["label_end_date"] = [global_dates[int(pos)] if int(pos) >= 0 else "" for pos in label_end_pos]
    sample_index["lookback_days"] = int(lookback_days)
    sample_index["history_bucket"] = sample_index["history_valid_ratio"].map(_history_bucket) if "history_valid_ratio" in sample_index.columns else "low"
    sample_index = _cap_samples_by_role(
        sample_index,
        max_samples_per_role=int(max_samples_per_role),
        max_samples_per_date_per_role=int(max_samples_per_date_per_role),
    )
    train_stocks = set(sample_index.loc[sample_index["role"].astype(str).eq("train"), "stock"].astype(str))
    sample_index["stock_seen_in_train"] = sample_index["stock"].astype(str).map(lambda stock: stock in train_stocks)
    sample_index.loc[sample_index["role"].astype(str).eq("train"), "stock_seen_in_train"] = True

    static_schema = dict(manifest.get("static_context_schema", {}) or {"enabled": False})
    static_context_ids: np.ndarray | None = None
    if bool(static_schema.get("enabled", False)):
        id_columns = [str(item) for item in static_schema.get("id_columns", []) or static_context_id_columns(static_schema.get("fields") or None)]
        static_context_ids = (
            sample_index.reindex(columns=id_columns, fill_value=0)
            .fillna(0)
            .astype("int64")
            .to_numpy(dtype=np.int64)
        )

    normalization = {
        "fit_role": "train_only",
        "method": "zscore",
        "feature_count": int(len(feature_columns)),
        "source": "qdp_sharded_memmap_training_view",
    }
    feature_mean = np.zeros((len(feature_columns),), dtype=np.float32)
    feature_std = np.ones((len(feature_columns),), dtype=np.float32)
    view_manifest_base = {key: value for key, value in dict(manifest).items() if key != "shards"}
    training_manifest = {
        **view_manifest_base,
        "dataset_mode": "memmap",
        "artifact_reused": True,
        "qdp_sharded_memmap_reused": True,
        "manifest_json": str(manifest_path.resolve()),
        "source_qdp_sharded_manifest_json": str(manifest_path.resolve()),
        "qdp_completed_shard_count": int(len(loaded_shards)),
        "role_years": {
            "train_start_year": int(train_start_year),
            "train_end_year": int(train_end_year),
            "validation_year": int(validation_year),
            "test_year": int(test_year),
        },
        "role_purge_trading_days": int(label_forward_offset),
        "next_open_label_extra_trading_day": int(next_open_extra_day),
        "sample_count": int(len(sample_index)),
        "sample_count_by_role": {
            role: int((sample_index["role"].astype(str) == role).sum())
            for role in ("train", "validation", "test")
        },
        "feature_columns": list(feature_columns),
        "feature_count": int(len(feature_columns)),
        "horizon": int(horizon),
        "forecast_horizon": int(horizon),
        "cumulative_horizons": [int(item) for item in cumulative_horizons],
        "risk_horizons": [int(item) for item in cumulative_horizons],
        "normalization": normalization,
        "static_context_schema": static_schema,
        "symbol_vocab_fingerprint": str(static_schema.get("symbol_vocab_fingerprint", "")),
        "industry_vocab_fingerprint": str(static_schema.get("industry_vocab_fingerprint", "")),
        "board_vocab_fingerprint": str(static_schema.get("board_vocab_fingerprint", "")),
    }
    dataset = ForecastShardedMemmapDataset(
        root=root,
        manifest_path=manifest_path,
        manifest=training_manifest,
        shards=loaded_shards,
        sample_index=sample_index,
        feature_columns=feature_columns,
        normalization_manifest=normalization,
        feature_mean=feature_mean,
        feature_std=feature_std,
        date_values=global_dates,
        static_context_ids=static_context_ids,
    )

    train_indices = dataset.role_indices("train")
    norm_cache_path = _qdp_sharded_normalization_cache_path(
        manifest_path,
        feature_columns=feature_columns,
        train_start_year=int(train_start_year),
        train_end_year=int(train_end_year),
    )
    sample_window_norm_max_samples = 4096
    use_sample_window_norm = (
        (int(max_samples_per_role) > 0 or int(max_samples_per_date_per_role) > 0)
        and int(len(train_indices)) <= sample_window_norm_max_samples
    )
    use_cache = not use_sample_window_norm
    if use_cache and norm_cache_path.exists():
        cached = json.loads(norm_cache_path.read_text(encoding="utf-8"))
        feature_mean = np.asarray(cached.get("feature_mean", []), dtype=np.float32)
        feature_std = np.asarray(cached.get("feature_std", []), dtype=np.float32)
    elif use_sample_window_norm:
        feature_mean, feature_std = _fit_qdp_sharded_sample_window_normalization(dataset, train_indices)
    else:
        feature_mean, feature_std = _fit_qdp_sharded_panel_normalization(
            shards=loaded_shards,
            feature_count=len(feature_columns),
            train_start_year=int(train_start_year),
            train_end_year=int(train_end_year),
        )
        write_json(
            norm_cache_path,
            {
                "fit_role": "train_only",
                "method": "zscore",
                "fit_scope": "train_year_panel",
                "feature_count": int(len(feature_columns)),
                "feature_mean": feature_mean.astype(float).tolist(),
                "feature_std": feature_std.astype(float).tolist(),
                "source": "qdp_sharded_memmap_training_view",
            },
        )
    if feature_mean.size != len(feature_columns) or feature_std.size != len(feature_columns):
        raise ValueError(f"qdp sharded memmap normalization does not match feature columns: {manifest_path}")
    normalization = {
        **normalization,
        "feature_mean": feature_mean.astype(float).tolist(),
        "feature_std": feature_std.astype(float).tolist(),
        "normalization_cache_json": str(norm_cache_path.resolve()) if use_cache else "",
        "normalization_fit_scope": "train_sample_windows" if use_sample_window_norm else "train_year_panel",
        "sample_window_normalization_max_samples": int(sample_window_norm_max_samples),
    }
    dataset.feature_mean = feature_mean.astype(np.float32, copy=False)
    dataset.feature_std = feature_std.astype(np.float32, copy=False)
    dataset.normalization_manifest = normalization
    dataset.manifest = {**dataset.manifest, "normalization": normalization}
    return dataset


def _contiguous_slice(values: np.ndarray) -> slice | np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if arr.size == 0:
        return arr
    if arr.size == 1:
        return slice(int(arr[0]), int(arr[0]) + 1)
    if np.all(np.diff(arr) == 1):
        return slice(int(arr[0]), int(arr[-1]) + 1)
    return arr


def build_qdp_training_pack(
    source_manifest_json: str | Path,
    *,
    output_root: str | Path | None = None,
    tag: str = "",
    train_start_year: int = 2012,
    train_end_year: int = 2023,
    validation_year: int = 2024,
    test_year: int = 2025,
    max_samples_per_role: int = 0,
    max_samples_per_date_per_role: int = 0,
    feature_dtype: str = "float16",
    stock_chunk_size: int = 64,
    resume: bool = True,
) -> dict[str, Any]:
    source_path = Path(source_manifest_json)
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    if str(source_manifest.get("artifact_type", "")) != "qdp_sharded_memmap":
        raise ValueError(f"qdp_training_pack_source_must_be_qdp_sharded_memmap: {source_path}")
    dtype = str(feature_dtype or "float16").strip().lower()
    if dtype not in {"float16", "float32"}:
        raise ValueError("--feature-dtype must be float16 or float32.")
    default_tag = (
        f"{source_path.parent.name}_training_pack_"
        f"{int(train_start_year)}_{int(train_end_year)}_{int(validation_year)}_{int(test_year)}_{dtype}"
    )
    safe_tag = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(tag or default_tag)).strip("_")
    if output_root is None:
        memmap_root = source_path.parent.parent.parent if source_path.parent.parent.name == "sharded" else source_path.parent.parent
        root = memmap_root / "training_pack" / safe_tag
    else:
        root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "qdp_training_pack_manifest.json"
    progress_path = root / "qdp_training_pack_progress.json"
    if bool(resume) and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(existing.get("status", "")) == "completed":
            return existing

    _qdp_training_pack_progress(progress_path, "loading_source_view", source_manifest_json=str(source_path.resolve()))
    dataset = _load_qdp_sharded_forecast_memmap_dataset(
        source_path,
        source_manifest,
        train_start_year=int(train_start_year),
        train_end_year=int(train_end_year),
        validation_year=int(validation_year),
        test_year=int(test_year),
        max_samples_per_role=int(max_samples_per_role),
        max_samples_per_date_per_role=int(max_samples_per_date_per_role),
    )
    sample_index = dataset.sample_index.reset_index(drop=True).copy()
    sample_index["_sample_pos"] = np.arange(len(sample_index), dtype=np.int64)
    feature_columns = list(dataset.feature_columns)
    feature_count = int(len(feature_columns))
    date_values = [str(item) for item in dataset.date_values.tolist()]
    stock_values = sorted({str(item).strip().upper() for item in sample_index["stock"].astype(str).tolist()})
    date_pos = {date: idx for idx, date in enumerate(date_values)}
    stock_pos = {stock: idx for idx, stock in enumerate(stock_values)}
    sample_index["global_date_pos"] = sample_index["date"].astype(str).map(date_pos).fillna(-1).astype("int32")
    sample_index["global_stock_pos"] = sample_index["stock"].astype(str).str.strip().str.upper().map(stock_pos).fillna(-1).astype("int32")
    if (sample_index["global_date_pos"] < 0).any() or (sample_index["global_stock_pos"] < 0).any():
        raise ValueError("qdp_training_pack_sample_index_has_unmapped_date_or_stock")

    sample_index_path = root / "sample_index.parquet"
    sample_index.to_parquet(sample_index_path, index=False)
    sample_count_by_role = {
        role: int((sample_index["role"].astype(str) == role).sum())
        for role in ("train", "validation", "test")
    }
    _qdp_training_pack_progress(
        progress_path,
        "source_view_loaded",
        sample_count=int(len(sample_index)),
        sample_count_by_role=sample_count_by_role,
        stock_count=int(len(stock_values)),
        date_count=int(len(date_values)),
        feature_count=int(feature_count),
    )

    feature_panel_path = root / f"feature_panel_stock_date_feature.{dtype}.dat"
    feature_shape = (int(len(stock_values)), int(len(date_values)), int(feature_count))
    feature_panel = np.memmap(feature_panel_path, dtype=dtype, mode="w+", shape=feature_shape)
    feature_panel[:] = 0
    feature_mean = np.asarray(dataset.normalization_manifest.get("feature_mean", []), dtype=np.float32).reshape(1, 1, -1)
    feature_std = np.asarray(dataset.normalization_manifest.get("feature_std", []), dtype=np.float32).reshape(1, 1, -1)
    if feature_mean.shape[-1] != feature_count or feature_std.shape[-1] != feature_count:
        raise ValueError("qdp_training_pack_normalization_feature_count_mismatch")
    feature_std = np.where(np.isfinite(feature_std) & (np.abs(feature_std) > 1.0e-8), feature_std, 1.0).astype(np.float32)
    written_shards = 0
    completed_shards = [shard for shard in dataset.shards if str(shard.get("status", "")) == "completed"]
    for shard_idx, shard in enumerate(completed_shards):
        label_manifest = dict(shard.get("label_manifest", {}) or {})
        shard_dates = [str(pd.Timestamp(item).strftime("%Y-%m-%d")) for item in list(label_manifest.get("date_values", []) or [])]
        shard_stocks = [str(item).strip().upper() for item in list(label_manifest.get("stock_values", []) or [])]
        local_date_positions = np.asarray([date_pos[item] for item in shard_dates if item in date_pos], dtype=np.int64)
        if local_date_positions.size == 0:
            continue
        local_stock_pairs = [(local_idx, stock_pos[stock]) for local_idx, stock in enumerate(shard_stocks) if stock in stock_pos]
        if not local_stock_pairs:
            continue
        feature_store_shape = tuple(int(item) for item in list(shard.get("feature_store_shape", []) or []))
        store = np.memmap(Path(str(shard.get("feature_store_path", "") or "")), dtype="float32", mode="r", shape=feature_store_shape)
        local_count = min(int(store.shape[0]), int(local_date_positions.size))
        target_date_idx = local_date_positions[:local_count]
        for start in range(0, len(local_stock_pairs), max(int(stock_chunk_size), 1)):
            pairs = local_stock_pairs[start : start + max(int(stock_chunk_size), 1)]
            local_stock_idx = np.asarray([item[0] for item in pairs], dtype=np.int64)
            target_stock_idx = np.asarray([item[1] for item in pairs], dtype=np.int64)
            values = np.asarray(store[:local_count, local_stock_idx, :], dtype=np.float32)
            values = np.nan_to_num((values - feature_mean) / feature_std, nan=0.0, posinf=0.0, neginf=0.0)
            values = np.transpose(values, (1, 0, 2)).astype(dtype, copy=False)
            stock_indexer = _contiguous_slice(target_stock_idx)
            date_indexer = _contiguous_slice(target_date_idx)
            if isinstance(stock_indexer, slice) and isinstance(date_indexer, slice):
                feature_panel[stock_indexer, date_indexer, :] = values
            else:
                for pos, global_stock_idx in enumerate(target_stock_idx.tolist()):
                    feature_panel[int(global_stock_idx), target_date_idx, :] = values[pos]
        written_shards += 1
        if written_shards == 1 or written_shards % 10 == 0 or written_shards == len(completed_shards):
            feature_panel.flush()
            _qdp_training_pack_progress(
                progress_path,
                "feature_panel_writing",
                completed_shards=int(written_shards),
                total_shards=int(len(completed_shards)),
                feature_panel_path=str(feature_panel_path.resolve()),
            )
    feature_panel.flush()

    static_context_meta: dict[str, Any] = {}
    if dataset.static_context_ids is not None:
        static_path = root / "static_context_ids.int32.dat"
        static_values = np.asarray(dataset.static_context_ids, dtype=np.int32)
        static_store = np.memmap(static_path, dtype="int32", mode="w+", shape=static_values.shape)
        static_store[:] = static_values
        static_store.flush()
        static_context_meta = {
            "path": str(static_path.resolve()),
            "dtype": "int32",
            "shape": [int(item) for item in static_values.shape],
        }

    horizon = int(dataset.manifest.get("horizon", PATH20_HORIZON) or PATH20_HORIZON)
    cumulative_horizons = dataset.cumulative_horizons
    all_label_specs = _label_array_specs(horizon, cumulative_horizons)
    common_label_arrays = _common_sharded_label_arrays(completed_shards)
    missing_required = sorted(set(_REQUIRED_QDP_LABEL_ARRAYS) - set(common_label_arrays))
    if missing_required:
        raise ValueError(f"qdp_training_pack_source_missing_required_label_arrays: {missing_required}")
    label_specs = {
        name: shape
        for name, shape in all_label_specs.items()
        if name in common_label_arrays
    }
    label_dir = root / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    label_arrays_manifest: dict[str, dict[str, Any]] = {}
    for label_name, tail_shape in label_specs.items():
        shape = (int(len(sample_index)), *tuple(int(item) for item in tail_shape))
        path = label_dir / f"{label_name}.float32.dat"
        out = np.memmap(path, dtype="float32", mode="w+", shape=shape)
        for shard_idx, group in sample_index.groupby("_shard_idx", sort=False):
            store = dataset._open_label_store_for_shard(int(shard_idx), label_name)
            date_idx = group["date_pos"].to_numpy(dtype=np.int64, copy=False)
            local_stock_idx = group["stock_pos"].to_numpy(dtype=np.int64, copy=False)
            sample_pos = group["_sample_pos"].to_numpy(dtype=np.int64, copy=False)
            values = np.asarray(store[date_idx, local_stock_idx], dtype=np.float32).reshape((len(group), *tail_shape))
            out[sample_pos] = values
        out.flush()
        label_arrays_manifest[label_name] = {
            "path": str(path.resolve()),
            "dtype": "float32",
            "shape": [int(item) for item in shape],
        }
        _qdp_training_pack_progress(
            progress_path,
            "label_array_written",
            label_name=str(label_name),
            sample_count=int(len(sample_index)),
        )

    source_base = {key: value for key, value in dict(source_manifest).items() if key != "shards"}
    manifest = {
        **source_base,
        "artifact_type": "qdp_training_pack_v1",
        "status": "completed",
        "created_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "dataset_mode": "memmap",
        "manifest_json": str(manifest_path.resolve()),
        "source_qdp_sharded_manifest_json": str(source_path.resolve()),
        "source_artifact_type": "qdp_sharded_memmap",
        "role_years": {
            "train_start_year": int(train_start_year),
            "train_end_year": int(train_end_year),
            "validation_year": int(validation_year),
            "test_year": int(test_year),
        },
        "max_samples_per_role": int(max_samples_per_role),
        "max_samples_per_date_per_role": int(max_samples_per_date_per_role),
        "sample_index_path": str(sample_index_path.resolve()),
        "sample_count": int(len(sample_index)),
        "sample_count_by_role": sample_count_by_role,
        "stock_values": stock_values,
        "date_values": date_values,
        "feature_columns": feature_columns,
        "feature_count": int(feature_count),
        "feature_panel_path": str(feature_panel_path.resolve()),
        "feature_panel_shape": [int(item) for item in feature_shape],
        "feature_dtype": dtype,
        "features_are_normalized": True,
        "normalization": dict(dataset.normalization_manifest),
        "label_arrays": label_arrays_manifest,
        "label_array_names": list(label_arrays_manifest.keys()),
        "label_schema_name": str(source_manifest.get("label_schema_name", source_manifest.get("label_metadata", {}).get("label_schema_name", "")) or ""),
        "label_schema_version": int(source_manifest.get("label_schema_version", source_manifest.get("label_metadata", {}).get("label_schema_version", 1)) or 1),
        "static_context_schema": dict(dataset.manifest.get("static_context_schema", {}) or {"enabled": False}),
        "static_context_ids": static_context_meta,
        "lookback_days": int(dataset.lookback_days),
        "horizon": int(horizon),
        "forecast_horizon": int(horizon),
        "cumulative_horizons": [int(item) for item in cumulative_horizons],
        "risk_horizons": [int(item) for item in cumulative_horizons],
        "training_pack_contract": {
            "feature_layout": "stock_date_feature",
            "feature_dtype": dtype,
            "feature_normalization": "train_only_zscore_nan_to_zero",
            "label_layout": "sample_major",
            "window_materialization": "runtime_sliding_window_view",
        },
    }
    write_json(manifest_path, _json_ready(manifest))
    _qdp_training_pack_progress(
        progress_path,
        "completed",
        manifest_json=str(manifest_path.resolve()),
        sample_count=int(len(sample_index)),
        feature_panel_shape=[int(item) for item in feature_shape],
    )
    return manifest


def _memmap_file_matches(path: Path, *, shape: tuple[int, ...], dtype: str | np.dtype) -> bool:
    if not path.exists():
        return False
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    return int(path.stat().st_size) == expected_bytes


def build_qdp_training_pack_date_major_layout(
    training_pack_manifest_json: str | Path,
    *,
    feature_dtype: str = "",
    stock_chunk_size: int = 64,
    resume: bool = True,
) -> dict[str, Any]:
    manifest_path = Path(training_pack_manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("artifact_type", "")) != "qdp_training_pack_v1":
        raise ValueError(f"qdp_regime_training_pack_source_must_be_qdp_training_pack_v1: {manifest_path}")
    root = manifest_path.parent
    source_feature_path = Path(str(manifest.get("feature_panel_path", "") or ""))
    if not source_feature_path.is_absolute():
        source_feature_path = root / source_feature_path
    source_shape = tuple(int(item) for item in list(manifest.get("feature_panel_shape", []) or []))
    if len(source_shape) != 3:
        raise ValueError(f"qdp training pack has invalid feature_panel_shape: {manifest_path}")
    source_dtype = str(manifest.get("feature_dtype", "float16") or "float16").strip().lower()
    _validate_memmap_file(
        source_feature_path,
        shape=source_shape,
        label="qdp_training_pack_feature_panel",
        dtype=source_dtype,
    )

    dtype = str(feature_dtype or source_dtype or "float16").strip().lower()
    if dtype not in {"float16", "float32"}:
        raise ValueError("--feature-dtype must be float16 or float32.")
    stock_count, date_count, feature_count = (int(source_shape[0]), int(source_shape[1]), int(source_shape[2]))
    target_shape = (date_count, stock_count, feature_count)
    target_path = root / f"feature_panel_date_stock_feature.{dtype}.dat"
    cross_section_index_path = root / "cross_section_index.parquet"
    companion_manifest_path = root / "qdp_regime_training_pack_manifest.json"
    progress_path = root / "qdp_regime_training_pack_progress.json"
    chunk = max(int(stock_chunk_size), 1)

    existing_ready = (
        bool(resume)
        and _memmap_file_matches(target_path, shape=target_shape, dtype=dtype)
    )
    if not existing_ready:
        _qdp_training_pack_progress(
            progress_path,
            "date_major_feature_panel_writing",
            source_feature_panel_path=str(source_feature_path.resolve()),
            target_feature_panel_path=str(target_path.resolve()),
            feature_dtype=dtype,
            source_shape=[int(item) for item in source_shape],
            target_shape=[int(item) for item in target_shape],
            stock_chunk_size=int(chunk),
            completed_stocks=0,
            total_stocks=int(stock_count),
        )
        source = np.memmap(source_feature_path, dtype=source_dtype, mode="r", shape=source_shape)
        target = np.memmap(target_path, dtype=dtype, mode="w+", shape=target_shape)
        for stock_start in range(0, stock_count, chunk):
            stock_end = min(stock_start + chunk, stock_count)
            values = np.asarray(source[stock_start:stock_end, :, :], dtype=dtype)
            target[:, stock_start:stock_end, :] = np.transpose(values, (1, 0, 2))
            if stock_start == 0 or stock_end == stock_count or (stock_end // chunk) % 10 == 0:
                target.flush()
                _qdp_training_pack_progress(
                    progress_path,
                    "date_major_feature_panel_writing",
                    target_feature_panel_path=str(target_path.resolve()),
                    completed_stocks=int(stock_end),
                    total_stocks=int(stock_count),
                    completion_ratio=float(stock_end / max(stock_count, 1)),
                )
        target.flush()

    cross_section_rows = 0
    sample_index_path = Path(str(manifest.get("sample_index_path", "") or ""))
    if not sample_index_path.is_absolute():
        sample_index_path = root / sample_index_path
    if sample_index_path.exists():
        columns = ["role", "date", "global_date_pos", "global_stock_pos"]
        sample_index = pd.read_parquet(sample_index_path, columns=columns)
        sample_index = sample_index.copy()
        sample_index["_sample_pos"] = np.arange(len(sample_index), dtype=np.int64)
        cross_section_index = (
            sample_index.groupby(["role", "date", "global_date_pos"], sort=True)
            .agg(
                sample_count=("global_stock_pos", "count"),
                stock_count=("global_stock_pos", "nunique"),
                first_sample_pos=("_sample_pos", "min"),
                last_sample_pos=("_sample_pos", "max"),
            )
            .reset_index()
        )
        cross_section_index["role"] = cross_section_index["role"].astype(str)
        cross_section_index["date"] = pd.to_datetime(cross_section_index["date"]).dt.strftime("%Y-%m-%d")
        cross_section_index["global_date_pos"] = pd.to_numeric(
            cross_section_index["global_date_pos"], errors="coerce"
        ).fillna(-1).astype("int32")
        cross_section_index["sample_count"] = cross_section_index["sample_count"].astype("int32")
        cross_section_index["stock_count"] = cross_section_index["stock_count"].astype("int32")
        cross_section_index["first_sample_pos"] = cross_section_index["first_sample_pos"].astype("int64")
        cross_section_index["last_sample_pos"] = cross_section_index["last_sample_pos"].astype("int64")
        cross_section_index.to_parquet(cross_section_index_path, index=False)
        cross_section_rows = int(len(cross_section_index))

    training_pack_contract = dict(manifest.get("training_pack_contract", {}) or {})
    training_pack_contract.update(
        {
            "date_batch_feature_layout": "date_stock_feature",
            "date_batch_feature_dtype": dtype,
            "date_batch_window_materialization": "precomputed_date_major_slice",
            "cross_section_index": "role_date_group_counts",
        }
    )
    regime_layout = {
        "enabled": True,
        "feature_layout": "date_stock_feature",
        "feature_panel_path": str(target_path.resolve()),
        "feature_panel_shape": [int(item) for item in target_shape],
        "feature_dtype": dtype,
        "source_feature_layout": "stock_date_feature",
        "source_feature_panel_path": str(source_feature_path.resolve()),
        "source_feature_panel_shape": [int(item) for item in source_shape],
        "cross_section_index_path": str(cross_section_index_path.resolve()) if cross_section_index_path.exists() else "",
        "cross_section_index_rows": int(cross_section_rows),
        "window_materialization": "date_major_runtime_slice",
    }
    manifest.update(
        {
            "date_major_feature_panel_path": str(target_path.resolve()),
            "date_major_feature_panel_shape": [int(item) for item in target_shape],
            "date_major_feature_dtype": dtype,
            "date_major_feature_layout": "date_stock_feature",
            "date_major_feature_panel": {
                "path": str(target_path.resolve()),
                "shape": [int(item) for item in target_shape],
                "dtype": dtype,
                "layout": "date_stock_feature",
            },
            "cross_section_index_path": str(cross_section_index_path.resolve()) if cross_section_index_path.exists() else "",
            "cross_section_index_rows": int(cross_section_rows),
            "regime_auxiliary_layout": regime_layout,
            "regime_training_pack_manifest_json": str(companion_manifest_path.resolve()),
            "training_pack_contract": training_pack_contract,
            "updated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        }
    )
    write_json(manifest_path, _json_ready(manifest))
    companion_manifest = {
        "artifact_type": "qdp_regime_training_pack_v1",
        "status": "completed",
        "created_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "source_training_pack_manifest_json": str(manifest_path.resolve()),
        "source_artifact_type": "qdp_training_pack_v1",
        "sample_count": int(manifest.get("sample_count", 0) or 0),
        "sample_count_by_role": dict(manifest.get("sample_count_by_role", {}) or {}),
        "stock_count": int(stock_count),
        "date_count": int(date_count),
        "feature_count": int(feature_count),
        "lookback_days": int(manifest.get("lookback_days", 0) or 0),
        "horizon": int(manifest.get("horizon", manifest.get("forecast_horizon", 0)) or 0),
        "cumulative_horizons": [int(item) for item in list(manifest.get("cumulative_horizons", []) or [])],
        "date_major_feature_panel": regime_layout,
        "training_pack_contract": training_pack_contract,
    }
    write_json(companion_manifest_path, _json_ready(companion_manifest))
    _qdp_training_pack_progress(
        progress_path,
        "completed",
        source_training_pack_manifest_json=str(manifest_path.resolve()),
        regime_training_pack_manifest_json=str(companion_manifest_path.resolve()),
        target_feature_panel_path=str(target_path.resolve()),
        cross_section_index_path=str(cross_section_index_path.resolve()) if cross_section_index_path.exists() else "",
        target_shape=[int(item) for item in target_shape],
        feature_dtype=dtype,
    )
    return manifest


def load_forecast_memmap_dataset(
    manifest_json: str | Path,
    *,
    train_start_year: int = 2019,
    train_end_year: int = 2022,
    validation_year: int = 2023,
    test_year: int = 2024,
    max_samples_per_role: int = 0,
    max_samples_per_date_per_role: int = 0,
) -> ForecastMemmapDataset | ForecastShardedMemmapDataset | ForecastTrainingPackDataset:
    manifest_path = Path(manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("artifact_type", "")) == "qdp_training_pack_v1":
        return _load_training_pack_forecast_memmap_dataset(
            manifest_path,
            manifest,
            max_samples_per_role=int(max_samples_per_role),
            max_samples_per_date_per_role=int(max_samples_per_date_per_role),
        )
    if str(manifest.get("artifact_type", "")) == "qdp_sharded_memmap":
        return _load_qdp_sharded_forecast_memmap_dataset(
            manifest_path,
            manifest,
            train_start_year=int(train_start_year),
            train_end_year=int(train_end_year),
            validation_year=int(validation_year),
            test_year=int(test_year),
            max_samples_per_role=int(max_samples_per_role),
            max_samples_per_date_per_role=int(max_samples_per_date_per_role),
        )
    if str(manifest.get("dataset_mode", "")) != "memmap":
        raise ValueError(f"forecast memmap manifest required dataset_mode=memmap: {manifest_path}")

    root = manifest_path.parent
    feature_store_shape = tuple(int(item) for item in manifest.get("feature_store_shape", []))
    if len(feature_store_shape) != 3:
        raise ValueError(f"forecast memmap manifest has invalid feature_store_shape: {manifest_path}")
    sample_index_path = _resolve_manifest_path(manifest.get("sample_index_csv"), manifest_path=manifest_path)
    if not sample_index_path.exists():
        raise ValueError(f"forecast memmap manifest missing sample_index_csv: {sample_index_path}")
    sample_index = pd.read_csv(sample_index_path)
    row_count = int(manifest.get("sample_count", 0) or len(sample_index))
    if row_count != len(sample_index):
        raise ValueError(
            f"forecast memmap manifest sample_count does not match sample_index_csv: "
            f"sample_count={row_count}, sample_index_rows={len(sample_index)} at {manifest_path}"
        )
    horizon = int(manifest.get("horizon", PATH20_HORIZON) or PATH20_HORIZON)
    cumulative_horizons = normalize_cumulative_horizons(
        manifest.get("cumulative_horizons", PATH20_CUMULATIVE_HORIZONS),
        horizon=horizon,
    )
    cumulative_count = len(cumulative_horizons)
    normalization = dict(manifest.get("normalization", {}) or {})
    feature_columns = [str(item) for item in manifest.get("feature_columns", [])]
    feature_mean = np.asarray(normalization.get("feature_mean", []), dtype=np.float32)
    feature_std = np.asarray(normalization.get("feature_std", []), dtype=np.float32)
    if len(feature_columns) != feature_mean.size or len(feature_columns) != feature_std.size:
        raise ValueError(f"forecast memmap manifest normalization does not match feature columns: {manifest_path}")
    if int(feature_store_shape[2]) != len(feature_columns):
        raise ValueError(
            f"forecast memmap manifest feature_store_shape does not match feature columns: "
            f"feature_store_shape={feature_store_shape}, feature_columns={len(feature_columns)} at {manifest_path}"
        )

    feature_store_path = _resolve_manifest_path(manifest.get("feature_store_path"), manifest_path=manifest_path)
    _validate_memmap_file(feature_store_path, shape=feature_store_shape, label="feature_store_path")
    _validate_memmap_file(root / "forecast_y_daily_excess.dat", shape=(row_count, horizon), label="forecast_y_daily_excess")
    _validate_memmap_file(root / "forecast_y_cum_excess.dat", shape=(row_count, cumulative_count), label="forecast_y_cum_excess")
    _validate_memmap_file(root / "forecast_y_rank_by_horizon.dat", shape=(row_count, cumulative_count), label="forecast_y_rank_by_horizon")
    if (root / "forecast_y_drawdown_by_horizon.dat").exists():
        _validate_memmap_file(root / "forecast_y_drawdown_by_horizon.dat", shape=(row_count, cumulative_count), label="forecast_y_drawdown_by_horizon")
        _validate_memmap_file(root / "forecast_y_worst_by_horizon.dat", shape=(row_count, cumulative_count), label="forecast_y_worst_by_horizon")
        _validate_memmap_file(root / "forecast_y_upside_by_horizon.dat", shape=(row_count, cumulative_count), label="forecast_y_upside_by_horizon")
    _validate_memmap_file(root / "forecast_y_rank_20d.dat", shape=(row_count,), label="forecast_y_rank_20d")
    _validate_memmap_file(root / "forecast_y_max_drawdown_20d.dat", shape=(row_count,), label="forecast_y_max_drawdown_20d")
    _validate_memmap_file(root / "forecast_y_worst_1d_20d.dat", shape=(row_count,), label="forecast_y_worst_1d_20d")
    _validate_memmap_file(root / "forecast_y_upside_20d.dat", shape=(row_count,), label="forecast_y_upside_20d")
    static_context_ids: np.memmap | None = None
    static_schema = dict(manifest.get("static_context_schema", {}) or {})
    if bool(static_schema.get("enabled", False)):
        static_fields = normalize_static_context_fields(static_schema.get("fields") or None)
        static_shape = tuple(int(item) for item in manifest.get("static_context_shape", []))
        if static_shape != (row_count, len(static_fields)):
            raise ValueError(f"forecast memmap manifest has invalid static_context_shape: {manifest_path}")
        static_context_path = _resolve_manifest_path(manifest.get("static_context_path"), manifest_path=manifest_path)
        if not static_context_path.exists():
            raise ValueError(f"forecast memmap manifest missing static_context_path: {static_context_path}")
        expected_bytes = int(np.prod(static_shape, dtype=np.int64)) * np.dtype("int64").itemsize
        actual_bytes = int(static_context_path.stat().st_size)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"forecast memmap manifest has invalid static_context_path size: "
                f"expected {expected_bytes} bytes for shape {static_shape}, got {actual_bytes} bytes at {static_context_path}"
            )
        static_context_ids = np.memmap(static_context_path, dtype="int64", mode="r", shape=static_shape)

    has_dynamic_risk = (root / "forecast_y_drawdown_by_horizon.dat").exists()
    y_drawdown_by_horizon = (
        np.memmap(root / "forecast_y_drawdown_by_horizon.dat", dtype="float32", mode="r", shape=(row_count, cumulative_count))
        if has_dynamic_risk
        else np.repeat(
            np.memmap(root / "forecast_y_max_drawdown_20d.dat", dtype="float32", mode="r", shape=(row_count,)).reshape(-1, 1),
            cumulative_count,
            axis=1,
        ).astype(np.float32)
    )
    y_worst_by_horizon = (
        np.memmap(root / "forecast_y_worst_by_horizon.dat", dtype="float32", mode="r", shape=(row_count, cumulative_count))
        if has_dynamic_risk
        else np.repeat(
            np.memmap(root / "forecast_y_worst_1d_20d.dat", dtype="float32", mode="r", shape=(row_count,)).reshape(-1, 1),
            cumulative_count,
            axis=1,
        ).astype(np.float32)
    )
    y_upside_by_horizon = (
        np.memmap(root / "forecast_y_upside_by_horizon.dat", dtype="float32", mode="r", shape=(row_count, cumulative_count))
        if has_dynamic_risk
        else np.repeat(
            np.memmap(root / "forecast_y_upside_20d.dat", dtype="float32", mode="r", shape=(row_count,)).reshape(-1, 1),
            cumulative_count,
            axis=1,
        ).astype(np.float32)
    )
    manifest = {
        **manifest,
        "artifact_reused": True,
        "manifest_json": str(manifest_path.resolve()),
        "cumulative_horizons": [int(item) for item in cumulative_horizons],
        "risk_horizons": [int(item) for item in cumulative_horizons],
        "forecast_horizon": int(horizon),
    }
    date_values = [str(item) for item in manifest.get("date_values", [])]
    stock_values = [str(item) for item in manifest.get("stock_values", [])]
    if not date_values and "date" in sample_index.columns:
        date_values = [pd.Timestamp(item).strftime("%Y-%m-%d") for item in pd.to_datetime(sample_index["date"]).tolist()]
    if not stock_values and "stock" in sample_index.columns:
        stock_values = [str(item).strip().upper() for item in sample_index["stock"].dropna().astype(str).tolist() if str(item).strip()]
    return ForecastMemmapDataset(
        root=root,
        feature_store_path=feature_store_path,
        feature_store_shape=feature_store_shape,
        sample_index=sample_index,
        y_daily_excess=np.memmap(root / "forecast_y_daily_excess.dat", dtype="float32", mode="r", shape=(row_count, horizon)),
        y_cum_excess=np.memmap(root / "forecast_y_cum_excess.dat", dtype="float32", mode="r", shape=(row_count, cumulative_count)),
        y_rank_by_horizon=np.memmap(root / "forecast_y_rank_by_horizon.dat", dtype="float32", mode="r", shape=(row_count, cumulative_count)),
        y_rank_20d=np.memmap(root / "forecast_y_rank_20d.dat", dtype="float32", mode="r", shape=(row_count,)),
        y_drawdown_by_horizon=y_drawdown_by_horizon,
        y_worst_by_horizon=y_worst_by_horizon,
        y_upside_by_horizon=y_upside_by_horizon,
        y_max_drawdown_20d=np.memmap(root / "forecast_y_max_drawdown_20d.dat", dtype="float32", mode="r", shape=(row_count,)),
        y_worst_1d_20d=np.memmap(root / "forecast_y_worst_1d_20d.dat", dtype="float32", mode="r", shape=(row_count,)),
        y_upside_20d=np.memmap(root / "forecast_y_upside_20d.dat", dtype="float32", mode="r", shape=(row_count,)),
        feature_columns=feature_columns,
        normalization_manifest=normalization,
        manifest=manifest,
        feature_mean=feature_mean,
        feature_std=feature_std,
        date_values=np.array(date_values, dtype=object),
        stock_values=np.array(stock_values, dtype=object),
        static_context_ids=static_context_ids,
    )


def _summary_stats(values: list[float]) -> dict[str, float]:
    finite = np.asarray([float(item) for item in values if np.isfinite(float(item))], dtype=float)
    if finite.size == 0:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def _write_memmap_build_progress(root: Path, stage: str, **payload: Any) -> None:
    progress = {
        "stage": str(stage),
        **{str(key): _json_ready(value) for key, value in payload.items()},
    }
    write_json(Path(root) / "forecast_memmap_build_progress.json", progress)


def _fit_memmap_train_normalization(
    *,
    feature_store_path: Path,
    feature_shape: tuple[int, int, int],
    sample_index: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    feature_count = int(feature_shape[2]) if len(feature_shape) == 3 else 0
    feature_mean = np.zeros((feature_count,), dtype=np.float32)
    feature_std = np.ones((feature_count,), dtype=np.float32)
    if feature_count <= 0 or sample_index.empty:
        return feature_mean, feature_std

    train_rows = sample_index.loc[
        sample_index["role"].astype(str).eq("train"),
        ["sequence_start_pos", "date_pos", "stock_pos"],
    ].copy()
    if train_rows.empty:
        return feature_mean, feature_std

    store = np.memmap(feature_store_path, dtype="float32", mode="r", shape=feature_shape)
    sums = np.zeros((feature_count,), dtype=np.float64)
    sq_sums = np.zeros((feature_count,), dtype=np.float64)
    counts = np.zeros((feature_count,), dtype=np.float64)
    zero_row = np.zeros((1, feature_count), dtype=np.float64)
    max_date = int(feature_shape[0])
    max_stock = int(feature_shape[1])
    for stock_pos, rows in train_rows.groupby("stock_pos", sort=False):
        stock_idx = int(stock_pos)
        if stock_idx < 0 or stock_idx >= max_stock:
            continue
        starts = rows["sequence_start_pos"].to_numpy(dtype=np.int64, copy=False)
        ends = rows["date_pos"].to_numpy(dtype=np.int64, copy=False) + 1
        valid = (starts >= 0) & (ends > starts) & (ends <= max_date)
        if not bool(valid.any()):
            continue
        starts = starts[valid]
        ends = ends[valid]
        values = np.asarray(store[:, stock_idx, :], dtype=np.float64)
        finite = np.isfinite(values)
        clean = np.where(finite, values, 0.0)
        sum_cum = np.vstack([zero_row, np.cumsum(clean, axis=0, dtype=np.float64)])
        sq_cum = np.vstack([zero_row, np.cumsum(np.square(clean), axis=0, dtype=np.float64)])
        count_cum = np.vstack([zero_row, np.cumsum(finite.astype(np.float64), axis=0, dtype=np.float64)])
        sums += (sum_cum[ends] - sum_cum[starts]).sum(axis=0)
        sq_sums += (sq_cum[ends] - sq_cum[starts]).sum(axis=0)
        counts += (count_cum[ends] - count_cum[starts]).sum(axis=0)

    feature_mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0).astype(np.float32)
    variance = np.divide(sq_sums, counts, out=np.zeros_like(sq_sums), where=counts > 0) - np.square(feature_mean.astype(np.float64))
    feature_std = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
    feature_std = np.where(np.isfinite(feature_std) & (np.abs(feature_std) > 1.0e-8), feature_std, 1.0).astype(np.float32)
    return feature_mean, feature_std


def build_forecast_memmap_dataset(
    prepared: PreparedPolicyInputs,
    *,
    root: Path,
    train_start_year: int = 2019,
    train_end_year: int = 2022,
    validation_year: int = 2023,
    test_year: int = 2024,
    lookback_days: int = 252,
    horizon: int = PATH20_HORIZON,
    cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    execution_mode: str = "next_open",
    max_samples_per_role: int = 0,
    max_samples_per_date_per_role: int = 0,
    feature_profile: str = DEFAULT_FORECAST_FEATURE_PROFILE,
    max_feature_columns: int = DEFAULT_FORECAST_MAX_FEATURE_COLUMNS,
    min_lookback_valid_ratio: float = 0.80,
    include_static_context: bool = False,
    static_context_fields: tuple[str, ...] | list[str] | str | None = None,
) -> ForecastMemmapDataset:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lookback_days = int(lookback_days)
    horizon = int(horizon)
    resolved_horizons = normalize_cumulative_horizons(cumulative_horizons, horizon=horizon)
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")
    if not (0.0 <= float(min_lookback_valid_ratio) <= 1.0):
        raise ValueError("min_lookback_valid_ratio must be in [0, 1].")

    dates = [pd.Timestamp(dt).normalize() for dt in prepared.close.index]
    universe = [str(stock).strip().upper() for stock in prepared.universe]
    _write_memmap_build_progress(
        root,
        "dataset_start",
        date_count=len(dates),
        universe_size=len(universe),
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        validation_year=validation_year,
        test_year=test_year,
        feature_profile=feature_profile,
        max_feature_columns=max_feature_columns,
        max_samples_per_role=max_samples_per_role,
        max_samples_per_date_per_role=max_samples_per_date_per_role,
    )
    date_to_pos = {dt: idx for idx, dt in enumerate(dates)}
    stock_to_pos = {stock: idx for idx, stock in enumerate(universe)}
    resolved_static_fields = normalize_static_context_fields(static_context_fields)
    static_id_columns = static_context_id_columns(resolved_static_fields)
    static_vocab = (
        build_static_context_vocab(prepared, static_context_fields=resolved_static_fields)
        if bool(include_static_context)
        else _static_context_schema_disabled(resolved_static_fields)
    )
    static_by_stock = _static_context_for_universe(prepared, universe=universe, vocab=static_vocab) if bool(include_static_context) else pd.DataFrame()
    liquidity_bucket_by_date_stock = (
        _cross_section_bucket(prepared.amount.reindex(index=dates, columns=universe), window=20, buckets=5)
        if bool(include_static_context)
        else {}
    )
    price_bucket_by_date_stock = (
        _cross_section_bucket(prepared.close.reindex(index=dates, columns=universe), window=20, buckets=5)
        if bool(include_static_context)
        else {}
    )
    next_open_extra_day = 1 if str(execution_mode or "next_open").strip().lower() == "next_open" else 0
    label_forward_offset = horizon + next_open_extra_day
    eligible_dates_by_role = _date_role_boundaries(
        dates,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        validation_year=validation_year,
        test_year=test_year,
        purge_trading_days=label_forward_offset,
    )
    _write_memmap_build_progress(
        root,
        "feature_store_start",
        eligible_date_counts={role: len(values) for role, values in eligible_dates_by_role.items()},
        label_forward_offset=label_forward_offset,
    )
    feature_store_path, feature_columns, feature_manifest, history_ratio = build_forecast_feature_store(
        prepared,
        dates,
        root=root,
        feature_profile=feature_profile,
        max_feature_columns=max_feature_columns,
        lookback_days=lookback_days,
        min_lookback_valid_ratio=float(min_lookback_valid_ratio),
    )
    feature_shape = tuple(int(item) for item in feature_manifest.get("feature_store_shape", [len(dates), len(universe), len(feature_columns)]))
    _write_memmap_build_progress(
        root,
        "feature_store_done",
        feature_store_shape=list(feature_shape),
        feature_count=len(feature_columns),
    )
    labels = build_path20_labels(prepared, execution_mode=execution_mode, horizon=horizon, cumulative_horizons=resolved_horizons)
    _write_memmap_build_progress(root, "labels_done", horizon=horizon, cumulative_horizons=list(resolved_horizons))

    sample_rows: list[dict[str, Any]] = []
    y_daily_rows: list[list[float]] = []
    y_cum_rows: list[list[float]] = []
    y_rank_by_horizon_rows: list[list[float]] = []
    y_rank_rows: list[float] = []
    y_drawdown_by_horizon_rows: list[list[float]] = []
    y_worst_by_horizon_rows: list[list[float]] = []
    y_upside_by_horizon_rows: list[list[float]] = []
    y_drawdown_rows: list[float] = []
    y_worst_rows: list[float] = []
    y_upside_rows: list[float] = []
    kept_history_values: list[float] = []
    dropped_target_nan = 0
    dropped_missing_lookback = 0
    dropped_low_history = 0
    capped = int(max_samples_per_role) > 0
    per_date_cap = max(int(max_samples_per_date_per_role), 0)
    capped_per_date = per_date_cap > 0
    sample_count_by_role = {"train": 0, "validation": 0, "test": 0}
    train_seen_stocks: set[str] = set()

    membership_frame = prepared.membership_frame.reindex(index=dates, columns=universe, fill_value=False).astype(bool)
    history_ratio = history_ratio.reindex(index=dates, columns=universe)
    for role in ("train", "validation", "test"):
        for date_idx, signal_dt in enumerate(eligible_dates_by_role.get(role, [])):
            if capped and sample_count_by_role[role] >= int(max_samples_per_role):
                break
            signal_pos = date_to_pos.get(signal_dt)
            if signal_pos is None or signal_pos < lookback_days - 1:
                dropped_missing_lookback += len(universe)
                continue
            label_end_pos = signal_pos + label_forward_offset
            if label_end_pos >= len(dates):
                dropped_missing_lookback += len(universe)
                continue
            sequence_start_pos = signal_pos - lookback_days + 1
            membership_row = membership_frame.loc[signal_dt] if signal_dt in membership_frame.index else pd.Series(False, index=universe)
            sample_count_this_date = 0
            stock_iter = _rotated_items(universe, date_idx * max(per_date_cap, 1)) if capped_per_date else universe
            for stock in stock_iter:
                if capped and sample_count_by_role[role] >= int(max_samples_per_role):
                    break
                if capped_per_date and sample_count_this_date >= per_date_cap:
                    break
                if not bool(membership_row.get(stock, False)):
                    continue
                valid_ratio = float(history_ratio.loc[signal_dt, stock]) if signal_dt in history_ratio.index and stock in history_ratio.columns else 0.0
                if not np.isfinite(valid_ratio) or valid_ratio < float(min_lookback_valid_ratio):
                    dropped_low_history += 1
                    continue
                daily_target = [
                    _safe_label_value(labels.daily_excess_return[step], signal_dt, str(stock))
                    for step in range(1, horizon + 1)
                ]
                cum_target = [
                    _safe_label_value(labels.cumulative_excess_return[step], signal_dt, str(stock))
                    for step in resolved_horizons
                ]
                rank_by_horizon_target = [
                    _safe_label_value(labels.forward_rank[step], signal_dt, str(stock))
                    for step in resolved_horizons
                ]
                rank_horizon = int(horizon) if int(horizon) in labels.forward_rank else int(resolved_horizons[-1])
                rank_target = _safe_label_value(labels.forward_rank[rank_horizon], signal_dt, str(stock))
                drawdown_by_horizon_target = [
                    _safe_label_value(labels.path_max_drawdown_by_horizon[step], signal_dt, str(stock))
                    for step in resolved_horizons
                ]
                worst_by_horizon_target = [
                    _safe_label_value(labels.path_worst_1d_by_horizon[step], signal_dt, str(stock))
                    for step in resolved_horizons
                ]
                upside_by_horizon_target = [
                    _safe_label_value(labels.path_upside_capture_by_horizon[step], signal_dt, str(stock))
                    for step in resolved_horizons
                ]
                drawdown_target = _safe_label_value(labels.path_max_drawdown_20d, signal_dt, str(stock))
                worst_target = _safe_label_value(labels.path_worst_1d_20d, signal_dt, str(stock))
                upside_target = _safe_label_value(labels.path_upside_capture_20d, signal_dt, str(stock))
                all_targets = [
                    *daily_target,
                    *cum_target,
                    *rank_by_horizon_target,
                    rank_target,
                    *drawdown_by_horizon_target,
                    *worst_by_horizon_target,
                    *upside_by_horizon_target,
                    drawdown_target,
                    worst_target,
                    upside_target,
                ]
                if not np.isfinite(np.asarray(all_targets, dtype=float)).all():
                    dropped_target_nan += 1
                    continue
                stock_seen = stock in train_seen_stocks if role != "train" else True
                sample_rows.append(
                    {
                        "row_id": int(len(sample_rows)),
                        "role": role,
                        "date": signal_dt.strftime("%Y-%m-%d"),
                        "stock": stock,
                        "date_pos": int(signal_pos),
                        "stock_pos": int(stock_to_pos[stock]),
                        "sequence_start_pos": int(sequence_start_pos),
                        "label_end_pos": int(label_end_pos),
                        "sequence_start_date": dates[sequence_start_pos].strftime("%Y-%m-%d"),
                        "label_end_date": dates[label_end_pos].strftime("%Y-%m-%d"),
                        "history_valid_ratio": float(valid_ratio),
                        "history_bucket": _history_bucket(valid_ratio),
                        "stock_seen_in_train": bool(stock_seen),
                        "lookback_days": int(lookback_days),
                    }
                )
                if bool(include_static_context):
                    static_row = static_by_stock.loc[stock]
                    sample_rows[-1].update({field: int(static_row.get(field, 0)) for field in static_id_columns})
                    sample_rows[-1]["liquidity_bucket_id"] = int(
                        liquidity_bucket_by_date_stock.get((pd.Timestamp(signal_dt), str(stock)), 0)
                    )
                    sample_rows[-1]["price_bucket_id"] = int(
                        price_bucket_by_date_stock.get((pd.Timestamp(signal_dt), str(stock)), 0)
                    )
                if role == "train":
                    train_seen_stocks.add(stock)
                kept_history_values.append(float(valid_ratio))
                y_daily_rows.append(daily_target)
                y_cum_rows.append(cum_target)
                y_rank_by_horizon_rows.append(rank_by_horizon_target)
                y_rank_rows.append(rank_target)
                y_drawdown_by_horizon_rows.append(drawdown_by_horizon_target)
                y_worst_by_horizon_rows.append(worst_by_horizon_target)
                y_upside_by_horizon_rows.append(upside_by_horizon_target)
                y_drawdown_rows.append(drawdown_target)
                y_worst_rows.append(worst_target)
                y_upside_rows.append(upside_target)
                sample_count_by_role[role] += 1
                sample_count_this_date += 1

    sample_index = pd.DataFrame(sample_rows)
    if not sample_index.empty:
        sample_index["stock_seen_in_train"] = sample_index["stock"].astype(str).map(lambda stock: stock in train_seen_stocks)
        sample_index.loc[sample_index["role"] == "train", "stock_seen_in_train"] = True
    sample_index_path = root / "forecast_sample_index.csv"
    sample_index.to_csv(sample_index_path, index=False, encoding="utf-8-sig")

    row_count = int(len(sample_index))
    _write_memmap_build_progress(root, "sample_index_done", sample_count=row_count, sample_count_by_role=sample_count_by_role)
    static_context_ids: np.memmap | None = None
    static_context_path = root / "forecast_static_context_ids.dat"
    if bool(include_static_context):
        static_values = (
            sample_index.reindex(columns=list(static_id_columns), fill_value=0)
            .fillna(0)
            .astype("int64")
            .to_numpy(dtype=np.int64)
        )
        static_context_ids = np.memmap(static_context_path, dtype="int64", mode="w+", shape=(row_count, len(static_id_columns)))
        if row_count:
            static_context_ids[...] = static_values.reshape((row_count, len(static_id_columns)))
        static_context_ids.flush()
        static_context_ids = np.memmap(static_context_path, dtype="int64", mode="r", shape=(row_count, len(static_id_columns)))
    y_daily = _write_array_memmap(root / "forecast_y_daily_excess.dat", np.asarray(y_daily_rows, dtype=np.float32), (row_count, horizon))
    y_cum = _write_array_memmap(root / "forecast_y_cum_excess.dat", np.asarray(y_cum_rows, dtype=np.float32), (row_count, len(resolved_horizons)))
    y_rank_by_horizon = _write_array_memmap(root / "forecast_y_rank_by_horizon.dat", np.asarray(y_rank_by_horizon_rows, dtype=np.float32), (row_count, len(resolved_horizons)))
    y_drawdown_by_horizon = _write_array_memmap(root / "forecast_y_drawdown_by_horizon.dat", np.asarray(y_drawdown_by_horizon_rows, dtype=np.float32), (row_count, len(resolved_horizons)))
    y_worst_by_horizon = _write_array_memmap(root / "forecast_y_worst_by_horizon.dat", np.asarray(y_worst_by_horizon_rows, dtype=np.float32), (row_count, len(resolved_horizons)))
    y_upside_by_horizon = _write_array_memmap(root / "forecast_y_upside_by_horizon.dat", np.asarray(y_upside_by_horizon_rows, dtype=np.float32), (row_count, len(resolved_horizons)))
    y_rank = _write_array_memmap(root / "forecast_y_rank_20d.dat", np.asarray(y_rank_rows, dtype=np.float32), (row_count,))
    y_drawdown = _write_array_memmap(root / "forecast_y_max_drawdown_20d.dat", np.asarray(y_drawdown_rows, dtype=np.float32), (row_count,))
    y_worst = _write_array_memmap(root / "forecast_y_worst_1d_20d.dat", np.asarray(y_worst_rows, dtype=np.float32), (row_count,))
    y_upside = _write_array_memmap(root / "forecast_y_upside_20d.dat", np.asarray(y_upside_rows, dtype=np.float32), (row_count,))
    _write_memmap_build_progress(root, "target_memmaps_done", sample_count=row_count)

    if row_count and sample_count_by_role["train"] > 0:
        feature_mean, feature_std = _fit_memmap_train_normalization(
            feature_store_path=feature_store_path,
            feature_shape=feature_shape,
            sample_index=sample_index,
        )
    else:
        feature_mean = np.zeros((len(feature_columns),), dtype=np.float32)
        feature_std = np.ones((len(feature_columns),), dtype=np.float32)
    _write_memmap_build_progress(root, "normalization_done", sample_count=row_count, feature_count=len(feature_columns))
    normalization_manifest: dict[str, Any] = {
        "fit_role": "train_only",
        "method": "zscore",
        "feature_count": int(len(feature_columns)),
        "feature_mean": feature_mean.astype(float).tolist(),
        "feature_std": feature_std.astype(float).tolist(),
        "raw_feature_nan_ratio": float(feature_manifest.get("feature_nan_ratio", 0.0) or 0.0),
    }
    raw_cache_meta = dict(getattr(prepared, "raw_cache_meta", {}) or {})
    pool_view_meta = dict(raw_cache_meta.get("pool_view", {}) or {})
    sector_board_meta = dict(raw_cache_meta.get("sector_board_view", {}) or {})
    manifest: dict[str, Any] = {
        "status": "completed" if row_count else "insufficient_or_incomplete",
        "stage": "forecast_sequence_dataset",
        "dataset_mode": "memmap",
        "source_market_dataset_id": str(pool_view_meta.get("source_market_dataset_id", "") or raw_cache_meta.get("dataset_id", "") or ""),
        "source_pool_view_id": str(pool_view_meta.get("dataset_id", "") or ""),
        "source_pool_view_kind": str(pool_view_meta.get("view_kind", "") or ""),
        "source_pool_view_name": str(pool_view_meta.get("view_name", "") or ""),
        "source_sector_board_view_id": str(sector_board_meta.get("dataset_id", "") or ""),
        "source_sector_board_view_kind": str(sector_board_meta.get("view_kind", "") or ""),
        "source_sector_board_snapshot_semantics": str(sector_board_meta.get("snapshot_semantics", "") or ""),
        "lookback_days": int(lookback_days),
        "horizon": int(horizon),
        "forecast_horizon": int(horizon),
        "execution_mode": str(execution_mode),
        "label_semantics": dict(labels.metadata),
        "role_years": {
            "train_start_year": int(train_start_year),
            "train_end_year": int(train_end_year),
            "validation_year": int(validation_year),
            "test_year": int(test_year),
        },
        "role_purge_trading_days": int(label_forward_offset),
        "next_open_label_extra_trading_day": int(next_open_extra_day),
        "feature_profile": str(feature_manifest.get("feature_profile", feature_profile)),
        "feature_manifest": dict(feature_manifest),
        "feature_columns": list(feature_columns),
        "feature_group_counts": dict(feature_manifest.get("feature_group_counts", {})),
        "feature_count_before_cap": int(feature_manifest.get("feature_count_before_cap", len(feature_columns))),
        "feature_count_after_cap": int(feature_manifest.get("feature_count_after_cap", len(feature_columns))),
        "raw_kline_feature_count": int(feature_manifest.get("raw_kline_feature_count", 0)),
        "market_context_feature_count": int(feature_manifest.get("market_context_feature_count", 0)),
        "peer_context_feature_count": int(feature_manifest.get("peer_context_feature_count", 0)),
        "sector_context_feature_count": int(feature_manifest.get("sector_context_feature_count", 0)),
        "sector_relative_context_feature_count": int(feature_manifest.get("sector_relative_context_feature_count", 0)),
        "regime_context_feature_count": int(feature_manifest.get("regime_context_feature_count", 0)),
        "alpha_prior_feature_count": int(feature_manifest.get("alpha_prior_feature_count", 0)),
        "history_quality_feature_count": int(feature_manifest.get("history_quality_feature_count", 0)),
        "feature_profile_audit": dict(feature_manifest.get("feature_profile_audit", {})),
        "feature_store_path": str(feature_store_path.resolve()),
        "feature_store_shape": [int(item) for item in feature_shape],
        "sample_index_csv": str(sample_index_path.resolve()),
        "date_values": [dt.strftime("%Y-%m-%d") for dt in dates],
        "stock_values": [str(stock) for stock in universe],
        "cumulative_horizons": [int(item) for item in resolved_horizons],
        "rank_horizons": [int(item) for item in resolved_horizons],
        "risk_horizons": [int(item) for item in resolved_horizons],
        "sample_count": int(row_count),
        "sample_count_by_role": {key: int(value) for key, value in sample_count_by_role.items()},
        "dropped_target_nan": int(dropped_target_nan),
        "dropped_missing_lookback": int(dropped_missing_lookback),
        "dropped_low_history": int(dropped_low_history),
        "min_lookback_valid_ratio": float(min_lookback_valid_ratio),
        "history_valid_ratio_summary": _summary_stats(kept_history_values),
        "max_samples_per_role": int(max_samples_per_role),
        "max_samples_per_date_per_role": int(max_samples_per_date_per_role),
        "normalization": normalization_manifest,
        "static_context_schema": {
            "enabled": bool(include_static_context),
            "fields": list(resolved_static_fields),
            "id_columns": list(static_id_columns),
            "vocab_sizes": dict(static_vocab.get("vocab_sizes", {}) or {}),
            "embedding_defaults": {
                "symbol": 16,
                "exchange": 4,
                "industry": 8,
                "board": 4,
                "liquidity_bucket": 4,
                "price_bucket": 4,
                "dropout": 0.20,
            },
        },
        "symbol_vocab_fingerprint": str(static_vocab.get("symbol_vocab_fingerprint", "")),
        "industry_vocab_fingerprint": str(static_vocab.get("industry_vocab_fingerprint", "")),
        "board_vocab_fingerprint": str(static_vocab.get("board_vocab_fingerprint", "")),
        "cross_section_batching_enabled": False,
    }
    if bool(include_static_context):
        manifest["static_context_path"] = str(static_context_path.resolve())
        manifest["static_context_shape"] = [int(row_count), int(len(static_id_columns))]
        manifest["static_context_vocab"] = {
            key: value
            for key, value in static_vocab.items()
            if key.endswith("_vocab") or key == "vocab_sizes"
        }
    if not row_count:
        manifest["reason"] = "no_forecast_samples"
    manifest_path = root / "forecast_dataset_manifest.json"
    write_json(manifest_path, _json_ready({**manifest, "manifest_json": str(manifest_path.resolve())}))
    _write_memmap_build_progress(root, "manifest_written", manifest_json=str(manifest_path.resolve()), sample_count=row_count)
    return ForecastMemmapDataset(
        root=root,
        feature_store_path=feature_store_path,
        feature_store_shape=feature_shape,
        sample_index=sample_index,
        y_daily_excess=y_daily,
        y_cum_excess=y_cum,
        y_rank_by_horizon=y_rank_by_horizon,
        y_rank_20d=y_rank,
        y_drawdown_by_horizon=y_drawdown_by_horizon,
        y_worst_by_horizon=y_worst_by_horizon,
        y_upside_by_horizon=y_upside_by_horizon,
        y_max_drawdown_20d=y_drawdown,
        y_worst_1d_20d=y_worst,
        y_upside_20d=y_upside,
        feature_columns=list(feature_columns),
        normalization_manifest=normalization_manifest,
        manifest=manifest,
        feature_mean=feature_mean,
        feature_std=feature_std,
        date_values=np.array([dt.strftime("%Y-%m-%d") for dt in dates], dtype=object),
        stock_values=np.array(universe, dtype=object),
        static_context_ids=static_context_ids,
    )


def save_forecast_sequence_dataset(dataset: ForecastSequenceDataset, root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / "forecast_dataset.npz"
    manifest_path = root / "forecast_dataset_manifest.json"
    np.savez_compressed(
        npz_path,
        x=dataset.x,
        y_daily_excess=dataset.y_daily_excess,
        y_cum_excess=dataset.y_cum_excess,
        y_rank_by_horizon=dataset.y_rank_by_horizon,
        y_rank_20d=dataset.y_rank_20d,
        y_drawdown_by_horizon=dataset.y_drawdown_by_horizon,
        y_worst_by_horizon=dataset.y_worst_by_horizon,
        y_upside_by_horizon=dataset.y_upside_by_horizon,
        y_max_drawdown_20d=dataset.y_max_drawdown_20d,
        y_worst_1d_20d=dataset.y_worst_1d_20d,
        y_upside_20d=dataset.y_upside_20d,
        date=np.array([pd.Timestamp(item).strftime("%Y-%m-%d") for item in dataset.date.tolist()], dtype=object),
        stock=dataset.stock.astype(str),
        role=dataset.role.astype(str),
        sequence_start_dates=np.array(
            [pd.Timestamp(item).strftime("%Y-%m-%d") for item in dataset.sequence_start_dates.tolist()],
            dtype=object,
        ),
        label_end_dates=np.array(
            [pd.Timestamp(item).strftime("%Y-%m-%d") for item in dataset.label_end_dates.tolist()],
            dtype=object,
        ),
        feature_columns=np.array(dataset.feature_columns, dtype=object),
    )
    manifest = {
        **dataset.manifest,
        "dataset_npz": str(npz_path.resolve()),
        "manifest_json": str(manifest_path.resolve()),
        "normalization": dataset.normalization_manifest,
    }
    manifest = _json_ready(manifest)
    write_json(manifest_path, manifest)
    return manifest
