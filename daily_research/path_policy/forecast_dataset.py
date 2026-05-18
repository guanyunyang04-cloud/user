from __future__ import annotations

from dataclasses import dataclass
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
)


@dataclass(frozen=True)
class ForecastSequenceDataset:
    x: np.ndarray
    y_daily_excess: np.ndarray
    y_cum_excess: np.ndarray
    y_rank_by_horizon: np.ndarray
    y_rank_20d: np.ndarray
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

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row_idx = int(self.indices[int(item)])
        if self._feature_store is None:
            self._feature_store = self.dataset.open_feature_store()
        x = self.dataset.input_window(row_idx, store=self._feature_store)
        y_daily = self.dataset.y_daily_excess[row_idx] * self.target_scale
        y_cum = self.dataset.y_cum_excess[row_idx] * self.target_scale
        y_risk = np.asarray(
            [
                self.dataset.y_max_drawdown_20d[row_idx],
                self.dataset.y_worst_1d_20d[row_idx],
                self.dataset.y_upside_20d[row_idx],
            ],
            dtype=np.float32,
        ) * self.target_scale
        return (
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y_daily, dtype=torch.float32),
            torch.as_tensor(y_cum, dtype=torch.float32),
            torch.as_tensor(y_risk, dtype=torch.float32),
            torch.as_tensor(row_idx, dtype=torch.long),
        )


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

    def role_indices(self, role: str) -> np.ndarray:
        return np.flatnonzero(self.sample_index["role"].astype(str).to_numpy() == str(role))

    def torch_dataset(self, indices: np.ndarray, *, target_scale: float = 100.0) -> ForecastMemmapTorchDataset:
        return ForecastMemmapTorchDataset(self, indices, target_scale=target_scale)


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
    manifest: dict[str, Any],
    normalization_manifest: dict[str, Any],
) -> ForecastSequenceDataset:
    x = np.empty((0, int(lookback_days), int(len(feature_columns))), dtype=np.float32)
    return ForecastSequenceDataset(
        x=x,
        y_daily_excess=np.empty((0, int(horizon)), dtype=np.float32),
        y_cum_excess=np.empty((0, len(PATH20_CUMULATIVE_HORIZONS)), dtype=np.float32),
        y_rank_by_horizon=np.empty((0, len(PATH20_CUMULATIVE_HORIZONS)), dtype=np.float32),
        y_rank_20d=np.empty((0,), dtype=np.float32),
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
    execution_mode: str = "next_open",
    max_samples_per_role: int = 0,
    feature_profile: str = DEFAULT_FORECAST_FEATURE_PROFILE,
    max_feature_columns: int = DEFAULT_FORECAST_MAX_FEATURE_COLUMNS,
) -> ForecastSequenceDataset:
    lookback_days = int(lookback_days)
    horizon = int(horizon)
    if horizon != PATH20_HORIZON:
        raise ValueError("Path20 forecast dataset currently requires horizon=20.")
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
    labels = build_path20_labels(prepared, execution_mode=execution_mode, horizon=horizon)

    x_rows: list[np.ndarray] = []
    y_daily_rows: list[list[float]] = []
    y_cum_rows: list[list[float]] = []
    y_rank_by_horizon_rows: list[list[float]] = []
    y_rank_rows: list[float] = []
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
    sample_count_by_role = {"train": 0, "validation": 0, "test": 0}

    for role in ("train", "validation", "test"):
        for signal_dt in eligible_dates_by_role.get(role, []):
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
            for stock in prepared.universe:
                if capped and sample_count_by_role[role] >= int(max_samples_per_role):
                    break
                if not bool(membership_row.get(stock, False)):
                    continue
                daily_target = [
                    _safe_label_value(labels.daily_excess_return[step], signal_dt, str(stock))
                    for step in range(1, horizon + 1)
                ]
                cum_target = [
                    _safe_label_value(labels.cumulative_excess_return[step], signal_dt, str(stock))
                    for step in PATH20_CUMULATIVE_HORIZONS
                ]
                rank_by_horizon_target = [
                    _safe_label_value(labels.forward_rank[step], signal_dt, str(stock))
                    for step in PATH20_CUMULATIVE_HORIZONS
                ]
                rank_target = _safe_label_value(labels.forward_rank[horizon], signal_dt, str(stock))
                drawdown_target = _safe_label_value(labels.path_max_drawdown_20d, signal_dt, str(stock))
                worst_target = _safe_label_value(labels.path_worst_1d_20d, signal_dt, str(stock))
                upside_target = _safe_label_value(labels.path_upside_capture_20d, signal_dt, str(stock))
                all_targets = [
                    *daily_target,
                    *cum_target,
                    *rank_by_horizon_target,
                    rank_target,
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
                y_drawdown_rows.append(drawdown_target)
                y_worst_rows.append(worst_target)
                y_upside_rows.append(upside_target)
                date_rows.append(signal_dt)
                stock_rows.append(str(stock))
                role_rows.append(role)
                sequence_start_rows.append(sequence_dates[0])
                label_end_rows.append(dates[label_end_pos])
                sample_count_by_role[role] += 1

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
        "alpha_prior_feature_count": int(feature_manifest.get("alpha_prior_feature_count", 0)),
        "cumulative_horizons": [int(item) for item in PATH20_CUMULATIVE_HORIZONS],
        "rank_horizons": [int(item) for item in PATH20_CUMULATIVE_HORIZONS],
        "sample_count_by_role": {key: int(value) for key, value in sample_count_by_role.items()},
        "dropped_target_nan": int(dropped_target_nan),
        "dropped_missing_lookback": int(dropped_missing_lookback),
        "max_samples_per_role": int(max_samples_per_role),
    }

    if not x_rows:
        base_manifest["status"] = "insufficient_or_incomplete"
        base_manifest["reason"] = "no_forecast_samples"
        return _empty_dataset(
            feature_columns=feature_columns,
            lookback_days=lookback_days,
            horizon=horizon,
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
    execution_mode: str = "next_open",
    max_samples_per_role: int = 0,
    feature_profile: str = DEFAULT_FORECAST_FEATURE_PROFILE,
    max_feature_columns: int = DEFAULT_FORECAST_MAX_FEATURE_COLUMNS,
    min_lookback_valid_ratio: float = 0.80,
) -> ForecastMemmapDataset:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lookback_days = int(lookback_days)
    horizon = int(horizon)
    if horizon != PATH20_HORIZON:
        raise ValueError("Path20 forecast memmap dataset currently requires horizon=20.")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")
    if not (0.0 <= float(min_lookback_valid_ratio) <= 1.0):
        raise ValueError("min_lookback_valid_ratio must be in [0, 1].")

    dates = [pd.Timestamp(dt).normalize() for dt in prepared.close.index]
    universe = [str(stock).strip().upper() for stock in prepared.universe]
    date_to_pos = {dt: idx for idx, dt in enumerate(dates)}
    stock_to_pos = {stock: idx for idx, stock in enumerate(universe)}
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
    labels = build_path20_labels(prepared, execution_mode=execution_mode, horizon=horizon)

    sample_rows: list[dict[str, Any]] = []
    y_daily_rows: list[list[float]] = []
    y_cum_rows: list[list[float]] = []
    y_rank_by_horizon_rows: list[list[float]] = []
    y_rank_rows: list[float] = []
    y_drawdown_rows: list[float] = []
    y_worst_rows: list[float] = []
    y_upside_rows: list[float] = []
    kept_history_values: list[float] = []
    dropped_target_nan = 0
    dropped_missing_lookback = 0
    dropped_low_history = 0
    capped = int(max_samples_per_role) > 0
    sample_count_by_role = {"train": 0, "validation": 0, "test": 0}
    train_seen_stocks: set[str] = set()

    membership_frame = prepared.membership_frame.reindex(index=dates, columns=universe, fill_value=False).astype(bool)
    history_ratio = history_ratio.reindex(index=dates, columns=universe)
    for role in ("train", "validation", "test"):
        for signal_dt in eligible_dates_by_role.get(role, []):
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
            for stock in universe:
                if capped and sample_count_by_role[role] >= int(max_samples_per_role):
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
                    for step in PATH20_CUMULATIVE_HORIZONS
                ]
                rank_by_horizon_target = [
                    _safe_label_value(labels.forward_rank[step], signal_dt, str(stock))
                    for step in PATH20_CUMULATIVE_HORIZONS
                ]
                rank_target = _safe_label_value(labels.forward_rank[horizon], signal_dt, str(stock))
                drawdown_target = _safe_label_value(labels.path_max_drawdown_20d, signal_dt, str(stock))
                worst_target = _safe_label_value(labels.path_worst_1d_20d, signal_dt, str(stock))
                upside_target = _safe_label_value(labels.path_upside_capture_20d, signal_dt, str(stock))
                all_targets = [
                    *daily_target,
                    *cum_target,
                    *rank_by_horizon_target,
                    rank_target,
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
                if role == "train":
                    train_seen_stocks.add(stock)
                kept_history_values.append(float(valid_ratio))
                y_daily_rows.append(daily_target)
                y_cum_rows.append(cum_target)
                y_rank_by_horizon_rows.append(rank_by_horizon_target)
                y_rank_rows.append(rank_target)
                y_drawdown_rows.append(drawdown_target)
                y_worst_rows.append(worst_target)
                y_upside_rows.append(upside_target)
                sample_count_by_role[role] += 1

    sample_index = pd.DataFrame(sample_rows)
    if not sample_index.empty:
        sample_index["stock_seen_in_train"] = sample_index["stock"].astype(str).map(lambda stock: stock in train_seen_stocks)
        sample_index.loc[sample_index["role"] == "train", "stock_seen_in_train"] = True
    sample_index_path = root / "forecast_sample_index.csv"
    sample_index.to_csv(sample_index_path, index=False, encoding="utf-8-sig")

    row_count = int(len(sample_index))
    y_daily = _write_array_memmap(root / "forecast_y_daily_excess.dat", np.asarray(y_daily_rows, dtype=np.float32), (row_count, horizon))
    y_cum = _write_array_memmap(root / "forecast_y_cum_excess.dat", np.asarray(y_cum_rows, dtype=np.float32), (row_count, len(PATH20_CUMULATIVE_HORIZONS)))
    y_rank_by_horizon = _write_array_memmap(root / "forecast_y_rank_by_horizon.dat", np.asarray(y_rank_by_horizon_rows, dtype=np.float32), (row_count, len(PATH20_CUMULATIVE_HORIZONS)))
    y_rank = _write_array_memmap(root / "forecast_y_rank_20d.dat", np.asarray(y_rank_rows, dtype=np.float32), (row_count,))
    y_drawdown = _write_array_memmap(root / "forecast_y_max_drawdown_20d.dat", np.asarray(y_drawdown_rows, dtype=np.float32), (row_count,))
    y_worst = _write_array_memmap(root / "forecast_y_worst_1d_20d.dat", np.asarray(y_worst_rows, dtype=np.float32), (row_count,))
    y_upside = _write_array_memmap(root / "forecast_y_upside_20d.dat", np.asarray(y_upside_rows, dtype=np.float32), (row_count,))

    feature_mean = np.zeros((len(feature_columns),), dtype=np.float32)
    feature_std = np.ones((len(feature_columns),), dtype=np.float32)
    if row_count and sample_count_by_role["train"] > 0:
        store = np.memmap(feature_store_path, dtype="float32", mode="r", shape=feature_shape)
        train_positions = sample_index.index[sample_index["role"].astype(str) == "train"].to_numpy(dtype=int)
        sums = np.zeros((len(feature_columns),), dtype=np.float64)
        sq_sums = np.zeros((len(feature_columns),), dtype=np.float64)
        counts = np.zeros((len(feature_columns),), dtype=np.float64)
        for row_idx in train_positions:
            row = sample_index.iloc[int(row_idx)]
            window = np.asarray(store[int(row["sequence_start_pos"]) : int(row["date_pos"]) + 1, int(row["stock_pos"]), :], dtype=np.float32)
            finite = np.isfinite(window)
            clean = np.where(finite, window, 0.0)
            sums += clean.sum(axis=0)
            sq_sums += np.square(clean).sum(axis=0)
            counts += finite.sum(axis=0)
        feature_mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0).astype(np.float32)
        variance = np.divide(sq_sums, counts, out=np.zeros_like(sq_sums), where=counts > 0) - np.square(feature_mean.astype(np.float64))
        feature_std = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
        feature_std = np.where(np.isfinite(feature_std) & (np.abs(feature_std) > 1.0e-8), feature_std, 1.0).astype(np.float32)
    normalization_manifest: dict[str, Any] = {
        "fit_role": "train_only",
        "method": "zscore",
        "feature_count": int(len(feature_columns)),
        "feature_mean": feature_mean.astype(float).tolist(),
        "feature_std": feature_std.astype(float).tolist(),
        "raw_feature_nan_ratio": float(feature_manifest.get("feature_nan_ratio", 0.0) or 0.0),
    }
    manifest: dict[str, Any] = {
        "status": "completed" if row_count else "insufficient_or_incomplete",
        "stage": "forecast_sequence_dataset",
        "dataset_mode": "memmap",
        "lookback_days": int(lookback_days),
        "horizon": int(horizon),
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
        "alpha_prior_feature_count": int(feature_manifest.get("alpha_prior_feature_count", 0)),
        "history_quality_feature_count": int(feature_manifest.get("history_quality_feature_count", 0)),
        "feature_store_path": str(feature_store_path.resolve()),
        "feature_store_shape": [int(item) for item in feature_shape],
        "sample_index_csv": str(sample_index_path.resolve()),
        "cumulative_horizons": [int(item) for item in PATH20_CUMULATIVE_HORIZONS],
        "rank_horizons": [int(item) for item in PATH20_CUMULATIVE_HORIZONS],
        "sample_count": int(row_count),
        "sample_count_by_role": {key: int(value) for key, value in sample_count_by_role.items()},
        "dropped_target_nan": int(dropped_target_nan),
        "dropped_missing_lookback": int(dropped_missing_lookback),
        "dropped_low_history": int(dropped_low_history),
        "min_lookback_valid_ratio": float(min_lookback_valid_ratio),
        "history_valid_ratio_summary": _summary_stats(kept_history_values),
        "max_samples_per_role": int(max_samples_per_role),
        "normalization": normalization_manifest,
    }
    if not row_count:
        manifest["reason"] = "no_forecast_samples"
    manifest_path = root / "forecast_dataset_manifest.json"
    write_json(manifest_path, _json_ready({**manifest, "manifest_json": str(manifest_path.resolve())}))
    return ForecastMemmapDataset(
        root=root,
        feature_store_path=feature_store_path,
        feature_store_shape=feature_shape,
        sample_index=sample_index,
        y_daily_excess=y_daily,
        y_cum_excess=y_cum,
        y_rank_by_horizon=y_rank_by_horizon,
        y_rank_20d=y_rank,
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
