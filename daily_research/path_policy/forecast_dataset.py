from __future__ import annotations

import json
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

    def torch_dataset(self, indices: np.ndarray, *, target_scale: float = 100.0) -> ForecastMemmapTorchDataset:
        return ForecastMemmapTorchDataset(self, indices, target_scale=target_scale)

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


def _validate_memmap_file(path: Path, *, shape: tuple[int, ...], label: str) -> None:
    if not path.exists():
        raise ValueError(f"forecast memmap manifest missing {label}: {path}")
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype("float32").itemsize
    actual_bytes = int(path.stat().st_size)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"forecast memmap manifest has invalid {label} size: "
            f"expected {expected_bytes} bytes for shape {shape}, got {actual_bytes} bytes at {path}"
        )


def load_forecast_memmap_dataset(manifest_json: str | Path) -> ForecastMemmapDataset:
    manifest_path = Path(manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
