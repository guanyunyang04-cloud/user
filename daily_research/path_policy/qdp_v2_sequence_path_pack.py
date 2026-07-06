from __future__ import annotations

import argparse
import gc
import json
import math
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from daily_research.path_policy.qdp_v2_raw_rising_path_atlas import (
    DAILY_RAW_COLUMNS,
    INTRADAY_SIGNAL_COLUMNS,
    LIMIT_SIGNAL_COLUMNS,
    RAW_SIGNAL_COLUMNS,
    _add_raw_daily_signals,
)


DEFAULT_QDP_ROOT = Path("quant_data_platform/data/qdp_v2")
DEFAULT_OUTPUT_ROOT = Path("quant_data_platform/data/qdp_v2/research/sequence_pack")
DEFAULT_LOOKBACK_DAYS = 100
DEFAULT_FORWARD_DAYS = 20
DEFAULT_START_DATE = "2012-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_TRAIN_YEARS = tuple(range(2012, 2024))
DEFAULT_VALIDATION_YEARS = (2024,)
DEFAULT_TEST_YEARS = (2025,)
DEFAULT_RUN_TAG = "qdp_v2_seq100_path20"

DAILY_RAW_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "open_ret_prev_close",
    "high_ret_prev_close",
    "low_ret_prev_close",
    "close_ret_prev_close",
    "volume_log",
    "amount_log",
    "intraday_range_raw",
]

PATH_OHLC_FIELDS = ["open", "high", "low", "close"]
PATH_OHLCVA_FIELDS = ["open", "high", "low", "close", "volume", "amount"]


def path_summary_columns(forward_days: int) -> list[str]:
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
        f"path_trade_value_{suffix}",
    ]


def path_value_column(forward_days: int) -> str:
    return f"path_trade_value_{int(forward_days)}d"


PATH_SUMMARY_COLUMNS = path_summary_columns(DEFAULT_FORWARD_DAYS)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return str(path.resolve())


def _parse_years(raw: str | Iterable[int] | None, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return default
    if not isinstance(raw, str):
        return tuple(sorted({int(item) for item in raw}))
    values: list[int] = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start, end = [int(part.strip()) for part in item.split("-", 1)]
            values.extend(range(start, end + 1))
        else:
            values.append(int(item))
    return tuple(sorted(set(values))) or default


def _read_active(root: Path) -> dict[str, Any]:
    return json.loads((root / "active" / "active.json").read_text(encoding="utf-8"))


def _relative_shard_paths(root: Path, manifest: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for shard in list(manifest.get("shards", []) or []):
        raw = Path(str(shard.get("path", "") or ""))
        paths.append(raw if raw.is_absolute() else root / raw)
    return paths


def _dataset_manifest(root: Path, active: Mapping[str, Any], domain: str) -> dict[str, Any]:
    dataset_id = str(dict(active.get("datasets", {}) or {}).get(domain, "") or "")
    if not dataset_id:
        raise KeyError(f"active manifest has no dataset for domain: {domain}")
    path = root / "datasets" / domain / dataset_id / "dataset.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _read_dataset_date_range(root: Path, active: Mapping[str, Any], domain: str, columns: list[str], start: str, end: str) -> pd.DataFrame:
    manifest = _dataset_manifest(root, active, domain)
    available = {str(item.get("name", "")) for item in list(manifest.get("schema", []) or [])}
    selected = [col for col in columns if col in available]
    if not selected:
        return pd.DataFrame(columns=columns)
    paths = _relative_shard_paths(root, manifest)
    filt = (ds.field("trade_date") >= str(start)) & (ds.field("trade_date") <= str(end))
    table = ds.dataset([str(path) for path in paths], format="parquet").to_table(columns=selected, filter=filt)
    frame = table.to_pandas()
    if "trade_date" in frame.columns:
        frame["trade_date"] = frame["trade_date"].astype(str)
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    return frame


def _read_trading_dates(root: Path, active: Mapping[str, Any], start: str, end: str) -> list[str]:
    calendar = _read_dataset_date_range(root, active, "trading_calendar", ["trade_date", "is_open"], start, end)
    if calendar.empty:
        raise ValueError("active trading_calendar is empty")
    return (
        calendar[calendar["is_open"].astype(bool)]["trade_date"]
        .astype(str)
        .sort_values(kind="mergesort")
        .drop_duplicates()
        .to_list()
    )


def _fill_float_memmap(path: Path, shape: tuple[int, ...], *, fill_value: float) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(path, dtype="float32", mode="w+", shape=shape)
    chunk = max(1, min(int(shape[0]), 128))
    for start in range(0, int(shape[0]), chunk):
        arr[start : start + chunk] = np.float32(fill_value)
    arr.flush()
    return arr


def _fill_bool_memmap(path: Path, shape: tuple[int, ...], *, fill_value: bool = False) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(path, dtype="bool", mode="w+", shape=shape)
    arr[:] = bool(fill_value)
    arr.flush()
    return arr


class DateShardedFloatStore:
    def __init__(self, *, directory: Path, name: str, shape: tuple[int, int, int, int], shard_size: int) -> None:
        self.directory = directory
        self.name = str(name)
        self.shape = tuple(int(item) for item in shape)
        self.shard_size = max(int(shard_size), 1)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._shards: dict[int, tuple[int, int, np.memmap, Path]] = {}
        self.shard_metas: list[dict[str, Any]] = []

    def _open_shard(self, date_idx: int) -> tuple[int, int, np.memmap, Path]:
        start = (int(date_idx) // self.shard_size) * self.shard_size
        end = min(start + self.shard_size, self.shape[0])
        if start in self._shards:
            return self._shards[start]
        path = self.directory / f"{self.name}.{start:06d}_{end - 1:06d}.float32.dat"
        arr = np.memmap(path, dtype="float32", mode="w+", shape=(end - start, *self.shape[1:]))
        arr[:] = np.nan
        arr.flush()
        item = (start, end, arr, path)
        self._shards[start] = item
        self.shard_metas.append(
            {
                "path": str(path.resolve()),
                "date_start_idx": int(start),
                "date_end_idx": int(end - 1),
                "shape": [int(end - start), *[int(dim) for dim in self.shape[1:]]],
            }
        )
        return item

    def set_date(self, date_idx: int, values: np.ndarray) -> None:
        start, _end, arr, _path = self._open_shard(int(date_idx))
        arr[int(date_idx) - start] = values

    def mask_date(self, date_idx: int, mask: np.ndarray) -> None:
        start, _end, arr, _path = self._open_shard(int(date_idx))
        arr[int(date_idx) - start, ~mask, :, :] = np.nan

    def flush(self) -> None:
        for _start, _end, arr, _path in self._shards.values():
            arr.flush()

    def manifest(self, *, fields: list[str], anchor: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "shape": [int(dim) for dim in self.shape],
            "fields": list(fields),
            "anchor": str(anchor),
            "shard_size": int(self.shard_size),
            "shards": sorted(self.shard_metas, key=lambda item: int(item["date_start_idx"])),
        }
        if extra:
            payload.update(dict(extra))
        return payload


def _label_fill_nan(store: Any) -> None:
    if store is None:
        return
    if isinstance(store, DateShardedFloatStore):
        return
    store[:] = np.nan


def _label_set_date(store: Any, date_idx: int, values: np.ndarray) -> None:
    if store is None:
        return
    if isinstance(store, DateShardedFloatStore):
        store.set_date(date_idx, values)
    else:
        store[date_idx] = values


def _label_mask_date(store: Any, date_idx: int, mask: np.ndarray) -> None:
    if store is None:
        return
    if isinstance(store, DateShardedFloatStore):
        store.mask_date(date_idx, mask)
    else:
        store[date_idx, ~mask, :, :] = np.nan


def _label_flush(store: Any) -> None:
    if store is None:
        return
    if hasattr(store, "flush"):
        store.flush()


def _index_frame(frame: pd.DataFrame, date_to_idx: Mapping[str, int], symbol_to_idx: Mapping[str, int]) -> pd.DataFrame:
    out = frame.copy()
    out["_date_idx"] = out["trade_date"].map(date_to_idx)
    out["_symbol_idx"] = out["symbol"].map(symbol_to_idx)
    out = out[out["_date_idx"].notna() & out["_symbol_idx"].notna()].copy()
    out["_date_idx"] = out["_date_idx"].astype("int32")
    out["_symbol_idx"] = out["_symbol_idx"].astype("int32")
    return out


def _write_panel_values(
    panel: np.memmap,
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    date_to_idx: Mapping[str, int],
    symbol_to_idx: Mapping[str, int],
) -> None:
    if frame.empty:
        return
    indexed = _index_frame(frame, date_to_idx, symbol_to_idx)
    if indexed.empty:
        return
    values = indexed[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    panel[indexed["_date_idx"].to_numpy(dtype=np.int64), indexed["_symbol_idx"].to_numpy(dtype=np.int64), :] = values
    panel.flush()


def _prepare_daily_frame(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.sort_values(["symbol", "trade_date"], kind="mergesort").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce").astype("float64")
    groups = daily.groupby("symbol", sort=False, group_keys=False)
    prev_close = groups["close"].shift(1).astype("float64")
    daily["open_ret_prev_close"] = daily["open"].div(prev_close.replace(0.0, np.nan)).sub(1.0)
    daily["high_ret_prev_close"] = daily["high"].div(prev_close.replace(0.0, np.nan)).sub(1.0)
    daily["low_ret_prev_close"] = daily["low"].div(prev_close.replace(0.0, np.nan)).sub(1.0)
    daily["close_ret_prev_close"] = daily["close"].div(prev_close.replace(0.0, np.nan)).sub(1.0)
    daily["volume_log"] = np.log1p(daily["volume"])
    daily["amount_log"] = np.log1p(daily["amount"])
    daily["intraday_range_raw"] = daily["high"].div(daily["low"].replace(0.0, np.nan)).sub(1.0)
    return daily.replace([np.inf, -np.inf], np.nan)


def _compute_future_path_and_masks(
    *,
    raw_panel: np.ndarray,
    up_limit_panel: np.ndarray,
    lookback_days: int,
    forward_days: int,
    price_anchor: str = "next_open",
    future_path_out: np.ndarray | None = None,
    future_ohlcva_path_out: np.ndarray | None = None,
    path_summary_out: np.ndarray | None = None,
    flush_every_dates: int = 8,
    write_legacy_ohlc_path: bool = True,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    open_panel = raw_panel[:, :, DAILY_RAW_FEATURES.index("open")]
    high_panel = raw_panel[:, :, DAILY_RAW_FEATURES.index("high")]
    low_panel = raw_panel[:, :, DAILY_RAW_FEATURES.index("low")]
    close_panel = raw_panel[:, :, DAILY_RAW_FEATURES.index("close")]
    volume_log_panel = raw_panel[:, :, DAILY_RAW_FEATURES.index("volume_log")]
    amount_log_panel = raw_panel[:, :, DAILY_RAW_FEATURES.index("amount_log")]
    n_dates, n_symbols = open_panel.shape
    input_valid = np.zeros((n_dates, n_symbols), dtype=bool)
    entry_buyable = np.zeros((n_dates, n_symbols), dtype=bool)
    label_valid = np.zeros((n_dates, n_symbols), dtype=bool)
    summary_columns = path_summary_columns(forward_days)
    future_path = None
    if bool(write_legacy_ohlc_path):
        future_path = (
            future_path_out
            if future_path_out is not None
            else np.full((n_dates, n_symbols, forward_days, 4), np.nan, dtype=np.float32)
        )
    future_ohlcva_path = (
        future_ohlcva_path_out
        if future_ohlcva_path_out is not None
        else np.full((n_dates, n_symbols, forward_days, 6), np.nan, dtype=np.float32)
    )
    path_summary = (
        path_summary_out
        if path_summary_out is not None
        else np.full((n_dates, n_symbols, len(summary_columns)), np.nan, dtype=np.float32)
    )
    _label_fill_nan(future_path)
    _label_fill_nan(future_ohlcva_path)
    path_summary[:] = np.nan
    finite_close = np.isfinite(close_panel)
    for date_idx in range(n_dates):
        start = date_idx - int(lookback_days) + 1
        entry_idx = date_idx + 1
        end_idx = entry_idx + int(forward_days)
        if start < 0:
            continue
        input_valid[date_idx] = finite_close[start : date_idx + 1].all(axis=0)
        if end_idx > n_dates:
            continue
        entry_open = open_panel[entry_idx].astype("float64", copy=False)
        signal_close = close_panel[date_idx].astype("float64", copy=False)
        entry_up_limit = up_limit_panel[entry_idx].astype("float64", copy=False)
        entry_ok = np.isfinite(entry_open)
        limit_blocked = np.isfinite(entry_up_limit) & entry_ok & (entry_open >= entry_up_limit * 0.999)
        entry_buyable[date_idx] = entry_ok & (~limit_blocked)
        fut_open = open_panel[entry_idx:end_idx]
        fut_high = high_panel[entry_idx:end_idx]
        fut_low = low_panel[entry_idx:end_idx]
        fut_close = close_panel[entry_idx:end_idx]
        fut_volume_log = volume_log_panel[entry_idx:end_idx]
        fut_amount_log = amount_log_panel[entry_idx:end_idx]
        history_start = max(0, date_idx - 19)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            trailing_volume_log = np.nanmean(volume_log_panel[history_start : date_idx + 1], axis=0)
            trailing_amount_log = np.nanmean(amount_log_panel[history_start : date_idx + 1], axis=0)
        path_ok = (
            np.isfinite(fut_open).all(axis=0)
            & np.isfinite(fut_high).all(axis=0)
            & np.isfinite(fut_low).all(axis=0)
            & np.isfinite(fut_close).all(axis=0)
            & np.isfinite(fut_volume_log).all(axis=0)
            & np.isfinite(fut_amount_log).all(axis=0)
            & np.isfinite(trailing_volume_log)
            & np.isfinite(trailing_amount_log)
            & entry_ok
        )
        if str(price_anchor) == "today_close":
            denom = np.where(signal_close != 0.0, signal_close, np.nan)
            path_ok &= np.isfinite(denom)
        elif str(price_anchor) == "next_open":
            denom = np.where(entry_open != 0.0, entry_open, np.nan)
        else:
            raise ValueError(f"unsupported price_anchor: {price_anchor}")
        label_valid[date_idx] = path_ok
        paths = [
            fut_open.T / denom[:, None] - 1.0,
            fut_high.T / denom[:, None] - 1.0,
            fut_low.T / denom[:, None] - 1.0,
            fut_close.T / denom[:, None] - 1.0,
        ]
        stacked = np.stack(paths, axis=2).astype(np.float32, copy=False)
        volume_rel = (fut_volume_log.T - trailing_volume_log[:, None]).astype(np.float32, copy=False)
        amount_rel = (fut_amount_log.T - trailing_amount_log[:, None]).astype(np.float32, copy=False)
        stacked_ohlcva = np.concatenate(
            [stacked, volume_rel[:, :, None], amount_rel[:, :, None]],
            axis=2,
        ).astype(np.float32, copy=False)
        _label_set_date(future_path, date_idx, stacked)
        _label_set_date(future_ohlcva_path, date_idx, stacked_ohlcva)
        if str(price_anchor) == "today_close":
            entry_anchor = np.maximum(1.0 + stacked[:, :1, 0].astype("float64", copy=False), 1.0e-6)
            summary_stacked = (1.0 + stacked.astype("float64", copy=False)) / entry_anchor[:, :, None] - 1.0
        else:
            summary_stacked = stacked.astype("float64", copy=False)
        high_ret = summary_stacked[:, :, 1]
        low_ret = summary_stacked[:, :, 2]
        close_ret = summary_stacked[:, :, 3]
        max_ret = np.max(np.where(np.isfinite(high_ret), high_ret, -np.inf), axis=1)
        min_ret = np.min(np.where(np.isfinite(low_ret), low_ret, np.inf), axis=1)
        max_ret[~np.isfinite(max_ret)] = np.nan
        min_ret[~np.isfinite(min_ret)] = np.nan
        final_ret = close_ret[:, -1]
        peak_idx = np.nanargmax(np.where(np.isfinite(high_ret), high_ret, -np.inf), axis=1)
        trough_idx = np.nanargmin(np.where(np.isfinite(low_ret), low_ret, np.inf), axis=1)
        min_after_peak = np.full(n_symbols, np.nan, dtype=np.float64)
        for symbol_idx in np.where(path_ok)[0]:
            pidx = int(peak_idx[symbol_idx])
            tidx = int(trough_idx[symbol_idx])
            min_after_peak[symbol_idx] = np.nanmin(low_ret[symbol_idx, pidx:])
            trough_idx[symbol_idx] = tidx
        drawdown_after_peak = (1.0 + min_after_peak) / (1.0 + max_ret) - 1.0
        time_above = (close_ret > 0).mean(axis=1)
        time_below = (close_ret < 0).mean(axis=1)
        value = final_ret + 0.50 * max_ret + 0.35 * min_ret + 0.20 * drawdown_after_peak
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
                value,
            ]
        ).astype(np.float32, copy=False)
        summary[~path_ok, :] = np.nan
        _label_mask_date(future_path, date_idx, path_ok)
        _label_mask_date(future_ohlcva_path, date_idx, path_ok)
        path_summary[date_idx] = summary
        if int(flush_every_dates) > 0 and (date_idx + 1) % int(flush_every_dates) == 0:
            _label_flush(future_path)
            _label_flush(future_ohlcva_path)
            if isinstance(path_summary, np.memmap):
                path_summary.flush()
            gc.collect()
    return future_path, future_ohlcva_path, path_summary, input_valid, entry_buyable, label_valid


def _fit_normalization(panel: np.ndarray, train_date_mask: np.ndarray) -> dict[str, list[float]]:
    selected = np.asarray(panel[train_date_mask], dtype=np.float32)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(selected, axis=(0, 1))
        std = np.nanstd(selected, axis=(0, 1))
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    return {"mean": mean.tolist(), "std": std.tolist()}


def _build_sample_index(
    *,
    date_values: list[str],
    symbol_values: list[str],
    start_date: str,
    end_date: str,
    train_years: tuple[int, ...],
    validation_years: tuple[int, ...],
    test_years: tuple[int, ...],
    input_valid: np.ndarray,
    entry_buyable: np.ndarray,
    label_valid: np.ndarray,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    date_arr = np.asarray(date_values, dtype=object)
    symbol_arr = np.asarray(symbol_values, dtype=object)
    for date_idx, trade_date in enumerate(date_values):
        if trade_date < str(start_date) or trade_date > str(end_date):
            continue
        year = int(str(trade_date)[:4])
        if year in train_years:
            split = "train"
        elif year in validation_years:
            split = "validation"
        elif year in test_years:
            split = "test"
        else:
            continue
        mask = input_valid[date_idx] & entry_buyable[date_idx] & label_valid[date_idx]
        symbol_idx = np.flatnonzero(mask).astype(np.int32)
        if len(symbol_idx) == 0:
            continue
        rows.append(
            pd.DataFrame(
                {
                    "sample_id": np.arange(len(symbol_idx), dtype=np.int64),
                    "split": split,
                    "year": np.int16(year),
                    "trade_date": str(trade_date),
                    "date_idx": np.full(len(symbol_idx), date_idx, dtype=np.int32),
                    "symbol_idx": symbol_idx,
                    "symbol": symbol_arr[symbol_idx],
                    "entry_trade_date": date_arr[date_idx + 1] if date_idx + 1 < len(date_arr) else "",
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["sample_id", "split", "year", "trade_date", "date_idx", "symbol_idx", "symbol", "entry_trade_date"])
    out = pd.concat(rows, ignore_index=True)
    out["sample_id"] = np.arange(len(out), dtype=np.int64)
    return out


@dataclass(frozen=True)
class SequencePackConfig:
    qdp_root: Path
    output_root: Path
    run_tag: str
    lookback_days: int
    forward_days: int
    start_date: str
    end_date: str
    train_years: tuple[int, ...]
    validation_years: tuple[int, ...]
    test_years: tuple[int, ...]
    write_legacy_ohlc_label: bool = True
    label_shard_size: int = 0
    price_anchor: str = "next_open"


def build_sequence_pack(config: SequencePackConfig) -> dict[str, Any]:
    root = config.qdp_root.resolve()
    active = _read_active(root)
    active_scope = dict(active.get("scope", {}) or {})
    scope_start = str(active_scope.get("start_date", "2011-11-22") or "2011-11-22")
    active_end = str(active.get("active_as_of_date", active_scope.get("end_date", config.end_date)) or config.end_date)
    output_dir = config.output_root / str(config.run_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    _write_json(progress_path, {"status": "started", "updated_at": _now()})
    all_open_dates = _read_trading_dates(root, active, scope_start, active_end)
    if not all_open_dates:
        raise ValueError("no open trading dates in active scope")
    start_positions = [idx for idx, date in enumerate(all_open_dates) if date >= str(config.start_date)]
    end_positions = [idx for idx, date in enumerate(all_open_dates) if date <= str(config.end_date)]
    if not start_positions or not end_positions:
        raise ValueError(f"requested window has no trading dates: {config.start_date}..{config.end_date}")
    sample_start_pos = int(start_positions[0])
    sample_end_pos = int(end_positions[-1])
    panel_start_pos = max(0, sample_start_pos - int(config.lookback_days) + 1)
    panel_end_pos = min(len(all_open_dates) - 1, sample_end_pos + int(config.forward_days))
    date_values = all_open_dates[panel_start_pos : panel_end_pos + 1]
    _write_json(progress_path, {"status": "reading_daily", "updated_at": _now()})
    daily = _read_dataset_date_range(root, active, "market_daily_raw", DAILY_RAW_COLUMNS, date_values[0], date_values[-1])
    if daily.empty:
        raise ValueError("market_daily_raw returned no rows")
    daily = _prepare_daily_frame(daily)
    symbol_values = sorted(daily["symbol"].astype(str).str.upper().str.strip().unique().tolist())
    date_to_idx = {date: idx for idx, date in enumerate(date_values)}
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(symbol_values)}
    n_dates = len(date_values)
    n_symbols = len(symbol_values)

    panel_dir = output_dir / "panels"
    label_dir = output_dir / "labels"
    mask_dir = output_dir / "masks"
    daily_raw = _fill_float_memmap(panel_dir / "daily_raw.float32.dat", (n_dates, n_symbols, len(DAILY_RAW_FEATURES)), fill_value=np.nan)
    daily_state = _fill_float_memmap(panel_dir / "daily_state.float32.dat", (n_dates, n_symbols, len(RAW_SIGNAL_COLUMNS)), fill_value=np.nan)
    intraday_summary = _fill_float_memmap(
        panel_dir / "intraday_summary.float32.dat", (n_dates, n_symbols, len(INTRADAY_SIGNAL_COLUMNS)), fill_value=np.nan
    )
    limit_structure = _fill_float_memmap(
        panel_dir / "limit_structure.float32.dat", (n_dates, n_symbols, len(LIMIT_SIGNAL_COLUMNS)), fill_value=np.nan
    )
    up_limit_panel = _fill_float_memmap(label_dir / "entry_up_limit.float32.dat", (n_dates, n_symbols), fill_value=np.nan)

    _write_json(progress_path, {"status": "writing_daily_panels", "updated_at": _now()})
    daily_with_state = _add_raw_daily_signals(daily.copy())
    _write_panel_values(daily_raw, daily_with_state, DAILY_RAW_FEATURES, date_to_idx=date_to_idx, symbol_to_idx=symbol_to_idx)
    _write_panel_values(daily_state, daily_with_state, RAW_SIGNAL_COLUMNS, date_to_idx=date_to_idx, symbol_to_idx=symbol_to_idx)
    del daily_with_state
    gc.collect()

    for domain, columns, feature_columns, panel in [
        ("intraday_daily_features", ["symbol", "trade_date", *INTRADAY_SIGNAL_COLUMNS], INTRADAY_SIGNAL_COLUMNS, intraday_summary),
        ("limit_intraday_features", ["symbol", "trade_date", *LIMIT_SIGNAL_COLUMNS], LIMIT_SIGNAL_COLUMNS, limit_structure),
    ]:
        for year in range(int(date_values[0][:4]), int(date_values[-1][:4]) + 1):
            _write_json(progress_path, {"status": f"writing_{domain}", "year": year, "updated_at": _now()})
            frame = _read_dataset_date_range(root, active, domain, columns, f"{year}-01-01", f"{year}-12-31")
            _write_panel_values(panel, frame, feature_columns, date_to_idx=date_to_idx, symbol_to_idx=symbol_to_idx)
            del frame
            gc.collect()

    for year in range(int(date_values[0][:4]), int(date_values[-1][:4]) + 1):
        _write_json(progress_path, {"status": "writing_entry_limits", "year": year, "updated_at": _now()})
        limit = _read_dataset_date_range(root, active, "limit_status", ["symbol", "trade_date", "up_limit"], f"{year}-01-01", f"{year}-12-31")
        if not limit.empty:
            limit = _index_frame(limit, date_to_idx, symbol_to_idx)
            values = pd.to_numeric(limit["up_limit"], errors="coerce").to_numpy(dtype=np.float32, copy=True)
            up_limit_panel[limit["_date_idx"].to_numpy(dtype=np.int64), limit["_symbol_idx"].to_numpy(dtype=np.int64)] = values
            up_limit_panel.flush()
        del limit
        gc.collect()

    _write_json(progress_path, {"status": "computing_labels", "updated_at": _now()})
    summary_columns = path_summary_columns(config.forward_days)
    future_path_store = (
        _fill_float_memmap(
            label_dir / "future_ohlc_path.float32.dat",
            (n_dates, n_symbols, int(config.forward_days), 4),
            fill_value=np.nan,
        )
        if bool(config.write_legacy_ohlc_label)
        else None
    )
    future_ohlcva_shape = (n_dates, n_symbols, int(config.forward_days), 6)
    future_ohlcva_path_store = (
        DateShardedFloatStore(
            directory=label_dir / "future_ohlcva_path_shards",
            name="future_ohlcva_path",
            shape=future_ohlcva_shape,
            shard_size=int(config.label_shard_size),
        )
        if int(config.label_shard_size) > 0
        else _fill_float_memmap(
            label_dir / "future_ohlcva_path.float32.dat",
            future_ohlcva_shape,
            fill_value=np.nan,
        )
    )
    path_summary_store = _fill_float_memmap(
        label_dir / "path_summary.float32.dat",
        (n_dates, n_symbols, len(summary_columns)),
        fill_value=np.nan,
    )
    future_path, future_ohlcva_path, path_summary, input_valid, entry_buyable, label_valid = _compute_future_path_and_masks(
        raw_panel=daily_raw,
        up_limit_panel=up_limit_panel,
        lookback_days=int(config.lookback_days),
        forward_days=int(config.forward_days),
        price_anchor=str(config.price_anchor),
        future_path_out=future_path_store,
        future_ohlcva_path_out=future_ohlcva_path_store,
        path_summary_out=path_summary_store,
        write_legacy_ohlc_path=bool(config.write_legacy_ohlc_label),
    )
    if future_path is not None:
        future_path.flush()
    future_ohlcva_path.flush()
    path_summary_store.flush()
    input_valid_store = _fill_bool_memmap(mask_dir / "input_valid.bool.dat", tuple(int(item) for item in input_valid.shape))
    entry_buyable_store = _fill_bool_memmap(mask_dir / "entry_buyable.bool.dat", tuple(int(item) for item in entry_buyable.shape))
    label_valid_store = _fill_bool_memmap(mask_dir / "label_valid.bool.dat", tuple(int(item) for item in label_valid.shape))
    input_valid_store[:] = input_valid
    entry_buyable_store[:] = entry_buyable
    label_valid_store[:] = label_valid
    input_valid_store.flush()
    entry_buyable_store.flush()
    label_valid_store.flush()

    _write_json(progress_path, {"status": "building_sample_index", "updated_at": _now()})
    sample_index = _build_sample_index(
        date_values=date_values,
        symbol_values=symbol_values,
        start_date=config.start_date,
        end_date=config.end_date,
        train_years=config.train_years,
        validation_years=config.validation_years,
        test_years=config.test_years,
        input_valid=input_valid,
        entry_buyable=entry_buyable,
        label_valid=label_valid,
    )
    sample_index_path = output_dir / "sample_index.parquet"
    sample_index.to_parquet(sample_index_path, index=False)

    train_date_mask = np.array([int(date[:4]) in set(config.train_years) for date in date_values], dtype=bool)
    normalization = {
        "fit_scope": "train_year_dates_only",
        "daily_raw": _fit_normalization(daily_raw, train_date_mask),
        "daily_state": _fit_normalization(daily_state, train_date_mask),
        "intraday_summary": _fit_normalization(intraday_summary, train_date_mask),
        "limit_structure": _fit_normalization(limit_structure, train_date_mask),
    }

    active_dataset_ids = dict(active.get("datasets", {}) or {})
    split_counts = sample_index["split"].value_counts().to_dict() if not sample_index.empty else {}
    path_anchor_name = "signal_day_close" if str(config.price_anchor) == "today_close" else "next_calendar_trading_day_open"
    if isinstance(future_ohlcva_path_store, DateShardedFloatStore):
        future_ohlcva_meta = future_ohlcva_path_store.manifest(
            fields=PATH_OHLCVA_FIELDS,
            anchor=path_anchor_name,
            extra={
                "price_anchor": str(config.price_anchor),
                "volume_amount_transform": "log1p(future_value) - trailing_20d_mean_log1p(value)_through_signal_date",
            },
        )
    else:
        future_ohlcva_meta = {
            "path": str((label_dir / "future_ohlcva_path.float32.dat").resolve()),
            "shape": [n_dates, n_symbols, int(config.forward_days), 6],
            "fields": PATH_OHLCVA_FIELDS,
            "anchor": path_anchor_name,
            "price_anchor": str(config.price_anchor),
            "volume_amount_transform": "log1p(future_value) - trailing_20d_mean_log1p(value)_through_signal_date",
        }
    label_arrays = {
        "future_ohlcva_path": future_ohlcva_meta,
        "path_summary": {
            "path": str((label_dir / "path_summary.float32.dat").resolve()),
            "shape": [n_dates, n_symbols, len(summary_columns)],
            "columns": summary_columns,
        },
    }
    if bool(config.write_legacy_ohlc_label):
        label_arrays = {
            "future_ohlc_path": {
                "path": str((label_dir / "future_ohlc_path.float32.dat").resolve()),
                "shape": [n_dates, n_symbols, int(config.forward_days), 4],
                "fields": PATH_OHLC_FIELDS,
                "anchor": path_anchor_name,
                "price_anchor": str(config.price_anchor),
            },
            **label_arrays,
        }
    manifest = {
        "artifact_type": "qdp_v2_sequence_path_pack",
        "created_at": _now(),
        "qdp_root": str(root),
        "active_manifest": str((root / "active" / "active.json").resolve()),
        "active_as_of_date": active.get("active_as_of_date", ""),
        "active_datasets": active_dataset_ids,
        "scope": active_scope,
        "lookback_days": int(config.lookback_days),
        "forward_days": int(config.forward_days),
        "start_date": str(config.start_date),
        "end_date": str(config.end_date),
        "train_years": list(config.train_years),
        "validation_years": list(config.validation_years),
        "test_years": list(config.test_years),
        "date_values": date_values,
        "symbol_values": symbol_values,
        "date_count": int(n_dates),
        "symbol_count": int(n_symbols),
        "sample_count": int(len(sample_index)),
        "sample_count_by_split": {str(key): int(value) for key, value in split_counts.items()},
        "feature_channels": {
            "daily_raw": {"path": str((panel_dir / "daily_raw.float32.dat").resolve()), "shape": [n_dates, n_symbols, len(DAILY_RAW_FEATURES)], "columns": DAILY_RAW_FEATURES},
            "daily_state": {"path": str((panel_dir / "daily_state.float32.dat").resolve()), "shape": [n_dates, n_symbols, len(RAW_SIGNAL_COLUMNS)], "columns": RAW_SIGNAL_COLUMNS},
            "intraday_summary": {
                "path": str((panel_dir / "intraday_summary.float32.dat").resolve()),
                "shape": [n_dates, n_symbols, len(INTRADAY_SIGNAL_COLUMNS)],
                "columns": INTRADAY_SIGNAL_COLUMNS,
            },
            "limit_structure": {
                "path": str((panel_dir / "limit_structure.float32.dat").resolve()),
                "shape": [n_dates, n_symbols, len(LIMIT_SIGNAL_COLUMNS)],
                "columns": LIMIT_SIGNAL_COLUMNS,
            },
        },
        "label_arrays": label_arrays,
        "masks": {
            "input_valid": {"path": str((mask_dir / "input_valid.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "entry_buyable": {"path": str((mask_dir / "entry_buyable.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "label_valid": {"path": str((mask_dir / "label_valid.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
        },
        "sample_index_path": str(sample_index_path.resolve()),
        "normalization": normalization,
        "label_semantics": {
            "entry_anchor": "signal day close decision, next calendar trading day open entry",
            "price_anchor": str(config.price_anchor),
            "future_ohlc_path": f"OHLC returns are relative to {path_anchor_name}.",
            "future_ohlcva_path": f"OHLC returns are relative to {path_anchor_name}; volume and amount are log-relative to trailing 20 trading days ending on signal date.",
            "path_trade_value_v2": "derived from next calendar trading day open entry even when price_anchor=today_close",
            path_value_column(config.forward_days): "future_final_return + 0.50*future_max_return + 0.35*future_min_return + 0.20*drawdown_after_peak",
            "path_type_labels": "derived_explanation_only_not_primary_training_target",
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(progress_path, {"status": "completed", "manifest_json": str(manifest_path.resolve()), "updated_at": _now()})
    return manifest


def _hardlink_file(source: Path, target: Path, *, overwrite: bool = False) -> Path:
    source = source.resolve()
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not overwrite:
            return target
        target.unlink()
    try:
        os.link(source, target)
        return target
    except OSError:
        return source


def reanchor_sequence_pack(
    *,
    source_manifest: str | Path,
    output_root: Path,
    run_tag: str,
    price_anchor: str,
    write_legacy_ohlc_label: bool = True,
    label_shard_size: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = Path(source_manifest).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("artifact_type") != "qdp_v2_sequence_path_pack":
        raise ValueError(f"not a sequence path pack manifest: {source_path}")
    output_dir = output_root / str(run_tag)
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(overwrite):
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    _write_json(progress_path, {"status": "started", "source_manifest": str(source_path), "updated_at": _now()})
    n_dates = int(source["date_count"])
    n_symbols = int(source["symbol_count"])
    forward_days = int(source["forward_days"])
    lookback_days = int(source["lookback_days"])
    panel_dir = output_dir / "panels"
    mask_dir = output_dir / "masks"
    label_dir = output_dir / "labels"

    _write_json(progress_path, {"status": "linking_inputs", "updated_at": _now()})
    feature_channels: dict[str, Any] = {}
    for name, meta in dict(source.get("feature_channels", {}) or {}).items():
        src = Path(str(meta["path"]))
        dst = _hardlink_file(src, panel_dir / src.name, overwrite=overwrite)
        copied = dict(meta)
        copied["path"] = str(dst.resolve())
        feature_channels[str(name)] = copied
    masks: dict[str, Any] = {}
    for name, meta in dict(source.get("masks", {}) or {}).items():
        src = Path(str(meta["path"]))
        dst = _hardlink_file(src, mask_dir / src.name, overwrite=overwrite)
        copied = dict(meta)
        copied["path"] = str(dst.resolve())
        masks[str(name)] = copied
    sample_index_src = Path(str(source["sample_index_path"]))
    sample_index_path = _hardlink_file(sample_index_src, output_dir / "sample_index.parquet", overwrite=overwrite)
    source_label_dir = Path(str(source_path.parent / "labels"))
    entry_up_limit_src = source_label_dir / "entry_up_limit.float32.dat"
    if not entry_up_limit_src.exists():
        raise FileNotFoundError(f"source pack is missing entry_up_limit label: {entry_up_limit_src}")
    entry_up_limit_path = _hardlink_file(entry_up_limit_src, label_dir / "entry_up_limit.float32.dat", overwrite=overwrite)

    daily_raw_meta = feature_channels["daily_raw"]
    raw_panel = np.memmap(
        daily_raw_meta["path"],
        dtype="float32",
        mode="r",
        shape=tuple(int(item) for item in daily_raw_meta["shape"]),
    )
    up_limit_panel = np.memmap(entry_up_limit_path, dtype="float32", mode="r", shape=(n_dates, n_symbols))

    _write_json(progress_path, {"status": "computing_labels", "updated_at": _now()})
    summary_columns = path_summary_columns(forward_days)
    future_path_store = (
        _fill_float_memmap(label_dir / "future_ohlc_path.float32.dat", (n_dates, n_symbols, forward_days, 4), fill_value=np.nan)
        if bool(write_legacy_ohlc_label)
        else None
    )
    future_ohlcva_shape = (n_dates, n_symbols, forward_days, 6)
    future_ohlcva_path_store = (
        DateShardedFloatStore(
            directory=label_dir / "future_ohlcva_path_shards",
            name="future_ohlcva_path",
            shape=future_ohlcva_shape,
            shard_size=int(label_shard_size),
        )
        if int(label_shard_size) > 0
        else _fill_float_memmap(label_dir / "future_ohlcva_path.float32.dat", future_ohlcva_shape, fill_value=np.nan)
    )
    path_summary_store = _fill_float_memmap(label_dir / "path_summary.float32.dat", (n_dates, n_symbols, len(summary_columns)), fill_value=np.nan)
    future_path, future_ohlcva_path, path_summary, _input_valid, _entry_buyable, _label_valid = _compute_future_path_and_masks(
        raw_panel=raw_panel,
        up_limit_panel=up_limit_panel,
        lookback_days=lookback_days,
        forward_days=forward_days,
        price_anchor=str(price_anchor),
        future_path_out=future_path_store,
        future_ohlcva_path_out=future_ohlcva_path_store,
        path_summary_out=path_summary_store,
        write_legacy_ohlc_path=bool(write_legacy_ohlc_label),
    )
    if future_path is not None:
        future_path.flush()
    future_ohlcva_path.flush()
    path_summary.flush()
    path_anchor_name = "signal_day_close" if str(price_anchor) == "today_close" else "next_calendar_trading_day_open"
    if isinstance(future_ohlcva_path_store, DateShardedFloatStore):
        future_ohlcva_meta = future_ohlcva_path_store.manifest(
            fields=PATH_OHLCVA_FIELDS,
            anchor=path_anchor_name,
            extra={
                "price_anchor": str(price_anchor),
                "volume_amount_transform": "log1p(future_value) - trailing_20d_mean_log1p(value)_through_signal_date",
            },
        )
    else:
        future_ohlcva_meta = {
            "path": str((label_dir / "future_ohlcva_path.float32.dat").resolve()),
            "shape": [n_dates, n_symbols, forward_days, 6],
            "fields": PATH_OHLCVA_FIELDS,
            "anchor": path_anchor_name,
            "price_anchor": str(price_anchor),
            "volume_amount_transform": "log1p(future_value) - trailing_20d_mean_log1p(value)_through_signal_date",
        }
    label_arrays: dict[str, Any] = {
        "future_ohlcva_path": future_ohlcva_meta,
        "path_summary": {
            "path": str((label_dir / "path_summary.float32.dat").resolve()),
            "shape": [n_dates, n_symbols, len(summary_columns)],
            "columns": summary_columns,
        },
    }
    if bool(write_legacy_ohlc_label):
        label_arrays = {
            "future_ohlc_path": {
                "path": str((label_dir / "future_ohlc_path.float32.dat").resolve()),
                "shape": [n_dates, n_symbols, forward_days, 4],
                "fields": PATH_OHLC_FIELDS,
                "anchor": path_anchor_name,
                "price_anchor": str(price_anchor),
            },
            **label_arrays,
        }
    manifest = dict(source)
    manifest.update(
        {
            "created_at": _now(),
            "derived_from_pack_manifest": str(source_path),
            "sample_index_path": str(sample_index_path.resolve()),
            "feature_channels": feature_channels,
            "label_arrays": label_arrays,
            "masks": masks,
            "label_semantics": {
                **dict(source.get("label_semantics", {}) or {}),
                "price_anchor": str(price_anchor),
                "future_ohlc_path": f"OHLC returns are relative to {path_anchor_name}.",
                "future_ohlcva_path": f"OHLC returns are relative to {path_anchor_name}; volume and amount are log-relative to trailing 20 trading days ending on signal date.",
                "path_trade_value_v2": "derived from next calendar trading day open entry even when price_anchor=today_close",
            },
        }
    )
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(progress_path, {"status": "completed", "manifest_json": str(manifest_path.resolve()), "updated_at": _now()})
    return manifest


def validate_sequence_pack(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    blockers: list[str] = []
    if manifest.get("artifact_type") != "qdp_v2_sequence_path_pack":
        blockers.append("not_qdp_v2_sequence_path_pack")
    for section in ["feature_channels", "label_arrays", "masks"]:
        for name, meta in dict(manifest.get(section, {}) or {}).items():
            dtype = "bool" if section == "masks" else "float32"
            shards = list(meta.get("shards", []) or [])
            if shards:
                for shard in shards:
                    file_path = Path(str(shard.get("path", "") or ""))
                    shape = tuple(int(item) for item in list(shard.get("shape", []) or []))
                    if not file_path.exists():
                        blockers.append(f"missing_{section}_{name}_shard")
                        continue
                    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
                    actual = int(file_path.stat().st_size)
                    if actual != expected:
                        blockers.append(f"size_mismatch_{section}_{name}_shard:{actual}!={expected}")
                continue
            file_path = Path(str(meta.get("path", "") or ""))
            shape = tuple(int(item) for item in list(meta.get("shape", []) or []))
            if not file_path.exists():
                blockers.append(f"missing_{section}_{name}")
                continue
            expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
            actual = int(file_path.stat().st_size)
            if actual != expected:
                blockers.append(f"size_mismatch_{section}_{name}:{actual}!={expected}")
    sample_index_path = Path(str(manifest.get("sample_index_path", "") or ""))
    if not sample_index_path.exists():
        blockers.append("missing_sample_index")
    return {
        "status": "blocked" if blockers else "ok",
        "blockers": blockers,
        "manifest_path": str(path.resolve()),
        "sample_count": int(manifest.get("sample_count", 0) or 0),
        "sample_count_by_split": dict(manifest.get("sample_count_by_split", {}) or {}),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate QDP v2 sequence path packs.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--qdp-root", type=Path, default=DEFAULT_QDP_ROOT)
    build.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    build.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    build.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    build.add_argument("--forward-days", type=int, default=DEFAULT_FORWARD_DAYS)
    build.add_argument("--start-date", default=DEFAULT_START_DATE)
    build.add_argument("--end-date", default=DEFAULT_END_DATE)
    build.add_argument("--train-years", default="2012-2023")
    build.add_argument("--validation-years", default="2024")
    build.add_argument("--test-years", default="2025")
    build.add_argument("--no-legacy-ohlc-label", action="store_true", help="Do not write the separate future_ohlc_path label; OHLC is available as the first four OHLCVA fields.")
    build.add_argument("--label-shard-size", type=int, default=0, help="Write future_ohlcva_path as date shards of this many dates; 0 writes a single memmap file.")
    build.add_argument(
        "--price-anchor",
        choices=("next_open", "today_close"),
        default="next_open",
        help="Anchor future OHLC labels to next trading day open or signal-day close.",
    )
    build.add_argument("--json", action="store_true")
    reanchor = sub.add_parser("reanchor")
    reanchor.add_argument("--source-manifest", type=Path, required=True)
    reanchor.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    reanchor.add_argument("--run-tag", required=True)
    reanchor.add_argument(
        "--price-anchor",
        choices=("next_open", "today_close"),
        required=True,
        help="New future OHLC label anchor.",
    )
    reanchor.add_argument("--no-legacy-ohlc-label", action="store_true")
    reanchor.add_argument("--label-shard-size", type=int, default=0)
    reanchor.add_argument("--overwrite", action="store_true")
    reanchor.add_argument("--json", action="store_true")
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        cfg = SequencePackConfig(
            qdp_root=Path(args.qdp_root),
            output_root=Path(args.output_root),
            run_tag=str(args.run_tag),
            lookback_days=int(args.lookback_days),
            forward_days=int(args.forward_days),
            start_date=str(args.start_date),
            end_date=str(args.end_date),
            train_years=_parse_years(args.train_years, default=DEFAULT_TRAIN_YEARS),
            validation_years=_parse_years(args.validation_years, default=DEFAULT_VALIDATION_YEARS),
            test_years=_parse_years(args.test_years, default=DEFAULT_TEST_YEARS),
            write_legacy_ohlc_label=not bool(args.no_legacy_ohlc_label),
            label_shard_size=int(args.label_shard_size),
            price_anchor=str(args.price_anchor),
        )
        result = build_sequence_pack(cfg)
    elif args.command == "reanchor":
        result = reanchor_sequence_pack(
            source_manifest=Path(args.source_manifest),
            output_root=Path(args.output_root),
            run_tag=str(args.run_tag),
            price_anchor=str(args.price_anchor),
            write_legacy_ohlc_label=not bool(args.no_legacy_ohlc_label),
            label_shard_size=int(args.label_shard_size),
            overwrite=bool(args.overwrite),
        )
    else:
        result = validate_sequence_pack(args.manifest)
    if bool(getattr(args, "json", False)):
        print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    else:
        if args.command in {"build", "reanchor"}:
            payload = {
                "status": "completed",
                "manifest_json": str((Path(args.output_root) / str(args.run_tag) / "manifest.json").resolve()),
                "sample_count": int(result.get("sample_count", 0) or 0),
                "sample_count_by_split": dict(result.get("sample_count_by_split", {}) or {}),
            }
        else:
            payload = result
        print(json.dumps(payload, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
