from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from daily_research.path_policy.qdp_v2_raw_rising_path_atlas import (
    DAILY_RAW_COLUMNS,
    DEFAULT_QDP_ROOT,
    INTRADAY_COLUMNS,
    INTRADAY_SIGNAL_COLUMNS,
    LIMIT_COLUMNS,
    LIMIT_SIGNAL_COLUMNS,
    RAW_SIGNAL_COLUMNS,
    _add_raw_daily_signals,
    _read_dataset,
    _relative_shard_paths,
)


DEFAULT_OUTPUT_ROOT = Path("daily_research/output/path_policy/stock_path_profile_atlas")
DEFAULT_EVENT_START = "2012-01-01"
DEFAULT_EVENT_END = "2025-12-02"
DEFAULT_FORWARD_DAYS = 60
DEFAULT_PRE_DAYS = 60
DEFAULT_CLUSTER_COUNT = 8
DEFAULT_RANDOM_SEED = 7


PATH_CLUSTER_FEATURES = [
    "future_max_return_60d",
    "future_min_return_60d",
    "future_final_return_60d",
    "future_peak_day_60d",
    "future_trough_day_60d",
    "drawdown_after_peak_60d",
    "runup_after_trough_60d",
    "path_range_60d",
    "path_efficiency_60d",
    "time_above_zero_60d",
    "time_below_zero_60d",
]


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
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


def _write_csv(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path.resolve())


def _parse_year_range(raw: str, *, default: tuple[int, ...]) -> tuple[int, ...]:
    values: list[int] = []
    for chunk in str(raw or "").split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start, end = [int(part) for part in item.split("-", 1)]
            values.extend(range(start, end + 1))
        else:
            values.append(int(item))
    return tuple(sorted(set(values))) or default


def _read_active(root: Path) -> dict[str, Any]:
    return json.loads((root / "active" / "active.json").read_text(encoding="utf-8"))


def _read_dataset_date_range(root: Path, active: Mapping[str, Any], domain: str, columns: list[str], start: str, end: str) -> pd.DataFrame:
    dataset_id = str(dict(active.get("datasets", {}) or {}).get(domain, "") or "")
    if not dataset_id:
        return pd.DataFrame(columns=columns)
    manifest_path = root / "datasets" / domain / dataset_id / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    available = {str(item.get("name", "")) for item in list(manifest.get("schema", []) or [])}
    selected = [col for col in columns if col in available]
    paths = _relative_shard_paths(root, manifest)
    filt = (ds.field("trade_date") >= str(start)) & (ds.field("trade_date") <= str(end))
    table = ds.dataset([str(path) for path in paths], format="parquet").to_table(columns=selected, filter=filt)
    frame = table.to_pandas()
    if "trade_date" in frame.columns:
        frame["trade_date"] = frame["trade_date"].astype(str)
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    return frame


def _load_daily_base(root: Path, active: Mapping[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    daily = _read_dataset(root, active, "market_daily_raw", DAILY_RAW_COLUMNS)
    daily = daily[(daily["trade_date"] >= "2011-11-22") & (daily["trade_date"] <= "2026-06-26")].copy()
    daily = daily.sort_values(["symbol", "trade_date"], kind="mergesort").reset_index(drop=True)
    daily["_row_pos"] = np.arange(len(daily), dtype=np.int64)
    calendar = _read_dataset(root, active, "trading_calendar", ["trade_date", "is_open"])
    open_days = calendar[calendar["is_open"].astype(bool)]["trade_date"].astype(str).sort_values(kind="mergesort").to_list()
    date_to_pos = {date: idx for idx, date in enumerate(open_days)}
    daily["calendar_pos"] = daily["trade_date"].map(date_to_pos).fillna(-1).astype("int32")
    symbol_codes = pd.factorize(daily["symbol"], sort=False)[0].astype(np.int32, copy=False)
    calendar_pos = daily["calendar_pos"].to_numpy(dtype=np.int32, copy=False)
    return daily, symbol_codes, calendar_pos


def _first_hit_day(path: np.ndarray, threshold: float, *, direction: str) -> np.ndarray:
    if direction == "up":
        hit = np.asarray(path >= float(threshold), dtype=bool)
    elif direction == "down":
        hit = np.asarray(path <= -abs(float(threshold)), dtype=bool)
    else:
        raise ValueError(direction)
    finite = np.isfinite(path)
    hit &= finite
    any_hit = hit.any(axis=1)
    first = np.argmax(hit, axis=1).astype(np.int16) + 1
    first[~any_hit] = 0
    return first


def _path_arrays_for_rows(
    *,
    daily: pd.DataFrame,
    row_pos: np.ndarray,
    symbol_codes: np.ndarray,
    calendar_pos: np.ndarray,
    forward_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_pos = np.asarray(row_pos, dtype=np.int64)
    n = int(len(row_pos))
    horizon = int(forward_days)
    open_values = daily["open"].to_numpy(dtype=np.float64, copy=False)
    high_values = daily["high"].to_numpy(dtype=np.float64, copy=False)
    low_values = daily["low"].to_numpy(dtype=np.float64, copy=False)
    close_values = daily["close"].to_numpy(dtype=np.float64, copy=False)
    entry_target = row_pos + 1
    entry_in_bounds = entry_target < len(daily)
    safe_entry_target = np.where(entry_in_bounds, entry_target, 0)
    entry_valid = (
        entry_in_bounds
        & (symbol_codes[safe_entry_target] == symbol_codes[row_pos])
        & (calendar_pos[safe_entry_target] == calendar_pos[row_pos] + 1)
    )
    entry_open = np.full(n, np.nan, dtype=np.float64)
    entry_open[entry_valid] = open_values[safe_entry_target[entry_valid]]
    high_ret = np.full((n, horizon), np.nan, dtype=np.float32)
    low_ret = np.full((n, horizon), np.nan, dtype=np.float32)
    close_ret = np.full((n, horizon), np.nan, dtype=np.float32)
    for day in range(1, horizon + 1):
        target = row_pos + day
        in_bounds = target < len(daily)
        safe_target = np.where(in_bounds, target, 0)
        valid = (
            entry_valid
            & in_bounds
            & (symbol_codes[safe_target] == symbol_codes[row_pos])
            & (calendar_pos[safe_target] == calendar_pos[row_pos] + day)
            & np.isfinite(entry_open)
            & (entry_open > 0)
        )
        if not np.any(valid):
            continue
        idx = day - 1
        valid_target = safe_target[valid]
        denom = entry_open[valid]
        high_ret[valid, idx] = (high_values[valid_target] / denom - 1.0).astype(np.float32, copy=False)
        low_ret[valid, idx] = (low_values[valid_target] / denom - 1.0).astype(np.float32, copy=False)
        close_ret[valid, idx] = (close_values[valid_target] / denom - 1.0).astype(np.float32, copy=False)
    return high_ret, low_ret, close_ret, entry_open.astype(np.float32), entry_valid, entry_target.astype(np.int64)


def _compute_path_metrics(
    *,
    daily: pd.DataFrame,
    year: int,
    event_start: str,
    event_end: str,
    symbol_codes: np.ndarray,
    calendar_pos: np.ndarray,
    forward_days: int,
    root: Path,
    active: Mapping[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    mask = (
        daily["trade_date"].str.slice(0, 4).astype(int).eq(int(year))
        & (daily["trade_date"] >= str(event_start))
        & (daily["trade_date"] <= str(event_end))
        & (daily["calendar_pos"] >= 0)
    )
    base = daily.loc[mask, ["symbol", "trade_date", "_row_pos", "calendar_pos"]].copy().reset_index(drop=True)
    row_pos = base["_row_pos"].to_numpy(dtype=np.int64, copy=True)
    high_ret, low_ret, close_ret, entry_open, entry_valid, entry_target = _path_arrays_for_rows(
        daily=daily,
        row_pos=row_pos,
        symbol_codes=symbol_codes,
        calendar_pos=calendar_pos,
        forward_days=forward_days,
    )
    observed = np.isfinite(close_ret).sum(axis=1).astype(np.int16)
    valid_path = (observed == int(forward_days)) & np.isfinite(high_ret).all(axis=1) & np.isfinite(low_ret).all(axis=1)
    high_filled = np.where(np.isfinite(high_ret), high_ret, -np.inf)
    low_filled = np.where(np.isfinite(low_ret), low_ret, np.inf)
    close_finite = np.isfinite(close_ret)
    max_ret = np.max(high_filled, axis=1).astype(np.float32)
    min_ret = np.min(low_filled, axis=1).astype(np.float32)
    max_ret[observed == 0] = np.nan
    min_ret[observed == 0] = np.nan
    peak_idx = np.argmax(high_filled, axis=1).astype(np.int16)
    trough_idx = np.argmin(low_filled, axis=1).astype(np.int16)
    peak_day = peak_idx + 1
    trough_day = trough_idx + 1
    peak_day[observed == 0] = 0
    trough_day[observed == 0] = 0
    final_ret = close_ret[:, int(forward_days) - 1].astype(np.float32)
    min_after_peak = np.full(len(base), np.inf, dtype=np.float32)
    max_after_trough = np.full(len(base), -np.inf, dtype=np.float32)
    for idx in range(int(forward_days)):
        low_values = low_ret[:, idx]
        high_values = high_ret[:, idx]
        min_after_peak = np.where((peak_idx <= idx) & np.isfinite(low_values), np.minimum(min_after_peak, low_values), min_after_peak)
        max_after_trough = np.where((trough_idx <= idx) & np.isfinite(high_values), np.maximum(max_after_trough, high_values), max_after_trough)
    drawdown_after_peak = (1.0 + min_after_peak) / (1.0 + max_ret) - 1.0
    runup_after_trough = (1.0 + max_after_trough) / (1.0 + min_ret) - 1.0
    drawdown_after_peak[~np.isfinite(min_after_peak)] = np.nan
    runup_after_trough[~np.isfinite(max_after_trough)] = np.nan
    path_range = max_ret - min_ret
    path_efficiency = final_ret / np.where(np.abs(path_range) > 1e-6, path_range, np.nan)
    time_above = np.divide((close_ret > 0).sum(axis=1), np.maximum(observed, 1), out=np.zeros(len(base), dtype=np.float64), where=observed > 0)
    time_below = np.divide((close_ret < 0).sum(axis=1), np.maximum(observed, 1), out=np.zeros(len(base), dtype=np.float64), where=observed > 0)
    for col, values in [
        ("entry_open_next", entry_open),
        ("future_max_return_60d", max_ret),
        ("future_min_return_60d", min_ret),
        ("future_final_return_60d", final_ret),
        ("drawdown_after_peak_60d", drawdown_after_peak),
        ("runup_after_trough_60d", runup_after_trough),
        ("path_range_60d", path_range),
        ("path_efficiency_60d", path_efficiency),
        ("time_above_zero_60d", time_above),
        ("time_below_zero_60d", time_below),
    ]:
        base[col] = values
    base["year"] = int(year)
    base["entry_valid"] = entry_valid
    trade_dates = daily["trade_date"].to_numpy(dtype=object)
    safe_entry_dates = np.full(len(base), "", dtype=object)
    safe_entry_dates[entry_valid] = trade_dates[entry_target[entry_valid]]
    base["entry_trade_date"] = safe_entry_dates
    base["observed_future_days_60d"] = observed
    base["path_valid_60d"] = valid_path
    base["future_peak_day_60d"] = peak_day
    base["future_trough_day_60d"] = trough_day
    base["time_to_profit_3pct"] = _first_hit_day(high_ret, 0.03, direction="up")
    base["time_to_profit_5pct"] = _first_hit_day(high_ret, 0.05, direction="up")
    base["time_to_profit_10pct"] = _first_hit_day(high_ret, 0.10, direction="up")
    base["time_to_profit_20pct"] = _first_hit_day(high_ret, 0.20, direction="up")
    base["time_to_loss_3pct"] = _first_hit_day(low_ret, 0.03, direction="down")
    base["time_to_loss_5pct"] = _first_hit_day(low_ret, 0.05, direction="down")
    base["time_to_loss_10pct"] = _first_hit_day(low_ret, 0.10, direction="down")
    base["path_trade_value_60d"] = (
        base["future_final_return_60d"].astype("float64")
        + 0.50 * base["future_max_return_60d"].astype("float64")
        + 0.35 * base["future_min_return_60d"].astype("float64")
        + 0.20 * base["drawdown_after_peak_60d"].astype("float64")
    ).astype("float32")

    limit_status = _read_dataset_date_range(root, active, "limit_status", ["symbol", "trade_date", "up_limit"], f"{year}-01-01", f"{year + 1}-03-31")
    if not limit_status.empty:
        limit_status = limit_status.rename(columns={"trade_date": "entry_trade_date", "up_limit": "entry_up_limit"})
        base = base.merge(limit_status, on=["symbol", "entry_trade_date"], how="left", validate="many_to_one")
    else:
        base["entry_up_limit"] = np.nan
    base["entry_open_limit_blocked"] = (
        pd.to_numeric(base["entry_up_limit"], errors="coerce").notna()
        & pd.to_numeric(base["entry_open_next"], errors="coerce").notna()
        & (pd.to_numeric(base["entry_open_next"], errors="coerce") >= pd.to_numeric(base["entry_up_limit"], errors="coerce") * 0.999)
    )
    base["entry_buyable"] = base["entry_valid"].astype(bool) & (~base["entry_open_limit_blocked"].astype(bool))
    return base, high_ret, low_ret, close_ret


def _write_year_path_metrics(
    *,
    daily: pd.DataFrame,
    symbol_codes: np.ndarray,
    calendar_pos: np.ndarray,
    root: Path,
    active: Mapping[str, Any],
    years: tuple[int, ...],
    event_start: str,
    event_end: str,
    forward_days: int,
    output_dir: Path,
    progress_path: Path,
) -> list[Path]:
    shard_dir = output_dir / "path_metric_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for year in years:
        _write_json(progress_path, {"status": "path_metrics", "year": int(year), "updated_at": _now()})
        frame, _high, _low, _close = _compute_path_metrics(
            daily=daily,
            year=int(year),
            event_start=event_start,
            event_end=event_end,
            symbol_codes=symbol_codes,
            calendar_pos=calendar_pos,
            forward_days=forward_days,
            root=root,
            active=active,
        )
        path = shard_dir / f"path_metrics_year_{int(year)}.parquet"
        frame.to_parquet(path, index=False)
        paths.append(path)
        del frame, _high, _low, _close
        gc.collect()
    return paths


def _fit_path_clusters(shard_paths: list[Path], *, cluster_count: int, seed: int, progress_path: Path) -> tuple[Any, Any]:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    fitted_rows = 0
    for path in shard_paths:
        frame = pd.read_parquet(path, columns=[*PATH_CLUSTER_FEATURES, "path_valid_60d", "entry_valid"])
        frame = frame[frame["path_valid_60d"].astype(bool) & frame["entry_valid"].astype(bool)]
        values = frame[PATH_CLUSTER_FEATURES].to_numpy(dtype=np.float32, copy=True)
        finite = np.isfinite(values).all(axis=1)
        if np.any(finite):
            scaler.partial_fit(values[finite])
            fitted_rows += int(finite.sum())
        _write_json(progress_path, {"status": "cluster_scaler", "fitted_rows": fitted_rows, "last_shard": str(path), "updated_at": _now()})
    model = MiniBatchKMeans(n_clusters=int(cluster_count), random_state=int(seed), batch_size=65536, n_init=3)
    for epoch in range(2):
        trained_rows = 0
        for path in shard_paths:
            frame = pd.read_parquet(path, columns=[*PATH_CLUSTER_FEATURES, "path_valid_60d", "entry_valid"])
            frame = frame[frame["path_valid_60d"].astype(bool) & frame["entry_valid"].astype(bool)]
            values = frame[PATH_CLUSTER_FEATURES].to_numpy(dtype=np.float32, copy=True)
            finite = np.isfinite(values).all(axis=1)
            if np.any(finite):
                model.partial_fit(scaler.transform(values[finite]))
                trained_rows += int(finite.sum())
        _write_json(progress_path, {"status": "cluster_kmeans", "epoch": int(epoch + 1), "trained_rows": trained_rows, "updated_at": _now()})
    return scaler, model


def _label_clusters(summary: pd.DataFrame) -> dict[int, str]:
    if summary.empty:
        return {}
    frame = summary.set_index("path_cluster")
    final_q75 = frame["future_final_return_60d_mean"].quantile(0.75)
    final_q25 = frame["future_final_return_60d_mean"].quantile(0.25)
    max_q75 = frame["future_max_return_60d_mean"].quantile(0.75)
    max_mid = frame["future_max_return_60d_mean"].median()
    min_q25 = frame["future_min_return_60d_mean"].quantile(0.25)
    range_q75 = frame["path_range_60d_mean"].quantile(0.75)
    range_q25 = frame["path_range_60d_mean"].quantile(0.25)
    draw_q25 = frame["drawdown_after_peak_60d_mean"].quantile(0.25)
    labels: dict[int, str] = {}
    for cluster, row in frame.iterrows():
        final = float(row["future_final_return_60d_mean"])
        max_ret = float(row["future_max_return_60d_mean"])
        min_ret = float(row["future_min_return_60d_mean"])
        path_range = float(row["path_range_60d_mean"])
        draw = float(row["drawdown_after_peak_60d_mean"])
        peak_day = float(row["future_peak_day_60d_mean"])
        trough_day = float(row["future_trough_day_60d_mean"])
        if final >= final_q75 and max_ret >= max_mid and draw > draw_q25:
            label = "persistent_up"
        elif max_ret >= max_q75 and (draw <= draw_q25 or final < max_ret * 0.45):
            label = "spike_fade"
        elif final >= frame["future_final_return_60d_mean"].median() and max_ret >= max_mid:
            label = "trend_up"
        elif final <= final_q25 and min_ret <= min_q25 and trough_day <= peak_day:
            label = "downtrend"
        elif final <= final_q25 and min_ret <= min_q25:
            label = "late_down"
        elif path_range >= range_q75 and abs(final) <= abs(frame["future_final_return_60d_mean"]).median():
            label = "volatile_chop"
        elif path_range <= range_q25 and abs(final) <= abs(frame["future_final_return_60d_mean"]).median():
            label = "sideways_compression"
        else:
            label = "mixed_path"
        labels[int(cluster)] = label
    return labels


def _assign_clusters(
    shard_paths: list[Path],
    *,
    scaler: Any,
    model: Any,
    output_dir: Path,
    progress_path: Path,
) -> tuple[list[Path], pd.DataFrame, dict[int, str]]:
    assigned_dir = output_dir / "assigned_path_shards"
    assigned_dir.mkdir(parents=True, exist_ok=True)
    assigned_paths: list[Path] = []
    summary_frames: list[pd.DataFrame] = []
    for path in shard_paths:
        frame = pd.read_parquet(path)
        cluster = np.full(len(frame), -1, dtype=np.int16)
        values = frame[PATH_CLUSTER_FEATURES].to_numpy(dtype=np.float32, copy=True)
        finite = frame["path_valid_60d"].astype(bool).to_numpy() & frame["entry_valid"].astype(bool).to_numpy() & np.isfinite(values).all(axis=1)
        if np.any(finite):
            cluster[finite] = model.predict(scaler.transform(values[finite])).astype(np.int16)
        frame["path_cluster"] = cluster
        valid = frame[frame["path_cluster"] >= 0].copy()
        if not valid.empty:
            agg = valid.groupby("path_cluster", sort=True).agg(
                sample_count=("path_cluster", "size"),
                buyable_rate=("entry_buyable", "mean"),
                future_max_return_60d_mean=("future_max_return_60d", "mean"),
                future_min_return_60d_mean=("future_min_return_60d", "mean"),
                future_final_return_60d_mean=("future_final_return_60d", "mean"),
                future_peak_day_60d_mean=("future_peak_day_60d", "mean"),
                future_trough_day_60d_mean=("future_trough_day_60d", "mean"),
                drawdown_after_peak_60d_mean=("drawdown_after_peak_60d", "mean"),
                runup_after_trough_60d_mean=("runup_after_trough_60d", "mean"),
                path_range_60d_mean=("path_range_60d", "mean"),
                path_efficiency_60d_mean=("path_efficiency_60d", "mean"),
                path_trade_value_60d_mean=("path_trade_value_60d", "mean"),
                time_above_zero_60d_mean=("time_above_zero_60d", "mean"),
                time_below_zero_60d_mean=("time_below_zero_60d", "mean"),
            ).reset_index()
            summary_frames.append(agg)
        out_path = assigned_dir / path.name.replace("path_metrics_", "assigned_paths_")
        frame.to_parquet(out_path, index=False)
        assigned_paths.append(out_path)
        _write_json(progress_path, {"status": "cluster_assign", "last_shard": str(path), "updated_at": _now()})
    raw_summary = pd.concat(summary_frames, ignore_index=True)
    weighted_rows: list[dict[str, Any]] = []
    for cluster, group in raw_summary.groupby("path_cluster", sort=True):
        weights = group["sample_count"].to_numpy(dtype=np.float64)
        row: dict[str, Any] = {"path_cluster": int(cluster), "sample_count": int(weights.sum())}
        for col in [c for c in raw_summary.columns if c not in {"path_cluster", "sample_count"}]:
            row[col] = float(np.average(group[col].to_numpy(dtype=np.float64), weights=weights))
        weighted_rows.append(row)
    summary = pd.DataFrame(weighted_rows).sort_values("path_cluster", kind="mergesort")
    labels = _label_clusters(summary)
    summary["path_type"] = summary["path_cluster"].map(labels)
    for path in assigned_paths:
        frame = pd.read_parquet(path)
        frame["path_type"] = frame["path_cluster"].map(labels).fillna("invalid_or_incomplete")
        frame.to_parquet(path, index=False)
    return assigned_paths, summary, labels


def _profile_aggregate_rows(frame: pd.DataFrame, *, feature_cols: list[str], year: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_means = frame[feature_cols].mean(numeric_only=True)
    for path_type, group in frame.groupby("path_type", sort=True):
        if path_type == "invalid_or_incomplete":
            continue
        means = group[feature_cols].mean(numeric_only=True)
        for feature in feature_cols:
            value = means.get(feature, np.nan)
            all_value = all_means.get(feature, np.nan)
            rows.append(
                {
                    "year": int(year),
                    "path_type": str(path_type),
                    "feature": feature,
                    "sample_count": int(len(group)),
                    "feature_mean": float(value) if math.isfinite(float(value)) else float("nan"),
                    "all_mean": float(all_value) if math.isfinite(float(all_value)) else float("nan"),
                    "path_minus_all": float(value - all_value) if math.isfinite(float(value)) and math.isfinite(float(all_value)) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _bincount_profile(values: np.ndarray, codes: np.ndarray, type_count: int) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values) & (codes >= 0)
    if not np.any(finite):
        return np.zeros(type_count, dtype=np.float64), np.zeros(type_count, dtype=np.int64)
    sums = np.bincount(codes[finite], weights=values[finite].astype(np.float64), minlength=type_count)
    counts = np.bincount(codes[finite], minlength=type_count).astype(np.int64)
    return sums, counts


def _path_profile_rows(
    *,
    daily: pd.DataFrame,
    assigned: pd.DataFrame,
    symbol_codes: np.ndarray,
    calendar_pos: np.ndarray,
    type_order: list[str],
    pre_days: int,
    forward_days: int,
    year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_pos = assigned["_row_pos"].to_numpy(dtype=np.int64, copy=True)
    type_to_code = {name: idx for idx, name in enumerate(type_order)}
    codes = assigned["path_type"].map(type_to_code).fillna(-1).to_numpy(dtype=np.int16, copy=True)
    close = daily["close"].to_numpy(dtype=np.float64, copy=False)
    high = daily["high"].to_numpy(dtype=np.float64, copy=False)
    low = daily["low"].to_numpy(dtype=np.float64, copy=False)
    volume = daily["volume"].to_numpy(dtype=np.float64, copy=False)
    amount = daily["amount"].to_numpy(dtype=np.float64, copy=False)
    open_values = daily["open"].to_numpy(dtype=np.float64, copy=False)
    type_count = len(type_order)
    past_rows: list[dict[str, Any]] = []
    future_rows: list[dict[str, Any]] = []
    base_close = close[row_pos]
    for offset in range(-int(pre_days), 1):
        target = row_pos + offset
        in_bounds = (target >= 0) & (target < len(daily))
        safe_target = np.where(in_bounds, target, 0)
        valid = in_bounds & (symbol_codes[safe_target] == symbol_codes[row_pos]) & np.isfinite(base_close) & (base_close > 0)
        values_by_kind = {
            "past_close_ret": np.where(valid, close[safe_target] / base_close - 1.0, np.nan),
            "past_high_ret": np.where(valid, high[safe_target] / base_close - 1.0, np.nan),
            "past_low_ret": np.where(valid, low[safe_target] / base_close - 1.0, np.nan),
            "past_volume_log": np.where(valid, np.log1p(volume[safe_target]), np.nan),
            "past_amount_log": np.where(valid, np.log1p(amount[safe_target]), np.nan),
        }
        for kind, values in values_by_kind.items():
            sums, counts = _bincount_profile(values, codes, type_count)
            for idx, path_type in enumerate(type_order):
                mean = sums[idx] / counts[idx] if counts[idx] else np.nan
                past_rows.append({"year": int(year), "path_type": path_type, "path_kind": kind, "step": int(offset), "sample_count": int(counts[idx]), "mean": float(mean)})
    entry_target = row_pos + 1
    entry_in_bounds = entry_target < len(daily)
    safe_entry_target = np.where(entry_in_bounds, entry_target, 0)
    entry_valid = (
        entry_in_bounds
        & (symbol_codes[safe_entry_target] == symbol_codes[row_pos])
        & (calendar_pos[safe_entry_target] == calendar_pos[row_pos] + 1)
    )
    entry_open = np.full(len(row_pos), np.nan, dtype=np.float64)
    entry_open[entry_valid] = open_values[safe_entry_target[entry_valid]]
    for day in range(1, int(forward_days) + 1):
        target = row_pos + day
        in_bounds = target < len(daily)
        safe_target = np.where(in_bounds, target, 0)
        valid = (
            entry_valid
            & in_bounds
            & (symbol_codes[safe_target] == symbol_codes[row_pos])
            & (calendar_pos[safe_target] == calendar_pos[row_pos] + day)
            & np.isfinite(entry_open)
            & (entry_open > 0)
        )
        values_by_kind = {
            "future_close_ret": np.where(valid, close[safe_target] / entry_open - 1.0, np.nan),
            "future_high_ret": np.where(valid, high[safe_target] / entry_open - 1.0, np.nan),
            "future_low_ret": np.where(valid, low[safe_target] / entry_open - 1.0, np.nan),
        }
        for kind, values in values_by_kind.items():
            sums, counts = _bincount_profile(values, codes, type_count)
            for idx, path_type in enumerate(type_order):
                mean = sums[idx] / counts[idx] if counts[idx] else np.nan
                future_rows.append({"year": int(year), "path_type": path_type, "path_kind": kind, "step": int(day), "sample_count": int(counts[idx]), "mean": float(mean)})
    return pd.DataFrame(past_rows), pd.DataFrame(future_rows)


def _weighted_average_profiles(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        weights = group["sample_count"].to_numpy(dtype=np.float64)
        values = group["mean"].to_numpy(dtype=np.float64)
        valid = np.isfinite(values) & (weights > 0)
        row = {col: key[idx] for idx, col in enumerate(group_cols)}
        row["year"] = "all"
        row["sample_count"] = int(weights[valid].sum()) if np.any(valid) else 0
        row["mean"] = float(np.average(values[valid], weights=weights[valid])) if np.any(valid) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _build_profiles(
    *,
    root: Path,
    active: Mapping[str, Any],
    daily: pd.DataFrame,
    assigned_paths: list[Path],
    symbol_codes: np.ndarray,
    calendar_pos: np.ndarray,
    type_order: list[str],
    pre_days: int,
    forward_days: int,
    progress_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = _add_raw_daily_signals(daily)
    daily["_row_pos"] = np.arange(len(daily), dtype=np.int64)
    raw_cols = [col for col in RAW_SIGNAL_COLUMNS if col in daily.columns]
    intraday_cols = [col for col in INTRADAY_COLUMNS if col in {"symbol", "trade_date"} or col in INTRADAY_SIGNAL_COLUMNS]
    limit_cols = [col for col in LIMIT_COLUMNS if col in {"symbol", "trade_date"} or col in LIMIT_SIGNAL_COLUMNS]
    feature_frames: list[pd.DataFrame] = []
    past_frames: list[pd.DataFrame] = []
    future_frames: list[pd.DataFrame] = []
    type_year_rows: list[pd.DataFrame] = []
    for path in assigned_paths:
        assigned = pd.read_parquet(path)
        assigned = assigned[assigned["path_type"].ne("invalid_or_incomplete")].copy()
        if assigned.empty:
            continue
        year = int(assigned["year"].iloc[0])
        row_pos = assigned["_row_pos"].to_numpy(dtype=np.int64, copy=True)
        base = daily.iloc[row_pos][["symbol", "trade_date", *raw_cols]].copy().reset_index(drop=True)
        work = assigned[["symbol", "trade_date", "path_cluster", "path_type"]].merge(base, on=["symbol", "trade_date"], how="left", validate="one_to_one")
        intraday = _read_dataset_date_range(root, active, "intraday_daily_features", intraday_cols, f"{year}-01-01", f"{year}-12-31")
        limit_features = _read_dataset_date_range(root, active, "limit_intraday_features", limit_cols, f"{year}-01-01", f"{year}-12-31")
        if not intraday.empty:
            work = work.merge(intraday, on=["symbol", "trade_date"], how="left", validate="one_to_one")
        if not limit_features.empty:
            work = work.merge(limit_features, on=["symbol", "trade_date"], how="left", validate="one_to_one")
        feature_cols = [col for col in [*raw_cols, *INTRADAY_SIGNAL_COLUMNS, *LIMIT_SIGNAL_COLUMNS] if col in work.columns]
        for col in feature_cols:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        feature_frames.append(_profile_aggregate_rows(work, feature_cols=feature_cols, year=year))
        past, future = _path_profile_rows(
            daily=daily,
            assigned=assigned,
            symbol_codes=symbol_codes,
            calendar_pos=calendar_pos,
            type_order=type_order,
            pre_days=pre_days,
            forward_days=forward_days,
            year=year,
        )
        past_frames.append(past)
        future_frames.append(future)
        counts = assigned.groupby(["year", "path_type"], sort=True).size().rename("sample_count").reset_index()
        type_year_rows.append(counts)
        _write_json(progress_path, {"status": "profiles", "year": int(year), "updated_at": _now()})
        del assigned, base, work, intraday, limit_features, past, future
        gc.collect()
    feature_profile = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    past_profile = pd.concat(past_frames, ignore_index=True) if past_frames else pd.DataFrame()
    future_profile = pd.concat(future_frames, ignore_index=True) if future_frames else pd.DataFrame()
    type_year_counts = pd.concat(type_year_rows, ignore_index=True) if type_year_rows else pd.DataFrame()
    if not past_profile.empty:
        past_profile = pd.concat([past_profile, _weighted_average_profiles(past_profile, ["path_type", "path_kind", "step"])], ignore_index=True)
    if not future_profile.empty:
        future_profile = pd.concat([future_profile, _weighted_average_profiles(future_profile, ["path_type", "path_kind", "step"])], ignore_index=True)
    if not feature_profile.empty:
        all_rows: list[dict[str, Any]] = []
        for (path_type, feature), group in feature_profile.groupby(["path_type", "feature"], sort=True):
            weights = group["sample_count"].to_numpy(dtype=np.float64)
            values = group["feature_mean"].to_numpy(dtype=np.float64)
            all_values = group["all_mean"].to_numpy(dtype=np.float64)
            valid = np.isfinite(values) & (weights > 0)
            all_rows.append(
                {
                    "year": "all",
                    "path_type": path_type,
                    "feature": feature,
                    "sample_count": int(weights[valid].sum()) if np.any(valid) else 0,
                    "feature_mean": float(np.average(values[valid], weights=weights[valid])) if np.any(valid) else float("nan"),
                    "all_mean": float(np.average(all_values[valid], weights=weights[valid])) if np.any(valid) else float("nan"),
                    "path_minus_all": float(np.average(values[valid] - all_values[valid], weights=weights[valid])) if np.any(valid) else float("nan"),
                }
            )
        feature_profile = pd.concat([feature_profile, pd.DataFrame(all_rows)], ignore_index=True)
    return feature_profile, past_profile, future_profile, type_year_counts


def _plot_outputs(output_dir: Path, cluster_summary: pd.DataFrame, past_profile: pd.DataFrame, future_profile: pd.DataFrame, feature_profile: pd.DataFrame) -> list[str]:
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    if not future_profile.empty:
        frame = future_profile[(future_profile["year"].astype(str).eq("all")) & (future_profile["path_kind"].eq("future_close_ret"))].copy()
        if not frame.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            for path_type, group in frame.groupby("path_type", sort=True):
                group = group.sort_values("step")
                ax.plot(group["step"], group["mean"] * 100.0, label=path_type)
            ax.axhline(0.0, color="#777777", linewidth=0.8)
            ax.set_title("Future 60-day close path by learned path type")
            ax.set_xlabel("Future trading day")
            ax.set_ylabel("Return from next open (%)")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = chart_dir / "future_close_path_by_type.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            outputs.append(str(path.resolve()))
    if not past_profile.empty:
        frame = past_profile[(past_profile["year"].astype(str).eq("all")) & (past_profile["path_kind"].eq("past_close_ret"))].copy()
        if not frame.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            for path_type, group in frame.groupby("path_type", sort=True):
                group = group.sort_values("step")
                ax.plot(group["step"], group["mean"] * 100.0, label=path_type)
            ax.axhline(0.0, color="#777777", linewidth=0.8)
            ax.set_title("Past 60-day close path before signal day")
            ax.set_xlabel("Trading day offset")
            ax.set_ylabel("Return to signal close (%)")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = chart_dir / "past_close_path_by_type.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            outputs.append(str(path.resolve()))
    if not cluster_summary.empty:
        frame = cluster_summary.sort_values("path_cluster")
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [f"{int(row.path_cluster)}:{row.path_type}" for row in frame.itertuples()]
        ax.bar(labels, frame["future_final_return_60d_mean"] * 100.0, label="final")
        ax.scatter(labels, frame["future_max_return_60d_mean"] * 100.0, color="#c44e52", label="max high")
        ax.scatter(labels, frame["future_min_return_60d_mean"] * 100.0, color="#4c72b0", label="min low")
        ax.axhline(0.0, color="#777777", linewidth=0.8)
        ax.set_title("Path type future outcome summary")
        ax.set_ylabel("Return (%)")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = chart_dir / "path_type_outcome_summary.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path.resolve()))
    if not feature_profile.empty:
        frame = feature_profile[feature_profile["year"].astype(str).eq("all")].copy()
        focus = frame.reindex(frame["path_minus_all"].abs().sort_values(ascending=False).index).head(40)
        if not focus.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            labels = [f"{row.path_type}:{row.feature}" for row in focus.itertuples()]
            ax.barh(range(len(focus)), focus["path_minus_all"], color=np.where(focus["path_minus_all"] >= 0, "#2f6f9f", "#a23b3b"))
            ax.set_yticks(range(len(focus)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.invert_yaxis()
            ax.set_title("Largest feature profile differences by path type")
            ax.set_xlabel("Path type mean - all mean")
            ax.grid(True, axis="x", alpha=0.25)
            fig.tight_layout()
            path = chart_dir / "top_feature_differences.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            outputs.append(str(path.resolve()))
    return outputs


def _build_report(
    *,
    output_dir: Path,
    summary: Mapping[str, Any],
    cluster_summary: pd.DataFrame,
    feature_profile: pd.DataFrame,
    chart_paths: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# QDP v2 Stock Path Profile Atlas")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("This atlas profiles all stock future paths, not only winners. It computes next-open anchored future 1-60 day path metrics, clusters those full path metrics, then describes each path type using only pre-signal raw daily paths, intraday summaries, and limit-board structure.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- event window: {summary.get('event_start')} to {summary.get('event_end')}")
    lines.append(f"- candidate rows: {summary.get('candidate_count'):,}")
    lines.append(f"- valid clustered rows: {summary.get('clustered_count'):,}")
    lines.append(f"- output_dir: `{summary.get('output_dir')}`")
    lines.append("")
    lines.append("## Learned Future Path Types")
    lines.append("")
    if not cluster_summary.empty:
        for row in cluster_summary.sort_values("path_cluster").to_dict("records"):
            lines.append(
                f"- cluster {int(row['path_cluster'])} / {row['path_type']}: "
                f"n={int(row['sample_count']):,}, "
                f"max={row['future_max_return_60d_mean'] * 100:.2f}%, "
                f"final={row['future_final_return_60d_mean'] * 100:.2f}%, "
                f"min={row['future_min_return_60d_mean'] * 100:.2f}%, "
                f"peak_day={row['future_peak_day_60d_mean']:.1f}, "
                f"drawdown_after_peak={row['drawdown_after_peak_60d_mean'] * 100:.2f}%"
            )
    lines.append("")
    lines.append("## Strongest Pre-Signal Profile Differences")
    lines.append("")
    focus = feature_profile[feature_profile["year"].astype(str).eq("all")].copy() if not feature_profile.empty else pd.DataFrame()
    if not focus.empty:
        for path_type, group in focus.groupby("path_type", sort=True):
            lines.append(f"### {path_type}")
            top = group.reindex(group["path_minus_all"].abs().sort_values(ascending=False).index).head(10)
            for row in top.to_dict("records"):
                lines.append(
                    f"- {row['feature']}: diff={row['path_minus_all']:.4f}, "
                    f"type_mean={row['feature_mean']:.4f}, all={row['all_mean']:.4f}"
                )
            lines.append("")
    lines.append("## Charts")
    lines.append("")
    for path in chart_paths:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## Boundaries")
    lines.append("")
    lines.append("- Path types are learned from future paths, so they are for market-structure understanding, not directly tradable labels.")
    lines.append("- This is not a trained selector. It is the prerequisite atlas for defining path-value targets and model inputs.")
    lines.append("- Entry anchoring uses next open; rows with incomplete 60-day future paths are kept in shards but excluded from clustering.")
    path = output_dir / "stock_path_profile_atlas_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.resolve())


@dataclass(frozen=True)
class AtlasConfig:
    qdp_root: Path
    output_root: Path
    run_tag: str
    event_start: str
    event_end: str
    years: tuple[int, ...]
    forward_days: int
    pre_days: int
    cluster_count: int
    seed: int


def build_stock_path_profile_atlas(config: AtlasConfig) -> dict[str, Any]:
    root = config.qdp_root.resolve()
    active = _read_active(root)
    output_dir = config.output_root / f"{config.run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    _write_json(progress_path, {"status": "loading_daily_base", "updated_at": _now()})
    daily, symbol_codes, calendar_pos = _load_daily_base(root, active)
    _write_json(progress_path, {"status": "daily_loaded", "daily_rows": int(len(daily)), "updated_at": _now()})
    path_shards = _write_year_path_metrics(
        daily=daily,
        symbol_codes=symbol_codes,
        calendar_pos=calendar_pos,
        root=root,
        active=active,
        years=config.years,
        event_start=config.event_start,
        event_end=config.event_end,
        forward_days=config.forward_days,
        output_dir=output_dir,
        progress_path=progress_path,
    )
    scaler, model = _fit_path_clusters(path_shards, cluster_count=config.cluster_count, seed=config.seed, progress_path=progress_path)
    assigned_paths, cluster_summary, cluster_labels = _assign_clusters(path_shards, scaler=scaler, model=model, output_dir=output_dir, progress_path=progress_path)
    type_order = sorted({value for value in cluster_labels.values()})
    feature_profile, past_profile, future_profile, type_year_counts = _build_profiles(
        root=root,
        active=active,
        daily=daily,
        assigned_paths=assigned_paths,
        symbol_codes=symbol_codes,
        calendar_pos=calendar_pos,
        type_order=type_order,
        pre_days=config.pre_days,
        forward_days=config.forward_days,
        progress_path=progress_path,
    )
    outputs = {
        "cluster_summary_csv": _write_csv(output_dir / "path_cluster_summary.csv", cluster_summary),
        "feature_profile_csv": _write_csv(output_dir / "feature_profile_by_path_type.csv", feature_profile),
        "past_path_profile_csv": _write_csv(output_dir / "past_path_profile_by_path_type.csv", past_profile),
        "future_path_profile_csv": _write_csv(output_dir / "future_path_profile_by_path_type.csv", future_profile),
        "type_year_counts_csv": _write_csv(output_dir / "path_type_year_counts.csv", type_year_counts),
        "path_metric_shards": [str(path.resolve()) for path in path_shards],
        "assigned_path_shards": [str(path.resolve()) for path in assigned_paths],
    }
    chart_paths = _plot_outputs(output_dir, cluster_summary, past_profile, future_profile, feature_profile)
    outputs["charts"] = chart_paths
    candidate_count = int(sum(int(pd.read_parquet(path, columns=["symbol"]).shape[0]) for path in path_shards))
    clustered_count = int(cluster_summary["sample_count"].sum()) if not cluster_summary.empty else 0
    summary: dict[str, Any] = {
        "artifact_type": "qdp_v2_stock_path_profile_atlas",
        "generated_at": _now(),
        "qdp_root": str(root),
        "active_manifest": str((root / "active" / "active.json").resolve()),
        "active_datasets": dict(active.get("datasets", {}) or {}),
        "output_dir": str(output_dir.resolve()),
        "event_start": config.event_start,
        "event_end": config.event_end,
        "years": list(config.years),
        "candidate_count": candidate_count,
        "clustered_count": clustered_count,
        "forward_days": int(config.forward_days),
        "pre_days": int(config.pre_days),
        "cluster_count": int(config.cluster_count),
        "path_cluster_features": list(PATH_CLUSTER_FEATURES),
        "cluster_labels": cluster_labels,
        "outputs": outputs,
    }
    report_path = _build_report(output_dir=output_dir, summary=summary, cluster_summary=cluster_summary, feature_profile=feature_profile, chart_paths=chart_paths)
    summary["outputs"]["report_md"] = report_path
    summary_path = _write_json(output_dir / "stock_path_profile_atlas_summary.json", summary)
    summary["outputs"]["summary_json"] = summary_path
    _write_json(progress_path, {"status": "completed", "summary_json": summary_path, "report_md": report_path, "updated_at": _now()})
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a full stock path profile atlas from QDP v2 active data.")
    parser.add_argument("--qdp-root", default=str(DEFAULT_QDP_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-tag", default="qdp_v2_stock_path_profile_atlas")
    parser.add_argument("--event-start", default=DEFAULT_EVENT_START)
    parser.add_argument("--event-end", default=DEFAULT_EVENT_END)
    parser.add_argument("--years", default="2012-2025")
    parser.add_argument("--forward-days", type=int, default=DEFAULT_FORWARD_DAYS)
    parser.add_argument("--pre-days", type=int, default=DEFAULT_PRE_DAYS)
    parser.add_argument("--cluster-count", type=int, default=DEFAULT_CLUSTER_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = build_stock_path_profile_atlas(
        AtlasConfig(
            qdp_root=Path(args.qdp_root),
            output_root=Path(args.output_root),
            run_tag=str(args.run_tag),
            event_start=str(args.event_start),
            event_end=str(args.event_end),
            years=_parse_year_range(str(args.years), default=tuple(range(2012, 2026))),
            forward_days=int(args.forward_days),
            pre_days=int(args.pre_days),
            cluster_count=int(args.cluster_count),
            seed=int(args.seed),
        )
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(summary.get("outputs", {}).get("report_md", ""))
    return summary


if __name__ == "__main__":
    main()
