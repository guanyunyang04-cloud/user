from __future__ import annotations

import argparse
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


DEFAULT_QDP_ROOT = Path("quant_data_platform/data/qdp_v2")
DEFAULT_OUTPUT_ROOT = Path("daily_research/output/path_policy/raw_rising_path_atlas")
DEFAULT_EVENT_START = "2012-01-01"
DEFAULT_EVENT_END = "2025-12-02"
DEFAULT_PRE_DAYS = 20
DEFAULT_FORWARD_DAYS = 20
DEFAULT_MAX_EVENTS_PER_LABEL = 50_000
DEFAULT_MAX_CLUSTER_SAMPLES = 40_000
DEFAULT_RANDOM_SEED = 7
FULL_STREAM_PATH_KINDS = ("pre_close", "future_close", "future_high", "future_low")


DAILY_RAW_COLUMNS = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"]
INDUSTRY_COLUMNS = ["symbol", "trade_date", "industry"]
INTRADAY_COLUMNS = [
    "symbol",
    "trade_date",
    "first_5m_ret",
    "opening_auction_ret",
    "opening_auction_amount_share",
    "first_30m_ret",
    "first_30m_amount_share",
    "open_gap",
    "open_gap_first_30m_follow_through",
    "open_gap_first_30m_reversal",
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
    "bar_count",
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
LIMIT_COLUMNS = [
    "symbol",
    "trade_date",
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


RAW_SIGNAL_COLUMNS = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "volatility_20d",
    "amount_mean_20d_log",
    "amount_ratio_5_20",
    "volume_ratio_5_20",
    "distance_to_20d_high",
    "distance_to_60d_high",
    "distance_to_20d_low",
    "drawdown_from_20d_high",
    "range_1d",
    "body_to_range_1d",
    "upper_shadow_to_range_1d",
    "lower_shadow_to_range_1d",
    "open_gap_1d",
    "close_to_open_1d",
]
INTRADAY_SIGNAL_COLUMNS = [col for col in INTRADAY_COLUMNS if col not in {"symbol", "trade_date"}]
LIMIT_SIGNAL_COLUMNS = [col for col in LIMIT_COLUMNS if col not in {"symbol", "trade_date"}]


LABELS = [
    "top10_return_20d",
    "top5_return_20d",
    "hit_up5_before_down3_20d",
    "hit_up10_before_down5_20d",
    "persistent_upside_10pct_20d",
    "rise_then_fade_10pct_20d",
    "fast_peak_10pct_20d",
    "late_peak_10pct_20d",
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
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
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


def _parse_csv_strings(raw: str | Iterable[str] | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        values = list(default)
    elif isinstance(raw, str):
        values = raw.split(",")
    else:
        values = list(raw)
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out or default)


def _relative_shard_paths(root: Path, manifest: Mapping[str, Any]) -> list[Path]:
    out: list[Path] = []
    for shard in list(manifest.get("shards", []) or []):
        if str(shard.get("status", "stored")) not in {"stored", "completed"}:
            continue
        path = Path(str(shard.get("path", "") or ""))
        if not path.is_absolute():
            path = root / path
        if path.exists():
            out.append(path)
    if not out:
        raise FileNotFoundError(f"no shard paths found for {manifest.get('dataset_id', '')}")
    return out


def _read_dataset(root: Path, active: Mapping[str, Any], domain: str, columns: list[str]) -> pd.DataFrame:
    dataset_id = str(dict(active.get("datasets", {}) or {}).get(domain, "") or "")
    if not dataset_id:
        raise KeyError(f"active dataset missing: {domain}")
    manifest_path = root / "datasets" / domain / dataset_id / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    available = {str(item.get("name", "")) for item in list(manifest.get("schema", []) or [])}
    selected_columns = [col for col in columns if col in available]
    paths = _relative_shard_paths(root, manifest)
    table = ds.dataset([str(path) for path in paths], format="parquet").to_table(columns=selected_columns)
    frame = table.to_pandas()
    if "trade_date" in frame.columns:
        frame["trade_date"] = frame["trade_date"].astype(str)
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    return frame


def _read_dataset_for_keys_by_year(root: Path, active: Mapping[str, Any], domain: str, columns: list[str], keys: pd.DataFrame) -> pd.DataFrame:
    if keys.empty:
        return pd.DataFrame(columns=[col for col in columns if col in {"symbol", "trade_date"}])
    dataset_id = str(dict(active.get("datasets", {}) or {}).get(domain, "") or "")
    if not dataset_id:
        return pd.DataFrame(columns=columns)
    manifest_path = root / "datasets" / domain / dataset_id / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    available = {str(item.get("name", "")) for item in list(manifest.get("schema", []) or [])}
    selected_columns = [col for col in columns if col in available]
    paths = _relative_shard_paths(root, manifest)
    dataset = ds.dataset([str(path) for path in paths], format="parquet")
    work_keys = keys[["symbol", "trade_date"]].drop_duplicates().copy()
    work_keys["trade_date"] = work_keys["trade_date"].astype(str)
    work_keys["symbol"] = work_keys["symbol"].astype(str).str.upper().str.strip()
    frames: list[pd.DataFrame] = []
    for year, group in work_keys.groupby(work_keys["trade_date"].str.slice(0, 4), sort=True):
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        filt = (ds.field("trade_date") >= start) & (ds.field("trade_date") <= end)
        table = dataset.to_table(columns=selected_columns, filter=filt)
        frame = table.to_pandas()
        if frame.empty:
            continue
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
        frames.append(frame.merge(group, on=["symbol", "trade_date"], how="inner"))
    if not frames:
        return pd.DataFrame(columns=selected_columns)
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(["symbol", "trade_date"], keep="last")


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator.astype("float64").div(denominator.astype("float64").replace(0.0, np.nan))
    return out.replace([np.inf, -np.inf], np.nan)


def _add_raw_daily_signals(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.sort_values(["symbol", "trade_date"], kind="mergesort").reset_index(drop=True)
    groups = daily.groupby("symbol", sort=False, group_keys=False)
    close = daily["close"].astype("float64")
    open_ = daily["open"].astype("float64")
    high = daily["high"].astype("float64")
    low = daily["low"].astype("float64")
    amount = daily["amount"].astype("float64")
    volume = daily["volume"].astype("float64")
    prev_close = groups["close"].shift(1).astype("float64")
    daily["prev_close"] = prev_close
    daily["open_gap_1d"] = _safe_div(open_, prev_close).sub(1.0)
    daily["close_to_open_1d"] = _safe_div(close, open_).sub(1.0)
    daily["range_1d"] = _safe_div(high, low).sub(1.0)
    body = close.sub(open_).abs()
    day_range = high.sub(low).replace(0.0, np.nan)
    daily["body_to_range_1d"] = body.div(day_range).replace([np.inf, -np.inf], np.nan)
    daily["upper_shadow_to_range_1d"] = high.sub(np.maximum(open_, close)).div(day_range).replace([np.inf, -np.inf], np.nan)
    daily["lower_shadow_to_range_1d"] = np.minimum(open_, close).sub(low).div(day_range).replace([np.inf, -np.inf], np.nan)
    for horizon in (1, 3, 5, 10, 20):
        daily[f"ret_{horizon}d"] = groups["close"].pct_change(horizon, fill_method=None)
    ret_1d = daily["ret_1d"].astype("float64")
    daily["volatility_20d"] = groups["ret_1d"].transform(lambda s: s.rolling(20, min_periods=10).std())
    amount_mean5 = groups["amount"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    amount_mean20 = groups["amount"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    volume_mean5 = groups["volume"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    volume_mean20 = groups["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    daily["amount_mean_20d_log"] = np.log1p(amount_mean20)
    daily["amount_ratio_5_20"] = amount_mean5.div(amount_mean20.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    daily["volume_ratio_5_20"] = volume_mean5.div(volume_mean20.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    high20 = groups["high"].transform(lambda s: s.rolling(20, min_periods=5).max())
    high60 = groups["high"].transform(lambda s: s.rolling(60, min_periods=20).max())
    low20 = groups["low"].transform(lambda s: s.rolling(20, min_periods=5).min())
    daily["distance_to_20d_high"] = close.div(high20.replace(0.0, np.nan)).sub(1.0).replace([np.inf, -np.inf], np.nan)
    daily["distance_to_60d_high"] = close.div(high60.replace(0.0, np.nan)).sub(1.0).replace([np.inf, -np.inf], np.nan)
    daily["distance_to_20d_low"] = close.div(low20.replace(0.0, np.nan)).sub(1.0).replace([np.inf, -np.inf], np.nan)
    daily["drawdown_from_20d_high"] = daily["distance_to_20d_high"]
    del ret_1d
    return daily.replace([np.inf, -np.inf], np.nan)


def _add_forward_labels(daily: pd.DataFrame, *, forward_days: int) -> pd.DataFrame:
    daily = daily.sort_values(["symbol", "trade_date"], kind="mergesort").reset_index(drop=True)
    groups = daily.groupby("symbol", sort=False, group_keys=False)
    entry_open = groups["open"].shift(-1).astype("float64")
    daily["entry_open_next"] = entry_open
    daily["future_open_ret_20d"] = groups["open"].shift(-(int(forward_days) + 1)).astype("float64").div(entry_open).sub(1.0)
    high_paths: list[np.ndarray] = []
    low_paths: list[np.ndarray] = []
    close_paths: list[np.ndarray] = []
    for day in range(1, int(forward_days) + 1):
        high_paths.append(groups["high"].shift(-day).astype("float64").div(entry_open).sub(1.0).to_numpy(dtype=np.float32, copy=True))
        low_paths.append(groups["low"].shift(-day).astype("float64").div(entry_open).sub(1.0).to_numpy(dtype=np.float32, copy=True))
        close_paths.append(groups["close"].shift(-day).astype("float64").div(entry_open).sub(1.0).to_numpy(dtype=np.float32, copy=True))
    future_high = np.column_stack(high_paths)
    future_low = np.column_stack(low_paths)
    future_close = np.column_stack(close_paths)
    valid_path = np.isfinite(future_high).all(axis=1) & np.isfinite(future_low).all(axis=1) & np.isfinite(future_close).all(axis=1)
    max_high = np.full(len(daily), np.nan, dtype=np.float32)
    min_low = np.full(len(daily), np.nan, dtype=np.float32)
    final_close = np.full(len(daily), np.nan, dtype=np.float32)
    if np.any(valid_path):
        max_high[valid_path] = np.max(future_high[valid_path], axis=1)
        min_low[valid_path] = np.min(future_low[valid_path], axis=1)
        final_close[valid_path] = future_close[valid_path, -1]
    daily["future_max_high_20d"] = max_high
    daily["future_min_low_20d"] = min_low
    daily["future_final_close_20d"] = final_close
    peak_idx = np.argmax(np.nan_to_num(future_high, nan=-999.0), axis=1).astype(np.int16) + 1
    daily["future_peak_day_20d"] = np.where(valid_path, peak_idx, 0)
    daily["path_valid_20d"] = valid_path
    profit5 = _first_hit_day(future_high, 0.05, direction="up")
    stop3 = _first_hit_day(future_low, 0.03, direction="down")
    profit10 = _first_hit_day(future_high, 0.10, direction="up")
    stop5 = _first_hit_day(future_low, 0.05, direction="down")
    daily["hit_up5_before_down3_20d"] = valid_path & (profit5 > 0) & ((stop3 == 0) | (profit5 <= stop3))
    daily["hit_up10_before_down5_20d"] = valid_path & (profit10 > 0) & ((stop5 == 0) | (profit10 <= stop5))
    daily["persistent_upside_10pct_20d"] = valid_path & (daily["future_max_high_20d"] >= 0.10) & (
        daily["future_final_close_20d"] >= daily["future_max_high_20d"].sub(0.02)
    )
    daily["rise_then_fade_10pct_20d"] = valid_path & (daily["future_max_high_20d"] >= 0.10) & (
        daily["future_final_close_20d"] <= daily["future_max_high_20d"].sub(0.03)
    )
    daily["fast_peak_10pct_20d"] = valid_path & (daily["future_max_high_20d"] >= 0.10) & (daily["future_peak_day_20d"] <= 5)
    daily["late_peak_10pct_20d"] = valid_path & (daily["future_max_high_20d"] >= 0.10) & (daily["future_peak_day_20d"] >= 15)
    ret20 = daily["future_open_ret_20d"].astype("float64")
    pct_rank = ret20.groupby(daily["trade_date"], sort=False).rank(method="average", pct=True)
    daily["future_open_ret_20d_rank"] = pct_rank
    daily["top10_return_20d"] = pct_rank >= 0.90
    daily["top5_return_20d"] = pct_rank >= 0.95
    return daily


def _first_hit_day(path: np.ndarray, threshold: float, *, direction: str) -> np.ndarray:
    if direction == "up":
        hit = np.asarray(path >= float(threshold), dtype=bool)
    elif direction == "down":
        hit = np.asarray(path <= -abs(float(threshold)), dtype=bool)
    else:
        raise ValueError(direction)
    any_hit = hit.any(axis=1)
    first = np.argmax(hit, axis=1).astype(np.int16) + 1
    first[~any_hit] = 0
    return first


def _sample_events(frame: pd.DataFrame, *, label: str, max_events: int, seed: int) -> pd.DataFrame:
    events = frame[frame[label].astype(bool)].copy()
    if len(events) <= int(max_events):
        return events
    return events.sample(n=int(max_events), random_state=int(seed)).sort_values(["trade_date", "symbol"], kind="mergesort").reset_index(drop=True)


def _add_match_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col, target in [
        ("amount_mean_20d_log", "_liquidity_bucket"),
        ("volatility_20d", "_volatility_bucket"),
        ("close", "_price_bucket"),
    ]:
        ranks = pd.to_numeric(out[col], errors="coerce").rank(method="average", pct=True)
        rank_values = np.nan_to_num(ranks.to_numpy(dtype=np.float64), nan=0.5, posinf=0.5, neginf=0.5)
        out[target] = np.floor(np.clip(rank_values, 0.0, 0.999999) * 5).astype(np.int16)
    out["_match_full_id"] = out.groupby(
        ["trade_date", "industry", "_liquidity_bucket", "_volatility_bucket", "_price_bucket"],
        sort=False,
    ).ngroup().astype(np.int64)
    out["_match_loose_id"] = out.groupby(["trade_date", "industry"], sort=False).ngroup().astype(np.int64)
    out["_match_date_id"] = out.groupby(["trade_date"], sort=False).ngroup().astype(np.int64)
    return out


def _match_level_weights(candidate: pd.DataFrame, label: str) -> tuple[np.ndarray, np.ndarray, list[tuple[str, np.ndarray, np.ndarray, int]], dict[str, Any]]:
    event_mask = candidate[label].astype(bool).to_numpy()
    control_mask = ~event_mask
    full_ids = candidate["_match_full_id"].to_numpy(dtype=np.int64, copy=False)
    loose_ids = candidate["_match_loose_id"].to_numpy(dtype=np.int64, copy=False)
    date_ids = candidate["_match_date_id"].to_numpy(dtype=np.int64, copy=False)
    n_full = int(full_ids.max()) + 1 if len(full_ids) else 0
    n_loose = int(loose_ids.max()) + 1 if len(loose_ids) else 0
    n_date = int(date_ids.max()) + 1 if len(date_ids) else 0

    control_full = np.bincount(full_ids[control_mask], minlength=n_full)
    event_full = np.bincount(full_ids[event_mask], minlength=n_full)
    full_weights = np.where(control_full > 0, event_full, 0).astype(np.float64)
    remaining_after_full = event_mask & (control_full[full_ids] == 0)

    control_loose = np.bincount(loose_ids[control_mask], minlength=n_loose)
    event_loose = np.bincount(loose_ids[remaining_after_full], minlength=n_loose)
    loose_weights = np.where(control_loose > 0, event_loose, 0).astype(np.float64)
    remaining_after_loose = remaining_after_full & (control_loose[loose_ids] == 0)

    control_date = np.bincount(date_ids[control_mask], minlength=n_date)
    event_date = np.bincount(date_ids[remaining_after_loose], minlength=n_date)
    date_weights = np.where(control_date > 0, event_date, 0).astype(np.float64)

    event_count = int(event_mask.sum())
    full_count = float(full_weights.sum())
    loose_count = float(loose_weights.sum())
    date_count = float(date_weights.sum())
    levels = [
        ("date_industry_liquidity_vol_price", full_ids, full_weights, n_full),
        ("date_industry", loose_ids, loose_weights, n_loose),
        ("date_only", date_ids, date_weights, n_date),
    ]
    summary = {
        "label": label,
        "event_count": event_count,
        "control_available_count_weighted": full_count + loose_count + date_count,
        "match_date_industry_liquidity_vol_price_share": full_count / event_count if event_count else np.nan,
        "match_date_industry_share": loose_count / event_count if event_count else np.nan,
        "match_date_only_share": date_count / event_count if event_count else np.nan,
        "unmatched_share": max(0.0, (event_count - full_count - loose_count - date_count) / event_count) if event_count else np.nan,
    }
    return event_mask, control_mask, levels, summary


def _weighted_control_mean(values: np.ndarray, control_mask: np.ndarray, levels: list[tuple[str, np.ndarray, np.ndarray, int]]) -> tuple[float, float]:
    weighted_sum, total_weight = _weighted_control_sum_weight(values, control_mask, levels)
    if total_weight <= 0:
        return float("nan"), 0.0
    return weighted_sum / total_weight, total_weight


def _weighted_control_sum_weight(values: np.ndarray, control_mask: np.ndarray, levels: list[tuple[str, np.ndarray, np.ndarray, int]]) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values) & control_mask
    if not np.any(finite):
        return 0.0, 0.0
    total_weight = 0.0
    weighted_sum = 0.0
    finite_values = values[finite]
    for _name, group_ids, group_weights, group_count in levels:
        if group_count <= 0 or float(np.sum(group_weights)) <= 0:
            continue
        gids = group_ids[finite]
        sums = np.bincount(gids, weights=finite_values, minlength=group_count)
        counts = np.bincount(gids, minlength=group_count).astype(np.float64)
        valid_groups = (group_weights > 0) & (counts > 0)
        if not np.any(valid_groups):
            continue
        means = sums[valid_groups] / counts[valid_groups]
        weights = group_weights[valid_groups]
        weighted_sum += float(np.sum(means * weights))
        total_weight += float(np.sum(weights))
    return weighted_sum, total_weight


def _full_weighted_feature_contrast(candidate: pd.DataFrame, labels: tuple[str, ...], feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    for label in labels:
        event_mask, control_mask, levels, match_summary = _match_level_weights(candidate, label)
        match_rows.append(match_summary)
        for feature in feature_columns:
            if feature not in candidate.columns:
                continue
            values = pd.to_numeric(candidate[feature], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            event_values = values[event_mask & np.isfinite(values)]
            control_mean, control_weight = _weighted_control_mean(values, control_mask, levels)
            event_mean = float(np.mean(event_values)) if len(event_values) else float("nan")
            rows.append(
                {
                    "label": label,
                    "feature": feature,
                    "event_count": int(len(event_values)),
                    "control_count_weighted": float(control_weight),
                    "event_mean": event_mean,
                    "control_weighted_mean": control_mean,
                    "event_minus_weighted_control": event_mean - control_mean if math.isfinite(event_mean) and math.isfinite(control_mean) else float("nan"),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(match_rows)


def _full_auxiliary_feature_contrast(
    *,
    root: Path,
    active: Mapping[str, Any],
    candidate: pd.DataFrame,
    labels: tuple[str, ...],
    progress_path: Path,
) -> pd.DataFrame:
    label_level_specs: dict[str, list[tuple[str, np.ndarray, int]]] = {}
    for label in labels:
        _event_mask, _control_mask, levels, _summary = _match_level_weights(candidate, label)
        label_level_specs[label] = [(name, weights, group_count) for name, _ids, weights, group_count in levels]

    candidate_key_cols = [
        "symbol",
        "trade_date",
        "year",
        "_match_full_id",
        "_match_loose_id",
        "_match_date_id",
        *labels,
    ]
    candidate_keys = candidate[candidate_key_cols].copy()
    accum: dict[tuple[str, str], dict[str, float]] = {}
    domain_specs = [
        ("intraday_daily_features", INTRADAY_COLUMNS, INTRADAY_SIGNAL_COLUMNS),
        ("limit_intraday_features", LIMIT_COLUMNS, LIMIT_SIGNAL_COLUMNS),
    ]
    id_column_for_level = {
        "date_industry_liquidity_vol_price": "_match_full_id",
        "date_industry": "_match_loose_id",
        "date_only": "_match_date_id",
    }
    for domain, columns, feature_columns in domain_specs:
        dataset_id = str(dict(active.get("datasets", {}) or {}).get(domain, "") or "")
        if not dataset_id:
            continue
        manifest_path = root / "datasets" / domain / dataset_id / "dataset.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        available = {str(item.get("name", "")) for item in list(manifest.get("schema", []) or [])}
        selected_columns = [col for col in columns if col in available]
        selected_features = [col for col in feature_columns if col in selected_columns]
        if not selected_features:
            continue
        paths = _relative_shard_paths(root, manifest)
        dataset = ds.dataset([str(path) for path in paths], format="parquet")
        for year, key_group in candidate_keys.groupby("year", sort=True):
            start = f"{int(year):04d}-01-01"
            end = f"{int(year):04d}-12-31"
            filt = (ds.field("trade_date") >= start) & (ds.field("trade_date") <= end)
            table = dataset.to_table(columns=selected_columns, filter=filt)
            aux = table.to_pandas()
            if aux.empty:
                continue
            aux["trade_date"] = aux["trade_date"].astype(str)
            aux["symbol"] = aux["symbol"].astype(str).str.upper().str.strip()
            frame = key_group.merge(aux, on=["symbol", "trade_date"], how="left", validate="one_to_one")
            for feature in selected_features:
                if feature not in frame.columns:
                    continue
                values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=np.float64, copy=False)
                finite = np.isfinite(values)
                for label in labels:
                    key = (label, feature)
                    slot = accum.setdefault(
                        key,
                        {
                            "event_sum": 0.0,
                            "event_count": 0.0,
                            "control_weighted_sum": 0.0,
                            "control_weight": 0.0,
                        },
                    )
                    event_mask = frame[label].astype(bool).to_numpy()
                    event_values = values[event_mask & finite]
                    if len(event_values):
                        slot["event_sum"] += float(np.sum(event_values))
                        slot["event_count"] += float(len(event_values))
                    control_mask = ~event_mask
                    levels_for_frame = [
                        (
                            level_name,
                            frame[id_column_for_level[level_name]].to_numpy(dtype=np.int64, copy=False),
                            weights,
                            group_count,
                        )
                        for level_name, weights, group_count in label_level_specs[label]
                    ]
                    weighted_sum, total_weight = _weighted_control_sum_weight(values, control_mask, levels_for_frame)
                    slot["control_weighted_sum"] += float(weighted_sum)
                    slot["control_weight"] += float(total_weight)
            _write_json(progress_path, {"status": "full_aux_feature_contrast", "domain": domain, "completed_year": int(year), "updated_at": _now()})

    rows: list[dict[str, Any]] = []
    for (label, feature), slot in sorted(accum.items()):
        event_count = float(slot["event_count"])
        control_weight = float(slot["control_weight"])
        event_mean = slot["event_sum"] / event_count if event_count > 0 else float("nan")
        control_mean = slot["control_weighted_sum"] / control_weight if control_weight > 0 else float("nan")
        rows.append(
            {
                "label": label,
                "feature": feature,
                "event_count": int(event_count),
                "control_count_weighted": control_weight,
                "event_mean": event_mean,
                "control_weighted_mean": control_mean,
                "event_minus_weighted_control": event_mean - control_mean if math.isfinite(event_mean) and math.isfinite(control_mean) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _path_values_for_kind(
    *,
    daily: pd.DataFrame,
    candidate: pd.DataFrame,
    symbol_codes: np.ndarray,
    kind: str,
    step: int,
) -> np.ndarray:
    row_pos = candidate["_row_pos"].to_numpy(dtype=np.int64, copy=False)
    close_values = daily["close"].to_numpy(dtype=np.float64, copy=False)
    high_values = daily["high"].to_numpy(dtype=np.float64, copy=False)
    low_values = daily["low"].to_numpy(dtype=np.float64, copy=False)
    entry_open = daily["entry_open_next"].to_numpy(dtype=np.float64, copy=False)
    if kind == "pre_close":
        target = row_pos + int(step)
        denom = close_values[row_pos]
        source = close_values
    elif kind == "future_close":
        target = row_pos + int(step)
        denom = entry_open[row_pos]
        source = close_values
    elif kind == "future_high":
        target = row_pos + int(step)
        denom = entry_open[row_pos]
        source = high_values
    elif kind == "future_low":
        target = row_pos + int(step)
        denom = entry_open[row_pos]
        source = low_values
    else:
        raise ValueError(f"unsupported full path kind: {kind}")
    valid = (
        (target >= 0)
        & (target < len(daily))
        & (symbol_codes[target] == symbol_codes[row_pos])
        & np.isfinite(denom)
        & (denom > 0)
        & np.isfinite(source[target])
    )
    out = np.full(len(candidate), np.nan, dtype=np.float32)
    out[valid] = (source[target[valid]] / denom[valid] - 1.0).astype(np.float32, copy=False)
    return out


def _full_weighted_path_profiles(
    *,
    daily: pd.DataFrame,
    candidate: pd.DataFrame,
    labels: tuple[str, ...],
    pre_days: int,
    forward_days: int,
    progress_path: Path,
) -> pd.DataFrame:
    symbol_codes = pd.factorize(daily["symbol"], sort=False)[0].astype(np.int32, copy=False)
    label_specs = {
        label: _match_level_weights(candidate, label)[:3]
        for label in labels
    }
    rows: list[dict[str, Any]] = []
    path_steps = [
        ("pre_close", list(range(-int(pre_days), 1))),
        ("future_close", list(range(1, int(forward_days) + 1))),
        ("future_high", list(range(1, int(forward_days) + 1))),
        ("future_low", list(range(1, int(forward_days) + 1))),
    ]
    for kind, steps in path_steps:
        for step in steps:
            values = _path_values_for_kind(
                daily=daily,
                candidate=candidate,
                symbol_codes=symbol_codes,
                kind=kind,
                step=int(step),
            )
            for label in labels:
                event_mask, control_mask, levels = label_specs[label]
                finite_event = event_mask & np.isfinite(values)
                event_mean = float(np.mean(values[finite_event])) if np.any(finite_event) else float("nan")
                control_mean, control_weight = _weighted_control_mean(values, control_mask, levels)
                rows.append(
                    {
                        "label": label,
                        "cohort": "event_full",
                        "path_kind": kind,
                        "step": int(step),
                        "sample_count": int(finite_event.sum()),
                        "mean": event_mean,
                        "p25": float("nan"),
                        "median": float("nan"),
                        "p75": float("nan"),
                        "stat_scope": "full_exact_event_mean",
                    }
                )
                rows.append(
                    {
                        "label": label,
                        "cohort": "matched_control_full_weighted",
                        "path_kind": kind,
                        "step": int(step),
                        "sample_count": float(control_weight),
                        "mean": control_mean,
                        "p25": float("nan"),
                        "median": float("nan"),
                        "p75": float("nan"),
                        "stat_scope": "full_weighted_matched_control_mean",
                    }
                )
        _write_json(progress_path, {"status": "full_stream_path_profiles", "completed_kind": kind, "updated_at": _now()})
    return pd.DataFrame(rows)


def _matched_controls(frame: pd.DataFrame, *, events: pd.DataFrame, label: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    pool = frame[~frame[label].astype(bool)].copy()
    if "industry" not in pool.columns:
        pool["industry"] = "UNKNOWN"
    if "industry" not in events.columns:
        events["industry"] = "UNKNOWN"
    for col, q in [
        ("amount_mean_20d_log", "liquidity_bucket"),
        ("volatility_20d", "volatility_bucket"),
        ("close", "price_bucket"),
    ]:
        combined = pd.concat([pool[[col]], events[[col]]], ignore_index=True)
        ranks = combined[col].rank(method="average", pct=True)
        rank_values = np.nan_to_num(ranks.to_numpy(dtype=np.float64), nan=0.5, posinf=0.5, neginf=0.5)
        buckets = np.floor(np.clip(rank_values, 0, 0.999999) * 5).astype(np.int16)
        pool[q] = buckets[: len(pool)]
        events[q] = buckets[len(pool) :]
    match_cols = ["trade_date", "industry", "liquidity_bucket", "volatility_bucket", "price_bucket"]
    control_candidates: dict[tuple[Any, ...], np.ndarray] = {
        key: group.index.to_numpy(dtype=np.int64, copy=True)
        for key, group in pool.groupby(match_cols, sort=False)
    }
    loose_candidates: dict[tuple[Any, ...], np.ndarray] = {
        key: group.index.to_numpy(dtype=np.int64, copy=True)
        for key, group in pool.groupby(["trade_date", "industry"], sort=False)
    }
    date_candidates: dict[str, np.ndarray] = {
        str(key): group.index.to_numpy(dtype=np.int64, copy=True)
        for key, group in pool.groupby("trade_date", sort=False)
    }
    chosen: list[int] = []
    methods: list[str] = []
    for row in events[match_cols].itertuples(index=False, name=None):
        candidates = control_candidates.get(tuple(row))
        method = "date_industry_liquidity_vol_price"
        if candidates is None or len(candidates) == 0:
            candidates = loose_candidates.get((row[0], row[1]))
            method = "date_industry"
        if candidates is None or len(candidates) == 0:
            candidates = date_candidates.get(str(row[0]))
            method = "date_only"
        if candidates is None or len(candidates) == 0:
            continue
        chosen.append(int(candidates[int(rng.integers(0, len(candidates)))]))
        methods.append(method)
    controls = pool.loc[chosen].copy().reset_index(drop=True)
    controls["match_method"] = methods
    return controls


def _feature_contrast(
    *,
    all_frame: pd.DataFrame,
    events: pd.DataFrame,
    controls: pd.DataFrame,
    label: str,
    columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in columns:
        if col not in events.columns or col not in controls.columns:
            continue
        if col in all_frame.columns:
            all_values = pd.to_numeric(all_frame[col], errors="coerce")
        else:
            all_values = pd.Series(dtype="float64")
        event_values = pd.to_numeric(events[col], errors="coerce")
        control_values = pd.to_numeric(controls[col], errors="coerce")
        rows.append(
            {
                "label": label,
                "feature": col,
                "all_count": int(all_values.notna().sum()),
                "event_count": int(event_values.notna().sum()),
                "control_count": int(control_values.notna().sum()),
                "all_mean": float(all_values.mean()) if len(all_values) else float("nan"),
                "event_mean": float(event_values.mean()),
                "control_mean": float(control_values.mean()),
                "event_minus_all": float(event_values.mean() - all_values.mean()) if len(all_values) else float("nan"),
                "event_minus_control": float(event_values.mean() - control_values.mean()),
                "event_median": float(event_values.median()),
                "control_median": float(control_values.median()),
            }
        )
    return pd.DataFrame(rows)


def _path_profile(
    *,
    sorted_daily: pd.DataFrame,
    samples: pd.DataFrame,
    label: str,
    cohort: str,
    pre_days: int,
    forward_days: int,
) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    daily = sorted_daily
    close_values = daily["close"].to_numpy(dtype=np.float64, copy=False)
    high_values = daily["high"].to_numpy(dtype=np.float64, copy=False)
    low_values = daily["low"].to_numpy(dtype=np.float64, copy=False)
    entry_open_values = daily["entry_open_next"].to_numpy(dtype=np.float64, copy=False)
    row_pos = samples["_row_pos"].to_numpy(dtype=np.int64, copy=True)
    pre_close_paths: list[np.ndarray] = []
    forward_close_paths: list[np.ndarray] = []
    forward_high_paths: list[np.ndarray] = []
    forward_low_paths: list[np.ndarray] = []
    for pos in row_pos:
        if pos - pre_days < 0 or pos + forward_days >= len(daily):
            continue
        if daily.at[int(pos), "symbol"] != daily.at[int(pos - pre_days), "symbol"] or daily.at[int(pos), "symbol"] != daily.at[int(pos + forward_days), "symbol"]:
            continue
        base_close = close_values[int(pos)]
        entry_open = entry_open_values[int(pos)]
        if not math.isfinite(base_close) or base_close <= 0 or not math.isfinite(entry_open) or entry_open <= 0:
            continue
        pre_close_paths.append(close_values[int(pos - pre_days) : int(pos) + 1] / base_close - 1.0)
        forward_close_paths.append(close_values[int(pos + 1) : int(pos + forward_days) + 1] / entry_open - 1.0)
        forward_high_paths.append(high_values[int(pos + 1) : int(pos + forward_days) + 1] / entry_open - 1.0)
        forward_low_paths.append(low_values[int(pos + 1) : int(pos + forward_days) + 1] / entry_open - 1.0)
    rows: list[dict[str, Any]] = []
    for kind, arrs, steps in [
        ("pre_close", pre_close_paths, range(-pre_days, 1)),
        ("future_close", forward_close_paths, range(1, forward_days + 1)),
        ("future_high", forward_high_paths, range(1, forward_days + 1)),
        ("future_low", forward_low_paths, range(1, forward_days + 1)),
    ]:
        if not arrs:
            continue
        matrix = np.vstack(arrs).astype(np.float32)
        for idx, step in enumerate(steps):
            col = matrix[:, idx]
            rows.append(
                {
                    "label": label,
                    "cohort": cohort,
                    "path_kind": kind,
                    "step": int(step),
                    "sample_count": int(len(col)),
                    "mean": float(np.nanmean(col)),
                    "p25": float(np.nanpercentile(col, 25)),
                    "median": float(np.nanmedian(col)),
                    "p75": float(np.nanpercentile(col, 75)),
                }
            )
    return pd.DataFrame(rows)


def _build_cluster_features(daily: pd.DataFrame, samples: pd.DataFrame, *, pre_days: int, max_samples: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    sample = samples.copy()
    if len(sample) > int(max_samples):
        sample = sample.sample(n=int(max_samples), random_state=int(seed)).sort_values(["trade_date", "symbol"], kind="mergesort")
    close_values = daily["close"].to_numpy(dtype=np.float64, copy=False)
    volume_values = daily["volume"].to_numpy(dtype=np.float64, copy=False)
    amount_values = daily["amount"].to_numpy(dtype=np.float64, copy=False)
    row_pos = sample["_row_pos"].to_numpy(dtype=np.int64, copy=True)
    columns: list[str] = []
    vectors: list[np.ndarray] = []
    valid_indices: list[int] = []
    for pos in row_pos:
        if pos - pre_days < 0 or daily.at[int(pos), "symbol"] != daily.at[int(pos - pre_days), "symbol"]:
            continue
        base_close = close_values[int(pos)]
        if not math.isfinite(base_close) or base_close <= 0:
            continue
        pre_close = close_values[int(pos - pre_days) : int(pos) + 1] / base_close - 1.0
        pre_volume = np.log1p(volume_values[int(pos - pre_days) : int(pos) + 1])
        pre_amount = np.log1p(amount_values[int(pos - pre_days) : int(pos) + 1])
        if not (np.isfinite(pre_close).all() and np.isfinite(pre_volume).all() and np.isfinite(pre_amount).all()):
            continue
        vectors.append(np.concatenate([pre_close, pre_volume, pre_amount]).astype(np.float32))
        valid_indices.append(int(pos))
    if not vectors:
        return pd.DataFrame(), np.empty((0, 0), dtype=np.float32), []
    if not columns:
        columns = [f"close_rel_D{step:+d}" for step in range(-pre_days, 1)]
        columns += [f"volume_log_D{step:+d}" for step in range(-pre_days, 1)]
        columns += [f"amount_log_D{step:+d}" for step in range(-pre_days, 1)]
    out_samples = daily.iloc[valid_indices][["symbol", "trade_date", "_row_pos"]].copy().reset_index(drop=True)
    return out_samples, np.vstack(vectors).astype(np.float32), columns


def _cluster_columns(pre_days: int) -> list[str]:
    columns = [f"close_rel_D{step:+d}" for step in range(-int(pre_days), 1)]
    columns += [f"volume_log_D{step:+d}" for step in range(-int(pre_days), 1)]
    columns += [f"amount_log_D{step:+d}" for step in range(-int(pre_days), 1)]
    return columns


def _cluster_matrix_from_positions(
    *,
    daily: pd.DataFrame,
    row_positions: np.ndarray,
    pre_days: int,
    symbol_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    close_values = daily["close"].to_numpy(dtype=np.float64, copy=False)
    volume_values = daily["volume"].to_numpy(dtype=np.float64, copy=False)
    amount_values = daily["amount"].to_numpy(dtype=np.float64, copy=False)
    vectors: list[np.ndarray] = []
    valid_positions: list[int] = []
    for pos_raw in np.asarray(row_positions, dtype=np.int64).reshape(-1):
        pos = int(pos_raw)
        if pos - int(pre_days) < 0 or pos >= len(daily):
            continue
        if symbol_codes[pos] != symbol_codes[pos - int(pre_days)]:
            continue
        base_close = close_values[pos]
        if not math.isfinite(base_close) or base_close <= 0:
            continue
        pre_close = close_values[pos - int(pre_days) : pos + 1] / base_close - 1.0
        pre_volume = np.log1p(volume_values[pos - int(pre_days) : pos + 1])
        pre_amount = np.log1p(amount_values[pos - int(pre_days) : pos + 1])
        if not (np.isfinite(pre_close).all() and np.isfinite(pre_volume).all() and np.isfinite(pre_amount).all()):
            continue
        vectors.append(np.concatenate([pre_close, pre_volume, pre_amount]).astype(np.float32))
        valid_positions.append(pos)
    if not vectors:
        return np.empty((0, len(_cluster_columns(pre_days))), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.vstack(vectors).astype(np.float32), np.asarray(valid_positions, dtype=np.int64)


def _iter_cluster_batches(
    *,
    daily: pd.DataFrame,
    samples: pd.DataFrame,
    pre_days: int,
    batch_size: int,
    symbol_codes: np.ndarray,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    row_positions = samples["_row_pos"].to_numpy(dtype=np.int64, copy=True)
    for slc in _row_slices(len(row_positions), int(batch_size)):
        matrix, valid_positions = _cluster_matrix_from_positions(
            daily=daily,
            row_positions=row_positions[slc],
            pre_days=pre_days,
            symbol_codes=symbol_codes,
        )
        if matrix.shape[0] > 0:
            yield matrix, valid_positions


def _row_slices(length: int, batch_size: int) -> Iterable[slice]:
    step = max(1, int(batch_size))
    for start in range(0, int(length), step):
        yield slice(start, min(start + step, int(length)))


def _cluster_rising_events_full(
    *,
    daily: pd.DataFrame,
    samples: pd.DataFrame,
    pre_days: int,
    seed: int,
    cluster_count: int,
    progress_path: Path,
    batch_size: int = 50_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - depends on local env
        return (
            pd.DataFrame({"status": ["skipped"], "reason": [f"full_cluster_assignments_not_materialized:{exc}"]}),
            pd.DataFrame({"status": ["skipped"], "reason": [f"sklearn_unavailable:{exc}"]}),
            pd.DataFrame(),
        )
    if samples.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    symbol_codes = pd.factorize(daily["symbol"], sort=False)[0].astype(np.int32, copy=False)
    scaler = StandardScaler()
    fitted_rows = 0
    for batch_idx, (matrix, _positions) in enumerate(
        _iter_cluster_batches(daily=daily, samples=samples, pre_days=pre_days, batch_size=batch_size, symbol_codes=symbol_codes),
        start=1,
    ):
        scaler.partial_fit(matrix)
        fitted_rows += int(matrix.shape[0])
        if batch_idx % 10 == 0:
            _write_json(progress_path, {"status": "full_cluster_scaler", "fitted_rows": int(fitted_rows), "updated_at": _now()})
    if fitted_rows < int(cluster_count) * 10:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    model = MiniBatchKMeans(n_clusters=int(cluster_count), random_state=int(seed), batch_size=min(max(int(batch_size), 1024), 65536), n_init=1)
    trained_rows = 0
    for epoch in range(2):
        for batch_idx, (matrix, _positions) in enumerate(
            _iter_cluster_batches(daily=daily, samples=samples, pre_days=pre_days, batch_size=batch_size, symbol_codes=symbol_codes),
            start=1,
        ):
            model.partial_fit(scaler.transform(matrix))
            trained_rows += int(matrix.shape[0])
            if batch_idx % 10 == 0:
                _write_json(progress_path, {"status": "full_cluster_kmeans", "epoch": int(epoch + 1), "trained_rows": int(trained_rows), "updated_at": _now()})

    cluster_n = int(cluster_count)
    count = np.zeros(cluster_n, dtype=np.int64)
    sums = {
        "future_open_ret_20d": np.zeros(cluster_n, dtype=np.float64),
        "future_max_high_20d": np.zeros(cluster_n, dtype=np.float64),
        "future_min_low_20d": np.zeros(cluster_n, dtype=np.float64),
        "future_final_close_20d": np.zeros(cluster_n, dtype=np.float64),
        "future_peak_day_20d": np.zeros(cluster_n, dtype=np.float64),
        "hit_up10_before_down5_20d": np.zeros(cluster_n, dtype=np.float64),
        "rise_then_fade_10pct_20d": np.zeros(cluster_n, dtype=np.float64),
        "persistent_upside_10pct_20d": np.zeros(cluster_n, dtype=np.float64),
    }
    assigned_rows = 0
    for batch_idx, (matrix, positions) in enumerate(
        _iter_cluster_batches(daily=daily, samples=samples, pre_days=pre_days, batch_size=batch_size, symbol_codes=symbol_codes),
        start=1,
    ):
        cluster = model.predict(scaler.transform(matrix)).astype(np.int16)
        assigned_rows += int(len(cluster))
        count += np.bincount(cluster, minlength=cluster_n).astype(np.int64)
        outcomes = daily.iloc[positions]
        for name, target in [
            ("future_open_ret_20d", "future_open_ret_20d"),
            ("future_max_high_20d", "future_max_high_20d"),
            ("future_min_low_20d", "future_min_low_20d"),
            ("future_final_close_20d", "future_final_close_20d"),
            ("future_peak_day_20d", "future_peak_day_20d"),
            ("hit_up10_before_down5_20d", "hit_up10_before_down5_20d"),
            ("rise_then_fade_10pct_20d", "rise_then_fade_10pct_20d"),
            ("persistent_upside_10pct_20d", "persistent_upside_10pct_20d"),
        ]:
            values = pd.to_numeric(outcomes[target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            sums[name] += np.bincount(cluster, weights=values, minlength=cluster_n)
        if batch_idx % 10 == 0:
            _write_json(progress_path, {"status": "full_cluster_assign", "assigned_rows": int(assigned_rows), "updated_at": _now()})

    rows: list[dict[str, Any]] = []
    for cluster_id in range(cluster_n):
        denom = int(count[cluster_id])
        row: dict[str, Any] = {"cluster": int(cluster_id), "sample_count": denom}
        if denom > 0:
            row.update(
                {
                    "mean_future_open_ret_20d": float(sums["future_open_ret_20d"][cluster_id] / denom),
                    "mean_max_high_20d": float(sums["future_max_high_20d"][cluster_id] / denom),
                    "mean_min_low_20d": float(sums["future_min_low_20d"][cluster_id] / denom),
                    "mean_final_close_20d": float(sums["future_final_close_20d"][cluster_id] / denom),
                    "mean_peak_day": float(sums["future_peak_day_20d"][cluster_id] / denom),
                    "hit_up10_before_down5_rate": float(sums["hit_up10_before_down5_20d"][cluster_id] / denom),
                    "rise_then_fade_rate": float(sums["rise_then_fade_10pct_20d"][cluster_id] / denom),
                    "persistent_rate": float(sums["persistent_upside_10pct_20d"][cluster_id] / denom),
                }
            )
        rows.append(row)
    centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=_cluster_columns(pre_days))
    centers.insert(0, "cluster", np.arange(len(centers), dtype=np.int16))
    assignment_note = pd.DataFrame(
        {
            "status": ["not_materialized"],
            "reason": ["full_cluster_assignments_are_streamed_to_summary_to_avoid_multi_gb_csv"],
            "assigned_rows": [int(assigned_rows)],
        }
    )
    return assignment_note, pd.DataFrame(rows), centers


def _cluster_rising_events(
    *,
    daily: pd.DataFrame,
    samples: pd.DataFrame,
    pre_days: int,
    max_samples: int,
    seed: int,
    cluster_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - depends on local env
        return (
            pd.DataFrame(),
            pd.DataFrame({"status": ["skipped"], "reason": [f"sklearn_unavailable:{exc}"]}),
            pd.DataFrame(),
        )
    cluster_samples, matrix, columns = _build_cluster_features(daily, samples, pre_days=pre_days, max_samples=max_samples, seed=seed)
    if matrix.shape[0] < int(cluster_count) * 10:
        return cluster_samples, pd.DataFrame(), pd.DataFrame()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    model = MiniBatchKMeans(n_clusters=int(cluster_count), random_state=int(seed), batch_size=4096, n_init=10)
    cluster = model.fit_predict(scaled)
    cluster_samples = cluster_samples.copy()
    cluster_samples["cluster"] = cluster.astype(np.int16)
    outcomes = daily.loc[cluster_samples["_row_pos"].to_numpy(dtype=np.int64, copy=True), [
        "future_open_ret_20d",
        "future_max_high_20d",
        "future_min_low_20d",
        "future_final_close_20d",
        "future_peak_day_20d",
        "hit_up10_before_down5_20d",
        "rise_then_fade_10pct_20d",
        "persistent_upside_10pct_20d",
    ]].reset_index(drop=True)
    cluster_samples = pd.concat([cluster_samples.reset_index(drop=True), outcomes], axis=1)
    summary = cluster_samples.groupby("cluster", sort=True).agg(
        sample_count=("symbol", "count"),
        mean_future_open_ret_20d=("future_open_ret_20d", "mean"),
        mean_max_high_20d=("future_max_high_20d", "mean"),
        mean_min_low_20d=("future_min_low_20d", "mean"),
        mean_final_close_20d=("future_final_close_20d", "mean"),
        mean_peak_day=("future_peak_day_20d", "mean"),
        hit_up10_before_down5_rate=("hit_up10_before_down5_20d", "mean"),
        rise_then_fade_rate=("rise_then_fade_10pct_20d", "mean"),
        persistent_rate=("persistent_upside_10pct_20d", "mean"),
    ).reset_index()
    centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=columns)
    centers.insert(0, "cluster", np.arange(len(centers), dtype=np.int16))
    return cluster_samples, summary, centers


def _plot_path_profiles(path_profiles: pd.DataFrame, output_dir: Path) -> list[str]:
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for label in ["top10_return_20d", "hit_up10_before_down5_20d", "persistent_upside_10pct_20d", "rise_then_fade_10pct_20d"]:
        frame = path_profiles[(path_profiles["label"] == label) & (path_profiles["path_kind"].isin(["pre_close", "future_close"]))].copy()
        if frame.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, kind, title in [(axes[0], "pre_close", "Pre-entry close path"), (axes[1], "future_close", "Future close path")]:
            sub = frame[frame["path_kind"] == kind]
            for cohort, group in sub.groupby("cohort", sort=False):
                group = group.sort_values("step")
                ax.plot(group["step"], group["mean"] * 100.0, label=cohort)
                if "p25" in group.columns and "p75" in group.columns and group["p25"].notna().any() and group["p75"].notna().any():
                    ax.fill_between(group["step"], group["p25"] * 100.0, group["p75"] * 100.0, alpha=0.15)
            ax.axhline(0.0, color="#777777", linewidth=0.8)
            ax.set_title(title)
            ax.set_xlabel("Trading day offset")
            ax.set_ylabel("Return (%)")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(label)
        fig.tight_layout()
        path = chart_dir / f"path_profile_{label}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path.resolve()))
    return outputs


def _plot_feature_contrast(feature_contrast: pd.DataFrame, output_dir: Path) -> list[str]:
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for label in ["top10_return_20d", "hit_up10_before_down5_20d"]:
        frame = feature_contrast[feature_contrast["label"] == label].copy()
        if frame.empty:
            continue
        frame["abs_diff"] = frame["event_minus_control"].abs()
        frame = frame.sort_values("abs_diff", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 7))
        y = np.arange(len(frame))
        ax.barh(y, frame["event_minus_control"], color=np.where(frame["event_minus_control"] >= 0, "#2f6f9f", "#b75c4a"))
        ax.set_yticks(y)
        ax.set_yticklabels(frame["feature"], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0.0, color="#777777", linewidth=0.8)
        ax.set_title(f"Event minus matched control: {label}")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        path = chart_dir / f"feature_contrast_{label}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path.resolve()))
    return outputs


def _plot_cluster_summary(cluster_summary: pd.DataFrame, output_dir: Path) -> list[str]:
    if cluster_summary.empty or "cluster" not in cluster_summary.columns:
        return []
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    frame = cluster_summary.sort_values("cluster")
    ax.bar(frame["cluster"].astype(str), frame["hit_up10_before_down5_rate"] * 100.0, label="+10 before -5")
    ax.plot(frame["cluster"].astype(str), frame["persistent_rate"] * 100.0, color="#c44e52", marker="o", label="persistent")
    ax.set_title("Rising-event cluster outcome rates")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Rate (%)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = chart_dir / "cluster_outcome_rates.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [str(path.resolve())]


def _event_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = frame.copy()
    work["year"] = work["trade_date"].str.slice(0, 4).astype(int)
    for scope, group in [("all", work), *[(f"year={year}", group) for year, group in work.groupby("year", sort=True)]]:
        row: dict[str, Any] = {"scope": scope, "sample_count": int(len(group))}
        for label in LABELS:
            row[f"{label}_count"] = int(group[label].sum())
            row[f"{label}_rate"] = float(group[label].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _build_markdown_report(
    *,
    output_dir: Path,
    summary: Mapping[str, Any],
    event_summary: pd.DataFrame,
    feature_contrast: pd.DataFrame,
    full_feature_contrast: pd.DataFrame,
    full_aux_feature_contrast: pd.DataFrame,
    full_match_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    chart_paths: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# QDP v2 原始上涨路径画像")
    lines.append("")
    lines.append("## 方法")
    lines.append("")
    lines.append("这份报告直接使用 active QDP v2 的 `market_daily_raw` 原始日线事实表，并接入由 1m 产生且已校验的 `intraday_daily_features` 与 `limit_intraday_features`。它不使用 307 维训练特征、不使用 symbol。")
    lines.append("")
    lines.append("## 范围")
    lines.append("")
    lines.append(f"- event window: {summary.get('event_start')} to {summary.get('event_end')}")
    lines.append(f"- candidate rows: {summary.get('candidate_count', 0):,}")
    lines.append(f"- output_dir: `{summary.get('output_dir', '')}`")
    lines.append("")
    lines.append("## 上涨事件基准率")
    lines.append("")
    if not event_summary.empty:
        row = event_summary[event_summary["scope"].eq("all")].iloc[0].to_dict()
        for label in LABELS:
            lines.append(f"- {label}: {row.get(label + '_rate', 0) * 100:.2f}% ({int(row.get(label + '_count', 0)):,})")
    lines.append("")
    lines.append("## 匹配对照后的主要差异")
    lines.append("")
    if not full_feature_contrast.empty and not full_aux_feature_contrast.empty:
        contrast_for_report = pd.concat([full_feature_contrast, full_aux_feature_contrast], ignore_index=True)
    elif not full_feature_contrast.empty:
        contrast_for_report = full_feature_contrast
    elif not full_aux_feature_contrast.empty:
        contrast_for_report = full_aux_feature_contrast
    else:
        contrast_for_report = feature_contrast
    diff_column = "event_minus_weighted_control" if "event_minus_weighted_control" in contrast_for_report.columns else "event_minus_control"
    control_mean_column = "control_weighted_mean" if "control_weighted_mean" in contrast_for_report.columns else "control_mean"
    if not contrast_for_report.empty:
        for label in ["top10_return_20d", "hit_up10_before_down5_20d"]:
            top = contrast_for_report[contrast_for_report["label"].eq(label)].copy()
            top["abs_diff"] = top[diff_column].abs()
            top = top.sort_values("abs_diff", ascending=False).head(12)
            if top.empty:
                continue
            lines.append(f"### {label}")
            for row in top.to_dict("records"):
                lines.append(f"- {row['feature']}: event-control={row[diff_column]:.4f}, event_mean={row['event_mean']:.4f}, control_mean={row[control_mean_column]:.4f}")
            lines.append("")
    if not full_match_summary.empty:
        lines.append("## 全量加权匹配覆盖")
        lines.append("")
        for row in full_match_summary.to_dict("records"):
            lines.append(
                f"- {row['label']}: exact_bucket={row['match_date_industry_liquidity_vol_price_share']:.2%}, "
                f"date_industry={row['match_date_industry_share']:.2%}, "
                f"date_only={row['match_date_only_share']:.2%}, "
                f"unmatched={row['unmatched_share']:.2%}"
            )
        lines.append("")
    lines.append("## 上涨事件内部聚类")
    lines.append("")
    if not cluster_summary.empty and "cluster" in cluster_summary.columns:
        for row in cluster_summary.sort_values("cluster").to_dict("records"):
            lines.append(
                f"- cluster {int(row['cluster'])}: n={int(row['sample_count']):,}, "
                f"open20={row['mean_future_open_ret_20d'] * 100:.2f}%, "
                f"max_high={row['mean_max_high_20d'] * 100:.2f}%, "
                f"+10/-5={row['hit_up10_before_down5_rate'] * 100:.2f}%, "
                f"persistent={row['persistent_rate'] * 100:.2f}%, "
                f"fade={row['rise_then_fade_rate'] * 100:.2f}%"
            )
    lines.append("")
    lines.append("## 图表")
    lines.append("")
    for path in chart_paths:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## 解释边界")
    lines.append("")
    lines.append("- 这是 raw daily path + 1m-derived daily structure 研究，不是全量 1m 序列模型。")
    lines.append("- 匹配对照控制了同日、行业、流动性、波动和价格分桶，但仍不是因果证明。")
    lines.append("- 事件标签使用未来路径定义，因此只能用于画像和规则开发；实盘必须把入场规则固定后做独立回测。")
    lines.append("- 下一步若要真正超脱人工摘要，需要构建 raw sequence training pack，让模型直接读取日线/5m/1m 序列。")
    path = output_dir / "raw_rising_path_atlas_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.resolve())


@dataclass(frozen=True)
class AtlasConfig:
    qdp_root: Path
    output_root: Path
    run_tag: str
    event_start: str
    event_end: str
    labels: tuple[str, ...]
    pre_days: int
    forward_days: int
    max_events_per_label: int
    max_cluster_samples: int
    cluster_count: int
    seed: int
    full_stream_stats: bool
    full_cluster: bool


def build_raw_rising_path_atlas(config: AtlasConfig) -> dict[str, Any]:
    root = config.qdp_root.resolve()
    active = json.loads((root / "active" / "active.json").read_text(encoding="utf-8"))
    output_dir = config.output_root / f"{config.run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    _write_json(progress_path, {"status": "loading_daily_raw", "updated_at": _now()})

    daily = _read_dataset(root, active, "market_daily_raw", DAILY_RAW_COLUMNS)
    daily = daily[(daily["trade_date"] >= "2011-11-22") & (daily["trade_date"] <= "2026-06-26")].copy()
    daily = _add_raw_daily_signals(daily)
    daily = _add_forward_labels(daily, forward_days=config.forward_days)
    daily["_row_pos"] = np.arange(len(daily), dtype=np.int64)

    _write_json(progress_path, {"status": "loading_auxiliary_domains", "updated_at": _now(), "daily_rows": int(len(daily))})
    industry = _read_dataset(root, active, "industry_concept", INDUSTRY_COLUMNS)
    daily = daily.merge(industry, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    daily["industry"] = daily["industry"].fillna("UNKNOWN")

    candidate = daily[
        (daily["trade_date"] >= str(config.event_start))
        & (daily["trade_date"] <= str(config.event_end))
        & daily["path_valid_20d"].astype(bool)
        & daily["entry_open_next"].notna()
        & daily["prev_close"].notna()
    ].copy()
    candidate["year"] = candidate["trade_date"].str.slice(0, 4).astype(int)
    labels = tuple(label for label in config.labels if label in candidate.columns)
    if not labels:
        raise ValueError("no requested labels found")

    _write_json(progress_path, {"status": "building_event_profiles", "candidate_rows": int(len(candidate)), "updated_at": _now()})
    event_summary = _event_summary(candidate)
    full_feature_contrast = pd.DataFrame()
    full_aux_feature_contrast = pd.DataFrame()
    full_match_summary = pd.DataFrame()
    full_path_profiles = pd.DataFrame()
    if config.full_stream_stats:
        _write_json(progress_path, {"status": "full_stream_prepare_match_buckets", "candidate_rows": int(len(candidate)), "updated_at": _now()})
        candidate = _add_match_buckets(candidate)
        _write_json(progress_path, {"status": "full_stream_feature_contrast", "candidate_rows": int(len(candidate)), "updated_at": _now()})
        full_feature_contrast, full_match_summary = _full_weighted_feature_contrast(
            candidate,
            labels,
            [col for col in RAW_SIGNAL_COLUMNS if col in candidate.columns],
        )
        _write_json(progress_path, {"status": "full_stream_path_profiles", "candidate_rows": int(len(candidate)), "updated_at": _now()})
        full_path_profiles = _full_weighted_path_profiles(
            daily=daily,
            candidate=candidate,
            labels=labels,
            pre_days=config.pre_days,
            forward_days=config.forward_days,
            progress_path=progress_path,
        )
        _write_json(progress_path, {"status": "full_aux_feature_contrast", "candidate_rows": int(len(candidate)), "updated_at": _now()})
        full_aux_feature_contrast = _full_auxiliary_feature_contrast(
            root=root,
            active=active,
            candidate=candidate,
            labels=labels,
            progress_path=progress_path,
        )
    path_frames: list[pd.DataFrame] = []
    match_frames: list[pd.DataFrame] = []
    sampled_frames: list[pd.DataFrame] = []
    for label in labels:
        events = _sample_events(candidate, label=label, max_events=config.max_events_per_label, seed=config.seed)
        controls = _matched_controls(candidate, events=events, label=label, seed=config.seed + len(label))
        events = events.reset_index(drop=True)
        controls = controls.reset_index(drop=True)
        events["__label"] = label
        events["__cohort"] = "event"
        controls["__label"] = label
        controls["__cohort"] = "matched_control"
        sampled_frames.extend([events, controls])
        match_frames.append(
            pd.DataFrame(
                {
                    "label": [label],
                    "event_count": [int(len(events))],
                    "control_count": [int(len(controls))],
                    "match_date_industry_liquidity_vol_price_share": [
                        float(controls["match_method"].eq("date_industry_liquidity_vol_price").mean()) if "match_method" in controls.columns and len(controls) else np.nan
                    ],
                    "match_date_industry_share": [
                        float(controls["match_method"].eq("date_industry").mean()) if "match_method" in controls.columns and len(controls) else np.nan
                    ],
                    "match_date_only_share": [
                        float(controls["match_method"].eq("date_only").mean()) if "match_method" in controls.columns and len(controls) else np.nan
                    ],
                }
            )
        )
        path_frames.append(_path_profile(sorted_daily=daily, samples=events, label=label, cohort="event", pre_days=config.pre_days, forward_days=config.forward_days))
        path_frames.append(_path_profile(sorted_daily=daily, samples=controls, label=label, cohort="matched_control", pre_days=config.pre_days, forward_days=config.forward_days))

    sampled = pd.concat(sampled_frames, ignore_index=True) if sampled_frames else pd.DataFrame()
    aux_keys = sampled[["symbol", "trade_date"]].drop_duplicates() if not sampled.empty else pd.DataFrame(columns=["symbol", "trade_date"])
    _write_json(progress_path, {"status": "loading_auxiliary_for_sampled_events", "sampled_keys": int(len(aux_keys)), "updated_at": _now()})
    intraday = _read_dataset_for_keys_by_year(root, active, "intraday_daily_features", INTRADAY_COLUMNS, aux_keys)
    limit_features = _read_dataset_for_keys_by_year(root, active, "limit_intraday_features", LIMIT_COLUMNS, aux_keys)
    if not intraday.empty:
        sampled = sampled.merge(intraday, on=["symbol", "trade_date"], how="left")
    if not limit_features.empty:
        sampled = sampled.merge(limit_features, on=["symbol", "trade_date"], how="left")
    for col in LIMIT_SIGNAL_COLUMNS:
        if col in sampled.columns and sampled[col].dtype == bool:
            sampled[col] = sampled[col].astype("float32")
    feature_columns = [col for col in [*RAW_SIGNAL_COLUMNS, *INTRADAY_SIGNAL_COLUMNS, *LIMIT_SIGNAL_COLUMNS] if col in sampled.columns]
    contrast_frames: list[pd.DataFrame] = []
    for label in labels:
        events = sampled[(sampled["__label"].eq(label)) & (sampled["__cohort"].eq("event"))].copy()
        controls = sampled[(sampled["__label"].eq(label)) & (sampled["__cohort"].eq("matched_control"))].copy()
        contrast_frames.append(_feature_contrast(all_frame=candidate, events=events, controls=controls, label=label, columns=feature_columns))

    if config.full_cluster:
        _write_json(progress_path, {"status": "full_cluster_prepare_events", "candidate_rows": int(len(candidate)), "updated_at": _now()})
        cluster_labels = [label for label in ("top10_return_20d", "hit_up10_before_down5_20d") if label in candidate.columns]
        if cluster_labels:
            cluster_mask = np.zeros(len(candidate), dtype=bool)
            for label in cluster_labels:
                cluster_mask |= candidate[label].astype(bool).to_numpy()
            all_top_events = candidate.loc[cluster_mask, ["symbol", "trade_date", "_row_pos"]].drop_duplicates(["symbol", "trade_date"])
        else:
            all_top_events = pd.DataFrame(columns=["symbol", "trade_date", "_row_pos"])
        cluster_samples, cluster_summary, cluster_centers = _cluster_rising_events_full(
            daily=daily,
            samples=all_top_events,
            pre_days=config.pre_days,
            seed=config.seed,
            cluster_count=config.cluster_count,
            progress_path=progress_path,
        )
    else:
        all_top_events = pd.concat(
            [
                _sample_events(candidate, label="top10_return_20d", max_events=config.max_cluster_samples // 2, seed=config.seed),
                _sample_events(candidate, label="hit_up10_before_down5_20d", max_events=config.max_cluster_samples // 2, seed=config.seed + 11),
            ],
            ignore_index=True,
        ).drop_duplicates(["symbol", "trade_date"])
        cluster_samples, cluster_summary, cluster_centers = _cluster_rising_events(
            daily=daily,
            samples=all_top_events,
            pre_days=config.pre_days,
            max_samples=config.max_cluster_samples,
            seed=config.seed,
            cluster_count=config.cluster_count,
        )

    feature_contrast = pd.concat(contrast_frames, ignore_index=True) if contrast_frames else pd.DataFrame()
    path_profiles = pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame()
    match_summary = pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame()

    outputs: dict[str, Any] = {
        "event_summary_csv": _write_csv(output_dir / "event_summary_by_year.csv", event_summary),
        "feature_contrast_csv": _write_csv(output_dir / "raw_feature_contrast_vs_matched_controls.csv", feature_contrast),
        "path_profiles_csv": _write_csv(output_dir / "raw_path_profiles.csv", path_profiles),
        "match_summary_csv": _write_csv(output_dir / "matched_control_summary.csv", match_summary),
        "full_stream_feature_contrast_csv": _write_csv(output_dir / "full_stream_raw_feature_contrast_weighted.csv", full_feature_contrast),
        "full_stream_aux_feature_contrast_csv": _write_csv(output_dir / "full_stream_aux_feature_contrast_weighted.csv", full_aux_feature_contrast),
        "full_stream_path_profiles_csv": _write_csv(output_dir / "full_stream_raw_path_profiles_weighted.csv", full_path_profiles),
        "full_stream_match_summary_csv": _write_csv(output_dir / "full_stream_match_summary.csv", full_match_summary),
        "cluster_samples_csv": _write_csv(output_dir / "rising_cluster_samples.csv", cluster_samples),
        "cluster_summary_csv": _write_csv(output_dir / "rising_cluster_summary.csv", cluster_summary),
        "cluster_centers_csv": _write_csv(output_dir / "rising_cluster_centers.csv", cluster_centers),
    }
    chart_paths: list[str] = []
    chart_paths.extend(_plot_path_profiles(full_path_profiles if not full_path_profiles.empty else path_profiles, output_dir))
    chart_paths.extend(_plot_feature_contrast(feature_contrast, output_dir))
    chart_paths.extend(_plot_cluster_summary(cluster_summary, output_dir))
    outputs["charts"] = chart_paths

    summary: dict[str, Any] = {
        "artifact_type": "qdp_v2_raw_rising_path_atlas",
        "generated_at": _now(),
        "qdp_root": str(root),
        "active_manifest": str((root / "active" / "active.json").resolve()),
        "output_dir": str(output_dir.resolve()),
        "event_start": str(config.event_start),
        "event_end": str(config.event_end),
        "candidate_count": int(len(candidate)),
        "labels": list(labels),
        "pre_days": int(config.pre_days),
        "forward_days": int(config.forward_days),
        "max_events_per_label": int(config.max_events_per_label),
        "max_cluster_samples": int(config.max_cluster_samples),
        "cluster_count": int(config.cluster_count),
        "full_stream_stats": bool(config.full_stream_stats),
        "full_cluster": bool(config.full_cluster),
        "inputs": {
            "daily": dict(active.get("datasets", {}) or {}).get("market_daily_raw", ""),
            "industry": dict(active.get("datasets", {}) or {}).get("industry_concept", ""),
            "intraday_daily_features": dict(active.get("datasets", {}) or {}).get("intraday_daily_features", ""),
            "limit_intraday_features": dict(active.get("datasets", {}) or {}).get("limit_intraday_features", ""),
        },
        "outputs": outputs,
    }
    report_path = _build_markdown_report(
        output_dir=output_dir,
        summary=summary,
        event_summary=event_summary,
        feature_contrast=feature_contrast,
        full_feature_contrast=full_feature_contrast,
        full_aux_feature_contrast=full_aux_feature_contrast,
        full_match_summary=full_match_summary,
        cluster_summary=cluster_summary,
        chart_paths=chart_paths,
    )
    summary["outputs"]["report_md"] = report_path
    summary_path = _write_json(output_dir / "raw_rising_path_atlas_summary.json", summary)
    summary["outputs"]["summary_json"] = summary_path
    _write_json(progress_path, {"status": "completed", "summary_json": summary_path, "report_md": report_path, "updated_at": _now()})
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a raw daily path atlas for rising A-share events from QDP v2 active data.")
    parser.add_argument("--qdp-root", default=str(DEFAULT_QDP_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-tag", default="qdp_v2_raw_rising_path_atlas")
    parser.add_argument("--event-start", default=DEFAULT_EVENT_START)
    parser.add_argument("--event-end", default=DEFAULT_EVENT_END)
    parser.add_argument("--labels", default=",".join(LABELS))
    parser.add_argument("--pre-days", type=int, default=DEFAULT_PRE_DAYS)
    parser.add_argument("--forward-days", type=int, default=DEFAULT_FORWARD_DAYS)
    parser.add_argument("--max-events-per-label", type=int, default=DEFAULT_MAX_EVENTS_PER_LABEL)
    parser.add_argument("--max-cluster-samples", type=int, default=DEFAULT_MAX_CLUSTER_SAMPLES)
    parser.add_argument("--cluster-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--full-stream-stats", action="store_true")
    parser.add_argument("--full-cluster", action="store_true", help="Cluster all rising events with streaming MiniBatchKMeans instead of sampling.")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = build_raw_rising_path_atlas(
        AtlasConfig(
            qdp_root=Path(args.qdp_root),
            output_root=Path(args.output_root),
            run_tag=str(args.run_tag),
            event_start=str(args.event_start),
            event_end=str(args.event_end),
            labels=_parse_csv_strings(args.labels, default=tuple(LABELS)),
            pre_days=int(args.pre_days),
            forward_days=int(args.forward_days),
            max_events_per_label=int(args.max_events_per_label),
            max_cluster_samples=int(args.max_cluster_samples),
            cluster_count=int(args.cluster_count),
            seed=int(args.seed),
            full_stream_stats=bool(args.full_stream_stats),
            full_cluster=bool(args.full_cluster),
        )
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(summary.get("outputs", {}).get("report_md", ""))
    return summary


if __name__ == "__main__":
    main()
