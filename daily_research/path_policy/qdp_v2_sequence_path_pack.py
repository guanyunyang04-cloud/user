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
DEFAULT_OUTPUT_ROOT = Path("daily_research/data/research_store/sequence_pack")
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

PRICE_ADJUSTMENT_NONE = "none"
PRICE_ADJUSTMENT_BACK = "back_adjust"
ENTRY_RULE_LEGACY = "legacy_999_tolerance"
ENTRY_RULE_OPEN_BELOW_LIMIT = "open_below_limit_tick"
SAMPLE_FILTER_COMPLETE_CASE = "complete_case_filled"
SAMPLE_FILTER_SIGNAL_ELIGIBLE = "signal_eligible_price_label"
SUSPENSION_FILL_NONE = "none"
SUSPENSION_FILL_CARRY_CLOSE = "carry_adjusted_close"
REQUIRED_PIT_MARKET_VIEW_OVERRIDES = {
    "market_daily_raw",
    "adjust_factor",
    "limit_status",
    "security_status",
    "pit_signal_universe",
}

PIT_MAINBOARD_PREFIXES = ("000", "001", "002", "003", "600", "601", "603", "605")


def _is_pit_mainboard_symbol(symbol: str) -> bool:
    normalized = str(symbol).upper().strip()
    code = normalized.split(".", 1)[0]
    exchange = normalized.split(".", 1)[1] if "." in normalized else ""
    return exchange in {"SH", "SZ"} and code.startswith(PIT_MAINBOARD_PREFIXES)


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


def _apply_research_dataset_view(
    root: Path,
    active: Mapping[str, Any],
    view_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    if view_path is None:
        return dict(active), None, ""
    path = Path(view_path).resolve()
    view = json.loads(path.read_text(encoding="utf-8-sig"))
    if str(view.get("kind", "") or "") != "qdp_v2_research_dataset_view":
        raise ValueError(f"not a qdp_v2 research dataset view: {path}")
    datasets = dict(view.get("datasets", {}) or {})
    overrides = dict(view.get("overrides", {}) or {})
    missing_overrides = sorted(REQUIRED_PIT_MARKET_VIEW_OVERRIDES.difference(overrides))
    if missing_overrides:
        raise ValueError(f"research dataset view missing required overrides: {missing_overrides}")
    for domain, dataset_id in datasets.items():
        manifest_path = root / "datasets" / str(domain) / str(dataset_id) / "dataset.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"research dataset view manifest not found: {manifest_path}")
    selected = dict(active)
    selected["datasets"] = datasets
    selected["research_dataset_view"] = {
        "view_id": str(view.get("view_id", "") or ""),
        "view_path": str(path),
        "overrides": overrides,
    }
    return selected, view, str(path)


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


def _read_dataset_symbols_date_range(root: Path, active: Mapping[str, Any], domain: str, start: str, end: str) -> list[str]:
    manifest = _dataset_manifest(root, active, domain)
    paths = _relative_shard_paths(root, manifest)
    filt = (ds.field("trade_date") >= str(start)) & (ds.field("trade_date") <= str(end))
    table = ds.dataset([str(path) for path in paths], format="parquet").to_table(columns=["symbol"], filter=filt)
    return sorted({str(item).upper().strip() for item in table.column("symbol").unique().to_pylist() if str(item).strip()})


def _sql_path_list(paths: Iterable[Path]) -> str:
    quoted = ["'" + str(Path(path).resolve()).replace("'", "''") + "'" for path in paths]
    return "[" + ",".join(quoted) + "]"


def _audit_pit_price_coverage(
    root: Path,
    active: Mapping[str, Any],
    *,
    start: str,
    end: str,
) -> dict[str, Any]:
    """Exact Gate-0 anti-join between PIT-eligible rows and the active daily substrate."""

    import duckdb  # type: ignore

    pit_manifest = _dataset_manifest(root, active, "pit_signal_universe")
    daily_manifest = _dataset_manifest(root, active, "market_daily_raw")
    pit_sql = _sql_path_list(_relative_shard_paths(root, pit_manifest))
    daily_sql = _sql_path_list(_relative_shard_paths(root, daily_manifest))
    start_sql = str(start).replace("'", "''")
    end_sql = str(end).replace("'", "''")
    con = duckdb.connect(":memory:")
    try:
        row = con.execute(
            f"""
            with missing as (
              select u.trade_date, u.symbol
              from read_parquet({pit_sql}, union_by_name=true) u
              left join read_parquet({daily_sql}, union_by_name=true) d
                on u.trade_date = d.trade_date and u.symbol = d.symbol
              where cast(u.eligible_for_signal as boolean)
                and u.trade_date between '{start_sql}' and '{end_sql}'
                and d.symbol is null
            )
            select
              count(*) as missing_rows,
              count(distinct symbol) as missing_symbols,
              min(trade_date) as first_missing_date,
              max(trade_date) as last_missing_date
            from missing
            """
        ).fetchone()
        examples = con.execute(
            f"""
            select u.trade_date, u.symbol
            from read_parquet({pit_sql}, union_by_name=true) u
            left join read_parquet({daily_sql}, union_by_name=true) d
              on u.trade_date = d.trade_date and u.symbol = d.symbol
            where cast(u.eligible_for_signal as boolean)
              and u.trade_date between '{start_sql}' and '{end_sql}'
              and d.symbol is null
            order by u.trade_date, u.symbol
            limit 5
            """
        ).fetchall()
    finally:
        con.close()
    return {
        "missing_rows": int(row[0] or 0),
        "missing_symbols": int(row[1] or 0),
        "first_missing_date": str(row[2] or ""),
        "last_missing_date": str(row[3] or ""),
        "examples": [{"trade_date": str(item[0]), "symbol": str(item[1])} for item in examples],
    }


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


def _write_scalar_panel_values(
    panel: np.ndarray,
    frame: pd.DataFrame,
    value_column: str,
    *,
    date_to_idx: Mapping[str, int],
    symbol_to_idx: Mapping[str, int],
) -> None:
    if frame.empty or value_column not in frame.columns:
        return
    indexed = _index_frame(frame, date_to_idx, symbol_to_idx)
    if indexed.empty:
        return
    values = pd.to_numeric(indexed[value_column], errors="coerce").to_numpy(dtype=np.float32, copy=True)
    panel[indexed["_date_idx"].to_numpy(dtype=np.int64), indexed["_symbol_idx"].to_numpy(dtype=np.int64)] = values
    if hasattr(panel, "flush"):
        panel.flush()


def _write_status_panels(
    *,
    frame: pd.DataFrame,
    status_valid: np.ndarray,
    is_st: np.ndarray,
    is_suspended: np.ndarray,
    is_delisted: np.ndarray,
    date_to_idx: Mapping[str, int],
    symbol_to_idx: Mapping[str, int],
) -> None:
    if frame.empty:
        return
    indexed = _index_frame(frame, date_to_idx, symbol_to_idx)
    if indexed.empty:
        return
    date_idx = indexed["_date_idx"].to_numpy(dtype=np.int64)
    symbol_idx = indexed["_symbol_idx"].to_numpy(dtype=np.int64)
    status_valid[date_idx, symbol_idx] = True
    for column, panel in (("is_st", is_st), ("is_suspended", is_suspended), ("is_delisted", is_delisted)):
        values = indexed[column].fillna(False).astype(bool).to_numpy(copy=True) if column in indexed.columns else np.zeros(len(indexed), dtype=bool)
        panel[date_idx, symbol_idx] = values
    for panel in (status_valid, is_st, is_suspended, is_delisted):
        if hasattr(panel, "flush"):
            panel.flush()


def _write_pit_universe_panels(
    *,
    frame: pd.DataFrame,
    status_valid: np.ndarray,
    is_st: np.ndarray,
    is_suspended: np.ndarray,
    is_delisted: np.ndarray,
    universe_has_bar: np.ndarray,
    signal_eligible: np.ndarray,
    date_to_idx: Mapping[str, int],
    symbol_to_idx: Mapping[str, int],
) -> None:
    if frame.empty:
        return
    indexed = _index_frame(frame, date_to_idx, symbol_to_idx)
    if indexed.empty:
        return
    date_idx = indexed["_date_idx"].to_numpy(dtype=np.int64)
    symbol_idx = indexed["_symbol_idx"].to_numpy(dtype=np.int64)
    status_valid[date_idx, symbol_idx] = True
    mappings = (
        ("is_st", is_st),
        ("is_suspended", is_suspended),
        ("is_delisted", is_delisted),
        ("has_bar", universe_has_bar),
        ("eligible_for_signal", signal_eligible),
    )
    for column, panel in mappings:
        if column not in indexed.columns:
            raise ValueError(f"PIT signal universe is missing required column: {column}")
        panel[date_idx, symbol_idx] = indexed[column].fillna(False).astype(bool).to_numpy(copy=True)
    for panel in (status_valid, is_st, is_suspended, is_delisted, universe_has_bar, signal_eligible):
        if hasattr(panel, "flush"):
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


def _apply_back_adjustment(daily: pd.DataFrame, factor: pd.DataFrame) -> pd.DataFrame:
    """Apply the canonical back-adjust factor to OHLC while preserving raw execution prices.

    The factor is a label/feature-side price transformation.  Volume and amount remain in
    their native units, and ``raw_open`` is retained so the next-open fill rule can still be
    evaluated in the exchange's 0.01-yuan tick space.
    """

    required_daily = {"symbol", "trade_date", *PATH_OHLC_FIELDS}
    missing_daily = sorted(required_daily.difference(daily.columns))
    if missing_daily:
        raise ValueError(f"daily frame is missing price columns: {missing_daily}")
    factor_column = "adjust_factor" if "adjust_factor" in factor.columns else "back_adjust_factor"
    required_factor = {"symbol", "trade_date", factor_column}
    missing_factor = sorted(required_factor.difference(factor.columns))
    if missing_factor:
        raise ValueError(f"adjust factor frame is missing columns: {missing_factor}")

    factor_values = factor[["symbol", "trade_date", factor_column]].copy()
    factor_values["symbol"] = factor_values["symbol"].astype(str).str.upper().str.strip()
    factor_values["trade_date"] = factor_values["trade_date"].astype(str)
    factor_values[factor_column] = pd.to_numeric(factor_values[factor_column], errors="coerce")
    if bool(factor_values.duplicated(["symbol", "trade_date"]).any()):
        raise ValueError("adjust factor contains duplicate symbol-day keys")

    out = daily.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out["trade_date"] = out["trade_date"].astype(str)
    out["raw_open"] = pd.to_numeric(out["open"], errors="coerce").astype("float64")
    out = out.merge(
        factor_values.rename(columns={factor_column: "price_adjust_factor"}),
        on=["symbol", "trade_date"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    valid_factor = np.isfinite(out["price_adjust_factor"].to_numpy(dtype=np.float64, copy=False)) & out[
        "price_adjust_factor"
    ].gt(0.0).to_numpy(dtype=bool, copy=False)
    if not bool(valid_factor.all()):
        bad = out.loc[~valid_factor, ["symbol", "trade_date"]].head(5).to_dict("records")
        raise ValueError(f"back-adjust factor must be finite and positive for every daily bar; examples={bad}")
    for column in PATH_OHLC_FIELDS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64") * out["price_adjust_factor"]
    return out


def _price_to_tick_units(values: np.ndarray, *, tick_size: float = 0.01) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float64)
    return np.floor(numeric / float(tick_size) + 0.5)


def _entry_fill_mask(
    entry_open: np.ndarray,
    entry_up_limit: np.ndarray,
    *,
    entry_suspended: np.ndarray | None = None,
    rule: str = ENTRY_RULE_LEGACY,
    tick_size: float = 0.01,
) -> np.ndarray:
    open_values = np.asarray(entry_open, dtype=np.float64)
    limit_values = np.asarray(entry_up_limit, dtype=np.float64)
    entry_ok = np.isfinite(open_values)
    if entry_suspended is not None:
        entry_ok &= ~np.asarray(entry_suspended, dtype=bool)
    normalized_rule = str(rule).strip().lower()
    if normalized_rule == ENTRY_RULE_LEGACY:
        limit_blocked = np.isfinite(limit_values) & entry_ok & (open_values >= limit_values * 0.999)
    elif normalized_rule == ENTRY_RULE_OPEN_BELOW_LIMIT:
        limit_blocked = (
            np.isfinite(limit_values)
            & entry_ok
            & (_price_to_tick_units(open_values, tick_size=tick_size) >= _price_to_tick_units(limit_values, tick_size=tick_size))
        )
    else:
        raise ValueError(f"unsupported entry rule: {rule}")
    return entry_ok & (~limit_blocked)


def _fill_suspended_daily_raw(
    raw_panel: np.ndarray,
    *,
    suspended_panel: np.ndarray,
    has_bar_panel: np.ndarray,
) -> None:
    """Carry the last observed adjusted close through explicit suspension days in-place."""

    if raw_panel.shape[:2] != suspended_panel.shape or raw_panel.shape[:2] != has_bar_panel.shape:
        raise ValueError("daily, suspended, and has_bar panels must share date/symbol dimensions")
    open_idx = DAILY_RAW_FEATURES.index("open")
    high_idx = DAILY_RAW_FEATURES.index("high")
    low_idx = DAILY_RAW_FEATURES.index("low")
    close_idx = DAILY_RAW_FEATURES.index("close")
    zero_columns = [
        DAILY_RAW_FEATURES.index("volume"),
        DAILY_RAW_FEATURES.index("amount"),
        DAILY_RAW_FEATURES.index("volume_log"),
        DAILY_RAW_FEATURES.index("amount_log"),
        DAILY_RAW_FEATURES.index("open_ret_prev_close"),
        DAILY_RAW_FEATURES.index("high_ret_prev_close"),
        DAILY_RAW_FEATURES.index("low_ret_prev_close"),
        DAILY_RAW_FEATURES.index("close_ret_prev_close"),
        DAILY_RAW_FEATURES.index("intraday_range_raw"),
    ]
    close_panel = raw_panel[:, :, close_idx]
    n_dates, n_symbols = close_panel.shape
    for symbol_idx in range(n_symbols):
        last_close = np.nan
        for date_idx in range(n_dates):
            current_close = float(close_panel[date_idx, symbol_idx])
            if bool(has_bar_panel[date_idx, symbol_idx]) and math.isfinite(current_close):
                last_close = current_close
                continue
            if not bool(suspended_panel[date_idx, symbol_idx]) or not math.isfinite(last_close):
                continue
            raw_panel[date_idx, symbol_idx, [open_idx, high_idx, low_idx, close_idx]] = np.float32(last_close)
            raw_panel[date_idx, symbol_idx, zero_columns] = np.float32(0.0)
    if hasattr(raw_panel, "flush"):
        raw_panel.flush()


def _compute_future_path_and_masks(
    *,
    raw_panel: np.ndarray,
    up_limit_panel: np.ndarray,
    lookback_days: int,
    forward_days: int,
    price_anchor: str = "next_open",
    raw_entry_open_panel: np.ndarray | None = None,
    suspended_panel: np.ndarray | None = None,
    entry_rule: str = ENTRY_RULE_LEGACY,
    separate_price_va_validity: bool = False,
    price_label_valid_out: np.ndarray | None = None,
    va_aux_valid_out: np.ndarray | None = None,
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
    execution_open_panel = open_panel if raw_entry_open_panel is None else np.asarray(raw_entry_open_panel)
    if execution_open_panel.shape != open_panel.shape:
        raise ValueError("raw_entry_open_panel must match the daily date/symbol shape")
    suspension_values = np.zeros_like(open_panel, dtype=bool) if suspended_panel is None else np.asarray(suspended_panel, dtype=bool)
    if suspension_values.shape != open_panel.shape:
        raise ValueError("suspended_panel must match the daily date/symbol shape")
    input_valid = np.zeros((n_dates, n_symbols), dtype=bool)
    entry_buyable = np.zeros((n_dates, n_symbols), dtype=bool)
    label_valid = np.zeros((n_dates, n_symbols), dtype=bool)
    price_label_valid = (
        np.zeros((n_dates, n_symbols), dtype=bool) if price_label_valid_out is None else price_label_valid_out
    )
    va_aux_valid = np.zeros((n_dates, n_symbols), dtype=bool) if va_aux_valid_out is None else va_aux_valid_out
    if price_label_valid.shape != (n_dates, n_symbols) or va_aux_valid.shape != (n_dates, n_symbols):
        raise ValueError("price_label_valid_out and va_aux_valid_out must match the daily date/symbol shape")
    price_label_valid[:] = False
    va_aux_valid[:] = False
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
        execution_entry_open = execution_open_panel[entry_idx].astype("float64", copy=False)
        signal_close = close_panel[date_idx].astype("float64", copy=False)
        entry_up_limit = up_limit_panel[entry_idx].astype("float64", copy=False)
        anchor_ok = np.isfinite(entry_open)
        entry_buyable[date_idx] = _entry_fill_mask(
            execution_entry_open,
            entry_up_limit,
            entry_suspended=suspension_values[entry_idx],
            rule=str(entry_rule),
        )
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
        price_ok = (
            np.isfinite(fut_open).all(axis=0)
            & np.isfinite(fut_high).all(axis=0)
            & np.isfinite(fut_low).all(axis=0)
            & np.isfinite(fut_close).all(axis=0)
            & anchor_ok
        )
        va_ok = (
            np.isfinite(fut_volume_log).all(axis=0)
            & np.isfinite(fut_amount_log).all(axis=0)
            & np.isfinite(trailing_volume_log)
            & np.isfinite(trailing_amount_log)
        )
        if str(price_anchor) == "today_close":
            denom = np.where(signal_close != 0.0, signal_close, np.nan)
        elif str(price_anchor) == "next_open":
            denom = np.where(entry_open != 0.0, entry_open, np.nan)
        else:
            raise ValueError(f"unsupported price_anchor: {price_anchor}")
        price_ok &= np.isfinite(denom)
        price_label_valid[date_idx] = price_ok
        va_aux_valid[date_idx] = va_ok
        path_ok = price_ok if bool(separate_price_va_validity) else (price_ok & va_ok)
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
    signal_eligible: np.ndarray | None = None,
    require_entry_filled: bool = True,
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
        mask = input_valid[date_idx] & label_valid[date_idx]
        if signal_eligible is not None:
            mask &= np.asarray(signal_eligible[date_idx], dtype=bool)
        if bool(require_entry_filled):
            mask &= entry_buyable[date_idx]
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
                    "entry_filled": entry_buyable[date_idx, symbol_idx],
                }
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "split",
                "year",
                "trade_date",
                "date_idx",
                "symbol_idx",
                "symbol",
                "entry_trade_date",
                "entry_filled",
            ]
        )
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
    price_adjustment: str = PRICE_ADJUSTMENT_NONE
    entry_rule: str = ENTRY_RULE_LEGACY
    sample_filter: str = SAMPLE_FILTER_COMPLETE_CASE
    suspension_fill: str = SUSPENSION_FILL_NONE
    pit_universe_manifest: Path | None = None
    dataset_view: Path | None = None


def build_sequence_pack(config: SequencePackConfig) -> dict[str, Any]:
    price_adjustment = str(config.price_adjustment).strip().lower()
    entry_rule = str(config.entry_rule).strip().lower()
    sample_filter = str(config.sample_filter).strip().lower()
    suspension_fill = str(config.suspension_fill).strip().lower()
    if price_adjustment not in {PRICE_ADJUSTMENT_NONE, PRICE_ADJUSTMENT_BACK}:
        raise ValueError(f"unsupported price adjustment: {config.price_adjustment}")
    if entry_rule not in {ENTRY_RULE_LEGACY, ENTRY_RULE_OPEN_BELOW_LIMIT}:
        raise ValueError(f"unsupported entry rule: {config.entry_rule}")
    if sample_filter not in {SAMPLE_FILTER_COMPLETE_CASE, SAMPLE_FILTER_SIGNAL_ELIGIBLE}:
        raise ValueError(f"unsupported sample filter: {config.sample_filter}")
    if suspension_fill not in {SUSPENSION_FILL_NONE, SUSPENSION_FILL_CARRY_CLOSE}:
        raise ValueError(f"unsupported suspension fill: {config.suspension_fill}")
    root = config.qdp_root.resolve()
    canonical_active = _read_active(root)
    active_manifest_dataset_ids = dict(canonical_active.get("datasets", {}) or {})
    active, research_dataset_view, resolved_dataset_view = _apply_research_dataset_view(
        root,
        canonical_active,
        config.dataset_view,
    )
    resolved_pit_manifest = ""
    if sample_filter == SAMPLE_FILTER_SIGNAL_ELIGIBLE:
        if config.pit_universe_manifest is not None:
            pit_manifest_path = Path(config.pit_universe_manifest).resolve()
            pit_manifest = json.loads(pit_manifest_path.read_text(encoding="utf-8"))
            if str(pit_manifest.get("domain", "")) != "pit_signal_universe":
                raise ValueError(f"not a pit_signal_universe manifest: {pit_manifest_path}")
            pit_dataset_id = str(pit_manifest.get("dataset_id", "") or "")
            expected_path = root / "datasets" / "pit_signal_universe" / pit_dataset_id / "dataset.json"
            if not pit_dataset_id or expected_path.resolve() != pit_manifest_path:
                raise ValueError(
                    "pit universe manifest must be the canonical QDP dataset manifest under the configured qdp_root"
                )
            active = dict(active)
            active_datasets = dict(active.get("datasets", {}) or {})
            active_datasets["pit_signal_universe"] = pit_dataset_id
            active["datasets"] = active_datasets
            resolved_pit_manifest = str(pit_manifest_path)
        elif str(dict(active.get("datasets", {}) or {}).get("pit_signal_universe", "") or ""):
            resolved_pit_manifest = str(
                (
                    root
                    / "datasets"
                    / "pit_signal_universe"
                    / str(dict(active.get("datasets", {}) or {})["pit_signal_universe"])
                    / "dataset.json"
                ).resolve()
            )
        else:
            raise ValueError(
                "sample_filter=signal_eligible_price_label requires --pit-universe-manifest or an active "
                "pit_signal_universe pointer"
            )
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
    pit_price_coverage = {
        "missing_rows": 0,
        "missing_symbols": 0,
        "first_missing_date": "",
        "last_missing_date": "",
        "examples": [],
    }
    if sample_filter == SAMPLE_FILTER_SIGNAL_ELIGIBLE:
        _write_json(progress_path, {"status": "auditing_pit_price_coverage", "updated_at": _now()})
        pit_price_coverage = _audit_pit_price_coverage(
            root,
            active,
            start=str(config.start_date),
            end=str(config.end_date),
        )
        if int(pit_price_coverage["missing_rows"]) > 0:
            _write_json(
                progress_path,
                {
                    "status": "blocked",
                    "blocker": "pit_eligible_rows_missing_active_daily_prices",
                    "pit_price_coverage": pit_price_coverage,
                    "updated_at": _now(),
                },
            )
            raise ValueError(
                "PIT signal universe is not aligned to the active market_daily_raw substrate; "
                f"missing_rows={pit_price_coverage['missing_rows']}, "
                f"missing_symbols={pit_price_coverage['missing_symbols']}, "
                f"dates={pit_price_coverage['first_missing_date']}..{pit_price_coverage['last_missing_date']}. "
                "Activate a matching PIT-complete price substrate before building."
            )
    _write_json(progress_path, {"status": "reading_daily", "updated_at": _now()})
    daily = _read_dataset_date_range(root, active, "market_daily_raw", DAILY_RAW_COLUMNS, date_values[0], date_values[-1])
    if daily.empty:
        raise ValueError("market_daily_raw returned no rows")
    daily["raw_open"] = pd.to_numeric(daily["open"], errors="coerce").astype("float64")
    if sample_filter == SAMPLE_FILTER_SIGNAL_ELIGIBLE:
        daily = daily[daily["symbol"].map(_is_pit_mainboard_symbol)].copy()
        if daily.empty:
            raise ValueError("PIT mainboard filter removed every market_daily_raw row")
    if price_adjustment == PRICE_ADJUSTMENT_BACK:
        _write_json(progress_path, {"status": "reading_adjust_factor", "updated_at": _now()})
        factor = _read_dataset_date_range(
            root,
            active,
            "adjust_factor",
            ["symbol", "trade_date", "adjust_factor", "back_adjust_factor"],
            date_values[0],
            date_values[-1],
        )
        daily = _apply_back_adjustment(daily, factor)
        del factor
        gc.collect()
    daily = _prepare_daily_frame(daily)
    daily_symbols = set(daily["symbol"].astype(str).str.upper().str.strip().unique().tolist())
    pit_symbols: set[str] = set()
    if sample_filter == SAMPLE_FILTER_SIGNAL_ELIGIBLE:
        _write_json(progress_path, {"status": "reading_pit_symbol_scope", "updated_at": _now()})
        pit_symbols = set(
            _read_dataset_symbols_date_range(
                root,
                active,
                "pit_signal_universe",
                date_values[0],
                date_values[-1],
            )
        )
        pit_symbols = {symbol for symbol in pit_symbols if _is_pit_mainboard_symbol(symbol)}
    symbol_values = sorted(daily_symbols | pit_symbols)
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
    raw_open_panel = _fill_float_memmap(label_dir / "entry_open_raw.float32.dat", (n_dates, n_symbols), fill_value=np.nan)
    has_bar_panel = _fill_bool_memmap(mask_dir / "has_bar.bool.dat", (n_dates, n_symbols))
    universe_has_bar_panel = _fill_bool_memmap(mask_dir / "pit_universe_has_bar.bool.dat", (n_dates, n_symbols))
    status_valid_panel = _fill_bool_memmap(mask_dir / "status_valid.bool.dat", (n_dates, n_symbols))
    is_st_panel = _fill_bool_memmap(mask_dir / "is_st.bool.dat", (n_dates, n_symbols))
    is_suspended_panel = _fill_bool_memmap(mask_dir / "is_suspended.bool.dat", (n_dates, n_symbols))
    is_delisted_panel = _fill_bool_memmap(mask_dir / "is_delisted.bool.dat", (n_dates, n_symbols))
    signal_eligible_panel = _fill_bool_memmap(mask_dir / "signal_eligible.bool.dat", (n_dates, n_symbols))
    tradable_panel = _fill_bool_memmap(mask_dir / "tradable.bool.dat", (n_dates, n_symbols))
    previous_close_valid_panel = _fill_bool_memmap(mask_dir / "previous_close_valid.bool.dat", (n_dates, n_symbols))
    corr_valid_panel = _fill_bool_memmap(mask_dir / "corr_valid.bool.dat", (n_dates, n_symbols))
    zero_range_panel = _fill_bool_memmap(mask_dir / "zero_range.bool.dat", (n_dates, n_symbols))

    _write_json(progress_path, {"status": "writing_daily_panels", "updated_at": _now()})
    daily_with_state = _add_raw_daily_signals(daily.copy())
    _write_panel_values(daily_raw, daily_with_state, DAILY_RAW_FEATURES, date_to_idx=date_to_idx, symbol_to_idx=symbol_to_idx)
    _write_panel_values(daily_state, daily_with_state, RAW_SIGNAL_COLUMNS, date_to_idx=date_to_idx, symbol_to_idx=symbol_to_idx)
    _write_scalar_panel_values(
        raw_open_panel,
        daily_with_state,
        "raw_open",
        date_to_idx=date_to_idx,
        symbol_to_idx=symbol_to_idx,
    )
    has_bar_panel[:] = np.isfinite(raw_open_panel)
    has_bar_panel.flush()
    if n_dates > 1:
        previous_close_valid_panel[1:] = has_bar_panel[1:] & np.maximum.accumulate(has_bar_panel[:-1], axis=0)
    high_values = daily_raw[:, :, DAILY_RAW_FEATURES.index("high")]
    low_values = daily_raw[:, :, DAILY_RAW_FEATURES.index("low")]
    zero_range_panel[:] = has_bar_panel & np.isfinite(high_values) & np.isfinite(low_values) & np.isclose(high_values, low_values)
    previous_close_valid_panel.flush()
    zero_range_panel.flush()
    del daily_with_state
    gc.collect()

    missing_eligible_bar_count = 0
    if sample_filter == SAMPLE_FILTER_SIGNAL_ELIGIBLE:
        for year in range(int(date_values[0][:4]), int(date_values[-1][:4]) + 1):
            _write_json(progress_path, {"status": "writing_pit_signal_universe", "year": year, "updated_at": _now()})
            pit_universe = _read_dataset_date_range(
                root,
                active,
                "pit_signal_universe",
                [
                    "symbol",
                    "trade_date",
                    "is_st",
                    "is_suspended",
                    "is_delisted",
                    "has_bar",
                    "eligible_for_signal",
                ],
                f"{year}-01-01",
                f"{year}-12-31",
            )
            _write_pit_universe_panels(
                frame=pit_universe,
                status_valid=status_valid_panel,
                is_st=is_st_panel,
                is_suspended=is_suspended_panel,
                is_delisted=is_delisted_panel,
                universe_has_bar=universe_has_bar_panel,
                signal_eligible=signal_eligible_panel,
                date_to_idx=date_to_idx,
                symbol_to_idx=symbol_to_idx,
            )
            del pit_universe
            gc.collect()
        requested_date_mask = np.asarray(
            [str(config.start_date) <= date <= str(config.end_date) for date in date_values],
            dtype=bool,
        )
        missing_eligible_bars = signal_eligible_panel[requested_date_mask] & (~has_bar_panel[requested_date_mask])
        missing_eligible_bar_count = int(np.count_nonzero(missing_eligible_bars))
        if missing_eligible_bar_count:
            raise ValueError(
                "PIT signal universe contains eligible symbol-days absent from the active market_daily_raw dataset; "
                f"count={missing_eligible_bar_count}. Activate a matching PIT-complete price substrate before building."
            )
        tradable_panel[:] = has_bar_panel & status_valid_panel & (~is_suspended_panel) & (~is_delisted_panel)
    elif suspension_fill == SUSPENSION_FILL_CARRY_CLOSE:
        for year in range(int(date_values[0][:4]), int(date_values[-1][:4]) + 1):
            _write_json(progress_path, {"status": "writing_security_status", "year": year, "updated_at": _now()})
            status = _read_dataset_date_range(
                root,
                active,
                "security_status",
                ["symbol", "trade_date", "is_st", "is_suspended", "is_delisted"],
                f"{year}-01-01",
                f"{year}-12-31",
            )
            _write_status_panels(
                frame=status,
                status_valid=status_valid_panel,
                is_st=is_st_panel,
                is_suspended=is_suspended_panel,
                is_delisted=is_delisted_panel,
                date_to_idx=date_to_idx,
                symbol_to_idx=symbol_to_idx,
            )
            del status
            gc.collect()
        universe_has_bar_panel[:] = has_bar_panel
        signal_eligible_panel[:] = has_bar_panel
        tradable_panel[:] = has_bar_panel & status_valid_panel & (~is_suspended_panel) & (~is_delisted_panel)
    else:
        universe_has_bar_panel[:] = has_bar_panel
        signal_eligible_panel[:] = has_bar_panel
        tradable_panel[:] = has_bar_panel
    universe_has_bar_panel.flush()
    signal_eligible_panel.flush()
    tradable_panel.flush()
    if suspension_fill == SUSPENSION_FILL_CARRY_CLOSE:
        _write_json(progress_path, {"status": "filling_suspended_prices", "updated_at": _now()})
        _fill_suspended_daily_raw(
            daily_raw,
            suspended_panel=is_suspended_panel,
            has_bar_panel=has_bar_panel,
        )

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

    corr_column = "intraday_price_volume_corr"
    if corr_column in INTRADAY_SIGNAL_COLUMNS:
        corr_valid_panel[:] = np.isfinite(intraday_summary[:, :, INTRADAY_SIGNAL_COLUMNS.index(corr_column)])
        corr_valid_panel.flush()

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
    price_label_valid_store = _fill_bool_memmap(mask_dir / "price_label_valid.bool.dat", (n_dates, n_symbols))
    va_aux_valid_store = _fill_bool_memmap(mask_dir / "va_aux_valid.bool.dat", (n_dates, n_symbols))
    separate_price_va_validity = sample_filter == SAMPLE_FILTER_SIGNAL_ELIGIBLE
    future_path, future_ohlcva_path, path_summary, input_valid, entry_buyable, label_valid = _compute_future_path_and_masks(
        raw_panel=daily_raw,
        up_limit_panel=up_limit_panel,
        lookback_days=int(config.lookback_days),
        forward_days=int(config.forward_days),
        price_anchor=str(config.price_anchor),
        raw_entry_open_panel=raw_open_panel,
        suspended_panel=is_suspended_panel,
        entry_rule=entry_rule,
        separate_price_va_validity=separate_price_va_validity,
        price_label_valid_out=price_label_valid_store,
        va_aux_valid_out=va_aux_valid_store,
        future_path_out=future_path_store,
        future_ohlcva_path_out=future_ohlcva_path_store,
        path_summary_out=path_summary_store,
        write_legacy_ohlc_path=bool(config.write_legacy_ohlc_label),
    )
    if future_path is not None:
        future_path.flush()
    future_ohlcva_path.flush()
    path_summary_store.flush()
    price_label_valid_store.flush()
    va_aux_valid_store.flush()
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
        signal_eligible=signal_eligible_panel if sample_filter == SAMPLE_FILTER_SIGNAL_ELIGIBLE else None,
        require_entry_filled=sample_filter == SAMPLE_FILTER_COMPLETE_CASE,
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

    source_dataset_ids = dict(active.get("datasets", {}) or {})
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
        "active_datasets": active_manifest_dataset_ids,
        "research_source_datasets": source_dataset_ids,
        "research_dataset_view": {
            "path": resolved_dataset_view or None,
            "view_id": str(dict(research_dataset_view or {}).get("view_id", "") or "") or None,
            "overrides": dict(dict(research_dataset_view or {}).get("overrides", {}) or {}),
        },
        "scope": active_scope,
        "lookback_days": int(config.lookback_days),
        "forward_days": int(config.forward_days),
        "start_date": str(config.start_date),
        "end_date": str(config.end_date),
        "train_years": list(config.train_years),
        "validation_years": list(config.validation_years),
        "test_years": list(config.test_years),
        "data_semantics": {
            "price_adjustment": price_adjustment,
            "price_adjustment_factor_column": "adjust_factor" if price_adjustment == PRICE_ADJUSTMENT_BACK else None,
            "entry_rule": entry_rule,
            "entry_tick_size": 0.01,
            "sample_filter": sample_filter,
            "suspension_fill": suspension_fill,
            "label_valid_alias": "price_label_valid" if separate_price_va_validity else "complete_ohlcva_label_valid",
            "unfilled_samples_retained": sample_filter == SAMPLE_FILTER_SIGNAL_ELIGIBLE,
            "pit_universe_domain": "pit_signal_universe" if sample_filter == SAMPLE_FILTER_SIGNAL_ELIGIBLE else None,
            "pit_universe_manifest": resolved_pit_manifest or None,
            "pit_universe_symbol_count": int(len(pit_symbols)),
            "pit_symbols_without_active_daily_rows": int(len(pit_symbols.difference(daily_symbols))),
            "eligible_symbol_days_without_active_daily_bar": int(missing_eligible_bar_count),
            "pit_price_coverage_gate": pit_price_coverage,
        },
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
        "execution_arrays": {
            "entry_open_raw": {
                "path": str((label_dir / "entry_open_raw.float32.dat").resolve()),
                "shape": [n_dates, n_symbols],
                "units": "CNY_raw_exchange_price",
            },
            "entry_up_limit_raw": {
                "path": str((label_dir / "entry_up_limit.float32.dat").resolve()),
                "shape": [n_dates, n_symbols],
                "units": "CNY_raw_exchange_price",
            },
        },
        "masks": {
            "input_valid": {"path": str((mask_dir / "input_valid.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "entry_buyable": {"path": str((mask_dir / "entry_buyable.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "entry_filled": {"path": str((mask_dir / "entry_buyable.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "label_valid": {"path": str((mask_dir / "label_valid.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "price_label_valid": {"path": str((mask_dir / "price_label_valid.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "va_aux_valid": {"path": str((mask_dir / "va_aux_valid.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "has_bar": {"path": str((mask_dir / "has_bar.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "price_observed": {"path": str((mask_dir / "has_bar.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "pit_universe_has_bar": {"path": str((mask_dir / "pit_universe_has_bar.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "status_valid": {"path": str((mask_dir / "status_valid.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "is_st": {"path": str((mask_dir / "is_st.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "is_suspended": {"path": str((mask_dir / "is_suspended.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "is_delisted": {"path": str((mask_dir / "is_delisted.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "signal_eligible": {"path": str((mask_dir / "signal_eligible.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "tradable": {"path": str((mask_dir / "tradable.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "previous_close_valid": {"path": str((mask_dir / "previous_close_valid.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "corr_valid": {"path": str((mask_dir / "corr_valid.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "zero_range": {"path": str((mask_dir / "zero_range.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
        },
        "mask_views": {
            "observed_price_path": {
                "source_mask": "price_observed",
                "shape": [n_dates, n_symbols, int(config.forward_days)],
                "indexing": "source[signal_date_idx+1:signal_date_idx+1+forward_days, symbol_idx]",
                "materialized": False,
            },
            "tradable_path": {
                "source_mask": "tradable",
                "shape": [n_dates, n_symbols, int(config.forward_days)],
                "indexing": "source[signal_date_idx+1:signal_date_idx+1+forward_days, symbol_idx]",
                "materialized": False,
            },
        },
        "sample_index_path": str(sample_index_path.resolve()),
        "normalization": normalization,
        "label_semantics": {
            "entry_anchor": "signal day close decision, next calendar trading day open entry",
            "entry_fill": "evaluated after ranking from raw next-day open, suspension state, and the configured exchange-tick limit rule",
            "price_anchor": str(config.price_anchor),
            "future_ohlc_path": f"OHLC returns are relative to {path_anchor_name}; price adjustment={price_adjustment}.",
            "future_ohlcva_path": f"OHLC returns are relative to {path_anchor_name}; volume and amount are log-relative to trailing 20 trading days ending on signal date.",
            "suspended_path": "explicit suspension days carry the last adjusted close with zero volume/amount only when suspension_fill=carry_adjusted_close; tradable remains false",
            "future_mask_view": "for signal index d, future observed/tradable masks are masks[d+1:d+1+forward_days] transposed by symbol",
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
    build.add_argument(
        "--price-adjustment",
        choices=(PRICE_ADJUSTMENT_NONE, PRICE_ADJUSTMENT_BACK),
        default=PRICE_ADJUSTMENT_NONE,
        help="Use raw prices or multiply OHLC by the active canonical back-adjust factor.",
    )
    build.add_argument(
        "--entry-rule",
        choices=(ENTRY_RULE_LEGACY, ENTRY_RULE_OPEN_BELOW_LIMIT),
        default=ENTRY_RULE_LEGACY,
        help="Next-open execution rule. open_below_limit_tick only blocks an open at the rounded exchange limit.",
    )
    build.add_argument(
        "--sample-filter",
        choices=(SAMPLE_FILTER_COMPLETE_CASE, SAMPLE_FILTER_SIGNAL_ELIGIBLE),
        default=SAMPLE_FILTER_COMPLETE_CASE,
        help="Legacy complete-case/filled samples or signal-day PIT universe with unfilled rows retained.",
    )
    build.add_argument(
        "--suspension-fill",
        choices=(SUSPENSION_FILL_NONE, SUSPENSION_FILL_CARRY_CLOSE),
        default=SUSPENSION_FILL_NONE,
        help="Optionally carry the last adjusted close through explicit suspension days while preserving masks.",
    )
    build.add_argument(
        "--pit-universe-manifest",
        type=Path,
        default=None,
        help="Immutable QDP pit_signal_universe dataset.json; avoids activating the research-scope pointer.",
    )
    build.add_argument(
        "--dataset-view",
        type=Path,
        default=None,
        help="Non-active qdp_v2 research dataset view. Must atomically override market_daily_raw, adjust_factor, limit_status, security_status, and pit_signal_universe.",
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
            price_adjustment=str(args.price_adjustment),
            entry_rule=str(args.entry_rule),
            sample_filter=str(args.sample_filter),
            suspension_fill=str(args.suspension_fill),
            pit_universe_manifest=Path(args.pit_universe_manifest) if args.pit_universe_manifest is not None else None,
            dataset_view=Path(args.dataset_view) if args.dataset_view is not None else None,
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
