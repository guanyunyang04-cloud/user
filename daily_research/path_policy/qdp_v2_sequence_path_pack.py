from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
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
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

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
DEFAULT_EXECUTION_TAIL_DAYS = 20
DEFAULT_MINIMUM_FREE_MEMORY_GB = 1.0
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
DEFAULT_LOT_SIZE = 100
DEFAULT_COMMISSION_BPS = 3.0
DEFAULT_MIN_COMMISSION_CNY = 5.0
DEFAULT_STAMP_TAX_BPS = 5.0
DEFAULT_STAMP_TAX_SCHEDULE = (("1900-01-01", 10.0), ("2023-08-28", 5.0))
DEFAULT_TRANSFER_FEE_BPS = 0.1
DEFAULT_SLIPPAGE_BPS = 7.0
DEFAULT_STRESS_SLIPPAGE_MULTIPLIER = 2.0
REQUIRED_PIT_MARKET_VIEW_OVERRIDES = {
    "market_daily_raw",
    "adjust_factor",
    "limit_status",
    "security_status",
    "pit_signal_universe",
}

PIT_MAINBOARD_PREFIXES = ("000", "001", "002", "003", "600", "601", "603", "605")


@dataclass(frozen=True)
class AShareExecutionCostConfig:
    """Deterministic research assumptions for one A-share round trip.

    Rates are explicit experiment inputs rather than claims about a broker account.  The
    minimum commission and board-lot constraint make small-account simulations materially
    different from a flat percentage haircut.
    """

    lot_size: int = DEFAULT_LOT_SIZE
    commission_bps: float = DEFAULT_COMMISSION_BPS
    minimum_commission_cny: float = DEFAULT_MIN_COMMISSION_CNY
    stamp_tax_bps: float = DEFAULT_STAMP_TAX_BPS
    stamp_tax_schedule: tuple[tuple[str, float], ...] = DEFAULT_STAMP_TAX_SCHEDULE
    transfer_fee_bps: float = DEFAULT_TRANSFER_FEE_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    stress_slippage_multiplier: float = DEFAULT_STRESS_SLIPPAGE_MULTIPLIER


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


def _available_physical_memory_gb() -> float | None:
    """Return currently available physical memory without making it a hard dependency."""

    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().available) / float(1024**3)
    except (ImportError, AttributeError, OSError):
        return None


def _trim_process_working_set() -> None:
    """Return released pages to Windows after large Arrow/pandas stages."""

    if os.name != "nt":
        return
    try:
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.EmptyWorkingSet(handle)
    except (AttributeError, OSError):
        return


def _enforce_memory_guard(
    *,
    stage: str,
    minimum_free_gb: float,
    progress_path: Path | None = None,
) -> float | None:
    minimum = float(minimum_free_gb)
    if not math.isfinite(minimum) or minimum < 0.0:
        raise ValueError("minimum_free_memory_gb must be finite and non-negative")
    available = _available_physical_memory_gb()
    if available is None or minimum <= 0.0:
        return available
    if available < minimum:
        payload = {
            "status": "blocked",
            "blocker": "minimum_free_memory_guard",
            "stage": str(stage),
            "available_memory_gb": float(available),
            "minimum_free_memory_gb": float(minimum),
            "updated_at": _now(),
        }
        if progress_path is not None:
            _write_json(progress_path, payload)
        raise MemoryError(
            f"memory guard blocked stage={stage}: available={available:.2f}GB < minimum={minimum:.2f}GB"
        )
    return available


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


def _bind_research_contract(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    raw_bytes = resolved.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8-sig"))
    declared_digest = str(payload.get("contract_sha256", "") or "").lower()
    semantic_payload = dict(payload)
    semantic_payload.pop("contract_sha256", None)
    semantic_bytes = json.dumps(
        semantic_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    semantic_digest = hashlib.sha256(semantic_bytes).hexdigest()
    if declared_digest and declared_digest != semantic_digest:
        raise ValueError(
            "research contract semantic digest mismatch: "
            f"declared={declared_digest}, actual={semantic_digest}"
        )
    return {
        "path": str(resolved),
        "contract_id": str(payload.get("contract_id", "") or ""),
        "contract_sha256": semantic_digest,
        "contract_file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


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


def _parse_rate_schedule(
    raw: str | Iterable[tuple[str, float]] | None,
    *,
    default: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    if raw is None:
        return default
    if not isinstance(raw, str):
        values = tuple((str(date), float(rate)) for date, rate in raw)
        return values or default
    values: list[tuple[str, float]] = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        date_text, separator, rate_text = item.partition(":")
        if not separator:
            raise ValueError(f"invalid effective-date rate: {item}")
        values.append((date_text.strip(), float(rate_text.strip())))
    return tuple(values) or default


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
    import duckdb  # type: ignore

    manifest = _dataset_manifest(root, active, domain)
    paths = _relative_shard_paths(root, manifest)
    path_sql = _sql_path_list(paths)
    start_text = str(start)
    end_text = str(end)
    symbols: set[str] = set()
    for year in range(int(start_text[:4]), int(end_text[:4]) + 1):
        part_start = max(start_text, f"{year}-01-01")
        part_end = min(end_text, f"{year}-12-31")
        start_sql = part_start.replace("'", "''")
        end_sql = part_end.replace("'", "''")
        con = duckdb.connect(":memory:")
        try:
            con.execute("set threads=2")
            con.execute("set memory_limit='512MB'")
            rows = con.execute(
                f"""
                select distinct upper(trim(cast(symbol as varchar))) as symbol
                from read_parquet({path_sql}, union_by_name=true)
                where cast(trade_date as varchar) between '{start_sql}' and '{end_sql}'
                  and symbol is not null
                  and trim(cast(symbol as varchar)) <> ''
                """
            ).fetchall()
            symbols.update(str(row[0]) for row in rows)
        finally:
            con.close()
        del rows
        gc.collect()
        _trim_process_working_set()
    return sorted(symbols)


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
        self._initialized_starts: set[int] = set()
        self.shard_metas: list[dict[str, Any]] = []

    def _close_open_shards(self) -> None:
        for key, (_start, _end, arr, _path) in list(self._shards.items()):
            arr.flush()
            mmap_handle = getattr(arr, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
            del self._shards[key]

    def _open_shard(self, date_idx: int) -> tuple[int, int, np.memmap, Path]:
        start = (int(date_idx) // self.shard_size) * self.shard_size
        end = min(start + self.shard_size, self.shape[0])
        if start in self._shards:
            return self._shards[start]
        # Label dates are written in ascending order.  Keeping every ~600-MB
        # mapping open lets Windows grow the process working set until the host
        # becomes unstable, so retain only the active shard mapping.
        self._close_open_shards()
        path = self.directory / f"{self.name}.{start:06d}_{end - 1:06d}.float32.dat"
        revisit = start in self._initialized_starts
        arr = np.memmap(
            path,
            dtype="float32",
            mode="r+" if revisit else "w+",
            shape=(end - start, *self.shape[1:]),
        )
        if not revisit:
            arr[:] = np.nan
            arr.flush()
            self._initialized_starts.add(start)
        item = (start, end, arr, path)
        self._shards[start] = item
        if not revisit:
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
        self._close_open_shards()

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


def _prepare_daily_with_state_memory_bounded(daily: pd.DataFrame) -> pd.DataFrame:
    """Build both daily feature groups with one symbol/date sort.

    The legacy composition sorted and copied the multi-million-row frame once in
    ``_prepare_daily_frame`` and then again in ``_add_raw_daily_signals``.  On the
    15-GB research workstation that leaves several full pandas frames resident at
    the same time.  The state builder already establishes the required order, so
    append the pack-only columns to that result in place.
    """

    out = _add_raw_daily_signals(daily)
    prev_close = pd.to_numeric(out["prev_close"], errors="coerce").astype("float64")
    for name, numerator in (
        ("open_ret_prev_close", out["open"]),
        ("high_ret_prev_close", out["high"]),
        ("low_ret_prev_close", out["low"]),
        ("close_ret_prev_close", out["close"]),
    ):
        out[name] = (
            pd.to_numeric(numerator, errors="coerce")
            .astype("float64")
            .div(prev_close.replace(0.0, np.nan))
            .sub(1.0)
            .replace([np.inf, -np.inf], np.nan)
        )
    out["volume_log"] = np.log1p(pd.to_numeric(out["volume"], errors="coerce")).replace(
        [np.inf, -np.inf], np.nan
    )
    out["amount_log"] = np.log1p(pd.to_numeric(out["amount"], errors="coerce")).replace(
        [np.inf, -np.inf], np.nan
    )
    out["intraday_range_raw"] = (
        pd.to_numeric(out["high"], errors="coerce")
        .astype("float64")
        .div(pd.to_numeric(out["low"], errors="coerce").astype("float64").replace(0.0, np.nan))
        .sub(1.0)
        .replace([np.inf, -np.inf], np.nan)
    )
    return out


def _apply_back_adjustment(daily: pd.DataFrame, factor: pd.DataFrame) -> pd.DataFrame:
    """Apply the canonical back-adjust factor to OHLC while preserving raw execution prices.

    The factor is a label/feature-side price transformation.  Volume and amount remain in
    their native units, and ``raw_open``/``raw_close`` are retained so entry and exit rules
    can still be evaluated in the exchange's 0.01-yuan tick space.
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
    out["raw_close"] = pd.to_numeric(out["close"], errors="coerce").astype("float64")
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


def _exit_fill_mask(
    exit_close: np.ndarray,
    exit_down_limit: np.ndarray,
    *,
    exit_observed: np.ndarray | None = None,
    exit_status_valid: np.ndarray | None = None,
    exit_suspended: np.ndarray | None = None,
    exit_delisted: np.ndarray | None = None,
    require_status_valid: bool = True,
    tick_size: float = 0.01,
) -> np.ndarray:
    """Return days on which a close-priced exit is deterministically executable.

    The rule is deliberately conservative: an observed close at the rounded down-limit is
    treated as queue-blocked.  A planned exit can therefore be retried on later dates using
    the 2-D ``exit_sellable`` panel without changing the prediction horizon.
    """

    close_values = np.asarray(exit_close, dtype=np.float64)
    limit_values = np.asarray(exit_down_limit, dtype=np.float64)
    if close_values.shape != limit_values.shape:
        raise ValueError("exit_close and exit_down_limit must share shape")
    exit_ok = np.isfinite(close_values)
    for name, values, expected in (
        ("exit_observed", exit_observed, True),
        ("exit_status_valid", exit_status_valid, True),
        ("exit_suspended", exit_suspended, False),
        ("exit_delisted", exit_delisted, False),
    ):
        if values is None:
            if name == "exit_status_valid" and bool(require_status_valid):
                raise ValueError("exit_status_valid is required when require_status_valid=True")
            continue
        mask = np.asarray(values, dtype=bool)
        if mask.shape != close_values.shape:
            raise ValueError(f"{name} must match the exit price shape")
        if name == "exit_status_valid" and not bool(require_status_valid):
            continue
        exit_ok &= mask if expected else ~mask
    down_limit_blocked = (
        np.isfinite(limit_values)
        & np.isfinite(close_values)
        & (_price_to_tick_units(close_values, tick_size=tick_size) <= _price_to_tick_units(limit_values, tick_size=tick_size))
    )
    return exit_ok & (~down_limit_blocked)


def _resolve_deferred_exit_days(
    planned_exit_days: np.ndarray,
    exit_sellable_path: np.ndarray,
    *,
    forward_days: int,
    execution_tail_days: int,
) -> np.ndarray:
    """Resolve 1-based planned exits to the first sellable day within the tail window."""

    planned = np.asarray(planned_exit_days, dtype=np.float64).reshape(-1)
    sellable = np.asarray(exit_sellable_path, dtype=bool)
    if sellable.ndim != 2 or sellable.shape[0] != planned.shape[0]:
        raise ValueError("exit_sellable_path must be [sample, day] and match planned_exit_days")
    horizon = int(forward_days) + int(execution_tail_days)
    if int(forward_days) <= 0 or int(execution_tail_days) < 0:
        raise ValueError("forward_days must be positive and execution_tail_days must be non-negative")
    if sellable.shape[1] < horizon:
        raise ValueError("exit_sellable_path is shorter than forward_days + execution_tail_days")
    if horizon < 2:
        raise ValueError("exit window must include T+1 after the entry day")
    resolved = np.full(planned.shape, np.nan, dtype=np.float32)
    for row_idx, raw_day in enumerate(planned):
        if not math.isfinite(float(raw_day)):
            continue
        # Path day 1 is the entry day; A-share T+1 makes path day 2 the earliest exit.
        planned_day = max(2, min(max(int(round(float(raw_day))), 1), int(forward_days)))
        future = np.flatnonzero(sellable[row_idx, planned_day - 1 : horizon])
        if future.size:
            resolved[row_idx] = np.float32(planned_day + int(future[0]))
    return resolved


def _execution_cost_contract(config: AShareExecutionCostConfig) -> dict[str, Any]:
    numeric = {
        "commission_bps": float(config.commission_bps),
        "minimum_commission_cny": float(config.minimum_commission_cny),
        "stamp_tax_bps": float(config.stamp_tax_bps),
        "transfer_fee_bps": float(config.transfer_fee_bps),
        "slippage_bps": float(config.slippage_bps),
        "stress_slippage_multiplier": float(config.stress_slippage_multiplier),
    }
    if int(config.lot_size) <= 0:
        raise ValueError("lot_size must be positive")
    if any((not math.isfinite(value)) or value < 0.0 for value in numeric.values()):
        raise ValueError("execution cost assumptions must be finite and non-negative")
    if numeric["stress_slippage_multiplier"] < 1.0:
        raise ValueError("stress_slippage_multiplier must be at least 1")
    stamp_schedule: list[dict[str, Any]] = []
    previous_date = ""
    for effective_date, rate_bps in config.stamp_tax_schedule:
        try:
            date_text = pd.Timestamp(str(effective_date)).strftime("%Y-%m-%d")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid stamp-tax effective date: {effective_date}") from exc
        rate = float(rate_bps)
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError("stamp-tax schedule rates must be finite and non-negative")
        if previous_date and date_text <= previous_date:
            raise ValueError("stamp-tax schedule must be strictly increasing by effective date")
        stamp_schedule.append({"effective_date": date_text, "stamp_tax_bps": rate})
        previous_date = date_text
    if not stamp_schedule:
        stamp_schedule.append({"effective_date": "1900-01-01", "stamp_tax_bps": numeric["stamp_tax_bps"]})
    return {
        "contract": "a_share_round_trip_cashflow_v1",
        "lot_size": int(config.lot_size),
        **numeric,
        "stamp_tax_schedule": stamp_schedule,
        "commission_sides": ["buy", "sell"],
        "minimum_commission_applied_per_order": True,
        "stamp_tax_sides": ["sell"],
        "transfer_fee_sides": ["buy", "sell"],
        "slippage_application": "buy_price*(1+bps/10000), sell_price*(1-bps/10000)",
        "unaffordable_or_unfilled_order": "retain_cash",
    }


def simulate_a_share_round_trip(
    *,
    allocated_cash: float,
    entry_price: float,
    exit_price: float,
    cost: AShareExecutionCostConfig = AShareExecutionCostConfig(),
    slippage_multiplier: float = 1.0,
    exit_trade_date: str | None = None,
) -> dict[str, float | int | bool]:
    """Apply board-lot, minimum-fee, tax, transfer-fee, and slippage assumptions."""

    contract = _execution_cost_contract(cost)
    cash = float(allocated_cash)
    entry = float(entry_price)
    exit_value = float(exit_price)
    multiplier = float(slippage_multiplier)
    if not all(math.isfinite(value) for value in (cash, entry, exit_value, multiplier)):
        raise ValueError("cash, prices, and slippage_multiplier must be finite")
    if cash < 0.0 or entry <= 0.0 or exit_value < 0.0 or multiplier < 0.0:
        raise ValueError("cash/prices/slippage_multiplier are outside the supported range")
    slip = float(contract["slippage_bps"]) * multiplier / 10_000.0
    buy_price = entry * (1.0 + slip)
    sell_price = exit_value * max(0.0, 1.0 - slip)
    lot_size = int(contract["lot_size"])
    shares = int(cash // (buy_price * lot_size)) * lot_size
    commission_rate = float(contract["commission_bps"]) / 10_000.0
    transfer_rate = float(contract["transfer_fee_bps"]) / 10_000.0
    minimum_commission = float(contract["minimum_commission_cny"])
    while shares > 0:
        buy_gross = float(shares) * buy_price
        buy_commission = max(minimum_commission, buy_gross * commission_rate)
        buy_transfer_fee = buy_gross * transfer_rate
        if buy_gross + buy_commission + buy_transfer_fee <= cash + 1.0e-9:
            break
        shares -= lot_size
    if shares <= 0:
        return {
            "filled": False,
            "shares": 0,
            "ending_cash": cash,
            "net_return": 0.0,
            "total_cost": 0.0,
        }
    buy_gross = float(shares) * buy_price
    buy_commission = max(minimum_commission, buy_gross * commission_rate)
    buy_transfer_fee = buy_gross * transfer_rate
    cash_after_buy = cash - buy_gross - buy_commission - buy_transfer_fee
    sell_gross = float(shares) * sell_price
    sell_commission = max(minimum_commission, sell_gross * commission_rate)
    sell_transfer_fee = sell_gross * transfer_rate
    stamp_tax_bps = float(contract["stamp_tax_bps"])
    if exit_trade_date is not None:
        try:
            trade_date = pd.Timestamp(str(exit_trade_date)).strftime("%Y-%m-%d")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid exit_trade_date: {exit_trade_date}") from exc
        matched = [
            float(item["stamp_tax_bps"])
            for item in list(contract["stamp_tax_schedule"])
            if str(item["effective_date"]) <= trade_date
        ]
        if not matched:
            raise ValueError("exit_trade_date predates the configured stamp-tax schedule")
        stamp_tax_bps = matched[-1]
    stamp_tax = sell_gross * stamp_tax_bps / 10_000.0
    ending_cash = cash_after_buy + sell_gross - sell_commission - sell_transfer_fee - stamp_tax
    explicit_cost = buy_commission + buy_transfer_fee + sell_commission + sell_transfer_fee + stamp_tax
    slippage_cost = float(shares) * ((buy_price - entry) + (exit_value - sell_price))
    return {
        "filled": True,
        "shares": int(shares),
        "ending_cash": float(ending_cash),
        "net_return": float(ending_cash / cash - 1.0) if cash > 0.0 else 0.0,
        "total_cost": float(explicit_cost + slippage_cost),
    }


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
            if (date_idx + 1) % max(int(flush_every_dates) * 4, 1) == 0:
                _trim_process_working_set()
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
    price_label_valid: np.ndarray | None = None,
    va_aux_valid: np.ndarray | None = None,
    signal_eligible: np.ndarray | None = None,
    require_entry_filled: bool = True,
    require_label_valid: bool = True,
    id_column: str = "sample_id",
) -> pd.DataFrame:
    if str(id_column) not in {"sample_id", "candidate_id"}:
        raise ValueError("id_column must be sample_id or candidate_id")
    expected_shape = (len(date_values), len(symbol_values))
    for name, values in (
        ("input_valid", input_valid),
        ("entry_buyable", entry_buyable),
        ("label_valid", label_valid),
        ("price_label_valid", price_label_valid),
        ("va_aux_valid", va_aux_valid),
        ("signal_eligible", signal_eligible),
    ):
        if values is not None and np.asarray(values).shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    price_valid_values = label_valid if price_label_valid is None else price_label_valid
    va_valid_values = label_valid if va_aux_valid is None else va_aux_valid
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
        mask = np.asarray(input_valid[date_idx], dtype=bool).copy()
        if signal_eligible is not None:
            mask &= np.asarray(signal_eligible[date_idx], dtype=bool)
        if bool(require_label_valid):
            mask &= np.asarray(label_valid[date_idx], dtype=bool)
        if bool(require_entry_filled):
            mask &= entry_buyable[date_idx]
        symbol_idx = np.flatnonzero(mask).astype(np.int32)
        if len(symbol_idx) == 0:
            continue
        rows.append(
            pd.DataFrame(
                {
                    str(id_column): np.arange(len(symbol_idx), dtype=np.int64),
                    "split": split,
                    "year": np.int16(year),
                    "trade_date": str(trade_date),
                    "date_idx": np.full(len(symbol_idx), date_idx, dtype=np.int32),
                    "symbol_idx": symbol_idx,
                    "symbol": symbol_arr[symbol_idx],
                    "entry_trade_date": date_arr[date_idx + 1] if date_idx + 1 < len(date_arr) else "",
                    "entry_filled": entry_buyable[date_idx, symbol_idx],
                    "label_valid": label_valid[date_idx, symbol_idx],
                    "price_label_valid": price_valid_values[date_idx, symbol_idx],
                    "va_aux_valid": va_valid_values[date_idx, symbol_idx],
                }
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                str(id_column),
                "split",
                "year",
                "trade_date",
                "date_idx",
                "symbol_idx",
                "symbol",
                "entry_trade_date",
                "entry_filled",
                "label_valid",
                "price_label_valid",
                "va_aux_valid",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    out[str(id_column)] = np.arange(len(out), dtype=np.int64)
    return out


def _write_sample_index_streaming(
    *,
    output_path: Path,
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
    price_label_valid: np.ndarray | None = None,
    va_aux_valid: np.ndarray | None = None,
    signal_eligible: np.ndarray | None = None,
    require_entry_filled: bool = True,
    require_label_valid: bool = True,
    id_column: str = "sample_id",
    target_row_group_size: int = 250_000,
) -> dict[str, Any]:
    """Write a candidate or supervision index without materializing all years in RAM."""

    if str(id_column) not in {"sample_id", "candidate_id"}:
        raise ValueError("id_column must be sample_id or candidate_id")
    expected_shape = (len(date_values), len(symbol_values))
    for name, values in (
        ("input_valid", input_valid),
        ("entry_buyable", entry_buyable),
        ("label_valid", label_valid),
        ("price_label_valid", price_label_valid),
        ("va_aux_valid", va_aux_valid),
        ("signal_eligible", signal_eligible),
    ):
        if values is not None and np.asarray(values).shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    price_valid_values = label_valid if price_label_valid is None else price_label_valid
    va_valid_values = label_valid if va_aux_valid is None else va_aux_valid
    date_arr = np.asarray(date_values, dtype=object)
    symbol_arr = np.asarray(symbol_values, dtype=object)
    pending: list[pd.DataFrame] = []
    pending_rows = 0
    total_rows = 0
    split_counts: dict[str, int] = {}
    writer: pq.ParquetWriter | None = None

    def flush_pending() -> None:
        nonlocal pending, pending_rows, writer
        if not pending:
            return
        frame = pd.concat(pending, ignore_index=True, copy=False)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(
                temporary,
                table.schema,
                compression="zstd",
                use_dictionary=["split", "symbol", "trade_date", "entry_trade_date"],
            )
        writer.write_table(table, row_group_size=max(int(target_row_group_size), 1))
        pending = []
        pending_rows = 0
        del frame, table
        gc.collect()
        _trim_process_working_set()

    try:
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
            mask = np.asarray(input_valid[date_idx], dtype=bool).copy()
            if signal_eligible is not None:
                mask &= np.asarray(signal_eligible[date_idx], dtype=bool)
            if bool(require_label_valid):
                mask &= np.asarray(label_valid[date_idx], dtype=bool)
            if bool(require_entry_filled):
                mask &= np.asarray(entry_buyable[date_idx], dtype=bool)
            symbol_idx = np.flatnonzero(mask).astype(np.int32)
            if not len(symbol_idx):
                continue
            row_count = int(len(symbol_idx))
            frame = pd.DataFrame(
                {
                    str(id_column): np.arange(total_rows, total_rows + row_count, dtype=np.int64),
                    "split": split,
                    "year": np.full(row_count, year, dtype=np.int16),
                    "trade_date": str(trade_date),
                    "date_idx": np.full(row_count, date_idx, dtype=np.int32),
                    "symbol_idx": symbol_idx,
                    "symbol": symbol_arr[symbol_idx],
                    "entry_trade_date": date_arr[date_idx + 1] if date_idx + 1 < len(date_arr) else "",
                    "entry_filled": np.asarray(entry_buyable[date_idx, symbol_idx], dtype=bool),
                    "label_valid": np.asarray(label_valid[date_idx, symbol_idx], dtype=bool),
                    "price_label_valid": np.asarray(price_valid_values[date_idx, symbol_idx], dtype=bool),
                    "va_aux_valid": np.asarray(va_valid_values[date_idx, symbol_idx], dtype=bool),
                }
            )
            pending.append(frame)
            pending_rows += row_count
            total_rows += row_count
            split_counts[split] = int(split_counts.get(split, 0) + row_count)
            if pending_rows >= max(int(target_row_group_size), 1):
                flush_pending()
        flush_pending()
        if writer is None:
            empty = _build_sample_index(
                date_values=[],
                symbol_values=[],
                start_date=start_date,
                end_date=end_date,
                train_years=train_years,
                validation_years=validation_years,
                test_years=test_years,
                input_valid=np.zeros((0, 0), dtype=bool),
                entry_buyable=np.zeros((0, 0), dtype=bool),
                label_valid=np.zeros((0, 0), dtype=bool),
                require_entry_filled=require_entry_filled,
                require_label_valid=require_label_valid,
                id_column=id_column,
            )
            empty.to_parquet(temporary, index=False)
    finally:
        if writer is not None:
            writer.close()
    os.replace(temporary, output_path)
    return {
        "path": str(output_path.resolve()),
        "row_count": int(total_rows),
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "write_policy": "date_ordered_streaming_zstd",
        "target_row_group_size": int(target_row_group_size),
    }


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
    execution_tail_days: int = DEFAULT_EXECUTION_TAIL_DAYS
    require_full_dependency_padding: bool = True
    minimum_free_memory_gb: float = DEFAULT_MINIMUM_FREE_MEMORY_GB
    unresolved_exit_recovery_fraction: float = 0.0
    execution_cost: AShareExecutionCostConfig = AShareExecutionCostConfig()
    research_contract: Path | None = None
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
    execution_tail_days = int(config.execution_tail_days)
    if execution_tail_days < 0:
        raise ValueError("execution_tail_days must be non-negative")
    unresolved_exit_recovery_fraction = float(config.unresolved_exit_recovery_fraction)
    if (
        not math.isfinite(unresolved_exit_recovery_fraction)
        or unresolved_exit_recovery_fraction < 0.0
        or unresolved_exit_recovery_fraction > 1.0
    ):
        raise ValueError("unresolved_exit_recovery_fraction must be finite and between 0 and 1")
    execution_cost_contract = _execution_cost_contract(config.execution_cost)
    research_contract_binding = _bind_research_contract(config.research_contract)
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
    memory_observation_path = output_dir / "memory_observations.json"
    memory_observations: list[dict[str, Any]] = []
    _write_json(progress_path, {"status": "started", "updated_at": _now()})

    def memory_guard(stage: str) -> float | None:
        available = _enforce_memory_guard(
            stage=stage,
            minimum_free_gb=float(config.minimum_free_memory_gb),
            progress_path=progress_path,
        )
        memory_observations.append(
            {
                "stage": str(stage),
                "available_memory_gb": float(available) if available is not None else None,
                "observed_at": _now(),
            }
        )
        _write_json(
            memory_observation_path,
            {
                "minimum_free_memory_gb": float(config.minimum_free_memory_gb),
                "observations": memory_observations,
            },
        )
        return available

    memory_guard("build_start")
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
    required_dependency_days = int(config.forward_days) + execution_tail_days
    required_panel_end_pos = sample_end_pos + required_dependency_days
    dependency_padding_complete = required_panel_end_pos < len(all_open_dates)
    if bool(config.require_full_dependency_padding) and not dependency_padding_complete:
        available = max(0, len(all_open_dates) - 1 - sample_end_pos)
        raise ValueError(
            "requested sample window lacks full forward+execution dependency padding: "
            f"required={required_dependency_days}, available={available}"
        )
    panel_end_pos = min(
        len(all_open_dates) - 1,
        required_panel_end_pos,
    )
    date_values = all_open_dates[panel_start_pos : panel_end_pos + 1]
    pit_price_coverage = {
        "missing_rows": 0,
        "missing_symbols": 0,
        "first_missing_date": "",
        "last_missing_date": "",
        "examples": [],
    }
    if sample_filter == SAMPLE_FILTER_SIGNAL_ELIGIBLE:
        memory_guard("audit_pit_price_coverage")
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
    memory_guard("read_daily_symbol_scope")
    _write_json(progress_path, {"status": "reading_daily_symbol_scope", "updated_at": _now()})
    daily_symbols = set(
        _read_dataset_symbols_date_range(
            root,
            active,
            "market_daily_raw",
            date_values[0],
            date_values[-1],
        )
    )
    if sample_filter == SAMPLE_FILTER_SIGNAL_ELIGIBLE:
        daily_symbols = {symbol for symbol in daily_symbols if _is_pit_mainboard_symbol(symbol)}
    if not daily_symbols:
        raise ValueError("market_daily_raw returned no symbols in the requested panel range")
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
    down_limit_panel = _fill_float_memmap(label_dir / "exit_down_limit.float32.dat", (n_dates, n_symbols), fill_value=np.nan)
    raw_open_panel = _fill_float_memmap(label_dir / "entry_open_raw.float32.dat", (n_dates, n_symbols), fill_value=np.nan)
    raw_close_panel = _fill_float_memmap(label_dir / "exit_close_raw.float32.dat", (n_dates, n_symbols), fill_value=np.nan)
    has_bar_panel = _fill_bool_memmap(mask_dir / "has_bar.bool.dat", (n_dates, n_symbols))
    universe_has_bar_panel = _fill_bool_memmap(mask_dir / "pit_universe_has_bar.bool.dat", (n_dates, n_symbols))
    status_valid_panel = _fill_bool_memmap(mask_dir / "status_valid.bool.dat", (n_dates, n_symbols))
    is_st_panel = _fill_bool_memmap(mask_dir / "is_st.bool.dat", (n_dates, n_symbols))
    is_suspended_panel = _fill_bool_memmap(mask_dir / "is_suspended.bool.dat", (n_dates, n_symbols))
    is_delisted_panel = _fill_bool_memmap(mask_dir / "is_delisted.bool.dat", (n_dates, n_symbols))
    signal_eligible_panel = _fill_bool_memmap(mask_dir / "signal_eligible.bool.dat", (n_dates, n_symbols))
    tradable_panel = _fill_bool_memmap(mask_dir / "tradable.bool.dat", (n_dates, n_symbols))
    exit_has_valid_bar_volume_panel = _fill_bool_memmap(
        mask_dir / "exit_has_valid_bar_volume.bool.dat",
        (n_dates, n_symbols),
    )
    exit_sellable_panel = _fill_bool_memmap(mask_dir / "exit_sellable.bool.dat", (n_dates, n_symbols))
    previous_close_valid_panel = _fill_bool_memmap(mask_dir / "previous_close_valid.bool.dat", (n_dates, n_symbols))
    corr_valid_panel = _fill_bool_memmap(mask_dir / "corr_valid.bool.dat", (n_dates, n_symbols))
    zero_range_panel = _fill_bool_memmap(mask_dir / "zero_range.bool.dat", (n_dates, n_symbols))

    # Initializing multi-gigabyte memmaps touches every page.  The data is
    # already flushed, so evict those clean pages before loading yearly pandas
    # chunks; otherwise the initialized mappings and the first chunk overlap in
    # the physical working set on 16-GB hosts.
    gc.collect()
    _trim_process_working_set()
    memory_guard("panel_allocation_completed")

    # Stream one calendar year at a time with enough pre-year history to make
    # every rolling feature exact for candidate-eligible rows.  This avoids the
    # previous 9.5-million-row daily/factor pandas merge that exhausted RAM.
    daily_history_days = max(int(config.lookback_days), 60)
    for year in range(int(date_values[0][:4]), int(date_values[-1][:4]) + 1):
        target_dates = [date for date in date_values if int(date[:4]) == int(year)]
        if not target_dates:
            continue
        target_start = str(target_dates[0])
        target_end = str(target_dates[-1])
        target_start_idx = int(date_to_idx[target_start])
        read_start = str(date_values[max(0, target_start_idx - daily_history_days)])
        memory_guard(f"daily_panel_{year}_read")
        _write_json(
            progress_path,
            {
                "status": "writing_daily_panels",
                "year": int(year),
                "read_start": read_start,
                "target_start": target_start,
                "target_end": target_end,
                "updated_at": _now(),
            },
        )
        daily_chunk = _read_dataset_date_range(
            root,
            active,
            "market_daily_raw",
            DAILY_RAW_COLUMNS,
            read_start,
            target_end,
        )
        if daily_chunk.empty:
            _write_json(
                progress_path,
                {
                    "status": "daily_panel_padding_year_without_rows",
                    "year": int(year),
                    "target_start": target_start,
                    "target_end": target_end,
                    "updated_at": _now(),
                },
            )
            continue
        daily_chunk["raw_open"] = pd.to_numeric(daily_chunk["open"], errors="coerce").astype("float64")
        daily_chunk["raw_close"] = pd.to_numeric(daily_chunk["close"], errors="coerce").astype("float64")
        if sample_filter == SAMPLE_FILTER_SIGNAL_ELIGIBLE:
            daily_chunk = daily_chunk[daily_chunk["symbol"].map(_is_pit_mainboard_symbol)].copy()
            if daily_chunk.empty:
                raise ValueError(f"PIT mainboard filter removed every daily row for streamed year {year}")
        if price_adjustment == PRICE_ADJUSTMENT_BACK:
            memory_guard(f"daily_panel_{year}_factor")
            factor_chunk = _read_dataset_date_range(
                root,
                active,
                "adjust_factor",
                ["symbol", "trade_date", "adjust_factor", "back_adjust_factor"],
                read_start,
                target_end,
            )
            daily_chunk = _apply_back_adjustment(daily_chunk, factor_chunk)
            del factor_chunk
            gc.collect()
        daily_with_state = _prepare_daily_with_state_memory_bounded(daily_chunk)
        del daily_chunk
        target_mask = daily_with_state["trade_date"].astype(str).between(target_start, target_end)
        target_frame = daily_with_state.loc[target_mask]
        _write_panel_values(
            daily_raw,
            target_frame,
            DAILY_RAW_FEATURES,
            date_to_idx=date_to_idx,
            symbol_to_idx=symbol_to_idx,
        )
        _write_panel_values(
            daily_state,
            target_frame,
            RAW_SIGNAL_COLUMNS,
            date_to_idx=date_to_idx,
            symbol_to_idx=symbol_to_idx,
        )
        _write_scalar_panel_values(
            raw_open_panel,
            target_frame,
            "raw_open",
            date_to_idx=date_to_idx,
            symbol_to_idx=symbol_to_idx,
        )
        _write_scalar_panel_values(
            raw_close_panel,
            target_frame,
            "raw_close",
            date_to_idx=date_to_idx,
            symbol_to_idx=symbol_to_idx,
        )
        del target_frame, target_mask, daily_with_state
        gc.collect()
        _trim_process_working_set()
        memory_guard(f"daily_panel_{year}_completed")

    has_bar_panel[:] = np.isfinite(raw_open_panel)
    has_bar_panel.flush()
    volume_values = daily_raw[:, :, DAILY_RAW_FEATURES.index("volume")]
    exit_has_valid_bar_volume_panel[:] = (
        has_bar_panel
        & np.isfinite(raw_close_panel)
        & np.isfinite(volume_values)
        & (volume_values > 0.0)
    )
    exit_has_valid_bar_volume_panel.flush()
    if n_dates > 1:
        previous_close_valid_panel[1:] = has_bar_panel[1:] & np.maximum.accumulate(has_bar_panel[:-1], axis=0)
    high_values = daily_raw[:, :, DAILY_RAW_FEATURES.index("high")]
    low_values = daily_raw[:, :, DAILY_RAW_FEATURES.index("low")]
    zero_range_panel[:] = has_bar_panel & np.isfinite(high_values) & np.isfinite(low_values) & np.isclose(high_values, low_values)
    previous_close_valid_panel.flush()
    zero_range_panel.flush()
    gc.collect()
    _trim_process_working_set()

    missing_eligible_bar_count = 0
    if sample_filter == SAMPLE_FILTER_SIGNAL_ELIGIBLE:
        for year in range(int(date_values[0][:4]), int(date_values[-1][:4]) + 1):
            memory_guard(f"pit_signal_universe_{year}")
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
            _trim_process_working_set()
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
            memory_guard(f"security_status_{year}")
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
            _trim_process_working_set()
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
            memory_guard(f"{domain}_{year}")
            _write_json(progress_path, {"status": f"writing_{domain}", "year": year, "updated_at": _now()})
            frame = _read_dataset_date_range(root, active, domain, columns, f"{year}-01-01", f"{year}-12-31")
            _write_panel_values(panel, frame, feature_columns, date_to_idx=date_to_idx, symbol_to_idx=symbol_to_idx)
            del frame
            gc.collect()
            _trim_process_working_set()

    corr_column = "intraday_price_volume_corr"
    if corr_column in INTRADAY_SIGNAL_COLUMNS:
        corr_valid_panel[:] = np.isfinite(intraday_summary[:, :, INTRADAY_SIGNAL_COLUMNS.index(corr_column)])
        corr_valid_panel.flush()

    for year in range(int(date_values[0][:4]), int(date_values[-1][:4]) + 1):
        memory_guard(f"limit_status_{year}")
        _write_json(progress_path, {"status": "writing_entry_limits", "year": year, "updated_at": _now()})
        limit = _read_dataset_date_range(
            root,
            active,
            "limit_status",
            ["symbol", "trade_date", "up_limit", "down_limit"],
            f"{year}-01-01",
            f"{year}-12-31",
        )
        if not limit.empty:
            limit = _index_frame(limit, date_to_idx, symbol_to_idx)
            date_idx = limit["_date_idx"].to_numpy(dtype=np.int64)
            symbol_idx = limit["_symbol_idx"].to_numpy(dtype=np.int64)
            up_values = pd.to_numeric(limit["up_limit"], errors="coerce").to_numpy(dtype=np.float32, copy=True)
            down_values = pd.to_numeric(limit["down_limit"], errors="coerce").to_numpy(dtype=np.float32, copy=True)
            up_limit_panel[date_idx, symbol_idx] = up_values
            down_limit_panel[date_idx, symbol_idx] = down_values
            up_limit_panel.flush()
            down_limit_panel.flush()
        del limit
        gc.collect()
        _trim_process_working_set()

    exit_sellable_panel[:] = _exit_fill_mask(
        raw_close_panel,
        down_limit_panel,
        exit_observed=exit_has_valid_bar_volume_panel,
        exit_status_valid=status_valid_panel,
        exit_suspended=is_suspended_panel,
        exit_delisted=is_delisted_panel,
        require_status_valid=(
            sample_filter == SAMPLE_FILTER_SIGNAL_ELIGIBLE or suspension_fill == SUSPENSION_FILL_CARRY_CLOSE
        ),
    )
    exit_sellable_panel.flush()

    memory_guard("computing_labels")
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
    candidate_eligible_store = _fill_bool_memmap(
        mask_dir / "candidate_eligible.bool.dat",
        tuple(int(item) for item in input_valid.shape),
    )
    input_valid_store[:] = input_valid
    entry_buyable_store[:] = entry_buyable
    label_valid_store[:] = label_valid
    candidate_eligible_store[:] = input_valid & np.asarray(signal_eligible_panel, dtype=bool)
    input_valid_store.flush()
    entry_buyable_store.flush()
    label_valid_store.flush()
    candidate_eligible_store.flush()

    memory_guard("labels_completed")
    _write_json(progress_path, {"status": "building_sample_index", "updated_at": _now()})
    sample_index_path = output_dir / "sample_index.parquet"
    sample_index_stats = _write_sample_index_streaming(
        output_path=sample_index_path,
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
        price_label_valid=price_label_valid_store,
        va_aux_valid=va_aux_valid_store,
        signal_eligible=signal_eligible_panel if sample_filter == SAMPLE_FILTER_SIGNAL_ELIGIBLE else None,
        require_entry_filled=sample_filter == SAMPLE_FILTER_COMPLETE_CASE,
    )
    memory_guard("sample_index_completed")
    _write_json(progress_path, {"status": "building_candidate_index", "updated_at": _now()})
    candidate_index_path = output_dir / "candidate_index.parquet"
    candidate_index_stats = _write_sample_index_streaming(
        output_path=candidate_index_path,
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
        price_label_valid=price_label_valid_store,
        va_aux_valid=va_aux_valid_store,
        signal_eligible=signal_eligible_panel,
        require_entry_filled=False,
        require_label_valid=False,
        id_column="candidate_id",
    )

    memory_guard("candidate_index_completed")
    _write_json(progress_path, {"status": "fitting_normalization", "updated_at": _now()})
    train_date_mask = np.array([int(date[:4]) in set(config.train_years) for date in date_values], dtype=bool)
    normalization = {
        "fit_scope": "train_year_dates_only",
        "daily_raw": _fit_normalization(daily_raw, train_date_mask),
        "daily_state": _fit_normalization(daily_state, train_date_mask),
        "intraday_summary": _fit_normalization(intraday_summary, train_date_mask),
        "limit_structure": _fit_normalization(limit_structure, train_date_mask),
    }
    memory_guard("manifest_assembly")

    source_dataset_ids = dict(active.get("datasets", {}) or {})
    split_counts = dict(sample_index_stats["split_counts"])
    candidate_split_counts = dict(candidate_index_stats["split_counts"])
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
        "research_contract": research_contract_binding,
        "development_contract": research_contract_binding,
        "scope": active_scope,
        "lookback_days": int(config.lookback_days),
        "forward_days": int(config.forward_days),
        "execution_tail_days": execution_tail_days,
        "max_label_dependency_days": required_dependency_days,
        "dependency_padding_complete": bool(dependency_padding_complete),
        "available_dependency_padding_days": int(panel_end_pos - sample_end_pos),
        "resource_guard": {
            "minimum_free_memory_gb": float(config.minimum_free_memory_gb),
            "available_memory_probe": "psutil.virtual_memory.available",
            "label_store_max_open_shards": 1,
            "daily_feature_sort_passes": 1,
            "memory_observations_path": str(memory_observation_path.resolve()),
            "memory_observations_sha256": hashlib.sha256(memory_observation_path.read_bytes()).hexdigest(),
            "minimum_observed_available_memory_gb": min(
                (
                    float(item["available_memory_gb"])
                    for item in memory_observations
                    if item["available_memory_gb"] is not None
                ),
                default=None,
            ),
        },
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
            "candidate_filter": "signal_day_input_valid_and_pit_signal_eligible_only",
            "supervision_filter": "candidate_filter_and_label_valid",
            "candidate_selection_uses_future_labels": False,
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
        "sample_count": int(sample_index_stats["row_count"]),
        "sample_count_by_split": {str(key): int(value) for key, value in split_counts.items()},
        "candidate_count": int(candidate_index_stats["row_count"]),
        "candidate_count_by_split": {str(key): int(value) for key, value in candidate_split_counts.items()},
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
            "exit_close_raw": {
                "path": str((label_dir / "exit_close_raw.float32.dat").resolve()),
                "shape": [n_dates, n_symbols],
                "units": "CNY_raw_exchange_price",
            },
            "exit_down_limit_raw": {
                "path": str((label_dir / "exit_down_limit.float32.dat").resolve()),
                "shape": [n_dates, n_symbols],
                "units": "CNY_raw_exchange_price",
            },
        },
        "execution_cost_contract": execution_cost_contract,
        "terminal_execution_contract": {
            "unresolved_after_tail": "apply_precommitted_recovery_fraction",
            "recovery_fraction_of_entry_notional": unresolved_exit_recovery_fraction,
            "default_interpretation": "zero_recovery_total_loss" if unresolved_exit_recovery_fraction == 0.0 else "configured_partial_recovery",
        },
        "execution_views": {
            "exit_close_raw_path": {
                "source_array": "exit_close_raw",
                "shape": [n_dates, n_symbols, required_dependency_days],
                "indexing": "source[signal_date_idx+1:signal_date_idx+1+forward_days+execution_tail_days, symbol_idx]",
                "materialized": False,
            },
            "exit_down_limit_raw_path": {
                "source_array": "exit_down_limit_raw",
                "shape": [n_dates, n_symbols, required_dependency_days],
                "indexing": "source[signal_date_idx+1:signal_date_idx+1+forward_days+execution_tail_days, symbol_idx]",
                "materialized": False,
            },
            "exit_sellable_path": {
                "source_mask": "exit_sellable",
                "shape": [n_dates, n_symbols, required_dependency_days],
                "indexing": "source[signal_date_idx+1:signal_date_idx+1+forward_days+execution_tail_days, symbol_idx]",
                "materialized": False,
            },
        },
        "masks": {
            "input_valid": {"path": str((mask_dir / "input_valid.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "candidate_eligible": {"path": str((mask_dir / "candidate_eligible.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
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
            "exit_has_valid_bar_volume": {"path": str((mask_dir / "exit_has_valid_bar_volume.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
            "exit_sellable": {"path": str((mask_dir / "exit_sellable.bool.dat").resolve()), "shape": [n_dates, n_symbols]},
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
            "exit_sellable_path": {
                "source_mask": "exit_sellable",
                "shape": [n_dates, n_symbols, int(config.forward_days) + execution_tail_days],
                "indexing": "source[signal_date_idx+1:signal_date_idx+1+forward_days+execution_tail_days, symbol_idx]",
                "materialized": False,
            },
        },
        "sample_index_path": str(sample_index_path.resolve()),
        "candidate_index_path": str(candidate_index_path.resolve()),
        "index_semantics": {
            "sample_index": "supervised rows; candidate eligibility plus label_valid (and legacy entry fill when configured)",
            "candidate_index": "full scoring universe; signal-day input_valid plus signal_eligible, never future-label or entry-fill gated",
            "join_key": ["date_idx", "symbol_idx"],
            "label_flags": ["label_valid", "price_label_valid", "va_aux_valid"],
            "sample_index_write": sample_index_stats,
            "candidate_index_write": candidate_index_stats,
        },
        "normalization": normalization,
        "label_semantics": {
            "entry_anchor": "signal day close decision, next calendar trading day open entry",
            "entry_fill": "evaluated after ranking from raw next-day open, suspension state, and the configured exchange-tick limit rule",
            "earliest_exit": "path day 2 (T+1 after the next-open entry day)",
            "exit_fill": "planned close exit defers to the first observed, status-valid, non-suspended, non-delisted day whose raw close is above the tick-rounded down limit",
            "exit_retry_window": f"prediction remains {int(config.forward_days)} days; exit may defer for {execution_tail_days} additional trading days",
            "unresolved_exit": f"after the retry window recover {unresolved_exit_recovery_fraction:.6f} of entry notional before costs",
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
    candidate_index_path: Path | None = None
    if str(source.get("candidate_index_path", "") or ""):
        candidate_index_src = Path(str(source["candidate_index_path"]))
        candidate_index_path = _hardlink_file(
            candidate_index_src,
            output_dir / "candidate_index.parquet",
            overwrite=overwrite,
        )
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
    if candidate_index_path is not None:
        manifest["candidate_index_path"] = str(candidate_index_path.resolve())
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
    for section in ["feature_channels", "label_arrays", "execution_arrays", "masks"]:
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
    candidate_index_raw = str(manifest.get("candidate_index_path", "") or "")
    if candidate_index_raw:
        if not Path(candidate_index_raw).exists():
            blockers.append("missing_candidate_index")
    elif bool(dict(manifest.get("data_semantics", {}) or {}).get("candidate_selection_uses_future_labels") is False):
        blockers.append("missing_candidate_index")
    if (
        bool(dict(manifest.get("data_semantics", {}) or {}).get("candidate_selection_uses_future_labels") is False)
        and manifest.get("dependency_padding_complete") is not True
    ):
        blockers.append("incomplete_forward_execution_dependency_padding")
    return {
        "status": "blocked" if blockers else "ok",
        "blockers": blockers,
        "manifest_path": str(path.resolve()),
        "sample_count": int(manifest.get("sample_count", 0) or 0),
        "sample_count_by_split": dict(manifest.get("sample_count_by_split", {}) or {}),
        "candidate_count": int(manifest.get("candidate_count", 0) or 0),
        "candidate_count_by_split": dict(manifest.get("candidate_count_by_split", {}) or {}),
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
        "--execution-tail-days",
        type=int,
        default=DEFAULT_EXECUTION_TAIL_DAYS,
        help="Additional trading days available only for deferred exit execution; does not extend the prediction target.",
    )
    build.add_argument(
        "--require-full-dependency-padding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Block unless every requested signal date has forward_days + execution_tail_days of calendar padding.",
    )
    build.add_argument(
        "--minimum-free-memory-gb",
        type=float,
        default=DEFAULT_MINIMUM_FREE_MEMORY_GB,
        help="Stop safely before a large stage when available physical memory falls below this threshold.",
    )
    build.add_argument(
        "--unresolved-exit-recovery-fraction",
        type=float,
        default=0.0,
        help="Precommitted terminal recovery fraction after the execution tail; 0 is total-loss recovery.",
    )
    build.add_argument("--lot-size", type=int, default=DEFAULT_LOT_SIZE)
    build.add_argument("--commission-bps", type=float, default=DEFAULT_COMMISSION_BPS)
    build.add_argument("--minimum-commission-cny", type=float, default=DEFAULT_MIN_COMMISSION_CNY)
    build.add_argument("--stamp-tax-bps", type=float, default=DEFAULT_STAMP_TAX_BPS)
    build.add_argument(
        "--stamp-tax-schedule",
        default=",".join(f"{date}:{rate:g}" for date, rate in DEFAULT_STAMP_TAX_SCHEDULE),
        help="Comma-separated effective-date rates, for example 1900-01-01:10,2023-08-28:5.",
    )
    build.add_argument("--transfer-fee-bps", type=float, default=DEFAULT_TRANSFER_FEE_BPS)
    build.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    build.add_argument(
        "--stress-slippage-multiplier",
        type=float,
        default=DEFAULT_STRESS_SLIPPAGE_MULTIPLIER,
    )
    build.add_argument(
        "--research-contract",
        type=Path,
        default=None,
        help="Approved immutable research contract; semantic and raw-file SHA-256 identities are bound into the pack manifest.",
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
            execution_tail_days=int(args.execution_tail_days),
            require_full_dependency_padding=bool(args.require_full_dependency_padding),
            minimum_free_memory_gb=float(args.minimum_free_memory_gb),
            unresolved_exit_recovery_fraction=float(args.unresolved_exit_recovery_fraction),
            execution_cost=AShareExecutionCostConfig(
                lot_size=int(args.lot_size),
                commission_bps=float(args.commission_bps),
                minimum_commission_cny=float(args.minimum_commission_cny),
                stamp_tax_bps=float(args.stamp_tax_bps),
                stamp_tax_schedule=_parse_rate_schedule(
                    args.stamp_tax_schedule,
                    default=DEFAULT_STAMP_TAX_SCHEDULE,
                ),
                transfer_fee_bps=float(args.transfer_fee_bps),
                slippage_bps=float(args.slippage_bps),
                stress_slippage_multiplier=float(args.stress_slippage_multiplier),
            ),
            research_contract=Path(args.research_contract) if args.research_contract is not None else None,
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
