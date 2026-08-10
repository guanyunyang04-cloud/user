from __future__ import annotations

"""Full-market multi-target forecasting with reusable LightGBM datasets."""

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Self

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from daily_research.path_policy import seq100_v4_economic_realizability as economic
from daily_research.path_policy.seq100_candidate_execution import (
    ExecutionCosts,
    parse_execution_costs,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_full_market_multitask_forecast_v1"
MAXIMUM_OUTCOME_DATE = "2025-12-31"
FORBIDDEN_YEAR = 2026
HORIZONS = (2, 3, 5, 10)
RETRY_DAYS = 20
COMMON_PURGE_DAYS = 30
INNER_VALIDATION_DAYS = 291
RESERVE_SYSTEM_GIB = 1.0
DEFAULT_FEATURE_VARIANT = "all_causal_557"
FEATURE_VARIANT_FAMILIES = {
    "all_causal_557": None,
    "price_path_context_364": (
        "daily_cross_sectional_technical",
        "daily_price_volume_technical",
        "traditional_technical_indicators",
        "market_state",
        "same_day_5m",
    ),
    "price_path_core_183": (
        "daily_price_volume_technical",
        "market_state",
        "same_day_5m",
    ),
}
FEATURE_VARIANT_COUNTS = {
    "all_causal_557": 557,
    "price_path_context_364": 364,
    "price_path_core_183": 183,
}
TRAINING_MODES = ("outer_early_stop", "causal_nested")
TAKE_PROFIT_THRESHOLD = 0.01
TAKE_PROFIT_EXIT_THRESHOLDS = {
    "take_profit_1pct_else_planned_close": 0.01,
    "take_profit_2pct_else_planned_close": 0.02,
    "take_profit_3pct_else_planned_close": 0.03,
    "take_profit_4pct_else_planned_close": 0.04,
    "take_profit_5pct_else_planned_close": 0.05,
}
PRIMARY_EXIT_POLICIES = ("planned_close",)
EXACT_NET_SCENARIOS = {"base": 1.0, "stress": 2.0}
LOWER_QUANTILE_ALPHA = 0.10

TARGET_COLUMNS = (
    "next_close_return",
    *(
        name
        for horizon in HORIZONS
        for name in (
            f"endpoint_return_d{horizon}",
            f"legal_exit_return_d{horizon}",
            f"market_excess_endpoint_return_d{horizon}",
            f"legal_mfe_d{horizon}",
            f"exposure_mae_d{horizon}",
            f"signed_path_efficiency_d{horizon}",
        )
    ),
)
EXACT_NET_TARGET_COLUMNS = tuple(
    f"exact_net_return_d{horizon}_{scenario}"
    for horizon in HORIZONS
    for scenario in EXACT_NET_SCENARIOS
)

TARGET_SCHEMA = "seq100_full_market_multitask_targets/1"
EXACT_NET_TARGET_SCHEMA = "seq100_full_market_exact_net_targets/1"
CACHE_SCHEMA = "seq100_full_market_lightgbm_cache/1"
TASK_SCHEMA = "seq100_full_market_multitask_task/1"
EVALUATION_SCHEMA = "seq100_full_market_multitask_oof_evaluation/1"
ACCOUNT_REPLAY_SCHEMA = "seq100_full_market_multitask_account_replay/1"
PAYOFF_EVALUATION_SCHEMA = "seq100_full_market_payoff_oof_evaluation/1"
MARKET_REGIME_SCHEMA = "seq100_full_market_market_regime_evaluation/1"

DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_full_market_multitask_forecast_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies"
    / "seq100_full_market_multitask_forecast_v1"
)


class FullMarketForecastError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (
        (WORKSPACE_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FullMarketForecastError(f"required_json_missing:{path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not_json_serializable:{type(value).__name__}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(partial, index=False, compression="zstd")
    os.replace(partial, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_record(path: Path, *, hash_file: bool = True, **extra: Any) -> dict[str, Any]:
    record = {
        "path": str(path.resolve()),
        "size": int(path.stat().st_size),
    }
    if hash_file:
        record["sha256"] = _sha256(path)
    record.update(extra)
    return record


def _emit(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            _json_safe({"event": event, "at": _now(), **payload}),
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        ),
        flush=True,
    )


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _read_json(path)
    if study.get("study_id") != STUDY_ID:
        raise FullMarketForecastError("study_id_mismatch")
    source = dict(study.get("sources", {}) or {})
    if (
        str(source.get("research_start_date")) != "2012-01-01"
        or str(source.get("maximum_outcome_date")) != MAXIMUM_OUTCOME_DATE
        or int(source.get("forbidden_year", -1)) != FORBIDDEN_YEAR
    ):
        raise FullMarketForecastError("source_period_contract_mismatch")
    targets = dict(study.get("targets", {}) or {})
    if (
        tuple(int(value) for value in targets.get("horizons", ())) != HORIZONS
        or str(targets.get("blocked_exit"))
        != "first sellable open within 20 trading days"
    ):
        raise FullMarketForecastError("target_contract_mismatch")
    validation = dict(study.get("validation", {}) or {})
    if (
        int(validation.get("forward_fold_count", -1)) != 5
        or int(validation.get("common_purge_trading_days", -1)) != COMMON_PURGE_DAYS
        or int(validation.get("inner_validation_trading_days", -1))
        != INNER_VALIDATION_DAYS
        or str(validation.get("validation_start_date")) != "2020-01-01"
        or str(validation.get("validation_end_date")) != MAXIMUM_OUTCOME_DATE
    ):
        raise FullMarketForecastError("validation_contract_mismatch")
    resources = dict(study.get("resources", {}) or {})
    if not math.isclose(
        float(resources.get("reserve_system_gib", -1.0)),
        RESERVE_SYSTEM_GIB,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise FullMarketForecastError("system_reserve_must_be_exactly_one_gib")
    configured_variants = dict(study.get("feature_variants", {}))
    for variant, expected_count in FEATURE_VARIANT_COUNTS.items():
        feature = dict(configured_variants.get(variant, {}))
        if int(feature.get("feature_count", -1)) != expected_count:
            raise FullMarketForecastError(f"feature_count_contract_mismatch:{variant}")
    return study


@dataclass(frozen=True)
class ResourcePlan:
    total_bytes: int
    available_bytes: int
    reserve_bytes: int
    usable_bytes: int
    cpu_threads: int
    histogram_pool_mb: int
    sequence_batch_size: int


def resource_plan(
    *,
    available_bytes: int | None = None,
    total_bytes: int | None = None,
    logical_cpus: int | None = None,
    reserve_gib: float = RESERVE_SYSTEM_GIB,
    maximum_threads: int = 16,
    histogram_pool_cap_mb: int = 1536,
    sequence_batch_cap: int = 65536,
) -> ResourcePlan:
    memory = psutil.virtual_memory()
    available = int(memory.available if available_bytes is None else available_bytes)
    total = int(memory.total if total_bytes is None else total_bytes)
    cpus = (
        int(psutil.cpu_count(logical=True) or 1)
        if logical_cpus is None
        else int(logical_cpus)
    )
    reserve = int(float(reserve_gib) * (1 << 30))
    usable = max(0, available - reserve)
    usable_mib = usable // (1 << 20)
    threads = max(1, min(int(maximum_threads), cpus))
    histogram_pool = max(
        256,
        min(int(histogram_pool_cap_mb), max(256, int(usable_mib * 0.20))),
    )
    if usable_mib >= 6144:
        batch = 65536
    elif usable_mib >= 3072:
        batch = 32768
    elif usable_mib >= 1536:
        batch = 16384
    else:
        batch = 8192
    return ResourcePlan(
        total_bytes=total,
        available_bytes=available,
        reserve_bytes=reserve,
        usable_bytes=usable,
        cpu_threads=threads,
        histogram_pool_mb=histogram_pool,
        sequence_batch_size=min(int(sequence_batch_cap), batch),
    )


def _assert_resource_capacity(plan: ResourcePlan, projected_bytes: int) -> None:
    if plan.usable_bytes <= 0:
        raise FullMarketForecastError("no_memory_available_after_one_gib_reserve")
    if int(projected_bytes) > int(plan.usable_bytes):
        raise FullMarketForecastError(
            f"projected_working_set_exceeds_dynamic_budget:{projected_bytes}:"
            f"{plan.usable_bytes}"
        )


def _trim_working_set() -> bool:
    if os.name != "nt" or not hasattr(ctypes, "WinDLL"):
        return False
    try:
        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        psapi = ctypes.WinDLL("psapi.dll", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
        psapi.EmptyWorkingSet.restype = ctypes.c_bool
        return bool(psapi.EmptyWorkingSet(kernel32.GetCurrentProcess()))
    except (AttributeError, OSError):
        return False


class ResourceMonitor(AbstractContextManager["ResourceMonitor"]):
    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = float(interval_seconds)
        self.process = psutil.Process()
        self.started = 0.0
        self.peak_rss_bytes = 0
        self.minimum_available_bytes = 2**63 - 1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_rss_bytes = max(
                self.peak_rss_bytes, int(self.process.memory_info().rss)
            )
            self.minimum_available_bytes = min(
                self.minimum_available_bytes, int(psutil.virtual_memory().available)
            )

    def __enter__(self) -> Self:
        self.started = time.perf_counter()
        self.peak_rss_bytes = int(self.process.memory_info().rss)
        self.minimum_available_bytes = int(psutil.virtual_memory().available)
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def metrics(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": float(time.perf_counter() - self.started),
            "peak_process_rss_bytes": int(self.peak_rss_bytes),
            "minimum_system_available_bytes": int(self.minimum_available_bytes),
        }


def _open_array(record: Mapping[str, Any], *, dtype: np.dtype[Any]) -> np.memmap:
    path = _resolve(str(record["path"]))
    shape = tuple(int(value) for value in record["shape"])
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if not path.is_file() or int(path.stat().st_size) != expected:
        raise FullMarketForecastError(f"source_array_invalid:{path}")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


@dataclass
class SourceContext:
    model_manifest_path: Path
    model_manifest: dict[str, Any]
    row_index: pd.DataFrame
    pack_path: Path
    pack: dict[str, Any]
    date_values: np.ndarray
    cutoff_idx: int


def _source_context(study: Mapping[str, Any]) -> SourceContext:
    model_path = _resolve(str(study["sources"]["model_input_manifest"]))
    model = _read_json(model_path)
    if model.get("status") != "completed" or int(model.get("row_count", -1)) <= 0:
        raise FullMarketForecastError("model_input_manifest_invalid")
    row_index = pd.read_parquet(Path(model["row_index"]["path"]))
    required = {
        "candidate_id",
        "date_idx",
        "symbol_idx",
        "trade_date",
        "symbol",
        "security_id",
    }
    if (
        not required.issubset(row_index.columns)
        or len(row_index) != int(model["row_count"])
        or not row_index["candidate_id"].is_unique
        or not row_index["date_idx"].is_monotonic_increasing
    ):
        raise FullMarketForecastError("model_row_index_contract_failed")
    dates_text = row_index["trade_date"].astype(str)
    years = tuple(sorted(dates_text.str[:4].astype(int).unique().tolist()))
    if years != tuple(range(2012, 2026)) or dates_text.str.startswith("2026-").any():
        raise FullMarketForecastError("formal_row_period_mismatch")
    label_manifest = _read_json(Path(model["label_manifest"]["path"]))
    pack_path = Path(label_manifest["source"]["pack_manifest"]).resolve()
    pack = _read_json(pack_path)
    date_values = np.asarray(pack["date_values"], dtype=str)
    cutoff = np.flatnonzero(date_values == MAXIMUM_OUTCOME_DATE)
    if len(cutoff) != 1:
        raise FullMarketForecastError("outcome_cutoff_missing_or_duplicate")
    cutoff_idx = int(cutoff[0])
    if not np.char.startswith(date_values[cutoff_idx + 1 :], "2026-").any():
        raise FullMarketForecastError("forbidden_year_boundary_not_explicit")
    symbols = row_index["symbol_idx"].to_numpy(dtype=np.int32)
    pack_symbols = np.asarray(pack["symbol_values"], dtype=str)
    if (
        bool((symbols < 0).any())
        or int(symbols.max()) >= len(pack_symbols)
        or not np.array_equal(
            pack_symbols[symbols], row_index["symbol"].astype(str).to_numpy()
        )
    ):
        raise FullMarketForecastError("row_symbol_pack_alignment_failed")
    return SourceContext(
        model_manifest_path=model_path,
        model_manifest=model,
        row_index=row_index,
        pack_path=pack_path,
        pack=pack,
        date_values=date_values,
        cutoff_idx=cutoff_idx,
    )


def _derive_next_open_sellable(
    *,
    daily_raw: np.ndarray,
    raw_open: np.ndarray,
    raw_down_limit: np.ndarray,
    status_valid: np.ndarray,
    suspended: np.ndarray,
    delisted: np.ndarray,
    cutoff_idx: int,
) -> np.ndarray:
    symbol_count = daily_raw.shape[1]
    output = np.full((cutoff_idx + 2, symbol_count), -1, dtype=np.int32)
    next_idx = np.full(symbol_count, -1, dtype=np.int32)
    for date_idx in range(int(cutoff_idx), -1, -1):
        raw = np.asarray(raw_open[date_idx], dtype=np.float64)
        limit = np.asarray(raw_down_limit[date_idx], dtype=np.float64)
        base = (
            np.isfinite(raw)
            & (raw > 0.0)
            & np.asarray(status_valid[date_idx], dtype=bool)
            & ~np.asarray(suspended[date_idx], dtype=bool)
            & ~np.asarray(delisted[date_idx], dtype=bool)
        )
        blocked = (
            base
            & np.isfinite(limit)
            & (np.floor(raw / 0.01 + 0.5) <= np.floor(limit / 0.01 + 0.5))
        )
        adjusted = np.asarray(daily_raw[date_idx, :, 0], dtype=np.float64)
        sellable = base & ~blocked & np.isfinite(adjusted) & (adjusted > 0.0)
        next_idx[sellable] = date_idx
        output[date_idx] = next_idx
    return output


def _derive_rows_for_date(
    *,
    signal_idx: int,
    symbols: np.ndarray,
    cutoff_idx: int,
    daily_raw: np.ndarray,
    entry_filled: np.ndarray,
    exit_sellable: np.ndarray,
    next_open_sellable: np.ndarray,
    retry_days: int = RETRY_DAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    symbols = np.asarray(symbols, dtype=np.int32)
    count = len(symbols)
    values = np.full((count, len(TARGET_COLUMNS)), np.nan, dtype=np.float32)
    valid = np.zeros((count, len(TARGET_COLUMNS)), dtype=np.uint8)
    fill_days = np.full((count, len(HORIZONS)), -1, dtype=np.int16)
    current_close = np.asarray(daily_raw[signal_idx, symbols, 3], dtype=np.float64)
    if signal_idx + 1 <= cutoff_idx:
        next_close = np.asarray(daily_raw[signal_idx + 1, symbols, 3], dtype=np.float64)
        next_valid = (
            np.isfinite(current_close)
            & (current_close > 0.0)
            & np.isfinite(next_close)
            & (next_close > 0.0)
        )
        values[next_valid, 0] = (
            next_close[next_valid] / current_close[next_valid] - 1.0
        ).astype(np.float32)
        valid[next_valid, 0] = 1
    if signal_idx + 1 > cutoff_idx:
        return values, valid, fill_days

    entry_idx = signal_idx + 1
    entry_open = np.asarray(daily_raw[entry_idx, symbols, 0], dtype=np.float64)
    entry_ok = (
        np.asarray(entry_filled[signal_idx, symbols], dtype=bool)
        & np.isfinite(entry_open)
        & (entry_open > 0.0)
    )
    for horizon_position, horizon in enumerate(HORIZONS):
        base = 1 + horizon_position * 6
        exit_idx = signal_idx + horizon
        if exit_idx > cutoff_idx:
            continue
        endpoint_close = np.asarray(daily_raw[exit_idx, symbols, 3], dtype=np.float64)
        endpoint_ok = entry_ok & np.isfinite(endpoint_close) & (endpoint_close > 0.0)
        endpoint = np.full(count, np.nan, dtype=np.float64)
        endpoint[endpoint_ok] = (
            endpoint_close[endpoint_ok] / entry_open[endpoint_ok] - 1.0
        )
        values[endpoint_ok, base] = endpoint[endpoint_ok].astype(np.float32)
        valid[endpoint_ok, base] = 1

        legal = np.full(count, np.nan, dtype=np.float64)
        direct = endpoint_ok & np.asarray(exit_sellable[exit_idx, symbols], dtype=bool)
        legal[direct] = endpoint[direct]
        fill_days[direct, horizon_position] = np.int16(horizon)
        blocked = entry_ok & ~direct
        if blocked.any():
            positions = np.flatnonzero(blocked)
            next_idx = np.asarray(
                next_open_sellable[exit_idx + 1, symbols[positions]], dtype=np.int32
            )
            delayed = (next_idx >= 0) & (
                next_idx <= min(cutoff_idx, exit_idx + int(retry_days))
            )
            if delayed.any():
                accepted = positions[delayed]
                accepted_dates = next_idx[delayed]
                delayed_open = np.asarray(
                    daily_raw[accepted_dates, symbols[accepted], 0], dtype=np.float64
                )
                price_ok = np.isfinite(delayed_open) & (delayed_open > 0.0)
                accepted = accepted[price_ok]
                accepted_dates = accepted_dates[price_ok]
                delayed_open = delayed_open[price_ok]
                legal[accepted] = delayed_open / entry_open[accepted] - 1.0
                fill_days[accepted, horizon_position] = (
                    accepted_dates - signal_idx
                ).astype(np.int16)
            if exit_idx + int(retry_days) <= cutoff_idx:
                unresolved = positions[~np.isfinite(legal[positions])]
                legal[unresolved] = -1.0
                fill_days[unresolved, horizon_position] = np.int16(
                    horizon + int(retry_days)
                )
        legal_ok = entry_ok & np.isfinite(legal)
        values[legal_ok, base + 1] = legal[legal_ok].astype(np.float32)
        valid[legal_ok, base + 1] = 1

        if endpoint_ok.any():
            mean_return = float(np.mean(endpoint[endpoint_ok]))
            values[endpoint_ok, base + 2] = (
                endpoint[endpoint_ok] - mean_return
            ).astype(np.float32)
            valid[endpoint_ok, base + 2] = 1

        legal_high = np.asarray(
            daily_raw[signal_idx + 2 : exit_idx + 1, symbols, 1], dtype=np.float64
        ).T
        if legal_high.shape[1]:
            high_ok = entry_ok & np.isfinite(legal_high).any(axis=1)
            maximum = np.max(
                np.where(np.isfinite(legal_high), legal_high, -np.inf), axis=1
            )
            values[high_ok, base + 3] = (
                maximum[high_ok] / entry_open[high_ok] - 1.0
            ).astype(np.float32)
            valid[high_ok, base + 3] = 1

        lows = np.asarray(
            daily_raw[entry_idx : exit_idx + 1, symbols, 2], dtype=np.float64
        ).T
        low_ok = entry_ok & np.isfinite(lows).any(axis=1)
        minimum = np.min(np.where(np.isfinite(lows), lows, np.inf), axis=1)
        values[low_ok, base + 4] = (minimum[low_ok] / entry_open[low_ok] - 1.0).astype(
            np.float32
        )
        valid[low_ok, base + 4] = 1

        closes = np.asarray(
            daily_raw[entry_idx : exit_idx + 1, symbols, 3], dtype=np.float64
        ).T
        closes_ok = entry_ok & np.isfinite(closes).all(axis=1)
        if closes.shape[1]:
            path = np.abs(closes[:, 0] - entry_open)
            if closes.shape[1] > 1:
                path += np.sum(np.abs(np.diff(closes, axis=1)), axis=1)
            efficiency_ok = closes_ok & (path > 0.0)
            values[efficiency_ok, base + 5] = (
                (closes[efficiency_ok, -1] - entry_open[efficiency_ok])
                / path[efficiency_ok]
            ).astype(np.float32)
            valid[efficiency_ok, base + 5] = 1
            flat = closes_ok & (path == 0.0)
            values[flat, base + 5] = 0.0
            valid[flat, base + 5] = 1
    return values, valid, fill_days


def _target_fingerprint(*, study_path: Path, context: SourceContext) -> str:
    return _stable_hash(
        {
            "schema": TARGET_SCHEMA,
            "study_sha256": _sha256(study_path),
            "model_input_fingerprint": context.model_manifest["input_fingerprint"],
            "row_index_sha256": context.model_manifest["row_index"]["sha256"],
            "pack_sha256": _sha256(context.pack_path),
            "target_columns": TARGET_COLUMNS,
            "horizons": HORIZONS,
            "retry_days": RETRY_DAYS,
            "maximum_outcome_date": MAXIMUM_OUTCOME_DATE,
            "forbidden_year": FORBIDDEN_YEAR,
        }
    )


def _target_manifest_valid(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        manifest = _read_json(path)
        if (
            manifest.get("schema") != TARGET_SCHEMA
            or manifest.get("status") != "completed"
            or manifest.get("fingerprint") != fingerprint
        ):
            return False
        for record in dict(manifest["files"]).values():
            file_path = Path(record["path"])
            if not file_path.is_file() or int(file_path.stat().st_size) != int(
                record["size"]
            ):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def prepare_targets(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    context = _source_context(study)
    fingerprint = _target_fingerprint(study_path=study_path, context=context)
    manifest_path = output_root / "targets/manifest.json"
    if _target_manifest_valid(manifest_path, fingerprint):
        return _read_json(manifest_path)
    row_index = context.row_index
    row_count = len(row_index)
    target_root = output_root / "targets"
    target_root.mkdir(parents=True, exist_ok=True)
    values_path = target_root / "values.float32.dat"
    valid_path = target_root / "valid.uint8.dat"
    fill_path = target_root / "legal_fill_days.int16.dat"
    partials = [
        Path(str(path) + ".partial") for path in (values_path, valid_path, fill_path)
    ]
    for partial in partials:
        partial.unlink(missing_ok=True)
    values = np.memmap(
        partials[0], dtype=np.float32, mode="w+", shape=(row_count, len(TARGET_COLUMNS))
    )
    valid = np.memmap(
        partials[1], dtype=np.uint8, mode="w+", shape=(row_count, len(TARGET_COLUMNS))
    )
    fill_days = np.memmap(
        partials[2], dtype=np.int16, mode="w+", shape=(row_count, len(HORIZONS))
    )
    values[:] = np.nan
    valid[:] = 0
    fill_days[:] = -1
    daily_raw = _open_array(
        context.pack["feature_channels"]["daily_raw"], dtype=np.float32
    )
    masks = context.pack["masks"]
    execution = context.pack["execution_arrays"]
    entry_filled = _open_array(masks["entry_filled"], dtype=np.bool_)
    exit_sellable = _open_array(masks["exit_sellable"], dtype=np.bool_)
    raw_open = _open_array(execution["entry_open_raw"], dtype=np.float32)
    raw_down_limit = _open_array(execution["exit_down_limit_raw"], dtype=np.float32)
    status_valid = _open_array(masks["status_valid"], dtype=np.bool_)
    suspended = _open_array(masks["is_suspended"], dtype=np.bool_)
    delisted = _open_array(masks["is_delisted"], dtype=np.bool_)
    plan = resource_plan(
        reserve_gib=float(study["resources"]["reserve_system_gib"]),
        maximum_threads=int(study["resources"]["maximum_cpu_threads"]),
        histogram_pool_cap_mb=int(study["resources"]["histogram_pool_cap_mb"]),
        sequence_batch_cap=int(study["resources"]["sequence_batch_cap"]),
    )
    _assert_resource_capacity(plan, 256 * (1 << 20))
    _emit("target_preparation_started", row_count=row_count, resource_plan=asdict(plan))
    with ResourceMonitor() as monitor:
        next_open_sellable = _derive_next_open_sellable(
            daily_raw=daily_raw,
            raw_open=raw_open,
            raw_down_limit=raw_down_limit,
            status_valid=status_valid,
            suspended=suspended,
            delisted=delisted,
            cutoff_idx=context.cutoff_idx,
        )
        dates = row_index["date_idx"].to_numpy(dtype=np.int32)
        symbols = row_index["symbol_idx"].to_numpy(dtype=np.int32)
        boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
        for group_number, (left, right) in enumerate(pairwise(boundaries), start=1):
            current_values, current_valid, current_fill = _derive_rows_for_date(
                signal_idx=int(dates[left]),
                symbols=symbols[left:right],
                cutoff_idx=context.cutoff_idx,
                daily_raw=daily_raw,
                entry_filled=entry_filled,
                exit_sellable=exit_sellable,
                next_open_sellable=next_open_sellable,
            )
            values[left:right] = current_values
            valid[left:right] = current_valid
            fill_days[left:right] = current_fill
            if group_number % 500 == 0:
                values.flush()
                valid.flush()
                fill_days.flush()
                _emit(
                    "target_preparation_progress",
                    signal_dates=group_number,
                    rows_completed=int(right),
                )
        values.flush()
        valid.flush()
        fill_days.flush()
        valid_counts = np.asarray(valid, dtype=np.uint8).sum(axis=0)
        target_means = []
        for column in range(len(TARGET_COLUMNS)):
            mask = np.asarray(valid[:, column], dtype=bool)
            target_means.append(
                float(np.asarray(values[:, column], dtype=np.float64)[mask].mean())
                if mask.any()
                else math.nan
            )
        monitor_metrics = monitor.metrics()
    del values, valid, fill_days, next_open_sellable
    gc.collect()
    os.replace(partials[0], values_path)
    os.replace(partials[1], valid_path)
    os.replace(partials[2], fill_path)
    files = {
        "values": _file_record(
            values_path,
            dtype="float32",
            shape=[row_count, len(TARGET_COLUMNS)],
            columns=list(TARGET_COLUMNS),
        ),
        "valid": _file_record(
            valid_path,
            dtype="uint8",
            shape=[row_count, len(TARGET_COLUMNS)],
            columns=list(TARGET_COLUMNS),
        ),
        "legal_fill_days": _file_record(
            fill_path,
            dtype="int16",
            shape=[row_count, len(HORIZONS)],
            horizons=list(HORIZONS),
        ),
    }
    manifest = {
        "schema": TARGET_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "fingerprint": fingerprint,
        "row_count": row_count,
        "target_columns": list(TARGET_COLUMNS),
        "valid_counts": {
            name: int(value)
            for name, value in zip(TARGET_COLUMNS, valid_counts, strict=True)
        },
        "target_means": {
            name: float(value)
            for name, value in zip(TARGET_COLUMNS, target_means, strict=True)
            if math.isfinite(value)
        },
        "contract": {
            "horizons": list(HORIZONS),
            "entry_day": 1,
            "earliest_legal_exit_day": 2,
            "retry_days": RETRY_DAYS,
            "maximum_outcome_date": MAXIMUM_OUTCOME_DATE,
            "maximum_source_date_idx_read": context.cutoff_idx,
            "maximum_source_date_read": str(context.date_values[context.cutoff_idx]),
            "forbidden_2026_read_count": 0,
        },
        "resource_plan": asdict(plan),
        "resource_metrics": monitor_metrics,
        "sources": {
            "study": _file_record(study_path),
            "model_inputs": _file_record(
                context.model_manifest_path,
                input_fingerprint=context.model_manifest["input_fingerprint"],
            ),
            "row_index": dict(context.model_manifest["row_index"]),
            "pack": _file_record(context.pack_path),
        },
        "files": files,
    }
    _write_json(manifest_path, manifest)
    _emit(
        "target_preparation_completed",
        elapsed_seconds=monitor_metrics["elapsed_seconds"],
    )
    return manifest


def _vectorized_exact_net_returns(
    *,
    signal_indices: np.ndarray,
    symbol_indices: np.ndarray,
    legal_gross_returns: np.ndarray,
    fill_days: np.ndarray,
    daily_raw: np.ndarray,
    raw_open: np.ndarray,
    date_values: np.ndarray,
    costs: ExecutionCosts,
    notional_cny: float,
    slippage_multiplier: float,
    maximum_date_idx: int,
    entry_filled: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the exact lot/fee/tax cashflow contract to a row batch."""

    signals = np.asarray(signal_indices, dtype=np.int64)
    symbols = np.asarray(symbol_indices, dtype=np.int64)
    gross_returns = np.asarray(legal_gross_returns, dtype=np.float64)
    exits_after = np.asarray(fill_days, dtype=np.int64)
    if not (
        signals.ndim == symbols.ndim == gross_returns.ndim == exits_after.ndim == 1
        and len(signals) == len(symbols) == len(gross_returns) == len(exits_after)
    ):
        raise FullMarketForecastError("exact_net_batch_shape_mismatch")
    if (
        not math.isfinite(float(notional_cny))
        or float(notional_cny) <= 0.0
        or not math.isfinite(float(slippage_multiplier))
        or float(slippage_multiplier) < 0.0
    ):
        raise FullMarketForecastError("exact_net_cash_or_slippage_invalid")

    count = len(signals)
    output = np.full(count, np.nan, dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    if not count:
        return output, valid
    symbol_count = int(raw_open.shape[1])
    entry_indices = signals + 1
    exit_indices = signals + exits_after
    index_ok = (
        (signals >= 0)
        & (symbols >= 0)
        & (symbols < symbol_count)
        & (entry_indices <= int(maximum_date_idx))
        & (exit_indices <= int(maximum_date_idx))
        & (exits_after >= 2)
    )
    safe_signals = np.clip(signals, 0, int(maximum_date_idx))
    safe_entries = np.clip(entry_indices, 0, int(maximum_date_idx))
    safe_exits = np.clip(exit_indices, 0, int(maximum_date_idx))
    safe_symbols = np.clip(symbols, 0, max(symbol_count - 1, 0))
    adjusted_entry = np.asarray(
        daily_raw[safe_entries, safe_symbols, 0], dtype=np.float64
    )
    raw_entry = np.asarray(raw_open[safe_entries, safe_symbols], dtype=np.float64)
    eligible = (
        index_ok
        & np.isfinite(gross_returns)
        & np.isfinite(adjusted_entry)
        & (adjusted_entry > 0.0)
        & np.isfinite(raw_entry)
        & (raw_entry > 0.0)
    )
    if entry_filled is not None:
        eligible &= np.asarray(entry_filled[safe_signals, safe_symbols], dtype=bool)
    if not eligible.any():
        return output, valid

    slippage_rate = float(costs.slippage_bps) * float(slippage_multiplier) / 10_000.0
    fill_price = raw_entry * (1.0 + slippage_rate)
    unit_fill_notional = fill_price * int(costs.lot_size)
    maximum_lots = np.zeros(count, dtype=np.int64)
    maximum_lots[eligible] = np.floor(
        float(notional_cny) / unit_fill_notional[eligible]
    ).astype(np.int64)

    # Fees can make the naive price-only lot count unaffordable.  A vectorized
    # binary search exactly matches the scalar execution engine without a
    # multi-million-row Python loop.
    lower = np.zeros(count, dtype=np.int64)
    upper = maximum_lots.copy()
    commission_rate = float(costs.commission_bps) / 10_000.0
    transfer_rate = float(costs.transfer_fee_bps) / 10_000.0
    while bool(np.any(lower < upper)):
        active = lower < upper
        middle = (lower + upper + 1) // 2
        trial_fill = middle.astype(np.float64) * unit_fill_notional
        trial_commission = np.maximum(
            float(costs.minimum_commission_cny), trial_fill * commission_rate
        )
        trial_outflow = trial_fill + trial_commission + trial_fill * transfer_rate
        affordable = trial_outflow <= float(notional_cny) + 1.0e-9
        lower = np.where(active & affordable, middle, lower)
        upper = np.where(active & ~affordable, middle - 1, upper)

    lots = lower
    shares = lots.astype(np.float64) * int(costs.lot_size)
    position_ok = eligible & (lots > 0)
    buy_fill_notional = shares * fill_price
    buy_commission = np.maximum(
        float(costs.minimum_commission_cny),
        buy_fill_notional * commission_rate,
    )
    buy_transfer = buy_fill_notional * transfer_rate
    buy_outflow = buy_fill_notional + buy_commission + buy_transfer
    residual_cash = float(notional_cny) - buy_outflow
    gross_entry_notional = shares * raw_entry

    proceeds = np.zeros(count, dtype=np.float64)
    ordinary_exit = position_ok & (gross_returns > -1.0)
    if ordinary_exit.any():
        gross_exit_value = gross_entry_notional[ordinary_exit] * (
            1.0 + gross_returns[ordinary_exit]
        )
        sell_fill = gross_exit_value * max(0.0, 1.0 - slippage_rate)
        sell_commission = np.maximum(
            float(costs.minimum_commission_cny), sell_fill * commission_rate
        )
        sell_transfer = sell_fill * transfer_rate
        exit_dates = np.asarray(date_values, dtype=str)[safe_exits[ordinary_exit]]
        stamp_bps = np.zeros(len(exit_dates), dtype=np.float64)
        for effective_date, rate in costs.stamp_tax_schedule:
            stamp_bps[exit_dates >= str(effective_date)] = float(rate)
        stamp_tax = sell_fill * stamp_bps / 10_000.0
        proceeds[ordinary_exit] = (
            sell_fill - sell_commission - sell_transfer - stamp_tax
        )

    valid = position_ok
    output[valid] = (
        (residual_cash[valid] + proceeds[valid]) / float(notional_cny) - 1.0
    ).astype(np.float32)
    return output, valid


def _exact_net_target_fingerprint(
    *,
    study_path: Path,
    context: SourceContext,
    path_target_manifest: Mapping[str, Any],
    costs: ExecutionCosts,
    notional_cny: float,
) -> str:
    return _stable_hash(
        {
            "schema": EXACT_NET_TARGET_SCHEMA,
            "study_sha256": _sha256(study_path),
            "path_target_fingerprint": path_target_manifest["fingerprint"],
            "model_input_fingerprint": context.model_manifest["input_fingerprint"],
            "pack_sha256": _sha256(context.pack_path),
            "target_columns": EXACT_NET_TARGET_COLUMNS,
            "horizons": HORIZONS,
            "notional_cny": float(notional_cny),
            "costs": asdict(costs),
            "scenarios": EXACT_NET_SCENARIOS,
            "maximum_outcome_date": MAXIMUM_OUTCOME_DATE,
            "forbidden_year": FORBIDDEN_YEAR,
        }
    )


def _exact_net_target_manifest_valid(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        manifest = _read_json(path)
        if (
            manifest.get("schema") != EXACT_NET_TARGET_SCHEMA
            or manifest.get("status") != "completed"
            or manifest.get("fingerprint") != fingerprint
        ):
            return False
        for record in dict(manifest["files"]).values():
            file_path = Path(record["path"])
            if not file_path.is_file() or int(file_path.stat().st_size) != int(
                record["size"]
            ):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def prepare_exact_net_targets(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Build fixed-notional executable returns from the legal path panel."""

    study = load_study(study_path)
    context = _source_context(study)
    path_manifest = prepare_targets(study_path=study_path, output_root=output_root)
    costs = parse_execution_costs(context.pack)
    if not math.isclose(
        float(EXACT_NET_SCENARIOS["stress"]),
        float(costs.stress_slippage_multiplier),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise FullMarketForecastError("stress_slippage_contract_mismatch")
    notional = float(study["targets"]["positive_after_cost_notional_cny"])
    fingerprint = _exact_net_target_fingerprint(
        study_path=study_path,
        context=context,
        path_target_manifest=path_manifest,
        costs=costs,
        notional_cny=notional,
    )
    target_root = output_root / "exact_net_targets"
    manifest_path = target_root / "manifest.json"
    if _exact_net_target_manifest_valid(manifest_path, fingerprint):
        return _read_json(manifest_path)

    row_count = int(path_manifest["row_count"])
    values_path = target_root / "values.float32.dat"
    valid_path = target_root / "valid.uint8.dat"
    target_root.mkdir(parents=True, exist_ok=True)
    partial_values = Path(str(values_path) + ".partial")
    partial_valid = Path(str(valid_path) + ".partial")
    partial_values.unlink(missing_ok=True)
    partial_valid.unlink(missing_ok=True)
    output_values = np.memmap(
        partial_values,
        dtype=np.float32,
        mode="w+",
        shape=(row_count, len(EXACT_NET_TARGET_COLUMNS)),
    )
    output_valid = np.memmap(
        partial_valid,
        dtype=np.uint8,
        mode="w+",
        shape=(row_count, len(EXACT_NET_TARGET_COLUMNS)),
    )
    output_values[:] = np.nan
    output_valid[:] = 0

    path_values_record = path_manifest["files"]["values"]
    path_valid_record = path_manifest["files"]["valid"]
    fill_record = path_manifest["files"]["legal_fill_days"]
    path_values = np.memmap(
        Path(path_values_record["path"]),
        dtype=np.float32,
        mode="r",
        shape=tuple(int(value) for value in path_values_record["shape"]),
    )
    path_valid = np.memmap(
        Path(path_valid_record["path"]),
        dtype=np.uint8,
        mode="r",
        shape=tuple(int(value) for value in path_valid_record["shape"]),
    )
    legal_fill_days = np.memmap(
        Path(fill_record["path"]),
        dtype=np.int16,
        mode="r",
        shape=tuple(int(value) for value in fill_record["shape"]),
    )
    daily_raw = _open_array(
        context.pack["feature_channels"]["daily_raw"], dtype=np.float32
    )
    raw_open = _open_array(
        context.pack["execution_arrays"]["entry_open_raw"], dtype=np.float32
    )
    entry_filled = _open_array(context.pack["masks"]["entry_filled"], dtype=np.bool_)
    row_dates = context.row_index["date_idx"].to_numpy(dtype=np.int32)
    row_symbols = context.row_index["symbol_idx"].to_numpy(dtype=np.int32)
    plan = resource_plan(
        reserve_gib=float(study["resources"]["reserve_system_gib"]),
        maximum_threads=int(study["resources"]["maximum_cpu_threads"]),
        histogram_pool_cap_mb=int(study["resources"]["histogram_pool_cap_mb"]),
        sequence_batch_cap=int(study["resources"]["sequence_batch_cap"]),
    )
    _assert_resource_capacity(plan, 512 * (1 << 20))
    chunk_size = max(65_536, int(plan.sequence_batch_size) * 4)
    counts = np.zeros(len(EXACT_NET_TARGET_COLUMNS), dtype=np.int64)
    sums = np.zeros(len(EXACT_NET_TARGET_COLUMNS), dtype=np.float64)
    path_columns = list(path_manifest["target_columns"])
    _emit(
        "exact_net_target_preparation_started",
        row_count=row_count,
        column_count=len(EXACT_NET_TARGET_COLUMNS),
        chunk_size=chunk_size,
        resource_plan=asdict(plan),
    )
    with ResourceMonitor() as monitor:
        for left in range(0, row_count, chunk_size):
            right = min(left + chunk_size, row_count)
            for horizon_position, horizon in enumerate(HORIZONS):
                source_column = path_columns.index(f"legal_exit_return_d{horizon}")
                source_valid = np.asarray(
                    path_valid[left:right, source_column], dtype=bool
                )
                gross_return = np.asarray(
                    path_values[left:right, source_column], dtype=np.float64
                )
                gross_return[~source_valid] = np.nan
                current_fill = np.asarray(
                    legal_fill_days[left:right, horizon_position], dtype=np.int16
                )
                for scenario_position, (scenario, multiplier) in enumerate(
                    EXACT_NET_SCENARIOS.items()
                ):
                    column = (
                        horizon_position * len(EXACT_NET_SCENARIOS) + scenario_position
                    )
                    current_values, current_valid = _vectorized_exact_net_returns(
                        signal_indices=row_dates[left:right],
                        symbol_indices=row_symbols[left:right],
                        legal_gross_returns=gross_return,
                        fill_days=current_fill,
                        daily_raw=daily_raw,
                        raw_open=raw_open,
                        date_values=context.date_values,
                        costs=costs,
                        notional_cny=notional,
                        slippage_multiplier=float(multiplier),
                        maximum_date_idx=context.cutoff_idx,
                        entry_filled=entry_filled,
                    )
                    output_values[left:right, column] = current_values
                    output_valid[left:right, column] = current_valid.astype(np.uint8)
                    counts[column] += int(current_valid.sum())
                    if current_valid.any():
                        sums[column] += float(
                            np.asarray(
                                current_values[current_valid], dtype=np.float64
                            ).sum()
                        )
            if right % (chunk_size * 4) == 0 or right == row_count:
                output_values.flush()
                output_valid.flush()
                _emit("exact_net_target_preparation_progress", rows_completed=right)
        output_values.flush()
        output_valid.flush()
        monitor_metrics = monitor.metrics()
    del output_values, output_valid
    gc.collect()
    os.replace(partial_values, values_path)
    os.replace(partial_valid, valid_path)
    files = {
        "values": _file_record(
            values_path,
            dtype="float32",
            shape=[row_count, len(EXACT_NET_TARGET_COLUMNS)],
            columns=list(EXACT_NET_TARGET_COLUMNS),
        ),
        "valid": _file_record(
            valid_path,
            dtype="uint8",
            shape=[row_count, len(EXACT_NET_TARGET_COLUMNS)],
            columns=list(EXACT_NET_TARGET_COLUMNS),
        ),
    }
    manifest = {
        "schema": EXACT_NET_TARGET_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "fingerprint": fingerprint,
        "row_count": row_count,
        "target_columns": list(EXACT_NET_TARGET_COLUMNS),
        "valid_counts": {
            name: int(value)
            for name, value in zip(EXACT_NET_TARGET_COLUMNS, counts, strict=True)
        },
        "target_means": {
            name: float(sums[position] / counts[position])
            for position, name in enumerate(EXACT_NET_TARGET_COLUMNS)
            if counts[position] > 0
        },
        "contract": {
            "horizons": list(HORIZONS),
            "notional_cny": notional,
            "costs": asdict(costs),
            "slippage_multipliers": dict(EXACT_NET_SCENARIOS),
            "entry_day": 1,
            "earliest_legal_exit_day": 2,
            "maximum_outcome_date": MAXIMUM_OUTCOME_DATE,
            "maximum_source_date_idx_read": context.cutoff_idx,
            "maximum_source_date_read": str(context.date_values[context.cutoff_idx]),
            "forbidden_2026_read_count": 0,
        },
        "resource_plan": asdict(plan),
        "resource_metrics": monitor_metrics,
        "sources": {
            "study": _file_record(study_path),
            "path_targets": _file_record(
                output_root / "targets/manifest.json",
                target_fingerprint=path_manifest["fingerprint"],
            ),
            "model_inputs": _file_record(
                context.model_manifest_path,
                input_fingerprint=context.model_manifest["input_fingerprint"],
            ),
            "pack": _file_record(context.pack_path),
        },
        "files": files,
    }
    _write_json(manifest_path, manifest)
    _emit(
        "exact_net_target_preparation_completed",
        elapsed_seconds=monitor_metrics["elapsed_seconds"],
    )
    return manifest


def build_forward_folds(
    *,
    date_idx: np.ndarray,
    trade_date: np.ndarray,
    validation_start_date: str = "2020-01-01",
    validation_end_date: str = MAXIMUM_OUTCOME_DATE,
    fold_count: int = 5,
    purge_days: int = COMMON_PURGE_DAYS,
) -> list[dict[str, Any]]:
    dates = np.asarray(date_idx, dtype=np.int32)
    labels = np.asarray(trade_date, dtype=str)
    unique_dates, first = np.unique(dates, return_index=True)
    unique_labels = labels[first]
    selected = unique_dates[
        (unique_labels >= str(validation_start_date))
        & (unique_labels <= str(validation_end_date))
    ]
    blocks = np.array_split(selected, int(fold_count))
    folds: list[dict[str, Any]] = []
    for fold_number, block in enumerate(blocks, start=1):
        if not len(block):
            raise FullMarketForecastError("empty_validation_fold")
        start_idx = int(block[0])
        end_idx = int(block[-1])
        train_max = start_idx - int(purge_days) - 1
        train_rows = int(np.sum(dates <= train_max))
        evaluation_rows = int(np.sum((dates >= start_idx) & (dates <= end_idx)))
        if train_rows <= 0 or evaluation_rows <= 0:
            raise FullMarketForecastError("fold_support_empty")
        folds.append(
            {
                "fold": fold_number,
                "validation_start_date_idx": start_idx,
                "validation_end_date_idx": end_idx,
                "validation_start_date": str(
                    unique_labels[np.searchsorted(unique_dates, start_idx)]
                ),
                "validation_end_date": str(
                    unique_labels[np.searchsorted(unique_dates, end_idx)]
                ),
                "validation_date_count": len(block),
                "training_maximum_date_idx": train_max,
                "training_maximum_date": str(
                    unique_labels[
                        np.searchsorted(unique_dates, train_max, side="right") - 1
                    ]
                ),
                "training_row_count": train_rows,
                "validation_row_count": evaluation_rows,
                "purge_days": int(purge_days),
            }
        )
    return folds


class MatrixSequence(lgb.Sequence):
    def __init__(
        self,
        matrix: np.ndarray,
        rows: np.ndarray,
        *,
        batch_size: int,
    ) -> None:
        self.matrix = matrix
        self.rows = np.asarray(rows, dtype=np.int64)
        self.batch_size = int(batch_size)

    def __len__(self) -> int:
        return len(self.rows)

    def _block(self, local: np.ndarray) -> np.ndarray:
        selected = self.rows[np.asarray(local, dtype=np.int64)]
        if len(selected) and np.all(np.diff(selected) == 1):
            return np.asarray(
                self.matrix[int(selected[0]) : int(selected[-1]) + 1],
                dtype=np.float64,
            )
        return np.asarray(self.matrix[selected], dtype=np.float64)

    def __getitem__(self, index: Any) -> np.ndarray:
        if isinstance(index, slice):
            start = 0 if index.start is None else int(index.start)
            stop = len(self.rows) if index.stop is None else int(index.stop)
            step = 1 if index.step is None else int(index.step)
            return self._block(np.arange(start, stop, step, dtype=np.int64))
        if isinstance(index, (list, tuple, np.ndarray)):
            return self._block(np.asarray(index, dtype=np.int64))
        if isinstance(index, (int, np.integer)):
            return self._block(np.asarray([int(index)], dtype=np.int64))[0]
        raise TypeError(type(index).__name__)


def _feature_variant_contract(
    study: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    feature_variant: str,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, Any]]:
    variant = str(feature_variant)
    if variant not in FEATURE_VARIANT_FAMILIES:
        raise FullMarketForecastError(f"unknown_feature_variant:{variant}")
    configured = dict(study["feature_variants"][variant])
    full_names = tuple(model_manifest["feature_groups"]["compact_core"])
    records = {
        str(record["feature_name"]): dict(record)
        for record in model_manifest["features"]
    }
    families = FEATURE_VARIANT_FAMILIES[variant]
    selected_names = tuple(
        name
        for name in full_names
        if families is None or str(records[name]["analytic_family"]) in families
    )
    columns = np.asarray(
        [int(records[name]["column_index"]) for name in selected_names],
        dtype=np.int32,
    )
    expected_count = int(FEATURE_VARIANT_COUNTS[variant])
    if (
        len(selected_names) != expected_count
        or int(configured["feature_count"]) != expected_count
        or len(np.unique(columns)) != expected_count
    ):
        raise FullMarketForecastError(f"feature_variant_resolution_failed:{variant}")
    if families is not None and tuple(configured["analytic_families"]) != tuple(
        families
    ):
        raise FullMarketForecastError(f"feature_family_contract_mismatch:{variant}")
    return selected_names, columns, configured


def _feature_view_valid(
    manifest_path: Path, *, fingerprint: str, expected_size: int
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = _read_json(manifest_path)
        path = Path(manifest["file"]["path"])
        return bool(
            manifest.get("status") == "completed"
            and manifest.get("fingerprint") == fingerprint
            and path.is_file()
            and int(path.stat().st_size) == int(expected_size)
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def prepare_feature_view(
    *,
    study: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    output_root: Path,
    feature_variant: str,
) -> tuple[np.memmap, tuple[str, ...], dict[str, Any]]:
    feature_names, columns, configured = _feature_variant_contract(
        study, model_manifest, feature_variant
    )
    source = dict(model_manifest["storage"]["compact"])
    source_shape = tuple(int(value) for value in source["shape"])
    if feature_variant == DEFAULT_FEATURE_VARIANT:
        matrix = np.memmap(
            Path(source["path"]),
            dtype=np.float32,
            mode="r",
            shape=source_shape,
        )
        return (
            matrix,
            feature_names,
            {
                "status": "source_matrix_reused",
                "feature_variant": feature_variant,
                "fingerprint": _stable_hash(
                    {
                        "source_sample_hash": source["sample_hash"],
                        "feature_names": feature_names,
                    }
                ),
                "file": source,
            },
        )
    fingerprint = _stable_hash(
        {
            "schema": "seq100_full_market_feature_view/1",
            "model_input_fingerprint": model_manifest["input_fingerprint"],
            "source_sample_hash": source["sample_hash"],
            "feature_variant": feature_variant,
            "feature_names": feature_names,
            "source_columns": columns.tolist(),
        }
    )
    view_root = output_root / "feature_views" / feature_variant
    view_path = view_root / "matrix.float32.dat"
    manifest_path = view_root / "manifest.json"
    shape = (source_shape[0], len(feature_names))
    expected_size = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    if not _feature_view_valid(
        manifest_path, fingerprint=fingerprint, expected_size=expected_size
    ):
        view_root.mkdir(parents=True, exist_ok=True)
        partial = Path(str(view_path) + ".partial")
        partial.unlink(missing_ok=True)
        source_matrix = np.memmap(
            Path(source["path"]),
            dtype=np.float32,
            mode="r",
            shape=source_shape,
        )
        destination = np.memmap(
            partial,
            dtype=np.float32,
            mode="w+",
            shape=shape,
        )
        resource = resource_plan(
            reserve_gib=float(study["resources"]["reserve_system_gib"]),
            maximum_threads=int(study["resources"]["maximum_cpu_threads"]),
            histogram_pool_cap_mb=int(study["resources"]["histogram_pool_cap_mb"]),
            sequence_batch_cap=int(study["resources"]["sequence_batch_cap"]),
        )
        batch_size = max(16_384, int(resource.sequence_batch_size))
        _emit(
            "feature_view_build_started",
            feature_variant=feature_variant,
            rows=shape[0],
            features=shape[1],
            batch_size=batch_size,
        )
        with ResourceMonitor() as monitor:
            for left in range(0, shape[0], batch_size):
                right = min(left + batch_size, shape[0])
                destination[left:right] = source_matrix[left:right, columns]
            destination.flush()
            metrics = monitor.metrics()
        sample_rows = np.asarray([0, shape[0] // 2, shape[0] - 1], dtype=np.int64)
        sample_hash = hashlib.sha256(
            np.asarray(destination[sample_rows], dtype=np.float32).tobytes()
        ).hexdigest()
        del destination, source_matrix
        gc.collect()
        os.replace(partial, view_path)
        manifest = {
            "schema": "seq100_full_market_feature_view/1",
            "status": "completed",
            "completed_at": _now(),
            "fingerprint": fingerprint,
            "feature_variant": feature_variant,
            "feature_count": len(feature_names),
            "analytic_families": configured["analytic_families"],
            "shape": list(shape),
            "source_columns": columns.tolist(),
            "sample_rows": sample_rows.tolist(),
            "sample_hash": sample_hash,
            "resource_plan": asdict(resource),
            "resource_metrics": metrics,
            "file": _file_record(view_path, hash_file=False),
        }
        _write_json(manifest_path, manifest)
        _trim_working_set()
    manifest = _read_json(manifest_path)
    matrix = np.memmap(
        Path(manifest["file"]["path"]),
        dtype=np.float32,
        mode="r",
        shape=shape,
    )
    return matrix, feature_names, manifest


def _cache_fingerprint(
    *,
    model_manifest: Mapping[str, Any],
    fold: Mapping[str, Any],
    split: str,
    max_bin: int,
    seed: int,
    feature_variant: str = DEFAULT_FEATURE_VARIANT,
    feature_count: int | None = None,
    feature_view_fingerprint: str | None = None,
) -> str:
    resolved_count = int(
        FEATURE_VARIANT_COUNTS[feature_variant]
        if feature_count is None
        else feature_count
    )
    return _stable_hash(
        {
            "schema": CACHE_SCHEMA,
            "model_input_fingerprint": model_manifest["input_fingerprint"],
            "storage_sample_hash": model_manifest["storage"]["compact"]["sample_hash"],
            "variant": feature_variant,
            "feature_count": resolved_count,
            "feature_view_fingerprint": feature_view_fingerprint,
            "fold": dict(fold),
            "split": split,
            "max_bin": int(max_bin),
            "seed": int(seed),
            "feature_pre_filter": False,
        }
    )


def _cache_valid(meta_path: Path, fingerprint: str) -> bool:
    if not meta_path.is_file():
        return False
    try:
        meta = _read_json(meta_path)
        binary = Path(meta["file"]["path"])
        return bool(
            meta.get("schema") == CACHE_SCHEMA
            and meta.get("status") == "completed"
            and meta.get("fingerprint") == fingerprint
            and binary.is_file()
            and int(binary.stat().st_size) == int(meta["file"]["size"])
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _fold_rows(
    row_index: pd.DataFrame, fold: Mapping[str, Any], split: str
) -> np.ndarray:
    dates = row_index["date_idx"].to_numpy(dtype=np.int32)
    if split == "train":
        return np.flatnonzero(dates <= int(fold["training_maximum_date_idx"])).astype(
            np.int64
        )
    if split == "validation":
        return np.flatnonzero(
            (dates >= int(fold["validation_start_date_idx"]))
            & (dates <= int(fold["validation_end_date_idx"]))
        ).astype(np.int64)
    raise FullMarketForecastError(f"unknown_dataset_split:{split}")


def _nested_inner_rows(
    train_dates: np.ndarray,
    *,
    validation_date_count: int = INNER_VALIDATION_DAYS,
    purge_days: int = COMMON_PURGE_DAYS,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    dates = np.asarray(train_dates, dtype=np.int32)
    unique_dates = np.unique(dates)
    validation_count = int(validation_date_count)
    purge_count = int(purge_days)
    validation_start_position = len(unique_dates) - validation_count
    training_end_position = validation_start_position - purge_count - 1
    if validation_count <= 0 or purge_count < 0 or training_end_position < 0:
        raise FullMarketForecastError("nested_inner_support_insufficient")
    validation_start = int(unique_dates[validation_start_position])
    validation_end = int(unique_dates[-1])
    training_maximum = int(unique_dates[training_end_position])
    training_rows = np.flatnonzero(dates <= training_maximum).astype(np.int32)
    validation_rows = np.flatnonzero(dates >= validation_start).astype(np.int32)
    if not len(training_rows) or not len(validation_rows):
        raise FullMarketForecastError("nested_inner_rows_empty")
    metadata = {
        "training_maximum_date_idx": training_maximum,
        "validation_start_date_idx": validation_start,
        "validation_end_date_idx": validation_end,
        "validation_date_count": len(np.unique(dates[validation_rows])),
        "purge_date_count": int(
            np.sum(
                (unique_dates > training_maximum) & (unique_dates < validation_start)
            )
        ),
        "training_row_count": len(training_rows),
        "validation_row_count": len(validation_rows),
    }
    if (
        metadata["validation_date_count"] != validation_count
        or metadata["purge_date_count"] != purge_count
    ):
        raise FullMarketForecastError("nested_inner_date_contract_failed")
    return training_rows, validation_rows, metadata


def build_dataset_cache(
    *,
    matrix: np.ndarray,
    rows: np.ndarray,
    feature_names: Sequence[str],
    binary_path: Path,
    meta_path: Path,
    fingerprint: str,
    max_bin: int,
    seed: int,
    resource: ResourcePlan,
    reference: lgb.Dataset | None = None,
) -> dict[str, Any]:
    if _cache_valid(meta_path, fingerprint):
        return _read_json(meta_path)
    projected = int(
        len(rows) * len(feature_names) * (1 if max_bin <= 255 else 2) * 1.35
    )
    projected += int(resource.histogram_pool_mb * (1 << 20))
    _assert_resource_capacity(resource, projected)
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(binary_path) + ".partial")
    partial.unlink(missing_ok=True)
    contiguous = bool(len(rows) and int(rows[-1]) - int(rows[0]) + 1 == len(rows))
    if contiguous:
        # All formal fold slices are contiguous in the date-sorted row-major
        # matrix. Passing the float32 memmap view directly avoids the Sequence
        # adapter's batch-local float64 conversion and its much larger native
        # construction working set.
        data: Any = matrix[int(rows[0]) : int(rows[-1]) + 1]
        adapter = "contiguous_float32_memmap"
    else:
        data = MatrixSequence(
            matrix,
            rows,
            batch_size=resource.sequence_batch_size,
        )
        adapter = "indexed_float64_sequence"
    dummy = np.zeros(len(rows), dtype=np.float32)
    params = {
        "max_bin": int(max_bin),
        "data_random_seed": int(seed),
        "feature_pre_filter": False,
        "num_threads": int(resource.cpu_threads),
        "verbosity": -1,
    }
    dataset = lgb.Dataset(
        data,
        label=dummy,
        feature_name=list(feature_names),
        reference=reference,
        free_raw_data=True,
        params=params,
    )
    _emit("dataset_cache_build_started", split=binary_path.stem, rows=len(rows))
    with ResourceMonitor() as monitor:
        dataset.construct()
        dataset.save_binary(str(partial))
        metrics = monitor.metrics()
    os.replace(partial, binary_path)
    del dataset, data, dummy
    gc.collect()
    trimmed = _trim_working_set()
    meta = {
        "schema": CACHE_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "fingerprint": fingerprint,
        "row_count": len(rows),
        "feature_count": len(feature_names),
        "max_bin": int(max_bin),
        "source_adapter": adapter,
        "working_set_trimmed": bool(trimmed),
        "resource_plan": asdict(resource),
        "resource_metrics": metrics,
        "file": _file_record(binary_path, hash_file=False),
    }
    _write_json(meta_path, meta)
    return meta


def _target_definition(target: str) -> tuple[str, str, float, str]:
    for horizon in HORIZONS:
        prefix = f"exact_net_return_d{horizon}"
        if target == prefix:
            return f"{prefix}_base", "regression", 0.0, "exact_net"
        if target == f"{prefix}_rank":
            return f"{prefix}_base", "ranking", 0.0, "exact_net"
        if target == f"{prefix}_positive":
            return f"{prefix}_base", "binary", 0.0, "exact_net"
        if target == f"{prefix}_q10":
            return f"{prefix}_base", "quantile", 0.0, "exact_net"
    if target == "next_close_up":
        source = "next_close_return"
        kind = "binary"
        binary_threshold = 0.0
    elif target == "legal_mfe_d2_ge_1pct":
        source = "legal_mfe_d2"
        kind = "binary"
        binary_threshold = TAKE_PROFIT_THRESHOLD
    elif target == "exposure_mae_d2_gt_minus_3pct":
        source = "exposure_mae_d2"
        kind = "binary"
        binary_threshold = -0.03
    elif target == "legal_exit_return_d2_gt_0p3pct":
        source = "legal_exit_return_d2"
        kind = "binary"
        binary_threshold = 0.003
    elif target.startswith("legal_exit_return_d"):
        source = target
        kind = "regression"
        binary_threshold = 0.0
    elif target.startswith("market_excess_endpoint_return_d") and target.endswith(
        "_rank"
    ):
        source = target.removesuffix("_rank")
        kind = "ranking"
        binary_threshold = 0.0
    else:
        raise FullMarketForecastError(f"unknown_training_target:{target}")
    return source, kind, binary_threshold, "path"


def _load_target_arrays(
    manifest: Mapping[str, Any], target: str
) -> tuple[np.memmap, np.memmap, int, str, float]:
    columns = list(manifest["target_columns"])
    source, kind, binary_threshold, _ = _target_definition(target)
    if source not in columns:
        raise FullMarketForecastError(f"training_target_source_missing:{source}")
    column = columns.index(source)
    values_record = manifest["files"]["values"]
    valid_record = manifest["files"]["valid"]
    values = np.memmap(
        Path(values_record["path"]),
        dtype=np.float32,
        mode="r",
        shape=tuple(int(value) for value in values_record["shape"]),
    )
    valid = np.memmap(
        Path(valid_record["path"]),
        dtype=np.uint8,
        mode="r",
        shape=tuple(int(value) for value in valid_record["shape"]),
    )
    return values, valid, column, kind, binary_threshold


def _equal_date_weights(dates: np.ndarray, valid: np.ndarray) -> np.ndarray:
    dates = np.asarray(dates, dtype=np.int32)
    valid = np.asarray(valid, dtype=bool)
    output = np.zeros(len(dates), dtype=np.float32)
    if not valid.any():
        return output
    unique, inverse, counts = np.unique(
        dates[valid], return_inverse=True, return_counts=True
    )
    del unique
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights *= len(weights) / weights.sum()
    output[np.flatnonzero(valid)] = weights.astype(np.float32)
    return output


def _date_group_sizes(dates: np.ndarray) -> np.ndarray:
    current = np.asarray(dates, dtype=np.int32)
    if current.ndim != 1 or not len(current):
        raise FullMarketForecastError("ranking_dates_empty_or_not_one_dimensional")
    if np.any(current[1:] < current[:-1]):
        raise FullMarketForecastError("ranking_dates_not_sorted")
    boundaries = np.flatnonzero(np.r_[True, current[1:] != current[:-1], True])
    return np.diff(boundaries).astype(np.int32)


def _date_relevance_labels(
    dates: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    *,
    levels: int = 10,
) -> np.ndarray:
    current_dates = np.asarray(dates, dtype=np.int32)
    current_values = np.asarray(values, dtype=np.float64)
    current_valid = np.asarray(valid, dtype=bool) & np.isfinite(current_values)
    output = np.zeros(len(current_dates), dtype=np.float32)
    boundaries = np.flatnonzero(
        np.r_[True, current_dates[1:] != current_dates[:-1], True]
    )
    for left, right in pairwise(boundaries):
        positions = left + np.flatnonzero(current_valid[left:right])
        if len(positions) < int(levels):
            continue
        order = np.argsort(current_values[positions], kind="stable")
        relevance = np.floor(
            np.arange(len(positions), dtype=np.float64) * int(levels) / len(positions)
        ).astype(np.float32)
        output[positions[order]] = relevance
    return output


def _predict_batches(
    booster: lgb.Booster,
    matrix: np.ndarray,
    rows: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    output = np.empty(len(rows), dtype=np.float32)
    for left in range(0, len(rows), int(batch_size)):
        right = min(left + int(batch_size), len(rows))
        values = np.asarray(matrix[rows[left:right]], dtype=np.float32)
        output[left:right] = np.asarray(
            booster.predict(values, num_iteration=booster.best_iteration),
            dtype=np.float32,
        )
    return output


def _daily_metrics(
    *,
    dates: np.ndarray,
    actual: np.ndarray,
    prediction: np.ndarray,
    kind: str,
    binary_threshold: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.DataFrame(
        {
            "date_idx": np.asarray(dates, dtype=np.int32),
            "actual": np.asarray(actual, dtype=np.float64),
            "prediction": np.asarray(prediction, dtype=np.float64),
        }
    ).dropna()
    frame["prediction_rank"] = frame.groupby("date_idx")["prediction"].rank(pct=True)
    frame["actual_rank"] = frame.groupby("date_idx")["actual"].rank(pct=True)
    daily = frame.groupby("date_idx").agg(
        baseline_actual=("actual", "mean"),
        row_count=("actual", "size"),
    )
    for name, fraction in (("top1pct", 0.01), ("top5pct", 0.05)):
        selected = frame[frame["prediction_rank"] >= 1.0 - fraction]
        daily = daily.join(
            selected.groupby("date_idx")["actual"].mean().rename(f"{name}_actual"),
            how="left",
        )
    rank_ic = frame.groupby("date_idx")[["prediction_rank", "actual_rank"]].corr()
    rank_ic = rank_ic.iloc[0::2, -1]
    daily["rank_ic"] = rank_ic.to_numpy(dtype=np.float64)
    metrics: dict[str, Any] = {
        "row_count": len(frame),
        "date_count": len(daily),
        "daily_rank_ic_mean": float(daily["rank_ic"].mean()),
        "daily_rank_ic_positive_fraction": float(daily["rank_ic"].gt(0.0).mean()),
        "top1pct_actual_mean": float(daily["top1pct_actual"].mean()),
        "top5pct_actual_mean": float(daily["top5pct_actual"].mean()),
        "baseline_actual_mean": float(daily["baseline_actual"].mean()),
    }
    if kind == "binary":
        binary = (frame["actual"].to_numpy() > float(binary_threshold)).astype(np.int8)
        score = frame["prediction"].to_numpy()
        metrics.update(
            {
                "auc": float(roc_auc_score(binary, score)),
                "log_loss": float(log_loss(binary, score, labels=[0, 1])),
                "accuracy_at_0p5": float(np.mean((score >= 0.5) == binary)),
            }
        )
    else:
        error = frame["prediction"].to_numpy() - frame["actual"].to_numpy()
        metrics.update(
            {
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "prediction_mean": float(frame["prediction"].mean()),
                "actual_mean": float(frame["actual"].mean()),
            }
        )
    return daily.reset_index(), metrics


def _model_parameters(
    study: Mapping[str, Any], *, kind: str, resource: ResourcePlan, profile: str
) -> dict[str, Any]:
    model = dict(study["lightgbm"])
    capacity = dict(model["profiles"][profile])
    params: dict[str, Any] = {
        "boosting_type": "gbdt",
        "device_type": "cpu",
        **capacity,
        "max_bin": int(model["max_bin"]),
        "num_threads": int(resource.cpu_threads),
        "histogram_pool_size": int(resource.histogram_pool_mb),
        "deterministic": True,
        "force_col_wise": True,
        "feature_pre_filter": False,
        "seed": int(study["validation"]["seed"]),
        "feature_fraction_seed": int(study["validation"]["seed"]),
        "bagging_seed": int(study["validation"]["seed"]),
        "data_random_seed": int(study["validation"]["seed"]),
        "verbosity": -1,
    }
    if kind == "binary":
        params.update({"objective": "binary", "metric": model["binary_metric"]})
    elif kind == "ranking":
        params.update(
            {
                "objective": "lambdarank",
                "metric": "ndcg",
                "label_gain": list(range(10)),
                "lambdarank_truncation_level": 50,
                "ndcg_eval_at": [10, 30, 100],
            }
        )
    elif kind == "quantile":
        params.update(
            {
                "objective": "quantile",
                "alpha": LOWER_QUANTILE_ALPHA,
                "metric": "quantile",
            }
        )
    else:
        params.update(
            {
                "objective": model["regression_objective"],
                "alpha": float(model["huber_alpha"]),
                "metric": "l1",
            }
        )
    return params


def _task_fingerprint(
    *,
    study_path: Path,
    target_manifest: Mapping[str, Any],
    fold: Mapping[str, Any],
    target: str,
    profile: str,
    feature_variant: str = DEFAULT_FEATURE_VARIANT,
    feature_view_fingerprint: str | None = None,
    training_mode: str = "outer_early_stop",
) -> str:
    payload = {
        "schema": TASK_SCHEMA,
        "study_sha256": _sha256(study_path),
        "target_fingerprint": target_manifest["fingerprint"],
        "fold": dict(fold),
        "target": target,
        "profile": profile,
        "feature_variant": feature_variant,
        "feature_view_fingerprint": feature_view_fingerprint,
    }
    if training_mode != "outer_early_stop":
        payload["training_mode"] = training_mode
        payload["inner_validation_date_count"] = INNER_VALIDATION_DAYS
        payload["inner_purge_days"] = COMMON_PURGE_DAYS
    return _stable_hash(payload)


def _task_id(
    *,
    fold: int,
    target: str,
    profile: str,
    feature_variant: str = DEFAULT_FEATURE_VARIANT,
    training_mode: str = "outer_early_stop",
) -> str:
    task_id = f"fold_{int(fold):02d}__{target}__{profile}"
    if feature_variant != DEFAULT_FEATURE_VARIANT:
        task_id += f"__{feature_variant}"
    if training_mode != "outer_early_stop":
        task_id += f"__{training_mode}"
    return task_id


def train_task(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    fold_number: int = 1,
    target: str = "next_close_up",
    profile: str | None = None,
    feature_variant: str = DEFAULT_FEATURE_VARIANT,
    training_mode: str = "outer_early_stop",
) -> dict[str, Any]:
    study = load_study(study_path)
    _, _, _, target_panel = _target_definition(target)
    if target_panel == "exact_net":
        target_manifest = prepare_exact_net_targets(
            study_path=study_path, output_root=output_root
        )
    else:
        target_manifest = prepare_targets(
            study_path=study_path, output_root=output_root
        )
    context = _source_context(study)
    folds = build_forward_folds(
        date_idx=context.row_index["date_idx"].to_numpy(dtype=np.int32),
        trade_date=context.row_index["trade_date"].astype(str).to_numpy(),
        validation_start_date=str(study["validation"]["validation_start_date"]),
        validation_end_date=str(study["validation"]["validation_end_date"]),
        fold_count=int(study["validation"]["forward_fold_count"]),
        purge_days=int(study["validation"]["common_purge_trading_days"]),
    )
    if not 1 <= int(fold_number) <= len(folds):
        raise FullMarketForecastError("fold_number_out_of_range")
    fold = folds[int(fold_number) - 1]
    profile_name = str(profile or study["lightgbm"]["primary_profile"])
    if profile_name not in study["lightgbm"]["profiles"]:
        raise FullMarketForecastError(f"unknown_model_profile:{profile_name}")
    if training_mode not in TRAINING_MODES:
        raise FullMarketForecastError(f"unknown_training_mode:{training_mode}")
    matrix, feature_names, feature_view = prepare_feature_view(
        study=study,
        model_manifest=context.model_manifest,
        output_root=output_root,
        feature_variant=feature_variant,
    )
    fingerprint = _task_fingerprint(
        study_path=study_path,
        target_manifest=target_manifest,
        fold=fold,
        target=target,
        profile=profile_name,
        feature_variant=feature_variant,
        feature_view_fingerprint=str(feature_view["fingerprint"]),
        training_mode=training_mode,
    )
    task_id = _task_id(
        fold=int(fold_number),
        target=target,
        profile=profile_name,
        feature_variant=feature_variant,
        training_mode=training_mode,
    )
    task_root = output_root / "tasks" / task_id
    result_path = task_root / "task_result.json"
    if result_path.is_file():
        current = _read_json(result_path)
        if (
            current.get("status") == "completed"
            and current.get("fingerprint") == fingerprint
        ):
            return current
    cache_resource = resource_plan(
        reserve_gib=float(study["resources"]["reserve_system_gib"]),
        maximum_threads=int(study["resources"]["maximum_cpu_threads"]),
        histogram_pool_cap_mb=int(study["resources"]["histogram_pool_cap_mb"]),
        sequence_batch_cap=int(study["resources"]["sequence_batch_cap"]),
    )
    seed = int(study["validation"]["seed"])
    max_bin = int(study["lightgbm"]["max_bin"])
    cache_root = output_root / "dataset_cache"
    if feature_variant != DEFAULT_FEATURE_VARIANT:
        cache_root /= feature_variant
    cache_root /= f"fold_{int(fold_number):02d}"
    train_rows = _fold_rows(context.row_index, fold, "train")
    validation_rows = _fold_rows(context.row_index, fold, "validation")
    binary_params = {
        "max_bin": max_bin,
        "data_random_seed": seed,
        "feature_pre_filter": False,
        "num_threads": cache_resource.cpu_threads,
        "verbosity": -1,
    }
    train_binary = cache_root / "train.bin"
    cache_records = {
        "train": build_dataset_cache(
            matrix=matrix,
            rows=train_rows,
            feature_names=feature_names,
            binary_path=train_binary,
            meta_path=cache_root / "train.json",
            fingerprint=_cache_fingerprint(
                model_manifest=context.model_manifest,
                fold=fold,
                split="train",
                max_bin=max_bin,
                seed=seed,
                feature_variant=feature_variant,
                feature_count=len(feature_names),
                feature_view_fingerprint=str(feature_view["fingerprint"]),
            ),
            max_bin=max_bin,
            seed=seed,
            resource=cache_resource,
        )
    }
    if training_mode == "outer_early_stop":
        # Validation bins must be derived from the training bin mapper.  A
        # binary cache built independently on validation data would create
        # incomparable feature bins even when max_bin and the seed match.
        train_reference = lgb.Dataset(
            cache_records["train"]["file"]["path"],
            params=binary_params,
            free_raw_data=True,
        ).construct()
        cache_records["validation"] = build_dataset_cache(
            matrix=matrix,
            rows=validation_rows,
            feature_names=feature_names,
            binary_path=cache_root / "validation.bin",
            meta_path=cache_root / "validation.json",
            fingerprint=_cache_fingerprint(
                model_manifest=context.model_manifest,
                fold=fold,
                split="validation",
                max_bin=max_bin,
                seed=seed,
                feature_variant=feature_variant,
                feature_count=len(feature_names),
                feature_view_fingerprint=str(feature_view["fingerprint"]),
            ),
            max_bin=max_bin,
            seed=seed,
            resource=cache_resource,
            reference=train_reference,
        )
        del train_reference
        gc.collect()
        _trim_working_set()
    values, valid, column, kind, binary_threshold = _load_target_arrays(
        target_manifest, target
    )
    train_raw = np.asarray(values[train_rows, column], dtype=np.float32)
    train_valid = np.asarray(valid[train_rows, column], dtype=bool) & np.isfinite(
        train_raw
    )
    validation_raw = np.asarray(values[validation_rows, column], dtype=np.float32)
    validation_valid = np.asarray(
        valid[validation_rows, column], dtype=bool
    ) & np.isfinite(validation_raw)
    if kind == "binary":
        train_label = (train_raw > binary_threshold).astype(np.float32)
        validation_label = (validation_raw > binary_threshold).astype(np.float32)
    elif kind == "ranking":
        source_dates = context.row_index["date_idx"].to_numpy(dtype=np.int32)
        train_label = _date_relevance_labels(
            source_dates[train_rows], train_raw, train_valid
        )
        validation_label = _date_relevance_labels(
            source_dates[validation_rows], validation_raw, validation_valid
        )
    else:
        train_label = np.where(train_valid, train_raw, 0.0).astype(np.float32)
        validation_label = np.where(validation_valid, validation_raw, 0.0).astype(
            np.float32
        )
    train_label[~train_valid] = 0.0
    validation_label[~validation_valid] = 0.0
    all_dates = context.row_index["date_idx"].to_numpy(dtype=np.int32)
    train_valid_count = int(train_valid.sum())
    validation_valid_count = int(validation_valid.sum())
    train_weight = _equal_date_weights(all_dates[train_rows], train_valid)
    validation_weight = _equal_date_weights(
        all_dates[validation_rows], validation_valid
    )
    train_set = lgb.Dataset(
        cache_records["train"]["file"]["path"],
        params=binary_params,
        free_raw_data=True,
    ).construct()
    train_set.set_label(train_label)
    train_set.set_weight(train_weight)
    validation_set: lgb.Dataset | None = None
    if training_mode == "outer_early_stop":
        validation_set = lgb.Dataset(
            cache_records["validation"]["file"]["path"],
            reference=train_set,
            params=binary_params,
            free_raw_data=True,
        ).construct()
        validation_set.set_label(validation_label)
        validation_set.set_weight(validation_weight)
        if kind == "ranking":
            train_set.set_group(_date_group_sizes(all_dates[train_rows]))
            validation_set.set_group(_date_group_sizes(all_dates[validation_rows]))
    gc.collect()
    # The two binary datasets are resident now, so this second snapshot is the
    # one that should size the histogram pool and prediction batches.
    training_resource = resource_plan(
        reserve_gib=float(study["resources"]["reserve_system_gib"]),
        maximum_threads=int(study["resources"]["maximum_cpu_threads"]),
        histogram_pool_cap_mb=int(study["resources"]["histogram_pool_cap_mb"]),
        sequence_batch_cap=int(study["resources"]["sequence_batch_cap"]),
    )
    _assert_resource_capacity(training_resource, 256 * (1 << 20))
    parameters = _model_parameters(
        study, kind=kind, resource=training_resource, profile=profile_name
    )
    _emit(
        "training_started",
        task_id=task_id,
        training_mode=training_mode,
        train_rows=len(train_rows),
        validation_rows=len(validation_rows),
        resource_plan=asdict(training_resource),
    )
    nested_selection: dict[str, Any] | None = None
    with ResourceMonitor() as monitor:
        if training_mode == "outer_early_stop":
            if validation_set is None:
                raise FullMarketForecastError("outer_validation_dataset_missing")
            booster = lgb.train(
                parameters,
                train_set,
                num_boost_round=int(study["lightgbm"]["num_boost_round"]),
                valid_sets=[validation_set],
                valid_names=["forward_validation"],
                callbacks=[
                    lgb.early_stopping(
                        int(study["lightgbm"]["early_stopping_rounds"]),
                        verbose=True,
                    ),
                    lgb.log_evaluation(period=25),
                ],
            )
            selected_iteration = int(
                booster.best_iteration or booster.current_iteration()
            )
        else:
            outer_train_dates = all_dates[train_rows]
            inner_train_rows, inner_validation_rows, inner_metadata = (
                _nested_inner_rows(
                    outer_train_dates,
                    validation_date_count=int(
                        study["validation"]["inner_validation_trading_days"]
                    ),
                    purge_days=int(study["validation"]["common_purge_trading_days"]),
                )
            )
            inner_train_set = train_set.subset(inner_train_rows).construct()
            inner_validation_set = train_set.subset(inner_validation_rows).construct()
            inner_train_set.set_label(train_label[inner_train_rows])
            inner_validation_set.set_label(train_label[inner_validation_rows])
            inner_train_set.set_weight(
                _equal_date_weights(
                    outer_train_dates[inner_train_rows],
                    train_valid[inner_train_rows],
                )
            )
            inner_validation_set.set_weight(
                _equal_date_weights(
                    outer_train_dates[inner_validation_rows],
                    train_valid[inner_validation_rows],
                )
            )
            if kind == "ranking":
                inner_train_set.set_group(
                    _date_group_sizes(outer_train_dates[inner_train_rows])
                )
                inner_validation_set.set_group(
                    _date_group_sizes(outer_train_dates[inner_validation_rows])
                )
            inner_booster = lgb.train(
                parameters,
                inner_train_set,
                num_boost_round=int(study["lightgbm"]["num_boost_round"]),
                valid_sets=[inner_validation_set],
                valid_names=["inner_forward_validation"],
                callbacks=[
                    lgb.early_stopping(
                        int(study["lightgbm"]["early_stopping_rounds"]),
                        verbose=True,
                    ),
                    lgb.log_evaluation(period=25),
                ],
            )
            selected_iteration = max(
                1,
                int(inner_booster.best_iteration or inner_booster.current_iteration()),
            )
            nested_selection = {
                **inner_metadata,
                "selected_iteration": selected_iteration,
                "outer_validation_used_for_iteration_selection": False,
            }
            del inner_booster, inner_train_set, inner_validation_set, train_set
            gc.collect()
            _trim_working_set()
            train_set = lgb.Dataset(
                cache_records["train"]["file"]["path"],
                params=binary_params,
                free_raw_data=True,
            ).construct()
            train_set.set_label(train_label)
            train_set.set_weight(train_weight)
            if kind == "ranking":
                train_set.set_group(_date_group_sizes(outer_train_dates))
            booster = lgb.train(
                parameters,
                train_set,
                num_boost_round=selected_iteration,
                callbacks=[lgb.log_evaluation(period=25)],
            )
        prediction = _predict_batches(
            booster,
            matrix,
            validation_rows,
            batch_size=training_resource.sequence_batch_size,
        )
        resource_metrics = monitor.metrics()
    final_iteration = int(booster.current_iteration())
    if final_iteration != selected_iteration:
        raise FullMarketForecastError("final_iteration_contract_failed")
    daily, metrics = _daily_metrics(
        dates=all_dates[validation_rows][validation_valid],
        actual=validation_raw[validation_valid],
        prediction=prediction[validation_valid],
        kind=kind,
        binary_threshold=binary_threshold,
    )
    task_root.mkdir(parents=True, exist_ok=True)
    model_path = task_root / "model.txt"
    model_partial = model_path.with_suffix(".txt.partial")
    booster.save_model(str(model_partial), num_iteration=final_iteration)
    os.replace(model_partial, model_path)
    prediction_path = task_root / "prediction.npy"
    prediction_partial = prediction_path.with_suffix(".npy.partial")
    with prediction_partial.open("wb") as stream:
        np.save(stream, prediction, allow_pickle=False)
    os.replace(prediction_partial, prediction_path)
    daily_path = task_root / "daily_metrics.parquet"
    _write_parquet(daily, daily_path)
    importance = pd.DataFrame(
        {
            "feature_name": feature_names,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain", "split"], ascending=False)
    importance_path = task_root / "feature_importance.parquet"
    _write_parquet(importance, importance_path)
    result = {
        "schema": TASK_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "task_id": task_id,
        "fingerprint": fingerprint,
        "target": target,
        "target_panel": target_panel,
        "kind": kind,
        "binary_threshold": binary_threshold if kind == "binary" else None,
        "profile": profile_name,
        "training_mode": training_mode,
        "fold": fold,
        "feature_variant": feature_variant,
        "feature_count": len(feature_names),
        "feature_view": feature_view,
        "best_iteration": final_iteration,
        "iteration_selection": nested_selection,
        "train_valid_count": train_valid_count,
        "validation_valid_count": validation_valid_count,
        "metrics": metrics,
        "parameters": parameters,
        "resource_plan": asdict(training_resource),
        "resource_metrics": resource_metrics,
        "dataset_cache": cache_records,
        "forbidden_2026_read_count": 0,
        "decision_boundary": {
            "outer_validation_used_for_iteration_selection": (
                training_mode == "outer_early_stop"
            ),
            "outer_validation_used_for_training": False,
        },
        "files": {
            "model": _file_record(model_path),
            "prediction": _file_record(
                prediction_path,
                shape=[len(prediction)],
                dtype="float32",
            ),
            "daily_metrics": _file_record(daily_path, row_count=len(daily)),
            "feature_importance": _file_record(
                importance_path, row_count=len(importance)
            ),
        },
    }
    _write_json(result_path, result)
    _emit(
        "training_completed",
        task_id=task_id,
        best_iteration=final_iteration,
        metrics=metrics,
        elapsed_seconds=resource_metrics["elapsed_seconds"],
    )
    return result


def _exact_unit_net_return(
    *,
    signal_idx: int,
    symbol_idx: int,
    legal_gross_return: float,
    fill_day: int,
    daily_raw: np.ndarray,
    raw_open: np.ndarray,
    date_values: np.ndarray,
    costs: ExecutionCosts,
    notional_cny: float,
    slippage_multiplier: float,
    entry_filled: np.ndarray | None = None,
) -> tuple[float, bool]:
    if entry_filled is not None and not bool(
        entry_filled[int(signal_idx), int(symbol_idx)]
    ):
        return 0.0, False
    if (
        not math.isfinite(legal_gross_return)
        or int(fill_day) < 2
        or int(signal_idx) + 1 >= len(date_values)
        or int(signal_idx) + int(fill_day) >= len(date_values)
    ):
        return 0.0, False
    entry_idx = int(signal_idx) + 1
    adjusted_entry = float(daily_raw[entry_idx, int(symbol_idx), 0])
    entry_raw = float(raw_open[entry_idx, int(symbol_idx)])
    position, _ = economic._buy_position(
        available_cash=float(notional_cny),
        allocated_cash=float(notional_cny),
        symbol_idx=int(symbol_idx),
        signal_date_idx=int(signal_idx),
        execution_date_idx=entry_idx,
        raw_open=entry_raw,
        adjusted_open=adjusted_entry,
        costs=costs,
        slippage_multiplier=float(slippage_multiplier),
    )
    if position is None:
        return 0.0, False
    residual_cash = float(notional_cny) - float(position.net_cash_outflow)
    if legal_gross_return <= -1.0:
        proceeds = 0.0
    else:
        adjusted_exit = adjusted_entry * (1.0 + float(legal_gross_return))
        exit_idx = int(signal_idx) + int(fill_day)
        proceeds, _ = economic._sell_position(
            position=position,
            adjusted_open=adjusted_exit,
            trade_date=str(date_values[exit_idx]),
            costs=costs,
            slippage_multiplier=float(slippage_multiplier),
        )
    return float((residual_cash + proceeds) / float(notional_cny) - 1.0), True


def _newey_west_interval(
    values: np.ndarray, *, lag: int = 20, z_value: float = 1.96
) -> dict[str, float | int | None]:
    current = np.asarray(values, dtype=np.float64)
    current = current[np.isfinite(current)]
    count = len(current)
    if count < 2:
        return {
            "count": count,
            "mean": float(current.mean()) if count else None,
            "standard_error": None,
            "lower": None,
            "upper": None,
        }
    demeaned = current - current.mean()
    maximum_lag = min(int(lag), count - 1)
    long_run = float(np.dot(demeaned, demeaned) / count)
    for offset in range(1, maximum_lag + 1):
        covariance = float(np.dot(demeaned[offset:], demeaned[:-offset]) / count)
        long_run += 2.0 * (1.0 - offset / (maximum_lag + 1.0)) * covariance
    standard_error = math.sqrt(max(long_run, 0.0) / count)
    mean = float(current.mean())
    return {
        "count": count,
        "mean": mean,
        "standard_error": standard_error,
        "lower": mean - float(z_value) * standard_error,
        "upper": mean + float(z_value) * standard_error,
    }


def _within_date_rank(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("date_idx", sort=False)[column].rank(pct=True)


def _oof_task_result(
    output_root: Path,
    *,
    fold: int,
    target: str,
    profile: str,
    feature_variant: str = DEFAULT_FEATURE_VARIANT,
    training_mode: str = "outer_early_stop",
) -> tuple[dict[str, Any], Path]:
    task_id = _task_id(
        fold=fold,
        target=target,
        profile=profile,
        feature_variant=feature_variant,
        training_mode=training_mode,
    )
    path = output_root / "tasks" / task_id / "task_result.json"
    result = _read_json(path)
    if result.get("status") != "completed" or result.get("task_id") != task_id:
        raise FullMarketForecastError(f"required_oof_task_incomplete:{task_id}")
    return result, path


def _load_ensemble_prediction(
    task_records: Mapping[str, Mapping[str, Any]],
    *,
    fold_number: int,
    target: str,
    model_profiles: Sequence[str],
    row_mask: np.ndarray,
) -> np.ndarray:
    predictions = [
        np.asarray(
            np.load(
                task_records[f"{fold_number}:{target}:{model_profile}"]["files"][
                    "prediction"
                ]["path"],
                mmap_mode="r",
                allow_pickle=False,
            )[row_mask],
            dtype=np.float32,
        )
        for model_profile in model_profiles
    ]
    if len(predictions) == 1:
        return predictions[0]
    return np.mean(np.stack(predictions, axis=0), axis=0, dtype=np.float32)


def _payoff_summary(daily: pd.DataFrame, decile_daily: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "date_count": int(daily["date_idx"].nunique()),
        "daily_rank_ic_mean": float(daily["rank_ic"].mean()),
        "daily_rank_ic_positive_fraction": float(daily["rank_ic"].gt(0.0).mean()),
    }
    years = daily["trade_date"].astype(str).str[:4]
    for top_k in (1, 3, 10):
        for scenario in EXACT_NET_SCENARIOS:
            column = f"top{top_k}_{scenario}"
            values = daily[column].to_numpy(dtype=np.float64)
            interval = _newey_west_interval(values, lag=20)
            annual = daily.assign(year=years).groupby("year")[column].mean()
            summary[column] = {
                **interval,
                "positive_date_fraction": float(np.mean(values > 0.0)),
                "positive_year_count": int(annual.gt(0.0).sum()),
                "year_count": len(annual),
                "worst_year_mean": float(annual.min()),
            }
        for scenario in EXACT_NET_SCENARIOS:
            difference = (
                daily[f"top{top_k}_{scenario}"] - daily[f"baseline_{scenario}"]
            ).to_numpy(dtype=np.float64)
            summary[f"top{top_k}_{scenario}_minus_baseline"] = _newey_west_interval(
                difference, lag=20
            )

    deciles = (
        decile_daily.groupby("decile", sort=True)[["base", "stress"]]
        .mean()
        .reset_index()
    )
    base_means = deciles["base"].to_numpy(dtype=np.float64)
    stress_means = deciles["stress"].to_numpy(dtype=np.float64)
    ordinal = deciles["decile"].to_numpy(dtype=np.float64)
    summary["deciles"] = deciles.to_dict("records")
    summary["decile_base_spearman"] = float(
        pd.Series(ordinal).corr(pd.Series(base_means), method="spearman")
    )
    summary["decile_stress_spearman"] = float(
        pd.Series(ordinal).corr(pd.Series(stress_means), method="spearman")
    )
    summary["decile_base_adjacent_increase_count"] = int(
        np.sum(np.diff(base_means) > 0.0)
    )
    summary["decile_stress_adjacent_increase_count"] = int(
        np.sum(np.diff(stress_means) > 0.0)
    )
    pivot = decile_daily.pivot(
        index="date_idx", columns="decile", values=["base", "stress"]
    )
    for scenario in EXACT_NET_SCENARIOS:
        difference = pivot[(scenario, 10)].to_numpy(dtype=np.float64) - pivot[
            (scenario, 1)
        ].to_numpy(dtype=np.float64)
        summary[f"top_decile_minus_bottom_{scenario}"] = _newey_west_interval(
            difference, lag=20
        )
    return summary


def evaluate_payoff_oof(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    target: str = "exact_net_return_d5_rank",
    profile: str | None = None,
    feature_variant: str = DEFAULT_FEATURE_VARIANT,
    training_mode: str = "causal_nested",
) -> dict[str, Any]:
    study = load_study(study_path)
    source, _, _, panel = _target_definition(target)
    if panel != "exact_net" or not source.endswith("_base"):
        raise FullMarketForecastError("payoff_evaluation_requires_exact_net_target")
    if training_mode not in TRAINING_MODES:
        raise FullMarketForecastError(f"unknown_training_mode:{training_mode}")
    profile_name = str(profile or study["lightgbm"]["primary_profile"])
    context = _source_context(study)
    target_manifest = prepare_exact_net_targets(
        study_path=study_path, output_root=output_root
    )
    folds = build_forward_folds(
        date_idx=context.row_index["date_idx"].to_numpy(dtype=np.int32),
        trade_date=context.row_index["trade_date"].astype(str).to_numpy(),
        validation_start_date=str(study["validation"]["validation_start_date"]),
        validation_end_date=str(study["validation"]["validation_end_date"]),
        fold_count=int(study["validation"]["forward_fold_count"]),
        purge_days=int(study["validation"]["common_purge_trading_days"]),
    )
    task_records: list[tuple[dict[str, Any], Path]] = [
        _oof_task_result(
            output_root,
            fold=int(fold["fold"]),
            target=target,
            profile=profile_name,
            feature_variant=feature_variant,
            training_mode=training_mode,
        )
        for fold in folds
    ]
    fingerprint = _stable_hash(
        {
            "schema": PAYOFF_EVALUATION_SCHEMA,
            "study_sha256": _sha256(study_path),
            "target_fingerprint": target_manifest["fingerprint"],
            "target": target,
            "profile": profile_name,
            "feature_variant": feature_variant,
            "training_mode": training_mode,
            "tasks": {str(path): _sha256(path) for _, path in task_records},
            "top_k": [1, 3, 10],
            "deciles": 10,
            "primary_weighting": "equal_signal_date",
        }
    )
    evaluation_name = f"{target}__{profile_name}"
    if feature_variant != DEFAULT_FEATURE_VARIANT:
        evaluation_name += f"__{feature_variant}"
    if training_mode != "outer_early_stop":
        evaluation_name += f"__{training_mode}"
    evaluation_root = output_root / "payoff_evaluation" / evaluation_name
    manifest_path = evaluation_root / "manifest.json"
    if manifest_path.is_file():
        current = _read_json(manifest_path)
        if (
            current.get("status") == "completed"
            and current.get("fingerprint") == fingerprint
        ):
            return current

    columns = list(target_manifest["target_columns"])
    base_column = columns.index(source)
    stress_column = columns.index(source.removesuffix("_base") + "_stress")
    values_record = target_manifest["files"]["values"]
    valid_record = target_manifest["files"]["valid"]
    values = np.memmap(
        Path(values_record["path"]),
        dtype=np.float32,
        mode="r",
        shape=tuple(int(value) for value in values_record["shape"]),
    )
    valid = np.memmap(
        Path(valid_record["path"]),
        dtype=np.uint8,
        mode="r",
        shape=tuple(int(value) for value in valid_record["shape"]),
    )
    daily_parts: list[pd.DataFrame] = []
    decile_parts: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    _emit("payoff_oof_evaluation_started", target=target, folds=len(folds))
    for fold, (task, _) in zip(folds, task_records, strict=True):
        fold_number = int(fold["fold"])
        positions = _fold_rows(context.row_index, fold, "validation")
        prediction = np.asarray(
            np.load(
                task["files"]["prediction"]["path"],
                mmap_mode="r",
                allow_pickle=False,
            ),
            dtype=np.float32,
        )
        if len(prediction) != len(positions):
            raise FullMarketForecastError("payoff_prediction_row_count_mismatch")
        usable = (
            np.asarray(valid[positions, base_column], dtype=bool)
            & np.asarray(valid[positions, stress_column], dtype=bool)
            & np.isfinite(prediction)
        )
        current_positions = positions[usable]
        frame = context.row_index.iloc[current_positions][
            [
                "candidate_id",
                "date_idx",
                "trade_date",
                "symbol",
                "symbol_idx",
            ]
        ].copy()
        frame["model_row_position"] = current_positions
        frame["fold"] = fold_number
        frame["prediction"] = prediction[usable]
        frame["base"] = np.asarray(
            values[current_positions, base_column], dtype=np.float32
        )
        frame["stress"] = np.asarray(
            values[current_positions, stress_column], dtype=np.float32
        )
        frame = frame.sort_values(
            ["date_idx", "prediction", "candidate_id"], kind="stable"
        ).reset_index(drop=True)
        frame["score_position"] = frame.groupby("date_idx", sort=False).cumcount()
        frame["date_size"] = frame.groupby("date_idx", sort=False)[
            "candidate_id"
        ].transform("size")
        frame["decile"] = np.minimum(
            10,
            np.floor(
                frame["score_position"].to_numpy(dtype=np.float64)
                * 10.0
                / frame["date_size"].to_numpy(dtype=np.float64)
            ).astype(np.int8)
            + 1,
        )
        frame["selection_rank"] = (frame["date_size"] - frame["score_position"]).astype(
            np.int32
        )
        baseline = frame.groupby(["date_idx", "trade_date"], sort=False)[
            ["base", "stress"]
        ].mean()
        baseline = baseline.rename(
            columns={"base": "baseline_base", "stress": "baseline_stress"}
        )
        fold_daily = baseline.reset_index()
        for top_k in (1, 3, 10):
            selected = frame.loc[frame["selection_rank"] <= top_k]
            means = selected.groupby("date_idx", sort=False)[["base", "stress"]].mean()
            means = means.rename(
                columns={
                    "base": f"top{top_k}_base",
                    "stress": f"top{top_k}_stress",
                }
            )
            fold_daily = fold_daily.merge(
                means.reset_index(), on="date_idx", how="left", validate="one_to_one"
            )
        task_daily = pd.read_parquet(task["files"]["daily_metrics"]["path"])[
            ["date_idx", "rank_ic"]
        ]
        fold_daily = fold_daily.merge(
            task_daily, on="date_idx", how="left", validate="one_to_one"
        )
        fold_daily["fold"] = fold_number
        fold_deciles = (
            frame.groupby(["date_idx", "trade_date", "decile"], sort=False)[
                ["base", "stress"]
            ]
            .mean()
            .reset_index()
        )
        fold_deciles["fold"] = fold_number
        fold_selection = frame.loc[frame["selection_rank"] <= 10].copy()
        fold_summary = _payoff_summary(fold_daily, fold_deciles)
        fold_summary["fold"] = fold_number
        fold_summaries.append(fold_summary)
        daily_parts.append(fold_daily)
        decile_parts.append(fold_deciles)
        selection_parts.append(fold_selection)
        _emit(
            "payoff_oof_fold_completed",
            fold=fold_number,
            top1_base=fold_summary["top1_base"]["mean"],
            top1_stress=fold_summary["top1_stress"]["mean"],
            rank_ic=fold_summary["daily_rank_ic_mean"],
        )
        del frame
        gc.collect()

    daily = pd.concat(daily_parts, ignore_index=True).sort_values("date_idx")
    decile_daily = pd.concat(decile_parts, ignore_index=True).sort_values(
        ["date_idx", "decile"]
    )
    selections = pd.concat(selection_parts, ignore_index=True).sort_values(
        ["date_idx", "selection_rank"]
    )
    combined = _payoff_summary(daily, decile_daily)
    rank_positive_folds = sum(
        float(item["daily_rank_ic_mean"]) > 0.0 for item in fold_summaries
    )
    top1_improvement_folds = sum(
        float(item["top1_base_minus_baseline"]["mean"]) > 0.0 for item in fold_summaries
    )
    top_decile_improvement_folds = sum(
        float(item["top_decile_minus_bottom_base"]["mean"]) > 0.0
        for item in fold_summaries
    )
    monotonicity_gate = {
        "rank_ic_positive_fold_count": int(rank_positive_folds),
        "top1_above_baseline_fold_count": int(top1_improvement_folds),
        "top_decile_above_bottom_fold_count": int(top_decile_improvement_folds),
        "combined_base_decile_spearman": combined["decile_base_spearman"],
        "passed": bool(
            rank_positive_folds == len(folds)
            and top1_improvement_folds >= 4
            and top_decile_improvement_folds >= 4
            and float(combined["decile_base_spearman"]) >= 0.80
        ),
    }
    eligible_top_k = [
        top_k
        for top_k in (1, 3, 10)
        if combined[f"top{top_k}_stress"]["lower"] is not None
        and float(combined[f"top{top_k}_stress"]["lower"]) > 0.0
        and int(combined[f"top{top_k}_stress"]["positive_year_count"]) >= 5
    ]
    account_replay_gate = {
        "requires_positive_stress_hac_lower_bound": True,
        "requires_at_least_five_of_six_positive_years": True,
        "eligible_top_k": eligible_top_k,
        "passed": bool(monotonicity_gate["passed"] and eligible_top_k),
    }
    daily_path = evaluation_root / "daily_metrics.parquet"
    decile_path = evaluation_root / "decile_daily.parquet"
    selections_path = evaluation_root / "top10_selections.parquet"
    _write_parquet(daily, daily_path)
    _write_parquet(decile_daily, decile_path)
    _write_parquet(selections, selections_path)
    manifest = {
        "schema": PAYOFF_EVALUATION_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "fingerprint": fingerprint,
        "target": target,
        "profile": profile_name,
        "feature_variant": feature_variant,
        "training_mode": training_mode,
        "fold_summaries": fold_summaries,
        "combined": combined,
        "monotonicity_gate": monotonicity_gate,
        "account_replay_gate": account_replay_gate,
        "forbidden_2026_read_count": 0,
        "files": {
            "daily_metrics": _file_record(daily_path, row_count=len(daily)),
            "decile_daily": _file_record(decile_path, row_count=len(decile_daily)),
            "top10_selections": _file_record(
                selections_path, row_count=len(selections)
            ),
        },
        "sources": {
            "exact_net_targets": _file_record(
                output_root / "exact_net_targets/manifest.json",
                target_fingerprint=target_manifest["fingerprint"],
            ),
            "tasks": [_file_record(path) for _, path in task_records],
        },
    }
    _write_json(manifest_path, manifest)
    _emit(
        "payoff_oof_evaluation_completed",
        target=target,
        monotonicity_passed=monotonicity_gate["passed"],
        account_replay_passed=account_replay_gate["passed"],
    )
    return manifest


def _market_gate_metrics(daily: pd.DataFrame) -> dict[str, Any]:
    stress = daily["stress"].to_numpy(dtype=np.float64)
    base = daily["base"].to_numpy(dtype=np.float64)
    annual = (
        daily.assign(year=daily["trade_date"].astype(str).str[:4])
        .groupby("year")[["base", "stress"]]
        .sum()
    )
    fold_means = daily.groupby("fold")[["base", "stress"]].mean()
    return {
        "date_count": len(daily),
        "trade_date_count": int(daily["trade_count"].gt(0).sum()),
        "trade_date_fraction": float(daily["trade_count"].gt(0).mean()),
        "trade_count": int(daily["trade_count"].sum()),
        "base": _newey_west_interval(base, lag=20),
        "stress": _newey_west_interval(stress, lag=20),
        "positive_stress_year_count": int(annual["stress"].gt(0.0).sum()),
        "year_count": len(annual),
        "positive_stress_fold_count": int(fold_means["stress"].gt(0.0).sum()),
        "fold_count": len(fold_means),
        "annual_sum": annual.reset_index().to_dict("records"),
        "fold_means": fold_means.reset_index().to_dict("records"),
    }


def evaluate_market_regime_oof(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    horizon: int = 5,
    ranking_target: str = "exact_net_return_d5_rank",
    profile: str | None = None,
    feature_variant: str = DEFAULT_FEATURE_VARIANT,
    training_mode: str = "causal_nested",
) -> dict[str, Any]:
    if int(horizon) != 5 or ranking_target != "exact_net_return_d5_rank":
        raise FullMarketForecastError("market_regime_v1_is_frozen_to_d5")
    study = load_study(study_path)
    if training_mode != "causal_nested":
        raise FullMarketForecastError("market_regime_requires_causal_nested_ranking")
    profile_name = str(profile or study["lightgbm"]["primary_profile"])
    context = _source_context(study)
    target_manifest = prepare_exact_net_targets(
        study_path=study_path, output_root=output_root
    )
    folds = build_forward_folds(
        date_idx=context.row_index["date_idx"].to_numpy(dtype=np.int32),
        trade_date=context.row_index["trade_date"].astype(str).to_numpy(),
        validation_start_date=str(study["validation"]["validation_start_date"]),
        validation_end_date=str(study["validation"]["validation_end_date"]),
        fold_count=int(study["validation"]["forward_fold_count"]),
        purge_days=int(study["validation"]["common_purge_trading_days"]),
    )
    task_records = [
        _oof_task_result(
            output_root,
            fold=int(fold["fold"]),
            target=ranking_target,
            profile=profile_name,
            feature_variant=feature_variant,
            training_mode=training_mode,
        )
        for fold in folds
    ]
    ridge_alphas = (0.1, 1.0, 10.0, 100.0, 1000.0)
    logistic_cs = (0.001, 0.01, 0.1, 1.0)
    market_features = tuple(
        record["feature_name"]
        for record in context.model_manifest["features"]
        if record["analytic_family"] == "market_state"
    )
    if len(market_features) != 54:
        raise FullMarketForecastError("market_state_feature_count_changed")
    fingerprint = _stable_hash(
        {
            "schema": MARKET_REGIME_SCHEMA,
            "study_sha256": _sha256(study_path),
            "target_fingerprint": target_manifest["fingerprint"],
            "ranking_target": ranking_target,
            "profile": profile_name,
            "feature_variant": feature_variant,
            "training_mode": training_mode,
            "market_features": market_features,
            "ridge_alphas": ridge_alphas,
            "logistic_cs": logistic_cs,
            "gates": {
                "ridge": "predicted_exact_net_return_above_zero",
                "logistic": "predicted_positive_probability_above_0p5",
                "consensus": "ridge_and_logistic",
            },
            "top_k": [1, 3, 10],
            "tasks": {str(path): _sha256(path) for _, path in task_records},
        }
    )
    output_path = output_root / "market_regime_evaluation/manifest.json"
    if output_path.is_file():
        current = _read_json(output_path)
        if (
            current.get("status") == "completed"
            and current.get("fingerprint") == fingerprint
        ):
            return current

    row_dates = context.row_index["date_idx"].to_numpy(dtype=np.int32)
    boundaries = np.flatnonzero(np.r_[True, row_dates[1:] != row_dates[:-1], True])
    first_rows = boundaries[:-1]
    last_rows = boundaries[1:] - 1
    unique_dates = row_dates[first_rows]
    unique_trade_dates = (
        context.row_index["trade_date"].astype(str).to_numpy()[first_rows]
    )
    feature_records = [
        record
        for record in context.model_manifest["features"]
        if record["analytic_family"] == "market_state"
    ]
    feature_columns = np.asarray(
        [int(record["column_index"]) for record in feature_records], dtype=np.int32
    )
    source_matrix = _open_array(
        context.model_manifest["storage"]["compact"], dtype=np.float32
    )
    market_matrix = np.asarray(
        source_matrix[np.ix_(first_rows, feature_columns)], dtype=np.float64
    )
    market_matrix_last = np.asarray(
        source_matrix[np.ix_(last_rows, feature_columns)], dtype=np.float64
    )
    if not np.allclose(
        market_matrix, market_matrix_last, rtol=0.0, atol=0.0, equal_nan=True
    ):
        raise FullMarketForecastError("market_features_are_not_date_invariant")

    columns = list(target_manifest["target_columns"])
    base_column = columns.index("exact_net_return_d5_base")
    stress_column = columns.index("exact_net_return_d5_stress")
    values_record = target_manifest["files"]["values"]
    valid_record = target_manifest["files"]["valid"]
    target_values = np.memmap(
        Path(values_record["path"]),
        dtype=np.float32,
        mode="r",
        shape=tuple(int(value) for value in values_record["shape"]),
    )
    target_valid = np.memmap(
        Path(valid_record["path"]),
        dtype=np.uint8,
        mode="r",
        shape=tuple(int(value) for value in valid_record["shape"]),
    )
    market_base = np.full(len(unique_dates), np.nan, dtype=np.float64)
    market_stress = np.full(len(unique_dates), np.nan, dtype=np.float64)
    for date_position, (left, right) in enumerate(pairwise(boundaries)):
        usable = np.asarray(
            target_valid[left:right, base_column], dtype=bool
        ) & np.asarray(target_valid[left:right, stress_column], dtype=bool)
        if usable.any():
            market_base[date_position] = float(
                np.asarray(target_values[left:right, base_column], dtype=np.float64)[
                    usable
                ].mean()
            )
            market_stress[date_position] = float(
                np.asarray(target_values[left:right, stress_column], dtype=np.float64)[
                    usable
                ].mean()
            )

    prediction_parts: list[pd.DataFrame] = []
    coefficient_parts: list[pd.DataFrame] = []
    fit_records: list[dict[str, Any]] = []
    for fold in folds:
        train_rows = np.flatnonzero(
            (unique_dates <= int(fold["training_maximum_date_idx"]))
            & np.isfinite(market_base)
        )
        validation_rows = np.flatnonzero(
            (unique_dates >= int(fold["validation_start_date_idx"]))
            & (unique_dates <= int(fold["validation_end_date_idx"]))
            & np.isfinite(market_base)
        )
        inner_train, inner_validation, inner_meta = _nested_inner_rows(
            unique_dates[train_rows]
        )
        ridge_choices: list[tuple[float, float]] = []
        for alpha in ridge_alphas:
            model = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                RobustScaler(),
                Ridge(alpha=float(alpha)),
            )
            model.fit(
                market_matrix[train_rows[inner_train]],
                market_base[train_rows[inner_train]],
            )
            prediction = model.predict(market_matrix[train_rows[inner_validation]])
            ridge_choices.append(
                (
                    mean_absolute_error(
                        market_base[train_rows[inner_validation]], prediction
                    ),
                    float(alpha),
                )
            )
        ridge_score, ridge_alpha = min(ridge_choices)
        logistic_choices: list[tuple[float, float]] = []
        for regularization in logistic_cs:
            model = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                RobustScaler(),
                LogisticRegression(C=float(regularization), max_iter=2000),
            )
            model.fit(
                market_matrix[train_rows[inner_train]],
                (market_base[train_rows[inner_train]] > 0.0).astype(np.int8),
            )
            probability = model.predict_proba(
                market_matrix[train_rows[inner_validation]]
            )[:, 1]
            logistic_choices.append(
                (
                    log_loss(
                        (market_base[train_rows[inner_validation]] > 0.0).astype(
                            np.int8
                        ),
                        probability,
                        labels=[0, 1],
                    ),
                    float(regularization),
                )
            )
        logistic_score, logistic_c = min(logistic_choices)
        ridge_model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RobustScaler(),
            Ridge(alpha=ridge_alpha),
        )
        logistic_model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RobustScaler(),
            LogisticRegression(C=logistic_c, max_iter=2000),
        )
        ridge_model.fit(market_matrix[train_rows], market_base[train_rows])
        logistic_model.fit(
            market_matrix[train_rows],
            (market_base[train_rows] > 0.0).astype(np.int8),
        )
        ridge_prediction = ridge_model.predict(market_matrix[validation_rows])
        positive_probability = logistic_model.predict_proba(
            market_matrix[validation_rows]
        )[:, 1]
        prediction_parts.append(
            pd.DataFrame(
                {
                    "date_idx": unique_dates[validation_rows],
                    "trade_date": unique_trade_dates[validation_rows],
                    "fold": int(fold["fold"]),
                    "actual_base": market_base[validation_rows],
                    "actual_stress": market_stress[validation_rows],
                    "ridge_prediction": ridge_prediction,
                    "positive_probability": positive_probability,
                    "ridge_gate": ridge_prediction > 0.0,
                    "logistic_gate": positive_probability > 0.5,
                    "consensus_gate": (ridge_prediction > 0.0)
                    & (positive_probability > 0.5),
                }
            )
        )
        for model_name, model in (
            ("ridge", ridge_model),
            ("logistic", logistic_model),
        ):
            transformed_names = model[:-1].get_feature_names_out(market_features)
            coefficients = np.asarray(model[-1].coef_, dtype=np.float64).reshape(-1)
            coefficient_parts.append(
                pd.DataFrame(
                    {
                        "fold": int(fold["fold"]),
                        "model": model_name,
                        "feature_name": transformed_names,
                        "coefficient": coefficients,
                    }
                )
            )
        fit_records.append(
            {
                "fold": int(fold["fold"]),
                "ridge_alpha": ridge_alpha,
                "ridge_inner_mae": ridge_score,
                "logistic_c": logistic_c,
                "logistic_inner_log_loss": logistic_score,
                "ridge_validation_correlation": float(
                    np.corrcoef(ridge_prediction, market_base[validation_rows])[0, 1]
                ),
                "ridge_gate_fraction": float(np.mean(ridge_prediction > 0.0)),
                "logistic_gate_fraction": float(np.mean(positive_probability > 0.5)),
                "consensus_gate_fraction": float(
                    np.mean((ridge_prediction > 0.0) & (positive_probability > 0.5))
                ),
                "inner_split": inner_meta,
            }
        )

    market_predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        "date_idx"
    )
    coefficients = pd.concat(coefficient_parts, ignore_index=True)
    selection_parts: list[pd.DataFrame] = []
    for fold, (task, _) in zip(folds, task_records, strict=True):
        positions = _fold_rows(context.row_index, fold, "validation")
        prediction = np.asarray(
            np.load(
                task["files"]["prediction"]["path"],
                mmap_mode="r",
                allow_pickle=False,
            ),
            dtype=np.float32,
        )
        usable = (
            np.asarray(target_valid[positions, base_column], dtype=bool)
            & np.asarray(target_valid[positions, stress_column], dtype=bool)
            & np.isfinite(prediction)
        )
        current_positions = positions[usable]
        frame = context.row_index.iloc[current_positions][
            ["candidate_id", "date_idx", "trade_date", "symbol", "symbol_idx"]
        ].copy()
        frame["model_row_position"] = current_positions
        frame["fold"] = int(fold["fold"])
        frame["ranking_score"] = prediction[usable]
        frame["base"] = np.asarray(
            target_values[current_positions, base_column], dtype=np.float32
        )
        frame["stress"] = np.asarray(
            target_values[current_positions, stress_column], dtype=np.float32
        )
        frame = frame.sort_values(
            ["date_idx", "ranking_score", "candidate_id"], kind="stable"
        )
        frame["selection_rank"] = (
            frame.groupby("date_idx", sort=False).cumcount(ascending=False) + 1
        )
        selection_parts.append(frame.loc[frame["selection_rank"] <= 10])
    selections = pd.concat(selection_parts, ignore_index=True).merge(
        market_predictions[
            [
                "date_idx",
                "ridge_prediction",
                "positive_probability",
                "ridge_gate",
                "logistic_gate",
                "consensus_gate",
            ]
        ],
        on="date_idx",
        how="left",
        validate="many_to_one",
    )
    daily_parts: list[pd.DataFrame] = []
    summaries: dict[str, Any] = {}
    for gate_name in ("ridge", "logistic", "consensus"):
        gate_column = f"{gate_name}_gate"
        for top_k in (1, 3, 10):
            selected = selections.loc[
                selections[gate_column] & (selections["selection_rank"] <= int(top_k))
            ]
            aggregate = selected.groupby("date_idx", sort=False)[
                ["base", "stress"]
            ].agg(["sum", "size"])
            daily = market_predictions[["date_idx", "trade_date", "fold"]].copy()
            return_values = aggregate.loc[:, [("base", "sum"), ("stress", "sum")]]
            return_values.columns = ["base", "stress"]
            return_values /= float(top_k)
            trade_count = aggregate[("base", "size")].rename("trade_count")
            daily = daily.merge(
                return_values.reset_index(), on="date_idx", how="left"
            ).merge(trade_count.reset_index(), on="date_idx", how="left")
            daily[["base", "stress", "trade_count"]] = daily[
                ["base", "stress", "trade_count"]
            ].fillna(0.0)
            daily["trade_count"] = daily["trade_count"].astype(np.int32)
            daily["gate"] = gate_name
            daily["top_k"] = top_k
            task_name = f"{gate_name}_top{top_k}"
            summaries[task_name] = _market_gate_metrics(daily)
            daily_parts.append(daily)
    daily_results = pd.concat(daily_parts, ignore_index=True)
    primary = summaries["consensus_top10"]
    primary_lower = primary["stress"]["lower"]
    gate = {
        "primary": "consensus_top10",
        "requires_all_folds_positive": True,
        "requires_all_years_positive": True,
        "requires_positive_stress_hac_lower_bound": True,
        "all_folds_positive": (
            int(primary["positive_stress_fold_count"]) == int(primary["fold_count"])
        ),
        "all_years_positive": (
            int(primary["positive_stress_year_count"]) == int(primary["year_count"])
        ),
        "positive_stress_hac_lower_bound": bool(
            primary_lower is not None and float(primary_lower) > 0.0
        ),
    }
    gate["passed"] = bool(
        gate["all_folds_positive"]
        and gate["all_years_positive"]
        and gate["positive_stress_hac_lower_bound"]
    )
    root = output_path.parent
    predictions_path = root / "market_predictions.parquet"
    coefficients_path = root / "coefficients.parquet"
    selections_path = root / "top10_selections.parquet"
    daily_path = root / "daily_gate_results.parquet"
    _write_parquet(market_predictions, predictions_path)
    _write_parquet(coefficients, coefficients_path)
    _write_parquet(selections, selections_path)
    _write_parquet(daily_results, daily_path)
    manifest = {
        "schema": MARKET_REGIME_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "fingerprint": fingerprint,
        "horizon": horizon,
        "ranking_target": ranking_target,
        "profile": profile_name,
        "feature_variant": feature_variant,
        "training_mode": training_mode,
        "market_feature_count": len(market_features),
        "market_features": list(market_features),
        "fit_records": fit_records,
        "summaries": summaries,
        "gate": gate,
        "account_replay_performed": False,
        "forbidden_2026_read_count": 0,
        "files": {
            "market_predictions": _file_record(
                predictions_path, row_count=len(market_predictions)
            ),
            "coefficients": _file_record(
                coefficients_path, row_count=len(coefficients)
            ),
            "top10_selections": _file_record(
                selections_path, row_count=len(selections)
            ),
            "daily_gate_results": _file_record(
                daily_path, row_count=len(daily_results)
            ),
        },
        "sources": {
            "model_inputs": _file_record(context.model_manifest_path),
            "exact_net_targets": _file_record(
                output_root / "exact_net_targets/manifest.json",
                target_fingerprint=target_manifest["fingerprint"],
            ),
            "ranking_tasks": [_file_record(path) for _, path in task_records],
        },
    }
    _write_json(output_path, manifest)
    _emit(
        "market_regime_evaluation_completed",
        gate_passed=gate["passed"],
        primary_stress_mean=primary["stress"]["mean"],
        primary_stress_lower=primary_lower,
    )
    return manifest


def evaluate_oof_selections(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    profile: str | None = None,
    ensemble_profiles: Sequence[str] | None = None,
    feature_variant: str = DEFAULT_FEATURE_VARIANT,
    candidate_only: bool = False,
    training_mode: str = "outer_early_stop",
) -> dict[str, Any]:
    study = load_study(study_path)
    if training_mode not in TRAINING_MODES:
        raise FullMarketForecastError(f"unknown_training_mode:{training_mode}")
    context = _source_context(study)
    target_manifest = prepare_targets(study_path=study_path, output_root=output_root)
    if ensemble_profiles:
        model_profiles = tuple(str(value) for value in ensemble_profiles)
        if len(model_profiles) < 2 or len(set(model_profiles)) != len(model_profiles):
            raise FullMarketForecastError("ensemble_profiles_must_be_unique")
        profile_name = "ensemble_" + "_".join(model_profiles)
    else:
        model_profiles = (str(profile or study["lightgbm"]["primary_profile"]),)
        profile_name = model_profiles[0]
    unknown_profiles = set(model_profiles) - set(study["lightgbm"]["profiles"])
    if unknown_profiles:
        raise FullMarketForecastError(
            f"unknown_ensemble_profiles:{sorted(unknown_profiles)}"
        )
    folds = build_forward_folds(
        date_idx=context.row_index["date_idx"].to_numpy(dtype=np.int32),
        trade_date=context.row_index["trade_date"].astype(str).to_numpy(),
        validation_start_date=str(study["validation"]["validation_start_date"]),
        validation_end_date=str(study["validation"]["validation_end_date"]),
        fold_count=int(study["validation"]["forward_fold_count"]),
        purge_days=int(study["validation"]["common_purge_trading_days"]),
    )
    if candidate_only:
        required_targets = (
            "next_close_up",
            "legal_mfe_d2_ge_1pct",
            "exposure_mae_d2_gt_minus_3pct",
        )
        variants = ("d2_upside_safety_minimum",)
        top_ks = (1, 3, 10, 30)
    else:
        required_targets = (
            "next_close_up",
            "market_excess_endpoint_return_d2_rank",
            "legal_mfe_d2_ge_1pct",
            "exposure_mae_d2_gt_minus_3pct",
            "legal_exit_return_d2_gt_0p3pct",
        )
        variants = (
            "next_direction",
            "d2_relative_rank",
            "average_rank",
            "maximin_rank",
            "d2_take_profit_probability",
            "d2_take_profit_path_average",
            "d2_take_profit_direction_average",
            "d2_take_profit_all_average",
            "d2_upside_safety_average",
            "d2_path_safety_average",
            "d2_upside_path_safety_average",
            "d2_upside_safety_minimum",
            "d2_profit_probability",
            "d2_profit_safety_minimum",
            "d2_profit_upside_average",
            "d2_profit_upside_safety_minimum",
            "d2_profit_path_safety_average",
        )
        top_ks = (1, 3)
    task_records: dict[str, dict[str, Any]] = {}
    task_paths: list[Path] = []
    for fold in folds:
        for target in required_targets:
            for model_profile in model_profiles:
                result, path = _oof_task_result(
                    output_root,
                    fold=int(fold["fold"]),
                    target=target,
                    profile=model_profile,
                    feature_variant=feature_variant,
                    training_mode=training_mode,
                )
                task_records[f"{int(fold['fold'])}:{target}:{model_profile}"] = result
                task_paths.append(path)
    fingerprint = _stable_hash(
        {
            "schema": EVALUATION_SCHEMA,
            "study_sha256": _sha256(study_path),
            "target_fingerprint": target_manifest["fingerprint"],
            "task_results": {str(path): _sha256(path) for path in task_paths},
            "variants": variants,
            "exit_policies": PRIMARY_EXIT_POLICIES,
            "gates": ("always", "daily_mean_direction_above_0p5"),
            "top_k": top_ks,
            "cost_scenarios": {"base": 1.0, "stress": 2.0},
            "notional_cny": float(study["targets"]["positive_after_cost_notional_cny"]),
            "training_mode": training_mode,
            "feature_variant": feature_variant,
            "model_profiles": model_profiles,
            "evaluation_scope": "candidate_only" if candidate_only else "full",
            "selection_universe": "all_signal_date_candidates_before_entry_fill",
            "unfilled_selected_order": "cash_no_rank_substitution",
            "maximum_evaluable_signal_offset": -(2 + RETRY_DAYS),
        }
    )
    evaluation_name = "oof_d2_selection"
    if feature_variant != DEFAULT_FEATURE_VARIANT:
        evaluation_name += f"_{feature_variant}"
    if profile_name != str(study["lightgbm"]["primary_profile"]):
        evaluation_name += f"_{profile_name}"
    if candidate_only:
        evaluation_name += "_candidate"
    if training_mode != "outer_early_stop":
        evaluation_name += f"_{training_mode}"
    evaluation_root = output_root / "evaluation" / evaluation_name
    manifest_path = evaluation_root / "manifest.json"
    if manifest_path.is_file():
        current = _read_json(manifest_path)
        if (
            current.get("status") == "completed"
            and current.get("fingerprint") == fingerprint
        ):
            return current

    values_record = target_manifest["files"]["values"]
    valid_record = target_manifest["files"]["valid"]
    fill_record = target_manifest["files"]["legal_fill_days"]
    values = np.memmap(
        Path(values_record["path"]),
        dtype=np.float32,
        mode="r",
        shape=tuple(int(value) for value in values_record["shape"]),
    )
    valid = np.memmap(
        Path(valid_record["path"]),
        dtype=np.uint8,
        mode="r",
        shape=tuple(int(value) for value in valid_record["shape"]),
    )
    fill_days = np.memmap(
        Path(fill_record["path"]),
        dtype=np.int16,
        mode="r",
        shape=tuple(int(value) for value in fill_record["shape"]),
    )
    legal_column = list(target_manifest["target_columns"]).index("legal_exit_return_d2")
    mfe_column = list(target_manifest["target_columns"]).index("legal_mfe_d2")
    daily_raw = _open_array(
        context.pack["feature_channels"]["daily_raw"], dtype=np.float32
    )
    raw_open = _open_array(
        context.pack["execution_arrays"]["entry_open_raw"], dtype=np.float32
    )
    entry_filled = _open_array(context.pack["masks"]["entry_filled"], dtype=np.bool_)
    costs = parse_execution_costs(context.pack)
    notional = float(study["targets"]["positive_after_cost_notional_cny"])
    exit_policies = PRIMARY_EXIT_POLICIES
    gates = ("always", "daily_mean_direction_above_0p5")
    cost_scenarios = {"base": 1.0, "stress": 2.0}
    selection_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    net_cache: dict[tuple[int, str, str], tuple[float, bool]] = {}
    row_index = context.row_index
    _emit(
        "oof_selection_evaluation_started",
        folds=len(folds),
        training_mode=training_mode,
    )
    for fold in folds:
        fold_number = int(fold["fold"])
        all_positions = _fold_rows(row_index, fold, "validation")
        evaluable = row_index.iloc[all_positions]["date_idx"].to_numpy(
            dtype=np.int32
        ) <= int(context.cutoff_idx - 2 - RETRY_DAYS)
        positions = all_positions[evaluable]

        direction = _load_ensemble_prediction(
            task_records,
            fold_number=fold_number,
            target="next_close_up",
            model_profiles=model_profiles,
            row_mask=evaluable,
        )
        take_profit_probability = _load_ensemble_prediction(
            task_records,
            fold_number=fold_number,
            target="legal_mfe_d2_ge_1pct",
            model_profiles=model_profiles,
            row_mask=evaluable,
        )
        safety_probability = _load_ensemble_prediction(
            task_records,
            fold_number=fold_number,
            target="exposure_mae_d2_gt_minus_3pct",
            model_profiles=model_profiles,
            row_mask=evaluable,
        )
        if candidate_only:
            path_score = np.full(len(positions), np.nan, dtype=np.float32)
            profit_probability = np.full(len(positions), np.nan, dtype=np.float32)
        else:
            path_score = _load_ensemble_prediction(
                task_records,
                fold_number=fold_number,
                target="market_excess_endpoint_return_d2_rank",
                model_profiles=model_profiles,
                row_mask=evaluable,
            )
            profit_probability = _load_ensemble_prediction(
                task_records,
                fold_number=fold_number,
                target="legal_exit_return_d2_gt_0p3pct",
                model_profiles=model_profiles,
                row_mask=evaluable,
            )
        prediction_lengths = (
            len(direction),
            len(take_profit_probability),
            len(safety_probability),
            len(path_score),
            len(profit_probability),
        )
        if any(length != len(positions) for length in prediction_lengths):
            raise FullMarketForecastError("oof_prediction_row_count_mismatch")
        frame = row_index.iloc[positions][
            ["candidate_id", "date_idx", "trade_date", "symbol", "symbol_idx"]
        ].copy()
        frame["model_row_position"] = positions
        frame["direction_probability"] = np.asarray(direction, dtype=np.float32)
        frame["path_score"] = np.asarray(path_score, dtype=np.float32)
        frame["take_profit_probability"] = np.asarray(
            take_profit_probability, dtype=np.float32
        )
        frame["safety_probability"] = np.asarray(safety_probability, dtype=np.float32)
        frame["profit_probability"] = np.asarray(profit_probability, dtype=np.float32)
        frame["legal_gross_return"] = np.asarray(
            values[positions, legal_column], dtype=np.float32
        )
        frame["legal_valid"] = np.asarray(valid[positions, legal_column], dtype=bool)
        frame["legal_mfe"] = np.asarray(values[positions, mfe_column], dtype=np.float32)
        frame["mfe_valid"] = np.asarray(valid[positions, mfe_column], dtype=bool)
        frame["fill_day"] = np.asarray(fill_days[positions, 0], dtype=np.int16)
        frame["entry_filled"] = np.asarray(
            entry_filled[
                frame["date_idx"].to_numpy(dtype=np.int32),
                frame["symbol_idx"].to_numpy(dtype=np.int32),
            ],
            dtype=bool,
        )
        frame["legal_gross_return"] = np.where(
            frame["legal_valid"] & np.isfinite(frame["legal_gross_return"]),
            frame["legal_gross_return"],
            0.0,
        )
        finite = (
            np.isfinite(frame["direction_probability"])
            & np.isfinite(frame["take_profit_probability"])
            & np.isfinite(frame["safety_probability"])
        )
        if not candidate_only:
            finite &= np.isfinite(frame["path_score"])
            finite &= np.isfinite(frame["profit_probability"])
        frame = frame[finite].copy()
        frame["take_profit_hit"] = (
            frame["mfe_valid"]
            & np.isfinite(frame["legal_mfe"])
            & frame["legal_mfe"].ge(TAKE_PROFIT_THRESHOLD)
        )
        frame["direction_rank"] = _within_date_rank(frame, "direction_probability")
        frame["take_profit_rank"] = _within_date_rank(frame, "take_profit_probability")
        frame["safety_rank"] = _within_date_rank(frame, "safety_probability")
        frame["score__d2_upside_safety_minimum"] = np.minimum(
            frame["take_profit_rank"], frame["safety_rank"]
        )
        if not candidate_only:
            frame["path_rank"] = _within_date_rank(frame, "path_score")
            frame["profit_rank"] = _within_date_rank(frame, "profit_probability")
            frame["score__next_direction"] = frame["direction_rank"]
            frame["score__d2_relative_rank"] = frame["path_rank"]
            frame["score__average_rank"] = 0.5 * (
                frame["direction_rank"] + frame["path_rank"]
            )
            frame["score__maximin_rank"] = np.minimum(
                frame["direction_rank"], frame["path_rank"]
            )
            frame["score__d2_take_profit_probability"] = frame["take_profit_rank"]
            frame["score__d2_take_profit_path_average"] = 0.5 * (
                frame["take_profit_rank"] + frame["path_rank"]
            )
            frame["score__d2_take_profit_direction_average"] = 0.5 * (
                frame["take_profit_rank"] + frame["direction_rank"]
            )
            frame["score__d2_take_profit_all_average"] = (
                frame["take_profit_rank"] + frame["path_rank"] + frame["direction_rank"]
            ) / 3.0
            frame["score__d2_upside_safety_average"] = 0.5 * (
                frame["take_profit_rank"] + frame["safety_rank"]
            )
            frame["score__d2_path_safety_average"] = 0.5 * (
                frame["path_rank"] + frame["safety_rank"]
            )
            frame["score__d2_upside_path_safety_average"] = (
                frame["take_profit_rank"] + frame["path_rank"] + frame["safety_rank"]
            ) / 3.0
            frame["score__d2_profit_probability"] = frame["profit_rank"]
            frame["score__d2_profit_safety_minimum"] = np.minimum(
                frame["profit_rank"], frame["safety_rank"]
            )
            frame["score__d2_profit_upside_average"] = 0.5 * (
                frame["profit_rank"] + frame["take_profit_rank"]
            )
            frame["score__d2_profit_upside_safety_minimum"] = np.minimum.reduce(
                [
                    frame["profit_rank"].to_numpy(dtype=np.float64),
                    frame["take_profit_rank"].to_numpy(dtype=np.float64),
                    frame["safety_rank"].to_numpy(dtype=np.float64),
                ]
            )
            frame["score__d2_profit_path_safety_average"] = (
                frame["profit_rank"] + frame["path_rank"] + frame["safety_rank"]
            ) / 3.0
        date_summary = frame.groupby("date_idx", sort=False).agg(
            trade_date=("trade_date", "first"),
            daily_mean_direction=("direction_probability", "mean"),
            candidate_count=("candidate_id", "size"),
        )
        for date_idx, group in frame.groupby("date_idx", sort=False):
            summary = date_summary.loc[int(date_idx)]
            for exit_policy in exit_policies:
                policy_group = group.copy()
                take_profit_threshold = TAKE_PROFIT_EXIT_THRESHOLDS.get(exit_policy)
                if take_profit_threshold is not None:
                    policy_group["policy_take_profit_hit"] = policy_group[
                        "mfe_valid"
                    ] & np.isfinite(policy_group["legal_mfe"])
                    policy_group["policy_take_profit_hit"] &= policy_group[
                        "legal_mfe"
                    ].ge(take_profit_threshold)
                    policy_group["policy_gross_return"] = np.where(
                        policy_group["policy_take_profit_hit"],
                        take_profit_threshold,
                        policy_group["legal_gross_return"],
                    )
                    policy_group["policy_fill_day"] = np.where(
                        policy_group["policy_take_profit_hit"],
                        2,
                        policy_group["fill_day"],
                    )
                else:
                    policy_group["policy_take_profit_hit"] = False
                    policy_group["policy_gross_return"] = policy_group[
                        "legal_gross_return"
                    ]
                    policy_group["policy_fill_day"] = policy_group["fill_day"]
                pool_policy_gross_mean = float(
                    policy_group["policy_gross_return"].mean()
                )
                for variant in variants:
                    score_column = f"score__{variant}"
                    ordered = policy_group.sort_values(
                        [score_column, "direction_probability", "symbol"],
                        ascending=[False, False, True],
                        kind="stable",
                    )
                    for gate in gates:
                        gate_passed = bool(
                            gate == "always"
                            or float(summary["daily_mean_direction"]) > 0.5
                        )
                        for top_k in top_ks:
                            selected = (
                                ordered.head(top_k) if gate_passed else ordered.head(0)
                            )
                            for cost_name, multiplier in cost_scenarios.items():
                                net_values: list[float] = []
                                gross_values: list[float] = []
                                for selection_rank, (_, row) in enumerate(
                                    selected.iterrows(), start=1
                                ):
                                    row_position = int(row["model_row_position"])
                                    cache_key = (
                                        row_position,
                                        cost_name,
                                        exit_policy,
                                    )
                                    if cache_key not in net_cache:
                                        net_cache[cache_key] = _exact_unit_net_return(
                                            signal_idx=int(row["date_idx"]),
                                            symbol_idx=int(row["symbol_idx"]),
                                            legal_gross_return=float(
                                                row["policy_gross_return"]
                                            ),
                                            fill_day=int(row["policy_fill_day"]),
                                            daily_raw=daily_raw,
                                            raw_open=raw_open,
                                            date_values=context.date_values,
                                            costs=costs,
                                            notional_cny=notional,
                                            slippage_multiplier=multiplier,
                                            entry_filled=entry_filled,
                                        )
                                    net_return, order_filled = net_cache[cache_key]
                                    net_values.append(float(net_return))
                                    gross_values.append(
                                        float(row["policy_gross_return"])
                                    )
                                    selection_rows.append(
                                        {
                                            "fold": fold_number,
                                            "date_idx": int(date_idx),
                                            "trade_date": str(row["trade_date"]),
                                            "variant": variant,
                                            "exit_policy": exit_policy,
                                            "gate": gate,
                                            "top_k": top_k,
                                            "cost_scenario": cost_name,
                                            "selection_rank": selection_rank,
                                            "candidate_id": int(row["candidate_id"]),
                                            "symbol": str(row["symbol"]),
                                            "entry_filled": bool(row["entry_filled"]),
                                            "direction_probability": float(
                                                row["direction_probability"]
                                            ),
                                            "path_score": float(row["path_score"]),
                                            "take_profit_probability": float(
                                                row["take_profit_probability"]
                                            ),
                                            "safety_probability": float(
                                                row["safety_probability"]
                                            ),
                                            "profit_probability": float(
                                                row["profit_probability"]
                                            ),
                                            "take_profit_hit": bool(
                                                row["policy_take_profit_hit"]
                                            ),
                                            "take_profit_threshold": (
                                                take_profit_threshold
                                            ),
                                            "combined_score": float(row[score_column]),
                                            "planned_legal_gross_return": float(
                                                row["legal_gross_return"]
                                            ),
                                            "legal_gross_return": float(
                                                row["policy_gross_return"]
                                            ),
                                            "exact_net_return": float(net_return),
                                            "fill_day": int(row["policy_fill_day"]),
                                            "order_filled": bool(order_filled),
                                        }
                                    )
                                basket_gross = (
                                    float(np.mean(gross_values))
                                    if gross_values
                                    else 0.0
                                )
                                basket_net = (
                                    float(np.mean(net_values)) if net_values else 0.0
                                )
                                daily_rows.append(
                                    {
                                        "fold": fold_number,
                                        "date_idx": int(date_idx),
                                        "trade_date": str(summary["trade_date"]),
                                        "year": int(str(summary["trade_date"])[:4]),
                                        "variant": variant,
                                        "exit_policy": exit_policy,
                                        "gate": gate,
                                        "top_k": top_k,
                                        "cost_scenario": cost_name,
                                        "gate_passed": gate_passed,
                                        "trade_count": len(net_values),
                                        "candidate_count": int(
                                            summary["candidate_count"]
                                        ),
                                        "daily_mean_direction": float(
                                            summary["daily_mean_direction"]
                                        ),
                                        "pool_legal_gross_mean": (
                                            pool_policy_gross_mean
                                        ),
                                        "basket_legal_gross_return": basket_gross,
                                        "basket_exact_net_return": basket_net,
                                        "paired_legal_gross_excess": (
                                            basket_gross - pool_policy_gross_mean
                                            if gate_passed
                                            else 0.0
                                        ),
                                    }
                                )
        _emit("oof_selection_fold_completed", fold=fold_number, rows=len(frame))
    selections = pd.DataFrame(selection_rows)
    daily = pd.DataFrame(daily_rows)
    metric_rows: list[dict[str, Any]] = []
    grouping = ["variant", "exit_policy", "gate", "top_k", "cost_scenario"]
    for keys, group in daily.groupby(grouping, sort=False):
        net = group["basket_exact_net_return"].to_numpy(dtype=np.float64)
        gross_excess = group["paired_legal_gross_excess"].to_numpy(dtype=np.float64)
        traded = group["trade_count"].to_numpy(dtype=np.int32) > 0
        trimmed = np.sort(net)
        keep_count = max(1, math.floor(len(trimmed) * 0.99))
        record = {
            **dict(zip(grouping, keys, strict=True)),
            "date_count": len(group),
            "traded_date_count": int(traded.sum()),
            "trade_rate": float(traded.mean()),
            "mean_exact_net_return": float(net.mean()),
            "median_exact_net_return": float(np.median(net)),
            "positive_net_date_fraction": float(np.mean(net > 0.0)),
            "mean_paired_gross_excess": float(gross_excess.mean()),
            "remove_top_one_percent_net_mean": float(trimmed[:keep_count].mean()),
            "hac20_exact_net": _newey_west_interval(net, lag=20),
            "hac20_paired_gross_excess": _newey_west_interval(gross_excess, lag=20),
            "positive_fold_count": int(
                group.groupby("fold")["basket_exact_net_return"].mean().gt(0.0).sum()
            ),
            "positive_year_count": int(
                group.groupby("year")["basket_exact_net_return"].mean().gt(0.0).sum()
            ),
        }
        metric_rows.append(record)
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["mean_exact_net_return", "mean_paired_gross_excess"], ascending=False
    )
    annual = (
        daily.groupby([*grouping, "year"], as_index=False)
        .agg(
            date_count=("date_idx", "size"),
            traded_date_count=("trade_count", lambda values: int((values > 0).sum())),
            mean_exact_net_return=("basket_exact_net_return", "mean"),
            mean_paired_gross_excess=("paired_legal_gross_excess", "mean"),
            positive_net_date_fraction=(
                "basket_exact_net_return",
                lambda values: float((values > 0.0).mean()),
            ),
        )
        .sort_values([*grouping, "year"])
    )
    evaluation_root.mkdir(parents=True, exist_ok=True)
    selection_path = evaluation_root / "selections.parquet"
    daily_path = evaluation_root / "daily_metrics.parquet"
    metrics_path = evaluation_root / "summary_metrics.parquet"
    annual_path = evaluation_root / "annual_metrics.parquet"
    _write_parquet(selections, selection_path)
    _write_parquet(daily, daily_path)
    _write_parquet(metrics, metrics_path)
    _write_parquet(annual, annual_path)
    best = metrics.iloc[0].to_dict() if len(metrics) else {}
    manifest = {
        "schema": EVALUATION_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "fingerprint": fingerprint,
        "profile": profile_name,
        "model_profiles": list(model_profiles),
        "feature_variant": feature_variant,
        "evaluation_scope": "candidate_only" if candidate_only else "full",
        "training_mode": training_mode,
        "variants": list(variants),
        "exit_policies": list(exit_policies),
        "take_profit_model_threshold": TAKE_PROFIT_THRESHOLD,
        "take_profit_exit_thresholds_previously_audited": (TAKE_PROFIT_EXIT_THRESHOLDS),
        "gates": list(gates),
        "top_k": list(top_ks),
        "cost_scenarios": cost_scenarios,
        "notional_cny": notional,
        "selection_row_count": len(selections),
        "daily_row_count": len(daily),
        "best_development_configuration": _json_safe(best),
        "decision_boundary": {
            "daily_event_evaluation_not_account_replay": True,
            "overlapping_d2_positions_not_compounded": True,
            "ranking_uses_future_entry_fill_status": False,
            "unfilled_selected_order_is_cash_without_substitution": True,
            "historical_result_is_adaptive_not_independent_confirmation": True,
            "outer_validation_used_for_iteration_selection": (
                training_mode == "outer_early_stop"
            ),
            "stable_profit_claim_allowed": False,
            "forbidden_2026_read_count": 0,
        },
        "sources": {
            "study": _file_record(study_path),
            "targets": _file_record(output_root / "targets/manifest.json"),
            "tasks": [_file_record(path) for path in task_paths],
        },
        "files": {
            "selections": _file_record(selection_path, row_count=len(selections)),
            "daily_metrics": _file_record(daily_path, row_count=len(daily)),
            "summary_metrics": _file_record(metrics_path, row_count=len(metrics)),
            "annual_metrics": _file_record(annual_path, row_count=len(annual)),
        },
    }
    _write_json(manifest_path, manifest)
    _emit(
        "oof_selection_evaluation_completed",
        selection_rows=len(selections),
        daily_rows=len(daily),
    )
    return manifest


@dataclass
class ReplayHolding:
    trade_id: int
    position: economic.Position
    signal_date_idx: int
    signal_date: str
    symbol: str
    selection_rank: int
    legal_gross_return: float
    exit_date_idx: int
    exit_phase: str
    buy_costs: dict[str, float]


def _account_specs(*, candidate_only: bool = False) -> tuple[dict[str, Any], ...]:
    configurations = (
        (
            "average_rank",
            "planned_close",
            "daily_mean_direction_above_0p5",
            3,
        ),
        (
            "d2_relative_rank",
            "planned_close",
            "daily_mean_direction_above_0p5",
            3,
        ),
        ("d2_relative_rank", "planned_close", "always", 1),
        (
            "d2_take_profit_probability",
            "planned_close",
            "daily_mean_direction_above_0p5",
            3,
        ),
        (
            "d2_take_profit_path_average",
            "planned_close",
            "daily_mean_direction_above_0p5",
            3,
        ),
        (
            "d2_upside_safety_minimum",
            "planned_close",
            "daily_mean_direction_above_0p5",
            1,
        ),
    )
    if candidate_only:
        configurations = tuple(
            (
                "d2_upside_safety_minimum",
                "planned_close",
                "daily_mean_direction_above_0p5",
                top_k,
            )
            for top_k in (1, 3, 10, 30)
        )
    specs = [
        {
            "variant": variant,
            "exit_policy": exit_policy,
            "gate": gate,
            "top_k": top_k,
            "cost_scenario": cost,
            "slippage_multiplier": 1.0 if cost == "base" else 2.0,
            "cohort_equity_fraction": 0.5,
        }
        for variant, exit_policy, gate, top_k in configurations
        for cost in ("base", "stress")
    ]
    for cap in (0.05, 0.10):
        for cost in ("base", "stress"):
            specs.append(
                {
                    "variant": "d2_upside_safety_minimum",
                    "exit_policy": "planned_close",
                    "gate": "daily_mean_direction_above_0p5",
                    "top_k": 1,
                    "cost_scenario": cost,
                    "slippage_multiplier": 1.0 if cost == "base" else 2.0,
                    "cohort_equity_fraction": 0.5,
                    "maximum_credited_gross_return": cap,
                }
            )
    return tuple(specs)


def _account_task_id(spec: Mapping[str, Any]) -> str:
    task_id = (
        f"{spec['variant']}__{spec['exit_policy']}__{spec['gate']}"
        f"__top{int(spec['top_k'])}"
        f"__{spec['cost_scenario']}"
    )
    cap = spec.get("maximum_credited_gross_return")
    if cap is not None:
        task_id += f"__cap{round(float(cap) * 100)}pct"
    return task_id


def _simulate_account_spec(
    *,
    spec: Mapping[str, Any],
    selections: pd.DataFrame,
    context: SourceContext,
    daily_raw: np.ndarray,
    raw_open: np.ndarray,
    entry_filled: np.ndarray,
    costs: ExecutionCosts,
    starting_cash: float = 1_000_000.0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    selected = selections[
        selections["variant"].eq(str(spec["variant"]))
        & selections["exit_policy"].eq(str(spec["exit_policy"]))
        & selections["gate"].eq(str(spec["gate"]))
        & selections["top_k"].eq(int(spec["top_k"]))
        & selections["cost_scenario"].eq("base")
    ].copy()
    selected = selected.drop_duplicates(
        ["date_idx", "candidate_id", "selection_rank"], keep="first"
    ).sort_values(["date_idx", "selection_rank"])
    if selected.empty:
        raise FullMarketForecastError(f"account_selection_schedule_empty:{spec}")
    selected["symbol_idx"] = selected["symbol"].map(
        {
            str(symbol): index
            for index, symbol in enumerate(context.pack["symbol_values"])
        }
    )
    if selected["symbol_idx"].isna().any():
        raise FullMarketForecastError("account_selection_symbol_missing_from_pack")
    signal_schedule = {
        int(date_idx): group.copy()
        for date_idx, group in selected.groupby("date_idx", sort=True)
    }
    first_signal_idx = int(selected["date_idx"].min())
    last_signal_idx = int(selected["date_idx"].max())
    cutoff_idx = int(context.cutoff_idx)
    holdings: dict[int, ReplayHolding] = {}
    open_exits: dict[int, list[int]] = {}
    close_exits: dict[int, list[int]] = {}
    cash = float(starting_cash)
    previous_equity = float(starting_cash)
    trade_id = 0
    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    total_costs = {
        "commission": 0.0,
        "transfer_fee": 0.0,
        "stamp_tax": 0.0,
        "slippage": 0.0,
    }

    def close_trade(identifier: int, *, current_idx: int) -> None:
        nonlocal cash
        holding = holdings.pop(identifier)
        if holding.legal_gross_return <= -1.0:
            proceeds = 0.0
            sale = {
                "commission": 0.0,
                "transfer_fee": 0.0,
                "stamp_tax": 0.0,
                "slippage": 0.0,
                "total_cost": 0.0,
            }
        else:
            adjusted_exit = holding.position.entry_adjusted_open * (
                1.0 + holding.legal_gross_return
            )
            proceeds, sale = economic._sell_position(
                position=holding.position,
                adjusted_open=adjusted_exit,
                trade_date=str(context.date_values[current_idx]),
                costs=costs,
                slippage_multiplier=float(spec["slippage_multiplier"]),
            )
        cash += float(proceeds)
        for name in total_costs:
            total_costs[name] += float(sale.get(name, 0.0))
        pnl = float(proceeds - holding.position.net_cash_outflow)
        trade_rows.append(
            {
                "trade_id": identifier,
                "signal_date_idx": holding.signal_date_idx,
                "signal_date": holding.signal_date,
                "entry_date_idx": holding.position.entry_date_idx,
                "entry_date": str(context.date_values[holding.position.entry_date_idx]),
                "exit_date_idx": current_idx,
                "exit_date": str(context.date_values[current_idx]),
                "exit_phase": holding.exit_phase,
                "symbol": holding.symbol,
                "symbol_idx": holding.position.symbol_idx,
                "selection_rank": holding.selection_rank,
                "shares": holding.position.shares,
                "gross_entry_notional": holding.position.gross_entry_notional,
                "net_cash_outflow": holding.position.net_cash_outflow,
                "proceeds": float(proceeds),
                "pnl": pnl,
                "trade_net_return": (
                    pnl / holding.position.net_cash_outflow
                    if holding.position.net_cash_outflow > 0.0
                    else math.nan
                ),
                "legal_gross_return": holding.legal_gross_return,
                "buy_cost": float(holding.buy_costs.get("total_cost", 0.0)),
                "sell_cost": float(sale.get("total_cost", 0.0)),
            }
        )

    for current_idx in range(first_signal_idx, cutoff_idx + 1):
        for identifier in open_exits.pop(current_idx, []):
            if identifier in holdings:
                close_trade(identifier, current_idx=current_idx)

        signal_idx = current_idx - 1
        orders = signal_schedule.get(signal_idx)
        requested = 0 if orders is None else len(orders)
        filled = 0
        cohort_budget = min(
            cash,
            previous_equity * float(spec["cohort_equity_fraction"]),
        )
        per_order_budget = (
            cohort_budget / int(spec["top_k"])
            if requested and int(spec["top_k"]) > 0
            else 0.0
        )
        if orders is not None:
            for row in orders.itertuples(index=False):
                symbol_idx = int(row.symbol_idx)
                if not bool(entry_filled[signal_idx, symbol_idx]):
                    continue
                adjusted_entry = float(daily_raw[current_idx, symbol_idx, 0])
                entry_raw = float(raw_open[current_idx, symbol_idx])
                position, buy = economic._buy_position(
                    available_cash=cash,
                    allocated_cash=per_order_budget,
                    symbol_idx=symbol_idx,
                    signal_date_idx=signal_idx,
                    execution_date_idx=current_idx,
                    raw_open=entry_raw,
                    adjusted_open=adjusted_entry,
                    costs=costs,
                    slippage_multiplier=float(spec["slippage_multiplier"]),
                )
                if position is None:
                    continue
                cash -= position.net_cash_outflow
                if cash < -1.0e-6:
                    raise FullMarketForecastError("account_cash_became_negative")
                cash = max(cash, 0.0)
                filled += 1
                trade_id += 1
                fill_day = int(row.fill_day)
                exit_idx = signal_idx + fill_day
                uncapped_legal_return = float(row.legal_gross_return)
                cap = spec.get("maximum_credited_gross_return")
                legal_return = (
                    min(uncapped_legal_return, float(cap))
                    if cap is not None
                    else uncapped_legal_return
                )
                if legal_return <= -1.0:
                    phase = "terminal_recovery"
                    close_exits.setdefault(exit_idx, []).append(trade_id)
                elif str(spec["exit_policy"]) in TAKE_PROFIT_EXIT_THRESHOLDS and bool(
                    row.take_profit_hit
                ):
                    phase = "take_profit_limit"
                    close_exits.setdefault(exit_idx, []).append(trade_id)
                elif fill_day == int(spec.get("planned_fill_day", 2)):
                    phase = "planned_close"
                    close_exits.setdefault(exit_idx, []).append(trade_id)
                else:
                    phase = "delayed_open"
                    open_exits.setdefault(exit_idx, []).append(trade_id)
                holdings[trade_id] = ReplayHolding(
                    trade_id=trade_id,
                    position=position,
                    signal_date_idx=signal_idx,
                    signal_date=str(context.date_values[signal_idx]),
                    symbol=str(row.symbol),
                    selection_rank=int(row.selection_rank),
                    legal_gross_return=legal_return,
                    exit_date_idx=exit_idx,
                    exit_phase=phase,
                    buy_costs={str(key): float(value) for key, value in buy.items()},
                )
                for name in total_costs:
                    total_costs[name] += float(buy.get(name, 0.0))

        for identifier in close_exits.pop(current_idx, []):
            if identifier in holdings:
                close_trade(identifier, current_idx=current_idx)

        adjusted_close = np.asarray(daily_raw[current_idx, :, 3], dtype=np.float64)
        equity = float(cash)
        for holding in holdings.values():
            equity += economic._position_value(
                holding.position,
                float(adjusted_close[holding.position.symbol_idx]),
            )
        daily_return = equity / previous_equity - 1.0 if previous_equity > 0.0 else 0.0
        equity_rows.append(
            {
                "date_idx": current_idx,
                "trade_date": str(context.date_values[current_idx]),
                "year": int(str(context.date_values[current_idx])[:4]),
                "cash": cash,
                "position_count": len(holdings),
                "requested_entry_count": requested,
                "filled_entry_count": filled,
                "equity": equity,
                "daily_net_return": daily_return,
            }
        )
        previous_equity = equity

    if holdings or open_exits or close_exits:
        raise FullMarketForecastError("account_positions_or_exits_unresolved_at_cutoff")
    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    equity["equity_peak"] = equity["equity"].cummax()
    equity["drawdown"] = equity["equity"] / equity["equity_peak"] - 1.0
    annual_rows = []
    for year, group in equity.groupby("year", sort=True):
        annual_rows.append(
            {
                "year": int(year),
                "net_return": float(
                    np.prod(1.0 + group["daily_net_return"].to_numpy(dtype=np.float64))
                    - 1.0
                ),
                "maximum_drawdown": float(group["drawdown"].min()),
                "trade_count": int(
                    trades["exit_date"]
                    .astype(str)
                    .str.startswith(f"{int(year)}-")
                    .sum()
                ),
            }
        )
    annual = pd.DataFrame(annual_rows)
    daily_returns = equity["daily_net_return"].to_numpy(dtype=np.float64)
    result = {
        "task_id": _account_task_id(spec),
        "spec": dict(spec),
        "first_signal_date": str(context.date_values[first_signal_idx]),
        "last_signal_date": str(context.date_values[last_signal_idx]),
        "last_account_date": str(context.date_values[cutoff_idx]),
        "starting_cash": float(starting_cash),
        "ending_equity": float(equity["equity"].iloc[-1]),
        "total_net_return": float(equity["equity"].iloc[-1] / starting_cash - 1.0),
        "maximum_drawdown": float(equity["drawdown"].min()),
        "annualized_daily_sharpe": (
            float(np.sqrt(242.0) * daily_returns.mean() / daily_returns.std(ddof=1))
            if daily_returns.std(ddof=1) > 0.0
            else math.nan
        ),
        "trade_count": len(trades),
        "positive_trade_fraction": (
            float(trades["pnl"].gt(0.0).mean()) if len(trades) else math.nan
        ),
        "mean_trade_net_return": (
            float(trades["trade_net_return"].mean()) if len(trades) else math.nan
        ),
        "positive_year_count": int(annual["net_return"].gt(0.0).sum()),
        "negative_year_count": int(annual["net_return"].lt(0.0).sum()),
        "worst_year_return": float(annual["net_return"].min()),
        "total_costs": total_costs,
        "minimum_cash": float(equity["cash"].min()),
        "maximum_position_count": int(equity["position_count"].max()),
        "unresolved_position_count": 0,
        "forbidden_2026_read_count": 0,
        "annual": annual.to_dict("records"),
    }
    return result, equity, trades


def replay_oof_accounts(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    profile: str | None = None,
    ensemble_profiles: Sequence[str] | None = None,
    feature_variant: str = DEFAULT_FEATURE_VARIANT,
    candidate_only: bool = False,
    training_mode: str = "outer_early_stop",
) -> dict[str, Any]:
    study = load_study(study_path)
    if training_mode not in TRAINING_MODES:
        raise FullMarketForecastError(f"unknown_training_mode:{training_mode}")
    context = _source_context(study)
    evaluation = evaluate_oof_selections(
        study_path=study_path,
        output_root=output_root,
        profile=profile,
        ensemble_profiles=ensemble_profiles,
        feature_variant=feature_variant,
        candidate_only=candidate_only,
        training_mode=training_mode,
    )
    profile_name = str(evaluation["profile"])
    selection_path = Path(evaluation["files"]["selections"]["path"])
    selections = pd.read_parquet(selection_path)
    daily_raw = _open_array(
        context.pack["feature_channels"]["daily_raw"], dtype=np.float32
    )
    raw_open = _open_array(
        context.pack["execution_arrays"]["entry_open_raw"], dtype=np.float32
    )
    entry_filled = _open_array(context.pack["masks"]["entry_filled"], dtype=np.bool_)
    costs = parse_execution_costs(context.pack)
    specs = _account_specs(candidate_only=candidate_only)
    fingerprint = _stable_hash(
        {
            "schema": ACCOUNT_REPLAY_SCHEMA,
            "study_sha256": _sha256(study_path),
            "evaluation_fingerprint": evaluation["fingerprint"],
            "selection_sha256": evaluation["files"]["selections"]["sha256"],
            "specs": specs,
            "starting_cash": 1_000_000.0,
            "training_mode": training_mode,
            "feature_variant": feature_variant,
            "evaluation_scope": "candidate_only" if candidate_only else "full",
            "unfilled_selected_order": "cash_no_rank_substitution",
        }
    )
    replay_name = "account_replay"
    if feature_variant != DEFAULT_FEATURE_VARIANT:
        replay_name += f"_{feature_variant}"
    if profile_name != str(study["lightgbm"]["primary_profile"]):
        replay_name += f"_{profile_name}"
    if candidate_only:
        replay_name += "_candidate"
    if training_mode != "outer_early_stop":
        replay_name += f"_{training_mode}"
    replay_root = output_root / replay_name
    manifest_path = replay_root / "manifest.json"
    if manifest_path.is_file():
        current = _read_json(manifest_path)
        if (
            current.get("status") == "completed"
            and current.get("fingerprint") == fingerprint
        ):
            return current
    summaries: list[dict[str, Any]] = []
    files: dict[str, dict[str, Any]] = {}
    for spec in specs:
        result, equity, trades = _simulate_account_spec(
            spec=spec,
            selections=selections,
            context=context,
            daily_raw=daily_raw,
            raw_open=raw_open,
            entry_filled=entry_filled,
            costs=costs,
        )
        task_id = str(result["task_id"])
        task_root = replay_root / "tasks" / task_id
        equity_path = task_root / "equity.parquet"
        trades_path = task_root / "trades.parquet"
        result_path = task_root / "task_result.json"
        _write_parquet(equity, equity_path)
        _write_parquet(trades, trades_path)
        task_payload = {
            "schema": ACCOUNT_REPLAY_SCHEMA,
            "status": "completed",
            "completed_at": _now(),
            "study_id": STUDY_ID,
            "fingerprint": fingerprint,
            **result,
            "files": {
                "equity": _file_record(equity_path, row_count=len(equity)),
                "trades": _file_record(trades_path, row_count=len(trades)),
            },
        }
        _write_json(result_path, task_payload)
        files[task_id] = _file_record(result_path)
        summaries.append(result)
        _emit(
            "account_replay_task_completed",
            task_id=task_id,
            total_net_return=result["total_net_return"],
            maximum_drawdown=result["maximum_drawdown"],
        )
    summary_frame = pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key not in {"annual", "total_costs", "spec"}
            }
            | {
                "variant": result["spec"]["variant"],
                "exit_policy": result["spec"]["exit_policy"],
                "gate": result["spec"]["gate"],
                "top_k": result["spec"]["top_k"],
                "cost_scenario": result["spec"]["cost_scenario"],
            }
            for result in summaries
        ]
    ).sort_values(["total_net_return", "maximum_drawdown"], ascending=[False, False])
    summary_path = replay_root / "summary.parquet"
    _write_parquet(summary_frame, summary_path)
    manifest = {
        "schema": ACCOUNT_REPLAY_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "fingerprint": fingerprint,
        "feature_variant": feature_variant,
        "evaluation_scope": "candidate_only" if candidate_only else "full",
        "training_mode": training_mode,
        "task_count": len(summaries),
        "best_development_account": _json_safe(
            summaries[
                int(np.argmax([result["total_net_return"] for result in summaries]))
            ]
        ),
        "decision_boundary": {
            "historical_result_is_adaptive_not_independent_confirmation": True,
            "outer_validation_used_for_iteration_selection": (
                training_mode == "outer_early_stop"
            ),
            "stable_profit_claim_allowed": False,
            "no_leverage": True,
            "entry_fill_checked_before_buy": True,
            "unfilled_selected_order_is_cash_without_substitution": True,
            "cohort_equity_fraction": 0.5,
            "forbidden_2026_read_count": 0,
        },
        "sources": {
            "study": _file_record(study_path),
            "oof_evaluation": _file_record(
                Path(evaluation["files"]["selections"]["path"]).parent / "manifest.json"
            ),
            "selections": _file_record(selection_path),
        },
        "files": {
            "summary": _file_record(summary_path, row_count=len(summary_frame)),
            **files,
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def status(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    context = _source_context(study)
    target_path = output_root / "targets/manifest.json"
    exact_net_target_path = output_root / "exact_net_targets/manifest.json"
    target_ready = target_path.is_file()
    folds = build_forward_folds(
        date_idx=context.row_index["date_idx"].to_numpy(dtype=np.int32),
        trade_date=context.row_index["trade_date"].astype(str).to_numpy(),
    )
    tasks = list((output_root / "tasks").glob("*/task_result.json"))
    return {
        "study_id": STUDY_ID,
        "target_panel_ready": target_ready,
        "exact_net_target_panel_ready": exact_net_target_path.is_file(),
        "folds": folds,
        "completed_task_count": len(tasks),
        "resource_plan": asdict(resource_plan()),
    }


def run_baseline(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    folds: Sequence[int] = (1, 2, 3, 4, 5),
    targets: Sequence[str] = (
        "next_close_up",
        "market_excess_endpoint_return_d2_rank",
    ),
    profile: str | None = None,
    feature_variant: str = DEFAULT_FEATURE_VARIANT,
    training_mode: str = "outer_early_stop",
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for fold_number in folds:
        for target in targets:
            result = train_task(
                study_path=study_path,
                output_root=output_root,
                fold_number=int(fold_number),
                target=str(target),
                profile=profile,
                feature_variant=feature_variant,
                training_mode=training_mode,
            )
            results.append(
                {
                    "task_id": result["task_id"],
                    "fold": int(fold_number),
                    "target": str(target),
                    "best_iteration": int(result["best_iteration"]),
                    "metrics": dict(result["metrics"]),
                }
            )
            gc.collect()
            _trim_working_set()
    return {
        "status": "completed",
        "study_id": STUDY_ID,
        "feature_variant": feature_variant,
        "training_mode": training_mode,
        "task_count": len(results),
        "results": results,
    }


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def _parse_text_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare-targets",
            "prepare-exact-net-targets",
            "train-task",
            "status",
            "run-first",
            "run-baseline",
            "evaluate-payoff",
            "evaluate-market-regime",
            "evaluate-oof",
            "replay-account",
        ),
    )
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--target", default="next_close_up")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--ensemble-profiles", default="")
    parser.add_argument(
        "--feature-variant",
        choices=tuple(FEATURE_VARIANT_FAMILIES),
        default=DEFAULT_FEATURE_VARIANT,
    )
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument(
        "--training-mode",
        choices=TRAINING_MODES,
        default="outer_early_stop",
    )
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument(
        "--targets",
        default="next_close_up,market_excess_endpoint_return_d2_rank",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-targets":
        result = prepare_targets(study_path=args.study, output_root=args.output_root)
    elif args.command == "prepare-exact-net-targets":
        result = prepare_exact_net_targets(
            study_path=args.study, output_root=args.output_root
        )
    elif args.command == "train-task":
        result = train_task(
            study_path=args.study,
            output_root=args.output_root,
            fold_number=args.fold,
            target=args.target,
            profile=args.profile,
            feature_variant=args.feature_variant,
            training_mode=args.training_mode,
        )
    elif args.command == "run-first":
        result = train_task(
            study_path=args.study,
            output_root=args.output_root,
            fold_number=1,
            target=str(load_study(args.study)["outputs"]["first_training_targets"][0]),
            profile=args.profile,
            feature_variant=args.feature_variant,
            training_mode=args.training_mode,
        )
    elif args.command == "run-baseline":
        result = run_baseline(
            study_path=args.study,
            output_root=args.output_root,
            folds=_parse_int_csv(args.folds),
            targets=_parse_text_csv(args.targets),
            profile=args.profile,
            feature_variant=args.feature_variant,
            training_mode=args.training_mode,
        )
    elif args.command == "evaluate-payoff":
        result = evaluate_payoff_oof(
            study_path=args.study,
            output_root=args.output_root,
            target=args.target,
            profile=args.profile,
            feature_variant=args.feature_variant,
            training_mode=args.training_mode,
        )
    elif args.command == "evaluate-market-regime":
        result = evaluate_market_regime_oof(
            study_path=args.study,
            output_root=args.output_root,
            profile=args.profile,
            feature_variant=args.feature_variant,
            training_mode=args.training_mode,
        )
    elif args.command == "evaluate-oof":
        ensemble_profiles = _parse_text_csv(args.ensemble_profiles)
        result = evaluate_oof_selections(
            study_path=args.study,
            output_root=args.output_root,
            profile=args.profile,
            ensemble_profiles=ensemble_profiles or None,
            feature_variant=args.feature_variant,
            candidate_only=args.candidate_only,
            training_mode=args.training_mode,
        )
    elif args.command == "replay-account":
        ensemble_profiles = _parse_text_csv(args.ensemble_profiles)
        result = replay_oof_accounts(
            study_path=args.study,
            output_root=args.output_root,
            profile=args.profile,
            ensemble_profiles=ensemble_profiles or None,
            feature_variant=args.feature_variant,
            candidate_only=args.candidate_only,
            training_mode=args.training_mode,
        )
    else:
        result = status(study_path=args.study, output_root=args.output_root)
    print(
        json.dumps(
            _json_safe(result),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
