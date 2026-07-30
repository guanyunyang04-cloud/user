from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from scipy import stats

from daily_research.path_policy.seq100_candidate_execution import (
    ExecutionCosts,
    parse_execution_costs,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_v4_economic_realizability_v1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_v4_economic_realizability_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_v4_economic_realizability_v1"
)
DEFAULT_RECORD_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/research_records/seq100/seq100_v4_economic_realizability_v1"
)
SIGNAL_SCHEMA = "seq100_v4_economic_signal_book/v1"
TASK_SCHEMA = "seq100_v4_economic_task/v1"
SUMMARY_SCHEMA = "seq100_v4_economic_summary/v1"
YEARS = (2023, 2024, 2025)
PHYSICAL_COLUMNS = (
    "mfe_10",
    "mfe_20",
    "state_low_10",
    "state_mid_10",
    "state_high_10",
    "pre_peak_mae_10",
    "pre_peak_mae_20",
)
RANK_COLUMNS = (
    *PHYSICAL_COLUMNS,
    "dual_mfe_rank",
    "noise10_rank",
    "noise20_rank",
    "noise_dual_rank",
)
ORDER_COLUMNS = ("mfe10", "mfe20", "dual", "noise")
FORMAL_FAMILIES = (
    "mfe10_primary",
    "mfe20_primary",
    "dual_mfe_maximin",
    "dual_mfe_state_veto",
    "dual_mfe_risk_veto",
    "dual_mfe_state_risk_veto",
)
NEGATIVE_CONTROL = "deterministic_noise"
FAMILIES = (*FORMAL_FAMILIES, NEGATIVE_CONTROL)
EXPOSURE_MODES = ("target_full", "strong_candidate_cash")
SLOT_COUNTS = (1, 3, 6, 12, 24, 48)
BUFFER_MULTIPLIERS = (0.0, 0.5, 1.0, 2.0)
COST_SCENARIOS = ("base", "stress")
STARTING_CASH_CNY = 1_000_000.0
MINIMUM_AVAILABLE_BYTES = int(0.5 * 1024**3)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (
        (WORKSPACE_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    )


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(WORKSPACE_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported JSON type: {type(value).__name__}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
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
    os.replace(temporary, path)


def _save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, **extra: Any) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": _display(resolved),
        "size": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
        **extra,
    }


def _verify_record(record: Mapping[str, Any]) -> Path:
    path = _resolve(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if "size" in record and int(record["size"]) != int(path.stat().st_size):
        raise ValueError(f"file size changed: {path}")
    if "sha256" in record and str(record["sha256"]) != _sha256(path):
        raise ValueError(f"file hash changed: {path}")
    return path


def _emit(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {"event": event, "at": _now(), **payload},
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )


def _memory_guard() -> None:
    available = int(psutil.virtual_memory().available)
    if available < MINIMUM_AVAILABLE_BYTES:
        raise MemoryError("available memory is below 0.5 GiB")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _study_hash(path: Path) -> str:
    return _sha256(path)


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("study_id") != STUDY_ID:
        raise ValueError(f"study_id must be {STUDY_ID}")
    period = dict(payload["period"])
    if tuple(int(value) for value in period["signal_years"]) != YEARS:
        raise ValueError("signal years changed")
    if period["maximum_consumed_date"] != "2025-12-31":
        raise ValueError("maximum consumed date changed")
    if int(period["forbidden_year"]) != 2026:
        raise ValueError("2026 must remain forbidden")
    if bool(period["uses_2020_2022_policy_outcomes"]):
        raise ValueError("2020-2022 policy outcomes may not be used")
    account = dict(payload["account"])
    if not math.isclose(
        float(account["starting_cash_cny"]),
        STARTING_CASH_CNY,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("starting cash changed")
    if tuple(int(value) for value in account["slot_counts"]) != SLOT_COUNTS:
        raise ValueError("slot grid changed")
    policies = dict(payload["policies"])
    if tuple(policies["formal_families"]) != FORMAL_FAMILIES:
        raise ValueError("formal policy families changed")
    if policies["negative_control"] != NEGATIVE_CONTROL:
        raise ValueError("negative control changed")
    if tuple(policies["exposure_modes"]) != EXPOSURE_MODES:
        raise ValueError("exposure modes changed")
    if (
        tuple(float(value) for value in policies["buffer_multipliers"])
        != BUFFER_MULTIPLIERS
    ):
        raise ValueError("buffer grid changed")
    cost_scenarios = tuple(dict(payload["execution"]["cost_scenarios"]))
    if cost_scenarios != COST_SCENARIOS:
        raise ValueError("cost scenarios changed")
    if int(payload["outputs"]["expected_task_count"]) != 672:
        raise ValueError("expected task count changed")
    return payload


def _open_array(record: Mapping[str, Any], *, dtype: str | np.dtype) -> np.memmap:
    path = _resolve(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    shape = tuple(int(value) for value in record["shape"])
    return np.memmap(path, mode="r", dtype=dtype, shape=shape)


def _rank01(values: np.ndarray) -> np.ndarray:
    current = np.asarray(values, dtype=np.float64)
    if current.ndim != 1 or not bool(np.isfinite(current).all()):
        raise ValueError("rank input must be a finite vector")
    ranked = stats.rankdata(current, method="average")
    denominator = max(len(current) - 1, 1)
    return ((ranked - 1.0) / denominator).astype(np.float32)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    current = np.asarray(values, dtype=np.uint64)
    current = current + np.uint64(0x9E3779B97F4A7C15)
    current = (current ^ (current >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    current = (current ^ (current >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return current ^ (current >> np.uint64(31))


def _noise_rank(
    *,
    date_idx: int,
    symbol_idx: np.ndarray,
    seed: int,
) -> np.ndarray:
    symbols = np.asarray(symbol_idx, dtype=np.uint64)
    date_key = np.uint64((int(date_idx) * 0xD6E8FEB86659FD93) & ((1 << 64) - 1))
    key = symbols ^ date_key ^ np.uint64(int(seed))
    hashed = _splitmix64(key)
    order = np.argsort(hashed, kind="stable")
    output = np.empty(len(symbols), dtype=np.float32)
    denominator = max(len(symbols) - 1, 1)
    output[order] = np.arange(len(symbols), dtype=np.float32) / denominator
    return output


def _price_to_ticks(values: np.ndarray, tick_size: float = 0.01) -> np.ndarray:
    return np.floor(np.asarray(values, dtype=np.float64) / tick_size + 0.5)


def derive_open_sellable(
    *,
    raw_open: np.ndarray,
    raw_down_limit: np.ndarray,
    status_valid: np.ndarray,
    suspended: np.ndarray,
    delisted: np.ndarray,
) -> np.ndarray:
    open_values = np.asarray(raw_open, dtype=np.float64)
    limit_values = np.asarray(raw_down_limit, dtype=np.float64)
    output = (
        np.isfinite(open_values)
        & np.asarray(status_valid, dtype=bool)
        & ~np.asarray(suspended, dtype=bool)
        & ~np.asarray(delisted, dtype=bool)
    )
    blocked = (
        np.isfinite(limit_values)
        & output
        & (_price_to_ticks(open_values) <= _price_to_ticks(limit_values))
    )
    return output & ~blocked


def _source_array(
    pack: Mapping[str, Any],
    group: str,
    name: str,
    *,
    dtype: str | np.dtype,
) -> np.memmap:
    record = dict(pack[group][name])
    return _open_array(record, dtype=dtype)


def _signal_complete(
    output_root: Path,
    *,
    study_sha256: str | None = None,
) -> bool:
    manifest_path = output_root / "signal_book/manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = _load_json(manifest_path)
        if (
            manifest.get("schema") != SIGNAL_SCHEMA
            or manifest.get("status") != "completed"
        ):
            return False
        if (
            study_sha256 is not None
            and manifest.get("study_config_sha256") != study_sha256
        ):
            return False
        for record in dict(manifest["files"]).values():
            _verify_record(record)
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def prepare_signal_book(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    study_sha256 = _study_hash(study_path)
    if _signal_complete(output_root, study_sha256=study_sha256):
        return _load_json(output_root / "signal_book/manifest.json")
    _memory_guard()
    source = dict(study["sources"])
    contract_path = _resolve(source["entry_contract_manifest"])
    feature_path = _resolve(source["base_feature_manifest"])
    pack_path = _resolve(source["pack_manifest"])
    contract = _load_json(contract_path)
    feature = _load_json(feature_path)
    pack = _load_json(pack_path)
    if tuple(contract["physical_columns"]) != PHYSICAL_COLUMNS:
        raise ValueError("v4 physical columns changed")
    if contract.get("schema") != "seq100_entry_contract_oos_manifest/v4":
        raise ValueError("entry contract is not v4")
    if not bool(contract["semantics"]["mfe"]["rank_first"]):
        raise ValueError("v4 MFE must remain rank-first")
    post_entry = _load_json(_resolve(source["post_entry_result"]))
    if (
        post_entry["decision"]["overall"]
        != "daily_recomputation_v4_sufficient_within_tested_supervised_LightGBM_scope"
    ):
        raise ValueError("post-entry v2 decision changed")
    rows = np.load(
        _verify_record(contract["files"]["candidate_rows"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    ranks = np.load(
        _verify_record(contract["files"]["date_rank_predictions"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    if ranks.shape != (len(rows), len(PHYSICAL_COLUMNS)):
        raise ValueError("v4 rank shape changed")
    candidate_count = int(feature["candidate_alignment"]["candidate_count"])
    feature_files = dict(feature["files"])
    candidate_date_idx = np.memmap(
        _resolve(feature_files["candidate_date_idx"]["path"]),
        mode="r",
        dtype=np.int32,
        shape=(candidate_count,),
    )
    candidate_symbol_idx = np.memmap(
        _resolve(feature_files["candidate_symbol_idx"]["path"]),
        mode="r",
        dtype=np.int32,
        shape=(candidate_count,),
    )
    date_values = np.asarray(pack["date_values"], dtype=str)
    symbol_count = int(pack["symbol_count"])
    contract_rows = np.asarray(rows, dtype=np.int64)
    date_idx_all = np.asarray(candidate_date_idx[contract_rows], dtype=np.int32)
    symbol_idx_all = np.asarray(candidate_symbol_idx[contract_rows], dtype=np.int32)
    years_all = np.asarray(
        [int(str(date_values[int(value)])[:4]) for value in date_idx_all],
        dtype=np.int16,
    )
    keep = np.isin(years_all, np.asarray(YEARS, dtype=np.int16))
    if bool(np.any(years_all[keep] == 2026)):
        raise AssertionError("2026 entered the signal book")
    selected_dates = date_idx_all[keep]
    selected_symbols = symbol_idx_all[keep]
    selected_ranks = np.asarray(ranks[keep], dtype=np.float32)
    if not bool(np.isfinite(selected_ranks).all()):
        raise ValueError("v4 decision ranks contain non-finite values")
    signal_date_idx = np.unique(selected_dates)
    if not signal_date_idx.size:
        raise ValueError("signal book has no 2023-2025 dates")
    signal_date_values = date_values[signal_date_idx]
    if signal_date_values[0][:4] != "2023" or signal_date_values[-1] != "2025-12-31":
        raise ValueError("unexpected signal date coverage")
    n_days = len(signal_date_idx)
    panel = np.full(
        (n_days, symbol_count, len(RANK_COLUMNS)),
        np.nan,
        dtype=np.float32,
    )
    orders = np.full(
        (len(ORDER_COLUMNS), n_days, symbol_count),
        -1,
        dtype=np.int32,
    )
    candidate_counts = np.zeros(n_days, dtype=np.int32)
    date_to_signal = {int(value): idx for idx, value in enumerate(signal_date_idx)}
    boundaries = np.flatnonzero(
        np.r_[True, selected_dates[1:] != selected_dates[:-1], True]
    )
    noise_seeds = tuple(int(value) for value in study["policies"]["noise_seeds"])
    if len(noise_seeds) != 2:
        raise ValueError("two noise seeds are required")
    for start, stop in pairwise(boundaries):
        _memory_guard()
        date_idx = int(selected_dates[start])
        day = int(date_to_signal[date_idx])
        symbols = np.asarray(selected_symbols[start:stop], dtype=np.int32)
        if len(np.unique(symbols)) != len(symbols):
            raise ValueError(f"duplicate candidate symbol on {date_values[date_idx]}")
        values = np.asarray(selected_ranks[start:stop], dtype=np.float32)
        panel[day, symbols, : len(PHYSICAL_COLUMNS)] = values
        dual = _rank01(np.minimum(values[:, 0], values[:, 1]))
        noise10 = _noise_rank(
            date_idx=date_idx,
            symbol_idx=symbols,
            seed=noise_seeds[0],
        )
        noise20 = _noise_rank(
            date_idx=date_idx,
            symbol_idx=symbols,
            seed=noise_seeds[1],
        )
        noise_dual = _rank01(np.minimum(noise10, noise20))
        panel[day, symbols, 7] = dual
        panel[day, symbols, 8] = noise10
        panel[day, symbols, 9] = noise20
        panel[day, symbols, 10] = noise_dual
        selector_values = (values[:, 0], values[:, 1], dual, noise_dual)
        for order_idx, score in enumerate(selector_values):
            ranked = np.lexsort((symbols, -np.asarray(score, dtype=np.float64)))
            orders[order_idx, day, : len(symbols)] = symbols[ranked]
        candidate_counts[day] = len(symbols)
    if bool(np.any(candidate_counts <= 0)):
        raise ValueError("a signal date has no candidates")
    masks = dict(pack["masks"])
    execution = dict(pack["execution_arrays"])
    entry_filled = _open_array(masks["entry_filled"], dtype=np.bool_)
    raw_open = _open_array(execution["entry_open_raw"], dtype=np.float32)
    raw_down_limit = _open_array(execution["exit_down_limit_raw"], dtype=np.float32)
    status_valid = _open_array(masks["status_valid"], dtype=np.bool_)
    suspended = _open_array(masks["is_suspended"], dtype=np.bool_)
    delisted = _open_array(masks["is_delisted"], dtype=np.bool_)
    next_buyable = np.zeros((n_days, symbol_count), dtype=np.bool_)
    next_sellable = np.zeros((n_days, symbol_count), dtype=np.bool_)
    for day in range(n_days - 1):
        signal_idx = int(signal_date_idx[day])
        execution_idx = int(signal_date_idx[day + 1])
        if execution_idx != signal_idx + 1:
            raise ValueError("signal calendar contains a gap")
        next_buyable[day] = np.asarray(entry_filled[signal_idx], dtype=bool)
        next_sellable[day] = derive_open_sellable(
            raw_open=np.asarray(raw_open[execution_idx], dtype=np.float64),
            raw_down_limit=np.asarray(raw_down_limit[execution_idx], dtype=np.float64),
            status_valid=np.asarray(status_valid[execution_idx], dtype=bool),
            suspended=np.asarray(suspended[execution_idx], dtype=bool),
            delisted=np.asarray(delisted[execution_idx], dtype=bool),
        )
    daily_raw = _open_array(pack["feature_channels"]["daily_raw"], dtype=np.float32)
    adjusted_close = daily_raw[:, :, 3]
    terminal_recovery = float(
        dict(pack.get("terminal_execution", {}) or {}).get(
            "recovery_fraction_of_entry_notional", 0.0
        )
        or 0.0
    )
    benchmark_returns = np.zeros(n_days, dtype=np.float64)
    benchmark_missing_count = 0
    for day in range(1, n_days):
        previous_symbols = orders[0, day - 1, : candidate_counts[day - 1]]
        previous_idx = int(signal_date_idx[day - 1])
        current_idx = int(signal_date_idx[day])
        previous_close = np.asarray(
            adjusted_close[previous_idx, previous_symbols],
            dtype=np.float64,
        )
        current_close = np.asarray(
            adjusted_close[current_idx, previous_symbols],
            dtype=np.float64,
        )
        valid_previous = np.isfinite(previous_close) & (previous_close > 0.0)
        returns = np.full(
            len(previous_symbols), terminal_recovery - 1.0, dtype=np.float64
        )
        valid = valid_previous & np.isfinite(current_close) & (current_close > 0.0)
        returns[valid] = current_close[valid] / previous_close[valid] - 1.0
        benchmark_missing_count += int(np.sum(valid_previous & ~valid))
        benchmark_returns[day] = float(np.mean(returns[valid_previous]))
    signal_root = output_root / "signal_book"
    files: dict[str, dict[str, Any]] = {}
    arrays = {
        "signal_date_idx": signal_date_idx.astype(np.int32),
        "candidate_counts": candidate_counts,
        "rank_panel": panel,
        "candidate_orders": orders,
        "next_open_buyable": next_buyable,
        "next_open_sellable": next_sellable,
        "benchmark_returns": benchmark_returns,
    }
    for name, values in arrays.items():
        path = signal_root / f"{name}.npy"
        _save_npy(path, values)
        files[name] = _file_record(
            path,
            shape=list(values.shape),
            dtype=str(values.dtype),
        )
    manifest = {
        "schema": SIGNAL_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "study_config_sha256": study_sha256,
        "entry_contract": {
            "path": _display(contract_path),
            "sha256": _sha256(contract_path),
            "candidate_rows_sha256": contract["files"]["candidate_rows"]["sha256"],
            "date_rank_predictions_sha256": contract["files"]["date_rank_predictions"][
                "sha256"
            ],
        },
        "pack_manifest": {
            "path": _display(pack_path),
            "sha256": _sha256(pack_path),
        },
        "rank_columns": list(RANK_COLUMNS),
        "order_columns": list(ORDER_COLUMNS),
        "signal_date_count": int(n_days),
        "candidate_row_count": int(np.sum(candidate_counts)),
        "candidate_count_minimum": int(np.min(candidate_counts)),
        "candidate_count_median": float(np.median(candidate_counts)),
        "candidate_count_maximum": int(np.max(candidate_counts)),
        "first_signal_date": str(signal_date_values[0]),
        "last_signal_date": str(signal_date_values[-1]),
        "last_execution_date": "2025-12-31",
        "benchmark_missing_next_price_count": int(benchmark_missing_count),
        "forbidden_2026_row_count": 0,
        "uses_2020_2022_policy_outcomes": False,
        "files": files,
    }
    _write_json(signal_root / "manifest.json", manifest)
    _emit(
        "signal_book_completed",
        signal_dates=n_days,
        candidate_rows=int(np.sum(candidate_counts)),
    )
    return manifest


@dataclass(frozen=True)
class TaskSpec:
    family: str
    exposure_mode: str
    slot_count: int
    buffer_multiplier: float
    cost_scenario: str

    @property
    def task_id(self) -> str:
        buffer_text = str(self.buffer_multiplier).replace(".", "p")
        return (
            f"{self.family}__{self.exposure_mode}__k{self.slot_count:02d}"
            f"__b{buffer_text}__{self.cost_scenario}"
        )


@dataclass
class Position:
    symbol_idx: int
    shares: int
    entry_signal_date_idx: int
    entry_date_idx: int
    entry_raw_open: float
    entry_adjusted_open: float
    gross_entry_notional: float
    net_cash_outflow: float
    gross_cash_outflow: float
    last_adjusted_price: float


@dataclass(frozen=True)
class PendingSell:
    symbol_idx: int
    reason: str
    paired_buy_symbol_idx: int | None


@dataclass(frozen=True)
class PendingBuy:
    symbol_idx: int
    reason: str


@dataclass(frozen=True)
class PendingOrders:
    signal_day: int
    sells: tuple[PendingSell, ...]
    unpaired_buys: tuple[PendingBuy, ...]


class SignalBook:
    def __init__(
        self,
        *,
        study: Mapping[str, Any],
        output_root: Path,
    ) -> None:
        manifest_path = output_root / "signal_book/manifest.json"
        if not _signal_complete(output_root):
            raise FileNotFoundError("signal book is incomplete")
        self.manifest = _load_json(manifest_path)
        self.study = dict(study)
        files = dict(self.manifest["files"])
        self.signal_date_idx = np.load(
            _verify_record(files["signal_date_idx"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.candidate_counts = np.load(
            _verify_record(files["candidate_counts"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.rank_panel = np.load(
            _verify_record(files["rank_panel"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.orders = np.load(
            _verify_record(files["candidate_orders"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.next_buyable = np.load(
            _verify_record(files["next_open_buyable"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.next_sellable = np.load(
            _verify_record(files["next_open_sellable"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.benchmark_returns = np.load(
            _verify_record(files["benchmark_returns"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        source = dict(study["sources"])
        self.pack_path = _resolve(source["pack_manifest"])
        self.pack = _load_json(self.pack_path)
        self.date_values = np.asarray(self.pack["date_values"], dtype=str)
        self.symbol_values = np.asarray(self.pack["symbol_values"], dtype=str)
        self.symbol_count = int(self.pack["symbol_count"])
        self.raw_open = _source_array(
            self.pack,
            "execution_arrays",
            "entry_open_raw",
            dtype=np.float32,
        )
        self.daily_raw = _source_array(
            self.pack,
            "feature_channels",
            "daily_raw",
            dtype=np.float32,
        )
        self.adjusted_open = self.daily_raw[:, :, 0]
        self.adjusted_close = self.daily_raw[:, :, 3]
        self.amount = self.daily_raw[:, :, 5]
        self.status_valid = _source_array(
            self.pack,
            "masks",
            "status_valid",
            dtype=np.bool_,
        )
        self.has_bar = _source_array(
            self.pack,
            "masks",
            "has_bar",
            dtype=np.bool_,
        )
        self.is_delisted = _source_array(
            self.pack,
            "masks",
            "is_delisted",
            dtype=np.bool_,
        )
        self.costs = parse_execution_costs(self.pack)
        self._validate_costs()
        self.terminal_recovery_fraction = float(
            dict(self.pack.get("terminal_execution", {}) or {}).get(
                "recovery_fraction_of_entry_notional", 0.0
            )
            or 0.0
        )
        if int(self.signal_date_idx[-1]) >= len(self.date_values):
            raise ValueError("signal date index exceeds the pack calendar")
        if str(self.date_values[int(self.signal_date_idx[-1])]) != "2025-12-31":
            raise ValueError("signal book does not end at 2025-12-31")

    def _validate_costs(self) -> None:
        expected = dict(self.study["execution"]["expected_pack_costs"])
        comparisons = {
            "lot_size": self.costs.lot_size,
            "commission_bps": self.costs.commission_bps,
            "minimum_commission_cny": self.costs.minimum_commission_cny,
            "transfer_fee_bps": self.costs.transfer_fee_bps,
            "slippage_bps": self.costs.slippage_bps,
            "stress_slippage_multiplier": self.costs.stress_slippage_multiplier,
        }
        for name, observed in comparisons.items():
            wanted = expected[name]
            if isinstance(observed, int):
                if int(observed) != int(wanted):
                    raise ValueError(f"execution cost {name} changed")
            elif not math.isclose(
                float(observed),
                float(wanted),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(f"execution cost {name} changed")
        expected_schedule = tuple(
            (
                pd.Timestamp(item["effective_date"]).strftime("%Y-%m-%d"),
                float(item["stamp_tax_bps"]),
            )
            for item in expected["stamp_tax_schedule"]
        )
        if tuple(self.costs.stamp_tax_schedule) != expected_schedule:
            raise ValueError("stamp-tax schedule changed")

    @property
    def day_count(self) -> int:
        return len(self.signal_date_idx)

    def date_text(self, day: int) -> str:
        return str(self.date_values[int(self.signal_date_idx[int(day)])])

    def selector_column(self, family: str) -> int:
        if family == "mfe10_primary":
            return 0
        if family == "mfe20_primary":
            return 1
        if family in {
            "dual_mfe_maximin",
            "dual_mfe_state_veto",
            "dual_mfe_risk_veto",
            "dual_mfe_state_risk_veto",
        }:
            return 7
        if family == NEGATIVE_CONTROL:
            return 10
        raise ValueError(f"unknown family: {family}")

    def order_column(self, family: str) -> int:
        if family == "mfe10_primary":
            return 0
        if family == "mfe20_primary":
            return 1
        if family.startswith("dual_mfe_"):
            return 2
        if family == NEGATIVE_CONTROL:
            return 3
        raise ValueError(f"unknown family: {family}")

    def symbols_for_day(self, family: str, day: int) -> np.ndarray:
        count = int(self.candidate_counts[int(day)])
        return np.asarray(
            self.orders[self.order_column(family), int(day), :count],
            dtype=np.int32,
        )

    def rank(self, day: int, symbol_idx: int, column: int) -> float:
        return float(self.rank_panel[int(day), int(symbol_idx), int(column)])

    def candidate_count(self, day: int) -> int:
        return int(self.candidate_counts[int(day)])

    def trailing_amount(self, *, signal_date_idx: int, symbol_idx: int) -> float:
        start = max(0, int(signal_date_idx) - 19)
        values = np.asarray(
            self.amount[start : int(signal_date_idx) + 1, int(symbol_idx)],
            dtype=np.float64,
        )
        valid = values[np.isfinite(values) & (values > 0.0)]
        return float(np.median(valid)) if valid.size else math.nan


def task_specs() -> list[TaskSpec]:
    output = [
        TaskSpec(
            family=family,
            exposure_mode=exposure,
            slot_count=slots,
            buffer_multiplier=buffer,
            cost_scenario=cost,
        )
        for family in FAMILIES
        for exposure in EXPOSURE_MODES
        for slots in SLOT_COUNTS
        for buffer in BUFFER_MULTIPLIERS
        for cost in COST_SCENARIOS
    ]
    if len(output) != 672 or len({item.task_id for item in output}) != len(output):
        raise AssertionError("task inventory changed")
    return output


def replacement_buffer(
    *,
    multiplier: float,
    slot_count: int,
    candidate_count: int,
) -> float:
    if slot_count <= 0 or candidate_count <= 0 or multiplier < 0.0:
        raise ValueError("invalid replacement-buffer input")
    return float(multiplier) * int(slot_count) / max(int(candidate_count) - 1, 1)


def _stamp_tax_bps(costs: ExecutionCosts, trade_date: str) -> float:
    matches = [
        float(rate)
        for effective_date, rate in costs.stamp_tax_schedule
        if str(effective_date) <= str(trade_date)
    ]
    return float(matches[-1]) if matches else 0.0


def _position_value(position: Position, adjusted_price: float) -> float:
    if (
        not math.isfinite(adjusted_price)
        or adjusted_price <= 0.0
        or not math.isfinite(position.entry_adjusted_open)
        or position.entry_adjusted_open <= 0.0
    ):
        return float(
            position.gross_entry_notional
            * position.last_adjusted_price
            / position.entry_adjusted_open
        )
    position.last_adjusted_price = float(adjusted_price)
    return float(
        position.gross_entry_notional
        * float(adjusted_price)
        / position.entry_adjusted_open
    )


def _portfolio_value(
    *,
    cash: float,
    positions: Mapping[int, Position],
    adjusted_prices: np.ndarray,
) -> float:
    total = float(cash)
    for symbol_idx, position in positions.items():
        total += _position_value(position, float(adjusted_prices[int(symbol_idx)]))
    return float(total)


def _buy_position(
    *,
    available_cash: float,
    allocated_cash: float,
    symbol_idx: int,
    signal_date_idx: int,
    execution_date_idx: int,
    raw_open: float,
    adjusted_open: float,
    costs: ExecutionCosts,
    slippage_multiplier: float,
) -> tuple[Position | None, dict[str, float]]:
    if (
        available_cash <= 0.0
        or allocated_cash <= 0.0
        or not math.isfinite(raw_open)
        or raw_open <= 0.0
        or not math.isfinite(adjusted_open)
        or adjusted_open <= 0.0
    ):
        return None, {}
    allocation = min(float(available_cash), float(allocated_cash))
    slippage_rate = float(costs.slippage_bps) * float(slippage_multiplier) / 10_000.0
    fill_price = float(raw_open) * (1.0 + slippage_rate)
    shares = int(
        math.floor(allocation / (fill_price * costs.lot_size)) * costs.lot_size
    )
    while shares > 0:
        fill_notional = shares * fill_price
        commission = max(
            float(costs.minimum_commission_cny),
            fill_notional * float(costs.commission_bps) / 10_000.0,
        )
        transfer = fill_notional * float(costs.transfer_fee_bps) / 10_000.0
        cash_outflow = fill_notional + commission + transfer
        if cash_outflow <= min(float(available_cash), allocation) + 1.0e-9:
            break
        shares -= int(costs.lot_size)
    if shares <= 0:
        return None, {}
    gross_notional = float(shares) * float(raw_open)
    fill_notional = float(shares) * fill_price
    commission = max(
        float(costs.minimum_commission_cny),
        fill_notional * float(costs.commission_bps) / 10_000.0,
    )
    transfer = fill_notional * float(costs.transfer_fee_bps) / 10_000.0
    slippage = fill_notional - gross_notional
    cash_outflow = fill_notional + commission + transfer
    position = Position(
        symbol_idx=int(symbol_idx),
        shares=int(shares),
        entry_signal_date_idx=int(signal_date_idx),
        entry_date_idx=int(execution_date_idx),
        entry_raw_open=float(raw_open),
        entry_adjusted_open=float(adjusted_open),
        gross_entry_notional=float(gross_notional),
        net_cash_outflow=float(cash_outflow),
        gross_cash_outflow=float(gross_notional),
        last_adjusted_price=float(adjusted_open),
    )
    details = {
        "gross_notional": float(gross_notional),
        "fill_notional": float(fill_notional),
        "commission": float(commission),
        "transfer_fee": float(transfer),
        "stamp_tax": 0.0,
        "slippage": float(slippage),
        "cash_flow": float(-cash_outflow),
        "total_cost": float(commission + transfer + slippage),
    }
    return position, details


def _sell_position(
    *,
    position: Position,
    adjusted_open: float,
    trade_date: str,
    costs: ExecutionCosts,
    slippage_multiplier: float,
) -> tuple[float, dict[str, float]]:
    gross_value = _position_value(position, float(adjusted_open))
    slippage_rate = float(costs.slippage_bps) * float(slippage_multiplier) / 10_000.0
    fill_notional = gross_value * max(0.0, 1.0 - slippage_rate)
    commission = max(
        float(costs.minimum_commission_cny),
        fill_notional * float(costs.commission_bps) / 10_000.0,
    )
    transfer = fill_notional * float(costs.transfer_fee_bps) / 10_000.0
    stamp = fill_notional * _stamp_tax_bps(costs, trade_date) / 10_000.0
    proceeds = fill_notional - commission - transfer - stamp
    slippage = gross_value - fill_notional
    details = {
        "gross_notional": float(gross_value),
        "fill_notional": float(fill_notional),
        "commission": float(commission),
        "transfer_fee": float(transfer),
        "stamp_tax": float(stamp),
        "slippage": float(slippage),
        "cash_flow": float(proceeds),
        "total_cost": float(commission + transfer + stamp + slippage),
    }
    return float(proceeds), details


def _gate_flags(family: str) -> tuple[bool, bool]:
    return (
        family in {"dual_mfe_state_veto", "dual_mfe_state_risk_veto"},
        family in {"dual_mfe_risk_veto", "dual_mfe_state_risk_veto"},
    )


def _entry_threshold_pass(
    *,
    book: SignalBook,
    spec: TaskSpec,
    day: int,
    symbol_idx: int,
    threshold: float,
) -> bool:
    if spec.exposure_mode == "target_full":
        return True
    if spec.family == "mfe10_primary":
        return book.rank(day, symbol_idx, 0) >= threshold
    if spec.family == "mfe20_primary":
        return book.rank(day, symbol_idx, 1) >= threshold
    if spec.family == NEGATIVE_CONTROL:
        return (
            book.rank(day, symbol_idx, 8) >= threshold
            and book.rank(day, symbol_idx, 9) >= threshold
        )
    return (
        book.rank(day, symbol_idx, 0) >= threshold
        and book.rank(day, symbol_idx, 1) >= threshold
    )


def _daily_gate_set(
    *,
    book: SignalBook,
    spec: TaskSpec,
    day: int,
    ordered_symbols: np.ndarray,
) -> set[int] | None:
    use_state, use_risk = _gate_flags(spec.family)
    if not use_state and not use_risk:
        return None
    top_fraction = float(book.study["policies"]["auxiliary_top_fraction"])
    retention = float(book.study["policies"]["auxiliary_retention_fraction"])
    pool_count = max(1, math.ceil(len(ordered_symbols) * top_fraction))
    pool = np.asarray(ordered_symbols[:pool_count], dtype=np.int32)
    keep = np.ones(len(pool), dtype=bool)
    if use_state:
        state_low = np.asarray(book.rank_panel[day, pool, 2], dtype=np.float64)
        state_order = np.lexsort((pool, state_low))
        retain_count = max(1, math.ceil(len(pool) * retention))
        state_keep = np.zeros(len(pool), dtype=bool)
        state_keep[state_order[:retain_count]] = True
        keep &= state_keep
    if use_risk:
        safety = np.minimum(
            np.asarray(book.rank_panel[day, pool, 5], dtype=np.float64),
            np.asarray(book.rank_panel[day, pool, 6], dtype=np.float64),
        )
        risk_order = np.lexsort((pool, -safety))
        retain_count = max(1, math.ceil(len(pool) * retention))
        risk_keep = np.zeros(len(pool), dtype=bool)
        risk_keep[risk_order[:retain_count]] = True
        keep &= risk_keep
    return {int(value) for value in pool[keep]}


def _pair_gate_pass(
    *,
    book: SignalBook,
    spec: TaskSpec,
    day: int,
    candidate: int,
    incumbent: int,
) -> bool:
    use_state, use_risk = _gate_flags(spec.family)
    tolerance = float(book.study["policies"]["auxiliary_pair_tolerance"])
    if use_state and (
        book.rank(day, candidate, 2) > book.rank(day, incumbent, 2) + tolerance
    ):
        return False
    if use_risk:
        candidate_safety = min(
            book.rank(day, candidate, 5),
            book.rank(day, candidate, 6),
        )
        incumbent_safety = min(
            book.rank(day, incumbent, 5),
            book.rank(day, incumbent, 6),
        )
        if candidate_safety < incumbent_safety - tolerance:
            return False
    return True


def plan_orders(
    *,
    book: SignalBook,
    spec: TaskSpec,
    day: int,
    positions: Mapping[int, Position],
) -> PendingOrders:
    selector_column = book.selector_column(spec.family)
    ordered = book.symbols_for_day(spec.family, day)
    gate_set = _daily_gate_set(
        book=book,
        spec=spec,
        day=day,
        ordered_symbols=ordered,
    )
    held = {int(value) for value in positions}
    forced: list[tuple[int, str]] = []
    retained: list[int] = []
    retention_threshold = float(book.study["policies"]["strong_retention_rank"])
    for symbol_idx in sorted(held):
        score = book.rank(day, symbol_idx, selector_column)
        if not math.isfinite(score):
            forced.append((symbol_idx, "missing_v4_signal"))
        elif not _entry_threshold_pass(
            book=book,
            spec=spec,
            day=day,
            symbol_idx=symbol_idx,
            threshold=retention_threshold,
        ):
            forced.append((symbol_idx, "below_retention_threshold"))
        else:
            retained.append(symbol_idx)
    entry_threshold = float(book.study["policies"]["strong_entry_rank"])
    candidate_list: list[int] = []
    if gate_set is None:
        # At most K unheld symbols can be consumed.  The first |held| + K
        # ranked names therefore contain every symbol that can affect the
        # deterministic greedy plan.
        scan_count = min(len(ordered), len(held) + int(spec.slot_count))
    else:
        # Auxiliary vetoes are defined only inside the selector's Top 5% pool.
        scan_count = max(
            1,
            math.ceil(
                len(ordered) * float(book.study["policies"]["auxiliary_top_fraction"])
            ),
        )
    for raw_symbol in ordered[:scan_count]:
        symbol_idx = int(raw_symbol)
        if symbol_idx in held:
            continue
        if gate_set is not None and symbol_idx not in gate_set:
            continue
        if not _entry_threshold_pass(
            book=book,
            spec=spec,
            day=day,
            symbol_idx=symbol_idx,
            threshold=entry_threshold,
        ):
            continue
        candidate_list.append(symbol_idx)
    reserved: set[int] = set()

    def take_candidate(incumbent: int | None = None) -> int | None:
        for symbol_idx in candidate_list:
            if symbol_idx in reserved:
                continue
            if incumbent is not None and not _pair_gate_pass(
                book=book,
                spec=spec,
                day=day,
                candidate=symbol_idx,
                incumbent=incumbent,
            ):
                continue
            reserved.add(symbol_idx)
            return symbol_idx
        return None

    unpaired: list[PendingBuy] = []
    current_empty = max(int(spec.slot_count) - len(positions), 0)
    for _ in range(current_empty):
        candidate = take_candidate()
        if candidate is None:
            break
        unpaired.append(PendingBuy(symbol_idx=candidate, reason="empty_slot"))
    sells: list[PendingSell] = []
    for symbol_idx, reason in forced:
        candidate = take_candidate()
        sells.append(
            PendingSell(
                symbol_idx=int(symbol_idx),
                reason=str(reason),
                paired_buy_symbol_idx=candidate,
            )
        )
    candidate_count = book.candidate_count(day)
    buffer_value = replacement_buffer(
        multiplier=spec.buffer_multiplier,
        slot_count=spec.slot_count,
        candidate_count=candidate_count,
    )
    retained_sorted = sorted(
        retained,
        key=lambda symbol: (
            book.rank(day, symbol, selector_column),
            int(symbol),
        ),
    )
    for incumbent in retained_sorted:
        incumbent_score = book.rank(day, incumbent, selector_column)
        candidate: int | None = None
        for proposed in candidate_list:
            if proposed in reserved:
                continue
            proposed_score = book.rank(day, proposed, selector_column)
            if proposed_score - incumbent_score + 1.0e-12 < buffer_value:
                # Remaining candidates cannot improve because candidate_list is ordered.
                break
            if not _pair_gate_pass(
                book=book,
                spec=spec,
                day=day,
                candidate=proposed,
                incumbent=incumbent,
            ):
                continue
            candidate = proposed
            reserved.add(proposed)
            break
        if candidate is None:
            continue
        sells.append(
            PendingSell(
                symbol_idx=int(incumbent),
                reason="rank_replacement",
                paired_buy_symbol_idx=int(candidate),
            )
        )
    return PendingOrders(
        signal_day=int(day),
        sells=tuple(sells),
        unpaired_buys=tuple(unpaired),
    )


def _drawdown(values: np.ndarray) -> float:
    current = np.asarray(values, dtype=np.float64)
    if not current.size or not bool(np.isfinite(current).all()):
        return math.nan
    running = np.maximum.accumulate(current)
    return float(np.min(current / running - 1.0))


def _annualized_metrics(
    *,
    equity: np.ndarray,
    daily_returns: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(equity, dtype=np.float64)
    returns = np.asarray(daily_returns, dtype=np.float64)
    if len(values) < 2 or not bool(np.isfinite(values).all()):
        raise ValueError("equity path is invalid")
    periods = max(len(returns) - 1, 1)
    cumulative = float(values[-1] / values[0] - 1.0)
    annualized = float((values[-1] / values[0]) ** (252.0 / periods) - 1.0)
    active = returns[1:]
    volatility = float(np.std(active, ddof=1) * math.sqrt(252.0))
    mean = float(np.mean(active))
    std = float(np.std(active, ddof=1))
    sharpe = float(mean / std * math.sqrt(252.0)) if std > 0.0 else math.nan
    downside = active[active < 0.0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) >= 2 else 0.0
    sortino = (
        float(mean / downside_std * math.sqrt(252.0))
        if downside_std > 0.0
        else math.nan
    )
    maximum_drawdown = _drawdown(values)
    calmar = (
        float(annualized / abs(maximum_drawdown))
        if math.isfinite(maximum_drawdown) and maximum_drawdown < 0.0
        else math.nan
    )
    return {
        "cumulative_return": cumulative,
        "cagr": annualized,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "maximum_drawdown": maximum_drawdown,
        "calmar": calmar,
    }


def _year_metrics(
    *,
    dates: np.ndarray,
    equity: np.ndarray,
    benchmark_wealth: np.ndarray,
) -> list[dict[str, Any]]:
    date_values = np.asarray(dates, dtype=str)
    output: list[dict[str, Any]] = []
    previous_equity = float(STARTING_CASH_CNY)
    previous_benchmark = float(STARTING_CASH_CNY)
    for year in YEARS:
        rows = np.flatnonzero(np.char.startswith(date_values, str(year)))
        if not rows.size:
            raise ValueError(f"missing calendar year {year}")
        ending_equity = float(equity[int(rows[-1])])
        ending_benchmark = float(benchmark_wealth[int(rows[-1])])
        annual_return = ending_equity / previous_equity - 1.0
        benchmark_return = ending_benchmark / previous_benchmark - 1.0
        excess = (1.0 + annual_return) / max(1.0 + benchmark_return, 1.0e-12) - 1.0
        output.append(
            {
                "year": int(year),
                "net_return": float(annual_return),
                "benchmark_return": float(benchmark_return),
                "relative_excess_return": float(excess),
                "ending_equity": ending_equity,
            }
        )
        previous_equity = ending_equity
        previous_benchmark = ending_benchmark
    return output


def _monthly_metrics(
    *,
    dates: np.ndarray,
    equity: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(np.asarray(dates, dtype=str)),
            "equity": np.asarray(equity, dtype=np.float64),
        }
    )
    frame["month"] = frame["trade_date"].dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    previous = float(STARTING_CASH_CNY)
    for month, part in frame.groupby("month", sort=True):
        ending = float(part["equity"].iloc[-1])
        rows.append(
            {
                "month": str(month),
                "net_return": float(ending / previous - 1.0),
                "ending_equity": ending,
            }
        )
        previous = ending
    return pd.DataFrame(rows)


def _attempt_row(
    *,
    task_id: str,
    side: str,
    status: str,
    signal_date: str,
    execution_date: str,
    symbol_idx: int,
    symbol: str,
    reason: str,
    details: Mapping[str, float] | None = None,
    holding_days: int | None = None,
    sell_delay_days: int | None = None,
    participation: float = math.nan,
) -> dict[str, Any]:
    values = dict(details or {})
    return {
        "task_id": task_id,
        "side": str(side),
        "status": str(status),
        "signal_date": str(signal_date),
        "execution_date": str(execution_date),
        "symbol_idx": int(symbol_idx),
        "symbol": str(symbol),
        "reason": str(reason),
        "holding_days": (int(holding_days) if holding_days is not None else pd.NA),
        "sell_delay_days": (
            int(sell_delay_days) if sell_delay_days is not None else pd.NA
        ),
        "gross_notional": float(values.get("gross_notional", math.nan)),
        "fill_notional": float(values.get("fill_notional", math.nan)),
        "commission": float(values.get("commission", 0.0)),
        "transfer_fee": float(values.get("transfer_fee", 0.0)),
        "stamp_tax": float(values.get("stamp_tax", 0.0)),
        "slippage": float(values.get("slippage", 0.0)),
        "cash_flow": float(values.get("cash_flow", 0.0)),
        "total_cost": float(values.get("total_cost", 0.0)),
        "participation_of_trailing20_median_amount": float(participation),
    }


def simulate_task(
    *,
    book: SignalBook,
    spec: TaskSpec,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    multiplier = float(
        dict(book.study["execution"]["cost_scenarios"])[spec.cost_scenario]
    )
    cash = float(book.study["account"]["starting_cash_cny"])
    gross_cash = cash
    positions: dict[int, Position] = {}
    pending: PendingOrders | None = None
    total_cost = 0.0
    cost_parts = defaultdict(float)
    traded_notional = 0.0
    attempts: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    holding_days: list[int] = []
    sell_delay_days: list[int] = []
    consecutive_blocked_sells: dict[int, int] = {}
    realized_contribution = defaultdict(float)
    counters = defaultdict(int)
    maximum_conservation_error = 0.0
    benchmark_wealth = float(book.study["account"]["starting_cash_cny"])
    previous_equity = float(book.study["account"]["starting_cash_cny"])
    previous_gross_equity = previous_equity
    for day in range(book.day_count):
        _memory_guard()
        date_idx = int(book.signal_date_idx[day])
        trade_date = book.date_text(day)
        adjusted_open_row = np.asarray(book.adjusted_open[date_idx], dtype=np.float64)
        if pending is not None:
            if int(pending.signal_day) != day - 1:
                raise AssertionError("pending orders skipped a signal day")
            signal_idx = int(book.signal_date_idx[pending.signal_day])
            signal_date = book.date_text(pending.signal_day)
            equity_open = _portfolio_value(
                cash=cash,
                positions=positions,
                adjusted_prices=adjusted_open_row,
            )
            successful_pairs: list[PendingBuy] = []
            for order in pending.sells:
                symbol_idx = int(order.symbol_idx)
                symbol = str(book.symbol_values[symbol_idx])
                position = positions.get(symbol_idx)
                if position is None:
                    counters["stale_sell_order_count"] += 1
                    continue
                sellable = bool(book.next_sellable[pending.signal_day, symbol_idx])
                t_plus_one = date_idx > int(position.entry_date_idx)
                if not sellable or not t_plus_one:
                    counters["failed_sell_count"] += 1
                    consecutive_blocked_sells[symbol_idx] = (
                        consecutive_blocked_sells.get(symbol_idx, 0) + 1
                    )
                    if not sellable:
                        counters["blocked_sell_count"] += 1
                    attempts.append(
                        _attempt_row(
                            task_id=spec.task_id,
                            side="sell",
                            status="failed",
                            signal_date=signal_date,
                            execution_date=trade_date,
                            symbol_idx=symbol_idx,
                            symbol=symbol,
                            reason=order.reason,
                        )
                    )
                    continue
                adjusted_open = float(adjusted_open_row[symbol_idx])
                proceeds, details = _sell_position(
                    position=position,
                    adjusted_open=adjusted_open,
                    trade_date=trade_date,
                    costs=book.costs,
                    slippage_multiplier=multiplier,
                )
                cash += proceeds
                gross_cash += float(details["gross_notional"])
                total_cost += float(details["total_cost"])
                traded_notional += float(details["fill_notional"])
                for name in ("commission", "transfer_fee", "stamp_tax", "slippage"):
                    cost_parts[name] += float(details[name])
                duration = int(date_idx - position.entry_date_idx)
                holding_days.append(duration)
                sell_delay = int(consecutive_blocked_sells.pop(symbol_idx, 0))
                if sell_delay > 0:
                    sell_delay_days.append(sell_delay)
                pnl = float(proceeds - position.net_cash_outflow)
                realized_contribution[symbol_idx] += pnl
                del positions[symbol_idx]
                counters["sell_count"] += 1
                trailing = book.trailing_amount(
                    signal_date_idx=signal_idx,
                    symbol_idx=symbol_idx,
                )
                participation = (
                    float(details["fill_notional"]) / trailing
                    if math.isfinite(trailing) and trailing > 0.0
                    else math.nan
                )
                attempts.append(
                    _attempt_row(
                        task_id=spec.task_id,
                        side="sell",
                        status="filled",
                        signal_date=signal_date,
                        execution_date=trade_date,
                        symbol_idx=symbol_idx,
                        symbol=symbol,
                        reason=order.reason,
                        details=details,
                        holding_days=duration,
                        sell_delay_days=sell_delay,
                        participation=participation,
                    )
                )
                if order.paired_buy_symbol_idx is not None:
                    successful_pairs.append(
                        PendingBuy(
                            symbol_idx=int(order.paired_buy_symbol_idx),
                            reason=f"paired_after_{order.reason}",
                        )
                    )
            buy_orders = [*pending.unpaired_buys, *successful_pairs]
            for order in buy_orders:
                symbol_idx = int(order.symbol_idx)
                symbol = str(book.symbol_values[symbol_idx])
                if len(positions) >= int(spec.slot_count) or symbol_idx in positions:
                    counters["skipped_buy_slot_count"] += 1
                    attempts.append(
                        _attempt_row(
                            task_id=spec.task_id,
                            side="buy",
                            status="skipped_slot",
                            signal_date=signal_date,
                            execution_date=trade_date,
                            symbol_idx=symbol_idx,
                            symbol=symbol,
                            reason=order.reason,
                        )
                    )
                    continue
                buyable = bool(book.next_buyable[pending.signal_day, symbol_idx])
                if not buyable:
                    counters["failed_buy_count"] += 1
                    attempts.append(
                        _attempt_row(
                            task_id=spec.task_id,
                            side="buy",
                            status="failed",
                            signal_date=signal_date,
                            execution_date=trade_date,
                            symbol_idx=symbol_idx,
                            symbol=symbol,
                            reason=order.reason,
                        )
                    )
                    continue
                raw_open = float(book.raw_open[date_idx, symbol_idx])
                adjusted_open = float(adjusted_open_row[symbol_idx])
                allocation = min(
                    cash,
                    max(equity_open, 0.0) / int(spec.slot_count),
                )
                position, details = _buy_position(
                    available_cash=cash,
                    allocated_cash=allocation,
                    symbol_idx=symbol_idx,
                    signal_date_idx=signal_idx,
                    execution_date_idx=date_idx,
                    raw_open=raw_open,
                    adjusted_open=adjusted_open,
                    costs=book.costs,
                    slippage_multiplier=multiplier,
                )
                if position is None:
                    counters["failed_buy_cash_or_lot_count"] += 1
                    attempts.append(
                        _attempt_row(
                            task_id=spec.task_id,
                            side="buy",
                            status="failed_cash_or_lot",
                            signal_date=signal_date,
                            execution_date=trade_date,
                            symbol_idx=symbol_idx,
                            symbol=symbol,
                            reason=order.reason,
                        )
                    )
                    continue
                cash += float(details["cash_flow"])
                gross_cash -= float(details["gross_notional"])
                total_cost += float(details["total_cost"])
                traded_notional += float(details["fill_notional"])
                for name in ("commission", "transfer_fee", "stamp_tax", "slippage"):
                    cost_parts[name] += float(details[name])
                positions[symbol_idx] = position
                counters["buy_count"] += 1
                trailing = book.trailing_amount(
                    signal_date_idx=signal_idx,
                    symbol_idx=symbol_idx,
                )
                participation = (
                    float(details["fill_notional"]) / trailing
                    if math.isfinite(trailing) and trailing > 0.0
                    else math.nan
                )
                attempts.append(
                    _attempt_row(
                        task_id=spec.task_id,
                        side="buy",
                        status="filled",
                        signal_date=signal_date,
                        execution_date=trade_date,
                        symbol_idx=symbol_idx,
                        symbol=symbol,
                        reason=order.reason,
                        details=details,
                        participation=participation,
                    )
                )
        adjusted_close_row = np.asarray(book.adjusted_close[date_idx], dtype=np.float64)
        if day == book.day_count - 1:
            terminal_symbols = [
                int(symbol_idx)
                for symbol_idx in positions
                if bool(book.is_delisted[date_idx, int(symbol_idx)])
                or (
                    not bool(book.status_valid[date_idx, int(symbol_idx)])
                    and not bool(book.has_bar[date_idx, int(symbol_idx)])
                )
            ]
            for symbol_idx in terminal_symbols:
                position = positions.pop(symbol_idx)
                recovery = float(
                    position.gross_entry_notional * book.terminal_recovery_fraction
                )
                cash += recovery
                gross_cash += recovery
                duration = int(date_idx - position.entry_date_idx)
                holding_days.append(duration)
                realized_contribution[symbol_idx] += float(
                    recovery - position.net_cash_outflow
                )
                consecutive_blocked_sells.pop(symbol_idx, None)
                counters["terminal_recovery_count"] += 1
                counters["terminal_recovery_notional"] += recovery
                attempts.append(
                    _attempt_row(
                        task_id=spec.task_id,
                        side="terminal",
                        status="recovery",
                        signal_date=trade_date,
                        execution_date=trade_date,
                        symbol_idx=symbol_idx,
                        symbol=str(book.symbol_values[symbol_idx]),
                        reason="pack_terminal_lifecycle_recovery",
                        details={
                            "gross_notional": recovery,
                            "fill_notional": recovery,
                            "cash_flow": recovery,
                            "total_cost": 0.0,
                        },
                        holding_days=duration,
                    )
                )
        equity = _portfolio_value(
            cash=cash,
            positions=positions,
            adjusted_prices=adjusted_close_row,
        )
        gross_equity = _portfolio_value(
            cash=gross_cash,
            positions=positions,
            adjusted_prices=adjusted_close_row,
        )
        conservation_error = abs((gross_equity - equity) - total_cost)
        maximum_conservation_error = max(
            maximum_conservation_error,
            conservation_error,
        )
        tolerance = max(1.0e-5, abs(gross_equity) * 1.0e-10)
        if conservation_error > tolerance:
            raise AssertionError(
                f"cashflow conservation failed by {conservation_error:.8f}"
            )
        daily_return = float(equity / previous_equity - 1.0)
        gross_daily_return = float(gross_equity / previous_gross_equity - 1.0)
        benchmark_return = float(book.benchmark_returns[day])
        benchmark_wealth *= 1.0 + benchmark_return
        excess_return = (
            float((1.0 + daily_return) / (1.0 + benchmark_return) - 1.0)
            if benchmark_return > -1.0
            else math.nan
        )
        equity_rows.append(
            {
                "task_id": spec.task_id,
                "trade_date": trade_date,
                "date_idx": date_idx,
                "cash": float(cash),
                "gross_cash": float(gross_cash),
                "position_count": len(positions),
                "net_equity": float(equity),
                "gross_equity": float(gross_equity),
                "daily_net_return": daily_return,
                "daily_gross_return": gross_daily_return,
                "benchmark_return": benchmark_return,
                "daily_relative_excess_return": excess_return,
                "benchmark_wealth": float(benchmark_wealth),
                "cash_fraction": float(cash / equity) if equity > 0.0 else math.nan,
                "cumulative_cost": float(total_cost),
                "conservation_error": float(conservation_error),
            }
        )
        previous_equity = float(equity)
        previous_gross_equity = float(gross_equity)
        pending = (
            plan_orders(
                book=book,
                spec=spec,
                day=day,
                positions=positions,
            )
            if day < book.day_count - 1
            else None
        )
        if pending is not None:
            requested_sells = {int(order.symbol_idx) for order in pending.sells}
            for symbol_idx in tuple(consecutive_blocked_sells):
                if symbol_idx not in positions or symbol_idx not in requested_sells:
                    consecutive_blocked_sells.pop(symbol_idx, None)
    equity_frame = pd.DataFrame(equity_rows)
    equity_frame["rolling_63d_net_return"] = (
        1.0 + equity_frame["daily_net_return"].astype(float)
    ).rolling(63, min_periods=63).apply(np.prod, raw=True) - 1.0
    equity_frame["rolling_63d_relative_excess_return"] = (
        1.0 + equity_frame["daily_relative_excess_return"].astype(float)
    ).rolling(63, min_periods=63).apply(np.prod, raw=True) - 1.0
    attempts_frame = pd.DataFrame(attempts)
    if attempts_frame.empty:
        attempts_frame = pd.DataFrame(
            columns=list(
                _attempt_row(
                    task_id=spec.task_id,
                    side="buy",
                    status="none",
                    signal_date="",
                    execution_date="",
                    symbol_idx=0,
                    symbol="",
                    reason="",
                )
            )
        )
    monthly = _monthly_metrics(
        dates=equity_frame["trade_date"].to_numpy(dtype=str),
        equity=equity_frame["net_equity"].to_numpy(dtype=np.float64),
    )
    net_metrics = _annualized_metrics(
        equity=np.r_[STARTING_CASH_CNY, equity_frame["net_equity"].to_numpy()],
        daily_returns=np.r_[0.0, equity_frame["daily_net_return"].to_numpy()],
    )
    gross_metrics = _annualized_metrics(
        equity=np.r_[STARTING_CASH_CNY, equity_frame["gross_equity"].to_numpy()],
        daily_returns=np.r_[0.0, equity_frame["daily_gross_return"].to_numpy()],
    )
    annual = _year_metrics(
        dates=equity_frame["trade_date"].to_numpy(dtype=str),
        equity=equity_frame["net_equity"].to_numpy(dtype=np.float64),
        benchmark_wealth=equity_frame["benchmark_wealth"].to_numpy(dtype=np.float64),
    )
    final_date = str(equity_frame["trade_date"].iloc[-1])
    final_date_idx = int(equity_frame["date_idx"].iloc[-1])
    terminal_equity = float(cash)
    terminal_cost = 0.0
    unrealized_contribution = defaultdict(float)
    for symbol_idx, position in positions.items():
        adjusted_close = float(book.adjusted_close[final_date_idx, symbol_idx])
        proceeds, details = _sell_position(
            position=position,
            adjusted_open=adjusted_close,
            trade_date=final_date,
            costs=book.costs,
            slippage_multiplier=multiplier,
        )
        terminal_equity += proceeds
        terminal_cost += float(details["total_cost"])
        unrealized_contribution[symbol_idx] += float(
            proceeds - position.net_cash_outflow
        )
    contribution = defaultdict(float, realized_contribution)
    for symbol_idx, value in unrealized_contribution.items():
        contribution[symbol_idx] += float(value)
    absolute_contribution = np.asarray(
        sorted((abs(value) for value in contribution.values()), reverse=True),
        dtype=np.float64,
    )
    contribution_total = float(np.sum(absolute_contribution))
    top5_concentration = (
        float(np.sum(absolute_contribution[:5]) / contribution_total)
        if contribution_total > 0.0
        else 0.0
    )
    filled = attempts_frame[attempts_frame["status"].astype(str).eq("filled")]
    participation = pd.to_numeric(
        filled["participation_of_trailing20_median_amount"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    participation = participation[np.isfinite(participation)]
    thresholds = tuple(
        float(value)
        for value in book.study["execution"]["amount_participation_thresholds"]
    )
    benchmark_end = float(equity_frame["benchmark_wealth"].iloc[-1])
    terminal_return = float(terminal_equity / STARTING_CASH_CNY - 1.0)
    terminal_excess = float(
        (terminal_equity / STARTING_CASH_CNY)
        / max(benchmark_end / STARTING_CASH_CNY, 1.0e-12)
        - 1.0
    )
    result = {
        "schema": TASK_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "task": asdict(spec),
        "task_id": spec.task_id,
        "signal_book_sha256": book.manifest["files"]["rank_panel"]["sha256"],
        "period": {
            "first_signal_date": str(equity_frame["trade_date"].iloc[0]),
            "last_mark_date": final_date,
            "continuous_years": list(YEARS),
            "annual_reset": False,
            "forbidden_2026_row_count": 0,
        },
        "metrics": {
            "net": net_metrics,
            "gross_same_trade_sequence": gross_metrics,
            "terminal_cost_accrued_equity": float(terminal_equity),
            "terminal_cost_accrued_return": terminal_return,
            "terminal_cost_accrued_relative_excess_return": terminal_excess,
            "terminal_accrued_exit_cost": float(terminal_cost),
            "benchmark_ending_wealth": benchmark_end,
            "turnover_to_starting_cash": float(traded_notional / STARTING_CASH_CNY),
            "average_cash_fraction": float(equity_frame["cash_fraction"].mean()),
            "maximum_cash_fraction": float(equity_frame["cash_fraction"].max()),
            "trade_attempt_count": len(attempts_frame),
            "filled_buy_count": int(counters["buy_count"]),
            "filled_sell_count": int(counters["sell_count"]),
            "failed_buy_count": int(
                counters["failed_buy_count"] + counters["failed_buy_cash_or_lot_count"]
            ),
            "failed_sell_count": int(counters["failed_sell_count"]),
            "blocked_sell_count": int(counters["blocked_sell_count"]),
            "buy_failure_rate": float(
                (
                    counters["failed_buy_count"]
                    + counters["failed_buy_cash_or_lot_count"]
                )
                / max(
                    counters["buy_count"]
                    + counters["failed_buy_count"]
                    + counters["failed_buy_cash_or_lot_count"],
                    1,
                )
            ),
            "sell_failure_rate": float(
                counters["failed_sell_count"]
                / max(counters["sell_count"] + counters["failed_sell_count"], 1)
            ),
            "delayed_sell_count": len(sell_delay_days),
            "sell_delay_days_mean": (
                float(np.mean(sell_delay_days)) if sell_delay_days else 0.0
            ),
            "sell_delay_days_p90": (
                float(np.quantile(sell_delay_days, 0.9)) if sell_delay_days else 0.0
            ),
            "unresolved_blocked_sell_count": len(consecutive_blocked_sells),
            "terminal_recovery_count": int(counters["terminal_recovery_count"]),
            "terminal_recovery_notional": float(counters["terminal_recovery_notional"]),
            "holding_days_mean": (
                float(np.mean(holding_days)) if holding_days else math.nan
            ),
            "holding_days_median": (
                float(np.median(holding_days)) if holding_days else math.nan
            ),
            "holding_days_p90": (
                float(np.quantile(holding_days, 0.9)) if holding_days else math.nan
            ),
            "top5_absolute_pnl_contribution_fraction": top5_concentration,
            "maximum_conservation_error": float(maximum_conservation_error),
            "participation": {
                "finite_order_count": len(participation),
                **{
                    f"fraction_above_{threshold:g}": (
                        float(np.mean(participation > threshold))
                        if len(participation)
                        else math.nan
                    )
                    for threshold in thresholds
                },
            },
            "costs": {
                "total": float(total_cost),
                **{name: float(cost_parts[name]) for name in cost_parts},
            },
        },
        "annual": annual,
    }
    return result, equity_frame, attempts_frame, monthly


def _task_dir(output_root: Path, spec: TaskSpec) -> Path:
    return output_root / "tasks" / spec.task_id


def _task_complete(
    *,
    output_root: Path,
    spec: TaskSpec,
    study_sha256: str,
    signal_sha256: str,
) -> bool:
    path = _task_dir(output_root, spec) / "task_result.json"
    if not path.is_file():
        return False
    try:
        result = _load_json(path)
        if (
            result.get("schema") != TASK_SCHEMA
            or result.get("status") != "completed"
            or result.get("task_id") != spec.task_id
            or result.get("study_config_sha256") != study_sha256
            or result.get("signal_book_sha256") != signal_sha256
            or dict(result.get("task", {})) != asdict(spec)
        ):
            return False
        for record in dict(result["files"]).values():
            _verify_record(record)
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _write_task(
    *,
    output_root: Path,
    result: dict[str, Any],
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    monthly: pd.DataFrame,
    study_sha256: str,
) -> dict[str, Any]:
    spec = TaskSpec(**dict(result["task"]))
    directory = _task_dir(output_root, spec)
    equity_path = directory / "equity.parquet"
    trades_path = directory / "trades.parquet"
    monthly_path = directory / "monthly.parquet"
    _write_parquet(equity_path, equity)
    _write_parquet(trades_path, trades)
    _write_parquet(monthly_path, monthly)
    payload = {
        **result,
        "study_config_sha256": str(study_sha256),
        "files": {
            "equity": _file_record(
                equity_path,
                row_count=len(equity),
                columns=list(equity.columns),
            ),
            "trades": _file_record(
                trades_path,
                row_count=len(trades),
                columns=list(trades.columns),
            ),
            "monthly": _file_record(
                monthly_path,
                row_count=len(monthly),
                columns=list(monthly.columns),
            ),
        },
    }
    _write_json(directory / "task_result.json", payload)
    return payload


def run_pending(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    signal_manifest = prepare_signal_book(
        study_path=study_path,
        output_root=output_root,
    )
    study_sha256 = _study_hash(study_path)
    signal_sha256 = str(signal_manifest["files"]["rank_panel"]["sha256"])
    specs = task_specs()
    complete_before = sum(
        _task_complete(
            output_root=output_root,
            spec=spec,
            study_sha256=study_sha256,
            signal_sha256=signal_sha256,
        )
        for spec in specs
    )
    _emit(
        "account_tasks_starting",
        completed=complete_before,
        total=len(specs),
    )
    book = SignalBook(study=study, output_root=output_root)
    completed = complete_before
    started = time.monotonic()
    for spec in specs:
        if _task_complete(
            output_root=output_root,
            spec=spec,
            study_sha256=study_sha256,
            signal_sha256=signal_sha256,
        ):
            continue
        result, equity, trades, monthly = simulate_task(book=book, spec=spec)
        _write_task(
            output_root=output_root,
            result=result,
            equity=equity,
            trades=trades,
            monthly=monthly,
            study_sha256=study_sha256,
        )
        completed += 1
        if completed % 10 == 0 or completed == len(specs):
            _emit(
                "account_task_progress",
                completed=completed,
                total=len(specs),
                task_id=spec.task_id,
                elapsed_seconds=round(time.monotonic() - started, 2),
            )
    return {
        "status": "completed",
        "completed": int(completed),
        "total": len(specs),
    }


def status(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    load_study(study_path)
    study_sha256 = _study_hash(study_path)
    signal_ready = _signal_complete(
        output_root,
        study_sha256=study_sha256,
    )
    signal_sha256 = ""
    if signal_ready:
        signal_sha256 = str(
            _load_json(output_root / "signal_book/manifest.json")["files"][
                "rank_panel"
            ]["sha256"]
        )
    specs = task_specs()
    complete = (
        sum(
            _task_complete(
                output_root=output_root,
                spec=spec,
                study_sha256=study_sha256,
                signal_sha256=signal_sha256,
            )
            for spec in specs
        )
        if signal_ready
        else 0
    )
    return {
        "study_id": STUDY_ID,
        "signal_book_complete": bool(signal_ready),
        "completed_tasks": int(complete),
        "pending_tasks": int(len(specs) - complete),
        "total_tasks": len(specs),
        "evaluation_complete": bool(
            (output_root / "evaluation/summary.json").is_file()
        ),
    }


def hac_mean_test(values: np.ndarray, *, maximum_lag: int) -> dict[str, float]:
    current = np.asarray(values, dtype=np.float64)
    current = current[np.isfinite(current)]
    if current.size < 3:
        return {
            "mean": math.nan,
            "standard_error": math.nan,
            "t_statistic": math.nan,
            "one_sided_p_value": math.nan,
        }
    centered = current - float(np.mean(current))
    n = len(current)
    lag = min(int(maximum_lag), n - 1)
    long_run = float(np.dot(centered, centered) / n)
    for offset in range(1, lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / n)
        long_run += 2.0 * (1.0 - offset / (lag + 1.0)) * covariance
    variance_mean = max(long_run / n, 0.0)
    standard_error = math.sqrt(variance_mean)
    mean = float(np.mean(current))
    if standard_error <= 0.0:
        statistic = math.inf if mean > 0.0 else -math.inf if mean < 0.0 else 0.0
        p_value = 0.0 if mean > 0.0 else 1.0
    else:
        statistic = mean / standard_error
        p_value = float(stats.norm.sf(statistic))
    return {
        "mean": mean,
        "standard_error": float(standard_error),
        "t_statistic": float(statistic),
        "one_sided_p_value": float(p_value),
    }


def moving_block_bootstrap_lower(
    values: np.ndarray,
    *,
    block_length: int,
    repetitions: int,
    seed: int,
    lower_quantile: float,
) -> dict[str, float]:
    current = np.asarray(values, dtype=np.float64)
    current = current[np.isfinite(current)]
    if len(current) < max(3, int(block_length)):
        return {
            "mean": math.nan,
            "lower_bound": math.nan,
            "positive_probability": math.nan,
        }
    rng = np.random.default_rng(int(seed))
    n = len(current)
    blocks = math.ceil(n / int(block_length))
    samples = np.empty(int(repetitions), dtype=np.float64)
    offsets = np.arange(int(block_length), dtype=np.int64)
    for repetition in range(int(repetitions)):
        starts = rng.integers(0, n, size=blocks, endpoint=False)
        indices = (starts[:, None] + offsets[None, :]) % n
        samples[repetition] = float(np.mean(current[indices.ravel()[:n]]))
    return {
        "mean": float(np.mean(current)),
        "lower_bound": float(np.quantile(samples, float(lower_quantile))),
        "positive_probability": float(np.mean(samples > 0.0)),
    }


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    current = np.asarray(p_values, dtype=np.float64)
    if current.ndim != 1 or not bool(np.isfinite(current).all()):
        raise ValueError("BH requires a finite vector")
    count = len(current)
    order = np.argsort(current, kind="stable")
    adjusted = np.empty(count, dtype=np.float64)
    running = 1.0
    for reverse_index in range(count - 1, -1, -1):
        position = int(order[reverse_index])
        rank = reverse_index + 1
        running = min(running, float(current[position]) * count / rank)
        adjusted[position] = min(running, 1.0)
    return adjusted


def _task_result(
    *,
    output_root: Path,
    spec: TaskSpec,
) -> dict[str, Any]:
    path = _task_dir(output_root, spec) / "task_result.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _load_json(path)


def _task_metric_row(result: Mapping[str, Any]) -> dict[str, Any]:
    spec = dict(result["task"])
    metrics = dict(result["metrics"])
    net = dict(metrics["net"])
    gross = dict(metrics["gross_same_trade_sequence"])
    benchmark_ending = float(metrics["benchmark_ending_wealth"])
    gross_relative_excess = (1.0 + float(gross["cumulative_return"])) / max(
        benchmark_ending / STARTING_CASH_CNY, 1.0e-12
    ) - 1.0
    annual = list(result["annual"])
    costs = dict(metrics["costs"])
    participation = dict(metrics["participation"])
    return {
        **spec,
        "task_id": str(result["task_id"]),
        "net_cumulative_return": float(net["cumulative_return"]),
        "net_cagr": float(net["cagr"]),
        "net_annualized_volatility": float(net["annualized_volatility"]),
        "net_sharpe": net["sharpe"],
        "net_sortino": net["sortino"],
        "maximum_drawdown": float(net["maximum_drawdown"]),
        "net_calmar": net["calmar"],
        "gross_cumulative_return": float(gross["cumulative_return"]),
        "gross_cagr": float(gross["cagr"]),
        "gross_relative_excess_return": float(gross_relative_excess),
        "benchmark_ending_wealth": benchmark_ending,
        "terminal_cost_accrued_return": float(metrics["terminal_cost_accrued_return"]),
        "terminal_cost_accrued_relative_excess_return": float(
            metrics["terminal_cost_accrued_relative_excess_return"]
        ),
        "turnover_to_starting_cash": float(metrics["turnover_to_starting_cash"]),
        "average_cash_fraction": float(metrics["average_cash_fraction"]),
        "maximum_cash_fraction": float(metrics["maximum_cash_fraction"]),
        "trade_attempt_count": int(metrics["trade_attempt_count"]),
        "filled_buy_count": int(metrics["filled_buy_count"]),
        "filled_sell_count": int(metrics["filled_sell_count"]),
        "failed_buy_count": int(metrics["failed_buy_count"]),
        "failed_sell_count": int(metrics["failed_sell_count"]),
        "buy_failure_rate": float(metrics["buy_failure_rate"]),
        "sell_failure_rate": float(metrics["sell_failure_rate"]),
        "delayed_sell_count": int(metrics["delayed_sell_count"]),
        "sell_delay_days_mean": float(metrics["sell_delay_days_mean"]),
        "sell_delay_days_p90": float(metrics["sell_delay_days_p90"]),
        "unresolved_blocked_sell_count": int(metrics["unresolved_blocked_sell_count"]),
        "terminal_recovery_count": int(metrics["terminal_recovery_count"]),
        "terminal_recovery_notional": float(metrics["terminal_recovery_notional"]),
        "holding_days_mean": metrics["holding_days_mean"],
        "holding_days_median": metrics["holding_days_median"],
        "holding_days_p90": metrics["holding_days_p90"],
        "top5_absolute_pnl_contribution_fraction": float(
            metrics["top5_absolute_pnl_contribution_fraction"]
        ),
        "maximum_conservation_error": float(metrics["maximum_conservation_error"]),
        "total_cost": float(costs["total"]),
        "commission_cost": float(costs.get("commission", 0.0)),
        "stamp_tax_cost": float(costs.get("stamp_tax", 0.0)),
        "transfer_fee_cost": float(costs.get("transfer_fee", 0.0)),
        "slippage_cost": float(costs.get("slippage", 0.0)),
        "participation_finite_order_count": int(participation["finite_order_count"]),
        **{
            str(name): (float(value) if value is not None else math.nan)
            for name, value in participation.items()
            if str(name).startswith("fraction_above_")
        },
        "positive_absolute_years": int(
            sum(float(item["net_return"]) > 0.0 for item in annual)
        ),
        "positive_excess_years": int(
            sum(float(item["relative_excess_return"]) > 0.0 for item in annual)
        ),
        **{
            f"net_return_{int(item['year'])}": float(item["net_return"])
            for item in annual
        },
        **{
            f"excess_return_{int(item['year'])}": float(item["relative_excess_return"])
            for item in annual
        },
    }


def _configuration_table(
    metrics: pd.DataFrame, study: Mapping[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["family", "exposure_mode", "slot_count", "buffer_multiplier"]
    minimum_absolute = int(study["statistics"]["minimum_positive_absolute_years"])
    minimum_excess = int(study["statistics"]["minimum_positive_excess_years"])
    for values, part in metrics.groupby(keys, sort=True):
        by_cost = {
            str(row["cost_scenario"]): dict(row) for row in part.to_dict("records")
        }
        if set(by_cost) != set(COST_SCENARIOS):
            raise ValueError(f"cost support changed for {values}")
        base = by_cost["base"]
        stress = by_cost["stress"]
        base_pass = bool(
            float(base["terminal_cost_accrued_return"]) > 0.0
            and float(base["terminal_cost_accrued_relative_excess_return"]) > 0.0
            and int(base["positive_absolute_years"]) >= minimum_absolute
            and int(base["positive_excess_years"]) >= minimum_excess
        )
        stress_pass = bool(
            float(stress["terminal_cost_accrued_return"]) > 0.0
            and float(stress["terminal_cost_accrued_relative_excess_return"]) > 0.0
        )
        gross_pass = bool(
            float(base["gross_cumulative_return"]) > 0.0
            and float(base["gross_relative_excess_return"]) > 0.0
        )
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "base_pass": base_pass,
                "stress_pass": stress_pass,
                "economic_pass": bool(base_pass and stress_pass),
                "gross_pass": gross_pass,
                "base_terminal_return": float(base["terminal_cost_accrued_return"]),
                "stress_terminal_return": float(stress["terminal_cost_accrued_return"]),
                "base_terminal_excess": float(
                    base["terminal_cost_accrued_relative_excess_return"]
                ),
                "stress_terminal_excess": float(
                    stress["terminal_cost_accrued_relative_excess_return"]
                ),
                "base_net_cagr": float(base["net_cagr"]),
                "stress_net_cagr": float(stress["net_cagr"]),
                "base_maximum_drawdown": float(base["maximum_drawdown"]),
                "base_turnover": float(base["turnover_to_starting_cash"]),
                "positive_absolute_years": int(base["positive_absolute_years"]),
                "positive_excess_years": int(base["positive_excess_years"]),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != len(FAMILIES) * len(EXPOSURE_MODES) * len(SLOT_COUNTS) * len(
        BUFFER_MULTIPLIERS
    ):
        raise AssertionError("configuration table row count changed")
    return frame


def robust_rectangles(
    frame: pd.DataFrame,
    *,
    flag: str,
    slot_width: int,
    buffer_width: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for slot_start in range(len(SLOT_COUNTS) - int(slot_width) + 1):
        slots = SLOT_COUNTS[slot_start : slot_start + int(slot_width)]
        for buffer_start in range(len(BUFFER_MULTIPLIERS) - int(buffer_width) + 1):
            buffers = BUFFER_MULTIPLIERS[
                buffer_start : buffer_start + int(buffer_width)
            ]
            selected = frame[
                frame["slot_count"].astype(int).isin(slots)
                & frame["buffer_multiplier"].astype(float).isin(buffers)
            ]
            expected = int(slot_width) * int(buffer_width)
            if len(selected) != expected:
                raise ValueError("rectangle support is incomplete")
            if bool(selected[flag].astype(bool).all()):
                output.append(
                    {
                        "slot_counts": list(slots),
                        "buffer_multipliers": list(buffers),
                        "cell_count": expected,
                    }
                )
    return output


def _family_daily_series(
    *,
    output_root: Path,
    family: str,
    exposure_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series: list[np.ndarray] = []
    dates: np.ndarray | None = None
    benchmark: np.ndarray | None = None
    for slots in SLOT_COUNTS:
        for buffer in BUFFER_MULTIPLIERS:
            spec = TaskSpec(
                family=family,
                exposure_mode=exposure_mode,
                slot_count=slots,
                buffer_multiplier=buffer,
                cost_scenario="base",
            )
            result = _task_result(output_root=output_root, spec=spec)
            path = _verify_record(result["files"]["equity"])
            frame = pd.read_parquet(
                path,
                columns=[
                    "trade_date",
                    "daily_relative_excess_return",
                    "daily_net_return",
                ],
            )
            current_dates = frame["trade_date"].astype(str).to_numpy()
            if dates is None:
                dates = current_dates
            elif not np.array_equal(dates, current_dates):
                raise ValueError("family equity dates differ")
            series.append(
                frame["daily_relative_excess_return"].to_numpy(dtype=np.float64)
            )
            if benchmark is None:
                benchmark = frame["daily_net_return"].to_numpy(dtype=np.float64)
    if dates is None or benchmark is None:
        raise AssertionError("family series is empty")
    return dates, np.median(np.vstack(series), axis=0), benchmark


def _family_statistics(
    *,
    output_root: Path,
    configurations: pd.DataFrame,
    study: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    statistics = dict(study["statistics"])
    family_rows: list[dict[str, Any]] = []
    daily_rows: list[pd.DataFrame] = []
    for family in FAMILIES:
        for exposure in EXPOSURE_MODES:
            selected = configurations[
                configurations["family"].astype(str).eq(family)
                & configurations["exposure_mode"].astype(str).eq(exposure)
            ]
            economic_rectangles = robust_rectangles(
                selected,
                flag="economic_pass",
                slot_width=int(statistics["robust_rectangle_slot_width"]),
                buffer_width=int(statistics["robust_rectangle_buffer_width"]),
            )
            gross_rectangles = robust_rectangles(
                selected,
                flag="gross_pass",
                slot_width=int(statistics["robust_rectangle_slot_width"]),
                buffer_width=int(statistics["robust_rectangle_buffer_width"]),
            )
            base_rectangles = robust_rectangles(
                selected,
                flag="base_pass",
                slot_width=int(statistics["robust_rectangle_slot_width"]),
                buffer_width=int(statistics["robust_rectangle_buffer_width"]),
            )
            dates, median_excess, _ = _family_daily_series(
                output_root=output_root,
                family=family,
                exposure_mode=exposure,
            )
            hac = hac_mean_test(
                median_excess,
                maximum_lag=int(statistics["hac_maximum_lag"]),
            )
            bootstrap = moving_block_bootstrap_lower(
                median_excess,
                block_length=int(statistics["bootstrap_block_length"]),
                repetitions=int(statistics["bootstrap_repetitions"]),
                seed=int(statistics["bootstrap_seed"]),
                lower_quantile=float(statistics["bootstrap_lower_quantile"]),
            )
            family_rows.append(
                {
                    "family": family,
                    "exposure_mode": exposure,
                    "economic_rectangle_count": len(economic_rectangles),
                    "gross_rectangle_count": len(gross_rectangles),
                    "base_rectangle_count": len(base_rectangles),
                    "economic_rectangles": json.dumps(
                        economic_rectangles,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "gross_rectangles": json.dumps(
                        gross_rectangles,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "base_rectangles": json.dumps(
                        base_rectangles,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "median_daily_excess_mean": float(hac["mean"]),
                    "hac_standard_error": float(hac["standard_error"]),
                    "hac_t_statistic": float(hac["t_statistic"]),
                    "hac_one_sided_p_value": float(hac["one_sided_p_value"]),
                    "bootstrap_lower_90": float(bootstrap["lower_bound"]),
                    "bootstrap_positive_probability": float(
                        bootstrap["positive_probability"]
                    ),
                    "isolated_economic_cell_count": int(
                        selected["economic_pass"].astype(bool).sum()
                    ),
                    "isolated_base_cell_count": int(
                        selected["base_pass"].astype(bool).sum()
                    ),
                    "isolated_gross_cell_count": int(
                        selected["gross_pass"].astype(bool).sum()
                    ),
                }
            )
            daily_rows.append(
                pd.DataFrame(
                    {
                        "trade_date": dates,
                        "family": family,
                        "exposure_mode": exposure,
                        "median_daily_relative_excess_return": median_excess,
                    }
                )
            )
    frame = pd.DataFrame(family_rows)
    formal_mask = frame["family"].astype(str).isin(FORMAL_FAMILIES)
    q_values = benjamini_hochberg(
        frame.loc[formal_mask, "hac_one_sided_p_value"].to_numpy(dtype=np.float64)
    )
    frame["bh_q_value"] = np.nan
    frame.loc[formal_mask, "bh_q_value"] = q_values
    frame["robustly_monetizable"] = (
        frame["economic_rectangle_count"].astype(int).gt(0)
        & frame["hac_one_sided_p_value"]
        .astype(float)
        .le(float(statistics["maximum_hac_p_value"]))
        & frame["bootstrap_lower_90"].astype(float).ge(0.0)
        & (
            ~formal_mask
            | frame["bh_q_value"]
            .astype(float)
            .le(float(statistics["maximum_bh_q_value"]))
        )
    )

    def classify(row: pd.Series) -> str:
        if bool(row["robustly_monetizable"]):
            return "robustly_monetizable"
        if int(row["economic_rectangle_count"]) > 0:
            return "economically_positive_but_statistically_unconfirmed"
        if int(row["isolated_economic_cell_count"]) > 0:
            return "fragile_parameter_dependent"
        if int(row["base_rectangle_count"]) > 0:
            return "economically_positive_but_cost_sensitive"
        if int(row["gross_rectangle_count"]) > 0:
            return "predictive_but_cost_limited"
        return "not_monetized_by_tested_daily_policy"

    frame["family_status"] = frame.apply(classify, axis=1)
    return frame, pd.concat(daily_rows, ignore_index=True)


def _paired_comparisons(
    *,
    output_root: Path,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    family_pairs = (
        ("mfe10_primary", "mfe20_primary", "mfe10_minus_mfe20"),
        ("dual_mfe_maximin", "mfe10_primary", "dual_minus_mfe10"),
        ("dual_mfe_maximin", "mfe20_primary", "dual_minus_mfe20"),
        (
            "dual_mfe_state_veto",
            "dual_mfe_maximin",
            "state_veto_minus_dual",
        ),
        (
            "dual_mfe_risk_veto",
            "dual_mfe_maximin",
            "risk_veto_minus_dual",
        ),
        (
            "dual_mfe_state_risk_veto",
            "dual_mfe_maximin",
            "state_risk_veto_minus_dual",
        ),
    )
    metric_map = {
        (
            str(row["family"]),
            str(row["exposure_mode"]),
            int(row["slot_count"]),
            float(row["buffer_multiplier"]),
            str(row["cost_scenario"]),
        ): dict(row)
        for row in metrics.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    equity_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def equity_series(spec: TaskSpec) -> tuple[np.ndarray, np.ndarray]:
        if spec.task_id not in equity_cache:
            result = _task_result(output_root=output_root, spec=spec)
            frame = pd.read_parquet(
                _verify_record(result["files"]["equity"]),
                columns=["trade_date", "daily_net_return"],
            )
            equity_cache[spec.task_id] = (
                frame["trade_date"].astype(str).to_numpy(),
                frame["daily_net_return"].to_numpy(dtype=np.float64),
            )
        return equity_cache[spec.task_id]

    def append_comparison(
        *,
        comparison_type: str,
        comparison: str,
        left_spec: TaskSpec,
        right_spec: TaskSpec,
    ) -> None:
        left_dates, left_returns = equity_series(left_spec)
        right_dates, right_returns = equity_series(right_spec)
        if not np.array_equal(left_dates, right_dates):
            raise ValueError("paired comparison dates differ")
        hac = hac_mean_test(left_returns - right_returns, maximum_lag=20)
        left_metrics = metric_map[
            (
                left_spec.family,
                left_spec.exposure_mode,
                left_spec.slot_count,
                left_spec.buffer_multiplier,
                left_spec.cost_scenario,
            )
        ]
        right_metrics = metric_map[
            (
                right_spec.family,
                right_spec.exposure_mode,
                right_spec.slot_count,
                right_spec.buffer_multiplier,
                right_spec.cost_scenario,
            )
        ]
        rows.append(
            {
                "comparison_type": comparison_type,
                "comparison": comparison,
                "left_task_id": left_spec.task_id,
                "right_task_id": right_spec.task_id,
                "left_family": left_spec.family,
                "right_family": right_spec.family,
                "left_exposure_mode": left_spec.exposure_mode,
                "right_exposure_mode": right_spec.exposure_mode,
                "slot_count": left_spec.slot_count,
                "buffer_multiplier": left_spec.buffer_multiplier,
                "left_cost_scenario": left_spec.cost_scenario,
                "right_cost_scenario": right_spec.cost_scenario,
                "terminal_return_delta": float(
                    left_metrics["terminal_cost_accrued_return"]
                    - right_metrics["terminal_cost_accrued_return"]
                ),
                "terminal_excess_delta": float(
                    left_metrics["terminal_cost_accrued_relative_excess_return"]
                    - right_metrics["terminal_cost_accrued_relative_excess_return"]
                ),
                "gross_return_delta": float(
                    left_metrics["gross_cumulative_return"]
                    - right_metrics["gross_cumulative_return"]
                ),
                "maximum_drawdown_delta": float(
                    left_metrics["maximum_drawdown"] - right_metrics["maximum_drawdown"]
                ),
                "turnover_delta": float(
                    left_metrics["turnover_to_starting_cash"]
                    - right_metrics["turnover_to_starting_cash"]
                ),
                "average_cash_fraction_delta": float(
                    left_metrics["average_cash_fraction"]
                    - right_metrics["average_cash_fraction"]
                ),
                "terminal_recovery_count_delta": int(
                    left_metrics["terminal_recovery_count"]
                    - right_metrics["terminal_recovery_count"]
                ),
                **{
                    f"net_return_delta_{year}": float(
                        left_metrics[f"net_return_{year}"]
                        - right_metrics[f"net_return_{year}"]
                    )
                    for year in YEARS
                },
                **{
                    f"excess_return_delta_{year}": float(
                        left_metrics[f"excess_return_{year}"]
                        - right_metrics[f"excess_return_{year}"]
                    )
                    for year in YEARS
                },
                "mean_daily_return_delta": float(hac["mean"]),
                "hac_t_statistic": float(hac["t_statistic"]),
                "hac_one_sided_p_value": float(hac["one_sided_p_value"]),
            }
        )

    for left, right, comparison in family_pairs:
        for exposure in EXPOSURE_MODES:
            for slots in SLOT_COUNTS:
                for buffer in BUFFER_MULTIPLIERS:
                    for cost in COST_SCENARIOS:
                        append_comparison(
                            comparison_type="family",
                            comparison=comparison,
                            left_spec=TaskSpec(
                                family=left,
                                exposure_mode=exposure,
                                slot_count=slots,
                                buffer_multiplier=buffer,
                                cost_scenario=cost,
                            ),
                            right_spec=TaskSpec(
                                family=right,
                                exposure_mode=exposure,
                                slot_count=slots,
                                buffer_multiplier=buffer,
                                cost_scenario=cost,
                            ),
                        )
    for family in FAMILIES:
        for slots in SLOT_COUNTS:
            for buffer in BUFFER_MULTIPLIERS:
                for cost in COST_SCENARIOS:
                    append_comparison(
                        comparison_type="exposure",
                        comparison="strong_candidate_cash_minus_target_full",
                        left_spec=TaskSpec(
                            family=family,
                            exposure_mode="strong_candidate_cash",
                            slot_count=slots,
                            buffer_multiplier=buffer,
                            cost_scenario=cost,
                        ),
                        right_spec=TaskSpec(
                            family=family,
                            exposure_mode="target_full",
                            slot_count=slots,
                            buffer_multiplier=buffer,
                            cost_scenario=cost,
                        ),
                    )
    for family in FAMILIES:
        for exposure in EXPOSURE_MODES:
            for slots in SLOT_COUNTS:
                for buffer in BUFFER_MULTIPLIERS:
                    append_comparison(
                        comparison_type="cost",
                        comparison="stress_minus_base",
                        left_spec=TaskSpec(
                            family=family,
                            exposure_mode=exposure,
                            slot_count=slots,
                            buffer_multiplier=buffer,
                            cost_scenario="stress",
                        ),
                        right_spec=TaskSpec(
                            family=family,
                            exposure_mode=exposure,
                            slot_count=slots,
                            buffer_multiplier=buffer,
                            cost_scenario="base",
                        ),
                    )
    return pd.DataFrame(rows)


def _annual_metric_table(
    results: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        task = dict(result["task"])
        for annual in result["annual"]:
            rows.append(
                {
                    **task,
                    "task_id": str(result["task_id"]),
                    "year": int(annual["year"]),
                    "net_return": float(annual["net_return"]),
                    "benchmark_return": float(annual["benchmark_return"]),
                    "relative_excess_return": float(annual["relative_excess_return"]),
                    "ending_equity": float(annual["ending_equity"]),
                }
            )
    return pd.DataFrame(rows)


def _surface_slice_table(metrics: pd.DataFrame) -> pd.DataFrame:
    measure_aggregations = {
        "median_terminal_return": ("terminal_cost_accrued_return", "median"),
        "median_terminal_excess": (
            "terminal_cost_accrued_relative_excess_return",
            "median",
        ),
        "median_gross_return": ("gross_cumulative_return", "median"),
        "median_maximum_drawdown": ("maximum_drawdown", "median"),
        "median_turnover": ("turnover_to_starting_cash", "median"),
        "median_cash_fraction": ("average_cash_fraction", "median"),
        "median_terminal_recovery_count": ("terminal_recovery_count", "median"),
        "positive_terminal_fraction": (
            "terminal_cost_accrued_return",
            lambda values: float(np.mean(np.asarray(values) > 0.0)),
        ),
        "task_count": ("task_id", "count"),
    }

    def grouped(
        frame: pd.DataFrame,
        *,
        name: str,
        scope: str,
        keys: Sequence[str],
    ) -> pd.DataFrame:
        result = (
            frame.groupby(list(keys), sort=True, dropna=False)
            .agg(**measure_aggregations)
            .reset_index()
        )
        result.insert(0, "slice_name", name)
        result.insert(1, "scope", scope)
        return result

    formal = metrics[metrics["family"].astype(str).isin(FORMAL_FAMILIES)]
    slices = [
        grouped(
            metrics,
            name="family_exposure_cost",
            scope="all_including_negative_control",
            keys=("family", "exposure_mode", "cost_scenario"),
        ),
        grouped(
            formal,
            name="slot_exposure_cost",
            scope="formal_families",
            keys=("exposure_mode", "slot_count", "cost_scenario"),
        ),
        grouped(
            formal,
            name="buffer_exposure_cost",
            scope="formal_families",
            keys=("exposure_mode", "buffer_multiplier", "cost_scenario"),
        ),
        grouped(
            formal,
            name="cost_exposure",
            scope="formal_families",
            keys=("exposure_mode", "cost_scenario"),
        ),
    ]
    records: list[dict[str, Any]] = []
    for frame in slices:
        records.extend(frame.to_dict("records"))
    return pd.DataFrame.from_records(records)


def evaluate(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    record_root: Path = DEFAULT_RECORD_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    signal_manifest = prepare_signal_book(
        study_path=study_path,
        output_root=output_root,
    )
    study_sha256 = _study_hash(study_path)
    signal_sha256 = str(signal_manifest["files"]["rank_panel"]["sha256"])
    specs = task_specs()
    incomplete = [
        spec.task_id
        for spec in specs
        if not _task_complete(
            output_root=output_root,
            spec=spec,
            study_sha256=study_sha256,
            signal_sha256=signal_sha256,
        )
    ]
    if incomplete:
        raise RuntimeError(
            f"{len(incomplete)} account tasks are incomplete; first={incomplete[0]}"
        )
    task_results = [_task_result(output_root=output_root, spec=spec) for spec in specs]
    metric_rows = [_task_metric_row(result) for result in task_results]
    metrics = pd.DataFrame(metric_rows)
    annual_metrics = _annual_metric_table(task_results)
    surface_slices = _surface_slice_table(metrics)
    configurations = _configuration_table(metrics, study)
    family_stats, family_daily = _family_statistics(
        output_root=output_root,
        configurations=configurations,
        study=study,
    )
    comparisons = _paired_comparisons(
        output_root=output_root,
        metrics=metrics,
    )
    accounting_invalid = bool(
        metrics["maximum_conservation_error"].astype(float).max()
        > STARTING_CASH_CNY * 1.0e-9
    )
    negative = family_stats[family_stats["family"].astype(str).eq(NEGATIVE_CONTROL)]
    negative_control_invalid = bool(negative["robustly_monetizable"].astype(bool).any())
    official = family_stats[family_stats["family"].astype(str).isin(FORMAL_FAMILIES)]
    robust = official[official["robustly_monetizable"].astype(bool)]
    isolated_economic = configurations[
        configurations["family"].astype(str).isin(FORMAL_FAMILIES)
        & configurations["economic_pass"].astype(bool)
    ].sort_values(
        ["base_terminal_excess", "base_terminal_return"],
        ascending=False,
    )
    if accounting_invalid or negative_control_invalid:
        overall = "audit_invalid_due_to_negative_control_or_accounting"
    elif not robust.empty:
        overall = "v4_economically_realizable_in_decision_folds"
    elif bool(official["gross_rectangle_count"].astype(int).gt(0).any()):
        overall = "v4_signal_exists_but_not_net_monetized"
    else:
        overall = "v4_not_monetized_by_preregistered_policy_surface"
    robust_families = [
        {
            "family": str(row["family"]),
            "exposure_mode": str(row["exposure_mode"]),
            "family_status": str(row["family_status"]),
            "economic_rectangle_count": int(row["economic_rectangle_count"]),
            "hac_p_value": float(row["hac_one_sided_p_value"]),
            "bh_q_value": float(row["bh_q_value"]),
            "bootstrap_lower_90": float(row["bootstrap_lower_90"]),
        }
        for row in robust.to_dict("records")
    ]
    isolated_configurations = [
        {
            "family": str(row["family"]),
            "exposure_mode": str(row["exposure_mode"]),
            "slot_count": int(row["slot_count"]),
            "buffer_multiplier": float(row["buffer_multiplier"]),
            "base_terminal_return": float(row["base_terminal_return"]),
            "stress_terminal_return": float(row["stress_terminal_return"]),
            "base_terminal_excess": float(row["base_terminal_excess"]),
            "stress_terminal_excess": float(row["stress_terminal_excess"]),
            "base_maximum_drawdown": float(row["base_maximum_drawdown"]),
            "base_turnover": float(row["base_turnover"]),
        }
        for row in isolated_economic.to_dict("records")
    ]
    evaluation_root = output_root / "evaluation"
    metrics_path = evaluation_root / "task_metrics.parquet"
    annual_path = evaluation_root / "annual_metrics.parquet"
    surface_slices_path = evaluation_root / "surface_slices.parquet"
    configurations_path = evaluation_root / "configurations.parquet"
    family_path = evaluation_root / "family_statistics.parquet"
    family_daily_path = evaluation_root / "family_daily_median_excess.parquet"
    comparisons_path = evaluation_root / "paired_comparisons.parquet"
    _write_parquet(metrics_path, metrics)
    _write_parquet(annual_path, annual_metrics)
    _write_parquet(surface_slices_path, surface_slices)
    _write_parquet(configurations_path, configurations)
    _write_parquet(family_path, family_stats)
    _write_parquet(family_daily_path, family_daily)
    _write_parquet(comparisons_path, comparisons)
    decision = {
        "status": "completed_without_policy_winner_selection",
        "overall": overall,
        "robust_family_exposure_pairs": robust_families,
        "robust_pair_count": len(robust_families),
        "isolated_economic_configurations": isolated_configurations,
        "isolated_economic_configuration_count": len(isolated_configurations),
        "negative_control_invalid": negative_control_invalid,
        "accounting_invalid": accounting_invalid,
        "single_policy_winner_selected": False,
        "policy_surface_role": (
            "pre-registered 2023-2025 development-fold economic realizability; "
            "not a pristine final holdout"
        ),
        "next_step": (
            "lock one policy from a robust region before any 2026 or prospective "
            "validation"
            if len(robust_families)
            else "do not select an isolated winner or consume 2026; diagnose why the "
            "state/risk-veto neighborhood is parameter-fragile and why ungated MFE "
            "opportunity is not converted into executable returns"
        ),
    }
    _write_json(evaluation_root / "decision.json", decision)
    files = {
        "task_metrics": _file_record(
            metrics_path,
            row_count=len(metrics),
        ),
        "annual_metrics": _file_record(
            annual_path,
            row_count=len(annual_metrics),
        ),
        "surface_slices": _file_record(
            surface_slices_path,
            row_count=len(surface_slices),
        ),
        "configurations": _file_record(
            configurations_path,
            row_count=len(configurations),
        ),
        "family_statistics": _file_record(
            family_path,
            row_count=len(family_stats),
        ),
        "family_daily_median_excess": _file_record(
            family_daily_path,
            row_count=len(family_daily),
        ),
        "paired_comparisons": _file_record(
            comparisons_path,
            row_count=len(comparisons),
        ),
        "decision": _file_record(evaluation_root / "decision.json"),
    }
    family_records = [
        {
            "family": str(row["family"]),
            "exposure_mode": str(row["exposure_mode"]),
            "family_status": str(row["family_status"]),
            "robustly_monetizable": bool(row["robustly_monetizable"]),
            "economic_rectangle_count": int(row["economic_rectangle_count"]),
            "base_rectangle_count": int(row["base_rectangle_count"]),
            "gross_rectangle_count": int(row["gross_rectangle_count"]),
            "isolated_economic_cell_count": int(row["isolated_economic_cell_count"]),
            "median_daily_excess_mean": float(row["median_daily_excess_mean"]),
            "hac_p_value": float(row["hac_one_sided_p_value"]),
            "bh_q_value": (
                float(row["bh_q_value"])
                if math.isfinite(float(row["bh_q_value"]))
                else None
            ),
            "bootstrap_lower_90": float(row["bootstrap_lower_90"]),
        }
        for row in family_stats.to_dict("records")
    ]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "study_config_sha256": study_sha256,
        "entry_contract": "seq100_entry_contract_oos_v4",
        "period": {
            "years": list(YEARS),
            "continuous_account": True,
            "annual_reset": False,
            "uses_2020_2022_policy_outcomes": False,
            "maximum_consumed_date": "2025-12-31",
            "forbidden_2026_row_count": 0,
        },
        "task_count": len(metrics),
        "annual_metric_count": len(annual_metrics),
        "surface_slice_count": len(surface_slices),
        "paired_comparison_count": len(comparisons),
        "configuration_count": len(configurations),
        "family_hypothesis_count": int(len(FORMAL_FAMILIES) * len(EXPOSURE_MODES)),
        "signal_book": {
            "signal_date_count": int(signal_manifest["signal_date_count"]),
            "candidate_row_count": int(signal_manifest["candidate_row_count"]),
            "manifest": _file_record(output_root / "signal_book/manifest.json"),
        },
        "decision": decision,
        "families": family_records,
        "validation": {
            "task_count_matches_preregistration": len(metrics) == 672,
            "maximum_conservation_error": float(
                metrics["maximum_conservation_error"].max()
            ),
            "negative_control_invalid": negative_control_invalid,
            "tasks_with_terminal_recovery": int(
                metrics["terminal_recovery_count"].astype(int).gt(0).sum()
            ),
            "maximum_terminal_recovery_count": int(
                metrics["terminal_recovery_count"].astype(int).max()
            ),
            "single_policy_winner_selected": False,
            "no_model_training": True,
            "forbidden_2026_row_count": 0,
        },
        "files": files,
        "does_not_select": list(study["non_selections"]),
    }
    summary_path = evaluation_root / "summary.json"
    _write_json(summary_path, summary)
    record_root.mkdir(parents=True, exist_ok=True)
    record = {
        **summary,
        "full_output": _file_record(summary_path),
    }
    _write_json(record_root / "result.json", record)
    _emit(
        "economic_evaluation_completed",
        overall=overall,
        robust_pair_count=len(robust_families),
    )
    return record


def self_test() -> dict[str, Any]:
    if len(task_specs()) != 672:
        raise AssertionError("task inventory self-test failed")
    expected_buffer = 2.0 * 6 / 3077
    if not math.isclose(
        replacement_buffer(
            multiplier=2.0,
            slot_count=6,
            candidate_count=3078,
        ),
        expected_buffer,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise AssertionError("replacement buffer self-test failed")
    sellable = derive_open_sellable(
        raw_open=np.asarray([10.0, 9.0, 8.0, math.nan]),
        raw_down_limit=np.asarray([9.0, 9.0, 7.0, 1.0]),
        status_valid=np.asarray([True, True, True, True]),
        suspended=np.asarray([False, False, True, False]),
        delisted=np.asarray([False, False, False, False]),
    )
    if sellable.tolist() != [True, False, False, False]:
        raise AssertionError("open sellability self-test failed")
    q = benjamini_hochberg([0.01, 0.04, 0.03])
    if not np.allclose(q, np.asarray([0.03, 0.04, 0.04])):
        raise AssertionError("BH self-test failed")
    synthetic = pd.DataFrame(
        [
            {
                "slot_count": slots,
                "buffer_multiplier": buffer,
                "economic_pass": slots in (1, 3, 6) and buffer in (0.0, 0.5),
            }
            for slots in SLOT_COUNTS
            for buffer in BUFFER_MULTIPLIERS
        ]
    )
    rectangles = robust_rectangles(
        synthetic,
        flag="economic_pass",
        slot_width=3,
        buffer_width=2,
    )
    if len(rectangles) != 1:
        raise AssertionError("rectangle self-test failed")
    return {
        "status": "passed",
        "task_count": 672,
        "checks": 5,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--run-pending", action="store_true")
    actions.add_argument("--evaluate", action="store_true")
    actions.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.status:
        payload = status(
            study_path=args.study,
            output_root=args.output_root,
        )
    elif args.prepare:
        payload = prepare_signal_book(
            study_path=args.study,
            output_root=args.output_root,
        )
    elif args.run_pending:
        payload = run_pending(
            study_path=args.study,
            output_root=args.output_root,
        )
    elif args.evaluate:
        payload = evaluate(
            study_path=args.study,
            output_root=args.output_root,
            record_root=args.record_root,
        )
    else:
        payload = self_test()
    print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
