from __future__ import annotations

"""Replay the canonical D1 return signal under an A-share T+1 account."""

import argparse
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_v4_economic_realizability as economic

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_quality_liquidity_d1_execution"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_quality_liquidity_d1_execution.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_model"
    / "direct_returns/d1_execution"
)
SIGNAL_SCHEMA = "seq100_quality_liquidity_d1_execution_signal_book/1"
TASK_SCHEMA = "seq100_quality_liquidity_d1_execution_task/1"
MANIFEST_SCHEMA = "seq100_quality_liquidity_d1_execution_manifest/1"
AUDIT_SCHEMA = "seq100_quality_liquidity_d1_execution_audit/1"
YEARS = (2023, 2024, 2025)
ENTRY_MODES = ("open", "first_5m_vwap")
FAMILIES = tuple(f"d1_{value}" for value in ENTRY_MODES)
SLOT_COUNTS = (1, 2, 3, 5, 10)
BUFFER_MULTIPLIERS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
COST_SCENARIOS = ("base", "stress")
STARTING_CASH_CNY = 1_000_000.0
FEATURE_COUNT = 557
LAST_ENTRY_SIGNAL_DATE = "2025-12-23"
LAST_MARK_DATE = "2025-12-31"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (
        (WORKSPACE_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(
            economic._json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=economic._json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    economic._write_parquet(path, frame)


def _file_record(path: Path, **extra: Any) -> dict[str, Any]:
    return economic._file_record(path, **extra)


def _study_hash(path: Path) -> str:
    return economic._sha256(path)


def _emit(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {"event": event, "at": _now(), **payload},
            ensure_ascii=False,
            default=economic._json_default,
        ),
        flush=True,
    )


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _load_json(path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError(f"study_id must be {STUDY_ID}")
    period = dict(study["period"])
    if tuple(int(value) for value in period["signal_years"]) != YEARS:
        raise ValueError("signal years changed")
    if str(period["last_entry_signal_date"]) != LAST_ENTRY_SIGNAL_DATE:
        raise ValueError("last entry signal date changed")
    if str(period["last_mark_date"]) != LAST_MARK_DATE:
        raise ValueError("last mark date changed")
    if int(period["forbidden_year"]) != 2026:
        raise ValueError("2026 must remain forbidden")
    model = dict(study["model_contract"])
    if (
        str(model["target"]) != "return_1"
        or str(model["source_label"]) != "g_1"
        or int(model["feature_count"]) != FEATURE_COUNT
    ):
        raise ValueError("D1 model contract changed")
    account = dict(study["account"])
    if tuple(int(value) for value in account["slot_counts"]) != SLOT_COUNTS:
        raise ValueError("slot counts changed")
    if not math.isclose(
        float(account["starting_cash_cny"]),
        STARTING_CASH_CNY,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("starting cash changed")
    policies = dict(study["policies"])
    if tuple(str(value) for value in policies["entry_modes"]) != ENTRY_MODES:
        raise ValueError("entry modes changed")
    if tuple(float(value) for value in policies["buffer_multipliers"]) != (
        BUFFER_MULTIPLIERS
    ):
        raise ValueError("replacement buffers changed")
    if tuple(dict(study["execution"]["cost_scenarios"])) != COST_SCENARIOS:
        raise ValueError("cost scenarios changed")
    expected = (
        len(ENTRY_MODES)
        * len(SLOT_COUNTS)
        * len(BUFFER_MULTIPLIERS)
        * len(COST_SCENARIOS)
    )
    if int(study["outputs"]["expected_task_count"]) != expected:
        raise ValueError("expected task count changed")
    return study


def task_specs(
    study_path: Path = DEFAULT_STUDY_PATH,
) -> list[economic.TaskSpec]:
    study = load_study(study_path)
    exposure = str(study["policies"]["exposure_mode"])
    specs = [
        economic.TaskSpec(
            family=f"d1_{entry_mode}",
            exposure_mode=exposure,
            slot_count=slots,
            buffer_multiplier=buffer,
            cost_scenario=cost,
        )
        for entry_mode in ENTRY_MODES
        for slots in SLOT_COUNTS
        for buffer in BUFFER_MULTIPLIERS
        for cost in COST_SCENARIOS
    ]
    expected_count = (
        len(ENTRY_MODES)
        * len(SLOT_COUNTS)
        * len(BUFFER_MULTIPLIERS)
        * len(COST_SCENARIOS)
    )
    if len(specs) != expected_count or len({item.task_id for item in specs}) != len(
        specs
    ):
        raise AssertionError("D1 execution task inventory changed")
    return specs


def _model_task_path(model_root: Path, year: int) -> Path:
    return (
        model_root
        / "direct_returns/tasks"
        / f"return_zscore__g_01__compact_core__{int(year)}"
        / "task_result.json"
    )


def _load_prediction_task(
    *, model_root: Path, year: int
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    path = _model_task_path(model_root, year)
    task = _load_json(path)
    if (
        task.get("status") != "completed"
        or task.get("target") != "return_1"
        or task.get("source_label") != "g_1"
        or int(task.get("evaluation_year", -1)) != int(year)
        or task.get("variant") != "compact_core"
        or int(task.get("feature_count", -1)) != FEATURE_COUNT
    ):
        raise ValueError(f"invalid D1 model task contract: {path}")
    candidate_path = economic._verify_record(task["files"]["candidate_id"])
    prediction_path = economic._verify_record(task["files"]["prediction"])
    candidate_ids = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    if prediction.shape != candidate_ids.shape:
        raise ValueError(f"D1 prediction shape changed: {year}")
    return task, candidate_ids, prediction


def _signal_complete(
    output_root: Path,
    *,
    study_sha256: str | None = None,
) -> bool:
    path = output_root / "signal_book/manifest.json"
    if not path.is_file():
        return False
    try:
        manifest = _load_json(path)
        if manifest.get("schema") != SIGNAL_SCHEMA or manifest.get("status") != (
            "completed"
        ):
            return False
        if study_sha256 is not None and manifest.get("study_config_sha256") != (
            study_sha256
        ):
            return False
        for record in dict(manifest["files"]).values():
            economic._verify_record(record)
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
    economic._memory_guard()
    sources = dict(study["sources"])
    model_root = _resolve(sources["model_root"])
    input_manifest_path = _resolve(sources["model_input_manifest"])
    row_index_path = _resolve(sources["row_index"])
    pack_path = _resolve(sources["pack_manifest"])
    close_d1_manifest_path = _resolve(sources["close_d1_manifest"])
    direct_audit = _load_json(model_root / "direct_returns/audit.json")
    input_manifest = _load_json(input_manifest_path)
    compact_features = tuple(
        str(value)
        for value in input_manifest.get("feature_groups", {}).get("compact_core", [])
    )
    if (
        direct_audit.get("status") != "ok"
        or int(direct_audit.get("task_count", -1)) != 15
    ):
        raise ValueError("direct-return model audit is not valid")
    if len(compact_features) != FEATURE_COUNT or len(set(compact_features)) != (
        FEATURE_COUNT
    ):
        raise ValueError("compact feature contract changed")
    if int(input_manifest.get("source", {}).get("forbidden_year", -1)) != 2026:
        raise ValueError("model input forbidden-year contract changed")
    close_d1_manifest = _load_json(close_d1_manifest_path)
    if close_d1_manifest.get("status") != "audited":
        raise ValueError("first-5m execution cache is not audited")
    outcome_record = dict(close_d1_manifest["preparation_files"]["execution_outcomes"])
    outcome_path = economic._verify_record(outcome_record)
    rows = pd.read_parquet(
        row_index_path,
        columns=["candidate_id", "date_idx", "trade_date", "symbol"],
        filters=[
            ("trade_date", ">=", "2023-01-01"),
            ("trade_date", "<=", LAST_MARK_DATE),
        ],
    ).sort_values("candidate_id", kind="stable")
    if rows.empty or rows["candidate_id"].duplicated().any():
        raise ValueError("D1 execution row index is empty or duplicated")
    if set(rows["trade_date"].astype(str).str[:4].astype(int)) != set(YEARS):
        raise ValueError("D1 execution row-index year coverage changed")
    outcomes = pd.read_parquet(
        outcome_path,
        columns=["candidate_id", "first_5m_vwap", "outcome_valid"],
    ).sort_values("candidate_id", kind="stable")
    expected_ids = rows["candidate_id"].to_numpy(dtype=np.int64)
    outcome_ids = outcomes["candidate_id"].to_numpy(dtype=np.int64)
    if outcomes["candidate_id"].duplicated().any() or bool(
        np.setdiff1d(outcome_ids, expected_ids, assume_unique=True).size
    ):
        raise ValueError("first-5m execution outcomes are not a model-row subset")
    outcome_valid = outcomes["outcome_valid"].to_numpy(dtype=bool)
    outcome_vwap = outcomes["first_5m_vwap"].to_numpy(dtype=np.float32)
    outcome_valid &= np.isfinite(outcome_vwap) & (outcome_vwap > 0.0)
    outcome_vwap = np.where(outcome_valid, outcome_vwap, np.nan).astype(np.float32)
    aligned_vwap = np.full(len(expected_ids), np.nan, dtype=np.float32)
    positions = np.searchsorted(outcome_ids, expected_ids)
    matched = positions < len(outcome_ids)
    matched[matched] &= outcome_ids[positions[matched]] == expected_ids[matched]
    aligned_vwap[matched] = outcome_vwap[positions[matched]]
    del outcomes
    pack = _load_json(pack_path)
    date_values = np.asarray(pack["date_values"], dtype=str)
    symbol_values = np.asarray(pack["symbol_values"], dtype=str)
    symbol_map = {str(symbol): idx for idx, symbol in enumerate(symbol_values)}
    symbol_indices = rows["symbol"].astype(str).map(symbol_map)
    if symbol_indices.isna().any():
        raise ValueError("quality-pool symbol is absent from the sequence pack")
    rows = rows.assign(symbol_idx=symbol_indices.astype(np.int32))
    signal_date_idx = np.sort(rows["date_idx"].astype(np.int32).unique())
    signal_dates = date_values[signal_date_idx]
    if (
        str(signal_dates[0]) != str(study["period"]["first_signal_date"])
        or str(signal_dates[-1]) != LAST_MARK_DATE
        or any(str(value).startswith("2026") for value in signal_dates)
    ):
        raise ValueError("D1 signal calendar changed")
    day_count = len(signal_date_idx)
    symbol_count = int(pack["symbol_count"])
    rank_panel = np.full((day_count, symbol_count), np.nan, dtype=np.float32)
    candidate_orders = np.full((day_count, symbol_count), -1, dtype=np.int32)
    candidate_counts = np.zeros(day_count, dtype=np.int32)
    first_5m_vwap = np.full((day_count, symbol_count), np.nan, dtype=np.float32)
    date_to_day = {int(value): idx for idx, value in enumerate(signal_date_idx)}
    source_tasks: list[dict[str, Any]] = []
    row_offset = 0
    for year in YEARS:
        year_rows = rows[rows["trade_date"].astype(str).str.startswith(str(year))]
        year_count = len(year_rows)
        year_ids = year_rows["candidate_id"].to_numpy(dtype=np.int64)
        task, candidate_ids, prediction = _load_prediction_task(
            model_root=model_root, year=year
        )
        if tuple(str(value) for value in task.get("feature_names", [])) != (
            compact_features
        ):
            raise ValueError(f"D1 feature order changed: {year}")
        if not np.array_equal(np.asarray(candidate_ids, dtype=np.int64), year_ids):
            raise ValueError(f"D1 prediction candidate IDs changed: {year}")
        date_idx = year_rows["date_idx"].to_numpy(dtype=np.int32)
        symbol_idx = year_rows["symbol_idx"].to_numpy(dtype=np.int32)
        year_vwap = aligned_vwap[row_offset : row_offset + year_count]
        row_offset += year_count
        boundaries = np.flatnonzero(np.r_[True, date_idx[1:] != date_idx[:-1], True])
        for start, stop in pairwise(boundaries):
            current_date_idx = int(date_idx[start])
            day = int(date_to_day[current_date_idx])
            symbols = np.asarray(symbol_idx[start:stop], dtype=np.int32)
            if len(np.unique(symbols)) != len(symbols):
                raise ValueError(f"duplicate symbol on {date_values[current_date_idx]}")
            scores = np.asarray(prediction[start:stop], dtype=np.float64)
            if not bool(np.isfinite(scores).all()):
                raise ValueError(
                    f"non-finite D1 prediction on {date_values[current_date_idx]}"
                )
            ranks = economic._rank01(scores)
            order = np.lexsort((symbols, -ranks))
            rank_panel[day, symbols] = ranks
            candidate_orders[day, : len(symbols)] = symbols[order]
            candidate_counts[day] = len(symbols)
            first_5m_vwap[day, symbols] = year_vwap[start:stop]
        source_tasks.append(
            {
                "year": year,
                "task_result": _file_record(_model_task_path(model_root, year)),
                "candidate_id_sha256": task["files"]["candidate_id"]["sha256"],
                "prediction_sha256": task["files"]["prediction"]["sha256"],
                "feature_count": int(task["feature_count"]),
            }
        )
        economic._memory_guard()
    if row_offset != len(rows) or bool(np.any(candidate_counts <= 0)):
        raise ValueError("D1 signal preparation did not consume every row")
    masks = dict(pack["masks"])
    execution = dict(pack["execution_arrays"])
    entry_filled = economic._open_array(masks["entry_filled"], dtype=np.bool_)
    raw_open = economic._open_array(execution["entry_open_raw"], dtype=np.float32)
    raw_down_limit = economic._open_array(
        execution["exit_down_limit_raw"], dtype=np.float32
    )
    status_valid = economic._open_array(masks["status_valid"], dtype=np.bool_)
    suspended = economic._open_array(masks["is_suspended"], dtype=np.bool_)
    delisted = economic._open_array(masks["is_delisted"], dtype=np.bool_)
    next_buyable = np.zeros((day_count, symbol_count), dtype=np.bool_)
    next_sellable = np.zeros((day_count, symbol_count), dtype=np.bool_)
    for day in range(day_count - 1):
        signal_idx = int(signal_date_idx[day])
        execution_idx = int(signal_date_idx[day + 1])
        if execution_idx != signal_idx + 1:
            raise ValueError("D1 signal calendar contains a gap")
        next_buyable[day] = np.asarray(entry_filled[signal_idx], dtype=bool)
        next_sellable[day] = economic.derive_open_sellable(
            raw_open=np.asarray(raw_open[execution_idx], dtype=np.float64),
            raw_down_limit=np.asarray(raw_down_limit[execution_idx], dtype=np.float64),
            status_valid=np.asarray(status_valid[execution_idx], dtype=bool),
            suspended=np.asarray(suspended[execution_idx], dtype=bool),
            delisted=np.asarray(delisted[execution_idx], dtype=bool),
        )
    daily_raw = economic._open_array(
        pack["feature_channels"]["daily_raw"], dtype=np.float32
    )
    adjusted_close = daily_raw[:, :, 3]
    terminal_recovery = float(
        dict(pack.get("terminal_execution", {}) or {}).get(
            "recovery_fraction_of_entry_notional", 0.0
        )
        or 0.0
    )
    benchmark_returns = np.zeros(day_count, dtype=np.float64)
    benchmark_missing_count = 0
    for day in range(1, day_count):
        previous_symbols = candidate_orders[day - 1, : candidate_counts[day - 1]]
        previous_idx = int(signal_date_idx[day - 1])
        current_idx = int(signal_date_idx[day])
        previous_close = np.asarray(
            adjusted_close[previous_idx, previous_symbols], dtype=np.float64
        )
        current_close = np.asarray(
            adjusted_close[current_idx, previous_symbols], dtype=np.float64
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
    arrays = {
        "signal_date_idx": signal_date_idx.astype(np.int32),
        "candidate_counts": candidate_counts,
        "rank_panel": rank_panel,
        "candidate_orders": candidate_orders,
        "first_5m_vwap": first_5m_vwap,
        "next_open_buyable": next_buyable,
        "next_open_sellable": next_sellable,
        "benchmark_returns": benchmark_returns,
    }
    files: dict[str, dict[str, Any]] = {}
    for name, values in arrays.items():
        path = signal_root / f"{name}.npy"
        economic._save_npy(path, values)
        files[name] = _file_record(
            path,
            shape=list(values.shape),
            dtype=str(values.dtype),
        )
    candidate_support = np.isfinite(rank_panel)
    entry_signal_days = signal_dates <= LAST_ENTRY_SIGNAL_DATE
    buyable_support = (
        candidate_support[entry_signal_days] & next_buyable[entry_signal_days]
    )
    vwap_valid = np.isfinite(first_5m_vwap[entry_signal_days]) & buyable_support
    vwap_coverage = float(np.sum(vwap_valid) / max(np.sum(buyable_support), 1))
    manifest = {
        "schema": SIGNAL_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "study_config_sha256": study_sha256,
        "direct_return_audit": _file_record(model_root / "direct_returns/audit.json"),
        "model_input_manifest": _file_record(input_manifest_path),
        "row_index": _file_record(row_index_path),
        "pack_manifest": _file_record(pack_path),
        "first_5m_source_manifest": _file_record(close_d1_manifest_path),
        "source_tasks": source_tasks,
        "feature_count": FEATURE_COUNT,
        "target": "return_1",
        "source_label": "g_1",
        "signal_date_count": int(day_count),
        "candidate_row_count": int(np.sum(candidate_counts)),
        "candidate_count_minimum": int(np.min(candidate_counts)),
        "candidate_count_median": float(np.median(candidate_counts)),
        "candidate_count_maximum": int(np.max(candidate_counts)),
        "first_5m_vwap_valid_count": int(np.sum(vwap_valid)),
        "first_5m_vwap_buyable_support_count": int(np.sum(buyable_support)),
        "first_5m_vwap_buyable_coverage": vwap_coverage,
        "first_5m_vwap_all_candidate_coverage": float(
            np.sum(np.isfinite(first_5m_vwap) & candidate_support)
            / max(np.sum(candidate_support), 1)
        ),
        "first_signal_date": str(signal_dates[0]),
        "last_entry_signal_date": LAST_ENTRY_SIGNAL_DATE,
        "last_mark_date": str(signal_dates[-1]),
        "benchmark": "previous_quality_pool_equal_weight_adjusted_close",
        "benchmark_missing_next_price_count": int(benchmark_missing_count),
        "forbidden_2026_row_count": 0,
        "files": files,
    }
    minimum_vwap = float(study["policies"]["minimum_first_5m_vwap_coverage"])
    if vwap_coverage < minimum_vwap:
        raise ValueError(
            f"first-5m VWAP coverage {vwap_coverage:.8f} below {minimum_vwap:.8f}"
        )
    _write_json(signal_root / "manifest.json", manifest)
    _emit(
        "d1_signal_book_completed",
        signal_dates=day_count,
        candidate_rows=int(np.sum(candidate_counts)),
        first_5m_vwap_buyable_coverage=vwap_coverage,
    )
    return manifest


class SignalBook:
    def __init__(self, *, study: Mapping[str, Any], output_root: Path) -> None:
        if not _signal_complete(output_root):
            raise FileNotFoundError("D1 execution signal book is incomplete")
        self.manifest = _load_json(output_root / "signal_book/manifest.json")
        self.study = dict(study)
        files = dict(self.manifest["files"])
        self.signal_date_idx = np.load(
            economic._verify_record(files["signal_date_idx"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.candidate_counts = np.load(
            economic._verify_record(files["candidate_counts"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.rank_panel = np.load(
            economic._verify_record(files["rank_panel"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.orders = np.load(
            economic._verify_record(files["candidate_orders"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.first_5m_vwap = np.load(
            economic._verify_record(files["first_5m_vwap"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.next_buyable = np.load(
            economic._verify_record(files["next_open_buyable"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.next_sellable = np.load(
            economic._verify_record(files["next_open_sellable"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.benchmark_returns = np.load(
            economic._verify_record(files["benchmark_returns"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        pack_path = _resolve(self.study["sources"]["pack_manifest"])
        self.pack = _load_json(pack_path)
        self.date_values = np.asarray(self.pack["date_values"], dtype=str)
        self.symbol_values = np.asarray(self.pack["symbol_values"], dtype=str)
        self.symbol_count = int(self.pack["symbol_count"])
        self.raw_open = economic._source_array(
            self.pack, "execution_arrays", "entry_open_raw", dtype=np.float32
        )
        self.daily_raw = economic._source_array(
            self.pack, "feature_channels", "daily_raw", dtype=np.float32
        )
        self.adjusted_open = self.daily_raw[:, :, 0]
        self.adjusted_close = self.daily_raw[:, :, 3]
        self.amount = self.daily_raw[:, :, 5]
        self.status_valid = economic._source_array(
            self.pack, "masks", "status_valid", dtype=np.bool_
        )
        self.has_bar = economic._source_array(
            self.pack, "masks", "has_bar", dtype=np.bool_
        )
        self.is_delisted = economic._source_array(
            self.pack, "masks", "is_delisted", dtype=np.bool_
        )
        self.costs = economic.parse_execution_costs(self.pack)
        self._validate_costs()
        self.terminal_recovery_fraction = float(
            dict(self.pack.get("terminal_execution", {}) or {}).get(
                "recovery_fraction_of_entry_notional", 0.0
            )
            or 0.0
        )
        if self.date_text(self.day_count - 1) != LAST_MARK_DATE:
            raise ValueError("D1 execution signal book end date changed")

    def _validate_costs(self) -> None:
        expected = dict(self.study["execution"]["expected_pack_costs"])
        observed = {
            "lot_size": self.costs.lot_size,
            "commission_bps": self.costs.commission_bps,
            "minimum_commission_cny": self.costs.minimum_commission_cny,
            "transfer_fee_bps": self.costs.transfer_fee_bps,
            "slippage_bps": self.costs.slippage_bps,
            "stress_slippage_multiplier": self.costs.stress_slippage_multiplier,
        }
        for name, value in observed.items():
            if not math.isclose(
                float(value),
                float(expected[name]),
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
        if family not in FAMILIES:
            raise ValueError(f"unknown D1 execution family: {family}")
        return 0

    def symbols_for_day(self, family: str, day: int) -> np.ndarray:
        self.selector_column(family)
        count = int(self.candidate_counts[int(day)])
        return np.asarray(self.orders[int(day), :count], dtype=np.int32)

    def rank(self, day: int, symbol_idx: int, column: int) -> float:
        if int(column) != 0:
            raise ValueError("D1 execution has one rank column")
        return float(self.rank_panel[int(day), int(symbol_idx)])

    def rank_values(self, *, day: int, symbols: np.ndarray, column: int) -> np.ndarray:
        if int(column) != 0:
            raise ValueError("D1 execution has one rank column")
        return np.asarray(
            self.rank_panel[int(day), np.asarray(symbols, dtype=np.int32)],
            dtype=np.float64,
        )

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

    def entry_is_buyable(
        self, *, spec: economic.TaskSpec, signal_day: int, symbol_idx: int
    ) -> bool:
        base = bool(self.next_buyable[int(signal_day), int(symbol_idx)])
        if spec.family == "d1_open":
            return base
        if spec.family == "d1_first_5m_vwap":
            vwap = float(self.first_5m_vwap[int(signal_day), int(symbol_idx)])
            return base and math.isfinite(vwap) and vwap > 0.0
        raise ValueError(f"unknown D1 execution family: {spec.family}")

    def entry_prices(
        self,
        *,
        spec: economic.TaskSpec,
        signal_day: int,
        execution_date_idx: int,
        symbol_idx: int,
    ) -> tuple[float, float]:
        raw_open = float(self.raw_open[int(execution_date_idx), int(symbol_idx)])
        adjusted_open = float(
            self.adjusted_open[int(execution_date_idx), int(symbol_idx)]
        )
        if spec.family == "d1_open":
            return raw_open, adjusted_open
        if spec.family != "d1_first_5m_vwap":
            raise ValueError(f"unknown D1 execution family: {spec.family}")
        raw_vwap = float(self.first_5m_vwap[int(signal_day), int(symbol_idx)])
        if (
            not math.isfinite(raw_open)
            or raw_open <= 0.0
            or not math.isfinite(adjusted_open)
            or adjusted_open <= 0.0
            or not math.isfinite(raw_vwap)
            or raw_vwap <= 0.0
        ):
            return math.nan, math.nan
        return raw_vwap, raw_vwap * adjusted_open / raw_open

    def plan_orders(
        self,
        *,
        spec: economic.TaskSpec,
        day: int,
        positions: Mapping[int, economic.Position],
    ) -> economic.PendingOrders:
        if self.date_text(day) <= LAST_ENTRY_SIGNAL_DATE:
            return economic.plan_orders(
                book=self,
                spec=spec,
                day=day,
                positions=positions,
            )
        return economic.PendingOrders(
            signal_day=int(day),
            sells=tuple(
                economic.PendingSell(
                    symbol_idx=int(symbol_idx),
                    reason="end_of_sample_liquidation",
                    paired_buy_symbol_idx=None,
                )
                for symbol_idx in sorted(positions)
            ),
            unpaired_buys=(),
        )


def _task_dir(output_root: Path, spec: economic.TaskSpec) -> Path:
    return output_root / "tasks" / spec.task_id


def _task_complete(
    *,
    output_root: Path,
    spec: economic.TaskSpec,
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
            or result.get("study_id") != STUDY_ID
            or result.get("task_id") != spec.task_id
            or result.get("study_config_sha256") != study_sha256
            or result.get("signal_book_sha256") != signal_sha256
            or dict(result.get("task", {})) != asdict(spec)
        ):
            return False
        for record in dict(result["files"]).values():
            economic._verify_record(record)
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
    spec = economic.TaskSpec(**dict(result["task"]))
    directory = _task_dir(output_root, spec)
    equity_path = directory / "equity.parquet"
    trades_path = directory / "trades.parquet"
    monthly_path = directory / "monthly.parquet"
    _write_parquet(equity_path, equity)
    _write_parquet(trades_path, trades)
    _write_parquet(monthly_path, monthly)
    payload = {
        **result,
        "schema": TASK_SCHEMA,
        "study_id": STUDY_ID,
        "study_config_sha256": study_sha256,
        "files": {
            "equity": _file_record(equity_path, row_count=len(equity)),
            "trades": _file_record(trades_path, row_count=len(trades)),
            "monthly": _file_record(monthly_path, row_count=len(monthly)),
        },
    }
    _write_json(directory / "task_result.json", payload)
    return payload


def run(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    study = load_study(study_path)
    signal_manifest = prepare_signal_book(
        study_path=study_path,
        output_root=output_root,
    )
    study_sha256 = _study_hash(study_path)
    signal_sha256 = str(signal_manifest["files"]["rank_panel"]["sha256"])
    specs = task_specs(study_path)
    complete_before = sum(
        _task_complete(
            output_root=output_root,
            spec=spec,
            study_sha256=study_sha256,
            signal_sha256=signal_sha256,
        )
        for spec in specs
    )
    book = SignalBook(study=study, output_root=output_root)
    completed = complete_before
    executed = 0
    started = time.monotonic()
    _emit("d1_account_tasks_starting", completed=completed, total=len(specs))
    for spec in specs:
        if _task_complete(
            output_root=output_root,
            spec=spec,
            study_sha256=study_sha256,
            signal_sha256=signal_sha256,
        ):
            continue
        if max_tasks is not None and executed >= int(max_tasks):
            break
        result, equity, trades, monthly = economic.simulate_task(book=book, spec=spec)
        result["schema"] = TASK_SCHEMA
        result["study_id"] = STUDY_ID
        result["period"]["last_entry_signal_date"] = LAST_ENTRY_SIGNAL_DATE
        result["period"]["research_semantics"] = str(
            study["period"]["research_semantics"]
        )
        _write_task(
            output_root=output_root,
            result=result,
            equity=equity,
            trades=trades,
            monthly=monthly,
            study_sha256=study_sha256,
        )
        completed += 1
        executed += 1
        if completed % 10 == 0 or completed == len(specs):
            _emit(
                "d1_account_task_progress",
                completed=completed,
                total=len(specs),
                task_id=spec.task_id,
                elapsed_seconds=round(time.monotonic() - started, 2),
            )
    return {
        "status": "completed" if completed == len(specs) else "in_progress",
        "completed": int(completed),
        "total": len(specs),
        "executed_this_run": int(executed),
    }


def status(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    load_study(study_path)
    study_sha256 = _study_hash(study_path)
    signal_ready = _signal_complete(output_root, study_sha256=study_sha256)
    signal_sha256 = ""
    if signal_ready:
        signal_sha256 = str(
            _load_json(output_root / "signal_book/manifest.json")["files"][
                "rank_panel"
            ]["sha256"]
        )
    specs = task_specs(study_path)
    completed = (
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
        "status": "completed" if completed == len(specs) else "in_progress",
        "signal_book_ready": bool(signal_ready),
        "completed": int(completed),
        "total": len(specs),
    }


def _task_result(output_root: Path, spec: economic.TaskSpec) -> dict[str, Any]:
    return _load_json(_task_dir(output_root, spec) / "task_result.json")


def _metric_tables(
    *,
    results: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    monthly_frames: list[pd.DataFrame] = []
    for raw in results:
        result = dict(raw)
        task = dict(result["task"])
        metrics = dict(result["metrics"])
        net = dict(metrics["net"])
        gross = dict(metrics["gross_same_trade_sequence"])
        costs = dict(metrics["costs"])
        equity = pd.read_parquet(
            economic._verify_record(result["files"]["equity"]),
            columns=["position_count", "cash_fraction"],
        )
        monthly = pd.read_parquet(economic._verify_record(result["files"]["monthly"]))
        monthly.insert(0, "task_id", result["task_id"])
        monthly.insert(1, "entry_mode", str(task["family"])[3:])
        monthly.insert(2, "slot_count", int(task["slot_count"]))
        monthly.insert(3, "buffer_multiplier", float(task["buffer_multiplier"]))
        monthly.insert(4, "cost_scenario", str(task["cost_scenario"]))
        monthly_frames.append(monthly)
        monthly_returns = monthly["net_return"].to_numpy(dtype=np.float64)
        task_rows.append(
            {
                "task_id": result["task_id"],
                "entry_mode": str(task["family"])[3:],
                "slot_count": int(task["slot_count"]),
                "buffer_multiplier": float(task["buffer_multiplier"]),
                "cost_scenario": str(task["cost_scenario"]),
                "terminal_return": float(metrics["terminal_cost_accrued_return"]),
                "terminal_relative_excess_return": float(
                    metrics["terminal_cost_accrued_relative_excess_return"]
                ),
                "cagr": float(net["cagr"]),
                "sharpe": float(net["sharpe"]),
                "sortino": float(net["sortino"]),
                "maximum_drawdown": float(net["maximum_drawdown"]),
                "calmar": float(net["calmar"]),
                "gross_cagr": float(gross["cagr"]),
                "gross_sharpe": float(gross["sharpe"]),
                "positive_month_count": int(np.sum(monthly_returns > 0.0)),
                "month_count": len(monthly_returns),
                "positive_month_fraction": float(np.mean(monthly_returns > 0.0)),
                "worst_month_return": float(np.min(monthly_returns)),
                "best_month_return": float(np.max(monthly_returns)),
                "average_position_count": float(
                    equity["position_count"].astype(float).mean()
                ),
                "maximum_position_count": int(equity["position_count"].max()),
                "average_cash_fraction": float(metrics["average_cash_fraction"]),
                "turnover_to_starting_cash": float(
                    metrics["turnover_to_starting_cash"]
                ),
                "filled_buy_count": int(metrics["filled_buy_count"]),
                "filled_sell_count": int(metrics["filled_sell_count"]),
                "buy_failure_rate": float(metrics["buy_failure_rate"]),
                "sell_failure_rate": float(metrics["sell_failure_rate"]),
                "blocked_sell_count": int(metrics["blocked_sell_count"]),
                "delayed_sell_count": int(metrics["delayed_sell_count"]),
                "sell_delay_days_mean": float(metrics["sell_delay_days_mean"]),
                "holding_days_mean": float(metrics["holding_days_mean"]),
                "holding_days_median": float(metrics["holding_days_median"]),
                "unresolved_blocked_sell_count": int(
                    metrics["unresolved_blocked_sell_count"]
                ),
                "total_cost": float(costs["total"]),
                "commission": float(costs.get("commission", 0.0)),
                "transfer_fee": float(costs.get("transfer_fee", 0.0)),
                "stamp_tax": float(costs.get("stamp_tax", 0.0)),
                "slippage": float(costs.get("slippage", 0.0)),
                "maximum_conservation_error": float(
                    metrics["maximum_conservation_error"]
                ),
            }
        )
        for annual in result["annual"]:
            annual_rows.append(
                {
                    "task_id": result["task_id"],
                    "entry_mode": str(task["family"])[3:],
                    "slot_count": int(task["slot_count"]),
                    "buffer_multiplier": float(task["buffer_multiplier"]),
                    "cost_scenario": str(task["cost_scenario"]),
                    **dict(annual),
                }
            )
    task_metrics = pd.DataFrame(task_rows).sort_values(
        ["entry_mode", "slot_count", "buffer_multiplier", "cost_scenario"],
        kind="stable",
    )
    annual_metrics = pd.DataFrame(annual_rows).sort_values(
        [
            "entry_mode",
            "slot_count",
            "buffer_multiplier",
            "cost_scenario",
            "year",
        ],
        kind="stable",
    )
    monthly_metrics = pd.concat(monthly_frames, ignore_index=True).sort_values(
        [
            "entry_mode",
            "slot_count",
            "buffer_multiplier",
            "cost_scenario",
            "month",
        ],
        kind="stable",
    )
    return task_metrics, annual_metrics, monthly_metrics


def _configuration_table(
    *,
    task_metrics: pd.DataFrame,
    annual_metrics: pd.DataFrame,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    selection = dict(study["selection"])
    indexed = {
        (
            str(row["entry_mode"]),
            int(row["slot_count"]),
            float(row["buffer_multiplier"]),
            str(row["cost_scenario"]),
        ): dict(row)
        for row in task_metrics.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    for entry_mode in ENTRY_MODES:
        for slots in SLOT_COUNTS:
            for buffer in BUFFER_MULTIPLIERS:
                base = indexed[(entry_mode, slots, buffer, "base")]
                stress = indexed[(entry_mode, slots, buffer, "stress")]
                annual = annual_metrics[
                    annual_metrics["entry_mode"].astype(str).eq(entry_mode)
                    & annual_metrics["slot_count"].astype(int).eq(slots)
                    & annual_metrics["buffer_multiplier"].astype(float).eq(buffer)
                ]
                base_years = annual[annual["cost_scenario"].astype(str).eq("base")][
                    "net_return"
                ].to_numpy(dtype=np.float64)
                stress_years = annual[annual["cost_scenario"].astype(str).eq("stress")][
                    "net_return"
                ].to_numpy(dtype=np.float64)
                stable = bool(
                    len(base_years) == len(YEARS)
                    and len(stress_years) == len(YEARS)
                    and np.sum(base_years > 0.0)
                    >= int(selection["minimum_positive_years"])
                    and np.sum(stress_years > 0.0)
                    >= int(selection["minimum_positive_years"])
                    and float(base["terminal_return"]) > 0.0
                    and float(stress["terminal_return"]) > 0.0
                    and float(stress["sharpe"])
                    >= float(selection["minimum_stress_sharpe"])
                    and float(stress["positive_month_fraction"])
                    >= float(selection["minimum_stress_positive_month_fraction"])
                    and float(stress["maximum_drawdown"])
                    >= -float(selection["maximum_stress_drawdown"])
                    and int(stress["unresolved_blocked_sell_count"]) == 0
                )
                rows.append(
                    {
                        "entry_mode": entry_mode,
                        "slot_count": slots,
                        "buffer_multiplier": buffer,
                        "stable": stable,
                        "base_positive_years": int(np.sum(base_years > 0.0)),
                        "stress_positive_years": int(np.sum(stress_years > 0.0)),
                        "base_worst_annual_return": float(np.min(base_years)),
                        "stress_worst_annual_return": float(np.min(stress_years)),
                        **{
                            f"{prefix}_{name}": float(current[name])
                            for prefix, current in (("base", base), ("stress", stress))
                            for name in (
                                "terminal_return",
                                "terminal_relative_excess_return",
                                "cagr",
                                "sharpe",
                                "maximum_drawdown",
                                "calmar",
                                "positive_month_fraction",
                                "worst_month_return",
                                "average_position_count",
                                "average_cash_fraction",
                                "turnover_to_starting_cash",
                                "buy_failure_rate",
                                "sell_failure_rate",
                                "total_cost",
                            )
                        },
                    }
                )
    frame = pd.DataFrame(rows)
    support = (
        frame.groupby("slot_count", sort=False)["stable"].sum().astype(int).to_dict()
    )
    frame["stable_variant_count_for_slot"] = frame["slot_count"].map(support)
    frame["selection_eligible"] = frame["stable"] & frame[
        "stable_variant_count_for_slot"
    ].ge(int(selection["minimum_stable_variants_per_slot"]))
    return frame.sort_values(
        ["slot_count", "entry_mode", "buffer_multiplier"], kind="stable"
    ).reset_index(drop=True)


def _decision(configurations: pd.DataFrame) -> dict[str, Any]:
    eligible = configurations[configurations["selection_eligible"].astype(bool)]
    if eligible.empty:
        selected: dict[str, Any] | None = None
        overall = "d1_not_stable_under_preregistered_t1_surface"
    else:
        minimum_slots = int(eligible["slot_count"].min())
        finalists = eligible[eligible["slot_count"].astype(int).eq(minimum_slots)]
        finalists = finalists.sort_values(
            [
                "stress_calmar",
                "stress_sharpe",
                "stress_terminal_return",
                "base_terminal_return",
            ],
            ascending=[False, False, False, False],
            kind="stable",
        )
        selected = economic._json_safe(dict(finalists.iloc[0]))
        overall = "d1_has_stable_minimum_holding_t1_execution"
    return {
        "status": "completed",
        "overall": overall,
        "selected_configuration": selected,
        "stable_configuration_count": int(configurations["stable"].sum()),
        "selection_eligible_configuration_count": int(
            configurations["selection_eligible"].sum()
        ),
        "production_policy_selected": False,
        "research_semantics": "retrospective_rolling_execution_research",
    }


def evaluate(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    current = status(study_path=study_path, output_root=output_root)
    if current["completed"] != current["total"]:
        raise RuntimeError("D1 execution tasks are incomplete")
    specs = task_specs(study_path)
    results = [_task_result(output_root, spec) for spec in specs]
    task_metrics, annual_metrics, monthly_metrics = _metric_tables(
        results=results, output_root=output_root
    )
    configurations = _configuration_table(
        task_metrics=task_metrics,
        annual_metrics=annual_metrics,
        study=study,
    )
    decision = _decision(configurations)
    evaluation_root = output_root / "evaluation"
    frames = {
        "task_metrics": task_metrics,
        "annual_metrics": annual_metrics,
        "monthly_metrics": monthly_metrics,
        "configurations": configurations,
    }
    files: dict[str, dict[str, Any]] = {}
    for name, frame in frames.items():
        path = evaluation_root / f"{name}.parquet"
        _write_parquet(path, frame)
        files[name] = _file_record(path, row_count=len(frame))
    decision_path = evaluation_root / "decision.json"
    _write_json(decision_path, decision)
    files["decision"] = _file_record(decision_path)
    signal = _load_json(output_root / "signal_book/manifest.json")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "evaluated",
        "created_at": _now(),
        "study_id": STUDY_ID,
        "study_config": _file_record(study_path),
        "study_config_sha256": _study_hash(study_path),
        "signal_book": _file_record(output_root / "signal_book/manifest.json"),
        "signal_book_rank_sha256": signal["files"]["rank_panel"]["sha256"],
        "feature_count": FEATURE_COUNT,
        "target": "return_1",
        "source_label": "g_1",
        "task_count": len(specs),
        "last_entry_signal_date": LAST_ENTRY_SIGNAL_DATE,
        "last_mark_date": LAST_MARK_DATE,
        "forbidden_2026_row_count": 0,
        "research_semantics": study["period"]["research_semantics"],
        "decision": decision,
        "files": files,
    }
    _write_json(output_root / "manifest.json", manifest)
    return decision


def audit(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    study_sha256 = _study_hash(study_path)
    signal_ready = _signal_complete(output_root, study_sha256=study_sha256)
    signal = (
        _load_json(output_root / "signal_book/manifest.json") if signal_ready else {}
    )
    signal_sha256 = str(signal.get("files", {}).get("rank_panel", {}).get("sha256", ""))
    specs = task_specs(study_path)
    tasks_valid = signal_ready and all(
        _task_complete(
            output_root=output_root,
            spec=spec,
            study_sha256=study_sha256,
            signal_sha256=signal_sha256,
        )
        for spec in specs
    )
    manifest_path = output_root / "manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    output_files_valid = True
    try:
        for record in dict(manifest.get("files", {})).values():
            economic._verify_record(record)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        output_files_valid = False
    t_plus_one_valid = bool(tasks_valid)
    position_limits_valid = bool(tasks_valid)
    cash_valid = bool(tasks_valid)
    terminal_flat = bool(tasks_valid)
    output_dates_valid = bool(tasks_valid)
    accounting_valid = bool(tasks_valid)
    if tasks_valid:
        for spec in specs:
            result = _task_result(output_root, spec)
            equity = pd.read_parquet(
                economic._verify_record(result["files"]["equity"]),
                columns=["trade_date", "cash", "position_count", "conservation_error"],
            )
            trades = pd.read_parquet(
                economic._verify_record(result["files"]["trades"]),
                columns=[
                    "side",
                    "status",
                    "signal_date",
                    "execution_date",
                    "holding_days",
                ],
            )
            filled_buys = trades[
                trades["side"].astype(str).eq("buy")
                & trades["status"].astype(str).eq("filled")
            ]
            filled_sells = trades[
                trades["side"].astype(str).eq("sell")
                & trades["status"].astype(str).eq("filled")
            ]
            t_plus_one_valid &= bool(
                (
                    filled_buys["execution_date"].astype(str)
                    > filled_buys["signal_date"].astype(str)
                ).all()
                and pd.to_numeric(filled_sells["holding_days"], errors="coerce")
                .ge(1)
                .all()
            )
            position_limits_valid &= bool(
                equity["position_count"].astype(int).between(0, spec.slot_count).all()
            )
            cash_valid &= bool(equity["cash"].astype(float).ge(-1.0e-6).all())
            terminal_flat &= bool(
                int(equity["position_count"].iloc[-1]) == 0
                and int(result["metrics"]["unresolved_blocked_sell_count"]) == 0
            )
            dates = equity["trade_date"].astype(str)
            output_dates_valid &= bool(
                dates.iloc[0] == "2023-01-03"
                and dates.iloc[-1] == LAST_MARK_DATE
                and not dates.str.startswith("2026").any()
                and not trades["signal_date"].astype(str).str.startswith("2026").any()
                and not trades["execution_date"]
                .astype(str)
                .str.startswith("2026")
                .any()
            )
            accounting_valid &= bool(
                equity["conservation_error"].astype(float).max()
                <= STARTING_CASH_CNY * 1.0e-9
            )
    checks = {
        "manifest_evaluated": manifest.get("status") in {"evaluated", "audited"},
        "signal_book_valid": signal_ready,
        "source_model_contract": int(signal.get("feature_count", -1)) == FEATURE_COUNT
        and signal.get("target") == "return_1"
        and signal.get("source_label") == "g_1",
        "signal_period_exact": signal.get("first_signal_date") == "2023-01-03"
        and signal.get("last_mark_date") == LAST_MARK_DATE,
        "first_5m_vwap_coverage": float(
            signal.get("first_5m_vwap_buyable_coverage", 0.0)
        )
        >= float(study["policies"]["minimum_first_5m_vwap_coverage"]),
        "forbidden_2026_rows": int(signal.get("forbidden_2026_row_count", -1)) == 0
        and int(manifest.get("forbidden_2026_row_count", -1)) == 0,
        "task_count_exact": int(manifest.get("task_count", -1)) == len(specs),
        "task_files_valid": bool(tasks_valid),
        "t_plus_one_valid": bool(t_plus_one_valid),
        "position_limits_valid": bool(position_limits_valid),
        "cash_nonnegative": bool(cash_valid),
        "terminal_account_flat": bool(terminal_flat),
        "output_dates_valid": bool(output_dates_valid),
        "accounting_conservation_valid": bool(accounting_valid),
        "evaluation_files_valid": bool(output_files_valid),
        "no_html_report": not any(output_root.rglob("*.html")),
    }
    payload = {
        "schema": AUDIT_SCHEMA,
        "study_id": STUDY_ID,
        "status": "ok" if all(checks.values()) else "failed",
        "created_at": _now(),
        "checks": checks,
        "task_count": len(specs),
        "research_semantics": study["period"]["research_semantics"],
    }
    _write_json(output_root / "audit.json", payload)
    if payload["status"] != "ok":
        raise RuntimeError("D1 execution audit failed")
    manifest["status"] = "audited"
    manifest["audit"] = _file_record(output_root / "audit.json")
    _write_json(manifest_path, manifest)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-path", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--evaluate", action="store_true")
    actions.add_argument("--audit", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.status:
        result = status(study_path=args.study_path, output_root=args.output_root)
    elif args.prepare:
        result = prepare_signal_book(
            study_path=args.study_path,
            output_root=args.output_root,
        )
    elif args.run:
        result = run(
            study_path=args.study_path,
            output_root=args.output_root,
            max_tasks=args.max_tasks,
        )
    elif args.evaluate:
        result = evaluate(study_path=args.study_path, output_root=args.output_root)
    else:
        result = audit(study_path=args.study_path, output_root=args.output_root)
    print(json.dumps(economic._json_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
