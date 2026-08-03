from __future__ import annotations

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
STUDY_ID = "seq100_quality_liquidity_execution"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_quality_liquidity_execution.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_execution"
)
SIGNAL_SCHEMA = "seq100_quality_liquidity_execution_signal_book/1"
TASK_SCHEMA = "seq100_quality_liquidity_execution_task/1"
MANIFEST_SCHEMA = "seq100_quality_liquidity_execution_manifest/1"
AUDIT_SCHEMA = "seq100_quality_liquidity_execution_audit/1"
YEARS = (2023, 2024, 2025)
TARGETS = ("mfe_10", "mfe_20", "risk_10", "risk_20", "state_10")
FAMILIES = (
    "mfe10_primary",
    "mfe20_primary",
    "dual_mfe_maximin",
    "dual_mfe_state_veto",
    "dual_mfe_risk_veto",
    "dual_mfe_state_risk_veto",
)
SLOT_COUNTS = (6, 12, 24)
BUFFER_MULTIPLIERS = (0.0, 0.5)
COST_SCENARIOS = ("base", "stress")
PHYSICAL_COLUMNS = (
    "mfe_10_rank",
    "mfe_20_rank",
    "state_badness_10_rank",
    "state_mid_10_rank",
    "state_high_10_rank",
    "risk_10_safety_rank",
    "risk_20_safety_rank",
)
RANK_COLUMNS = (*PHYSICAL_COLUMNS, "dual_mfe_rank")
ORDER_COLUMNS = ("mfe10", "mfe20", "dual")
STARTING_CASH_CNY = 1_000_000.0


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
    temporary = path.with_suffix(path.suffix + ".tmp")
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
    if int(period["forbidden_year"]) != 2026:
        raise ValueError("2026 must remain forbidden")
    if str(period["last_entry_signal_date"]) != "2025-12-03":
        raise ValueError("last entry signal date changed")
    if str(period["last_mark_date"]) != "2025-12-31":
        raise ValueError("last mark date changed")
    contract = dict(study["model_contract"])
    if int(contract["feature_count"]) != 557:
        raise ValueError("execution study requires all 557 compact features")
    if tuple(contract["targets"]) != TARGETS:
        raise ValueError("model targets changed")
    if tuple(int(value) for value in contract["prediction_years"]) != YEARS:
        raise ValueError("prediction years changed")
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
    if tuple(policies["families"]) != FAMILIES:
        raise ValueError("policy families changed")
    if tuple(float(value) for value in policies["buffer_multipliers"]) != (
        BUFFER_MULTIPLIERS
    ):
        raise ValueError("buffer multipliers changed")
    if tuple(dict(study["execution"]["cost_scenarios"])) != COST_SCENARIOS:
        raise ValueError("cost scenarios changed")
    expected = (
        len(FAMILIES) * len(SLOT_COUNTS) * len(BUFFER_MULTIPLIERS) * len(COST_SCENARIOS)
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
            family=family,
            exposure_mode=exposure,
            slot_count=slots,
            buffer_multiplier=buffer,
            cost_scenario=cost,
        )
        for family in FAMILIES
        for slots in SLOT_COUNTS
        for buffer in BUFFER_MULTIPLIERS
        for cost in COST_SCENARIOS
    ]
    if len(specs) != 72 or len({item.task_id for item in specs}) != len(specs):
        raise AssertionError("execution task inventory changed")
    return specs


def _model_task_path(model_root: Path, target: str, year: int) -> Path:
    stage = "mfe_core" if target.startswith("mfe_") else "risk_state_core"
    return (
        model_root
        / "tasks"
        / f"{stage}__{target}__compact_core__{int(year)}"
        / "task_result.json"
    )


def _load_prediction_task(
    *, model_root: Path, target: str, year: int
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    path = _model_task_path(model_root, target, year)
    task = _load_json(path)
    if (
        task.get("status") != "completed"
        or task.get("target") != target
        or int(task.get("evaluation_year", -1)) != int(year)
        or task.get("variant") != "compact_core"
        or int(task.get("feature_count", -1)) != 557
    ):
        raise ValueError(f"invalid model task contract: {path}")
    candidate_path = economic._verify_record(task["files"]["candidate_id"])
    prediction_path = economic._verify_record(task["files"]["prediction"])
    candidate_ids = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (
        (len(candidate_ids), 3) if target == "state_10" else (len(candidate_ids),)
    )
    if prediction.shape != expected_shape:
        raise ValueError(f"prediction shape changed: {target} {year}")
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


def _daily_prediction_ranks(
    *,
    mfe10: np.ndarray,
    mfe20: np.ndarray,
    risk10: np.ndarray,
    risk20: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    vectors = (mfe10, mfe20, risk10, risk20)
    if any(np.asarray(value).ndim != 1 for value in vectors):
        raise ValueError("scalar prediction vector shape changed")
    count = len(np.asarray(mfe10))
    if any(len(np.asarray(value)) != count for value in vectors):
        raise ValueError("prediction vectors are not aligned")
    state_values = np.asarray(state, dtype=np.float64)
    if state_values.shape != (count, 3):
        raise ValueError("state prediction shape changed")
    raw = (
        np.asarray(mfe10, dtype=np.float64),
        np.asarray(mfe20, dtype=np.float64),
        state_values[:, 0] - state_values[:, 2],
        state_values[:, 1],
        state_values[:, 2],
        np.asarray(risk10, dtype=np.float64),
        np.asarray(risk20, dtype=np.float64),
    )
    if not all(bool(np.isfinite(value).all()) for value in raw):
        raise ValueError("model prediction contains non-finite values")
    return np.column_stack([economic._rank01(value) for value in raw]).astype(
        np.float32,
        copy=False,
    )


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
    pack_path = _resolve(sources["pack_manifest"])
    row_index_path = _resolve(sources["row_index"])
    model_input_manifest_path = _resolve(sources["model_input_manifest"])
    model_manifest = _load_json(model_root / "manifest.json")
    model_audit = _load_json(model_root / "audit.json")
    model_input_manifest = _load_json(model_input_manifest_path)
    compact_features = tuple(
        str(value)
        for value in model_input_manifest.get("feature_groups", {}).get(
            "compact_core", []
        )
    )
    if (
        model_manifest.get("status") != "audited"
        or model_audit.get("status") != "ok"
        or int(model_manifest.get("task_count", -1)) != 24
    ):
        raise ValueError("canonical model is not audited")
    if len(compact_features) != 557 or len(set(compact_features)) != 557:
        raise ValueError("canonical compact feature contract changed")
    if int(model_input_manifest.get("source", {}).get("forbidden_year", -1)) != 2026:
        raise ValueError("model input forbidden-year contract changed")
    model_input_record = dict(model_manifest.get("model_inputs", {}))
    if model_input_record.get("sha256") != _study_hash(
        model_input_manifest_path
    ) or model_input_record.get("input_fingerprint") != model_input_manifest.get(
        "input_fingerprint"
    ):
        raise ValueError("model and canonical input manifests are not aligned")
    rows = pd.read_parquet(
        row_index_path,
        columns=["candidate_id", "date_idx", "trade_date", "symbol"],
        filters=[
            ("trade_date", ">=", "2023-01-01"),
            ("trade_date", "<=", "2025-12-31"),
        ],
    ).sort_values("candidate_id", kind="stable")
    if rows.empty or rows["candidate_id"].duplicated().any():
        raise ValueError("model row index is empty or duplicated")
    if set(rows["trade_date"].astype(str).str[:4].astype(int)) != set(YEARS):
        raise ValueError("row index year coverage changed")
    pack = _load_json(pack_path)
    date_values = np.asarray(pack["date_values"], dtype=str)
    symbol_values = np.asarray(pack["symbol_values"], dtype=str)
    symbol_map = {str(symbol): idx for idx, symbol in enumerate(symbol_values)}
    symbol_idx_values = rows["symbol"].astype(str).map(symbol_map)
    if symbol_idx_values.isna().any():
        raise ValueError("quality-pool symbol is absent from the sequence pack")
    rows = rows.assign(symbol_idx=symbol_idx_values.astype(np.int32))
    signal_date_idx = np.sort(rows["date_idx"].astype(np.int32).unique())
    signal_dates = date_values[signal_date_idx]
    if (
        str(signal_dates[0]) != str(study["period"]["first_signal_date"])
        or str(signal_dates[-1]) != str(study["period"]["last_mark_date"])
        or any(str(value).startswith("2026") for value in signal_dates)
    ):
        raise ValueError("signal calendar changed")
    day_count = len(signal_date_idx)
    symbol_count = int(pack["symbol_count"])
    panel = np.full(
        (day_count, symbol_count, len(RANK_COLUMNS)),
        np.nan,
        dtype=np.float32,
    )
    orders = np.full(
        (len(ORDER_COLUMNS), day_count, symbol_count),
        -1,
        dtype=np.int32,
    )
    candidate_counts = np.zeros(day_count, dtype=np.int32)
    date_to_day = {int(value): idx for idx, value in enumerate(signal_date_idx)}
    source_tasks: list[dict[str, Any]] = []
    for year in YEARS:
        year_rows = rows[rows["trade_date"].astype(str).str.startswith(str(year))]
        expected_ids = year_rows["candidate_id"].to_numpy(dtype=np.int64)
        loaded: dict[str, np.ndarray] = {}
        canonical_ids: np.ndarray | None = None
        for target in TARGETS:
            task, candidate_ids, prediction = _load_prediction_task(
                model_root=model_root,
                target=target,
                year=year,
            )
            if tuple(str(value) for value in task.get("feature_names", [])) != (
                compact_features
            ):
                raise ValueError(f"model feature order changed: {target} {year}")
            current_ids = np.asarray(candidate_ids, dtype=np.int64)
            if not np.array_equal(current_ids, expected_ids):
                raise ValueError(f"prediction candidate IDs changed: {target} {year}")
            if canonical_ids is not None and not np.array_equal(
                current_ids, canonical_ids
            ):
                raise ValueError(f"head candidate IDs differ in {year}")
            canonical_ids = current_ids
            loaded[target] = np.asarray(prediction, dtype=np.float32)
            task_path = _model_task_path(model_root, target, year)
            source_tasks.append(
                {
                    "target": target,
                    "year": year,
                    "task_result": _file_record(task_path),
                    "candidate_id_sha256": task["files"]["candidate_id"]["sha256"],
                    "prediction_sha256": task["files"]["prediction"]["sha256"],
                    "feature_count": int(task["feature_count"]),
                }
            )
        date_idx = year_rows["date_idx"].to_numpy(dtype=np.int32)
        symbol_idx = year_rows["symbol_idx"].to_numpy(dtype=np.int32)
        boundaries = np.flatnonzero(np.r_[True, date_idx[1:] != date_idx[:-1], True])
        for start, stop in pairwise(boundaries):
            current_date_idx = int(date_idx[start])
            day = int(date_to_day[current_date_idx])
            symbols = np.asarray(symbol_idx[start:stop], dtype=np.int32)
            if len(np.unique(symbols)) != len(symbols):
                raise ValueError(f"duplicate symbol on {date_values[current_date_idx]}")
            ranks = _daily_prediction_ranks(
                mfe10=loaded["mfe_10"][start:stop],
                mfe20=loaded["mfe_20"][start:stop],
                risk10=loaded["risk_10"][start:stop],
                risk20=loaded["risk_20"][start:stop],
                state=loaded["state_10"][start:stop],
            )
            dual = economic._rank01(np.minimum(ranks[:, 0], ranks[:, 1]))
            panel[day, symbols, : len(PHYSICAL_COLUMNS)] = ranks
            panel[day, symbols, 7] = dual
            selectors = (ranks[:, 0], ranks[:, 1], dual)
            for order_idx, score in enumerate(selectors):
                ranked = np.lexsort((symbols, -np.asarray(score, dtype=np.float64)))
                orders[order_idx, day, : len(symbols)] = symbols[ranked]
            candidate_counts[day] = len(symbols)
        del loaded
        economic._memory_guard()
    if bool(np.any(candidate_counts <= 0)):
        raise ValueError("a signal date has no candidates")
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
            raise ValueError("signal calendar contains a gap")
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
        previous_symbols = orders[0, day - 1, : candidate_counts[day - 1]]
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
        "rank_panel": panel,
        "candidate_orders": orders,
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
    manifest = {
        "schema": SIGNAL_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "study_config_sha256": study_sha256,
        "model_manifest": _file_record(model_root / "manifest.json"),
        "model_audit": _file_record(model_root / "audit.json"),
        "model_input_manifest": _file_record(model_input_manifest_path),
        "row_index": _file_record(row_index_path),
        "pack_manifest": _file_record(pack_path),
        "source_tasks": source_tasks,
        "feature_count": 557,
        "all_557_inputs_used_by_every_head": True,
        "rank_columns": list(RANK_COLUMNS),
        "order_columns": list(ORDER_COLUMNS),
        "signal_date_count": int(day_count),
        "candidate_row_count": int(np.sum(candidate_counts)),
        "candidate_count_minimum": int(np.min(candidate_counts)),
        "candidate_count_median": float(np.median(candidate_counts)),
        "candidate_count_maximum": int(np.max(candidate_counts)),
        "first_signal_date": str(signal_dates[0]),
        "last_entry_signal_date": str(study["period"]["last_entry_signal_date"]),
        "last_mark_date": str(signal_dates[-1]),
        "benchmark": "previous_quality_pool_equal_weight_adjusted_close",
        "benchmark_missing_next_price_count": int(benchmark_missing_count),
        "forbidden_2026_row_count": 0,
        "files": files,
    }
    _write_json(signal_root / "manifest.json", manifest)
    _emit(
        "signal_book_completed",
        signal_dates=day_count,
        candidate_rows=int(np.sum(candidate_counts)),
    )
    return manifest


class SignalBook(economic.SignalBook):
    def __init__(self, *, study: Mapping[str, Any], output_root: Path) -> None:
        if not _signal_complete(output_root):
            raise FileNotFoundError("execution signal book is incomplete")
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
        self.pack_path = _resolve(self.study["sources"]["pack_manifest"])
        self.pack = _load_json(self.pack_path)
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
        self.exit_sellable = economic._source_array(
            self.pack, "masks", "exit_sellable", dtype=np.bool_
        )
        self.costs = economic.parse_execution_costs(self.pack)
        self._validate_costs()
        self.terminal_recovery_fraction = float(
            dict(self.pack.get("terminal_execution", {}) or {}).get(
                "recovery_fraction_of_entry_notional", 0.0
            )
            or 0.0
        )
        self.last_entry_signal_date = str(
            self.study["period"]["last_entry_signal_date"]
        )
        if self.date_text(self.day_count - 1) != "2025-12-31":
            raise ValueError("signal book must end on 2025-12-31")

    def trading_allowed(self, day: int) -> bool:
        return self.date_text(day) <= self.last_entry_signal_date


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
    _emit("account_tasks_starting", completed=completed, total=len(specs))
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
        result["period"]["last_entry_signal_date"] = str(
            study["period"]["last_entry_signal_date"]
        )
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
                "account_task_progress",
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


def _tail_path_tables(
    book: SignalBook, study: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = dict(study["diagnostics"])
    fractions = tuple(float(value) for value in diagnostics["tail_fractions"])
    horizons = tuple(int(value) for value in diagnostics["horizons"])
    families = tuple(str(value) for value in diagnostics["families"])
    daily_rows: list[dict[str, Any]] = []
    for day in range(book.day_count):
        signal_idx = int(book.signal_date_idx[day])
        signal_date = book.date_text(day)
        year = int(signal_date[:4])
        count = book.candidate_count(day)
        all_symbols = book.symbols_for_day("mfe10_primary", day)
        local_position = np.full(book.symbol_count, -1, dtype=np.int32)
        local_position[all_symbols] = np.arange(count, dtype=np.int32)
        signal_close = np.asarray(
            book.adjusted_close[signal_idx, all_symbols], dtype=np.float64
        )
        entry_idx = signal_idx + 1
        if entry_idx >= len(book.date_values):
            continue
        entry_open = np.asarray(
            book.adjusted_open[entry_idx, all_symbols], dtype=np.float64
        )
        open_gap = entry_open / signal_close - 1.0
        buyable = np.asarray(book.next_buyable[day, all_symbols], dtype=bool)
        for horizon in horizons:
            endpoint_idx = signal_idx + horizon
            if (
                endpoint_idx >= len(book.date_values)
                or str(book.date_values[endpoint_idx]) > "2025-12-31"
            ):
                continue
            closes = np.asarray(
                book.adjusted_close[entry_idx : endpoint_idx + 1, all_symbols],
                dtype=np.float64,
            ).T
            sellable = np.asarray(
                book.exit_sellable[entry_idx : endpoint_idx + 1, all_symbols],
                dtype=bool,
            ).T
            returns = closes / entry_open[:, None] - 1.0
            opportunity = np.where(
                sellable[:, 1:] & np.isfinite(returns[:, 1:]), returns[:, 1:], np.nan
            )
            path_valid = (
                np.isfinite(entry_open)
                & (entry_open > 0.0)
                & np.any(np.isfinite(opportunity), axis=1)
            )
            mfe = np.full(count, np.nan, dtype=np.float64)
            peak_day = np.full(count, np.nan, dtype=np.float64)
            adversity = np.full(count, np.nan, dtype=np.float64)
            valid_positions = np.flatnonzero(path_valid)
            if valid_positions.size:
                valid_opp = opportunity[valid_positions]
                peak_local = np.nanargmax(valid_opp, axis=1)
                mfe[valid_positions] = valid_opp[
                    np.arange(len(valid_positions)), peak_local
                ]
                peak_day[valid_positions] = peak_local + 2
                for row_pos, peak in zip(valid_positions, peak_local, strict=True):
                    through_peak = returns[row_pos, : int(peak) + 2]
                    finite = through_peak[np.isfinite(through_peak)]
                    adversity[row_pos] = max(
                        0.0,
                        -float(min(0.0, np.min(finite))) if finite.size else math.nan,
                    )
            fixed_return = returns[:, -1]
            endpoint_sellable = sellable[:, -1] & np.isfinite(fixed_return)
            for family in families:
                ordered = book.symbols_for_day(family, day)
                for fraction in fractions:
                    selected_count = (
                        count
                        if math.isclose(fraction, 1.0)
                        else max(1, math.ceil(count * fraction))
                    )
                    selected_symbols = ordered[:selected_count]
                    selected = local_position[selected_symbols]
                    if bool(np.any(selected < 0)):
                        raise ValueError("tail diagnostic symbol alignment failed")
                    eligible = path_valid[selected] & buyable[selected]
                    usable = selected[eligible]
                    daily_rows.append(
                        {
                            "trade_date": signal_date,
                            "year": year,
                            "family": family,
                            "horizon": horizon,
                            "tail_fraction": fraction,
                            "selected_count": int(selected_count),
                            "buyable_fraction": float(np.mean(buyable[selected])),
                            "path_valid_fraction": float(np.mean(path_valid[selected])),
                            "eligible_count": len(usable),
                            "open_gap_mean": float(np.nanmean(open_gap[usable]))
                            if len(usable)
                            else math.nan,
                            "mfe_mean": float(np.nanmean(mfe[usable]))
                            if len(usable)
                            else math.nan,
                            "mfe_median": float(np.nanmedian(mfe[usable]))
                            if len(usable)
                            else math.nan,
                            "fixed_return_mean": float(np.nanmean(fixed_return[usable]))
                            if len(usable)
                            else math.nan,
                            "fixed_return_median": float(
                                np.nanmedian(fixed_return[usable])
                            )
                            if len(usable)
                            else math.nan,
                            "pre_peak_adversity_median": float(
                                np.nanmedian(adversity[usable])
                            )
                            if len(usable)
                            else math.nan,
                            "peak_day_median": float(np.nanmedian(peak_day[usable]))
                            if len(usable)
                            else math.nan,
                            "endpoint_sellable_fraction": float(
                                np.mean(endpoint_sellable[usable])
                            )
                            if len(usable)
                            else math.nan,
                        }
                    )
    daily = pd.DataFrame(daily_rows)
    metric_columns = [
        "buyable_fraction",
        "path_valid_fraction",
        "open_gap_mean",
        "mfe_mean",
        "mfe_median",
        "fixed_return_mean",
        "fixed_return_median",
        "pre_peak_adversity_median",
        "peak_day_median",
        "endpoint_sellable_fraction",
    ]
    annual = (
        daily.groupby(["year", "family", "horizon", "tail_fraction"], sort=True)
        .agg(
            date_count=("trade_date", "count"),
            mean_selected_count=("selected_count", "mean"),
            mean_eligible_count=("eligible_count", "mean"),
            **{name: (name, "mean") for name in metric_columns},
        )
        .reset_index()
    )
    overall = (
        daily.groupby(["family", "horizon", "tail_fraction"], sort=True)
        .agg(
            date_count=("trade_date", "count"),
            mean_selected_count=("selected_count", "mean"),
            mean_eligible_count=("eligible_count", "mean"),
            **{name: (name, "mean") for name in metric_columns},
        )
        .reset_index()
    )
    overall.insert(0, "year", 0)
    annual = pd.concat([annual, overall], ignore_index=True)
    return daily, annual


def _configuration_table(
    metrics: pd.DataFrame, study: Mapping[str, Any]
) -> pd.DataFrame:
    selection = dict(study["selection"])
    indexed = {
        (
            str(row["family"]),
            int(row["slot_count"]),
            float(row["buffer_multiplier"]),
            str(row["cost_scenario"]),
        ): dict(row)
        for row in metrics.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for slots in SLOT_COUNTS:
            for buffer in BUFFER_MULTIPLIERS:
                base = indexed[(family, slots, buffer, "base")]
                stress = indexed[(family, slots, buffer, "stress")]
                economic_pass = bool(
                    base["positive_absolute_years"]
                    >= int(selection["minimum_positive_absolute_years"])
                    and stress["positive_absolute_years"]
                    >= int(selection["minimum_positive_absolute_years"])
                    and base["positive_excess_years"]
                    >= int(selection["minimum_positive_excess_years"])
                    and stress["positive_excess_years"]
                    >= int(selection["minimum_positive_excess_years"])
                    and float(base["terminal_cost_accrued_return"]) > 0.0
                    and float(stress["terminal_cost_accrued_return"]) > 0.0
                    and float(base["terminal_cost_accrued_relative_excess_return"])
                    > 0.0
                    and float(stress["terminal_cost_accrued_relative_excess_return"])
                    > 0.0
                )
                rows.append(
                    {
                        "family": family,
                        "slot_count": slots,
                        "buffer_multiplier": buffer,
                        "economic_pass": economic_pass,
                        **{
                            f"{prefix}_{name}": current[name]
                            for prefix, current in (("base", base), ("stress", stress))
                            for name in (
                                "terminal_cost_accrued_return",
                                "terminal_cost_accrued_relative_excess_return",
                                "maximum_drawdown",
                                "turnover_to_starting_cash",
                                "positive_absolute_years",
                                "positive_excess_years",
                                "buy_failure_rate",
                                "sell_failure_rate",
                                "fraction_above_0.001",
                                "fraction_above_0.005",
                                "fraction_above_0.01",
                            )
                        },
                    }
                )
    return pd.DataFrame(rows)


def _stable_regions(configurations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for buffer in BUFFER_MULTIPLIERS:
            current = configurations[
                configurations["family"].astype(str).eq(family)
                & configurations["buffer_multiplier"].astype(float).eq(buffer)
            ]
            passing = sorted(
                current.loc[current["economic_pass"].astype(bool), "slot_count"]
                .astype(int)
                .tolist()
            )
            pairs = [
                (left, right)
                for left, right in pairwise(SLOT_COUNTS)
                if left in passing and right in passing
            ]
            rows.append(
                {
                    "family": family,
                    "buffer_multiplier": buffer,
                    "passing_slot_counts": json.dumps(passing),
                    "adjacent_passing_pairs": json.dumps(pairs),
                    "robust_region": bool(pairs),
                }
            )
    return pd.DataFrame(rows)


def _buffer_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    indexed = {
        (
            str(row["family"]),
            int(row["slot_count"]),
            float(row["buffer_multiplier"]),
            str(row["cost_scenario"]),
        ): dict(row)
        for row in metrics.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for slots in SLOT_COUNTS:
            for cost in COST_SCENARIOS:
                plain = indexed[(family, slots, 0.0, cost)]
                buffered = indexed[(family, slots, 0.5, cost)]
                rows.append(
                    {
                        "family": family,
                        "slot_count": slots,
                        "cost_scenario": cost,
                        "terminal_return_delta": float(
                            buffered["terminal_cost_accrued_return"]
                            - plain["terminal_cost_accrued_return"]
                        ),
                        "terminal_excess_delta": float(
                            buffered["terminal_cost_accrued_relative_excess_return"]
                            - plain["terminal_cost_accrued_relative_excess_return"]
                        ),
                        "turnover_delta": float(
                            buffered["turnover_to_starting_cash"]
                            - plain["turnover_to_starting_cash"]
                        ),
                        "maximum_drawdown_delta": float(
                            buffered["maximum_drawdown"] - plain["maximum_drawdown"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def evaluate(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    current_status = status(study_path=study_path, output_root=output_root)
    if current_status["completed"] != current_status["total"]:
        raise RuntimeError("execution tasks are incomplete")
    specs = task_specs(study_path)
    results = [_task_result(output_root, spec) for spec in specs]
    metrics = pd.DataFrame([economic._task_metric_row(result) for result in results])
    annual = economic._annual_metric_table(results)
    configurations = _configuration_table(metrics, study)
    regions = _stable_regions(configurations)
    buffer_deltas = _buffer_deltas(metrics)
    book = SignalBook(study=study, output_root=output_root)
    tail_daily, tail_annual = _tail_path_tables(book, study)
    accounting_invalid = bool(
        metrics["maximum_conservation_error"].astype(float).max()
        > STARTING_CASH_CNY * 1.0e-9
    )
    robust = regions[regions["robust_region"].astype(bool)]
    passing_cell_count = int(configurations["economic_pass"].astype(bool).sum())
    isolated_count = 0
    for row in configurations[configurations["economic_pass"].astype(bool)].to_dict(
        "records"
    ):
        peers = configurations[
            configurations["family"].astype(str).eq(str(row["family"]))
            & configurations["buffer_multiplier"]
            .astype(float)
            .eq(float(row["buffer_multiplier"]))
            & configurations["economic_pass"].astype(bool)
        ]["slot_count"].astype(int)
        slot = int(row["slot_count"])
        slot_position = SLOT_COUNTS.index(slot)
        neighbours = {
            SLOT_COUNTS[index]
            for index in (slot_position - 1, slot_position + 1)
            if 0 <= index < len(SLOT_COUNTS)
        }
        if neighbours.isdisjoint(set(peers.tolist())):
            isolated_count += 1
    if accounting_invalid:
        overall = "execution_audit_invalid_due_to_accounting"
    elif not robust.empty:
        overall = "compact_model_has_robust_simple_execution_region"
    elif isolated_count:
        overall = "compact_model_has_isolated_execution_cells_only"
    else:
        overall = "compact_model_not_monetized_by_simple_execution_surface"
    decision = {
        "status": "completed_without_production_policy_selection",
        "overall": overall,
        "research_semantics": study["period"]["research_semantics"],
        "robust_regions": robust.to_dict("records"),
        "robust_region_count": len(robust),
        "economic_passing_cell_count": passing_cell_count,
        "isolated_economic_cell_count": isolated_count,
        "accounting_invalid": accounting_invalid,
        "production_policy_selected": False,
        "next_step": (
            "run execution-price sensitivity only for robust regions"
            if len(robust)
            else "attribute the failure to selection, exit timing, costs, or fill constraints"
        ),
    }
    evaluation_root = output_root / "evaluation"
    output_frames = {
        "task_metrics": metrics,
        "annual_metrics": annual,
        "configurations": configurations,
        "stable_regions": regions,
        "buffer_deltas": buffer_deltas,
        "tail_path_daily": tail_daily,
        "tail_path_annual": tail_annual,
    }
    files: dict[str, dict[str, Any]] = {}
    for name, frame in output_frames.items():
        path = evaluation_root / f"{name}.parquet"
        _write_parquet(path, frame)
        files[name] = _file_record(path, row_count=len(frame))
    decision_path = evaluation_root / "decision.json"
    _write_json(decision_path, decision)
    files["decision"] = _file_record(decision_path)
    signal_manifest = _load_json(output_root / "signal_book/manifest.json")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "evaluated",
        "created_at": _now(),
        "study_id": STUDY_ID,
        "study_config": _file_record(study_path),
        "study_config_sha256": _study_hash(study_path),
        "signal_book": _file_record(output_root / "signal_book/manifest.json"),
        "signal_book_rank_sha256": signal_manifest["files"]["rank_panel"]["sha256"],
        "feature_count": 557,
        "all_557_inputs_used_by_every_head": True,
        "task_count": len(specs),
        "research_semantics": study["period"]["research_semantics"],
        "last_entry_signal_date": study["period"]["last_entry_signal_date"],
        "last_mark_date": study["period"]["last_mark_date"],
        "forbidden_2026_row_count": 0,
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
    manifest_path = output_root / "manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
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
    task_periods_valid = bool(tasks_valid)
    output_dates_valid = bool(tasks_valid)
    trading_cutoff_valid = bool(tasks_valid)
    if tasks_valid:
        cutoff = str(study["period"]["last_entry_signal_date"])
        last_mark = str(study["period"]["last_mark_date"])
        for spec in specs:
            result = _task_result(output_root, spec)
            period = dict(result.get("period", {}))
            task_periods_valid &= (
                period.get("last_entry_signal_date") == cutoff
                and period.get("last_mark_date") == last_mark
                and int(period.get("forbidden_2026_row_count", -1)) == 0
            )
            equity = pd.read_parquet(
                _resolve(result["files"]["equity"]["path"]),
                columns=["trade_date"],
            )
            equity_dates = equity["trade_date"].astype(str)
            output_dates_valid &= bool(
                len(equity_dates) == int(signal.get("signal_date_count", -1))
                and equity_dates.iloc[0] == "2023-01-03"
                and equity_dates.iloc[-1] == last_mark
                and not equity_dates.str.startswith("2026").any()
            )
            trades = pd.read_parquet(
                _resolve(result["files"]["trades"]["path"]),
                columns=["side", "signal_date", "execution_date"],
            )
            active = trades["side"].astype(str).isin({"buy", "sell"})
            if bool(active.any()):
                trading_cutoff_valid &= bool(
                    trades.loc[active, "signal_date"].astype(str).le(cutoff).all()
                    and trades.loc[active, "execution_date"]
                    .astype(str)
                    .le(last_mark)
                    .all()
                )
            output_dates_valid &= bool(
                not trades["signal_date"].astype(str).str.startswith("2026").any()
                and not trades["execution_date"]
                .astype(str)
                .str.startswith("2026")
                .any()
            )
    output_files_valid = True
    try:
        for record in dict(manifest.get("files", {})).values():
            economic._verify_record(record)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        output_files_valid = False
    accounting_valid = False
    if output_files_valid and "task_metrics" in manifest.get("files", {}):
        metrics = pd.read_parquet(
            economic._verify_record(manifest["files"]["task_metrics"]),
            columns=["maximum_conservation_error"],
        )
        accounting_valid = bool(
            metrics["maximum_conservation_error"].astype(float).max()
            <= STARTING_CASH_CNY * 1.0e-9
        )
    checks = {
        "manifest_evaluated": manifest.get("status") in {"evaluated", "audited"},
        "signal_book_valid": signal_ready,
        "all_557_features_used": int(signal.get("feature_count", -1)) == 557
        and bool(signal.get("all_557_inputs_used_by_every_head")),
        "candidate_row_count_positive": int(signal.get("candidate_row_count", 0)) > 0,
        "signal_years_exact": signal.get("first_signal_date") == "2023-01-03"
        and signal.get("last_mark_date") == "2025-12-31",
        "last_entry_signal_frozen": signal.get("last_entry_signal_date")
        == "2025-12-03",
        "forbidden_2026_rows": int(signal.get("forbidden_2026_row_count", -1)) == 0
        and int(manifest.get("forbidden_2026_row_count", -1)) == 0,
        "task_count_exact": int(manifest.get("task_count", -1)) == 72,
        "task_files_valid": bool(tasks_valid),
        "task_period_contracts_valid": bool(task_periods_valid),
        "output_dates_valid": bool(output_dates_valid),
        "actual_trading_cutoff_valid": bool(trading_cutoff_valid),
        "evaluation_files_valid": bool(output_files_valid),
        "accounting_conservation_valid": bool(accounting_valid),
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
        raise RuntimeError("quality-liquidity execution audit failed")
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
