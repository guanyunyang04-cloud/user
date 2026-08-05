from __future__ import annotations

"""Explore executable low-position policies for the two strict T+1 models."""

import argparse
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_quality_liquidity_d1_execution as d1
from daily_research.path_policy import seq100_v4_economic_realizability as economic

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_quality_liquidity_t1_execution"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_quality_liquidity_t1_execution.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies"
    / "seq100_quality_liquidity_t1_return_models/execution"
)
SIGNAL_SCHEMA = "seq100_quality_liquidity_t1_execution_signal_book/1"
TASK_SCHEMA = "seq100_quality_liquidity_t1_execution_task/1"
MANIFEST_SCHEMA = "seq100_quality_liquidity_t1_execution_manifest/1"
AUDIT_SCHEMA = "seq100_quality_liquidity_t1_execution_audit/1"

YEARS = (2023, 2024, 2025)
TARGETS = ("open_to_open_1", "open_to_close_2")
SCORE_FAMILIES = (
    "open_to_open",
    "open_to_close_2",
    "blend_open_weight_25",
    "blend_open_weight_50",
    "blend_open_weight_75",
    "agreement_min",
)
SCORE_OPEN_WEIGHTS = {
    "open_to_open": 1.0,
    "open_to_close_2": 0.0,
    "blend_open_weight_25": 0.25,
    "blend_open_weight_50": 0.50,
    "blend_open_weight_75": 0.75,
}
ENTRY_MODES = ("open", "first_5m_vwap")
EXIT_MODES = (
    "rolling_open",
    "fixed_open_2",
    "fixed_close_2",
    "renewed_close_2",
)
CONFIDENCE_MODES = (
    "always",
    "causal_top_score_q50",
    "causal_top_score_q80",
)
FIXED_CLOSE_CADENCES = ("full_cohort", "staggered")
SLOT_COUNTS = (1, 2, 3)
ROLLING_BUFFERS = (0.0, 1.0, 2.0, 5.0, 10.0)
RENEWED_CLOSE_BUFFERS = (0.0, 1.0, 2.0, 5.0, 10.0)
COST_SCENARIOS = ("base", "stress")
STARTING_CASH_CNY = 1_000_000.0
FEATURE_COUNT = 557
FIRST_SIGNAL_DATE = "2023-01-03"
LAST_ENTRY_SIGNAL_DATE = "2025-12-29"
LAST_MARK_DATE = "2025-12-31"


@dataclass(frozen=True)
class ExecutionSpec:
    score_family: str
    entry_mode: str
    exit_mode: str
    cadence: str
    confidence_mode: str
    slot_count: int
    buffer_multiplier: float
    cost_scenario: str

    @property
    def task_id(self) -> str:
        buffer_text = str(float(self.buffer_multiplier)).replace(".", "p")
        return (
            f"{self.score_family}__{self.entry_mode}__{self.exit_mode}"
            f"__{self.cadence}__{self.confidence_mode}__k{self.slot_count:02d}"
            f"__b{buffer_text}__{self.cost_scenario}"
        )


@dataclass(frozen=True)
class FixedBuy:
    signal_day: int
    symbol_idx: int
    reason: str


def _resolve(value: str | Path) -> Path:
    return d1._resolve(value)


def _load_json(path: Path) -> dict[str, Any]:
    return d1._load_json(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    d1._write_json(path, payload)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    d1._write_parquet(path, frame)


def _file_record(path: Path, **extra: Any) -> dict[str, Any]:
    return d1._file_record(path, **extra)


def _study_hash(path: Path) -> str:
    return d1._study_hash(path)


def _engine_sha256() -> str:
    return economic._sha256(Path(__file__).resolve())


def _emit(event: str, **payload: Any) -> None:
    d1._emit(event, **payload)


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _load_json(path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError(f"study_id must be {STUDY_ID}")
    period = dict(study["period"])
    if tuple(int(value) for value in period["signal_years"]) != YEARS:
        raise ValueError("signal years changed")
    if str(period["first_signal_date"]) != FIRST_SIGNAL_DATE:
        raise ValueError("first signal date changed")
    if str(period["last_entry_signal_date"]) != LAST_ENTRY_SIGNAL_DATE:
        raise ValueError("last entry signal date changed")
    if str(period["last_mark_date"]) != LAST_MARK_DATE:
        raise ValueError("last mark date changed")
    if int(period["forbidden_year"]) != 2026:
        raise ValueError("2026 must remain forbidden")
    contract = dict(study["model_contract"])
    if (
        tuple(str(value) for value in contract["targets"]) != TARGETS
        or str(contract["variant"]) != "compact_core"
        or int(contract["feature_count"]) != FEATURE_COUNT
        or bool(contract["candidate_selection_uses_future_fill"])
    ):
        raise ValueError("strict T+1 model contract changed")
    signals = dict(study["signals"])
    if tuple(str(value) for value in signals["score_families"]) != SCORE_FAMILIES:
        raise ValueError("score family grid changed")
    if tuple(str(value) for value in signals["confidence_modes"]) != (CONFIDENCE_MODES):
        raise ValueError("confidence grid changed")
    account = dict(study["account"])
    if tuple(int(value) for value in account["slot_counts"]) != SLOT_COUNTS:
        raise ValueError("slot grid changed")
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
    if tuple(str(value) for value in policies["exit_modes"]) != EXIT_MODES:
        raise ValueError("exit modes changed")
    if tuple(str(value) for value in policies["fixed_close_cadences"]) != (
        FIXED_CLOSE_CADENCES
    ):
        raise ValueError("fixed-close cadence grid changed")
    if tuple(float(value) for value in policies["rolling_buffer_multipliers"]) != (
        ROLLING_BUFFERS
    ):
        raise ValueError("rolling buffer grid changed")
    if tuple(
        float(value) for value in policies["renewed_close_buffer_multipliers"]
    ) != (RENEWED_CLOSE_BUFFERS):
        raise ValueError("renewed-close buffer grid changed")
    if tuple(dict(study["execution"]["cost_scenarios"])) != COST_SCENARIOS:
        raise ValueError("cost scenarios changed")
    expected = len(task_specs(path, validate_study=False))
    if int(study["outputs"]["expected_task_count"]) != expected:
        raise ValueError("expected task count changed")
    return study


def task_specs(
    study_path: Path = DEFAULT_STUDY_PATH,
    *,
    validate_study: bool = True,
) -> list[ExecutionSpec]:
    if validate_study:
        load_study(study_path)
    specs: list[ExecutionSpec] = []
    for family in SCORE_FAMILIES:
        for entry_mode in ENTRY_MODES:
            for confidence in CONFIDENCE_MODES:
                for slots in SLOT_COUNTS:
                    for cost in COST_SCENARIOS:
                        for buffer in ROLLING_BUFFERS:
                            specs.append(
                                ExecutionSpec(
                                    score_family=family,
                                    entry_mode=entry_mode,
                                    exit_mode="rolling_open",
                                    cadence="daily",
                                    confidence_mode=confidence,
                                    slot_count=slots,
                                    buffer_multiplier=buffer,
                                    cost_scenario=cost,
                                )
                            )
                        specs.append(
                            ExecutionSpec(
                                score_family=family,
                                entry_mode=entry_mode,
                                exit_mode="fixed_open_2",
                                cadence="full_cohort",
                                confidence_mode=confidence,
                                slot_count=slots,
                                buffer_multiplier=0.0,
                                cost_scenario=cost,
                            )
                        )
                        close_cadences = (
                            ("full_cohort",) if slots == 1 else FIXED_CLOSE_CADENCES
                        )
                        for cadence in close_cadences:
                            specs.append(
                                ExecutionSpec(
                                    score_family=family,
                                    entry_mode=entry_mode,
                                    exit_mode="fixed_close_2",
                                    cadence=cadence,
                                    confidence_mode=confidence,
                                    slot_count=slots,
                                    buffer_multiplier=0.0,
                                    cost_scenario=cost,
                                )
                            )
                            for buffer in RENEWED_CLOSE_BUFFERS:
                                specs.append(
                                    ExecutionSpec(
                                        score_family=family,
                                        entry_mode=entry_mode,
                                        exit_mode="renewed_close_2",
                                        cadence=cadence,
                                        confidence_mode=confidence,
                                        slot_count=slots,
                                        buffer_multiplier=buffer,
                                        cost_scenario=cost,
                                    )
                                )
    if len(specs) != 3456 or len({item.task_id for item in specs}) != len(specs):
        raise AssertionError("T1 execution task inventory changed")
    return specs


def _model_task_path(model_root: Path, *, target: str, year: int) -> Path:
    return (
        model_root
        / "tasks"
        / f"t1_return_zscore__{target}__compact_core__{int(year)}"
        / "task_result.json"
    )


def _load_prediction_task(
    *, model_root: Path, target: str, year: int
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    path = _model_task_path(model_root, target=target, year=year)
    task = _load_json(path)
    if (
        task.get("status") != "completed"
        or task.get("target") != target
        or int(task.get("evaluation_year", -1)) != int(year)
        or task.get("variant") != "compact_core"
        or int(task.get("feature_count", -1)) != FEATURE_COUNT
        or int(task.get("horizon", -1)) != 2
    ):
        raise ValueError(f"invalid strict T+1 model task contract: {path}")
    candidate_ids = np.load(
        economic._verify_record(task["files"]["candidate_id"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    prediction = np.load(
        economic._verify_record(task["files"]["prediction"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    if candidate_ids.shape != prediction.shape:
        raise ValueError(f"prediction shape changed: {target} {year}")
    return task, candidate_ids, prediction


def _family_rank_and_confidence(
    open_prediction: np.ndarray,
    close_prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    open_values = np.asarray(open_prediction, dtype=np.float64)
    close_values = np.asarray(close_prediction, dtype=np.float64)
    if (
        open_values.ndim != 1
        or close_values.shape != open_values.shape
        or not bool(np.isfinite(open_values).all())
        or not bool(np.isfinite(close_values).all())
    ):
        raise ValueError("family score inputs must be aligned finite vectors")
    open_rank = economic._rank01(open_values)
    close_rank = economic._rank01(close_values)
    ranks = np.empty((len(SCORE_FAMILIES), len(open_values)), dtype=np.float32)
    confidence = np.empty_like(ranks)
    for family_idx, family in enumerate(SCORE_FAMILIES):
        if family == "agreement_min":
            composite_rank = np.minimum(open_rank, close_rank)
            raw_confidence = np.minimum(open_values, close_values)
        else:
            weight = float(SCORE_OPEN_WEIGHTS[family])
            composite_rank = weight * open_rank + (1.0 - weight) * close_rank
            raw_confidence = weight * open_values + (1.0 - weight) * close_values
        ranks[family_idx] = economic._rank01(composite_rank)
        confidence[family_idx] = raw_confidence.astype(np.float32)
    return ranks, confidence


def _causal_percentile(
    values: np.ndarray,
    *,
    lookback: int,
    minimum_history: int,
) -> np.ndarray:
    current = np.asarray(values, dtype=np.float64)
    if current.ndim != 1 or not bool(np.isfinite(current).all()):
        raise ValueError("causal percentile input must be a finite vector")
    if lookback <= 0 or minimum_history <= 0 or minimum_history > lookback:
        raise ValueError("invalid causal percentile window")
    output = np.ones(len(current), dtype=np.float32)
    for index, value in enumerate(current):
        start = max(0, index - int(lookback))
        history = current[start:index]
        if len(history) < int(minimum_history):
            continue
        lower = float(np.sum(history < value))
        equal = float(np.sum(history == value))
        output[index] = np.float32((lower + 0.5 * equal) / len(history))
    return output


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
        if manifest.get("signal_builder_source_sha256") != _engine_sha256():
            return False
        for record in dict(manifest["files"]).values():
            economic._verify_record(record)
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _load_aligned_first_5m_vwap(
    *,
    rows: pd.DataFrame,
    close_d1_manifest_path: Path,
) -> np.ndarray:
    close_manifest = _load_json(close_d1_manifest_path)
    if close_manifest.get("status") != "audited":
        raise ValueError("first-5m execution cache is not audited")
    record = dict(close_manifest["preparation_files"]["execution_outcomes"])
    outcomes = pd.read_parquet(
        economic._verify_record(record),
        columns=["candidate_id", "first_5m_vwap", "outcome_valid"],
    ).sort_values("candidate_id", kind="stable")
    expected_ids = rows["candidate_id"].to_numpy(dtype=np.int64)
    outcome_ids = outcomes["candidate_id"].to_numpy(dtype=np.int64)
    if outcomes["candidate_id"].duplicated().any() or bool(
        np.setdiff1d(outcome_ids, expected_ids, assume_unique=True).size
    ):
        raise ValueError("first-5m execution outcomes are not a model-row subset")
    valid = outcomes["outcome_valid"].to_numpy(dtype=bool)
    values = outcomes["first_5m_vwap"].to_numpy(dtype=np.float32)
    valid &= np.isfinite(values) & (values > 0.0)
    values = np.where(valid, values, np.nan).astype(np.float32)
    aligned = np.full(len(expected_ids), np.nan, dtype=np.float32)
    positions = np.searchsorted(outcome_ids, expected_ids)
    matched = positions < len(outcome_ids)
    matched[matched] &= outcome_ids[positions[matched]] == expected_ids[matched]
    aligned[matched] = values[positions[matched]]
    return aligned


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
    model_manifest = _load_json(model_root / "manifest.json")
    model_audit = _load_json(model_root / "audit.json")
    input_manifest = _load_json(input_manifest_path)
    compact_features = tuple(
        str(value)
        for value in input_manifest.get("feature_groups", {}).get("compact_core", [])
    )
    if (
        model_manifest.get("status") != "audited"
        or model_audit.get("status") != "ok"
        or int(model_audit.get("task_count", -1)) != 6
    ):
        raise ValueError("strict T+1 model study is not audited")
    if len(compact_features) != FEATURE_COUNT or len(set(compact_features)) != (
        FEATURE_COUNT
    ):
        raise ValueError("compact feature contract changed")
    rows = pd.read_parquet(
        row_index_path,
        columns=["candidate_id", "date_idx", "trade_date", "symbol"],
        filters=[
            ("trade_date", ">=", "2023-01-01"),
            ("trade_date", "<=", LAST_MARK_DATE),
        ],
    ).sort_values("candidate_id", kind="stable")
    if rows.empty or rows["candidate_id"].duplicated().any():
        raise ValueError("T1 execution row index is empty or duplicated")
    if set(rows["trade_date"].astype(str).str[:4].astype(int)) != set(YEARS):
        raise ValueError("T1 execution row-index year coverage changed")
    aligned_vwap = _load_aligned_first_5m_vwap(
        rows=rows,
        close_d1_manifest_path=close_d1_manifest_path,
    )
    pack = _load_json(pack_path)
    date_values = np.asarray(pack["date_values"], dtype=str)
    symbol_values = np.asarray(pack["symbol_values"], dtype=str)
    symbol_map = {str(symbol): index for index, symbol in enumerate(symbol_values)}
    symbol_indices = rows["symbol"].astype(str).map(symbol_map)
    if symbol_indices.isna().any():
        raise ValueError("quality-pool symbol is absent from the sequence pack")
    rows = rows.assign(symbol_idx=symbol_indices.astype(np.int32))
    signal_date_idx = np.sort(rows["date_idx"].astype(np.int32).unique())
    signal_dates = date_values[signal_date_idx]
    if (
        str(signal_dates[0]) != FIRST_SIGNAL_DATE
        or str(signal_dates[-1]) != LAST_MARK_DATE
        or any(str(value).startswith("2026") for value in signal_dates)
    ):
        raise ValueError("T1 execution signal calendar changed")
    if bool(np.any(np.diff(signal_date_idx) != 1)):
        raise ValueError("T1 execution signal calendar contains a pack-date gap")
    day_count = len(signal_date_idx)
    symbol_count = int(pack["symbol_count"])
    rank_panel = np.full(
        (len(SCORE_FAMILIES), day_count, symbol_count),
        np.nan,
        dtype=np.float32,
    )
    orders = np.full(
        (len(SCORE_FAMILIES), day_count, symbol_count),
        -1,
        dtype=np.int32,
    )
    candidate_counts = np.zeros(day_count, dtype=np.int32)
    first_5m_vwap = np.full((day_count, symbol_count), np.nan, dtype=np.float32)
    top_confidence = np.full((len(SCORE_FAMILIES), day_count), np.nan, dtype=np.float32)
    date_to_day = {int(value): index for index, value in enumerate(signal_date_idx)}
    source_tasks: list[dict[str, Any]] = []
    row_offset = 0
    for year in YEARS:
        year_rows = rows[rows["trade_date"].astype(str).str.startswith(str(year))]
        year_count = len(year_rows)
        expected_ids = year_rows["candidate_id"].to_numpy(dtype=np.int64)
        loaded: dict[str, np.ndarray] = {}
        canonical_ids: np.ndarray | None = None
        for target in TARGETS:
            task, candidate_ids, prediction = _load_prediction_task(
                model_root=model_root,
                target=target,
                year=year,
            )
            current_ids = np.asarray(candidate_ids, dtype=np.int64)
            if not np.array_equal(current_ids, expected_ids):
                raise ValueError(f"prediction candidate IDs changed: {target} {year}")
            if canonical_ids is not None and not np.array_equal(
                canonical_ids, current_ids
            ):
                raise ValueError(f"target prediction candidate IDs differ: {year}")
            if tuple(str(value) for value in task.get("feature_names", [])) != (
                compact_features
            ):
                raise ValueError(f"feature order changed: {target} {year}")
            canonical_ids = current_ids
            loaded[target] = np.asarray(prediction, dtype=np.float32)
            source_tasks.append(
                {
                    "target": target,
                    "year": year,
                    "task_result": _file_record(
                        _model_task_path(model_root, target=target, year=year)
                    ),
                    "candidate_id_sha256": task["files"]["candidate_id"]["sha256"],
                    "prediction_sha256": task["files"]["prediction"]["sha256"],
                    "feature_count": int(task["feature_count"]),
                }
            )
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
            ranks, confidence = _family_rank_and_confidence(
                loaded["open_to_open_1"][start:stop],
                loaded["open_to_close_2"][start:stop],
            )
            for family_idx in range(len(SCORE_FAMILIES)):
                order = np.lexsort((symbols, -ranks[family_idx].astype(np.float64)))
                rank_panel[family_idx, day, symbols] = ranks[family_idx]
                orders[family_idx, day, : len(symbols)] = symbols[order]
                top_confidence[family_idx, day] = confidence[family_idx, int(order[0])]
            candidate_counts[day] = len(symbols)
            first_5m_vwap[day, symbols] = year_vwap[start:stop]
        del loaded
        economic._memory_guard()
    if row_offset != len(rows) or bool(np.any(candidate_counts <= 0)):
        raise ValueError("T1 signal preparation did not consume every model row")
    signals = dict(study["signals"])
    confidence_percentile = np.vstack(
        [
            _causal_percentile(
                top_confidence[index],
                lookback=int(signals["confidence_lookback_days"]),
                minimum_history=int(signals["confidence_min_history_days"]),
            )
            for index in range(len(SCORE_FAMILIES))
        ]
    ).astype(np.float32)
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
    next_open_sellable = np.zeros((day_count, symbol_count), dtype=np.bool_)
    for day in range(day_count - 1):
        signal_idx = int(signal_date_idx[day])
        execution_idx = int(signal_date_idx[day + 1])
        next_buyable[day] = np.asarray(entry_filled[signal_idx], dtype=bool)
        next_open_sellable[day] = economic.derive_open_sellable(
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
        symbols = orders[0, day - 1, : candidate_counts[day - 1]]
        previous_idx = int(signal_date_idx[day - 1])
        current_idx = int(signal_date_idx[day])
        previous_close = np.asarray(
            adjusted_close[previous_idx, symbols], dtype=np.float64
        )
        current_close = np.asarray(
            adjusted_close[current_idx, symbols], dtype=np.float64
        )
        valid_previous = np.isfinite(previous_close) & (previous_close > 0.0)
        returns = np.full(len(symbols), terminal_recovery - 1.0, dtype=np.float64)
        valid = valid_previous & np.isfinite(current_close) & (current_close > 0.0)
        returns[valid] = current_close[valid] / previous_close[valid] - 1.0
        benchmark_missing_count += int(np.sum(valid_previous & ~valid))
        benchmark_returns[day] = float(np.mean(returns[valid_previous]))
    signal_root = output_root / "signal_book"
    arrays = {
        "signal_date_idx": signal_date_idx.astype(np.int32),
        "candidate_counts": candidate_counts,
        "rank_panel": rank_panel,
        "candidate_orders": orders,
        "top_confidence": top_confidence,
        "causal_confidence_percentile": confidence_percentile,
        "first_5m_vwap": first_5m_vwap,
        "next_open_buyable": next_buyable,
        "next_open_sellable": next_open_sellable,
        "benchmark_returns": benchmark_returns,
    }
    files: dict[str, dict[str, Any]] = {}
    for name, values in arrays.items():
        path = signal_root / f"{name}.npy"
        economic._save_npy(path, values)
        files[name] = _file_record(
            path, shape=list(values.shape), dtype=str(values.dtype)
        )
    candidate_support = np.isfinite(rank_panel[0])
    entry_signal_days = signal_dates <= LAST_ENTRY_SIGNAL_DATE
    buyable_support = (
        candidate_support[entry_signal_days] & next_buyable[entry_signal_days]
    )
    vwap_valid = np.isfinite(first_5m_vwap[entry_signal_days]) & buyable_support
    vwap_coverage = float(np.sum(vwap_valid) / max(np.sum(buyable_support), 1))
    minimum_vwap = float(study["policies"]["minimum_first_5m_vwap_coverage"])
    if vwap_coverage < minimum_vwap:
        raise ValueError(
            f"first-5m VWAP coverage {vwap_coverage:.8f} below {minimum_vwap:.8f}"
        )
    manifest = {
        "schema": SIGNAL_SCHEMA,
        "status": "completed",
        "completed_at": d1._now(),
        "study_id": STUDY_ID,
        "study_config_sha256": study_sha256,
        "signal_builder_source_sha256": _engine_sha256(),
        "model_manifest": _file_record(model_root / "manifest.json"),
        "model_audit": _file_record(model_root / "audit.json"),
        "model_input_manifest": _file_record(input_manifest_path),
        "row_index": _file_record(row_index_path),
        "pack_manifest": _file_record(pack_path),
        "first_5m_source_manifest": _file_record(close_d1_manifest_path),
        "source_tasks": source_tasks,
        "targets": list(TARGETS),
        "score_families": list(SCORE_FAMILIES),
        "feature_count": FEATURE_COUNT,
        "signal_date_count": int(day_count),
        "candidate_row_count": int(np.sum(candidate_counts)),
        "candidate_count_minimum": int(np.min(candidate_counts)),
        "candidate_count_median": float(np.median(candidate_counts)),
        "candidate_count_maximum": int(np.max(candidate_counts)),
        "first_5m_vwap_buyable_coverage": vwap_coverage,
        "first_signal_date": str(signal_dates[0]),
        "last_entry_signal_date": LAST_ENTRY_SIGNAL_DATE,
        "last_mark_date": str(signal_dates[-1]),
        "confidence_is_causal": True,
        "candidate_selection_uses_future_fill": False,
        "benchmark": "previous_quality_pool_equal_weight_adjusted_close",
        "benchmark_missing_next_price_count": int(benchmark_missing_count),
        "forbidden_2026_row_count": 0,
        "files": files,
    }
    _write_json(signal_root / "manifest.json", manifest)
    _emit(
        "t1_execution_signal_book_completed",
        signal_dates=day_count,
        candidate_rows=int(np.sum(candidate_counts)),
        first_5m_vwap_buyable_coverage=vwap_coverage,
    )
    return manifest


class SignalBook:
    def __init__(self, *, study: Mapping[str, Any], output_root: Path) -> None:
        if not _signal_complete(output_root):
            raise FileNotFoundError("T1 execution signal book is incomplete")
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
        self.top_confidence = np.load(
            economic._verify_record(files["top_confidence"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.confidence_percentile = np.load(
            economic._verify_record(files["causal_confidence_percentile"]),
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
        self.pack = _load_json(_resolve(self.study["sources"]["pack_manifest"]))
        self.date_values = np.asarray(self.pack["date_values"], dtype=str)
        self.symbol_values = np.asarray(self.pack["symbol_values"], dtype=str)
        self.symbol_count = int(self.pack["symbol_count"])
        self.raw_open = economic._source_array(
            self.pack, "execution_arrays", "entry_open_raw", dtype=np.float32
        )
        self.raw_close = economic._source_array(
            self.pack, "execution_arrays", "exit_close_raw", dtype=np.float32
        )
        self.raw_down_limit = economic._source_array(
            self.pack, "execution_arrays", "exit_down_limit_raw", dtype=np.float32
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
        self.is_suspended = economic._source_array(
            self.pack, "masks", "is_suspended", dtype=np.bool_
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
        if self.date_text(self.day_count - 1) != LAST_MARK_DATE:
            raise ValueError("T1 execution signal book end date changed")

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
                float(value), float(expected[name]), rel_tol=0.0, abs_tol=1.0e-12
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

    @staticmethod
    def family_index(family: str) -> int:
        try:
            return SCORE_FAMILIES.index(str(family))
        except ValueError as exc:
            raise ValueError(f"unknown T1 score family: {family}") from exc

    def symbols_for_day(self, family: str, day: int) -> np.ndarray:
        family_idx = self.family_index(family)
        count = int(self.candidate_counts[int(day)])
        return np.asarray(self.orders[family_idx, int(day), :count], dtype=np.int32)

    def rank_value(self, *, family: str, day: int, symbol_idx: int) -> float:
        return float(
            self.rank_panel[self.family_index(family), int(day), int(symbol_idx)]
        )

    def candidate_count(self, day: int) -> int:
        return int(self.candidate_counts[int(day)])

    def confidence_pass(self, *, family: str, day: int, mode: str) -> bool:
        if mode == "always":
            return True
        percentile = float(
            self.confidence_percentile[self.family_index(family), int(day)]
        )
        if mode == "causal_top_score_q50":
            return percentile >= 0.50
        if mode == "causal_top_score_q80":
            return percentile >= 0.80
        raise ValueError(f"unknown confidence mode: {mode}")

    def trailing_amount(self, *, signal_date_idx: int, symbol_idx: int) -> float:
        start = max(0, int(signal_date_idx) - 19)
        values = np.asarray(
            self.amount[start : int(signal_date_idx) + 1, int(symbol_idx)],
            dtype=np.float64,
        )
        valid = values[np.isfinite(values) & (values > 0.0)]
        return float(np.median(valid)) if valid.size else math.nan

    def open_sellable(self, *, date_idx: int, symbol_idx: int) -> bool:
        index = int(date_idx)
        symbol = int(symbol_idx)
        value = economic.derive_open_sellable(
            raw_open=np.asarray([self.raw_open[index, symbol]], dtype=np.float64),
            raw_down_limit=np.asarray(
                [self.raw_down_limit[index, symbol]], dtype=np.float64
            ),
            status_valid=np.asarray([self.status_valid[index, symbol]], dtype=bool),
            suspended=np.asarray([self.is_suspended[index, symbol]], dtype=bool),
            delisted=np.asarray([self.is_delisted[index, symbol]], dtype=bool),
        )
        return bool(value[0])

    def close_sellable(self, *, date_idx: int, symbol_idx: int) -> bool:
        index = int(date_idx)
        symbol = int(symbol_idx)
        raw_close = float(self.raw_close[index, symbol])
        return (
            bool(self.exit_sellable[index, symbol])
            and math.isfinite(raw_close)
            and raw_close > 0.0
        )

    def entry_is_buyable(
        self, *, spec: economic.TaskSpec, signal_day: int, symbol_idx: int
    ) -> bool:
        _, entry_mode, _ = _parse_rolling_family(spec.family)
        base = bool(self.next_buyable[int(signal_day), int(symbol_idx)])
        if entry_mode == "open":
            return base
        vwap = float(self.first_5m_vwap[int(signal_day), int(symbol_idx)])
        return base and math.isfinite(vwap) and vwap > 0.0

    def entry_prices(
        self,
        *,
        spec: economic.TaskSpec,
        signal_day: int,
        execution_date_idx: int,
        symbol_idx: int,
    ) -> tuple[float, float]:
        _, entry_mode, _ = _parse_rolling_family(spec.family)
        return self.execution_entry_prices(
            entry_mode=entry_mode,
            signal_day=signal_day,
            execution_date_idx=execution_date_idx,
            symbol_idx=symbol_idx,
        )

    def execution_entry_prices(
        self,
        *,
        entry_mode: str,
        signal_day: int,
        execution_date_idx: int,
        symbol_idx: int,
    ) -> tuple[float, float]:
        raw_open = float(self.raw_open[int(execution_date_idx), int(symbol_idx)])
        adjusted_open = float(
            self.adjusted_open[int(execution_date_idx), int(symbol_idx)]
        )
        if entry_mode == "open":
            return raw_open, adjusted_open
        if entry_mode != "first_5m_vwap":
            raise ValueError(f"unknown entry mode: {entry_mode}")
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
        family, _, confidence = _parse_rolling_family(spec.family)
        if self.date_text(day) > LAST_ENTRY_SIGNAL_DATE:
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
        return _plan_rolling_orders(
            book=self,
            spec=spec,
            score_family=family,
            confidence_mode=confidence,
            day=day,
            positions=positions,
        )


def _rolling_family(spec: ExecutionSpec) -> str:
    return f"t1r::{spec.score_family}::{spec.entry_mode}::{spec.confidence_mode}"


def _parse_rolling_family(value: str) -> tuple[str, str, str]:
    parts = str(value).split("::")
    if len(parts) != 4 or parts[0] != "t1r":
        raise ValueError(f"invalid rolling family: {value}")
    family, entry_mode, confidence = parts[1:]
    if family not in SCORE_FAMILIES or entry_mode not in ENTRY_MODES:
        raise ValueError(f"invalid rolling family: {value}")
    if confidence not in CONFIDENCE_MODES:
        raise ValueError(f"invalid rolling confidence mode: {confidence}")
    return family, entry_mode, confidence


def _rolling_adapter(spec: ExecutionSpec) -> economic.TaskSpec:
    if spec.exit_mode != "rolling_open" or spec.cadence != "daily":
        raise ValueError("rolling adapter requires the rolling-open daily policy")
    return economic.TaskSpec(
        family=_rolling_family(spec),
        exposure_mode="target_full",
        slot_count=int(spec.slot_count),
        buffer_multiplier=float(spec.buffer_multiplier),
        cost_scenario=str(spec.cost_scenario),
    )


def _plan_rolling_orders(
    *,
    book: SignalBook,
    spec: economic.TaskSpec,
    score_family: str,
    confidence_mode: str,
    day: int,
    positions: Mapping[int, economic.Position],
) -> economic.PendingOrders:
    if not book.confidence_pass(
        family=score_family,
        day=day,
        mode=confidence_mode,
    ):
        return economic.PendingOrders(
            signal_day=int(day),
            sells=tuple(
                economic.PendingSell(
                    symbol_idx=int(symbol_idx),
                    reason="low_confidence_cash",
                    paired_buy_symbol_idx=None,
                )
                for symbol_idx in sorted(positions)
            ),
            unpaired_buys=(),
        )
    ordered = book.symbols_for_day(score_family, day)
    if not len(ordered):
        return economic.PendingOrders(signal_day=int(day), sells=(), unpaired_buys=())
    held = {int(value) for value in positions}
    missing = [
        symbol
        for symbol in sorted(held)
        if not math.isfinite(
            book.rank_value(family=score_family, day=day, symbol_idx=symbol)
        )
    ]
    scan_count = min(len(ordered), len(held) + int(spec.slot_count))
    candidates = [
        int(symbol) for symbol in ordered[:scan_count] if int(symbol) not in held
    ]
    reserved: set[int] = set()

    def take_candidate() -> int | None:
        for candidate in candidates:
            if candidate not in reserved:
                reserved.add(candidate)
                return candidate
        return None

    unpaired: list[economic.PendingBuy] = []
    for _ in range(max(int(spec.slot_count) - len(positions), 0)):
        candidate = take_candidate()
        if candidate is None:
            break
        unpaired.append(economic.PendingBuy(candidate, "empty_slot"))
    sells: list[economic.PendingSell] = []
    for symbol in missing:
        sells.append(
            economic.PendingSell(
                symbol_idx=symbol,
                reason="missing_signal",
                paired_buy_symbol_idx=take_candidate(),
            )
        )
    buffer_value = economic.replacement_buffer(
        multiplier=float(spec.buffer_multiplier),
        slot_count=int(spec.slot_count),
        candidate_count=len(ordered),
    )
    retained = sorted(
        held.difference(missing),
        key=lambda symbol: (
            book.rank_value(family=score_family, day=day, symbol_idx=symbol),
            symbol,
        ),
    )
    for incumbent in retained:
        incumbent_score = book.rank_value(
            family=score_family, day=day, symbol_idx=incumbent
        )
        candidate: int | None = None
        for proposed in candidates:
            if proposed in reserved:
                continue
            proposed_score = book.rank_value(
                family=score_family, day=day, symbol_idx=proposed
            )
            if proposed_score - incumbent_score + 1.0e-12 < buffer_value:
                break
            candidate = proposed
            reserved.add(proposed)
            break
        if candidate is not None:
            sells.append(
                economic.PendingSell(
                    symbol_idx=incumbent,
                    reason="rank_replacement",
                    paired_buy_symbol_idx=candidate,
                )
            )
    return economic.PendingOrders(
        signal_day=int(day),
        sells=tuple(sells),
        unpaired_buys=tuple(unpaired),
    )


def _plan_fixed_buys(
    *,
    book: SignalBook,
    spec: ExecutionSpec,
    day: int,
    positions: Mapping[int, economic.Position],
    exit_due: Mapping[int, int],
) -> tuple[FixedBuy, ...]:
    if spec.exit_mode not in {
        "fixed_open_2",
        "fixed_close_2",
        "renewed_close_2",
    }:
        raise ValueError("fixed buy planner requires a fixed exit mode")
    if book.date_text(day) > LAST_ENTRY_SIGNAL_DATE or not book.confidence_pass(
        family=spec.score_family,
        day=day,
        mode=spec.confidence_mode,
    ):
        return ()
    signal_idx = int(book.signal_date_idx[int(day)])
    next_idx = signal_idx + 1
    if next_idx >= len(book.date_values):
        return ()
    held = {int(value) for value in positions}
    due_next: set[int] = set()
    if spec.exit_mode == "fixed_open_2":
        due_next = {
            int(symbol)
            for symbol, due_idx in exit_due.items()
            if int(due_idx) <= next_idx
        }
        capacity = max(int(spec.slot_count) - (len(held) - len(due_next)), 0)
        desired = capacity
    else:
        capacity = max(int(spec.slot_count) - len(held), 0)
        if spec.cadence == "full_cohort":
            desired = int(spec.slot_count) if not held else 0
        elif spec.cadence == "staggered":
            desired = min(math.ceil(int(spec.slot_count) / 2), capacity)
        else:
            raise ValueError(f"invalid fixed-close cadence: {spec.cadence}")
    desired = min(int(desired), int(capacity))
    if desired <= 0:
        return ()
    allowed_held = due_next if spec.exit_mode == "fixed_open_2" else set()
    output: list[FixedBuy] = []
    for raw_symbol in book.symbols_for_day(spec.score_family, day):
        symbol = int(raw_symbol)
        if symbol in held and symbol not in allowed_held:
            continue
        output.append(
            FixedBuy(
                signal_day=int(day),
                symbol_idx=symbol,
                reason="fixed_horizon_cohort",
            )
        )
        if len(output) >= desired:
            break
    return tuple(output)


def _renew_close_position(
    *,
    book: SignalBook,
    spec: ExecutionSpec,
    day: int,
    symbol_idx: int,
) -> bool:
    if spec.exit_mode != "renewed_close_2" or day <= 0:
        return False
    renewal_signal_day = int(day) - 1
    if book.date_text(renewal_signal_day) > LAST_ENTRY_SIGNAL_DATE:
        return False
    if not book.confidence_pass(
        family=spec.score_family,
        day=renewal_signal_day,
        mode=spec.confidence_mode,
    ):
        return False
    candidate_count = book.candidate_count(renewal_signal_day)
    retention_count = min(
        candidate_count,
        max(
            int(spec.slot_count),
            math.ceil(int(spec.slot_count) * (1.0 + float(spec.buffer_multiplier))),
        ),
    )
    retained = book.symbols_for_day(spec.score_family, renewal_signal_day)[
        :retention_count
    ]
    return bool(np.any(np.asarray(retained, dtype=np.int32) == int(symbol_idx)))


def _confidence_active_fraction(book: SignalBook, spec: ExecutionSpec) -> float:
    active: list[bool] = []
    for day in range(book.day_count):
        if book.date_text(day) > LAST_ENTRY_SIGNAL_DATE:
            break
        active.append(
            book.confidence_pass(
                family=spec.score_family,
                day=day,
                mode=spec.confidence_mode,
            )
        )
    return float(np.mean(active)) if active else math.nan


def _normalize_rolling_result(
    *,
    spec: ExecutionSpec,
    result: dict[str, Any],
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    monthly: pd.DataFrame,
    book: SignalBook,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_task_id = str(result["task_id"])
    result["schema"] = TASK_SCHEMA
    result["study_id"] = STUDY_ID
    result["task"] = asdict(spec)
    result["task_id"] = spec.task_id
    result["source_engine_task_id"] = source_task_id
    result["signal_book_sha256"] = book.manifest["files"]["rank_panel"]["sha256"]
    result["period"]["last_entry_signal_date"] = LAST_ENTRY_SIGNAL_DATE
    result["period"]["research_semantics"] = str(
        book.study["period"]["research_semantics"]
    )
    result["execution_contract"] = {
        "entry_mode": spec.entry_mode,
        "exit_mode": spec.exit_mode,
        "cadence": spec.cadence,
        "target_aligned": spec.score_family == "open_to_open",
        "confidence_is_causal": True,
    }
    equity = equity.copy()
    trades = trades.copy()
    equity["task_id"] = spec.task_id
    trades["task_id"] = spec.task_id
    result["metrics"]["average_position_count"] = float(
        equity["position_count"].astype(float).mean()
    )
    result["metrics"]["maximum_position_count"] = int(
        equity["position_count"].astype(int).max()
    )
    result["metrics"]["confidence_active_day_fraction"] = _confidence_active_fraction(
        book, spec
    )
    filled_sells = trades[
        trades["side"].astype(str).eq("sell")
        & trades["status"].astype(str).eq("filled")
    ]
    holding = pd.to_numeric(filled_sells["holding_days"], errors="coerce").dropna()
    result["metrics"]["minimum_filled_sell_holding_days"] = (
        int(holding.min()) if len(holding) else None
    )
    return result, equity, trades, monthly


def simulate_rolling_task(
    *,
    book: SignalBook,
    spec: ExecutionSpec,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result, equity, trades, monthly = economic.simulate_task(
        book=book,
        spec=_rolling_adapter(spec),
    )
    return _normalize_rolling_result(
        spec=spec,
        result=result,
        equity=equity,
        trades=trades,
        monthly=monthly,
        book=book,
    )


def simulate_fixed_task(
    *,
    book: SignalBook,
    spec: ExecutionSpec,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if spec.exit_mode not in {
        "fixed_open_2",
        "fixed_close_2",
        "renewed_close_2",
    }:
        raise ValueError("fixed simulator requires a fixed exit mode")
    multiplier = float(
        dict(book.study["execution"]["cost_scenarios"])[spec.cost_scenario]
    )
    cash = float(book.study["account"]["starting_cash_cny"])
    gross_cash = cash
    positions: dict[int, economic.Position] = {}
    exit_due: dict[int, int] = {}
    pending_buys: tuple[FixedBuy, ...] = ()
    total_cost = 0.0
    cost_parts = defaultdict(float)
    traded_notional = 0.0
    attempts: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    holding_days: list[int] = []
    sell_delay_days: list[int] = []
    blocked_days: dict[int, int] = {}
    realized_contribution = defaultdict(float)
    counters = defaultdict(int)
    maximum_conservation_error = 0.0
    benchmark_wealth = float(book.study["account"]["starting_cash_cny"])
    previous_equity = benchmark_wealth
    previous_gross_equity = previous_equity

    for day in range(book.day_count):
        economic._memory_guard()
        date_idx = int(book.signal_date_idx[day])
        trade_date = book.date_text(day)
        adjusted_open_row = np.asarray(book.adjusted_open[date_idx], dtype=np.float64)
        if spec.exit_mode == "fixed_open_2":
            due_symbols = sorted(
                symbol
                for symbol, due_idx in exit_due.items()
                if int(due_idx) <= date_idx
            )
            for symbol_idx in due_symbols:
                position = positions.get(symbol_idx)
                if position is None:
                    exit_due.pop(symbol_idx, None)
                    continue
                symbol = str(book.symbol_values[symbol_idx])
                signal_date = str(book.date_values[int(position.entry_signal_date_idx)])
                sellable = book.open_sellable(
                    date_idx=date_idx,
                    symbol_idx=symbol_idx,
                )
                t_plus_one = date_idx > int(position.entry_date_idx)
                if not sellable or not t_plus_one:
                    counters["failed_sell_count"] += 1
                    if not sellable:
                        counters["blocked_sell_count"] += 1
                    blocked_days[symbol_idx] = blocked_days.get(symbol_idx, 0) + 1
                    attempts.append(
                        economic._attempt_row(
                            task_id=spec.task_id,
                            side="sell",
                            status="failed",
                            signal_date=signal_date,
                            execution_date=trade_date,
                            symbol_idx=symbol_idx,
                            symbol=symbol,
                            reason="fixed_open_2_exit",
                        )
                    )
                    continue
                proceeds, details = economic._sell_position(
                    position=position,
                    adjusted_open=float(adjusted_open_row[symbol_idx]),
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
                delay = int(blocked_days.pop(symbol_idx, 0))
                if delay > 0:
                    sell_delay_days.append(delay)
                realized_contribution[symbol_idx] += float(
                    proceeds - position.net_cash_outflow
                )
                del positions[symbol_idx]
                exit_due.pop(symbol_idx, None)
                counters["sell_count"] += 1
                trailing = book.trailing_amount(
                    signal_date_idx=int(position.entry_signal_date_idx),
                    symbol_idx=symbol_idx,
                )
                participation = (
                    float(details["fill_notional"]) / trailing
                    if math.isfinite(trailing) and trailing > 0.0
                    else math.nan
                )
                attempts.append(
                    economic._attempt_row(
                        task_id=spec.task_id,
                        side="sell",
                        status="filled",
                        signal_date=signal_date,
                        execution_date=trade_date,
                        symbol_idx=symbol_idx,
                        symbol=symbol,
                        reason="fixed_open_2_exit",
                        details=details,
                        holding_days=duration,
                        sell_delay_days=delay,
                        participation=participation,
                    )
                )

        equity_open = economic._portfolio_value(
            cash=cash,
            positions=positions,
            adjusted_prices=adjusted_open_row,
        )
        for order in pending_buys:
            symbol_idx = int(order.symbol_idx)
            symbol = str(book.symbol_values[symbol_idx])
            signal_idx = int(book.signal_date_idx[int(order.signal_day)])
            signal_date = book.date_text(order.signal_day)
            if len(positions) >= int(spec.slot_count) or symbol_idx in positions:
                counters["skipped_buy_slot_count"] += 1
                attempts.append(
                    economic._attempt_row(
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
            buyable = bool(book.next_buyable[int(order.signal_day), symbol_idx])
            if spec.entry_mode == "first_5m_vwap":
                vwap = float(book.first_5m_vwap[int(order.signal_day), symbol_idx])
                buyable &= math.isfinite(vwap) and vwap > 0.0
            if not buyable:
                counters["failed_buy_count"] += 1
                attempts.append(
                    economic._attempt_row(
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
            raw_entry, adjusted_entry = book.execution_entry_prices(
                entry_mode=spec.entry_mode,
                signal_day=int(order.signal_day),
                execution_date_idx=date_idx,
                symbol_idx=symbol_idx,
            )
            allocation = min(
                float(cash),
                max(float(equity_open), 0.0) / int(spec.slot_count),
            )
            position, details = economic._buy_position(
                available_cash=cash,
                allocated_cash=allocation,
                symbol_idx=symbol_idx,
                signal_date_idx=signal_idx,
                execution_date_idx=date_idx,
                raw_open=raw_entry,
                adjusted_open=adjusted_entry,
                costs=book.costs,
                slippage_multiplier=multiplier,
            )
            if position is None:
                counters["failed_buy_cash_or_lot_count"] += 1
                attempts.append(
                    economic._attempt_row(
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
            exit_due[symbol_idx] = signal_idx + 2
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
                economic._attempt_row(
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
        pending_buys = ()

        adjusted_close_row = np.asarray(book.adjusted_close[date_idx], dtype=np.float64)
        if spec.exit_mode in {"fixed_close_2", "renewed_close_2"}:
            due_symbols = sorted(
                symbol
                for symbol, due_idx in exit_due.items()
                if int(due_idx) <= date_idx
            )
            for symbol_idx in due_symbols:
                position = positions.get(symbol_idx)
                if position is None:
                    exit_due.pop(symbol_idx, None)
                    continue
                if _renew_close_position(
                    book=book,
                    spec=spec,
                    day=day,
                    symbol_idx=symbol_idx,
                ):
                    exit_due[symbol_idx] = date_idx + 1
                    counters["renewal_count"] += 1
                    attempts.append(
                        economic._attempt_row(
                            task_id=spec.task_id,
                            side="renew",
                            status="retained",
                            signal_date=book.date_text(day - 1),
                            execution_date=trade_date,
                            symbol_idx=symbol_idx,
                            symbol=str(book.symbol_values[symbol_idx]),
                            reason="renewed_d2_signal",
                        )
                    )
                    continue
                symbol = str(book.symbol_values[symbol_idx])
                signal_date = str(book.date_values[int(position.entry_signal_date_idx)])
                sellable = book.close_sellable(
                    date_idx=date_idx,
                    symbol_idx=symbol_idx,
                )
                t_plus_one = date_idx > int(position.entry_date_idx)
                if not sellable or not t_plus_one:
                    counters["failed_sell_count"] += 1
                    if not sellable:
                        counters["blocked_sell_count"] += 1
                    blocked_days[symbol_idx] = blocked_days.get(symbol_idx, 0) + 1
                    attempts.append(
                        economic._attempt_row(
                            task_id=spec.task_id,
                            side="sell",
                            status="failed",
                            signal_date=signal_date,
                            execution_date=trade_date,
                            symbol_idx=symbol_idx,
                            symbol=symbol,
                            reason=f"{spec.exit_mode}_exit",
                        )
                    )
                    continue
                proceeds, details = economic._sell_position(
                    position=position,
                    adjusted_open=float(adjusted_close_row[symbol_idx]),
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
                delay = int(blocked_days.pop(symbol_idx, 0))
                if delay > 0:
                    sell_delay_days.append(delay)
                realized_contribution[symbol_idx] += float(
                    proceeds - position.net_cash_outflow
                )
                del positions[symbol_idx]
                exit_due.pop(symbol_idx, None)
                counters["sell_count"] += 1
                trailing = book.trailing_amount(
                    signal_date_idx=int(position.entry_signal_date_idx),
                    symbol_idx=symbol_idx,
                )
                participation = (
                    float(details["fill_notional"]) / trailing
                    if math.isfinite(trailing) and trailing > 0.0
                    else math.nan
                )
                attempts.append(
                    economic._attempt_row(
                        task_id=spec.task_id,
                        side="sell",
                        status="filled",
                        signal_date=signal_date,
                        execution_date=trade_date,
                        symbol_idx=symbol_idx,
                        symbol=symbol,
                        reason=f"{spec.exit_mode}_exit",
                        details=details,
                        holding_days=duration,
                        sell_delay_days=delay,
                        participation=participation,
                    )
                )

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
                exit_due.pop(symbol_idx, None)
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
                blocked_days.pop(symbol_idx, None)
                counters["terminal_recovery_count"] += 1
                counters["terminal_recovery_notional"] += recovery
                attempts.append(
                    economic._attempt_row(
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

        equity = economic._portfolio_value(
            cash=cash,
            positions=positions,
            adjusted_prices=adjusted_close_row,
        )
        gross_equity = economic._portfolio_value(
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
        if day < book.day_count - 1:
            pending_buys = _plan_fixed_buys(
                book=book,
                spec=spec,
                day=day,
                positions=positions,
                exit_due=exit_due,
            )

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
                economic._attempt_row(
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
    monthly = economic._monthly_metrics(
        dates=equity_frame["trade_date"].to_numpy(dtype=str),
        equity=equity_frame["net_equity"].to_numpy(dtype=np.float64),
    )
    net_metrics = economic._annualized_metrics(
        equity=np.r_[STARTING_CASH_CNY, equity_frame["net_equity"].to_numpy()],
        daily_returns=np.r_[0.0, equity_frame["daily_net_return"].to_numpy()],
    )
    gross_metrics = economic._annualized_metrics(
        equity=np.r_[STARTING_CASH_CNY, equity_frame["gross_equity"].to_numpy()],
        daily_returns=np.r_[0.0, equity_frame["daily_gross_return"].to_numpy()],
    )
    annual = economic._year_metrics(
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
        proceeds, details = economic._sell_position(
            position=position,
            adjusted_open=float(book.adjusted_close[final_date_idx, symbol_idx]),
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
        filled["participation_of_trailing20_median_amount"], errors="coerce"
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
    holding_array = np.asarray(holding_days, dtype=np.int64)
    result = {
        "schema": TASK_SCHEMA,
        "status": "completed",
        "completed_at": d1._now(),
        "study_id": STUDY_ID,
        "task": asdict(spec),
        "task_id": spec.task_id,
        "signal_book_sha256": book.manifest["files"]["rank_panel"]["sha256"],
        "period": {
            "first_signal_date": str(equity_frame["trade_date"].iloc[0]),
            "last_entry_signal_date": LAST_ENTRY_SIGNAL_DATE,
            "last_mark_date": final_date,
            "continuous_years": list(YEARS),
            "annual_reset": False,
            "forbidden_2026_row_count": 0,
            "research_semantics": str(book.study["period"]["research_semantics"]),
        },
        "execution_contract": {
            "entry_mode": spec.entry_mode,
            "exit_mode": spec.exit_mode,
            "cadence": spec.cadence,
            "target_aligned": (
                (
                    spec.score_family == "open_to_open"
                    and spec.exit_mode == "fixed_open_2"
                )
                or (
                    spec.score_family == "open_to_close_2"
                    and spec.exit_mode == "fixed_close_2"
                )
            ),
            "confidence_is_causal": True,
            "fixed_exit_is_forced": spec.exit_mode != "renewed_close_2",
            "renewal_uses_previous_signal_day": spec.exit_mode == "renewed_close_2",
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
            "average_position_count": float(
                equity_frame["position_count"].astype(float).mean()
            ),
            "maximum_position_count": int(
                equity_frame["position_count"].astype(int).max()
            ),
            "confidence_active_day_fraction": _confidence_active_fraction(book, spec),
            "trade_attempt_count": len(attempts_frame),
            "filled_buy_count": int(counters["buy_count"]),
            "filled_sell_count": int(counters["sell_count"]),
            "failed_buy_count": int(
                counters["failed_buy_count"] + counters["failed_buy_cash_or_lot_count"]
            ),
            "failed_sell_count": int(counters["failed_sell_count"]),
            "blocked_sell_count": int(counters["blocked_sell_count"]),
            "renewal_count": int(counters["renewal_count"]),
            "delayed_sell_count": len(sell_delay_days),
            "sell_delay_days_mean": (
                float(np.mean(sell_delay_days)) if sell_delay_days else 0.0
            ),
            "sell_delay_days_p90": (
                float(np.quantile(sell_delay_days, 0.9)) if sell_delay_days else 0.0
            ),
            "unresolved_blocked_sell_count": len(blocked_days),
            "terminal_recovery_count": int(counters["terminal_recovery_count"]),
            "terminal_recovery_notional": float(counters["terminal_recovery_notional"]),
            "holding_days_mean": (
                float(np.mean(holding_array)) if holding_array.size else math.nan
            ),
            "holding_days_median": (
                float(np.median(holding_array)) if holding_array.size else math.nan
            ),
            "holding_days_p90": (
                float(np.quantile(holding_array, 0.9))
                if holding_array.size
                else math.nan
            ),
            "minimum_filled_sell_holding_days": (
                int(np.min(holding_array)) if holding_array.size else None
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


def _task_dir(output_root: Path, spec: ExecutionSpec) -> Path:
    return output_root / "tasks" / spec.task_id


def _task_complete(
    *,
    output_root: Path,
    spec: ExecutionSpec,
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
            or result.get("execution_engine_source_sha256") != _engine_sha256()
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
    spec = ExecutionSpec(**dict(result["task"]))
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
        "execution_engine_source_sha256": _engine_sha256(),
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
    _emit("t1_execution_tasks_starting", completed=completed, total=len(specs))
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
        if spec.exit_mode == "rolling_open":
            result, equity, trades, monthly = simulate_rolling_task(
                book=book,
                spec=spec,
            )
        else:
            result, equity, trades, monthly = simulate_fixed_task(
                book=book,
                spec=spec,
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
        if completed % 25 == 0 or completed == len(specs):
            _emit(
                "t1_execution_task_progress",
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


def _task_result(output_root: Path, spec: ExecutionSpec) -> dict[str, Any]:
    return _load_json(_task_dir(output_root, spec) / "task_result.json")


def _configuration_key(spec: ExecutionSpec) -> tuple[Any, ...]:
    return (
        spec.score_family,
        spec.entry_mode,
        spec.exit_mode,
        spec.cadence,
        spec.confidence_mode,
        int(spec.slot_count),
        float(spec.buffer_multiplier),
    )


def _task_metric_row(
    *,
    spec: ExecutionSpec,
    result: Mapping[str, Any],
    monthly: pd.DataFrame,
) -> dict[str, Any]:
    metrics = dict(result["metrics"])
    net = dict(metrics["net"])
    participation = dict(metrics.get("participation", {}) or {})
    annual = list(result["annual"])
    annual_returns = [float(item["net_return"]) for item in annual]
    return {
        **asdict(spec),
        "task_id": spec.task_id,
        "terminal_return": float(metrics["terminal_cost_accrued_return"]),
        "terminal_relative_excess_return": float(
            metrics["terminal_cost_accrued_relative_excess_return"]
        ),
        "cagr": float(net["cagr"]),
        "sharpe": float(net["sharpe"]),
        "sortino": float(net["sortino"]),
        "maximum_drawdown": float(net["maximum_drawdown"]),
        "positive_year_count": int(np.sum(np.asarray(annual_returns) > 0.0)),
        "minimum_annual_return": float(np.min(annual_returns)),
        "annual_return_std": float(np.std(annual_returns, ddof=0)),
        "positive_month_fraction": float(
            np.mean(monthly["net_return"].astype(float).to_numpy() > 0.0)
        ),
        "turnover_to_starting_cash": float(metrics["turnover_to_starting_cash"]),
        "average_cash_fraction": float(metrics["average_cash_fraction"]),
        "average_position_count": float(metrics["average_position_count"]),
        "maximum_position_count": int(metrics["maximum_position_count"]),
        "confidence_active_day_fraction": float(
            metrics["confidence_active_day_fraction"]
        ),
        "filled_buy_count": int(metrics["filled_buy_count"]),
        "filled_sell_count": int(metrics["filled_sell_count"]),
        "failed_buy_count": int(metrics["failed_buy_count"]),
        "failed_sell_count": int(metrics["failed_sell_count"]),
        "blocked_sell_count": int(metrics["blocked_sell_count"]),
        "renewal_count": int(metrics.get("renewal_count", 0)),
        "holding_days_median": float(metrics["holding_days_median"]),
        "minimum_filled_sell_holding_days": metrics.get(
            "minimum_filled_sell_holding_days"
        ),
        "maximum_conservation_error": float(metrics["maximum_conservation_error"]),
        "participation_above_0p1pct": float(
            participation.get("fraction_above_0.001", math.nan)
        ),
        "participation_above_0p5pct": float(
            participation.get("fraction_above_0.005", math.nan)
        ),
        "participation_above_1pct": float(
            participation.get("fraction_above_0.01", math.nan)
        ),
    }


def _strict_pass(row: Mapping[str, Any], selection: Mapping[str, Any]) -> bool:
    return bool(
        float(row["base_terminal_return"]) > 0.0
        and float(row["stress_terminal_return"]) > 0.0
        and float(row["stress_terminal_relative_excess_return"]) > 0.0
        and int(row["stress_positive_year_count"])
        >= int(selection["minimum_positive_years"])
        and float(row["stress_sharpe"]) >= float(selection["minimum_stress_sharpe"])
        and float(row["stress_positive_month_fraction"])
        >= float(selection["minimum_stress_positive_month_fraction"])
        and float(row["stress_maximum_drawdown"])
        >= -float(selection["maximum_stress_drawdown"])
    )


def _economic_pass(row: Mapping[str, Any]) -> bool:
    return bool(
        float(row["base_terminal_return"]) > 0.0
        and float(row["stress_terminal_return"]) > 0.0
        and int(row["stress_positive_year_count"]) >= 2
        and float(row["stress_sharpe"]) > 0.0
        and float(row["stress_positive_month_fraction"]) >= 0.50
        and float(row["stress_maximum_drawdown"]) >= -0.50
    )


def _neighbor_count(frame: pd.DataFrame, *, passing_column: str) -> np.ndarray:
    score_order = (
        "open_to_close_2",
        "blend_open_weight_25",
        "blend_open_weight_50",
        "blend_open_weight_75",
        "open_to_open",
    )
    score_index = {value: index for index, value in enumerate(score_order)}
    confidence_index = {value: index for index, value in enumerate(CONFIDENCE_MODES)}
    slot_index = {value: index for index, value in enumerate(SLOT_COUNTS)}
    buffer_index = {value: index for index, value in enumerate(ROLLING_BUFFERS)}
    records = frame.to_dict("records")
    output = np.zeros(len(records), dtype=np.int32)
    for left_index, left in enumerate(records):
        if not bool(left[passing_column]):
            continue
        for right_index, right in enumerate(records):
            if left_index == right_index or not bool(right[passing_column]):
                continue
            if any(
                left[name] != right[name]
                for name in ("entry_mode", "exit_mode", "cadence")
            ):
                continue
            distances: list[int] = []
            left_family = str(left["score_family"])
            right_family = str(right["score_family"])
            if left_family == right_family:
                distances.append(0)
            elif left_family in score_index and right_family in score_index:
                distances.append(
                    abs(score_index[left_family] - score_index[right_family])
                )
            else:
                distances.append(2)
            distances.extend(
                [
                    abs(
                        confidence_index[str(left["confidence_mode"])]
                        - confidence_index[str(right["confidence_mode"])]
                    ),
                    abs(
                        slot_index[int(left["slot_count"])]
                        - slot_index[int(right["slot_count"])]
                    ),
                    (
                        abs(
                            buffer_index[float(left["buffer_multiplier"])]
                            - buffer_index[float(right["buffer_multiplier"])]
                        )
                        if str(left["exit_mode"]) in {"rolling_open", "renewed_close_2"}
                        else 0
                    ),
                ]
            )
            if sum(value > 0 for value in distances) == 1 and max(distances) == 1:
                output[left_index] += 1
    return output


def _pareto_mask(frame: pd.DataFrame) -> np.ndarray:
    values = frame[
        [
            "slot_count",
            "stress_terminal_return",
            "stress_sharpe",
            "stress_maximum_drawdown",
            "stress_turnover_to_starting_cash",
        ]
    ].to_numpy(dtype=np.float64)
    output = np.ones(len(values), dtype=bool)
    for index, current in enumerate(values):
        if not output[index]:
            continue
        dominated = (
            (values[:, 0] <= current[0])
            & (values[:, 1] >= current[1])
            & (values[:, 2] >= current[2])
            & (values[:, 3] >= current[3])
            & (values[:, 4] <= current[4])
            & np.any(values != current, axis=1)
        )
        dominated[index] = False
        if bool(np.any(dominated)):
            output[index] = False
    return output


def evaluate(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    current_status = status(study_path=study_path, output_root=output_root)
    if current_status["status"] != "completed":
        raise ValueError("T1 execution tasks are incomplete")
    task_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    monthly_rows: list[pd.DataFrame] = []
    for spec in task_specs(study_path):
        result = _task_result(output_root, spec)
        monthly = pd.read_parquet(_resolve(result["files"]["monthly"]["path"]))
        task_rows.append(_task_metric_row(spec=spec, result=result, monthly=monthly))
        for item in result["annual"]:
            annual_rows.append({**asdict(spec), **dict(item)})
        current_monthly = monthly.copy()
        for name, value in asdict(spec).items():
            current_monthly[name] = value
        monthly_rows.append(current_monthly)
    task_metrics = pd.DataFrame(task_rows).sort_values(
        [
            "score_family",
            "entry_mode",
            "exit_mode",
            "cadence",
            "confidence_mode",
            "slot_count",
            "buffer_multiplier",
            "cost_scenario",
        ],
        kind="stable",
    )
    annual_metrics = pd.DataFrame(annual_rows).sort_values(
        [
            "score_family",
            "entry_mode",
            "exit_mode",
            "cadence",
            "confidence_mode",
            "slot_count",
            "buffer_multiplier",
            "cost_scenario",
            "year",
        ],
        kind="stable",
    )
    monthly_metrics = pd.concat(monthly_rows, ignore_index=True).sort_values(
        [
            "score_family",
            "entry_mode",
            "exit_mode",
            "cadence",
            "confidence_mode",
            "slot_count",
            "buffer_multiplier",
            "cost_scenario",
            "month",
        ],
        kind="stable",
    )
    spec_fields = tuple(ExecutionSpec.__dataclass_fields__)
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in task_metrics.to_dict("records"):
        spec = ExecutionSpec(**{name: row[name] for name in spec_fields})
        indexed[(_configuration_key(spec), str(row["cost_scenario"]))] = row
    configuration_rows: list[dict[str, Any]] = []
    config_fields = (
        "score_family",
        "entry_mode",
        "exit_mode",
        "cadence",
        "confidence_mode",
        "slot_count",
        "buffer_multiplier",
    )
    metric_fields = (
        "terminal_return",
        "terminal_relative_excess_return",
        "cagr",
        "sharpe",
        "sortino",
        "maximum_drawdown",
        "positive_year_count",
        "minimum_annual_return",
        "annual_return_std",
        "positive_month_fraction",
        "turnover_to_starting_cash",
        "average_cash_fraction",
        "average_position_count",
        "maximum_position_count",
        "confidence_active_day_fraction",
        "filled_buy_count",
        "filled_sell_count",
        "failed_buy_count",
        "failed_sell_count",
        "blocked_sell_count",
        "renewal_count",
        "holding_days_median",
        "maximum_conservation_error",
        "participation_above_0p1pct",
        "participation_above_0p5pct",
        "participation_above_1pct",
    )
    for key in sorted({value[0] for value in indexed}, key=str):
        base = indexed[(key, "base")]
        stress = indexed[(key, "stress")]
        row: dict[str, Any] = {name: base[name] for name in config_fields}
        row["configuration_id"] = "__".join(
            str(row[name]).replace(".", "p") for name in config_fields
        )
        for prefix, source in (("base", base), ("stress", stress)):
            for name in metric_fields:
                row[f"{prefix}_{name}"] = source[name]
        row["stress_cost_return_drag"] = float(base["terminal_return"]) - float(
            stress["terminal_return"]
        )
        row["strict_pass"] = _strict_pass(row, study["selection"])
        row["economic_pass"] = _economic_pass(row)
        configuration_rows.append(row)
    configurations = pd.DataFrame(configuration_rows)
    configurations["strict_neighbor_count"] = _neighbor_count(
        configurations,
        passing_column="strict_pass",
    )
    configurations["economic_neighbor_count"] = _neighbor_count(
        configurations,
        passing_column="economic_pass",
    )
    minimum_neighbors = int(study["selection"]["minimum_passing_neighbor_count"])
    configurations["stable_strict_pass"] = configurations["strict_pass"] & (
        configurations["strict_neighbor_count"] >= minimum_neighbors
    )
    configurations["stable_economic_pass"] = configurations["economic_pass"] & (
        configurations["economic_neighbor_count"] >= minimum_neighbors
    )
    finite_for_pareto = configurations[
        np.isfinite(configurations["stress_terminal_return"].astype(float))
        & np.isfinite(configurations["stress_sharpe"].astype(float))
        & np.isfinite(configurations["stress_maximum_drawdown"].astype(float))
    ].copy()
    finite_for_pareto["pareto_frontier"] = _pareto_mask(finite_for_pareto)
    pareto = finite_for_pareto[finite_for_pareto["pareto_frontier"]].copy()
    configurations = configurations.sort_values(
        [
            "stable_strict_pass",
            "strict_pass",
            "stable_economic_pass",
            "economic_pass",
            "slot_count",
            "stress_sharpe",
            "stress_terminal_return",
        ],
        ascending=[False, False, False, False, True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    stable_strict = configurations[configurations["stable_strict_pass"]]
    strict = configurations[configurations["strict_pass"]]
    stable_economic = configurations[configurations["stable_economic_pass"]]
    recommended_configuration: dict[str, Any] | None = None
    if len(stable_strict):
        smallest_k = int(stable_strict["slot_count"].min())
        candidates = stable_strict[stable_strict["slot_count"].eq(smallest_k)]
        conclusion = "stable_low_position_execution_region_found"
    elif len(strict):
        smallest_k = None
        candidates = strict
        conclusion = "only_isolated_strict_execution_cells_found"
    elif len(stable_economic):
        smallest_k = None
        candidates = stable_economic
        conclusion = "relaxed_economic_region_found_but_strict_gate_failed"
    else:
        smallest_k = None
        candidates = configurations
        conclusion = "no_stable_profitable_execution_region_found"
    candidates = candidates.sort_values(
        ["stress_sharpe", "stress_terminal_return", "stress_maximum_drawdown"],
        ascending=[False, False, False],
        kind="stable",
    )
    best_diagnostic = candidates.iloc[0].to_dict()
    if len(stable_strict):
        recommended_configuration = best_diagnostic
    decision = {
        "schema": "seq100_quality_liquidity_t1_execution_decision/1",
        "status": "completed",
        "completed_at": d1._now(),
        "study_id": STUDY_ID,
        "conclusion": conclusion,
        "configuration_count": len(configurations),
        "strict_pass_count": int(configurations["strict_pass"].sum()),
        "stable_strict_pass_count": int(configurations["stable_strict_pass"].sum()),
        "economic_pass_count": int(configurations["economic_pass"].sum()),
        "stable_economic_pass_count": int(configurations["stable_economic_pass"].sum()),
        "smallest_recommended_slot_count": smallest_k,
        "recommended_configuration": recommended_configuration,
        "best_available_diagnostic_configuration": best_diagnostic,
        "selection_order": [
            "strict base/stress and cross-year gate",
            "passing adjacent parameter region",
            "smallest slot count",
            "stress Sharpe",
            "stress terminal return",
            "stress drawdown",
        ],
        "retrospective_policy_selection": True,
        "production_policy_selected": False,
    }
    outputs = {
        "task_metrics": output_root / "task_metrics.parquet",
        "configuration_metrics": output_root / "configuration_metrics.parquet",
        "annual_metrics": output_root / "annual_metrics.parquet",
        "monthly_metrics": output_root / "monthly_metrics.parquet",
        "pareto_frontier": output_root / "pareto_frontier.parquet",
        "decision": output_root / "decision.json",
    }
    _write_parquet(outputs["task_metrics"], task_metrics)
    _write_parquet(outputs["configuration_metrics"], configurations)
    _write_parquet(outputs["annual_metrics"], annual_metrics)
    _write_parquet(outputs["monthly_metrics"], monthly_metrics)
    _write_parquet(outputs["pareto_frontier"], pareto)
    _write_json(outputs["decision"], decision)
    signal_manifest = _load_json(output_root / "signal_book/manifest.json")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "evaluated",
        "updated_at": d1._now(),
        "study_id": STUDY_ID,
        "study_config": _file_record(study_path),
        "signal_book": _file_record(output_root / "signal_book/manifest.json"),
        "source_model_manifest": signal_manifest["model_manifest"],
        "source_model_audit": signal_manifest["model_audit"],
        "task_count": len(task_metrics),
        "configuration_count": len(configurations),
        "expected_task_count": int(study["outputs"]["expected_task_count"]),
        "continuous_account": True,
        "last_entry_signal_date": LAST_ENTRY_SIGNAL_DATE,
        "last_mark_date": LAST_MARK_DATE,
        "forbidden_2026_row_count": 0,
        "decision": decision,
        "execution_replay_performed": True,
        "production_policy_selected": False,
        "outputs": {name: _file_record(path) for name, path in outputs.items()},
    }
    _write_json(output_root / "manifest.json", manifest)
    _emit(
        "t1_execution_evaluated",
        conclusion=conclusion,
        strict_pass_count=decision["strict_pass_count"],
        stable_strict_pass_count=decision["stable_strict_pass_count"],
    )
    return manifest


def audit(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("T1 execution manifest is missing")
    manifest = _load_json(manifest_path)
    signal_manifest = _load_json(output_root / "signal_book/manifest.json")
    study_sha256 = _study_hash(study_path)
    signal_sha256 = str(signal_manifest["files"]["rank_panel"]["sha256"])
    specs = task_specs(study_path)
    results = [_task_result(output_root, spec) for spec in specs]
    task_files_valid = all(
        _task_complete(
            output_root=output_root,
            spec=spec,
            study_sha256=study_sha256,
            signal_sha256=signal_sha256,
        )
        for spec in specs
    )
    configuration_path = economic._verify_record(
        manifest["outputs"]["configuration_metrics"]
    )
    task_metrics_path = economic._verify_record(manifest["outputs"]["task_metrics"])
    configurations = pd.read_parquet(configuration_path)
    task_metrics = pd.read_parquet(task_metrics_path)
    pair_counts = task_metrics.groupby(
        [
            "score_family",
            "entry_mode",
            "exit_mode",
            "cadence",
            "confidence_mode",
            "slot_count",
            "buffer_multiplier",
        ],
        sort=False,
    )["cost_scenario"].nunique()
    minimum_holding = [
        dict(result["metrics"]).get("minimum_filled_sell_holding_days")
        for result in results
    ]
    minimum_holding = [int(value) for value in minimum_holding if value is not None]
    maximum_positions = [
        int(dict(result["metrics"])["maximum_position_count"]) for result in results
    ]
    configured_slots = [int(result["task"]["slot_count"]) for result in results]
    maximum_conservation = max(
        float(dict(result["metrics"])["maximum_conservation_error"])
        for result in results
    )
    fixed_holding_ok = True
    for result in results:
        task = dict(result["task"])
        holding = dict(result["metrics"]).get("minimum_filled_sell_holding_days")
        if task["exit_mode"] in {
            "fixed_open_2",
            "fixed_close_2",
            "renewed_close_2",
        } and (holding is None or int(holding) < 1):
            fixed_holding_ok = False
            break
    checks = {
        "manifest_schema": manifest.get("schema") == MANIFEST_SCHEMA,
        "manifest_status": manifest.get("status") in {"evaluated", "audited"},
        "signal_schema": signal_manifest.get("schema") == SIGNAL_SCHEMA,
        "signal_status": signal_manifest.get("status") == "completed",
        "source_model_audited": _load_json(
            _resolve(signal_manifest["model_audit"]["path"])
        ).get("status")
        == "ok",
        "task_count_exact": len(results)
        == int(study["outputs"]["expected_task_count"]),
        "task_files_valid": bool(task_files_valid),
        "configuration_count_exact": len(configurations) * 2 == len(results),
        "base_stress_pairs_complete": bool((pair_counts == 2).all()),
        "maximum_positions_respected": all(
            observed <= allowed
            for observed, allowed in zip(
                maximum_positions, configured_slots, strict=True
            )
        ),
        "t_plus_one_respected": bool(min(minimum_holding, default=1) >= 1),
        "fixed_contract_holding_respected": bool(fixed_holding_ok),
        "cash_position_conservation": maximum_conservation <= 1.0e-4,
        "continuous_account": all(
            not bool(result["period"].get("annual_reset", True)) for result in results
        ),
        "last_entry_signal_date_exact": all(
            str(result["period"]["last_entry_signal_date"]) == LAST_ENTRY_SIGNAL_DATE
            for result in results
        ),
        "last_mark_date_exact": all(
            str(result["period"]["last_mark_date"]) == LAST_MARK_DATE
            for result in results
        ),
        "forbidden_2026_zero": (
            int(signal_manifest.get("forbidden_2026_row_count", -1)) == 0
            and all(
                int(result["period"].get("forbidden_2026_row_count", -1)) == 0
                for result in results
            )
        ),
        "candidate_selection_does_not_use_future_fill": not bool(
            signal_manifest.get("candidate_selection_uses_future_fill", True)
        ),
        "confidence_is_causal": bool(signal_manifest.get("confidence_is_causal", False))
        and all(
            bool(result["execution_contract"].get("confidence_is_causal", False))
            for result in results
        ),
        "close_sellability_source_present": "exit_sellable"
        in dict(_load_json(_resolve(study["sources"]["pack_manifest"]))["masks"]),
        "outputs_hashed": all(
            economic._verify_record(record).is_file()
            for record in dict(manifest["outputs"]).values()
        ),
        "production_policy_not_selected": not bool(
            manifest.get("production_policy_selected", True)
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not bool(passed))
    audit_payload = {
        "schema": AUDIT_SCHEMA,
        "status": "ok" if not failed else "failed",
        "completed_at": d1._now(),
        "study_id": STUDY_ID,
        "checks": checks,
        "failed_checks": failed,
        "task_count": len(results),
        "configuration_count": len(configurations),
        "maximum_conservation_error": float(maximum_conservation),
        "minimum_filled_sell_holding_days": min(minimum_holding, default=None),
        "maximum_observed_position_count": max(maximum_positions, default=0),
        "forbidden_2026_row_count": 0,
    }
    audit_path = output_root / "audit.json"
    _write_json(audit_path, audit_payload)
    if failed:
        raise ValueError(f"T1 execution audit failed: {failed}")
    manifest["status"] = "audited"
    manifest["updated_at"] = d1._now()
    manifest["audit"] = _file_record(audit_path)
    _write_json(manifest_path, manifest)
    _emit(
        "t1_execution_audited",
        task_count=len(results),
        configuration_count=len(configurations),
    )
    return audit_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "run", "evaluate", "audit", "all", "status"),
    )
    parser.add_argument("--study-path", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-tasks", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    study_path = args.study_path.resolve()
    output_root = args.output_root.resolve()
    if args.command == "prepare":
        payload = prepare_signal_book(
            study_path=study_path,
            output_root=output_root,
        )
    elif args.command == "run":
        payload = run(
            study_path=study_path,
            output_root=output_root,
            max_tasks=args.max_tasks,
        )
    elif args.command == "evaluate":
        payload = evaluate(study_path=study_path, output_root=output_root)
    elif args.command == "audit":
        payload = audit(study_path=study_path, output_root=output_root)
    elif args.command == "status":
        payload = status(study_path=study_path, output_root=output_root)
    else:
        prepare_signal_book(study_path=study_path, output_root=output_root)
        run_status = run(
            study_path=study_path,
            output_root=output_root,
            max_tasks=args.max_tasks,
        )
        if run_status["status"] != "completed":
            payload = run_status
        else:
            evaluate(study_path=study_path, output_root=output_root)
            payload = audit(study_path=study_path, output_root=output_root)
    print(json.dumps(economic._json_safe(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
