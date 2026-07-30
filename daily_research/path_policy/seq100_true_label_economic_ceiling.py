from __future__ import annotations

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

from daily_research.path_policy import (
    seq100_v4_economic_realizability as economic,
)
from daily_research.path_policy.seq100_candidate_execution import (
    parse_execution_costs,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_true_label_economic_ceiling_v1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_true_label_economic_ceiling_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies"
    / "seq100_true_label_economic_ceiling_v1"
)
DEFAULT_RECORD_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/research_records/seq100"
    / "seq100_true_label_economic_ceiling_v1"
)
BOOK_SCHEMA = "seq100_true_label_oracle_book/v1"
TASK_SCHEMA = "seq100_true_label_economic_task/v1"
SUMMARY_SCHEMA = "seq100_true_label_economic_summary/v1"
YEARS = economic.YEARS
FAMILIES = economic.FORMAL_FAMILIES
EXPOSURE_MODES = economic.EXPOSURE_MODES
SLOT_COUNTS = economic.SLOT_COUNTS
BUFFER_MULTIPLIERS = economic.BUFFER_MULTIPLIERS
COST_SCENARIOS = economic.COST_SCENARIOS
MODES = (
    "true_label_daily_rerank",
    "true_label_peak_close_hold",
    "true_label_post_peak_next_open_hold",
)
RANK_COLUMNS = economic.RANK_COLUMNS
ORDER_COLUMNS = ("mfe10", "mfe20", "dual")
STARTING_CASH_CNY = economic.STARTING_CASH_CNY


def _resolve(value: str | Path) -> Path:
    return economic._resolve(value)


def _load_json(path: Path) -> dict[str, Any]:
    return economic._load_json(path)


def _write_json(path: Path, payload: Any) -> None:
    economic._write_json(path, payload)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    economic._write_parquet(path, frame)


def _file_record(path: Path, **extra: Any) -> dict[str, Any]:
    return economic._file_record(path, **extra)


def _verify_record(record: Mapping[str, Any]) -> Path:
    return economic._verify_record(record)


def _emit(event: str, **payload: Any) -> None:
    economic._emit(event, **payload)


def _memory_guard() -> None:
    economic._memory_guard()


def _study_hash(path: Path) -> str:
    return economic._study_hash(path)


def _save_npy(path: Path, values: np.ndarray) -> None:
    economic._save_npy(path, values)


def _open_array(
    record: Mapping[str, Any],
    *,
    dtype: str | np.dtype,
) -> np.memmap:
    return economic._open_array(record, dtype=dtype)


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("study_id") != STUDY_ID:
        raise ValueError(f"study_id must be {STUDY_ID}")
    period = dict(payload["period"])
    if tuple(int(value) for value in period["signal_years"]) != YEARS:
        raise ValueError("signal years changed")
    if period["maximum_consumed_outcome_date"] != "2025-12-31":
        raise ValueError("outcome cutoff changed")
    if int(period["forbidden_year"]) != 2026:
        raise ValueError("2026 must remain forbidden")
    oracle = dict(payload["oracle"])
    if tuple(oracle["modes"]) != MODES:
        raise ValueError("oracle modes changed")
    if tuple(oracle["families"]) != FAMILIES:
        raise ValueError("oracle families changed")
    if tuple(oracle["exposure_modes"]) != EXPOSURE_MODES:
        raise ValueError("exposure modes changed")
    if tuple(int(value) for value in oracle["slot_counts"]) != SLOT_COUNTS:
        raise ValueError("slot counts changed")
    if (
        tuple(float(value) for value in oracle["daily_rerank_buffer_multipliers"])
        != BUFFER_MULTIPLIERS
    ):
        raise ValueError("buffer multipliers changed")
    if tuple(oracle["cost_scenarios"]) != COST_SCENARIOS:
        raise ValueError("cost scenarios changed")
    if int(oracle["expected_task_count"]) != 864:
        raise ValueError("expected task count changed")
    if int(oracle["daily_rerank_expected_tasks"]) != 576:
        raise ValueError("daily rerank task count changed")
    if int(oracle["scheduled_hold_expected_tasks"]) != 288:
        raise ValueError("scheduled task count changed")
    return payload


@dataclass(frozen=True)
class OracleTaskSpec:
    mode: str
    family: str
    exposure_mode: str
    slot_count: int
    buffer_multiplier: float | None
    cost_scenario: str

    @property
    def task_id(self) -> str:
        if self.buffer_multiplier is None:
            buffer_text = "scheduled"
        else:
            buffer_text = "b" + str(float(self.buffer_multiplier)).replace(".", "p")
        return (
            f"{self.mode}__{self.family}__{self.exposure_mode}"
            f"__k{int(self.slot_count):02d}__{buffer_text}__{self.cost_scenario}"
        )


def task_specs() -> list[OracleTaskSpec]:
    output: list[OracleTaskSpec] = []
    for mode in MODES:
        buffers: tuple[float | None, ...] = (
            tuple(BUFFER_MULTIPLIERS) if mode == "true_label_daily_rerank" else (None,)
        )
        output.extend(
            OracleTaskSpec(
                mode=mode,
                family=family,
                exposure_mode=exposure,
                slot_count=slots,
                buffer_multiplier=buffer,
                cost_scenario=cost,
            )
            for family in FAMILIES
            for exposure in EXPOSURE_MODES
            for slots in SLOT_COUNTS
            for buffer in buffers
            for cost in COST_SCENARIOS
        )
    if len(output) != 864 or len({item.task_id for item in output}) != len(output):
        raise AssertionError("oracle task inventory changed")
    return output


class FuturePathReader:
    def __init__(self, pack: Mapping[str, Any]) -> None:
        path = dict(pack["label_arrays"]["future_ohlcva_path"])
        self.shards = tuple(dict(item) for item in path["shards"])
        self._shard_by_date: dict[int, int] = {}
        for index, record in enumerate(self.shards):
            for date_idx in range(
                int(record["date_start_idx"]),
                int(record["date_end_idx"]) + 1,
            ):
                self._shard_by_date[date_idx] = index
        self._cached_index: int | None = None
        self._cached: np.memmap | None = None

    def future_close_and_d1_open(
        self,
        *,
        date_idx: int,
        symbols: np.ndarray,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        index = self._shard_by_date.get(int(date_idx))
        if index is None:
            raise ValueError(f"no future path shard covers date {date_idx}")
        record = self.shards[index]
        if self._cached_index != index:
            self._cached = np.memmap(
                _resolve(record["path"]),
                dtype=np.float32,
                mode="r",
                shape=tuple(int(value) for value in record["shape"]),
            )
            self._cached_index = index
        if self._cached is None:
            raise AssertionError("future path cache is empty")
        local = int(date_idx) - int(record["date_start_idx"])
        symbol_idx = np.asarray(symbols, dtype=np.int64)
        close = np.asarray(
            self._cached[local, symbol_idx, : int(horizon), 3],
            dtype=np.float64,
        )
        d1_open = np.asarray(
            self._cached[local, symbol_idx, 0, 0],
            dtype=np.float64,
        )
        return close, d1_open


def derive_peak_close_dates(
    *,
    date_idx: int,
    symbols: np.ndarray,
    horizon: int,
    label_mfe: np.ndarray,
    future_close: np.ndarray,
    d1_open: np.ndarray,
    exit_sellable: np.ndarray,
    cutoff_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    symbol_idx = np.asarray(symbols, dtype=np.int64)
    labels = np.asarray(label_mfe, dtype=np.float64)
    close = np.asarray(future_close, dtype=np.float64)
    anchor = np.asarray(d1_open, dtype=np.float64)
    if close.shape != (len(symbol_idx), int(horizon)):
        raise ValueError("future close shape is invalid")
    if int(date_idx) + int(horizon) > int(cutoff_idx):
        raise ValueError("peak derivation would cross the outcome cutoff")
    denominator = 1.0 + anchor
    relative = (
        np.divide(
            1.0 + close,
            denominator[:, None],
            out=np.full_like(close, np.nan),
            where=np.isfinite(denominator[:, None]) & (denominator[:, None] > 1.0e-8),
        )
        - 1.0
    )
    legal = np.asarray(
        exit_sellable[
            int(date_idx) + 1 : int(date_idx) + int(horizon) + 1,
            symbol_idx,
        ],
        dtype=bool,
    ).T
    legal[:, 0] = False
    choice = legal & np.isfinite(relative)
    valid = np.isfinite(labels) & choice.any(axis=1)
    peak_dates = np.full(len(symbol_idx), -1, dtype=np.int32)
    derived = np.full(len(symbol_idx), np.nan, dtype=np.float32)
    if bool(valid.any()):
        masked = np.where(choice, relative, -np.inf)
        peak_offset = np.argmax(masked, axis=1)
        rows = np.arange(len(symbol_idx), dtype=np.int64)
        values = relative[rows, peak_offset]
        derived[valid] = values[valid].astype(np.float32)
        peak_dates[valid] = (int(date_idx) + 1 + peak_offset[valid]).astype(np.int32)
        maximum_error = float(np.max(np.abs(values[valid] - labels[valid])))
        if maximum_error > 2.0e-5:
            raise AssertionError(
                f"derived MFE differs from stored label by {maximum_error:.8f}"
            )
    return peak_dates, derived


def _rank_subset(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=np.float32)
    keep = np.asarray(valid, dtype=bool)
    if bool(keep.any()):
        output[keep] = economic._rank01(np.asarray(values, dtype=np.float64)[keep])
    return output


def _day_true_ranks(
    *,
    mfe10: np.ndarray,
    mfe20: np.ndarray,
    risk10: np.ndarray,
    risk20: np.ndarray,
    state10: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(mfe10)
    panel = np.full((count, len(RANK_COLUMNS)), np.nan, dtype=np.float32)
    valid10 = (
        np.isfinite(mfe10)
        & np.isfinite(risk10)
        & (np.asarray(state10, dtype=np.int16) >= 0)
    )
    valid20 = (
        np.isfinite(mfe20)
        & np.isfinite(risk20)
        & np.isfinite(risk10)
        & (np.asarray(state10, dtype=np.int16) >= 0)
    )
    panel[:, 0] = _rank_subset(mfe10, valid10)
    panel[:, 1] = _rank_subset(mfe20, valid20)
    panel[:, 2] = _rank_subset(
        (np.asarray(state10, dtype=np.int16) == 0).astype(np.float64),
        valid10,
    )
    panel[:, 3] = _rank_subset(
        (np.asarray(state10, dtype=np.int16) == 1).astype(np.float64),
        valid10,
    )
    panel[:, 4] = _rank_subset(
        (np.asarray(state10, dtype=np.int16) == 2).astype(np.float64),
        valid10,
    )
    panel[:, 5] = _rank_subset(risk10, valid10)
    panel[:, 6] = _rank_subset(risk20, valid20)
    dual_valid = valid10 & valid20
    if bool(dual_valid.any()):
        minimum = np.minimum(panel[dual_valid, 0], panel[dual_valid, 1])
        panel[dual_valid, 7] = economic._rank01(minimum)
    return panel, valid10, valid20, dual_valid


def _oracle_book_complete(
    output_root: Path,
    *,
    study_sha256: str | None = None,
) -> bool:
    path = output_root / "oracle_book/manifest.json"
    if not path.is_file():
        return False
    try:
        payload = _load_json(path)
        if payload.get("schema") != BOOK_SCHEMA or payload.get("status") != "completed":
            return False
        if (
            study_sha256 is not None
            and payload.get("study_config_sha256") != study_sha256
        ):
            return False
        for record in dict(payload["files"]).values():
            _verify_record(record)
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
    return True


def prepare_oracle_book(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    study_sha256 = _study_hash(study_path)
    if _oracle_book_complete(output_root, study_sha256=study_sha256):
        return _load_json(output_root / "oracle_book/manifest.json")
    _memory_guard()
    sources = dict(study["sources"])
    contract_path = _resolve(sources["entry_contract_manifest"])
    feature_path = _resolve(sources["base_feature_manifest"])
    label_path = _resolve(sources["label_manifest"])
    pack_path = _resolve(sources["pack_manifest"])
    v4_signal_path = _resolve(sources["v4_signal_book_manifest"])
    contract = _load_json(contract_path)
    feature = _load_json(feature_path)
    labels_manifest = _load_json(label_path)
    pack = _load_json(pack_path)
    v4_signal = _load_json(v4_signal_path)
    if contract.get("schema") != "seq100_entry_contract_oos_manifest/v4":
        raise ValueError("entry contract is not v4")
    if labels_manifest.get("schema") != "seq100_future_path_candidate_labels/v1":
        raise ValueError("true label manifest changed")
    if labels_manifest["maximum_outcome_date_read"] != "2025-12-31":
        raise ValueError("true labels consume another outcome cutoff")
    if int(labels_manifest["forbidden_outcome_year"]) != 2026:
        raise ValueError("true labels do not preserve the 2026 ban")
    boundaries = dict(labels_manifest["horizon_boundaries"])
    if boundaries["10"]["latest_signal_date_with_complete_outcome"] != str(
        study["period"]["mfe10_latest_complete_signal_date"]
    ):
        raise ValueError("D10 complete-label boundary changed")
    if boundaries["20"]["latest_signal_date_with_complete_outcome"] != str(
        study["period"]["mfe20_latest_complete_signal_date"]
    ):
        raise ValueError("D20 complete-label boundary changed")
    rows = np.load(
        _verify_record(contract["files"]["candidate_rows"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    candidate_count = int(feature["candidate_alignment"]["candidate_count"])
    feature_files = dict(feature["files"])
    candidate_date_idx = np.memmap(
        _resolve(feature_files["candidate_date_idx"]["path"]),
        dtype=np.int32,
        mode="r",
        shape=(candidate_count,),
    )
    candidate_symbol_idx = np.memmap(
        _resolve(feature_files["candidate_symbol_idx"]["path"]),
        dtype=np.int32,
        mode="r",
        shape=(candidate_count,),
    )
    label_files = dict(labels_manifest["files"])
    label_columns = tuple(labels_manifest["label_columns"])
    label_values = np.memmap(
        _resolve(label_files["candidate_labels"]["path"]),
        dtype=np.float32,
        mode="r",
        shape=tuple(int(value) for value in label_files["candidate_labels"]["shape"]),
    )
    state_values = np.memmap(
        _resolve(label_files["state_labels"]["path"]),
        dtype=np.int8,
        mode="r",
        shape=tuple(int(value) for value in label_files["state_labels"]["shape"]),
    )
    label_index = {name: idx for idx, name in enumerate(label_columns)}
    state_index = {
        name: idx for idx, name in enumerate(labels_manifest["state_columns"])
    }
    contract_rows = np.asarray(rows, dtype=np.int64)
    all_dates = np.asarray(candidate_date_idx[contract_rows], dtype=np.int32)
    all_symbols = np.asarray(candidate_symbol_idx[contract_rows], dtype=np.int32)
    date_values = np.asarray(pack["date_values"], dtype=str)
    years = np.asarray(
        [int(str(date_values[int(value)])[:4]) for value in all_dates],
        dtype=np.int16,
    )
    keep = np.isin(years, np.asarray(YEARS, dtype=np.int16))
    selected_rows = contract_rows[keep]
    selected_dates = all_dates[keep]
    selected_symbols = all_symbols[keep]
    signal_date_idx = np.unique(selected_dates)
    source_signal_dates = np.load(
        _verify_record(v4_signal["files"]["signal_date_idx"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    if not np.array_equal(signal_date_idx, source_signal_dates):
        raise ValueError("oracle and v4 signal calendars differ")
    if str(date_values[int(signal_date_idx[-1])]) != "2025-12-31":
        raise ValueError("oracle book does not end at 2025-12-31")
    cutoff_rows = np.flatnonzero(date_values == "2025-12-31")
    if len(cutoff_rows) != 1:
        raise ValueError("2025 cutoff date is absent or duplicated")
    cutoff_idx = int(cutoff_rows[0])
    n_days = len(signal_date_idx)
    symbol_count = int(pack["symbol_count"])
    panel = np.full(
        (n_days, symbol_count, len(RANK_COLUMNS)),
        np.nan,
        dtype=np.float32,
    )
    raw_panel = np.full((n_days, symbol_count, 4), np.nan, dtype=np.float32)
    peak10 = np.full((n_days, symbol_count), -1, dtype=np.int32)
    peak20 = np.full((n_days, symbol_count), -1, dtype=np.int32)
    orders = np.full(
        (len(ORDER_COLUMNS), n_days, symbol_count),
        -1,
        dtype=np.int32,
    )
    order_counts = np.zeros((len(ORDER_COLUMNS), n_days), dtype=np.int32)
    exit_sellable = _open_array(pack["masks"]["exit_sellable"], dtype=np.bool_)
    next_buyable = np.load(
        _verify_record(v4_signal["files"]["next_open_buyable"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    if next_buyable.shape != (n_days, symbol_count):
        raise ValueError("next-open buyability shape changed")
    future_reader = FuturePathReader(pack)
    date_to_day = {int(value): idx for idx, value in enumerate(signal_date_idx)}
    group_boundaries = np.flatnonzero(
        np.r_[True, selected_dates[1:] != selected_dates[:-1], True]
    )
    label_error_maximum = 0.0
    complete_counts = {10: 0, 20: 0}
    started = time.monotonic()
    last_emit = started
    for start, stop in pairwise(group_boundaries):
        _memory_guard()
        date_idx = int(selected_dates[start])
        day = int(date_to_day[date_idx])
        symbols = np.asarray(selected_symbols[start:stop], dtype=np.int32)
        source_rows = np.asarray(selected_rows[start:stop], dtype=np.int64)
        if len(np.unique(symbols)) != len(symbols):
            raise ValueError(f"duplicate candidate on {date_values[date_idx]}")
        mfe10 = np.asarray(
            label_values[source_rows, label_index["mfe_10"]],
            dtype=np.float32,
        )
        mfe20 = np.asarray(
            label_values[source_rows, label_index["mfe_20"]],
            dtype=np.float32,
        )
        risk10 = np.asarray(
            label_values[source_rows, label_index["pre_peak_mae_10"]],
            dtype=np.float32,
        )
        risk20 = np.asarray(
            label_values[source_rows, label_index["pre_peak_mae_20"]],
            dtype=np.float32,
        )
        state10 = np.asarray(
            state_values[source_rows, state_index["state_10"]],
            dtype=np.int8,
        )
        local_panel, valid10, valid20, dual_valid = _day_true_ranks(
            mfe10=mfe10,
            mfe20=mfe20,
            risk10=risk10,
            risk20=risk20,
            state10=state10,
        )
        panel[day, symbols] = local_panel
        raw_panel[day, symbols, 0] = mfe10
        raw_panel[day, symbols, 1] = mfe20
        raw_panel[day, symbols, 2] = risk10
        raw_panel[day, symbols, 3] = risk20
        support = (valid10, valid20, dual_valid)
        score_columns = (0, 1, 7)
        for order_idx, (current_valid, score_column) in enumerate(
            zip(support, score_columns, strict=True)
        ):
            executable = current_valid & np.asarray(
                next_buyable[day, symbols],
                dtype=bool,
            )
            current_symbols = symbols[executable]
            scores = local_panel[executable, score_column]
            ranked = np.lexsort(
                (current_symbols, -np.asarray(scores, dtype=np.float64))
            )
            count = len(current_symbols)
            orders[order_idx, day, :count] = current_symbols[ranked]
            order_counts[order_idx, day] = count
        for horizon, mfe, valid, destination in (
            (10, mfe10, valid10, peak10),
            (20, mfe20, valid20, peak20),
        ):
            if not bool(valid.any()):
                continue
            if date_idx + horizon > cutoff_idx:
                raise AssertionError("finite label crosses the 2025 cutoff")
            current_symbols = symbols[valid]
            future_close, d1_open = future_reader.future_close_and_d1_open(
                date_idx=date_idx,
                symbols=current_symbols,
                horizon=horizon,
            )
            dates, derived = derive_peak_close_dates(
                date_idx=date_idx,
                symbols=current_symbols,
                horizon=horizon,
                label_mfe=mfe[valid],
                future_close=future_close,
                d1_open=d1_open,
                exit_sellable=exit_sellable,
                cutoff_idx=cutoff_idx,
            )
            destination[day, current_symbols] = dates
            label_error_maximum = max(
                label_error_maximum,
                float(np.nanmax(np.abs(derived - mfe[valid]))),
            )
            complete_counts[horizon] += len(current_symbols)
        now = time.monotonic()
        if now - last_emit >= 30.0:
            _emit(
                "oracle_book_progress",
                signal_date=str(date_values[date_idx]),
                elapsed_seconds=round(now - started, 2),
            )
            last_emit = now
    if label_error_maximum > 2.0e-5:
        raise AssertionError("stored and derived MFE differ")
    book_root = output_root / "oracle_book"
    files: dict[str, dict[str, Any]] = {}
    arrays = {
        "signal_date_idx": signal_date_idx.astype(np.int32),
        "rank_panel": panel,
        "raw_label_panel": raw_panel,
        "candidate_orders": orders,
        "candidate_order_counts": order_counts,
        "peak_close_date_idx_10": peak10,
        "peak_close_date_idx_20": peak20,
    }
    for name, values in arrays.items():
        path = book_root / f"{name}.npy"
        _save_npy(path, values)
        files[name] = _file_record(
            path,
            shape=list(values.shape),
            dtype=str(values.dtype),
        )
    for name in (
        "next_open_buyable",
        "next_open_sellable",
        "benchmark_returns",
    ):
        files[name] = dict(v4_signal["files"][name])
    candidate_diagnostics = _candidate_diagnostics(
        study=study,
        signal_date_idx=signal_date_idx,
        date_values=date_values,
        raw_panel=raw_panel,
        peak10=peak10,
        peak20=peak20,
    )
    diagnostics_path = book_root / "candidate_diagnostics.parquet"
    _write_parquet(diagnostics_path, candidate_diagnostics)
    files["candidate_diagnostics"] = _file_record(
        diagnostics_path,
        row_count=len(candidate_diagnostics),
        columns=list(candidate_diagnostics.columns),
    )
    manifest = {
        "schema": BOOK_SCHEMA,
        "status": "completed",
        "completed_at": economic._now(),
        "study_id": STUDY_ID,
        "study_config_sha256": study_sha256,
        "period": {
            "first_signal_date": str(date_values[int(signal_date_idx[0])]),
            "last_mark_date": str(date_values[int(signal_date_idx[-1])]),
            "maximum_outcome_date_read": "2025-12-31",
            "forbidden_2026_row_count": 0,
            "mfe10_last_complete_signal_date": str(
                study["period"]["mfe10_latest_complete_signal_date"]
            ),
            "mfe20_last_complete_signal_date": str(
                study["period"]["mfe20_latest_complete_signal_date"]
            ),
        },
        "rank_columns": list(RANK_COLUMNS),
        "raw_label_columns": [
            "mfe_10",
            "mfe_20",
            "pre_peak_mae_10",
            "pre_peak_mae_20",
        ],
        "order_columns": list(ORDER_COLUMNS),
        "signal_date_count": int(n_days),
        "candidate_row_count": len(selected_rows),
        "complete_label_counts": {
            "mfe_10": int(complete_counts[10]),
            "mfe_20": int(complete_counts[20]),
        },
        "maximum_mfe_reconstruction_error": float(label_error_maximum),
        "entry_fill_foresight": True,
        "source_hashes": {
            "entry_contract_manifest": economic._sha256(contract_path),
            "base_feature_manifest": economic._sha256(feature_path),
            "label_manifest": economic._sha256(label_path),
            "pack_manifest": economic._sha256(pack_path),
            "v4_signal_book_manifest": economic._sha256(v4_signal_path),
        },
        "files": files,
    }
    _write_json(book_root / "manifest.json", manifest)
    _emit(
        "oracle_book_completed",
        signal_dates=n_days,
        candidate_rows=len(selected_rows),
        mfe10_labels=complete_counts[10],
        mfe20_labels=complete_counts[20],
    )
    return manifest


def _approximate_roundtrip_return(
    *,
    mfe: np.ndarray,
    sell_dates: np.ndarray,
    study: Mapping[str, Any],
    cost_scenario: str,
    costs: Any | None = None,
    slippage_multiplier: float | None = None,
) -> np.ndarray:
    current_costs = costs
    if current_costs is None:
        pack = _load_json(_resolve(study["sources"]["pack_manifest"]))
        current_costs = parse_execution_costs(pack)
    if slippage_multiplier is None:
        source_study = _load_json(_resolve(study["sources"]["v4_economic_study"]))
        multiplier = float(
            dict(source_study["execution"]["cost_scenarios"])[cost_scenario]
        )
    else:
        multiplier = float(slippage_multiplier)
    buy_slippage = current_costs.slippage_bps * multiplier / 10_000.0
    sell_slippage = current_costs.slippage_bps * multiplier / 10_000.0
    commission = current_costs.commission_bps / 10_000.0
    transfer = current_costs.transfer_fee_bps / 10_000.0
    dates = np.asarray(sell_dates, dtype=str)
    stamp = np.asarray(
        [
            economic._stamp_tax_bps(current_costs, str(value)) / 10_000.0
            for value in dates
        ],
        dtype=np.float64,
    )
    buy_factor = (1.0 + buy_slippage) * (1.0 + commission + transfer)
    sell_factor = (1.0 - sell_slippage) * (1.0 - commission - transfer - stamp)
    return (1.0 + np.asarray(mfe, dtype=np.float64)) * sell_factor / buy_factor - 1.0


def _candidate_diagnostics(
    *,
    study: Mapping[str, Any],
    signal_date_idx: np.ndarray,
    date_values: np.ndarray,
    raw_panel: np.ndarray,
    peak10: np.ndarray,
    peak20: np.ndarray,
) -> pd.DataFrame:
    source_study = _load_json(_resolve(study["sources"]["v4_economic_study"]))
    costs = parse_execution_costs(
        _load_json(_resolve(study["sources"]["pack_manifest"]))
    )
    scenario_multipliers = dict(source_study["execution"]["cost_scenarios"])
    records: list[dict[str, Any]] = []
    for horizon, column, peaks in ((10, 0, peak10), (20, 1, peak20)):
        for day, date_idx in enumerate(np.asarray(signal_date_idx, dtype=np.int32)):
            values = np.asarray(raw_panel[day, :, column], dtype=np.float64)
            valid = np.isfinite(values) & (np.asarray(peaks[day]) >= 0)
            if not bool(valid.any()):
                continue
            current = values[valid]
            peak_delay = np.asarray(peaks[day, valid], dtype=np.int32) - int(date_idx)
            ranks = economic._rank01(current)
            sell_dates = np.asarray(
                date_values[np.asarray(peaks[day, valid], dtype=np.int64)],
                dtype=str,
            )
            net_base = _approximate_roundtrip_return(
                mfe=current,
                sell_dates=sell_dates,
                study=study,
                cost_scenario="base",
                costs=costs,
                slippage_multiplier=float(scenario_multipliers["base"]),
            )
            net_stress = _approximate_roundtrip_return(
                mfe=current,
                sell_dates=sell_dates,
                study=study,
                cost_scenario="stress",
                costs=costs,
                slippage_multiplier=float(scenario_multipliers["stress"]),
            )
            for fraction, name in ((1.0, "all"), (0.05, "top5"), (0.01, "top1")):
                keep = ranks >= 1.0 - fraction - 1.0e-12
                records.append(
                    {
                        "year": int(str(date_values[int(date_idx)])[:4]),
                        "signal_date": str(date_values[int(date_idx)]),
                        "horizon": int(horizon),
                        "segment": name,
                        "candidate_count": int(np.sum(keep)),
                        "mean_mfe": float(np.mean(current[keep])),
                        "median_mfe": float(np.median(current[keep])),
                        "mean_peak_day": float(np.mean(peak_delay[keep])),
                        "median_peak_day": float(np.median(peak_delay[keep])),
                        "base_positive_net_fraction": float(
                            np.mean(net_base[keep] > 0.0)
                        ),
                        "stress_positive_net_fraction": float(
                            np.mean(net_stress[keep] > 0.0)
                        ),
                        "mean_base_approx_net_return": float(np.mean(net_base[keep])),
                        "mean_stress_approx_net_return": float(
                            np.mean(net_stress[keep])
                        ),
                    }
                )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("candidate oracle diagnostics are empty")
    return frame


class OracleSignalBook:
    def __init__(
        self,
        *,
        study: Mapping[str, Any],
        output_root: Path,
    ) -> None:
        if not _oracle_book_complete(output_root):
            raise FileNotFoundError("oracle book is incomplete")
        self.oracle_study = dict(study)
        self.study = _load_json(
            _resolve(self.oracle_study["sources"]["v4_economic_study"])
        )
        self.manifest = _load_json(output_root / "oracle_book/manifest.json")
        files = dict(self.manifest["files"])
        self.signal_date_idx = np.load(
            _verify_record(files["signal_date_idx"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.rank_panel = np.load(
            _verify_record(files["rank_panel"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.raw_label_panel = np.load(
            _verify_record(files["raw_label_panel"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.orders = np.load(
            _verify_record(files["candidate_orders"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.order_counts = np.load(
            _verify_record(files["candidate_order_counts"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.peak10 = np.load(
            _verify_record(files["peak_close_date_idx_10"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.peak20 = np.load(
            _verify_record(files["peak_close_date_idx_20"]),
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
        self.pack_path = _resolve(self.oracle_study["sources"]["pack_manifest"])
        self.pack = _load_json(self.pack_path)
        self.date_values = np.asarray(self.pack["date_values"], dtype=str)
        self.symbol_values = np.asarray(self.pack["symbol_values"], dtype=str)
        self.symbol_count = int(self.pack["symbol_count"])
        self.raw_open = economic._source_array(
            self.pack,
            "execution_arrays",
            "entry_open_raw",
            dtype=np.float32,
        )
        self.daily_raw = economic._source_array(
            self.pack,
            "feature_channels",
            "daily_raw",
            dtype=np.float32,
        )
        self.adjusted_open = self.daily_raw[:, :, 0]
        self.adjusted_close = self.daily_raw[:, :, 3]
        self.amount = self.daily_raw[:, :, 5]
        self.status_valid = economic._source_array(
            self.pack,
            "masks",
            "status_valid",
            dtype=np.bool_,
        )
        self.has_bar = economic._source_array(
            self.pack,
            "masks",
            "has_bar",
            dtype=np.bool_,
        )
        self.is_delisted = economic._source_array(
            self.pack,
            "masks",
            "is_delisted",
            dtype=np.bool_,
        )
        self.close_sellable = economic._source_array(
            self.pack,
            "masks",
            "exit_sellable",
            dtype=np.bool_,
        )
        self.costs = parse_execution_costs(self.pack)
        economic.SignalBook._validate_costs(self)
        self.terminal_recovery_fraction = float(
            dict(self.pack.get("terminal_execution", {}) or {}).get(
                "recovery_fraction_of_entry_notional",
                0.0,
            )
            or 0.0
        )
        self.cutoff_idx = int(np.flatnonzero(self.date_values == "2025-12-31")[0])

    @property
    def day_count(self) -> int:
        return len(self.signal_date_idx)

    def date_text(self, day: int) -> str:
        return str(self.date_values[int(self.signal_date_idx[int(day)])])

    def selector_column(self, family: str) -> int:
        return economic.SignalBook.selector_column(self, family)

    def order_column(self, family: str) -> int:
        if family == "mfe10_primary":
            return 0
        if family == "mfe20_primary":
            return 1
        if family.startswith("dual_mfe_"):
            return 2
        raise ValueError(f"unknown oracle family: {family}")

    def symbols_for_day(self, family: str, day: int) -> np.ndarray:
        column = self.order_column(family)
        count = int(self.order_counts[column, int(day)])
        return np.asarray(
            self.orders[column, int(day), :count],
            dtype=np.int32,
        )

    def rank(self, day: int, symbol_idx: int, column: int) -> float:
        return float(self.rank_panel[int(day), int(symbol_idx), int(column)])

    def candidate_count(self, day: int) -> int:
        return int(np.max(self.order_counts[:, int(day)]))

    def trailing_amount(self, *, signal_date_idx: int, symbol_idx: int) -> float:
        start = max(0, int(signal_date_idx) - 19)
        values = np.asarray(
            self.amount[start : int(signal_date_idx) + 1, int(symbol_idx)],
            dtype=np.float64,
        )
        valid = values[np.isfinite(values) & (values > 0.0)]
        return float(np.median(valid)) if valid.size else math.nan

    def exit_horizon(self, family: str) -> int:
        return 10 if family == "mfe10_primary" else 20

    def peak_close_date(self, *, day: int, symbol_idx: int, family: str) -> int:
        source = self.peak10 if self.exit_horizon(family) == 10 else self.peak20
        return int(source[int(day), int(symbol_idx)])


@dataclass(frozen=True)
class ScheduledBuy:
    symbol_idx: int
    signal_day: int
    reason: str


@dataclass(frozen=True)
class ScheduledSell:
    symbol_idx: int
    signal_day: int
    reason: str
    paired_buy: ScheduledBuy | None


def _internal_spec(spec: OracleTaskSpec) -> economic.TaskSpec:
    return economic.TaskSpec(
        family=spec.family,
        exposure_mode=spec.exposure_mode,
        slot_count=int(spec.slot_count),
        buffer_multiplier=float(spec.buffer_multiplier or 0.0),
        cost_scenario=spec.cost_scenario,
    )


def _eligible_scheduled_candidates(
    *,
    book: OracleSignalBook,
    spec: OracleTaskSpec,
    day: int,
    held: set[int],
    limit: int,
    require_post_peak_open: bool,
) -> list[ScheduledBuy]:
    if limit <= 0:
        return []
    internal = _internal_spec(spec)
    ordered = book.symbols_for_day(spec.family, day)
    if len(ordered) == 0:
        return []
    gate_set = economic._daily_gate_set(
        book=book,
        spec=internal,
        day=day,
        ordered_symbols=ordered,
    )
    threshold = float(book.study["policies"]["strong_entry_rank"])
    if gate_set is None:
        scan_count = len(ordered)
    else:
        scan_count = max(
            1,
            math.ceil(
                len(ordered) * float(book.study["policies"]["auxiliary_top_fraction"])
            ),
        )
    output: list[ScheduledBuy] = []
    for raw_symbol in ordered[:scan_count]:
        symbol_idx = int(raw_symbol)
        if symbol_idx in held:
            continue
        if gate_set is not None and symbol_idx not in gate_set:
            continue
        if not economic._entry_threshold_pass(
            book=book,
            spec=internal,
            day=day,
            symbol_idx=symbol_idx,
            threshold=threshold,
        ):
            continue
        peak_date_idx = book.peak_close_date(
            day=day,
            symbol_idx=symbol_idx,
            family=spec.family,
        )
        if peak_date_idx < 0:
            continue
        if require_post_peak_open and peak_date_idx + 1 > book.cutoff_idx:
            continue
        output.append(
            ScheduledBuy(
                symbol_idx=symbol_idx,
                signal_day=int(day),
                reason="oracle_empty_or_peak_slot",
            )
        )
        held.add(symbol_idx)
        if len(output) >= int(limit):
            break
    return output


def _rewrite_rerank_result(
    *,
    spec: OracleTaskSpec,
    result: dict[str, Any],
    equity: pd.DataFrame,
    trades: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    old_task_id = str(result["task_id"])
    new_task_id = spec.task_id
    result = {
        **result,
        "schema": TASK_SCHEMA,
        "study_id": STUDY_ID,
        "task": asdict(spec),
        "task_id": new_task_id,
        "oracle_semantics": {
            "future_labels_used": True,
            "mode": spec.mode,
            "peak_timing_used": False,
            "entry_fill_foresight": True,
            "not_oos": True,
        },
    }
    for frame in (equity, trades):
        if "task_id" in frame:
            frame = frame.copy()
            frame["task_id"] = new_task_id
        if frame is equity:
            equity = frame
        else:
            trades = frame
    if old_task_id == new_task_id:
        raise AssertionError("oracle rerank task id did not change")
    return result, equity, trades


def _process_buy(
    *,
    book: OracleSignalBook,
    spec: OracleTaskSpec,
    order: ScheduledBuy,
    date_idx: int,
    trade_date: str,
    equity_open: float,
    cash: float,
    gross_cash: float,
    positions: dict[int, economic.Position],
    scheduled_peak: dict[int, int],
    attempts: list[dict[str, Any]],
    counters: defaultdict[str, int],
    multiplier: float,
) -> tuple[float, float, float, float, dict[str, float]]:
    symbol_idx = int(order.symbol_idx)
    signal_idx = int(book.signal_date_idx[int(order.signal_day)])
    signal_date = book.date_text(int(order.signal_day))
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
                symbol=str(book.symbol_values[symbol_idx]),
                reason=order.reason,
            )
        )
        return cash, gross_cash, 0.0, 0.0, {}
    if not bool(book.next_buyable[int(order.signal_day), symbol_idx]):
        counters["failed_buy_count"] += 1
        attempts.append(
            economic._attempt_row(
                task_id=spec.task_id,
                side="buy",
                status="failed",
                signal_date=signal_date,
                execution_date=trade_date,
                symbol_idx=symbol_idx,
                symbol=str(book.symbol_values[symbol_idx]),
                reason=order.reason,
            )
        )
        return cash, gross_cash, 0.0, 0.0, {}
    raw_open = float(book.raw_open[date_idx, symbol_idx])
    adjusted_open = float(book.adjusted_open[date_idx, symbol_idx])
    allocation = min(
        float(cash),
        max(float(equity_open), 0.0) / int(spec.slot_count),
    )
    position, details = economic._buy_position(
        available_cash=float(cash),
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
            economic._attempt_row(
                task_id=spec.task_id,
                side="buy",
                status="failed_cash_or_lot",
                signal_date=signal_date,
                execution_date=trade_date,
                symbol_idx=symbol_idx,
                symbol=str(book.symbol_values[symbol_idx]),
                reason=order.reason,
            )
        )
        return cash, gross_cash, 0.0, 0.0, {}
    peak_date_idx = book.peak_close_date(
        day=int(order.signal_day),
        symbol_idx=symbol_idx,
        family=spec.family,
    )
    if peak_date_idx <= date_idx:
        raise AssertionError("scheduled peak does not satisfy T+1")
    positions[symbol_idx] = position
    scheduled_peak[symbol_idx] = int(peak_date_idx)
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
            symbol=str(book.symbol_values[symbol_idx]),
            reason=order.reason,
            details=details,
            participation=participation,
        )
    )
    return (
        float(cash + details["cash_flow"]),
        float(gross_cash - details["gross_notional"]),
        float(details["total_cost"]),
        float(details["fill_notional"]),
        dict(details),
    )


def _execute_scheduled_sale(
    *,
    book: OracleSignalBook,
    spec: OracleTaskSpec,
    symbol_idx: int,
    signal_date: str,
    execution_date: str,
    execution_date_idx: int,
    adjusted_price: float,
    sellable: bool,
    reason: str,
    cash: float,
    gross_cash: float,
    positions: dict[int, economic.Position],
    scheduled_peak: dict[int, int],
    attempts: list[dict[str, Any]],
    counters: defaultdict[str, int],
    consecutive_blocked_sells: dict[int, int],
    holding_days: list[int],
    sell_delay_days: list[int],
    realized_contribution: defaultdict[int, float],
    multiplier: float,
) -> tuple[bool, float, float, float, float, dict[str, float]]:
    position = positions.get(int(symbol_idx))
    if position is None:
        counters["stale_sell_order_count"] += 1
        return False, cash, gross_cash, 0.0, 0.0, {}
    t_plus_one = int(execution_date_idx) > int(position.entry_date_idx)
    if not bool(sellable) or not t_plus_one:
        counters["failed_sell_count"] += 1
        if not bool(sellable):
            counters["blocked_sell_count"] += 1
        consecutive_blocked_sells[int(symbol_idx)] = (
            consecutive_blocked_sells.get(int(symbol_idx), 0) + 1
        )
        attempts.append(
            economic._attempt_row(
                task_id=spec.task_id,
                side="sell",
                status="failed",
                signal_date=signal_date,
                execution_date=execution_date,
                symbol_idx=int(symbol_idx),
                symbol=str(book.symbol_values[int(symbol_idx)]),
                reason=reason,
            )
        )
        return False, cash, gross_cash, 0.0, 0.0, {}
    proceeds, details = economic._sell_position(
        position=position,
        adjusted_open=float(adjusted_price),
        trade_date=execution_date,
        costs=book.costs,
        slippage_multiplier=multiplier,
    )
    duration = int(execution_date_idx) - int(position.entry_date_idx)
    holding_days.append(duration)
    delay = int(consecutive_blocked_sells.pop(int(symbol_idx), 0))
    if delay > 0:
        sell_delay_days.append(delay)
    realized_contribution[int(symbol_idx)] += float(
        proceeds - position.net_cash_outflow
    )
    del positions[int(symbol_idx)]
    scheduled_peak.pop(int(symbol_idx), None)
    counters["sell_count"] += 1
    trailing = book.trailing_amount(
        signal_date_idx=int(execution_date_idx),
        symbol_idx=int(symbol_idx),
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
            execution_date=execution_date,
            symbol_idx=int(symbol_idx),
            symbol=str(book.symbol_values[int(symbol_idx)]),
            reason=reason,
            details=details,
            holding_days=duration,
            sell_delay_days=delay,
            participation=participation,
        )
    )
    return (
        True,
        float(cash + proceeds),
        float(gross_cash + details["gross_notional"]),
        float(details["total_cost"]),
        float(details["fill_notional"]),
        dict(details),
    )


def _finalize_scheduled_result(
    *,
    book: OracleSignalBook,
    spec: OracleTaskSpec,
    cash: float,
    positions: Mapping[int, economic.Position],
    equity_frame: pd.DataFrame,
    attempts_frame: pd.DataFrame,
    total_cost: float,
    cost_parts: Mapping[str, float],
    traded_notional: float,
    counters: Mapping[str, int],
    holding_days: Sequence[int],
    sell_delay_days: Sequence[int],
    consecutive_blocked_sells: Mapping[int, int],
    realized_contribution: Mapping[int, float],
    maximum_conservation_error: float,
    scheduled_peak: Mapping[int, int],
    multiplier: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    monthly = economic._monthly_metrics(
        dates=equity_frame["trade_date"].to_numpy(dtype=str),
        equity=equity_frame["net_equity"].to_numpy(dtype=np.float64),
    )
    net_metrics = economic._annualized_metrics(
        equity=np.r_[
            STARTING_CASH_CNY,
            equity_frame["net_equity"].to_numpy(dtype=np.float64),
        ],
        daily_returns=np.r_[
            0.0,
            equity_frame["daily_net_return"].to_numpy(dtype=np.float64),
        ],
    )
    gross_metrics = economic._annualized_metrics(
        equity=np.r_[
            STARTING_CASH_CNY,
            equity_frame["gross_equity"].to_numpy(dtype=np.float64),
        ],
        daily_returns=np.r_[
            0.0,
            equity_frame["daily_gross_return"].to_numpy(dtype=np.float64),
        ],
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
    contribution = defaultdict(float, realized_contribution)
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
        contribution[int(symbol_idx)] += float(proceeds - position.net_cash_outflow)
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
        "completed_at": economic._now(),
        "study_id": STUDY_ID,
        "task": asdict(spec),
        "task_id": spec.task_id,
        "signal_book_sha256": book.manifest["files"]["rank_panel"]["sha256"],
        "oracle_semantics": {
            "future_labels_used": True,
            "mode": spec.mode,
            "peak_timing_used": True,
            "entry_fill_foresight": True,
            "not_oos": True,
        },
        "period": {
            "first_signal_date": str(equity_frame["trade_date"].iloc[0]),
            "last_mark_date": final_date,
            "continuous_years": list(YEARS),
            "annual_reset": False,
            "maximum_outcome_date_read": "2025-12-31",
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
            "unresolved_scheduled_exit_count": len(scheduled_peak),
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
                **{
                    name: float(cost_parts.get(name, 0.0))
                    for name in (
                        "commission",
                        "transfer_fee",
                        "stamp_tax",
                        "slippage",
                    )
                },
            },
        },
        "annual": annual,
    }
    return result, monthly


def simulate_scheduled_task(
    *,
    book: OracleSignalBook,
    spec: OracleTaskSpec,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if spec.mode not in {
        "true_label_peak_close_hold",
        "true_label_post_peak_next_open_hold",
    }:
        raise ValueError("scheduled simulator received another oracle mode")
    post_peak_open = spec.mode == "true_label_post_peak_next_open_hold"
    multiplier = float(
        dict(book.study["execution"]["cost_scenarios"])[spec.cost_scenario]
    )
    cash = float(book.study["account"]["starting_cash_cny"])
    gross_cash = cash
    positions: dict[int, economic.Position] = {}
    scheduled_peak: dict[int, int] = {}
    pending_sells: tuple[ScheduledSell, ...] = ()
    pending_buys: tuple[ScheduledBuy, ...] = ()
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
    benchmark_wealth = float(STARTING_CASH_CNY)
    previous_equity = float(STARTING_CASH_CNY)
    previous_gross_equity = float(STARTING_CASH_CNY)
    for day in range(book.day_count):
        _memory_guard()
        date_idx = int(book.signal_date_idx[day])
        trade_date = book.date_text(day)
        adjusted_open_row = np.asarray(book.adjusted_open[date_idx], dtype=np.float64)
        equity_open = economic._portfolio_value(
            cash=cash,
            positions=positions,
            adjusted_prices=adjusted_open_row,
        )
        successful_pairs: list[ScheduledBuy] = []
        if post_peak_open:
            for order in pending_sells:
                sold, cash, gross_cash, cost, notional, details = (
                    _execute_scheduled_sale(
                        book=book,
                        spec=spec,
                        symbol_idx=int(order.symbol_idx),
                        signal_date=book.date_text(int(order.signal_day)),
                        execution_date=trade_date,
                        execution_date_idx=date_idx,
                        adjusted_price=float(adjusted_open_row[int(order.symbol_idx)]),
                        sellable=bool(
                            book.next_sellable[
                                int(order.signal_day),
                                int(order.symbol_idx),
                            ]
                        ),
                        reason=order.reason,
                        cash=cash,
                        gross_cash=gross_cash,
                        positions=positions,
                        scheduled_peak=scheduled_peak,
                        attempts=attempts,
                        counters=counters,
                        consecutive_blocked_sells=consecutive_blocked_sells,
                        holding_days=holding_days,
                        sell_delay_days=sell_delay_days,
                        realized_contribution=realized_contribution,
                        multiplier=multiplier,
                    )
                )
                total_cost += cost
                traded_notional += notional
                for name in ("commission", "transfer_fee", "stamp_tax", "slippage"):
                    cost_parts[name] += float(details.get(name, 0.0))
                if sold and order.paired_buy is not None:
                    successful_pairs.append(order.paired_buy)
        for order in (*pending_buys, *successful_pairs):
            cash, gross_cash, cost, notional, details = _process_buy(
                book=book,
                spec=spec,
                order=order,
                date_idx=date_idx,
                trade_date=trade_date,
                equity_open=equity_open,
                cash=cash,
                gross_cash=gross_cash,
                positions=positions,
                scheduled_peak=scheduled_peak,
                attempts=attempts,
                counters=counters,
                multiplier=multiplier,
            )
            total_cost += cost
            traded_notional += notional
            for name in ("commission", "transfer_fee", "stamp_tax", "slippage"):
                cost_parts[name] += float(details.get(name, 0.0))
        pending_sells = ()
        pending_buys = ()
        adjusted_close_row = np.asarray(
            book.adjusted_close[date_idx],
            dtype=np.float64,
        )
        if not post_peak_open:
            due = sorted(
                symbol_idx
                for symbol_idx, peak_date_idx in scheduled_peak.items()
                if int(peak_date_idx) <= date_idx
            )
            for symbol_idx in due:
                sold, cash, gross_cash, cost, notional, details = (
                    _execute_scheduled_sale(
                        book=book,
                        spec=spec,
                        symbol_idx=int(symbol_idx),
                        signal_date=trade_date,
                        execution_date=trade_date,
                        execution_date_idx=date_idx,
                        adjusted_price=float(adjusted_close_row[int(symbol_idx)]),
                        sellable=bool(book.close_sellable[date_idx, int(symbol_idx)]),
                        reason="true_label_peak_close",
                        cash=cash,
                        gross_cash=gross_cash,
                        positions=positions,
                        scheduled_peak=scheduled_peak,
                        attempts=attempts,
                        counters=counters,
                        consecutive_blocked_sells=consecutive_blocked_sells,
                        holding_days=holding_days,
                        sell_delay_days=sell_delay_days,
                        realized_contribution=realized_contribution,
                        multiplier=multiplier,
                    )
                )
                total_cost += cost
                traded_notional += notional
                for name in ("commission", "transfer_fee", "stamp_tax", "slippage"):
                    cost_parts[name] += float(details.get(name, 0.0))
                if not sold:
                    counters["unexpected_peak_close_failure_count"] += 1
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
                scheduled_peak.pop(symbol_idx, None)
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
                f"oracle cashflow conservation failed by {conservation_error:.8f}"
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
            held_for_selection = {int(value) for value in positions}
            if post_peak_open:
                due = sorted(
                    symbol_idx
                    for symbol_idx, peak_date_idx in scheduled_peak.items()
                    if int(peak_date_idx) <= date_idx
                )
                empty_count = max(int(spec.slot_count) - len(positions), 0)
                candidates = _eligible_scheduled_candidates(
                    book=book,
                    spec=spec,
                    day=day,
                    held=set(held_for_selection),
                    limit=empty_count + len(due),
                    require_post_peak_open=True,
                )
                unpaired = candidates[:empty_count]
                paired = candidates[empty_count:]
                pending_buys = tuple(unpaired)
                pending_sells = tuple(
                    ScheduledSell(
                        symbol_idx=int(symbol_idx),
                        signal_day=int(day),
                        reason="post_true_peak_next_open",
                        paired_buy=(paired[index] if index < len(paired) else None),
                    )
                    for index, symbol_idx in enumerate(due)
                )
            else:
                empty_count = max(int(spec.slot_count) - len(positions), 0)
                pending_buys = tuple(
                    _eligible_scheduled_candidates(
                        book=book,
                        spec=spec,
                        day=day,
                        held=set(held_for_selection),
                        limit=empty_count,
                        require_post_peak_open=False,
                    )
                )
            requested = {int(order.symbol_idx) for order in pending_sells}
            for symbol_idx in tuple(consecutive_blocked_sells):
                if symbol_idx not in positions or (
                    post_peak_open
                    and symbol_idx not in requested
                    and scheduled_peak.get(symbol_idx, book.cutoff_idx + 1) > date_idx
                ):
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
    result, monthly = _finalize_scheduled_result(
        book=book,
        spec=spec,
        cash=cash,
        positions=positions,
        equity_frame=equity_frame,
        attempts_frame=attempts_frame,
        total_cost=total_cost,
        cost_parts=cost_parts,
        traded_notional=traded_notional,
        counters=counters,
        holding_days=holding_days,
        sell_delay_days=sell_delay_days,
        consecutive_blocked_sells=consecutive_blocked_sells,
        realized_contribution=realized_contribution,
        maximum_conservation_error=maximum_conservation_error,
        scheduled_peak=scheduled_peak,
        multiplier=multiplier,
    )
    return result, equity_frame, attempts_frame, monthly


def simulate_task(
    *,
    book: OracleSignalBook,
    spec: OracleTaskSpec,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if spec.mode == "true_label_daily_rerank":
        result, equity, trades, monthly = economic.simulate_task(
            book=book,
            spec=_internal_spec(spec),
        )
        result, equity, trades = _rewrite_rerank_result(
            spec=spec,
            result=result,
            equity=equity,
            trades=trades,
        )
        return result, equity, trades, monthly
    return simulate_scheduled_task(book=book, spec=spec)


def _task_dir(output_root: Path, spec: OracleTaskSpec) -> Path:
    return output_root / "tasks" / spec.task_id


def _task_complete(
    *,
    output_root: Path,
    spec: OracleTaskSpec,
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
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
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
    spec = OracleTaskSpec(**dict(result["task"]))
    directory = _task_dir(output_root, spec)
    equity_path = directory / "equity.parquet"
    trades_path = directory / "trades.parquet"
    monthly_path = directory / "monthly.parquet"
    _write_parquet(equity_path, equity)
    _write_parquet(trades_path, trades)
    _write_parquet(monthly_path, monthly)
    payload = {
        **result,
        "study_config_sha256": study_sha256,
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
    manifest = prepare_oracle_book(
        study_path=study_path,
        output_root=output_root,
    )
    study_sha256 = _study_hash(study_path)
    signal_sha256 = str(manifest["files"]["rank_panel"]["sha256"])
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
        "oracle_account_tasks_starting",
        completed=complete_before,
        total=len(specs),
    )
    book = OracleSignalBook(study=study, output_root=output_root)
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
                "oracle_account_task_progress",
                completed=completed,
                total=len(specs),
                task_id=spec.task_id,
                elapsed_seconds=round(time.monotonic() - started, 2),
            )
    return {
        "status": "completed",
        "completed": completed,
        "total": len(specs),
    }


def status(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    load_study(study_path)
    study_sha256 = _study_hash(study_path)
    book_ready = _oracle_book_complete(
        output_root,
        study_sha256=study_sha256,
    )
    signal_sha256 = ""
    if book_ready:
        signal_sha256 = str(
            _load_json(output_root / "oracle_book/manifest.json")["files"][
                "rank_panel"
            ]["sha256"]
        )
    specs = task_specs()
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
        if book_ready
        else 0
    )
    return {
        "study_id": STUDY_ID,
        "oracle_book_complete": bool(book_ready),
        "completed_tasks": completed,
        "pending_tasks": len(specs) - completed,
        "total_tasks": len(specs),
        "evaluation_complete": (output_root / "evaluation/summary.json").is_file(),
    }


def _task_result(
    *,
    output_root: Path,
    spec: OracleTaskSpec,
) -> dict[str, Any]:
    path = _task_dir(output_root, spec) / "task_result.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _load_json(path)


def _task_metric_row(result: Mapping[str, Any]) -> dict[str, Any]:
    return economic._task_metric_row(result)


def _annual_metric_table(
    results: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    return economic._annual_metric_table(results)


def _monthly_metric_table(
    *,
    results: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        task = dict(result["task"])
        monthly = pd.read_parquet(_verify_record(result["files"]["monthly"]))
        equity = pd.read_parquet(
            _verify_record(result["files"]["equity"]),
            columns=["trade_date", "benchmark_wealth"],
        )
        equity["month"] = (
            pd.to_datetime(equity["trade_date"]).dt.to_period("M").astype(str)
        )
        benchmark_end = equity.groupby("month", sort=True)["benchmark_wealth"].last()
        previous_benchmark = STARTING_CASH_CNY
        benchmark_returns: dict[str, float] = {}
        for month, ending in benchmark_end.items():
            benchmark_returns[str(month)] = float(ending) / previous_benchmark - 1.0
            previous_benchmark = float(ending)
        for item in monthly.to_dict("records"):
            month = str(item["month"])
            net_return = float(item["net_return"])
            benchmark_return = float(benchmark_returns[month])
            rows.append(
                {
                    **task,
                    "task_id": str(result["task_id"]),
                    "month": month,
                    "net_return": net_return,
                    "benchmark_return": benchmark_return,
                    "relative_excess_return": (
                        (1.0 + net_return) / max(1.0 + benchmark_return, 1.0e-12) - 1.0
                    ),
                    "ending_equity": float(item["ending_equity"]),
                }
            )
    return pd.DataFrame.from_records(rows)


def _relative_accounting_errors(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for result in results:
        frame = pd.read_parquet(
            _verify_record(result["files"]["equity"]),
            columns=["gross_equity", "conservation_error"],
        )
        denominator = np.maximum(
            np.abs(frame["gross_equity"].to_numpy(dtype=np.float64)),
            STARTING_CASH_CNY,
        )
        relative = np.divide(
            frame["conservation_error"].to_numpy(dtype=np.float64),
            denominator,
            out=np.full(len(frame), np.inf, dtype=np.float64),
            where=np.isfinite(denominator) & (denominator > 0.0),
        )
        output[str(result["task_id"])] = float(np.max(relative))
    return output


def _configuration_table(
    metrics: pd.DataFrame,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    statistics = dict(study["statistics"])
    minimum_absolute = int(statistics["minimum_positive_absolute_years"])
    minimum_excess = int(statistics["minimum_positive_excess_years"])
    keys = [
        "mode",
        "family",
        "exposure_mode",
        "slot_count",
        "buffer_multiplier",
    ]
    records: list[dict[str, Any]] = []
    for values, part in metrics.groupby(keys, sort=True, dropna=False):
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
        mode, family, exposure, slots, buffer = values
        records.append(
            {
                "mode": str(mode),
                "family": str(family),
                "exposure_mode": str(exposure),
                "slot_count": int(slots),
                "buffer_multiplier": (
                    float(buffer) if math.isfinite(float(buffer)) else math.nan
                ),
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
    frame = pd.DataFrame.from_records(records)
    expected = 288 + 72 + 72
    if len(frame) != expected:
        raise AssertionError(
            f"configuration table has {len(frame)} rows, expected {expected}"
        )
    return frame


def stable_slot_bands(
    frame: pd.DataFrame,
    *,
    flag: str,
    width: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for start in range(len(SLOT_COUNTS) - int(width) + 1):
        slots = SLOT_COUNTS[start : start + int(width)]
        selected = frame[frame["slot_count"].astype(int).isin(slots)]
        if len(selected) != int(width):
            raise ValueError("scheduled slot-band support is incomplete")
        if bool(selected[flag].astype(bool).all()):
            output.append({"slot_counts": list(slots), "cell_count": int(width)})
    return output


def _mode_family_daily_series(
    *,
    output_root: Path,
    mode: str,
    family: str,
    exposure_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    series: list[np.ndarray] = []
    dates: np.ndarray | None = None
    buffers: tuple[float | None, ...] = (
        tuple(BUFFER_MULTIPLIERS) if mode == "true_label_daily_rerank" else (None,)
    )
    for slots in SLOT_COUNTS:
        for buffer in buffers:
            spec = OracleTaskSpec(
                mode=mode,
                family=family,
                exposure_mode=exposure_mode,
                slot_count=slots,
                buffer_multiplier=buffer,
                cost_scenario="base",
            )
            result = _task_result(output_root=output_root, spec=spec)
            frame = pd.read_parquet(
                _verify_record(result["files"]["equity"]),
                columns=["trade_date", "daily_relative_excess_return"],
            )
            current_dates = frame["trade_date"].astype(str).to_numpy()
            if dates is None:
                dates = current_dates
            elif not np.array_equal(dates, current_dates):
                raise ValueError("oracle family equity dates differ")
            series.append(
                frame["daily_relative_excess_return"].to_numpy(dtype=np.float64)
            )
    if dates is None or not series:
        raise AssertionError("oracle family series is empty")
    return dates, np.median(np.vstack(series), axis=0)


def _family_mode_statistics(
    *,
    output_root: Path,
    configurations: pd.DataFrame,
    study: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    statistics = dict(study["statistics"])
    rows: list[dict[str, Any]] = []
    daily_rows: list[pd.DataFrame] = []
    for mode in MODES:
        for family in FAMILIES:
            for exposure in EXPOSURE_MODES:
                selected = configurations[
                    configurations["mode"].astype(str).eq(mode)
                    & configurations["family"].astype(str).eq(family)
                    & configurations["exposure_mode"].astype(str).eq(exposure)
                ]
                if mode == "true_label_daily_rerank":
                    regions = economic.robust_rectangles(
                        selected,
                        flag="economic_pass",
                        slot_width=int(statistics["daily_rerank_slot_width"]),
                        buffer_width=int(statistics["daily_rerank_buffer_width"]),
                    )
                    gross_regions = economic.robust_rectangles(
                        selected,
                        flag="gross_pass",
                        slot_width=int(statistics["daily_rerank_slot_width"]),
                        buffer_width=int(statistics["daily_rerank_buffer_width"]),
                    )
                else:
                    regions = stable_slot_bands(
                        selected,
                        flag="economic_pass",
                        width=int(statistics["scheduled_slot_width"]),
                    )
                    gross_regions = stable_slot_bands(
                        selected,
                        flag="gross_pass",
                        width=int(statistics["scheduled_slot_width"]),
                    )
                dates, median_excess = _mode_family_daily_series(
                    output_root=output_root,
                    mode=mode,
                    family=family,
                    exposure_mode=exposure,
                )
                hac = economic.hac_mean_test(
                    median_excess,
                    maximum_lag=int(statistics["hac_maximum_lag"]),
                )
                bootstrap = economic.moving_block_bootstrap_lower(
                    median_excess,
                    block_length=int(statistics["bootstrap_block_length"]),
                    repetitions=int(statistics["bootstrap_repetitions"]),
                    seed=int(statistics["bootstrap_seed"]),
                    lower_quantile=float(statistics["bootstrap_lower_quantile"]),
                )
                rows.append(
                    {
                        "mode": mode,
                        "family": family,
                        "exposure_mode": exposure,
                        "economic_region_count": len(regions),
                        "gross_region_count": len(gross_regions),
                        "economic_regions": json.dumps(
                            regions,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "gross_regions": json.dumps(
                            gross_regions,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "economic_cell_count": int(
                            selected["economic_pass"].astype(bool).sum()
                        ),
                        "gross_cell_count": int(
                            selected["gross_pass"].astype(bool).sum()
                        ),
                        "median_daily_excess_mean": float(hac["mean"]),
                        "hac_standard_error": float(hac["standard_error"]),
                        "hac_t_statistic": float(hac["t_statistic"]),
                        "hac_one_sided_p_value": float(hac["one_sided_p_value"]),
                        "bootstrap_lower_90": float(bootstrap["lower_bound"]),
                        "bootstrap_positive_probability": float(
                            bootstrap["positive_probability"]
                        ),
                    }
                )
                daily_rows.append(
                    pd.DataFrame(
                        {
                            "trade_date": dates,
                            "mode": mode,
                            "family": family,
                            "exposure_mode": exposure,
                            "median_daily_relative_excess_return": (median_excess),
                        }
                    )
                )
    frame = pd.DataFrame.from_records(rows)
    frame["bh_q_value"] = economic.benjamini_hochberg(
        frame["hac_one_sided_p_value"].to_numpy(dtype=np.float64)
    )
    frame["preregistered_strong_region"] = (
        frame["economic_region_count"].astype(int).gt(0)
    )
    frame["statistically_confirmed"] = (
        frame["hac_one_sided_p_value"]
        .astype(float)
        .le(float(statistics["maximum_hac_p_value"]))
        & frame["bootstrap_lower_90"].astype(float).ge(0.0)
        & frame["bh_q_value"].astype(float).le(float(statistics["maximum_bh_q_value"]))
    )
    frame["strong_and_statistically_confirmed"] = frame[
        "preregistered_strong_region"
    ].astype(bool) & frame["statistically_confirmed"].astype(bool)
    return frame, pd.concat(daily_rows, ignore_index=True)


def _predicted_v4_gaps(
    *,
    study: Mapping[str, Any],
    output_root: Path,
    oracle_metrics: pd.DataFrame,
) -> pd.DataFrame:
    v4_root = _resolve(study["sources"]["v4_signal_book_manifest"]).parents[1]
    current = oracle_metrics[
        oracle_metrics["mode"].astype(str).eq("true_label_daily_rerank")
    ]
    rows: list[dict[str, Any]] = []
    for item in current.to_dict("records"):
        old_spec = economic.TaskSpec(
            family=str(item["family"]),
            exposure_mode=str(item["exposure_mode"]),
            slot_count=int(item["slot_count"]),
            buffer_multiplier=float(item["buffer_multiplier"]),
            cost_scenario=str(item["cost_scenario"]),
        )
        old_result = economic._task_result(
            output_root=v4_root,
            spec=old_spec,
        )
        old = economic._task_metric_row(old_result)
        oracle_spec = OracleTaskSpec(
            mode="true_label_daily_rerank",
            family=old_spec.family,
            exposure_mode=old_spec.exposure_mode,
            slot_count=old_spec.slot_count,
            buffer_multiplier=old_spec.buffer_multiplier,
            cost_scenario=old_spec.cost_scenario,
        )
        oracle_result = _task_result(
            output_root=output_root,
            spec=oracle_spec,
        )
        oracle_equity = pd.read_parquet(
            _verify_record(oracle_result["files"]["equity"]),
            columns=["trade_date", "daily_net_return"],
        )
        old_equity = pd.read_parquet(
            _verify_record(old_result["files"]["equity"]),
            columns=["trade_date", "daily_net_return"],
        )
        if not np.array_equal(
            oracle_equity["trade_date"].astype(str).to_numpy(),
            old_equity["trade_date"].astype(str).to_numpy(),
        ):
            raise ValueError("predicted/oracle account calendars differ")
        hac = economic.hac_mean_test(
            oracle_equity["daily_net_return"].to_numpy(dtype=np.float64)
            - old_equity["daily_net_return"].to_numpy(dtype=np.float64),
            maximum_lag=20,
        )
        rows.append(
            {
                "oracle_task_id": str(item["task_id"]),
                "predicted_task_id": old_spec.task_id,
                "family": old_spec.family,
                "exposure_mode": old_spec.exposure_mode,
                "slot_count": old_spec.slot_count,
                "buffer_multiplier": old_spec.buffer_multiplier,
                "cost_scenario": old_spec.cost_scenario,
                "oracle_terminal_return": float(item["terminal_cost_accrued_return"]),
                "predicted_terminal_return": float(old["terminal_cost_accrued_return"]),
                "terminal_return_gap": float(
                    item["terminal_cost_accrued_return"]
                    - old["terminal_cost_accrued_return"]
                ),
                "oracle_terminal_excess": float(
                    item["terminal_cost_accrued_relative_excess_return"]
                ),
                "predicted_terminal_excess": float(
                    old["terminal_cost_accrued_relative_excess_return"]
                ),
                "terminal_excess_gap": float(
                    item["terminal_cost_accrued_relative_excess_return"]
                    - old["terminal_cost_accrued_relative_excess_return"]
                ),
                "oracle_cagr": float(item["net_cagr"]),
                "predicted_cagr": float(old["net_cagr"]),
                "cagr_gap": float(item["net_cagr"] - old["net_cagr"]),
                "mean_daily_return_gap": float(hac["mean"]),
                "hac_t_statistic": float(hac["t_statistic"]),
                "hac_one_sided_p_value": float(hac["one_sided_p_value"]),
                **{
                    f"net_return_gap_{year}": float(
                        item[f"net_return_{year}"] - old[f"net_return_{year}"]
                    )
                    for year in YEARS
                },
            }
        )
    return pd.DataFrame.from_records(rows)


def weighted_interval_ceiling(
    trades: pd.DataFrame,
    *,
    first_date_idx: int,
    last_date_idx: int,
) -> tuple[float, np.ndarray]:
    required = {
        "signal_date_idx",
        "exit_date_idx",
        "log_wealth_increment",
    }
    if not required.issubset(trades.columns):
        raise ValueError("weighted interval trades are incomplete")
    size = int(last_date_idx) - int(first_date_idx) + 1
    if size <= 0:
        raise ValueError("weighted interval date range is empty")
    grouped: dict[int, list[int]] = defaultdict(list)
    for row_idx, row in enumerate(trades.to_dict("records")):
        start = int(row["signal_date_idx"])
        end = int(row["exit_date_idx"])
        weight = float(row["log_wealth_increment"])
        if (
            start < int(first_date_idx)
            or end > int(last_date_idx)
            or end <= start
            or not math.isfinite(weight)
        ):
            raise ValueError("weighted interval trade is invalid")
        grouped[start].append(row_idx)
    best = np.full(size, -np.inf, dtype=np.float64)
    predecessor = np.full(size, -1, dtype=np.int32)
    edge_choice = np.full(size, -1, dtype=np.int64)
    best[0] = 0.0
    for date_idx in range(int(first_date_idx), int(last_date_idx) + 1):
        local = date_idx - int(first_date_idx)
        if local > 0 and best[local - 1] > best[local]:
            best[local] = best[local - 1]
            predecessor[local] = local - 1
            edge_choice[local] = -1
        if not math.isfinite(float(best[local])):
            continue
        for row_idx in grouped.get(date_idx, []):
            end = int(trades.iloc[row_idx]["exit_date_idx"])
            end_local = end - int(first_date_idx)
            candidate = best[local] + float(
                trades.iloc[row_idx]["log_wealth_increment"]
            )
            if candidate > best[end_local] + 1.0e-15:
                best[end_local] = candidate
                predecessor[end_local] = local
                edge_choice[end_local] = row_idx
    cursor = size - 1
    chosen: list[int] = []
    while cursor >= 0:
        row_idx = int(edge_choice[cursor])
        previous = int(predecessor[cursor])
        if row_idx >= 0:
            chosen.append(row_idx)
        if previous < 0:
            break
        cursor = previous
    chosen.reverse()
    return float(best[-1]), np.asarray(chosen, dtype=np.int64)


def _global_ceiling(
    *,
    book: OracleSignalBook,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    first_date_idx = int(book.signal_date_idx[0])
    last_date_idx = int(book.signal_date_idx[-1])
    benchmark_ending = float(
        STARTING_CASH_CNY
        * np.prod(1.0 + np.asarray(book.benchmark_returns, dtype=np.float64))
    )
    for horizon, column, peaks in (
        (10, 0, book.peak10),
        (20, 1, book.peak20),
    ):
        for cost_scenario in COST_SCENARIOS:
            edges: list[dict[str, Any]] = []
            for day, signal_date_idx in enumerate(book.signal_date_idx):
                values = np.asarray(
                    book.raw_label_panel[day, :, column],
                    dtype=np.float64,
                )
                exit_dates = np.asarray(peaks[day], dtype=np.int32)
                valid = (
                    np.isfinite(values)
                    & (exit_dates > int(signal_date_idx))
                    & np.asarray(book.next_buyable[day], dtype=bool)
                )
                if not bool(valid.any()):
                    continue
                symbols = np.flatnonzero(valid)
                returns = _approximate_roundtrip_return(
                    mfe=values[valid],
                    sell_dates=np.asarray(
                        book.date_values[exit_dates[valid]],
                        dtype=str,
                    ),
                    study=book.oracle_study,
                    cost_scenario=cost_scenario,
                    costs=book.costs,
                    slippage_multiplier=float(
                        dict(book.study["execution"]["cost_scenarios"])[cost_scenario]
                    ),
                )
                for exit_date_idx in np.unique(exit_dates[valid]):
                    local = np.flatnonzero(exit_dates[valid] == int(exit_date_idx))
                    best_local = int(local[np.argmax(returns[local])])
                    net_return = float(returns[best_local])
                    if net_return <= 0.0:
                        continue
                    symbol_idx = int(symbols[best_local])
                    edges.append(
                        {
                            "horizon": horizon,
                            "cost_scenario": cost_scenario,
                            "signal_date_idx": int(signal_date_idx),
                            "entry_date_idx": int(signal_date_idx) + 1,
                            "exit_date_idx": int(exit_date_idx),
                            "symbol_idx": symbol_idx,
                            "symbol": str(book.symbol_values[symbol_idx]),
                            "gross_mfe": float(values[symbol_idx]),
                            "net_return": net_return,
                            "log_wealth_increment": math.log1p(net_return),
                        }
                    )
            edge_frame = pd.DataFrame.from_records(edges)
            if edge_frame.empty:
                raise ValueError("global ceiling has no positive edges")
            log_wealth, chosen = weighted_interval_ceiling(
                edge_frame,
                first_date_idx=first_date_idx,
                last_date_idx=last_date_idx,
            )
            selected = edge_frame.iloc[chosen].copy()
            selected["signal_date"] = book.date_values[
                selected["signal_date_idx"].to_numpy(dtype=np.int64)
            ]
            selected["entry_date"] = book.date_values[
                selected["entry_date_idx"].to_numpy(dtype=np.int64)
            ]
            selected["exit_date"] = book.date_values[
                selected["exit_date_idx"].to_numpy(dtype=np.int64)
            ]
            selected["sequence"] = np.arange(len(selected), dtype=np.int32)
            selected["cumulative_wealth"] = np.exp(
                np.cumsum(selected["log_wealth_increment"].to_numpy(dtype=np.float64))
            )
            wealth = math.exp(log_wealth)
            result_rows.append(
                {
                    "horizon": horizon,
                    "cost_scenario": cost_scenario,
                    "edge_count": len(edge_frame),
                    "selected_trade_count": len(selected),
                    "cumulative_return": wealth - 1.0,
                    "cagr": wealth ** (252.0 / max(book.day_count, 1)) - 1.0,
                    "benchmark_cumulative_return": (
                        benchmark_ending / STARTING_CASH_CNY - 1.0
                    ),
                    "relative_excess_return": (
                        wealth
                        / max(
                            benchmark_ending / STARTING_CASH_CNY,
                            1.0e-12,
                        )
                        - 1.0
                    ),
                    "mean_trade_net_return": float(selected["net_return"].mean()),
                    "median_trade_net_return": float(selected["net_return"].median()),
                    "relaxed_upper_bound": True,
                    "uses_board_lots": False,
                    "uses_minimum_commission": False,
                }
            )
            trade_frames.append(selected)
    result = pd.DataFrame.from_records(result_rows)
    trades = pd.concat(trade_frames, ignore_index=True)
    root = output_root / "evaluation"
    _write_parquet(root / "global_interval_ceiling.parquet", result)
    _write_parquet(root / "global_interval_trades.parquet", trades)
    return result, trades


def _candidate_diagnostic_summary(
    manifest: Mapping[str, Any],
) -> pd.DataFrame:
    frame = pd.read_parquet(_verify_record(manifest["files"]["candidate_diagnostics"]))
    return (
        frame.groupby(["year", "horizon", "segment"], sort=True)
        .agg(
            trading_day_count=("signal_date", "count"),
            median_daily_candidate_count=("candidate_count", "median"),
            mean_mfe=("mean_mfe", "mean"),
            median_mfe=("median_mfe", "mean"),
            mean_peak_day=("mean_peak_day", "mean"),
            median_peak_day=("median_peak_day", "mean"),
            base_positive_net_fraction=(
                "base_positive_net_fraction",
                "mean",
            ),
            stress_positive_net_fraction=(
                "stress_positive_net_fraction",
                "mean",
            ),
            mean_base_approx_net_return=(
                "mean_base_approx_net_return",
                "mean",
            ),
            mean_stress_approx_net_return=(
                "mean_stress_approx_net_return",
                "mean",
            ),
        )
        .reset_index()
    )


def evaluate(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    record_root: Path = DEFAULT_RECORD_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    manifest = prepare_oracle_book(
        study_path=study_path,
        output_root=output_root,
    )
    study_sha256 = _study_hash(study_path)
    signal_sha256 = str(manifest["files"]["rank_panel"]["sha256"])
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
            f"{len(incomplete)} oracle tasks incomplete; first={incomplete[0]}"
        )
    results = [_task_result(output_root=output_root, spec=spec) for spec in specs]
    metrics = pd.DataFrame([_task_metric_row(result) for result in results])
    relative_errors = _relative_accounting_errors(results)
    metrics["maximum_relative_conservation_error"] = metrics["task_id"].map(
        relative_errors
    )
    annual = _annual_metric_table(results)
    monthly = _monthly_metric_table(results=results)
    configurations = _configuration_table(metrics, study)
    family_stats, family_daily = _family_mode_statistics(
        output_root=output_root,
        configurations=configurations,
        study=study,
    )
    predicted_gaps = _predicted_v4_gaps(
        study=study,
        output_root=output_root,
        oracle_metrics=metrics,
    )
    book = OracleSignalBook(study=study, output_root=output_root)
    global_ceiling, global_trades = _global_ceiling(
        book=book,
        output_root=output_root,
    )
    candidate_summary = _candidate_diagnostic_summary(manifest)
    accounting_invalid = bool(
        not np.isfinite(
            metrics["maximum_relative_conservation_error"].to_numpy(dtype=np.float64)
        ).all()
        or metrics["maximum_relative_conservation_error"].astype(float).max() > 1.0e-9
    )

    def mode_is_strong(mode: str) -> bool:
        return bool(
            family_stats[family_stats["mode"].astype(str).eq(mode)][
                "preregistered_strong_region"
            ]
            .astype(bool)
            .any()
        )

    rerank_strong = mode_is_strong("true_label_daily_rerank")
    peak_strong = mode_is_strong("true_label_peak_close_hold")
    post_peak_strong = mode_is_strong("true_label_post_peak_next_open_hold")
    global_strong = bool(
        global_ceiling.groupby("horizon", sort=True)
        .filter(
            lambda part: bool(
                (part["cumulative_return"].astype(float) > 0.0).all()
                and (part["relative_excess_return"].astype(float) > 0.0).all()
            )
        )["horizon"]
        .nunique()
        > 0
    )
    if accounting_invalid:
        overall = "audit_invalid_due_to_accounting"
    elif post_peak_strong or rerank_strong:
        overall = "mfe_direction_has_strong_executable_true_label_ceiling"
    elif peak_strong:
        overall = (
            "mfe_label_ceiling_exists_but_next_open_execution_breaks_realizability"
        )
    elif global_strong:
        overall = "market_ceiling_exists_but_mfe_label_policy_not_economically_aligned"
    else:
        overall = "tested_market_and_mfe_ceiling_weak_under_frozen_scope"
    strong_pairs = family_stats[
        family_stats["preregistered_strong_region"].astype(bool)
    ].sort_values(["mode", "family", "exposure_mode"])
    decision = {
        "status": "completed_without_oracle_policy_selection",
        "overall": overall,
        "true_label_daily_rerank_strong": rerank_strong,
        "true_peak_close_hold_strong": peak_strong,
        "post_peak_next_open_hold_strong": post_peak_strong,
        "global_single_slot_interval_ceiling_strong": global_strong,
        "accounting_invalid": accounting_invalid,
        "strong_family_mode_exposure_pairs": [
            {
                "mode": str(row["mode"]),
                "family": str(row["family"]),
                "exposure_mode": str(row["exposure_mode"]),
                "economic_region_count": int(row["economic_region_count"]),
                "statistically_confirmed": bool(row["statistically_confirmed"]),
                "hac_p_value": float(row["hac_one_sided_p_value"]),
                "bh_q_value": float(row["bh_q_value"]),
                "bootstrap_lower_90": float(row["bootstrap_lower_90"]),
            }
            for row in strong_pairs.to_dict("records")
        ],
        "single_oracle_winner_selected": False,
        "interpretation": {
            "daily_rerank": (
                "tests whether perfect versions of the current labels can be "
                "monetized by the frozen daily mapping"
            ),
            "peak_close": (
                "label-native hindsight ceiling with the otherwise realistic "
                "finite-capital account"
            ),
            "post_peak_next_open": (
                "tests how much of the label-native ceiling survives a feasible "
                "next-open exit after the true peak is observed"
            ),
            "global_interval": (
                "relaxed single-slot market upper bound; it omits board lots and "
                "minimum commission and is not a deployable strategy"
            ),
        },
        "does_not_validate": (
            "future knowledge is intentionally used, so these results do not "
            "constitute OOS profitability"
        ),
    }
    evaluation_root = output_root / "evaluation"
    paths = {
        "task_metrics": evaluation_root / "task_metrics.parquet",
        "annual_metrics": evaluation_root / "annual_metrics.parquet",
        "monthly_metrics": evaluation_root / "monthly_metrics.parquet",
        "configurations": evaluation_root / "configurations.parquet",
        "family_mode_statistics": (evaluation_root / "family_mode_statistics.parquet"),
        "family_mode_daily": (
            evaluation_root / "family_mode_daily_median_excess.parquet"
        ),
        "predicted_v4_gaps": evaluation_root / "predicted_v4_gaps.parquet",
        "candidate_summary": (evaluation_root / "candidate_diagnostic_summary.parquet"),
        "global_interval_ceiling": (
            evaluation_root / "global_interval_ceiling.parquet"
        ),
        "global_interval_trades": (evaluation_root / "global_interval_trades.parquet"),
    }
    frames = {
        "task_metrics": metrics,
        "annual_metrics": annual,
        "monthly_metrics": monthly,
        "configurations": configurations,
        "family_mode_statistics": family_stats,
        "family_mode_daily": family_daily,
        "predicted_v4_gaps": predicted_gaps,
        "candidate_summary": candidate_summary,
        "global_interval_ceiling": global_ceiling,
        "global_interval_trades": global_trades,
    }
    for name, path in paths.items():
        _write_parquet(path, frames[name])
    decision_path = evaluation_root / "decision.json"
    _write_json(decision_path, decision)
    files = {
        name: _file_record(
            path,
            row_count=len(frames[name]),
            columns=list(frames[name].columns),
        )
        for name, path in paths.items()
    }
    files["decision"] = _file_record(decision_path)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "completed_at": economic._now(),
        "study_id": STUDY_ID,
        "study_config_sha256": study_sha256,
        "oracle_role": (
            "deliberate hindsight economic ceiling, not an OOS strategy result"
        ),
        "task_count": len(metrics),
        "annual_metric_count": len(annual),
        "monthly_metric_count": len(monthly),
        "configuration_count": len(configurations),
        "family_mode_hypothesis_count": len(family_stats),
        "period": {
            "years": list(YEARS),
            "continuous_account": True,
            "annual_reset": False,
            "maximum_consumed_outcome_date": "2025-12-31",
            "forbidden_2026_row_count": 0,
        },
        "oracle_book": {
            "signal_date_count": int(manifest["signal_date_count"]),
            "candidate_row_count": int(manifest["candidate_row_count"]),
            "manifest": _file_record(output_root / "oracle_book/manifest.json"),
        },
        "decision": decision,
        "validation": {
            "task_count_matches_preregistration": len(metrics) == 864,
            "maximum_conservation_error": float(
                metrics["maximum_conservation_error"].max()
            ),
            "maximum_relative_conservation_error": float(
                metrics["maximum_relative_conservation_error"].max()
            ),
            "accounting_invalid": accounting_invalid,
            "no_model_training": True,
            "single_oracle_winner_selected": False,
            "forbidden_2026_row_count": 0,
            "maximum_mfe_reconstruction_error": float(
                manifest["maximum_mfe_reconstruction_error"]
            ),
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
        "true_label_economic_ceiling_completed",
        overall=overall,
        strong_pair_count=len(strong_pairs),
    )
    return record


def self_test() -> dict[str, Any]:
    study = load_study(DEFAULT_STUDY_PATH)
    if len(task_specs()) != 864:
        raise AssertionError("oracle task inventory self-test failed")
    if int(study["period"]["forbidden_year"]) != 2026:
        raise AssertionError("2026 guard self-test failed")
    peaks, derived = derive_peak_close_dates(
        date_idx=10,
        symbols=np.asarray([0, 1], dtype=np.int32),
        horizon=3,
        label_mfe=np.asarray([0.2, 0.1], dtype=np.float64),
        future_close=np.asarray(
            [[0.1, 0.2, 0.2], [0.1, 0.1, 0.05]],
            dtype=np.float64,
        ),
        d1_open=np.asarray([0.0, 0.0], dtype=np.float64),
        exit_sellable=np.ones((20, 2), dtype=bool),
        cutoff_idx=19,
    )
    if peaks.tolist() != [12, 12] or not np.allclose(
        derived,
        np.asarray([0.2, 0.1]),
    ):
        raise AssertionError("first legal peak self-test failed")
    synthetic = pd.DataFrame(
        {
            "slot_count": list(SLOT_COUNTS),
            "economic_pass": [value in (3, 6, 12) for value in SLOT_COUNTS],
        }
    )
    if len(stable_slot_bands(synthetic, flag="economic_pass", width=3)) != 1:
        raise AssertionError("slot-band self-test failed")
    trades = pd.DataFrame(
        [
            {
                "signal_date_idx": 0,
                "exit_date_idx": 2,
                "log_wealth_increment": math.log(1.2),
            },
            {
                "signal_date_idx": 2,
                "exit_date_idx": 4,
                "log_wealth_increment": math.log(1.3),
            },
            {
                "signal_date_idx": 0,
                "exit_date_idx": 4,
                "log_wealth_increment": math.log(1.5),
            },
        ]
    )
    log_wealth, chosen = weighted_interval_ceiling(
        trades,
        first_date_idx=0,
        last_date_idx=4,
    )
    if not math.isclose(
        math.exp(log_wealth),
        1.2 * 1.3,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ) or chosen.tolist() != [0, 1]:
        raise AssertionError("weighted-interval self-test failed")
    return {
        "status": "passed",
        "task_count": 864,
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
        payload = prepare_oracle_book(
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
    print(
        json.dumps(
            economic._json_safe(payload),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
