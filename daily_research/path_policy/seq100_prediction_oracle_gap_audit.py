from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import (
    seq100_true_label_economic_ceiling as oracle,
)
from daily_research.path_policy import (
    seq100_v4_economic_realizability as economic,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_prediction_oracle_gap_audit_v1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_prediction_oracle_gap_audit_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies"
    / "seq100_prediction_oracle_gap_audit_v1"
)
DEFAULT_RECORD_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/research_records/seq100"
    / "seq100_prediction_oracle_gap_audit_v1"
)
BOOK_SCHEMA = "seq100_prediction_oracle_gap_book/v1"
TASK_SCHEMA = "seq100_prediction_oracle_gap_task/v1"
SUMMARY_SCHEMA = "seq100_prediction_oracle_gap_summary/v1"
YEARS = economic.YEARS
EXPOSURE_MODES = economic.EXPOSURE_MODES
SLOT_COUNTS = economic.SLOT_COUNTS
COST_SCENARIOS = economic.COST_SCENARIOS
DAILY_BUFFERS = (0.0, 2.0)
MODES = ("daily_rerank", "true_peak_close", "post_peak_next_open")
STARTING_CASH_CNY = economic.STARTING_CASH_CNY
COMPONENT_KEYS = (
    "mfe10_predicted_d10",
    "mfe10_true_d10",
    "mfe10_predicted_d20",
    "mfe10_true_d20",
    "mfe20_predicted_d20",
    "mfe20_true_d20",
)
DUAL_KEYS = (
    "dual_predicted_predicted",
    "dual_true_predicted",
    "dual_predicted_true",
    "dual_true_true",
)
ORDER_KEYS = (
    "mfe10_predicted",
    "mfe10_true",
    "mfe20_predicted",
    "mfe20_true",
    *DUAL_KEYS,
)


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


def _save_npy(path: Path, values: np.ndarray) -> None:
    economic._save_npy(path, values)


def _emit(event: str, **payload: Any) -> None:
    economic._emit(event, **payload)


def _memory_guard() -> None:
    economic._memory_guard()


def _study_hash(path: Path) -> str:
    return economic._study_hash(path)


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    family: str
    mfe10_source: str
    mfe20_source: str
    state_source: str
    risk_source: str
    role: str


@dataclass(frozen=True)
class GapTaskSpec:
    mode: str
    variant_id: str
    exposure_mode: str
    slot_count: int
    buffer_multiplier: float | None
    cost_scenario: str

    @property
    def task_id(self) -> str:
        buffer = (
            "scheduled"
            if self.buffer_multiplier is None
            else "b" + str(float(self.buffer_multiplier)).replace(".", "p")
        )
        return (
            f"{self.mode}__{self.variant_id}__{self.exposure_mode}"
            f"__k{int(self.slot_count):02d}__{buffer}__{self.cost_scenario}"
        )


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
    variants = [VariantSpec(**dict(item)) for item in payload["variants"]]
    if len(variants) != 17:
        raise ValueError("variant inventory changed")
    if len({item.variant_id for item in variants}) != len(variants):
        raise ValueError("variant IDs are duplicated")
    allowed_sources = {"predicted", "true"}
    for variant in variants:
        if variant.family not in economic.FORMAL_FAMILIES:
            raise ValueError(f"unknown family: {variant.family}")
        if {
            variant.mfe10_source,
            variant.mfe20_source,
            variant.state_source,
            variant.risk_source,
        } - allowed_sources:
            raise ValueError(f"unknown source in {variant.variant_id}")
    account = dict(payload["account"])
    if not math.isclose(
        float(account["starting_cash_cny"]),
        STARTING_CASH_CNY,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("starting cash changed")
    if account["allocation_mode"] != "fixed_initial_notional_per_slot":
        raise ValueError("allocation mode changed")
    if bool(account["profits_reinvested"]):
        raise ValueError("gap audit may not reinvest profits")
    if tuple(int(value) for value in account["slot_counts"]) != SLOT_COUNTS:
        raise ValueError("slot grid changed")
    if tuple(account["exposure_modes"]) != EXPOSURE_MODES:
        raise ValueError("exposure modes changed")
    if (
        tuple(float(value) for value in account["daily_rerank_buffer_multipliers"])
        != DAILY_BUFFERS
    ):
        raise ValueError("buffer grid changed")
    if tuple(account["cost_scenarios"]) != COST_SCENARIOS:
        raise ValueError("cost scenarios changed")
    if int(payload["modes"]["expected_task_count"]) != 1200:
        raise ValueError("task count changed")
    return payload


def variant_specs(
    study: Mapping[str, Any] | None = None,
) -> dict[str, VariantSpec]:
    payload = load_study() if study is None else study
    return {
        item.variant_id: item
        for item in (VariantSpec(**dict(record)) for record in payload["variants"])
    }


def task_specs(
    study: Mapping[str, Any] | None = None,
) -> list[GapTaskSpec]:
    payload = load_study() if study is None else study
    variants = variant_specs(payload)
    scheduled = tuple(payload["modes"]["scheduled_exit"]["variant_ids"])
    output: list[GapTaskSpec] = []
    for mode in MODES:
        ids = tuple(variants) if mode == "daily_rerank" else scheduled
        buffers: tuple[float | None, ...] = (
            tuple(DAILY_BUFFERS) if mode == "daily_rerank" else (None,)
        )
        output.extend(
            GapTaskSpec(
                mode=mode,
                variant_id=variant_id,
                exposure_mode=exposure,
                slot_count=slots,
                buffer_multiplier=buffer,
                cost_scenario=cost,
            )
            for variant_id in ids
            for exposure in EXPOSURE_MODES
            for slots in SLOT_COUNTS
            for buffer in buffers
            for cost in COST_SCENARIOS
        )
    if len(output) != 1200 or len({item.task_id for item in output}) != len(output):
        raise AssertionError("gap task inventory changed")
    return output


def _prepared_complete(
    output_root: Path,
    *,
    study_sha256: str | None = None,
) -> bool:
    path = output_root / "gap_book/manifest.json"
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


def _finite_correlation(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    if int(valid.sum()) < 3:
        return math.nan
    first = first[valid]
    second = second[valid]
    if float(np.std(first)) <= 0.0 or float(np.std(second)) <= 0.0:
        return math.nan
    return float(np.corrcoef(first, second)[0, 1])


def _rank_on_support(
    source: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    symbols = np.flatnonzero(np.asarray(support, dtype=bool)).astype(np.int32)
    output = np.full(len(source), np.nan, dtype=np.float32)
    if len(symbols):
        output[symbols] = economic._rank01(
            np.asarray(source[symbols], dtype=np.float64)
        )
    return output, symbols


def _set_order(
    *,
    orders: np.ndarray,
    counts: np.ndarray,
    order_index: int,
    day: int,
    ranks: np.ndarray,
    symbols: np.ndarray,
) -> None:
    if len(symbols) == 0:
        return
    order = np.lexsort(
        (
            np.asarray(symbols, dtype=np.int32),
            -np.asarray(ranks[symbols], dtype=np.float64),
        )
    )
    ranked = np.asarray(symbols[order], dtype=np.int32)
    orders[int(order_index), int(day), : len(ranked)] = ranked
    counts[int(order_index), int(day)] = len(ranked)


def _top_metrics(
    *,
    predicted_rank: np.ndarray,
    true_rank: np.ndarray,
    true_mfe: np.ndarray,
    peak_day: np.ndarray,
    fraction: float,
) -> dict[str, Any]:
    predicted = np.asarray(predicted_rank, dtype=np.float64)
    truth = np.asarray(true_rank, dtype=np.float64)
    mfe = np.asarray(true_mfe, dtype=np.float64)
    peak = np.asarray(peak_day, dtype=np.float64)
    valid = (
        np.isfinite(predicted)
        & np.isfinite(truth)
        & np.isfinite(mfe)
        & np.isfinite(peak)
    )
    predicted_top = valid & (predicted >= 1.0 - float(fraction) - 1.0e-12)
    true_top = valid & (truth >= 1.0 - float(fraction) - 1.0e-12)
    intersection = predicted_top & true_top
    union = predicted_top | true_top
    predicted_count = int(predicted_top.sum())
    true_count = int(true_top.sum())
    intersection_count = int(intersection.sum())
    predicted_mean = float(np.mean(mfe[predicted_top])) if predicted_count else math.nan
    oracle_mean = float(np.mean(mfe[true_top])) if true_count else math.nan
    return {
        "fraction": float(fraction),
        "predicted_count": predicted_count,
        "true_count": true_count,
        "intersection_count": intersection_count,
        "overlap_over_smaller": (
            intersection_count / max(min(predicted_count, true_count), 1)
        ),
        "jaccard": intersection_count / max(int(union.sum()), 1),
        "tail_hit_rate": intersection_count / max(predicted_count, 1),
        "predicted_top_true_mfe": predicted_mean,
        "oracle_top_true_mfe": oracle_mean,
        "opportunity_capture_ratio": (
            predicted_mean / oracle_mean
            if math.isfinite(predicted_mean)
            and math.isfinite(oracle_mean)
            and oracle_mean > 0.0
            else math.nan
        ),
        "predicted_top_peak_day": (
            float(np.mean(peak[predicted_top])) if predicted_count else math.nan
        ),
        "oracle_top_peak_day": (
            float(np.mean(peak[true_top])) if true_count else math.nan
        ),
    }


def prepare_gap_book(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    study_sha256 = _study_hash(study_path)
    if _prepared_complete(output_root, study_sha256=study_sha256):
        return _load_json(output_root / "gap_book/manifest.json")
    _memory_guard()
    sources = dict(study["sources"])
    v4_path = _resolve(sources["v4_signal_book_manifest"])
    oracle_path = _resolve(sources["oracle_book_manifest"])
    v4 = _load_json(v4_path)
    truth_manifest = _load_json(oracle_path)
    if v4.get("schema") != economic.SIGNAL_SCHEMA:
        raise ValueError("v4 signal book schema changed")
    if truth_manifest.get("schema") != oracle.BOOK_SCHEMA:
        raise ValueError("oracle book schema changed")
    predicted = np.load(
        _verify_record(v4["files"]["rank_panel"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    truth = np.load(
        _verify_record(truth_manifest["files"]["rank_panel"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    raw_truth = np.load(
        _verify_record(truth_manifest["files"]["raw_label_panel"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    peak10 = np.load(
        _verify_record(truth_manifest["files"]["peak_close_date_idx_10"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    peak20 = np.load(
        _verify_record(truth_manifest["files"]["peak_close_date_idx_20"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    signal_dates = np.load(
        _verify_record(v4["files"]["signal_date_idx"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    truth_dates = np.load(
        _verify_record(truth_manifest["files"]["signal_date_idx"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    next_buyable = np.load(
        _verify_record(v4["files"]["next_open_buyable"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    if not np.array_equal(signal_dates, truth_dates):
        raise ValueError("v4 and oracle calendars differ")
    if predicted.shape != truth.shape or predicted.shape[2] != len(
        economic.RANK_COLUMNS
    ):
        raise ValueError("rank panel shape changed")
    day_count, symbol_count, _ = predicted.shape
    component_ranks = np.full(
        (len(COMPONENT_KEYS), day_count, symbol_count),
        np.nan,
        dtype=np.float32,
    )
    dual_ranks = np.full(
        (len(DUAL_KEYS), day_count, symbol_count),
        np.nan,
        dtype=np.float32,
    )
    orders = np.full(
        (len(ORDER_KEYS), day_count, symbol_count),
        -1,
        dtype=np.int32,
    )
    counts = np.zeros((len(ORDER_KEYS), day_count), dtype=np.int32)
    support_counts = np.zeros((2, day_count), dtype=np.int32)
    component_index = {name: idx for idx, name in enumerate(COMPONENT_KEYS)}
    dual_index = {name: idx for idx, name in enumerate(DUAL_KEYS)}
    order_index = {name: idx for idx, name in enumerate(ORDER_KEYS)}
    oracle_study_path = _resolve(sources["oracle_study"])
    oracle_study = oracle.load_study(oracle_study_path)
    oracle_root = oracle_path.parents[1]
    base_book = oracle.OracleSignalBook(
        study=oracle_study,
        output_root=oracle_root,
    )
    date_values = np.asarray(base_book.date_values, dtype=str)
    top_fractions = tuple(
        float(value) for value in study["candidate_diagnostics"]["top_fractions"]
    )
    diagnostic_rows: list[dict[str, Any]] = []
    capacity_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    last_emit = started
    for day, raw_date_idx in enumerate(signal_dates):
        _memory_guard()
        date_idx = int(raw_date_idx)
        buyable = np.asarray(next_buyable[day], dtype=bool)
        support10 = (
            buyable
            & np.isfinite(predicted[day, :, 0])
            & np.isfinite(truth[day, :, 0])
            & np.isfinite(raw_truth[day, :, 0])
            & (np.asarray(peak10[day], dtype=np.int32) > date_idx)
        )
        support20 = (
            buyable
            & np.isfinite(predicted[day, :, 0])
            & np.isfinite(predicted[day, :, 1])
            & np.isfinite(predicted[day, :, 2])
            & np.isfinite(predicted[day, :, 4])
            & np.isfinite(predicted[day, :, 5])
            & np.isfinite(predicted[day, :, 6])
            & np.isfinite(truth[day, :, 0])
            & np.isfinite(truth[day, :, 1])
            & np.isfinite(truth[day, :, 2])
            & np.isfinite(truth[day, :, 4])
            & np.isfinite(truth[day, :, 5])
            & np.isfinite(truth[day, :, 6])
            & np.isfinite(raw_truth[day, :, 0])
            & np.isfinite(raw_truth[day, :, 1])
            & np.isfinite(raw_truth[day, :, 2])
            & np.isfinite(raw_truth[day, :, 3])
            & (np.asarray(peak20[day], dtype=np.int32) > date_idx)
        )
        support_counts[:, day] = (int(support10.sum()), int(support20.sum()))
        ranked: dict[str, np.ndarray] = {}
        for key, source, support in (
            ("mfe10_predicted_d10", predicted[day, :, 0], support10),
            ("mfe10_true_d10", truth[day, :, 0], support10),
            ("mfe10_predicted_d20", predicted[day, :, 0], support20),
            ("mfe10_true_d20", truth[day, :, 0], support20),
            ("mfe20_predicted_d20", predicted[day, :, 1], support20),
            ("mfe20_true_d20", truth[day, :, 1], support20),
        ):
            current, _ = _rank_on_support(source, support)
            ranked[key] = current
            component_ranks[component_index[key], day] = current
        symbols10 = np.flatnonzero(support10).astype(np.int32)
        symbols20 = np.flatnonzero(support20).astype(np.int32)
        for key in ("mfe10_predicted", "mfe10_true"):
            source_key = f"{key}_d10"
            _set_order(
                orders=orders,
                counts=counts,
                order_index=order_index[key],
                day=day,
                ranks=ranked[source_key],
                symbols=symbols10,
            )
        for key in ("mfe20_predicted", "mfe20_true"):
            source_key = f"{key}_d20"
            _set_order(
                orders=orders,
                counts=counts,
                order_index=order_index[key],
                day=day,
                ranks=ranked[source_key],
                symbols=symbols20,
            )
        dual_sources = (
            (
                "dual_predicted_predicted",
                "mfe10_predicted_d20",
                "mfe20_predicted_d20",
            ),
            (
                "dual_true_predicted",
                "mfe10_true_d20",
                "mfe20_predicted_d20",
            ),
            (
                "dual_predicted_true",
                "mfe10_predicted_d20",
                "mfe20_true_d20",
            ),
            (
                "dual_true_true",
                "mfe10_true_d20",
                "mfe20_true_d20",
            ),
        )
        for key, left_key, right_key in dual_sources:
            current = np.full(symbol_count, np.nan, dtype=np.float32)
            if len(symbols20):
                minimum = np.minimum(
                    ranked[left_key][symbols20],
                    ranked[right_key][symbols20],
                )
                current[symbols20] = economic._rank01(minimum)
            dual_ranks[dual_index[key], day] = current
            _set_order(
                orders=orders,
                counts=counts,
                order_index=order_index[key],
                day=day,
                ranks=current,
                symbols=symbols20,
            )
        trade_date = str(date_values[date_idx])
        for horizon, support, pred_key, true_key, raw_column, peak_source in (
            (
                10,
                support10,
                "mfe10_predicted_d10",
                "mfe10_true_d10",
                0,
                peak10,
            ),
            (
                20,
                support20,
                "mfe20_predicted_d20",
                "mfe20_true_d20",
                1,
                peak20,
            ),
        ):
            peak_day = np.asarray(peak_source[day], dtype=np.float64) - date_idx
            true_top5 = support & (ranked[true_key] >= 0.95 - 1.0e-12)
            risk_column = 5 if horizon == 10 else 6
            predicted_state_score = np.asarray(
                predicted[day, :, 4], dtype=np.float64
            ) - np.asarray(predicted[day, :, 2], dtype=np.float64)
            true_state_score = np.asarray(
                truth[day, :, 4], dtype=np.float64
            ) - np.asarray(truth[day, :, 2], dtype=np.float64)
            common = {
                "signal_date": trade_date,
                "year": int(trade_date[:4]),
                "horizon": horizon,
                "common_support_count": int(support.sum()),
                "daily_rank_ic": _finite_correlation(
                    ranked[pred_key][support],
                    ranked[true_key][support],
                ),
                "risk_rank_ic": _finite_correlation(
                    predicted[day, support, risk_column],
                    truth[day, support, risk_column],
                ),
                "risk_rank_ic_inside_true_mfe_top5": _finite_correlation(
                    predicted[day, true_top5, risk_column],
                    truth[day, true_top5, risk_column],
                ),
                "state_ordinal_ic": _finite_correlation(
                    predicted_state_score[support],
                    true_state_score[support],
                ),
                "state_ordinal_ic_inside_true_mfe_top5": _finite_correlation(
                    predicted_state_score[true_top5],
                    true_state_score[true_top5],
                ),
            }
            for fraction in top_fractions:
                diagnostic_rows.append(
                    {
                        **common,
                        **_top_metrics(
                            predicted_rank=ranked[pred_key],
                            true_rank=ranked[true_key],
                            true_mfe=raw_truth[day, :, raw_column],
                            peak_day=peak_day,
                            fraction=fraction,
                        ),
                    }
                )
            window_start = max(0, date_idx - 19)
            amount = np.asarray(
                base_book.amount[window_start : date_idx + 1],
                dtype=np.float64,
            )
            amount[~np.isfinite(amount) | (amount <= 0.0)] = np.nan
            with np.errstate(all="ignore"):
                trailing_amount = np.nanmedian(amount, axis=0)
            for source_name, source_rank in (
                ("predicted", ranked[pred_key]),
                ("true", ranked[true_key]),
            ):
                selected = support & (source_rank >= 0.95 - 1.0e-12)
                for slots in SLOT_COUNTS:
                    budget = STARTING_CASH_CNY / int(slots)
                    participation = np.divide(
                        budget,
                        trailing_amount[selected],
                        out=np.full(int(selected.sum()), np.inf),
                        where=np.isfinite(trailing_amount[selected])
                        & (trailing_amount[selected] > 0.0),
                    )
                    for threshold in study["account"][
                        "participation_diagnostic_thresholds"
                    ]:
                        capacity_rows.append(
                            {
                                "signal_date": trade_date,
                                "year": int(trade_date[:4]),
                                "horizon": horizon,
                                "selector_source": source_name,
                                "slot_count": int(slots),
                                "participation_threshold": float(threshold),
                                "selected_count": len(participation),
                                "eligible_fraction": float(
                                    np.mean(participation <= float(threshold))
                                )
                                if len(participation)
                                else math.nan,
                                "median_participation": float(np.median(participation))
                                if len(participation)
                                else math.nan,
                            }
                        )
        now = time.monotonic()
        if now - last_emit >= 30.0:
            _emit(
                "gap_book_progress",
                signal_date=trade_date,
                elapsed_seconds=round(now - started, 2),
            )
            last_emit = now
    book_root = output_root / "gap_book"
    arrays = {
        "component_rank_panel": component_ranks,
        "dual_rank_panel": dual_ranks,
        "candidate_orders": orders,
        "candidate_order_counts": counts,
        "support_counts": support_counts,
    }
    files: dict[str, dict[str, Any]] = {}
    for name, values in arrays.items():
        path = book_root / f"{name}.npy"
        _save_npy(path, values)
        files[name] = _file_record(
            path,
            shape=list(values.shape),
            dtype=str(values.dtype),
        )
    diagnostics = pd.DataFrame.from_records(diagnostic_rows)
    capacity = pd.DataFrame.from_records(capacity_rows)
    diagnostics_path = book_root / "candidate_diagnostics.parquet"
    capacity_path = book_root / "candidate_capacity.parquet"
    _write_parquet(diagnostics_path, diagnostics)
    _write_parquet(capacity_path, capacity)
    files["candidate_diagnostics"] = _file_record(
        diagnostics_path,
        row_count=len(diagnostics),
        columns=list(diagnostics.columns),
    )
    files["candidate_capacity"] = _file_record(
        capacity_path,
        row_count=len(capacity),
        columns=list(capacity.columns),
    )
    signal_identity = economic._sha256(book_root / "candidate_orders.npy")
    manifest = {
        "schema": BOOK_SCHEMA,
        "status": "completed",
        "completed_at": economic._now(),
        "study_id": STUDY_ID,
        "study_config_sha256": study_sha256,
        "signal_identity_sha256": signal_identity,
        "period": {
            "first_signal_date": str(date_values[int(signal_dates[0])]),
            "last_mark_date": str(date_values[int(signal_dates[-1])]),
            "maximum_outcome_date_read": "2025-12-31",
            "forbidden_2026_row_count": 0,
        },
        "day_count": int(day_count),
        "symbol_count": int(symbol_count),
        "component_keys": list(COMPONENT_KEYS),
        "dual_keys": list(DUAL_KEYS),
        "order_keys": list(ORDER_KEYS),
        "support": {
            "D10_minimum": int(np.min(support_counts[0])),
            "D10_median": float(np.median(support_counts[0])),
            "D10_maximum": int(np.max(support_counts[0])),
            "D20_minimum": int(np.min(support_counts[1])),
            "D20_median": float(np.median(support_counts[1])),
            "D20_maximum": int(np.max(support_counts[1])),
        },
        "allocation_semantics": {
            "mode": "fixed_initial_notional_per_slot",
            "profits_reinvested": False,
        },
        "source_hashes": {
            "v4_signal_book_manifest": economic._sha256(v4_path),
            "oracle_book_manifest": economic._sha256(oracle_path),
        },
        "source_files": {
            "v4_rank_panel": dict(v4["files"]["rank_panel"]),
            "oracle_rank_panel": dict(truth_manifest["files"]["rank_panel"]),
            "oracle_raw_label_panel": dict(truth_manifest["files"]["raw_label_panel"]),
        },
        "files": files,
    }
    _write_json(book_root / "manifest.json", manifest)
    _emit(
        "gap_book_completed",
        day_count=day_count,
        diagnostic_rows=len(diagnostics),
        capacity_rows=len(capacity),
    )
    return manifest


class MixedSignalBook:
    def __init__(
        self,
        *,
        study: Mapping[str, Any],
        output_root: Path,
        variant: VariantSpec,
        slot_count: int,
    ) -> None:
        if not _prepared_complete(output_root):
            raise FileNotFoundError("gap book is incomplete")
        self.gap_study = dict(study)
        self.variant = variant
        self.fixed_slot_count = int(slot_count)
        sources = dict(study["sources"])
        oracle_study_path = _resolve(sources["oracle_study"])
        oracle_study = oracle.load_study(oracle_study_path)
        oracle_root = _resolve(sources["oracle_book_manifest"]).parents[1]
        self.base = oracle.OracleSignalBook(
            study=oracle_study,
            output_root=oracle_root,
        )
        self.study = self.base.study
        prepared = _load_json(output_root / "gap_book/manifest.json")
        self.prepared_manifest = prepared
        self.manifest = {
            "files": {
                "rank_panel": {
                    "sha256": prepared["signal_identity_sha256"],
                }
            }
        }
        files = dict(prepared["files"])
        self.component_ranks = np.load(
            _verify_record(files["component_rank_panel"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.dual_ranks = np.load(
            _verify_record(files["dual_rank_panel"]),
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
        v4 = _load_json(_resolve(sources["v4_signal_book_manifest"]))
        truth = _load_json(_resolve(sources["oracle_book_manifest"]))
        self.predicted_rank_panel = np.load(
            _verify_record(v4["files"]["rank_panel"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.true_rank_panel = np.load(
            _verify_record(truth["files"]["rank_panel"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.component_index = {
            name: idx for idx, name in enumerate(prepared["component_keys"])
        }
        self.dual_index = {name: idx for idx, name in enumerate(prepared["dual_keys"])}
        self.order_index = {
            name: idx for idx, name in enumerate(prepared["order_keys"])
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def _component_key(self, column: int) -> str:
        family = self.variant.family
        suffix = "d10" if family == "mfe10_primary" else "d20"
        if int(column) == 0:
            return f"mfe10_{self.variant.mfe10_source}_{suffix}"
        if int(column) == 1:
            return f"mfe20_{self.variant.mfe20_source}_d20"
        raise ValueError("component key requested for another column")

    def _dual_key(self) -> str:
        return f"dual_{self.variant.mfe10_source}_{self.variant.mfe20_source}"

    def _order_key(self) -> str:
        if self.variant.family == "mfe10_primary":
            return f"mfe10_{self.variant.mfe10_source}"
        if self.variant.family == "mfe20_primary":
            return f"mfe20_{self.variant.mfe20_source}"
        return self._dual_key()

    def selector_column(self, family: str) -> int:
        return economic.SignalBook.selector_column(self, family)

    def symbols_for_day(self, family: str, day: int) -> np.ndarray:
        if family != self.variant.family:
            raise ValueError("task family and mixed-book variant differ")
        index = self.order_index[self._order_key()]
        count = int(self.order_counts[index, int(day)])
        return np.asarray(
            self.orders[index, int(day), :count],
            dtype=np.int32,
        )

    def rank_values(
        self,
        *,
        day: int,
        symbols: np.ndarray,
        column: int,
    ) -> np.ndarray:
        symbol_idx = np.asarray(symbols, dtype=np.int32)
        if int(column) in (0, 1):
            key = self._component_key(int(column))
            return np.asarray(
                self.component_ranks[
                    self.component_index[key],
                    int(day),
                    symbol_idx,
                ],
                dtype=np.float64,
            )
        if int(column) == 7:
            return np.asarray(
                self.dual_ranks[
                    self.dual_index[self._dual_key()],
                    int(day),
                    symbol_idx,
                ],
                dtype=np.float64,
            )
        if int(column) in (2, 3, 4):
            source = (
                self.true_rank_panel
                if self.variant.state_source == "true"
                else self.predicted_rank_panel
            )
            return np.asarray(
                source[int(day), symbol_idx, int(column)],
                dtype=np.float64,
            )
        if int(column) in (5, 6):
            source = (
                self.true_rank_panel
                if self.variant.risk_source == "true"
                else self.predicted_rank_panel
            )
            return np.asarray(
                source[int(day), symbol_idx, int(column)],
                dtype=np.float64,
            )
        raise ValueError(f"unsupported mixed rank column: {column}")

    def rank(self, day: int, symbol_idx: int, column: int) -> float:
        return float(
            self.rank_values(
                day=int(day),
                symbols=np.asarray([symbol_idx], dtype=np.int32),
                column=int(column),
            )[0]
        )

    def candidate_count(self, day: int) -> int:
        index = self.order_index[self._order_key()]
        return int(self.order_counts[index, int(day)])

    def position_allocation(
        self,
        *,
        spec: economic.TaskSpec,
        cash: float,
        equity_open: float,
    ) -> float:
        del equity_open
        if int(spec.slot_count) != self.fixed_slot_count:
            raise ValueError("fixed-notional slot count changed")
        return min(
            float(cash),
            STARTING_CASH_CNY / int(spec.slot_count),
        )


def _internal_spec(
    *,
    task: GapTaskSpec,
    variant: VariantSpec,
) -> economic.TaskSpec:
    return economic.TaskSpec(
        family=variant.family,
        exposure_mode=task.exposure_mode,
        slot_count=int(task.slot_count),
        buffer_multiplier=float(task.buffer_multiplier or 0.0),
        cost_scenario=task.cost_scenario,
    )


def _rewrite_result(
    *,
    task: GapTaskSpec,
    variant: VariantSpec,
    result: dict[str, Any],
    equity: pd.DataFrame,
    trades: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    task_id = task.task_id
    result = {
        **result,
        "schema": TASK_SCHEMA,
        "study_id": STUDY_ID,
        "task": asdict(task),
        "task_id": task_id,
        "variant": asdict(variant),
        "gap_semantics": {
            "fixed_initial_notional_per_slot": True,
            "profits_reinvested": False,
            "candidate_aligned_common_support": True,
            "future_label_sources": [
                name
                for name, source in (
                    ("mfe10", variant.mfe10_source),
                    ("mfe20", variant.mfe20_source),
                    ("state", variant.state_source),
                    ("risk", variant.risk_source),
                )
                if source == "true"
            ],
            "exit_mode": task.mode,
            "not_oos_when_any_true_source_or_scheduled_exit": bool(
                task.mode != "daily_rerank"
                or "true"
                in {
                    variant.mfe10_source,
                    variant.mfe20_source,
                    variant.state_source,
                    variant.risk_source,
                }
            ),
        },
    }
    output_frames: list[pd.DataFrame] = []
    for frame in (equity, trades):
        current = frame.copy()
        if "task_id" in current:
            current["task_id"] = task_id
        output_frames.append(current)
    return result, output_frames[0], output_frames[1]


def simulate_task(
    *,
    study: Mapping[str, Any],
    output_root: Path,
    task: GapTaskSpec,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    variant = variant_specs(study)[task.variant_id]
    book = MixedSignalBook(
        study=study,
        output_root=output_root,
        variant=variant,
        slot_count=task.slot_count,
    )
    internal = _internal_spec(task=task, variant=variant)
    if task.mode == "daily_rerank":
        result, equity, trades, monthly = economic.simulate_task(
            book=book,
            spec=internal,
        )
    else:
        oracle_mode = (
            "true_label_peak_close_hold"
            if task.mode == "true_peak_close"
            else "true_label_post_peak_next_open_hold"
        )
        oracle_task = oracle.OracleTaskSpec(
            mode=oracle_mode,
            family=variant.family,
            exposure_mode=task.exposure_mode,
            slot_count=int(task.slot_count),
            buffer_multiplier=None,
            cost_scenario=task.cost_scenario,
        )
        result, equity, trades, monthly = oracle.simulate_scheduled_task(
            book=book,
            spec=oracle_task,
        )
    result, equity, trades = _rewrite_result(
        task=task,
        variant=variant,
        result=result,
        equity=equity,
        trades=trades,
    )
    return result, equity, trades, monthly


def _task_dir(output_root: Path, task: GapTaskSpec) -> Path:
    return output_root / "tasks" / task.task_id


def _task_complete(
    *,
    output_root: Path,
    task: GapTaskSpec,
    study_sha256: str,
    signal_sha256: str,
) -> bool:
    path = _task_dir(output_root, task) / "task_result.json"
    if not path.is_file():
        return False
    try:
        result = _load_json(path)
        if (
            result.get("schema") != TASK_SCHEMA
            or result.get("status") != "completed"
            or result.get("task_id") != task.task_id
            or result.get("study_config_sha256") != study_sha256
            or result.get("signal_book_sha256") != signal_sha256
            or dict(result.get("task", {})) != asdict(task)
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
    task = GapTaskSpec(**dict(result["task"]))
    directory = _task_dir(output_root, task)
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
    manifest = prepare_gap_book(
        study_path=study_path,
        output_root=output_root,
    )
    study_sha256 = _study_hash(study_path)
    signal_sha256 = str(manifest["signal_identity_sha256"])
    tasks = task_specs(study)
    complete_before = sum(
        _task_complete(
            output_root=output_root,
            task=task,
            study_sha256=study_sha256,
            signal_sha256=signal_sha256,
        )
        for task in tasks
    )
    _emit(
        "gap_tasks_starting",
        completed=complete_before,
        total=len(tasks),
    )
    completed = complete_before
    started = time.monotonic()
    for task in tasks:
        if _task_complete(
            output_root=output_root,
            task=task,
            study_sha256=study_sha256,
            signal_sha256=signal_sha256,
        ):
            continue
        result, equity, trades, monthly = simulate_task(
            study=study,
            output_root=output_root,
            task=task,
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
        if completed % 10 == 0 or completed == len(tasks):
            _emit(
                "gap_task_progress",
                completed=completed,
                total=len(tasks),
                task_id=task.task_id,
                elapsed_seconds=round(time.monotonic() - started, 2),
            )
    return {
        "status": "completed",
        "completed": completed,
        "total": len(tasks),
    }


def status(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    study_sha256 = _study_hash(study_path)
    ready = _prepared_complete(output_root, study_sha256=study_sha256)
    signal_sha256 = ""
    if ready:
        signal_sha256 = str(
            _load_json(output_root / "gap_book/manifest.json")["signal_identity_sha256"]
        )
    tasks = task_specs(study)
    completed = (
        sum(
            _task_complete(
                output_root=output_root,
                task=task,
                study_sha256=study_sha256,
                signal_sha256=signal_sha256,
            )
            for task in tasks
        )
        if ready
        else 0
    )
    return {
        "study_id": STUDY_ID,
        "gap_book_complete": bool(ready),
        "completed_tasks": completed,
        "pending_tasks": len(tasks) - completed,
        "total_tasks": len(tasks),
        "evaluation_complete": (output_root / "evaluation/summary.json").is_file(),
    }


def _task_result(
    *,
    output_root: Path,
    task: GapTaskSpec,
) -> dict[str, Any]:
    path = _task_dir(output_root, task) / "task_result.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _load_json(path)


def _task_metric_row(result: Mapping[str, Any]) -> dict[str, Any]:
    row = economic._task_metric_row(result)
    variant = dict(result["variant"])
    return {
        **row,
        "net_terminal_pnl_cny": float(row["net_cumulative_return"]) * STARTING_CASH_CNY,
        "terminal_cost_accrued_pnl_cny": float(row["terminal_cost_accrued_return"])
        * STARTING_CASH_CNY,
        "family": str(variant["family"]),
        "mfe10_source": str(variant["mfe10_source"]),
        "mfe20_source": str(variant["mfe20_source"]),
        "state_source": str(variant["state_source"]),
        "risk_source": str(variant["risk_source"]),
        "variant_role": str(variant["role"]),
    }


def _relative_accounting_errors(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    return oracle._relative_accounting_errors(results)


def _add_period_cash_metrics(
    frame: pd.DataFrame,
    *,
    period_column: str,
) -> pd.DataFrame:
    required = {"task_id", period_column, "ending_equity"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"period metric columns missing: {sorted(missing)}")
    ordered = frame.sort_values(["task_id", period_column]).copy()
    prior = ordered.groupby("task_id", sort=False)["ending_equity"].shift(1)
    ordered["starting_equity"] = prior.fillna(STARTING_CASH_CNY)
    ordered["period_pnl_cny"] = ordered["ending_equity"].astype(float) - ordered[
        "starting_equity"
    ].astype(float)
    ordered["period_pnl_on_initial_cash"] = (
        ordered["period_pnl_cny"] / STARTING_CASH_CNY
    )
    ordered["cumulative_pnl_cny"] = (
        ordered["ending_equity"].astype(float) - STARTING_CASH_CNY
    )
    return ordered.sort_index()


def _candidate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "common_support_count",
        "daily_rank_ic",
        "risk_rank_ic",
        "risk_rank_ic_inside_true_mfe_top5",
        "state_ordinal_ic",
        "state_ordinal_ic_inside_true_mfe_top5",
        "predicted_count",
        "true_count",
        "intersection_count",
        "overlap_over_smaller",
        "jaccard",
        "tail_hit_rate",
        "predicted_top_true_mfe",
        "oracle_top_true_mfe",
        "opportunity_capture_ratio",
        "predicted_top_peak_day",
        "oracle_top_peak_day",
    ]
    return (
        frame.groupby(["year", "horizon", "fraction"], sort=True)[numeric]
        .mean()
        .reset_index()
    )


def _capacity_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            [
                "year",
                "horizon",
                "selector_source",
                "slot_count",
                "participation_threshold",
            ],
            sort=True,
        )
        .agg(
            trading_day_count=("signal_date", "count"),
            mean_eligible_fraction=("eligible_fraction", "mean"),
            median_eligible_fraction=("eligible_fraction", "median"),
            mean_median_participation=("median_participation", "mean"),
        )
        .reset_index()
    )


def _metric_map(metrics: pd.DataFrame) -> dict[tuple[Any, ...], dict[str, Any]]:
    output: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in metrics.to_dict("records"):
        buffer = row["buffer_multiplier"]
        normalized_buffer = (
            None
            if buffer is None or not math.isfinite(float(buffer))
            else float(buffer)
        )
        key = (
            str(row["mode"]),
            str(row["variant_id"]),
            str(row["exposure_mode"]),
            int(row["slot_count"]),
            normalized_buffer,
            str(row["cost_scenario"]),
        )
        output[key] = dict(row)
    return output


def _append_effect(
    rows: list[dict[str, Any]],
    *,
    effect: str,
    layer: str,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> None:
    if (
        left["exposure_mode"] != right["exposure_mode"]
        or int(left["slot_count"]) != int(right["slot_count"])
        or left["cost_scenario"] != right["cost_scenario"]
    ):
        raise ValueError("effect comparison is not matched")
    rows.append(
        {
            "effect": effect,
            "layer": layer,
            "left_task_id": str(left["task_id"]),
            "right_task_id": str(right["task_id"]),
            "left_mode": str(left["mode"]),
            "right_mode": str(right["mode"]),
            "left_variant": str(left["variant_id"]),
            "right_variant": str(right["variant_id"]),
            "exposure_mode": str(left["exposure_mode"]),
            "slot_count": int(left["slot_count"]),
            "buffer_multiplier": left["buffer_multiplier"],
            "cost_scenario": str(left["cost_scenario"]),
            "terminal_return_delta": float(
                left["terminal_cost_accrued_return"]
                - right["terminal_cost_accrued_return"]
            ),
            "terminal_pnl_delta_cny": float(
                (
                    left["terminal_cost_accrued_return"]
                    - right["terminal_cost_accrued_return"]
                )
                * STARTING_CASH_CNY
            ),
            "cagr_delta": float(left["net_cagr"] - right["net_cagr"]),
            "maximum_drawdown_delta": float(
                left["maximum_drawdown"] - right["maximum_drawdown"]
            ),
            "turnover_delta": float(
                left["turnover_to_starting_cash"] - right["turnover_to_starting_cash"]
            ),
            "total_cost_delta": float(left["total_cost"] - right["total_cost"]),
            **{
                f"net_return_delta_{year}": float(
                    left[f"net_return_{year}"] - right[f"net_return_{year}"]
                )
                for year in YEARS
            },
        }
    )


def _effect_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    lookup = _metric_map(metrics)
    rows: list[dict[str, Any]] = []

    def get(
        mode: str,
        variant: str,
        exposure: str,
        slots: int,
        buffer: float | None,
        cost: str,
    ) -> dict[str, Any]:
        return lookup[(mode, variant, exposure, slots, buffer, cost)]

    daily_pairs = (
        (
            "D10_MFE_prediction",
            "MFE_prediction",
            "mfe10_true",
            "mfe10_predicted",
        ),
        (
            "D20_MFE_prediction",
            "MFE_prediction",
            "mfe20_true",
            "mfe20_predicted",
        ),
        (
            "dual_D10_prediction",
            "MFE_prediction",
            "dual_true_mfe10_only",
            "dual_predicted",
        ),
        (
            "dual_D20_prediction",
            "MFE_prediction",
            "dual_true_mfe20_only",
            "dual_predicted",
        ),
        (
            "dual_both_MFE_prediction",
            "MFE_prediction",
            "dual_true_mfe",
            "dual_predicted",
        ),
        (
            "risk_prediction_conditional_true_MFE",
            "risk_prediction",
            "dual_true_mfe_true_risk",
            "dual_true_mfe_predicted_risk",
        ),
        (
            "state_prediction_conditional_true_MFE",
            "state_prediction",
            "dual_true_mfe_true_state",
            "dual_true_mfe_predicted_state",
        ),
        (
            "joint_aux_prediction_conditional_true_MFE",
            "joint_aux_prediction",
            "dual_true_all",
            "dual_true_mfe_predicted_state_risk",
        ),
        (
            "predicted_risk_veto_value",
            "risk_policy",
            "dual_predicted_risk",
            "dual_predicted",
        ),
        (
            "predicted_state_veto_value",
            "state_policy",
            "dual_predicted_state",
            "dual_predicted",
        ),
    )
    for effect, layer, left_variant, right_variant in daily_pairs:
        for exposure in EXPOSURE_MODES:
            for slots in SLOT_COUNTS:
                for cost in COST_SCENARIOS:
                    left = get(
                        "daily_rerank",
                        left_variant,
                        exposure,
                        slots,
                        2.0,
                        cost,
                    )
                    right = get(
                        "daily_rerank",
                        right_variant,
                        exposure,
                        slots,
                        2.0,
                        cost,
                    )
                    _append_effect(
                        rows,
                        effect=effect,
                        layer=layer,
                        left=left,
                        right=right,
                    )
    scheduled_pairs = (
        ("D10_entry_selection", "mfe10_true", "mfe10_predicted"),
        ("D20_entry_selection", "mfe20_true", "mfe20_predicted"),
        ("dual_entry_selection", "dual_true_mfe", "dual_predicted"),
        (
            "dual_risk_entry_and_aux",
            "dual_true_mfe_true_risk",
            "dual_predicted_risk",
        ),
    )
    for mode in ("true_peak_close", "post_peak_next_open"):
        for prefix, left_variant, right_variant in scheduled_pairs:
            for exposure in EXPOSURE_MODES:
                for slots in SLOT_COUNTS:
                    for cost in COST_SCENARIOS:
                        _append_effect(
                            rows,
                            effect=f"{prefix}__{mode}",
                            layer="entry_selection_under_fixed_exit",
                            left=get(
                                mode,
                                left_variant,
                                exposure,
                                slots,
                                None,
                                cost,
                            ),
                            right=get(
                                mode,
                                right_variant,
                                exposure,
                                slots,
                                None,
                                cost,
                            ),
                        )
    scheduled_variants = (
        "mfe10_predicted",
        "mfe10_true",
        "mfe20_predicted",
        "mfe20_true",
        "dual_predicted",
        "dual_true_mfe",
        "dual_predicted_risk",
        "dual_true_mfe_true_risk",
    )
    for variant in scheduled_variants:
        for exposure in EXPOSURE_MODES:
            for slots in SLOT_COUNTS:
                for cost in COST_SCENARIOS:
                    peak = get(
                        "true_peak_close",
                        variant,
                        exposure,
                        slots,
                        None,
                        cost,
                    )
                    post = get(
                        "post_peak_next_open",
                        variant,
                        exposure,
                        slots,
                        None,
                        cost,
                    )
                    daily = get(
                        "daily_rerank",
                        variant,
                        exposure,
                        slots,
                        2.0,
                        cost,
                    )
                    _append_effect(
                        rows,
                        effect=f"peak_close_minus_post_open__{variant}",
                        layer="peak_to_next_open_realization",
                        left=peak,
                        right=post,
                    )
                    _append_effect(
                        rows,
                        effect=f"post_peak_exit_minus_daily__{variant}",
                        layer="exit_realization",
                        left=post,
                        right=daily,
                    )
                    _append_effect(
                        rows,
                        effect=f"peak_exit_minus_daily__{variant}",
                        layer="exit_realization_upper_bound",
                        left=peak,
                        right=daily,
                    )
    return pd.DataFrame.from_records(rows)


def _effect_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["effect", "layer", "cost_scenario"], sort=True)
        .agg(
            comparison_count=("terminal_return_delta", "count"),
            median_terminal_return_delta=("terminal_return_delta", "median"),
            minimum_terminal_return_delta=("terminal_return_delta", "min"),
            maximum_terminal_return_delta=("terminal_return_delta", "max"),
            positive_fraction=(
                "terminal_return_delta",
                lambda values: float(
                    np.mean(np.asarray(values, dtype=np.float64) > 0.0)
                ),
            ),
            median_terminal_pnl_delta_cny=(
                "terminal_pnl_delta_cny",
                "median",
            ),
            median_cagr_delta=("cagr_delta", "median"),
            median_drawdown_delta=("maximum_drawdown_delta", "median"),
            median_turnover_delta=("turnover_delta", "median"),
            median_total_cost_delta=("total_cost_delta", "median"),
        )
        .reset_index()
    )


def _endpoint_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(
            ["mode", "variant_id", "cost_scenario"],
            sort=True,
        )
        .agg(
            configuration_count=("task_id", "count"),
            median_terminal_return=(
                "terminal_cost_accrued_return",
                "median",
            ),
            minimum_terminal_return=("terminal_cost_accrued_return", "min"),
            maximum_terminal_return=("terminal_cost_accrued_return", "max"),
            positive_terminal_fraction=(
                "terminal_cost_accrued_return",
                lambda values: float(
                    np.mean(np.asarray(values, dtype=np.float64) > 0.0)
                ),
            ),
            median_cagr=("net_cagr", "median"),
            median_maximum_drawdown=("maximum_drawdown", "median"),
            median_turnover=("turnover_to_starting_cash", "median"),
            median_total_cost=("total_cost", "median"),
            median_holding_days=("holding_days_mean", "median"),
            median_fraction_above_0p1=(
                "fraction_above_0.001",
                "median",
            ),
            median_fraction_above_0p5=(
                "fraction_above_0.005",
                "median",
            ),
            median_fraction_above_1p0=("fraction_above_0.01", "median"),
        )
        .reset_index()
    )


def _task_capacity_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(
            ["mode", "variant_id", "slot_count", "cost_scenario"],
            sort=True,
        )
        .agg(
            task_count=("task_id", "count"),
            median_filled_order_count=(
                "participation_finite_order_count",
                "median",
            ),
            median_fraction_above_0p1=(
                "fraction_above_0.001",
                "median",
            ),
            median_fraction_above_0p5=(
                "fraction_above_0.005",
                "median",
            ),
            median_fraction_above_1p0=("fraction_above_0.01", "median"),
        )
        .reset_index()
    )


def _dominant_loss(effect_summary: pd.DataFrame) -> dict[str, Any]:
    candidates = (
        "D10_MFE_prediction",
        "D20_MFE_prediction",
        "dual_both_MFE_prediction",
        "risk_prediction_conditional_true_MFE",
        "state_prediction_conditional_true_MFE",
        "joint_aux_prediction_conditional_true_MFE",
        "D10_entry_selection__post_peak_next_open",
        "D20_entry_selection__post_peak_next_open",
        "dual_entry_selection__post_peak_next_open",
        "post_peak_exit_minus_daily__dual_predicted",
        "peak_close_minus_post_open__dual_true_mfe",
    )
    selected = effect_summary[
        effect_summary["cost_scenario"].astype(str).eq("base")
        & effect_summary["effect"].astype(str).isin(candidates)
    ].copy()
    selected = selected.sort_values(
        "median_terminal_return_delta",
        ascending=False,
    )
    if selected.empty:
        raise ValueError("dominant-loss evidence is empty")
    records = [
        {
            "effect": str(row["effect"]),
            "layer": str(row["layer"]),
            "median_terminal_return_delta": float(row["median_terminal_return_delta"]),
            "positive_fraction": float(row["positive_fraction"]),
        }
        for row in selected.to_dict("records")
    ]
    return {
        "dominant_effect": records[0]["effect"],
        "dominant_layer": records[0]["layer"],
        "ranked_base_cost_effects": records,
    }


def evaluate(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    record_root: Path = DEFAULT_RECORD_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    manifest = prepare_gap_book(
        study_path=study_path,
        output_root=output_root,
    )
    study_sha256 = _study_hash(study_path)
    signal_sha256 = str(manifest["signal_identity_sha256"])
    tasks = task_specs(study)
    incomplete = [
        task.task_id
        for task in tasks
        if not _task_complete(
            output_root=output_root,
            task=task,
            study_sha256=study_sha256,
            signal_sha256=signal_sha256,
        )
    ]
    if incomplete:
        raise RuntimeError(
            f"{len(incomplete)} gap tasks incomplete; first={incomplete[0]}"
        )
    results = [_task_result(output_root=output_root, task=task) for task in tasks]
    metrics = pd.DataFrame([_task_metric_row(result) for result in results])
    relative_errors = _relative_accounting_errors(results)
    metrics["maximum_relative_conservation_error"] = metrics["task_id"].map(
        relative_errors
    )
    annual = _add_period_cash_metrics(
        economic._annual_metric_table(results),
        period_column="year",
    )
    monthly = _add_period_cash_metrics(
        oracle._monthly_metric_table(results=results),
        period_column="month",
    )
    diagnostics = pd.read_parquet(
        _verify_record(manifest["files"]["candidate_diagnostics"])
    )
    candidate_summary = _candidate_summary(diagnostics)
    capacity = pd.read_parquet(_verify_record(manifest["files"]["candidate_capacity"]))
    candidate_capacity = _capacity_summary(capacity)
    effects = _effect_rows(metrics)
    effect_summary = _effect_summary(effects)
    endpoints = _endpoint_summary(metrics)
    task_capacity = _task_capacity_summary(metrics)
    dominant = _dominant_loss(effect_summary)
    accounting_invalid = bool(
        not np.isfinite(
            metrics["maximum_relative_conservation_error"].to_numpy(dtype=np.float64)
        ).all()
        or metrics["maximum_relative_conservation_error"].astype(float).max() > 1.0e-9
    )

    def effect_record(name: str) -> dict[str, Any]:
        selected = effect_summary[
            effect_summary["effect"].astype(str).eq(name)
            & effect_summary["cost_scenario"].astype(str).eq("base")
        ]
        if len(selected) != 1:
            raise ValueError(f"effect support changed: {name}")
        row = selected.iloc[0]
        return {
            "median_terminal_return_delta": float(row["median_terminal_return_delta"]),
            "minimum_terminal_return_delta": float(
                row["minimum_terminal_return_delta"]
            ),
            "maximum_terminal_return_delta": float(
                row["maximum_terminal_return_delta"]
            ),
            "positive_fraction": float(row["positive_fraction"]),
        }

    candidate_top5 = candidate_summary[
        np.isclose(candidate_summary["fraction"].astype(float), 0.05)
    ].sort_values(["horizon", "year"])
    opportunity_capture = [
        {
            "year": int(row["year"]),
            "horizon": int(row["horizon"]),
            "rank_ic": float(row["daily_rank_ic"]),
            "top5_overlap": float(row["overlap_over_smaller"]),
            "top5_true_mfe_capture_ratio": float(row["opportunity_capture_ratio"]),
            "predicted_top5_true_mfe": float(row["predicted_top_true_mfe"]),
            "oracle_top5_true_mfe": float(row["oracle_top_true_mfe"]),
            "predicted_top5_peak_day": float(row["predicted_top_peak_day"]),
            "oracle_top5_peak_day": float(row["oracle_top_peak_day"]),
        }
        for row in candidate_top5.to_dict("records")
    ]
    risk_effect = effect_record("risk_prediction_conditional_true_MFE")
    state_effect = effect_record("state_prediction_conditional_true_MFE")
    mfe_effects = {
        name: effect_record(name)
        for name in (
            "D10_MFE_prediction",
            "D20_MFE_prediction",
            "dual_both_MFE_prediction",
        )
    }
    predicted_dual_exit = effect_record("post_peak_exit_minus_daily__dual_predicted")
    peak_to_open = effect_record("peak_close_minus_post_open__dual_true_mfe")
    if accounting_invalid:
        overall = "audit_invalid_due_to_accounting"
    elif all(
        item["median_terminal_return_delta"] > 0.0 for item in mfe_effects.values()
    ):
        overall = "MFE_prediction_is_a_material_economic_bottleneck"
    elif predicted_dual_exit["median_terminal_return_delta"] > 0.0:
        overall = "exit_realization_is_the_primary_tested_bottleneck"
    else:
        overall = "gap_is_distributed_without_a_single_positive_substitution"
    decision = {
        "status": "completed_without_policy_selection",
        "overall": overall,
        "dominant_loss": dominant,
        "MFE_prediction_effects": mfe_effects,
        "risk_prediction_effect": risk_effect,
        "state_prediction_effect": state_effect,
        "predicted_dual_post_peak_exit_effect": predicted_dual_exit,
        "true_dual_peak_to_next_open_loss": peak_to_open,
        "opportunity_capture": opportunity_capture,
        "accounting_invalid": accounting_invalid,
        "fixed_notional_semantics": {
            "starting_cash_cny": STARTING_CASH_CNY,
            "position_budget": "starting_cash / slots",
            "profits_reinvested": False,
        },
        "single_policy_selected": False,
        "interpretation_boundary": (
            "true-source substitutions and true-peak exits are hindsight "
            "diagnostics; only all-predicted daily variants remain OOS signals"
        ),
    }
    evaluation_root = output_root / "evaluation"
    paths = {
        "task_metrics": evaluation_root / "task_metrics.parquet",
        "annual_metrics": evaluation_root / "annual_metrics.parquet",
        "monthly_metrics": evaluation_root / "monthly_metrics.parquet",
        "candidate_daily": evaluation_root / "candidate_daily.parquet",
        "candidate_summary": evaluation_root / "candidate_summary.parquet",
        "candidate_capacity": evaluation_root / "candidate_capacity_summary.parquet",
        "paired_effects": evaluation_root / "paired_effects.parquet",
        "effect_summary": evaluation_root / "effect_summary.parquet",
        "endpoint_summary": evaluation_root / "endpoint_summary.parquet",
        "task_capacity": evaluation_root / "task_capacity_summary.parquet",
    }
    frames = {
        "task_metrics": metrics,
        "annual_metrics": annual,
        "monthly_metrics": monthly,
        "candidate_daily": diagnostics,
        "candidate_summary": candidate_summary,
        "candidate_capacity": candidate_capacity,
        "paired_effects": effects,
        "effect_summary": effect_summary,
        "endpoint_summary": endpoints,
        "task_capacity": task_capacity,
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
        "task_count": len(metrics),
        "annual_metric_count": len(annual),
        "monthly_metric_count": len(monthly),
        "effect_comparison_count": len(effects),
        "period": {
            "years": list(YEARS),
            "continuous_account": True,
            "annual_reset": False,
            "maximum_consumed_outcome_date": "2025-12-31",
            "forbidden_2026_row_count": 0,
        },
        "gap_book": {
            "day_count": int(manifest["day_count"]),
            "symbol_count": int(manifest["symbol_count"]),
            "support": dict(manifest["support"]),
            "manifest": _file_record(output_root / "gap_book/manifest.json"),
        },
        "decision": decision,
        "validation": {
            "task_count_matches_preregistration": len(metrics) == 1200,
            "maximum_relative_conservation_error": float(
                metrics["maximum_relative_conservation_error"].max()
            ),
            "accounting_invalid": accounting_invalid,
            "fixed_notional": True,
            "profits_reinvested": False,
            "single_policy_selected": False,
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
        "prediction_oracle_gap_completed",
        overall=overall,
        dominant_effect=dominant["dominant_effect"],
    )
    return record


def self_test() -> dict[str, Any]:
    study = load_study(DEFAULT_STUDY_PATH)
    if len(task_specs(study)) != 1200:
        raise AssertionError("task inventory self-test failed")
    if int(study["period"]["forbidden_year"]) != 2026:
        raise AssertionError("2026 guard self-test failed")
    ranks, symbols = _rank_on_support(
        np.asarray([0.4, 0.1, 0.3, 0.2]),
        np.asarray([True, False, True, True]),
    )
    if symbols.tolist() != [0, 2, 3]:
        raise AssertionError("common support self-test failed")
    if not np.allclose(
        ranks[symbols],
        np.asarray([1.0, 0.5, 0.0]),
    ):
        raise AssertionError("support reranking self-test failed")
    top = _top_metrics(
        predicted_rank=np.asarray([1.0, 0.5, 0.0]),
        true_rank=np.asarray([0.5, 1.0, 0.0]),
        true_mfe=np.asarray([0.2, 0.4, 0.1]),
        peak_day=np.asarray([3.0, 4.0, 2.0]),
        fraction=0.5,
    )
    if top["intersection_count"] != 2:
        raise AssertionError("top overlap self-test failed")

    class FixedBook:
        @staticmethod
        def position_allocation(
            *,
            spec: economic.TaskSpec,
            cash: float,
            equity_open: float,
        ) -> float:
            del spec, cash, equity_open
            return 123.0

    allocation = economic._position_allocation(
        book=FixedBook(),
        spec=economic.TaskSpec(
            family="mfe10_primary",
            exposure_mode="target_full",
            slot_count=3,
            buffer_multiplier=0.0,
            cost_scenario="base",
        ),
        cash=1_000.0,
        equity_open=2_000.0,
    )
    if allocation != 123.0:
        raise AssertionError("fixed allocation self-test failed")
    return {
        "status": "passed",
        "task_count": 1200,
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
        payload = prepare_gap_book(
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
