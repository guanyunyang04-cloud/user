from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from daily_research.path_policy import seq100_entry_role_synthesis as entry_role
from daily_research.path_policy import seq100_mfe_feature_family_audit as feature_audit
from daily_research.path_policy import seq100_path_label_learnability as base

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_post_entry_incremental_information_v1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_post_entry_incremental_information_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies"
    / "seq100_post_entry_incremental_information_v1"
)
DEFAULT_RECORD_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/research_records/seq100"
    / "seq100_post_entry_incremental_information_v1"
)
FOLD_YEARS = (2023, 2024, 2025)
LANDMARK_AGES = (1, 3, 5)
TARGET_NAMES = (
    "mfe_10",
    "mfe_20",
    "state_10",
    "pre_peak_mae_10",
    "pre_peak_mae_20",
)
CONTROL_NAMES = (
    "current_mfe10_rank",
    "current_mfe20_rank",
    "current_state_low_rank",
    "current_state_high_rank",
    "current_pre_peak_mae10_rank",
    "current_pre_peak_mae20_rank",
)
BASELINE_SCORE_BY_TARGET = {
    "mfe_10": "mfe_10",
    "mfe_20": "mfe_20",
    "state_10": "state_expected_10",
    "pre_peak_mae_10": "pre_peak_mae_10",
    "pre_peak_mae_20": "pre_peak_mae_20",
}
SUMMARY_SCHEMA = "seq100_post_entry_incremental_information_summary/v1"
AGE_SCHEMA = "seq100_post_entry_incremental_information_age/v1"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)
    return _file_record(path, rows=len(frame), columns=list(frame.columns))


def _file_record(path: Path, **extra: Any) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(WORKSPACE_ROOT).as_posix()
        if resolved.is_relative_to(WORKSPACE_ROOT)
        else str(resolved),
        "size": int(resolved.stat().st_size),
        **extra,
    }


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def _config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("study_id") != STUDY_ID:
        raise ValueError(f"study_id must be {STUDY_ID}")
    folds = dict(payload["folds"])
    if tuple(int(value) for value in folds["fold_years"]) != FOLD_YEARS:
        raise ValueError("fold years changed")
    if folds["maximum_outcome_date"] != "2025-12-31":
        raise ValueError("outcome boundary changed")
    if int(folds["forbidden_outcome_year"]) != 2026:
        raise ValueError("2026 boundary changed")
    landmarks = dict(payload["landmarks"])
    if tuple(int(value) for value in landmarks["ages"]) != LANDMARK_AGES:
        raise ValueError("landmark ages changed")
    if not bool(landmarks["ages_are_diagnostic_slices_not_holding_periods"]):
        raise ValueError("landmark ages cannot select holding periods")
    exclusions = dict(landmarks["age_feature_exclusions"])
    if tuple(sorted(int(value) for value in exclusions)) != LANDMARK_AGES:
        raise ValueError("age feature exclusions changed")
    contract = dict(payload["entry_contract"])
    if not bool(contract["frozen"]) or bool(contract["fusion"]):
        raise ValueError("entry contract must remain frozen and unfused")
    targets = dict(payload["targets"])
    if bool(targets["holding_target_requires_new_buy_fill"]):
        raise ValueError("holding targets cannot require a new buy fill")
    decision = dict(payload["decision"])
    if not bool(decision["formal_nested_models_are_not_part_of_this_audit"]):
        raise ValueError("this audit cannot train a formal holding model")
    return payload


def _date_boundaries(date_idx: np.ndarray) -> np.ndarray:
    dates = np.asarray(date_idx, dtype=np.int32)
    if dates.ndim != 1 or not dates.size:
        raise ValueError("date boundaries require a non-empty one-dimensional array")
    if bool(np.any(dates[1:] < dates[:-1])):
        raise ValueError("date rows must be ordered")
    return np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])


def _rank_by_date(values: np.ndarray, date_idx: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    dates = np.asarray(date_idx, dtype=np.int32)
    if raw.shape != dates.shape:
        raise ValueError("date ranks require aligned arrays")
    result = np.full(raw.shape, np.nan, dtype=np.float32)
    for start, stop in pairwise(_date_boundaries(dates)):
        finite = np.isfinite(raw[start:stop])
        count = int(finite.sum())
        if count == 0:
            continue
        local = np.flatnonzero(finite)
        if count == 1:
            result[start + local] = 1.0
        else:
            ranks = stats.rankdata(raw[start:stop][finite], method="average")
            result[start + local] = ((ranks - 1.0) / (count - 1.0)).astype(
                np.float32
            )
    return result


@dataclass(frozen=True)
class FrozenContract:
    rows: np.ndarray
    raw: dict[str, np.ndarray]
    rank: dict[str, np.ndarray]
    source_alignment: tuple[dict[str, Any], ...]
    consumed_files: tuple[str, ...]

    def positions(self, candidate_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        requested = np.asarray(candidate_rows, dtype=np.int64)
        position = np.searchsorted(self.rows, requested)
        safe = np.minimum(position, len(self.rows) - 1)
        found = (position < len(self.rows)) & (self.rows[safe] == requested)
        return safe.astype(np.int64, copy=False), found


def _result_files(bundle: entry_role.ScoreBundle) -> set[str]:
    files: set[str] = set()
    for record in dict(bundle.result.get("files", {})).values():
        if isinstance(record, Mapping) and "path" in record:
            files.add(str(_resolve(str(record["path"]))))
    return files


def load_frozen_contract(
    *,
    study: Mapping[str, Any],
    feature_study: Mapping[str, Any],
    path_study: Mapping[str, Any],
) -> FrozenContract:
    sources = dict(study["sources"])
    feature_root = _resolve(sources["feature_output_root"])
    path_root = _resolve(sources["path_output_root"])
    contract = dict(study["entry_contract"])
    row_parts: list[np.ndarray] = []
    raw_parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "mfe_10",
            "mfe_20",
            "state_low_10",
            "state_mid_10",
            "state_high_10",
            "state_expected_10",
            "pre_peak_mae_10",
            "pre_peak_mae_20",
        )
    }
    rank_parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "mfe_10",
            "mfe_20",
            "state_low_10",
            "state_high_10",
            "state_expected_10",
            "pre_peak_mae_10",
            "pre_peak_mae_20",
        )
    }
    consumed: set[str] = set()
    source_alignment: list[dict[str, Any]] = []
    for year in FOLD_YEARS:
        mfe10 = entry_role._load_mfe_bundle(
            feature_study=feature_study,
            output_root=feature_root,
            year=year,
            horizon=10,
            variant=str(contract["mfe_10_variant"]),
        )
        mfe20 = entry_role._load_mfe_bundle(
            feature_study=feature_study,
            output_root=feature_root,
            year=year,
            horizon=20,
            variant=str(contract["mfe_20_variant"]),
        )
        state = entry_role._load_path_bundle(
            path_study=path_study,
            output_root=path_root,
            year=year,
            horizon=10,
            target="state",
        )
        risk10 = entry_role._load_path_bundle(
            path_study=path_study,
            output_root=path_root,
            year=year,
            horizon=10,
            target="pre_peak_mae",
        )
        risk20 = entry_role._load_path_bundle(
            path_study=path_study,
            output_root=path_root,
            year=year,
            horizon=20,
            target="pre_peak_mae",
        )
        bundles = (mfe10, mfe20, state, risk10, risk20)
        bundle_names = ("mfe_10", "mfe_20", "state_10", "risk_10", "risk_20")
        rows = np.asarray(mfe10.rows, dtype=np.int64)
        for bundle in bundles[1:]:
            rows = np.intersect1d(
                rows, np.asarray(bundle.rows, dtype=np.int64), assume_unique=True
            )
        if not rows.size:
            raise ValueError(f"frozen contract intersection is empty in fold {year}")
        positions: list[np.ndarray] = []
        source_counts: dict[str, Any] = {}
        for name, bundle in zip(bundle_names, bundles, strict=True):
            source_rows = np.asarray(bundle.rows, dtype=np.int64)
            position = np.searchsorted(source_rows, rows)
            if bool(np.any(position >= len(source_rows))) or not np.array_equal(
                source_rows[position], rows
            ):
                raise AssertionError(f"frozen contract intersection failed: {name}")
            positions.append(position)
            source_counts[name] = {
                "source_row_count": len(source_rows),
                "common_row_count": len(rows),
                "rows_outside_common_contract": len(source_rows) - len(rows),
            }
        source_alignment.append({"year": int(year), "sources": source_counts})
        mfe10_position, mfe20_position, state_position, risk10_position, risk20_position = positions
        probability = np.asarray(state.prediction, dtype=np.float32)[state_position]
        if probability.shape != (len(rows), 3):
            raise ValueError("state prediction must have three columns")
        if not np.allclose(probability.sum(axis=1), 1.0, rtol=1.0e-5, atol=1.0e-5):
            raise ValueError("state probabilities do not sum to one")
        current_raw = {
            "mfe_10": np.asarray(mfe10.prediction, dtype=np.float32)[mfe10_position],
            "mfe_20": np.asarray(mfe20.prediction, dtype=np.float32)[mfe20_position],
            "state_low_10": probability[:, 0],
            "state_mid_10": probability[:, 1],
            "state_high_10": probability[:, 2],
            "state_expected_10": probability[:, 1] + 2.0 * probability[:, 2],
            "pre_peak_mae_10": np.asarray(risk10.prediction, dtype=np.float32)[
                risk10_position
            ],
            "pre_peak_mae_20": np.asarray(risk20.prediction, dtype=np.float32)[
                risk20_position
            ],
        }
        date_idx = np.asarray(feature_study_inputs_date(feature_study, rows), dtype=np.int32)
        row_parts.append(rows)
        for name, values in current_raw.items():
            raw_parts[name].append(np.asarray(values, dtype=np.float32))
        for name, parts in rank_parts.items():
            parts.append(_rank_by_date(current_raw[name], date_idx))
        for bundle in bundles:
            consumed.update(_result_files(bundle))
    all_rows = np.concatenate(row_parts)
    if bool(np.any(all_rows[1:] <= all_rows[:-1])):
        raise ValueError("frozen contract rows must be unique and ordered")
    return FrozenContract(
        rows=all_rows,
        raw={name: np.concatenate(parts) for name, parts in raw_parts.items()},
        rank={name: np.concatenate(parts) for name, parts in rank_parts.items()},
        source_alignment=tuple(source_alignment),
        consumed_files=tuple(sorted(consumed)),
    )


_FEATURE_INPUT_CACHE: base.LearnabilityInputs | None = None


def feature_study_inputs_date(
    feature_study: Mapping[str, Any], rows: np.ndarray
) -> np.ndarray:
    global _FEATURE_INPUT_CACHE
    if _FEATURE_INPUT_CACHE is None:
        _FEATURE_INPUT_CACHE, _ = feature_audit._load_validated_inputs(feature_study)
    return np.asarray(_FEATURE_INPUT_CACHE.candidate_date_idx[rows], dtype=np.int32)


def _candidate_lookup(
    candidate_keys: np.ndarray, requested_keys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(candidate_keys, dtype=np.int64)
    requested = np.asarray(requested_keys, dtype=np.int64)
    if source.ndim != 1 or requested.ndim != 1 or not source.size:
        raise ValueError("candidate lookup requires non-empty one-dimensional keys")
    if bool(np.any(source[1:] <= source[:-1])):
        raise ValueError("candidate keys must be unique and ordered")
    position = np.searchsorted(source, requested)
    safe = np.minimum(position, len(source) - 1)
    found = (position < len(source)) & (source[safe] == requested)
    rows = np.where(found, safe, -1).astype(np.int64)
    return rows, found


def _load_memmap(record: Mapping[str, Any], dtype: Any) -> np.memmap:
    return np.memmap(
        _resolve(str(record["path"])),
        dtype=dtype,
        mode="r",
        shape=tuple(int(value) for value in record["shape"]),
    )


def _load_state_model(
    inputs: base.LearnabilityInputs, horizon: int
) -> dict[str, np.ndarray]:
    record = dict(inputs.label_manifest["state_models"][str(int(horizon))])
    with np.load(_resolve(record["path"]), allow_pickle=False) as stored:
        model = {name: np.asarray(stored[name]) for name in stored.files}
    model["raw_to_state"] = np.asarray(
        record["raw_cluster_to_ordered_state"], dtype=np.int8
    )
    if int(model["horizon"][0]) != int(horizon):
        raise ValueError("state model horizon changed")
    if int(model["window_year"][0]) != 2022:
        raise ValueError("state model is not frozen through 2022")
    return model


def _assign_state(close: np.ndarray, model: Mapping[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(close, dtype=np.float32)
    horizon = int(np.asarray(model["horizon"])[0])
    if values.ndim != 2 or values.shape[1] != horizon:
        raise ValueError("state path has the wrong horizon")
    scaled = (
        np.clip(values, model["clip_low"], model["clip_high"])
        - model["center"]
    ) / model["scale"]
    transformed = (scaled - model["pca_mean"]) @ np.asarray(
        model["pca_components"], dtype=np.float32
    ).T
    centers = np.asarray(model["fixed_k3_centers"], dtype=np.float32)
    squared = (
        np.square(transformed).sum(axis=1, keepdims=True)
        - 2.0 * transformed @ centers.T
        + np.square(centers).sum(axis=1)[None, :]
    )
    raw = np.argmin(squared, axis=1)
    return np.asarray(model["raw_to_state"], dtype=np.int8)[raw]


def _numeric_path_labels(
    close: np.ndarray,
    sellable: np.ndarray,
    valid: np.ndarray,
    *,
    first_legal_day: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(close, dtype=np.float64)
    legal = np.asarray(sellable, dtype=bool)
    base_valid = np.asarray(valid, dtype=bool)
    if values.ndim != 2 or legal.shape != values.shape:
        raise ValueError("numeric labels require aligned two-dimensional paths")
    start = int(first_legal_day) - 1
    if start < 0 or start >= values.shape[1]:
        raise ValueError("first legal day is outside the path")
    g = np.full(len(values), np.nan, dtype=np.float32)
    mfe = np.full(len(values), np.nan, dtype=np.float32)
    adverse = np.full(len(values), np.nan, dtype=np.float32)
    g[base_valid] = values[base_valid, -1].astype(np.float32)
    legal_choice = legal[:, start:] & np.isfinite(values[:, start:])
    has_legal = legal_choice.any(axis=1)
    label_valid = base_valid & has_legal
    if bool(label_valid.any()):
        masked = np.where(legal_choice, values[:, start:], -np.inf)
        peak = np.argmax(masked, axis=1) + start
        rows = np.arange(len(values), dtype=np.int64)
        cumulative_min = np.minimum.accumulate(values, axis=1)
        mfe[label_valid] = values[rows[label_valid], peak[label_valid]].astype(
            np.float32
        )
        adverse[label_valid] = np.minimum(
            0.0, cumulative_min[rows[label_valid], peak[label_valid]]
        ).astype(np.float32)
    return g, mfe, adverse


class HoldingPathReader:
    def __init__(
        self, pack: Mapping[str, Any], inputs: base.LearnabilityInputs
    ) -> None:
        self.pack = dict(pack)
        self.date_values = np.asarray(pack["date_values"], dtype=str)
        self.symbol_count = int(pack["symbol_count"])
        cutoff = np.flatnonzero(self.date_values == inputs.maximum_outcome_date)
        if cutoff.size != 1:
            raise ValueError("outcome cutoff is absent or duplicated")
        self.cutoff_idx = int(cutoff[0])
        channels = dict(pack["feature_channels"])
        self.daily_raw = _load_memmap(channels["daily_raw"], np.float32)
        self.turnover = _load_memmap(channels["turnover"], np.float32)
        masks = dict(pack["masks"])
        self.has_bar = _load_memmap(masks["has_bar"], bool)
        self.is_suspended = _load_memmap(masks["is_suspended"], bool)
        self.tradable = _load_memmap(masks["tradable"], bool)
        self.exit_sellable = _load_memmap(masks["exit_sellable"], bool)
        path = dict(pack["label_arrays"]["future_ohlcva_path"])
        self.shards = tuple(dict(item) for item in path["shards"])
        self._shard_by_date: dict[int, int] = {}
        for index, record in enumerate(self.shards):
            for date_idx in range(
                int(record["date_start_idx"]), int(record["date_end_idx"]) + 1
            ):
                self._shard_by_date[date_idx] = index
        self.cached_index: int | None = None
        self.cached: np.memmap | None = None
        self.state10_model = _load_state_model(inputs, 10)

    def _future(self, date_idx: int, symbols: np.ndarray, horizon: int) -> np.ndarray:
        index = self._shard_by_date.get(int(date_idx))
        if index is None:
            raise ValueError(f"no future path shard covers date {date_idx}")
        record = self.shards[index]
        if self.cached_index != index:
            self.cached = np.memmap(
                _resolve(record["path"]),
                dtype=np.float32,
                mode="r",
                shape=tuple(int(value) for value in record["shape"]),
            )
            self.cached_index = index
        if self.cached is None:
            raise AssertionError("future path cache is empty")
        local = int(date_idx) - int(record["date_start_idx"])
        return np.asarray(
            self.cached[local, np.asarray(symbols, dtype=np.int64), :horizon, :4],
            dtype=np.float32,
        )

    def realized_features(
        self, signal_dates: np.ndarray, symbols: np.ndarray, age: int
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        dates = np.asarray(signal_dates, dtype=np.int32)
        symbol = np.asarray(symbols, dtype=np.int64)
        count = len(dates)
        entry_open = np.asarray(self.daily_raw[dates + 1, symbol, 0], dtype=np.float64)
        signal_close = np.asarray(self.daily_raw[dates, symbol, 3], dtype=np.float64)
        close = np.empty((count, int(age)), dtype=np.float64)
        relative_turnover = np.empty_like(close)
        tradable = np.empty((count, int(age)), dtype=bool)
        for day in range(1, int(age) + 1):
            current = dates + day
            close[:, day - 1] = np.asarray(
                self.daily_raw[current, symbol, 3], dtype=np.float64
            )
            relative_turnover[:, day - 1] = np.asarray(
                self.turnover[current, symbol, 1], dtype=np.float64
            )
            tradable[:, day - 1] = np.asarray(
                self.tradable[current, symbol], dtype=bool
            )
        denominator = np.column_stack([entry_open, close[:, :-1]])
        day_return = np.divide(
            close,
            denominator,
            out=np.full_like(close, np.nan),
            where=np.isfinite(denominator) & (denominator > 1.0e-8),
        ) - 1.0
        entry_return = np.divide(
            close,
            entry_open[:, None],
            out=np.full_like(close, np.nan),
            where=np.isfinite(entry_open[:, None]) & (entry_open[:, None] > 1.0e-8),
        ) - 1.0
        path_valid = (
            np.isfinite(signal_close)
            & (signal_close > 1.0e-8)
            & np.isfinite(entry_open)
            & (entry_open > 1.0e-8)
            & np.isfinite(close).all(axis=1)
            & np.isfinite(day_return).all(axis=1)
        )
        running_peak = np.maximum.accumulate(
            np.column_stack([entry_open, close]), axis=1
        )[:, -1]
        absolute_move = np.abs(day_return).sum(axis=1)
        current_return = entry_return[:, -1]
        finite_turnover = np.isfinite(relative_turnover)
        turnover_count = finite_turnover.sum(axis=1)
        turnover_mean = np.divide(
            np.where(finite_turnover, relative_turnover, 0.0).sum(axis=1),
            turnover_count,
            out=np.full(count, np.nan, dtype=np.float64),
            where=turnover_count > 0,
        )
        signal_turnover = np.asarray(
            self.turnover[dates, symbol, 1], dtype=np.float64
        )
        features = {
            "entry_gap_return": np.divide(
                entry_open,
                signal_close,
                out=np.full(count, np.nan, dtype=np.float64),
                where=np.isfinite(signal_close) & (signal_close > 1.0e-8),
            )
            - 1.0,
            "entry_to_close_return": current_return,
            "realized_close_mfe": np.nanmax(entry_return, axis=1),
            "realized_close_mae": np.minimum(0.0, np.nanmin(entry_return, axis=1)),
            "drawdown_from_realized_peak": np.divide(
                close[:, -1],
                running_peak,
                out=np.full(count, np.nan, dtype=np.float64),
                where=np.isfinite(running_peak) & (running_peak > 1.0e-8),
            )
            - 1.0,
            "last_close_return": day_return[:, -1],
            "realized_volatility": np.nanstd(day_return, axis=1),
            "positive_day_fraction": np.mean(day_return > 0.0, axis=1),
            "path_efficiency": np.divide(
                current_return,
                absolute_move,
                out=np.zeros(count, dtype=np.float64),
                where=absolute_move > 1.0e-12,
            ),
            "relative_turnover_mean": turnover_mean,
            "relative_turnover_last": relative_turnover[:, -1],
            "relative_turnover_change": relative_turnover[:, -1]
            - signal_turnover,
            "tradable_day_fraction": tradable.mean(axis=1),
        }
        for name, values in features.items():
            current = np.asarray(values, dtype=np.float32)
            current[~path_valid] = np.nan
            features[name] = current
        quality = {
            "realized_path_valid": path_valid,
            "realized_path_all_days_tradable": tradable.all(axis=1),
            "activity_feature_complete": np.logical_and.reduce(
                [
                    np.isfinite(features["relative_turnover_mean"]),
                    np.isfinite(features["relative_turnover_last"]),
                    np.isfinite(features["relative_turnover_change"]),
                ]
            ),
        }
        return features, quality

    def holding_targets(
        self, decision_dates: np.ndarray, symbols: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        dates = np.asarray(decision_dates, dtype=np.int32)
        symbol = np.asarray(symbols, dtype=np.int64)
        count = len(dates)
        names = (
            "mfe_10",
            "mfe_20",
            "state_10",
            "pre_peak_mae_10",
            "pre_peak_mae_20",
            "g_10",
            "g_20",
            "matched_mfe_10",
            "matched_mfe_20",
            "matched_pre_peak_mae_10",
            "matched_pre_peak_mae_20",
        )
        targets = {
            name: np.full(count, np.nan, dtype=np.float32) for name in names
        }
        flags = {
            "anchor_observed": np.zeros(count, dtype=bool),
            "anchor_suspended": np.zeros(count, dtype=bool),
            "path_complete_10": np.zeros(count, dtype=bool),
            "path_complete_20": np.zeros(count, dtype=bool),
        }
        for start, stop in pairwise(_date_boundaries(dates)):
            date_idx = int(dates[start])
            current_symbols = symbol[start:stop]
            maximum = min(20, self.cutoff_idx - date_idx)
            if maximum < 10:
                continue
            future = self._future(date_idx, current_symbols, maximum)
            anchor = 1.0 + np.asarray(future[:, 0, 0], dtype=np.float64)
            next_date = date_idx + 1
            observed = (
                np.asarray(self.has_bar[next_date, current_symbols], dtype=bool)
                & ~np.asarray(
                    self.is_suspended[next_date, current_symbols], dtype=bool
                )
                & np.isfinite(anchor)
                & (anchor > 1.0e-8)
            )
            flags["anchor_observed"][start:stop] = observed
            flags["anchor_suspended"][start:stop] = np.asarray(
                self.is_suspended[next_date, current_symbols], dtype=bool
            )
            close = np.divide(
                1.0 + np.asarray(future[:, :, 3], dtype=np.float64),
                anchor[:, None],
                out=np.full((stop - start, maximum), np.nan, dtype=np.float64),
                where=np.isfinite(anchor[:, None]) & (anchor[:, None] > 1.0e-8),
            ) - 1.0
            complete_day = np.isfinite(future).all(axis=2) & np.isfinite(close)
            complete_prefix = np.logical_and.accumulate(complete_day, axis=1)
            sellable = np.asarray(
                self.exit_sellable[
                    date_idx + 1 : date_idx + 1 + maximum, current_symbols
                ],
                dtype=bool,
            ).T
            for horizon in (10, 20):
                if maximum < horizon:
                    continue
                valid = observed & complete_prefix[:, horizon - 1]
                flags[f"path_complete_{horizon}"][start:stop] = valid
                g, mfe, adverse = _numeric_path_labels(
                    close[:, :horizon],
                    sellable[:, :horizon],
                    valid,
                    first_legal_day=1,
                )
                _, matched_mfe, matched_adverse = _numeric_path_labels(
                    close[:, :horizon],
                    sellable[:, :horizon],
                    valid,
                    first_legal_day=2,
                )
                targets[f"g_{horizon}"][start:stop] = g
                targets[f"mfe_{horizon}"][start:stop] = mfe
                targets[f"pre_peak_mae_{horizon}"][start:stop] = adverse
                targets[f"matched_mfe_{horizon}"][start:stop] = matched_mfe
                targets[f"matched_pre_peak_mae_{horizon}"][start:stop] = (
                    matched_adverse
                )
                if horizon == 10 and bool(valid.any()):
                    state = np.full(stop - start, np.nan, dtype=np.float32)
                    state[valid] = _assign_state(
                        close[valid, :10], self.state10_model
                    ).astype(np.float32)
                    targets["state_10"][start:stop] = state
        return targets, flags


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 3:
        return math.nan
    x = x[finite] - float(x[finite].mean())
    y = y[finite] - float(y[finite].mean())
    denominator = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    return float(np.dot(x, y) / denominator) if denominator > 1.0e-12 else math.nan


def _partial_rank_statistics(
    controls: np.ndarray,
    feature: np.ndarray,
    target: np.ndarray,
    *,
    minimum_rows: int,
    quantile_count: int,
) -> dict[str, Any]:
    z = np.asarray(controls, dtype=np.float64)
    x = np.asarray(feature, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if z.ndim != 2 or x.shape != y.shape or len(x) != len(z):
        raise ValueError("partial-rank inputs are not aligned")
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z).all(axis=1)
    count = int(valid.sum())
    if count < int(minimum_rows):
        return {
            "row_count": count,
            "partial_rank_ic": math.nan,
            "residual_quintile_spread": math.nan,
        }
    ranked = np.column_stack(
        [
            stats.rankdata(z[valid], axis=0, method="average"),
            stats.rankdata(x[valid], method="average"),
            stats.rankdata(y[valid], method="average"),
        ]
    ).astype(np.float64)
    design = np.column_stack(
        [np.ones(count, dtype=np.float64), ranked[:, : z.shape[1]]]
    )
    values = ranked[:, z.shape[1] :]
    coefficient = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values - design @ coefficient
    correlation = _safe_correlation(residual[:, 0], residual[:, 1])
    order = np.argsort(residual[:, 0], kind="mergesort")
    tail = max(1, math.ceil(count / int(quantile_count)))
    spread = float(
        residual[order[-tail:], 1].mean() - residual[order[:tail], 1].mean()
    )
    return {
        "row_count": count,
        "partial_rank_ic": correlation,
        "residual_quintile_spread": spread,
    }


def _partial_rank_matrix_statistics(
    controls: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    minimum_rows: int,
    quantile_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(controls, dtype=np.float64)
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if z.ndim != 2 or x.ndim != 2 or y.ndim != 2:
        raise ValueError("partial-rank matrices must be two-dimensional")
    if not (len(z) == len(x) == len(y)):
        raise ValueError("partial-rank matrices are not aligned")
    feature_count = x.shape[1]
    target_count = y.shape[1]
    counts = np.zeros((feature_count, target_count), dtype=np.int32)
    correlations = np.full((feature_count, target_count), np.nan, dtype=np.float64)
    spreads = np.full((feature_count, target_count), np.nan, dtype=np.float64)
    base_valid = np.isfinite(z).all(axis=1) & np.isfinite(y).all(axis=1)

    def compute(feature_indices: np.ndarray, valid: np.ndarray) -> None:
        count = int(valid.sum())
        if count < int(minimum_rows) or not len(feature_indices):
            return
        feature_indices = np.asarray(
            [
                index
                for index in feature_indices
                if float(np.ptp(x[valid, int(index)])) > 1.0e-12
            ],
            dtype=np.int64,
        )
        if not len(feature_indices):
            return
        ranked_controls = stats.rankdata(z[valid], axis=0, method="average")
        ranked_features = stats.rankdata(
            x[valid][:, feature_indices], axis=0, method="average"
        )
        ranked_targets = stats.rankdata(y[valid], axis=0, method="average")
        design = np.column_stack(
            [np.ones(count, dtype=np.float64), ranked_controls]
        )
        values = np.column_stack([ranked_features, ranked_targets])
        coefficient = np.linalg.lstsq(design, values, rcond=None)[0]
        residual = values - design @ coefficient
        feature_residual = residual[:, : len(feature_indices)]
        target_residual = residual[:, len(feature_indices) :]
        feature_centered = feature_residual - feature_residual.mean(
            axis=0, keepdims=True
        )
        target_centered = target_residual - target_residual.mean(
            axis=0, keepdims=True
        )
        numerator = feature_centered.T @ target_centered
        denominator = np.sqrt(
            np.square(feature_centered).sum(axis=0)[:, None]
            * np.square(target_centered).sum(axis=0)[None, :]
        )
        current_correlation = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 1.0e-12,
        )
        tail = max(1, math.ceil(count / int(quantile_count)))
        for local_index, global_index in enumerate(feature_indices):
            order = np.argsort(feature_residual[:, local_index], kind="mergesort")
            current_spread = (
                target_residual[order[-tail:]].mean(axis=0)
                - target_residual[order[:tail]].mean(axis=0)
            )
            counts[global_index, :] = count
            correlations[global_index, :] = current_correlation[local_index]
            spreads[global_index, :] = current_spread

    complete = np.asarray(
        [bool(np.isfinite(x[base_valid, index]).all()) for index in range(feature_count)]
    )
    complete_indices = np.flatnonzero(complete)
    compute(complete_indices, base_valid)
    for feature_index in np.flatnonzero(~complete):
        valid = base_valid & np.isfinite(x[:, feature_index])
        compute(np.asarray([feature_index], dtype=np.int64), valid)
    return counts, correlations, spreads


def _baseline_daily_statistics(
    score: np.ndarray,
    target: np.ndarray,
    *,
    minimum_rows: int,
) -> dict[str, Any]:
    predicted = np.asarray(score, dtype=np.float64)
    actual = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(predicted) & np.isfinite(actual)
    count = int(valid.sum())
    if count < int(minimum_rows):
        return {
            "row_count": count,
            "rank_ic": math.nan,
            "top5_lift": math.nan,
            "target_mean": math.nan,
        }
    current_score = predicted[valid]
    current_target = actual[valid]
    score_variable = float(np.ptp(current_score)) > 1.0e-12
    target_variable = float(np.ptp(current_target)) > 1.0e-12
    rank_ic = (
        float(stats.spearmanr(current_score, current_target).statistic)
        if score_variable and target_variable
        else math.nan
    )
    top_count = max(1, math.ceil(0.05 * count))
    order = np.argsort(current_score, kind="mergesort")
    mean = float(current_target.mean())
    return {
        "row_count": count,
        "rank_ic": rank_ic,
        "top5_lift": float(current_target[order[-top_count:]].mean() - mean)
        if score_variable
        else math.nan,
        "target_mean": mean,
    }


def _strata(entry_mfe10_rank: np.ndarray, entry_mfe20_rank: np.ndarray) -> dict[str, np.ndarray]:
    short = np.asarray(entry_mfe10_rank, dtype=np.float64) >= 0.95
    long = np.asarray(entry_mfe20_rank, dtype=np.float64) >= 0.95
    return {
        "all": np.ones(len(short), dtype=bool),
        "entry_mfe10_top5": short,
        "entry_mfe20_top5": long,
        "entry_both_top5": short & long,
        "entry_either_top5": short | long,
        "entry_short_only_top5": short & ~long,
        "entry_long_only_top5": long & ~short,
    }


def _feature_block_map(study: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for block, names in dict(study["feature_blocks"]).items():
        if block == "A_current_contract":
            continue
        for name in names:
            if name in result:
                raise ValueError(f"added feature is duplicated: {name}")
            result[str(name)] = str(block)
    return result


def _count_by_year(
    date_values: np.ndarray, date_idx: np.ndarray, masks: Mapping[str, np.ndarray]
) -> list[dict[str, Any]]:
    years = np.asarray([int(str(value)[:4]) for value in date_values[date_idx]])
    rows: list[dict[str, Any]] = []
    for year in FOLD_YEARS:
        current = years == year
        rows.append(
            {
                "year": int(year),
                "cohort_count": int(current.sum()),
                **{
                    name: int(np.sum(current & np.asarray(mask, dtype=bool)))
                    for name, mask in masks.items()
                },
            }
        )
    return rows


def _crosscheck_matched_labels(
    *,
    inputs: base.LearnabilityInputs,
    current_rows: np.ndarray,
    found: np.ndarray,
    targets: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    rows = np.asarray(current_rows, dtype=np.int64)
    available = np.asarray(found, dtype=bool) & (rows >= 0)
    checks: dict[str, Any] = {}
    pairs = (
        ("matched_mfe_10", "mfe", 10),
        ("matched_mfe_20", "mfe", 20),
        ("matched_pre_peak_mae_10", "pre_peak_mae", 10),
        ("matched_pre_peak_mae_20", "pre_peak_mae", 20),
    )
    for derived_name, target_name, horizon in pairs:
        expected = np.full(len(rows), np.nan, dtype=np.float32)
        expected[available] = np.asarray(
            inputs.label_values(target_name, horizon)[rows[available]],
            dtype=np.float32,
        )
        derived = np.asarray(targets[derived_name], dtype=np.float32)
        comparable = np.isfinite(expected) & np.isfinite(derived)
        error = (
            float(np.max(np.abs(expected[comparable] - derived[comparable])))
            if bool(comparable.any())
            else math.nan
        )
        if bool(comparable.any()) and not np.allclose(
            expected[comparable], derived[comparable], rtol=2.0e-5, atol=2.0e-5
        ):
            raise AssertionError(f"holding label reconstruction drifted: {derived_name}")
        checks[derived_name] = {
            "comparable_count": int(comparable.sum()),
            "maximum_absolute_error": error,
        }
    expected_state = np.full(len(rows), -1, dtype=np.int8)
    expected_state[available] = np.asarray(
        inputs.label_values("state", 10)[rows[available]], dtype=np.int8
    )
    derived_state = np.asarray(targets["state_10"], dtype=np.float32)
    comparable_state = (expected_state >= 0) & np.isfinite(derived_state)
    mismatch = int(
        np.sum(
            expected_state[comparable_state]
            != derived_state[comparable_state].astype(np.int8)
        )
    )
    if mismatch:
        raise AssertionError("holding state reconstruction drifted")
    checks["state_10"] = {
        "comparable_count": int(comparable_state.sum()),
        "mismatch_count": mismatch,
    }
    return checks


def _run_age(
    *,
    age: int,
    study: Mapping[str, Any],
    study_hash: str,
    output_root: Path,
    inputs: base.LearnabilityInputs,
    contract: FrozenContract,
    reader: HoldingPathReader,
    candidate_keys: np.ndarray,
) -> dict[str, Any]:
    age_root = output_root / f"age_{int(age):02d}"
    complete_path = age_root / "complete.json"
    if complete_path.is_file():
        completed = json.loads(complete_path.read_text(encoding="utf-8"))
        if (
            completed.get("status") == "completed"
            and completed.get("schema") == AGE_SCHEMA
            and int(completed.get("age", -1)) == int(age)
            and completed.get("study_config_sha256") == study_hash
            and all(_resolve(record["path"]).is_file() for record in completed["files"].values())
        ):
            return completed

    original_rows = np.asarray(contract.rows, dtype=np.int64)
    if not bool(np.all(np.asarray(inputs.entry_fill[original_rows]) == 1)):
        raise ValueError("post-entry cohort contains an unfilled original entry")
    original_dates = np.asarray(
        inputs.candidate_date_idx[original_rows], dtype=np.int32
    )
    symbols = np.asarray(inputs.candidate_symbol_idx[original_rows], dtype=np.int32)
    decision_dates = original_dates + int(age)
    if bool(np.any(decision_dates > reader.cutoff_idx)):
        raise AssertionError("landmark observation crossed the 2025 cutoff")
    current_keys = decision_dates.astype(np.int64) * reader.symbol_count + symbols
    current_rows, current_candidate_found = _candidate_lookup(
        candidate_keys, current_keys
    )
    score_lookup_rows = np.where(current_candidate_found, current_rows, 0)
    current_score_position, current_score_found = contract.positions(score_lookup_rows)
    current_score_found &= current_candidate_found

    realized_features, realized_quality = reader.realized_features(
        original_dates, symbols, age
    )
    targets, target_flags = reader.holding_targets(decision_dates, symbols)
    matched_checks = _crosscheck_matched_labels(
        inputs=inputs,
        current_rows=current_rows,
        found=current_candidate_found,
        targets=targets,
    )

    current_rank = {
        name: np.where(
            current_score_found,
            contract.rank[source][current_score_position],
            np.nan,
        ).astype(np.float32)
        for name, source in {
            "current_mfe10_rank": "mfe_10",
            "current_mfe20_rank": "mfe_20",
            "current_state_low_rank": "state_low_10",
            "current_state_high_rank": "state_high_10",
            "current_state_expected_rank": "state_expected_10",
            "current_pre_peak_mae10_rank": "pre_peak_mae_10",
            "current_pre_peak_mae20_rank": "pre_peak_mae_20",
        }.items()
    }
    current_raw = {
        name: np.where(
            current_score_found,
            contract.raw[name][current_score_position],
            np.nan,
        ).astype(np.float32)
        for name in BASELINE_SCORE_BY_TARGET.values()
    }
    entry_rank = contract.rank
    memory_features = {
        "mfe10_rank_change": current_rank["current_mfe10_rank"]
        - entry_rank["mfe_10"],
        "mfe20_rank_change": current_rank["current_mfe20_rank"]
        - entry_rank["mfe_20"],
        "state_expected_rank_change": current_rank["current_state_expected_rank"]
        - entry_rank["state_expected_10"],
        "state_high_rank_change": current_rank["current_state_high_rank"]
        - entry_rank["state_high_10"],
        "pre_peak_mae10_rank_change": current_rank[
            "current_pre_peak_mae10_rank"
        ]
        - entry_rank["pre_peak_mae_10"],
        "pre_peak_mae20_rank_change": current_rank[
            "current_pre_peak_mae20_rank"
        ]
        - entry_rank["pre_peak_mae_20"],
    }
    added_features = {**memory_features, **realized_features}
    feature_blocks = _feature_block_map(study)
    if set(added_features) != set(feature_blocks):
        missing = sorted(set(feature_blocks) - set(added_features))
        extra = sorted(set(added_features) - set(feature_blocks))
        raise ValueError(f"added feature contract differs: missing={missing}, extra={extra}")

    target_matrix = np.column_stack([targets[name] for name in TARGET_NAMES])
    controls = np.column_stack([current_rank[name] for name in CONTROL_NAMES])
    target_complete = np.isfinite(target_matrix).all(axis=1)
    control_complete = np.isfinite(controls).all(axis=1)
    realized_valid = np.asarray(
        realized_quality["realized_path_valid"], dtype=bool
    )
    common_support = (
        current_score_found & target_complete & control_complete & realized_valid
    )
    all_feature_complete = common_support & np.logical_and.reduce(
        [np.isfinite(values) for values in added_features.values()]
    )
    target_valid_without_score = target_complete & realized_valid
    missing_score_with_target = target_valid_without_score & ~current_score_found

    strata = _strata(entry_rank["mfe_10"], entry_rank["mfe_20"])
    expected_strata = tuple(str(value) for value in study["population"]["entry_strata"])
    if tuple(strata) != expected_strata:
        raise ValueError("entry strata changed")
    evaluation = dict(study["evaluation"])
    minimum_rows = int(evaluation["minimum_rows_per_date"])
    quantile_count = int(evaluation["auxiliary_quantile_count"])
    excluded_features = {
        str(value)
        for value in study["landmarks"]["age_feature_exclusions"][str(int(age))]
    }
    unknown_exclusions = excluded_features - set(feature_blocks)
    if unknown_exclusions:
        raise ValueError(f"unknown age feature exclusions: {sorted(unknown_exclusions)}")
    feature_names = tuple(
        name for name in feature_blocks if name not in excluded_features
    )
    added_feature_matrix = np.column_stack(
        [added_features[name] for name in feature_names]
    )
    daily_baseline: list[dict[str, Any]] = []
    daily_partial: list[dict[str, Any]] = []
    date_values = reader.date_values
    for start, stop in pairwise(_date_boundaries(decision_dates)):
        date_idx = int(decision_dates[start])
        trade_date = str(date_values[date_idx])
        year = int(trade_date[:4])
        if year not in FOLD_YEARS:
            continue
        for stratum_name, stratum_values in strata.items():
            local = np.asarray(stratum_values[start:stop], dtype=bool)
            local &= common_support[start:stop]
            if int(local.sum()) < minimum_rows:
                continue
            local_controls = controls[start:stop][local]
            local_targets = target_matrix[start:stop][local]
            local_features = added_feature_matrix[start:stop][local]
            partial_count, partial_ic, partial_spread = (
                _partial_rank_matrix_statistics(
                    local_controls,
                    local_features,
                    local_targets,
                    minimum_rows=minimum_rows,
                    quantile_count=quantile_count,
                )
            )
            for target_index, target_name in enumerate(TARGET_NAMES):
                local_target = local_targets[:, target_index]
                score_name = BASELINE_SCORE_BY_TARGET[target_name]
                baseline = _baseline_daily_statistics(
                    current_raw[score_name][start:stop][local],
                    local_target,
                    minimum_rows=minimum_rows,
                )
                daily_baseline.append(
                    {
                        "date_idx": date_idx,
                        "trade_date": trade_date,
                        "year": year,
                        "age": int(age),
                        "stratum": stratum_name,
                        "target": target_name,
                        **baseline,
                    }
                )
                for feature_index, feature_name in enumerate(feature_names):
                    daily_partial.append(
                        {
                            "date_idx": date_idx,
                            "trade_date": trade_date,
                            "year": year,
                            "age": int(age),
                            "stratum": stratum_name,
                            "target": target_name,
                            "feature": feature_name,
                            "block": feature_blocks[feature_name],
                            "row_count": int(
                                partial_count[feature_index, target_index]
                            ),
                            "partial_rank_ic": float(
                                partial_ic[feature_index, target_index]
                            ),
                            "residual_quintile_spread": float(
                                partial_spread[feature_index, target_index]
                            ),
                        }
                    )

    baseline_frame = pd.DataFrame.from_records(daily_baseline)
    partial_frame = pd.DataFrame.from_records(daily_partial)
    if baseline_frame.empty or partial_frame.empty:
        raise ValueError(f"landmark D{age} produced no audit rows")
    if bool((baseline_frame["year"] == 2026).any()) or bool(
        (partial_frame["year"] == 2026).any()
    ):
        raise AssertionError("forbidden 2026 row reached the audit")
    baseline_record = _write_parquet(
        age_root / "daily_baseline.parquet", baseline_frame
    )
    partial_record = _write_parquet(
        age_root / "daily_partial.parquet", partial_frame
    )

    masks = {
        "realized_path_valid": realized_valid,
        "current_candidate_found": current_candidate_found,
        "current_contract_available": current_score_found,
        "holding_target_complete": target_complete,
        "primary_common_support": common_support,
        "all_added_features_complete": all_feature_complete,
        "target_valid_but_current_contract_missing": missing_score_with_target,
        "next_anchor_suspended": np.asarray(
            target_flags["anchor_suspended"], dtype=bool
        ),
    }
    t1_effect: dict[str, Any] = {}
    for horizon in (10, 20):
        holding = np.asarray(targets[f"mfe_{horizon}"], dtype=np.float64)
        matched = np.asarray(targets[f"matched_mfe_{horizon}"], dtype=np.float64)
        valid = np.isfinite(holding) & np.isfinite(matched)
        difference = holding[valid] - matched[valid]
        t1_effect[str(horizon)] = {
            "comparable_count": int(valid.sum()),
            "mean_holding_minus_entry_semantics": float(difference.mean())
            if difference.size
            else math.nan,
            "different_row_fraction": float(np.mean(np.abs(difference) > 1.0e-12))
            if difference.size
            else math.nan,
            "maximum_difference": float(difference.max())
            if difference.size
            else math.nan,
        }
    quality = {
        "age": int(age),
        "cohort_count": len(original_rows),
        "counts": {name: int(np.sum(mask)) for name, mask in masks.items()},
        "rates": {
            name: float(np.mean(mask)) for name, mask in masks.items()
        },
        "counts_by_decision_year": _count_by_year(
            date_values, decision_dates, masks
        ),
        "matched_entry_label_reconstruction": matched_checks,
        "holding_D1_legality_effect": t1_effect,
        "evaluated_added_features": list(feature_names),
        "structurally_excluded_features": sorted(excluded_features),
        "duplicate_original_date_symbol_count": int(
            len(original_rows)
            - len(
                np.unique(
                    original_dates.astype(np.int64) * reader.symbol_count + symbols
                )
            )
        ),
        "maximum_observation_date": str(date_values[int(decision_dates.max())]),
        "maximum_outcome_date_read": str(date_values[reader.cutoff_idx]),
        "forbidden_2026_row_count": 0,
    }
    if quality["duplicate_original_date_symbol_count"]:
        raise AssertionError("landmark cohort is not date-symbol unique")
    quality_path = age_root / "quality.json"
    _write_json(quality_path, quality)
    completed = {
        "schema": AGE_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "study_config_sha256": study_hash,
        "age": int(age),
        "quality": quality,
        "files": {
            "daily_baseline": baseline_record,
            "daily_partial": partial_record,
            "quality": _file_record(quality_path),
        },
    }
    _write_json(complete_path, completed)
    del target_matrix, controls, baseline_frame, partial_frame
    gc.collect()
    return completed


def _load_age_frames(
    completed: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = dict(completed["files"])
    baseline = pd.read_parquet(_resolve(files["daily_baseline"]["path"]))
    partial = pd.read_parquet(_resolve(files["daily_partial"]["path"]))
    return baseline, partial


def _hac_record(values: pd.Series, lag: int) -> dict[str, Any]:
    return base._hac_mean_test(values.to_numpy(dtype=np.float64), maximum_lag=lag)


def _aggregate_baseline(frame: pd.DataFrame, *, lag: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["year", "age", "stratum", "target"]
    for group_key, group in frame.groupby(keys, sort=True):
        row = dict(zip(keys, group_key, strict=True))
        row["date_count"] = int(group["date_idx"].nunique())
        row["row_count_mean"] = float(group["row_count"].mean())
        row["rank_ic"] = _hac_record(group["rank_ic"], lag)
        row["top5_lift"] = _hac_record(group["top5_lift"], lag)
        row["target_mean"] = float(group["target_mean"].mean())
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    result = np.full(p.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(p)
    if not bool(finite.any()):
        return result
    current = p[finite]
    order = np.argsort(current, kind="mergesort")
    ranked = current[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[finite] = restored
    return result


def _aggregate_partial(frame: pd.DataFrame, *, lag: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["year", "age", "stratum", "target", "feature", "block"]
    for group_key, group in frame.groupby(keys, sort=True):
        row = dict(zip(keys, group_key, strict=True))
        row["date_count"] = int(group["date_idx"].nunique())
        valid_date = np.isfinite(group["partial_rank_ic"].to_numpy(dtype=np.float64))
        row["valid_date_count"] = int(valid_date.sum())
        row["valid_date_fraction"] = float(valid_date.mean())
        row["row_count_mean"] = float(group.loc[valid_date, "row_count"].mean())
        if not bool(valid_date.any()):
            row["row_count_mean"] = math.nan
        partial = _hac_record(group["partial_rank_ic"], lag)
        spread = _hac_record(group["residual_quintile_spread"], lag)
        row.update(
            {
                "partial_rank_ic_mean": float(partial["mean"]),
                "partial_rank_ic_standard_error": float(partial["standard_error"]),
                "partial_rank_ic_t_statistic": float(partial["t_statistic"]),
                "partial_rank_ic_p_value": float(partial["p_value_two_sided"]),
                "residual_quintile_spread_mean": float(spread["mean"]),
                "residual_quintile_spread_p_value": float(
                    spread["p_value_two_sided"]
                ),
            }
        )
        rows.append(row)
    result = pd.DataFrame.from_records(rows)
    result["partial_rank_ic_q_value"] = np.nan
    fdr_keys = ["year", "age", "stratum", "target"]
    for index in result.groupby(fdr_keys, sort=False).groups.values():
        positions = np.asarray(list(index), dtype=np.int64)
        result.loc[positions, "partial_rank_ic_q_value"] = _benjamini_hochberg(
            result.loc[positions, "partial_rank_ic_p_value"].to_numpy(
                dtype=np.float64
            )
        )
    return result


def _stable_evidence(
    annual: pd.DataFrame, study: Mapping[str, Any]
) -> pd.DataFrame:
    rules = dict(study["decision"])
    alpha = float(rules["maximum_hac_p_value"])
    fdr_alpha = float(study["evaluation"]["fdr_alpha"])
    keys = ["age", "stratum", "target", "feature", "block"]
    rows: list[dict[str, Any]] = []
    for group_key, group in annual.groupby(keys, sort=True):
        group = group.sort_values("year")
        if tuple(group["year"].astype(int)) != FOLD_YEARS:
            continue
        means = group["partial_rank_ic_mean"].to_numpy(dtype=np.float64)
        signs = np.sign(means)
        same_sign = bool(np.all(signs == signs[0]) and signs[0] != 0.0)
        hac_years = int(np.sum(group["partial_rank_ic_p_value"] <= alpha))
        fdr_years = int(np.sum(group["partial_rank_ic_q_value"] <= fdr_alpha))
        worst_abs = float(np.min(np.abs(means)))
        coverage = group["valid_date_fraction"].to_numpy(dtype=np.float64)
        statistical_support = (
            same_sign
            and len(means) >= int(rules["minimum_same_sign_years"])
            and hac_years >= int(rules["minimum_hac_supported_years"])
            and fdr_years >= int(rules["minimum_fdr_supported_years"])
            and worst_abs >= float(rules["minimum_abs_worst_year_partial_ic"])
        )
        coverage_supported = bool(
            np.all(
                coverage
                >= float(rules["minimum_valid_date_fraction_each_year"])
            )
        )
        stable = statistical_support and coverage_supported
        rows.append(
            {
                **dict(zip(keys, group_key, strict=True)),
                "partial_rank_ic_2023_2024_2025": means.tolist(),
                "p_value_2023_2024_2025": group[
                    "partial_rank_ic_p_value"
                ].tolist(),
                "q_value_2023_2024_2025": group[
                    "partial_rank_ic_q_value"
                ].tolist(),
                "valid_date_fraction_2023_2024_2025": coverage.tolist(),
                "minimum_valid_date_fraction": float(np.min(coverage)),
                "same_sign_years": int(np.sum(signs == signs[0]))
                if signs[0] != 0.0
                else 0,
                "hac_supported_years": hac_years,
                "fdr_supported_years": fdr_years,
                "worst_year_absolute_partial_ic": worst_abs,
                "mean_absolute_partial_ic": float(np.mean(np.abs(means))),
                "statistically_supported_before_coverage": bool(
                    statistical_support
                ),
                "coverage_supported": coverage_supported,
                "sparse_event_coordinate": bool(
                    statistical_support and not coverage_supported
                ),
                "stable": bool(stable),
            }
        )
    return pd.DataFrame.from_records(rows)


def _build_decision(
    stable_frame: pd.DataFrame, study: Mapping[str, Any]
) -> dict[str, Any]:
    stable = stable_frame[stable_frame["stable"]].copy()
    sparse_event = stable_frame[stable_frame["sparse_event_coordinate"]].copy()
    candidate_strata = {
        "entry_mfe10_top5",
        "entry_mfe20_top5",
        "entry_both_top5",
        "entry_either_top5",
        "entry_short_only_top5",
        "entry_long_only_top5",
    }
    path_blocks = {"B2_realized_price_path", "B3_activity_and_tradability"}
    path = stable[
        stable["block"].isin(path_blocks)
        & stable["stratum"].isin(candidate_strata)
    ]
    memory = stable[
        (stable["block"] == "B1_entry_memory")
        & stable["stratum"].isin(candidate_strata)
    ]
    opportunity_targets = {"mfe_10", "mfe_20"}
    path_quality_targets = {
        "state_10",
        "pre_peak_mae_10",
        "pre_peak_mae_20",
    }
    path_opportunity = path[path["target"].isin(opportunity_targets)]
    path_quality = path[path["target"].isin(path_quality_targets)]
    supported_ages = sorted(int(value) for value in path["age"].unique())
    if path.empty:
        verdict = (
            "entry_memory_only"
            if not memory.empty
            else "current_contract_conditionally_sufficient"
        )
    elif path_opportunity.empty and not path_quality.empty:
        verdict = "path_updates_state_or_risk_only"
    elif len(supported_ages) < 2:
        verdict = "age_specific_path_increment"
    else:
        verdict = "broad_post_entry_updater_candidate"
    strongest = stable.sort_values(
        ["mean_absolute_partial_ic", "fdr_supported_years"],
        ascending=[False, False],
    ).head(100)
    return {
        "status": "completed_policy_free_incremental_information_audit",
        "verdict": verdict,
        "frozen_entry_contract_rejected_as_conditionally_sufficient": bool(
            not path.empty
        ),
        "dedicated_path_updater_candidate": bool(not path.empty),
        "supported_path_ages": supported_ages,
        "stable_coordinate_count": len(stable),
        "stable_candidate_path_coordinate_count": len(path),
        "stable_candidate_entry_memory_coordinate_count": len(memory),
        "stable_candidate_path_opportunity_coordinate_count": len(
            path_opportunity
        ),
        "stable_candidate_path_quality_or_risk_coordinate_count": len(
            path_quality
        ),
        "sparse_event_coordinate_count": len(sparse_event),
        "strongest_sparse_event_coordinates": sparse_event.sort_values(
            "mean_absolute_partial_ic", ascending=False
        )
        .head(30)
        .to_dict(orient="records"),
        "strongest_stable_coordinates": strongest.to_dict(orient="records"),
        "next_step": (
            "fit matched-capacity nested OOS challengers A versus A+B for only the supported ages, targets, and feature blocks"
            if not path.empty
            else "retain the daily recomputed frozen entry contract; do not train a dedicated path updater"
        ),
        "does_not_select": list(study["non_selections"]),
    }


def _compact_record(
    *,
    summary: Mapping[str, Any],
    stable: pd.DataFrame,
) -> dict[str, Any]:
    decision = dict(summary["decision"])
    strongest = stable[stable["stable"]].sort_values(
        "mean_absolute_partial_ic", ascending=False
    ).head(30)
    return {
        "schema_version": 1,
        "artifact_type": "seq100_post_entry_incremental_information_research_record",
        "study_id": STUDY_ID,
        "status": summary["status"],
        "completed_at": summary["completed_at"],
        "question": "Does original entry memory or realized D1/D3/D5 path add stable conditional information beyond recomputing the frozen five-block entry contract at the current close?",
        "scope": summary["scope"],
        "decision": decision,
        "data_quality": summary["data_quality"],
        "strongest_stable_coordinates": strongest.to_dict(orient="records"),
        "limitations": [
            "The 2023-2025 folds are repeatedly reused research evidence, not a pristine final holdout.",
            "Partial rank evidence is a policy-free conditional-information diagnostic, not a realizable holding return or exit rule.",
            "The holding MFE target includes legal D1 closes for an existing position; the matched D2 diagnostic exists only to quantify the T+1 semantic difference from a new entry.",
            "Stable-coordinate counts include correlated path transformations and overlapping entry strata; they are evidence instances, not independent discoveries.",
            "Stable conditional coordinates justify only a matched-capacity nested prediction test, not a fused continuation value or account policy.",
        ],
        "outputs": {
            "summary": "daily_research/output/path_policy/studies/seq100_post_entry_incremental_information_v1/summary.json",
            "decision": "daily_research/output/path_policy/studies/seq100_post_entry_incremental_information_v1/decision.json",
            "annual_baseline": "daily_research/output/path_policy/studies/seq100_post_entry_incremental_information_v1/annual_baseline.parquet",
            "annual_partial": "daily_research/output/path_policy/studies/seq100_post_entry_incremental_information_v1/annual_partial.parquet",
            "stable_evidence": "daily_research/output/path_policy/studies/seq100_post_entry_incremental_information_v1/stable_evidence.parquet",
        },
    }


def run_audit(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    record_root: Path = DEFAULT_RECORD_ROOT,
) -> dict[str, Any]:
    started = datetime.now().astimezone()
    study = load_study(study_path)
    study_hash = _config_hash(study_path)
    sources = dict(study["sources"])
    entry_decision = json.loads(
        _resolve(sources["entry_role_decision"]).read_text(encoding="utf-8")
    )
    expected_coordinates = [
        "mfe_10",
        "mfe_20",
        "state_10",
        "pre_peak_mae_10",
        "pre_peak_mae_20",
    ]
    if [row["name"] for row in entry_decision["entry_output_coordinates"]] != expected_coordinates:
        raise ValueError("frozen entry output contract changed")
    feature_study = feature_audit.load_study(_resolve(sources["feature_study"]))
    path_study = base.load_study(_resolve(sources["path_study"]))
    inputs, pack = feature_audit._load_validated_inputs(feature_study)
    global _FEATURE_INPUT_CACHE
    _FEATURE_INPUT_CACHE = inputs
    if inputs.maximum_outcome_date != study["folds"]["maximum_outcome_date"]:
        raise ValueError("input outcome boundary differs from study")
    contract = load_frozen_contract(
        study=study, feature_study=feature_study, path_study=path_study
    )
    reader = HoldingPathReader(pack, inputs)
    candidate_keys = (
        np.asarray(inputs.candidate_date_idx, dtype=np.int64) * reader.symbol_count
        + np.asarray(inputs.candidate_symbol_idx, dtype=np.int64)
    )
    if bool(np.any(candidate_keys[1:] <= candidate_keys[:-1])):
        raise ValueError("candidate date-symbol keys are not unique and ordered")

    completed_ages: list[dict[str, Any]] = []
    for age in LANDMARK_AGES:
        completed = _run_age(
            age=age,
            study=study,
            study_hash=study_hash,
            output_root=output_root,
            inputs=inputs,
            contract=contract,
            reader=reader,
            candidate_keys=candidate_keys,
        )
        completed_ages.append(completed)
        print(
            json.dumps(
                {
                    "event": "landmark_completed",
                    "age": age,
                    "primary_common_support": completed["quality"]["counts"][
                        "primary_common_support"
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    baseline_frames: list[pd.DataFrame] = []
    partial_frames: list[pd.DataFrame] = []
    for completed in completed_ages:
        baseline, partial = _load_age_frames(completed)
        baseline_frames.append(baseline)
        partial_frames.append(partial)
    daily_baseline = pd.concat(baseline_frames, ignore_index=True)
    daily_partial = pd.concat(partial_frames, ignore_index=True)
    lag = int(study["evaluation"]["hac_lag"])
    annual_baseline = _aggregate_baseline(daily_baseline, lag=lag)
    annual_partial = _aggregate_partial(daily_partial, lag=lag)
    stable = _stable_evidence(annual_partial, study)
    decision = _build_decision(stable, study)
    annual_baseline_file = _write_parquet(
        output_root / "annual_baseline.parquet", annual_baseline
    )
    annual_partial_file = _write_parquet(
        output_root / "annual_partial.parquet", annual_partial
    )
    stable_file = _write_parquet(output_root / "stable_evidence.parquet", stable)
    data_quality = [dict(item["quality"]) for item in completed_ages]
    if any(row["forbidden_2026_row_count"] for row in data_quality):
        raise AssertionError("audit consumed a 2026 row")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed_without_training_or_policy_selection",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "scope": {
            "candidate_universe_count": int(inputs.candidate_count),
            "oos_original_entry_count": len(contract.rows),
            "fold_years": list(FOLD_YEARS),
            "landmark_ages": list(LANDMARK_AGES),
            "maximum_consumed_outcome_date": inputs.maximum_outcome_date,
            "new_booster_count": 0,
            "retained_meta_model_count": 0,
            "frozen_contract_prediction_blocks_reused": 5 * len(FOLD_YEARS),
            "consumed_prediction_and_model_file_count": len(
                contract.consumed_files
            ),
            "frozen_contract_source_alignment": list(contract.source_alignment),
        },
        "target_semantics": dict(study["targets"]),
        "data_quality": data_quality,
        "baseline_summary": annual_baseline.to_dict(orient="records"),
        "conditional_increment_summary": {
            "annual_record_count": len(annual_partial),
            "stable_test_count": int(stable["stable"].sum()),
            "sparse_event_test_count": int(
                stable["sparse_event_coordinate"].sum()
            ),
            "stable_by_block": {
                str(name): int(count)
                for name, count in stable[stable["stable"]]["block"]
                .value_counts()
                .sort_index()
                .items()
            },
            "stable_by_target": {
                str(name): int(count)
                for name, count in stable[stable["stable"]]["target"]
                .value_counts()
                .sort_index()
                .items()
            },
        },
        "decision": decision,
        "files": {
            "annual_baseline": annual_baseline_file,
            "annual_partial": annual_partial_file,
            "stable_evidence": stable_file,
        },
        "runtime": {
            "elapsed_seconds": (
                datetime.now().astimezone() - started
            ).total_seconds()
        },
        "disclosure": {
            "2023_2025_reuse": "These owner-authorized folds have been reused for research decisions; this is comparative recent-market evidence, not a pristine-holdout claim.",
            "2026": "No 2026 row, outcome, feature, prediction, metric, or decision input was read.",
            "causality": "Original entry scores are OOS; realized path features end at the landmark close; targets start at the following open.",
            "population": "Every original OOS filled entry remains in landmark accounting regardless of interim path. Missing current-contract states are reported before common-support analysis.",
            "dependence": "Coordinate counts span correlated path transformations and overlapping entry strata; they must not be interpreted as independent factors.",
            "policy_free": "No fused score, holding model, exit, switching rule, cost buffer, slot count, or account policy was fit or selected.",
        },
    }
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "decision.json", decision)
    record = _compact_record(summary=summary, stable=stable)
    _write_json(record_root / "result.json", record)
    return summary


def self_test() -> dict[str, Any]:
    close = np.asarray(
        [[0.20, 0.05, 0.10], [-0.05, -0.10, -0.02]], dtype=np.float64
    )
    sellable = np.ones_like(close, dtype=bool)
    valid = np.ones(2, dtype=bool)
    _, holding, holding_risk = _numeric_path_labels(
        close, sellable, valid, first_legal_day=1
    )
    _, entry, entry_risk = _numeric_path_labels(
        close, sellable, valid, first_legal_day=2
    )
    if not (holding[0] > entry[0] and holding_risk[0] == 0.0):
        raise AssertionError("holding D1 legality semantics drifted")
    if not (holding[1] == entry[1] and entry_risk[1] == -0.10):
        raise AssertionError("matched path risk semantics drifted")

    source = np.asarray([10, 20, 30, 40], dtype=np.int64)
    rows, found = _candidate_lookup(source, np.asarray([20, 25, 40]))
    if rows.tolist() != [1, -1, 3] or found.tolist() != [True, False, True]:
        raise AssertionError("candidate lookup drifted")

    rng = np.random.default_rng(7)
    controls = rng.normal(size=(500, 3))
    feature = rng.normal(size=500)
    target = 2.0 * controls[:, 0] + feature + rng.normal(scale=0.05, size=500)
    partial = _partial_rank_statistics(
        controls,
        feature,
        target,
        minimum_rows=20,
        quantile_count=5,
    )
    if float(partial["partial_rank_ic"]) < 0.90:
        raise AssertionError("partial-rank increment drifted")
    return {"status": "passed", "test_count": 3}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit incremental post-entry information at D1/D3/D5."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("command", choices=("self-test", "audit"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "self-test":
        result = self_test()
    else:
        summary = run_audit(
            study_path=args.config.resolve(),
            output_root=args.output_root.resolve(),
            record_root=args.record_root.resolve(),
        )
        result = {
            "status": summary["status"],
            "study_id": summary["study_id"],
            "decision": summary["decision"],
            "runtime": summary["runtime"],
        }
    print(json.dumps(result, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
