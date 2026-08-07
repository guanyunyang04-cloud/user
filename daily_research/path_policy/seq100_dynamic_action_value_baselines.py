"""Strictly causal transparent baselines for dynamic buy action value.

The study learns only from annual prefix-oracle experiences that were available
before each out-of-sample year and remained at least ten sessions past branch
coalescence.  It decomposes action value into a date-level market component and
within-date stock-state effects, then compares transparent shrinkage and
matched-state estimators without selecting a portfolio policy.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from scipy.stats import rankdata

from daily_research.path_policy import seq100_market_replay as replay

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = WORKSPACE_ROOT / (
    "daily_research/studies/seq100_dynamic_action_value_baselines_v1.json"
)
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / (
    "daily_research/output/path_policy/studies/"
    "seq100_dynamic_action_value_baselines_v1"
)
STUDY_ID = "seq100_dynamic_action_value_baselines_v1"
SCHEMA_VERSION = 1

VERSION_COLUMNS = (
    "cost_scenario",
    "version_role",
    "signal_year",
    "as_of_year",
    "as_of_date",
    "as_of_date_idx",
    "trade_date",
    "date_idx",
    "symbol_idx",
    "prefix_oracle_cash_buy_selected",
    "buy_advantage_vs_cash",
    "cash_resolution_date_idx",
    "sessions_after_resolution",
    "naturally_resolved",
)


def _resolve(path: str | Path) -> Path:
    return replay.resolve_path(path)


def _read_json(path: str | Path) -> dict[str, Any]:
    return replay.read_json(path)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _file_record(path: str | Path) -> dict[str, Any]:
    target = _resolve(path)
    return {
        "path": str(target.resolve()),
        "sha256": replay.sha256(target),
        "size": int(target.stat().st_size),
    }


def _write_frame(
    path: str | Path, frame: pd.DataFrame, *, row_group_size: int
) -> dict[str, Any]:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    frame.to_parquet(
        temporary,
        index=False,
        compression="zstd",
        row_group_size=int(row_group_size),
    )
    os.replace(temporary, target)
    return {**_file_record(target), "rows": len(frame)}


def _write_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, target)
    return _file_record(target)


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _read_json(path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("dynamic_action_baseline_study_id")
    source = dict(study["source"])
    if str(source["quality_pool_name"]) != "quality_liquidity_pit":
        raise ValueError("dynamic_action_baseline_pool")
    if int(source["expected_feature_count"]) != 557:
        raise ValueError("dynamic_action_baseline_feature_count")
    if int(source["forbidden_year"]) != 2026:
        raise ValueError("dynamic_action_baseline_forbidden_year")
    target = dict(study["target"])
    if int(target["minimum_sessions_after_resolution"]) != 10:
        raise ValueError("dynamic_action_baseline_maturity")
    if not bool(target["continuous_target"]):
        raise ValueError("dynamic_action_baseline_continuous_target")
    if bool(target["fixed_holding_horizon_used"]):
        raise ValueError("dynamic_action_baseline_fixed_horizon")
    if bool(target["binary_good_stock_label_used"]):
        raise ValueError("dynamic_action_baseline_binary_label")
    if bool(target["final_2025_oracle_row_labels_allowed"]):
        raise ValueError("dynamic_action_baseline_final_oracle_labels")
    coordinates = dict(study["coordinates"])
    specs = list(coordinates["specifications"])
    names = [str(item["name"]) for item in specs]
    if len(names) != len(set(names)) or not names:
        raise ValueError("dynamic_action_baseline_coordinate_names")
    families = {str(item["family"]) for item in specs}
    if "market" not in families or len(families) < 2:
        raise ValueError("dynamic_action_baseline_coordinate_families")
    matched = list(coordinates["matched_coordinates"])
    if not matched or not set(matched).issubset(names):
        raise ValueError("dynamic_action_baseline_matched_coordinates")
    estimators = dict(study["estimators"])
    sensitivity = tuple(
        float(value)
        for value in estimators["matched_smoothing_sensitivity_date_equivalents"]
    )
    if sensitivity != (0.0, 5.0, 20.0, 50.0, 100.0, 200.0):
        raise ValueError("dynamic_action_baseline_smoothing_sensitivity")
    if float(estimators["matched_cell_smoothing_date_equivalents"]) not in sensitivity:
        raise ValueError("dynamic_action_baseline_fixed_smoothing_absent")
    period = dict(study["period"])
    years = tuple(int(value) for value in period["formal_years"])
    oos_years = tuple(int(value) for value in period["strict_oos_years"])
    if years != tuple(range(2012, 2026)) or oos_years != tuple(range(2013, 2026)):
        raise ValueError("dynamic_action_baseline_period")
    if bool(period["daily_first_availability_claimed"]):
        raise ValueError("dynamic_action_baseline_daily_availability")
    return study


def _record_by_key(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    result: dict[tuple[str, int, int], dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        key = (
            str(record["cost_scenario"]),
            int(record["signal_year"]),
            int(record["as_of_year"]),
        )
        if key in result:
            raise ValueError("dynamic_action_baseline_duplicate_version")
        result[key] = record
    return result


def _source_contract(
    study: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[tuple[str, int, int], dict[str, Any]],
    dict[str, int],
]:
    source = dict(study["source"])
    requests = (
        (
            "causal_prefix_manifest",
            "expected_causal_prefix_manifest_sha256",
        ),
        (
            "causal_prefix_validation",
            "expected_causal_prefix_validation_sha256",
        ),
        ("model_input_manifest", "expected_model_input_manifest_sha256"),
    )
    contract: dict[str, Any] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for path_key, hash_key in requests:
        path = _resolve(source[path_key])
        digest = replay.sha256(path)
        if digest != str(source[hash_key]):
            raise ValueError(f"dynamic_action_baseline_source_hash:{path_key}")
        contract[path_key] = {"path": str(path.resolve()), "sha256": digest}
        payloads[path_key] = _read_json(path)

    prefix_manifest = payloads["causal_prefix_manifest"]
    prefix_validation = payloads["causal_prefix_validation"]
    input_manifest = payloads["model_input_manifest"]
    if prefix_manifest.get("status") != "completed":
        raise ValueError("dynamic_action_baseline_prefix_incomplete")
    if prefix_validation.get("status") != "passed":
        raise ValueError("dynamic_action_baseline_prefix_validation")
    if input_manifest.get("status") != "completed":
        raise ValueError("dynamic_action_baseline_input_incomplete")
    if bool(prefix_manifest.get("daily_first_availability_claimed")):
        raise ValueError("dynamic_action_baseline_bad_availability_claim")
    if bool(prefix_manifest.get("final_oracle_row_labels_in_causal_outputs")):
        raise ValueError("dynamic_action_baseline_final_label_leak")

    compact_names = list(
        dict(input_manifest["feature_groups"])[str(source["feature_variant"])]
    )
    features = list(input_manifest["features"])
    if (
        len(compact_names) != int(source["expected_feature_count"])
        or [str(item["feature_name"]) for item in features] != compact_names
    ):
        raise ValueError("dynamic_action_baseline_feature_contract")
    feature_index = {
        str(item["feature_name"]): int(item["column_index"]) for item in features
    }
    required = {
        str(source_name)
        for spec in dict(study["coordinates"])["specifications"]
        for source_name in spec["sources"]
    }
    missing = required - set(feature_index)
    if missing:
        raise ValueError(f"dynamic_action_baseline_missing_features:{sorted(missing)}")

    storage = dict(dict(input_manifest["storage"])["compact"])
    shape = tuple(int(value) for value in storage["shape"])
    feature_path = Path(str(storage["path"]))
    if shape != (int(input_manifest["row_count"]), len(compact_names)):
        raise ValueError("dynamic_action_baseline_storage_shape")
    expected_size = int(np.prod(shape) * np.dtype("float32").itemsize)
    if not feature_path.is_file() or feature_path.stat().st_size != expected_size:
        raise ValueError("dynamic_action_baseline_storage_file")
    row_record = dict(input_manifest["row_index"])
    row_path = Path(str(row_record["path"]))
    if not row_path.is_file() or replay.sha256(row_path) != str(row_record["sha256"]):
        raise ValueError("dynamic_action_baseline_row_index")

    contract["feature_storage"] = storage
    contract["row_index"] = row_record
    contract["feature_count"] = len(compact_names)
    records = _record_by_key(prefix_manifest["experience_versions"])
    return contract, prefix_manifest, input_manifest, records, feature_index


def _packed_key(date_idx: np.ndarray, symbol_idx: np.ndarray) -> np.ndarray:
    dates = np.asarray(date_idx, dtype=np.int64)
    symbols = np.asarray(symbol_idx, dtype=np.int64)
    if bool((dates < 0).any()) or bool((symbols < 0).any()):
        raise ValueError("dynamic_action_baseline_negative_key")
    return (dates << np.int64(32)) | symbols


def _same_date_percentile(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64), copy=False)
    groups = series.groupby(pd.Series(dates, copy=False), sort=False)
    ranks = groups.rank(method="average")
    counts = groups.transform("count")
    result = ((ranks - 0.5) / counts).to_numpy(np.float64)
    result[~np.isfinite(series.to_numpy(np.float64))] = np.nan
    return result


def _causal_expanding_percentile(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    result = np.full(len(source), np.nan, dtype=np.float64)
    for index, value in enumerate(source):
        if not np.isfinite(value):
            continue
        history = source[: index + 1]
        finite = np.isfinite(history)
        count = int(finite.sum())
        if count:
            comparable = history[finite]
            result[index] = (
                float(np.count_nonzero(comparable < value))
                + 0.5 * float(np.count_nonzero(comparable == value))
            ) / count
    return result


def _coordinate_metadata(study: Mapping[str, Any]) -> tuple[
    list[dict[str, Any]],
    list[str],
    list[str],
    dict[str, list[int]],
    np.ndarray,
]:
    specs = [dict(item) for item in dict(study["coordinates"])["specifications"]]
    names = [str(item["name"]) for item in specs]
    market_names = [str(item["name"]) for item in specs if item["family"] == "market"]
    stock_names = [str(item["name"]) for item in specs if item["family"] != "market"]
    stock_name_to_index = {name: index for index, name in enumerate(stock_names)}
    families: dict[str, list[int]] = {}
    for item in specs:
        family = str(item["family"])
        if family == "market":
            continue
        families.setdefault(family, []).append(stock_name_to_index[str(item["name"])])
    name_to_all = {name: index for index, name in enumerate(names)}
    matched = np.asarray(
        [name_to_all[name] for name in dict(study["coordinates"])["matched_coordinates"]],
        dtype=np.int16,
    )
    return specs, names, market_names, families, matched


def build_coordinate_partitions(
    *,
    study: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    feature_index: Mapping[str, int],
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_group_size = int(dict(study["resources"])["parquet_row_group_size"])
    specs, coordinate_names, _market_names, _families, _matched = (
        _coordinate_metadata(study)
    )
    row_record = dict(input_manifest["row_index"])
    row_index = pd.read_parquet(
        row_record["path"], columns=["date_idx", "symbol_idx", "trade_date"]
    )
    if len(row_index) != int(input_manifest["row_count"]):
        raise ValueError("dynamic_action_baseline_row_count")
    keys = _packed_key(row_index["date_idx"], row_index["symbol_idx"])
    if bool((keys[1:] <= keys[:-1]).any()):
        raise ValueError("dynamic_action_baseline_row_order")
    years = row_index["trade_date"].astype(str).str[:4].astype(np.int16).to_numpy()
    if bool((years >= 2026).any()):
        raise ValueError("dynamic_action_baseline_forbidden_input_year")

    storage = dict(dict(input_manifest["storage"])["compact"])
    shape = tuple(int(value) for value in storage["shape"])
    feature_values = np.memmap(
        storage["path"], dtype=np.float32, mode="r", shape=shape
    )
    source_names = list(
        dict.fromkeys(
            str(source)
            for spec in specs
            for source in list(spec["sources"])
        )
    )
    source_indices = np.asarray(
        [int(feature_index[name]) for name in source_names], dtype=np.int64
    )
    source_to_local = {name: index for index, name in enumerate(source_names)}

    date_values = row_index["date_idx"].to_numpy(np.int64)
    unique_dates, date_starts = np.unique(date_values, return_index=True)
    date_ends = np.r_[date_starts[1:] - 1, len(row_index) - 1]
    market_specs = [item for item in specs if item["family"] == "market"]
    market_indices = np.asarray(
        [feature_index[str(item["sources"][0])] for item in market_specs],
        dtype=np.int64,
    )
    market_first = np.asarray(
        feature_values[date_starts[:, None], market_indices[None, :]],
        dtype=np.float64,
    )
    market_last = np.asarray(
        feature_values[date_ends[:, None], market_indices[None, :]],
        dtype=np.float64,
    )
    if not np.allclose(market_first, market_last, atol=1.0e-7, rtol=0.0, equal_nan=True):
        raise ValueError("dynamic_action_baseline_market_not_date_constant")
    market_percentiles = {
        str(spec["name"]): _causal_expanding_percentile(market_first[:, index])
        for index, spec in enumerate(market_specs)
    }

    records: list[dict[str, Any]] = []
    missing_rows: dict[str, int] = {name: 0 for name in coordinate_names}
    formal_years = [int(value) for value in dict(study["period"])["formal_years"]]
    for year in formal_years:
        positions = np.flatnonzero(years == year)
        if not len(positions):
            raise ValueError(f"dynamic_action_baseline_year_absent:{year}")
        start = int(positions[0])
        stop = int(positions[-1]) + 1
        if stop - start != len(positions):
            raise ValueError("dynamic_action_baseline_year_not_contiguous")
        raw = np.asarray(
            feature_values[start:stop, source_indices], dtype=np.float64
        )
        local_dates = date_values[start:stop]
        date_offsets = np.searchsorted(unique_dates, local_dates)
        coordinate_values: dict[str, np.ndarray] = {}
        for spec in specs:
            name = str(spec["name"])
            transform = str(spec["transform"])
            sources = [str(value) for value in spec["sources"]]
            if transform == "identity_percentile":
                values = raw[:, source_to_local[sources[0]]].copy()
                finite = np.isfinite(values)
                if bool(((values[finite] < -1.0e-6) | (values[finite] > 1.000001)).any()):
                    raise ValueError(f"dynamic_action_baseline_percentile_range:{name}")
                values[finite] = np.clip(values[finite], 0.0, 1.0)
            elif transform == "same_date_percentile":
                values = _same_date_percentile(
                    raw[:, source_to_local[sources[0]]], local_dates
                )
            elif transform == "causal_expanding_date_percentile":
                values = market_percentiles[name][date_offsets]
            elif transform == "large_moneyflow_imbalance_same_date_percentile":
                buy = (
                    raw[:, source_to_local[sources[0]]]
                    + raw[:, source_to_local[sources[2]]]
                )
                sell = (
                    raw[:, source_to_local[sources[1]]]
                    + raw[:, source_to_local[sources[3]]]
                )
                denominator = buy + sell
                imbalance = np.divide(
                    buy - sell,
                    denominator,
                    out=np.full(len(raw), np.nan, dtype=np.float64),
                    where=np.isfinite(denominator) & (denominator > 0.0),
                )
                values = _same_date_percentile(imbalance, local_dates)
            else:
                raise ValueError(f"dynamic_action_baseline_transform:{transform}")
            coordinate_values[name] = values.astype(np.float32)
            missing_rows[name] += int((~np.isfinite(values)).sum())

        output = pd.DataFrame(
            {
                "input_row_idx": np.arange(start, stop, dtype=np.int64),
                "trade_date": row_index["trade_date"].iloc[start:stop].astype(str).to_numpy(),
                "date_idx": row_index["date_idx"].iloc[start:stop].to_numpy(np.int32),
                "symbol_idx": row_index["symbol_idx"].iloc[start:stop].to_numpy(np.int32),
                **coordinate_values,
            }
        )
        path = output_root / "coordinates" / f"year={year}" / "part-0000.parquet"
        record = _write_frame(path, output, row_group_size=row_group_size)
        record.update(
            {
                "year": year,
                "first_date": str(output["trade_date"].iloc[0]),
                "last_date": str(output["trade_date"].iloc[-1]),
            }
        )
        records.append(record)
        del raw, coordinate_values, output
        gc.collect()

    del feature_values
    audit = {
        "rows": int(sum(int(record["rows"]) for record in records)),
        "coordinate_count": len(coordinate_names),
        "coordinate_names": coordinate_names,
        "missing_rows": missing_rows,
        "forbidden_2026_rows": 0,
        "causal_market_percentiles": True,
        "same_date_stock_percentiles": True,
    }
    return records, audit


@dataclass(frozen=True)
class CoordinateYear:
    year: int
    frame: pd.DataFrame
    values: np.ndarray
    keys: np.ndarray


class CoordinateStore:
    def __init__(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        coordinate_names: Sequence[str],
        cache_years: int,
    ) -> None:
        self.records = {int(record["year"]): dict(record) for record in records}
        self.coordinate_names = list(coordinate_names)
        self.cache_years = max(int(cache_years), 1)
        self.cache: OrderedDict[int, CoordinateYear] = OrderedDict()

    def load(self, year: int) -> CoordinateYear:
        requested = int(year)
        cached = self.cache.get(requested)
        if cached is not None:
            self.cache.move_to_end(requested)
            return cached
        record = self.records[requested]
        path = Path(str(record["path"]))
        if replay.sha256(path) != str(record["sha256"]):
            raise ValueError("dynamic_action_baseline_coordinate_hash")
        columns = [
            "input_row_idx",
            "trade_date",
            "date_idx",
            "symbol_idx",
            *self.coordinate_names,
        ]
        frame = pd.read_parquet(path, columns=columns)
        if len(frame) != int(record["rows"]):
            raise ValueError("dynamic_action_baseline_coordinate_rows")
        keys = _packed_key(frame["date_idx"], frame["symbol_idx"])
        if bool((keys[1:] <= keys[:-1]).any()):
            raise ValueError("dynamic_action_baseline_coordinate_order")
        values = frame[self.coordinate_names].to_numpy(np.float32, copy=True)
        result = CoordinateYear(
            year=requested,
            frame=frame,
            values=values,
            keys=keys,
        )
        self.cache[requested] = result
        self.cache.move_to_end(requested)
        while len(self.cache) > self.cache_years:
            self.cache.popitem(last=False)
        return result

    def clear(self) -> None:
        self.cache.clear()
        gc.collect()


@dataclass(frozen=True)
class AlignedExperience:
    cost_scenario: str
    signal_year: int
    as_of_year: int
    version_role: str
    coordinate_positions: np.ndarray
    coordinates: np.ndarray
    trade_date: np.ndarray
    date_idx: np.ndarray
    symbol_idx: np.ndarray
    target: np.ndarray
    selected: np.ndarray
    sessions_after_resolution: np.ndarray
    source_rows: int
    maturity_rows: int
    missing_coordinate_rows: int


def _read_aligned_experience(
    *,
    record: Mapping[str, Any],
    coordinates: CoordinateYear,
    maturity_sessions: int,
) -> AlignedExperience:
    path = Path(str(record["path"]))
    if replay.sha256(path) != str(record["sha256"]):
        raise ValueError("dynamic_action_baseline_version_hash")
    frame = pd.read_parquet(path, columns=list(VERSION_COLUMNS))
    if len(frame) != int(record["rows"]):
        raise ValueError("dynamic_action_baseline_version_rows")
    natural = frame["naturally_resolved"].astype(bool).to_numpy()
    sessions = frame["sessions_after_resolution"].to_numpy(np.int64)
    eligible = natural & (sessions >= int(maturity_sessions))
    maturity_rows = int(eligible.sum())
    frame = frame.loc[eligible].reset_index(drop=True)
    sessions = sessions[eligible]
    label_keys = _packed_key(frame["date_idx"], frame["symbol_idx"])
    if len(label_keys) and bool((label_keys[1:] <= label_keys[:-1]).any()):
        raise ValueError("dynamic_action_baseline_version_order")
    offsets = np.searchsorted(coordinates.keys, label_keys)
    found = offsets < len(coordinates.keys)
    valid_offsets = offsets[found]
    found_indices = np.flatnonzero(found)
    found[found_indices] = coordinates.keys[valid_offsets] == label_keys[found_indices]
    positions = offsets[found].astype(np.int64, copy=False)
    aligned = frame.loc[found].reset_index(drop=True)
    aligned_sessions = sessions[found]
    if len(aligned) and not aligned["trade_date"].astype(str).str[:4].astype(int).eq(
        int(record["signal_year"])
    ).all():
        raise ValueError("dynamic_action_baseline_signal_year_alignment")
    target = aligned["buy_advantage_vs_cash"].to_numpy(np.float64)
    if not np.isfinite(target).all():
        raise ValueError("dynamic_action_baseline_nonfinite_target")
    return AlignedExperience(
        cost_scenario=str(record["cost_scenario"]),
        signal_year=int(record["signal_year"]),
        as_of_year=int(record["as_of_year"]),
        version_role=str(record["version_role"]),
        coordinate_positions=positions,
        coordinates=coordinates.values[positions].astype(np.float64, copy=False),
        trade_date=aligned["trade_date"].astype(str).to_numpy(),
        date_idx=aligned["date_idx"].to_numpy(np.int32),
        symbol_idx=aligned["symbol_idx"].to_numpy(np.int32),
        target=target,
        selected=aligned["prefix_oracle_cash_buy_selected"].astype(bool).to_numpy(),
        sessions_after_resolution=aligned_sessions.astype(np.int32, copy=False),
        source_rows=len(eligible),
        maturity_rows=maturity_rows,
        missing_coordinate_rows=int((~found).sum()),
    )


def _univariate_bins(values: np.ndarray, *, bin_count: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    result = np.full(source.shape, int(bin_count), dtype=np.int16)
    finite = np.isfinite(source)
    clipped = np.clip(source[finite], 0.0, 1.0)
    result[finite] = np.minimum(
        np.floor(clipped * int(bin_count)).astype(np.int16),
        int(bin_count) - 1,
    )
    return result


def _matched_codes(
    values: np.ndarray, *, matched_indices: np.ndarray, bin_count: int
) -> np.ndarray:
    selected = np.asarray(values, dtype=np.float64)[:, matched_indices]
    bins = _univariate_bins(selected, bin_count=int(bin_count)).astype(np.int64)
    base = int(bin_count) + 1
    multipliers = np.power(base, np.arange(bins.shape[1], dtype=np.int64))
    return (bins * multipliers[None, :]).sum(axis=1).astype(np.int32)


def _date_presence(codes: np.ndarray, group_index: np.ndarray, size: int) -> np.ndarray:
    packed = group_index.astype(np.int64) * int(size) + codes.astype(np.int64)
    unique = np.unique(packed)
    return np.bincount(unique % int(size), minlength=int(size)).astype(np.float64)


class ActionValueAccumulator:
    def __init__(
        self,
        *,
        market_indices: np.ndarray,
        stock_indices: np.ndarray,
        matched_indices: np.ndarray,
        univariate_bins: int,
        matched_bins: int,
    ) -> None:
        self.market_indices = np.asarray(market_indices, dtype=np.int16)
        self.stock_indices = np.asarray(stock_indices, dtype=np.int16)
        self.matched_indices = np.asarray(matched_indices, dtype=np.int16)
        self.univariate_bins = int(univariate_bins)
        self.matched_bins = int(matched_bins)
        self.bin_size = self.univariate_bins + 1
        self.cell_size = (self.matched_bins + 1) ** len(self.matched_indices)

        self.global_date_count = 0.0
        self.global_sum = 0.0
        self.global_sum2 = 0.0
        market_shape = (len(self.market_indices), self.bin_size)
        stock_shape = (len(self.stock_indices), self.bin_size)
        self.market_count = np.zeros(market_shape, dtype=np.float64)
        self.market_sum = np.zeros(market_shape, dtype=np.float64)
        self.market_sum2 = np.zeros(market_shape, dtype=np.float64)
        self.stock_weight = np.zeros(stock_shape, dtype=np.float64)
        self.stock_sum = np.zeros(stock_shape, dtype=np.float64)
        self.stock_sum2 = np.zeros(stock_shape, dtype=np.float64)
        self.stock_date_count = np.zeros(stock_shape, dtype=np.float64)
        self.cell_weight = np.zeros(self.cell_size, dtype=np.float64)
        self.cell_sum = np.zeros(self.cell_size, dtype=np.float64)
        self.cell_sum2 = np.zeros(self.cell_size, dtype=np.float64)
        self.cell_positive_weight = np.zeros(self.cell_size, dtype=np.float64)
        self.cell_date_count = np.zeros(self.cell_size, dtype=np.float64)
        self.training_rows = 0

    def apply(self, experience: AlignedExperience, *, sign: int) -> dict[str, Any]:
        direction = int(sign)
        if direction not in (-1, 1):
            raise ValueError("dynamic_action_baseline_accumulator_sign")
        y = np.asarray(experience.target, dtype=np.float64)
        x = np.asarray(experience.coordinates, dtype=np.float64)
        dates = np.asarray(experience.date_idx, dtype=np.int64)
        if not len(y):
            return {
                "rows": 0,
                "dates": 0,
                "missing_coordinate_rows": experience.missing_coordinate_rows,
            }
        if x.shape[0] != len(y) or bool((dates[1:] < dates[:-1]).any()):
            raise ValueError("dynamic_action_baseline_accumulator_alignment")
        _unique_dates, starts, counts = np.unique(
            dates, return_index=True, return_counts=True
        )
        group_index = np.repeat(np.arange(len(starts), dtype=np.int64), counts)
        daily_mean = np.add.reduceat(y, starts) / counts.astype(np.float64)
        residual = y - daily_mean[group_index]
        weights = 1.0 / counts[group_index].astype(np.float64)

        self.global_date_count += direction * len(starts)
        self.global_sum += direction * float(daily_mean.sum())
        self.global_sum2 += direction * float(np.square(daily_mean).sum())

        market_bins = _univariate_bins(
            x[starts][:, self.market_indices], bin_count=self.univariate_bins
        )
        for column in range(market_bins.shape[1]):
            code = market_bins[:, column]
            self.market_count[column] += direction * np.bincount(
                code, minlength=self.bin_size
            )
            self.market_sum[column] += direction * np.bincount(
                code, weights=daily_mean, minlength=self.bin_size
            )
            self.market_sum2[column] += direction * np.bincount(
                code, weights=np.square(daily_mean), minlength=self.bin_size
            )

        stock_bins = _univariate_bins(
            x[:, self.stock_indices], bin_count=self.univariate_bins
        )
        for column in range(stock_bins.shape[1]):
            code = stock_bins[:, column]
            self.stock_weight[column] += direction * np.bincount(
                code, weights=weights, minlength=self.bin_size
            )
            self.stock_sum[column] += direction * np.bincount(
                code, weights=weights * residual, minlength=self.bin_size
            )
            self.stock_sum2[column] += direction * np.bincount(
                code, weights=weights * np.square(residual), minlength=self.bin_size
            )
            self.stock_date_count[column] += direction * _date_presence(
                code, group_index, self.bin_size
            )

        cell_code = _matched_codes(
            x,
            matched_indices=self.matched_indices,
            bin_count=self.matched_bins,
        )
        self.cell_weight += direction * np.bincount(
            cell_code, weights=weights, minlength=self.cell_size
        )
        self.cell_sum += direction * np.bincount(
            cell_code, weights=weights * y, minlength=self.cell_size
        )
        self.cell_sum2 += direction * np.bincount(
            cell_code, weights=weights * np.square(y), minlength=self.cell_size
        )
        self.cell_positive_weight += direction * np.bincount(
            cell_code, weights=weights * (y > 0.0), minlength=self.cell_size
        )
        self.cell_date_count += direction * _date_presence(
            cell_code, group_index, self.cell_size
        )
        self.training_rows += direction * len(y)
        return {
            "rows": len(y),
            "dates": len(starts),
            "missing_coordinate_rows": experience.missing_coordinate_rows,
            "target_mean": float(y.mean()),
            "positive_fraction": float(np.mean(y > 0.0)),
        }

    def assert_valid(self) -> None:
        tolerance = 1.0e-8
        arrays = (
            self.market_count,
            self.stock_weight,
            self.stock_date_count,
            self.cell_weight,
            self.cell_date_count,
            self.cell_positive_weight,
        )
        if self.global_date_count <= 0.0 or self.training_rows <= 0:
            raise ValueError("dynamic_action_baseline_empty_training_state")
        if any(bool((values < -tolerance).any()) for values in arrays):
            raise ValueError("dynamic_action_baseline_negative_sufficient_stat")
        if bool((self.cell_positive_weight - self.cell_weight > tolerance).any()):
            raise ValueError("dynamic_action_baseline_positive_weight")


def _mean_and_se(
    *,
    count: np.ndarray,
    total: np.ndarray,
    total2: np.ndarray,
    date_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_mean = np.divide(
        total,
        count,
        out=np.zeros_like(total, dtype=np.float64),
        where=count > 1.0e-12,
    )
    variance = np.divide(
        total2,
        count,
        out=np.zeros_like(total2, dtype=np.float64),
        where=count > 1.0e-12,
    ) - np.square(raw_mean)
    standard_error = np.sqrt(
        np.maximum(variance, 0.0) / np.maximum(date_count, 1.0)
    )
    standard_error[count <= 1.0e-12] = np.nan
    return raw_mean, standard_error


def build_snapshot(
    *,
    accumulator: ActionValueAccumulator,
    study: Mapping[str, Any],
    coordinate_names: Sequence[str],
    market_indices: np.ndarray,
    stock_indices: np.ndarray,
    families: Mapping[str, Sequence[int]],
    matched_indices: np.ndarray,
    training_cutoff_year: int,
    maximum_label_as_of_year: int,
    maximum_signal_year: int,
) -> dict[str, np.ndarray]:
    accumulator.assert_valid()
    estimators = dict(study["estimators"])
    market_alpha = float(
        estimators["market_univariate_smoothing_date_equivalents"]
    )
    stock_alpha = float(
        estimators["stock_univariate_smoothing_date_equivalents"]
    )
    matched_alpha = float(
        estimators["matched_cell_smoothing_date_equivalents"]
    )
    prior = accumulator.global_sum / accumulator.global_date_count
    prior_variance = max(
        accumulator.global_sum2 / accumulator.global_date_count - prior * prior,
        0.0,
    )
    prior_se = math.sqrt(prior_variance / accumulator.global_date_count)

    market_mean, market_raw_se = _mean_and_se(
        count=accumulator.market_count,
        total=accumulator.market_sum,
        total2=accumulator.market_sum2,
        date_count=accumulator.market_count,
    )
    market_lambda = accumulator.market_count / (
        accumulator.market_count + market_alpha
    )
    market_effect = market_lambda * (market_mean - prior)
    market_se = market_lambda * np.nan_to_num(market_raw_se, nan=prior_se)

    stock_mean, stock_raw_se = _mean_and_se(
        count=accumulator.stock_weight,
        total=accumulator.stock_sum,
        total2=accumulator.stock_sum2,
        date_count=accumulator.stock_date_count,
    )
    stock_lambda = accumulator.stock_weight / (
        accumulator.stock_weight + stock_alpha
    )
    stock_effect = stock_lambda * stock_mean
    stock_se = stock_lambda * np.nan_to_num(stock_raw_se, nan=prior_se)

    cell_mean, cell_raw_se = _mean_and_se(
        count=accumulator.cell_weight,
        total=accumulator.cell_sum,
        total2=accumulator.cell_sum2,
        date_count=accumulator.cell_date_count,
    )
    cell_lambda = accumulator.cell_weight / (
        accumulator.cell_weight + matched_alpha
    )
    cell_positive_fraction = np.divide(
        accumulator.cell_positive_weight,
        accumulator.cell_weight,
        out=np.full(accumulator.cell_size, np.nan, dtype=np.float64),
        where=accumulator.cell_weight > 1.0e-12,
    )
    family_names = list(families)
    family_index_by_stock = np.full(len(stock_indices), -1, dtype=np.int16)
    for family_index, family in enumerate(family_names):
        family_index_by_stock[np.asarray(families[family], dtype=np.int64)] = family_index
    if bool((family_index_by_stock < 0).any()):
        raise ValueError("dynamic_action_baseline_family_coverage")
    return {
        "prior": np.asarray([prior], dtype=np.float64),
        "prior_se": np.asarray([prior_se], dtype=np.float64),
        "market_effect": market_effect.astype(np.float64),
        "market_se": market_se.astype(np.float64),
        "stock_effect": stock_effect.astype(np.float64),
        "stock_se": stock_se.astype(np.float64),
        "cell_mean": cell_mean.astype(np.float64),
        "cell_se": np.nan_to_num(cell_raw_se, nan=prior_se).astype(np.float64),
        "cell_lambda": cell_lambda.astype(np.float64),
        "cell_weight": accumulator.cell_weight.astype(np.float64),
        "cell_date_count": accumulator.cell_date_count.astype(np.float64),
        "cell_positive_fraction": cell_positive_fraction.astype(np.float64),
        "coordinate_names": np.asarray(list(coordinate_names), dtype="U64"),
        "market_indices": np.asarray(market_indices, dtype=np.int16),
        "stock_indices": np.asarray(stock_indices, dtype=np.int16),
        "matched_indices": np.asarray(matched_indices, dtype=np.int16),
        "family_names": np.asarray(family_names, dtype="U32"),
        "family_index_by_stock": family_index_by_stock,
        "univariate_bins": np.asarray([accumulator.univariate_bins], dtype=np.int16),
        "matched_bins": np.asarray([accumulator.matched_bins], dtype=np.int16),
        "training_rows": np.asarray([accumulator.training_rows], dtype=np.int64),
        "training_dates": np.asarray(
            [round(accumulator.global_date_count)], dtype=np.int32
        ),
        "training_cutoff_year": np.asarray([training_cutoff_year], dtype=np.int16),
        "maximum_label_as_of_year": np.asarray(
            [maximum_label_as_of_year], dtype=np.int16
        ),
        "maximum_signal_year": np.asarray([maximum_signal_year], dtype=np.int16),
    }


def predict_from_snapshot(
    values: np.ndarray, snapshot: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    prior = float(np.asarray(snapshot["prior"])[0])
    prior_se = float(np.asarray(snapshot["prior_se"])[0])
    univariate_bins = int(np.asarray(snapshot["univariate_bins"])[0])
    matched_bins = int(np.asarray(snapshot["matched_bins"])[0])
    market_indices = np.asarray(snapshot["market_indices"], dtype=np.int64)
    stock_indices = np.asarray(snapshot["stock_indices"], dtype=np.int64)
    matched_indices = np.asarray(snapshot["matched_indices"], dtype=np.int64)
    family_names = [str(value) for value in np.asarray(snapshot["family_names"])]
    family_index_by_stock = np.asarray(
        snapshot["family_index_by_stock"], dtype=np.int64
    )

    market_bins = _univariate_bins(
        x[:, market_indices], bin_count=univariate_bins
    )
    market_effect_table = np.asarray(snapshot["market_effect"], dtype=np.float64)
    market_se_table = np.asarray(snapshot["market_se"], dtype=np.float64)
    market_effects = np.column_stack(
        [market_effect_table[index, market_bins[:, index]] for index in range(len(market_indices))]
    )
    market_errors = np.column_stack(
        [market_se_table[index, market_bins[:, index]] for index in range(len(market_indices))]
    )
    predicted_market = prior + market_effects.mean(axis=1)
    market_uncertainty = prior_se + market_errors.mean(axis=1)

    stock_bins = _univariate_bins(x[:, stock_indices], bin_count=univariate_bins)
    stock_effect_table = np.asarray(snapshot["stock_effect"], dtype=np.float64)
    stock_se_table = np.asarray(snapshot["stock_se"], dtype=np.float64)
    stock_effects = np.column_stack(
        [stock_effect_table[index, stock_bins[:, index]] for index in range(len(stock_indices))]
    )
    stock_errors = np.column_stack(
        [stock_se_table[index, stock_bins[:, index]] for index in range(len(stock_indices))]
    )
    family_effects: dict[str, np.ndarray] = {}
    family_errors: dict[str, np.ndarray] = {}
    for family_index, family in enumerate(family_names):
        columns = np.flatnonzero(family_index_by_stock == family_index)
        family_effects[family] = stock_effects[:, columns].mean(axis=1)
        family_errors[family] = stock_errors[:, columns].mean(axis=1)
    family_matrix = np.column_stack([family_effects[name] for name in family_names])
    family_error_matrix = np.column_stack(
        [family_errors[name] for name in family_names]
    )
    predicted_additive = predicted_market + family_matrix.mean(axis=1)
    additive_uncertainty = market_uncertainty + family_error_matrix.mean(axis=1)

    cell_code = _matched_codes(
        x, matched_indices=matched_indices, bin_count=matched_bins
    )
    cell_lambda = np.asarray(snapshot["cell_lambda"], dtype=np.float64)[cell_code]
    cell_mean = np.asarray(snapshot["cell_mean"], dtype=np.float64)[cell_code]
    cell_se = np.asarray(snapshot["cell_se"], dtype=np.float64)[cell_code]
    predicted_matched = (
        cell_lambda * cell_mean + (1.0 - cell_lambda) * predicted_additive
    )
    matched_uncertainty = (
        cell_lambda * cell_se + (1.0 - cell_lambda) * additive_uncertainty
    )
    output: dict[str, np.ndarray] = {
        "predicted_prior": np.full(len(x), prior, dtype=np.float64),
        "predicted_market": predicted_market,
        "predicted_additive": predicted_additive,
        "predicted_matched_history": predicted_matched,
        "additive_uncertainty_proxy": additive_uncertainty,
        "matched_uncertainty_proxy": matched_uncertainty,
        "matched_cell_code": cell_code,
        "matched_cell_weight": np.asarray(snapshot["cell_weight"], dtype=np.float64)[
            cell_code
        ],
        "matched_cell_date_count": np.asarray(
            snapshot["cell_date_count"], dtype=np.float64
        )[cell_code],
        "matched_cell_positive_fraction": np.asarray(
            snapshot["cell_positive_fraction"], dtype=np.float64
        )[cell_code],
    }
    for family, effect in family_effects.items():
        output[f"effect_{family}"] = effect
    return output


def select_latest_records(
    *,
    version_records: Mapping[tuple[str, int, int], Mapping[str, Any]],
    cost_scenario: str,
    cutoff_year: int,
    maximum_signal_year: int,
) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for signal_year in range(2012, int(maximum_signal_year) + 1):
        candidates = [
            dict(record)
            for (cost, signal, as_of), record in version_records.items()
            if cost == str(cost_scenario)
            and signal == signal_year
            and as_of <= int(cutoff_year)
        ]
        if not candidates:
            continue
        selected[signal_year] = max(candidates, key=lambda item: int(item["as_of_year"]))
    return selected


def select_evaluation_record(
    *,
    version_records: Mapping[tuple[str, int, int], Mapping[str, Any]],
    cost_scenario: str,
    signal_year: int,
) -> dict[str, Any]:
    maximum_as_of = min(int(signal_year) + 1, 2025)
    selected = select_latest_records(
        version_records=version_records,
        cost_scenario=cost_scenario,
        cutoff_year=maximum_as_of,
        maximum_signal_year=signal_year,
    ).get(int(signal_year))
    if selected is None:
        raise ValueError("dynamic_action_baseline_evaluation_record")
    return selected


def _snapshot_record(
    *,
    root: Path,
    cost_scenario: str,
    oos_year: int,
    snapshot: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    path = (
        root
        / "model_states"
        / f"cost={cost_scenario}"
        / f"oos_year={int(oos_year)}"
        / "state.npz"
    )
    record = _write_npz(path, snapshot)
    record.update({"cost_scenario": cost_scenario, "oos_year": int(oos_year)})
    return record


def _prediction_frame(
    *,
    experience: AlignedExperience,
    coordinate_year: CoordinateYear,
    predictions: Mapping[str, np.ndarray],
    family_names: Sequence[str],
    training_cutoff_year: int,
    training_rows: int,
    training_dates: int,
    maximum_label_as_of_year: int,
    maximum_signal_year: int,
) -> pd.DataFrame:
    positions = experience.coordinate_positions
    output: dict[str, Any] = {
        "cost_scenario": np.full(
            len(positions), experience.cost_scenario, dtype=object
        ),
        "oos_year": np.full(len(positions), experience.signal_year, dtype=np.int16),
        "training_cutoff_year": np.full(
            len(positions), int(training_cutoff_year), dtype=np.int16
        ),
        "training_rows": np.full(len(positions), int(training_rows), dtype=np.int64),
        "training_dates": np.full(
            len(positions), int(training_dates), dtype=np.int32
        ),
        "maximum_training_label_as_of_year": np.full(
            len(positions), int(maximum_label_as_of_year), dtype=np.int16
        ),
        "maximum_training_signal_year": np.full(
            len(positions), int(maximum_signal_year), dtype=np.int16
        ),
        "evaluation_as_of_year": np.full(
            len(positions), experience.as_of_year, dtype=np.int16
        ),
        "evaluation_version_role": np.full(
            len(positions), experience.version_role, dtype=object
        ),
        "input_row_idx": coordinate_year.frame["input_row_idx"].to_numpy(np.int64)[
            positions
        ],
        "trade_date": experience.trade_date,
        "date_idx": experience.date_idx,
        "symbol_idx": experience.symbol_idx,
        "actual_buy_advantage_vs_cash": experience.target,
        "actual_positive": experience.target > 0.0,
        "prefix_oracle_cash_buy_selected": experience.selected,
        "evaluation_sessions_after_resolution": experience.sessions_after_resolution,
    }
    direct = (
        "predicted_prior",
        "predicted_market",
        "predicted_additive",
        "predicted_matched_history",
        "additive_uncertainty_proxy",
        "matched_uncertainty_proxy",
        "matched_cell_weight",
        "matched_cell_date_count",
        "matched_cell_positive_fraction",
    )
    for name in direct:
        output[name] = np.asarray(predictions[name], dtype=np.float32)
    output["matched_cell_code"] = np.asarray(
        predictions["matched_cell_code"], dtype=np.int32
    )
    matched_positive = np.asarray(
        predictions["matched_cell_positive_fraction"], dtype=np.float64
    )
    matched_weight = np.asarray(predictions["matched_cell_weight"], dtype=np.float64)
    output["matched_failure_available"] = (
        (matched_weight > 0.0)
        & np.isfinite(matched_positive)
        & (matched_positive < 1.0 - 1.0e-12)
    )
    for family in family_names:
        output[f"effect_{family}"] = np.asarray(
            predictions[f"effect_{family}"], dtype=np.float32
        )
    return pd.DataFrame(output)


def _model_predictions(
    frame: pd.DataFrame, family_names: Sequence[str]
) -> dict[str, np.ndarray]:
    market = frame["predicted_market"].to_numpy(np.float64)
    effects = {
        family: frame[f"effect_{family}"].to_numpy(np.float64)
        for family in family_names
    }
    result = {
        "prior": frame["predicted_prior"].to_numpy(np.float64),
        "market": market,
        "additive": frame["predicted_additive"].to_numpy(np.float64),
        "matched_history": frame["predicted_matched_history"].to_numpy(np.float64),
    }
    effect_matrix = np.column_stack([effects[family] for family in family_names])
    for index, family in enumerate(family_names):
        result[f"family_{family}"] = market + effect_matrix[:, index]
        other = np.delete(effect_matrix, index, axis=1)
        result[f"without_{family}"] = market + other.mean(axis=1)
    return result


def _safe_spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    finite = np.isfinite(y) & np.isfinite(p)
    if int(finite.sum()) < 3:
        return math.nan
    y = y[finite]
    p = p[finite]
    if np.ptp(y) <= 1.0e-15 or np.ptp(p) <= 1.0e-15:
        return math.nan
    return float(np.corrcoef(rankdata(y), rankdata(p))[0, 1])


def _weighted_calibration(
    *, actual: np.ndarray, predicted: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0.0)
    if int(finite.sum()) < 3:
        return math.nan, math.nan
    y = y[finite]
    p = p[finite]
    w = w[finite]
    total = float(w.sum())
    mean_y = float(np.dot(w, y) / total)
    mean_p = float(np.dot(w, p) / total)
    centered = p - mean_p
    variance = float(np.dot(w, np.square(centered)))
    if variance <= 1.0e-15:
        return mean_y, 0.0
    slope = float(np.dot(w, centered * (y - mean_y)) / variance)
    return mean_y - slope * mean_p, slope


def _evaluate_model_partition(
    *,
    frame: pd.DataFrame,
    model_name: str,
    predicted: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    actual = frame["actual_buy_advantage_vs_cash"].to_numpy(np.float64)
    dates = frame["date_idx"].to_numpy(np.int64)
    if bool((dates[1:] < dates[:-1]).any()):
        raise ValueError("dynamic_action_baseline_prediction_order")
    unique_dates, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    group_index = np.repeat(np.arange(len(starts), dtype=np.int64), counts)
    weights = 1.0 / counts[group_index].astype(np.float64)
    calibration_intercept, calibration_slope = _weighted_calibration(
        actual=actual, predicted=predicted, weights=weights
    )
    rows: list[dict[str, Any]] = []
    trade_dates = frame["trade_date"].astype(str).to_numpy()
    symbols = frame["symbol_idx"].to_numpy(np.int32)
    failure_available = frame["matched_failure_available"].astype(bool).to_numpy()
    cell_positive = frame["matched_cell_positive_fraction"].to_numpy(np.float64)
    cell_dates = frame["matched_cell_date_count"].to_numpy(np.float64)
    for date_position, (start, count) in enumerate(zip(starts, counts, strict=True)):
        stop = int(start + count)
        y = actual[start:stop]
        p = predicted[start:stop]
        order = np.argsort(-p, kind="mergesort")
        selected_local = int(order[0])
        selected = int(start + selected_local)
        top_count = min(5, len(order))
        predicted_positive = p > 0.0
        realized_positive_mean = (
            float(np.mean(y[predicted_positive]))
            if bool(predicted_positive.any())
            else math.nan
        )
        selected_action = bool(p[selected_local] > 0.0)
        selected_actual = float(y[selected_local])
        oracle_value = float(max(float(np.max(y)), 0.0))
        policy_value = selected_actual if selected_action else 0.0
        fraction_metrics: dict[str, float] = {}
        for fraction in (0.01, 0.05, 0.10):
            fraction_count = max(1, math.ceil(len(order) * fraction))
            chosen = y[order[:fraction_count]]
            label = f"top{int(fraction * 100)}pct"
            fraction_metrics[f"{label}_realized_mean"] = float(np.mean(chosen))
            fraction_metrics[f"{label}_positive_fraction"] = float(
                np.mean(chosen > 0.0)
            )
        rows.append(
            {
                "cost_scenario": str(frame["cost_scenario"].iloc[0]),
                "oos_year": int(frame["oos_year"].iloc[0]),
                "trade_date": str(trade_dates[start]),
                "date_idx": int(unique_dates[date_position]),
                "model": model_name,
                "row_count": int(count),
                "mse": float(np.mean(np.square(y - p))),
                "mae": float(np.mean(np.abs(y - p))),
                "spearman": _safe_spearman(y, p),
                "actual_mean": float(np.mean(y)),
                "predicted_mean": float(np.mean(p)),
                "predicted_positive_fraction": float(np.mean(predicted_positive)),
                "realized_mean_predicted_positive": realized_positive_mean,
                "selected_action": selected_action,
                "selected_symbol_idx": int(symbols[selected]),
                "selected_predicted": float(p[selected_local]),
                "selected_actual": selected_actual,
                "selected_actual_positive": selected_actual > 0.0,
                "top5_realized_mean": float(np.mean(y[order[:top_count]])),
                **fraction_metrics,
                "oracle_daily_value": oracle_value,
                "policy_value": policy_value,
                "regret": oracle_value - policy_value,
                "selected_matched_failure_available": bool(
                    failure_available[selected]
                ),
                "selected_matched_cell_positive_fraction": float(
                    cell_positive[selected]
                ),
                "selected_matched_cell_date_count": float(cell_dates[selected]),
            }
        )
    daily = pd.DataFrame(rows)
    annual = {
        "cost_scenario": str(frame["cost_scenario"].iloc[0]),
        "oos_year": int(frame["oos_year"].iloc[0]),
        "model": model_name,
        "dates": len(daily),
        "rows": len(frame),
        "date_equal_mse": float(daily["mse"].mean()),
        "date_equal_mae": float(daily["mae"].mean()),
        "mean_daily_spearman": float(daily["spearman"].mean()),
        "mean_top1_realized_advantage": float(daily["selected_actual"].mean()),
        "mean_top5_realized_advantage": float(daily["top5_realized_mean"].mean()),
        "mean_top1pct_realized_advantage": float(
            daily["top1pct_realized_mean"].mean()
        ),
        "mean_top1pct_positive_fraction": float(
            daily["top1pct_positive_fraction"].mean()
        ),
        "mean_top5pct_realized_advantage": float(
            daily["top5pct_realized_mean"].mean()
        ),
        "mean_top5pct_positive_fraction": float(
            daily["top5pct_positive_fraction"].mean()
        ),
        "mean_top10pct_realized_advantage": float(
            daily["top10pct_realized_mean"].mean()
        ),
        "mean_top10pct_positive_fraction": float(
            daily["top10pct_positive_fraction"].mean()
        ),
        "zero_threshold_policy_value": float(daily["policy_value"].mean()),
        "daily_oracle_value": float(daily["oracle_daily_value"].mean()),
        "daily_oracle_regret": float(daily["regret"].mean()),
        "selected_action_fraction": float(daily["selected_action"].mean()),
        "selected_actual_positive_fraction": float(
            daily["selected_actual_positive"].mean()
        ),
        "matched_failure_available_fraction": float(
            daily["selected_matched_failure_available"].mean()
        ),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
    }
    return daily, annual


def _hac_mean(values: np.ndarray, *, lag: int) -> tuple[float, float, float]:
    source = np.asarray(values, dtype=np.float64)
    source = source[np.isfinite(source)]
    if not len(source):
        return math.nan, math.nan, math.nan
    mean = float(source.mean())
    residual = source - mean
    long_run = float(np.dot(residual, residual) / len(source))
    maximum_lag = min(int(lag), max(len(source) - 1, 0))
    for offset in range(1, maximum_lag + 1):
        weight = 1.0 - offset / (maximum_lag + 1.0)
        covariance = float(
            np.dot(residual[offset:], residual[:-offset]) / len(source)
        )
        long_run += 2.0 * weight * covariance
    standard_error = math.sqrt(max(long_run, 0.0) / len(source))
    return mean, standard_error, mean - 1.96 * standard_error


def evaluate_predictions(
    *,
    prediction_records: Sequence[Mapping[str, Any]],
    family_names: Sequence[str],
    output_root: Path,
    row_group_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    daily_frames: list[pd.DataFrame] = []
    annual_rows: list[dict[str, Any]] = []
    for record in prediction_records:
        path = Path(str(record["path"]))
        if replay.sha256(path) != str(record["sha256"]):
            raise ValueError("dynamic_action_baseline_prediction_hash")
        frame = pd.read_parquet(path)
        for model_name, predicted in _model_predictions(frame, family_names).items():
            daily, annual = _evaluate_model_partition(
                frame=frame, model_name=model_name, predicted=predicted
            )
            daily_frames.append(daily)
            annual_rows.append(annual)
        del frame
        gc.collect()
    daily_frame = pd.concat(daily_frames, ignore_index=True)
    annual_frame = pd.DataFrame(annual_rows).sort_values(
        ["cost_scenario", "oos_year", "model"], kind="mergesort"
    )
    aggregate_rows: list[dict[str, Any]] = []
    for (cost, model), group in daily_frame.groupby(
        ["cost_scenario", "model"], sort=True
    ):
        policy_mean, policy_se, policy_lower = _hac_mean(
            group["policy_value"].to_numpy(np.float64), lag=20
        )
        top_mean, top_se, top_lower = _hac_mean(
            group["selected_actual"].to_numpy(np.float64), lag=20
        )
        annual = annual_frame[
            annual_frame["cost_scenario"].eq(cost)
            & annual_frame["model"].eq(model)
        ]
        aggregate_rows.append(
            {
                "cost_scenario": str(cost),
                "model": str(model),
                "dates": len(group),
                "oos_years": len(annual),
                "date_equal_mse": float(group["mse"].mean()),
                "date_equal_mae": float(group["mae"].mean()),
                "mean_daily_spearman": float(group["spearman"].mean()),
                "mean_top1_realized_advantage": top_mean,
                "top1_hac_standard_error": top_se,
                "top1_hac_lower_95": top_lower,
                "mean_top5_realized_advantage": float(
                    group["top5_realized_mean"].mean()
                ),
                "mean_top1pct_realized_advantage": float(
                    group["top1pct_realized_mean"].mean()
                ),
                "mean_top1pct_positive_fraction": float(
                    group["top1pct_positive_fraction"].mean()
                ),
                "mean_top5pct_realized_advantage": float(
                    group["top5pct_realized_mean"].mean()
                ),
                "mean_top5pct_positive_fraction": float(
                    group["top5pct_positive_fraction"].mean()
                ),
                "mean_top10pct_realized_advantage": float(
                    group["top10pct_realized_mean"].mean()
                ),
                "mean_top10pct_positive_fraction": float(
                    group["top10pct_positive_fraction"].mean()
                ),
                "zero_threshold_policy_value": policy_mean,
                "policy_hac_standard_error": policy_se,
                "policy_hac_lower_95": policy_lower,
                "daily_oracle_value": float(group["oracle_daily_value"].mean()),
                "daily_oracle_regret": float(group["regret"].mean()),
                "selected_action_fraction": float(group["selected_action"].mean()),
                "selected_actual_positive_fraction": float(
                    group["selected_actual_positive"].mean()
                ),
                "positive_policy_year_fraction": float(
                    np.mean(annual["zero_threshold_policy_value"] > 0.0)
                ),
                "matched_failure_available_fraction": float(
                    group["selected_matched_failure_available"].mean()
                ),
                "matched_failure_available_when_acting": float(
                    group.loc[
                        group["selected_action"],
                        "selected_matched_failure_available",
                    ].mean()
                ),
            }
        )
    aggregate_frame = pd.DataFrame(aggregate_rows).sort_values(
        ["cost_scenario", "model"], kind="mergesort"
    )
    daily_record = _write_frame(
        output_root / "daily_model_metrics.parquet",
        daily_frame,
        row_group_size=row_group_size,
    )
    annual_record = _write_frame(
        output_root / "annual_model_metrics.parquet",
        annual_frame,
        row_group_size=row_group_size,
    )
    aggregate_record = _write_frame(
        output_root / "aggregate_model_metrics.parquet",
        aggregate_frame,
        row_group_size=row_group_size,
    )

    gate_rows: list[dict[str, Any]] = []
    for model in sorted(set(aggregate_frame["model"])):
        scoped = aggregate_frame[aggregate_frame["model"].eq(model)]
        market = aggregate_frame[aggregate_frame["model"].eq("market")].set_index(
            "cost_scenario"
        )
        checks: list[bool] = []
        for row in scoped.itertuples(index=False):
            checks.extend(
                [
                    float(row.policy_hac_lower_95) > 0.0,
                    float(row.positive_policy_year_fraction) > 0.5,
                    float(row.mean_daily_spearman) > 0.0,
                    float(row.daily_oracle_regret)
                    < float(market.loc[row.cost_scenario, "daily_oracle_regret"]),
                ]
            )
        gate_rows.append(
            {
                "model": model,
                "cost_scenarios_present": len(scoped),
                "all_gate_checks_passed": bool(len(scoped) == 2 and all(checks)),
            }
        )
    gate = {
        "schema": "seq100_dynamic_action_value_promotion_gate/1",
        "passed_models": [
            row["model"] for row in gate_rows if row["all_gate_checks_passed"]
        ],
        "rows": gate_rows,
        "account_replay_allowed": any(
            row["all_gate_checks_passed"] for row in gate_rows
        ),
        "profit_claim_allowed": False,
    }
    _write_json(output_root / "promotion_gate.json", gate)
    outputs = {
        "daily_model_metrics": daily_record,
        "annual_model_metrics": annual_record,
        "aggregate_model_metrics": aggregate_record,
        "promotion_gate": _file_record(output_root / "promotion_gate.json"),
    }
    audit = {
        "daily_metric_rows": len(daily_frame),
        "annual_metric_rows": len(annual_frame),
        "aggregate_metric_rows": len(aggregate_frame),
        "models": sorted(set(aggregate_frame["model"])),
        "passed_models": gate["passed_models"],
        "account_replay_allowed": gate["account_replay_allowed"],
    }
    return outputs, audit


def build_prediction_quantile_diagnostics(
    *,
    prediction_records: Sequence[Mapping[str, Any]],
    output_root: Path,
    row_group_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_columns = {
        "additive": "predicted_additive",
        "matched_history": "predicted_matched_history",
    }
    daily_frames: list[pd.DataFrame] = []
    for record in prediction_records:
        frame = pd.read_parquet(
            record["path"],
            columns=[
                "cost_scenario",
                "oos_year",
                "trade_date",
                "actual_buy_advantage_vs_cash",
                "prefix_oracle_cash_buy_selected",
                *model_columns.values(),
            ],
        )
        actual = frame["actual_buy_advantage_vs_cash"].to_numpy(np.float64)
        for model, column in model_columns.items():
            percentile = frame.groupby("trade_date", sort=False)[column].rank(
                method="first", pct=True
            )
            bins = np.minimum(
                np.floor(percentile.to_numpy(np.float64) * 10.0).astype(np.int16),
                9,
            )
            temporary = pd.DataFrame(
                {
                    "trade_date": frame["trade_date"].astype(str).to_numpy(),
                    "prediction_quantile": bins,
                    "actual": actual,
                    "positive": actual > 0.0,
                    "upside": np.maximum(actual, 0.0),
                    "downside": np.minimum(actual, 0.0),
                    "oracle_selected": frame[
                        "prefix_oracle_cash_buy_selected"
                    ].astype(bool).to_numpy(),
                }
            )
            daily = (
                temporary.groupby(
                    ["trade_date", "prediction_quantile"], sort=False
                )
                .agg(
                    actual_mean=("actual", "mean"),
                    positive_fraction=("positive", "mean"),
                    upside_component=("upside", "mean"),
                    downside_component=("downside", "mean"),
                    oracle_selected_rate=("oracle_selected", "mean"),
                    rows=("actual", "size"),
                )
                .reset_index()
            )
            daily.insert(0, "model", model)
            daily.insert(0, "oos_year", int(frame["oos_year"].iloc[0]))
            daily.insert(0, "cost_scenario", str(frame["cost_scenario"].iloc[0]))
            daily_frames.append(daily)
    daily_frame = pd.concat(daily_frames, ignore_index=True)
    metric_columns = [
        "actual_mean",
        "positive_fraction",
        "upside_component",
        "downside_component",
        "oracle_selected_rate",
    ]
    annual = (
        daily_frame.groupby(
            ["cost_scenario", "oos_year", "model", "prediction_quantile"],
            sort=True,
        )[metric_columns]
        .mean()
        .reset_index()
    )
    annual["dates"] = (
        daily_frame.groupby(
            ["cost_scenario", "oos_year", "model", "prediction_quantile"],
            sort=True,
        )["trade_date"]
        .count()
        .to_numpy(np.int32)
    )
    aggregate = (
        daily_frame.groupby(
            ["cost_scenario", "model", "prediction_quantile"], sort=True
        )[metric_columns]
        .mean()
        .reset_index()
    )
    aggregate["dates"] = (
        daily_frame.groupby(
            ["cost_scenario", "model", "prediction_quantile"], sort=True
        )["trade_date"]
        .count()
        .to_numpy(np.int32)
    )
    spread_rows: list[dict[str, Any]] = []
    for (cost, model), group in aggregate.groupby(
        ["cost_scenario", "model"], sort=True
    ):
        indexed = group.set_index("prediction_quantile")
        low = indexed.loc[0]
        high = indexed.loc[9]
        spread_rows.append(
            {
                "cost_scenario": cost,
                "model": model,
                **{
                    f"high_minus_low_{name}": float(high[name] - low[name])
                    for name in metric_columns
                },
            }
        )
    spread = pd.DataFrame(spread_rows)
    annual_record = _write_frame(
        output_root / "annual_prediction_quantiles.parquet",
        annual,
        row_group_size=row_group_size,
    )
    aggregate_record = _write_frame(
        output_root / "aggregate_prediction_quantiles.parquet",
        aggregate,
        row_group_size=row_group_size,
    )
    spread_record = _write_frame(
        output_root / "prediction_quantile_spreads.parquet",
        spread,
        row_group_size=row_group_size,
    )
    return (
        {
            "annual_prediction_quantiles": annual_record,
            "aggregate_prediction_quantiles": aggregate_record,
            "prediction_quantile_spreads": spread_record,
        },
        {
            "annual_prediction_quantile_rows": len(annual),
            "aggregate_prediction_quantile_rows": len(aggregate),
        },
    )


def build_coordinate_diagnostics(
    *,
    prediction_records: Sequence[Mapping[str, Any]],
    coordinate_records: Sequence[Mapping[str, Any]],
    coordinate_names: Sequence[str],
    output_root: Path,
    row_group_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    coordinate_by_year = {
        int(record["year"]): dict(record) for record in coordinate_records
    }
    metric_names = (
        "actual_mean",
        "positive_fraction",
        "upside_component",
        "downside_component",
        "oracle_selected_rate",
    )
    aggregate_sum: dict[tuple[str, str, int], np.ndarray] = {}
    aggregate_count: dict[tuple[str, str, int], int] = {}
    annual_rows: list[dict[str, Any]] = []
    for record in prediction_records:
        cost = str(record["cost_scenario"])
        year = int(record["oos_year"])
        prediction = pd.read_parquet(
            record["path"],
            columns=[
                "trade_date",
                "input_row_idx",
                "actual_buy_advantage_vs_cash",
                "prefix_oracle_cash_buy_selected",
            ],
        )
        coordinate_record = coordinate_by_year[year]
        coordinate = pd.read_parquet(
            coordinate_record["path"],
            columns=["input_row_idx", *coordinate_names],
        )
        first_input = int(coordinate["input_row_idx"].iloc[0])
        local = prediction["input_row_idx"].to_numpy(np.int64) - first_input
        if bool((local < 0).any()) or bool((local >= len(coordinate)).any()):
            raise ValueError("dynamic_action_baseline_coordinate_diagnostic_alignment")
        values = coordinate.iloc[local][list(coordinate_names)].to_numpy(
            np.float64
        )
        actual = prediction["actual_buy_advantage_vs_cash"].to_numpy(np.float64)
        dates = prediction["trade_date"].astype(str).to_numpy()
        selected = prediction["prefix_oracle_cash_buy_selected"].astype(bool).to_numpy()
        bins = _univariate_bins(values, bin_count=10)
        for column, coordinate_name in enumerate(coordinate_names):
            temporary = pd.DataFrame(
                {
                    "trade_date": dates,
                    "coordinate_bin": bins[:, column],
                    "actual": actual,
                    "positive": actual > 0.0,
                    "upside": np.maximum(actual, 0.0),
                    "downside": np.minimum(actual, 0.0),
                    "oracle_selected": selected,
                }
            )
            daily = (
                temporary.groupby(["trade_date", "coordinate_bin"], sort=False)
                .agg(
                    actual_mean=("actual", "mean"),
                    positive_fraction=("positive", "mean"),
                    upside_component=("upside", "mean"),
                    downside_component=("downside", "mean"),
                    oracle_selected_rate=("oracle_selected", "mean"),
                    rows=("actual", "size"),
                )
                .reset_index()
            )
            annual = daily.groupby("coordinate_bin", sort=True)[
                list(metric_names)
            ].mean()
            date_counts = daily.groupby("coordinate_bin", sort=True)[
                "trade_date"
            ].count()
            row_counts = daily.groupby("coordinate_bin", sort=True)["rows"].sum()
            for coordinate_bin in annual.index:
                metrics = annual.loc[coordinate_bin]
                annual_rows.append(
                    {
                        "cost_scenario": cost,
                        "oos_year": year,
                        "coordinate": coordinate_name,
                        "coordinate_bin": int(coordinate_bin),
                        **{name: float(metrics[name]) for name in metric_names},
                        "dates": int(date_counts.loc[coordinate_bin]),
                        "rows": int(row_counts.loc[coordinate_bin]),
                    }
                )
                scoped = daily[daily["coordinate_bin"].eq(coordinate_bin)]
                key = (cost, coordinate_name, int(coordinate_bin))
                contribution = scoped[list(metric_names)].sum().to_numpy(np.float64)
                aggregate_sum[key] = aggregate_sum.get(
                    key, np.zeros(len(metric_names), dtype=np.float64)
                ) + contribution
                aggregate_count[key] = aggregate_count.get(key, 0) + len(scoped)
        del prediction, coordinate, values
        gc.collect()

    aggregate_rows: list[dict[str, Any]] = []
    for key in sorted(aggregate_sum):
        cost, coordinate_name, coordinate_bin = key
        count = aggregate_count[key]
        metrics = aggregate_sum[key] / count
        aggregate_rows.append(
            {
                "cost_scenario": cost,
                "coordinate": coordinate_name,
                "coordinate_bin": coordinate_bin,
                **{
                    name: float(metrics[index])
                    for index, name in enumerate(metric_names)
                },
                "dates": count,
            }
        )
    annual_frame = pd.DataFrame(annual_rows).sort_values(
        ["cost_scenario", "oos_year", "coordinate", "coordinate_bin"],
        kind="mergesort",
    )
    aggregate_frame = pd.DataFrame(aggregate_rows).sort_values(
        ["cost_scenario", "coordinate", "coordinate_bin"], kind="mergesort"
    )
    spread_rows: list[dict[str, Any]] = []
    for (cost, coordinate_name), group in aggregate_frame.groupby(
        ["cost_scenario", "coordinate"], sort=True
    ):
        indexed = group.set_index("coordinate_bin")
        if 0 not in indexed.index or 9 not in indexed.index:
            continue
        low = indexed.loc[0]
        high = indexed.loc[9]
        positive_spread = float(high["positive_fraction"] - low["positive_fraction"])
        mean_spread = float(high["actual_mean"] - low["actual_mean"])
        spread_rows.append(
            {
                "cost_scenario": cost,
                "coordinate": coordinate_name,
                "high_minus_low_actual_mean": mean_spread,
                "high_minus_low_positive_fraction": positive_spread,
                "high_minus_low_upside_component": float(
                    high["upside_component"] - low["upside_component"]
                ),
                "high_minus_low_downside_component": float(
                    high["downside_component"] - low["downside_component"]
                ),
                "high_minus_low_oracle_selected_rate": float(
                    high["oracle_selected_rate"] - low["oracle_selected_rate"]
                ),
                "opportunity_without_net_edge": bool(
                    positive_spread > 0.0 and mean_spread <= 0.0
                ),
            }
        )
    spread_frame = pd.DataFrame(spread_rows).sort_values(
        ["cost_scenario", "coordinate"], kind="mergesort"
    )
    annual_record = _write_frame(
        output_root / "annual_coordinate_diagnostics.parquet",
        annual_frame,
        row_group_size=row_group_size,
    )
    aggregate_record = _write_frame(
        output_root / "aggregate_coordinate_diagnostics.parquet",
        aggregate_frame,
        row_group_size=row_group_size,
    )
    spread_record = _write_frame(
        output_root / "coordinate_extreme_spreads.parquet",
        spread_frame,
        row_group_size=row_group_size,
    )
    opportunity_coordinates = sorted(
        set(
            spread_frame.loc[
                spread_frame["opportunity_without_net_edge"], "coordinate"
            ].astype(str)
        )
    )
    return (
        {
            "annual_coordinate_diagnostics": annual_record,
            "aggregate_coordinate_diagnostics": aggregate_record,
            "coordinate_extreme_spreads": spread_record,
        },
        {
            "annual_coordinate_diagnostic_rows": len(annual_frame),
            "aggregate_coordinate_diagnostic_rows": len(aggregate_frame),
            "opportunity_without_net_edge_coordinates": opportunity_coordinates,
        },
    )


def build_matched_smoothing_sensitivity(
    *,
    study: Mapping[str, Any],
    prediction_records: Sequence[Mapping[str, Any]],
    state_records: Sequence[Mapping[str, Any]],
    output_root: Path,
    row_group_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_by_fold = {
        (str(record["cost_scenario"]), int(record["oos_year"])): dict(record)
        for record in state_records
    }
    alphas = [
        float(value)
        for value in dict(study["estimators"])[
            "matched_smoothing_sensitivity_date_equivalents"
        ]
    ]
    daily_rows: list[dict[str, Any]] = []
    for record in prediction_records:
        cost = str(record["cost_scenario"])
        year = int(record["oos_year"])
        state_record = state_by_fold[(cost, year)]
        with np.load(state_record["path"], allow_pickle=False) as state:
            cell_mean = state["cell_mean"].astype(np.float64)
            cell_weight = state["cell_weight"].astype(np.float64)
        prediction = pd.read_parquet(
            record["path"],
            columns=[
                "trade_date",
                "actual_buy_advantage_vs_cash",
                "predicted_additive",
                "matched_cell_code",
            ],
        )
        actual = prediction["actual_buy_advantage_vs_cash"].to_numpy(np.float64)
        additive = prediction["predicted_additive"].to_numpy(np.float64)
        codes = prediction["matched_cell_code"].to_numpy(np.int64)
        dates = prediction["trade_date"].astype(str).to_numpy()
        unique_dates, starts, counts = np.unique(
            dates, return_index=True, return_counts=True
        )
        for alpha in alphas:
            if alpha == 0.0:
                shrinkage = (cell_weight[codes] > 0.0).astype(np.float64)
            else:
                shrinkage = cell_weight[codes] / (cell_weight[codes] + alpha)
            predicted = (
                shrinkage * cell_mean[codes] + (1.0 - shrinkage) * additive
            )
            for date, start, count in zip(
                unique_dates, starts, counts, strict=True
            ):
                stop = int(start + count)
                local = int(np.argmax(predicted[start:stop]))
                selected = int(start + local)
                action = bool(predicted[selected] > 0.0)
                realized = float(actual[selected])
                daily_rows.append(
                    {
                        "cost_scenario": cost,
                        "oos_year": year,
                        "trade_date": str(date),
                        "smoothing_date_equivalents": alpha,
                        "selected_predicted": float(predicted[selected]),
                        "selected_actual": realized,
                        "selected_actual_positive": realized > 0.0,
                        "selected_action": action,
                        "policy_value": realized if action else 0.0,
                    }
                )
    daily = pd.DataFrame(daily_rows)
    annual = (
        daily.groupby(
            ["cost_scenario", "oos_year", "smoothing_date_equivalents"],
            sort=True,
        )
        .agg(
            dates=("trade_date", "count"),
            mean_top1_realized_advantage=("selected_actual", "mean"),
            zero_threshold_policy_value=("policy_value", "mean"),
            selected_action_fraction=("selected_action", "mean"),
            selected_actual_positive_fraction=(
                "selected_actual_positive",
                "mean",
            ),
            maximum_selected_prediction=("selected_predicted", "max"),
        )
        .reset_index()
    )
    aggregate_rows: list[dict[str, Any]] = []
    for (cost, alpha), group in daily.groupby(
        ["cost_scenario", "smoothing_date_equivalents"], sort=True
    ):
        policy_mean, policy_se, policy_lower = _hac_mean(
            group["policy_value"].to_numpy(np.float64), lag=20
        )
        scoped_annual = annual[
            annual["cost_scenario"].eq(cost)
            & annual["smoothing_date_equivalents"].eq(alpha)
        ]
        aggregate_rows.append(
            {
                "cost_scenario": cost,
                "smoothing_date_equivalents": float(alpha),
                "dates": len(group),
                "mean_top1_realized_advantage": float(
                    group["selected_actual"].mean()
                ),
                "zero_threshold_policy_value": policy_mean,
                "policy_hac_standard_error": policy_se,
                "policy_hac_lower_95": policy_lower,
                "selected_action_fraction": float(
                    group["selected_action"].mean()
                ),
                "selected_actual_positive_fraction": float(
                    group["selected_actual_positive"].mean()
                ),
                "positive_policy_years": int(
                    (scoped_annual["zero_threshold_policy_value"] > 0.0).sum()
                ),
                "maximum_selected_prediction": float(
                    group["selected_predicted"].max()
                ),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["cost_scenario", "smoothing_date_equivalents"], kind="mergesort"
    )
    annual_record = _write_frame(
        output_root / "annual_matched_smoothing_sensitivity.parquet",
        annual,
        row_group_size=row_group_size,
    )
    aggregate_record = _write_frame(
        output_root / "aggregate_matched_smoothing_sensitivity.parquet",
        aggregate,
        row_group_size=row_group_size,
    )
    return (
        {
            "annual_matched_smoothing_sensitivity": annual_record,
            "aggregate_matched_smoothing_sensitivity": aggregate_record,
        },
        {
            "matched_smoothing_sensitivity_rows": len(aggregate),
            "matched_smoothing_any_positive_policy_year": bool(
                (aggregate["positive_policy_years"] > 0).any()
            ),
        },
    )


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    memory_start = int(psutil.virtual_memory().available / 1024**2)
    minimum_available = memory_start
    study_file = _resolve(study_path)
    study = load_study(study_file)
    root = _resolve(output_root or study["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    resources = dict(study["resources"])
    reserve_memory = int(resources["reserve_memory_mb"])
    row_group_size = int(resources["parquet_row_group_size"])
    maturity_sessions = int(dict(study["target"])["minimum_sessions_after_resolution"])
    source_contract, prefix_manifest, input_manifest, version_records, feature_index = (
        _source_contract(study)
    )
    coordinate_records, coordinate_audit = build_coordinate_partitions(
        study=study,
        input_manifest=input_manifest,
        feature_index=feature_index,
        output_root=root,
    )
    minimum_available = min(
        minimum_available, int(psutil.virtual_memory().available / 1024**2)
    )
    specs, coordinate_names, market_names, families, matched_indices = (
        _coordinate_metadata(study)
    )
    name_to_index = {name: index for index, name in enumerate(coordinate_names)}
    market_indices = np.asarray(
        [name_to_index[name] for name in market_names], dtype=np.int16
    )
    stock_indices = np.asarray(
        [
            name_to_index[str(item["name"])]
            for item in specs
            if str(item["family"]) != "market"
        ],
        dtype=np.int16,
    )
    family_names = list(families)
    coordinates = CoordinateStore(
        records=coordinate_records,
        coordinate_names=coordinate_names,
        cache_years=int(resources["coordinate_cache_years"]),
    )
    univariate_bins = int(dict(study["coordinates"])["univariate_bins"])
    matched_bins = int(dict(study["coordinates"])["matched_bins"])
    oos_years = [int(value) for value in dict(study["period"])["strict_oos_years"]]
    cost_scenarios = [str(value) for value in dict(study["evaluation"])["cost_scenarios"]]

    prediction_records: list[dict[str, Any]] = []
    state_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    for cost_scenario in cost_scenarios:
        accumulator = ActionValueAccumulator(
            market_indices=market_indices,
            stock_indices=stock_indices,
            matched_indices=matched_indices,
            univariate_bins=univariate_bins,
            matched_bins=matched_bins,
        )
        active_records: dict[int, dict[str, Any]] = {}
        for oos_year in oos_years:
            available = int(psutil.virtual_memory().available / 1024**2)
            if available <= reserve_memory:
                raise MemoryError(
                    f"dynamic_action_baseline_memory_reserve:{available}"
                )
            fold_started = time.perf_counter()
            training_cutoff = int(oos_year) - 1
            desired_records = select_latest_records(
                version_records=version_records,
                cost_scenario=cost_scenario,
                cutoff_year=training_cutoff,
                maximum_signal_year=training_cutoff,
            )
            for signal_year, previous in list(active_records.items()):
                desired = desired_records.get(signal_year)
                if desired is not None and int(desired["as_of_year"]) == int(
                    previous["as_of_year"]
                ):
                    continue
                coordinate_year = coordinates.load(signal_year)
                experience = _read_aligned_experience(
                    record=previous,
                    coordinates=coordinate_year,
                    maturity_sessions=maturity_sessions,
                )
                audit = accumulator.apply(experience, sign=-1)
                update_rows.append(
                    {
                        "cost_scenario": cost_scenario,
                        "oos_year": oos_year,
                        "operation": "remove",
                        "signal_year": signal_year,
                        "as_of_year": int(previous["as_of_year"]),
                        **audit,
                    }
                )
                del active_records[signal_year]
            for signal_year, desired in desired_records.items():
                previous = active_records.get(signal_year)
                if previous is not None and int(previous["as_of_year"]) == int(
                    desired["as_of_year"]
                ):
                    continue
                coordinate_year = coordinates.load(signal_year)
                experience = _read_aligned_experience(
                    record=desired,
                    coordinates=coordinate_year,
                    maturity_sessions=maturity_sessions,
                )
                audit = accumulator.apply(experience, sign=1)
                update_rows.append(
                    {
                        "cost_scenario": cost_scenario,
                        "oos_year": oos_year,
                        "operation": "add",
                        "signal_year": signal_year,
                        "as_of_year": int(desired["as_of_year"]),
                        **audit,
                    }
                )
                active_records[signal_year] = dict(desired)

            maximum_label_as_of = max(
                int(record["as_of_year"]) for record in active_records.values()
            )
            maximum_signal_year = max(active_records)
            if maximum_label_as_of > training_cutoff or maximum_signal_year >= oos_year:
                raise ValueError("dynamic_action_baseline_training_cutoff_leak")
            snapshot = build_snapshot(
                accumulator=accumulator,
                study=study,
                coordinate_names=coordinate_names,
                market_indices=market_indices,
                stock_indices=stock_indices,
                families=families,
                matched_indices=matched_indices,
                training_cutoff_year=training_cutoff,
                maximum_label_as_of_year=maximum_label_as_of,
                maximum_signal_year=maximum_signal_year,
            )
            state_record = _snapshot_record(
                root=root,
                cost_scenario=cost_scenario,
                oos_year=oos_year,
                snapshot=snapshot,
            )
            state_records.append(state_record)

            evaluation_record = select_evaluation_record(
                version_records=version_records,
                cost_scenario=cost_scenario,
                signal_year=oos_year,
            )
            coordinate_year = coordinates.load(oos_year)
            evaluation = _read_aligned_experience(
                record=evaluation_record,
                coordinates=coordinate_year,
                maturity_sessions=maturity_sessions,
            )
            predicted = predict_from_snapshot(evaluation.coordinates, snapshot)
            prediction = _prediction_frame(
                experience=evaluation,
                coordinate_year=coordinate_year,
                predictions=predicted,
                family_names=family_names,
                training_cutoff_year=training_cutoff,
                training_rows=int(snapshot["training_rows"][0]),
                training_dates=int(snapshot["training_dates"][0]),
                maximum_label_as_of_year=maximum_label_as_of,
                maximum_signal_year=maximum_signal_year,
            )
            prediction_path = (
                root
                / "predictions"
                / f"cost={cost_scenario}"
                / f"oos_year={oos_year}"
                / "part-0000.parquet"
            )
            prediction_record = _write_frame(
                prediction_path, prediction, row_group_size=row_group_size
            )
            prediction_record.update(
                {
                    "cost_scenario": cost_scenario,
                    "oos_year": oos_year,
                    "training_cutoff_year": training_cutoff,
                    "maximum_training_label_as_of_year": maximum_label_as_of,
                    "maximum_training_signal_year": maximum_signal_year,
                    "evaluation_as_of_year": int(evaluation_record["as_of_year"]),
                    "evaluation_signal_year": int(
                        evaluation_record["signal_year"]
                    ),
                }
            )
            prediction_records.append(prediction_record)
            fold_elapsed = float(time.perf_counter() - fold_started)
            fold_records.append(
                {
                    "cost_scenario": cost_scenario,
                    "oos_year": oos_year,
                    "training_cutoff_year": training_cutoff,
                    "training_rows": int(snapshot["training_rows"][0]),
                    "training_dates": int(snapshot["training_dates"][0]),
                    "maximum_training_label_as_of_year": maximum_label_as_of,
                    "maximum_training_signal_year": maximum_signal_year,
                    "training_version_count": len(active_records),
                    "training_versions": [
                        {
                            "signal_year": signal_year,
                            "as_of_year": int(record["as_of_year"]),
                            "version_role": str(record["version_role"]),
                            "path": str(record["path"]),
                            "sha256": str(record["sha256"]),
                        }
                        for signal_year, record in sorted(active_records.items())
                    ],
                    "evaluation_as_of_year": int(evaluation_record["as_of_year"]),
                    "evaluation_rows": len(prediction),
                    "evaluation_missing_coordinate_rows": int(
                        evaluation.missing_coordinate_rows
                    ),
                    "elapsed_seconds": fold_elapsed,
                    "state": state_record,
                    "prediction": prediction_record,
                }
            )
            print(
                json.dumps(
                    {
                        "cost_scenario": cost_scenario,
                        "oos_year": oos_year,
                        "training_rows": int(snapshot["training_rows"][0]),
                        "prediction_rows": len(prediction),
                        "elapsed_seconds": fold_elapsed,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            minimum_available = min(
                minimum_available,
                int(psutil.virtual_memory().available / 1024**2),
            )
            del evaluation, predicted, prediction, snapshot
            gc.collect()
        coordinates.clear()

    update_frame = pd.DataFrame(update_rows).sort_values(
        ["cost_scenario", "oos_year", "operation", "signal_year"],
        kind="mergesort",
    )
    fold_summary = pd.DataFrame(
        [
            {key: value for key, value in record.items() if key not in {"training_versions", "state", "prediction"}}
            for record in fold_records
        ]
    ).sort_values(["cost_scenario", "oos_year"], kind="mergesort")
    training_update_record = _write_frame(
        root / "training_version_updates.parquet",
        update_frame,
        row_group_size=row_group_size,
    )
    fold_summary_record = _write_frame(
        root / "fold_summary.parquet",
        fold_summary,
        row_group_size=row_group_size,
    )
    evaluation_outputs, evaluation_audit = evaluate_predictions(
        prediction_records=prediction_records,
        family_names=family_names,
        output_root=root,
        row_group_size=row_group_size,
    )
    quantile_outputs, quantile_audit = build_prediction_quantile_diagnostics(
        prediction_records=prediction_records,
        output_root=root,
        row_group_size=row_group_size,
    )
    coordinate_outputs, diagnostic_audit = build_coordinate_diagnostics(
        prediction_records=prediction_records,
        coordinate_records=coordinate_records,
        coordinate_names=coordinate_names,
        output_root=root,
        row_group_size=row_group_size,
    )
    smoothing_outputs, smoothing_audit = build_matched_smoothing_sensitivity(
        study=study,
        prediction_records=prediction_records,
        state_records=state_records,
        output_root=root,
        row_group_size=row_group_size,
    )
    memory_end = int(psutil.virtual_memory().available / 1024**2)
    manifest = {
        "schema": f"seq100_dynamic_action_value_baselines/{SCHEMA_VERSION}",
        "status": "completed",
        "study_id": STUDY_ID,
        "study": {
            "path": str(study_file.resolve()),
            "sha256": replay.sha256(study_file),
        },
        "source_contract": source_contract,
        "source_audit": {
            "prefix_experience_version_rows": int(
                dict(prefix_manifest["audit"])["experience_version_rows"]
            ),
            "model_input_rows": int(input_manifest["row_count"]),
            "forbidden_2026_rows": 0,
        },
        "coordinates": coordinate_records,
        "coordinate_audit": coordinate_audit,
        "model_states": state_records,
        "predictions": prediction_records,
        "folds": fold_records,
        "outputs": {
            "training_version_updates": training_update_record,
            "fold_summary": fold_summary_record,
            **evaluation_outputs,
            **quantile_outputs,
            **coordinate_outputs,
            **smoothing_outputs,
        },
        "audit": {
            "cost_scenarios": cost_scenarios,
            "oos_years": oos_years,
            "fold_count": len(fold_records),
            "prediction_partitions": len(prediction_records),
            "prediction_rows": int(
                sum(int(record["rows"]) for record in prediction_records)
            ),
            "minimum_sessions_after_resolution": maturity_sessions,
            "training_cutoff_violations": 0,
            "forbidden_2026_rows": 0,
            "final_oracle_row_labels_used": False,
            "fixed_holding_horizon_used": False,
            "binary_good_stock_label_used": False,
            **evaluation_audit,
            **quantile_audit,
            **diagnostic_audit,
            **smoothing_audit,
        },
        "runtime": {
            "elapsed_seconds": float(time.perf_counter() - started),
            "available_memory_mb_at_start": memory_start,
            "available_memory_mb_at_end": memory_end,
            "minimum_available_memory_mb_observed": minimum_available,
            "reserve_memory_mb": reserve_memory,
            "logical_cpu_count": int(psutil.cpu_count() or 1),
            "adaptive_memory": bool(resources["adaptive_memory"]),
        },
        "training_performed": True,
        "prediction_performed": True,
        "causal_action_value_evaluation_performed": True,
        "portfolio_execution_performed": False,
        "profit_claim_allowed": False,
        "production_policy_selected": False,
        "deep_model_selected": False,
        "report_generation_performed": False,
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run causal transparent dynamic action-value baselines."
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    result = run_study(study_path=args.study, output_root=args.output_root)
    print(
        json.dumps(
            {
                "status": result["status"],
                "fold_count": result["audit"]["fold_count"],
                "prediction_rows": result["audit"]["prediction_rows"],
                "passed_models": result["audit"]["passed_models"],
                "elapsed_seconds": result["runtime"]["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
