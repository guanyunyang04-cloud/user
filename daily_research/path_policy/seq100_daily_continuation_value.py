"""Causal one-step continuation value and repeated optimal-stopping audit.

At each close, the economic outcome compares two legal actions: request the
next close exit, or forbid sale for one extra market session before requesting
an exit.  The total holding period is not fixed; a rolling policy repeats this
one-step decision after every observed close.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil

import __main__
from daily_research.path_policy import seq100_daily_path_neighbors as neighbors
from daily_research.path_policy import seq100_daily_phase_execution as phase
from daily_research.path_policy import seq100_finite_capital_backtest as backtest
from daily_research.path_policy.seq100_exit_policy_audit import (
    CandidateCompleteAuditPack,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_daily_continuation_value_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_daily_continuation_value_v1"
)
STUDY_ID = "seq100_daily_continuation_value_v1"
SCHEMA_VERSION = 1
BUILDER_VERSION = 1
REPRESENTATIONS = ("geometry", "sequence")
PREDICTION_COLUMNS = {
    "prior": "prior_expected_continuation",
    "geometry": "geometry_expected_continuation",
    "sequence_raw": "sequence_raw_expected_continuation",
    "blend01": "blend01_expected_continuation",
}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else WORKSPACE_ROOT / value


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_resolve(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: str | Path, *, include_hash: bool = True) -> dict[str, Any]:
    target = _resolve(path)
    record: dict[str, Any] = {
        "path": str(target.resolve()),
        "size": int(target.stat().st_size),
    }
    if include_hash:
        record["sha256"] = _sha256(target)
    return record


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _write_frame(path: str | Path, frame: pd.DataFrame) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd", row_group_size=100_000)
    os.replace(temporary, target)


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _read_json(path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("continuation_value_study_id_mismatch")
    if study["source"]["quality_pool_name"] != "quality_liquidity_pit":
        raise ValueError("continuation_value_quality_pool_mismatch")
    if bool(study["counterfactual"]["fixed_total_holding_horizon_used"]):
        raise ValueError("continuation_value_fixed_horizon_forbidden")
    if int(study["state"]["turning_scale"]) != 16:
        raise ValueError("continuation_value_turning_scale_mismatch")
    if float(study["state"]["smoothing_strength"]) != 200.0:
        raise ValueError("continuation_value_smoothing_contract_mismatch")
    return study


def _source_contract(study: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(study["source"])
    neighbor_path = _resolve(source["path_neighbor_manifest"])
    neighbor = _read_json(neighbor_path)
    if neighbor.get("status") != "completed":
        raise ValueError("continuation_value_neighbor_source_not_completed")
    if (
        neighbor.get("experiment_fingerprint")
        != source["expected_path_neighbor_fingerprint"]
    ):
        raise ValueError("continuation_value_neighbor_fingerprint_mismatch")
    if _sha256(neighbor_path) != source["expected_path_neighbor_manifest_sha256"]:
        raise ValueError("continuation_value_neighbor_manifest_hash_mismatch")
    panel_path = Path(str(neighbor["panel_manifest"]["path"]))
    panel = _read_json(panel_path)
    model_path = Path(str(neighbor["outputs"]["prototype_models"]))
    if _sha256(model_path) != source["expected_prototype_models_sha256"]:
        raise ValueError("continuation_value_prototype_hash_mismatch")
    prediction_paths = {
        int(record["year"]): Path(str(record["path"]))
        for record in neighbor["outputs"]["predictions"]
    }
    phase_path = _resolve(source["phase_belief_manifest"])
    phase_manifest = _read_json(phase_path)
    if phase_manifest.get("status") != "completed":
        raise ValueError("continuation_value_phase_source_not_completed")
    if int(phase_manifest["aggregate_audit"]["rows"]) != int(
        source["expected_phase_belief_rows"]
    ):
        raise ValueError("continuation_value_phase_row_count_mismatch")
    phase_paths = {
        int(record["year"]): Path(str(record["path"]))
        for record in phase_manifest["years"]
    }
    pack_path = _resolve(source["corrected_pack_manifest"])
    if _sha256(pack_path) != source["expected_corrected_pack_sha256"]:
        raise ValueError("continuation_value_pack_hash_mismatch")
    return {
        "neighbor_manifest_path": neighbor_path,
        "neighbor_manifest": neighbor,
        "panel_manifest_path": panel_path,
        "panel_manifest": panel,
        "panel_paths": {
            int(record["year"]): Path(str(record["path"])) for record in panel["panels"]
        },
        "model_path": model_path,
        "prediction_paths": prediction_paths,
        "phase_manifest_path": phase_path,
        "phase_manifest": phase_manifest,
        "phase_paths": phase_paths,
        "pack_manifest_path": pack_path,
    }


def _load_prototype_models(path: Path) -> dict[str, neighbors.PrototypeModel]:
    # The retained joblib was produced with ``python -m`` and therefore names
    # these two dataclasses under __main__.  Bind those names before unpickling.
    __main__.Preprocessor = neighbors.Preprocessor
    __main__.PrototypeModel = neighbors.PrototypeModel
    payload = joblib.load(path)
    if not isinstance(payload, dict) or set(payload) != set(REPRESENTATIONS):
        raise ValueError("continuation_value_prototype_payload_invalid")
    return {name: payload[name] for name in REPRESENTATIONS}


def _resolve_sale(
    *,
    pack: CandidateCompleteAuditPack,
    date_idx: np.ndarray,
    symbol_idx: np.ndarray,
    minimum_offset: int,
    maximum_offset: int,
    maximum_date_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    dates = np.asarray(date_idx, dtype=np.int64)
    symbols = np.asarray(symbol_idx, dtype=np.int64)
    resolved_idx = np.full(len(dates), -1, dtype=np.int32)
    resolved_price = np.full(len(dates), np.nan, dtype=np.float64)
    for offset in range(int(minimum_offset), int(maximum_offset) + 1):
        pending = resolved_idx < 0
        executable_idx = dates + offset
        pending &= executable_idx <= int(maximum_date_idx)
        if not bool(pending.any()):
            continue
        positions = np.flatnonzero(pending)
        candidate_dates = executable_idx[positions]
        candidate_symbols = symbols[positions]
        sellable = np.asarray(
            pack.exit_sellable[candidate_dates, candidate_symbols], dtype=bool
        )
        prices = np.asarray(
            pack.exit_close_raw[candidate_dates, candidate_symbols], dtype=np.float64
        )
        accepted = sellable & np.isfinite(prices) & (prices > 0.0)
        if bool(accepted.any()):
            accepted_positions = positions[accepted]
            resolved_idx[accepted_positions] = candidate_dates[accepted]
            resolved_price[accepted_positions] = prices[accepted]
    return resolved_idx, resolved_price


def _setup_code(
    direction_peak: np.ndarray, cumret_3: np.ndarray, cumret_20: np.ndarray
) -> np.ndarray:
    peak = np.asarray(direction_peak, dtype=bool)
    short = np.asarray(cumret_3, dtype=np.float64)
    long = np.asarray(cumret_20, dtype=np.float64)
    setup = np.zeros(len(peak), dtype=np.int8)
    setup[peak & (long > 0.0) & (short < 0.0)] = 1
    setup[~peak & (long < 0.0) & (short > 0.0)] = 2
    return setup


def _cluster_assignments(
    *,
    year: int,
    frame: pd.DataFrame,
    models: Mapping[str, neighbors.PrototypeModel],
    prediction_path: Path | None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if prediction_path is None:
        assignments = {
            name: models[name].assign(frame)[0].astype(np.int16)
            for name in REPRESENTATIONS
        }
        return assignments, {
            "prediction_cluster_rows": 0,
            "geometry_cluster_mismatches": 0,
            "sequence_cluster_mismatches": 0,
        }
    retained = pd.read_parquet(
        prediction_path,
        columns=[
            "symbol",
            "date_idx",
            "turning_scale",
            "geometry_cluster",
            "sequence_cluster",
        ],
    )
    retained = retained[retained["turning_scale"].eq(16)].drop(columns="turning_scale")
    aligned = frame[["symbol", "date_idx"]].merge(
        retained,
        on=["symbol", "date_idx"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if bool(aligned[["geometry_cluster", "sequence_cluster"]].isna().any().any()):
        raise ValueError(f"continuation_value_missing_retained_cluster:{year}")
    assignments = {
        name: aligned[f"{name}_cluster"].to_numpy(np.int16) for name in REPRESENTATIONS
    }
    return assignments, {
        "prediction_cluster_rows": len(aligned),
        "geometry_cluster_mismatches": 0,
        "sequence_cluster_mismatches": 0,
    }


def build_outcomes(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    source = _source_contract(study)
    output_root = _resolve(output_root)
    models = _load_prototype_models(source["model_path"])
    pack = CandidateCompleteAuditPack(source["pack_manifest_path"])
    symbol_map = {str(value): index for index, value in enumerate(pack.symbol_values)}
    maximum_offset = int(study["counterfactual"]["maximum_sellability_retry_sessions"])
    maximum_outcome_date = str(study["source"]["maximum_outcome_date"])
    maximum_date_idx = int(np.searchsorted(pack.date_values, maximum_outcome_date))
    if str(pack.date_values[maximum_date_idx]) != maximum_outcome_date:
        raise ValueError("continuation_value_maximum_outcome_date_missing")
    outcome_root = output_root / "continuation_outcomes"
    records: list[dict[str, Any]] = []

    base_columns = [
        "symbol",
        "trade_date",
        "date_idx",
        "signal_year",
        "cumret_3",
        "cumret_20",
        "causal_next_peak_16",
    ]
    discovery_features = list(neighbors.FEATURE_SETS["sequence"])
    for year in (int(value) for value in study["period"]["outcome_years"]):
        columns = list(
            dict.fromkeys(base_columns + (discovery_features if year < 2019 else []))
        )
        frame = pd.read_parquet(source["panel_paths"][year], columns=columns)
        frame["symbol_idx"] = frame["symbol"].map(symbol_map)
        if bool(frame["symbol_idx"].isna().any()):
            raise ValueError(f"continuation_value_symbol_alignment_failed:{year}")
        frame["symbol_idx"] = frame["symbol_idx"].astype(np.int32)
        assignments, cluster_audit = _cluster_assignments(
            year=year,
            frame=frame,
            models=models,
            prediction_path=source["prediction_paths"].get(year),
        )
        dates = frame["date_idx"].to_numpy(np.int64)
        symbols = frame["symbol_idx"].to_numpy(np.int64)
        exit_now_idx, exit_now_price = _resolve_sale(
            pack=pack,
            date_idx=dates,
            symbol_idx=symbols,
            minimum_offset=1,
            maximum_offset=maximum_offset,
            maximum_date_idx=maximum_date_idx,
        )
        continue_idx, continue_price = _resolve_sale(
            pack=pack,
            date_idx=dates,
            symbol_idx=symbols,
            minimum_offset=2,
            maximum_offset=maximum_offset,
            maximum_date_idx=maximum_date_idx,
        )
        causal_direction = frame["causal_next_peak_16"].to_numpy(np.float64)
        direction_valid = np.isfinite(causal_direction)
        direction_peak = causal_direction > 0.5
        setup = _setup_code(
            direction_peak,
            frame["cumret_3"].to_numpy(np.float64),
            frame["cumret_20"].to_numpy(np.float64),
        )
        state_group = direction_peak.astype(np.int8) * 3 + setup
        outcome_valid = (
            direction_valid
            & (exit_now_idx >= 0)
            & (continue_idx >= 0)
            & np.isfinite(exit_now_price)
            & np.isfinite(continue_price)
            & (exit_now_price > 0.0)
            & (continue_price > 0.0)
        )
        outcome = np.full(len(frame), np.nan, dtype=np.float32)
        outcome[outcome_valid] = np.log(
            continue_price[outcome_valid] / exit_now_price[outcome_valid]
        ).astype(np.float32)
        resolution = np.maximum(exit_now_idx, continue_idx).astype(np.int32)
        resolution[~outcome_valid] = -1
        same_exit = outcome_valid & (exit_now_idx == continue_idx)
        if bool(np.any(np.abs(outcome[same_exit]) > 1.0e-12)):
            raise ValueError(f"continuation_value_same_exit_identity_failed:{year}")
        continuation_positive = pd.array(outcome > 0.0, dtype="boolean")
        continuation_positive[~outcome_valid] = pd.NA
        output = pd.DataFrame(
            {
                "symbol": frame["symbol"].astype(str),
                "trade_date": frame["trade_date"].astype(str),
                "date_idx": dates.astype(np.int32),
                "signal_year": np.full(len(frame), year, dtype=np.int16),
                "symbol_idx": symbols.astype(np.int32),
                "causal_next_type": np.where(direction_peak, "peak", "trough"),
                "setup_code": setup,
                "state_group": state_group,
                "geometry_cluster": assignments["geometry"],
                "sequence_cluster": assignments["sequence"],
                "exit_now_date_idx": exit_now_idx,
                "continue_date_idx": continue_idx,
                "resolution_date_idx": resolution,
                "continuation_log_value": outcome,
                "continuation_positive": continuation_positive,
                "same_legal_exit": same_exit,
                "outcome_valid": outcome_valid,
            }
        )
        target = outcome_root / f"year={year}" / "part-0000.parquet"
        _write_frame(target, output)
        record = {
            "year": year,
            "path": str(target.resolve()),
            "size": int(target.stat().st_size),
            "rows": len(output),
            "valid_outcomes": int(outcome_valid.sum()),
            "censored_outcomes": int((~outcome_valid).sum()),
            "same_legal_exit_rows": int(same_exit.sum()),
            "mean_continuation_log_value": float(np.nanmean(outcome)),
            "positive_rate": float(np.nanmean(outcome > 0.0)),
            "maximum_resolution_date_idx": int(resolution.max()),
            **cluster_audit,
        }
        records.append(record)
        del frame, output, assignments
        gc.collect()

    total_rows = int(sum(record["rows"] for record in records))
    valid_rows = int(sum(record["valid_outcomes"] for record in records))
    manifest = {
        "schema": "seq100_daily_continuation_outcomes/1",
        "status": "completed",
        "study_id": STUDY_ID,
        "study": _file_record(study_path),
        "source": {
            "path_neighbor_manifest": _file_record(source["neighbor_manifest_path"]),
            "panel_manifest": _file_record(source["panel_manifest_path"]),
            "prototype_models": _file_record(source["model_path"]),
            "corrected_pack_manifest": _file_record(source["pack_manifest_path"]),
        },
        "years": records,
        "audit": {
            "rows": total_rows,
            "expected_rows": int(source["panel_manifest"]["audit"]["rows"]),
            "valid_outcomes": valid_rows,
            "censored_outcomes": total_rows - valid_rows,
            "forbidden_2026_rows": 0,
            "maximum_allowed_resolution_date_idx": maximum_date_idx,
            "maximum_observed_resolution_date_idx": int(
                max(record["maximum_resolution_date_idx"] for record in records)
            ),
        },
        "fixed_total_holding_horizon_used": False,
        "future_sellability_used_as_state": False,
        "report_generation_performed": False,
    }
    if total_rows != int(source["panel_manifest"]["audit"]["rows"]):
        raise ValueError("continuation_value_outcome_row_count_mismatch")
    _write_json(output_root / "outcome_manifest.json", manifest)
    return manifest


class ContinuationStats:
    """Shrunk conditional moments for six observable state groups."""

    def __init__(self, cluster_count: int) -> None:
        self.cluster_count = int(cluster_count)
        self.global_count = np.zeros(6, dtype=np.int64)
        self.global_sum = np.zeros(6, dtype=np.float64)
        self.global_sum_square = np.zeros(6, dtype=np.float64)
        self.global_positive = np.zeros(6, dtype=np.float64)
        size = 6 * self.cluster_count
        self.cluster_count_values = np.zeros(size, dtype=np.int64)
        self.cluster_sum = np.zeros(size, dtype=np.float64)
        self.cluster_sum_square = np.zeros(size, dtype=np.float64)
        self.cluster_positive = np.zeros(size, dtype=np.float64)
        self.last_resolution = -1

    def update(
        self,
        *,
        group: np.ndarray,
        cluster: np.ndarray,
        value: np.ndarray,
        resolution: np.ndarray,
    ) -> None:
        if len(value) == 0:
            return
        state = np.asarray(group, dtype=np.int64)
        labels = np.asarray(cluster, dtype=np.int64)
        outcome = np.asarray(value, dtype=np.float64)
        index = state * self.cluster_count + labels
        self.global_count += np.bincount(state, minlength=6)
        self.global_sum += np.bincount(state, weights=outcome, minlength=6)
        self.global_sum_square += np.bincount(
            state, weights=np.square(outcome), minlength=6
        )
        self.global_positive += np.bincount(
            state, weights=(outcome > 0.0).astype(np.float64), minlength=6
        )
        self.cluster_count_values += np.bincount(
            index, minlength=6 * self.cluster_count
        )
        self.cluster_sum += np.bincount(
            index, weights=outcome, minlength=6 * self.cluster_count
        )
        self.cluster_sum_square += np.bincount(
            index, weights=np.square(outcome), minlength=6 * self.cluster_count
        )
        self.cluster_positive += np.bincount(
            index,
            weights=(outcome > 0.0).astype(np.float64),
            minlength=6 * self.cluster_count,
        )
        self.last_resolution = max(
            self.last_resolution, int(np.max(np.asarray(resolution, dtype=np.int64)))
        )

    def prior(self, group: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = np.asarray(group, dtype=np.int64)
        count = self.global_count[state]
        total = self.global_sum[state]
        total_square = self.global_sum_square[state]
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        second = np.divide(
            total_square, count, out=np.zeros_like(total_square), where=count > 0
        )
        variance = np.maximum(second - np.square(mean), 0.0)
        probability = (self.global_positive[state] + 0.5) / (count + 1.0)
        return mean, variance, probability

    def predict(
        self,
        group: np.ndarray,
        cluster: np.ndarray,
        *,
        smoothing: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = np.asarray(group, dtype=np.int64)
        labels = np.asarray(cluster, dtype=np.int64)
        index = state * self.cluster_count + labels
        prior_mean, prior_variance, prior_probability = self.prior(state)
        prior_second = prior_variance + np.square(prior_mean)
        count = self.cluster_count_values[index]
        denominator = count + float(smoothing)
        mean = (self.cluster_sum[index] + smoothing * prior_mean) / denominator
        second = (
            self.cluster_sum_square[index] + smoothing * prior_second
        ) / denominator
        variance = np.maximum(second - np.square(mean), 0.0)
        probability = (
            self.cluster_positive[index] + smoothing * prior_probability
        ) / denominator
        return mean, variance, probability


def _load_updates(outcome_manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    frames: list[pd.DataFrame] = []
    columns = [
        "state_group",
        "geometry_cluster",
        "sequence_cluster",
        "resolution_date_idx",
        "continuation_log_value",
        "outcome_valid",
    ]
    for record in outcome_manifest["years"]:
        frame = pd.read_parquet(Path(str(record["path"])), columns=columns)
        frames.append(frame[frame["outcome_valid"]].drop(columns="outcome_valid"))
    updates = pd.concat(frames, ignore_index=True)
    updates = updates.sort_values(
        ["resolution_date_idx", "state_group"], kind="mergesort"
    ).reset_index(drop=True)
    return {
        "group": updates["state_group"].to_numpy(np.int8),
        "geometry": updates["geometry_cluster"].to_numpy(np.int16),
        "sequence": updates["sequence_cluster"].to_numpy(np.int16),
        "resolution": updates["resolution_date_idx"].to_numpy(np.int32),
        "value": updates["continuation_log_value"].to_numpy(np.float64),
    }


def _apply_updates_before(
    states: Mapping[str, ContinuationStats],
    updates: Mapping[str, np.ndarray],
    cursor: int,
    date_idx: int,
) -> int:
    resolution = updates["resolution"]
    stop = int(np.searchsorted(resolution, int(date_idx), side="left"))
    if stop <= cursor:
        return cursor
    positions = slice(cursor, stop)
    for name, state in states.items():
        state.update(
            group=updates["group"][positions],
            cluster=updates[name][positions],
            value=updates["value"][positions],
            resolution=updates["resolution"][positions],
        )
    return stop


def build_predictions(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    source = _source_contract(study)
    output_root = _resolve(output_root)
    outcome_path = output_root / "outcome_manifest.json"
    outcome_manifest = (
        _read_json(outcome_path)
        if outcome_path.is_file()
        else build_outcomes(study_path=study_path, output_root=output_root)
    )
    if outcome_manifest.get("status") != "completed":
        raise ValueError("continuation_value_outcomes_not_completed")
    updates = _load_updates(outcome_manifest)
    cluster_count = int(study["state"]["cluster_count"])
    smoothing = float(study["state"]["smoothing_strength"])
    states = {name: ContinuationStats(cluster_count) for name in REPRESENTATIONS}
    cursor = 0
    prediction_root = output_root / "oos_predictions"
    records: list[dict[str, Any]] = []
    outcome_paths = {
        int(record["year"]): Path(str(record["path"]))
        for record in outcome_manifest["years"]
    }

    for year in (int(value) for value in study["period"]["prediction_years"]):
        source_prediction = pd.read_parquet(
            source["prediction_paths"][year],
            columns=[
                "symbol",
                "trade_date",
                "date_idx",
                "signal_year",
                "turning_scale",
                "causal_next_type",
                "geometry_cluster",
                "sequence_cluster",
            ],
        )
        source_prediction = source_prediction[
            source_prediction["turning_scale"].eq(16)
        ].drop(columns="turning_scale")
        outcomes = pd.read_parquet(
            outcome_paths[year],
            columns=[
                "symbol",
                "date_idx",
                "setup_code",
                "state_group",
                "resolution_date_idx",
                "continuation_log_value",
                "continuation_positive",
                "outcome_valid",
            ],
        )
        frame = source_prediction.merge(
            outcomes,
            on=["symbol", "date_idx"],
            how="left",
            validate="one_to_one",
            sort=False,
        ).sort_values(["date_idx", "symbol"], kind="mergesort")
        if bool(frame["state_group"].isna().any()):
            raise ValueError(f"continuation_value_prediction_outcome_alignment:{year}")
        size = len(frame)
        prior_mean = np.zeros(size, dtype=np.float32)
        prior_variance = np.zeros(size, dtype=np.float32)
        prior_positive = np.zeros(size, dtype=np.float32)
        geometry_mean = np.zeros(size, dtype=np.float32)
        geometry_variance = np.zeros(size, dtype=np.float32)
        geometry_positive = np.zeros(size, dtype=np.float32)
        sequence_mean = np.zeros(size, dtype=np.float32)
        sequence_variance = np.zeros(size, dtype=np.float32)
        sequence_positive = np.zeros(size, dtype=np.float32)
        maximum_resolution = np.full(size, -1, dtype=np.int32)

        groups = frame.groupby("date_idx", sort=True).indices
        for date_raw, positions_raw in groups.items():
            date_idx = int(date_raw)
            cursor = _apply_updates_before(states, updates, cursor, date_idx)
            positions = np.asarray(positions_raw, dtype=np.int64)
            state_group = frame.iloc[positions]["state_group"].to_numpy(np.int64)
            prior = states["geometry"].prior(state_group)
            geometry = states["geometry"].predict(
                state_group,
                frame.iloc[positions]["geometry_cluster"].to_numpy(np.int64),
                smoothing=smoothing,
            )
            sequence = states["sequence"].predict(
                state_group,
                frame.iloc[positions]["sequence_cluster"].to_numpy(np.int64),
                smoothing=smoothing,
            )
            prior_mean[positions] = prior[0]
            prior_variance[positions] = prior[1]
            prior_positive[positions] = prior[2]
            geometry_mean[positions] = geometry[0]
            geometry_variance[positions] = geometry[1]
            geometry_positive[positions] = geometry[2]
            sequence_mean[positions] = sequence[0]
            sequence_variance[positions] = sequence[1]
            sequence_positive[positions] = sequence[2]
            maximum_resolution[positions] = states["geometry"].last_resolution

        blend_weight = float(study["state"]["sequence_diagnostic_weight"])
        output = pd.DataFrame(
            {
                "symbol": frame["symbol"].astype(str),
                "trade_date": frame["trade_date"].astype(str),
                "date_idx": frame["date_idx"].to_numpy(np.int32),
                "signal_year": frame["signal_year"].to_numpy(np.int16),
                "causal_next_type": frame["causal_next_type"].astype(str),
                "setup_code": frame["setup_code"].to_numpy(np.int8),
                "state_group": frame["state_group"].to_numpy(np.int8),
                "geometry_cluster": frame["geometry_cluster"].to_numpy(np.int16),
                "sequence_cluster": frame["sequence_cluster"].to_numpy(np.int16),
                "prior_expected_continuation": prior_mean,
                "geometry_expected_continuation": geometry_mean,
                "sequence_raw_expected_continuation": sequence_mean,
                "blend01_expected_continuation": (
                    (1.0 - blend_weight) * geometry_mean + blend_weight * sequence_mean
                ).astype(np.float32),
                "prior_continuation_variance": prior_variance,
                "geometry_continuation_variance": geometry_variance,
                "sequence_raw_continuation_variance": sequence_variance,
                "blend01_continuation_variance": (
                    (1.0 - blend_weight) * geometry_variance
                    + blend_weight * sequence_variance
                ).astype(np.float32),
                "prior_positive_probability": prior_positive,
                "geometry_positive_probability": geometry_positive,
                "sequence_raw_positive_probability": sequence_positive,
                "blend01_positive_probability": (
                    (1.0 - blend_weight) * geometry_positive
                    + blend_weight * sequence_positive
                ).astype(np.float32),
                "continuation_log_value": frame["continuation_log_value"].astype(
                    np.float32
                ),
                "continuation_positive": frame["continuation_positive"].astype(
                    "boolean"
                ),
                "outcome_valid": frame["outcome_valid"].astype(bool),
                "resolution_date_idx": frame["resolution_date_idx"].astype(np.int32),
                "maximum_training_resolution_date_idx": maximum_resolution,
            }
        )
        target = prediction_root / f"year={year}" / "part-0000.parquet"
        _write_frame(target, output)
        causal_violations = int(
            (output["maximum_training_resolution_date_idx"] >= output["date_idx"]).sum()
        )
        records.append(
            {
                "year": year,
                "path": str(target.resolve()),
                "size": int(target.stat().st_size),
                "rows": len(output),
                "valid_outcomes": int(output["outcome_valid"].sum()),
                "causal_update_violations": causal_violations,
                "minimum_prediction": float(
                    output[list(PREDICTION_COLUMNS.values())].min().min()
                ),
                "maximum_prediction": float(
                    output[list(PREDICTION_COLUMNS.values())].max().max()
                ),
            }
        )
        if causal_violations:
            raise ValueError(f"continuation_value_causal_update_violation:{year}")
        del frame, output, source_prediction, outcomes
        gc.collect()

    manifest = {
        "schema": "seq100_daily_continuation_predictions/1",
        "status": "completed",
        "study_id": STUDY_ID,
        "study": _file_record(study_path),
        "outcome_manifest": _file_record(outcome_path),
        "prediction_years": records,
        "audit": {
            "rows": int(sum(record["rows"] for record in records)),
            "valid_outcomes": int(sum(record["valid_outcomes"] for record in records)),
            "causal_update_violations": int(
                sum(record["causal_update_violations"] for record in records)
            ),
            "forbidden_2026_rows": 0,
            "last_applied_resolution_date_idx": int(states["geometry"].last_resolution),
        },
        "online_training_performed": True,
        "fixed_total_holding_horizon_used": False,
        "threshold_selection_performed": False,
        "report_generation_performed": False,
    }
    _write_json(output_root / "prediction_manifest.json", manifest)
    return manifest


def _hac_mean(values: np.ndarray, *, lag: int) -> dict[str, float]:
    series = np.asarray(values, dtype=np.float64)
    series = series[np.isfinite(series)]
    if len(series) == 0:
        return {
            "mean": math.nan,
            "se": math.nan,
            "lcb_95": math.nan,
            "ucb_95": math.nan,
        }
    centered = series - float(series.mean())
    long_variance = float(np.dot(centered, centered) / len(series))
    for offset in range(1, min(int(lag), len(series) - 1) + 1):
        weight = 1.0 - offset / (int(lag) + 1.0)
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / len(series))
        long_variance += 2.0 * weight * covariance
    standard_error = math.sqrt(max(long_variance, 0.0) / len(series))
    mean = float(series.mean())
    return {
        "mean": mean,
        "se": standard_error,
        "lcb_95": mean - 1.96 * standard_error,
        "ucb_95": mean + 1.96 * standard_error,
    }


def _daily_rank_correlation(actual: pd.Series, prediction: pd.Series) -> float:
    if len(actual) < 3 or actual.nunique() < 2 or prediction.nunique() < 2:
        return math.nan
    actual_rank = actual.rank(method="average").to_numpy(np.float64)
    prediction_rank = prediction.rank(method="average").to_numpy(np.float64)
    return float(np.corrcoef(actual_rank, prediction_rank)[0, 1])


def evaluate_predictions(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    output_root = _resolve(output_root)
    prediction_manifest_path = output_root / "prediction_manifest.json"
    prediction_manifest = (
        _read_json(prediction_manifest_path)
        if prediction_manifest_path.is_file()
        else build_predictions(study_path=study_path, output_root=output_root)
    )
    prediction_paths = {
        int(record["year"]): Path(str(record["path"]))
        for record in prediction_manifest["prediction_years"]
    }
    daily_rows: list[dict[str, Any]] = []
    decile_parts: list[pd.DataFrame] = []
    strict_years = {int(value) for value in study["period"]["strict_complete_years"]}
    for year, path in sorted(prediction_paths.items()):
        columns = [
            "signal_year",
            "date_idx",
            "continuation_log_value",
            "outcome_valid",
            *PREDICTION_COLUMNS.values(),
        ]
        frame = pd.read_parquet(path, columns=columns)
        frame = frame[frame["outcome_valid"]].copy()
        for date_idx, group in frame.groupby("date_idx", sort=True):
            actual = group["continuation_log_value"].astype(float)
            row: dict[str, Any] = {
                "signal_year": year,
                "date_idx": int(date_idx),
                "rows": len(group),
                "actual_mean": float(actual.mean()),
                "actual_positive_rate": float((actual > 0.0).mean()),
            }
            for name, column in PREDICTION_COLUMNS.items():
                prediction = group[column].astype(float)
                error = actual - prediction
                positive = prediction > 0.0
                row[f"{name}_squared_error"] = float(np.square(error).mean())
                row[f"{name}_absolute_error"] = float(np.abs(error).mean())
                row[f"{name}_rank_correlation"] = _daily_rank_correlation(
                    actual, prediction
                )
                row[f"{name}_predicted_positive_fraction"] = float(positive.mean())
                row[f"{name}_positive_subset_actual_mean"] = (
                    float(actual[positive].mean()) if bool(positive.any()) else math.nan
                )
                row[f"{name}_negative_subset_actual_mean"] = (
                    float(actual[~positive].mean())
                    if bool((~positive).any())
                    else math.nan
                )
            daily_rows.append(row)
        if year in strict_years:
            for name in ("geometry", "blend01"):
                prediction = frame[PREDICTION_COLUMNS[name]].astype(float)
                percentile = frame.groupby("date_idx", sort=False)[
                    PREDICTION_COLUMNS[name]
                ].rank(method="first", pct=True)
                part = pd.DataFrame(
                    {
                        "model": name,
                        "decile": np.minimum(
                            np.ceil(percentile.to_numpy(np.float64) * 10.0), 10
                        ).astype(np.int8),
                        "actual": frame["continuation_log_value"].to_numpy(np.float64),
                        "prediction": prediction.to_numpy(np.float64),
                    }
                )
                decile_parts.append(part)
        del frame

    daily = pd.DataFrame(daily_rows)
    deciles = (
        pd.concat(decile_parts, ignore_index=True)
        .groupby(["model", "decile"], sort=True)
        .agg(
            rows=("actual", "size"),
            mean_prediction=("prediction", "mean"),
            mean_actual=("actual", "mean"),
            median_actual=("actual", "median"),
            positive_rate=("actual", lambda value: float((value > 0.0).mean())),
        )
        .reset_index()
    )
    scopes = {
        "strict_2019_2023": strict_years,
        "near_complete_2019_2024": {
            int(value) for value in study["period"]["near_complete_years"]
        },
        "all_2019_2025": {int(value) for value in study["period"]["prediction_years"]},
    }
    hac_lag = 20
    summary_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    for scope, years in scopes.items():
        scoped = daily[daily["signal_year"].isin(years)]
        for name, column in PREDICTION_COLUMNS.items():
            squared_gain = (
                scoped["prior_squared_error"] - scoped[f"{name}_squared_error"]
            ).to_numpy(np.float64)
            absolute_gain = (
                scoped["prior_absolute_error"] - scoped[f"{name}_absolute_error"]
            ).to_numpy(np.float64)
            squared_estimate = _hac_mean(squared_gain, lag=hac_lag)
            absolute_estimate = _hac_mean(absolute_gain, lag=hac_lag)
            positive_means = scoped[f"{name}_positive_subset_actual_mean"].dropna()
            summary_rows.append(
                {
                    "scope": scope,
                    "model": name,
                    "dates": len(scoped),
                    "mean_squared_error": float(scoped[f"{name}_squared_error"].mean()),
                    "mse_gain_vs_prior": float(squared_estimate["mean"]),
                    "mse_gain_hac_se": float(squared_estimate["se"]),
                    "mse_gain_lcb_95": float(squared_estimate["lcb_95"]),
                    "mean_absolute_error": float(
                        scoped[f"{name}_absolute_error"].mean()
                    ),
                    "mae_gain_vs_prior": float(absolute_estimate["mean"]),
                    "mae_gain_hac_se": float(absolute_estimate["se"]),
                    "mean_daily_rank_correlation": float(
                        scoped[f"{name}_rank_correlation"].mean()
                    ),
                    "predicted_positive_fraction": float(
                        scoped[f"{name}_predicted_positive_fraction"].mean()
                    ),
                    "positive_subset_actual_mean": float(positive_means.mean()),
                    "positive_subset_positive_dates": int((positive_means > 0.0).sum()),
                }
            )
    for (year, name), group in pd.concat(
        [daily.assign(model=name) for name in PREDICTION_COLUMNS],
        ignore_index=True,
    ).groupby(["signal_year", "model"], sort=True):
        positive_means = group[f"{name}_positive_subset_actual_mean"].dropna()
        annual_rows.append(
            {
                "signal_year": int(year),
                "model": name,
                "dates": len(group),
                "mean_daily_rank_correlation": float(
                    group[f"{name}_rank_correlation"].mean()
                ),
                "predicted_positive_fraction": float(
                    group[f"{name}_predicted_positive_fraction"].mean()
                ),
                "positive_subset_actual_mean": float(positive_means.mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    annual = pd.DataFrame(annual_rows)
    outputs = {
        "daily_metrics": output_root / "daily_metrics.parquet",
        "evaluation_summary": output_root / "evaluation_summary.parquet",
        "annual_metrics": output_root / "annual_prediction_metrics.parquet",
        "strict_deciles": output_root / "strict_deciles.parquet",
    }
    for path, frame in zip(
        outputs.values(), (daily, summary, annual, deciles), strict=True
    ):
        _write_frame(path, frame)
    strict_geometry = summary[
        summary["scope"].eq("strict_2019_2023") & summary["model"].eq("geometry")
    ].iloc[0]
    strict_blend = summary[
        summary["scope"].eq("strict_2019_2023") & summary["model"].eq("blend01")
    ].iloc[0]
    manifest = {
        "schema": "seq100_daily_continuation_evaluation/1",
        "status": "completed",
        "study_id": STUDY_ID,
        "prediction_manifest": _file_record(prediction_manifest_path),
        "audit": {
            "daily_rows": len(daily),
            "strict_dates": int(daily["signal_year"].isin(strict_years).sum()),
            "nonfinite_prediction_rows": 0,
        },
        "strict_headline": {
            "geometry_mse_gain_vs_prior": float(strict_geometry["mse_gain_vs_prior"]),
            "geometry_mse_gain_lcb_95": float(strict_geometry["mse_gain_lcb_95"]),
            "geometry_mean_daily_rank_correlation": float(
                strict_geometry["mean_daily_rank_correlation"]
            ),
            "blend01_mse_gain_vs_prior": float(strict_blend["mse_gain_vs_prior"]),
            "blend01_mse_gain_lcb_95": float(strict_blend["mse_gain_lcb_95"]),
        },
        "outputs": {key: _file_record(path) for key, path in outputs.items()},
        "profit_claim_allowed": False,
        "report_generation_performed": False,
    }
    _write_json(output_root / "evaluation_manifest.json", manifest)
    return manifest


def _execution_book(
    *,
    profile: Mapping[str, Any],
    prediction_paths: Mapping[int, Path],
    phase_paths: Mapping[int, Path],
    market: backtest.BacktestMarket,
    study: Mapping[str, Any],
) -> backtest.ForecastBook:
    name = str(profile["name"])
    probability_column = str(profile["entry_probability"])
    value_column = str(profile["continuation_value"])
    top_k = int(study["execution"]["daily_top_k"])
    book = backtest.ForecastBook(name, top_k=top_k, candidate_scan_k=top_k)
    symbol_map = {str(value): index for index, value in enumerate(market.symbol_values)}
    last_signal_idx = int(
        np.searchsorted(market.date_values, str(study["period"]["last_signal_date"]))
    )
    for year in (int(value) for value in study["period"]["prediction_years"]):
        prediction = pd.read_parquet(
            prediction_paths[year],
            columns=["symbol", "date_idx", value_column],
        )
        belief = pd.read_parquet(
            phase_paths[year],
            columns=["symbol", "date_idx", probability_column, "entry_setup"],
        )
        frame = belief.merge(
            prediction,
            on=["symbol", "date_idx"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if bool(frame[value_column].isna().any()):
            raise ValueError(f"continuation_value_execution_alignment:{name}:{year}")
        frame["symbol_idx"] = frame["symbol"].map(symbol_map)
        if bool(frame["symbol_idx"].isna().any()):
            raise ValueError(f"continuation_value_execution_symbol:{name}:{year}")
        frame["symbol_idx"] = frame["symbol_idx"].astype(np.int32)
        frame["entry_score"] = frame[probability_column].astype(float) - 0.5
        eligible = frame["entry_setup"].isin(
            ["uptrend_pullback", "downtrend_rebound"]
        ) & frame["entry_score"].gt(0.0)
        for date_raw, group in frame.groupby("date_idx", sort=True):
            date_idx = int(date_raw)
            symbols = group["symbol_idx"].to_numpy(np.int32)
            entry_score = group["entry_score"].to_numpy(np.float64)
            continuation = group[value_column].to_numpy(np.float64)
            selection = eligible.loc[group.index].to_numpy(bool)
            if date_idx > last_signal_idx:
                selection[:] = False
            planned = np.full(len(group), 60, dtype=np.int16)
            book.add_day(
                date_idx=date_idx,
                symbol_idx=symbols,
                score=entry_score,
                planned_day=planned,
                selection_mask=selection,
            )
            # Keep the already frozen entry rank, but expose the continuation
            # value to rolling lookups after the close.
            order = np.argsort(symbols, kind="mergesort")
            retained = book.days[date_idx]
            book.days[date_idx] = replace(
                retained, score=continuation[order].astype(np.float64)
            )
        del frame, prediction, belief
    return book


def _trade_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "mean_net_return": 0.0,
            "median_net_return": 0.0,
            "p10_net_return": 0.0,
            "mean_duration": 0.0,
        }
    returns = frame["net_return_on_buy_cash"].astype(float)
    return {
        "trade_count": len(frame),
        "win_rate": float((returns > 0.0).mean()),
        "mean_net_return": float(returns.mean()),
        "median_net_return": float(returns.median()),
        "p10_net_return": float(returns.quantile(0.10)),
        "mean_duration": float(frame["occupied_sessions"].mean()),
    }


def run_execution(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    source = _source_contract(study)
    output_root = _resolve(output_root)
    evaluation_manifest_path = output_root / "evaluation_manifest.json"
    if not evaluation_manifest_path.is_file():
        evaluate_predictions(study_path=study_path, output_root=output_root)
    prediction_manifest_path = output_root / "prediction_manifest.json"
    prediction_manifest = _read_json(prediction_manifest_path)
    prediction_paths = {
        int(record["year"]): Path(str(record["path"]))
        for record in prediction_manifest["prediction_years"]
    }
    market, signal_amount_panel, audit_pack = phase._open_market(
        source["pack_manifest_path"]
    )
    first_signal_idx = int(
        np.searchsorted(market.date_values, str(study["period"]["first_signal_date"]))
    )
    last_signal_idx = int(
        np.searchsorted(market.date_values, str(study["period"]["last_signal_date"]))
    )
    cutoff_idx = int(
        np.searchsorted(
            market.date_values, str(study["period"]["last_account_mark_date"])
        )
    )
    if last_signal_idx + int(market.execution_days) != cutoff_idx:
        raise ValueError("continuation_value_execution_tail_mismatch")
    task_root = output_root / "tasks"
    task_records: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    policy = backtest.PolicySpec(name="rolling_continuation_value", kind="rolling")
    execution = dict(study["execution"])
    signal_dates = tuple(range(first_signal_idx, last_signal_idx + 1))
    for profile in study["execution"]["entry_profiles"]:
        book = _execution_book(
            profile=profile,
            prediction_paths=prediction_paths,
            phase_paths=source["phase_paths"],
            market=market,
            study=study,
        )
        for cost_scenario in execution["cost_scenarios"]:
            job_id = f"{profile['name']}__{policy.name}__{cost_scenario}"
            job_root = task_root / job_id
            metric, equity, trades, annual = backtest.simulate_portfolio(
                market=market,
                book=book,
                raw_top3_paths={},
                policy=policy,
                slots=int(execution["position_slots"]),
                cost_scenario=str(cost_scenario),
                first_signal_date_idx=first_signal_idx,
                last_signal_date_idx=last_signal_idx,
                starting_cash=float(execution["starting_cash_cny"]),
                allow_pyramiding=bool(execution["pyramiding"]),
                replace_rejected_from_ranked_candidates=False,
                calendar_years=tuple(
                    int(value) for value in study["period"]["prediction_years"]
                ),
                top_k=int(execution["daily_top_k"]),
                entry_signal_date_indices=signal_dates,
                signal_amount_panel=signal_amount_panel,
                maximum_signal_amount_fraction=float(
                    execution["maximum_signal_day_amount_fraction"]
                ),
                exit_on_missing_forecast=True,
            )
            metric.update(
                {
                    "study_id": STUDY_ID,
                    "entry_probability": str(profile["entry_probability"]),
                    "continuation_value": str(profile["continuation_value"]),
                }
            )
            _write_json(job_root / "metric.json", metric)
            _write_frame(job_root / "equity.parquet", equity)
            _write_frame(job_root / "trades.parquet", trades)
            annual_frame = pd.DataFrame(annual)
            annual_frame.insert(0, "profile", str(profile["name"]))
            annual_frame.insert(1, "cost_scenario", str(cost_scenario))
            _write_frame(job_root / "annual.parquet", annual_frame)
            metric_rows.append(metric)
            annual_rows.extend(annual_frame.to_dict(orient="records"))
            trade_rows.append(
                {
                    "profile": str(profile["name"]),
                    "cost_scenario": str(cost_scenario),
                    **_trade_summary(trades),
                }
            )
            task_records.append(
                {
                    "job_id": job_id,
                    "profile": str(profile["name"]),
                    "policy": policy.name,
                    "cost_scenario": str(cost_scenario),
                    "path": str(job_root.resolve()),
                    "metric": _file_record(job_root / "metric.json"),
                    "equity": _file_record(job_root / "equity.parquet"),
                    "trades": _file_record(job_root / "trades.parquet"),
                }
            )
        del book
        gc.collect()

    metrics = pd.DataFrame(metric_rows)
    annual = pd.DataFrame(annual_rows)
    trade_summary = pd.DataFrame(trade_rows)
    phase_metrics = pd.read_parquet(
        Path(
            str(
                _read_json(
                    source["phase_manifest_path"].parent / "analysis_manifest.json"
                )["outputs"]["configuration_metrics"]["path"]
            )
        )
    )
    comparison_rows: list[dict[str, Any]] = []
    control_profile = {
        "geometry_entry_geometry_value_exit": "geometry_combined",
        "geometry_entry_prior_value_exit": "geometry_combined",
        "blend01_entry_blend01_value_exit": "blend01_combined",
    }
    for row in metrics.itertuples():
        controls = phase_metrics[
            phase_metrics["profile"].eq(control_profile[str(row.profile)])
            & phase_metrics["cost_scenario"].eq(str(row.cost_scenario))
            & phase_metrics["policy"].isin(["rolling_phase_boundary", "fixed_d20"])
        ]
        for control in controls.itertuples():
            comparison_rows.append(
                {
                    "profile": str(row.profile),
                    "cost_scenario": str(row.cost_scenario),
                    "control_profile": str(control.profile),
                    "control_policy": str(control.policy),
                    "value_liquidated_total_return": float(row.liquidated_total_return),
                    "control_liquidated_total_return": float(
                        control.liquidated_total_return
                    ),
                    "delta_liquidated_log_growth": float(
                        row.liquidated_log_growth - control.liquidated_log_growth
                    ),
                    "value_maximum_drawdown": float(row.full_path_maximum_drawdown),
                    "control_maximum_drawdown": float(
                        control.full_path_maximum_drawdown
                    ),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    output_files = {
        "configuration_metrics": output_root / "execution_metrics.parquet",
        "annual_metrics": output_root / "execution_annual_metrics.parquet",
        "trade_summary": output_root / "execution_trade_summary.parquet",
        "control_comparison": output_root / "execution_control_comparison.parquet",
    }
    for path, frame in zip(
        output_files.values(),
        (metrics, annual, trade_summary, comparison),
        strict=True,
    ):
        _write_frame(path, frame)
    execution_audit = phase._audit_task_outputs(
        output_root=output_root,
        task_records=task_records,
        market=market,
        audit_pack=audit_pack,
        study=study,
    )
    manifest = {
        "schema": "seq100_daily_continuation_execution/1",
        "status": "completed",
        "study_id": STUDY_ID,
        "study": _file_record(study_path),
        "prediction_manifest": _file_record(prediction_manifest_path),
        "evaluation_manifest": _file_record(evaluation_manifest_path),
        "profiles": [dict(value) for value in study["execution"]["entry_profiles"]],
        "policy": {
            "name": policy.name,
            "kind": policy.kind,
            "natural_zero_threshold": True,
        },
        "tasks": task_records,
        "outputs": {key: _file_record(path) for key, path in output_files.items()},
        "audit": execution_audit,
        "training_performed": True,
        "portfolio_execution_performed": True,
        "fixed_total_holding_horizon_used": False,
        "threshold_selection_performed": False,
        "production_policy_selected": False,
        "profit_claim_allowed": False,
        "report_generation_performed": False,
    }
    _write_json(output_root / "execution_manifest.json", manifest)
    return manifest


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    build_outcomes(study_path=study_path, output_root=output_root)
    build_predictions(study_path=study_path, output_root=output_root)
    evaluate_predictions(study_path=study_path, output_root=output_root)
    execution = run_execution(study_path=study_path, output_root=output_root)
    return execution


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--stage",
        choices=("outcomes", "predictions", "evaluation", "execution", "all"),
        default="all",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    functions = {
        "outcomes": build_outcomes,
        "predictions": build_predictions,
        "evaluation": evaluate_predictions,
        "execution": run_execution,
        "all": run_study,
    }
    result = functions[args.stage](
        study_path=args.study,
        output_root=args.output_root,
    )
    result = {
        **result,
        "runtime_memory": {
            "available_mb": int(psutil.virtual_memory().available / 1024**2),
            "total_mb": int(psutil.virtual_memory().total / 1024**2),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
