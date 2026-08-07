"""Independent validation for causal transparent action-value baselines."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_dynamic_action_value_baselines as baseline
from daily_research.path_policy import seq100_market_replay as replay

VALIDATION_SCHEMA = "seq100_dynamic_action_value_baselines_validation/1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_frame(
    record: Mapping[str, Any], *, blocking: defaultdict[str, int], name: str
) -> pd.DataFrame:
    path = Path(str(record["path"]))
    if not path.is_file() or replay.sha256(path) != str(record["sha256"]):
        blocking[f"{name}_hash"] += 1
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if len(frame) != int(record["rows"]):
        blocking[f"{name}_rows"] += 1
    return frame


def _load_snapshot(
    record: Mapping[str, Any], *, blocking: defaultdict[str, int]
) -> dict[str, np.ndarray]:
    path = Path(str(record["path"]))
    if not path.is_file() or replay.sha256(path) != str(record["sha256"]):
        blocking["state_hash"] += 1
        return {}
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _safe_offsets(keys: np.ndarray, wanted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.searchsorted(keys, wanted)
    found = offsets < len(keys)
    found_indices = np.flatnonzero(found)
    found[found_indices] = keys[offsets[found]] == wanted[found_indices]
    return offsets, found


def validate(
    *,
    study_path: str | Path = baseline.DEFAULT_STUDY_PATH,
    output_root: str | Path | None = None,
    trace_rows_per_partition: int | None = None,
) -> dict[str, Any]:
    study_file = baseline._resolve(study_path)
    study = baseline.load_study(study_file)
    root = baseline._resolve(output_root or study["output_root"])
    manifest_path = root / "manifest.json"
    manifest = baseline._read_json(manifest_path)
    blocking: defaultdict[str, int] = defaultdict(int)
    if manifest.get("status") != "completed":
        blocking["manifest_status"] += 1
    if manifest.get("study_id") != baseline.STUDY_ID:
        blocking["manifest_study_id"] += 1
    if replay.sha256(study_file) != str(dict(manifest["study"])["sha256"]):
        blocking["study_hash"] += 1

    _contract, prefix_manifest, input_manifest, version_records, _feature_index = (
        baseline._source_contract(study)
    )
    if int(dict(manifest["source_audit"])["model_input_rows"]) != int(
        input_manifest["row_count"]
    ):
        blocking["source_input_rows"] += 1
    if int(dict(manifest["source_audit"])["prefix_experience_version_rows"]) != int(
        dict(prefix_manifest["audit"])["experience_version_rows"]
    ):
        blocking["source_version_rows"] += 1

    _specs, coordinate_names, _market_names, family_map, _matched_indices = (
        baseline._coordinate_metadata(study)
    )
    family_names = list(family_map)
    row_index = pd.read_parquet(
        dict(input_manifest["row_index"])["path"],
        columns=["date_idx", "symbol_idx", "trade_date"],
    )
    input_keys = baseline._packed_key(row_index["date_idx"], row_index["symbol_idx"])
    coordinate_records = {
        int(record["year"]): dict(record) for record in manifest["coordinates"]
    }
    expected_years = set(range(2012, 2026))
    if set(coordinate_records) != expected_years:
        blocking["coordinate_year_coverage"] += len(
            expected_years.symmetric_difference(coordinate_records)
        )
    coordinate_cache: dict[int, pd.DataFrame] = {}
    coordinate_rows = 0
    for year, record in sorted(coordinate_records.items()):
        frame = _read_frame(record, blocking=blocking, name="coordinate")
        coordinate_cache[year] = frame
        coordinate_rows += len(frame)
        required = {
            "input_row_idx",
            "trade_date",
            "date_idx",
            "symbol_idx",
            *coordinate_names,
        }
        if required - set(frame.columns):
            blocking["coordinate_columns"] += 1
            continue
        if frame["trade_date"].astype(str).str[:4].astype(int).ne(year).any():
            blocking["coordinate_year"] += 1
        if frame["trade_date"].astype(str).str[:4].astype(int).ge(2026).any():
            blocking["coordinate_forbidden_year"] += 1
        positions = frame["input_row_idx"].to_numpy(np.int64)
        if len(positions) and not np.array_equal(
            positions, np.arange(positions[0], positions[0] + len(positions))
        ):
            blocking["coordinate_input_positions"] += 1
        if len(positions):
            expected_keys = input_keys[positions]
            actual_keys = baseline._packed_key(frame["date_idx"], frame["symbol_idx"])
            if not np.array_equal(expected_keys, actual_keys):
                blocking["coordinate_key_alignment"] += 1
        values = frame[coordinate_names].to_numpy(np.float64)
        if np.isinf(values).any():
            blocking["coordinate_infinity"] += 1
        finite = np.isfinite(values)
        if bool(((values[finite] < -1.0e-6) | (values[finite] > 1.000001)).any()):
            blocking["coordinate_range"] += 1
    if coordinate_rows != int(input_manifest["row_count"]):
        blocking["coordinate_total_rows"] += 1

    state_records = {
        (str(record["cost_scenario"]), int(record["oos_year"])): dict(record)
        for record in manifest["model_states"]
    }
    prediction_records = {
        (str(record["cost_scenario"]), int(record["oos_year"])): dict(record)
        for record in manifest["predictions"]
    }
    expected_folds = {
        (cost, year)
        for cost in ("base", "double_slippage")
        for year in range(2013, 2026)
    }
    if set(state_records) != expected_folds:
        blocking["state_fold_coverage"] += len(
            expected_folds.symmetric_difference(state_records)
        )
    if set(prediction_records) != expected_folds:
        blocking["prediction_fold_coverage"] += len(
            expected_folds.symmetric_difference(prediction_records)
        )

    folds = {
        (str(record["cost_scenario"]), int(record["oos_year"])): dict(record)
        for record in manifest["folds"]
    }
    if set(folds) != expected_folds:
        blocking["manifest_fold_coverage"] += len(
            expected_folds.symmetric_difference(folds)
        )
    trace_rows = int(
        trace_rows_per_partition
        if trace_rows_per_partition is not None
        else dict(study["resources"])["validator_trace_rows_per_partition"]
    )
    traced = 0
    prediction_rows = 0
    for cost_scenario, oos_year in sorted(expected_folds):
        fold = folds.get((cost_scenario, oos_year))
        state_record = state_records.get((cost_scenario, oos_year))
        prediction_record = prediction_records.get((cost_scenario, oos_year))
        if fold is None or state_record is None or prediction_record is None:
            continue
        cutoff = oos_year - 1
        if int(fold["training_cutoff_year"]) != cutoff:
            blocking["training_cutoff"] += 1
        if int(fold["maximum_training_label_as_of_year"]) > cutoff:
            blocking["training_label_as_of_leak"] += 1
        if int(fold["maximum_training_signal_year"]) >= oos_year:
            blocking["training_signal_year_leak"] += 1
        for training_record in fold["training_versions"]:
            signal_year = int(training_record["signal_year"])
            as_of_year = int(training_record["as_of_year"])
            if signal_year >= oos_year or as_of_year > cutoff:
                blocking["training_version_leak"] += 1
            expected = version_records.get((cost_scenario, signal_year, as_of_year))
            if expected is None:
                blocking["training_version_source"] += 1
            elif str(expected["sha256"]) != str(training_record["sha256"]):
                blocking["training_version_hash"] += 1

        snapshot = _load_snapshot(state_record, blocking=blocking)
        if not snapshot:
            continue
        if int(snapshot["training_cutoff_year"][0]) != cutoff:
            blocking["state_cutoff"] += 1
        if int(snapshot["maximum_label_as_of_year"][0]) > cutoff:
            blocking["state_as_of_leak"] += 1
        if int(snapshot["maximum_signal_year"][0]) >= oos_year:
            blocking["state_signal_leak"] += 1
        if [str(value) for value in snapshot["coordinate_names"]] != coordinate_names:
            blocking["state_coordinate_names"] += 1
        if [str(value) for value in snapshot["family_names"]] != family_names:
            blocking["state_family_names"] += 1

        prediction = _read_frame(
            prediction_record, blocking=blocking, name="prediction"
        )
        prediction_rows += len(prediction)
        required_prediction = {
            "trade_date",
            "date_idx",
            "symbol_idx",
            "input_row_idx",
            "actual_buy_advantage_vs_cash",
            "evaluation_sessions_after_resolution",
            "predicted_prior",
            "predicted_market",
            "predicted_additive",
            "predicted_matched_history",
            *{f"effect_{family}" for family in family_names},
        }
        if required_prediction - set(prediction.columns):
            blocking["prediction_columns"] += 1
            continue
        if prediction["trade_date"].astype(str).str[:4].astype(int).ne(oos_year).any():
            blocking["prediction_year"] += 1
        if prediction["trade_date"].astype(str).str[:4].astype(int).ge(2026).any():
            blocking["prediction_forbidden_year"] += 1
        if prediction["training_cutoff_year"].astype(int).ne(cutoff).any():
            blocking["prediction_cutoff"] += 1
        if prediction["maximum_training_label_as_of_year"].astype(int).gt(cutoff).any():
            blocking["prediction_as_of_leak"] += 1
        if prediction["maximum_training_signal_year"].astype(int).ge(oos_year).any():
            blocking["prediction_signal_leak"] += 1
        maturity = int(dict(study["target"])["minimum_sessions_after_resolution"])
        if prediction["evaluation_sessions_after_resolution"].astype(int).lt(
            maturity
        ).any():
            blocking["prediction_immature_evaluation"] += 1
        numeric = prediction.select_dtypes(include=["number"]).to_numpy(np.float64)
        if np.isinf(numeric).any():
            blocking["prediction_infinity"] += 1

        coordinate = coordinate_cache[oos_year]
        first_input = int(coordinate["input_row_idx"].iloc[0])
        sample = prediction.sample(
            n=min(trace_rows, len(prediction)),
            random_state=20260807 + 1009 * oos_year + (0 if cost_scenario == "base" else 1),
        )
        local = sample["input_row_idx"].to_numpy(np.int64) - first_input
        if bool((local < 0).any()) or bool((local >= len(coordinate)).any()):
            blocking["trace_coordinate_position"] += 1
            continue
        trace_values = coordinate.iloc[local][coordinate_names].to_numpy(np.float64)
        recomputed = baseline.predict_from_snapshot(trace_values, snapshot)
        comparisons = (
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
        for name in comparisons:
            if not np.allclose(
                sample[name].to_numpy(np.float64),
                recomputed[name],
                atol=2.0e-6,
                rtol=0.0,
                equal_nan=True,
            ):
                blocking[f"trace_{name}"] += 1
        if not np.array_equal(
            sample["matched_cell_code"].to_numpy(np.int32),
            recomputed["matched_cell_code"].astype(np.int32),
        ):
            blocking["trace_cell_code"] += 1
        for family in family_names:
            name = f"effect_{family}"
            if not np.allclose(
                sample[name].to_numpy(np.float64),
                recomputed[name],
                atol=2.0e-6,
                rtol=0.0,
            ):
                blocking[f"trace_effect_{family}"] += 1

        evaluation_record = baseline.select_evaluation_record(
            version_records=version_records,
            cost_scenario=cost_scenario,
            signal_year=oos_year,
        )
        label = pd.read_parquet(
            evaluation_record["path"],
            columns=[
                "date_idx",
                "symbol_idx",
                "buy_advantage_vs_cash",
                "sessions_after_resolution",
                "naturally_resolved",
            ],
        )
        label_keys = baseline._packed_key(label["date_idx"], label["symbol_idx"])
        sample_keys = baseline._packed_key(sample["date_idx"], sample["symbol_idx"])
        offsets, found = _safe_offsets(label_keys, sample_keys)
        if not bool(found.all()):
            blocking["trace_label_alignment"] += 1
        else:
            source_rows = label.iloc[offsets]
            if not source_rows["naturally_resolved"].astype(bool).all():
                blocking["trace_label_natural"] += 1
            if source_rows["sessions_after_resolution"].astype(int).lt(maturity).any():
                blocking["trace_label_maturity"] += 1
            if not np.allclose(
                source_rows["buy_advantage_vs_cash"].to_numpy(np.float64),
                sample["actual_buy_advantage_vs_cash"].to_numpy(np.float64),
                atol=1.0e-12,
                rtol=0.0,
            ):
                blocking["trace_actual_target"] += 1
        traced += len(sample)

    outputs = dict(manifest["outputs"])
    for name in (
        "training_version_updates",
        "fold_summary",
        "daily_model_metrics",
        "annual_model_metrics",
        "aggregate_model_metrics",
        "annual_prediction_quantiles",
        "aggregate_prediction_quantiles",
        "prediction_quantile_spreads",
        "annual_coordinate_diagnostics",
        "aggregate_coordinate_diagnostics",
        "coordinate_extreme_spreads",
        "annual_matched_smoothing_sensitivity",
        "aggregate_matched_smoothing_sensitivity",
    ):
        _read_frame(outputs[name], blocking=blocking, name=name)
    gate_path = Path(str(dict(outputs["promotion_gate"])["path"]))
    if not gate_path.is_file() or replay.sha256(gate_path) != str(
        dict(outputs["promotion_gate"])["sha256"]
    ):
        blocking["promotion_gate_hash"] += 1
        gate = {}
    else:
        gate = baseline._read_json(gate_path)
    if bool(gate.get("profit_claim_allowed", True)):
        blocking["promotion_profit_claim"] += 1
    if bool(manifest.get("portfolio_execution_performed")):
        blocking["manifest_portfolio_execution"] += 1
    if bool(manifest.get("profit_claim_allowed")):
        blocking["manifest_profit_claim"] += 1
    if bool(manifest.get("production_policy_selected")):
        blocking["manifest_production_policy"] += 1
    if bool(manifest.get("deep_model_selected")):
        blocking["manifest_deep_model"] += 1
    if int(dict(manifest["audit"])["prediction_rows"]) != prediction_rows:
        blocking["manifest_prediction_rows"] += 1
    if int(dict(manifest["audit"])["minimum_sessions_after_resolution"]) != 10:
        blocking["manifest_maturity"] += 1
    if bool(dict(manifest["audit"])["final_oracle_row_labels_used"]):
        blocking["manifest_final_label_leak"] += 1

    result = {
        "schema": VALIDATION_SCHEMA,
        "status": "passed" if not blocking else "failed",
        "study_id": baseline.STUDY_ID,
        "blocking": dict(blocking),
        "coordinate_rows": coordinate_rows,
        "prediction_rows": prediction_rows,
        "folds": len(expected_folds),
        "independently_recomputed_prediction_rows": traced,
        "minimum_sessions_after_resolution": 10,
        "final_oracle_row_labels_used": False,
    }
    _write_json(root / "validation.json", result)
    if blocking:
        raise ValueError(
            f"dynamic_action_baseline_validation_failed:{dict(blocking)}"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate causal transparent action-value baseline outputs."
    )
    parser.add_argument("--study", default=str(baseline.DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--trace-rows-per-partition", type=int, default=None)
    args = parser.parse_args(argv)
    result = validate(
        study_path=args.study,
        output_root=args.output_root,
        trace_rows_per_partition=args.trace_rows_per_partition,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
