"""Independent validation for dynamic action distribution outputs."""

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

from daily_research.path_policy import seq100_dynamic_action_distribution as study
from daily_research.path_policy import seq100_dynamic_action_value_baselines as baseline
from daily_research.path_policy import seq100_market_replay as replay

VALIDATION_SCHEMA = "seq100_dynamic_action_distribution_validation/1"


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
    study_path: str | Path = study.DEFAULT_STUDY_PATH,
    output_root: str | Path | None = None,
    trace_rows_per_partition: int | None = None,
) -> dict[str, Any]:
    study_file = study._resolve(study_path)
    contract = study.load_study(study_file)
    root = study._resolve(output_root or contract["output_root"])
    manifest_path = root / "manifest.json"
    manifest = study._read_json(manifest_path)
    blocking: defaultdict[str, int] = defaultdict(int)
    if manifest.get("status") != "completed":
        blocking["manifest_status"] += 1
    if manifest.get("study_id") != study.STUDY_ID:
        blocking["manifest_study_id"] += 1
    if replay.sha256(study_file) != str(dict(manifest["study"])["sha256"]):
        blocking["study_hash"] += 1

    (
        _source_contract,
        prefix_manifest,
        input_manifest,
        version_records,
        _feature_index,
    ) = study._source_contract(contract)
    source_audit = dict(manifest["source_audit"])
    if int(source_audit["model_input_rows"]) != int(input_manifest["row_count"]):
        blocking["source_input_rows"] += 1
    if int(source_audit["prefix_experience_version_rows"]) != int(
        dict(prefix_manifest["audit"])["experience_version_rows"]
    ):
        blocking["source_version_rows"] += 1

    _specs, coordinate_names, _market_names, _families, _matched_indices = (
        baseline._coordinate_metadata(contract)
    )
    coordinate_records = {
        int(record["year"]): dict(record) for record in manifest["coordinates"]
    }
    expected_years = set(range(2012, 2026))
    if set(coordinate_records) != expected_years:
        blocking["coordinate_year_coverage"] += len(
            expected_years.symmetric_difference(coordinate_records)
        )
    coordinate_store = baseline.CoordinateStore(
        records=list(coordinate_records.values()),
        coordinate_names=coordinate_names,
        cache_years=2,
    )
    coordinate_rows = 0
    for year in sorted(coordinate_records):
        coordinate = coordinate_store.load(year)
        coordinate_rows += len(coordinate.frame)
        if coordinate.frame["trade_date"].astype(str).str[:4].astype(int).ne(year).any():
            blocking["coordinate_year"] += 1
        if coordinate.frame["trade_date"].astype(str).str[:4].astype(int).ge(2026).any():
            blocking["coordinate_forbidden_year"] += 1
        if np.isinf(coordinate.values).any():
            blocking["coordinate_infinity"] += 1
        finite = np.isfinite(coordinate.values)
        if bool(
            (
                (coordinate.values[finite] < -1.0e-6)
                | (coordinate.values[finite] > 1.000001)
            ).any()
        ):
            blocking["coordinate_range"] += 1
    if coordinate_rows != int(input_manifest["row_count"]):
        blocking["coordinate_total_rows"] += 1

    name_to_index = {name: index for index, name in enumerate(coordinate_names)}
    gate = dict(contract["activity_gate"])
    gate_indices = np.asarray(
        [name_to_index[str(name)] for name in gate["coordinates"]], dtype=np.int16
    )
    activity_store = study.ActivityRankStore(
        coordinates=coordinate_store,
        coordinate_indices=gate_indices,
        minimum_finite=int(gate["minimum_finite_coordinates"]),
        cache_years=2,
    )
    risk_indices = np.asarray(
        [
            name_to_index[str(name)]
            for name in dict(contract["coordinates"])["risk_match_coordinates"]
        ],
        dtype=np.int16,
    )
    risk_bins = int(dict(contract["coordinates"])["risk_match_bins"])
    scopes = [str(value) for value in dict(contract["estimators"])["training_scopes"]]

    state_records = {
        (
            str(record["cost_scenario"]),
            int(record["oos_year"]),
            str(record["scope"]),
        ): dict(record)
        for record in manifest["model_states"]
    }
    prediction_records = {
        (str(record["cost_scenario"]), int(record["oos_year"])): dict(record)
        for record in manifest["predictions"]
    }
    folds = {
        (str(record["cost_scenario"]), int(record["oos_year"])): dict(record)
        for record in manifest["folds"]
    }
    expected_folds = {
        (cost, year)
        for cost in ("base", "double_slippage")
        for year in range(2013, 2026)
    }
    expected_states = {
        (cost, year, scope)
        for cost, year in expected_folds
        for scope in scopes
    }
    if set(state_records) != expected_states:
        blocking["state_fold_coverage"] += len(
            expected_states.symmetric_difference(state_records)
        )
    if set(prediction_records) != expected_folds:
        blocking["prediction_fold_coverage"] += len(
            expected_folds.symmetric_difference(prediction_records)
        )
    if set(folds) != expected_folds:
        blocking["manifest_fold_coverage"] += len(
            expected_folds.symmetric_difference(folds)
        )

    trace_rows = int(
        trace_rows_per_partition
        if trace_rows_per_partition is not None
        else dict(contract["resources"])["validator_trace_rows_per_partition"]
    )
    prediction_rows = 0
    traced = 0
    prediction_suffixes = (
        "positive_probability",
        "upside_component",
        "downside_component",
        "direct_expected_value",
        "expected_value",
        "reconstruction_gap",
    )
    for cost_scenario, oos_year in sorted(expected_folds):
        fold = folds.get((cost_scenario, oos_year))
        prediction_record = prediction_records.get((cost_scenario, oos_year))
        if fold is None or prediction_record is None:
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

        snapshots: dict[str, dict[str, np.ndarray]] = {}
        for scope in scopes:
            record = state_records.get((cost_scenario, oos_year, scope))
            if record is None:
                continue
            snapshot = _load_snapshot(record, blocking=blocking)
            snapshots[scope] = snapshot
            if not snapshot:
                continue
            if str(snapshot["scope"][0]) != scope:
                blocking["state_scope"] += 1
            if int(snapshot["training_cutoff_year"][0]) != cutoff:
                blocking["state_cutoff"] += 1
            if int(snapshot["maximum_label_as_of_year"][0]) > cutoff:
                blocking["state_as_of_leak"] += 1
            if int(snapshot["maximum_signal_year"][0]) >= oos_year:
                blocking["state_signal_leak"] += 1
            if [str(value) for value in snapshot["coordinate_names"]] != coordinate_names:
                blocking["state_coordinate_names"] += 1
            if tuple(str(value) for value in snapshot["moment_names"]) != study.MOMENT_NAMES:
                blocking["state_moment_names"] += 1

        prediction = _read_frame(
            prediction_record, blocking=blocking, name="prediction"
        )
        prediction_rows += len(prediction)
        required = {
            "trade_date",
            "date_idx",
            "symbol_idx",
            "input_row_idx",
            "activity_rank",
            "risk_match_cell_code",
            "actual_buy_advantage_vs_cash",
            "actual_positive",
            "evaluation_sessions_after_resolution",
        }
        for scope in scopes:
            required.update(
                {
                    f"predicted_{scope}_{model}_{suffix}"
                    for model in study.MODEL_NAMES
                    for suffix in prediction_suffixes
                }
            )
            required.update(
                {
                    f"{scope}_matched_cell_code",
                    f"{scope}_matched_cell_weight",
                    f"{scope}_matched_cell_date_count",
                }
            )
        if required - set(prediction.columns):
            blocking["prediction_columns"] += 1
            continue
        if prediction["trade_date"].astype(str).str[:4].astype(int).ne(oos_year).any():
            blocking["prediction_year"] += 1
        if prediction["trade_date"].astype(str).str[:4].astype(int).ge(2026).any():
            blocking["prediction_forbidden_year"] += 1
        if prediction["training_cutoff_year"].astype(int).ne(cutoff).any():
            blocking["prediction_cutoff"] += 1
        maturity = int(dict(contract["target"])["minimum_sessions_after_resolution"])
        if prediction["evaluation_sessions_after_resolution"].astype(int).lt(
            maturity
        ).any():
            blocking["prediction_immature_evaluation"] += 1
        numeric = prediction.select_dtypes(include=["number"]).to_numpy(np.float64)
        if np.isinf(numeric).any():
            blocking["prediction_infinity"] += 1

        coordinate = coordinate_store.load(oos_year)
        first_input = int(coordinate.frame["input_row_idx"].iloc[0])
        sample = prediction.sample(
            n=min(trace_rows, len(prediction)),
            random_state=20260807 + 1009 * oos_year + (0 if cost_scenario == "base" else 1),
        ).sort_values("input_row_idx")
        local = sample["input_row_idx"].to_numpy(np.int64) - first_input
        if bool((local < 0).any()) or bool((local >= len(coordinate.frame)).any()):
            blocking["trace_coordinate_position"] += 1
            continue
        trace_values = coordinate.values[local].astype(np.float64)
        full_activity = activity_store.load(oos_year)
        if not np.allclose(
            sample["activity_rank"].to_numpy(np.float64),
            full_activity[local],
            atol=2.0e-6,
            rtol=0.0,
            equal_nan=True,
        ):
            blocking["trace_activity_rank"] += 1
        if bool((sample["activity_rank"].to_numpy(np.float64) < 0.8).any()):
            blocking["trace_hot_gate"] += 1
        recomputed_risk = baseline._matched_codes(
            trace_values, matched_indices=risk_indices, bin_count=risk_bins
        )
        if not np.array_equal(
            sample["risk_match_cell_code"].to_numpy(np.int32), recomputed_risk
        ):
            blocking["trace_risk_cell"] += 1

        for scope in scopes:
            snapshot = snapshots.get(scope)
            if not snapshot:
                continue
            recomputed = study.predict_from_snapshot(trace_values, snapshot)
            for model in study.MODEL_NAMES:
                for suffix in prediction_suffixes:
                    stored_name = f"predicted_{scope}_{model}_{suffix}"
                    recomputed_name = f"predicted_{model}_{suffix}"
                    if not np.allclose(
                        sample[stored_name].to_numpy(np.float64),
                        recomputed[recomputed_name],
                        atol=2.0e-6,
                        rtol=0.0,
                    ):
                        blocking[f"trace_{scope}_{model}_{suffix}"] += 1
            for suffix in (
                "matched_cell_code",
                "matched_cell_weight",
                "matched_cell_date_count",
            ):
                stored = sample[f"{scope}_{suffix}"].to_numpy()
                recomputed_values = recomputed[suffix]
                if suffix == "matched_cell_code":
                    equal = np.array_equal(stored.astype(np.int32), recomputed_values)
                else:
                    equal = np.allclose(
                        stored.astype(np.float64),
                        recomputed_values,
                        atol=2.0e-6,
                        rtol=0.0,
                    )
                if not equal:
                    blocking[f"trace_{scope}_{suffix}"] += 1

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
        "daily_gate_diagnostics",
        "annual_gate_diagnostics",
        "aggregate_gate_diagnostics",
        "daily_model_metrics",
        "annual_model_metrics",
        "aggregate_model_metrics",
        "selection_decomposition",
        "daily_matched_failure_diagnostics",
        "annual_matched_failure_diagnostics",
        "aggregate_matched_failure_diagnostics",
        "matched_failure_availability",
    ):
        _read_frame(outputs[name], blocking=blocking, name=name)
    gate_record = dict(outputs["promotion_gate"])
    gate_path = Path(str(gate_record["path"]))
    if not gate_path.is_file() or replay.sha256(gate_path) != str(
        gate_record["sha256"]
    ):
        blocking["promotion_gate_hash"] += 1
        promotion_gate = {}
    else:
        promotion_gate = study._read_json(gate_path)
    if bool(promotion_gate.get("profit_claim_allowed", True)):
        blocking["promotion_profit_claim"] += 1
    if bool(manifest.get("portfolio_execution_performed")):
        blocking["manifest_portfolio_execution"] += 1
    if bool(manifest.get("profit_claim_allowed")):
        blocking["manifest_profit_claim"] += 1
    if bool(manifest.get("production_policy_selected")):
        blocking["manifest_production_policy"] += 1
    if bool(manifest.get("deep_model_selected")):
        blocking["manifest_deep_model"] += 1
    audit = dict(manifest["audit"])
    if int(audit["hot_prediction_rows"]) != prediction_rows:
        blocking["manifest_prediction_rows"] += 1
    if int(audit["minimum_sessions_after_resolution"]) != 10:
        blocking["manifest_maturity"] += 1
    if bool(audit["final_oracle_row_labels_used"]):
        blocking["manifest_final_label_leak"] += 1
    if bool(audit["binary_good_stock_label_used"]):
        blocking["manifest_binary_label"] += 1
    if bool(audit["activity_gate_is_buy_rule"]):
        blocking["manifest_gate_rule"] += 1
    if bool(audit["pristine_confirmation_claimed"]):
        blocking["manifest_pristine_claim"] += 1

    result = {
        "schema": VALIDATION_SCHEMA,
        "status": "passed" if not blocking else "failed",
        "study_id": study.STUDY_ID,
        "blocking": dict(blocking),
        "coordinate_rows": coordinate_rows,
        "prediction_rows": prediction_rows,
        "folds": len(expected_folds),
        "states": len(expected_states),
        "independently_recomputed_prediction_rows": traced,
        "minimum_sessions_after_resolution": 10,
        "full_pool_activity_rank_recomputed": True,
        "final_oracle_row_labels_used": False,
    }
    _write_json(root / "validation.json", result)
    if blocking:
        raise ValueError(
            f"dynamic_action_distribution_validation_failed:{dict(blocking)}"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate dynamic action distribution study outputs."
    )
    parser.add_argument("--study", default=str(study.DEFAULT_STUDY_PATH))
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
