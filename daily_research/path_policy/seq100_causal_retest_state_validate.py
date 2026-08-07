"""Independent validation for the continuous causal breakout-retest study."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_causal_path_structure as structure
from daily_research.path_policy import (
    seq100_causal_path_structure_validate as structure_validator,
)
from daily_research.path_policy import seq100_causal_retest_state as study_module
from daily_research.path_policy import seq100_hot_path_atlas as atlas


def _assert_equal(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    name: str,
    tolerance: float,
) -> float:
    left = np.asarray(expected)
    right = np.asarray(actual)
    if len(left) != len(right):
        raise AssertionError(f"causal_retest_validation_length:{name}")
    is_float = name in study_module.STATE_FLOAT_FEATURES or name in structure.FLOAT_FEATURES
    if is_float:
        left_float = left.astype(np.float64)
        right_float = right.astype(np.float64)
        if not np.allclose(
            left_float,
            right_float,
            rtol=tolerance,
            atol=tolerance,
            equal_nan=True,
        ):
            difference = np.abs(left_float - right_float)
            difference[~np.isfinite(difference)] = 0.0
            position = int(np.argmax(difference))
            raise AssertionError(
                f"causal_retest_validation_float:{name}:{position}:"
                f"{left_float[position]}:{right_float[position]}"
            )
        finite = np.isfinite(left_float) & np.isfinite(right_float)
        return float(np.max(np.abs(left_float[finite] - right_float[finite]))) if finite.any() else 0.0
    if not np.array_equal(left, right):
        mismatch = np.flatnonzero(left != right)
        position = int(mismatch[0]) if len(mismatch) else -1
        raise AssertionError(f"causal_retest_validation_discrete:{name}:{position}")
    return 0.0


def _derive(frame: pd.DataFrame, source_study: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    parsed = structure.parse_causal_path(
        adj_high=frame["adj_high"].to_numpy(float),
        adj_low=frame["adj_low"].to_numpy(float),
        adj_close=frame["adj_close"].to_numpy(float),
        amount=frame["amount"].to_numpy(float),
        date_indices=frame["date_idx"].to_numpy(np.int64),
        study=source_study,
    )
    state = study_module.derive_causal_retest_state(
        adj_high=frame["adj_high"].to_numpy(float),
        adj_low=frame["adj_low"].to_numpy(float),
        adj_close=frame["adj_close"].to_numpy(float),
        amount=frame["amount"].to_numpy(float),
        date_indices=frame["date_idx"].to_numpy(np.int64),
        path_features=parsed,
    )
    return parsed, state


def verify_prefix_invariance(
    frame: pd.DataFrame,
    source_study: Mapping[str, Any],
    *,
    cutoffs: Sequence[int],
    tolerance: float,
) -> dict[str, Any]:
    _, full = _derive(frame, source_study)
    comparisons = 0
    maximum_difference = 0.0
    for cutoff in cutoffs:
        end = int(cutoff)
        if not 1 <= end <= len(frame):
            raise ValueError("causal_retest_validation_cutoff")
        _, prefix = _derive(frame.iloc[:end], source_study)
        for name in sorted(
            study_module.STATE_BOOLEAN_FEATURES
            | study_module.STATE_INTEGER_FEATURES
            | study_module.STATE_FLOAT_FEATURES
        ):
            maximum_difference = max(
                maximum_difference,
                _assert_equal(
                    full[name][:end],
                    prefix[name],
                    name=name,
                    tolerance=tolerance,
                ),
            )
            comparisons += end
    return {
        "cutoffs": len(cutoffs),
        "feature_cells": comparisons,
        "maximum_float_difference": maximum_difference,
    }


def _persisted_trace(
    connection: duckdb.DuckDBPyConnection,
    *,
    panel_paths: Sequence[Path],
    symbol_idx: int,
    frame: pd.DataFrame,
    parsed: Mapping[str, np.ndarray],
    state: Mapping[str, np.ndarray],
    sample_rows: int,
    tolerance: float,
) -> tuple[int, int, float]:
    scan = ", ".join(atlas._sql_quote(path) for path in panel_paths)
    columns = ", ".join(study_module.PANEL_FEATURE_COLUMNS)
    persisted = connection.execute(
        f"""
        SELECT date_idx, {columns}
        FROM read_parquet([{scan}])
        WHERE symbol_idx = ?
        ORDER BY date_idx
        """,
        [int(symbol_idx)],
    ).fetchdf()
    if persisted.empty:
        raise ValueError(f"causal_retest_validation_empty_panel_symbol:{symbol_idx}")
    if len(persisted) > int(sample_rows):
        selected = np.unique(
            np.linspace(0, len(persisted) - 1, int(sample_rows), dtype=np.int64)
        )
        persisted = persisted.iloc[selected].reset_index(drop=True)
    position_by_date = {
        int(value): position
        for position, value in enumerate(frame["date_idx"].to_numpy(np.int64))
    }
    positions = np.asarray(
        [position_by_date[int(value)] for value in persisted["date_idx"]],
        dtype=np.int64,
    )
    maximum_difference = 0.0
    for name in study_module.PANEL_FEATURE_COLUMNS:
        expected = state[name] if name in state else parsed[name]
        maximum_difference = max(
            maximum_difference,
            _assert_equal(
                np.asarray(expected)[positions],
                persisted[name].to_numpy(),
                name=name,
                tolerance=tolerance,
            ),
        )
    return (
        len(persisted),
        len(persisted) * len(study_module.PANEL_FEATURE_COLUMNS),
        maximum_difference,
    )


def _reference_hac(values: Sequence[float], lag: int) -> dict[str, float]:
    source = np.asarray(values, dtype=np.float64)
    source = source[np.isfinite(source)]
    count = len(source)
    if count == 0:
        return {"mean": math.nan, "se": math.nan, "lcb_95": math.nan, "ucb_95": math.nan}
    mean = float(source.mean())
    if count == 1:
        return {"mean": mean, "se": math.nan, "lcb_95": math.nan, "ucb_95": math.nan}
    centered = source - mean
    maximum_lag = min(int(lag), count - 1)
    long_run_variance = float(np.sum(np.square(centered)) / count)
    for offset in range(1, maximum_lag + 1):
        covariance = float(np.sum(centered[offset:] * centered[:-offset]) / count)
        weight = 1.0 - offset / (maximum_lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / count)
    return {
        "mean": mean,
        "se": standard_error,
        "lcb_95": mean - 1.96 * standard_error,
        "ucb_95": mean + 1.96 * standard_error,
    }


def _assert_scalar(left: float, right: float, *, name: str, tolerance: float = 1e-11) -> None:
    if math.isnan(left) and math.isnan(right):
        return
    if not math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"causal_retest_validation_scalar:{name}:{left}:{right}")


def _validate_outputs(
    manifest: Mapping[str, Any],
    *,
    study: Mapping[str, Any],
) -> dict[str, Any]:
    for name, record in dict(manifest["outputs"]).items():
        path = Path(str(record["path"]))
        if not path.is_file() or study_module._sha256(path) != str(record["sha256"]):
            raise ValueError(f"causal_retest_validation_output_hash:{name}")
    for record in manifest["predictions"]:
        path = Path(str(record["path"]))
        if not path.is_file() or study_module._sha256(path) != str(record["sha256"]):
            raise ValueError("causal_retest_validation_prediction_hash")
        if int(record["training_cutoff_year"]) != int(record["oos_year"]) - 1:
            raise AssertionError("causal_retest_validation_prediction_cutoff")
        if int(record["maximum_training_label_as_of_year"]) > int(record["training_cutoff_year"]):
            raise AssertionError("causal_retest_validation_label_cutoff")
        if int(record["maximum_training_signal_year"]) >= int(record["oos_year"]):
            raise AssertionError("causal_retest_validation_signal_cutoff")

    daily_record = dict(manifest["outputs"])["daily_model_evaluation"]
    aggregate_record = dict(manifest["outputs"])["aggregate_model_evaluation"]
    daily = pd.read_parquet(daily_record["path"])
    aggregate = pd.read_parquet(aggregate_record["path"])
    lag = int(dict(study["evaluation"])["hac_lag"])
    model_hac_rows = 0
    for _, row in aggregate.iterrows():
        group = daily[
            daily["cost_scenario"].eq(row["cost_scenario"])
            & daily["scope"].eq(row["scope"])
            & daily["model"].eq(row["model"])
        ].sort_values("date_idx", kind="mergesort")
        policy = _reference_hac(group["policy_realized_action_value"], lag)
        top1 = _reference_hac(group["top1_realized_action_value"], lag)
        _assert_scalar(policy["mean"], float(row["policy_action_value_mean"]), name="policy_mean")
        _assert_scalar(policy["se"], float(row["policy_action_value_hac_se"]), name="policy_se")
        _assert_scalar(top1["mean"], float(row["top1_action_value_mean"]), name="top1_mean")
        _assert_scalar(top1["se"], float(row["top1_action_value_hac_se"]), name="top1_se")
        model_hac_rows += 1

    coordinate_daily_record = dict(manifest["outputs"])["daily_coordinate_diagnostics"]
    coordinate_aggregate_record = dict(manifest["outputs"])["aggregate_coordinate_diagnostics"]
    coordinate_daily = pd.read_parquet(coordinate_daily_record["path"])
    coordinate_aggregate = pd.read_parquet(coordinate_aggregate_record["path"])
    coordinate_hac_rows = 0
    for _, row in coordinate_aggregate.iterrows():
        group = coordinate_daily[
            coordinate_daily["cost_scenario"].eq(row["cost_scenario"])
            & coordinate_daily["scope"].eq(row["scope"])
            & coordinate_daily["coordinate"].eq(row["coordinate"])
        ].sort_values("date_idx", kind="mergesort")
        winner = _reference_hac(group["winner_minus_failure_coordinate"], lag)
        contrast = _reference_hac(group["high_minus_low_action_value"], lag)
        _assert_scalar(
            winner["mean"],
            float(row["winner_minus_failure_coordinate"]),
            name="winner_mean",
        )
        _assert_scalar(
            winner["se"],
            float(row["winner_minus_failure_hac_se"]),
            name="winner_se",
        )
        _assert_scalar(
            contrast["mean"],
            float(row["high_minus_low_action_value"]),
            name="contrast_mean",
        )
        _assert_scalar(
            contrast["se"],
            float(row["high_minus_low_action_value_hac_se"]),
            name="contrast_se",
        )
        coordinate_hac_rows += 1
    return {
        "model_hac_rows_recomputed": model_hac_rows,
        "coordinate_hac_rows_recomputed": coordinate_hac_rows,
    }


def validate_study(
    *,
    study_path: str | Path = study_module.DEFAULT_STUDY_PATH,
    output_root: str | Path = study_module.DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study_file = study_module._resolve(study_path)
    root = study_module._resolve(output_root)
    study = study_module.load_study(study_file)
    manifest = study_module._read_json(root / "manifest.json")
    panel_manifest = study_module._read_json(root / "panel_manifest.json")
    if manifest.get("status") != "completed" or panel_manifest.get("status") != "prepared":
        raise ValueError("causal_retest_validation_study_incomplete")
    contract, _, _, source_meta = study_module._source_contract(study_file, study)
    if str(manifest["experiment_fingerprint"]) != str(source_meta["fingerprint"]):
        raise ValueError("causal_retest_validation_fingerprint")
    if study_module._sha256(root / "panel_manifest.json") != str(manifest["panel_manifest"]["sha256"]):
        raise ValueError("causal_retest_validation_panel_manifest_hash")
    panel_paths = [Path(str(record["path"])) for record in panel_manifest["state_panels"]]
    source_panel_paths = [Path(str(record["path"])) for record in contract["path_structure_panels"]]
    coordinate_paths = [Path(str(record["path"])) for record in contract["coordinate_partitions"]]
    prediction_paths = [Path(str(record["path"])) for record in manifest["predictions"]]
    runtime = study_module._runtime_resources(study)
    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={int(runtime['threads'])}")
    connection.execute("PRAGMA preserve_insertion_order=false")
    connection.execute(f"PRAGMA memory_limit='{max(int(runtime['usable_memory_mb']), 1024)}MB'")
    panel_scan = ", ".join(atlas._sql_quote(path) for path in panel_paths)
    source_scan = ", ".join(atlas._sql_quote(path) for path in source_panel_paths)
    coordinate_scan = ", ".join(atlas._sql_quote(path) for path in coordinate_paths)
    prediction_scan = ", ".join(atlas._sql_quote(path) for path in prediction_paths)
    try:
        panel_audit = connection.execute(
            f"""
            SELECT
                count(*) AS rows,
                count(DISTINCT input_row_idx) AS distinct_rows,
                sum((substr(trade_date, 1, 4) = '2026')::INTEGER) AS forbidden_rows,
                sum(up_breakout_event::INTEGER) AS breakout_events,
                sum(up_retest_event::INTEGER) AS retest_events,
                sum(up_reentry_event::INTEGER) AS reentry_events,
                sum(up_breakout_active::INTEGER) AS active_rows,
                sum((up_breakout_event AND sessions_since_up_breakout <> 0)::INTEGER) AS breakout_age_errors,
                sum((up_retest_event AND (NOT up_breakout_active OR NOT up_retest_history))::INTEGER) AS retest_state_errors,
                sum((up_repeated_retest AND up_retest_episode_count < 2)::INTEGER) AS repeat_errors,
                sum((up_reentry_event AND isfinite(boundary_close_distance_units) = false)::INTEGER) AS reentry_geometry_errors
            FROM read_parquet([{panel_scan}])
            """
        ).fetchone()
        if int(panel_audit[0]) != int(panel_manifest["rows"]) or int(panel_audit[1]) != int(panel_audit[0]):
            raise AssertionError("causal_retest_validation_panel_rows")
        if any(int(value) != 0 for value in panel_audit[2:3]) or any(
            int(value) != 0 for value in panel_audit[7:]
        ):
            raise AssertionError(f"causal_retest_validation_panel_invariant:{panel_audit}")
        source_alignment = connection.execute(
            f"""
            SELECT
                count(*) AS rows,
                sum((r.up_breakout_event <> (s.chan_breakout_event = 1))::INTEGER) AS breakout_mismatch,
                sum((r.up_retest_event <> (s.chan_retest_event = 1))::INTEGER) AS retest_mismatch,
                sum((r.up_reentry_event <> (s.chan_reentry_event = 1))::INTEGER) AS reentry_mismatch,
                sum((r.up_breakout_active <> (s.chan_breakout_active = 1))::INTEGER) AS active_mismatch,
                sum((r.dc_small_mode <> s.dc_small_mode)::INTEGER) AS small_mode_mismatch,
                sum((r.dc_medium_mode <> s.dc_medium_mode)::INTEGER) AS medium_mode_mismatch,
                sum((r.dc_large_mode <> s.dc_large_mode)::INTEGER) AS large_mode_mismatch
            FROM read_parquet([{panel_scan}]) r
            INNER JOIN read_parquet([{source_scan}]) s USING (input_row_idx)
            """
        ).fetchone()
        if int(source_alignment[0]) != int(panel_manifest["rows"]) or any(
            int(value) != 0 for value in source_alignment[1:]
        ):
            raise AssertionError(f"causal_retest_validation_source_alignment:{source_alignment}")

        prediction_audit = connection.execute(
            f"""
            SELECT
                count(*) AS rows,
                sum((substr(p.trade_date, 1, 4) = '2026')::INTEGER) AS forbidden_rows,
                sum((p.activity_rank < 0.8 OR NOT isfinite(p.activity_rank))::INTEGER) AS gate_errors,
                sum((abs(p.actual_buy_advantage_vs_cash - (p.actual_upside_component - p.actual_downside_component)) > 1e-10)::INTEGER) AS target_errors,
                sum((p.model_maximum_label_as_of_year > p.model_training_cutoff_year)::INTEGER) AS label_cutoff_errors,
                sum((p.model_maximum_signal_year >= p.oos_year)::INTEGER) AS signal_cutoff_errors,
                sum((p.scope = 'up_retest_event' AND NOT r.up_retest_event)::INTEGER) AS retest_scope_errors,
                sum((p.scope = 'up_price_amount_confirmed_breakout' AND (NOT r.up_breakout_event OR c.price_amount_correlation_5d_rank < 0.5 OR c.minute_last_30m_return_rank >= 0.5))::INTEGER) AS breakout_scope_errors,
                sum((abs(p.predicted_causal_rank_spline_expected_value - (p.predicted_causal_rank_spline_upside_component - p.predicted_causal_rank_spline_downside_component)) > 1e-6)::INTEGER) AS spline_component_errors,
                sum((abs(p.predicted_historical_neighbor_expected_value - (p.predicted_historical_neighbor_upside_component - p.predicted_historical_neighbor_downside_component)) > 1e-6)::INTEGER) AS neighbor_component_errors
            FROM read_parquet([{prediction_scan}]) p
            INNER JOIN read_parquet([{panel_scan}]) r USING (input_row_idx)
            INNER JOIN read_parquet([{coordinate_scan}]) c USING (input_row_idx)
            """
        ).fetchone()
        if int(prediction_audit[0]) != int(manifest["audit"]["prediction_rows"]) or any(
            int(value) != 0 for value in prediction_audit[1:]
        ):
            raise AssertionError(f"causal_retest_validation_predictions:{prediction_audit}")

        validation = dict(study["validation"])
        samples = structure_validator._sample_symbols(
            connection,
            Path(str(contract["row_index"]["path"])),
            int(validation["prefix_symbols"]),
        )
        source_study = structure.load_study(Path(str(contract["path_structure_study"]["path"])))
        symbol_rows: list[dict[str, Any]] = []
        prefix_cells = 0
        persisted_cells = 0
        persisted_rows = 0
        maximum_difference = 0.0
        for _, sample in samples.iterrows():
            frame = structure_validator._path_frame(
                connection,
                Path(str(contract["dense_base"]["path"])),
                str(sample["symbol"]),
            )
            cutoffs = np.unique(
                np.linspace(
                    1,
                    len(frame),
                    int(validation["prefix_cutoffs_per_symbol"]),
                    dtype=np.int64,
                )
            )
            prefix = verify_prefix_invariance(
                frame,
                source_study,
                cutoffs=cutoffs,
                tolerance=float(validation["floating_tolerance"]),
            )
            parsed, state = _derive(frame, source_study)
            trace = _persisted_trace(
                connection,
                panel_paths=panel_paths,
                symbol_idx=int(sample["symbol_idx"]),
                frame=frame,
                parsed=parsed,
                state=state,
                sample_rows=int(validation["persisted_rows_per_symbol"]),
                tolerance=float(validation["floating_tolerance"]),
            )
            prefix_cells += int(prefix["feature_cells"])
            persisted_rows += int(trace[0])
            persisted_cells += int(trace[1])
            maximum_difference = max(
                maximum_difference,
                float(prefix["maximum_float_difference"]),
                float(trace[2]),
            )
            symbol_rows.append(
                {
                    "symbol": str(sample["symbol"]),
                    "symbol_idx": int(sample["symbol_idx"]),
                    "path_rows": len(frame),
                    "pool_rows": int(sample["pool_rows"]),
                    "cutoffs": int(prefix["cutoffs"]),
                    "persisted_rows_traced": int(trace[0]),
                }
            )
    finally:
        connection.close()

    output_audit = _validate_outputs(manifest, study=study)
    result = {
        "schema": study_module.VALIDATION_SCHEMA_VERSION,
        "status": "passed",
        "study_id": study_module.STUDY_ID,
        "blocking": {},
        "panel_rows": int(panel_audit[0]),
        "duplicate_panel_rows": int(panel_audit[0] - panel_audit[1]),
        "forbidden_2026_rows": int(panel_audit[2]),
        "up_breakout_events": int(panel_audit[3]),
        "up_retest_events": int(panel_audit[4]),
        "up_reentry_events": int(panel_audit[5]),
        "up_breakout_active_rows": int(panel_audit[6]),
        "source_structure_rows_aligned": int(source_alignment[0]),
        "prediction_rows": int(prediction_audit[0]),
        "prediction_partitions": len(manifest["predictions"]),
        "sampled_symbols": len(symbol_rows),
        "prefix_cutoffs": int(sum(item["cutoffs"] for item in symbol_rows)),
        "prefix_feature_cells_compared": prefix_cells,
        "persisted_panel_rows_recomputed": persisted_rows,
        "persisted_feature_cells_compared": persisted_cells,
        "maximum_float_difference": maximum_difference,
        "future_confirmed_structure_written_back": False,
        "training_cutoff_violations": 0,
        "fixed_holding_horizon_used": False,
        "account_replay_allowed": bool(manifest["promotion_gate"]["account_replay_allowed"]),
        **output_audit,
        "symbols": symbol_rows,
    }
    study_module._write_json(root / "validation.json", result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the continuous causal breakout-retest study."
    )
    parser.add_argument("--study", default=str(study_module.DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(study_module.DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = validate_study(study_path=args.study, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
