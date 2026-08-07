"""Independent prefix and persisted-panel checks for causal path structures."""

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

from daily_research.path_policy import seq100_causal_path_structure as study_module
from daily_research.path_policy import seq100_hot_path_atlas as atlas


def _assert_feature_equal(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    name: str,
    tolerance: float,
) -> float:
    if len(expected) != len(actual):
        raise AssertionError(f"causal_path_validation_length:{name}")
    if name in study_module.FLOAT_FEATURES:
        left = np.asarray(expected, dtype=np.float64)
        right = np.asarray(actual, dtype=np.float64)
        if not np.allclose(
            left,
            right,
            rtol=tolerance,
            atol=tolerance,
            equal_nan=True,
        ):
            difference = np.abs(left - right)
            difference[~np.isfinite(difference)] = 0.0
            position = int(np.argmax(difference))
            raise AssertionError(
                f"causal_path_validation_float:{name}:{position}:"
                f"{left[position]}:{right[position]}"
            )
        finite = np.isfinite(left) & np.isfinite(right)
        return (
            float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
        )
    if not np.array_equal(np.asarray(expected), np.asarray(actual)):
        mismatch = np.flatnonzero(np.asarray(expected) != np.asarray(actual))
        position = int(mismatch[0]) if len(mismatch) else -1
        raise AssertionError(f"causal_path_validation_discrete:{name}:{position}")
    return 0.0


def verify_prefix_invariance(
    frame: pd.DataFrame,
    study: Mapping[str, Any],
    *,
    cutoffs: Sequence[int],
    tolerance: float,
) -> dict[str, Any]:
    full = study_module.parse_causal_path(
        adj_high=frame["adj_high"].to_numpy(float),
        adj_low=frame["adj_low"].to_numpy(float),
        adj_close=frame["adj_close"].to_numpy(float),
        amount=frame["amount"].to_numpy(float),
        date_indices=frame["date_idx"].to_numpy(np.int64),
        study=study,
    )
    comparisons = 0
    maximum_difference = 0.0
    for cutoff in cutoffs:
        end = int(cutoff)
        if not 1 <= end <= len(frame):
            raise ValueError("causal_path_validation_cutoff_out_of_range")
        prefix = frame.iloc[:end]
        parsed = study_module.parse_causal_path(
            adj_high=prefix["adj_high"].to_numpy(float),
            adj_low=prefix["adj_low"].to_numpy(float),
            adj_close=prefix["adj_close"].to_numpy(float),
            amount=prefix["amount"].to_numpy(float),
            date_indices=prefix["date_idx"].to_numpy(np.int64),
            study=study,
        )
        for name in study_module.FEATURE_COLUMNS:
            maximum_difference = max(
                maximum_difference,
                _assert_feature_equal(
                    full[name][:end],
                    parsed[name],
                    name=name,
                    tolerance=tolerance,
                ),
            )
            comparisons += end
    return {
        "cutoffs": len(cutoffs),
        "feature_prefix_cells": int(comparisons),
        "maximum_float_difference": float(maximum_difference),
    }


def _sample_symbols(
    connection: duckdb.DuckDBPyConnection, row_index_path: Path, count: int
) -> pd.DataFrame:
    symbols = connection.execute(
        f"""
        SELECT symbol_idx, any_value(symbol) AS symbol, count(*) AS pool_rows
        FROM read_parquet({atlas._sql_quote(row_index_path)})
        GROUP BY symbol_idx
        ORDER BY symbol_idx
        """
    ).fetchdf()
    if len(symbols) <= count:
        return symbols
    positions = np.unique(
        np.linspace(0, len(symbols) - 1, num=int(count), dtype=np.int64)
    )
    return symbols.iloc[positions].reset_index(drop=True)


def _path_frame(
    connection: duckdb.DuckDBPyConnection, dense_path: Path, symbol: str
) -> pd.DataFrame:
    frame = connection.execute(
        f"""
        SELECT trade_date, date_idx, adj_high, adj_low, adj_close, amount
        FROM read_parquet({atlas._sql_quote(dense_path)})
        WHERE symbol = ? AND bar_valid
          AND adj_high > 0 AND adj_low > 0 AND adj_close > 0
          AND isfinite(adj_high) AND isfinite(adj_low) AND isfinite(adj_close)
        ORDER BY date_idx
        """,
        [symbol],
    ).fetchdf()
    if frame.empty:
        raise ValueError(f"causal_path_validation_empty_symbol:{symbol}")
    return frame


def _panel_trace(
    connection: duckdb.DuckDBPyConnection,
    panel_paths: Sequence[Path],
    *,
    symbol_idx: int,
    frame: pd.DataFrame,
    parsed: Mapping[str, np.ndarray],
    tolerance: float,
) -> tuple[int, int, float]:
    scan = ", ".join(atlas._sql_quote(path) for path in panel_paths)
    columns = ", ".join(study_module.FEATURE_COLUMNS)
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
        raise ValueError(f"causal_path_validation_empty_panel_symbol:{symbol_idx}")
    position_by_date = {
        int(date_idx): pos
        for pos, date_idx in enumerate(frame["date_idx"].to_numpy(np.int64))
    }
    positions = np.asarray(
        [position_by_date[int(value)] for value in persisted["date_idx"]],
        dtype=np.int64,
    )
    maximum_difference = 0.0
    for name in study_module.FEATURE_COLUMNS:
        maximum_difference = max(
            maximum_difference,
            _assert_feature_equal(
                np.asarray(parsed[name])[positions],
                persisted[name].to_numpy(),
                name=name,
                tolerance=tolerance,
            ),
        )
    return (
        len(persisted),
        len(persisted) * len(study_module.FEATURE_COLUMNS),
        maximum_difference,
    )


def _reference_hac(values: Sequence[float], lag: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    count = len(array)
    if count == 0:
        return {"mean": math.nan, "se": math.nan}
    mean = float(array.sum() / count)
    if count == 1:
        return {"mean": mean, "se": math.nan}
    centered = array - mean
    long_run = float(np.sum(centered * centered) / count)
    maximum_lag = min(int(lag), count - 1)
    for offset in range(1, maximum_lag + 1):
        covariance = float(np.sum(centered[offset:] * centered[:-offset]) / count)
        weight = 1.0 - offset / (maximum_lag + 1.0)
        long_run += 2.0 * weight * covariance
    return {"mean": mean, "se": math.sqrt(max(long_run, 0.0) / count)}


def _assert_scalar_close(
    actual: float, expected: float, *, name: str, tolerance: float = 1.0e-10
) -> None:
    if math.isnan(actual) and math.isnan(expected):
        return
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(
            f"causal_path_validation_scalar:{name}:{actual}:{expected}"
        )


def _independent_daily_values(
    frame: pd.DataFrame, *, pattern: str, date_idx: int
) -> dict[str, float]:
    day = frame[frame["date_idx"].eq(int(date_idx))].copy()
    candidate = day[f"pattern_{pattern}"].fillna(False).to_numpy(bool)
    selected = day[candidate]
    if selected.empty:
        raise ValueError("causal_path_validation_sample_without_candidate")
    result = {
        "candidate_rows": float(len(selected)),
        "candidate_action_value": float(
            selected["actual_buy_advantage_vs_cash"].mean()
        ),
        "candidate_positive_fraction": float(selected["actual_positive"].mean()),
        "candidate_upside_component": float(selected["actual_upside_component"].mean()),
        "candidate_downside_component": float(
            selected["actual_downside_component"].mean()
        ),
    }
    cell_values: list[dict[str, float]] = []
    matched_positive_rows = 0
    matched_failure_positive_rows = 0
    for _, cell in day.groupby("risk_match_cell_code", sort=True):
        cell_candidate = cell[f"pattern_{pattern}"].fillna(False).to_numpy(bool)
        if not cell_candidate.any() or cell_candidate.all():
            continue
        candidate_cell = cell[cell_candidate]
        control_cell = cell[~cell_candidate]
        cell_values.append(
            {
                "value": float(
                    candidate_cell["actual_buy_advantage_vs_cash"].mean()
                    - control_cell["actual_buy_advantage_vs_cash"].mean()
                ),
                "positive": float(
                    candidate_cell["actual_positive"].mean()
                    - control_cell["actual_positive"].mean()
                ),
                "upside": float(
                    candidate_cell["actual_upside_component"].mean()
                    - control_cell["actual_upside_component"].mean()
                ),
                "downside": float(
                    candidate_cell["actual_downside_component"].mean()
                    - control_cell["actual_downside_component"].mean()
                ),
            }
        )
        positives = int(candidate_cell["actual_positive"].sum())
        negatives = int((~candidate_cell["actual_positive"].astype(bool)).sum())
        matched_positive_rows += positives
        if negatives > 0:
            matched_failure_positive_rows += positives
    if cell_values:
        result.update(
            {
                "matched_cells": float(len(cell_values)),
                "matched_action_value_difference": float(
                    np.mean([item["value"] for item in cell_values])
                ),
                "matched_positive_fraction_difference": float(
                    np.mean([item["positive"] for item in cell_values])
                ),
                "matched_upside_difference": float(
                    np.mean([item["upside"] for item in cell_values])
                ),
                "matched_downside_difference": float(
                    np.mean([item["downside"] for item in cell_values])
                ),
                "matched_candidate_positive_rows": float(matched_positive_rows),
                "matched_failure_positive_rows": float(matched_failure_positive_rows),
            }
        )
    return result


def _validate_diagnostics(
    connection: duckdb.DuckDBPyConnection,
    *,
    study: Mapping[str, Any],
    output_root: Path,
    panel_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    contract = dict(panel_manifest["source_contract"])
    structure_by_year = study_module._records_by_year(
        panel_manifest["structure_panels"]
    )
    coordinate_by_year = study_module._records_by_year(
        contract["coordinate_partitions"]
    )
    predictions = study_module._prediction_records(contract["prediction_partitions"])
    prediction_map = {
        (str(record["cost_scenario"]), int(record["oos_year"])): Path(record["path"])
        for record in predictions
    }
    daily = pd.read_parquet(output_root / "daily_pattern_diagnostics.parquet")
    aggregate = pd.read_parquet(output_root / "aggregate_pattern_diagnostics.parquet")
    cases = (
        (
            "base",
            2020,
            ("nested_up_pullback", "zone_breakout_up", "zone_retest_hold_up"),
        ),
        (
            "double_slippage",
            2024,
            (
                "nested_up_reacceleration",
                "up_exhaustion",
                "up_breakout_late_chase",
            ),
        ),
    )
    sampled_daily_rows = 0
    sampled_quantities = 0
    for cost, year, patterns in cases:
        frame = study_module._load_evaluation_partition(
            connection,
            prediction_path=prediction_map[(cost, year)],
            structure_path=structure_by_year[year],
            coordinate_path=coordinate_by_year[year],
        )
        for pattern in patterns:
            rows = daily[
                daily["cost_scenario"].eq(cost)
                & daily["evaluation_year"].eq(year)
                & daily["pattern"].eq(pattern)
            ].sort_values("date_idx")
            positions = np.unique(
                np.linspace(0, len(rows) - 1, num=min(5, len(rows)), dtype=np.int64)
            )
            for position in positions:
                persisted = rows.iloc[int(position)]
                expected = _independent_daily_values(
                    frame, pattern=pattern, date_idx=int(persisted["date_idx"])
                )
                for name, value in expected.items():
                    _assert_scalar_close(
                        float(persisted[name]),
                        float(value),
                        name=f"{cost}:{year}:{pattern}:{persisted['date_idx']}:{name}",
                    )
                    sampled_quantities += 1
                sampled_daily_rows += 1

    hac_lag = int(dict(study["evaluation"])["hac_lag"])
    aggregate_rows = 0
    decomposition_rows = 0
    for row in aggregate.itertuples(index=False):
        group = daily[
            daily["cost_scenario"].eq(row.cost_scenario)
            & daily["pattern"].eq(row.pattern)
        ].sort_values("date_idx")
        absolute = _reference_hac(group["candidate_action_value"], hac_lag)
        matched = group[group["matched_action_value_difference"].notna()]
        matched_hac = _reference_hac(
            matched["matched_action_value_difference"], hac_lag
        )
        _assert_scalar_close(
            float(row.action_value_mean),
            absolute["mean"],
            name=f"aggregate:{row.cost_scenario}:{row.pattern}:mean",
        )
        _assert_scalar_close(
            float(row.action_value_hac_se),
            absolute["se"],
            name=f"aggregate:{row.cost_scenario}:{row.pattern}:se",
        )
        _assert_scalar_close(
            float(row.matched_action_value_difference),
            matched_hac["mean"],
            name=f"aggregate:{row.cost_scenario}:{row.pattern}:matched_mean",
        )
        _assert_scalar_close(
            float(row.matched_action_value_hac_se),
            matched_hac["se"],
            name=f"aggregate:{row.cost_scenario}:{row.pattern}:matched_se",
        )
        _assert_scalar_close(
            float(row.action_value_mean),
            float(row.upside_component - row.downside_component),
            name=f"decomposition:{row.cost_scenario}:{row.pattern}:absolute",
        )
        _assert_scalar_close(
            float(row.matched_action_value_difference),
            float(row.matched_upside_difference - row.matched_downside_difference),
            name=f"decomposition:{row.cost_scenario}:{row.pattern}:matched",
        )
        aggregate_rows += 1
        decomposition_rows += 1
    return {
        "sampled_daily_rows_recomputed": sampled_daily_rows,
        "sampled_daily_quantities_recomputed": sampled_quantities,
        "aggregate_hac_rows_recomputed": aggregate_rows,
        "component_decomposition_rows_recomputed": decomposition_rows,
    }


def validate_study(
    *,
    study_path: str | Path = study_module.DEFAULT_STUDY_PATH,
    output_root: str | Path = study_module.DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study_path = study_module._resolve(study_path)
    output_root = study_module._resolve(output_root)
    study = study_module.load_study(study_path)
    manifest_path = output_root / "manifest.json"
    manifest = study_module._read_json(manifest_path)
    if (
        manifest.get("schema") != study_module.MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != "completed"
    ):
        raise ValueError("causal_path_validation_study_incomplete")
    panel_manifest = study_module._read_json(output_root / "panel_manifest.json")
    if str(panel_manifest.get("experiment_fingerprint")) != str(
        manifest.get("experiment_fingerprint")
    ):
        raise ValueError("causal_path_validation_fingerprint_mismatch")
    contract = dict(panel_manifest["source_contract"])
    row_index_path = Path(str(dict(contract["row_index"])["path"]))
    dense_path = Path(str(dict(contract["dense_base"])["path"]))
    panel_paths = [
        Path(str(record["path"])) for record in panel_manifest["structure_panels"]
    ]
    validation = dict(study["validation"])
    tolerance = float(validation["floating_tolerance"])
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=2")
    connection.execute("PRAGMA enable_progress_bar=false")
    try:
        scan = ", ".join(atlas._sql_quote(path) for path in panel_paths)
        audit = (
            connection.execute(
                f"""
            SELECT count(*) AS rows,
                   count(*) - count(DISTINCT input_row_idx) AS duplicates,
                   count(*) FILTER (WHERE trade_date >= '2026-01-01') AS forbidden,
                   count(*) FILTER (
                       WHERE pattern_up_continuation !=
                           (pattern_nested_up_reacceleration OR pattern_zone_breakout_up)
                   ) AS continuation_contract_violations,
                   min(input_row_idx) AS minimum_row_idx,
                   max(input_row_idx) AS maximum_row_idx
            FROM read_parquet([{scan}])
            """
            )
            .fetchdf()
            .iloc[0]
        )
        if (
            int(audit["rows"]) != int(contract["row_index_rows"])
            or int(audit["duplicates"]) != 0
            or int(audit["forbidden"]) != 0
            or int(audit["continuation_contract_violations"]) != 0
            or int(audit["minimum_row_idx"]) != 0
            or int(audit["maximum_row_idx"]) != int(contract["row_index_rows"]) - 1
        ):
            raise ValueError("causal_path_validation_panel_audit_failed")
        symbols = _sample_symbols(
            connection,
            row_index_path,
            int(validation["prefix_symbols"]),
        )
        prefix_cells = 0
        panel_cells = 0
        panel_rows = 0
        cutoff_count = 0
        maximum_difference = 0.0
        symbol_records: list[dict[str, Any]] = []
        for row in symbols.itertuples(index=False):
            frame = _path_frame(connection, dense_path, str(row.symbol))
            number = min(int(validation["prefix_cutoffs_per_symbol"]), len(frame))
            cutoffs = np.unique(
                np.linspace(1, len(frame), num=number, dtype=np.int64)
            ).tolist()
            prefix = verify_prefix_invariance(
                frame,
                study,
                cutoffs=cutoffs,
                tolerance=tolerance,
            )
            parsed = study_module.parse_causal_path(
                adj_high=frame["adj_high"].to_numpy(float),
                adj_low=frame["adj_low"].to_numpy(float),
                adj_close=frame["adj_close"].to_numpy(float),
                amount=frame["amount"].to_numpy(float),
                date_indices=frame["date_idx"].to_numpy(np.int64),
                study=study,
            )
            traced_rows, traced_cells, traced_difference = _panel_trace(
                connection,
                panel_paths,
                symbol_idx=int(row.symbol_idx),
                frame=frame,
                parsed=parsed,
                tolerance=tolerance,
            )
            prefix_cells += int(prefix["feature_prefix_cells"])
            cutoff_count += int(prefix["cutoffs"])
            panel_rows += traced_rows
            panel_cells += traced_cells
            maximum_difference = max(
                maximum_difference,
                float(prefix["maximum_float_difference"]),
                float(traced_difference),
            )
            symbol_records.append(
                {
                    "symbol": str(row.symbol),
                    "symbol_idx": int(row.symbol_idx),
                    "path_rows": len(frame),
                    "panel_rows": traced_rows,
                    "cutoffs": len(cutoffs),
                }
            )
        diagnostic_audit = _validate_diagnostics(
            connection,
            study=study,
            output_root=output_root,
            panel_manifest=panel_manifest,
        )
    finally:
        connection.close()
    result = {
        "schema": study_module.VALIDATION_SCHEMA_VERSION,
        "status": "passed",
        "study_id": study_module.STUDY_ID,
        "panel_rows": int(audit["rows"]),
        "duplicate_panel_rows": int(audit["duplicates"]),
        "forbidden_2026_rows": int(audit["forbidden"]),
        "continuation_contract_violations": int(
            audit["continuation_contract_violations"]
        ),
        "sampled_symbols": len(symbol_records),
        "prefix_cutoffs": cutoff_count,
        "prefix_feature_cells_compared": prefix_cells,
        "persisted_panel_rows_recomputed": panel_rows,
        "persisted_feature_cells_compared": panel_cells,
        "maximum_float_difference": maximum_difference,
        "floating_tolerance": tolerance,
        "future_confirmed_structure_written_back": False,
        "diagnostic_validation": diagnostic_audit,
        "symbols": symbol_records,
    }
    study_module._write_json(output_root / "validation.json", result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate prefix invariance and persisted causal path panels."
    )
    parser.add_argument("--study", default=str(study_module.DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(study_module.DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = validate_study(study_path=args.study, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=atlas._json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
