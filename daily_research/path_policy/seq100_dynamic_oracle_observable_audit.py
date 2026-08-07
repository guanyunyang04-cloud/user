"""Observable-state audit for the dynamic hindsight action values.

This study does not fit a policy.  It asks a narrower question: do the
future-informed, no-fixed-horizon action values have stable structure in the
information that was observable at the signal close?  All feature alignment is
by ``(date_idx, symbol_idx)`` and every causal limitation remains explicit.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from scipy.special import ndtr

from daily_research.path_policy import seq100_market_replay as replay

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = WORKSPACE_ROOT / (
    "daily_research/studies/seq100_dynamic_oracle_observable_audit_v1.json"
)
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / (
    "daily_research/output/path_policy/studies/"
    "seq100_dynamic_oracle_observable_audit_v1"
)
STUDY_ID = "seq100_dynamic_oracle_observable_audit_v1"
SCHEMA_VERSION = 1

LABEL_COLUMNS = (
    "trade_date",
    "date_idx",
    "symbol_idx",
    "buy_action_valid",
    "oracle_cash_buy_selected",
    "buy_advantage_vs_cash",
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
    return {
        "path": str(target.resolve()),
        "size": int(target.stat().st_size),
        "sha256": replay.sha256(target),
        "rows": len(frame),
    }


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _read_json(path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("dynamic_oracle_observable_study_id_mismatch")
    source = dict(study["source"])
    if str(source["quality_pool_name"]) != "quality_liquidity_pit":
        raise ValueError("dynamic_oracle_observable_pool_mismatch")
    if int(source["expected_feature_count"]) != 557:
        raise ValueError("dynamic_oracle_observable_feature_count_mismatch")
    target = dict(study["target"])
    if bool(target["fixed_holding_horizon_used"]):
        raise ValueError("dynamic_oracle_observable_fixed_horizon_forbidden")
    if bool(target["binary_good_stock_label_used"]):
        raise ValueError("dynamic_oracle_observable_binary_label_forbidden")
    boundaries = dict(study["boundaries"])
    if tuple(boundaries["alignment_key"]) != ("date_idx", "symbol_idx"):
        raise ValueError("dynamic_oracle_observable_alignment_contract")
    return study


def _record_by_year(records: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {int(record["year"]): dict(record) for record in records}
    if len(result) != len(records):
        raise ValueError("dynamic_oracle_observable_duplicate_source_year")
    return result


def _source_contract(
    study: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[int, dict[str, Any]]],
]:
    source = dict(study["source"])
    oracle_path = _resolve(source["oracle_manifest"])
    input_path = _resolve(source["model_input_manifest"])
    if replay.sha256(oracle_path) != str(source["expected_oracle_manifest_sha256"]):
        raise ValueError("dynamic_oracle_observable_oracle_hash_mismatch")
    if replay.sha256(input_path) != str(
        source["expected_model_input_manifest_sha256"]
    ):
        raise ValueError("dynamic_oracle_observable_input_hash_mismatch")
    oracle_manifest = _read_json(oracle_path)
    input_manifest = _read_json(input_path)
    if oracle_manifest.get("status") != "completed":
        raise ValueError("dynamic_oracle_observable_oracle_incomplete")
    if input_manifest.get("status") != "completed":
        raise ValueError("dynamic_oracle_observable_input_incomplete")
    if bool(oracle_manifest.get("training_performed")):
        raise ValueError("dynamic_oracle_observable_oracle_trained")

    scenario_records: dict[str, dict[int, dict[str, Any]]] = {}
    expected_costs = tuple(dict(study["diagnostics"])["cost_scenarios"])
    raw_scenarios = {
        str(record["cost_scenario"]): dict(record)
        for record in oracle_manifest["relaxed_oracles"]
    }
    if tuple(raw_scenarios) != expected_costs:
        raise ValueError("dynamic_oracle_observable_cost_scenarios")
    for cost, record in raw_scenarios.items():
        scenario_records[cost] = _record_by_year(record["action_labels"])

    compact_names = list(
        dict(input_manifest["feature_groups"])[str(source["feature_variant"])]
    )
    features = list(input_manifest["features"])
    if (
        len(compact_names) != int(source["expected_feature_count"])
        or len(set(compact_names)) != len(compact_names)
        or [str(row["feature_name"]) for row in features] != compact_names
    ):
        raise ValueError("dynamic_oracle_observable_feature_contract")
    storage = dict(dict(input_manifest["storage"])["compact"])
    shape = tuple(int(value) for value in storage["shape"])
    feature_path = Path(str(storage["path"]))
    if shape != (int(input_manifest["row_count"]), len(compact_names)):
        raise ValueError("dynamic_oracle_observable_storage_shape")
    if not feature_path.is_file() or feature_path.stat().st_size != int(
        np.prod(shape) * np.dtype("float32").itemsize
    ):
        raise ValueError("dynamic_oracle_observable_storage_file")
    row_record = dict(input_manifest["row_index"])
    row_path = Path(str(row_record["path"]))
    if not row_path.is_file() or replay.sha256(row_path) != str(row_record["sha256"]):
        raise ValueError("dynamic_oracle_observable_row_index_hash")

    contract = {
        "oracle_manifest": {
            "path": str(oracle_path.resolve()),
            "sha256": replay.sha256(oracle_path),
        },
        "model_input_manifest": {
            "path": str(input_path.resolve()),
            "sha256": replay.sha256(input_path),
        },
        "row_index": row_record,
        "feature_storage": storage,
        "feature_count": len(compact_names),
    }
    return contract, oracle_manifest, input_manifest, scenario_records


def _packed_key(date_idx: np.ndarray, symbol_idx: np.ndarray) -> np.ndarray:
    dates = np.asarray(date_idx, dtype=np.int64)
    symbols = np.asarray(symbol_idx, dtype=np.int64)
    if bool((dates < 0).any()) or bool((symbols < 0).any()):
        raise ValueError("dynamic_oracle_observable_negative_key")
    return (dates << np.int64(32)) | symbols


def _read_label(record: Mapping[str, Any]) -> pd.DataFrame:
    path = Path(str(record["path"]))
    if replay.sha256(path) != str(record["sha256"]):
        raise ValueError("dynamic_oracle_observable_label_hash")
    frame = pd.read_parquet(path, columns=list(LABEL_COLUMNS))
    if len(frame) != int(record["rows"]):
        raise ValueError("dynamic_oracle_observable_label_rows")
    keys = _packed_key(frame["date_idx"], frame["symbol_idx"])
    if bool((keys[1:] <= keys[:-1]).any()):
        raise ValueError("dynamic_oracle_observable_label_order")
    return frame


def _align_input_rows(
    *,
    input_dates: np.ndarray,
    input_symbols: np.ndarray,
    label: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    label_dates = label["date_idx"].to_numpy(np.int64)
    minimum = int(label_dates.min())
    maximum = int(label_dates.max())
    positions = np.flatnonzero(
        (input_dates >= minimum) & (input_dates <= maximum)
    ).astype(np.int64, copy=False)
    input_keys = _packed_key(input_dates[positions], input_symbols[positions])
    if bool((input_keys[1:] <= input_keys[:-1]).any()):
        raise ValueError("dynamic_oracle_observable_input_order")
    label_keys = _packed_key(label_dates, label["symbol_idx"].to_numpy(np.int64))
    offsets = np.searchsorted(label_keys, input_keys)
    found = (offsets < len(label_keys)) & (label_keys[offsets] == input_keys)
    if not bool(found.all()):
        raise ValueError(
            f"dynamic_oracle_observable_input_not_in_oracle:{int((~found).sum())}"
        )
    return positions, offsets.astype(np.int64, copy=False)


def _midrank_percentile_at_row(values: np.ndarray, row: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    selected = matrix[int(row)]
    finite = np.isfinite(matrix)
    selected_valid = np.isfinite(selected)
    counts = finite.sum(axis=0)
    less = ((matrix < selected) & finite).sum(axis=0)
    equal = ((matrix == selected) & finite).sum(axis=0)
    result = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    valid = selected_valid & (counts > 0)
    result[valid] = (
        less[valid].astype(np.float64) + 0.5 * equal[valid].astype(np.float64)
    ) / counts[valid]
    return result


def _standardize_and_correlate(
    values: np.ndarray, target: np.ndarray, *, clip: float
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if matrix.ndim != 2 or y.shape != (matrix.shape[0],):
        raise ValueError("dynamic_oracle_observable_correlation_shape")
    finite = np.isfinite(matrix) & np.isfinite(y)[:, None]
    x0 = np.where(finite, matrix, 0.0)
    y0 = np.where(np.isfinite(y), y, 0.0)
    counts = finite.sum(axis=0).astype(np.float64)
    sum_x = x0.sum(axis=0)
    sum_x2 = np.square(x0).sum(axis=0)
    sum_y = finite.T @ y0
    sum_y2 = finite.T @ np.square(y0)
    sum_xy = x0.T @ y0
    valid = counts >= 3.0
    covariance = np.zeros(matrix.shape[1], dtype=np.float64)
    variance_x = np.zeros(matrix.shape[1], dtype=np.float64)
    variance_y = np.zeros(matrix.shape[1], dtype=np.float64)
    covariance[valid] = (
        sum_xy[valid] - sum_x[valid] * sum_y[valid] / counts[valid]
    )
    variance_x[valid] = (
        sum_x2[valid] - np.square(sum_x[valid]) / counts[valid]
    )
    variance_y[valid] = (
        sum_y2[valid] - np.square(sum_y[valid]) / counts[valid]
    )
    denominator = np.sqrt(np.maximum(variance_x * variance_y, 0.0))
    correlation = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    comparable = valid & (denominator > 1.0e-15)
    correlation[comparable] = covariance[comparable] / denominator[comparable]

    means = np.divide(
        sum_x,
        counts,
        out=np.zeros_like(sum_x),
        where=counts > 0.0,
    )
    variances = np.divide(
        variance_x,
        counts,
        out=np.zeros_like(variance_x),
        where=counts > 0.0,
    )
    scales = np.sqrt(np.maximum(variances, 0.0))
    standardized = np.full_like(matrix, np.nan, dtype=np.float64)
    usable = finite & (scales > 1.0e-12)[None, :]
    centered = matrix - means[None, :]
    np.divide(
        centered,
        scales[None, :],
        out=standardized,
        where=usable,
    )
    np.clip(standardized, -float(clip), float(clip), out=standardized)
    return standardized, correlation


def _nearest_neighbors(
    *,
    standardized: np.ndarray,
    target: np.ndarray,
    symbol_idx: np.ndarray,
    selected_row: int,
    feature_weights: np.ndarray,
    neighbor_count: int,
    minimum_coverage: float,
) -> dict[str, Any]:
    matrix = np.asarray(standardized, dtype=np.float64)
    selected = matrix[int(selected_row)]
    selected_valid = np.isfinite(selected)
    selected_weight = float(feature_weights[selected_valid].sum())
    if selected_weight <= 0.0:
        return {"neighbor_available": False, "failure_neighbor_available": False}
    common = np.isfinite(matrix) & selected_valid[None, :]
    common_weight = common @ feature_weights
    difference = np.where(common, matrix - selected[None, :], 0.0)
    squared = np.square(difference) @ feature_weights
    distance = np.divide(
        squared,
        common_weight,
        out=np.full(len(matrix), np.inf, dtype=np.float64),
        where=common_weight > 0.0,
    )
    coverage = common_weight / selected_weight
    eligible = (
        np.isfinite(target)
        & np.isfinite(distance)
        & (coverage >= float(minimum_coverage))
    )
    eligible[int(selected_row)] = False
    candidate_rows = np.flatnonzero(eligible)
    result: dict[str, Any] = {
        "neighbor_available": bool(len(candidate_rows)),
        "failure_neighbor_available": False,
    }
    if len(candidate_rows):
        order = candidate_rows[np.argsort(distance[candidate_rows], kind="mergesort")]
        chosen = order[: min(int(neighbor_count), len(order))]
        nearest = int(chosen[0])
        result.update(
            {
                "nearest_symbol_idx": int(symbol_idx[nearest]),
                "nearest_distance": float(distance[nearest]),
                "nearest_coverage": float(coverage[nearest]),
                "nearest_advantage": float(target[nearest]),
                "neighbor_k": len(chosen),
                "neighbor_k_mean_advantage": float(np.mean(target[chosen])),
                "neighbor_k_positive_fraction": float(np.mean(target[chosen] > 0.0)),
                "median_candidate_distance": float(np.median(distance[candidate_rows])),
            }
        )
    failure_rows = np.flatnonzero(eligible & (target <= 0.0))
    if len(failure_rows):
        failure = int(failure_rows[np.argmin(distance[failure_rows])])
        result.update(
            {
                "failure_neighbor_available": True,
                "nearest_failure_symbol_idx": int(symbol_idx[failure]),
                "nearest_failure_distance": float(distance[failure]),
                "nearest_failure_coverage": float(coverage[failure]),
                "nearest_failure_advantage": float(target[failure]),
            }
        )
    return result


def _hac_mean(values: np.ndarray, *, lag: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0).astype(np.float64)
    means = np.divide(
        np.where(finite, matrix, 0.0).sum(axis=0),
        counts,
        out=np.full(matrix.shape[1], np.nan, dtype=np.float64),
        where=counts > 0.0,
    )
    residual = np.where(finite, matrix - means[None, :], 0.0)
    denominator = np.maximum(counts, 1.0)
    long_run = np.square(residual).sum(axis=0) / denominator
    maximum_lag = min(int(lag), max(matrix.shape[0] - 1, 0))
    for offset in range(1, maximum_lag + 1):
        weight = 1.0 - offset / (maximum_lag + 1.0)
        covariance = (
            residual[offset:] * residual[:-offset]
        ).sum(axis=0) / denominator
        long_run += 2.0 * weight * covariance
    standard_error = np.sqrt(np.maximum(long_run, 0.0) / denominator)
    standard_error[counts < 2.0] = np.nan
    return means, standard_error


def _nanmean_columns(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0)
    return np.divide(
        np.where(finite, matrix, 0.0).sum(axis=0),
        counts,
        out=np.full(matrix.shape[1], np.nan, dtype=np.float64),
        where=counts > 0,
    )


def _close_memmap(values: np.memmap) -> None:
    mapping = getattr(values, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    result = np.full_like(values, np.nan)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if not len(valid_positions):
        return result
    order = valid_positions[np.argsort(values[valid_positions], kind="mergesort")]
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _feature_metric_frame(
    *,
    feature_metadata: pd.DataFrame,
    daily_dates: np.ndarray,
    winner_percentiles: np.ndarray,
    daily_correlations: np.ndarray,
    hac_lag: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = np.asarray([int(str(value)[:4]) for value in daily_dates], dtype=np.int16)
    winner_effect = winner_percentiles.astype(np.float64) - 0.5
    winner_mean, winner_se = _hac_mean(winner_effect, lag=hac_lag)
    ic_mean, ic_se = _hac_mean(daily_correlations, lag=hac_lag)
    winner_z = np.divide(
        winner_mean,
        winner_se,
        out=np.full_like(winner_mean, np.nan),
        where=winner_se > 0.0,
    )
    ic_z = np.divide(
        ic_mean,
        ic_se,
        out=np.full_like(ic_mean, np.nan),
        where=ic_se > 0.0,
    )
    winner_p = 2.0 * ndtr(-np.abs(winner_z))
    ic_p = 2.0 * ndtr(-np.abs(ic_z))
    metrics = feature_metadata.copy()
    metrics["winner_observations"] = np.isfinite(winner_effect).sum(axis=0)
    metrics["winner_percentile_effect"] = winner_mean
    metrics["winner_hac_se"] = winner_se
    metrics["winner_hac_ci_low"] = winner_mean - 1.96 * winner_se
    metrics["winner_hac_ci_high"] = winner_mean + 1.96 * winner_se
    metrics["winner_p_value"] = winner_p
    metrics["winner_fdr_q"] = _benjamini_hochberg(winner_p)
    metrics["ic_observations"] = np.isfinite(daily_correlations).sum(axis=0)
    metrics["mean_daily_linear_ic"] = ic_mean
    metrics["ic_hac_se"] = ic_se
    metrics["ic_hac_ci_low"] = ic_mean - 1.96 * ic_se
    metrics["ic_hac_ci_high"] = ic_mean + 1.96 * ic_se
    metrics["ic_p_value"] = ic_p
    metrics["ic_fdr_q"] = _benjamini_hochberg(ic_p)

    annual_rows: list[dict[str, Any]] = []
    for year in sorted(set(years.tolist())):
        mask = years == int(year)
        winner_annual = _nanmean_columns(winner_effect[mask])
        ic_annual = _nanmean_columns(daily_correlations[mask])
        for index, name in enumerate(metrics["feature_name"].astype(str)):
            annual_rows.append(
                {
                    "year": int(year),
                    "feature_name": name,
                    "winner_percentile_effect": float(winner_annual[index]),
                    "mean_daily_linear_ic": float(ic_annual[index]),
                    "winner_observations": int(
                        np.isfinite(winner_effect[mask, index]).sum()
                    ),
                    "ic_observations": int(
                        np.isfinite(daily_correlations[mask, index]).sum()
                    ),
                }
            )
    annual = pd.DataFrame(annual_rows)
    annual_winner = annual.pivot(
        index="feature_name", columns="year", values="winner_percentile_effect"
    ).reindex(metrics["feature_name"])
    annual_ic = annual.pivot(
        index="feature_name", columns="year", values="mean_daily_linear_ic"
    ).reindex(metrics["feature_name"])
    winner_sign = np.sign(metrics["winner_percentile_effect"].to_numpy())[:, None]
    ic_sign = np.sign(metrics["mean_daily_linear_ic"].to_numpy())[:, None]
    metrics["winner_annual_same_sign_count"] = np.sum(
        np.sign(annual_winner.to_numpy()) == winner_sign, axis=1
    )
    metrics["ic_annual_same_sign_count"] = np.sum(
        np.sign(annual_ic.to_numpy()) == ic_sign, axis=1
    )
    metrics["annual_year_count"] = annual_winner.notna().sum(axis=1).to_numpy()
    return metrics, annual


def _feature_weights(feature_metadata: pd.DataFrame) -> np.ndarray:
    counts = feature_metadata.groupby("analytic_family")["feature_name"].transform(
        "count"
    )
    weights = 1.0 / counts.to_numpy(np.float64)
    return weights / float(weights.sum())


def _family_summary(metrics: pd.DataFrame, *, fdr_level: float) -> pd.DataFrame:
    grouped = metrics.groupby("analytic_family", sort=True)
    rows: list[dict[str, Any]] = []
    for family, frame in grouped:
        rows.append(
            {
                "analytic_family": str(family),
                "feature_count": len(frame),
                "median_abs_winner_percentile_effect": float(
                    frame["winner_percentile_effect"].abs().median()
                ),
                "maximum_abs_winner_percentile_effect": float(
                    frame["winner_percentile_effect"].abs().max()
                ),
                "winner_fdr_significant_features": int(
                    frame["winner_fdr_q"].le(float(fdr_level)).sum()
                ),
                "median_abs_daily_linear_ic": float(
                    frame["mean_daily_linear_ic"].abs().median()
                ),
                "maximum_abs_daily_linear_ic": float(
                    frame["mean_daily_linear_ic"].abs().max()
                ),
                "ic_fdr_significant_features": int(
                    frame["ic_fdr_q"].le(float(fdr_level)).sum()
                ),
                "maximum_winner_annual_same_sign_count": int(
                    frame["winner_annual_same_sign_count"].max()
                ),
                "maximum_ic_annual_same_sign_count": int(
                    frame["ic_annual_same_sign_count"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _cost_and_margin_diagnostics(
    *,
    study: Mapping[str, Any],
    oracle_manifest: Mapping[str, Any],
    target_differences: Sequence[np.ndarray],
    cost_counts: Mapping[str, int | float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root_records = {
        str(record["cost_scenario"]): dict(record)
        for record in oracle_manifest["relaxed_oracles"]
    }
    daily: dict[str, pd.DataFrame] = {}
    trades: dict[str, pd.DataFrame] = {}
    for cost, record in root_records.items():
        daily_record = dict(record["daily_cash_actions"])
        trade_record = dict(record["selected_trades"])
        if replay.sha256(daily_record["path"]) != str(daily_record["sha256"]):
            raise ValueError("dynamic_oracle_observable_daily_hash")
        if replay.sha256(trade_record["path"]) != str(trade_record["sha256"]):
            raise ValueError("dynamic_oracle_observable_trade_hash")
        daily[cost] = pd.read_parquet(daily_record["path"])
        trades[cost] = pd.read_parquet(trade_record["path"])

    base = daily["base"].sort_values("date_idx", kind="mergesort").reset_index(
        drop=True
    )
    stress = daily["double_slippage"].sort_values(
        "date_idx", kind="mergesort"
    ).reset_index(drop=True)
    if not np.array_equal(base["date_idx"], stress["date_idx"]):
        raise ValueError("dynamic_oracle_observable_daily_alignment")
    same_action = base["cash_action"].astype(str).eq(stress["cash_action"].astype(str))
    same_symbol = base["cash_selected_symbol_idx"].astype(int).eq(
        stress["cash_selected_symbol_idx"].astype(int)
    )

    perturbation_rows: list[dict[str, Any]] = []
    for cost, frame in daily.items():
        buy = frame["cash_action"].eq("buy").to_numpy()
        margin = frame["selected_action_margin"].to_numpy(np.float64)
        advantage = frame["selected_advantage_over_stay"].to_numpy(np.float64)
        for bps in dict(study["diagnostics"])["bounded_value_perturbations_bps"]:
            epsilon = float(bps) / 10_000.0
            perturbation_rows.append(
                {
                    "cost_scenario": str(cost),
                    "perturbation_bps_per_action_value": int(bps),
                    "buy_dates": int(buy.sum()),
                    "argmax_guaranteed_stable_dates": int(
                        (buy & (margin > 2.0 * epsilon)).sum()
                    ),
                    "argmax_guaranteed_stable_fraction": float(
                        np.mean(margin[buy] > 2.0 * epsilon) if buy.any() else math.nan
                    ),
                    "buy_vs_cash_guaranteed_stable_dates": int(
                        (buy & (advantage > 2.0 * epsilon)).sum()
                    ),
                    "buy_vs_cash_guaranteed_stable_fraction": float(
                        np.mean(advantage[buy] > 2.0 * epsilon)
                        if buy.any()
                        else math.nan
                    ),
                }
            )

    base_path = {
        (int(row.signal_date_idx), int(row.symbol_idx))
        for row in trades["base"].itertuples(index=False)
    }
    stress_path = {
        (int(row.signal_date_idx), int(row.symbol_idx))
        for row in trades["double_slippage"].itertuples(index=False)
    }
    intersection = base_path & stress_path
    union = base_path | stress_path
    differences = (
        np.concatenate(target_differences).astype(np.float64, copy=False)
        if target_differences
        else np.empty(0, dtype=np.float64)
    )
    summary = {
        **{key: int(value) for key, value in cost_counts.items()},
        "daily_rows": len(base),
        "same_cash_action_dates": int(same_action.sum()),
        "same_cash_action_fraction": float(same_action.mean()),
        "same_selected_symbol_dates": int(same_symbol.sum()),
        "same_selected_symbol_fraction": float(same_symbol.mean()),
        "base_path_trades": len(base_path),
        "double_slippage_path_trades": len(stress_path),
        "path_trade_intersection": len(intersection),
        "path_trade_union": len(union),
        "path_trade_jaccard": float(len(intersection) / len(union)),
        "target_difference_mean": float(differences.mean()),
        "target_difference_mean_abs": float(np.abs(differences).mean()),
        "target_difference_q01": float(np.quantile(differences, 0.01)),
        "target_difference_median": float(np.median(differences)),
        "target_difference_q99": float(np.quantile(differences, 0.99)),
    }
    overlap = pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value in summary.items()
        ]
    )
    return overlap, pd.DataFrame(perturbation_rows), summary


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    memory_start = int(psutil.virtual_memory().available / 1024**2)
    study = load_study(study_path)
    root = _resolve(output_root or study["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    row_group_size = int(dict(study["resources"])["parquet_row_group_size"])
    diagnostics = dict(study["diagnostics"])
    formal_years = tuple(int(value) for value in dict(study["period"])["formal_years"])
    contract, oracle_manifest, input_manifest, scenario_records = _source_contract(
        study
    )

    row_path = Path(str(dict(input_manifest["row_index"])["path"]))
    row_index = pd.read_parquet(row_path, columns=["date_idx", "symbol_idx"])
    if len(row_index) != int(input_manifest["row_count"]):
        raise ValueError("dynamic_oracle_observable_input_row_count")
    input_dates = row_index["date_idx"].to_numpy(np.int64)
    input_symbols = row_index["symbol_idx"].to_numpy(np.int64)
    input_keys = _packed_key(input_dates, input_symbols)
    duplicate_input_keys = int((input_keys[1:] == input_keys[:-1]).sum())
    if duplicate_input_keys or bool((input_keys[1:] < input_keys[:-1]).any()):
        raise ValueError("dynamic_oracle_observable_input_keys")

    storage = dict(dict(input_manifest["storage"])["compact"])
    shape = tuple(int(value) for value in storage["shape"])
    feature_path = Path(str(storage["path"]))
    feature_metadata = pd.DataFrame(input_manifest["features"])[
        ["feature_name", "column_index", "block", "analytic_family", "eligibility"]
    ].copy()
    feature_weights = _feature_weights(feature_metadata)

    winner_percentiles: list[np.ndarray] = []
    daily_correlations: list[np.ndarray] = []
    daily_dates: list[str] = []
    neighbor_rows: list[dict[str, Any]] = []
    target_differences: list[np.ndarray] = []
    cost_counts: defaultdict[str, int] = defaultdict(int)
    aligned_rows = 0
    oracle_rows = 0
    selected_missing_input = 0
    forbidden_rows = 0
    reserve_memory_mb = int(dict(study["resources"])["reserve_memory_mb"])
    minimum_available_memory_mb = memory_start
    memory_reopen_count = 0

    for year in formal_years:
        feature_values = np.memmap(
            feature_path, dtype=np.float32, mode="r", shape=shape
        )
        base = _read_label(scenario_records["base"][year])
        stress = _read_label(scenario_records["double_slippage"][year])
        oracle_rows += len(base)
        forbidden_rows += int(
            base["trade_date"].astype(str).str[:4].astype(int).ge(2026).sum()
        )
        base_keys = _packed_key(base["date_idx"], base["symbol_idx"])
        stress_keys = _packed_key(stress["date_idx"], stress["symbol_idx"])
        if not np.array_equal(base_keys, stress_keys):
            raise ValueError(f"dynamic_oracle_observable_cost_key_alignment:{year}")
        positions, offsets = _align_input_rows(
            input_dates=input_dates,
            input_symbols=input_symbols,
            label=base,
        )
        aligned_rows += len(positions)
        base_adv_all = base["buy_advantage_vs_cash"].to_numpy(np.float64)
        stress_adv_all = stress["buy_advantage_vs_cash"].to_numpy(np.float64)
        base_valid_all = base["buy_action_valid"].to_numpy(bool)
        stress_valid_all = stress["buy_action_valid"].to_numpy(bool)
        both = (
            base_valid_all
            & stress_valid_all
            & np.isfinite(base_adv_all)
            & np.isfinite(stress_adv_all)
        )
        differences = stress_adv_all[both] - base_adv_all[both]
        target_differences.append(differences.astype(np.float32, copy=False))
        cost_counts["comparable_action_rows"] += int(both.sum())
        cost_counts["base_positive_action_rows"] += int(
            (both & (base_adv_all > 0.0)).sum()
        )
        cost_counts["double_slippage_positive_action_rows"] += int(
            (both & (stress_adv_all > 0.0)).sum()
        )
        cost_counts["positive_to_nonpositive_flips"] += int(
            (both & (base_adv_all > 0.0) & (stress_adv_all <= 0.0)).sum()
        )
        cost_counts["nonpositive_to_positive_flips"] += int(
            (both & (base_adv_all <= 0.0) & (stress_adv_all > 0.0)).sum()
        )

        base_adv = base_adv_all[offsets]
        stress_adv = stress_adv_all[offsets]
        base_valid = base_valid_all[offsets]
        base_selected = base["oracle_cash_buy_selected"].to_numpy(bool)[offsets]
        stress_selected = stress["oracle_cash_buy_selected"].to_numpy(bool)[offsets]
        aligned_dates = input_dates[positions]
        aligned_symbols = input_symbols[positions]
        date_values = base["trade_date"].astype(str).to_numpy()[offsets]
        starts = np.flatnonzero(
            np.r_[True, aligned_dates[1:] != aligned_dates[:-1]]
        )
        stops = np.r_[starts[1:], len(aligned_dates)]
        for group_number, (start, stop) in enumerate(
            zip(starts, stops, strict=True), start=1
        ):
            absolute_positions = positions[start:stop]
            if not np.array_equal(
                absolute_positions,
                np.arange(absolute_positions[0], absolute_positions[-1] + 1),
            ):
                raise ValueError("dynamic_oracle_observable_date_rows_not_contiguous")
            valid = (
                base_valid[start:stop]
                & np.isfinite(base_adv[start:stop])
            )
            selected = base_selected[start:stop] & valid
            if int(selected.sum()) == 0:
                full_date = int(aligned_dates[start])
                source_date_mask = base["date_idx"].to_numpy(np.int64) == full_date
                source_selected = int(
                    base.loc[source_date_mask, "oracle_cash_buy_selected"].sum()
                )
                if source_selected:
                    selected_missing_input += 1
                continue
            if int(selected.sum()) != 1:
                raise ValueError("dynamic_oracle_observable_selected_multiplicity")
            local_valid_rows = np.flatnonzero(valid)
            selected_input_row = int(np.flatnonzero(selected)[0])
            selected_row = int(np.flatnonzero(local_valid_rows == selected_input_row)[0])
            matrix = np.asarray(
                feature_values[absolute_positions[valid]], dtype=np.float64
            )
            target = base_adv[start:stop][valid]
            stress_target = stress_adv[start:stop][valid]
            symbols = aligned_symbols[start:stop][valid]
            standardized, correlation = _standardize_and_correlate(
                matrix,
                target,
                clip=float(diagnostics["standardized_feature_clip"]),
            )
            winner_percentiles.append(
                _midrank_percentile_at_row(matrix, selected_row).astype(np.float32)
            )
            daily_correlations.append(correlation.astype(np.float32))
            trade_date = str(date_values[start])
            daily_dates.append(trade_date)
            neighbors = _nearest_neighbors(
                standardized=standardized,
                target=target,
                symbol_idx=symbols,
                selected_row=selected_row,
                feature_weights=feature_weights,
                neighbor_count=int(diagnostics["neighbor_count"]),
                minimum_coverage=float(
                    diagnostics["minimum_common_feature_weight_fraction"]
                ),
            )
            stress_choice = np.flatnonzero(stress_selected[start:stop] & valid)
            neighbor_rows.append(
                {
                    "trade_date": trade_date,
                    "year": int(year),
                    "date_idx": int(aligned_dates[start]),
                    "candidate_count": int(valid.sum()),
                    "selected_symbol_idx": int(symbols[selected_row]),
                    "selected_advantage": float(target[selected_row]),
                    "selected_stress_advantage": float(stress_target[selected_row]),
                    "stress_selected_symbol_idx": (
                        int(aligned_symbols[start:stop][stress_choice[0]])
                        if len(stress_choice) == 1
                        else -1
                    ),
                    **neighbors,
                }
            )
            del matrix, standardized, correlation
            if group_number % 32 == 0:
                available = int(psutil.virtual_memory().available / 1024**2)
                minimum_available_memory_mb = min(
                    minimum_available_memory_mb, available
                )
                if available < reserve_memory_mb:
                    _close_memmap(feature_values)
                    del feature_values
                    gc.collect()
                    feature_values = np.memmap(
                        feature_path, dtype=np.float32, mode="r", shape=shape
                    )
                    memory_reopen_count += 1
        _close_memmap(feature_values)
        del feature_values, base, stress
        gc.collect()
        minimum_available_memory_mb = min(
            minimum_available_memory_mb,
            int(psutil.virtual_memory().available / 1024**2),
        )

    if aligned_rows != len(row_index):
        raise ValueError(
            f"dynamic_oracle_observable_alignment_rows:{aligned_rows}:{len(row_index)}"
        )
    if not winner_percentiles:
        raise ValueError("dynamic_oracle_observable_no_selected_dates")
    winner_matrix = np.vstack(winner_percentiles)
    correlation_matrix = np.vstack(daily_correlations)
    feature_metrics, annual_metrics = _feature_metric_frame(
        feature_metadata=feature_metadata,
        daily_dates=np.asarray(daily_dates, dtype=object),
        winner_percentiles=winner_matrix,
        daily_correlations=correlation_matrix,
        hac_lag=int(diagnostics["hac_lag_sessions"]),
    )
    family_metrics = _family_summary(
        feature_metrics, fdr_level=float(diagnostics["fdr_level"])
    )
    neighbor_frame = pd.DataFrame(neighbor_rows).sort_values(
        "date_idx", kind="mergesort"
    )
    overlap, perturbations, cost_summary = _cost_and_margin_diagnostics(
        study=study,
        oracle_manifest=oracle_manifest,
        target_differences=target_differences,
        cost_counts=cost_counts,
    )

    outputs = {
        "feature_metrics": _write_frame(
            root / "feature_metrics.parquet",
            feature_metrics,
            row_group_size=row_group_size,
        ),
        "feature_annual_metrics": _write_frame(
            root / "feature_annual_metrics.parquet",
            annual_metrics,
            row_group_size=row_group_size,
        ),
        "family_metrics": _write_frame(
            root / "family_metrics.parquet",
            family_metrics,
            row_group_size=row_group_size,
        ),
        "matched_neighbors": _write_frame(
            root / "matched_neighbors.parquet",
            neighbor_frame,
            row_group_size=row_group_size,
        ),
        "cost_overlap": _write_frame(
            root / "cost_overlap.parquet", overlap, row_group_size=row_group_size
        ),
        "perturbation_stability": _write_frame(
            root / "perturbation_stability.parquet",
            perturbations,
            row_group_size=row_group_size,
        ),
    }
    audit = {
        "status": "passed",
        "oracle_rows": int(oracle_rows),
        "model_input_rows": len(row_index),
        "aligned_rows": int(aligned_rows),
        "oracle_rows_without_model_input": int(oracle_rows - aligned_rows),
        "duplicate_input_keys": duplicate_input_keys,
        "forbidden_2026_rows": forbidden_rows,
        "selected_dates_analyzed": len(neighbor_frame),
        "selected_dates_missing_model_input": selected_missing_input,
        "feature_count": len(feature_metrics),
        "neighbor_available_dates": int(
            neighbor_frame["neighbor_available"].astype(bool).sum()
        ),
        "failure_neighbor_available_dates": int(
            neighbor_frame["failure_neighbor_available"].astype(bool).sum()
        ),
        "cost_summary": cost_summary,
    }
    memory_end = int(psutil.virtual_memory().available / 1024**2)
    manifest = {
        "schema": f"seq100_dynamic_oracle_observable_audit/{SCHEMA_VERSION}",
        "status": "completed",
        "study_id": STUDY_ID,
        "study": {
            "path": str(_resolve(study_path).resolve()),
            "sha256": replay.sha256(study_path),
        },
        "source_contract": contract,
        "audit": audit,
        "outputs": outputs,
        "runtime": {
            "elapsed_seconds": float(time.perf_counter() - started),
            "available_memory_mb_at_start": memory_start,
            "available_memory_mb_at_end": memory_end,
            "minimum_available_memory_mb_observed": minimum_available_memory_mb,
            "reserve_memory_mb": reserve_memory_mb,
            "memory_reopen_count": memory_reopen_count,
            "logical_cpu_count": int(psutil.cpu_count() or 1),
            "adaptive_memory": bool(dict(study["resources"])["adaptive_memory"]),
        },
        "fixed_holding_horizon_used": False,
        "binary_good_stock_label_used": False,
        "future_path_used": True,
        "training_performed": False,
        "causal_policy_evaluated": False,
        "profit_claim_allowed": False,
        "production_policy_selected": False,
        "report_generation_performed": False,
    }
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "audit.json", audit)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit observable structure in dynamic hindsight action values."
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    result = run_study(study_path=args.study, output_root=args.output_root)
    print(json.dumps(result["audit"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
