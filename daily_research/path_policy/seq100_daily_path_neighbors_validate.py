"""Independent validation and nonparametric phase-feature diagnostics."""

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
import pyarrow.parquet as pq
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from daily_research.path_policy import seq100_daily_path_neighbors as neighbors
from daily_research.path_policy import seq100_hot_path_atlas as atlas

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_ROOT = neighbors.DEFAULT_OUTPUT_ROOT
VALIDATION_SCHEMA = "seq100_daily_path_neighbor_validation/1"
MODEL_COLUMNS = {
    "prior": "prior_probability",
    "geometry": "geometry_probability",
    "sequence_raw": "sequence_raw_probability",
    "sequence": "sequence_probability",
}
COMPARISONS = {
    "geometry_vs_prior": ("prior", "geometry"),
    "sequence_raw_vs_geometry": ("geometry", "sequence_raw"),
    "sequence_vs_geometry": ("geometry", "sequence"),
    "sequence_vs_prior": ("prior", "sequence"),
}
PHASE_PERIODS = {
    "discovery_2012_2018": set(range(2012, 2019)),
    "strict_oos_2019_2023": set(range(2019, 2024)),
    "near_complete_2024": {2024},
    "provisional_2025": {2025},
}
PHASE_FEATURES = tuple(dict.fromkeys(neighbors.GEOMETRY_FEATURES))


def _prediction_glob(analysis_root: Path) -> Path:
    return analysis_root / "oos_predictions" / "year=*" / "part-0000.parquet"


def _clip_sql(column: str) -> str:
    return f"greatest(1e-6, least(1.0 - 1e-6, {column}))"


def _independent_daily_scores(
    connection: duckdb.DuckDBPyConnection, prediction_glob: Path
) -> pd.DataFrame:
    expressions: list[str] = []
    for model, column in MODEL_COLUMNS.items():
        clipped = _clip_sql(column)
        expressions.extend(
            [
                (
                    "avg(-(extreme_ahead::INTEGER) * ln("
                    f"{clipped}) - (1 - extreme_ahead::INTEGER) * "
                    f"ln(1 - {clipped})) AS {model}_log_loss"
                ),
                (f"avg(pow(extreme_ahead::INTEGER - {column}, 2)) AS {model}_brier"),
            ]
        )
    return connection.execute(
        f"""
        SELECT
            signal_year,
            date_idx,
            turning_scale,
            causal_next_type,
            count(*) AS rows,
            avg(extreme_ahead::INTEGER) AS actual_rate,
            {", ".join(expressions)}
        FROM read_parquet(
            {atlas._sql_quote(prediction_glob)}, hive_partitioning=true
        )
        WHERE extreme_ahead IS NOT NULL
        GROUP BY ALL
        ORDER BY signal_year, date_idx, turning_scale, causal_next_type
        """
    ).fetchdf()


def _independent_information(
    daily: pd.DataFrame, study: Mapping[str, Any]
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    keys = ["signal_year", "date_idx", "turning_scale", "causal_next_type"]
    for comparison, (baseline, candidate) in COMPARISONS.items():
        part = daily[keys].copy()
        part["comparison"] = comparison
        part["information_gain_nats"] = (
            daily[f"{baseline}_log_loss"] - daily[f"{candidate}_log_loss"]
        )
        parts.append(part)
    cells = pd.concat(parts, ignore_index=True)
    dates = (
        cells.groupby(["signal_year", "date_idx", "comparison"], sort=True)[
            "information_gain_nats"
        ]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for scope, years in neighbors._evaluation_scopes(study).items():
        scoped = dates[dates["signal_year"].isin(years)]
        for comparison, group in scoped.groupby("comparison", sort=True):
            mean_nats = float(group["information_gain_nats"].mean())
            rows.append(
                {
                    "scope": scope,
                    "comparison": comparison,
                    "dates": len(group),
                    "information_gain_nats": mean_nats,
                    "information_gain_bits": mean_nats / math.log(2.0),
                }
            )
    return pd.DataFrame(rows)


def _equal_frequency_ece(
    actual: np.ndarray, probability: np.ndarray, bins: int
) -> float:
    order = np.argsort(probability, kind="mergesort")
    weighted_gap = 0.0
    for positions in np.array_split(order, bins):
        if len(positions) == 0:
            continue
        weighted_gap += len(positions) * abs(
            float(actual[positions].mean() - probability[positions].mean())
        )
    return weighted_gap / len(actual)


def _daily_auc(
    years: np.ndarray,
    dates: np.ndarray,
    actual: np.ndarray,
    probability: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "signal_year": years,
            "date_idx": dates,
            "actual": actual,
            "probability": probability,
        }
    )
    frame["rank"] = frame.groupby("date_idx", sort=False)["probability"].rank(
        method="average"
    )
    frame["positive_rank"] = frame["rank"] * frame["actual"]
    grouped = (
        frame.groupby(["signal_year", "date_idx"], sort=True)
        .agg(
            rows=("actual", "size"),
            positives=("actual", "sum"),
            positive_rank_sum=("positive_rank", "sum"),
        )
        .reset_index()
    )
    negatives = grouped["rows"] - grouped["positives"]
    valid = (grouped["positives"] > 0) & (negatives > 0)
    grouped["auc"] = np.nan
    grouped.loc[valid, "auc"] = (
        grouped.loc[valid, "positive_rank_sum"]
        - grouped.loc[valid, "positives"]
        * (grouped.loc[valid, "positives"] + 1.0)
        / 2.0
    ) / (grouped.loc[valid, "positives"] * negatives[valid])
    return grouped


def _auc_calibration_metrics(
    connection: duckdb.DuckDBPyConnection,
    prediction_glob: Path,
    study: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    probability_columns = ", ".join(MODEL_COLUMNS.values())
    for scale in neighbors.TURNING_SCALES:
        for next_type in ("peak", "trough"):
            frame = connection.execute(
                f"""
                SELECT signal_year, date_idx,
                       extreme_ahead::TINYINT AS actual,
                       {probability_columns}
                FROM read_parquet(
                    {atlas._sql_quote(prediction_glob)}, hive_partitioning=true
                )
                WHERE extreme_ahead IS NOT NULL
                  AND turning_scale = {scale}
                  AND causal_next_type = {atlas._sql_quote(next_type)}
                ORDER BY date_idx, symbol
                """
            ).fetchdf()
            years = frame["signal_year"].to_numpy(np.int16)
            dates = frame["date_idx"].to_numpy(np.int32)
            actual = frame["actual"].to_numpy(np.int8)
            for model, column in MODEL_COLUMNS.items():
                probability = frame[column].to_numpy(np.float64)
                daily = _daily_auc(years, dates, actual, probability)
                for scope, scope_years in neighbors._evaluation_scopes(study).items():
                    mask = np.isin(years, tuple(scope_years))
                    daily_scope = daily[daily["signal_year"].isin(scope_years)]
                    target = actual[mask]
                    prediction = probability[mask]
                    order = np.argsort(prediction, kind="mergesort")
                    tail_count = max(1, len(order) // 10)
                    clipped = np.clip(prediction, 1e-6, 1.0 - 1e-6)
                    rows.append(
                        {
                            "scope": scope,
                            "turning_scale": scale,
                            "causal_next_type": next_type,
                            "model": model,
                            "rows": len(target),
                            "event_auc": float(roc_auc_score(target, prediction)),
                            "mean_daily_auc": float(daily_scope["auc"].mean()),
                            "daily_auc_dates": int(daily_scope["auc"].notna().sum()),
                            "log_loss": float(
                                np.mean(
                                    -target * np.log(clipped)
                                    - (1 - target) * np.log(1.0 - clipped)
                                )
                            ),
                            "brier": float(np.mean(np.square(target - prediction))),
                            "mean_probability": float(prediction.mean()),
                            "actual_rate": float(target.mean()),
                            "ece_10": _equal_frequency_ece(target, prediction, bins=10),
                            "lowest_decile_actual_rate": float(
                                target[order[:tail_count]].mean()
                            ),
                            "highest_decile_actual_rate": float(
                                target[order[-tail_count:]].mean()
                            ),
                        }
                    )
            del frame
    by_cell = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for keys, group in by_cell.groupby(["scope", "model"], sort=True):
        scope, model = keys
        summary_rows.append(
            {
                "scope": scope,
                "model": model,
                "cells": len(group),
                "rows": int(group["rows"].sum()),
                "macro_event_auc": float(group["event_auc"].mean()),
                "mean_daily_auc": float(
                    np.average(
                        group["mean_daily_auc"], weights=group["daily_auc_dates"]
                    )
                ),
                "row_weighted_log_loss": float(
                    np.average(group["log_loss"], weights=group["rows"])
                ),
                "row_weighted_brier": float(
                    np.average(group["brier"], weights=group["rows"])
                ),
                "row_weighted_ece_10": float(
                    np.average(group["ece_10"], weights=group["rows"])
                ),
                "mean_lowest_decile_actual_rate": float(
                    group["lowest_decile_actual_rate"].mean()
                ),
                "mean_highest_decile_actual_rate": float(
                    group["highest_decile_actual_rate"].mean()
                ),
            }
        )
    return by_cell, pd.DataFrame(summary_rows)


def _prediction_audit(
    connection: duckdb.DuckDBPyConnection, prediction_glob: Path
) -> tuple[dict[str, Any], pd.DataFrame]:
    continuous_differences = ", ".join(
        (
            f"max(abs(predicted_{metric}_sequence - "
            f"predicted_{metric}_geometry)) AS {metric}_maximum_difference"
        )
        for metric in neighbors.CONTINUOUS_METRICS
    )
    row = (
        connection.execute(
            f"""
        SELECT
            count(*) AS rows,
            count(*) - count(DISTINCT
                symbol || '|' || date_idx::VARCHAR || '|'
                || turning_scale::VARCHAR
            ) AS duplicate_keys,
            count(*) FILTER (WHERE signal_year = 2026) AS forbidden_rows,
            count(*) FILTER (
                WHERE NOT isfinite(prior_probability)
                   OR NOT isfinite(geometry_probability)
                   OR NOT isfinite(sequence_raw_probability)
                   OR NOT isfinite(sequence_probability)
            ) AS nonfinite_probability_rows,
            count(*) FILTER (
                WHERE maximum_training_resolution_date_idx >= date_idx
            ) AS causal_update_violations,
            max(abs(sequence_probability - geometry_probability))
                AS nested_probability_maximum_difference,
            {continuous_differences}
        FROM read_parquet(
            {atlas._sql_quote(prediction_glob)}, hive_partitioning=true
        )
        """
        )
        .fetchdf()
        .iloc[0]
    )
    audit = {
        key: float(value) if "difference" in key else int(value)
        for key, value in row.to_dict().items()
    }
    censoring = connection.execute(
        f"""
        SELECT signal_year, turning_scale,
               count(*) AS rows,
               count(*) FILTER (WHERE extreme_ahead IS NULL) AS censored_rows
        FROM read_parquet(
            {atlas._sql_quote(prediction_glob)}, hive_partitioning=true
        )
        GROUP BY ALL
        ORDER BY signal_year, turning_scale
        """
    ).fetchdf()
    return audit, censoring


def _blend_increment_controls(
    connection: duckdb.DuckDBPyConnection,
    prediction_glob: Path,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    weights = tuple(
        float(value)
        for value in dict(study["representation"])[
            "sequence_incremental_weight_candidates"
        ]
    )
    sources = {
        "sequence_raw": "sequence_raw_probability",
        "prior": "prior_probability",
        "half": "0.5",
    }
    expressions: list[str] = []
    aliases: dict[tuple[str, float], str] = {}
    for source_index, (source, column) in enumerate(sources.items()):
        for weight_index, weight in enumerate(weights):
            alias = f"loss_{source_index}_{weight_index}"
            aliases[(source, weight)] = alias
            probability = _clip_sql(
                f"({1.0 - weight:.12g}) * geometry_probability "
                f"+ ({weight:.12g}) * ({column})"
            )
            expressions.append(
                "avg(-(extreme_ahead::INTEGER) * ln("
                f"{probability}) - (1 - extreme_ahead::INTEGER) * "
                f"ln(1 - {probability})) AS {alias}"
            )
    cells = connection.execute(
        f"""
        SELECT signal_year, date_idx, turning_scale, causal_next_type,
               {", ".join(expressions)}
        FROM read_parquet(
            {atlas._sql_quote(prediction_glob)}, hive_partitioning=true
        )
        WHERE extreme_ahead IS NOT NULL
        GROUP BY ALL
        ORDER BY signal_year, date_idx, turning_scale, causal_next_type
        """
    ).fetchdf()
    loss_columns = list(aliases.values())
    dates = (
        cells.groupby(["signal_year", "date_idx"], sort=True)[loss_columns]
        .mean()
        .reset_index()
    )
    baseline = dates[aliases[("sequence_raw", 0.0)]]
    hac_lag = int(dict(study["evaluation"])["hac_lag"])
    windows: dict[str, set[int]] = {
        **neighbors._evaluation_scopes(study),
        **{str(year): {year} for year in sorted(dates["signal_year"].unique())},
    }
    rows: list[dict[str, Any]] = []
    for scope, years in windows.items():
        mask = dates["signal_year"].isin(years)
        for (source, weight), alias in aliases.items():
            gain = (baseline[mask] - dates.loc[mask, alias]).to_numpy(np.float64)
            estimate = atlas._hac_mean(gain, lag=hac_lag)
            annual_gain = (
                pd.DataFrame(
                    {
                        "signal_year": dates.loc[mask, "signal_year"],
                        "gain": gain,
                    }
                )
                .groupby("signal_year", sort=True)["gain"]
                .mean()
            )
            rows.append(
                {
                    "scope": scope,
                    "source": source,
                    "weight": weight,
                    "dates": len(gain),
                    "gain_bits_vs_geometry": float(estimate["mean"] / math.log(2.0)),
                    "gain_hac_se_bits": float(estimate["se"] / math.log(2.0)),
                    "gain_lcb_95_bits": float(estimate["lcb_95"] / math.log(2.0)),
                    "gain_ucb_95_bits": float(estimate["ucb_95"] / math.log(2.0)),
                    "positive_years": int(annual_gain.gt(0).sum()),
                }
            )
    return pd.DataFrame(rows)


def _geometry_path_deciles(
    connection: duckdb.DuckDBPyConnection,
    prediction_glob: Path,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    strict_years = next(iter(neighbors._evaluation_scopes(study).values()))
    year_values = ",".join(str(year) for year in sorted(strict_years))
    return connection.execute(
        f"""
        WITH base AS (
            SELECT *, ntile(10) OVER (
                PARTITION BY date_idx, turning_scale, causal_next_type
                ORDER BY geometry_probability
            ) AS probability_decile
            FROM read_parquet(
                {atlas._sql_quote(prediction_glob)}, hive_partitioning=true
            )
            WHERE signal_year IN ({year_values})
              AND extreme_ahead IS NOT NULL
        ), daily AS (
            SELECT
                date_idx,
                turning_scale,
                causal_next_type,
                probability_decile,
                count(*) AS row_count,
                avg(geometry_probability) AS predicted_ahead_rate,
                avg(extreme_ahead::INTEGER) AS actual_ahead_rate,
                avg(CASE WHEN extreme_ahead
                         THEN directional_extreme_log_return ELSE 0 END)
                    AS unconditional_directional_opportunity,
                avg(directional_extreme_log_return)
                    FILTER (WHERE extreme_ahead)
                    AS mean_remaining_directional_return,
                median(directional_extreme_log_return)
                    FILTER (WHERE extreme_ahead)
                    AS median_remaining_directional_return,
                avg(extreme_wait) FILTER (WHERE extreme_ahead)
                    AS mean_days_to_extreme,
                median(extreme_wait) FILTER (WHERE extreme_ahead)
                    AS median_days_to_extreme,
                avg(confirmation_wait) AS mean_days_to_confirmation,
                avg(entry_confirmation_log_return)
                    AS mean_entry_confirmation_return,
                median(entry_confirmation_log_return)
                    AS median_entry_confirmation_return,
                quantile_cont(entry_confirmation_log_return, 0.10)
                    AS p10_entry_confirmation_return,
                quantile_cont(entry_confirmation_log_return, 0.90)
                    AS p90_entry_confirmation_return,
                avg((entry_confirmation_log_return < 0)::INTEGER)
                    AS negative_confirmation_return_rate,
                avg((entry_confirmation_log_return <= -0.05)::INTEGER)
                    AS loss_5pct_confirmation_rate,
                avg((entry_confirmation_log_return >= 0.05)::INTEGER)
                    AS gain_5pct_confirmation_rate
            FROM base
            GROUP BY ALL
        )
        SELECT
            turning_scale,
            causal_next_type,
            probability_decile,
            count(*) AS dates,
            sum(row_count) AS rows,
            avg(predicted_ahead_rate) AS predicted_ahead_rate,
            avg(actual_ahead_rate) AS actual_ahead_rate,
            avg(unconditional_directional_opportunity)
                AS unconditional_directional_opportunity,
            avg(mean_remaining_directional_return)
                AS mean_remaining_directional_return,
            avg(median_remaining_directional_return)
                AS median_remaining_directional_return,
            avg(mean_days_to_extreme) AS mean_days_to_extreme,
            avg(median_days_to_extreme) AS median_days_to_extreme,
            avg(mean_days_to_confirmation) AS mean_days_to_confirmation,
            avg(mean_entry_confirmation_return)
                AS mean_entry_confirmation_return,
            avg(median_entry_confirmation_return)
                AS median_entry_confirmation_return,
            avg(p10_entry_confirmation_return) AS p10_entry_confirmation_return,
            avg(p90_entry_confirmation_return) AS p90_entry_confirmation_return,
            avg(negative_confirmation_return_rate)
                AS negative_confirmation_return_rate,
            avg(loss_5pct_confirmation_rate) AS loss_5pct_confirmation_rate,
            avg(gain_5pct_confirmation_rate) AS gain_5pct_confirmation_rate
        FROM daily
        GROUP BY ALL
        ORDER BY turning_scale, causal_next_type, probability_decile
        """
    ).fetchdf()


def _conditional_setup_quintiles(
    connection: duckdb.DuckDBPyConnection,
    prediction_glob: Path,
    panel_glob: Path,
    study: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strict_years = next(iter(neighbors._evaluation_scopes(study).values()))
    year_values = ",".join(str(year) for year in sorted(strict_years))
    daily = connection.execute(
        f"""
        WITH joined AS (
            SELECT
                p.*,
                x.cumret_3,
                x.cumret_20,
                x.amount_shock_current,
                x.range_shock_current,
                x.attention_score,
                CASE
                    WHEN p.causal_next_type = 'peak'
                         AND x.cumret_20 > 0 AND x.cumret_3 < 0
                    THEN 'uptrend_pullback'
                    WHEN p.causal_next_type = 'trough'
                         AND x.cumret_20 < 0 AND x.cumret_3 > 0
                    THEN 'downtrend_rebound'
                END AS setup,
                CASE WHEN p.causal_next_type = 'peak'
                     THEN p.extreme_ahead::INTEGER
                     ELSE 1 - p.extreme_ahead::INTEGER END AS setup_success,
                CASE WHEN p.causal_next_type = 'peak'
                     THEN p.geometry_probability
                     ELSE 1 - p.geometry_probability END
                    AS setup_success_probability
            FROM read_parquet(
                {atlas._sql_quote(prediction_glob)}, hive_partitioning=true
            ) p
            JOIN read_parquet(
                {atlas._sql_quote(panel_glob)}, hive_partitioning=true
            ) x USING (symbol, date_idx)
            WHERE p.signal_year IN ({year_values})
              AND p.extreme_ahead IS NOT NULL
        ), ranked AS (
            SELECT *, ntile(5) OVER (
                PARTITION BY date_idx, turning_scale, setup
                ORDER BY setup_success_probability
            ) AS score_quintile
            FROM joined
            WHERE setup IS NOT NULL
        )
        SELECT
            signal_year,
            date_idx,
            turning_scale,
            setup,
            score_quintile,
            count(*) AS rows,
            avg(setup_success_probability) AS predicted_success_rate,
            avg(setup_success) AS success_rate,
            avg(entry_confirmation_log_return) AS mean_confirmation_return,
            median(entry_confirmation_log_return)
                AS median_confirmation_return,
            quantile_cont(entry_confirmation_log_return, 0.10)
                AS p10_confirmation_return,
            avg((entry_confirmation_log_return < 0)::INTEGER)
                AS negative_return_rate,
            avg(CASE WHEN extreme_ahead
                     THEN directional_extreme_log_return ELSE 0 END)
                AS directional_opportunity,
            avg(amount_shock_current) AS amount_shock,
            avg(range_shock_current) AS range_shock,
            avg(attention_score) AS attention
        FROM ranked
        GROUP BY ALL
        ORDER BY signal_year, date_idx, turning_scale, setup, score_quintile
        """
    ).fetchdf()
    mean_columns = [
        "predicted_success_rate",
        "success_rate",
        "mean_confirmation_return",
        "median_confirmation_return",
        "p10_confirmation_return",
        "negative_return_rate",
        "directional_opportunity",
        "amount_shock",
        "range_shock",
        "attention",
    ]

    def summarize(group_columns: list[str]) -> pd.DataFrame:
        means = daily.groupby(group_columns, sort=True)[mean_columns].mean()
        counts = daily.groupby(group_columns, sort=True).agg(
            dates=("date_idx", "count"), rows=("rows", "sum")
        )
        return counts.join(means).reset_index()

    summary = summarize(["turning_scale", "setup", "score_quintile"])
    annual = summarize(["signal_year", "turning_scale", "setup", "score_quintile"])
    return summary, annual


def _daily_score_recompute_audit(
    main_daily_path: str | Path, independent: pd.DataFrame
) -> pd.DataFrame:
    keys = ["signal_year", "date_idx", "turning_scale", "causal_next_type"]
    score_columns = [
        f"{model}_{metric}"
        for model in MODEL_COLUMNS
        for metric in ("log_loss", "brier")
    ]
    main = pd.read_csv(main_daily_path, usecols=[*keys, *score_columns])
    merged = main.merge(
        independent[[*keys, *score_columns]],
        on=keys,
        suffixes=("_main", "_independent"),
        validate="one_to_one",
    )
    if len(merged) != len(main) or len(merged) != len(independent):
        raise ValueError("daily_path_neighbors_daily_score_key_mismatch")
    rows: list[dict[str, Any]] = []
    for column in score_columns:
        difference = abs(merged[f"{column}_main"] - merged[f"{column}_independent"])
        rows.append(
            {
                "score": column,
                "cells": len(difference),
                "maximum_absolute_difference": float(difference.max()),
                "mean_absolute_difference": float(difference.mean()),
                "p99_absolute_difference": float(difference.quantile(0.99)),
            }
        )
    return pd.DataFrame(rows)


def _daily_rank_relationship(
    frame: pd.DataFrame,
    *,
    feature: str,
    target: str,
    continuous_target: bool,
    minimum_rows: int = 20,
) -> pd.DataFrame:
    columns = ["signal_year", "date_idx", "direction", feature, target]
    work = frame[columns].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work[target] = pd.to_numeric(work[target], errors="coerce")
    valid = (
        np.isfinite(work[feature].to_numpy(np.float64))
        & np.isfinite(work[target].to_numpy(np.float64))
        & work["direction"].isin((0, 1)).to_numpy()
    )
    work = work.loc[valid]
    if work.empty:
        return pd.DataFrame()
    groups = ["signal_year", "date_idx", "direction"]
    work["x"] = work.groupby(groups, sort=False)[feature].rank(
        method="average", pct=True
    )
    if continuous_target:
        work["y"] = work.groupby(groups, sort=False)[target].rank(
            method="average", pct=True
        )
    else:
        work["y"] = work[target]
    work["x2"] = np.square(work["x"])
    work["y2"] = np.square(work["y"])
    work["xy"] = work["x"] * work["y"]
    grouped = (
        work.groupby(groups, sort=True)
        .agg(
            rows=("x", "size"),
            sum_x=("x", "sum"),
            sum_y=("y", "sum"),
            sum_x2=("x2", "sum"),
            sum_y2=("y2", "sum"),
            sum_xy=("xy", "sum"),
            base_mean=(target, "mean"),
        )
        .reset_index()
    )
    numerator = grouped["rows"] * grouped["sum_xy"] - (
        grouped["sum_x"] * grouped["sum_y"]
    )
    variance_x = grouped["rows"] * grouped["sum_x2"] - np.square(grouped["sum_x"])
    variance_y = grouped["rows"] * grouped["sum_y2"] - np.square(grouped["sum_y"])
    denominator = np.sqrt(np.maximum(variance_x, 0) * np.maximum(variance_y, 0))
    grouped["correlation"] = np.where(denominator > 0, numerator / denominator, np.nan)
    top = (
        work[work["x"] >= 0.8]
        .groupby(groups, sort=True)[target]
        .mean()
        .rename("top_mean")
    )
    bottom = (
        work[work["x"] <= 0.2]
        .groupby(groups, sort=True)[target]
        .mean()
        .rename("bottom_mean")
    )
    grouped = grouped.join(top, on=groups).join(bottom, on=groups)
    grouped["top_minus_bottom"] = grouped["top_mean"] - grouped["bottom_mean"]
    return grouped[grouped["rows"] >= minimum_rows].reset_index(drop=True)


def _phase_feature_daily(
    panel_paths: Mapping[int, Path],
    output_root: Path,
    *,
    reuse_existing: bool,
) -> tuple[Path, list[dict[str, Any]]]:
    daily_root = output_root / "phase_feature_daily"
    records: list[dict[str, Any]] = []
    if reuse_existing:
        expected_paths = {
            year: daily_root / f"year={year}" / "part-0000.parquet"
            for year in panel_paths
        }
        if not all(path.exists() for path in expected_paths.values()):
            raise ValueError("daily_path_neighbors_phase_daily_reuse_incomplete")
        identity = pd.read_parquet(
            daily_root,
            columns=["signal_year", "turning_scale", "outcome", "feature"],
        )
        if set(identity["signal_year"].unique()) != set(panel_paths):
            raise ValueError("daily_path_neighbors_phase_daily_year_mismatch")
        if set(identity["turning_scale"].unique()) != set(neighbors.TURNING_SCALES):
            raise ValueError("daily_path_neighbors_phase_daily_scale_mismatch")
        if set(identity["feature"].unique()) != set(PHASE_FEATURES):
            raise ValueError("daily_path_neighbors_phase_daily_feature_mismatch")
        if set(identity["outcome"].unique()) != {
            "extreme_ahead_phase",
            "remaining_directional_return",
        }:
            raise ValueError("daily_path_neighbors_phase_daily_outcome_mismatch")
        for year, path in sorted(expected_paths.items()):
            records.append(
                {
                    "year": year,
                    **atlas._file_record(path, include_hash=False),
                    "rows": int(pq.ParquetFile(path).metadata.num_rows),
                }
            )
        return daily_root, records
    label_columns = tuple(
        column
        for scale in neighbors.TURNING_SCALES
        for column in (
            f"causal_next_peak_{scale}",
            f"extreme_ahead_{scale}",
            f"directional_extreme_log_return_{scale}",
        )
    )
    columns = tuple(
        dict.fromkeys(("signal_year", "date_idx", *PHASE_FEATURES, *label_columns))
    )
    for year, path in sorted(panel_paths.items()):
        frame = pd.read_parquet(path, columns=list(columns))
        year_parts: list[pd.DataFrame] = []
        for scale in neighbors.TURNING_SCALES:
            direction = frame[f"causal_next_peak_{scale}"].to_numpy(np.float64)
            ahead = frame[f"extreme_ahead_{scale}"].astype("boolean")
            relation = frame[["signal_year", "date_idx", *PHASE_FEATURES]].copy()
            relation["direction"] = direction
            relation["phase_target"] = ahead.fillna(False).to_numpy(np.int8)
            known = ahead.notna().to_numpy() & np.isfinite(direction)
            phase_relation = relation.loc[known]
            return_relation = relation.loc[
                known & ahead.fillna(False).to_numpy(bool)
            ].copy()
            return_relation["remaining_return"] = frame.loc[
                return_relation.index,
                f"directional_extreme_log_return_{scale}",
            ].to_numpy(np.float64)
            for feature in PHASE_FEATURES:
                phase_daily = _daily_rank_relationship(
                    phase_relation,
                    feature=feature,
                    target="phase_target",
                    continuous_target=False,
                )
                if not phase_daily.empty:
                    phase_daily["outcome"] = "extreme_ahead_phase"
                    phase_daily["feature"] = feature
                    phase_daily["turning_scale"] = scale
                    year_parts.append(phase_daily)
                return_daily = _daily_rank_relationship(
                    return_relation,
                    feature=feature,
                    target="remaining_return",
                    continuous_target=True,
                )
                if not return_daily.empty:
                    return_daily["outcome"] = "remaining_directional_return"
                    return_daily["feature"] = feature
                    return_daily["turning_scale"] = scale
                    year_parts.append(return_daily)
        year_output = pd.concat(year_parts, ignore_index=True)
        year_output["causal_next_type"] = np.where(
            year_output["direction"].eq(1), "peak", "trough"
        )
        keep = [
            "signal_year",
            "date_idx",
            "turning_scale",
            "causal_next_type",
            "outcome",
            "feature",
            "rows",
            "correlation",
            "top_minus_bottom",
            "base_mean",
        ]
        year_output = year_output[keep].sort_values(
            [
                "date_idx",
                "turning_scale",
                "causal_next_type",
                "outcome",
                "feature",
            ],
            kind="mergesort",
        )
        output_path = daily_root / f"year={year}" / "part-0000.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        year_output.to_parquet(output_path, index=False, compression="zstd")
        records.append(
            {
                "year": year,
                **atlas._file_record(output_path, include_hash=False),
                "rows": len(year_output),
            }
        )
        del frame, year_output
    return daily_root, records


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    result = np.full(len(values), np.nan, dtype=np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        return result
    positions = np.flatnonzero(valid)
    order = positions[np.argsort(values[valid], kind="mergesort")]
    ranks = np.arange(1, len(order) + 1, dtype=np.float64)
    adjusted = values[order] * len(order) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _phase_feature_summary(
    daily_root: Path, *, hac_lag: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_parquet(daily_root)
    group_columns = [
        "turning_scale",
        "causal_next_type",
        "outcome",
        "feature",
    ]
    annual = (
        daily.groupby(["signal_year", *group_columns], sort=True)
        .agg(
            dates=("correlation", "count"),
            mean_daily_correlation=("correlation", "mean"),
            mean_top_minus_bottom=("top_minus_bottom", "mean"),
            mean_base=("base_mean", "mean"),
        )
        .reset_index()
    )
    summary_rows: list[dict[str, Any]] = []
    for period, years in PHASE_PERIODS.items():
        scoped = daily[daily["signal_year"].isin(years)]
        for keys, group in scoped.groupby(group_columns, sort=True):
            scale, next_type, outcome, feature = keys
            ordered = group.sort_values("date_idx", kind="mergesort")
            correlation = ordered["correlation"].dropna().to_numpy(np.float64)
            top_bottom = ordered["top_minus_bottom"].dropna().to_numpy(np.float64)
            if len(correlation) == 0:
                continue
            corr_estimate = atlas._hac_mean(correlation, lag=hac_lag)
            top_estimate = (
                atlas._hac_mean(top_bottom, lag=hac_lag)
                if len(top_bottom) > 0
                else {"mean": math.nan, "se": math.nan}
            )
            annual_group = annual[
                annual["signal_year"].isin(years)
                & annual["turning_scale"].eq(scale)
                & annual["causal_next_type"].eq(next_type)
                & annual["outcome"].eq(outcome)
                & annual["feature"].eq(feature)
            ]
            se = float(corr_estimate["se"])
            z_score = (
                float(corr_estimate["mean"]) / se
                if se > 0
                else math.copysign(math.inf, float(corr_estimate["mean"]))
                if float(corr_estimate["mean"]) != 0
                else 0.0
            )
            mean_correlation = float(corr_estimate["mean"])
            summary_rows.append(
                {
                    "period": period,
                    "turning_scale": int(scale),
                    "causal_next_type": next_type,
                    "outcome": outcome,
                    "feature": feature,
                    "dates": len(correlation),
                    "mean_daily_correlation": mean_correlation,
                    "correlation_hac_se": se,
                    "correlation_lcb_95": float(corr_estimate["lcb_95"]),
                    "correlation_ucb_95": float(corr_estimate["ucb_95"]),
                    "correlation_p_value": float(2.0 * norm.sf(abs(z_score))),
                    "mean_top_minus_bottom": float(top_estimate["mean"]),
                    "top_minus_bottom_hac_se": float(top_estimate["se"]),
                    "mean_base": float(ordered["base_mean"].mean()),
                    "years": int(annual_group["signal_year"].nunique()),
                    "same_sign_years": int(
                        (
                            annual_group["mean_daily_correlation"] * mean_correlation
                            > 0
                        ).sum()
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary["correlation_bh_q"] = np.nan
    for positions in summary.groupby(["period", "outcome"]).groups.values():
        index = np.asarray(list(positions), dtype=np.int64)
        summary.loc[index, "correlation_bh_q"] = _bh_adjust(
            summary.loc[index, "correlation_p_value"].to_numpy(np.float64)
        )
    discovery = summary[summary["period"].eq("discovery_2012_2018")].copy()
    strict = summary[summary["period"].eq("strict_oos_2019_2023")].copy()
    extension = summary[summary["period"].eq("near_complete_2024")].copy()
    keys = group_columns
    replication = discovery.merge(
        strict,
        on=keys,
        suffixes=("_discovery", "_strict"),
        validate="one_to_one",
    )
    extension_columns = [
        *keys,
        "mean_daily_correlation",
        "correlation_bh_q",
    ]
    replication = replication.merge(
        extension[extension_columns],
        on=keys,
        how="left",
        validate="one_to_one",
    ).rename(
        columns={
            "mean_daily_correlation": "mean_daily_correlation_2024",
            "correlation_bh_q": "correlation_bh_q_2024",
        }
    )
    replication["same_sign_discovery_strict"] = (
        replication["mean_daily_correlation_discovery"]
        * replication["mean_daily_correlation_strict"]
        > 0
    )
    replication["same_sign_strict_2024"] = (
        replication["mean_daily_correlation_strict"]
        * replication["mean_daily_correlation_2024"]
        > 0
    )
    replication["replicated_strict"] = (
        replication["same_sign_discovery_strict"]
        & replication["correlation_bh_q_discovery"].le(0.05)
        & replication["correlation_bh_q_strict"].le(0.05)
        & replication["same_sign_years_strict"].ge(4)
    )
    return summary, annual, replication


def run_validation(
    *,
    analysis_root: str | Path = DEFAULT_ANALYSIS_ROOT,
    reuse_phase_daily: bool = False,
) -> dict[str, Any]:
    analysis_root = atlas._resolve_path(analysis_root)
    manifest_path = analysis_root / "analysis_manifest.json"
    manifest = atlas._read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError("daily_path_neighbors_analysis_not_completed")
    study_path = Path(str(dict(manifest["study"])["path"]))
    study = neighbors.load_study(study_path)
    panel_manifest = atlas._read_json(analysis_root / "panel_manifest.json")
    panel_paths = neighbors._panel_paths(panel_manifest)
    output_root = analysis_root / "validation"
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_glob = _prediction_glob(analysis_root)
    panel_glob = analysis_root / "daily_path_panels" / "year=*" / "part-0000.parquet"

    connection = atlas._connect(output_root, study)
    try:
        daily_scores = _independent_daily_scores(connection, prediction_glob)
        information = _independent_information(daily_scores, study)
        auc_by_cell, auc_summary = _auc_calibration_metrics(
            connection, prediction_glob, study
        )
        blend_controls = _blend_increment_controls(connection, prediction_glob, study)
        geometry_path_deciles = _geometry_path_deciles(
            connection, prediction_glob, study
        )
        conditional_setups, conditional_setups_annual = _conditional_setup_quintiles(
            connection, prediction_glob, panel_glob, study
        )
        prediction_audit, censoring = _prediction_audit(connection, prediction_glob)
        runtime = {
            **atlas._duckdb_runtime_resources(study),
            "active_memory_limit": str(
                connection.execute("SELECT current_setting('memory_limit')").fetchone()[
                    0
                ]
            ),
            "active_threads": int(
                connection.execute("SELECT current_setting('threads')").fetchone()[0]
            ),
        }
    finally:
        connection.close()

    main_headline = pd.read_csv(dict(manifest["outputs"])["headline_information"])
    comparison = information.merge(
        main_headline[["scope", "comparison", "information_gain_bits"]].rename(
            columns={"information_gain_bits": "main_information_gain_bits"}
        ),
        on=["scope", "comparison"],
        how="left",
        validate="one_to_one",
    )
    comparison["absolute_difference_bits"] = abs(
        comparison["information_gain_bits"] - comparison["main_information_gain_bits"]
    )
    daily_score_audit = _daily_score_recompute_audit(
        dict(manifest["outputs"])["daily_metrics"], daily_scores
    )

    daily_root, phase_records = _phase_feature_daily(
        panel_paths,
        output_root,
        reuse_existing=reuse_phase_daily,
    )
    phase_summary, phase_annual, phase_replication = _phase_feature_summary(
        daily_root, hac_lag=int(dict(study["evaluation"])["hac_lag"])
    )
    outputs = {
        "independent_daily_scores": output_root / "independent_daily_scores.parquet",
        "information_comparison": output_root / "information_comparison.csv",
        "daily_score_recompute_audit": output_root / "daily_score_recompute_audit.csv",
        "auc_calibration_by_cell": output_root / "auc_calibration_by_cell.csv",
        "auc_calibration_summary": output_root / "auc_calibration_summary.csv",
        "blend_increment_controls": output_root / "blend_increment_controls.csv",
        "geometry_path_deciles": output_root / "geometry_probability_path_deciles.csv",
        "conditional_setup_quintiles": output_root
        / "conditional_pullback_rebound_quintiles.csv",
        "conditional_setup_annual": output_root
        / "conditional_pullback_rebound_annual.csv",
        "censoring_by_year_scale": output_root / "censoring_by_year_scale.csv",
        "phase_feature_summary": output_root / "phase_feature_summary.csv",
        "phase_feature_annual": output_root / "phase_feature_annual.csv",
        "phase_feature_replication": output_root / "phase_feature_replication.csv",
    }
    daily_scores.to_parquet(
        outputs["independent_daily_scores"], index=False, compression="zstd"
    )
    comparison.to_csv(outputs["information_comparison"], index=False)
    daily_score_audit.to_csv(outputs["daily_score_recompute_audit"], index=False)
    auc_by_cell.to_csv(outputs["auc_calibration_by_cell"], index=False)
    auc_summary.to_csv(outputs["auc_calibration_summary"], index=False)
    blend_controls.to_csv(outputs["blend_increment_controls"], index=False)
    geometry_path_deciles.to_csv(outputs["geometry_path_deciles"], index=False)
    conditional_setups.to_csv(outputs["conditional_setup_quintiles"], index=False)
    conditional_setups_annual.to_csv(outputs["conditional_setup_annual"], index=False)
    censoring.to_csv(outputs["censoring_by_year_scale"], index=False)
    phase_summary.to_csv(outputs["phase_feature_summary"], index=False)
    phase_annual.to_csv(outputs["phase_feature_annual"], index=False)
    phase_replication.to_csv(outputs["phase_feature_replication"], index=False)

    blocking_keys = (
        "duplicate_keys",
        "forbidden_rows",
        "nonfinite_probability_rows",
        "causal_update_violations",
    )
    maximum_nested_difference = max(
        value for key, value in prediction_audit.items() if "difference" in key
    )
    strict_scope = next(iter(neighbors._evaluation_scopes(study)))
    sequence_tenth = blend_controls[
        blend_controls["scope"].eq(strict_scope)
        & blend_controls["source"].eq("sequence_raw")
        & blend_controls["weight"].eq(0.1)
    ].iloc[0]
    audit = {
        **prediction_audit,
        "maximum_information_recompute_difference_bits": float(
            comparison["absolute_difference_bits"].max()
        ),
        "maximum_daily_score_recompute_difference": float(
            daily_score_audit["maximum_absolute_difference"].max()
        ),
        "maximum_nested_geometry_difference": float(maximum_nested_difference),
        "strict_censored_rows": int(
            censoring[censoring["signal_year"].isin(range(2019, 2024))][
                "censored_rows"
            ].sum()
        ),
        "replicated_phase_relationships": int(
            phase_replication["replicated_strict"].sum()
        ),
        "diagnostic_sequence_weight_0_1_strict_gain_bits": float(
            sequence_tenth["gain_bits_vs_geometry"]
        ),
        "diagnostic_sequence_weight_0_1_strict_lcb_95_bits": float(
            sequence_tenth["gain_lcb_95_bits"]
        ),
    }
    if any(audit[key] != 0 for key in blocking_keys):
        raise ValueError(f"daily_path_neighbors_validation_audit_failed:{audit}")
    if audit["maximum_information_recompute_difference_bits"] > 1e-9:
        raise ValueError("daily_path_neighbors_information_recompute_mismatch")
    if audit["maximum_daily_score_recompute_difference"] > 1e-6:
        raise ValueError("daily_path_neighbors_daily_score_recompute_mismatch")
    if audit["maximum_nested_geometry_difference"] != 0.0:
        raise ValueError("daily_path_neighbors_zero_weight_nesting_mismatch")
    if audit["strict_censored_rows"] != 0:
        raise ValueError("daily_path_neighbors_strict_scope_is_censored")

    validation = {
        "schema": VALIDATION_SCHEMA,
        "status": "completed",
        "study_id": neighbors.STUDY_ID,
        "analysis_manifest": atlas._file_record(manifest_path),
        "runtime": runtime,
        "audit": audit,
        "outputs": {
            **{key: str(path.resolve()) for key, path in outputs.items()},
            "phase_feature_daily": phase_records,
        },
        "phase_feature_periods": {
            key: sorted(value) for key, value in PHASE_PERIODS.items()
        },
        "phase_features": list(PHASE_FEATURES),
        "formal_sequence_incremental_weight": float(
            dict(manifest["selected_configurations"])["sequence"]["incremental_weight"]
        ),
        "blend_increment_is_diagnostic_only": True,
        "conditional_pullback_rebound_is_exploratory": True,
        "fixed_holding_horizon_used": False,
        "execution_backtest_performed": False,
        "profit_claim_allowed": False,
        "causal_effect_claim_allowed": False,
        "report_generation_performed": False,
    }
    validation_path = output_root / "validation_manifest.json"
    atlas._write_json(validation_path, validation)
    return validation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently validate daily path-neighbor outputs."
    )
    parser.add_argument("--analysis-root", default=str(DEFAULT_ANALYSIS_ROOT))
    parser.add_argument(
        "--reuse-phase-daily",
        action="store_true",
        help="Reuse complete phase-feature daily partitions after validating identity.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_validation(
        analysis_root=args.analysis_root,
        reuse_phase_daily=args.reuse_phase_daily,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
