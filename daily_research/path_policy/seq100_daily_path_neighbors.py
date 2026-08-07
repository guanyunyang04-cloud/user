"""Causal daily path prototypes and natural-time turning outcomes.

The study keeps every point-in-time quality-liquidity stock day.  It observes
only the path available at the signal close, maps that state to the next
confirmed multi-scale directional-change event, and updates prototype outcome
statistics only after the event confirmation date has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from daily_research.path_policy import seq100_hot_path_atlas as atlas

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_daily_path_neighbors_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_daily_path_neighbors_v1"
)
STUDY_ID = "seq100_daily_path_neighbors_v1"
PANEL_SCHEMA = "seq100_daily_path_neighbor_panel/1"
ANALYSIS_SCHEMA = "seq100_daily_path_neighbor_analysis/1"
BUILDER_VERSION = 1

TURNING_SCALES = (4, 8, 16)
RETURN_FEATURES = tuple(f"ret_lag_{index}" for index in range(20))
AMOUNT_FEATURES = tuple(f"amount_shock_lag_{index}" for index in range(10))
RANGE_FEATURES = tuple(f"range_shock_lag_{index}" for index in range(10))
LOCATION_FEATURES = tuple(f"close_location_lag_{index}" for index in range(5))
GAP_FEATURES = tuple(f"observed_gap_lag_{index}" for index in range(5))
PATH_FEATURES = (
    *RETURN_FEATURES,
    *AMOUNT_FEATURES,
    *RANGE_FEATURES,
    *LOCATION_FEATURES,
    *GAP_FEATURES,
)
CORE_GEOMETRY_FEATURES = (
    "cumret_1",
    "cumret_3",
    "cumret_5",
    "cumret_10",
    "cumret_20",
    "cumret_60",
    "volatility_5",
    "volatility_20",
    "distance_close_high20",
    "distance_close_high60",
    "distance_close_low20",
    "trend_efficiency20",
    "amount_shock_current",
    "range_shock_current",
    "close_location_current",
    "market_gap20",
)
AUXILIARY_FEATURES = (
    "attention_score",
    "attention_velocity",
    "market_relative_ret_1d",
    "industry_relative_ret_1d",
    "pullback_from_high5",
    "distance_ma20",
    "distance_ma60",
    "volatility_ratio5_20",
    "log_volume_ratio20",
    "legacy_state_missing",
)
SCALE_GEOMETRY_FEATURES = tuple(
    feature
    for scale in TURNING_SCALES
    for feature in (f"causal_next_peak_{scale}", f"log_confirmation_age_{scale}")
)
GEOMETRY_FEATURES = (
    *CORE_GEOMETRY_FEATURES,
    *AUXILIARY_FEATURES,
    *SCALE_GEOMETRY_FEATURES,
)
FEATURE_SETS = {
    "geometry": GEOMETRY_FEATURES,
    "sequence": (*GEOMETRY_FEATURES, *PATH_FEATURES),
}
CONTINUOUS_METRICS = (
    "log_confirmation_wait",
    "entry_confirmation_log_return",
    "log_extreme_wait_ahead",
    "directional_extreme_log_return_ahead",
)
PREDICTION_MODELS = ("prior", "geometry", "sequence_raw", "sequence")


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study_path = atlas._resolve_path(path)
    study = atlas._read_json(study_path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("daily_path_neighbors_study_id_mismatch")
    if bool(dict(study["outcomes"])["fixed_holding_horizon"]):
        raise ValueError("daily_path_neighbors_fixed_horizon_forbidden")
    if str(dict(study["source"])["quality_pool_name"]) != "quality_liquidity_pit":
        raise ValueError("daily_path_neighbors_quality_pool_mismatch")
    scales = tuple(int(value) for value in dict(study["path_state"])["turning_scales"])
    if scales != TURNING_SCALES:
        raise ValueError("daily_path_neighbors_scale_contract_mismatch")
    period = dict(study["period"])
    strict_years = {int(value) for value in period["strict_complete_evaluation_years"]}
    near_years = {int(value) for value in period["near_complete_evaluation_years"]}
    rolling_years = {int(value) for value in period["rolling_evaluation_years"]}
    if not strict_years or not strict_years <= near_years <= rolling_years:
        raise ValueError("daily_path_neighbors_evaluation_scope_mismatch")
    return study


def _evaluation_scopes(study: Mapping[str, Any]) -> dict[str, set[int]]:
    period = dict(study["period"])
    strict = {int(value) for value in period["strict_complete_evaluation_years"]}
    near = {int(value) for value in period["near_complete_evaluation_years"]}
    rolling = {int(value) for value in period["rolling_evaluation_years"]}
    return {
        f"strict_complete_{min(strict)}_{max(strict)}": strict,
        f"near_complete_{min(near)}_{max(near)}": near,
        f"all_{min(rolling)}_{max(rolling)}_provisional": rolling,
    }


def _year_from_partition_path(path: Path) -> int:
    for part in path.parts:
        if part.startswith("year="):
            return int(part.split("=", maxsplit=1)[1])
    raise ValueError(f"daily_path_neighbors_year_missing:{path}")


def _source_contract(
    study_path: Path, study: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[int, Path], dict[int, Path], Path, Path, str]:
    source = dict(study["source"])
    quality_manifest_path = atlas._resolve_path(source["quality_turning_manifest"])
    hot_manifest_path = atlas._resolve_path(source["hot_path_manifest"])
    turning_manifest_path = atlas._resolve_path(source["turning_atlas_manifest"])
    quality_manifest = atlas._read_json(quality_manifest_path)
    hot_manifest = atlas._read_json(hot_manifest_path)
    turning_manifest = atlas._read_json(turning_manifest_path)
    expected = (
        (
            quality_manifest,
            "seq100_turning_path_quality_pool_v1",
            str(source["expected_quality_turning_fingerprint"]),
        ),
        (
            hot_manifest,
            "seq100_hot_path_atlas_v1",
            str(source["expected_hot_path_fingerprint"]),
        ),
        (
            turning_manifest,
            "seq100_turning_path_atlas_v1",
            str(source["expected_turning_atlas_fingerprint"]),
        ),
    )
    for manifest, study_id, fingerprint in expected:
        if manifest.get("study_id") != study_id:
            raise ValueError(f"daily_path_neighbors_source_study_mismatch:{study_id}")
        if str(manifest.get("experiment_fingerprint")) != fingerprint:
            raise ValueError(
                f"daily_path_neighbors_source_fingerprint_mismatch:{study_id}"
            )

    quality_source = dict(quality_manifest["source_contract"])
    quality_paths = {
        _year_from_partition_path(Path(str(record["path"]))): Path(str(record["path"]))
        for record in quality_source["quality_spines"]
    }
    state_paths = {
        int(record["year"]): Path(str(record["path"]))
        for record in hot_manifest["panels"]
    }
    formal_years = tuple(int(value) for value in dict(study["period"])["formal_years"])
    if tuple(sorted(quality_paths)) != formal_years:
        raise ValueError("daily_path_neighbors_quality_years_mismatch")
    if tuple(sorted(state_paths)) != formal_years:
        raise ValueError("daily_path_neighbors_state_years_mismatch")
    dense_path = Path(str(quality_source["dense_base"]["path"]))
    event_path = Path(str(quality_source["turning_events"]["path"]))
    for path in (
        *quality_paths.values(),
        *state_paths.values(),
        dense_path,
        event_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    payload = {
        "builder_version": BUILDER_VERSION,
        "study_sha256": atlas._sha256_file(study_path),
        "quality_manifest_sha256": atlas._sha256_file(quality_manifest_path),
        "hot_manifest_sha256": atlas._sha256_file(hot_manifest_path),
        "turning_manifest_sha256": atlas._sha256_file(turning_manifest_path),
        "quality_fingerprint": quality_manifest["experiment_fingerprint"],
        "hot_fingerprint": hot_manifest["experiment_fingerprint"],
        "turning_fingerprint": turning_manifest["experiment_fingerprint"],
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    contract = {
        "quality_turning_manifest": atlas._file_record(quality_manifest_path),
        "hot_path_manifest": atlas._file_record(hot_manifest_path),
        "turning_atlas_manifest": atlas._file_record(turning_manifest_path),
        "quality_pool_rows": int(quality_source["quality_pool_rows"]),
        "quality_spines": [
            atlas._file_record(quality_paths[year], include_hash=False)
            for year in formal_years
        ],
        "state_panels": [
            atlas._file_record(state_paths[year], include_hash=False)
            for year in formal_years
        ],
        "dense_base": atlas._file_record(dense_path, include_hash=False),
        "turning_events": atlas._file_record(event_path, include_hash=False),
        "fingerprint_payload": payload,
    }
    return contract, quality_paths, state_paths, dense_path, event_path, fingerprint


def _event_ctes(event_path: Path) -> str:
    quoted = atlas._sql_quote(event_path)
    return ",\n".join(
        f"e{scale} AS (SELECT * FROM read_parquet({quoted}) "
        f"WHERE threshold_multiplier = {scale})"
        for scale in TURNING_SCALES
    )


def _event_joins() -> str:
    joins: list[str] = []
    for scale in TURNING_SCALES:
        joins.extend(
            [
                (
                    f"ASOF LEFT JOIN e{scale} f{scale} ON q.symbol = f{scale}.symbol "
                    f"AND q.date_idx < f{scale}.confirmation_date_idx"
                ),
                (
                    f"ASOF LEFT JOIN e{scale} p{scale} ON q.symbol = p{scale}.symbol "
                    f"AND q.date_idx >= p{scale}.confirmation_date_idx"
                ),
            ]
        )
    return "\n".join(joins)


def _event_columns(scale: int) -> list[str]:
    return [
        f"(f{scale}.event_order IS NULL) AS right_censored_{scale}",
        f"(f{scale}.event_type = 'peak') AS next_peak_{scale}",
        (
            f"CASE WHEN p{scale}.event_type = 'trough' THEN 1.0 "
            f"WHEN p{scale}.event_type = 'peak' THEN 0.0 ELSE NULL END::REAL "
            f"AS causal_next_peak_{scale}"
        ),
        (
            f"ln(1.0 + greatest(q.date_idx - p{scale}.confirmation_date_idx, 0))::REAL "
            f"AS log_confirmation_age_{scale}"
        ),
        f"f{scale}.extreme_date_idx AS extreme_date_idx_{scale}",
        f"f{scale}.confirmation_date_idx AS resolution_date_idx_{scale}",
        (
            f"CASE WHEN f{scale}.event_order IS NOT NULL "
            f"THEN f{scale}.extreme_date_idx > q.date_idx ELSE NULL END "
            f"AS extreme_ahead_{scale}"
        ),
        (f"(f{scale}.extreme_date_idx - q.date_idx)::INTEGER AS extreme_wait_{scale}"),
        (
            f"(f{scale}.confirmation_date_idx - q.date_idx)::INTEGER "
            f"AS confirmation_wait_{scale}"
        ),
        (
            f"CASE WHEN s.entry_open > 0 AND f{scale}.confirmation_log_price IS NOT NULL "
            f"THEN f{scale}.confirmation_log_price - ln(s.entry_open) END::REAL "
            f"AS entry_confirmation_log_return_{scale}"
        ),
        (
            f"CASE WHEN s.entry_open > 0 AND f{scale}.extreme_date_idx >= s.entry_date_idx "
            f"THEN f{scale}.extreme_log_price - ln(s.entry_open) END::REAL "
            f"AS entry_extreme_log_return_{scale}"
        ),
        (
            f"CASE WHEN s.entry_open > 0 AND f{scale}.extreme_date_idx >= s.entry_date_idx "
            f"AND f{scale}.event_type = 'peak' "
            f"THEN f{scale}.extreme_log_price - ln(s.entry_open) "
            f"WHEN s.entry_open > 0 AND f{scale}.extreme_date_idx >= s.entry_date_idx "
            f"AND f{scale}.event_type = 'trough' "
            f"THEN ln(s.entry_open) - f{scale}.extreme_log_price END::REAL "
            f"AS directional_extreme_log_return_{scale}"
        ),
    ]


def _panel_query(
    *,
    year: int,
    quality_path: Path,
    state_path: Path,
    dense_path: Path,
    event_path: Path,
) -> str:
    input_start = f"{year - 2}-01-01"
    input_end = f"{min(year + 1, 2025)}-12-31"
    ret_lags = ",\n            ".join(
        f"lag(ret1, {index}) OVER w2::REAL AS ret_lag_{index}" for index in range(20)
    )
    amount_lags = ",\n            ".join(
        f"lag(amount_shock, {index}) OVER w2::REAL AS amount_shock_lag_{index}"
        for index in range(10)
    )
    range_lags = ",\n            ".join(
        f"lag(range_shock, {index}) OVER w2::REAL AS range_shock_lag_{index}"
        for index in range(10)
    )
    location_lags = ",\n            ".join(
        f"lag(close_location, {index}) OVER w2::REAL AS close_location_lag_{index}"
        for index in range(5)
    )
    gap_lags = ",\n            ".join(
        f"lag(observed_gap, {index}) OVER w2::REAL AS observed_gap_lag_{index}"
        for index in range(5)
    )
    outcome_columns = ",\n        ".join(
        column for scale in TURNING_SCALES for column in _event_columns(scale)
    )
    quality = atlas._sql_quote(quality_path)
    state = atlas._sql_quote(state_path)
    dense = atlas._sql_quote(dense_path)
    return f"""
    WITH raw AS (
        SELECT
            symbol,
            trade_date,
            date_idx,
            ln(adj_close) AS log_close,
            CASE WHEN amount > 0 THEN ln(amount) END AS log_amount,
            CASE WHEN adj_high > 0 AND adj_low > 0
                 THEN ln(adj_high / adj_low) END AS log_range,
            CASE WHEN adj_high > adj_low
                 THEN (adj_close - adj_low) / (adj_high - adj_low)
                 ELSE 0.5 END AS close_location,
            date_idx - lag(date_idx) OVER w AS observed_gap,
            lead(date_idx) OVER w AS entry_date_idx,
            lead(adj_open) OVER w AS entry_open
        FROM read_parquet({dense})
        WHERE bar_valid
          AND trade_date BETWEEN '{input_start}' AND '{input_end}'
        WINDOW w AS (PARTITION BY symbol ORDER BY date_idx)
    ), step AS (
        SELECT
            *,
            log_close - lag(log_close) OVER w AS ret1,
            log_amount - avg(log_amount) OVER (
                PARTITION BY symbol ORDER BY date_idx
                ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
            ) AS amount_shock,
            log_range - avg(log_range) OVER (
                PARTITION BY symbol ORDER BY date_idx
                ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
            ) AS range_shock,
            max(log_close) OVER (
                PARTITION BY symbol ORDER BY date_idx
                ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
            ) AS past_high20,
            max(log_close) OVER (
                PARTITION BY symbol ORDER BY date_idx
                ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING
            ) AS past_high60,
            min(log_close) OVER (
                PARTITION BY symbol ORDER BY date_idx
                ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
            ) AS past_low20,
            log_close - lag(log_close, 3) OVER w AS cumret3,
            log_close - lag(log_close, 5) OVER w AS cumret5,
            log_close - lag(log_close, 10) OVER w AS cumret10,
            log_close - lag(log_close, 20) OVER w AS cumret20,
            log_close - lag(log_close, 60) OVER w AS cumret60,
            date_idx - lag(date_idx, 20) OVER w AS gap20
        FROM raw
        WINDOW w AS (PARTITION BY symbol ORDER BY date_idx)
    ), state_calc AS (
        SELECT
            *,
            stddev_samp(ret1) OVER (
                PARTITION BY symbol ORDER BY date_idx
                ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
            ) AS volatility5,
            stddev_samp(ret1) OVER (
                PARTITION BY symbol ORDER BY date_idx
                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            ) AS volatility20,
            sum(abs(ret1)) OVER (
                PARTITION BY symbol ORDER BY date_idx
                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            ) AS path_length20
        FROM step
    ), sequence AS (
        SELECT
            *,
            {ret_lags},
            {amount_lags},
            {range_lags},
            {location_lags},
            {gap_lags}
        FROM state_calc
        WINDOW w2 AS (PARTITION BY symbol ORDER BY date_idx)
    ), {_event_ctes(event_path)}
    SELECT
        q.candidate_id,
        q.security_id,
        q.symbol,
        s.trade_date,
        q.date_idx,
        {year}::SMALLINT AS signal_year,
        s.entry_date_idx,
        s.entry_open::REAL AS entry_open,
        (aux.symbol IS NOT NULL) AS legacy_state_present,
        s.ret1::REAL AS cumret_1,
        s.cumret3::REAL AS cumret_3,
        s.cumret5::REAL AS cumret_5,
        s.cumret10::REAL AS cumret_10,
        s.cumret20::REAL AS cumret_20,
        s.cumret60::REAL AS cumret_60,
        s.volatility5::REAL AS volatility_5,
        s.volatility20::REAL AS volatility_20,
        (s.log_close - s.past_high20)::REAL AS distance_close_high20,
        (s.log_close - s.past_high60)::REAL AS distance_close_high60,
        (s.log_close - s.past_low20)::REAL AS distance_close_low20,
        (abs(s.cumret20) / NULLIF(s.path_length20, 0))::REAL AS trend_efficiency20,
        s.amount_shock::REAL AS amount_shock_current,
        s.range_shock::REAL AS range_shock_current,
        s.close_location::REAL AS close_location_current,
        s.gap20::REAL AS market_gap20,
        aux.attention_score::REAL AS attention_score,
        aux.attention_velocity::REAL AS attention_velocity,
        aux.market_relative_ret_1d::REAL AS market_relative_ret_1d,
        aux.industry_relative_ret_1d::REAL AS industry_relative_ret_1d,
        aux.pullback_from_high5::REAL AS pullback_from_high5,
        aux.distance_ma20::REAL AS distance_ma20,
        aux.distance_ma60::REAL AS distance_ma60,
        aux.volatility_ratio5_20::REAL AS volatility_ratio5_20,
        aux.log_volume_ratio20::REAL AS log_volume_ratio20,
        (aux.symbol IS NULL)::REAL AS legacy_state_missing,
        {", ".join(f"s.{name}" for name in PATH_FEATURES)},
        {outcome_columns}
    FROM read_parquet({quality}) q
    LEFT JOIN sequence s USING (symbol, date_idx)
    LEFT JOIN read_parquet({state}) aux USING (symbol, date_idx)
    {_event_joins()}
    """


def _panel_audit(
    connection: duckdb.DuckDBPyConnection, panel_path: Path, expected_rows: int
) -> dict[str, Any]:
    scale_check_parts: list[str] = []
    for scale in TURNING_SCALES:
        scale_check_parts.extend(
            [
                (
                    f"count(*) FILTER (WHERE NOT right_censored_{scale} "
                    f"AND resolution_date_idx_{scale} <= date_idx) "
                    f"AS bad_resolution_{scale}"
                ),
                (
                    f"count(*) FILTER (WHERE NOT right_censored_{scale} AND "
                    f"next_peak_{scale} != (causal_next_peak_{scale} > 0.5)) "
                    f"AS direction_mismatch_{scale}"
                ),
            ]
        )
    scale_checks = ",\n".join(scale_check_parts)
    row = (
        connection.execute(
            f"""
            SELECT
                count(*) AS row_count,
                count(*) - count(DISTINCT symbol || '|' || date_idx::VARCHAR)
                    AS duplicate_keys,
                count(*) FILTER (WHERE trade_date IS NULL) AS missing_dense_rows,
                count(*) FILTER (WHERE NOT legacy_state_present)
                    AS missing_auxiliary_rows,
                count(*) FILTER (WHERE left(trade_date, 4) = '2026')
                    AS forbidden_rows,
                count(*) FILTER (WHERE entry_open IS NULL OR entry_open <= 0)
                    AS missing_entry_rows,
                {scale_checks}
            FROM read_parquet({atlas._sql_quote(panel_path)})
            """
        )
        .fetchdf()
        .iloc[0]
        .to_dict()
    )
    audit = {key: int(value) for key, value in row.items()}
    if audit["row_count"] != int(expected_rows):
        raise ValueError("daily_path_neighbors_panel_row_count_mismatch")
    blocking = (
        "duplicate_keys",
        "missing_dense_rows",
        "forbidden_rows",
        *(f"bad_resolution_{scale}" for scale in TURNING_SCALES),
        *(f"direction_mismatch_{scale}" for scale in TURNING_SCALES),
    )
    if any(audit[key] != 0 for key in blocking):
        raise ValueError(f"daily_path_neighbors_panel_audit_failed:{audit}")
    return audit


def build_panels(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study_path = atlas._resolve_path(study_path)
    output_root = atlas._resolve_path(output_root)
    study = load_study(study_path)
    (
        contract,
        quality_paths,
        state_paths,
        dense_path,
        event_path,
        fingerprint,
    ) = _source_contract(study_path, study)
    panel_root = output_root / "daily_path_panels"
    panel_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "panel_progress.json"
    progress: dict[str, Any] = {}
    if progress_path.exists():
        candidate = atlas._read_json(progress_path)
        if candidate.get("experiment_fingerprint") == fingerprint:
            progress = candidate
    panel_manifest_path = output_root / "panel_manifest.json"
    if not progress and panel_manifest_path.exists():
        previous = atlas._read_json(panel_manifest_path)
        previous_source = dict(previous.get("source_contract", {}))
        source_keys = (
            "quality_turning_manifest",
            "hot_path_manifest",
            "turning_atlas_manifest",
        )
        same_sources = all(
            dict(previous_source.get(key, {})).get("sha256")
            == dict(contract[key]).get("sha256")
            for key in source_keys
        )
        if previous.get("schema") == PANEL_SCHEMA and same_sources:
            progress = {"completed_years": previous.get("panels", [])}
    records = {
        int(record["year"]): record for record in progress.get("completed_years", [])
    }
    connection = atlas._connect(output_root, study)
    runtime = {
        **atlas._duckdb_runtime_resources(study),
        "active_memory_limit": str(
            connection.execute("SELECT current_setting('memory_limit')").fetchone()[0]
        ),
        "active_threads": int(
            connection.execute("SELECT current_setting('threads')").fetchone()[0]
        ),
    }
    try:
        for year in (int(value) for value in dict(study["period"])["formal_years"]):
            panel_path = panel_root / f"year={year}" / "part-0000.parquet"
            existing = records.get(year)
            if (
                existing is not None
                and panel_path.exists()
                and int(existing.get("size", -1)) == panel_path.stat().st_size
            ):
                continue
            panel_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = panel_path.with_suffix(".parquet.tmp")
            temporary.unlink(missing_ok=True)
            expected_rows = int(
                connection.execute(
                    f"SELECT count(*) FROM read_parquet({atlas._sql_quote(quality_paths[year])})"
                ).fetchone()[0]
            )
            query = _panel_query(
                year=year,
                quality_path=quality_paths[year],
                state_path=state_paths[year],
                dense_path=dense_path,
                event_path=event_path,
            )
            row_group_size = int(dict(study["resources"])["parquet_row_group_size"])
            connection.execute(
                f"COPY ({query}) TO {atlas._sql_quote(temporary)} "
                f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {row_group_size})"
            )
            temporary.replace(panel_path)
            audit = _panel_audit(connection, panel_path, expected_rows)
            records[year] = {
                "year": year,
                **atlas._file_record(panel_path, include_hash=False),
                "audit": audit,
            }
            atlas._write_json(
                progress_path,
                {
                    "status": "building_panels",
                    "experiment_fingerprint": fingerprint,
                    "completed_years": [records[key] for key in sorted(records)],
                },
            )
    finally:
        connection.close()

    ordered = [records[year] for year in sorted(records)]
    aggregate = {
        "rows": sum(int(record["audit"]["row_count"]) for record in ordered),
        "duplicate_keys": sum(
            int(record["audit"]["duplicate_keys"]) for record in ordered
        ),
        "missing_dense_rows": sum(
            int(record["audit"]["missing_dense_rows"]) for record in ordered
        ),
        "missing_auxiliary_rows": sum(
            int(record["audit"]["missing_auxiliary_rows"]) for record in ordered
        ),
        "missing_entry_rows": sum(
            int(record["audit"]["missing_entry_rows"]) for record in ordered
        ),
        "forbidden_rows": sum(
            int(record["audit"]["forbidden_rows"]) for record in ordered
        ),
    }
    if aggregate["rows"] != int(contract["quality_pool_rows"]):
        raise ValueError("daily_path_neighbors_total_quality_rows_mismatch")
    manifest = {
        "schema": PANEL_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "builder_version": BUILDER_VERSION,
        "experiment_fingerprint": fingerprint,
        "study": atlas._file_record(study_path),
        "source_contract": contract,
        "runtime": runtime,
        "panels": ordered,
        "audit": aggregate,
        "training_performed": False,
        "fixed_holding_horizon_used": False,
        "report_generation_performed": False,
    }
    manifest_path = panel_manifest_path
    atlas._write_json(manifest_path, manifest)
    atlas._write_json(
        progress_path,
        {
            "status": "completed",
            "experiment_fingerprint": fingerprint,
            "panel_manifest": str(manifest_path.resolve()),
            "completed_years": ordered,
        },
    )
    return manifest


@dataclass
class Preprocessor:
    feature_names: tuple[str, ...]
    medians: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    pca: PCA

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, self.feature_names].to_numpy(np.float64, copy=True)
        values[~np.isfinite(values)] = np.nan
        missing = np.isnan(values)
        if missing.any():
            values[missing] = np.take(self.medians, np.nonzero(missing)[1])
        values = np.clip(values, self.lower_bounds, self.upper_bounds)
        values = (values - self.means) / self.scales
        transformed = self.pca.transform(values).astype(np.float32)
        if not np.isfinite(transformed).all():
            raise ValueError("daily_path_neighbors_nonfinite_embedding")
        return transformed


@dataclass
class PrototypeModel:
    name: str
    preprocessor: Preprocessor
    clusters: MiniBatchKMeans

    def assign(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        embedding = self.preprocessor.transform(frame)
        labels = self.clusters.predict(embedding).astype(np.int32)
        centers = self.clusters.cluster_centers_[labels]
        distance = np.sqrt(np.sum(np.square(embedding - centers), axis=1)).astype(
            np.float32
        )
        return labels, distance


def _panel_paths(panel_manifest: Mapping[str, Any]) -> dict[int, Path]:
    return {
        int(record["year"]): Path(str(record["path"]))
        for record in panel_manifest["panels"]
    }


def _sample_features(
    panel_paths: Mapping[int, Path],
    years: Sequence[int],
    features: Sequence[str],
    *,
    maximum_rows: int,
    random_seed: int,
) -> pd.DataFrame:
    years = tuple(int(value) for value in years)
    per_year = max(1, math.ceil(maximum_rows / len(years)))
    samples: list[pd.DataFrame] = []
    for offset, year in enumerate(years):
        frame = pd.read_parquet(panel_paths[year], columns=list(features))
        if len(frame) > per_year:
            rng = np.random.default_rng(random_seed + year * 101 + offset)
            positions = np.sort(rng.choice(len(frame), size=per_year, replace=False))
            frame = frame.iloc[positions]
        samples.append(frame)
    sample = pd.concat(samples, ignore_index=True)
    if len(sample) > maximum_rows:
        rng = np.random.default_rng(random_seed)
        positions = np.sort(rng.choice(len(sample), size=maximum_rows, replace=False))
        sample = sample.iloc[positions].reset_index(drop=True)
    return sample


def _fit_preprocessor(
    sample: pd.DataFrame,
    features: Sequence[str],
    *,
    components: int,
    lower_quantile: float,
    upper_quantile: float,
    random_seed: int,
) -> tuple[Preprocessor, np.ndarray]:
    names = tuple(features)
    values = sample.loc[:, names].to_numpy(np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    medians = np.nanmedian(values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    missing = np.isnan(values)
    if missing.any():
        values[missing] = np.take(medians, np.nonzero(missing)[1])
    lower = np.quantile(values, lower_quantile, axis=0)
    upper = np.quantile(values, upper_quantile, axis=0)
    values = np.clip(values, lower, upper)
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales = np.where(scales > 1e-8, scales, 1.0)
    standardized = (values - means) / scales
    component_count = min(int(components), standardized.shape[1])
    pca = PCA(
        n_components=component_count,
        whiten=True,
        svd_solver="randomized",
        random_state=random_seed,
    )
    embedding = pca.fit_transform(standardized).astype(np.float32)
    preprocessor = Preprocessor(
        feature_names=names,
        medians=medians,
        lower_bounds=lower,
        upper_bounds=upper,
        means=means,
        scales=scales,
        pca=pca,
    )
    return preprocessor, embedding


def _fit_cluster_models(
    *,
    name: str,
    preprocessor: Preprocessor,
    embedding: np.ndarray,
    cluster_counts: Sequence[int],
    study: Mapping[str, Any],
) -> dict[int, PrototypeModel]:
    representation = dict(study["representation"])
    seed = int(representation["random_seed"])
    models: dict[int, PrototypeModel] = {}
    for cluster_count in (int(value) for value in cluster_counts):
        clusters = MiniBatchKMeans(
            n_clusters=cluster_count,
            batch_size=int(representation["mini_batch_size"]),
            max_iter=int(representation["maximum_kmeans_iterations"]),
            n_init=int(representation["kmeans_n_init"]),
            random_state=seed + cluster_count,
            reassignment_ratio=0.01,
        )
        clusters.fit(embedding)
        models[cluster_count] = PrototypeModel(
            name=name,
            preprocessor=preprocessor,
            clusters=clusters,
        )
    return models


@dataclass
class PrototypeStats:
    cluster_count: int
    global_count: np.ndarray = field(init=False)
    global_success: np.ndarray = field(init=False)
    cluster_observations: np.ndarray = field(init=False)
    cluster_success: np.ndarray = field(init=False)
    global_metric_count: dict[str, np.ndarray] = field(init=False)
    global_metric_sum: dict[str, np.ndarray] = field(init=False)
    cluster_metric_count: dict[str, np.ndarray] = field(init=False)
    cluster_metric_sum: dict[str, np.ndarray] = field(init=False)
    last_applied_resolution: int = -1

    def __post_init__(self) -> None:
        self.global_count = np.zeros(6, dtype=np.int64)
        self.global_success = np.zeros(6, dtype=np.float64)
        self.cluster_observations = np.zeros(6 * self.cluster_count, dtype=np.int64)
        self.cluster_success = np.zeros(6 * self.cluster_count, dtype=np.float64)
        self.global_metric_count = {
            name: np.zeros(6, dtype=np.int64) for name in CONTINUOUS_METRICS
        }
        self.global_metric_sum = {
            name: np.zeros(6, dtype=np.float64) for name in CONTINUOUS_METRICS
        }
        self.cluster_metric_count = {
            name: np.zeros(6 * self.cluster_count, dtype=np.int64)
            for name in CONTINUOUS_METRICS
        }
        self.cluster_metric_sum = {
            name: np.zeros(6 * self.cluster_count, dtype=np.float64)
            for name in CONTINUOUS_METRICS
        }

    def update(
        self,
        *,
        scale_index: np.ndarray,
        direction: np.ndarray,
        target: np.ndarray,
        cluster: np.ndarray,
        resolution: np.ndarray,
        metrics: Mapping[str, np.ndarray],
    ) -> None:
        if len(target) == 0:
            return
        group = scale_index.astype(np.int64) * 2 + direction.astype(np.int64)
        cluster_group = group * self.cluster_count + cluster.astype(np.int64)
        self.global_count += np.bincount(group, minlength=6)
        self.global_success += np.bincount(group, weights=target, minlength=6)
        self.cluster_observations += np.bincount(
            cluster_group, minlength=6 * self.cluster_count
        )
        self.cluster_success += np.bincount(
            cluster_group, weights=target, minlength=6 * self.cluster_count
        )
        for name, values in metrics.items():
            finite = np.isfinite(values)
            if not finite.any():
                continue
            metric_group = group[finite]
            metric_cluster_group = cluster_group[finite]
            metric_values = values[finite]
            self.global_metric_count[name] += np.bincount(metric_group, minlength=6)
            self.global_metric_sum[name] += np.bincount(
                metric_group, weights=metric_values, minlength=6
            )
            self.cluster_metric_count[name] += np.bincount(
                metric_cluster_group, minlength=6 * self.cluster_count
            )
            self.cluster_metric_sum[name] += np.bincount(
                metric_cluster_group,
                weights=metric_values,
                minlength=6 * self.cluster_count,
            )
        self.last_applied_resolution = max(
            self.last_applied_resolution, int(np.max(resolution))
        )

    def prior_probability(
        self, scale_index: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        group = scale_index.astype(np.int64) * 2 + direction.astype(np.int64)
        return (self.global_success[group] + 0.5) / (self.global_count[group] + 1.0)

    def probability(
        self,
        scale_index: np.ndarray,
        direction: np.ndarray,
        cluster: np.ndarray,
        smoothing: float,
    ) -> np.ndarray:
        group = scale_index.astype(np.int64) * 2 + direction.astype(np.int64)
        cluster_group = group * self.cluster_count + cluster.astype(np.int64)
        prior = self.prior_probability(scale_index, direction)
        return (self.cluster_success[cluster_group] + smoothing * prior) / (
            self.cluster_observations[cluster_group] + smoothing
        )

    def prior_mean(
        self, name: str, scale_index: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        group = scale_index.astype(np.int64) * 2 + direction.astype(np.int64)
        count = self.global_metric_count[name][group]
        total = self.global_metric_sum[name][group]
        return np.divide(total, count, out=np.zeros_like(total), where=count > 0)

    def metric_mean(
        self,
        name: str,
        scale_index: np.ndarray,
        direction: np.ndarray,
        cluster: np.ndarray,
        smoothing: float,
    ) -> np.ndarray:
        group = scale_index.astype(np.int64) * 2 + direction.astype(np.int64)
        cluster_group = group * self.cluster_count + cluster.astype(np.int64)
        prior = self.prior_mean(name, scale_index, direction)
        count = self.cluster_metric_count[name][cluster_group]
        total = self.cluster_metric_sum[name][cluster_group]
        return (total + smoothing * prior) / (count + smoothing)


def _model_columns(*, include_features: bool = True) -> list[str]:
    columns = [
        "symbol",
        "trade_date",
        "date_idx",
        "signal_year",
        "entry_date_idx",
        "entry_open",
        "legacy_state_present",
    ]
    if include_features:
        columns.extend(FEATURE_SETS["sequence"])
    for scale in TURNING_SCALES:
        columns.extend(
            [
                f"right_censored_{scale}",
                f"next_peak_{scale}",
                f"causal_next_peak_{scale}",
                f"extreme_date_idx_{scale}",
                f"resolution_date_idx_{scale}",
                f"extreme_ahead_{scale}",
                f"extreme_wait_{scale}",
                f"confirmation_wait_{scale}",
                f"entry_confirmation_log_return_{scale}",
                f"directional_extreme_log_return_{scale}",
            ]
        )
    return list(dict.fromkeys(columns))


def _outcome_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {
        "scale_index": [],
        "direction": [],
        "target": [],
        "resolution": [],
        "log_confirmation_wait": [],
        "entry_confirmation_log_return": [],
        "log_extreme_wait_ahead": [],
        "directional_extreme_log_return_ahead": [],
    }
    for scale_index, scale in enumerate(TURNING_SCALES):
        target = frame[f"extreme_ahead_{scale}"].astype("boolean")
        direction = frame[f"causal_next_peak_{scale}"].to_numpy(np.float64)
        resolution = frame[f"resolution_date_idx_{scale}"].to_numpy(np.float64)
        confirmation_wait = frame[f"confirmation_wait_{scale}"].to_numpy(np.float64)
        extreme_wait = frame[f"extreme_wait_{scale}"].to_numpy(np.float64)
        directional_return = frame[f"directional_extreme_log_return_{scale}"].to_numpy(
            np.float64
        )
        arrays["scale_index"].append(np.full(len(frame), scale_index, dtype=np.int8))
        arrays["direction"].append(
            np.where(np.isfinite(direction), direction, -1).astype(np.int8)
        )
        arrays["target"].append(target.fillna(False).to_numpy(dtype=np.float64))
        arrays["resolution"].append(resolution)
        arrays["log_confirmation_wait"].append(np.log1p(confirmation_wait))
        arrays["entry_confirmation_log_return"].append(
            frame[f"entry_confirmation_log_return_{scale}"].to_numpy(np.float64)
        )
        ahead = target.fillna(False).to_numpy(bool)
        log_extreme_wait = np.full(len(frame), np.nan, dtype=np.float64)
        log_extreme_wait[ahead] = np.log1p(extreme_wait[ahead])
        arrays["log_extreme_wait_ahead"].append(log_extreme_wait)
        arrays["directional_extreme_log_return_ahead"].append(
            np.where(ahead, directional_return, np.nan)
        )
    result = {key: np.concatenate(values) for key, values in arrays.items()}
    valid = (
        np.isfinite(result["resolution"])
        & (result["direction"] >= 0)
        & np.isfinite(result["target"])
    )
    return {key: value[valid] for key, value in result.items()}


def _build_updates(
    frame: pd.DataFrame, assignments: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    updates = _outcome_arrays(frame)
    valid_count = len(updates["target"])
    raw_valid: list[np.ndarray] = []
    for scale in TURNING_SCALES:
        raw_valid.append(
            frame[f"resolution_date_idx_{scale}"].notna().to_numpy()
            & frame[f"causal_next_peak_{scale}"].notna().to_numpy()
            & frame[f"extreme_ahead_{scale}"].notna().to_numpy()
        )
    valid = np.concatenate(raw_valid)
    for name, values in assignments.items():
        tiled = np.concatenate([values] * len(TURNING_SCALES))
        updates[name] = tiled[valid].astype(np.int32)
    if any(len(value) != valid_count for value in updates.values()):
        raise AssertionError("daily_path_neighbors_update_length_mismatch")
    order = np.argsort(updates["resolution"], kind="mergesort")
    return {key: value[order] for key, value in updates.items()}


def _merge_updates(
    pending: Mapping[str, np.ndarray] | None, current: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    if not pending or len(pending["resolution"]) == 0:
        return {key: value.copy() for key, value in current.items()}
    merged = {key: np.concatenate([pending[key], current[key]]) for key in current}
    order = np.argsort(merged["resolution"], kind="mergesort")
    return {key: value[order] for key, value in merged.items()}


def _apply_updates_before(
    states: Mapping[str, PrototypeStats],
    updates: Mapping[str, np.ndarray],
    cursor: int,
    date_idx: int,
) -> int:
    end = int(np.searchsorted(updates["resolution"], int(date_idx), side="left"))
    if end <= cursor:
        return cursor
    section = slice(cursor, end)
    metrics = {name: updates[name][section] for name in CONTINUOUS_METRICS}
    for name, state in states.items():
        state.update(
            scale_index=updates["scale_index"][section],
            direction=updates["direction"][section],
            target=updates["target"][section],
            cluster=updates[name][section],
            resolution=updates["resolution"][section],
            metrics=metrics,
        )
        if state.last_applied_resolution >= int(date_idx):
            raise AssertionError("daily_path_neighbors_same_day_label_leakage")
    return end


def _binary_log_loss(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return -(target * np.log(clipped) + (1.0 - target) * np.log1p(-clipped))


def _selection_replay(
    *,
    feature_set: str,
    models: Mapping[int, PrototypeModel],
    panel_paths: Mapping[int, Path],
    study: Mapping[str, Any],
) -> pd.DataFrame:
    selection_years = {int(value) for value in dict(study["period"])["selection_years"]}
    end_year = max(selection_years)
    alphas = tuple(
        float(value)
        for value in dict(study["representation"])["smoothing_strength_candidates"]
    )
    states = {
        f"k{cluster_count}": PrototypeStats(cluster_count) for cluster_count in models
    }
    pending: dict[str, np.ndarray] | None = None
    rows: list[dict[str, Any]] = []
    columns = _model_columns(include_features=True)
    for year in sorted(year for year in panel_paths if year <= end_year):
        frame = pd.read_parquet(panel_paths[year], columns=columns)
        frame = frame.sort_values(["date_idx", "symbol"], kind="mergesort").reset_index(
            drop=True
        )
        assignments = {
            f"k{cluster_count}": model.assign(frame)[0]
            for cluster_count, model in models.items()
        }
        current_updates = _build_updates(frame, assignments)
        updates = _merge_updates(pending, current_updates)
        cursor = 0
        dates = frame["date_idx"].to_numpy(np.int64)
        for date_idx in np.unique(dates):
            cursor = _apply_updates_before(states, updates, cursor, int(date_idx))
            if year not in selection_years:
                continue
            positions = np.flatnonzero(dates == date_idx)
            prior_cells: list[float] = []
            candidate_cells: dict[tuple[int, float], list[float]] = {
                (cluster_count, alpha): []
                for cluster_count in models
                for alpha in alphas
            }
            for scale_index, scale in enumerate(TURNING_SCALES):
                target_series = frame.loc[positions, f"extreme_ahead_{scale}"].astype(
                    "boolean"
                )
                direction_values = frame.loc[
                    positions, f"causal_next_peak_{scale}"
                ].to_numpy(np.float64)
                valid = target_series.notna().to_numpy() & np.isfinite(direction_values)
                if not valid.any():
                    continue
                target = target_series.fillna(False).to_numpy(np.float64)[valid]
                direction = direction_values[valid].astype(np.int8)
                scale_array = np.full(len(target), scale_index, dtype=np.int8)
                for direction_value in (0, 1):
                    group = direction == direction_value
                    if not group.any():
                        continue
                    first_state = states[next(iter(states))]
                    prior = first_state.prior_probability(
                        scale_array[group], direction[group]
                    )
                    prior_cells.append(
                        float(_binary_log_loss(target[group], prior).mean())
                    )
                    for cluster_count in models:
                        key = f"k{cluster_count}"
                        cluster = assignments[key][positions][valid][group]
                        for alpha in alphas:
                            probability = states[key].probability(
                                scale_array[group], direction[group], cluster, alpha
                            )
                            candidate_cells[(cluster_count, alpha)].append(
                                float(
                                    _binary_log_loss(target[group], probability).mean()
                                )
                            )
            if not prior_cells:
                continue
            prior_loss = float(np.mean(prior_cells))
            for (cluster_count, alpha), losses in candidate_cells.items():
                rows.append(
                    {
                        "feature_set": feature_set,
                        "evaluation_year": year,
                        "date_idx": int(date_idx),
                        "cluster_count": int(cluster_count),
                        "smoothing_strength": float(alpha),
                        "prior_log_loss": prior_loss,
                        "model_log_loss": float(np.mean(losses)),
                    }
                )
        pending = {key: value[cursor:] for key, value in updates.items()}
    return pd.DataFrame(rows)


def _select_configuration(
    daily: pd.DataFrame, *, hac_lag: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for keys, group in daily.groupby(
        ["feature_set", "cluster_count", "smoothing_strength"], sort=True
    ):
        feature_set, cluster_count, smoothing = keys
        loss = group["model_log_loss"].to_numpy(np.float64)
        gain = (group["prior_log_loss"] - group["model_log_loss"]).to_numpy(np.float64)
        loss_estimate = atlas._hac_mean(loss, lag=int(hac_lag))
        gain_estimate = atlas._hac_mean(gain, lag=int(hac_lag))
        diagnostics.append(
            {
                "feature_set": feature_set,
                "cluster_count": int(cluster_count),
                "smoothing_strength": float(smoothing),
                "dates": len(group),
                "mean_log_loss": float(loss.mean()),
                "log_loss_hac_se": float(loss_estimate["se"]),
                "information_gain_bits": float(gain.mean() / math.log(2.0)),
                "information_lcb_95_bits": float(
                    gain_estimate["lcb_95"] / math.log(2.0)
                ),
            }
        )
    result = pd.DataFrame(diagnostics)
    selected: dict[str, Any] = {}
    result["selected"] = False
    for feature_set, group in result.groupby("feature_set", sort=True):
        best = group.sort_values(
            ["mean_log_loss", "cluster_count", "smoothing_strength"],
            ascending=[True, True, False],
            kind="mergesort",
        ).iloc[0]
        threshold = float(best["mean_log_loss"] + best["log_loss_hac_se"])
        eligible = group[group["mean_log_loss"] <= threshold].sort_values(
            ["cluster_count", "smoothing_strength", "mean_log_loss"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        choice = eligible.iloc[0]
        index = choice.name
        result.loc[index, "selected"] = True
        selected[str(feature_set)] = {
            "cluster_count": int(choice["cluster_count"]),
            "smoothing_strength": float(choice["smoothing_strength"]),
            "selection_threshold": threshold,
        }
    return result, selected


def _blend_selection_replay(
    *,
    geometry_model: PrototypeModel,
    sequence_model: PrototypeModel,
    selected: Mapping[str, Mapping[str, Any]],
    panel_paths: Mapping[int, Path],
    study: Mapping[str, Any],
) -> pd.DataFrame:
    selection_years = {int(value) for value in dict(study["period"])["selection_years"]}
    end_year = max(selection_years)
    weights = tuple(
        float(value)
        for value in dict(study["representation"])[
            "sequence_incremental_weight_candidates"
        ]
    )
    models = {"geometry": geometry_model, "sequence": sequence_model}
    states = {
        name: PrototypeStats(int(model.clusters.n_clusters))
        for name, model in models.items()
    }
    pending: dict[str, np.ndarray] | None = None
    rows: list[dict[str, Any]] = []
    columns = _model_columns(include_features=True)
    for year in sorted(year for year in panel_paths if year <= end_year):
        frame = pd.read_parquet(panel_paths[year], columns=columns)
        frame = frame.sort_values(["date_idx", "symbol"], kind="mergesort").reset_index(
            drop=True
        )
        assignments = {name: model.assign(frame)[0] for name, model in models.items()}
        updates = _merge_updates(pending, _build_updates(frame, assignments))
        cursor = 0
        dates = frame["date_idx"].to_numpy(np.int64)
        unique_dates, starts = np.unique(dates, return_index=True)
        stops = np.append(starts[1:], len(frame))
        for date_idx, start, stop in zip(unique_dates, starts, stops):
            cursor = _apply_updates_before(states, updates, cursor, int(date_idx))
            if year not in selection_years:
                continue
            positions = np.arange(start, stop, dtype=np.int64)
            geometry_cells: list[float] = []
            blend_cells = {weight: [] for weight in weights}
            for scale_index, scale in enumerate(TURNING_SCALES):
                target_series = frame.loc[positions, f"extreme_ahead_{scale}"].astype(
                    "boolean"
                )
                direction_values = frame.loc[
                    positions, f"causal_next_peak_{scale}"
                ].to_numpy(np.float64)
                valid = target_series.notna().to_numpy() & np.isfinite(direction_values)
                if not valid.any():
                    continue
                target = target_series.fillna(False).to_numpy(np.float64)[valid]
                direction = direction_values[valid].astype(np.int8)
                scale_array = np.full(len(target), scale_index, dtype=np.int8)
                geometry_probability = states["geometry"].probability(
                    scale_array,
                    direction,
                    assignments["geometry"][positions][valid],
                    float(selected["geometry"]["smoothing_strength"]),
                )
                raw_sequence_probability = states["sequence"].probability(
                    scale_array,
                    direction,
                    assignments["sequence"][positions][valid],
                    float(selected["sequence"]["smoothing_strength"]),
                )
                for direction_value in (0, 1):
                    group = direction == direction_value
                    if not group.any():
                        continue
                    geometry_probability_group = geometry_probability[group]
                    raw_sequence_group = raw_sequence_probability[group]
                    target_group = target[group]
                    geometry_cells.append(
                        float(
                            _binary_log_loss(
                                target_group, geometry_probability_group
                            ).mean()
                        )
                    )
                    for weight in weights:
                        blended = (
                            1.0 - weight
                        ) * geometry_probability_group + weight * raw_sequence_group
                        blend_cells[weight].append(
                            float(_binary_log_loss(target_group, blended).mean())
                        )
            if not geometry_cells:
                continue
            geometry_loss = float(np.mean(geometry_cells))
            for weight, losses in blend_cells.items():
                rows.append(
                    {
                        "evaluation_year": year,
                        "date_idx": int(date_idx),
                        "sequence_incremental_weight": float(weight),
                        "geometry_log_loss": geometry_loss,
                        "blend_log_loss": float(np.mean(losses)),
                    }
                )
        pending = {key: value[cursor:] for key, value in updates.items()}
    return pd.DataFrame(rows)


def _select_blend_weight(
    daily: pd.DataFrame, *, hac_lag: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for weight, group in daily.groupby("sequence_incremental_weight", sort=True):
        loss = group["blend_log_loss"].to_numpy(np.float64)
        gain = (group["geometry_log_loss"] - group["blend_log_loss"]).to_numpy(
            np.float64
        )
        loss_estimate = atlas._hac_mean(loss, lag=hac_lag)
        gain_estimate = atlas._hac_mean(gain, lag=hac_lag)
        rows.append(
            {
                "sequence_incremental_weight": float(weight),
                "dates": len(group),
                "mean_log_loss": float(loss.mean()),
                "log_loss_hac_se": float(loss_estimate["se"]),
                "incremental_information_bits": float(gain.mean() / math.log(2.0)),
                "incremental_information_lcb_95_bits": float(
                    gain_estimate["lcb_95"] / math.log(2.0)
                ),
            }
        )
    diagnostics = pd.DataFrame(rows)
    best = diagnostics.sort_values(
        ["mean_log_loss", "sequence_incremental_weight"], kind="mergesort"
    ).iloc[0]
    threshold = float(best["mean_log_loss"] + best["log_loss_hac_se"])
    eligible = diagnostics[diagnostics["mean_log_loss"] <= threshold].sort_values(
        "sequence_incremental_weight", kind="mergesort"
    )
    choice = eligible.iloc[0]
    diagnostics["selected"] = False
    diagnostics.loc[choice.name, "selected"] = True
    return diagnostics, {
        "incremental_weight": float(choice["sequence_incremental_weight"]),
        "selection_threshold": threshold,
    }


def _fit_representation_models(
    *,
    panel_paths: Mapping[int, Path],
    years: Sequence[int],
    study: Mapping[str, Any],
    cluster_counts: Mapping[str, Sequence[int]],
    maximum_rows: int,
) -> tuple[dict[str, dict[int, PrototypeModel]], pd.DataFrame]:
    representation = dict(study["representation"])
    seed = int(representation["random_seed"])
    fitted: dict[str, dict[int, PrototypeModel]] = {}
    metadata: list[dict[str, Any]] = []
    for offset, feature_set in enumerate(("geometry", "sequence")):
        features = FEATURE_SETS[feature_set]
        sample = _sample_features(
            panel_paths,
            years,
            features,
            maximum_rows=maximum_rows,
            random_seed=seed + offset * 1009,
        )
        components = int(representation[f"{feature_set}_pca_components"])
        preprocessor, embedding = _fit_preprocessor(
            sample,
            features,
            components=components,
            lower_quantile=float(representation["winsor_lower"]),
            upper_quantile=float(representation["winsor_upper"]),
            random_seed=seed + offset,
        )
        models = _fit_cluster_models(
            name=feature_set,
            preprocessor=preprocessor,
            embedding=embedding,
            cluster_counts=cluster_counts[feature_set],
            study=study,
        )
        fitted[feature_set] = models
        for cluster_count, model in models.items():
            labels = model.clusters.labels_
            counts = np.bincount(labels, minlength=cluster_count)
            metadata.append(
                {
                    "feature_set": feature_set,
                    "fit_years": ",".join(str(int(year)) for year in years),
                    "sample_rows": len(sample),
                    "input_features": len(features),
                    "pca_components": int(model.preprocessor.pca.n_components_),
                    "pca_explained_variance": float(
                        np.sum(model.preprocessor.pca.explained_variance_ratio_)
                    ),
                    "cluster_count": int(cluster_count),
                    "minimum_cluster_rows": int(counts.min()),
                    "median_cluster_rows": float(np.median(counts)),
                    "maximum_cluster_rows": int(counts.max()),
                    "inertia": float(model.clusters.inertia_),
                }
            )
    return fitted, pd.DataFrame(metadata)


def _safe_auc(target: np.ndarray, probability: np.ndarray) -> float:
    if len(target) == 0 or np.unique(target).size < 2:
        return math.nan
    return float(roc_auc_score(target, probability))


def _prediction_frame(
    *,
    frame: pd.DataFrame,
    scale_index: int,
    probabilities: Mapping[str, np.ndarray],
    metric_predictions: Mapping[str, Mapping[str, np.ndarray]],
    assignments: Mapping[str, np.ndarray],
    distances: Mapping[str, np.ndarray],
    maximum_resolution_used: np.ndarray,
) -> pd.DataFrame:
    scale = TURNING_SCALES[scale_index]
    target = frame[f"extreme_ahead_{scale}"].astype("boolean")
    direction = frame[f"causal_next_peak_{scale}"].to_numpy(np.float64)
    phase = np.full(len(frame), "right_censored", dtype=object)
    known = target.notna().to_numpy() & np.isfinite(direction)
    ahead = target.fillna(False).to_numpy(bool)
    peak = direction > 0.5
    phase[known & peak & ahead] = "approach_peak"
    phase[known & peak & ~ahead] = "post_peak"
    phase[known & ~peak & ahead] = "approach_trough"
    phase[known & ~peak & ~ahead] = "post_trough"
    output = pd.DataFrame(
        {
            "symbol": frame["symbol"].astype(str),
            "trade_date": frame["trade_date"].astype(str),
            "date_idx": frame["date_idx"].to_numpy(np.int32),
            "signal_year": frame["signal_year"].to_numpy(np.int16),
            "turning_scale": np.full(len(frame), scale, dtype=np.int8),
            "causal_next_type": np.where(peak, "peak", "trough"),
            "phase": phase,
            "extreme_ahead": target,
            "extreme_date_idx": frame[f"extreme_date_idx_{scale}"].astype("Int32"),
            "resolution_date_idx": frame[f"resolution_date_idx_{scale}"].astype(
                "Int32"
            ),
            "extreme_wait": frame[f"extreme_wait_{scale}"].astype("Int32"),
            "confirmation_wait": frame[f"confirmation_wait_{scale}"].astype("Int32"),
            "entry_confirmation_log_return": frame[
                f"entry_confirmation_log_return_{scale}"
            ].astype(np.float32),
            "directional_extreme_log_return": frame[
                f"directional_extreme_log_return_{scale}"
            ].astype(np.float32),
            "prior_probability": probabilities["prior"][scale_index],
            "geometry_probability": probabilities["geometry"][scale_index],
            "sequence_raw_probability": probabilities["sequence_raw"][scale_index],
            "sequence_probability": probabilities["sequence"][scale_index],
            "geometry_cluster": assignments["geometry"].astype(np.int16),
            "sequence_cluster": assignments["sequence"].astype(np.int16),
            "geometry_distance": distances["geometry"].astype(np.float32),
            "sequence_distance": distances["sequence"].astype(np.float32),
            "maximum_training_resolution_date_idx": maximum_resolution_used,
            "legacy_state_present": frame["legacy_state_present"].astype(bool),
        }
    )
    for metric in CONTINUOUS_METRICS:
        for model in PREDICTION_MODELS:
            output[f"predicted_{metric}_{model}"] = metric_predictions[metric][model][
                scale_index
            ].astype(np.float32)
    return output


def _daily_cell_metrics(
    *,
    frame: pd.DataFrame,
    positions: np.ndarray,
    date_idx: int,
    scale_index: int,
    probabilities: Mapping[str, np.ndarray],
    metric_predictions: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    scale = TURNING_SCALES[scale_index]
    target_series = frame.loc[positions, f"extreme_ahead_{scale}"].astype("boolean")
    direction_values = frame.loc[positions, f"causal_next_peak_{scale}"].to_numpy(
        np.float64
    )
    valid = target_series.notna().to_numpy() & np.isfinite(direction_values)
    if not valid.any():
        return []
    target_all = target_series.fillna(False).to_numpy(np.float64)
    year = int(frame.loc[positions[0], "signal_year"])
    rows: list[dict[str, Any]] = []
    for direction_value, direction_name in ((0, "trough"), (1, "peak")):
        group = valid & (direction_values.astype(np.int8) == direction_value)
        if not group.any():
            continue
        local_positions = positions[group]
        target = target_all[group]
        record: dict[str, Any] = {
            "signal_year": year,
            "date_idx": int(date_idx),
            "turning_scale": scale,
            "causal_next_type": direction_name,
            "rows": len(target),
            "actual_rate": float(target.mean()),
        }
        for model in PREDICTION_MODELS:
            probability = probabilities[model][scale_index][local_positions]
            loss = _binary_log_loss(target, probability)
            record[f"{model}_log_loss"] = float(loss.mean())
            record[f"{model}_brier"] = float(np.mean(np.square(target - probability)))
            record[f"{model}_auc"] = _safe_auc(target, probability)
        extreme_wait = frame.loc[local_positions, f"extreme_wait_{scale}"].to_numpy(
            np.float64
        )
        log_extreme_wait = np.full(len(target), np.nan, dtype=np.float64)
        ahead = target.astype(bool)
        log_extreme_wait[ahead] = np.log1p(extreme_wait[ahead])
        actual_metrics = {
            "log_confirmation_wait": np.log1p(
                frame.loc[local_positions, f"confirmation_wait_{scale}"].to_numpy(
                    np.float64
                )
            ),
            "entry_confirmation_log_return": frame.loc[
                local_positions, f"entry_confirmation_log_return_{scale}"
            ].to_numpy(np.float64),
            "log_extreme_wait_ahead": log_extreme_wait,
            "directional_extreme_log_return_ahead": np.where(
                target.astype(bool),
                frame.loc[
                    local_positions, f"directional_extreme_log_return_{scale}"
                ].to_numpy(np.float64),
                np.nan,
            ),
        }
        for metric, actual in actual_metrics.items():
            for model in PREDICTION_MODELS:
                predicted = metric_predictions[metric][model][scale_index][
                    local_positions
                ]
                metric_valid = np.isfinite(actual) & np.isfinite(predicted)
                if metric_valid.any():
                    error = actual[metric_valid] - predicted[metric_valid]
                    record[f"{metric}_{model}_mae"] = float(np.mean(np.abs(error)))
                    record[f"{metric}_{model}_mse"] = float(np.mean(np.square(error)))
                    record[f"{metric}_rows"] = int(metric_valid.sum())
                else:
                    record[f"{metric}_{model}_mae"] = math.nan
                    record[f"{metric}_{model}_mse"] = math.nan
                    record[f"{metric}_rows"] = 0
        rows.append(record)
    return rows


def _final_replay(
    *,
    models: Mapping[str, PrototypeModel],
    selected: Mapping[str, Mapping[str, Any]],
    panel_paths: Mapping[int, Path],
    prediction_root: Path,
    study: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    evaluation_years = {
        int(value) for value in dict(study["period"])["rolling_evaluation_years"]
    }
    states = {
        name: PrototypeStats(int(model.clusters.n_clusters))
        for name, model in models.items()
    }
    pending: dict[str, np.ndarray] | None = None
    daily_rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    columns = _model_columns(include_features=True)
    prediction_root.mkdir(parents=True, exist_ok=True)
    for year in sorted(panel_paths):
        frame = pd.read_parquet(panel_paths[year], columns=columns)
        frame = frame.sort_values(["date_idx", "symbol"], kind="mergesort").reset_index(
            drop=True
        )
        assigned = {name: model.assign(frame) for name, model in models.items()}
        assignments = {name: values[0] for name, values in assigned.items()}
        distances = {name: values[1] for name, values in assigned.items()}
        updates = _merge_updates(pending, _build_updates(frame, assignments))
        cursor = 0
        row_count = len(frame)
        probabilities = {
            model: np.full((len(TURNING_SCALES), row_count), np.nan, dtype=np.float32)
            for model in PREDICTION_MODELS
        }
        metric_predictions = {
            metric: {
                model: np.full(
                    (len(TURNING_SCALES), row_count), np.nan, dtype=np.float32
                )
                for model in PREDICTION_MODELS
            }
            for metric in CONTINUOUS_METRICS
        }
        maximum_resolution_used = np.full(row_count, -1, dtype=np.int32)
        dates = frame["date_idx"].to_numpy(np.int64)
        unique_dates, starts = np.unique(dates, return_index=True)
        stops = np.append(starts[1:], len(frame))
        for date_idx, start, stop in zip(unique_dates, starts, stops):
            cursor = _apply_updates_before(states, updates, cursor, int(date_idx))
            positions = np.arange(start, stop, dtype=np.int64)
            maximum_resolution_used[positions] = states[
                "geometry"
            ].last_applied_resolution
            for scale_index, scale in enumerate(TURNING_SCALES):
                direction_values = frame.loc[
                    positions, f"causal_next_peak_{scale}"
                ].to_numpy(np.float64)
                valid = np.isfinite(direction_values)
                if not valid.any():
                    continue
                local_positions = positions[valid]
                direction = direction_values[valid].astype(np.int8)
                scale_array = np.full(len(local_positions), scale_index, dtype=np.int8)
                prior = states["geometry"].prior_probability(scale_array, direction)
                probabilities["prior"][scale_index, local_positions] = prior
                geometry_probability = states["geometry"].probability(
                    scale_array,
                    direction,
                    assignments["geometry"][local_positions],
                    float(selected["geometry"]["smoothing_strength"]),
                )
                raw_sequence_probability = states["sequence"].probability(
                    scale_array,
                    direction,
                    assignments["sequence"][local_positions],
                    float(selected["sequence"]["smoothing_strength"]),
                )
                sequence_weight = float(selected["sequence"]["incremental_weight"])
                probabilities["geometry"][scale_index, local_positions] = (
                    geometry_probability
                )
                probabilities["sequence_raw"][scale_index, local_positions] = (
                    raw_sequence_probability
                )
                probabilities["sequence"][scale_index, local_positions] = (
                    1.0 - sequence_weight
                ) * geometry_probability + sequence_weight * raw_sequence_probability
                for metric in CONTINUOUS_METRICS:
                    metric_predictions[metric]["prior"][
                        scale_index, local_positions
                    ] = states["geometry"].prior_mean(metric, scale_array, direction)
                    geometry_mean = states["geometry"].metric_mean(
                        metric,
                        scale_array,
                        direction,
                        assignments["geometry"][local_positions],
                        float(selected["geometry"]["smoothing_strength"]),
                    )
                    raw_sequence_mean = states["sequence"].metric_mean(
                        metric,
                        scale_array,
                        direction,
                        assignments["sequence"][local_positions],
                        float(selected["sequence"]["smoothing_strength"]),
                    )
                    metric_predictions[metric]["geometry"][
                        scale_index, local_positions
                    ] = geometry_mean
                    metric_predictions[metric]["sequence_raw"][
                        scale_index, local_positions
                    ] = raw_sequence_mean
                    metric_predictions[metric]["sequence"][
                        scale_index, local_positions
                    ] = (
                        1.0 - sequence_weight
                    ) * geometry_mean + sequence_weight * raw_sequence_mean
            if year in evaluation_years:
                for scale_index in range(len(TURNING_SCALES)):
                    daily_rows.extend(
                        _daily_cell_metrics(
                            frame=frame,
                            positions=positions,
                            date_idx=int(date_idx),
                            scale_index=scale_index,
                            probabilities=probabilities,
                            metric_predictions=metric_predictions,
                        )
                    )
        pending = {key: value[cursor:] for key, value in updates.items()}
        if year not in evaluation_years:
            continue
        year_output = pd.concat(
            [
                _prediction_frame(
                    frame=frame,
                    scale_index=scale_index,
                    probabilities=probabilities,
                    metric_predictions=metric_predictions,
                    assignments=assignments,
                    distances=distances,
                    maximum_resolution_used=maximum_resolution_used,
                )
                for scale_index in range(len(TURNING_SCALES))
            ],
            ignore_index=True,
        )
        output_path = prediction_root / f"year={year}" / "part-0000.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        year_output.to_parquet(
            output_path,
            index=False,
            compression="zstd",
            row_group_size=int(dict(study["resources"])["parquet_row_group_size"]),
        )
        prediction_records.append(
            {
                "year": year,
                **atlas._file_record(output_path, include_hash=False),
                "rows": len(year_output),
            }
        )
        del year_output, frame
    return pd.DataFrame(daily_rows), prediction_records


def _information_summaries(
    daily: pd.DataFrame, study: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparisons = {
        "geometry_vs_prior": ("prior", "geometry"),
        "sequence_raw_vs_geometry": ("geometry", "sequence_raw"),
        "sequence_vs_prior": ("prior", "sequence"),
        "sequence_vs_geometry": ("geometry", "sequence"),
    }
    rows: list[pd.DataFrame] = []
    for name, (baseline, candidate) in comparisons.items():
        part = daily[
            ["signal_year", "date_idx", "turning_scale", "causal_next_type"]
        ].copy()
        part["comparison"] = name
        part["information_gain_nats"] = (
            daily[f"{baseline}_log_loss"] - daily[f"{candidate}_log_loss"]
        )
        rows.append(part)
    cell_information = pd.concat(rows, ignore_index=True)
    date_information = (
        cell_information.groupby(["signal_year", "date_idx", "comparison"], sort=True)[
            "information_gain_nats"
        ]
        .mean()
        .reset_index()
    )
    annual = (
        date_information.groupby(["signal_year", "comparison"], sort=True)[
            "information_gain_nats"
        ]
        .agg(dates="count", information_gain_nats="mean")
        .reset_index()
    )
    annual["information_gain_bits"] = annual["information_gain_nats"] / math.log(2.0)
    year_scopes = _evaluation_scopes(study)
    scopes = {
        name: date_information[date_information["signal_year"].isin(years)]
        for name, years in year_scopes.items()
    }
    headline_rows: list[dict[str, Any]] = []
    hac_lag = int(dict(study["evaluation"])["hac_lag"])
    for scope, values in scopes.items():
        for comparison, group in values.groupby("comparison", sort=True):
            gain = group["information_gain_nats"].to_numpy(np.float64)
            estimate = atlas._hac_mean(gain, lag=hac_lag)
            headline_rows.append(
                {
                    "scope": scope,
                    "comparison": comparison,
                    "dates": len(group),
                    "information_gain_nats": float(estimate["mean"]),
                    "information_gain_bits": float(estimate["mean"] / math.log(2.0)),
                    "information_hac_se_nats": float(estimate["se"]),
                    "information_lcb_95_bits": float(
                        estimate["lcb_95"] / math.log(2.0)
                    ),
                    "information_ucb_95_bits": float(
                        estimate["ucb_95"] / math.log(2.0)
                    ),
                    "positive_years": int(
                        annual[
                            annual["comparison"].eq(comparison)
                            & annual["signal_year"].isin(group["signal_year"].unique())
                        ]["information_gain_nats"]
                        .gt(0)
                        .sum()
                    ),
                }
            )
    strict_years = next(iter(year_scopes.values()))
    scale = (
        cell_information[cell_information["signal_year"].isin(strict_years)]
        .groupby(["comparison", "turning_scale", "causal_next_type"], sort=True)[
            "information_gain_nats"
        ]
        .agg(cell_dates="count", information_gain_nats="mean")
        .reset_index()
    )
    scale["information_gain_bits"] = scale["information_gain_nats"] / math.log(2.0)
    return pd.DataFrame(headline_rows), annual, scale


def _continuous_summary(daily: pd.DataFrame, study: Mapping[str, Any]) -> pd.DataFrame:
    scopes = {
        name: daily[daily["signal_year"].isin(years)]
        for name, years in _evaluation_scopes(study).items()
    }
    rows: list[dict[str, Any]] = []
    for scope, values in scopes.items():
        for metric in CONTINUOUS_METRICS:
            for model in PREDICTION_MODELS:
                mae = values[f"{metric}_{model}_mae"].to_numpy(np.float64)
                mse = values[f"{metric}_{model}_mse"].to_numpy(np.float64)
                rows.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "model": model,
                        "cell_dates": int(np.isfinite(mae).sum()),
                        "mean_cell_mae": float(np.nanmean(mae)),
                        "root_mean_cell_mse": float(np.sqrt(np.nanmean(mse))),
                    }
                )
    return pd.DataFrame(rows)


def _calibration_table(
    connection: duckdb.DuckDBPyConnection,
    prediction_root: Path,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    strict_years = next(iter(_evaluation_scopes(study).values()))
    complete_years = ",".join(str(value) for value in sorted(strict_years))
    glob = prediction_root / "year=*" / "part-0000.parquet"
    return connection.execute(
        f"""
        WITH base AS (
            SELECT *
            FROM read_parquet({atlas._sql_quote(glob)}, hive_partitioning=true)
            WHERE signal_year IN ({complete_years})
              AND extreme_ahead IS NOT NULL
        ), long AS (
            SELECT signal_year, date_idx, turning_scale, causal_next_type,
                   extreme_ahead::INTEGER AS actual,
                   'prior' AS model, prior_probability AS probability
            FROM base
            UNION ALL
            SELECT signal_year, date_idx, turning_scale, causal_next_type,
                   extreme_ahead::INTEGER, 'geometry', geometry_probability
            FROM base
            UNION ALL
            SELECT signal_year, date_idx, turning_scale, causal_next_type,
                   extreme_ahead::INTEGER, 'sequence_raw', sequence_raw_probability
            FROM base
            UNION ALL
            SELECT signal_year, date_idx, turning_scale, causal_next_type,
                   extreme_ahead::INTEGER, 'sequence', sequence_probability
            FROM base
        ), ranked AS (
            SELECT *, ntile({int(dict(study["evaluation"])["calibration_bins"])}) OVER (
                PARTITION BY model, turning_scale, causal_next_type
                ORDER BY probability
            ) AS calibration_bin
            FROM long
        )
        SELECT
            model,
            turning_scale,
            causal_next_type,
            calibration_bin,
            count(*) AS rows,
            avg(probability) AS predicted,
            avg(actual) AS actual
        FROM ranked
        GROUP BY ALL
        ORDER BY model, turning_scale, causal_next_type, calibration_bin
        """
    ).fetchdf()


def _prediction_audit(
    connection: duckdb.DuckDBPyConnection, prediction_root: Path
) -> dict[str, Any]:
    glob = prediction_root / "year=*" / "part-0000.parquet"
    row = (
        connection.execute(
            f"""
            SELECT
                count(*) AS prediction_rows,
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
                count(*) FILTER (WHERE extreme_ahead IS NULL) AS censored_rows,
                min(least(prior_probability, geometry_probability,
                          sequence_raw_probability,
                          sequence_probability)) AS minimum_probability,
                max(greatest(prior_probability, geometry_probability,
                             sequence_raw_probability,
                             sequence_probability)) AS maximum_probability,
                min(trade_date) AS minimum_date,
                max(trade_date) AS maximum_date
            FROM read_parquet({atlas._sql_quote(glob)}, hive_partitioning=true)
            """
        )
        .fetchdf()
        .iloc[0]
        .to_dict()
    )
    audit = {
        key: (
            float(value)
            if key in {"minimum_probability", "maximum_probability"}
            else str(value)
            if key in {"minimum_date", "maximum_date"}
            else int(value)
        )
        for key, value in row.items()
    }
    blocking = (
        "duplicate_keys",
        "forbidden_rows",
        "nonfinite_probability_rows",
        "causal_update_violations",
    )
    if any(audit[key] != 0 for key in blocking):
        raise ValueError(f"daily_path_neighbors_prediction_audit_failed:{audit}")
    if not 0.0 < audit["minimum_probability"] < audit["maximum_probability"] < 1.0:
        raise ValueError("daily_path_neighbors_probability_range_invalid")
    return audit


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study_path = atlas._resolve_path(study_path)
    output_root = atlas._resolve_path(output_root)
    study = load_study(study_path)
    output_root.mkdir(parents=True, exist_ok=True)
    panel_manifest = build_panels(study_path=study_path, output_root=output_root)
    panel_paths = _panel_paths(panel_manifest)
    fingerprint = str(panel_manifest["experiment_fingerprint"])
    progress_path = output_root / "analysis_progress.json"
    atlas._write_json(
        progress_path,
        {"status": "fitting_selection_representations", "fingerprint": fingerprint},
    )
    representation = dict(study["representation"])
    period = dict(study["period"])
    candidates = tuple(int(value) for value in representation["cluster_candidates"])
    selection_models, selection_metadata = _fit_representation_models(
        panel_paths=panel_paths,
        years=period["representation_initial_fit_years"],
        study=study,
        cluster_counts={name: candidates for name in FEATURE_SETS},
        maximum_rows=int(representation["initial_fit_max_rows"]),
    )
    selection_daily_parts: list[pd.DataFrame] = []
    for feature_set in ("geometry", "sequence"):
        atlas._write_json(
            progress_path,
            {
                "status": f"selecting_{feature_set}_prototype",
                "fingerprint": fingerprint,
            },
        )
        selection_daily_parts.append(
            _selection_replay(
                feature_set=feature_set,
                models=selection_models[feature_set],
                panel_paths=panel_paths,
                study=study,
            )
        )
    selection_daily = pd.concat(selection_daily_parts, ignore_index=True)
    selection_diagnostics, selected = _select_configuration(
        selection_daily, hac_lag=int(dict(study["evaluation"])["hac_lag"])
    )
    atlas._write_json(
        progress_path,
        {
            "status": "selecting_sequence_incremental_weight",
            "fingerprint": fingerprint,
            "selected": selected,
        },
    )
    blend_daily = _blend_selection_replay(
        geometry_model=selection_models["geometry"][
            int(selected["geometry"]["cluster_count"])
        ],
        sequence_model=selection_models["sequence"][
            int(selected["sequence"]["cluster_count"])
        ],
        selected=selected,
        panel_paths=panel_paths,
        study=study,
    )
    blend_diagnostics, blend_selected = _select_blend_weight(
        blend_daily, hac_lag=int(dict(study["evaluation"])["hac_lag"])
    )
    selected["sequence"].update(blend_selected)
    del selection_models

    atlas._write_json(
        progress_path,
        {
            "status": "fitting_final_representations",
            "fingerprint": fingerprint,
            "selected": selected,
        },
    )
    final_nested, final_metadata = _fit_representation_models(
        panel_paths=panel_paths,
        years=period["final_representation_fit_years"],
        study=study,
        cluster_counts={
            name: (int(selected[name]["cluster_count"]),) for name in FEATURE_SETS
        },
        maximum_rows=int(representation["final_fit_max_rows"]),
    )
    final_models = {
        name: next(iter(final_nested[name].values())) for name in FEATURE_SETS
    }
    model_path = output_root / "prototype_models.joblib"
    joblib.dump(final_models, model_path, compress=3)
    prediction_root = output_root / "oos_predictions"
    atlas._write_json(
        progress_path,
        {
            "status": "running_daily_causal_replay",
            "fingerprint": fingerprint,
            "selected": selected,
        },
    )
    daily_metrics, prediction_records = _final_replay(
        models=final_models,
        selected=selected,
        panel_paths=panel_paths,
        prediction_root=prediction_root,
        study=study,
    )
    headline, annual, scale = _information_summaries(daily_metrics, study)
    continuous = _continuous_summary(daily_metrics, study)

    connection = atlas._connect(output_root, study)
    try:
        calibration = _calibration_table(connection, prediction_root, study)
        prediction_audit = _prediction_audit(connection, prediction_root)
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

    outputs = {
        "selection_daily": output_root / "selection_daily.csv",
        "selection_diagnostics": output_root / "selection_diagnostics.csv",
        "blend_selection_daily": output_root / "blend_selection_daily.csv",
        "blend_selection_diagnostics": output_root / "blend_selection_diagnostics.csv",
        "selection_representation_metadata": output_root
        / "selection_representation_metadata.csv",
        "final_representation_metadata": output_root
        / "final_representation_metadata.csv",
        "daily_metrics": output_root / "daily_metrics.csv",
        "headline_information": output_root / "headline_information.csv",
        "annual_information": output_root / "annual_information.csv",
        "scale_information": output_root / "scale_information.csv",
        "continuous_metrics": output_root / "continuous_metrics.csv",
        "calibration": output_root / "calibration.csv",
    }
    selection_daily.to_csv(outputs["selection_daily"], index=False)
    selection_diagnostics.to_csv(outputs["selection_diagnostics"], index=False)
    blend_daily.to_csv(outputs["blend_selection_daily"], index=False)
    blend_diagnostics.to_csv(outputs["blend_selection_diagnostics"], index=False)
    selection_metadata.to_csv(outputs["selection_representation_metadata"], index=False)
    final_metadata.to_csv(outputs["final_representation_metadata"], index=False)
    daily_metrics.to_csv(outputs["daily_metrics"], index=False)
    headline.to_csv(outputs["headline_information"], index=False)
    annual.to_csv(outputs["annual_information"], index=False)
    scale.to_csv(outputs["scale_information"], index=False)
    continuous.to_csv(outputs["continuous_metrics"], index=False)
    calibration.to_csv(outputs["calibration"], index=False)

    manifest = {
        "schema": ANALYSIS_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "builder_version": BUILDER_VERSION,
        "experiment_fingerprint": fingerprint,
        "study": atlas._file_record(study_path),
        "panel_manifest": atlas._file_record(output_root / "panel_manifest.json"),
        "runtime": runtime,
        "selected_configurations": selected,
        "audit": {
            **prediction_audit,
            "quality_pool_rows": int(panel_manifest["audit"]["rows"]),
            "missing_auxiliary_rows": int(
                panel_manifest["audit"]["missing_auxiliary_rows"]
            ),
            "prediction_year_files": len(prediction_records),
        },
        "outputs": {
            **{key: str(path.resolve()) for key, path in outputs.items()},
            "prototype_models": str(model_path.resolve()),
            "predictions": prediction_records,
        },
        "training_performed": True,
        "online_label_updates_performed": True,
        "fixed_holding_horizon_used": False,
        "portfolio_selection_performed": False,
        "execution_backtest_performed": False,
        "profit_claim_allowed": False,
        "causal_effect_claim_allowed": False,
        "report_generation_performed": False,
    }
    manifest_path = output_root / "analysis_manifest.json"
    atlas._write_json(manifest_path, manifest)
    atlas._write_json(
        progress_path,
        {
            "status": "completed",
            "fingerprint": fingerprint,
            "analysis_manifest": str(manifest_path.resolve()),
        },
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and replay causal daily path prototype neighbors."
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--panels-only", action="store_true", help="Build causal panels and stop."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.panels_only:
        result = build_panels(study_path=args.study, output_root=args.output_root)
    else:
        result = run_study(study_path=args.study, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
