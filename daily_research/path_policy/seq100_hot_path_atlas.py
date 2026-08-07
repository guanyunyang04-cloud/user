"""Adjusted, execution-aware path atlas for daily A-share hot-stock research.

The study deliberately keeps three objects separate:

1. the causal state visible at the signal-day close;
2. the complete adjusted D1-D20 path after a next-market-day entry anchor;
3. any later rule or model used to compress a validated conditional law.

No single future-return label defines the discovery sample.  The dense security
status panel is the time skeleton, so suspensions remain missing market days
instead of being silently skipped by a per-stock ``lead`` operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_hot_path_atlas_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_hot_path_atlas_v1"
)
STUDY_ID = "seq100_hot_path_atlas_v1"
MANIFEST_SCHEMA = "seq100_hot_path_atlas_manifest/1"
ANALYSIS_SCHEMA = "seq100_hot_path_atlas_analysis/1"
BUILDER_VERSION = 1
FORMAL_START_YEAR = 2012
FORMAL_END_YEAR = 2025
FORBIDDEN_YEAR = 2026
PATH_DAYS = tuple(range(1, 21))
LEGAL_EXIT_DAYS = tuple(range(2, 21))
RETURN_HORIZONS = (5, 10, 20)
REQUIRED_DOMAINS = (
    "market_daily_raw",
    "adjust_factor",
    "security_status",
    "universe_snapshot",
    "industry_concept",
    "market_intraday_5m",
)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported_json_type:{type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, include_hash: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size": int(path.stat().st_size),
    }
    if include_hash:
        result["sha256"] = _sha256_file(path)
    return result


def _sql_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _parquet_scan(paths: Sequence[Path]) -> str:
    if not paths:
        raise ValueError("empty_parquet_path_list")
    values = ",".join(_sql_quote(path) for path in paths)
    return f"read_parquet([{values}], union_by_name=true)"


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study_path = _resolve_path(path)
    study = _read_json(study_path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("study_id_mismatch")
    source = dict(study.get("source", {}) or {})
    period = dict(study.get("period", {}) or {})
    price = dict(study.get("price_and_path", {}) or {})
    if int(source.get("forbidden_year", -1)) != FORBIDDEN_YEAR:
        raise ValueError("forbidden_year_contract_missing")
    if str(source.get("maximum_outcome_date")) != "2025-12-31":
        raise ValueError("maximum_outcome_date_contract_mismatch")
    if str(period.get("signal_start")) != "2012-01-01":
        raise ValueError("signal_start_contract_mismatch")
    if str(period.get("signal_end")) != "2025-12-31":
        raise ValueError("signal_end_contract_mismatch")
    if int(price.get("horizon_days", -1)) != 20:
        raise ValueError("path_horizon_contract_mismatch")
    if int(price.get("t_plus_one_first_legal_exit_day", -1)) != 2:
        raise ValueError("t_plus_one_contract_mismatch")
    return study


def _dataset_paths_and_manifest(
    qdp_root: Path, active: Mapping[str, Any], domain: str
) -> tuple[list[Path], Path, dict[str, Any]]:
    dataset_id = str(dict(active.get("datasets", {}) or {}).get(domain, "") or "")
    if not dataset_id:
        raise ValueError(f"active_dataset_missing:{domain}")
    manifest_path = qdp_root / "datasets" / domain / dataset_id / "dataset.json"
    manifest = _read_json(manifest_path)
    paths: list[Path] = []
    for shard in list(manifest.get("shards", []) or []):
        if str(shard.get("status", "stored")) not in {"stored", "completed"}:
            continue
        path = Path(str(shard.get("path", "") or ""))
        if not path.is_absolute():
            path = qdp_root / path
        if path.is_file():
            paths.append(path.resolve())
    if not paths:
        raise FileNotFoundError(f"active_dataset_has_no_files:{domain}")
    return paths, manifest_path, manifest


def _input_contract(
    study_path: Path, qdp_root: Path
) -> tuple[dict[str, Any], dict[str, list[Path]], str]:
    active_path = qdp_root / "active" / "active.json"
    active = _read_json(active_path)
    paths: dict[str, list[Path]] = {}
    datasets: dict[str, Any] = {}
    for domain in REQUIRED_DOMAINS:
        domain_paths, manifest_path, manifest = _dataset_paths_and_manifest(
            qdp_root, active, domain
        )
        paths[domain] = domain_paths
        datasets[domain] = {
            "dataset_id": str(dict(active.get("datasets", {}) or {})[domain]),
            "manifest": _file_record(manifest_path),
            "row_count": int(manifest.get("row_count", 0) or 0),
            "start_date": str(manifest.get("start_date", "")),
            "end_date": str(manifest.get("end_date", "")),
            "shard_count": int(len(domain_paths)),
        }
    payload = {
        "builder_version": BUILDER_VERSION,
        "study_sha256": _sha256_file(study_path),
        "active_sha256": _sha256_file(active_path),
        "datasets": {
            key: value["manifest"]["sha256"] for key, value in datasets.items()
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return (
        {
            "active_manifest": _file_record(active_path),
            "datasets": datasets,
            "fingerprint_payload": payload,
        },
        paths,
        fingerprint,
    )


def _connect(output_root: Path, study: Mapping[str, Any]) -> duckdb.DuckDBPyConnection:
    resources = dict(study.get("resources", {}) or {})
    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={int(resources.get('duckdb_threads', 2))}")
    connection.execute("PRAGMA preserve_insertion_order=false")
    memory_limit = str(resources.get("duckdb_memory_limit", "512MB"))
    connection.execute(f"PRAGMA memory_limit={_sql_quote(memory_limit)}")
    temp_dir = output_root / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection.execute("PRAGMA temp_directory=?", [str(temp_dir)])
    return connection


def _copy_query(
    connection: duckdb.DuckDBPyConnection, query: str, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    try:
        connection.execute(
            f"COPY ({query}) TO {_sql_quote(temporary)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _dense_base_query(paths: Mapping[str, Sequence[Path]]) -> str:
    status = _parquet_scan(paths["security_status"])
    universe = _parquet_scan(paths["universe_snapshot"])
    daily = _parquet_scan(paths["market_daily_raw"])
    factor = _parquet_scan(paths["adjust_factor"])
    industry = _parquet_scan(paths["industry_concept"])
    return f"""
WITH status AS (
    SELECT symbol, trade_date, is_st, is_suspended, is_delisted
    FROM {status}
    WHERE trade_date >= '2010-01-01' AND trade_date <= '2025-12-31'
), universe AS (
    SELECT symbol, trade_date, exchange, board, list_date
    FROM {universe}
    WHERE trade_date >= '2010-01-01' AND trade_date <= '2025-12-31'
      AND lower(board) = 'main' AND exchange IN ('SH', 'SZ')
), daily AS (
    SELECT symbol, trade_date, open, high, low, close, volume, amount
    FROM {daily}
    WHERE trade_date >= '2010-01-01' AND trade_date <= '2025-12-31'
), factor AS (
    SELECT symbol, trade_date, adjust_factor
    FROM {factor}
    WHERE trade_date >= '2010-01-01' AND trade_date <= '2025-12-31'
), industry AS (
    SELECT symbol, trade_date, industry
    FROM {industry}
    WHERE trade_date >= '2010-01-01' AND trade_date <= '2025-12-31'
), joined AS (
    SELECT
        s.symbol,
        s.trade_date,
        u.exchange,
        u.board,
        u.list_date,
        coalesce(i.industry, 'UNKNOWN') AS industry,
        s.is_st,
        s.is_suspended,
        s.is_delisted,
        d.open AS raw_open,
        d.high AS raw_high,
        d.low AS raw_low,
        d.close AS raw_close,
        d.volume,
        d.amount,
        f.adjust_factor
    FROM status s
    INNER JOIN universe u USING(symbol, trade_date)
    LEFT JOIN daily d USING(symbol, trade_date)
    LEFT JOIN factor f USING(symbol, trade_date)
    LEFT JOIN industry i USING(symbol, trade_date)
)
SELECT
    *,
    dense_rank() OVER (ORDER BY trade_date) - 1 AS date_idx,
    row_number() OVER (PARTITION BY symbol ORDER BY trade_date) AS listed_open_days,
    raw_open * adjust_factor AS adj_open,
    raw_high * adjust_factor AS adj_high,
    raw_low * adjust_factor AS adj_low,
    raw_close * adjust_factor AS adj_close,
    (raw_open > 0 AND raw_high > 0 AND raw_low > 0 AND raw_close > 0
     AND adjust_factor > 0 AND isfinite(adjust_factor)) AS bar_valid
FROM joined
ORDER BY symbol, trade_date
"""


def _path_lead_expressions() -> list[str]:
    expressions: list[str] = []
    for day in PATH_DAYS:
        expressions.extend(
            [
                f"lead(adj_high, {day}) OVER w AS path_high_{day}",
                f"lead(adj_low, {day}) OVER w AS path_low_{day}",
                f"lead(adj_close, {day}) OVER w AS path_close_{day}",
            ]
        )
        if day == 1:
            expressions.extend(
                [
                    "lead(trade_date, 1) OVER w AS entry_trade_date",
                    "lead(adj_open, 1) OVER w AS entry_open",
                    "lead(is_st, 1) OVER w AS entry_is_st",
                    "lead(is_suspended, 1) OVER w AS entry_is_suspended",
                    "lead(is_delisted, 1) OVER w AS entry_is_delisted",
                ]
            )
    return expressions


def _first_hit_case(prefix: str, threshold: float, *, direction: str) -> str:
    cases: list[str] = []
    for day in LEGAL_EXIT_DAYS:
        operator = ">=" if direction == "up" else "<="
        value = abs(float(threshold)) if direction == "up" else -abs(float(threshold))
        cases.append(
            f"WHEN path_{prefix}_{day} / NULLIF(entry_open, 0) - 1.0 "
            f"{operator} {value:.12g} THEN {day}"
        )
    return "CASE " + " ".join(cases) + " ELSE NULL END"


def _greatest_path(prefix: str, days: Iterable[int]) -> str:
    return "greatest(" + ",".join(f"path_{prefix}_{day}" for day in days) + ")"


def _least_path(prefix: str, days: Iterable[int]) -> str:
    return "least(" + ",".join(f"path_{prefix}_{day}" for day in days) + ")"


def _valid_path_count(prefix: str, days: Iterable[int]) -> str:
    return " + ".join(
        f"CASE WHEN path_{prefix}_{day} > 0 AND isfinite(path_{prefix}_{day}) "
        "THEN 1 ELSE 0 END"
        for day in days
    )


def _year_panel_query(
    dense_base_path: Path,
    formal_pool_path: Path,
    *,
    year: int,
    minimum_listed_days: int,
    minimum_prior_valid: int,
    cost_bps: float,
) -> str:
    input_start = f"{int(year) - 1}-01-01"
    input_end = "2025-12-31" if int(year) == FORMAL_END_YEAR else f"{int(year) + 1}-12-31"
    path_leads = ",\n        ".join(_path_lead_expressions())
    path_columns = ",\n    ".join(
        column
        for day in PATH_DAYS
        for column in (
            f"path_high_{day}",
            f"path_low_{day}",
            f"path_close_{day}",
        )
    )
    horizon_expressions: list[str] = []
    for horizon in RETURN_HORIZONS:
        legal_days = range(2, horizon + 1)
        exposure_days = range(1, horizon + 1)
        horizon_expressions.extend(
            [
                f"CASE WHEN path_close_{horizon} > 0 AND entry_open > 0 "
                f"THEN ln(path_close_{horizon} / entry_open) ELSE NULL END AS terminal_log_return_{horizon}",
                f"{_greatest_path('high', legal_days)} / NULLIF(entry_open, 0) - 1.0 AS mfe_{horizon}",
                f"{_least_path('low', exposure_days)} / NULLIF(entry_open, 0) - 1.0 AS mae_{horizon}",
                f"{_greatest_path('close', legal_days)} / NULLIF(entry_open, 0) - 1.0 - {float(cost_bps) / 10000.0:.12g} AS oracle_best_close_net_{horizon}",
                f"({_valid_path_count('close', exposure_days)}) AS valid_close_days_{horizon}",
            ]
        )
    horizon_sql = ",\n        ".join(horizon_expressions)
    up5 = _first_hit_case("high", 0.05, direction="up")
    up10 = _first_hit_case("high", 0.10, direction="up")
    down3 = _first_hit_case("low", 0.03, direction="down")
    down5 = _first_hit_case("low", 0.05, direction="down")
    return f"""
WITH dense AS (
    SELECT *
    FROM read_parquet({_sql_quote(dense_base_path)})
    WHERE trade_date >= '{input_start}' AND trade_date <= '{input_end}'
), lagged AS (
    SELECT *,
        lag(adj_close, 1) OVER w AS close_lag1,
        lag(adj_close, 3) OVER w AS close_lag3,
        lag(adj_close, 5) OVER w AS close_lag5,
        lag(adj_close, 10) OVER w AS close_lag10,
        lag(adj_close, 20) OVER w AS close_lag20,
        lag(adj_close, 60) OVER w AS close_lag60,
        lag(amount, 1) OVER w AS amount_lag1
    FROM dense
    WINDOW w AS (PARTITION BY symbol ORDER BY date_idx)
), returns AS (
    SELECT *,
        adj_close / NULLIF(close_lag1, 0) - 1.0 AS ret_1d,
        adj_close / NULLIF(close_lag3, 0) - 1.0 AS ret_3d,
        adj_close / NULLIF(close_lag5, 0) - 1.0 AS ret_5d,
        adj_close / NULLIF(close_lag10, 0) - 1.0 AS ret_10d,
        adj_close / NULLIF(close_lag20, 0) - 1.0 AS ret_20d,
        adj_close / NULLIF(close_lag60, 0) - 1.0 AS ret_60d,
        CASE WHEN adj_close > 0 AND close_lag1 > 0
             THEN ln(adj_close / close_lag1) ELSE NULL END AS log_ret_1d,
        adj_high / NULLIF(adj_low, 0) - 1.0 AS range_1d,
        (adj_close - adj_low) / NULLIF(adj_high - adj_low, 0) AS close_location_1d,
        (adj_close - adj_open) / NULLIF(adj_high - adj_low, 0) AS body_location_1d,
        (adj_high - greatest(adj_open, adj_close)) / NULLIF(adj_high - adj_low, 0) AS upper_shadow_1d,
        (least(adj_open, adj_close) - adj_low) / NULLIF(adj_high - adj_low, 0) AS lower_shadow_1d
    FROM lagged
), rolling AS (
    SELECT *,
        count(adj_close) OVER (PARTITION BY symbol ORDER BY date_idx ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS prior_valid_bars_20,
        median(amount) OVER (PARTITION BY symbol ORDER BY date_idx ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS amount_median20_prev,
        median(volume) OVER (PARTITION BY symbol ORDER BY date_idx ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS volume_median20_prev,
        median(range_1d) OVER (PARTITION BY symbol ORDER BY date_idx ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS range_median20_prev,
        stddev_samp(log_ret_1d) OVER (PARTITION BY symbol ORDER BY date_idx ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS volatility20_prev,
        stddev_samp(log_ret_1d) OVER (PARTITION BY symbol ORDER BY date_idx ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS volatility5_prev,
        sum(abs(log_ret_1d)) OVER (PARTITION BY symbol ORDER BY date_idx ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS absolute_path20,
        max(adj_high) OVER (PARTITION BY symbol ORDER BY date_idx ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS prev20_high,
        max(adj_high) OVER (PARTITION BY symbol ORDER BY date_idx ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING) AS prev60_high,
        min(adj_low) OVER (PARTITION BY symbol ORDER BY date_idx ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS prev20_low,
        avg(adj_close) OVER (PARTITION BY symbol ORDER BY date_idx ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS ma20_prev,
        avg(adj_close) OVER (PARTITION BY symbol ORDER BY date_idx ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING) AS ma60_prev,
        max(adj_high) OVER (PARTITION BY symbol ORDER BY date_idx ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS high5_including_current,
        avg(amount) OVER (PARTITION BY symbol ORDER BY date_idx ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS amount_mean5,
        avg(amount) OVER (PARTITION BY symbol ORDER BY date_idx ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS amount_mean20
    FROM returns
), state0 AS (
    SELECT *,
        amount / NULLIF(amount_median20_prev, 0) AS amount_ratio20,
        volume / NULLIF(volume_median20_prev, 0) AS volume_ratio20,
        range_1d / NULLIF(range_median20_prev, 0) AS range_ratio20,
        volatility5_prev / NULLIF(volatility20_prev, 0) AS volatility_ratio5_20,
        amount_mean5 / NULLIF(amount_mean20, 0) AS amount_ratio5_20,
        adj_close / NULLIF(prev20_high, 0) - 1.0 AS distance_prev20_high,
        adj_close / NULLIF(prev60_high, 0) - 1.0 AS distance_prev60_high,
        adj_close / NULLIF(prev20_low, 0) - 1.0 AS distance_prev20_low,
        adj_close / NULLIF(ma20_prev, 0) - 1.0 AS distance_ma20,
        adj_close / NULLIF(ma60_prev, 0) - 1.0 AS distance_ma60,
        adj_close / NULLIF(high5_including_current, 0) - 1.0 AS pullback_from_high5,
        CASE WHEN adj_close > 0 AND close_lag20 > 0 AND absolute_path20 > 0
             THEN abs(ln(adj_close / close_lag20)) / absolute_path20 ELSE NULL END AS trend_efficiency20,
        CASE WHEN amount > 0 AND amount_median20_prev > 0
             THEN ln(amount / amount_median20_prev) ELSE NULL END AS log_amount_ratio20,
        CASE WHEN volume > 0 AND volume_median20_prev > 0
             THEN ln(volume / volume_median20_prev) ELSE NULL END AS log_volume_ratio20,
        CASE WHEN range_1d >= 0 AND range_median20_prev > 0
             THEN ln(greatest(range_1d / range_median20_prev, 1e-8)) ELSE NULL END AS log_range_ratio20,
        CASE WHEN amount > 0 AND amount_lag1 > 0
             THEN ln(amount / amount_lag1) ELSE NULL END AS log_amount_change1
    FROM rolling
), future AS (
    SELECT *,
        {path_leads}
    FROM state0
    WINDOW w AS (PARTITION BY symbol ORDER BY date_idx)
), legal AS (
    SELECT *
    FROM future
    WHERE trade_date >= '{int(year)}-01-01' AND trade_date <= '{int(year)}-12-31'
      AND is_st = false AND is_suspended = false AND is_delisted = false
      AND bar_valid
      AND listed_open_days >= {int(minimum_listed_days)}
      AND prior_valid_bars_20 >= {int(minimum_prior_valid)}
      AND amount > 0 AND volume > 0
      AND amount_ratio20 > 0 AND volume_ratio20 > 0
      AND range_ratio20 >= 0 AND isfinite(ret_1d)
), ranked AS (
    SELECT *,
        percent_rank() OVER (PARTITION BY trade_date ORDER BY log_amount_ratio20) AS amount_shock_rank,
        percent_rank() OVER (PARTITION BY trade_date ORDER BY log_volume_ratio20) AS volume_shock_rank,
        percent_rank() OVER (PARTITION BY trade_date ORDER BY log_range_ratio20) AS range_shock_rank,
        percent_rank() OVER (PARTITION BY trade_date ORDER BY abs(ret_1d)) AS absolute_return_rank,
        percent_rank() OVER (PARTITION BY trade_date ORDER BY amount) AS amount_cross_section_rank,
        percent_rank() OVER (PARTITION BY trade_date ORDER BY ret_1d) AS ret1_rank,
        percent_rank() OVER (PARTITION BY trade_date ORDER BY ret_5d) AS ret5_rank,
        percent_rank() OVER (PARTITION BY trade_date ORDER BY distance_prev20_high) AS position20_rank,
        avg(ret_1d) OVER (PARTITION BY trade_date) AS market_ret_1d,
        avg(CASE WHEN ret_1d > 0 THEN 1.0 ELSE 0.0 END) OVER (PARTITION BY trade_date) AS market_breadth_up,
        stddev_samp(ret_1d) OVER (PARTITION BY trade_date) AS market_dispersion_1d,
        avg(ret_1d) OVER (PARTITION BY trade_date, industry) AS industry_ret_1d,
        count(*) OVER (PARTITION BY trade_date, industry) AS industry_member_count
    FROM legal
), attention AS (
    SELECT *,
        (amount_shock_rank + volume_shock_rank + range_shock_rank + absolute_return_rank) / 4.0 AS attention_score,
        ret_1d - market_ret_1d AS market_relative_ret_1d,
        ret_1d - industry_ret_1d AS industry_relative_ret_1d
    FROM ranked
), sequence0 AS (
    SELECT *,
        lag(attention_score) OVER w AS previous_attention_score,
        lag(date_idx) OVER w AS previous_legal_date_idx,
        max(attention_score) OVER (PARTITION BY symbol ORDER BY date_idx RANGE BETWEEN 4 PRECEDING AND CURRENT ROW) AS attention_max5,
        avg(attention_score) OVER (PARTITION BY symbol ORDER BY date_idx RANGE BETWEEN 4 PRECEDING AND CURRENT ROW) AS attention_mean5
    FROM attention
    WINDOW w AS (PARTITION BY symbol ORDER BY date_idx)
), onset AS (
    SELECT *,
        CASE WHEN previous_legal_date_idx = date_idx - 1
             THEN attention_score - previous_attention_score ELSE NULL END AS attention_velocity,
        (attention_score >= 0.90 AND
         (previous_legal_date_idx IS NULL OR previous_legal_date_idx <> date_idx - 1
          OR previous_attention_score < 0.90)) AS attention_onset
    FROM sequence0
), lifecycle AS (
    SELECT *,
        max(CASE WHEN attention_onset THEN date_idx ELSE NULL END)
            OVER (PARTITION BY symbol ORDER BY date_idx ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS last_attention_onset_idx
    FROM onset
), joined_pool AS (
    SELECT l.*, (q.symbol IS NOT NULL) AS in_formal_quality_pool
    FROM lifecycle l
    LEFT JOIN read_parquet({_sql_quote(formal_pool_path)}) q
      ON l.symbol = q.symbol AND l.trade_date = CAST(q.trade_date AS VARCHAR)
), outcomes0 AS (
    SELECT *,
        (entry_open > 0 AND entry_is_st = false AND entry_is_suspended = false
         AND entry_is_delisted = false) AS entry_observed_legal,
        (entry_open > 0 AND entry_is_st = false AND entry_is_suspended = false
         AND entry_is_delisted = false
         AND entry_open / NULLIF(adj_close, 0) - 1.0 < 0.095) AS entry_buyable_approx,
        {up5} AS up5_day,
        {up10} AS up10_day,
        {down3} AS down3_day,
        {down5} AS down5_day,
        {horizon_sql}
    FROM joined_pool
), outcomes AS (
    SELECT *,
        (up5_day IS NOT NULL AND (down3_day IS NULL OR up5_day < down3_day)) AS up5_before_down3,
        (up10_day IS NOT NULL AND (down5_day IS NULL OR up10_day < down5_day)) AS up10_before_down5,
        (up5_day IS NOT NULL AND down3_day IS NOT NULL AND up5_day = down3_day) AS up5_down3_same_day_ambiguous,
        (up10_day IS NOT NULL AND down5_day IS NOT NULL AND up10_day = down5_day) AS up10_down5_same_day_ambiguous,
        CASE WHEN last_attention_onset_idx IS NOT NULL AND date_idx - last_attention_onset_idx <= 20
             THEN date_idx - last_attention_onset_idx ELSE NULL END AS days_since_attention_onset
    FROM outcomes0
)
SELECT
    symbol, trade_date, date_idx, exchange, industry, listed_open_days,
    in_formal_quality_pool,
    raw_open, raw_high, raw_low, raw_close, adj_open, adj_high, adj_low, adj_close,
    volume, amount, adjust_factor,
    ret_1d, ret_3d, ret_5d, ret_10d, ret_20d, ret_60d,
    range_1d, close_location_1d, body_location_1d, upper_shadow_1d, lower_shadow_1d,
    volatility20_prev, volatility5_prev, amount_ratio20, volume_ratio20,
    range_ratio20, volatility_ratio5_20, amount_ratio5_20,
    log_amount_ratio20, log_volume_ratio20, log_range_ratio20, log_amount_change1,
    distance_prev20_high, distance_prev60_high, distance_prev20_low,
    distance_ma20, distance_ma60, pullback_from_high5, trend_efficiency20,
    amount_shock_rank, volume_shock_rank, range_shock_rank, absolute_return_rank,
    amount_cross_section_rank, ret1_rank, ret5_rank, position20_rank,
    attention_score, previous_attention_score, attention_velocity,
    attention_max5, attention_mean5, attention_onset, days_since_attention_onset,
    market_ret_1d, market_breadth_up, market_dispersion_1d,
    industry_ret_1d, industry_member_count,
    market_relative_ret_1d, industry_relative_ret_1d,
    entry_trade_date, entry_open, entry_is_st, entry_is_suspended, entry_is_delisted,
    entry_observed_legal, entry_buyable_approx,
    valid_close_days_20, up5_day, up10_day, down3_day, down5_day,
    up5_before_down3, up10_before_down5,
    up5_down3_same_day_ambiguous, up10_down5_same_day_ambiguous,
    terminal_log_return_5, terminal_log_return_10, terminal_log_return_20,
    mfe_5, mfe_10, mfe_20, mae_5, mae_10, mae_20,
    oracle_best_close_net_5, oracle_best_close_net_10, oracle_best_close_net_20,
    valid_close_days_5, valid_close_days_10,
    {path_columns}
FROM outcomes
"""


def _parquet_row_count(
    connection: duckdb.DuckDBPyConnection, path: Path
) -> int:
    return int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet({_sql_quote(path)})"
        ).fetchone()[0]
    )


def _existing_manifest_valid(
    manifest: Mapping[str, Any], *, fingerprint: str
) -> bool:
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("status") != "prepared"
        or str(manifest.get("experiment_fingerprint")) != str(fingerprint)
    ):
        return False
    dense = Path(str(dict(manifest.get("dense_base", {}) or {}).get("path", "")))
    if not dense.is_file():
        return False
    panels = list(manifest.get("panels", []) or [])
    if len(panels) != FORMAL_END_YEAR - FORMAL_START_YEAR + 1:
        return False
    return all(Path(str(record.get("path", ""))).is_file() for record in panels)


def prepare_daily_panel(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    qdp_root: str | Path = "quant_data_platform/data/qdp_v2",
    force: bool = False,
) -> dict[str, Any]:
    """Build adjusted broad-main-board state/path panels with bounded memory."""

    study_path = _resolve_path(study_path)
    output_root = _resolve_path(output_root)
    qdp_root = _resolve_path(qdp_root)
    study = load_study(study_path)
    input_contract, paths, fingerprint = _input_contract(study_path, qdp_root)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file() and not force:
        current = _read_json(manifest_path)
        if _existing_manifest_valid(current, fingerprint=fingerprint):
            return current
        if str(current.get("experiment_fingerprint", "")) not in {"", fingerprint}:
            raise ValueError("existing_hot_path_atlas_fingerprint_mismatch")

    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.json"
    dense_base_path = output_root / "dense_mainboard_2010_2025.parquet"
    connection = _connect(output_root, study)
    try:
        if force or not dense_base_path.is_file():
            _write_json(
                progress_path,
                {"status": "building_dense_base", "experiment_fingerprint": fingerprint},
            )
            _copy_query(connection, _dense_base_query(paths), dense_base_path)
        dense_rows = _parquet_row_count(connection, dense_base_path)
        dense_audit = connection.execute(
            f"""
            SELECT
                count(*) AS row_count,
                count(DISTINCT symbol) AS symbol_count,
                count(DISTINCT trade_date) AS date_count,
                min(trade_date) AS minimum_date,
                max(trade_date) AS maximum_date,
                count(*) FILTER (WHERE trade_date >= '2026-01-01') AS forbidden_rows,
                count(*) FILTER (WHERE bar_valid AND (adjust_factor IS NULL OR adjust_factor <= 0)) AS invalid_factor_rows,
                count(*) - count(DISTINCT symbol || '|' || trade_date) AS duplicate_keys
            FROM read_parquet({_sql_quote(dense_base_path)})
            """
        ).fetchdf().iloc[0].to_dict()
        if int(dense_audit["forbidden_rows"]) != 0:
            raise ValueError("dense_base_reads_forbidden_2026")
        if int(dense_audit["duplicate_keys"]) != 0:
            raise ValueError("dense_base_duplicate_keys")
        if int(dense_audit["invalid_factor_rows"]) != 0:
            raise ValueError("dense_base_invalid_adjustment_factor")
        if int(dense_rows) != int(dense_audit["row_count"]):
            raise AssertionError("dense_base_row_count_drift")

        universe = dict(study.get("universe", {}) or {})
        price = dict(study.get("price_and_path", {}) or {})
        formal_pool_path = _resolve_path(
            str(dict(study.get("source", {}) or {})["formal_quality_pool_index"])
        )
        if not formal_pool_path.is_file():
            raise FileNotFoundError("formal_quality_pool_comparison_index_missing")
        panel_root = output_root / "daily_panels"
        panel_root.mkdir(parents=True, exist_ok=True)
        panel_records: list[dict[str, Any]] = []
        for year in range(FORMAL_START_YEAR, FORMAL_END_YEAR + 1):
            panel_path = panel_root / f"state_path_{year}.parquet"
            if force or not panel_path.is_file():
                _write_json(
                    progress_path,
                    {
                        "status": "building_year_panel",
                        "year": int(year),
                        "completed_years": [
                            int(record["year"]) for record in panel_records
                        ],
                        "experiment_fingerprint": fingerprint,
                    },
                )
                query = _year_panel_query(
                    dense_base_path,
                    formal_pool_path,
                    year=year,
                    minimum_listed_days=int(
                        universe.get("minimum_listed_open_days", 60)
                    ),
                    minimum_prior_valid=int(
                        universe.get("minimum_prior_valid_bars_20", 15)
                    ),
                    cost_bps=float(price.get("round_trip_cost_bps", 60.0)),
                )
                _copy_query(connection, query, panel_path)
            audit = connection.execute(
                f"""
                SELECT
                    count(*) AS row_count,
                    count(DISTINCT symbol) AS symbol_count,
                    count(DISTINCT trade_date) AS date_count,
                    min(trade_date) AS minimum_date,
                    max(trade_date) AS maximum_date,
                    count(*) FILTER (WHERE trade_date < '{year}-01-01' OR trade_date > '{year}-12-31') AS outside_year_rows,
                    count(*) FILTER (WHERE entry_observed_legal) AS entry_observed_legal_rows,
                    count(*) FILTER (WHERE entry_buyable_approx) AS entry_buyable_approx_rows,
                    count(*) FILTER (WHERE valid_close_days_20 = 20) AS complete_close_path_rows,
                    count(*) FILTER (WHERE in_formal_quality_pool) AS formal_quality_pool_rows,
                    count(*) - count(DISTINCT symbol || '|' || trade_date) AS duplicate_keys
                FROM read_parquet({_sql_quote(panel_path)})
                """
            ).fetchdf().iloc[0].to_dict()
            if int(audit["outside_year_rows"]) != 0:
                raise ValueError(f"panel_outside_year_rows:{year}")
            if int(audit["duplicate_keys"]) != 0:
                raise ValueError(f"panel_duplicate_keys:{year}")
            panel_records.append(
                {
                    "year": int(year),
                    **_file_record(panel_path, include_hash=False),
                    "row_count": int(audit["row_count"]),
                    "symbol_count": int(audit["symbol_count"]),
                    "date_count": int(audit["date_count"]),
                    "minimum_date": str(audit["minimum_date"]),
                    "maximum_date": str(audit["maximum_date"]),
                    "entry_observed_legal_rows": int(
                        audit["entry_observed_legal_rows"]
                    ),
                    "entry_buyable_approx_rows": int(
                        audit["entry_buyable_approx_rows"]
                    ),
                    "complete_close_path_rows": int(
                        audit["complete_close_path_rows"]
                    ),
                    "formal_quality_pool_rows": int(
                        audit["formal_quality_pool_rows"]
                    ),
                }
            )

        total_rows = int(sum(record["row_count"] for record in panel_records))
        formal_rows = int(
            sum(record["formal_quality_pool_rows"] for record in panel_records)
        )
        manifest: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "status": "prepared",
            "study_id": STUDY_ID,
            "builder_version": BUILDER_VERSION,
            "experiment_fingerprint": fingerprint,
            "study": _file_record(study_path),
            "input_contract": input_contract,
            "dense_base": {
                **_file_record(dense_base_path, include_hash=False),
                **{key: _json_default(value) if isinstance(value, np.generic) else value for key, value in dense_audit.items()},
            },
            "panels": panel_records,
            "formal_signal_rows": total_rows,
            "formal_quality_pool_rows": formal_rows,
            "outside_formal_quality_pool_rows": total_rows - formal_rows,
            "semantics": {
                "price_adjustment": "raw_ohlc_times_adjust_factor",
                "time_axis": "dense_market_trading_days_from_security_status",
                "candidate_filter": "known non-ST non-suspended non-delisted main-board status; 60 listed open days; 15 valid prior bars; no quality/cap/liquidity threshold",
                "formal_quality_pool": "comparison flag only",
                "path": "next-market-day adjusted open anchor and D1-D20 adjusted high/low/close",
                "same_day_barrier_order": "ambiguous when upside and downside barriers first occur on the same daily bar",
                "training_performed": False,
                "portfolio_selection_performed": False,
                "forbidden_2026_rows": 0,
            },
        }
        _write_json(manifest_path, manifest)
        _write_json(
            progress_path,
            {
                "status": "prepared",
                "manifest": str(manifest_path.resolve()),
                "formal_signal_rows": total_rows,
            },
        )
        return manifest
    finally:
        connection.close()


def _hac_mean(values: Sequence[float], lag: int = 20) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    n = int(len(array))
    if n == 0:
        return {"mean": np.nan, "se": np.nan, "lcb_95": np.nan, "ucb_95": np.nan}
    mean = float(np.mean(array))
    if n == 1:
        return {"mean": mean, "se": np.nan, "lcb_95": np.nan, "ucb_95": np.nan}
    centered = array - mean
    maximum_lag = min(int(lag), n - 1)
    long_run = float(np.dot(centered, centered) / n)
    for offset in range(1, maximum_lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / n)
        long_run += 2.0 * (1.0 - offset / (maximum_lag + 1.0)) * covariance
    se = math.sqrt(max(long_run, 0.0) / n)
    return {
        "mean": mean,
        "se": se,
        "lcb_95": mean - 1.96 * se,
        "ucb_95": mean + 1.96 * se,
    }


def _panel_paths(manifest: Mapping[str, Any]) -> list[Path]:
    paths = [Path(str(record["path"])) for record in manifest.get("panels", [])]
    if not paths or not all(path.is_file() for path in paths):
        raise FileNotFoundError("prepared_hot_path_panels_missing")
    return paths


def _attention_date_table(
    connection: duckdb.DuckDBPyConnection,
    panel_paths: Sequence[Path],
    *,
    cost_bps: float,
) -> pd.DataFrame:
    scan = _parquet_scan(panel_paths)
    cost = float(cost_bps) / 10_000.0
    return connection.execute(
        f"""
        WITH base AS (
            SELECT *,
                CAST(least(9.0, greatest(0.0, floor(attention_score * 10.0))) AS INTEGER) AS attention_decile
            FROM {scan}
        ), expanded AS (
            SELECT *, 'all_broad_legal' AS pool_segment FROM base
            UNION ALL
            SELECT *, CASE WHEN in_formal_quality_pool
                           THEN 'inside_old_quality_pool'
                           ELSE 'outside_old_quality_pool' END AS pool_segment
            FROM base
        )
        SELECT
            trade_date,
            CAST(left(trade_date, 4) AS INTEGER) AS signal_year,
            attention_decile,
            pool_segment,
            count(*) AS candidate_rows,
            count(*) FILTER (WHERE entry_buyable_approx) AS buyable_rows,
            avg(exp(terminal_log_return_5) - 1.0 - {cost:.12g}) FILTER (WHERE entry_buyable_approx) AS net_return_5,
            avg(exp(terminal_log_return_10) - 1.0 - {cost:.12g}) FILTER (WHERE entry_buyable_approx) AS net_return_10,
            avg(exp(terminal_log_return_20) - 1.0 - {cost:.12g}) FILTER (WHERE entry_buyable_approx) AS net_return_20,
            avg(mfe_20) FILTER (WHERE entry_buyable_approx AND valid_close_days_20 = 20) AS mfe_20,
            avg(mae_20) FILTER (WHERE entry_buyable_approx AND valid_close_days_20 = 20) AS mae_20,
            avg(oracle_best_close_net_20) FILTER (WHERE entry_buyable_approx AND valid_close_days_20 = 20) AS oracle_best_close_net_20,
            avg(CAST(up5_before_down3 AS DOUBLE)) FILTER (WHERE entry_buyable_approx AND valid_close_days_20 = 20 AND NOT up5_down3_same_day_ambiguous) AS up5_before_down3_rate,
            avg(CAST(up10_before_down5 AS DOUBLE)) FILTER (WHERE entry_buyable_approx AND valid_close_days_20 = 20 AND NOT up10_down5_same_day_ambiguous) AS up10_before_down5_rate
        FROM expanded
        GROUP BY trade_date, signal_year, attention_decile, pool_segment
        ORDER BY trade_date, attention_decile, pool_segment
        """
    ).fetchdf()


def _summarize_attention_dates(date_table: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "net_return_5",
        "net_return_10",
        "net_return_20",
        "mfe_20",
        "mae_20",
        "oracle_best_close_net_20",
        "up5_before_down3_rate",
        "up10_before_down5_rate",
    )
    rows: list[dict[str, Any]] = []
    group_columns = ["attention_decile", "pool_segment"]
    for keys, group in date_table.groupby(group_columns, sort=True, observed=True):
        decile, pool_segment = keys
        row: dict[str, Any] = {
            "attention_decile": int(decile),
            "pool_segment": str(pool_segment),
            "dates": int(group["trade_date"].nunique()),
            "candidate_rows": int(group["candidate_rows"].sum()),
            "buyable_rows": int(group["buyable_rows"].sum()),
        }
        for metric in metrics:
            estimate = _hac_mean(group[metric].to_numpy(dtype=np.float64), lag=20)
            for name, value in estimate.items():
                row[f"{metric}_{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _annual_attention_summary(date_table: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "net_return_5",
        "net_return_10",
        "net_return_20",
        "mfe_20",
        "mae_20",
        "oracle_best_close_net_20",
        "up5_before_down3_rate",
        "up10_before_down5_rate",
    )
    aggregations: dict[str, str] = {metric: "mean" for metric in metrics}
    aggregations.update({"candidate_rows": "sum", "buyable_rows": "sum"})
    return (
        date_table.groupby(
            ["signal_year", "attention_decile", "pool_segment"],
            observed=True,
            sort=True,
        )
        .agg(aggregations)
        .reset_index()
    )


def _attention_threshold_table(
    connection: duckdb.DuckDBPyConnection,
    panel_paths: Sequence[Path],
    thresholds: Sequence[float],
    *,
    cost_bps: float,
) -> pd.DataFrame:
    scan = _parquet_scan(panel_paths)
    cost = float(cost_bps) / 10_000.0
    rows: list[pd.DataFrame] = []
    for threshold in thresholds:
        rows.append(
            connection.execute(
                f"""
                WITH base AS (
                    SELECT *, attention_score >= {float(threshold):.12g} AS selected
                    FROM {scan}
                    WHERE entry_buyable_approx AND valid_close_days_20 = 20
                ), daily AS (
                    SELECT
                        trade_date,
                        count(*) AS universe_rows,
                        count(*) FILTER (WHERE selected) AS selected_rows,
                        sum(CAST(up10_before_down5 AS INTEGER)) FILTER (WHERE NOT up10_down5_same_day_ambiguous) AS universe_ordered_events,
                        sum(CAST(up10_before_down5 AS INTEGER)) FILTER (WHERE selected AND NOT up10_down5_same_day_ambiguous) AS selected_ordered_events,
                        avg(exp(terminal_log_return_5) - 1.0 - {cost:.12g}) FILTER (WHERE selected) AS selected_net_return_5,
                        avg(exp(terminal_log_return_20) - 1.0 - {cost:.12g}) FILTER (WHERE selected) AS selected_net_return_20,
                        avg(CAST(up10_before_down5 AS DOUBLE)) FILTER (WHERE selected AND NOT up10_down5_same_day_ambiguous) AS selected_up10_before_down5_rate,
                        avg(CAST(up10_before_down5 AS DOUBLE)) FILTER (WHERE NOT up10_down5_same_day_ambiguous) AS universe_up10_before_down5_rate
                    FROM base GROUP BY trade_date
                )
                SELECT
                    {float(threshold):.12g} AS threshold,
                    trade_date,
                    selected_rows / NULLIF(universe_rows, 0)::DOUBLE AS selected_share,
                    selected_ordered_events / NULLIF(universe_ordered_events, 0)::DOUBLE AS opportunity_capture_share,
                    selected_net_return_5,
                    selected_net_return_20,
                    selected_up10_before_down5_rate,
                    universe_up10_before_down5_rate
                FROM daily ORDER BY trade_date
                """
            ).fetchdf()
        )
    date_table = pd.concat(rows, ignore_index=True)
    summary_rows: list[dict[str, Any]] = []
    for threshold, group in date_table.groupby("threshold", sort=True):
        row: dict[str, Any] = {"attention_threshold": float(threshold)}
        for metric in (
            "selected_share",
            "opportunity_capture_share",
            "selected_net_return_5",
            "selected_net_return_20",
            "selected_up10_before_down5_rate",
            "universe_up10_before_down5_rate",
        ):
            estimate = _hac_mean(group[metric].to_numpy(dtype=np.float64), lag=20)
            for name, value in estimate.items():
                row[f"{metric}_{name}"] = value
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _cluster_sample(
    connection: duckdb.DuckDBPyConnection,
    panel_paths: Sequence[Path],
    *,
    maximum_per_date_decile: int,
) -> pd.DataFrame:
    scan = _parquet_scan(panel_paths)
    path_columns = ",\n            ".join(
        f"p.{column}"
        for day in PATH_DAYS
        for column in (
            f"path_high_{day}",
            f"path_low_{day}",
            f"path_close_{day}",
        )
    )
    return connection.execute(
        f"""
        WITH eligible_keys AS (
            SELECT symbol, trade_date,
                CAST(least(9.0, greatest(0.0, floor(attention_score * 10.0))) AS INTEGER) AS attention_decile
            FROM {scan}
            WHERE entry_buyable_approx AND valid_close_days_20 = 20
              AND entry_open > 0
        ), sampled_keys AS (
            SELECT symbol, trade_date, attention_decile,
                row_number() OVER (
                    PARTITION BY trade_date, attention_decile
                    ORDER BY hash(symbol || '|' || trade_date)
                ) AS sample_order
            FROM eligible_keys
        )
        SELECT
            p.symbol, p.trade_date, p.date_idx,
            CAST(left(p.trade_date, 4) AS INTEGER) AS signal_year,
            k.attention_decile, p.in_formal_quality_pool, p.entry_open,
            p.attention_score, p.attention_velocity, p.attention_mean5,
            p.ret1_rank, p.ret5_rank, p.position20_rank,
            p.ret_1d, p.ret_5d, p.ret_20d,
            p.amount_shock_rank, p.volume_shock_rank, p.range_shock_rank,
            p.amount_cross_section_rank, p.close_location_1d,
            p.distance_prev20_high, p.distance_ma20, p.pullback_from_high5,
            p.trend_efficiency20, p.volatility20_prev,
            p.market_ret_1d, p.market_breadth_up, p.market_dispersion_1d,
            p.days_since_attention_onset,
            p.terminal_log_return_5, p.terminal_log_return_10, p.terminal_log_return_20,
            p.mfe_5, p.mfe_10, p.mfe_20, p.mae_5, p.mae_10, p.mae_20,
            p.oracle_best_close_net_20,
            p.up5_before_down3, p.up10_before_down5,
            {path_columns}
        FROM {scan} p
        INNER JOIN sampled_keys k USING(symbol, trade_date)
        WHERE k.sample_order <= {int(maximum_per_date_decile)}
        ORDER BY p.trade_date, k.attention_decile, p.symbol
        """
    ).fetchdf()


def _path_representation(
    frame: pd.DataFrame,
    discovery_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    entry = frame["entry_open"].to_numpy(dtype=np.float64)
    close = np.column_stack(
        [frame[f"path_close_{day}"].to_numpy(dtype=np.float64) for day in PATH_DAYS]
    )
    high = np.column_stack(
        [frame[f"path_high_{day}"].to_numpy(dtype=np.float64) for day in PATH_DAYS]
    )
    low = np.column_stack(
        [frame[f"path_low_{day}"].to_numpy(dtype=np.float64) for day in PATH_DAYS]
    )
    close_log = np.log(close / entry[:, None])
    high_log = np.log(high / entry[:, None])
    low_log = np.log(low / entry[:, None])
    if not (
        np.isfinite(close_log).all()
        and np.isfinite(high_log).all()
        and np.isfinite(low_log).all()
    ):
        raise ValueError("cluster_sample_contains_nonfinite_path")

    # Shape is normalized by its own RMS, with a floor that prevents a nearly
    # flat path from amplifying numerical noise.  Economic amplitude remains a
    # separate block, so clustering cannot silently discard return magnitude.
    rms = np.sqrt(np.mean(np.square(close_log), axis=1))
    shape_scale = np.maximum(rms, 0.02)
    shape = close_log / shape_scale[:, None]
    anchors = np.asarray([4, 9, 19], dtype=np.int64)
    running_high = np.maximum.accumulate(high_log, axis=1)[:, anchors]
    running_low = np.minimum.accumulate(low_log, axis=1)[:, anchors]
    amplitude = np.column_stack(
        [
            close_log[:, 4],
            close_log[:, 9],
            close_log[:, 19],
            running_high,
            running_low,
        ]
    )
    shape_scaler = StandardScaler().fit(shape[discovery_mask])
    amplitude_scaler = StandardScaler().fit(amplitude[discovery_mask])
    shape_z = shape_scaler.transform(shape) / math.sqrt(shape.shape[1])
    amplitude_z = amplitude_scaler.transform(amplitude) / math.sqrt(
        amplitude.shape[1]
    )
    combined = np.column_stack([shape_z, amplitude_z]).astype(np.float32)
    components = min(12, combined.shape[1])
    pca = PCA(n_components=components, random_state=0)
    pca.fit(combined[discovery_mask])
    representation = pca.transform(combined).astype(np.float32)
    metadata = {
        "shape_dimensions": int(shape.shape[1]),
        "amplitude_dimensions": int(amplitude.shape[1]),
        "pca_components": int(components),
        "pca_explained_variance_ratio": [
            float(value) for value in pca.explained_variance_ratio_
        ],
        "pca_explained_variance_total": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "shape_normalization": "future_close_log_path_divided_by_max(path_rms,0.02)",
        "block_weighting": "equal_total_shape_and_amplitude_euclidean_weight",
    }
    return representation, metadata


def _choose_path_clusters(
    representation: np.ndarray,
    discovery_mask: np.ndarray,
    cluster_candidates: Sequence[int],
    *,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    discovery_positions = np.flatnonzero(discovery_mask)
    if len(discovery_positions) < 10_000:
        raise ValueError("insufficient_path_cluster_discovery_rows")
    rng = np.random.default_rng(int(seed))
    fit_positions = discovery_positions
    if len(fit_positions) > 180_000:
        fit_positions = np.sort(rng.choice(fit_positions, 180_000, replace=False))
    evaluation_positions = discovery_positions
    if len(evaluation_positions) > 4_000:
        evaluation_positions = np.sort(
            rng.choice(evaluation_positions, 4_000, replace=False)
        )
    stability_positions = discovery_positions
    if len(stability_positions) > 20_000:
        stability_positions = np.sort(
            rng.choice(stability_positions, 20_000, replace=False)
        )

    rows: list[dict[str, Any]] = []
    fitted: dict[int, MiniBatchKMeans] = {}
    for cluster_count in cluster_candidates:
        first = MiniBatchKMeans(
            n_clusters=int(cluster_count),
            random_state=int(seed),
            batch_size=4096,
            n_init=5,
        ).fit(representation[fit_positions])
        second = MiniBatchKMeans(
            n_clusters=int(cluster_count),
            random_state=int(seed) + 101,
            batch_size=4096,
            n_init=5,
        ).fit(representation[fit_positions])
        evaluation_labels = first.predict(representation[evaluation_positions])
        stability_first = first.predict(representation[stability_positions])
        stability_second = second.predict(representation[stability_positions])
        silhouette = float(
            silhouette_score(
                representation[evaluation_positions], evaluation_labels
            )
        )
        stability = float(adjusted_rand_score(stability_first, stability_second))
        composite = silhouette + 0.10 * stability - 0.002 * int(cluster_count)
        rows.append(
            {
                "cluster_count": int(cluster_count),
                "silhouette": silhouette,
                "seed_stability_adjusted_rand": stability,
                "complexity_penalized_score": composite,
                "inertia": float(first.inertia_),
                "fit_rows": int(len(fit_positions)),
                "silhouette_rows": int(len(evaluation_positions)),
            }
        )
        fitted[int(cluster_count)] = first
    selection = pd.DataFrame(rows).sort_values(
        ["complexity_penalized_score", "cluster_count"],
        ascending=[False, True],
        kind="mergesort",
    )
    chosen_count = int(selection.iloc[0]["cluster_count"])
    model = fitted[chosen_count]
    labels = model.predict(representation).astype(np.int16)
    metadata = {
        "chosen_cluster_count": chosen_count,
        "selection_rule": "maximum silhouette + 0.10*seed_ARI - 0.002*K; ties favor smaller K",
        "fit_rows": int(len(fit_positions)),
        "discovery_rows": int(len(discovery_positions)),
        "random_seed": int(seed),
    }
    return labels, selection.sort_values("cluster_count"), metadata


def _canonicalize_cluster_labels(
    frame: pd.DataFrame, labels: np.ndarray, discovery_mask: np.ndarray
) -> tuple[np.ndarray, dict[int, int]]:
    work = pd.DataFrame(
        {
            "cluster": labels[discovery_mask],
            "terminal": frame.loc[
                discovery_mask, "terminal_log_return_20"
            ].to_numpy(dtype=np.float64),
        }
    )
    order = (
        work.groupby("cluster", sort=True)["terminal"]
        .mean()
        .sort_values(kind="mergesort")
        .index.tolist()
    )
    mapping = {int(old): int(new) for new, old in enumerate(order)}
    canonical = np.asarray([mapping[int(value)] for value in labels], dtype=np.int16)
    return canonical, mapping


def _cluster_outputs(
    frame: pd.DataFrame,
    labels: np.ndarray,
    *,
    discovery_years: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["path_cluster"] = labels
    discovery_set = set(int(year) for year in discovery_years)
    work["assignment_period"] = np.where(
        work["signal_year"].isin(discovery_set),
        "archetype_discovery",
        "fixed_center_assignment",
    )
    summary_metrics = [
        "terminal_log_return_5",
        "terminal_log_return_10",
        "terminal_log_return_20",
        "mfe_20",
        "mae_20",
        "oracle_best_close_net_20",
        "up5_before_down3",
        "up10_before_down5",
        "attention_score",
        "ret_5d",
        "distance_prev20_high",
        "close_location_1d",
    ]
    summary = (
        work.groupby(["assignment_period", "path_cluster"], sort=True)[
            summary_metrics
        ]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    center_rows: list[dict[str, Any]] = []
    for (period, cluster), group in work.groupby(
        ["assignment_period", "path_cluster"], sort=True
    ):
        entry = group["entry_open"].to_numpy(dtype=np.float64)
        for day in PATH_DAYS:
            center_rows.append(
                {
                    "assignment_period": str(period),
                    "path_cluster": int(cluster),
                    "path_day": int(day),
                    "mean_close_log_return": float(
                        np.mean(
                            np.log(
                                group[f"path_close_{day}"].to_numpy(
                                    dtype=np.float64
                                )
                                / entry
                            )
                        )
                    ),
                    "mean_high_log_return": float(
                        np.mean(
                            np.log(
                                group[f"path_high_{day}"].to_numpy(dtype=np.float64)
                                / entry
                            )
                        )
                    ),
                    "mean_low_log_return": float(
                        np.mean(
                            np.log(
                                group[f"path_low_{day}"].to_numpy(dtype=np.float64)
                                / entry
                            )
                        )
                    ),
                    "rows": int(len(group)),
                }
            )
    centers = pd.DataFrame(center_rows)
    distribution = (
        work.groupby(
            ["assignment_period", "attention_decile", "path_cluster"],
            observed=True,
            sort=True,
        )
        .size()
        .rename("rows")
        .reset_index()
    )
    totals = distribution.groupby(
        ["assignment_period", "attention_decile"], observed=True
    )["rows"].transform("sum")
    distribution["conditional_probability"] = distribution["rows"] / totals
    return summary, centers, distribution


def _analysis_markdown(
    manifest: Mapping[str, Any],
    attention_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    cluster_selection: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    cluster_metadata: Mapping[str, Any],
) -> str:
    lines = [
        "# Hot Path Atlas V1",
        "",
        "## Research object",
        "",
        "The signal universe is the broad point-in-time SH/SZ main board with known non-ST, non-suspended and non-delisted signal-day status. It does not inherit the old quality, size or liquidity screen. Prices are adjusted before returns are formed, and D1-D20 are global market-day offsets, so a suspension is missing/censored rather than silently skipped.",
        "",
        f"- broad legal signal rows: {int(manifest.get('formal_signal_rows', 0)):,}",
        f"- rows inside the old quality pool: {int(manifest.get('formal_quality_pool_rows', 0)):,}",
        f"- rows outside the old quality pool: {int(manifest.get('outside_formal_quality_pool_rows', 0)):,}",
        "- entry: next market-day adjusted open; T+1 legal path starts on D2",
        "- same-day upside/downside barrier hits are marked ambiguous rather than ordered from daily OHLC",
        "",
        "## Attention distribution",
        "",
        "Attention is a rank aggregation of stock-local amount, volume and range innovations plus absolute return. It is a continuous state coordinate, not a buy label. Net terminal returns below subtract the frozen 60 bp round-trip cost.",
        "",
    ]
    focus = attention_summary.loc[
        attention_summary["pool_segment"].eq("all_broad_legal"),
        [
            "attention_decile",
            "dates",
            "candidate_rows",
            "net_return_5_mean",
            "net_return_20_mean",
            "mfe_20_mean",
            "mae_20_mean",
            "up10_before_down5_rate_mean",
        ],
    ]
    if focus.empty:
        focus = attention_summary[
            [
                "attention_decile",
                "dates",
                "candidate_rows",
                "net_return_5_mean",
                "net_return_20_mean",
                "mfe_20_mean",
                "mae_20_mean",
                "up10_before_down5_rate_mean",
            ]
        ]
    lines.append(focus.to_markdown(index=False, floatfmt=".4f"))
    lines.extend(
        [
            "",
            "## Opportunity concentration",
            "",
            threshold_summary.to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Future path archetypes",
            "",
            "Path clusters are fit only on 2012-2018 after separating normalized path shape from economic amplitude. Fixed centers are then assigned to 2019-2025. These are descriptions of future paths, not names such as 'main rise' or 'washout' imposed in advance.",
            "",
            cluster_selection.to_markdown(index=False, floatfmt=".4f"),
            "",
            f"Chosen K: {cluster_metadata.get('chosen_cluster_count')}",
            "",
        ]
    )
    cluster_focus_columns = [
        column
        for column in (
            "assignment_period",
            "path_cluster",
            "terminal_log_return_20_count",
            "terminal_log_return_5_mean",
            "terminal_log_return_20_mean",
            "mfe_20_mean",
            "mae_20_mean",
            "up10_before_down5_mean",
            "attention_score_mean",
        )
        if column in cluster_summary.columns
    ]
    lines.append(
        cluster_summary[cluster_focus_columns].to_markdown(
            index=False, floatfmt=".4f"
        )
    )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This stage can show whether opportunity and path type concentrate in observable states. It does not authorize a strategy: the path archetypes still contain future information, the entry limit-up flag is an approximation, and no causal exit rule or continuous account has been selected. The next stage must test whether signal-time state can predict these fixed path laws in rolling years and whether a frozen rule has positive net utility.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze_daily_panel(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study_path = _resolve_path(study_path)
    output_root = _resolve_path(output_root)
    study = load_study(study_path)
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("hot_path_atlas_manifest_missing")
    manifest = _read_json(manifest_path)
    panel_paths = _panel_paths(manifest)
    analysis_root = output_root / "analysis"
    analysis_manifest_path = analysis_root / "manifest.json"
    if analysis_manifest_path.is_file() and not force:
        current = _read_json(analysis_manifest_path)
        if (
            current.get("schema") == ANALYSIS_SCHEMA
            and current.get("status") == "completed"
            and current.get("experiment_fingerprint")
            == manifest.get("experiment_fingerprint")
        ):
            return current
    analysis_root.mkdir(parents=True, exist_ok=True)
    connection = _connect(output_root, study)
    try:
        price = dict(study.get("price_and_path", {}) or {})
        analysis = dict(study.get("analysis", {}) or {})
        date_table = _attention_date_table(
            connection,
            panel_paths,
            cost_bps=float(price.get("round_trip_cost_bps", 60.0)),
        )
        attention_summary = _summarize_attention_dates(date_table)
        annual_attention = _annual_attention_summary(date_table)
        threshold_summary = _attention_threshold_table(
            connection,
            panel_paths,
            [float(value) for value in analysis.get("attention_quantiles", [])],
            cost_bps=float(price.get("round_trip_cost_bps", 60.0)),
        )
        sample = _cluster_sample(
            connection,
            panel_paths,
            maximum_per_date_decile=int(
                analysis.get("maximum_cluster_rows_per_date_attention_decile", 8)
            ),
        )
    finally:
        connection.close()

    discovery_years = [
        int(value)
        for value in dict(study.get("period", {}) or {}).get(
            "path_archetype_discovery_years", []
        )
    ]
    discovery_mask = sample["signal_year"].isin(discovery_years).to_numpy()
    representation, representation_metadata = _path_representation(
        sample, discovery_mask
    )
    labels, cluster_selection, cluster_metadata = _choose_path_clusters(
        representation,
        discovery_mask,
        [int(value) for value in analysis.get("path_cluster_candidates", [])],
        seed=int(analysis.get("random_seed", 17)),
    )
    labels, canonical_mapping = _canonicalize_cluster_labels(
        sample, labels, discovery_mask
    )
    cluster_summary, cluster_centers, attention_cluster_distribution = (
        _cluster_outputs(sample, labels, discovery_years=discovery_years)
    )
    sample_assignments = sample[
        [
            "symbol",
            "trade_date",
            "date_idx",
            "signal_year",
            "attention_decile",
            "in_formal_quality_pool",
            "attention_score",
            "attention_velocity",
            "attention_mean5",
            "amount_shock_rank",
            "volume_shock_rank",
            "range_shock_rank",
            "amount_cross_section_rank",
            "ret1_rank",
            "ret5_rank",
            "position20_rank",
            "ret_1d",
            "ret_5d",
            "ret_20d",
            "distance_prev20_high",
            "distance_ma20",
            "pullback_from_high5",
            "close_location_1d",
            "trend_efficiency20",
            "volatility20_prev",
            "market_ret_1d",
            "market_breadth_up",
            "market_dispersion_1d",
            "days_since_attention_onset",
            "terminal_log_return_5",
            "terminal_log_return_10",
            "terminal_log_return_20",
            "mfe_20",
            "mae_20",
            "up5_before_down3",
            "up10_before_down5",
        ]
    ].copy()
    sample_assignments["path_cluster"] = labels
    sample_assignments.to_parquet(
        analysis_root / "path_cluster_assignments.parquet",
        index=False,
        compression="zstd",
    )
    _write_csv(analysis_root / "attention_date_table.csv", date_table)
    _write_csv(analysis_root / "attention_decile_summary.csv", attention_summary)
    _write_csv(analysis_root / "attention_decile_annual.csv", annual_attention)
    _write_csv(
        analysis_root / "attention_opportunity_concentration.csv", threshold_summary
    )
    _write_csv(analysis_root / "path_cluster_selection.csv", cluster_selection)
    _write_csv(analysis_root / "path_cluster_summary.csv", cluster_summary)
    _write_csv(analysis_root / "path_cluster_centers.csv", cluster_centers)
    _write_csv(
        analysis_root / "attention_path_cluster_distribution.csv",
        attention_cluster_distribution,
    )
    combined_cluster_metadata = {
        **representation_metadata,
        **cluster_metadata,
        "canonical_label_mapping": {
            str(key): int(value) for key, value in canonical_mapping.items()
        },
        "sample_rows": int(len(sample)),
        "discovery_rows": int(discovery_mask.sum()),
    }
    _write_json(analysis_root / "path_cluster_metadata.json", combined_cluster_metadata)
    report = _analysis_markdown(
        manifest,
        attention_summary,
        threshold_summary,
        cluster_selection,
        cluster_summary,
        combined_cluster_metadata,
    )
    report_path = analysis_root / "research_record.md"
    report_path.write_text(report, encoding="utf-8")
    result = {
        "schema": ANALYSIS_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "experiment_fingerprint": manifest.get("experiment_fingerprint"),
        "sample_rows": int(len(sample)),
        "path_cluster": combined_cluster_metadata,
        "outputs": {
            "attention_decile_summary": str(
                (analysis_root / "attention_decile_summary.csv").resolve()
            ),
            "attention_opportunity_concentration": str(
                (analysis_root / "attention_opportunity_concentration.csv").resolve()
            ),
            "path_cluster_assignments": str(
                (analysis_root / "path_cluster_assignments.parquet").resolve()
            ),
            "path_cluster_summary": str(
                (analysis_root / "path_cluster_summary.csv").resolve()
            ),
            "research_record": str(report_path.resolve()),
        },
        "training_performed": False,
        "portfolio_selection_performed": False,
        "profit_claim_allowed": False,
    }
    _write_json(analysis_manifest_path, result)
    return result


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    qdp_root: str | Path = "quant_data_platform/data/qdp_v2",
    force_prepare: bool = False,
    force_analysis: bool = False,
) -> dict[str, Any]:
    prepared = prepare_daily_panel(
        study_path=study_path,
        output_root=output_root,
        qdp_root=qdp_root,
        force=force_prepare,
    )
    analysis = analyze_daily_panel(
        study_path=study_path,
        output_root=output_root,
        force=force_analysis,
    )
    return {"prepared": prepared, "analysis": analysis}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and analyze an adjusted daily hot-stock path atlas."
    )
    parser.add_argument(
        "command", choices=("prepare", "analyze", "run"), nargs="?", default="run"
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--qdp-root", default=str(WORKSPACE_ROOT / "quant_data_platform/data/qdp_v2")
    )
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-analysis", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_daily_panel(
            study_path=args.study,
            output_root=args.output_root,
            qdp_root=args.qdp_root,
            force=bool(args.force_prepare),
        )
    elif args.command == "analyze":
        result = analyze_daily_panel(
            study_path=args.study,
            output_root=args.output_root,
            force=bool(args.force_analysis),
        )
    else:
        result = run_study(
            study_path=args.study,
            output_root=args.output_root,
            qdp_root=args.qdp_root,
            force_prepare=bool(args.force_prepare),
            force_analysis=bool(args.force_analysis),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return result


if __name__ == "__main__":
    main()
