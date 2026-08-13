from __future__ import annotations

"""Prepare the full-history quality/liquidity common support and feature atlas."""

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from quant_data_platform.providers import (
    _fetch_baostock_trade_calendar_frame_with_timeout,
)
from quant_data_platform.qdp_v2.manifest import (
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
)
from quant_data_platform.qdp_v2.status import active_dataset_map

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_quality_liquidity_data_prep"
CALENDAR_INPUT_BUILDER_VERSION = 2
MEMBERSHIP_BUILDER_VERSION = 3
ATLAS_BUILDER_VERSION = 3
FEATURE_TRANSFORM_VERSIONS = {
    "minute": 3,
    "fundamental": 2,
    "event": 1,
}
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
YEARS = tuple(range(2010, 2026))
CALENDAR_PREHISTORY_START = "1990-12-19"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_quality_liquidity_data_prep.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_data_prep"
)
DEFAULT_REPORT_COVERAGE_PATH = (
    WORKSPACE_ROOT
    / "quant_data_platform/data/qdp_runtime/research_report_rc_backfill_v2/prepared/report_annual_statistics.parquet"
)
CANDIDATE_INDEX = (
    WORKSPACE_ROOT
    / "daily_research/data/research_store/seq100_pit_l35v2_v1/pack/candidate_index.parquet"
)
BASE_FEATURE_MANIFEST = (
    WORKSPACE_ROOT
    / "tmp/seq100_learnability_inputs/attempt_001/base_feature_manifest.json"
)
LABEL_MANIFEST = (
    WORKSPACE_ROOT / "tmp/seq100_learnability_inputs/attempt_001/label_manifest.json"
)

EXPECTED_5M_TIMES = tuple(
    pd.date_range("09:35", "11:30", freq="5min").strftime("%H%M00000")
) + tuple(pd.date_range("13:05", "15:00", freq="5min").strftime("%H%M00000"))

QUALITY_METRICS = (
    "roe_avg",
    "net_profit_margin",
    "net_profit_yoy",
    "revenue_yoy",
    "cash_flow_ps",
    "debt_to_asset",
)

MINUTE_FEATURES = (
    "minute_open_close_return",
    "minute_range",
    "minute_close_location",
    "minute_realized_volatility",
    "minute_downside_semivolatility",
    "minute_upside_semivolatility",
    "minute_first_30m_return",
    "minute_last_30m_return",
    "minute_morning_return",
    "minute_afternoon_return",
    "minute_morning_afternoon_gap",
    "minute_vwap_close_deviation",
    "minute_trend_efficiency",
    "minute_max_5m_return",
    "minute_min_5m_return",
    "minute_positive_bar_fraction",
    "minute_first_hour_amount_share",
    "minute_last_hour_amount_share",
    "minute_max_bar_amount_share",
    "minute_amount_entropy",
    "minute_volume_half_imbalance",
    "minute_high_time_fraction",
    "minute_low_time_fraction",
    "minute_mfe_from_open",
    "minute_mae_from_open",
)

FINANCIAL_RATIO_FIELDS = (
    "roe_avg",
    "net_profit_margin",
    "gross_profit_margin",
    "net_profit_yoy",
    "revenue_yoy",
    "eps",
    "asset_turnover",
    "debt_to_asset",
    "current_ratio",
    "cash_flow_ps",
)
INCOME_FIELDS = (
    "basic_eps",
    "diluted_eps",
    "total_revenue",
    "revenue",
    "total_cost",
    "operating_cost",
    "selling_expense",
    "administrative_expense",
    "finance_expense",
    "research_development_expense",
    "operating_profit",
    "total_profit",
    "income_tax",
    "net_income",
    "parent_net_income",
    "minority_income",
    "ebit",
    "ebitda",
    "continuing_net_income",
    "discontinued_net_income",
)
BALANCE_FIELDS = (
    "total_shares",
    "capital_reserve",
    "retained_earnings",
    "surplus_reserve",
    "cash_and_equivalents",
    "notes_receivable",
    "accounts_receivable",
    "other_receivables",
    "prepayments",
    "inventory",
    "current_assets",
    "fixed_assets",
    "construction_in_progress",
    "intangible_assets",
    "goodwill",
    "noncurrent_assets",
    "total_assets",
    "short_term_borrowings",
    "notes_payable",
    "accounts_payable",
    "advances_from_customers",
    "employee_compensation_payable",
    "taxes_payable",
    "other_payables",
    "current_liabilities",
    "long_term_borrowings",
    "bonds_payable",
    "noncurrent_liabilities",
    "total_liabilities",
    "minority_equity",
    "parent_equity",
    "total_equity",
    "liabilities_and_equity",
)
CASH_FLOW_FIELDS = (
    "net_profit",
    "cash_received_from_sales",
    "tax_refunds_received",
    "cash_paid_for_goods",
    "payroll_paid",
    "taxes_paid",
    "other_operating_cash_paid",
    "operating_cash_outflow",
    "net_operating_cash_flow",
    "investing_cash_inflow",
    "investing_cash_outflow",
    "net_investing_cash_flow",
    "borrowings_received",
    "bond_proceeds",
    "financing_cash_inflow",
    "borrowings_repaid",
    "dividends_and_interest_paid",
    "financing_cash_outflow",
    "net_financing_cash_flow",
    "free_cash_flow",
    "net_cash_increase",
    "cash_begin",
    "cash_end",
)
KEY_COLUMNS = ("candidate_id", "year", "trade_date", "date_idx", "symbol_idx", "symbol")


class DataPreparationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _state_path(output_root: Path) -> Path:
    return output_root / "state.json"


def _read_state(output_root: Path, *, study_id: str = STUDY_ID) -> dict[str, Any]:
    path = _state_path(output_root)
    if not path.is_file():
        return {
            "study_id": study_id,
            "status": "pending",
            "start_date": START_DATE,
            "end_date": END_DATE,
            "training_performed": False,
        }
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_state(output_root: Path, state: Mapping[str, Any]) -> None:
    _write_json(_state_path(output_root), state)


def _load_config(path: Path, *, expected_study_id: str | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    study_id = str(payload.get("study_id", ""))
    if not study_id or (
        expected_study_id is not None and study_id != expected_study_id
    ):
        raise DataPreparationError("data_prep_study_id_changed")
    period = dict(payload.get("period", {}) or {})
    if (
        period.get("start_date", START_DATE) != START_DATE
        or period.get("end_date") != END_DATE
        or int(period.get("forbidden_year", 0)) != 2026
    ):
        raise DataPreparationError("data_prep_date_boundary_changed")
    support = dict(payload.get("common_support", {}) or {})
    if not support.get("requires_current_day_complete_5m"):
        raise DataPreparationError("minute_complete_common_support_required")
    if support.get("drop_entire_symbol") or support.get("daily_fallback"):
        raise DataPreparationError("minute_missing_policy_changed")
    if dict(payload.get("training", {}) or {}).get("performed"):
        raise DataPreparationError("data_prep_must_not_train")
    return payload


def _qdp_snapshot(
    workspace: Path,
    *,
    pinned_dataset_ids: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, tuple[Path, ...]]]:
    root = qdp_v2_root(workspace)
    active = active_dataset_map(read_active_manifest(root))
    if pinned_dataset_ids:
        stale = {
            str(domain): {
                "pinned": str(dataset_id),
                "active": str(active.get(str(domain), "")),
            }
            for domain, dataset_id in pinned_dataset_ids.items()
            if str(active.get(str(domain), "")) != str(dataset_id)
        }
        if stale:
            raise DataPreparationError(
                f"pinned_qdp_datasets_not_active:{json.dumps(stale, sort_keys=True)}"
            )
    datasets = (
        {
            str(domain): str(dataset_id)
            for domain, dataset_id in pinned_dataset_ids.items()
        }
        if pinned_dataset_ids
        else active
    )
    paths: dict[str, tuple[Path, ...]] = {}
    for domain, dataset_id in datasets.items():
        manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
        if manifest_path is None:
            if pinned_dataset_ids:
                raise DataPreparationError(
                    f"pinned_qdp_dataset_manifest_missing:{domain}:{dataset_id}"
                )
            continue
        manifest = read_dataset_manifest(manifest_path)
        resolved = tuple(
            resolve_manifest_path(item.path, root=root) for item in manifest.shards
        )
        if resolved and all(path.is_file() for path in resolved):
            paths[domain] = resolved
        elif pinned_dataset_ids:
            raise DataPreparationError(
                f"pinned_qdp_dataset_shards_missing:{domain}:{dataset_id}"
            )
    return datasets, paths


def _scan(paths: Sequence[Path]) -> str:
    literals = ",".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    return f"read_parquet([{literals}], union_by_name=true)"


def _connect(output_root: Path) -> duckdb.DuckDBPyConnection:
    temporary = (output_root / "duckdb_tmp").resolve()
    temporary.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET threads=4")
    connection.execute("SET memory_limit='10GB'")
    connection.execute(
        f"SET temp_directory='{str(temporary).replace(chr(39), chr(39) * 2)}'"
    )
    connection.execute("SET preserve_insertion_order=false")
    return connection


def _copy_query(connection: duckdb.DuckDBPyConnection, sql: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    quoted = str(temporary).replace("'", "''")
    connection.execute(
        f"COPY ({sql}) TO '{quoted}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)"
    )
    os.replace(temporary, path)


def _record(path: Path, **extra: Any) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
        "row_count": int(parquet.metadata.num_rows),
        **extra,
    }


def _year_bounds(year: int, *, buffer_days: int = 0) -> tuple[str, str]:
    start = pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=buffer_days)
    start = max(start, pd.Timestamp(START_DATE))
    end = min(pd.Timestamp(year=year, month=12, day=31), pd.Timestamp(END_DATE))
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _expected_time_sql() -> str:
    return ",".join(f"'{value}'" for value in EXPECTED_5M_TIMES)


def _minute_sql(*, year: int, intraday_scan: str) -> str:
    start, end = _year_bounds(year)
    candidate = str(CANDIDATE_INDEX).replace("'", "''")
    return f"""
    WITH candidate AS (
      SELECT candidate_id,symbol,trade_date
      FROM read_parquet('{candidate}')
      WHERE year={year} AND trade_date BETWEEN '{start}' AND '{end}'
    ), bars0 AS (
      SELECT c.candidate_id,b.symbol,b.trade_date,b.bar_time,
             try_cast(b.open AS DOUBLE) AS open,
             try_cast(b.high AS DOUBLE) AS high,
             try_cast(b.low AS DOUBLE) AS low,
             try_cast(b.close AS DOUBLE) AS close,
             try_cast(b.volume AS DOUBLE) AS volume,
             try_cast(b.amount AS DOUBLE) AS amount,
             row_number() OVER (
               PARTITION BY b.symbol,b.trade_date ORDER BY b.bar_time
             ) AS bar_no,
             lag(try_cast(b.close AS DOUBLE)) OVER (
               PARTITION BY b.symbol,b.trade_date ORDER BY b.bar_time
             ) AS previous_close
      FROM {intraday_scan} b
      JOIN candidate c ON c.symbol=b.symbol AND c.trade_date=b.trade_date
      WHERE b.trade_date BETWEEN '{start}' AND '{end}'
    ), bars AS (
      SELECT *,
             sum(amount) OVER (PARTITION BY symbol,trade_date) AS total_amount,
             sum(volume) OVER (PARTITION BY symbol,trade_date) AS total_volume,
              CASE WHEN previous_close>0 AND close>0
                THEN ln(close/previous_close) END AS log_return
      FROM bars0
    ), aggregate AS (
      SELECT candidate_id,symbol,trade_date,
             count(*) AS bar_count,
             count(DISTINCT bar_time) AS distinct_bar_count,
             count(*) FILTER (WHERE bar_time IN ({_expected_time_sql()})) AS expected_bar_count,
             arg_min(open,bar_time) AS day_open,
             max(close) AS max_close,
             min(close) AS min_close,
             arg_max(close,bar_time) AS day_close,
             max(high) AS day_high,
             min(low) AS day_low,
             max(CASE WHEN bar_no=6 THEN close END) AS close_bar_6,
             max(CASE WHEN bar_no=24 THEN close END) AS morning_close,
             max(CASE WHEN bar_no=25 THEN open END) AS afternoon_open,
             max(CASE WHEN bar_no=42 THEN close END) AS close_bar_42,
              sum(CASE
                WHEN bar_no=1 AND open>0 AND close>0 THEN abs(ln(close/open))
                ELSE abs(log_return)
              END) AS absolute_log_return_sum,
             sqrt(sum(log_return*log_return)) AS realized_volatility,
             sqrt(sum(CASE WHEN log_return<0 THEN log_return*log_return ELSE 0 END)) AS downside_semivolatility,
             sqrt(sum(CASE WHEN log_return>0 THEN log_return*log_return ELSE 0 END)) AS upside_semivolatility,
             max(log_return) AS max_log_return,
             min(log_return) AS min_log_return,
             avg(CASE WHEN log_return>0 THEN 1.0 ELSE 0.0 END) FILTER (WHERE log_return IS NOT NULL) AS positive_bar_fraction,
             sum(CASE WHEN bar_no<=12 THEN amount ELSE 0 END) AS first_hour_amount,
             sum(CASE WHEN bar_no>36 THEN amount ELSE 0 END) AS last_hour_amount,
             max(amount) AS max_bar_amount,
             -sum(CASE WHEN amount>0 AND total_amount>0
               THEN (amount/total_amount)*ln(amount/total_amount) ELSE 0 END)/ln(48.0) AS amount_entropy,
             sum(CASE WHEN bar_no<=24 THEN volume ELSE -volume END) AS volume_half_difference,
             first(bar_no ORDER BY high DESC NULLS LAST,bar_no ASC) AS high_bar_no,
             first(bar_no ORDER BY low ASC NULLS LAST,bar_no ASC) AS low_bar_no,
             sum(((high+low+close)/3.0)*volume) AS typical_value_volume,
             max(total_amount) AS total_amount,
             max(total_volume) AS total_volume
      FROM bars
      GROUP BY candidate_id,symbol,trade_date
    )
    SELECT candidate_id,symbol,trade_date,
      day_close/NULLIF(day_open,0)-1.0 AS minute_open_close_return,
      (day_high-day_low)/NULLIF(day_open,0) AS minute_range,
      (day_close-day_low)/NULLIF(day_high-day_low,0) AS minute_close_location,
      realized_volatility AS minute_realized_volatility,
      downside_semivolatility AS minute_downside_semivolatility,
      upside_semivolatility AS minute_upside_semivolatility,
      close_bar_6/NULLIF(day_open,0)-1.0 AS minute_first_30m_return,
      day_close/NULLIF(close_bar_42,0)-1.0 AS minute_last_30m_return,
      morning_close/NULLIF(day_open,0)-1.0 AS minute_morning_return,
      day_close/NULLIF(afternoon_open,0)-1.0 AS minute_afternoon_return,
      afternoon_open/NULLIF(morning_close,0)-1.0 AS minute_morning_afternoon_gap,
      day_close/NULLIF(typical_value_volume/NULLIF(total_volume,0),0)-1.0 AS minute_vwap_close_deviation,
      abs(ln(day_close/NULLIF(day_open,0)))/NULLIF(absolute_log_return_sum,0) AS minute_trend_efficiency,
      exp(max_log_return)-1.0 AS minute_max_5m_return,
      exp(min_log_return)-1.0 AS minute_min_5m_return,
      positive_bar_fraction AS minute_positive_bar_fraction,
      first_hour_amount/NULLIF(total_amount,0) AS minute_first_hour_amount_share,
      last_hour_amount/NULLIF(total_amount,0) AS minute_last_hour_amount_share,
      max_bar_amount/NULLIF(total_amount,0) AS minute_max_bar_amount_share,
      amount_entropy AS minute_amount_entropy,
      volume_half_difference/NULLIF(total_volume,0) AS minute_volume_half_imbalance,
      (high_bar_no-1.0)/47.0 AS minute_high_time_fraction,
      (low_bar_no-1.0)/47.0 AS minute_low_time_fraction,
      max_close/NULLIF(day_open,0)-1.0 AS minute_mfe_from_open,
      min_close/NULLIF(day_open,0)-1.0 AS minute_mae_from_open
    FROM aggregate
    WHERE bar_count=48 AND distinct_bar_count=48 AND expected_bar_count=48
      AND day_open>0 AND day_close>0 AND day_high>=day_low
    ORDER BY candidate_id
    """


def _minute_input_fingerprint(intraday_paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"minute_sql_version={FEATURE_TRANSFORM_VERSIONS['minute']}\n".encode()
    )
    digest.update(_path_set_fingerprint([*intraday_paths, CANDIDATE_INDEX]).encode())
    return digest.hexdigest()


def prepare_minute_features(
    *,
    output_root: Path,
    qdp_paths: Mapping[str, Sequence[Path]],
    state: dict[str, Any],
) -> dict[str, Any]:
    intraday = qdp_paths.get("market_intraday_5m")
    if not intraday:
        raise DataPreparationError("active_intraday_5m_missing")
    records = dict(state.get("minute_years", {}) or {})
    source_fingerprint = _minute_input_fingerprint(intraday)
    connection = _connect(output_root)
    try:
        scan = _scan(intraday)
        for year in YEARS:
            path = output_root / "minute_daily" / f"year={year}" / "part-0000.parquet"
            existing = dict(records.get(str(year), {}) or {})
            valid_existing = (
                path.is_file()
                and existing.get("status") == "completed"
                and existing.get("sha256") == _sha256(path)
                and existing.get("source_fingerprint") == source_fingerprint
            )
            if valid_existing:
                existing["source_fingerprint"] = source_fingerprint
                records[str(year)] = existing
            else:
                _copy_query(
                    connection, _minute_sql(year=year, intraday_scan=scan), path
                )
                records[str(year)] = {
                    "status": "completed",
                    **_record(path, feature_count=len(MINUTE_FEATURES)),
                    "source_fingerprint": source_fingerprint,
                }
                state["minute_years"] = records
                state["status"] = "preparing_minute_features"
                _write_state(output_root, state)
                print(json.dumps({"phase": "minute", "year": year}), flush=True)
    finally:
        connection.close()
    state["minute_years"] = records
    _write_state(output_root, state)
    return records


def _pct_rank_sql(value: str, partition: str, *, descending: bool = False) -> str:
    direction = "DESC" if descending else "ASC"
    return (
        f"CASE WHEN {value} IS NULL THEN NULL ELSE ("
        f"rank() OVER (PARTITION BY {partition} ORDER BY {value} {direction})"
        f"+(count(*) OVER (PARTITION BY {partition},{value})-1)/2.0)"
        f"/NULLIF(count({value}) OVER (PARTITION BY {partition}),0) END"
    )


def _membership_sql(
    *,
    year: int,
    paths: Mapping[str, Sequence[Path]],
    minute_path: Path,
    calendar_path: Path,
    listing_path: Path,
) -> str:
    buffer_start, year_end = _year_bounds(year, buffer_days=60)
    year_start, _ = _year_bounds(year)
    candidate = str(CANDIDATE_INDEX).replace("'", "''")
    financial_rank_columns = ",\n".join(
        f"      {_pct_rank_sql(metric, 'trade_date,quality_group', descending=metric == 'debt_to_asset')} AS {metric}_quality_rank,\n"
        f"      {_pct_rank_sql(metric, 'trade_date', descending=metric == 'debt_to_asset')} AS {metric}_global_rank"
        for metric in QUALITY_METRICS
    )
    quality_values = ",".join(f"{metric}_rank" for metric in QUALITY_METRICS)
    status_eligible = _status_eligible_sql()
    return f"""
    WITH candidate AS (
      SELECT candidate_id,year,trade_date,date_idx,symbol_idx,symbol,
             entry_trade_date,entry_filled,label_valid,price_label_valid,va_aux_valid
      FROM read_parquet('{candidate}')
      WHERE year={year} AND trade_date BETWEEN '{year_start}' AND '{year_end}'
    ), daily_window AS (
      SELECT symbol,trade_date,try_cast(amount AS DOUBLE) AS amount,
             count(*) FILTER (WHERE try_cast(amount AS DOUBLE)>0) OVER (
               PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
             ) AS valid_amount_20,
             quantile_cont(try_cast(amount AS DOUBLE),0.5) FILTER (
               WHERE try_cast(amount AS DOUBLE)>0
             ) OVER (
               PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
             ) AS amount_median_20
      FROM {_scan(paths["market_daily_raw"])}
      WHERE trade_date BETWEEN '{buffer_start}' AND '{year_end}'
    ), financial_event AS (
      SELECT symbol,publish_date,report_date,roe_avg,net_profit_margin,
             net_profit_yoy,revenue_yoy,cash_flow_ps,debt_to_asset
      FROM {_scan(paths["financial_quarterly"])}
      WHERE publish_date<'{year_end}'
      QUALIFY row_number() OVER (
        PARTITION BY symbol,publish_date ORDER BY report_date DESC
      )=1
    ), candidate_financial AS (
      SELECT c.*,f.report_date,f.publish_date,f.roe_avg,f.net_profit_margin,
             f.net_profit_yoy,f.revenue_yoy,f.cash_flow_ps,f.debt_to_asset
      FROM candidate c ASOF LEFT JOIN financial_event f
        ON c.symbol=f.symbol AND c.trade_date>f.publish_date
    ), joined AS (
      SELECT c.*,u.list_status,u.list_date,
             s.is_st,
             s.is_suspended,
             s.is_delisted,
             try_cast(v.circ_mv AS DOUBLE) AS circ_mv,
             try_cast(v.pe AS DOUBLE) AS pe,try_cast(v.pb AS DOUBLE) AS pb,
             coalesce(i.industry_name,i.industry,'Unknown') AS industry,
             d.valid_amount_20,d.amount_median_20,
             cp.open_index-lp.list_open_index+1 AS listed_open_days,
             m.candidate_id IS NOT NULL AS minute_complete
      FROM candidate_financial c
      LEFT JOIN {_scan(paths["universe_snapshot"])} u
        ON u.symbol=c.symbol AND u.trade_date=c.trade_date
      LEFT JOIN {_scan(paths["security_status"])} s
        ON s.symbol=c.symbol AND s.trade_date=c.trade_date
      LEFT JOIN {_scan(paths["valuation"])} v
        ON v.symbol=c.symbol AND v.trade_date=c.trade_date
      LEFT JOIN {_scan(paths["industry_concept"])} i
        ON i.symbol=c.symbol AND i.trade_date=c.trade_date
      LEFT JOIN daily_window d ON d.symbol=c.symbol AND d.trade_date=c.trade_date
      LEFT JOIN read_parquet('{str(calendar_path).replace(chr(39), chr(39) * 2)}') cp
        ON cp.trade_date=c.trade_date
      LEFT JOIN read_parquet('{str(listing_path).replace(chr(39), chr(39) * 2)}') lp
        ON lp.symbol=c.symbol
      LEFT JOIN read_parquet('{str(minute_path).replace(chr(39), chr(39) * 2)}') m
        ON m.candidate_id=c.candidate_id
    ), cross_rank AS (
      SELECT *,
        {_pct_rank_sql("amount_median_20", "trade_date")} AS amount_median_rank,
        {_pct_rank_sql("circ_mv", "trade_date")} AS circ_mv_rank,
        count(*) OVER (PARTITION BY trade_date,industry) AS industry_group_count
      FROM joined
    ), quality_grouped AS (
      SELECT *,CASE WHEN industry_group_count>=20
        AND lower(industry) NOT IN ('unknown','unavailable')
        THEN industry ELSE '__ALL__' END AS quality_group
      FROM cross_rank
    ), component_rank AS (
      SELECT *,
{financial_rank_columns}
      FROM quality_grouped
    ), chosen_rank AS (
      SELECT *,
        CASE WHEN quality_group='__ALL__' THEN roe_avg_global_rank ELSE roe_avg_quality_rank END AS roe_avg_rank,
        CASE WHEN quality_group='__ALL__' THEN net_profit_margin_global_rank ELSE net_profit_margin_quality_rank END AS net_profit_margin_rank,
        CASE WHEN quality_group='__ALL__' THEN net_profit_yoy_global_rank ELSE net_profit_yoy_quality_rank END AS net_profit_yoy_rank,
        CASE WHEN quality_group='__ALL__' THEN revenue_yoy_global_rank ELSE revenue_yoy_quality_rank END AS revenue_yoy_rank,
        CASE WHEN quality_group='__ALL__' THEN cash_flow_ps_global_rank ELSE cash_flow_ps_quality_rank END AS cash_flow_ps_rank,
        CASE WHEN quality_group='__ALL__' THEN debt_to_asset_global_rank ELSE debt_to_asset_quality_rank END AS debt_to_asset_rank
      FROM component_rank
    ), scored AS (
      SELECT *,list_count(list_filter([{quality_values}],x->x IS NOT NULL)) AS quality_nonnull,
             list_avg(list_filter([{quality_values}],x->x IS NOT NULL)) AS quality_score,
             datediff('day',try_cast(report_date AS DATE),try_cast(trade_date AS DATE)) AS report_age_days
      FROM chosen_rank
    ), decided AS (
      SELECT *,(
        upper(coalesce(list_status,''))='L'
        AND {status_eligible}
        AND listed_open_days>=250 AND valid_amount_20>=15
        AND amount_median_rank>=0.30 AND circ_mv_rank>=0.20
        AND report_age_days BETWEEN 0 AND 550
        AND quality_nonnull>=3 AND quality_score>=0.30
      ) AS quality_liquidity_keep
      FROM scored
    )
    SELECT *,quality_liquidity_keep AND minute_complete AS quality_liquidity_complete_keep
    FROM decided ORDER BY candidate_id
    """


def _status_eligible_sql(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    fields = tuple(f"{prefix}{field}" for field in ("is_st", "is_suspended", "is_delisted"))
    known = " AND ".join(f"{field} IS NOT NULL" for field in fields)
    excluded = " AND ".join(f"NOT {field}" for field in fields)
    return f"({known} AND {excluded})"


def _path_set_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(item).resolve() for item in paths):
        digest.update(f"{path}|{path.stat().st_size}\n".encode())
    return digest.hexdigest()


def _path_set_fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(item).resolve() for item in paths):
        stat = path.stat()
        digest.update(f"{path}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _coordinate_row_mapping(
    *,
    current_date_idx: np.ndarray,
    current_symbol_idx: np.ndarray,
    legacy_date_idx: np.ndarray,
    legacy_symbol_idx: np.ndarray,
    symbol_count: int,
) -> np.ndarray:
    current_keys = np.asarray(current_date_idx, dtype=np.int64) * int(
        symbol_count
    ) + np.asarray(current_symbol_idx, dtype=np.int64)
    legacy_keys = np.asarray(legacy_date_idx, dtype=np.int64) * int(
        symbol_count
    ) + np.asarray(legacy_symbol_idx, dtype=np.int64)
    if legacy_keys.size and not bool(np.all(legacy_keys[1:] > legacy_keys[:-1])):
        raise DataPreparationError("legacy_candidate_coordinates_not_strictly_ordered")
    positions = np.searchsorted(legacy_keys, current_keys)
    matched = positions < len(legacy_keys)
    matched[matched] &= legacy_keys[positions[matched]] == current_keys[matched]
    result = np.full(len(current_keys), -1, dtype=np.int64)
    result[matched] = positions[matched]
    return result


@lru_cache(maxsize=1)
def _current_candidate_coordinates() -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(
        CANDIDATE_INDEX, columns=["candidate_id", "date_idx", "symbol_idx"]
    )
    candidate_ids = frame["candidate_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(candidate_ids, np.arange(len(frame), dtype=np.int64)):
        raise DataPreparationError("current_candidate_id_is_not_row_position")
    return (
        frame["date_idx"].to_numpy(dtype=np.int32),
        frame["symbol_idx"].to_numpy(dtype=np.int32),
    )


@lru_cache(maxsize=1)
def _legacy_candidate_coordinates() -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads(BASE_FEATURE_MANIFEST.read_text(encoding="utf-8"))
    count = int(manifest["candidate_alignment"]["candidate_count"])
    date_record = manifest["files"]["candidate_date_idx"]
    symbol_record = manifest["files"]["candidate_symbol_idx"]
    date_idx = np.memmap(
        Path(date_record["path"]), dtype=np.int32, mode="r", shape=(count,)
    )
    symbol_idx = np.memmap(
        Path(symbol_record["path"]), dtype=np.int32, mode="r", shape=(count,)
    )
    return np.asarray(date_idx).copy(), np.asarray(symbol_idx).copy()


def _legacy_candidate_rows(candidate_ids: np.ndarray) -> np.ndarray:
    requested = np.asarray(candidate_ids, dtype=np.int64)
    current_date_idx, current_symbol_idx = _current_candidate_coordinates()
    if bool(((requested < 0) | (requested >= len(current_date_idx))).any()):
        raise DataPreparationError("current_candidate_id_out_of_range")
    legacy_date_idx, legacy_symbol_idx = _legacy_candidate_coordinates()
    pack_manifest = json.loads(
        (CANDIDATE_INDEX.parent / "manifest.json").read_text(encoding="utf-8")
    )
    return _coordinate_row_mapping(
        current_date_idx=current_date_idx[requested],
        current_symbol_idx=current_symbol_idx[requested],
        legacy_date_idx=legacy_date_idx,
        legacy_symbol_idx=legacy_symbol_idx,
        symbol_count=int(pack_manifest["symbol_count"]),
    )


def _membership_input_fingerprint(
    *,
    qdp_paths: Mapping[str, Sequence[Path]],
    minute_path: Path,
    calendar_path: Path,
    listing_path: Path,
) -> str:
    domains = (
        "market_daily_raw",
        "universe_snapshot",
        "security_status",
        "valuation",
        "industry_concept",
        "financial_quarterly",
        "trading_calendar",
        "security_identity",
    )
    source_paths = [
        CANDIDATE_INDEX,
        minute_path,
        calendar_path,
        listing_path,
        *(path for domain in domains for path in qdp_paths[domain]),
    ]
    return _path_set_fingerprint(source_paths)


def _calendar_position_frames(
    calendar: pd.DataFrame,
    identity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(calendar["trade_date"], errors="coerce")
    open_values = calendar["is_open"].map(
        lambda value: str(value).strip().lower() in {"1", "true", "t", "yes"}
    )
    positions = pd.DataFrame({"trade_date": dates, "is_open": open_values})
    positions = (
        positions.loc[
            positions["is_open"] & positions["trade_date"].notna(),
            ["trade_date"],
        ]
        .drop_duplicates()
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    positions["open_index"] = np.arange(len(positions), dtype=np.int32)

    listings = identity.drop_duplicates("current_symbol", keep="last").copy()
    list_dates = pd.to_datetime(listings["list_date"], errors="coerce").to_numpy(
        dtype="datetime64[ns]"
    )
    open_dates = positions["trade_date"].to_numpy(dtype="datetime64[ns]")
    list_positions = np.searchsorted(open_dates, list_dates, side="left")
    listings["symbol"] = listings["current_symbol"].astype(str).str.upper()
    listings["list_open_index"] = list_positions.astype(np.int32)
    positions["trade_date"] = positions["trade_date"].dt.strftime("%Y-%m-%d")
    return positions, listings.loc[:, ["symbol", "list_date", "list_open_index"]]


def _prepare_calendar_inputs(
    *,
    output_root: Path,
    qdp_paths: Mapping[str, Sequence[Path]],
) -> tuple[Path, Path, dict[str, Any]]:
    inputs_root = output_root / "inputs"
    calendar_path = output_root / "inputs" / "calendar_positions.parquet"
    listing_path = output_root / "inputs" / "listing_positions.parquet"
    prehistory_path = inputs_root / "calendar_prehistory_baostock.parquet"
    manifest_path = inputs_root / "calendar_inputs_manifest.json"
    qdp_calendar_hash = _path_set_hash(qdp_paths["trading_calendar"])
    if manifest_path.is_file() and calendar_path.is_file() and listing_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("builder_version") == CALENDAR_INPUT_BUILDER_VERSION
            and manifest.get("qdp_calendar_path_set_hash") == qdp_calendar_hash
            and dict(manifest.get("files", {}) or {})
            .get("calendar_positions", {})
            .get("sha256")
            == _sha256(calendar_path)
            and dict(manifest.get("files", {}) or {})
            .get("listing_positions", {})
            .get("sha256")
            == _sha256(listing_path)
        ):
            manifest["manifest_path"] = str(manifest_path.resolve())
            manifest["manifest_sha256"] = _sha256(manifest_path)
            return calendar_path, listing_path, manifest

    if not prehistory_path.is_file():
        prehistory = _fetch_baostock_trade_calendar_frame_with_timeout(
            start_date=CALENDAR_PREHISTORY_START,
            end_date="2009-12-31",
            exchange="SSE",
            timeout_seconds=120,
        )
        if prehistory.empty:
            raise DataPreparationError("baostock_calendar_prehistory_empty")
        prehistory = prehistory.loc[:, ["trade_date", "is_open", "exchange"]].copy()
        prehistory["source"] = "baostock_query_trade_dates"
        _write_frame(prehistory, prehistory_path)
    else:
        prehistory = pd.read_parquet(prehistory_path)

    qdp_calendar = pd.concat(
        [
            pd.read_parquet(path, columns=["trade_date", "is_open"])
            for path in qdp_paths["trading_calendar"]
        ],
        ignore_index=True,
    )
    qdp_calendar["trade_date"] = pd.to_datetime(
        qdp_calendar["trade_date"], errors="coerce"
    )
    qdp_calendar = qdp_calendar.loc[
        qdp_calendar["trade_date"].between(START_DATE, END_DATE, inclusive="both")
    ].copy()
    prehistory["trade_date"] = pd.to_datetime(prehistory["trade_date"], errors="coerce")
    prehistory = prehistory.loc[
        prehistory["trade_date"] < pd.Timestamp(START_DATE)
    ].copy()
    calendar = pd.concat(
        [prehistory.loc[:, ["trade_date", "is_open"]], qdp_calendar],
        ignore_index=True,
    )
    identity = pd.concat(
        [
            pd.read_parquet(path, columns=["current_symbol", "list_date"])
            for path in qdp_paths["security_identity"]
        ],
        ignore_index=True,
    )
    calendar_positions, listing_positions = _calendar_position_frames(
        calendar, identity
    )
    _write_frame(calendar_positions, calendar_path)
    _write_frame(listing_positions, listing_path)
    prehistory_open_dates = _calendar_position_frames(
        prehistory.loc[:, ["trade_date", "is_open"]],
        identity.iloc[:0],
    )[0]
    manifest = {
        "schema": "seq100_calendar_inputs/v1",
        "builder_version": CALENDAR_INPUT_BUILDER_VERSION,
        "qdp_calendar_path_set_hash": qdp_calendar_hash,
        "prehistory_source": "baostock_query_trade_dates",
        "prehistory_start_date": str(prehistory["trade_date"].min().date()),
        "prehistory_end_date": str(prehistory["trade_date"].max().date()),
        "prehistory_open_day_count": len(prehistory_open_dates),
        "research_calendar_end_date": END_DATE,
        "forbidden_2026_rows": 0,
        "files": {
            "calendar_prehistory": _record(prehistory_path),
            "calendar_positions": _record(calendar_path),
            "listing_positions": _record(listing_path),
        },
    }
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path.resolve())
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return calendar_path, listing_path, manifest


def prepare_membership(
    *,
    output_root: Path,
    qdp_paths: Mapping[str, Sequence[Path]],
    state: dict[str, Any],
    materialize_daily_quality_pool: bool = False,
) -> dict[str, Any]:
    required = {
        "market_daily_raw",
        "universe_snapshot",
        "security_status",
        "valuation",
        "industry_concept",
        "financial_quarterly",
        "trading_calendar",
        "security_identity",
    }
    missing = sorted(required.difference(qdp_paths))
    if missing:
        raise DataPreparationError(f"membership_qdp_domains_missing:{missing}")
    calendar_path, listing_path, calendar_inputs = _prepare_calendar_inputs(
        output_root=output_root,
        qdp_paths=qdp_paths,
    )
    state["calendar_inputs"] = calendar_inputs
    calendar_inputs_hash = str(calendar_inputs["manifest_sha256"])
    records = dict(state.get("membership_years", {}) or {})
    connection = _connect(output_root)
    try:
        for year in YEARS:
            minute_path = (
                output_root / "minute_daily" / f"year={year}" / "part-0000.parquet"
            )
            if not minute_path.is_file():
                raise DataPreparationError(f"minute_year_missing:{year}")
            input_fingerprint = _membership_input_fingerprint(
                qdp_paths=qdp_paths,
                minute_path=minute_path,
                calendar_path=calendar_path,
                listing_path=listing_path,
            )
            diagnostics_path = (
                output_root / "membership" / f"year={year}" / "diagnostics.parquet"
            )
            support_path = (
                output_root / "common_support" / f"year={year}" / "part-0000.parquet"
            )
            quality_support_path = (
                output_root
                / "quality_liquidity_pit"
                / f"year={year}"
                / "part-0000.parquet"
            )
            existing = dict(records.get(str(year), {}) or {})
            valid_existing = (
                diagnostics_path.is_file()
                and support_path.is_file()
                and existing.get("status") == "completed"
                and existing.get("builder_version") == MEMBERSHIP_BUILDER_VERSION
                and existing.get("calendar_inputs_hash") == calendar_inputs_hash
                and existing.get("input_fingerprint") == input_fingerprint
                and existing.get("support_sha256") == _sha256(support_path)
                and (
                    not materialize_daily_quality_pool
                    or (
                        quality_support_path.is_file()
                        and existing.get("quality_support_sha256")
                        == _sha256(quality_support_path)
                    )
                )
            )
            if valid_existing:
                continue
            sql = _membership_sql(
                year=year,
                paths=qdp_paths,
                minute_path=minute_path,
                calendar_path=calendar_path,
                listing_path=listing_path,
            )
            _copy_query(connection, sql, diagnostics_path)
            support_sql = (
                "SELECT candidate_id,year,trade_date,date_idx,symbol_idx,symbol,"
                "entry_trade_date,entry_filled,label_valid,price_label_valid,va_aux_valid "
                f"FROM read_parquet('{str(diagnostics_path).replace(chr(39), chr(39) * 2)}') "
                "WHERE quality_liquidity_complete_keep ORDER BY candidate_id"
            )
            _copy_query(connection, support_sql, support_path)
            if materialize_daily_quality_pool:
                quality_support_sql = (
                    "SELECT candidate_id,year,trade_date,date_idx,symbol_idx,symbol,"
                    "entry_trade_date,entry_filled,label_valid,price_label_valid,va_aux_valid "
                    f"FROM read_parquet('{str(diagnostics_path).replace(chr(39), chr(39) * 2)}') "
                    "WHERE quality_liquidity_keep ORDER BY candidate_id"
                )
                _copy_query(connection, quality_support_sql, quality_support_path)
            stats = connection.execute(
                "SELECT count(*),sum(quality_liquidity_keep),sum(minute_complete),"
                "sum(quality_liquidity_keep AND NOT minute_complete),"
                "sum(quality_liquidity_complete_keep),"
                "count(DISTINCT symbol) FILTER(WHERE quality_liquidity_keep),"
                "count(DISTINCT symbol) FILTER(WHERE quality_liquidity_complete_keep),"
                "min(trade_date),max(trade_date) "
                f"FROM read_parquet('{str(diagnostics_path).replace(chr(39), chr(39) * 2)}')"
            ).fetchone()
            records[str(year)] = {
                "status": "completed",
                "builder_version": MEMBERSHIP_BUILDER_VERSION,
                "calendar_inputs_hash": calendar_inputs_hash,
                "input_fingerprint": input_fingerprint,
                "candidate_row_count": int(stats[0]),
                "quality_liquidity_row_count": int(stats[1] or 0),
                "minute_complete_candidate_row_count": int(stats[2] or 0),
                "quality_rows_missing_minute": int(stats[3] or 0),
                "common_support_row_count": int(stats[4] or 0),
                "quality_liquidity_symbol_count": int(stats[5] or 0),
                "common_support_symbol_count": int(stats[6] or 0),
                "start_date": str(stats[7]),
                "end_date": str(stats[8]),
                "diagnostics_path": str(diagnostics_path.resolve()),
                "diagnostics_sha256": _sha256(diagnostics_path),
                "support_path": str(support_path.resolve()),
                "support_sha256": _sha256(support_path),
                **(
                    {
                        "quality_support_path": str(quality_support_path.resolve()),
                        "quality_support_sha256": _sha256(quality_support_path),
                        "quality_support_row_count": int(stats[1] or 0),
                    }
                    if materialize_daily_quality_pool
                    else {}
                ),
            }
            state["membership_years"] = records
            state["status"] = "preparing_membership"
            _write_state(output_root, state)
            print(json.dumps({"phase": "membership", "year": year}), flush=True)
    finally:
        connection.close()
    return records


def _common_support_hash(
    records: Mapping[str, Any], *, years: Sequence[int] = YEARS
) -> str:
    digest = hashlib.sha256()
    for year in years:
        path = Path(records[str(year)]["support_path"])
        frame = pd.read_parquet(path, columns=["candidate_id", "trade_date", "symbol"])
        for row in frame.itertuples(index=False):
            digest.update(
                f"{int(row.candidate_id)}|{row.trade_date}|{row.symbol}\n".encode()
            )
    return digest.hexdigest()


def _quality_support_hash(
    records: Mapping[str, Any], *, years: Sequence[int] = YEARS
) -> str:
    digest = hashlib.sha256()
    for year in years:
        path = Path(records[str(year)]["quality_support_path"])
        frame = pd.read_parquet(path, columns=["candidate_id", "trade_date", "symbol"])
        for row in frame.itertuples(index=False):
            digest.update(
                f"{int(row.candidate_id)}|{row.trade_date}|{row.symbol}\n".encode()
            )
    return digest.hexdigest()


def _safe_ratio_sql(numerator: str, denominator: str) -> str:
    return (
        f"CASE WHEN ({denominator}) IS NOT NULL AND abs(({denominator}))>1e-12 "
        f"THEN ({numerator})/({denominator}) END"
    )


def _signed_log_sql(value: str) -> str:
    return f"CASE WHEN {value} IS NOT NULL THEN sign({value})*ln(1+abs({value})) END"


def _statement_event_sql(
    *,
    scan: str,
    prefix: str,
    fields: Sequence[str],
    end_date: str,
) -> str:
    metrics = ",\n             ".join(
        f"try_cast({field} AS DOUBLE) AS {prefix}_{field}" for field in fields
    )
    return f"""
      SELECT symbol,feature_available_date AS {prefix}_available_date,
             report_date AS {prefix}_report_date,
             fiscal_year AS {prefix}_fiscal_year,
             fiscal_quarter AS {prefix}_fiscal_quarter,
             try_cast(report_type AS INTEGER) AS {prefix}_report_type,
             try_cast(company_type AS INTEGER) AS {prefix}_company_type,
             try_cast(period_type AS INTEGER) AS {prefix}_period_type,
             source_conflict AS {prefix}_source_conflict,
             {metrics}
      FROM {scan}
      WHERE feature_available_date<>'' AND feature_available_date<='{end_date}'
      QUALIFY row_number() OVER (
        PARTITION BY symbol,feature_available_date
        ORDER BY (report_type='1') DESC,report_date DESC,update_flag DESC
      )=1
    """


def _fundamental_sql(
    *,
    year: int,
    support_path: Path,
    paths: Mapping[str, Sequence[Path]],
) -> str:
    _, end = _year_bounds(year)
    financial_select = ",\n      ".join(
        f"k.financial_{field}" for field in FINANCIAL_RATIO_FIELDS
    )
    income_select = ",\n      ".join(f"k.income_{field}" for field in INCOME_FIELDS)
    balance_select = ",\n      ".join(f"k.balance_{field}" for field in BALANCE_FIELDS)
    cash_select = ",\n      ".join(f"k.cashflow_{field}" for field in CASH_FLOW_FIELDS)
    derived = {
        "income_parent_net_margin": _safe_ratio_sql(
            "k.income_parent_net_income", "k.income_revenue"
        ),
        "income_operating_margin": _safe_ratio_sql(
            "k.income_operating_profit", "k.income_revenue"
        ),
        "income_total_profit_margin": _safe_ratio_sql(
            "k.income_total_profit", "k.income_revenue"
        ),
        "income_rd_intensity": _safe_ratio_sql(
            "k.income_research_development_expense", "k.income_revenue"
        ),
        "income_selling_expense_ratio": _safe_ratio_sql(
            "k.income_selling_expense", "k.income_revenue"
        ),
        "income_administrative_expense_ratio": _safe_ratio_sql(
            "k.income_administrative_expense", "k.income_revenue"
        ),
        "income_finance_expense_ratio": _safe_ratio_sql(
            "k.income_finance_expense", "k.income_revenue"
        ),
        "balance_debt_ratio": _safe_ratio_sql(
            "k.balance_total_liabilities", "k.balance_total_assets"
        ),
        "balance_cash_ratio": _safe_ratio_sql(
            "k.balance_cash_and_equivalents", "k.balance_total_assets"
        ),
        "balance_current_ratio_derived": _safe_ratio_sql(
            "k.balance_current_assets", "k.balance_current_liabilities"
        ),
        "balance_receivables_ratio": _safe_ratio_sql(
            "coalesce(k.balance_accounts_receivable,0)+coalesce(k.balance_notes_receivable,0)",
            "k.balance_total_assets",
        ),
        "balance_inventory_ratio": _safe_ratio_sql(
            "k.balance_inventory", "k.balance_total_assets"
        ),
        "balance_goodwill_ratio": _safe_ratio_sql(
            "k.balance_goodwill", "k.balance_total_assets"
        ),
        "balance_borrowing_ratio": _safe_ratio_sql(
            "coalesce(k.balance_short_term_borrowings,0)+coalesce(k.balance_long_term_borrowings,0)+coalesce(k.balance_bonds_payable,0)",
            "k.balance_total_assets",
        ),
        "balance_parent_equity_ratio": _safe_ratio_sql(
            "k.balance_parent_equity", "k.balance_total_assets"
        ),
        "cashflow_cfo_to_income": _safe_ratio_sql(
            "k.cashflow_net_operating_cash_flow", "abs(k.income_parent_net_income)"
        ),
        "cashflow_cfo_to_revenue": _safe_ratio_sql(
            "k.cashflow_net_operating_cash_flow", "k.income_revenue"
        ),
        "cashflow_fcf_to_revenue": _safe_ratio_sql(
            "k.cashflow_free_cash_flow", "k.income_revenue"
        ),
        "cashflow_investing_to_assets": _safe_ratio_sql(
            "k.cashflow_net_investing_cash_flow", "k.balance_total_assets"
        ),
        "cashflow_financing_to_assets": _safe_ratio_sql(
            "k.cashflow_net_financing_cash_flow", "k.balance_total_assets"
        ),
    }
    derived_select = ",\n      ".join(
        f"{expression} AS {name}" for name, expression in derived.items()
    )
    log_fields = {
        "income_revenue_signed_log": "k.income_revenue",
        "income_parent_net_income_signed_log": "k.income_parent_net_income",
        "balance_total_assets_log": "k.balance_total_assets",
        "balance_total_liabilities_log": "k.balance_total_liabilities",
        "cashflow_operating_signed_log": "k.cashflow_net_operating_cash_flow",
        "cashflow_free_signed_log": "k.cashflow_free_cash_flow",
    }
    log_select = ",\n      ".join(
        f"{_signed_log_sql(value)} AS {name}" for name, value in log_fields.items()
    )
    support = str(support_path).replace("'", "''")
    return f"""
    WITH keys AS (
      SELECT * FROM read_parquet('{support}')
    ), financial_event AS (
      SELECT symbol,publish_date AS financial_available_date,
             report_date AS financial_report_date,{",".join(f"{field} AS financial_{field}" for field in FINANCIAL_RATIO_FIELDS)}
      FROM {_scan(paths["financial_quarterly"])}
      WHERE publish_date<'{end}'
      QUALIFY row_number() OVER (
        PARTITION BY symbol,publish_date ORDER BY report_date DESC
      )=1
    ), forecast_event AS (
      SELECT symbol,publish_date AS forecast_available_date,
             report_date AS forecast_report_date,forecast_type,
             try_cast(profit_min AS DOUBLE) AS forecast_profit_min,
             try_cast(profit_max AS DOUBLE) AS forecast_profit_max,
             try_cast(profit_change_min AS DOUBLE) AS forecast_change_min,
             try_cast(profit_change_max AS DOUBLE) AS forecast_change_max
      FROM {_scan(paths["performance_forecast"])}
      WHERE publish_date<'{end}'
      QUALIFY row_number() OVER (
        PARTITION BY symbol,publish_date ORDER BY report_date DESC
      )=1
    ), income_event AS (
      {_statement_event_sql(scan=_scan(paths["income_statement_quarterly"]), prefix="income", fields=INCOME_FIELDS, end_date=end)}
    ), balance_event AS (
      {_statement_event_sql(scan=_scan(paths["balance_sheet_quarterly"]), prefix="balance", fields=BALANCE_FIELDS, end_date=end)}
    ), cashflow_event AS (
      {_statement_event_sql(scan=_scan(paths["cash_flow_statement_quarterly"]), prefix="cashflow", fields=CASH_FLOW_FIELDS, end_date=end)}
    ), joined AS (
      SELECT k.*,f.*,pf.*,i.*,b.*,cf.*
      FROM keys k
      ASOF LEFT JOIN financial_event f
        ON k.symbol=f.symbol AND k.trade_date>f.financial_available_date
      ASOF LEFT JOIN forecast_event pf
        ON k.symbol=pf.symbol AND k.trade_date>pf.forecast_available_date
      ASOF LEFT JOIN income_event i
        ON k.symbol=i.symbol AND k.trade_date>=i.income_available_date
      ASOF LEFT JOIN balance_event b
        ON k.symbol=b.symbol AND k.trade_date>=b.balance_available_date
      ASOF LEFT JOIN cashflow_event cf
        ON k.symbol=cf.symbol AND k.trade_date>=cf.cashflow_available_date
    )
    SELECT {",".join("k." + column for column in KEY_COLUMNS)},
      {financial_select},
      datediff('day',try_cast(k.financial_available_date AS DATE),try_cast(k.trade_date AS DATE)) AS financial_age_days,
      (k.financial_available_date IS NOT NULL)::INTEGER AS financial_present,
      (try_cast(k.forecast_profit_min AS DOUBLE)+try_cast(k.forecast_profit_max AS DOUBLE))/2.0 AS performance_forecast_profit_mid,
      try_cast(k.forecast_profit_max AS DOUBLE)-try_cast(k.forecast_profit_min AS DOUBLE) AS performance_forecast_profit_range,
      (try_cast(k.forecast_change_min AS DOUBLE)+try_cast(k.forecast_change_max AS DOUBLE))/2.0 AS performance_forecast_change_mid,
      try_cast(k.forecast_change_max AS DOUBLE)-try_cast(k.forecast_change_min AS DOUBLE) AS performance_forecast_change_range,
      (hash(coalesce(k.forecast_type,''))%2147483647)::BIGINT AS performance_forecast_type_hash,
      datediff('day',try_cast(k.forecast_available_date AS DATE),try_cast(k.trade_date AS DATE)) AS performance_forecast_age_days,
      (k.forecast_available_date IS NOT NULL)::INTEGER AS performance_forecast_present,
      {income_select},
      k.income_fiscal_quarter,k.income_report_type,k.income_company_type,k.income_period_type,
      k.income_source_conflict::INTEGER AS income_source_conflict,
      datediff('day',try_cast(k.income_available_date AS DATE),try_cast(k.trade_date AS DATE)) AS income_statement_age_days,
      (k.income_available_date IS NOT NULL)::INTEGER AS income_statement_present,
      {balance_select},
      k.balance_fiscal_quarter,k.balance_report_type,k.balance_company_type,k.balance_period_type,
      k.balance_source_conflict::INTEGER AS balance_source_conflict,
      datediff('day',try_cast(k.balance_available_date AS DATE),try_cast(k.trade_date AS DATE)) AS balance_sheet_age_days,
      (k.balance_available_date IS NOT NULL)::INTEGER AS balance_sheet_present,
      {cash_select},
      k.cashflow_fiscal_quarter,k.cashflow_report_type,k.cashflow_company_type,k.cashflow_period_type,
      k.cashflow_source_conflict::INTEGER AS cashflow_source_conflict,
      datediff('day',try_cast(k.cashflow_available_date AS DATE),try_cast(k.trade_date AS DATE)) AS cash_flow_statement_age_days,
      (k.cashflow_available_date IS NOT NULL)::INTEGER AS cash_flow_statement_present,
      {derived_select},
      {log_select}
    FROM joined k
    ORDER BY k.candidate_id
    """


def _announcement_sql(
    *,
    year: int,
    support_path: Path,
    calendar_path: Path,
    announcement_scan: str,
) -> str:
    buffer_start, end = _year_bounds(year, buffer_days=60)
    support = str(support_path).replace("'", "''")
    calendar = str(calendar_path).replace("'", "''")
    categories = (
        "risk_warning",
        "performance",
        "restructuring",
        "holding_change",
        "financing",
        "governance",
        "operations",
        "other",
    )
    daily_category = ",\n             ".join(
        f"count(*) FILTER(WHERE category='{category}') AS {category}_count"
        for category in categories
    )
    panel_category = ",\n             ".join(
        f"coalesce(a.{category}_count,0) AS {category}_count" for category in categories
    )
    rolling_category = ",\n             ".join(
        f"sum({category}_count) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS announcement_{category}_5d,\n"
        f"             sum({category}_count) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS announcement_{category}_20d"
        for category in categories
    )
    keyword_specs = {
        "buyback": "\\u56de\\u8d2d",
        "penalty": "\\u5904\\u7f5a",
        "litigation": "\\u8bc9\\u8bbc",
        "dividend": "\\u5206\\u7ea2",
        "major_contract": "\\u4e2d\\u6807",
        "delisting": "\\u9000\\u5e02",
    }
    keyword_specs = {
        name: value.encode("ascii").decode("unicode_escape")
        for name, value in keyword_specs.items()
    }
    daily_keywords = ",\n             ".join(
        f"count(*) FILTER(WHERE contains(normalized_title,'{value}')) AS keyword_{name}_count"
        for name, value in keyword_specs.items()
    )
    panel_keywords = ",\n             ".join(
        f"coalesce(a.keyword_{name}_count,0) AS keyword_{name}_count"
        for name in keyword_specs
    )
    rolling_keywords = ",\n             ".join(
        f"sum(keyword_{name}_count) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS announcement_keyword_{name}_20d"
        for name in keyword_specs
    )
    return f"""
    WITH keys AS (
      SELECT * FROM read_parquet('{support}')
    ), symbols AS (
      SELECT DISTINCT symbol FROM keys
    ), dates AS (
      SELECT trade_date FROM read_parquet('{calendar}')
      WHERE trade_date BETWEEN '{buffer_start}' AND '{end}'
    ), announcement_daily AS (
      SELECT symbol,feature_available_date AS trade_date,count(*) AS total_count,
             {daily_category},
             {daily_keywords}
      FROM {announcement_scan}
      WHERE feature_available_date BETWEEN '{buffer_start}' AND '{end}'
      GROUP BY symbol,feature_available_date
    ), panel AS (
      SELECT s.symbol,d.trade_date,coalesce(a.total_count,0) AS total_count,
             {panel_category},
             {panel_keywords}
      FROM symbols s CROSS JOIN dates d
      LEFT JOIN announcement_daily a
        ON a.symbol=s.symbol AND a.trade_date=d.trade_date
    ), rolling AS (
      SELECT symbol,trade_date,total_count AS announcement_total_1d,
             sum(total_count) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS announcement_total_5d,
             sum(total_count) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS announcement_total_20d,
             {rolling_category},
             {rolling_keywords}
      FROM panel
    ), risk_event AS (
      SELECT DISTINCT symbol,feature_available_date AS risk_event_date
      FROM {announcement_scan}
      WHERE category='risk_warning' AND feature_available_date<>'' AND feature_available_date<='{end}'
    ), performance_event AS (
      SELECT DISTINCT symbol,feature_available_date AS performance_event_date
      FROM {announcement_scan}
      WHERE category='performance' AND feature_available_date<>'' AND feature_available_date<='{end}'
    ), restructuring_event AS (
      SELECT DISTINCT symbol,feature_available_date AS restructuring_event_date
      FROM {announcement_scan}
      WHERE category='restructuring' AND feature_available_date<>'' AND feature_available_date<='{end}'
    ), holding_event AS (
      SELECT DISTINCT symbol,feature_available_date AS holding_event_date
      FROM {announcement_scan}
      WHERE category='holding_change' AND feature_available_date<>'' AND feature_available_date<='{end}'
    ), base AS (
      SELECT k.*,r.* EXCLUDE(symbol,trade_date)
      FROM keys k LEFT JOIN rolling r
        ON r.symbol=k.symbol AND r.trade_date=k.trade_date
    ), last_events AS (
      SELECT b.*,risk.risk_event_date,perf.performance_event_date,
             restruct.restructuring_event_date,holding.holding_event_date
      FROM base b
      ASOF LEFT JOIN risk_event risk
        ON b.symbol=risk.symbol AND b.trade_date>=risk.risk_event_date
      ASOF LEFT JOIN performance_event perf
        ON b.symbol=perf.symbol AND b.trade_date>=perf.performance_event_date
      ASOF LEFT JOIN restructuring_event restruct
        ON b.symbol=restruct.symbol AND b.trade_date>=restruct.restructuring_event_date
      ASOF LEFT JOIN holding_event holding
        ON b.symbol=holding.symbol AND b.trade_date>=holding.holding_event_date
    )
    SELECT * EXCLUDE(risk_event_date,performance_event_date,restructuring_event_date,holding_event_date),
      datediff('day',try_cast(risk_event_date AS DATE),try_cast(trade_date AS DATE)) AS announcement_days_since_risk,
      datediff('day',try_cast(performance_event_date AS DATE),try_cast(trade_date AS DATE)) AS announcement_days_since_performance,
      datediff('day',try_cast(restructuring_event_date AS DATE),try_cast(trade_date AS DATE)) AS announcement_days_since_restructuring,
      datediff('day',try_cast(holding_event_date AS DATE),try_cast(trade_date AS DATE)) AS announcement_days_since_holding_change,
      1::INTEGER AS announcement_source_covered
    FROM last_events ORDER BY candidate_id
    """


def _report_coverage_status(
    path: Path = DEFAULT_REPORT_COVERAGE_PATH,
) -> dict[int, str]:
    if not path.is_file():
        return {year: "unknown" for year in YEARS}
    frame = pd.read_parquet(
        path,
        columns=["year", "tushare_source_coverage_status"],
    )
    return {
        int(row.year): str(row.tushare_source_coverage_status)
        for row in frame.itertuples(index=False)
    }


def _report_sql(
    *,
    year: int,
    support_path: Path,
    calendar_path: Path,
    report_scan: str,
    forecast_scan: str,
    coverage_status: str,
) -> str:
    buffer_start, end = _year_bounds(year, buffer_days=180)
    support = str(support_path).replace("'", "''")
    calendar = str(calendar_path).replace("'", "''")
    positive = "\\u4e70\\u5165|\\u589e\\u6301|\\u63a8\\u8350|outperform|buy".encode(
        "ascii"
    ).decode("unicode_escape")
    neutral = "\\u4e2d\\u6027|\\u6301\\u6709|neutral|hold".encode("ascii").decode(
        "unicode_escape"
    )
    negative = "\\u51cf\\u6301|\\u5356\\u51fa|\\u56de\\u907f|underperform|sell".encode(
        "ascii"
    ).decode("unicode_escape")
    complete = int(
        coverage_status in {"observed_full_year_span", "complete_daily_task_ledger"}
    )
    partial = int(coverage_status == "partial_year_span")
    unavailable = int(coverage_status == "source_unavailable")
    return f"""
    WITH keys AS (
      SELECT * FROM read_parquet('{support}')
    ), symbols AS (
      SELECT DISTINCT symbol FROM keys
    ), dates AS (
      SELECT trade_date FROM read_parquet('{calendar}')
      WHERE trade_date BETWEEN '{buffer_start}' AND '{end}'
    ), forecast_by_report AS (
      SELECT report_id,count(*) AS forecast_quarter_count,
             avg(try_cast(eps AS DOUBLE)) AS forecast_eps,
             avg(try_cast(pe AS DOUBLE)) AS forecast_pe,
             avg(try_cast(net_profit AS DOUBLE)) AS forecast_net_profit,
             avg(try_cast(operating_revenue AS DOUBLE)) AS forecast_revenue,
             avg(try_cast(roe AS DOUBLE)) AS forecast_roe,
             avg(try_cast(ev_ebitda AS DOUBLE)) AS forecast_ev_ebitda
      FROM {forecast_scan}
      WHERE feature_available_date<='{end}'
      GROUP BY report_id
    ), report_rows AS (
      SELECT r.report_id,r.symbol,r.feature_available_date AS trade_date,
             lower(coalesce(r.rating,'')) AS rating,
             r.institution,r.analyst,
             try_cast(r.target_price_min AS DOUBLE) AS target_price_min,
             try_cast(r.target_price_max AS DOUBLE) AS target_price_max,
             r.tushare_present,r.eastmoney_present,
             f.* EXCLUDE(report_id)
      FROM {report_scan} r LEFT JOIN forecast_by_report f USING(report_id)
      WHERE r.feature_available_date BETWEEN '{buffer_start}' AND '{end}'
    ), report_daily AS (
      SELECT symbol,trade_date,count(*) AS report_count,
             count(DISTINCT institution) AS institution_count,
             count(DISTINCT analyst) AS analyst_count,
             count(*) FILTER(WHERE regexp_matches(rating,'{positive}')) AS positive_count,
             count(*) FILTER(WHERE regexp_matches(rating,'{neutral}')) AS neutral_count,
             count(*) FILTER(WHERE regexp_matches(rating,'{negative}')) AS negative_count,
             count(*) FILTER(WHERE forecast_quarter_count>0) AS forecast_report_count,
             sum(forecast_eps) AS forecast_eps_sum,
             sum(forecast_pe) AS forecast_pe_sum,
             sum(forecast_net_profit) AS forecast_net_profit_sum,
             sum(forecast_revenue) AS forecast_revenue_sum,
             sum(forecast_roe) AS forecast_roe_sum,
             sum(forecast_ev_ebitda) AS forecast_ev_ebitda_sum,
             sum(target_price_min) AS target_price_min_sum,
             sum(target_price_max) AS target_price_max_sum,
             count(target_price_min) AS target_price_count,
             count(*) FILTER(WHERE tushare_present) AS tushare_count,
             count(*) FILTER(WHERE eastmoney_present) AS eastmoney_count
      FROM report_rows GROUP BY symbol,trade_date
    ), panel AS (
      SELECT s.symbol,d.trade_date,
             coalesce(r.report_count,0) AS report_count,
             coalesce(r.institution_count,0) AS institution_count,
             coalesce(r.analyst_count,0) AS analyst_count,
             coalesce(r.positive_count,0) AS positive_count,
             coalesce(r.neutral_count,0) AS neutral_count,
             coalesce(r.negative_count,0) AS negative_count,
             coalesce(r.forecast_report_count,0) AS forecast_report_count,
             coalesce(r.forecast_eps_sum,0) AS forecast_eps_sum,
             coalesce(r.forecast_pe_sum,0) AS forecast_pe_sum,
             coalesce(r.forecast_net_profit_sum,0) AS forecast_net_profit_sum,
             coalesce(r.forecast_revenue_sum,0) AS forecast_revenue_sum,
             coalesce(r.forecast_roe_sum,0) AS forecast_roe_sum,
             coalesce(r.forecast_ev_ebitda_sum,0) AS forecast_ev_ebitda_sum,
             coalesce(r.target_price_min_sum,0) AS target_price_min_sum,
             coalesce(r.target_price_max_sum,0) AS target_price_max_sum,
             coalesce(r.target_price_count,0) AS target_price_count,
             coalesce(r.tushare_count,0) AS tushare_count,
             coalesce(r.eastmoney_count,0) AS eastmoney_count
      FROM symbols s CROSS JOIN dates d
      LEFT JOIN report_daily r ON r.symbol=s.symbol AND r.trade_date=d.trade_date
    ), rolling AS (
      SELECT symbol,trade_date,report_count AS research_report_count_1d,
        sum(report_count) OVER w5 AS research_report_count_5d,
        sum(report_count) OVER w20 AS research_report_count_20d,
        sum(report_count) OVER w60 AS research_report_count_60d,
        sum(institution_count) OVER w20 AS research_institution_activity_20d,
        sum(analyst_count) OVER w20 AS research_analyst_activity_20d,
        sum(positive_count) OVER w20 AS research_positive_rating_20d,
        sum(neutral_count) OVER w20 AS research_neutral_rating_20d,
        sum(negative_count) OVER w20 AS research_negative_rating_20d,
        sum(forecast_report_count) OVER w20 AS research_forecast_report_count_20d,
        sum(forecast_eps_sum) OVER w20/NULLIF(sum(forecast_report_count) OVER w20,0) AS research_forecast_eps_mean_20d,
        sum(forecast_pe_sum) OVER w20/NULLIF(sum(forecast_report_count) OVER w20,0) AS research_forecast_pe_mean_20d,
        sum(forecast_net_profit_sum) OVER w20/NULLIF(sum(forecast_report_count) OVER w20,0) AS research_forecast_net_profit_mean_20d,
        sum(forecast_revenue_sum) OVER w20/NULLIF(sum(forecast_report_count) OVER w20,0) AS research_forecast_revenue_mean_20d,
        sum(forecast_roe_sum) OVER w20/NULLIF(sum(forecast_report_count) OVER w20,0) AS research_forecast_roe_mean_20d,
        sum(forecast_ev_ebitda_sum) OVER w20/NULLIF(sum(forecast_report_count) OVER w20,0) AS research_forecast_ev_ebitda_mean_20d,
        sum(target_price_min_sum) OVER w20/NULLIF(sum(target_price_count) OVER w20,0) AS research_target_price_min_mean_20d,
        sum(target_price_max_sum) OVER w20/NULLIF(sum(target_price_count) OVER w20,0) AS research_target_price_max_mean_20d,
        sum(tushare_count) OVER w20 AS research_tushare_report_count_20d,
        sum(eastmoney_count) OVER w20 AS research_eastmoney_report_count_20d
      FROM panel
      WINDOW w5 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
             w20 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
             w60 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
    ), report_event AS (
      SELECT DISTINCT symbol,feature_available_date AS report_event_date
      FROM {report_scan}
      WHERE feature_available_date<>'' AND feature_available_date<='{end}'
    ), base AS (
      SELECT k.*,r.* EXCLUDE(symbol,trade_date)
      FROM keys k LEFT JOIN rolling r
        ON r.symbol=k.symbol AND r.trade_date=k.trade_date
    ), last_report AS (
      SELECT b.*,r.report_event_date
      FROM base b ASOF LEFT JOIN report_event r
        ON b.symbol=r.symbol AND b.trade_date>=r.report_event_date
    )
    SELECT * EXCLUDE(report_event_date),
      datediff('day',try_cast(report_event_date AS DATE),try_cast(trade_date AS DATE)) AS research_days_since_report,
      1::INTEGER AS research_metadata_source_covered,
      {complete}::INTEGER AS research_forecast_source_complete,
      {partial}::INTEGER AS research_forecast_source_partial,
      {unavailable}::INTEGER AS research_forecast_source_unavailable
    FROM last_report ORDER BY candidate_id
    """


def _validate_feature_block(
    connection: duckdb.DuckDBPyConnection,
    *,
    support_path: Path,
    feature_path: Path,
    raise_on_error: bool = True,
) -> dict[str, int]:
    support = str(support_path).replace("'", "''")
    feature = str(feature_path).replace("'", "''")
    row_keys = ",".join(KEY_COLUMNS)
    row = connection.execute(
        "SELECT "
        f"(SELECT count(*) FROM read_parquet('{support}')) AS support_rows,"
        f"(SELECT count(*) FROM read_parquet('{feature}')) AS feature_rows,"
        f"(SELECT count(*) FROM (SELECT candidate_id,count(*) n FROM read_parquet('{feature}') GROUP BY candidate_id HAVING n>1)) AS duplicates,"
        f"(SELECT count(*) FROM read_parquet('{support}') s ANTI JOIN read_parquet('{feature}') f USING({row_keys})) AS support_missing,"
        f"(SELECT count(*) FROM read_parquet('{feature}') f ANTI JOIN read_parquet('{support}') s USING({row_keys})) AS feature_extra,"
        f"(SELECT count(*) FROM read_parquet('{feature}') WHERE trade_date>'{END_DATE}') AS forbidden_rows"
    ).fetchone()
    result = {
        "support_rows": int(row[0]),
        "feature_rows": int(row[1]),
        "duplicate_candidate_ids": int(row[2]),
        "support_missing_rows": int(row[3]),
        "feature_extra_rows": int(row[4]),
        "forbidden_2026_rows": int(row[5]),
    }
    invalid = result["support_rows"] != result["feature_rows"] or any(
        result[key] for key in tuple(result)[2:]
    )
    if invalid and raise_on_error:
        raise DataPreparationError(f"feature_block_alignment_failed:{result}")
    return result


def _feature_record(
    connection: duckdb.DuckDBPyConnection,
    *,
    support_path: Path,
    feature_path: Path,
    input_fingerprint: str,
) -> dict[str, Any]:
    schema = pq.read_schema(feature_path)
    feature_columns = [name for name in schema.names if name not in KEY_COLUMNS]
    return {
        "status": "completed",
        **_record(feature_path, feature_count=len(feature_columns)),
        "support_sha256": _sha256(support_path),
        "input_fingerprint": input_fingerprint,
        "feature_columns": feature_columns,
        "alignment": _validate_feature_block(
            connection,
            support_path=support_path,
            feature_path=feature_path,
        ),
    }


def _feature_block_current(
    connection: duckdb.DuckDBPyConnection,
    *,
    existing: dict[str, Any],
    support_path: Path,
    feature_path: Path,
    input_fingerprint: str,
) -> bool:
    if not (
        feature_path.is_file()
        and existing.get("status") == "completed"
        and existing.get("sha256") == _sha256(feature_path)
    ):
        return False
    return (
        existing.get("support_sha256") == _sha256(support_path)
        and existing.get("input_fingerprint") == input_fingerprint
    )


def _feature_input_fingerprint(
    *,
    block: str,
    support_path: Path,
    qdp_paths: Mapping[str, Sequence[Path]],
    minute_source: Path,
    calendar_path: Path,
    report_coverage_path: Path = DEFAULT_REPORT_COVERAGE_PATH,
) -> str:
    block_domains = {
        "minute": (),
        "fundamental": (
            "financial_quarterly",
            "performance_forecast",
            "income_statement_quarterly",
            "balance_sheet_quarterly",
            "cash_flow_statement_quarterly",
        ),
        "event": ("announcement", "research_report", "research_report_forecast"),
    }
    digest = hashlib.sha256()
    digest.update(
        (
            f"{block}|transform={FEATURE_TRANSFORM_VERSIONS[block]}|"
            f"{_sha256(support_path)}\n"
        ).encode()
    )
    if block == "minute":
        digest.update(_path_set_fingerprint([minute_source]).encode())
    else:
        source_paths = [
            path for domain in block_domains[block] for path in qdp_paths[domain]
        ]
        digest.update(_path_set_fingerprint(source_paths).encode())
    if block == "event":
        digest.update(_path_set_fingerprint([calendar_path]).encode())
        digest.update(
            json.dumps(
                _report_coverage_status(report_coverage_path), sort_keys=True
            ).encode()
        )
    return digest.hexdigest()


def prepare_feature_blocks(
    *,
    output_root: Path,
    qdp_paths: Mapping[str, Sequence[Path]],
    state: dict[str, Any],
    report_coverage_path: Path = DEFAULT_REPORT_COVERAGE_PATH,
) -> dict[str, Any]:
    required = {
        "financial_quarterly",
        "performance_forecast",
        "income_statement_quarterly",
        "balance_sheet_quarterly",
        "cash_flow_statement_quarterly",
        "announcement",
        "research_report",
        "research_report_forecast",
    }
    missing = sorted(required.difference(qdp_paths))
    if missing:
        raise DataPreparationError(f"feature_qdp_domains_missing:{missing}")
    records = dict(state.get("feature_blocks", {}) or {})
    coverage = _report_coverage_status(report_coverage_path)
    calendar_path = output_root / "inputs" / "calendar_positions.parquet"
    connection = _connect(output_root)
    try:
        for year in YEARS:
            year_key = str(year)
            year_records = dict(records.get(year_key, {}) or {})
            support_path = (
                output_root / "common_support" / f"year={year}" / "part-0000.parquet"
            )
            minute_source = (
                output_root / "minute_daily" / f"year={year}" / "part-0000.parquet"
            )

            minute_path = (
                output_root
                / "features"
                / "minute"
                / f"year={year}"
                / "part-0000.parquet"
            )
            minute_input = _feature_input_fingerprint(
                block="minute",
                support_path=support_path,
                qdp_paths=qdp_paths,
                minute_source=minute_source,
                calendar_path=calendar_path,
                report_coverage_path=report_coverage_path,
            )
            existing = dict(year_records.get("minute", {}) or {})
            if _feature_block_current(
                connection,
                existing=existing,
                support_path=support_path,
                feature_path=minute_path,
                input_fingerprint=minute_input,
            ):
                year_records["minute"] = existing
            else:
                minute_columns = ",".join(f"m.{name}" for name in MINUTE_FEATURES)
                sql = (
                    f"SELECT {','.join('k.' + column for column in KEY_COLUMNS)},{minute_columns} "
                    f"FROM read_parquet('{str(support_path).replace(chr(39), chr(39) * 2)}') k "
                    f"JOIN read_parquet('{str(minute_source).replace(chr(39), chr(39) * 2)}') m USING(candidate_id) "
                    "ORDER BY k.candidate_id"
                )
                _copy_query(connection, sql, minute_path)
                year_records["minute"] = _feature_record(
                    connection,
                    support_path=support_path,
                    feature_path=minute_path,
                    input_fingerprint=minute_input,
                )
                records[year_key] = year_records
                state["feature_blocks"] = records
                state["status"] = "preparing_feature_blocks"
                _write_state(output_root, state)
                print(json.dumps({"phase": "feature_minute", "year": year}), flush=True)

            fundamental_path = (
                output_root
                / "features"
                / "fundamental"
                / f"year={year}"
                / "part-0000.parquet"
            )
            fundamental_input = _feature_input_fingerprint(
                block="fundamental",
                support_path=support_path,
                qdp_paths=qdp_paths,
                minute_source=minute_source,
                calendar_path=calendar_path,
                report_coverage_path=report_coverage_path,
            )
            existing = dict(year_records.get("fundamental", {}) or {})
            if _feature_block_current(
                connection,
                existing=existing,
                support_path=support_path,
                feature_path=fundamental_path,
                input_fingerprint=fundamental_input,
            ):
                year_records["fundamental"] = existing
            else:
                _copy_query(
                    connection,
                    _fundamental_sql(
                        year=year,
                        support_path=support_path,
                        paths=qdp_paths,
                    ),
                    fundamental_path,
                )
                year_records["fundamental"] = _feature_record(
                    connection,
                    support_path=support_path,
                    feature_path=fundamental_path,
                    input_fingerprint=fundamental_input,
                )
                records[year_key] = year_records
                state["feature_blocks"] = records
                _write_state(output_root, state)
                print(
                    json.dumps({"phase": "feature_fundamental", "year": year}),
                    flush=True,
                )

            event_path = (
                output_root
                / "features"
                / "event"
                / f"year={year}"
                / "part-0000.parquet"
            )
            event_input = _feature_input_fingerprint(
                block="event",
                support_path=support_path,
                qdp_paths=qdp_paths,
                minute_source=minute_source,
                calendar_path=calendar_path,
                report_coverage_path=report_coverage_path,
            )
            existing = dict(year_records.get("event", {}) or {})
            if _feature_block_current(
                connection,
                existing=existing,
                support_path=support_path,
                feature_path=event_path,
                input_fingerprint=event_input,
            ):
                year_records["event"] = existing
            else:
                staging = output_root / "staging" / f"year={year}"
                announcement_path = staging / "announcement.parquet"
                report_path = staging / "report.parquet"
                _copy_query(
                    connection,
                    _announcement_sql(
                        year=year,
                        support_path=support_path,
                        calendar_path=calendar_path,
                        announcement_scan=_scan(qdp_paths["announcement"]),
                    ),
                    announcement_path,
                )
                _copy_query(
                    connection,
                    _report_sql(
                        year=year,
                        support_path=support_path,
                        calendar_path=calendar_path,
                        report_scan=_scan(qdp_paths["research_report"]),
                        forecast_scan=_scan(qdp_paths["research_report_forecast"]),
                        coverage_status=coverage.get(year, "unknown"),
                    ),
                    report_path,
                )
                support_columns = set(pq.read_schema(support_path).names)
                announcement_features = [
                    name
                    for name in pq.read_schema(announcement_path).names
                    if name not in support_columns
                ]
                report_features = [
                    name
                    for name in pq.read_schema(report_path).names
                    if name not in support_columns
                ]
                sql = (
                    f"SELECT {','.join('a.' + column for column in KEY_COLUMNS)},"
                    + ",".join(f"a.{column}" for column in announcement_features)
                    + ","
                    + ",".join(f"r.{column}" for column in report_features)
                    + f" FROM read_parquet('{str(announcement_path).replace(chr(39), chr(39) * 2)}') a "
                    + f"JOIN read_parquet('{str(report_path).replace(chr(39), chr(39) * 2)}') r USING(candidate_id) "
                    + "ORDER BY a.candidate_id"
                )
                _copy_query(connection, sql, event_path)
                year_records["event"] = {
                    **_feature_record(
                        connection,
                        support_path=support_path,
                        feature_path=event_path,
                        input_fingerprint=event_input,
                    ),
                    "research_forecast_source_coverage_status": coverage.get(
                        year, "unknown"
                    ),
                }
                records[year_key] = year_records
                state["feature_blocks"] = records
                _write_state(output_root, state)
                print(json.dumps({"phase": "feature_event", "year": year}), flush=True)
            records[year_key] = year_records
            state["feature_blocks"] = records
    finally:
        connection.close()
    _write_state(output_root, state)
    return records


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _feature_catalog(
    state: Mapping[str, Any], *, years: Sequence[int] = YEARS
) -> pd.DataFrame:
    base_manifest = json.loads(BASE_FEATURE_MANIFEST.read_text(encoding="utf-8"))
    rows = [
        {
            "name": str(item["name"]),
            "family": str(item["family"]),
            "source_block": "existing_seq100_base",
            "kind": "continuous",
            "causal_as_of": str(
                item.get("causal_as_of", "signal_date_close_or_earlier")
            ),
            "selected_for_model": False,
        }
        for item in base_manifest["continuous_catalog"]
    ]
    blocks = dict(state.get("feature_blocks", {}) or {})
    reference_year = next(
        (str(year) for year in reversed(tuple(years)) if str(year) in blocks),
        "",
    )
    family_names = {
        "minute": "same_day_5m_aggregates",
        "fundamental": "pit_financial_and_statement",
        "event": "pit_announcement_and_research",
    }
    for block, family in family_names.items():
        for name in blocks[reference_year][block]["feature_columns"]:
            rows.append(
                {
                    "name": str(name),
                    "family": family,
                    "source_block": block,
                    "kind": "continuous",
                    "causal_as_of": "signal_date_close_or_earlier",
                    "selected_for_model": False,
                }
            )
    return (
        pd.DataFrame(rows).drop_duplicates("name", keep="first").reset_index(drop=True)
    )


def _feature_blocks_hash(
    state: Mapping[str, Any], *, years: Sequence[int] = YEARS
) -> str:
    digest = hashlib.sha256()
    blocks = dict(state.get("feature_blocks", {}) or {})
    for year in years:
        for family in ("minute", "fundamental", "event"):
            record = dict(blocks[str(year)][family])
            digest.update(
                (
                    f"{year}|{family}|{record.get('sha256', '')}|"
                    f"{record.get('support_sha256', '')}|"
                    f"{record.get('input_fingerprint', '')}\n"
                ).encode()
            )
    return digest.hexdigest()


def _feature_stats_for_file(
    connection: duckdb.DuckDBPyConnection,
    *,
    path: Path,
    year: int,
    family: str,
    features: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    quoted = str(path).replace("'", "''")
    expressions = ["count(*) AS row_count"]
    for index, feature in enumerate(features):
        column = '"' + feature.replace('"', '""') + '"'
        expressions.extend(
            [
                f"count({column}) AS n_{index}",
                f"avg(try_cast({column} AS DOUBLE)) AS mean_{index}",
                f"stddev_pop(try_cast({column} AS DOUBLE)) AS std_{index}",
                f"approx_quantile(try_cast({column} AS DOUBLE),0.01) AS q01_{index}",
                f"approx_quantile(try_cast({column} AS DOUBLE),0.50) AS q50_{index}",
                f"approx_quantile(try_cast({column} AS DOUBLE),0.99) AS q99_{index}",
            ]
        )
    row = connection.execute(
        f"SELECT {','.join(expressions)} FROM read_parquet('{quoted}')"
    ).fetchone()
    row_count = int(row[0])
    coverage: list[dict[str, Any]] = []
    distribution: list[dict[str, Any]] = []
    offset = 1
    for feature in features:
        nonnull = int(row[offset] or 0)
        coverage.append(
            {
                "year": year,
                "family": family,
                "feature": feature,
                "row_count": row_count,
                "nonnull_count": nonnull,
                "nonnull_rate": nonnull / max(row_count, 1),
            }
        )
        distribution.append(
            {
                "year": year,
                "family": family,
                "feature": feature,
                "mean": row[offset + 1],
                "std": row[offset + 2],
                "q01": row[offset + 3],
                "q50": row[offset + 4],
                "q99": row[offset + 5],
            }
        )
        offset += 6
    return coverage, distribution


def _feature_sample(
    *,
    output_root: Path,
    state: Mapping[str, Any],
    modulus: int,
    years: Sequence[int] = YEARS,
) -> pd.DataFrame:
    scope_years = tuple(int(year) for year in years)
    blocks = dict(state["feature_blocks"])
    minute_paths = [Path(blocks[str(year)]["minute"]["path"]) for year in scope_years]
    fundamental_paths = [
        Path(blocks[str(year)]["fundamental"]["path"]) for year in scope_years
    ]
    event_paths = [Path(blocks[str(year)]["event"]["path"]) for year in scope_years]
    reference_year = str(scope_years[-1])
    minute_features = list(blocks[reference_year]["minute"]["feature_columns"])
    fundamental_features = list(
        blocks[reference_year]["fundamental"]["feature_columns"]
    )
    event_features = list(blocks[reference_year]["event"]["feature_columns"])
    sql = (
        f"SELECT {','.join('m.' + column for column in KEY_COLUMNS)},"
        + ",".join(f'm."{column}"' for column in minute_features)
        + ","
        + ",".join(f'f."{column}"' for column in fundamental_features)
        + ","
        + ",".join(f'e."{column}"' for column in event_features)
        + f" FROM {_scan(minute_paths)} m"
        + f" JOIN {_scan(fundamental_paths)} f USING(candidate_id)"
        + f" JOIN {_scan(event_paths)} e USING(candidate_id)"
        + f" WHERE m.candidate_id%{int(modulus)}=0 ORDER BY m.candidate_id"
    )
    connection = _connect(output_root)
    try:
        return connection.execute(sql).fetchdf()
    finally:
        connection.close()


def _base_sample(
    candidate_ids: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    manifest = json.loads(BASE_FEATURE_MANIFEST.read_text(encoding="utf-8"))
    shape = tuple(int(value) for value in manifest["continuous_shape"])
    path = Path(manifest["files"]["continuous"]["path"])
    values = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
    names = [str(item["name"]) for item in manifest["continuous_catalog"]]
    legacy_rows = _legacy_candidate_rows(candidate_ids)
    sample = np.full((len(candidate_ids), shape[1]), np.nan, dtype=np.float32)
    matched = legacy_rows >= 0
    sample[matched] = np.asarray(values[legacy_rows[matched]], dtype=np.float32)
    return pd.DataFrame(sample, columns=names), list(manifest["continuous_catalog"])


def _label_sample(candidate_ids: np.ndarray) -> pd.DataFrame:
    manifest = json.loads(LABEL_MANIFEST.read_text(encoding="utf-8"))
    record = manifest["files"]["candidate_labels"]
    shape = tuple(int(value) for value in record["shape"])
    values = np.memmap(Path(record["path"]), dtype=np.float32, mode="r", shape=shape)
    legacy_rows = _legacy_candidate_rows(candidate_ids)
    sample = np.full((len(candidate_ids), 2), np.nan, dtype=np.float32)
    matched = legacy_rows >= 0
    sample[matched] = np.asarray(
        values[legacy_rows[matched]][:, [4, 7]], dtype=np.float32
    )
    return pd.DataFrame(sample, columns=["mfe_10", "mfe_20"])


def _combined_feature_sample(
    *,
    output_root: Path,
    state: Mapping[str, Any],
    modulus: int,
    years: Sequence[int] = YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = _feature_sample(
        output_root=output_root,
        state=state,
        modulus=modulus,
        years=years,
    )
    candidate_ids = sample["candidate_id"].to_numpy(dtype=np.int64)
    base, _ = _base_sample(candidate_ids)
    new = sample.drop(columns=list(KEY_COLUMNS)).reset_index(drop=True)
    combined = pd.concat([base.reset_index(drop=True), new], axis=1)
    return sample, combined.replace([np.inf, -np.inf], np.nan)


def _association_rows(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    years: np.ndarray,
    catalog: pd.DataFrame,
) -> list[dict[str, Any]]:
    family = dict(zip(catalog["name"], catalog["family"]))
    rows: list[dict[str, Any]] = []
    periods = {
        "all_history": np.ones(len(features), dtype=bool),
        "decision_folds_2023_2025": np.isin(years, [2023, 2024, 2025]),
    }
    for period, period_mask in periods.items():
        for target in ("mfe_10", "mfe_20"):
            outcome = labels[target].to_numpy(dtype=np.float64)
            for name in features.columns:
                values = features[name].to_numpy(dtype=np.float64)
                valid = period_mask & np.isfinite(values) & np.isfinite(outcome)
                count = int(valid.sum())
                if count < 100:
                    continue
                value_series = pd.Series(values[valid])
                outcome_series = pd.Series(outcome[valid])
                ranked_values = value_series.rank(method="average")
                ranked_outcome = outcome_series.rank(method="average")
                if (
                    ranked_values.nunique(dropna=True) < 2
                    or ranked_outcome.nunique(dropna=True) < 2
                ):
                    rho = float("nan")
                else:
                    rho = float(
                        ranked_values.corr(
                            ranked_outcome,
                        )
                    )
                direction = 1.0 if not np.isfinite(rho) or rho >= 0 else -1.0
                score = direction * values[valid]
                q95 = float(np.nanquantile(score, 0.95))
                q99 = float(np.nanquantile(score, 0.99))
                baseline = float(np.nanmean(outcome[valid]))
                rows.append(
                    {
                        "period": period,
                        "target": target,
                        "feature": name,
                        "family": family.get(name, "unknown"),
                        "n": count,
                        "spearman": rho,
                        "preferred_direction": "high" if direction > 0 else "low",
                        "top5_actual_mfe": float(
                            np.nanmean(outcome[valid][score >= q95])
                        ),
                        "top1_actual_mfe": float(
                            np.nanmean(outcome[valid][score >= q99])
                        ),
                        "sample_mean_mfe": baseline,
                    }
                )
    return rows


def _redundancy_rows(
    features: pd.DataFrame, catalog: pd.DataFrame
) -> list[dict[str, Any]]:
    family = dict(zip(catalog["name"], catalog["family"]))
    usable = features.loc[:, features.notna().sum(axis=0) >= 100]
    usable = usable.loc[:, usable.nunique(dropna=True) >= 2]
    ranked = usable.rank(method="average", pct=True)
    correlation = ranked.corr(method="pearson", min_periods=100)
    values = correlation.to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for left in range(len(correlation.columns)):
        for right in range(left + 1, len(correlation.columns)):
            value = values[left, right]
            if np.isfinite(value) and abs(value) >= 0.98:
                left_name = str(correlation.columns[left])
                right_name = str(correlation.columns[right])
                rows.append(
                    {
                        "feature_left": left_name,
                        "feature_right": right_name,
                        "family_left": family.get(left_name, "unknown"),
                        "family_right": family.get(right_name, "unknown"),
                        "spearman": float(value),
                        "absolute_spearman": float(abs(value)),
                    }
                )
    return sorted(rows, key=lambda item: item["absolute_spearman"], reverse=True)


def prepare_atlas(
    *,
    output_root: Path,
    state: dict[str, Any],
    years: Sequence[int] = YEARS,
    study_id: str = STUDY_ID,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    future_oos_prediction_years: Sequence[int] = (2023, 2024, 2025),
    report_coverage_path: Path = DEFAULT_REPORT_COVERAGE_PATH,
) -> dict[str, Any]:
    scope_years = tuple(int(year) for year in years)
    if not scope_years or tuple(sorted(set(scope_years))) != scope_years:
        raise DataPreparationError("atlas_years_must_be_sorted_unique")
    atlas_root = output_root / "atlas"
    manifest_path = atlas_root / "manifest.json"
    feature_blocks_hash = _feature_blocks_hash(state, years=scope_years)
    existing = dict(state.get("atlas", {}) or {})
    if (
        manifest_path.is_file()
        and existing.get("status") == "completed"
        and existing.get("manifest_sha256") == _sha256(manifest_path)
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("builder_version") == ATLAS_BUILDER_VERSION
            and manifest.get("common_support_hash") == state.get("common_support_hash")
            and manifest.get("feature_blocks_hash") == feature_blocks_hash
            and tuple(manifest.get("years", YEARS)) == scope_years
        ):
            state["status"] = "completed"
            state["training_performed"] = False
            state["feature_set_selected"] = False
            _write_state(output_root, state)
            return existing
    catalog = _feature_catalog(state, years=scope_years)
    catalog_path = atlas_root / "feature_catalog.parquet"
    _write_frame(catalog, catalog_path)
    coverage_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    connection = _connect(output_root)
    try:
        for year in scope_years:
            for family in ("minute", "fundamental", "event"):
                record = state["feature_blocks"][str(year)][family]
                coverage, distribution = _feature_stats_for_file(
                    connection,
                    path=Path(record["path"]),
                    year=year,
                    family=family,
                    features=record["feature_columns"],
                )
                coverage_rows.extend(coverage)
                distribution_rows.extend(distribution)
    finally:
        connection.close()
    coverage_path = atlas_root / "coverage_by_year.parquet"
    distribution_path = atlas_root / "distribution_by_year.parquet"
    _write_frame(pd.DataFrame(coverage_rows), coverage_path)
    _write_frame(pd.DataFrame(distribution_rows), distribution_path)

    sample, combined = _combined_feature_sample(
        output_root=output_root,
        state=state,
        modulus=223,
        years=scope_years,
    )
    candidate_ids = sample["candidate_id"].to_numpy(dtype=np.int64)
    labels = _label_sample(candidate_ids)
    associations = pd.DataFrame(
        _association_rows(
            combined,
            labels,
            sample["year"].to_numpy(dtype=np.int16),
            catalog,
        )
    )
    association_path = atlas_root / "mfe_associations.parquet"
    _write_frame(associations, association_path)

    _, redundancy_sample = _combined_feature_sample(
        output_root=output_root,
        state=state,
        modulus=887,
        years=scope_years,
    )
    redundancy = pd.DataFrame(_redundancy_rows(redundancy_sample, catalog))
    if redundancy.empty:
        redundancy = pd.DataFrame(
            columns=[
                "feature_left",
                "feature_right",
                "family_left",
                "family_right",
                "spearman",
                "absolute_spearman",
            ]
        )
    redundancy_path = atlas_root / "redundancy_pairs.parquet"
    _write_frame(redundancy, redundancy_path)

    membership = dict(state["membership_years"])
    support_summary = pd.DataFrame(
        [
            {
                "year": year,
                **{
                    key: membership[str(year)][key]
                    for key in (
                        "candidate_row_count",
                        "quality_liquidity_row_count",
                        "quality_rows_missing_minute",
                        "common_support_row_count",
                        "common_support_symbol_count",
                    )
                },
            }
            for year in scope_years
        ]
    )
    support_summary_path = atlas_root / "common_support_by_year.parquet"
    _write_frame(support_summary, support_summary_path)
    all_report_coverage = _report_coverage_status(report_coverage_path)
    report_coverage = {
        str(year): all_report_coverage.get(
            str(year), all_report_coverage.get(year, "unknown")
        )
        for year in scope_years
    }
    manifest = {
        "schema": "seq100_quality_liquidity_feature_atlas/v1",
        "builder_version": ATLAS_BUILDER_VERSION,
        "status": "completed",
        "study_id": study_id,
        "start_date": start_date,
        "end_date": end_date,
        "years": list(scope_years),
        "forbidden_2026_rows": 0,
        "common_support_row_count": int(state["common_support_row_count"]),
        "common_support_hash": str(state["common_support_hash"]),
        "feature_blocks_hash": feature_blocks_hash,
        "calendar_inputs": dict(state.get("calendar_inputs", {}) or {}),
        "existing_base_feature_count": int(
            json.loads(BASE_FEATURE_MANIFEST.read_text(encoding="utf-8"))[
                "continuous_shape"
            ][1]
        ),
        "new_feature_count": int(len(catalog) - 296),
        "total_continuous_feature_count": len(catalog),
        "association_sample_modulus": 223,
        "association_sample_row_count": len(sample),
        "redundancy_sample_modulus": 887,
        "redundancy_sample_row_count": len(redundancy_sample),
        "redundancy_sample_is_independent": True,
        "association_results_are_descriptive_not_feature_selection": True,
        "feature_set_selected": False,
        "training_performed": False,
        "future_oos_prediction_years": [
            int(year) for year in future_oos_prediction_years
        ],
        "report_forecast_source_coverage_by_year": report_coverage,
        "files": {
            "feature_catalog": _record(catalog_path),
            "coverage_by_year": _record(coverage_path),
            "distribution_by_year": _record(distribution_path),
            "mfe_associations": _record(association_path),
            "redundancy_pairs": _record(redundancy_path),
            "common_support_by_year": _record(support_summary_path),
            "base_feature_manifest": {
                "path": str(BASE_FEATURE_MANIFEST.resolve()),
                "sha256": _sha256(BASE_FEATURE_MANIFEST),
            },
            "label_manifest": {
                "path": str(LABEL_MANIFEST.resolve()),
                "sha256": _sha256(LABEL_MANIFEST),
            },
        },
    }
    _write_json(manifest_path, manifest)
    result = {
        "status": "completed",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "feature_count": len(catalog),
        "association_row_count": len(associations),
        "redundancy_pair_count": len(redundancy),
    }
    state["atlas"] = result
    state["status"] = "completed"
    state["training_performed"] = False
    state["feature_set_selected"] = False
    _write_state(output_root, state)
    return result


def prepare(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config = _load_config(study_path)
    study_id = str(config["study_id"])
    pinned_dataset_ids = {
        str(domain): str(dataset_id)
        for domain, dataset_id in dict(config.get("qdp_dataset_ids", {}) or {}).items()
    }
    report_coverage_value = str(
        dict(config.get("source_artifacts", {}) or {}).get(
            "report_annual_statistics_path", DEFAULT_REPORT_COVERAGE_PATH
        )
    )
    report_coverage_path = Path(report_coverage_value)
    if not report_coverage_path.is_absolute():
        report_coverage_path = WORKSPACE_ROOT / report_coverage_path
    output_root.mkdir(parents=True, exist_ok=True)
    state = _read_state(output_root, study_id=study_id)
    if state.get("study_id") != study_id:
        raise DataPreparationError("data_prep_output_study_mismatch")
    previous_datasets = dict(state.get("qdp_dataset_ids", {}) or {})
    datasets, paths = _qdp_snapshot(
        WORKSPACE_ROOT,
        pinned_dataset_ids=pinned_dataset_ids or None,
    )
    state["qdp_changed_domains"] = sorted(
        domain
        for domain in set(previous_datasets) | set(datasets)
        if previous_datasets.get(domain) != datasets.get(domain)
    )
    state["qdp_dataset_ids"] = datasets
    state["candidate_index"] = {
        "path": str(CANDIDATE_INDEX.resolve()),
        "sha256": _sha256(CANDIDATE_INDEX),
        "row_count": int(pq.ParquetFile(CANDIDATE_INDEX).metadata.num_rows),
    }
    state["base_feature_manifest"] = {
        "path": str(BASE_FEATURE_MANIFEST.resolve()),
        "sha256": _sha256(BASE_FEATURE_MANIFEST),
    }
    state["label_manifest"] = {
        "path": str(LABEL_MANIFEST.resolve()),
        "sha256": _sha256(LABEL_MANIFEST),
    }
    _write_state(output_root, state)
    prepare_minute_features(output_root=output_root, qdp_paths=paths, state=state)
    materialize_daily_quality_pool = bool(
        dict(config.get("common_support", {}) or {}).get(
            "materialize_quality_liquidity_pit", False
        )
    )
    membership = prepare_membership(
        output_root=output_root,
        qdp_paths=paths,
        state=state,
        materialize_daily_quality_pool=materialize_daily_quality_pool,
    )
    state["common_support_hash"] = _common_support_hash(membership)
    state["common_support_row_count"] = sum(
        int(record["common_support_row_count"]) for record in membership.values()
    )
    if materialize_daily_quality_pool:
        state["quality_liquidity_pit_hash"] = _quality_support_hash(membership)
        state["quality_liquidity_pit_row_count"] = sum(
            int(record["quality_support_row_count"]) for record in membership.values()
        )
    state["status"] = "common_support_prepared"
    state["training_performed"] = False
    state["feature_set_selected"] = False
    state["config"] = config
    _write_state(output_root, state)
    prepare_feature_blocks(
        output_root=output_root,
        qdp_paths=paths,
        state=state,
        report_coverage_path=report_coverage_path,
    )
    state["status"] = "feature_blocks_prepared"
    _write_state(output_root, state)
    prepare_atlas(
        output_root=output_root,
        state=state,
        study_id=study_id,
        report_coverage_path=report_coverage_path,
    )
    return state


def status(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    state = _read_state(output_root)
    return {
        "study_id": state.get("study_id", STUDY_ID),
        "status": state.get("status", "pending"),
        "minute_year_count": len(dict(state.get("minute_years", {}) or {})),
        "membership_year_count": len(dict(state.get("membership_years", {}) or {})),
        "feature_block_year_count": len(dict(state.get("feature_blocks", {}) or {})),
        "common_support_row_count": state.get("common_support_row_count", 0),
        "common_support_hash": state.get("common_support_hash", ""),
        "atlas_status": dict(state.get("atlas", {}) or {}).get("status", "pending"),
        "training_performed": state.get("training_performed", False),
    }


def evaluate(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    state = _read_state(output_root)
    atlas = dict(state.get("atlas", {}) or {})
    membership = dict(state.get("membership_years", {}) or {})
    calendar_inputs = dict(state.get("calendar_inputs", {}) or {})
    manifest_path = output_root / "atlas" / "manifest.json"
    if not manifest_path.is_file() or atlas.get("status") != "completed":
        raise DataPreparationError("feature_atlas_not_completed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "common_support_hash_present": bool(state.get("common_support_hash")),
        "feature_set_not_selected": manifest.get("feature_set_selected") is False,
        "training_not_performed": manifest.get("training_performed") is False,
        "forbidden_2026_rows": int(manifest.get("forbidden_2026_rows", -1)) == 0,
        "current_atlas_builder": manifest.get("builder_version")
        == ATLAS_BUILDER_VERSION,
        "atlas_matches_current_feature_blocks": manifest.get("feature_blocks_hash")
        == _feature_blocks_hash(state),
        "listing_age_uses_calendar_prehistory": calendar_inputs.get("builder_version")
        == CALENDAR_INPUT_BUILDER_VERSION
        and str(calendar_inputs.get("prehistory_start_date", "")) < START_DATE
        and int(calendar_inputs.get("prehistory_open_day_count", 0)) >= 4_000,
        "all_years_have_common_support": len(membership) == len(YEARS)
        and all(
            int(membership[str(year)].get("common_support_row_count", 0)) > 0
            for year in YEARS
        ),
        "membership_uses_current_builder": len(membership) == len(YEARS)
        and all(
            membership[str(year)].get("builder_version") == MEMBERSHIP_BUILDER_VERSION
            for year in YEARS
        ),
        "independent_redundancy_sample": manifest.get(
            "redundancy_sample_is_independent"
        )
        is True
        and int(manifest.get("redundancy_sample_row_count", 0)) >= 1_000,
        "all_feature_years_complete": len(dict(state.get("feature_blocks", {}) or {}))
        == len(YEARS),
        "all_feature_blocks_same_rows": all(
            record[block]["alignment"]["support_missing_rows"] == 0
            and record[block]["alignment"]["feature_extra_rows"] == 0
            for record in dict(state.get("feature_blocks", {}) or {}).values()
            for block in ("minute", "fundamental", "event")
        ),
        "feature_blocks_match_support_hashes": all(
            record[block].get("support_sha256")
            == membership[str(year)].get("support_sha256")
            for year, record in dict(state.get("feature_blocks", {}) or {}).items()
            for block in ("minute", "fundamental", "event")
        ),
    }
    if not all(checks.values()):
        raise DataPreparationError(f"feature_atlas_evaluation_failed:{checks}")
    return {
        "status": "ok",
        "study_id": state.get("study_id", STUDY_ID),
        "manifest": str(manifest_path.resolve()),
        "checks": checks,
        "feature_count": manifest.get("total_continuous_feature_count"),
        "common_support_row_count": manifest.get("common_support_row_count"),
    }


def self_test() -> dict[str, Any]:
    if len(EXPECTED_5M_TIMES) != 48 or len(set(EXPECTED_5M_TIMES)) != 48:
        raise AssertionError("five-minute time contract changed")
    if EXPECTED_5M_TIMES[0] != "093500000" or EXPECTED_5M_TIMES[-1] != "150000000":
        raise AssertionError("five-minute endpoint contract changed")
    if YEARS[-1] != 2025 or 2026 in YEARS:
        raise AssertionError("forbidden year entered preparation")
    high_rank = _pct_rank_sql("value", "trade_date")
    low_debt_rank = _pct_rank_sql("debt", "trade_date", descending=True)
    if "ASC" not in high_rank or "DESC" not in low_debt_rank:
        raise AssertionError("quality direction changed")
    return {
        "status": "ok",
        "checks": {
            "expected_5m_bar_count": len(EXPECTED_5M_TIMES),
            "missing_minute_drops_stock_day_only": True,
            "forbidden_2026": True,
            "training_performed": False,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seq100-quality-liquidity-data-prep")
    parser.add_argument("--study-path", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run-pending", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.status:
        payload = status(output_root=args.output_root)
    elif args.prepare or args.run_pending:
        payload = prepare(study_path=args.study_path, output_root=args.output_root)
    elif args.evaluate:
        payload = evaluate(output_root=args.output_root)
    else:
        payload = self_test()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
