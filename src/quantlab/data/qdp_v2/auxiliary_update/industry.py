"""Auxiliary update industry operations."""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.qdp_v2 import normalization as _normalization
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb

from .baostock import (
    _fetch_baostock_snapshots,
    _valid_parquet_columns,
)
from .context import (
    SECONDARY_VALIDATION_WORKERS,
    AuxiliaryContext,
    AuxiliaryUpdateError,
    _copy_query,
    _external_with_retry,
    _open_dates,
    _paths,
    _replace_domain,
    _scan_sql,
    _snapshot_dates,
)


def _industry_query_dates(ctx: AuxiliaryContext) -> list[str]:
    open_dates = _open_dates(ctx)
    monthly = set(_snapshot_dates(open_dates))
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    current = _scan_sql(_paths(ctx, "industry_concept"))
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "industry_plan_spill",
        threads=2,
    ) as con:
        seed_dates = con.execute(
            f"""
            WITH daily_first AS (
              SELECT symbol, min(trade_date) AS first_date
              FROM {daily}
              WHERE trade_date<=?
              GROUP BY symbol
            ), valid_sources AS (
              SELECT symbol,
                     min(coalesce(nullif(industry_source_date, ''), trade_date))
                       AS first_source_date
              FROM {current}
              WHERE coalesce(trim(industry), '')<>''
                AND lower(trim(industry)) NOT IN ('unknown', 'unclassified')
                AND coalesce(industry_fill_method, '')<>'unavailable'
                AND coalesce(original_source, '')<>
                    'qdp_v2_industry_initial_bfill'
                AND try_cast(
                      coalesce(nullif(industry_source_date, ''), trade_date)
                    AS DATE)<=try_cast(trade_date AS DATE)
              GROUP BY symbol
            )
            SELECT DISTINCT d.first_date
            FROM daily_first d
            LEFT JOIN valid_sources v USING(symbol)
            WHERE v.symbol IS NULL OR
                  try_cast(v.first_source_date AS DATE)>
                  try_cast(d.first_date AS DATE)
            """,
            [ctx.target_date],
        ).fetchall()
    shutil.rmtree(ctx.runtime / "industry_plan_spill", ignore_errors=True)
    monthly.update(str(item[0]) for item in seed_dates if item[0])
    return sorted(monthly)


def _industry_cninfo_symbols(
    ctx: AuxiliaryContext,
    *,
    fetched_paths: Sequence[Path],
) -> list[str]:
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    current = _scan_sql(_paths(ctx, "industry_concept"))
    fetched = _scan_sql(fetched_paths)
    spill = ctx.runtime / "industry_cninfo_plan_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        rows = con.execute(
            f"""
            WITH daily_first AS (
              SELECT symbol, min(trade_date) AS first_trade_date
              FROM {daily}
              WHERE trade_date<=?
              GROUP BY symbol
            ), candidates AS (
              SELECT symbol,
                     coalesce(nullif(industry_source_date, ''), trade_date)
                       AS source_date
              FROM {current}
              WHERE coalesce(trim(industry), '')<>''
                AND coalesce(original_source, '')<>
                    'qdp_v2_industry_initial_bfill'
                AND try_cast(industry_source_date AS DATE)<=try_cast(trade_date AS DATE)
              UNION ALL
              SELECT symbol, source_date
              FROM {fetched}
              WHERE coalesce(trim(industry), '')<>''
                AND try_cast(source_date AS DATE)<=try_cast(snapshot_query_date AS DATE)
            ), first_sources AS (
              SELECT symbol, min(source_date) AS first_source_date
              FROM candidates
              GROUP BY symbol
            )
            SELECT d.symbol
            FROM daily_first d
            LEFT JOIN first_sources s USING(symbol)
            WHERE s.symbol IS NULL OR
                  try_cast(s.first_source_date AS DATE)>
                  try_cast(d.first_trade_date AS DATE)
            ORDER BY d.symbol
            """,
            [ctx.target_date],
        ).fetchall()
    shutil.rmtree(spill, ignore_errors=True)
    return [str(item[0]) for item in rows]


def _fetch_cninfo_industry_parts(
    ctx: AuxiliaryContext,
    symbols: Sequence[str],
) -> list[Path]:
    import akshare as ak

    output_dir = ctx.runtime / "cninfo_industry_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not symbols:
        path = output_dir / "industry_empty.parquet"
        if not _valid_parquet_columns(path, CNINFO_INDUSTRY_COLUMNS):
            pd.DataFrame(columns=CNINFO_INDUSTRY_COLUMNS).to_parquet(
                path,
                index=False,
                compression="zstd",
            )
        return [path]

    def fetch_one(symbol: str) -> Path:
        path = output_dir / f"industry_{symbol.replace('.', '_')}.parquet"
        if _valid_parquet_columns(path, CNINFO_INDUSTRY_COLUMNS):
            return path
        path.unlink(missing_ok=True)
        code = symbol.split(".", 1)[0]

        def query() -> pd.DataFrame:
            try:
                return ak.stock_industry_change_cninfo(
                    symbol=code,
                    start_date="19900101",
                    end_date=ctx.target_date.replace("-", ""),
                )
            except KeyError:
                return pd.DataFrame()

        raw = _external_with_retry(
            query,
            label=f"industry_history:{symbol}",
        )
        frame = _normalize_cninfo_industry_history(
            raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(),
            symbol=symbol,
            target_date=ctx.target_date,
        )
        if frame.empty:
            frame = pd.DataFrame(columns=CNINFO_INDUSTRY_COLUMNS)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        try:
            frame.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    outputs: list[Path] = []
    with ThreadPoolExecutor(max_workers=SECONDARY_VALIDATION_WORKERS) as pool:
        futures = {pool.submit(fetch_one, symbol): symbol for symbol in symbols}
        completed = 0
        for future in as_completed(futures):
            outputs.append(future.result())
            completed += 1
            if completed % 200 == 0:
                print(
                    f"auxiliary_industry_cninfo={completed}/{len(symbols)}",
                    flush=True,
                )
    return sorted(outputs)


def _industry_repair_query(
    *,
    target_date: str,
    daily: str,
    current: str,
    fetched: str,
    cninfo: str,
) -> str:
    return f"""
    WITH candidates AS (
      SELECT symbol, industry, coalesce(nullif(original_source, ''), source) AS source,
             coalesce(nullif(industry_source_date, ''), trade_date) AS source_date,
             '证监会行业分类' AS industry_standard, 2 AS priority
      FROM {current}
      WHERE coalesce(trim(industry), '') <> ''
        AND lower(trim(industry)) NOT IN ('unknown', 'unclassified')
        AND coalesce(industry_fill_method, '') <> 'unavailable'
        AND coalesce(original_source, '') <> 'qdp_v2_industry_initial_bfill'
        AND try_cast(coalesce(nullif(industry_source_date, ''), trade_date) AS DATE)
            <= try_cast(trade_date AS DATE)
      UNION ALL
      SELECT symbol, industry, source, source_date, industry_standard, 1 AS priority
      FROM {fetched}
      WHERE coalesce(trim(industry), '') <> ''
        AND try_cast(source_date AS DATE) <= try_cast(snapshot_query_date AS DATE)
      UNION ALL
      SELECT symbol, industry, source, source_date, industry_standard,
             3 AS priority
      FROM {cninfo}
      WHERE coalesce(trim(industry), '')<>''
        AND try_cast(source_date AS DATE)<='{target_date}'
    ), dedup AS (
      SELECT * EXCLUDE(priority)
      FROM candidates
      QUALIFY row_number() OVER (
        PARTITION BY symbol, source_date ORDER BY priority DESC
      ) = 1
    ), resolved AS (
      SELECT d.symbol, d.trade_date,
             coalesce(c.industry, f.industry, 'Unknown') AS industry,
             coalesce(c.source, f.source, 'pit_history_industry_unavailable') AS source,
             coalesce(c.industry, f.original_industry, f.industry, 'Unknown')
               AS original_industry,
             coalesce(c.source, f.original_source, f.source,
                      'pit_history_industry_unavailable') AS original_source,
             CASE
               WHEN c.symbol IS NOT NULL AND c.source_date=d.trade_date
                 THEN 'direct_snapshot'
               WHEN c.symbol IS NOT NULL THEN 'prior_ffill'
               ELSE 'unavailable'
             END AS industry_fill_method,
             coalesce(c.source_date, f.industry_source_date, d.trade_date)
               AS industry_source_date,
             coalesce(nullif(c.industry_standard, ''),
                      nullif(f.industry_standard, ''), 'Unclassified')
               AS industry_standard
      FROM (SELECT * FROM {daily} WHERE trade_date <= '{target_date}') d
      ASOF LEFT JOIN dedup c
        ON d.symbol=c.symbol AND d.trade_date>=c.source_date
      LEFT JOIN {current} f
        ON d.symbol=f.symbol AND d.trade_date=f.trade_date
    )
    SELECT symbol, trade_date, industry,
           nullif(regexp_extract(industry, '^([A-Z][0-9]{{2}})', 1), '')
             AS industry_code,
           CASE
             WHEN regexp_matches(industry, '^[A-Z][0-9]{{2}}')
               THEN regexp_replace(industry, '^[A-Z][0-9]{{2}}', '')
             ELSE industry
           END AS industry_name,
           CASE
             WHEN lower(industry) IN ('unknown', 'unclassified')
               THEN 'unclassified'
             WHEN regexp_matches(industry, '^[A-Z][0-9]{{2}}')
               THEN 'csrc_coded'
             ELSE 'csrc_uncoded_historical_label'
           END AS industry_taxonomy_version,
           CASE
             WHEN regexp_matches(industry, '^[A-Z][0-9]{{2}}')
               THEN substr(industry, 1, 1)
             ELSE NULL
           END AS industry_section_code,
           source, original_industry, original_source, industry_fill_method,
           industry_source_date, industry_standard
    FROM resolved
    WHERE industry IS NOT NULL
    ORDER BY trade_date, symbol
    """


def _validate_industry_repair(
    ctx: AuxiliaryContext,
    *,
    daily: str,
    prepared: Path,
) -> tuple[int, int]:
    produced = _scan_sql([prepared])
    spill = ctx.runtime / "industry_validate_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        missing = int(
            con.execute(
                f"SELECT count(*) FROM ("
                f"SELECT symbol, trade_date FROM {daily} WHERE trade_date <= ? EXCEPT "
                f"SELECT symbol, trade_date FROM {produced})",
                [ctx.target_date],
            ).fetchone()[0]
        )
        invalid = int(
            con.execute(
                f"SELECT count(*) FROM {produced} WHERE "
                "industry IS NULL OR trim(industry)='' OR "
                "industry_name IS NULL OR trim(industry_name)='' OR "
                "industry_taxonomy_version NOT IN ("
                "'unclassified','csrc_coded','csrc_uncoded_historical_label') OR "
                "industry_fill_method NOT IN "
                "('direct_snapshot','prior_ffill','unavailable') OR "
                "try_cast(industry_source_date AS DATE)>try_cast(trade_date AS DATE)"
            ).fetchone()[0]
        )
    shutil.rmtree(spill, ignore_errors=True)
    return missing, invalid


def repair_industry(ctx: AuxiliaryContext) -> dict[str, Any]:
    dates = _industry_query_dates(ctx)
    parts = _fetch_baostock_snapshots(ctx, kind="industry", dates=dates)
    cninfo_symbols = _industry_cninfo_symbols(ctx, fetched_paths=parts)
    cninfo_parts = _fetch_cninfo_industry_parts(ctx, cninfo_symbols)
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    current = _scan_sql(_paths(ctx, "industry_concept"))
    fetched = _scan_sql(parts)
    cninfo = _scan_sql(cninfo_parts)
    prepared = ctx.runtime / "industry_concept.prepared.parquet"
    _copy_query(
        ctx,
        sql=_industry_repair_query(
            target_date=ctx.target_date,
            daily=daily,
            current=current,
            fetched=fetched,
            cninfo=cninfo,
        ),
        target=prepared,
    )
    missing, invalid = _validate_industry_repair(ctx, daily=daily, prepared=prepared)
    if missing or invalid:
        raise AuxiliaryUpdateError(f"industry_contract_failed:missing={missing}:invalid={invalid}")
    result = _replace_domain(
        ctx,
        domain="industry_concept",
        prepared=prepared,
        primary_key=("trade_date", "symbol"),
        contract_version="qdp_v2_industry_strict_pit_v7",
        source_contract=(
            "BaoStock/CNInfo dated snapshots with past-only asof fill; raw "
            "historical labels are preserved alongside non-retroactive taxonomy fields"
        ),
        validation={
            "secondary_compared_count": 0,
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "pending_current_cninfo_sample",
        },
    )
    shutil.rmtree(ctx.runtime / "baostock_industry_parts", ignore_errors=True)
    shutil.rmtree(ctx.runtime / "cninfo_industry_parts", ignore_errors=True)
    return {
        **result,
        "snapshot_date_count": len(dates),
        "cninfo_fallback_symbol_count": len(cninfo_symbols),
    }


_normalize_cninfo_industry_history = _normalization.normalize_cninfo_industry_history


CNINFO_INDUSTRY_COLUMNS = _normalization.CNINFO_INDUSTRY_COLUMNS
