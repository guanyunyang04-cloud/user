"""Auxiliary update shares operations."""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantlab.data.qdp_v2 import normalization as _normalization
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb

from .baostock import (
    _valid_parquet_columns,
)
from .context import (
    SECONDARY_VALIDATION_WORKERS,
    TUSHARE_WORKERS,
    AuxiliaryContext,
    AuxiliaryUpdateError,
    _copy_query,
    _external_with_retry,
    _manifest,
    _paths,
    _replace_domain,
    _resolve_tushare_token,
    _scan_sql,
    _TushareClient,
)

LEGACY_CNINFO_SHARE_SOURCE = "akshare_cninfo_historical_share_fallback"


CNINFO_A_SHARE_SOURCE = "akshare_cninfo_historical_a_share_fallback_v2"


def _share_repair_dates(ctx: AuxiliaryContext) -> list[str]:
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    share = _scan_sql(_paths(ctx, "share_capital"))
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "share_plan_spill",
        threads=2,
    ) as con:
        rows = con.execute(
            f"""
            SELECT DISTINCT d.trade_date
            FROM {daily} d
            LEFT JOIN {share} s USING(symbol, trade_date)
            WHERE d.trade_date <= ? AND (
              s.symbol IS NULL OR s.total_share IS NULL OR s.float_share IS NULL OR
              try_cast(s.total_share_source_date AS DATE)>try_cast(d.trade_date AS DATE) OR
              try_cast(s.float_share_source_date AS DATE)>try_cast(d.trade_date AS DATE)
            )
            ORDER BY d.trade_date
            """,
            [ctx.target_date],
        ).fetchall()
    shutil.rmtree(ctx.runtime / "share_plan_spill", ignore_errors=True)
    return [str(item[0]) for item in rows]


def _fetch_daily_basic_parts(
    ctx: AuxiliaryContext,
    dates: Sequence[str],
) -> list[Path]:
    token = _resolve_tushare_token(ctx.workspace)
    if not token:
        raise AuxiliaryUpdateError("tushare_token_required_for_historical_share_capital_repair")
    output_dir = ctx.runtime / "daily_basic_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_dates = sorted(set(str(item) for item in dates))
    client = _TushareClient(token, workspace_root=ctx.workspace)

    def fetch_one(trade_date: str) -> Path:
        path = output_dir / f"daily_basic_{trade_date.replace('-', '')}.parquet"
        if _valid_parquet_columns(path, DAILY_BASIC_NORMALIZED_COLUMNS):
            return path
        path.unlink(missing_ok=True)
        raw = client.fetch(
            "daily_basic",
            params={"trade_date": trade_date.replace("-", "")},
            fields=DAILY_BASIC_FIELDS,
        )
        frame = _normalize_daily_basic(raw, trade_date)
        if frame.empty:
            raise AuxiliaryUpdateError(f"tushare_daily_basic_empty:{trade_date}")
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        try:
            frame.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    outputs: list[Path] = []
    with ThreadPoolExecutor(max_workers=TUSHARE_WORKERS) as pool:
        futures = {pool.submit(fetch_one, item): item for item in unique_dates}
        for future in as_completed(futures):
            trade_date = futures[future]
            try:
                outputs.append(future.result())
            except Exception as exc:
                raise AuxiliaryUpdateError(
                    f"daily_basic_partition_failed:{trade_date}:{type(exc).__name__}:{exc}"
                ) from exc
    return sorted(outputs)


def _fetch_cninfo_share_fallback_parts(
    ctx: AuxiliaryContext,
    symbols: Sequence[str],
) -> list[Path]:
    import akshare as ak

    output_dir = ctx.runtime / "cninfo_share_fallback_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_symbols = sorted(set(str(item).upper() for item in symbols))

    def fetch_one(symbol: str) -> Path:
        path = output_dir / f"share_{symbol.replace('.', '_')}.parquet"
        if _valid_parquet_columns(path, CNINFO_SHARE_NORMALIZED_COLUMNS):
            return path
        path.unlink(missing_ok=True)
        code = symbol.split(".", 1)[0]
        raw = _external_with_retry(
            lambda: ak.stock_share_change_cninfo(
                symbol=code,
                start_date="19900101",
                end_date=ctx.target_date.replace("-", ""),
            ),
            label=f"share_fallback:{symbol}",
        )
        frame = _normalize_cninfo_share_change(
            raw,
            symbol=symbol,
            target_date=ctx.target_date,
            source=CNINFO_A_SHARE_SOURCE,
        )
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
        futures = {pool.submit(fetch_one, symbol): symbol for symbol in unique_symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                outputs.append(future.result())
            except Exception as exc:
                raise AuxiliaryUpdateError(f"cninfo_share_fallback_failed:{symbol}:{type(exc).__name__}:{exc}") from exc
    return sorted(outputs)


def _legacy_cninfo_share_symbols(
    ctx: AuxiliaryContext,
    *,
    current: str | None = None,
) -> list[str]:
    source = current or _scan_sql(_paths(ctx, "share_capital"))
    spill = ctx.runtime / "legacy_cninfo_share_inventory_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=1) as con:
        rows = con.execute(
            f"SELECT DISTINCT symbol FROM {source} WHERE source=? ORDER BY symbol",
            [LEGACY_CNINFO_SHARE_SOURCE],
        ).fetchall()
    shutil.rmtree(spill, ignore_errors=True)
    return [str(item[0]) for item in rows]


def _repair_legacy_cninfo_share_rows(
    ctx: AuxiliaryContext,
    *,
    current: str,
    symbols: Sequence[str],
) -> tuple[dict[str, Any], list[Path]]:
    expected_rows = int(_manifest(ctx.root, ctx.datasets, "share_capital").row_count)
    fallback_parts = _fetch_cninfo_share_fallback_parts(ctx, symbols)
    fallback = _scan_sql(fallback_parts)
    prepared = ctx.runtime / "share_capital.prepared.parquet"
    sql = f"""
    WITH fallback AS (
      SELECT symbol, source_date, total_share, float_share, source
      FROM {fallback}
      QUALIFY row_number() OVER (
        PARTITION BY symbol, source_date
        ORDER BY variation_date DESC NULLS LAST
      )=1
    )
    SELECT c.symbol, c.trade_date,
           CASE WHEN c.source='{LEGACY_CNINFO_SHARE_SOURCE}'
                THEN f.total_share ELSE c.total_share END AS total_share,
           CASE WHEN c.source='{LEGACY_CNINFO_SHARE_SOURCE}'
                THEN f.float_share ELSE c.float_share END AS float_share,
           CASE WHEN c.source='{LEGACY_CNINFO_SHARE_SOURCE}'
                THEN greatest(f.total_share-f.float_share,0.0)
                ELSE c.restricted_share END AS restricted_share,
           CASE WHEN c.source='{LEGACY_CNINFO_SHARE_SOURCE}'
                THEN f.source_date ELSE c.total_share_source_date END
                AS total_share_source_date,
           CASE WHEN c.source='{LEGACY_CNINFO_SHARE_SOURCE}'
                THEN f.source_date ELSE c.float_share_source_date END
                AS float_share_source_date,
           CASE WHEN c.source='{LEGACY_CNINFO_SHARE_SOURCE}'
                THEN f.source_date ELSE c.restricted_share_source_date END
                AS restricted_share_source_date,
           CASE WHEN c.source='{LEGACY_CNINFO_SHARE_SOURCE}' THEN
                  CASE WHEN f.source_date=c.trade_date
                       THEN 'direct_daily' ELSE 'prior_ffill' END
                ELSE c.share_fill_method END AS share_fill_method,
           CASE WHEN c.source='{LEGACY_CNINFO_SHARE_SOURCE}'
                THEN f.source ELSE c.source END AS source
    FROM {current} c
    ASOF LEFT JOIN fallback f
      ON c.symbol=f.symbol AND c.trade_date>=f.source_date
    ORDER BY c.trade_date,c.symbol
    """
    _copy_query(ctx, sql=sql, target=prepared)
    produced = _scan_sql([prepared])
    spill = ctx.runtime / "legacy_cninfo_share_validate_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        row = con.execute(
            f"""
            SELECT count(*),
                   count(*) FILTER(WHERE p.source='{LEGACY_CNINFO_SHARE_SOURCE}'),
                   count(*) FILTER(WHERE p.total_share IS NULL OR p.float_share IS NULL OR
                     p.total_share<=0 OR p.float_share<0 OR p.float_share>p.total_share OR
                     try_cast(p.total_share_source_date AS DATE)>
                       try_cast(p.trade_date AS DATE) OR
                     try_cast(p.float_share_source_date AS DATE)>
                       try_cast(p.trade_date AS DATE)),
                   count(*) FILTER(WHERE c.source='{LEGACY_CNINFO_SHARE_SOURCE}' AND
                     abs(c.float_share-p.float_share)>0.5)
            FROM {produced} p
            JOIN {current} c USING(symbol,trade_date)
            """
        ).fetchone()
    shutil.rmtree(spill, ignore_errors=True)
    if int(row[0] or 0) != expected_rows or int(row[1] or 0) or int(row[2] or 0):
        raise AuxiliaryUpdateError(
            "legacy_cninfo_share_migration_failed:"
            f"rows={int(row[0] or 0)}:legacy={int(row[1] or 0)}:"
            f"invalid={int(row[2] or 0)}"
        )
    result = _replace_domain(
        ctx,
        domain="share_capital",
        prepared=prepared,
        primary_key=("trade_date", "symbol"),
        contract_version="qdp_v2_share_capital_strict_pit_v3",
        source_contract=("same-day Tushare gap repair plus past-only CNInfo domestic A-share fallback and state carry"),
        validation={
            "secondary_compared_count": 0,
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "pending_cninfo_share_sample",
        },
    )
    shutil.rmtree(ctx.runtime / "cninfo_share_fallback_parts", ignore_errors=True)
    return (
        {
            **result,
            "legacy_cninfo_symbol_count": len(symbols),
            "legacy_cninfo_changed_float_row_count": int(row[3] or 0),
        },
        [],
    )


def _share_candidate_ctes(
    *,
    target_date: str,
    fetched: str,
    current: str,
    fallback: str = "",
) -> str:
    fallback_union = (
        f"""
        UNION ALL
        SELECT symbol, source_date, total_share, float_share,
               source, 3 AS priority
        FROM {fallback}
        WHERE total_share>0 AND float_share>=0 AND float_share<=total_share
          AND try_cast(source_date AS DATE)<=DATE '{target_date}'
        """
        if fallback
        else ""
    )
    return f"""
    candidates AS (
      SELECT symbol, trade_date AS source_date, total_share, float_share,
             source, 2 AS priority
      FROM {fetched}
      WHERE total_share>0 AND float_share>=0 AND float_share<=total_share
      UNION ALL
      SELECT symbol,
             cast(greatest(
               try_cast(total_share_source_date AS DATE),
               try_cast(float_share_source_date AS DATE)
             ) AS VARCHAR) AS source_date,
             total_share, float_share, source, 1 AS priority
      FROM {current}
      WHERE total_share>0 AND float_share>=0 AND float_share<=total_share
        AND try_cast(total_share_source_date AS DATE)<=try_cast(trade_date AS DATE)
        AND try_cast(float_share_source_date AS DATE)<=try_cast(trade_date AS DATE)
      {fallback_union}
    ), dedup AS (
      SELECT symbol, source_date, total_share, float_share, source
      FROM candidates
      QUALIFY row_number() OVER (
        PARTITION BY symbol, source_date ORDER BY priority DESC
      )=1
    )
    """


def _unresolved_share_rows(
    ctx: AuxiliaryContext,
    *,
    daily: str,
    candidate_ctes: str,
) -> Any:
    spill = ctx.runtime / "share_fallback_plan_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        unresolved = con.execute(
            f"""
            WITH {candidate_ctes}, resolved AS (
              SELECT d.symbol, d.trade_date, c.total_share, c.float_share
              FROM (SELECT * FROM {daily} WHERE trade_date<=?) d
              ASOF LEFT JOIN dedup c
                ON d.symbol=c.symbol AND d.trade_date>=c.source_date
            )
            SELECT symbol, min(trade_date) AS first_missing_date,
                   count(*) AS missing_key_count
            FROM resolved
            WHERE total_share IS NULL OR float_share IS NULL
            GROUP BY symbol
            ORDER BY symbol
            """,
            [ctx.target_date],
        ).fetchdf()
    shutil.rmtree(spill, ignore_errors=True)
    return unresolved


def _share_repair_query(
    *,
    target_date: str,
    daily: str,
    candidate_ctes: str,
) -> str:
    return f"""
    WITH {candidate_ctes}, resolved AS (
      SELECT d.symbol, d.trade_date, c.total_share, c.float_share,
             greatest(c.total_share-c.float_share, 0.0) AS restricted_share,
             c.source_date AS total_share_source_date,
             c.source_date AS float_share_source_date,
             c.source_date AS restricted_share_source_date,
             CASE WHEN c.source_date=d.trade_date THEN 'direct_daily'
                  ELSE 'prior_ffill' END AS share_fill_method,
             c.source
      FROM (SELECT * FROM {daily} WHERE trade_date<='{target_date}') d
      ASOF LEFT JOIN dedup c
        ON d.symbol=c.symbol AND d.trade_date>=c.source_date
    )
    SELECT * FROM resolved
    WHERE total_share IS NOT NULL AND float_share IS NOT NULL
    ORDER BY trade_date, symbol
    """


@dataclass(frozen=True)
class _ShareRepairDiagnostics:
    missing: int
    invalid: int
    compared: int
    mismatches: int


def _share_repair_diagnostics(
    ctx: AuxiliaryContext,
    *,
    daily: str,
    current: str,
    fetched: str,
    produced: str,
) -> _ShareRepairDiagnostics:
    spill = ctx.runtime / "share_validate_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        missing = int(
            con.execute(
                f"SELECT count(*) FROM (SELECT symbol,trade_date FROM {daily} "
                "WHERE trade_date<=? EXCEPT SELECT symbol,trade_date FROM " + produced + ")",
                [ctx.target_date],
            ).fetchone()[0]
        )
        invalid = int(
            con.execute(
                f"SELECT count(*) FROM {produced} WHERE total_share<=0 OR "
                "float_share<0 OR float_share>total_share OR restricted_share<0 OR "
                "try_cast(total_share_source_date AS DATE)>try_cast(trade_date AS DATE) OR "
                "try_cast(float_share_source_date AS DATE)>try_cast(trade_date AS DATE) OR "
                "try_cast(restricted_share_source_date AS DATE)>try_cast(trade_date AS DATE)"
            ).fetchone()[0]
        )
        compared = int(
            con.execute(
                f"SELECT count(*) FROM {current} c JOIN {fetched} t "
                "USING(symbol,trade_date) WHERE c.total_share IS NOT NULL AND t.total_share IS NOT NULL"
            ).fetchone()[0]
        )
        mismatches = int(
            con.execute(
                f"SELECT count(*) FROM {current} c JOIN {fetched} t "
                "USING(symbol,trade_date) WHERE c.total_share IS NOT NULL AND t.total_share IS NOT NULL "
                "AND abs(c.total_share/t.total_share-1)>0.0001"
            ).fetchone()[0]
        )
    shutil.rmtree(spill, ignore_errors=True)
    return _ShareRepairDiagnostics(missing, invalid, compared, mismatches)


def _fallback_candidate_count(ctx: AuxiliaryContext, fallback: str) -> int:
    if not fallback:
        return 0
    spill = ctx.runtime / "share_fallback_count_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=1) as con:
        count = int(con.execute(f"SELECT count(*) FROM {fallback}").fetchone()[0])
    shutil.rmtree(spill, ignore_errors=True)
    return count


def repair_share_capital(
    ctx: AuxiliaryContext,
    *,
    daily_basic_parts: Sequence[Path] | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    current = _scan_sql(_paths(ctx, "share_capital"))
    legacy_symbols = _legacy_cninfo_share_symbols(ctx, current=current)
    if legacy_symbols:
        return _repair_legacy_cninfo_share_rows(
            ctx,
            current=current,
            symbols=legacy_symbols,
        )
    dates = _share_repair_dates(ctx)
    parts = list(daily_basic_parts or ())
    required_paths = {
        ctx.runtime / "daily_basic_parts" / f"daily_basic_{item.replace('-', '')}.parquet" for item in dates
    }
    if not required_paths.issubset(set(parts)):
        parts = _fetch_daily_basic_parts(ctx, dates)
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    fetched = _scan_sql(parts)
    prepared = ctx.runtime / "share_capital.prepared.parquet"
    candidate_ctes = _share_candidate_ctes(
        target_date=ctx.target_date,
        fetched=fetched,
        current=current,
    )
    unresolved = _unresolved_share_rows(
        ctx,
        daily=daily,
        candidate_ctes=candidate_ctes,
    )
    fallback_parts = (
        _fetch_cninfo_share_fallback_parts(
            ctx,
            unresolved["symbol"].astype(str).tolist(),
        )
        if not unresolved.empty
        else []
    )
    fallback = _scan_sql(fallback_parts) if fallback_parts else ""
    candidate_ctes = _share_candidate_ctes(
        target_date=ctx.target_date,
        fetched=fetched,
        current=current,
        fallback=fallback,
    )
    _copy_query(
        ctx,
        sql=_share_repair_query(
            target_date=ctx.target_date,
            daily=daily,
            candidate_ctes=candidate_ctes,
        ),
        target=prepared,
    )
    produced = _scan_sql([prepared])
    diagnostics = _share_repair_diagnostics(
        ctx,
        daily=daily,
        current=current,
        fetched=fetched,
        produced=produced,
    )
    if diagnostics.missing or diagnostics.invalid:
        raise AuxiliaryUpdateError(
            f"share_capital_contract_failed:missing={diagnostics.missing}:invalid={diagnostics.invalid}"
        )
    result = _replace_domain(
        ctx,
        domain="share_capital",
        prepared=prepared,
        primary_key=("trade_date", "symbol"),
        contract_version="qdp_v2_share_capital_strict_pit_v3",
        source_contract=("same-day Tushare gap repair plus past-only CNInfo official fallback and state carry"),
        validation={
            "secondary_compared_count": 0,
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "pending_cninfo_share_sample",
        },
    )
    fallback_candidate_count = _fallback_candidate_count(ctx, fallback)
    shutil.rmtree(ctx.runtime / "cninfo_share_fallback_parts", ignore_errors=True)
    return (
        {
            **result,
            "repair_date_count": len(dates),
            "replaced_overlap_compared_count": diagnostics.compared,
            "replaced_overlap_difference_count": diagnostics.mismatches,
            "cninfo_fallback_symbol_count": len(unresolved),
            "cninfo_fallback_candidate_count": fallback_candidate_count,
        },
        parts,
    )


_normalize_daily_basic = _normalization.normalize_daily_basic


_normalize_cninfo_share_change = _normalization.normalize_cninfo_share_change


_tushare_date_series = _normalization.tushare_date_series


CNINFO_SHARE_NORMALIZED_COLUMNS = _normalization.CNINFO_SHARE_NORMALIZED_COLUMNS


DAILY_BASIC_FIELDS = _normalization.DAILY_BASIC_FIELDS


DAILY_BASIC_NORMALIZED_COLUMNS = _normalization.DAILY_BASIC_NORMALIZED_COLUMNS
