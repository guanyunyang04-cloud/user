"""Margin Eligibility Update: prepare responsibilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from quantlab.data.domains.contracts import DataDomain

from .config import (
    END_DATE,
    MARGIN_DETAIL_COLUMNS,
    START_DATE,
    MarginEligibilityUpdateError,
)
from .context import (
    _dataset_paths,
    _raw_day_path,
    _read_state,
    _runtime,
    _sql_paths,
    _trade_dates,
    _workspace,
    _write_state,
)


@dataclass(frozen=True)
class _PrepareContext:
    workspace: Path
    runtime: Path
    dates: list[str]
    universe_dataset_id: str
    margin_dataset_id: str
    raw_scan: str
    universe_scan: str
    margin_scan: str
    history_scan: str


@dataclass(frozen=True)
class _PreparedFiles:
    eligibility_paths: list[Path]
    margin_paths: list[Path]
    missing_path: Path
    conflicts_path: Path
    missing_count: int
    conflict_count: int
    annual_stats: list[dict[str, Any]]


def _copy_query(
    connection: duckdb.DuckDBPyConnection,
    *,
    query: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    quoted = temporary.resolve().as_posix().replace("'", "''")
    connection.execute(f"COPY ({query}) TO '{quoted}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    os.replace(temporary, path)


def _eligibility_query(*, universe_scan: str, year: int) -> str:
    """Build daily eligibility states without inferring an SSE negative list.

    SZSE publishes an explicit historical eligibility table, so an absent or
    false entry is a known negative.  SSE only publishes daily margin detail;
    presence proves eligibility, while absence carries no eligibility meaning.
    """

    return f"""
      SELECT u.symbol,u.trade_date,u.exchange,
        CASE
          WHEN u.exchange='SH' AND d.symbol IS NOT NULL THEN 'eligible_observed'
          WHEN u.exchange='SH' THEN 'source_unavailable'
          WHEN coalesce(e.eligible,false) THEN 'eligible_observed'
          ELSE 'known_ineligible'
        END AS eligibility_state,
        CASE
          WHEN u.exchange='SH' AND d.symbol IS NOT NULL THEN true
          WHEN u.exchange='SH' THEN NULL::BOOLEAN
          ELSE coalesce(e.eligible,false)
        END AS eligible,
        CASE WHEN u.exchange='SZ' THEN coalesce(e.finance_eligible,false)
             ELSE NULL::BOOLEAN END AS finance_eligible,
        CASE WHEN u.exchange='SZ' THEN coalesce(e.securities_lending_eligible,false)
             ELSE NULL::BOOLEAN END AS securities_lending_eligible,
        d.symbol IS NOT NULL AS detail_observed,
        CASE WHEN u.exchange='SH' THEN d.symbol IS NOT NULL ELSE true END
          AS source_available,
        CASE WHEN u.exchange='SH' THEN d.symbol IS NOT NULL ELSE true END
          AS eligibility_source_available,
        true AS detail_source_available,
        u.trade_date AS source_date,n.feature_available_date,
        false AS burn_in_only,
        CASE WHEN u.exchange='SH' THEN 'sse_official_margin_detail_only'
             ELSE 'szse_official_margin_eligibility+detail' END AS source
      FROM {universe_scan} u
      JOIN next_open_dates n USING(trade_date)
      LEFT JOIN official_eligibility e USING(symbol,trade_date,exchange)
      LEFT JOIN official_detail d USING(symbol,trade_date,exchange)
      WHERE u.trade_date BETWEEN '{year}-01-01' AND '{year}-12-31'
        AND lower(u.board)='main' AND u.exchange IN ('SH','SZ')
      ORDER BY u.trade_date,u.symbol
    """


def _scan(paths: list[Path]) -> str:
    return f"read_parquet([{_sql_paths(paths)}], union_by_name=true, hive_partitioning=false)"


def _prepare_context(workspace: Path) -> _PrepareContext:
    dates = _trade_dates(workspace)
    raw_paths = [_raw_day_path(workspace, trade_date) for trade_date in dates]
    if any(not path.is_file() for path in raw_paths):
        raise MarginEligibilityUpdateError("margin_exchange_raw_day_missing")
    universe_manifest, universe_paths = _dataset_paths(workspace, DataDomain.UNIVERSE_SNAPSHOT)
    margin_manifest, margin_paths = _dataset_paths(workspace, DataDomain.MARGIN_DETAIL)
    _, history_paths = _dataset_paths(workspace, DataDomain.SYMBOL_HISTORY)
    return _PrepareContext(
        workspace=workspace,
        runtime=_runtime(workspace),
        dates=dates,
        universe_dataset_id=universe_manifest.dataset_id,
        margin_dataset_id=margin_manifest.dataset_id,
        raw_scan=_scan(raw_paths),
        universe_scan=_scan(universe_paths),
        margin_scan=_scan(margin_paths),
        history_scan=_scan(history_paths),
    )


def _connect(runtime: Path) -> duckdb.DuckDBPyConnection:
    temp = runtime / "duckdb_tmp"
    temp.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET threads=4")
    quoted = temp.resolve().as_posix().replace("'", "''")
    connection.execute(f"SET temp_directory='{quoted}'")
    return connection


def _register_official_tables(
    connection: duckdb.DuckDBPyConnection,
    context: _PrepareContext,
) -> None:
    connection.execute(
        f"""
        CREATE TEMP TABLE official_detail AS
        SELECT symbol,trade_date,exchange,name,rzye,rqye,rzmre,rqyl,
               rzche,rqchl,rqmcl,rzrqye,source
        FROM {context.raw_scan}
        WHERE endpoint IN ('sse_detail','szse_detail')
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE official_eligibility AS
        SELECT symbol,trade_date,exchange,eligible,
               finance_eligible,securities_lending_eligible,source
        FROM {context.raw_scan}
        WHERE endpoint='szse_eligibility'
        """
    )
    next_dates = pd.DataFrame({"trade_date": context.dates})
    next_dates["feature_available_date"] = [*context.dates[1:], ""]
    connection.register("next_open_dates", next_dates)


def _write_eligibility(
    connection: duckdb.DuckDBPyConnection,
    context: _PrepareContext,
) -> tuple[list[Path], list[dict[str, Any]]]:
    paths: list[Path] = []
    annual_stats: list[dict[str, Any]] = []
    for year in range(2011, 2026):
        path = context.runtime / "prepared" / "margin_eligibility" / f"year={year}" / "part-0000.parquet"
        query = _eligibility_query(universe_scan=context.universe_scan, year=year)
        _copy_query(connection, query=query, path=path)
        paths.append(path)
        stats = connection.execute(
            f"SELECT count(*),count(*) FILTER(WHERE eligible),"
            "count(*) FILTER(WHERE detail_observed),"
            "count(*) FILTER(WHERE eligibility_state='source_unavailable'),"
            "count(*) FILTER(WHERE eligibility_state='known_ineligible') "
            f"FROM ({query})"
        ).fetchone()
        annual_stats.append(
            {
                "year": year,
                "row_count": int(stats[0]),
                "eligible_count": int(stats[1]),
                "detail_observed_count": int(stats[2]),
                "source_unavailable_count": int(stats[3]),
                "known_ineligible_count": int(stats[4]),
            }
        )
    return paths, annual_stats


def _missing_detail_query(context: _PrepareContext) -> str:
    return f"""
          WITH official_main AS (
            SELECT d.* FROM official_detail d
            JOIN {context.universe_scan} u USING(symbol,trade_date,exchange)
            WHERE lower(u.board)='main'
          )
          SELECT h.security_id,o.symbol,o.symbol AS ts_code,o.trade_date,
                 o.rzye,o.rqye,o.rzmre,o.rqyl,o.rzche,o.rqchl,o.rqmcl,o.rzrqye,
                 o.trade_date AS source_date,n.feature_available_date,
                 false AS burn_in_only,o.source
          FROM official_main o
          JOIN {context.history_scan} h ON h.symbol=o.symbol
            AND o.trade_date BETWEEN h.effective_from AND h.effective_to
          JOIN next_open_dates n USING(trade_date)
          LEFT JOIN {context.margin_scan} m
            ON m.symbol=o.symbol AND m.trade_date=o.trade_date
          WHERE m.symbol IS NULL
          ORDER BY o.trade_date,o.symbol
    """


def _conflicts_query(context: _PrepareContext) -> str:
    return f"""
          WITH official_main AS (
            SELECT d.* FROM official_detail d
            JOIN {context.universe_scan} u USING(symbol,trade_date,exchange)
            WHERE lower(u.board)='main'
          ), paired AS (
            SELECT o.symbol,o.trade_date,o.exchange,o.source,
                   unnest(['rzye','rqye','rzmre','rqyl','rzche','rqchl','rqmcl','rzrqye']) AS field,
                   unnest([o.rzye,o.rqye,o.rzmre,o.rqyl,o.rzche,o.rqchl,o.rqmcl,o.rzrqye]) AS official_value,
                   unnest([m.rzye,m.rqye,m.rzmre,m.rqyl,m.rzche,m.rqchl,m.rqmcl,m.rzrqye]) AS qdp_value
            FROM official_main o JOIN {context.margin_scan} m USING(symbol,trade_date)
          )
          SELECT *,abs(official_value-qdp_value) AS absolute_difference,
            abs(official_value-qdp_value)/greatest(abs(official_value),abs(qdp_value),1)
              AS relative_difference
          FROM paired
          WHERE official_value IS NOT NULL AND qdp_value IS NOT NULL
            AND abs(official_value-qdp_value)>greatest(1e-6,
                greatest(abs(official_value),abs(qdp_value),1)*1e-10)
          ORDER BY trade_date,symbol,field
    """


def _write_inventories(
    connection: duckdb.DuckDBPyConnection,
    context: _PrepareContext,
) -> tuple[str, Path, Path, int, int]:
    missing_query = _missing_detail_query(context)
    missing_path = context.runtime / "inventory" / "exchange_detail_missing_from_qdp.parquet"
    _copy_query(connection, query=missing_query, path=missing_path)
    missing_count = int(connection.execute(f"SELECT count(*) FROM ({missing_query})").fetchone()[0])
    conflicts_query = _conflicts_query(context)
    conflicts_path = context.runtime / "inventory" / "exchange_qdp_conflicts.parquet"
    _copy_query(connection, query=conflicts_query, path=conflicts_path)
    conflict_count = int(connection.execute(f"SELECT count(*) FROM ({conflicts_query})").fetchone()[0])
    return missing_query, missing_path, conflicts_path, missing_count, conflict_count


def _write_margin_detail(
    connection: duckdb.DuckDBPyConnection,
    context: _PrepareContext,
    missing_query: str,
) -> list[Path]:
    paths: list[Path] = []
    for year in range(2010, 2026):
        path = context.runtime / "prepared" / "margin_detail" / f"year={year}" / "part-0000.parquet"
        query = f"""
              SELECT {",".join(MARGIN_DETAIL_COLUMNS)}
              FROM {context.margin_scan}
              WHERE trade_date BETWEEN '{year}-01-01' AND '{year}-12-31'
              UNION ALL
              SELECT {",".join(MARGIN_DETAIL_COLUMNS)}
              FROM ({missing_query})
              WHERE trade_date BETWEEN '{year}-01-01' AND '{year}-12-31'
              ORDER BY trade_date,symbol
        """
        _copy_query(connection, query=query, path=path)
        paths.append(path)
    return paths


def _validation_stats(
    connection: duckdb.DuckDBPyConnection,
    *,
    context: _PrepareContext,
    eligibility_paths: list[Path],
    margin_paths: list[Path],
    missing_query: str,
) -> dict[str, Any]:
    eligibility_scan = _scan(eligibility_paths)
    margin_scan = _scan(margin_paths)
    eligibility_stats = connection.execute(
        f"SELECT count(*),count(DISTINCT trade_date),"
        f"count(*) FILTER(WHERE eligibility_state='source_unavailable'),"
        f"count(*) FILTER(WHERE trade_date>'{END_DATE}'),"
        "count(*) FILTER(WHERE trade_date='2024-01-02' AND eligible),"
        "count(*) FILTER(WHERE trade_date='2011-01-04' AND eligible),"
        "count(*) FILTER(WHERE exchange='SH' AND eligibility_state='known_ineligible'),"
        "count(*) FILTER(WHERE exchange='SH' AND eligibility_state='source_unavailable'),"
        "count(*) FILTER(WHERE exchange='SH' AND NOT detail_observed),"
        "count(*) FILTER(WHERE exchange='SZ' AND eligibility_state='source_unavailable'),"
        "count(*) FILTER(WHERE NOT coalesce(("
        "  (eligibility_state='eligible_observed' AND eligible IS true) OR"
        "  (eligibility_state='known_ineligible' AND eligible IS false) OR"
        "  (eligibility_state='source_unavailable' AND eligible IS NULL)),false)),"
        "count(*) FILTER(WHERE NOT coalesce("
        "  source_available=eligibility_source_available AND "
        "  source_available=(eligibility_state<>'source_unavailable') AND "
        "  detail_source_available,false)) "
        f"FROM {eligibility_scan}"
    ).fetchone()
    expected_rows = int(
        connection.execute(
            f"SELECT count(*) FROM {context.universe_scan} "
            f"WHERE trade_date BETWEEN '{START_DATE}' AND '{END_DATE}' "
            "AND lower(board)='main' AND exchange IN ('SH','SZ')"
        ).fetchone()[0]
    )
    margin_stats = connection.execute(
        f"SELECT count(*),count(*)-count(DISTINCT symbol||'|'||trade_date),"
        f"count(*) FILTER(WHERE trade_date>'{END_DATE}') "
        f"FROM {margin_scan}"
    ).fetchone()
    known_000527 = int(
        connection.execute(
            f"SELECT count(*) FROM ({missing_query}) WHERE symbol='000527.SZ' AND trade_date='2011-01-04'"
        ).fetchone()[0]
    )
    return {
        "eligibility": eligibility_stats,
        "expected_rows": expected_rows,
        "margin": margin_stats,
        "known_000527": known_000527,
    }


def _prepare_checks(
    *,
    context: _PrepareContext,
    state: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, bool]:
    eligibility_stats = validation["eligibility"]
    margin_stats = validation["margin"]
    return {
        "every_trade_date_covered": int(eligibility_stats[1]) == len(context.dates),
        "eligibility_row_count_matches_universe": int(eligibility_stats[0]) == validation["expected_rows"],
        "source_unavailable_is_explicit": int(eligibility_stats[2]) > 0,
        "forbidden_2026_eligibility_rows": int(eligibility_stats[3]) == 0,
        "benchmark_2024_01_02_mainboard_eligible_count": int(eligibility_stats[4]) == 1_888,
        "benchmark_2011_01_04_mainboard_eligible_count": int(eligibility_stats[5]) == 90,
        "sse_absence_is_not_known_ineligible": int(eligibility_stats[6]) == 0,
        "sse_absence_is_source_unavailable": int(eligibility_stats[7]) == int(eligibility_stats[8]),
        "szse_explicit_eligibility_has_no_unknown_state": int(eligibility_stats[9]) == 0,
        "eligibility_state_matches_nullable_value": int(eligibility_stats[10]) == 0,
        "source_availability_flags_are_consistent": int(eligibility_stats[11]) == 0,
        "known_000527_gap_repaired": validation["known_000527"] == 1,
        "margin_detail_primary_key_unique": int(margin_stats[1]) == 0,
        "margin_detail_forbidden_2026_rows": int(margin_stats[2]) == 0,
        "request_2026_count_zero": int(state.get("request_2026_count", -1)) == 0,
    }


def prepare(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if state.get("status") in {"prepared", "applied"}:
        return state
    if state.get("status") != "downloaded":
        raise MarginEligibilityUpdateError(f"margin_eligibility_not_downloaded:{state.get('status')}")
    context = _prepare_context(workspace)
    connection = _connect(context.runtime)
    try:
        _register_official_tables(connection, context)
        eligibility_paths, annual_stats = _write_eligibility(connection, context)
        missing_query, missing_path, conflicts_path, missing_count, conflict_count = _write_inventories(
            connection, context
        )
        margin_paths = _write_margin_detail(connection, context, missing_query)
        validation = _validation_stats(
            connection,
            context=context,
            eligibility_paths=eligibility_paths,
            margin_paths=margin_paths,
            missing_query=missing_query,
        )
    finally:
        connection.close()
    checks = _prepare_checks(context=context, state=state, validation=validation)
    if not all(checks.values()):
        raise MarginEligibilityUpdateError(f"margin_prepare_contract_failed:{checks}")
    state.update(
        {
            "status": "prepared",
            "input_dataset_ids": {
                DataDomain.MARGIN_DETAIL: context.margin_dataset_id,
                DataDomain.UNIVERSE_SNAPSHOT: context.universe_dataset_id,
            },
            "prepared": {
                DataDomain.MARGIN_ELIGIBILITY: [str(path) for path in eligibility_paths],
                DataDomain.MARGIN_DETAIL: [str(path) for path in margin_paths],
            },
            "annual_statistics": annual_stats,
            "exchange_detail_missing_count": missing_count,
            "exchange_qdp_conflict_field_count": conflict_count,
            "known_000527_repair_count": validation["known_000527"],
            "inventories": {
                "missing_detail": str(missing_path),
                "conflicts": str(conflicts_path),
            },
            "checks": checks,
        }
    )
    _write_state(workspace, state)
    return state
