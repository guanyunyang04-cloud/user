"""Auxiliary update names operations."""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.qdp_v2 import normalization as _normalization
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb

from .baostock import (
    _valid_parquet_columns,
)
from .context import (
    TUSHARE_WORKERS,
    AuxiliaryContext,
    AuxiliaryUpdateError,
    _copy_query,
    _paths,
    _replace_domain,
    _resolve_tushare_token,
    _scan_sql,
    _TushareClient,
)
from .corporate import (
    _current_symbols,
)


def _fetch_name_change_parts(ctx: AuxiliaryContext) -> list[Path]:
    token = _resolve_tushare_token(ctx.workspace)
    if not token:
        raise AuxiliaryUpdateError("tushare_token_required_for_name_change_repair")
    output_dir = ctx.runtime / "name_change_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    client = _TushareClient(token, workspace_root=ctx.workspace)
    symbols = _current_symbols(ctx)

    def fetch_one(symbol: str) -> Path:
        path = output_dir / f"namechange_{symbol.replace('.', '_')}.parquet"
        if _valid_parquet_columns(path, NAME_INTERVAL_COLUMNS):
            return path
        path.unlink(missing_ok=True)
        raw = client.fetch(
            "namechange",
            params={"ts_code": symbol},
            fields=NAME_CHANGE_FIELDS,
        )
        frame = _normalize_name_intervals(raw, target_date=ctx.target_date)
        if frame.empty:
            frame = pd.DataFrame(columns=NAME_INTERVAL_COLUMNS)
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
        futures = {pool.submit(fetch_one, symbol): symbol for symbol in symbols}
        completed = 0
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                outputs.append(future.result())
            except Exception as exc:
                raise AuxiliaryUpdateError(f"name_change_partition_failed:{symbol}:{type(exc).__name__}:{exc}") from exc
            completed += 1
            if completed % 200 == 0:
                print(
                    f"auxiliary_name_change={completed}/{len(symbols)}",
                    flush=True,
                )
    return sorted(outputs)


def _name_change_repair_query(
    *,
    intervals: str,
    universe: str,
    target_date: str,
) -> str:
    return f"""
    WITH interval_transitions AS (
      SELECT symbol, start_date AS trade_date,
             lag(name) OVER (PARTITION BY symbol ORDER BY start_date) AS old_name,
             name AS new_name
      FROM {intervals}
      WHERE start_date<='{target_date}'
    ), tushare_events AS (
      SELECT symbol, trade_date, old_name, new_name,
             'short_name' AS change_type,
             'tushare_namechange_intervals' AS source
      FROM interval_transitions
      WHERE trade_date>='2010-01-04'
        AND old_name IS NOT NULL AND old_name<>new_name
    ), universe_normalized AS (
      SELECT symbol, trade_date, trim(name) AS name
      FROM {universe}
      WHERE trade_date BETWEEN '2010-01-04' AND '{target_date}'
        AND coalesce(trim(name), '')<>''
    ), universe_transitions AS (
      SELECT symbol, trade_date,
             lag(name) OVER(PARTITION BY symbol ORDER BY trade_date) AS old_name,
             name AS new_name
      FROM universe_normalized
    ), universe_events AS (
      SELECT symbol, trade_date, old_name, new_name,
             'short_name' AS change_type,
             'qdp_universe_observed_name_transition' AS source
      FROM universe_transitions
      WHERE old_name IS NOT NULL AND old_name<>new_name
    ), latest_interval AS (
      SELECT symbol, name
      FROM {intervals}
      QUALIFY row_number() OVER(
        PARTITION BY symbol ORDER BY start_date DESC
      )=1
    ), latest_universe AS (
      SELECT symbol, trim(name) AS name
      FROM {universe}
      WHERE trade_date='{target_date}'
    ), conflicts AS (
      SELECT u.symbol
      FROM latest_universe u
      LEFT JOIN latest_interval i USING(symbol)
      WHERE i.symbol IS NULL OR i.name<>u.name
    ), resolved_events AS (
      SELECT * FROM tushare_events
      WHERE symbol NOT IN (SELECT symbol FROM conflicts)
      UNION ALL
      SELECT * FROM universe_events
      WHERE symbol IN (SELECT symbol FROM conflicts)
    )
    SELECT symbol, trade_date, old_name, new_name,
           change_type, source
    FROM resolved_events
    ORDER BY trade_date, symbol
    """


@dataclass(frozen=True)
class _NameChangeDiagnostics:
    invalid: int
    latest_missing: int
    latest_mismatch: int
    compared: int
    source_conflicts: int


def _name_change_diagnostics(
    ctx: AuxiliaryContext,
    *,
    intervals: str,
    universe: str,
    produced: str,
) -> _NameChangeDiagnostics:
    spill = ctx.runtime / "name_validate_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=1) as con:
        invalid = int(
            con.execute(
                f"SELECT count(*) FROM {produced} WHERE "
                "coalesce(trim(old_name),'')='' OR coalesce(trim(new_name),'')='' "
                "OR old_name=new_name"
            ).fetchone()[0]
        )
        latest = con.execute(
            f"""
            WITH latest_interval AS (
              SELECT symbol, name
              FROM {intervals}
              QUALIFY row_number() OVER (
                PARTITION BY symbol ORDER BY start_date DESC
              )=1
            ), latest_universe AS (
              SELECT symbol, trim(name) AS name
              FROM {universe}
              WHERE trade_date='{ctx.target_date}'
            ), first_universe AS (
              SELECT symbol, trim(name) AS name
              FROM {universe}
              WHERE trade_date<='{ctx.target_date}'
              QUALIFY row_number() OVER(
                PARTITION BY symbol ORDER BY trade_date
              )=1
            ), latest_event AS (
              SELECT symbol, new_name
              FROM {produced}
              QUALIFY row_number() OVER(
                PARTITION BY symbol ORDER BY trade_date DESC
              )=1
            )
            SELECT count(*) FILTER(WHERE i.symbol IS NULL),
                   count(*) FILTER(WHERE coalesce(e.new_name, f.name)<>u.name),
                   count(*) FILTER(WHERE i.symbol IS NOT NULL),
                   count(*) FILTER(WHERE i.symbol IS NULL OR i.name<>u.name)
            FROM latest_universe u
            LEFT JOIN latest_interval i USING(symbol)
            LEFT JOIN first_universe f USING(symbol)
            LEFT JOIN latest_event e USING(symbol)
            """
        ).fetchone()
    shutil.rmtree(spill, ignore_errors=True)
    return _NameChangeDiagnostics(
        invalid=invalid,
        latest_missing=int(latest[0] or 0),
        latest_mismatch=int(latest[1] or 0),
        compared=int(latest[2] or 0),
        source_conflicts=int(latest[3] or 0),
    )


def repair_name_change(ctx: AuxiliaryContext) -> dict[str, Any]:
    parts = _fetch_name_change_parts(ctx)
    intervals = _scan_sql(parts)
    universe = _scan_sql(_paths(ctx, "universe_snapshot"))
    prepared = ctx.runtime / "name_change.prepared.parquet"
    _copy_query(
        ctx,
        sql=_name_change_repair_query(
            intervals=intervals,
            universe=universe,
            target_date=ctx.target_date,
        ),
        target=prepared,
    )
    produced = _scan_sql([prepared])
    diagnostics = _name_change_diagnostics(
        ctx,
        intervals=intervals,
        universe=universe,
        produced=produced,
    )
    if diagnostics.invalid or diagnostics.latest_missing or diagnostics.latest_mismatch:
        raise AuxiliaryUpdateError(
            "name_change_contract_failed:"
            f"invalid={diagnostics.invalid}:latest_missing={diagnostics.latest_missing}:"
            f"latest_mismatch={diagnostics.latest_mismatch}"
        )
    result = _replace_domain(
        ctx,
        domain="name_change",
        prepared=prepared,
        primary_key=("symbol", "trade_date", "change_type"),
        contract_version="qdp_v2_name_change_strict_pit_v2",
        source_contract=(
            "Tushare name intervals; QDP daily-universe observed transitions "
            "arbitrate symbols whose latest provider name conflicts"
        ),
        validation={
            "secondary_compared_count": diagnostics.compared,
            "secondary_material_mismatch_count": diagnostics.latest_mismatch,
            "secondary_validation_status": "ok",
        },
    )
    shutil.rmtree(ctx.runtime / "name_change_parts", ignore_errors=True)
    return {
        **result,
        "symbol_request_count": len(parts),
        "tushare_universe_conflict_count": diagnostics.source_conflicts,
    }


_normalize_name_intervals = _normalization.normalize_name_intervals


NAME_CHANGE_FIELDS = _normalization.NAME_CHANGE_FIELDS


NAME_INTERVAL_COLUMNS = _normalization.NAME_INTERVAL_COLUMNS
