"""Tushare Extended Backfill: prepare responsibilities."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

from quantlab.data.domains.contracts import DataDomain
from quantlab.data.qdp_v2.manifest import (
    _manifest_schema_from_arrow,
    atomic_write_json,
)

from .config import (
    BURN_IN_START,
    END_DATE,
    FORBIDDEN_YEAR,
    MARGIN_DETAIL_FIELDS,
    MARGIN_FIELDS,
    MONEYFLOW_FIELDS,
    PREPARED_TRANSFORM_VERSION,
    RESEARCH_START,
    EndpointSpec,
    TushareExtendedBackfillError,
)
from .context import (
    _active_paths,
    _runtime,
    _scan,
    _sha256,
    _stable_hash,
    _validate_schema,
)
from .download import (
    _task_keys,
    _valid_success,
)


def _year_page_paths(workspace: Path, spec: EndpointSpec, year: int) -> list[Path]:
    root = _runtime(workspace) / "raw" / spec.name / f"year={year}"
    if not root.is_dir():
        return []
    if spec.mode == "year":
        paths = sorted(root.glob(f"task={year}/offset=*.parquet"))
    else:
        paths = sorted(
            path
            for task_root in root.glob("task=*")
            if (task_key := task_root.name.removeprefix("task=")).isdigit() and len(task_key) == 8
            for path in task_root.glob("offset=*.parquet")
        )
    return [path for path in paths if path.is_file()]


def _sql_date(column: str) -> str:
    value = f"cast({column} AS VARCHAR)"
    return (
        f"CASE WHEN length({value})=8 THEN substr({value},1,4)||'-'||"
        f"substr({value},5,2)||'-'||substr({value},7,2) ELSE substr({value},1,10) END"
    )


def _copy_query(connection: duckdb.DuckDBPyConnection, sql: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    quoted = str(temporary).replace("'", "''")
    connection.execute(f"COPY ({sql}) TO '{quoted}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)")
    os.replace(temporary, path)


def _provider_fields(paths: Sequence[Path]) -> tuple[str, ...]:
    if not paths:
        return ()
    return tuple(pq.read_schema(paths[0]).names)


def _quoted(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _provider_value_expression(spec: EndpointSpec, *, alias: str, field: str) -> str:
    reference = f"{alias}.{_quoted(field)}"
    text_fields = {"exchange_id"}
    if spec.name == "margin-secs":
        text_fields.update({"name", "exchange"})
    target_type = "VARCHAR" if field in text_fields else "DOUBLE"
    return f"try_cast({reference} AS {target_type}) AS {_quoted(field)}"


def _prepared_sql(
    *,
    spec: EndpointSpec,
    pages: Sequence[Path],
    history_paths: Sequence[Path],
    calendar_paths: Sequence[Path],
) -> str:
    fields = _provider_fields(pages)
    _validate_schema(spec, fields)
    date_expr = _sql_date("r.trade_date")
    provider_values = [field for field in fields if field not in {"ts_code", "trade_date"}]
    provider_sql = ",\n             ".join(
        _provider_value_expression(spec, alias="r", field=field) for field in provider_values
    )
    provider_sql = (",\n             " + provider_sql) if provider_sql else ""
    page_scan = _scan(pages)
    calendar_scan = _scan(calendar_paths)
    availability = (
        "n.trade_date"
        if spec.name == "stk-factor-pro"
        else "CASE WHEN c.next_trade_date<='2025-12-31' THEN c.next_trade_date ELSE '' END"
    )
    common = f"""
    WITH open_calendar AS (
      SELECT trade_date,
             lead(trade_date) OVER (ORDER BY trade_date) AS next_trade_date
      FROM (
        SELECT DISTINCT cast(trade_date AS VARCHAR) AS trade_date
        FROM {calendar_scan}
        WHERE is_open=true AND cast(trade_date AS VARCHAR)<='2026-01-31'
      )
    ), raw AS (
      SELECT * FROM {page_scan}
    ), normalized AS (
      SELECT cast(r.ts_code AS VARCHAR) AS ts_code,
             upper(cast(r.ts_code AS VARCHAR)) AS symbol,
             {date_expr} AS trade_date{provider_sql}
      FROM raw r
    ), history AS (
      SELECT symbol,min(security_id) AS security_id
      FROM {_scan(history_paths)}
      GROUP BY symbol
      HAVING count(DISTINCT security_id)=1
    ), mapped AS (
      SELECT h.security_id,n.*,n.trade_date AS source_date,
             {availability} AS feature_available_date,
             n.trade_date<'{RESEARCH_START}' AS burn_in_only,
             'tushare_compatible.{spec.api_name}' AS source
      FROM normalized n
      JOIN history h USING(symbol)
      LEFT JOIN open_calendar c USING(trade_date)
      WHERE n.trade_date BETWEEN '{BURN_IN_START}' AND '{END_DATE}'
    )
    """
    if spec.name == "margin":
        values = [field for field in fields if field != "trade_date"]
        selected = ",\n                 ".join(
            _provider_value_expression(spec, alias="r", field=field) for field in values
        )
        return f"""
        WITH open_calendar AS (
          SELECT trade_date,
                 lead(trade_date) OVER (ORDER BY trade_date) AS next_trade_date
          FROM (
            SELECT DISTINCT cast(trade_date AS VARCHAR) AS trade_date
            FROM {calendar_scan}
            WHERE is_open=true AND cast(trade_date AS VARCHAR)<='2026-01-31'
          )
        ), raw AS (
          SELECT * FROM {page_scan}
        ), normalized AS (
          SELECT {_sql_date("r.trade_date")} AS trade_date,
                 {selected}
          FROM raw r
        )
        SELECT trade_date,{",".join(_quoted(field) for field in values)},
               trade_date AS source_date,
               CASE WHEN c.next_trade_date<='{END_DATE}'
                    THEN c.next_trade_date ELSE '' END AS feature_available_date,
               trade_date<'{RESEARCH_START}' AS burn_in_only,
               'tushare_compatible.margin' AS source
        FROM normalized n
        LEFT JOIN open_calendar c USING(trade_date)
        WHERE trade_date BETWEEN '{BURN_IN_START}' AND '{END_DATE}'
        QUALIFY row_number() OVER (
          PARTITION BY trade_date,exchange_id ORDER BY trade_date
        )=1
        ORDER BY trade_date,exchange_id
        """
    output_fields = ["security_id", "symbol", "ts_code", "trade_date", *provider_values]
    return (
        common
        + f"""
        SELECT {",".join(_quoted(field) for field in output_fields)},
               source_date,feature_available_date,burn_in_only,source
        FROM mapped
        QUALIFY row_number() OVER (
          PARTITION BY trade_date,security_id ORDER BY trade_date
        )=1
        ORDER BY trade_date,security_id
        """
    )


def _validate_prepared(path: Path, spec: EndpointSpec, year: int) -> dict[str, Any]:
    numeric_fields: tuple[str, ...] = ()
    allowed_negative_fields: tuple[str, ...] = ()
    if spec.name == "margin":
        numeric_fields = tuple(field for field in MARGIN_FIELDS if field not in {"trade_date", "exchange_id", "rzche"})
    elif spec.name == "margin-detail":
        numeric_fields = tuple(
            field for field in MARGIN_DETAIL_FIELDS if field not in {"trade_date", "ts_code", "rzche", "rqchl"}
        )
    elif spec.name == "moneyflow":
        numeric_fields = tuple(
            field for field in MONEYFLOW_FIELDS if field not in {"trade_date", "ts_code", "net_mf_vol", "net_mf_amount"}
        )
        allowed_negative_fields = ("net_mf_vol", "net_mf_amount")
    negative_sql = (
        "+".join(f"count(*) FILTER(WHERE try_cast({_quoted(field)} AS DOUBLE)<0)" for field in numeric_fields) or "0"
    )
    allowed_negative_sql = (
        "+".join(f"count(*) FILTER(WHERE try_cast({_quoted(field)} AS DOUBLE)<0)" for field in allowed_negative_fields)
        or "0"
    )
    relation_sql = "0"
    source_exception_sql = "0"
    if spec.name == "margin":
        negative_sql = f"({negative_sql})+count(*) FILTER(WHERE try_cast(rzche AS DOUBLE)<0 AND exchange_id<>'BSE')"
        source_exception_sql = "count(*) FILTER(WHERE try_cast(rzche AS DOUBLE)<0 AND exchange_id='BSE')"
    elif spec.name == "margin-detail":
        source_exception_sql = (
            "count(*) FILTER(WHERE try_cast(rzche AS DOUBLE)<0)+count(*) FILTER(WHERE try_cast(rqchl AS DOUBLE)<0)"
        )
    if spec.name == "moneyflow":
        relation_sql = """
        count(*) FILTER(
          WHERE net_mf_amount IS NOT NULL
            AND buy_lg_amount IS NOT NULL AND buy_elg_amount IS NOT NULL
            AND sell_lg_amount IS NOT NULL AND sell_elg_amount IS NOT NULL
            AND abs(try_cast(net_mf_amount AS DOUBLE)-(
              try_cast(buy_lg_amount AS DOUBLE)+try_cast(buy_elg_amount AS DOUBLE)-
              try_cast(sell_lg_amount AS DOUBLE)-try_cast(sell_elg_amount AS DOUBLE)
            ))>greatest(1.0,abs(try_cast(net_mf_amount AS DOUBLE))*0.000001)
        )
        """
    with duckdb.connect() as connection:
        quoted = str(path).replace("'", "''")
        keys = ",".join(_quoted(item) for item in spec.primary_key)
        row = connection.execute(
            f"""
            SELECT count(*),count(*)-count(DISTINCT ({keys})),
                   min(trade_date),max(trade_date),
                   count(*) FILTER(WHERE trade_date>='{FORBIDDEN_YEAR}-01-01'),
                   count(*) FILTER(WHERE feature_available_date<>'' AND
                     feature_available_date<source_date),
                   count(*) FILTER(WHERE burn_in_only<>(trade_date<'{RESEARCH_START}')),
                   {negative_sql},
                   {allowed_negative_sql},
                   {relation_sql},
                   {source_exception_sql}
            FROM read_parquet('{quoted}')
            """
        ).fetchone()
    if int(row[1] or 0) or int(row[4] or 0) or int(row[5] or 0) or int(row[6] or 0) or int(row[7] or 0):
        raise TushareExtendedBackfillError(f"prepared_contract_failed:{spec.domain}:{year}:{tuple(row)}")
    return {
        "year": int(year),
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "row_count": int(row[0]),
        "start_date": str(row[2] or ""),
        "end_date": str(row[3] or ""),
        "primary_key_unique": True,
        "forbidden_2026_rows": int(row[4] or 0),
        "availability_before_source_rows": int(row[5] or 0),
        "burn_in_flag_mismatch_rows": int(row[6] or 0),
        "disallowed_negative_value_count": int(row[7] or 0),
        "allowed_negative_net_value_count": int(row[8] or 0),
        "moneyflow_main_net_relation_mismatch_count": int(row[9] or 0),
        "source_exception_negative_value_count": int(row[10] or 0),
    }


def _prepare_domain(workspace: Path, spec: EndpointSpec, open_dates: Sequence[str]) -> list[dict[str, Any]]:
    history_paths = _active_paths(workspace, DataDomain.SYMBOL_HISTORY)
    calendar_paths = _active_paths(workspace, DataDomain.TRADING_CALENDAR)
    records: list[dict[str, Any]] = []
    for year in range(2010, 2026):
        expected_keys = _task_keys(spec, [date for date in open_dates if date.startswith(str(year))])
        if spec.mode == "year":
            expected_keys = [str(year)]
        incomplete = [key for key in expected_keys if _valid_success(workspace, spec, key) is None]
        if incomplete:
            raise TushareExtendedBackfillError(f"raw_tasks_incomplete:{spec.name}:{year}:{len(incomplete)}")
        pages = _year_page_paths(workspace, spec, year)
        if not pages:
            raise TushareExtendedBackfillError(f"raw_pages_missing:{spec.name}:{year}")
        path = _runtime(workspace) / "prepared" / spec.domain / f"year={year}" / "part-0000.parquet"
        input_hash = _stable_hash(
            {
                "prepared_transform_version": PREPARED_TRANSFORM_VERSION,
                "endpoint": spec.name,
                "contract_version": spec.contract_version,
                "page_sha256": [_sha256(page) for page in pages],
            }
        )
        sidecar = path.with_suffix(".json")
        reusable = False
        if path.is_file() and sidecar.is_file():
            previous = json.loads(sidecar.read_text(encoding="utf-8"))
            reusable = previous.get("input_hash") == input_hash and previous.get("sha256") == _sha256(path)
        if not reusable:
            with duckdb.connect() as connection:
                connection.execute("SET threads=4")
                connection.execute("SET memory_limit='12GB'")
                _copy_query(
                    connection,
                    _prepared_sql(
                        spec=spec,
                        pages=pages,
                        history_paths=history_paths,
                        calendar_paths=calendar_paths,
                    ),
                    path,
                )
            validation = _validate_prepared(path, spec, year)
            atomic_write_json(sidecar, {**validation, "input_hash": input_hash})
        records.append({**_validate_prepared(path, spec, year), "input_hash": input_hash})
        print(json.dumps({"prepared": spec.domain, "year": year}), flush=True)
    return records


def _assert_uniform_prepared_schema(spec: EndpointSpec, records: Sequence[Mapping[str, Any]]) -> None:
    schemas: dict[str, list[int]] = {}
    for record in records:
        path = Path(str(record["path"]))
        digest = _stable_hash(_manifest_schema_from_arrow(pq.read_schema(path)))
        schemas.setdefault(digest, []).append(int(record["year"]))
    if len(schemas) != 1:
        detail = ";".join(
            f"{digest[:12]}={','.join(str(year) for year in years)}" for digest, years in sorted(schemas.items())
        )
        raise TushareExtendedBackfillError(f"prepared_schema_drift:{spec.domain}:{detail}")


def _factor_validation(workspace: Path, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    factor_paths = [Path(str(item["path"])) for item in records]
    daily_paths = _active_paths(workspace, "market_daily_raw")
    adjust_paths = _active_paths(workspace, DataDomain.ADJUST_FACTOR)
    with duckdb.connect() as connection:
        connection.execute("SET threads=4")
        connection.execute("SET memory_limit='12GB'")
        row = connection.execute(
            f"""
            WITH f AS (
              SELECT symbol,trade_date,
                     try_cast(open AS DOUBLE) AS open,
                     try_cast(high AS DOUBLE) AS high,
                     try_cast(low AS DOUBLE) AS low,
                     try_cast(close AS DOUBLE) AS close,
                     try_cast(close_hfq AS DOUBLE) AS close_hfq,
                     try_cast(adj_factor AS DOUBLE) AS adj_factor
              FROM {_scan(factor_paths)}
            ), first_factor AS (
              SELECT symbol,arg_min(adj_factor,trade_date) AS first_adj_factor
              FROM f WHERE adj_factor>0 GROUP BY symbol
            ), pairs AS (
              SELECT f.*,d.open AS d_open,d.high AS d_high,d.low AS d_low,
                     d.close AS d_close,a.back_adjust_factor,ff.first_adj_factor
              FROM f
              JOIN {_scan(daily_paths)} d USING(symbol,trade_date)
              JOIN {_scan(adjust_paths)} a USING(symbol,trade_date)
              JOIN first_factor ff USING(symbol)
              WHERE f.trade_date BETWEEN '{BURN_IN_START}' AND '{END_DATE}'
                AND abs(hash(f.symbol||f.trade_date)%97)=0
            )
            SELECT count(*),
              count(*) FILTER(WHERE
                abs(open/d_open-1)>0.005 OR abs(high/d_high-1)>0.005 OR
                abs(low/d_low-1)>0.005 OR abs(close/d_close-1)>0.005),
              count(*) FILTER(WHERE close>0 AND adj_factor>0 AND
                abs(close_hfq/(close*adj_factor)-1)>0.0001),
              count(*) FILTER(WHERE first_adj_factor>0 AND back_adjust_factor>0 AND
                abs((adj_factor/first_adj_factor)/back_adjust_factor-1)>0.005)
            FROM pairs
            """
        ).fetchone()
    compared = int(row[0] or 0)
    raw_mismatch = int(row[1] or 0)
    hfq_identity_mismatch = int(row[2] or 0)
    factor_mismatch = int(row[3] or 0)
    raw_rate = raw_mismatch / compared if compared else 1.0
    hfq_rate = (hfq_identity_mismatch + factor_mismatch) / compared if compared else 1.0
    return {
        "sample_compared_count": compared,
        "raw_price_mismatch_count": raw_mismatch,
        "raw_price_mismatch_rate": raw_rate,
        "hfq_identity_mismatch_count": hfq_identity_mismatch,
        "normalized_adjust_factor_mismatch_count": factor_mismatch,
        "hfq_validation_mismatch_rate": hfq_rate,
        "raw_price_validation_passed": compared > 0 and raw_rate <= 0.005,
        "hfq_validation_passed": compared > 0 and hfq_rate <= 0.005,
    }
