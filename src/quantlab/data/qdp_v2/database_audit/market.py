"""Database audit market checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from quantlab.data.qdp_v2.manifest import (
    EXPECTED_BAR_TIMES,
    DatasetManifest,
    resolve_manifest_path,
)
from quantlab.data.qdp_v2.repair.common import _sql_literal

from .common import (
    _path_texts,
)


def _latest_5m_stats(
    con: Any,
    *,
    daily_paths: Sequence[Path],
    intraday_paths: Sequence[Path],
    trade_date: str,
) -> dict[str, Any]:
    times = ",".join(_sql_literal(item) for item in EXPECTED_BAR_TIMES)
    params = [
        _path_texts(intraday_paths),
        trade_date,
        _path_texts(daily_paths),
        trade_date,
    ]
    row = con.execute(
        f"""
        WITH five AS (
          SELECT cast(symbol AS VARCHAR) AS symbol,
                 count(*) AS rows,
                 count(DISTINCT cast(bar_time AS VARCHAR)) AS distinct_times,
                 sum(CASE WHEN cast(bar_time AS VARCHAR) IN ({times}) THEN 0 ELSE 1 END) AS unexpected
          FROM read_parquet(?, union_by_name=true)
          WHERE cast(trade_date AS VARCHAR)=?
          GROUP BY symbol
        ), complete AS (
          SELECT symbol FROM five WHERE rows=48 AND distinct_times=48 AND unexpected=0
        ), traded AS (
          SELECT DISTINCT cast(symbol AS VARCHAR) AS symbol
          FROM read_parquet(?, union_by_name=true)
          WHERE cast(trade_date AS VARCHAR)=? AND coalesce(try_cast(volume AS DOUBLE),0)>0
        )
        SELECT
          (SELECT count(*) FROM traded) AS expected_days,
          (SELECT count(*) FROM complete) AS complete_days,
          (SELECT count(*) FROM five WHERE NOT (rows=48 AND distinct_times=48 AND unexpected=0)) AS invalid_days,
          (SELECT count(*) FROM (SELECT symbol FROM traded EXCEPT SELECT symbol FROM complete)) AS missing_days,
          (SELECT count(*) FROM (SELECT symbol FROM complete EXCEPT SELECT symbol FROM traded)) AS extra_days
        """,
        params,
    ).fetchone()
    expected = int(row[0])
    complete = int(row[1])
    return {
        "expected_traded_day_count": expected,
        "complete_day_count": complete,
        "invalid_day_count": int(row[2]),
        "missing_complete_day_count": int(row[3]),
        "extra_complete_day_count": int(row[4]),
        "coverage_ratio": float(complete / expected) if expected else 1.0,
    }


def _status_daily_partition_check(
    con: Any,
    *,
    status_paths: Sequence[Path],
    daily_paths: Sequence[Path],
    start_date: str,
    end_date: str,
    sample_limit: int,
) -> dict[str, Any]:
    base = """
      WITH daily AS (
        SELECT cast(symbol AS VARCHAR) AS symbol,
               cast(trade_date AS VARCHAR) AS trade_date,
               max(try_cast(volume AS DOUBLE)) AS volume
        FROM read_parquet(?, union_by_name=true)
        WHERE cast(trade_date AS VARCHAR) BETWEEN ? AND ?
        GROUP BY symbol,trade_date
      ), joined AS (
        SELECT cast(s.symbol AS VARCHAR) AS symbol,
               cast(s.trade_date AS VARCHAR) AS trade_date,
               try_cast(s.is_suspended AS BOOLEAN) AS old_suspended,
               d.volume AS daily_volume,
               (d.symbol IS NULL OR coalesce(d.volume,0)<=0) AS expected_suspended
        FROM read_parquet(?, union_by_name=true) s
        LEFT JOIN daily d
          ON cast(s.symbol AS VARCHAR)=d.symbol
         AND cast(s.trade_date AS VARCHAR)=d.trade_date
        WHERE cast(s.trade_date AS VARCHAR) BETWEEN ? AND ?
      )
    """
    params = [
        [str(path) for path in daily_paths],
        str(start_date),
        str(end_date),
        [str(path) for path in status_paths],
        str(start_date),
        str(end_date),
    ]
    row = con.execute(
        base
        + """
          SELECT count(*) AS status_rows,
                 count(*) FILTER (WHERE old_suspended IS NULL) AS null_flags,
                 count(*) FILTER (
                   WHERE coalesce(old_suspended,false)<>expected_suspended
                 ) AS mismatches,
                 count(*) FILTER (
                   WHERE coalesce(old_suspended,false)=false AND expected_suspended
                 ) AS unsuspended_without_positive_daily,
                 count(*) FILTER (
                   WHERE old_suspended=true AND NOT expected_suspended
                 ) AS suspended_with_positive_daily
          FROM joined
        """,
        params,
    ).fetchone()
    remaining = max(0, int(sample_limit))
    examples: list[dict[str, Any]] = []
    if remaining and (int(row[1] or 0) or int(row[2] or 0)):
        examples = (
            con.execute(
                base
                + f"""
              SELECT symbol,trade_date,old_suspended,expected_suspended,daily_volume,
                     CASE
                       WHEN old_suspended IS NULL THEN 'null_suspension_flag'
                       WHEN coalesce(old_suspended,false)=false AND expected_suspended
                         THEN 'unsuspended_without_positive_daily'
                       WHEN old_suspended=true AND NOT expected_suspended
                         THEN 'suspended_with_positive_daily'
                       ELSE 'other'
                     END AS reason
              FROM joined
              WHERE old_suspended IS NULL
                 OR coalesce(old_suspended,false)<>expected_suspended
              ORDER BY trade_date,symbol
              LIMIT {remaining}
            """,
                params,
            )
            .fetchdf()
            .to_dict("records")
        )
    return {
        "status_row_count": int(row[0] or 0),
        "null_suspension_flag_count": int(row[1] or 0),
        "semantic_mismatch_count": int(row[2] or 0),
        "unsuspended_without_positive_daily_count": int(row[3] or 0),
        "suspended_with_positive_daily_count": int(row[4] or 0),
        "examples": examples,
    }


def _status_daily_consistency_check(
    con: Any,
    *,
    root: Path,
    daily: DatasetManifest,
    status: DatasetManifest,
    sample_limit: int,
) -> dict[str, Any]:
    totals = {
        "status_row_count": 0,
        "null_suspension_flag_count": 0,
        "semantic_mismatch_count": 0,
        "unsuspended_without_positive_daily_count": 0,
        "suspended_with_positive_daily_count": 0,
    }
    examples: list[dict[str, Any]] = []
    daily_entries = [(resolve_manifest_path(item.path, root=root), item) for item in daily.shards]
    for status_entry in status.shards:
        start_date = str(status_entry.start_date or "")
        end_date = str(status_entry.end_date or "")
        if not start_date or not end_date:
            raise ValueError("status_consistency_shard_range_missing")
        status_path = resolve_manifest_path(status_entry.path, root=root)
        daily_paths = [
            path
            for path, entry in daily_entries
            if (
                (not entry.start_date or str(entry.start_date) <= end_date)
                and (not entry.end_date or str(entry.end_date) >= start_date)
            )
        ]
        if not daily_paths:
            raise ValueError(f"status_consistency_daily_paths_missing:{start_date}")
        result = _status_daily_partition_check(
            con,
            status_paths=[status_path],
            daily_paths=daily_paths,
            start_date=start_date,
            end_date=end_date,
            sample_limit=max(0, int(sample_limit) - len(examples)),
        )
        for key in totals:
            totals[key] += int(result[key])
        examples.extend(result["examples"])
    return {
        "status": (
            "ok" if not totals["semantic_mismatch_count"] and not totals["null_suspension_flag_count"] else "error"
        ),
        **totals,
        "semantics": "is_suspended iff daily row is absent or volume<=0",
        "examples": examples,
    }


def _consistency_paths(
    *,
    root: Path,
    daily: DatasetManifest,
    intraday: DatasetManifest,
    history: DatasetManifest | None,
) -> tuple[list[str], list[str], list[str], str, str]:
    intraday_entries = [(resolve_manifest_path(item.path, root=root), item) for item in intraday.shards]
    if not intraday_entries:
        raise ValueError("intraday_consistency_shards_missing")
    starts = [str(item.start_date or "") for _, item in intraday_entries]
    ends = [str(item.end_date or "") for _, item in intraday_entries]
    if any(not item for item in starts + ends):
        raise ValueError("intraday_consistency_shard_range_missing")
    start_date = min(starts)
    end_date = max(ends)
    daily_paths = [
        resolve_manifest_path(item.path, root=root)
        for item in daily.shards
        if (
            (not item.start_date or str(item.start_date) <= end_date)
            and (not item.end_date or str(item.end_date) >= start_date)
        )
    ]
    if not daily_paths:
        raise ValueError(f"intraday_consistency_daily_paths_missing:{start_date}")
    intraday_paths = [str(path) for path, _ in intraday_entries]
    history_paths = (
        [str(resolve_manifest_path(item.path, root=root)) for item in history.shards] if history is not None else []
    )
    return (
        [str(path) for path in daily_paths],
        intraday_paths,
        history_paths,
        start_date,
        end_date,
    )


def _five_minute_consistency_tables(
    *,
    intraday_paths: list[str],
    history_paths: list[str],
) -> tuple[str, list[Any]]:
    five_source = """
      five_source AS (
        SELECT upper(cast(symbol AS VARCHAR)) AS source_symbol,
               cast(trade_date AS VARCHAR) AS trade_date,
               count(*) AS bars,
               arg_min(try_cast(open AS DOUBLE),cast(bar_time AS VARCHAR)) AS open,
               max(try_cast(high AS DOUBLE)) AS high,
               min(try_cast(low AS DOUBLE)) AS low,
               arg_max(try_cast(close AS DOUBLE),cast(bar_time AS VARCHAR)) AS close,
               sum(try_cast(volume AS DOUBLE)) AS volume,
               sum(try_cast(amount AS DOUBLE)) AS amount
        FROM read_parquet(?, union_by_name=true)
        GROUP BY source_symbol,trade_date
      )
    """
    if history_paths:
        five_tables = (
            """
              lifecycle_history AS (
                SELECT cast(security_id AS VARCHAR) AS security_id,
                       upper(cast(symbol AS VARCHAR)) AS symbol,
                       cast(effective_from AS VARCHAR) AS effective_from,
                       cast(effective_to AS VARCHAR) AS effective_to
                FROM read_parquet(?, union_by_name=true)
              ), lifecycle_multi_ids AS (
                SELECT security_id FROM lifecycle_history
                GROUP BY security_id HAVING count(DISTINCT symbol)>1
              ), lifecycle AS (
                SELECT h.* FROM lifecycle_history h
                JOIN lifecycle_multi_ids m USING(security_id)
              ),
            """
            + five_source
            + """,
              five_mapped AS (
                SELECT coalesce(target.symbol,f.source_symbol) AS canonical_symbol,
                       f.*
                FROM five_source f
                LEFT JOIN lifecycle source
                  ON f.source_symbol=source.symbol
                LEFT JOIN lifecycle target
                  ON source.security_id=target.security_id
                 AND f.trade_date BETWEEN target.effective_from AND target.effective_to
              ), five AS (
                SELECT canonical_symbol AS symbol,trade_date,bars,open,high,low,
                       close,volume,amount
                FROM five_mapped
                QUALIFY row_number() OVER(
                  PARTITION BY canonical_symbol,trade_date
                  ORDER BY CASE WHEN source_symbol=canonical_symbol THEN 0 ELSE 1 END,
                           source_symbol
                )=1
              )
            """
        )
        params: list[Any] = [history_paths, intraday_paths]
    else:
        five_tables = (
            five_source
            + """,
              five AS (
                SELECT source_symbol AS symbol,trade_date,bars,open,high,low,
                       close,volume,amount
                FROM five_source
              )
            """
        )
        params = [intraday_paths]
    return five_tables, params


def _daily_intraday_join(
    *,
    five_tables: str,
    five_params: list[Any],
    daily_paths: list[str],
    start_date: str,
    end_date: str,
) -> tuple[str, list[Any]]:
    query = f"""
      WITH {five_tables}, day AS (
        SELECT cast(symbol AS VARCHAR) AS symbol,
               cast(trade_date AS VARCHAR) AS trade_date,
               try_cast(open AS DOUBLE) AS open,
               try_cast(high AS DOUBLE) AS high,
               try_cast(low AS DOUBLE) AS low,
               try_cast(close AS DOUBLE) AS close,
               try_cast(volume AS DOUBLE) AS volume,
               try_cast(amount AS DOUBLE) AS amount
        FROM read_parquet(?, union_by_name=true)
        WHERE cast(trade_date AS VARCHAR) BETWEEN ? AND ?
      ), joined AS (
        SELECT coalesce(d.symbol,f.symbol) AS symbol,
               coalesce(d.trade_date,f.trade_date) AS trade_date,
               d.open AS dopen,d.high AS dhigh,d.low AS dlow,d.close AS dclose,
               d.volume AS dvolume,d.amount AS damount,
               f.bars,f.open AS fopen,f.high AS fhigh,f.low AS flow,f.close AS fclose,
               f.volume AS fvolume,f.amount AS famount,
               CASE WHEN d.volume>0 THEN abs(f.volume-d.volume)/greatest(abs(d.volume),1) END AS volume_rel,
               CASE WHEN d.amount>0 THEN abs(f.amount-d.amount)/greatest(abs(d.amount),1) END AS amount_rel,
               CASE WHEN d.volume>0 THEN f.volume/d.volume END AS volume_ratio,
               greatest(
                 abs(f.open-d.open)/greatest(abs(d.open),1),
                 abs(f.high-d.high)/greatest(abs(d.high),1),
                 abs(f.low-d.low)/greatest(abs(d.low),1),
                 abs(f.close-d.close)/greatest(abs(d.close),1)
               ) AS price_rel
        FROM day d FULL OUTER JOIN five f USING(symbol,trade_date)
      )
    """
    params: list[Any] = [
        *five_params,
        daily_paths,
        start_date,
        end_date,
    ]
    return query, params


def _daily_intraday_totals(
    con: Any,
    *,
    joined: str,
    params: list[Any],
) -> dict[str, int]:
    row = con.execute(
        joined
        + """
          SELECT
            count(*) FILTER (WHERE dvolume>0) AS positive_daily,
            count(*) FILTER (WHERE dvolume>0 AND bars=48) AS complete_positive,
            count(*) FILTER (WHERE dvolume>0 AND bars IS NULL) AS missing_positive,
            count(*) FILTER (WHERE dvolume IS NULL AND bars IS NOT NULL) AS missing_daily,
            count(*) FILTER (WHERE coalesce(dvolume,0)<=0 AND bars IS NOT NULL) AS suspended_nonzero,
            count(*) FILTER (WHERE dvolume>0 AND bars IS NOT NULL AND price_rel>0.10) AS price_mismatch,
            count(*) FILTER (WHERE dvolume>0 AND bars IS NOT NULL AND volume_rel>0.05 AND amount_rel>0.05) AS flow_mismatch,
            count(*) FILTER (WHERE dvolume>0 AND bars IS NOT NULL AND (volume_ratio BETWEEN 95 AND 105 OR volume_ratio BETWEEN 0.0095 AND 0.0105)) AS volume_100x,
            count(*) FILTER (WHERE bars IS NOT NULL AND (
              dvolume IS NULL OR coalesce(dvolume,0)<=0 OR price_rel>0.10
              OR (volume_rel>0.05 AND amount_rel>0.05)
              OR volume_ratio BETWEEN 95 AND 105 OR volume_ratio BETWEEN 0.0095 AND 0.0105
            )) AS invalid_days
          FROM joined
        """,
        params,
    ).fetchone()
    keys = (
        "positive_daily_count",
        "complete_positive_daily_count",
        "missing_positive_daily_count",
        "missing_daily_for_5m_count",
        "suspended_nonzero_5m_count",
        "price_mismatch_day_count",
        "flow_mismatch_day_count",
        "volume_100x_day_count",
        "invalid_day_count",
    )
    return {key: int(value or 0) for key, value in zip(keys, row, strict=True)}


def _daily_intraday_examples(
    con: Any,
    *,
    joined: str,
    params: list[Any],
    totals: Mapping[str, int],
    sample_limit: int,
) -> list[dict[str, Any]]:
    remaining = max(0, int(sample_limit))
    if not remaining or not (totals["missing_positive_daily_count"] or totals["invalid_day_count"]):
        return []
    return (
        con.execute(
            joined
            + f"""
                  SELECT symbol,trade_date,bars,dvolume,fvolume,damount,famount,
                         price_rel,volume_rel,amount_rel,volume_ratio,
                         CASE
                           WHEN dvolume>0 AND bars IS NULL THEN 'missing_positive_daily'
                           WHEN dvolume IS NULL AND bars IS NOT NULL THEN 'missing_daily'
                           WHEN coalesce(dvolume,0)<=0 AND bars IS NOT NULL THEN 'suspended_nonzero_5m'
                           WHEN volume_ratio BETWEEN 95 AND 105 OR volume_ratio BETWEEN 0.0095 AND 0.0105 THEN 'volume_100x'
                           WHEN price_rel>0.10 THEN 'price_mismatch'
                           WHEN volume_rel>0.05 AND amount_rel>0.05 THEN 'flow_mismatch'
                           ELSE 'other'
                         END AS reason
                  FROM joined
                  WHERE (dvolume>0 AND bars IS NULL) OR (bars IS NOT NULL AND (
                    dvolume IS NULL OR coalesce(dvolume,0)<=0 OR price_rel>0.10
                    OR (volume_rel>0.05 AND amount_rel>0.05)
                    OR volume_ratio BETWEEN 95 AND 105 OR volume_ratio BETWEEN 0.0095 AND 0.0105
                  ))
                  LIMIT {remaining}
                """,
            params,
        )
        .fetchdf()
        .to_dict("records")
    )


def _daily_intraday_consistency_check(
    con: Any,
    *,
    root: Path,
    daily: DatasetManifest,
    intraday: DatasetManifest,
    history: DatasetManifest | None = None,
    sample_limit: int,
) -> dict[str, Any]:
    daily_paths, intraday_paths, history_paths, start_date, end_date = _consistency_paths(
        root=root,
        daily=daily,
        intraday=intraday,
        history=history,
    )
    five_tables, five_params = _five_minute_consistency_tables(
        intraday_paths=intraday_paths,
        history_paths=history_paths,
    )
    joined, params = _daily_intraday_join(
        five_tables=five_tables,
        five_params=five_params,
        daily_paths=daily_paths,
        start_date=start_date,
        end_date=end_date,
    )
    totals = _daily_intraday_totals(con, joined=joined, params=params)
    examples = _daily_intraday_examples(
        con,
        joined=joined,
        params=params,
        totals=totals,
        sample_limit=sample_limit,
    )
    positive = totals["positive_daily_count"]
    complete = totals["complete_positive_daily_count"]
    return {
        "status": "ok" if not totals["invalid_day_count"] else "error",
        **totals,
        "complete_coverage_ratio": float(complete / positive) if positive else 1.0,
        "examples": examples,
    }


def _factor_change_cte() -> str:
    return """
      WITH factors AS (
        SELECT cast(symbol AS VARCHAR) AS symbol,
               cast(trade_date AS VARCHAR) AS trade_date,
               try_cast(adjust_factor AS DOUBLE) AS factor,
               lag(try_cast(adjust_factor AS DOUBLE)) OVER(
                 PARTITION BY symbol ORDER BY cast(trade_date AS VARCHAR)
               ) AS prior_factor
        FROM read_parquet(?, union_by_name=true)
      ), prices AS (
        SELECT cast(symbol AS VARCHAR) AS symbol,
               cast(trade_date AS VARCHAR) AS trade_date,
               try_cast(open AS DOUBLE) AS open,
               lag(try_cast(close AS DOUBLE)) OVER(
                 PARTITION BY symbol ORDER BY cast(trade_date AS VARCHAR)
               ) AS prior_close,
               lag(cast(trade_date AS VARCHAR)) OVER(
                 PARTITION BY symbol ORDER BY cast(trade_date AS VARCHAR)
               ) AS prior_date
        FROM read_parquet(?, union_by_name=true)
      ), changes AS (
        SELECT f.symbol,f.trade_date,f.factor/f.prior_factor AS factor_ratio,
               p.open/p.prior_close AS raw_open_ratio,
               (f.factor/f.prior_factor)*(p.open/p.prior_close) AS adjusted_open_ratio,
               date_diff('day',try_cast(p.prior_date AS DATE),try_cast(f.trade_date AS DATE)) AS gap_days
        FROM factors f JOIN prices p USING(symbol,trade_date)
        WHERE f.prior_factor>0 AND p.prior_close>0
          AND (f.factor/f.prior_factor<0.99 OR f.factor/f.prior_factor>1.01)
      )
    """


def _factor_change_examples(
    con: Any,
    *,
    cte: str,
    params: list[list[str]],
    condition: str,
    sample_limit: int,
) -> list[dict[str, Any]]:
    return (
        con.execute(
            cte
            + f"""
              SELECT symbol,trade_date,gap_days,factor_ratio,raw_open_ratio,adjusted_open_ratio
              FROM changes
              WHERE {condition}
              ORDER BY abs(adjusted_open_ratio-1) DESC
              LIMIT {int(sample_limit)}
            """,
            params,
        )
        .fetchdf()
        .to_dict("records")
    )


def _factor_change_totals(
    con: Any,
    *,
    cte: str,
    params: list[list[str]],
) -> tuple[tuple[Any, ...], int]:
    row = con.execute(
        cte
        + """
          SELECT count(*) AS changes,
                 count(*) FILTER (WHERE (adjusted_open_ratio<0.70 OR adjusted_open_ratio>1.30)
                   AND gap_days<=10 AND raw_open_ratio BETWEEN 0.70 AND 1.30) AS factor_induced,
                 count(*) FILTER (WHERE (adjusted_open_ratio<0.70 OR adjusted_open_ratio>1.30)
                   AND NOT (gap_days<=10 AND raw_open_ratio BETWEEN 0.70 AND 1.30)) AS raw_discontinuity,
                 coalesce(max(abs(adjusted_open_ratio-1)),0) AS max_error,
                 count(*) FILTER (WHERE symbol='600076.SH' AND substr(trade_date,1,4)='2024') AS regression_changes,
                 coalesce(max(abs(adjusted_open_ratio-1)) FILTER (
                   WHERE symbol='600076.SH' AND substr(trade_date,1,4)='2024'
                 ),0) AS regression_max_error
          FROM changes
        """,
        params,
    ).fetchone()
    excessive = con.execute(
        cte
        + """
          SELECT count(*) FROM (
            SELECT symbol,substr(trade_date,1,4) AS year,count(*) AS changes
            FROM changes GROUP BY symbol,year HAVING count(*)>12
          )
        """,
        params,
    ).fetchone()[0]
    return row, int(excessive or 0)


def _factor_semantic_check(
    con: Any,
    *,
    root: Path,
    daily: DatasetManifest,
    factor: DatasetManifest,
    sample_limit: int,
) -> dict[str, Any]:
    daily_paths = [str(resolve_manifest_path(item.path, root=root)) for item in daily.shards]
    factor_paths = [str(resolve_manifest_path(item.path, root=root)) for item in factor.shards]
    cte = _factor_change_cte()
    params = [factor_paths, daily_paths]
    row, excessive = _factor_change_totals(con, cte=cte, params=params)
    factor_induced = (
        "(adjusted_open_ratio<0.70 OR adjusted_open_ratio>1.30) "
        "AND gap_days<=10 AND raw_open_ratio BETWEEN 0.70 AND 1.30"
    )
    examples = (
        _factor_change_examples(
            con,
            cte=cte,
            params=params,
            condition=factor_induced,
            sample_limit=sample_limit,
        )
        if int(row[1] or 0)
        else []
    )
    raw_examples = (
        _factor_change_examples(
            con,
            cte=cte,
            params=params,
            condition=f"(adjusted_open_ratio<0.70 OR adjusted_open_ratio>1.30) AND NOT ({factor_induced})",
            sample_limit=sample_limit,
        )
        if int(row[2] or 0)
        else []
    )
    return {
        "status": "ok" if not int(row[1] or 0) and not int(excessive or 0) else "error",
        "factor_change_count": int(row[0] or 0),
        "uncompensated_change_count": int(row[1] or 0),
        "raw_discontinuity_count": int(row[2] or 0),
        "max_compensation_error": float(row[3] or 0),
        "excessive_change_symbol_year_count": int(excessive or 0),
        "regression_600076_2024_change_count": int(row[4] or 0),
        "regression_600076_2024_max_compensation_error": float(row[5] or 0),
        "examples": examples,
        "raw_discontinuity_examples": raw_examples,
    }
