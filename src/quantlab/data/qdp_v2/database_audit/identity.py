"""Database audit identity checks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    dataset_manifest_for_id,
    read_dataset_manifest,
    resolve_manifest_path,
)

from .common import (
    _path_texts,
)


def _except_count(
    con: Any,
    left_paths: Sequence[Path],
    right_paths: Sequence[Path],
    trade_date: str,
) -> int:
    return int(
        con.execute(
            """
            SELECT count(*) FROM (
              SELECT cast(symbol AS VARCHAR), cast(trade_date AS VARCHAR)
              FROM read_parquet(?, union_by_name=true)
              WHERE cast(trade_date AS VARCHAR)=?
              EXCEPT
              SELECT cast(symbol AS VARCHAR), cast(trade_date AS VARCHAR)
              FROM read_parquet(?, union_by_name=true)
              WHERE cast(trade_date AS VARCHAR)=?
            )
            """,
            [_path_texts(left_paths), trade_date, _path_texts(right_paths), trade_date],
        ).fetchone()[0]
    )


def _active_manifest(root: Path, dataset_id: str, domain: str) -> DatasetManifest:
    path = dataset_manifest_for_id(root, dataset_id, domain)
    if path is None:
        raise FileNotFoundError(f"active_dataset_manifest_missing:{domain}:{dataset_id}")
    return read_dataset_manifest(path)


def _paths_for_date(root: Path, manifest: DatasetManifest, trade_date: str) -> tuple[Path, ...]:
    return tuple(
        resolve_manifest_path(item.path, root=root)
        for item in manifest.shards
        if (
            (not item.start_date or str(item.start_date) <= trade_date)
            and (not item.end_date or str(item.end_date) >= trade_date)
        )
    )


def _identity_history_consistency_check(
    con: Any,
    *,
    root: Path,
    identity: DatasetManifest,
    history: DatasetManifest,
    sample_limit: int,
) -> dict[str, Any]:
    identity_paths = [str(resolve_manifest_path(item.path, root=root)) for item in identity.shards]
    history_paths = [str(resolve_manifest_path(item.path, root=root)) for item in history.shards]
    sql = """
      WITH identities AS (
        SELECT DISTINCT cast(security_id AS VARCHAR) AS security_id
        FROM read_parquet(?, union_by_name=true)
      ), histories AS (
        SELECT DISTINCT cast(security_id AS VARCHAR) AS security_id,
               upper(trim(cast(symbol AS VARCHAR))) AS symbol
        FROM read_parquet(?, union_by_name=true)
      )
    """
    row = con.execute(
        sql
        + """
        SELECT
          (SELECT count(*) FROM histories h ANTI JOIN identities i USING(security_id)),
          (SELECT count(*) FROM identities i ANTI JOIN histories h USING(security_id)),
          (SELECT count(*) FROM (
             SELECT symbol FROM histories GROUP BY symbol
             HAVING count(DISTINCT security_id)>1
           ) conflicts)
        """,
        [identity_paths, history_paths],
    ).fetchone()
    orphan_history = int(row[0] or 0)
    identity_without_history = int(row[1] or 0)
    symbol_conflicts = int(row[2] or 0)
    examples = (
        con.execute(
            sql
            + f"""
        SELECT 'orphan_history' AS reason,h.security_id,h.symbol
        FROM histories h ANTI JOIN identities i USING(security_id)
        LIMIT {int(sample_limit)}
        """,
            [identity_paths, history_paths],
        )
        .fetchdf()
        .to_dict("records")
        if orphan_history
        else []
    )
    return {
        "status": ("ok" if not orphan_history and not identity_without_history and not symbol_conflicts else "error"),
        "orphan_symbol_history_count": orphan_history,
        "identity_without_symbol_history_count": identity_without_history,
        "symbol_multiple_identity_count": symbol_conflicts,
        "examples": examples,
    }


def _long_suspension_check(
    con: Any,
    *,
    root: Path,
    status: DatasetManifest,
    calendar: DatasetManifest,
    sample_limit: int,
    minimum_open_days: int = 20,
) -> dict[str, Any]:
    status_paths = [str(resolve_manifest_path(item.path, root=root)) for item in status.shards]
    calendar_paths = [str(resolve_manifest_path(item.path, root=root)) for item in calendar.shards]
    base = """
      WITH open_dates AS (
        SELECT DISTINCT cast(trade_date AS VARCHAR) AS trade_date
        FROM read_parquet(?, union_by_name=true)
        WHERE coalesce(try_cast(is_open AS BOOLEAN), false)
      ), ordered AS (
        SELECT cast(s.symbol AS VARCHAR) AS symbol,
               cast(s.trade_date AS VARCHAR) AS trade_date,
               coalesce(try_cast(s.is_suspended AS BOOLEAN), false) AS is_suspended,
               lead(coalesce(try_cast(s.is_suspended AS BOOLEAN), false)) OVER(
                 PARTITION BY cast(s.symbol AS VARCHAR)
                 ORDER BY cast(s.trade_date AS VARCHAR)
               ) AS next_is_suspended,
               sum(CASE WHEN coalesce(try_cast(s.is_suspended AS BOOLEAN), false)
                        THEN 0 ELSE 1 END) OVER(
                 PARTITION BY cast(s.symbol AS VARCHAR)
                 ORDER BY cast(s.trade_date AS VARCHAR)
                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               ) AS segment_id
        FROM read_parquet(?, union_by_name=true) s
        JOIN open_dates d ON d.trade_date=cast(s.trade_date AS VARCHAR)
      ), runs AS (
        SELECT symbol,min(trade_date) AS start_date,max(trade_date) AS end_date,
               count(*) AS suspended_open_days,
               max(CASE WHEN next_is_suspended=false THEN 1 ELSE 0 END) AS has_reopen_break
        FROM ordered
        WHERE is_suspended
        GROUP BY symbol,segment_id
        HAVING count(*) >= ?
      )
    """
    params = [calendar_paths, status_paths, int(minimum_open_days)]
    row = con.execute(
        base
        + """
        SELECT count(*),count(DISTINCT symbol),
               coalesce(sum(has_reopen_break),0),
               coalesce(sum(suspended_open_days),0)
        FROM runs
        """,
        params,
    ).fetchone()
    examples = (
        con.execute(
            base
            + f"""
        SELECT symbol,start_date,end_date,suspended_open_days,
               cast(has_reopen_break AS BOOLEAN) AS continuity_break
        FROM runs ORDER BY suspended_open_days DESC,symbol,start_date
        LIMIT {int(sample_limit)}
        """,
            params,
        )
        .fetchdf()
        .to_dict("records")
    )
    return {
        "status": "ok",
        "minimum_suspended_open_days": int(minimum_open_days),
        "long_suspension_interval_count": int(row[0] or 0),
        "affected_symbol_count": int(row[1] or 0),
        "continuity_break_count": int(row[2] or 0),
        "long_suspension_open_day_count": int(row[3] or 0),
        "research_rule": ("lookbacks, forward labels, and execution tails may not cross a qualifying reopen break"),
        "examples": examples,
    }


def _reopen_discontinuity_sql() -> str:
    return """
      WITH prices AS (
        SELECT cast(symbol AS VARCHAR) AS symbol,
               cast(trade_date AS VARCHAR) AS trade_date,
               try_cast(open AS DOUBLE) AS open,
               lag(try_cast(close AS DOUBLE)) OVER(
                 PARTITION BY cast(symbol AS VARCHAR)
                 ORDER BY cast(trade_date AS VARCHAR)
               ) AS prior_close,
               lag(cast(trade_date AS VARCHAR)) OVER(
                 PARTITION BY cast(symbol AS VARCHAR)
                 ORDER BY cast(trade_date AS VARCHAR)
               ) AS prior_trade_date,
               row_number() OVER(
                 PARTITION BY cast(symbol AS VARCHAR)
                 ORDER BY cast(trade_date AS VARCHAR)
               ) AS observation_number,
               min(cast(trade_date AS VARCHAR)) OVER(
                 PARTITION BY cast(symbol AS VARCHAR)
               ) AS first_trade_date
        FROM read_parquet(?, union_by_name=true)
        WHERE try_cast(close AS DOUBLE)>0
      ), factors AS (
        SELECT cast(symbol AS VARCHAR) AS symbol,
               cast(trade_date AS VARCHAR) AS trade_date,
               try_cast(adjust_factor AS DOUBLE) AS factor,
               lag(try_cast(adjust_factor AS DOUBLE)) OVER(
                 PARTITION BY cast(symbol AS VARCHAR)
                 ORDER BY cast(trade_date AS VARCHAR)
               ) AS prior_factor
        FROM read_parquet(?, union_by_name=true)
      ), gaps AS (
        SELECT p.symbol,p.prior_trade_date,p.trade_date,
               p.observation_number,p.first_trade_date,i.list_date,
               p.open/p.prior_close AS raw_open_ratio,
               f.factor/f.prior_factor AS factor_ratio,
               (p.open/p.prior_close)*(f.factor/f.prior_factor) AS adjusted_open_ratio
        FROM prices p
        LEFT JOIN factors f USING(symbol,trade_date)
        LEFT JOIN (
          SELECT upper(trim(cast(current_symbol AS VARCHAR))) AS symbol,
                 cast(list_date AS VARCHAR) AS list_date
          FROM read_parquet(?, union_by_name=true)
        ) i USING(symbol)
        WHERE p.prior_close>0
          AND (p.open/p.prior_close<0.70 OR p.open/p.prior_close>1.30)
      ), open_dates AS (
        SELECT DISTINCT cast(trade_date AS VARCHAR) AS trade_date
        FROM read_parquet(?, union_by_name=true)
        WHERE coalesce(try_cast(is_open AS BOOLEAN), false)
      ), status_rows AS (
        SELECT cast(symbol AS VARCHAR) AS symbol,
               cast(trade_date AS VARCHAR) AS trade_date,
               coalesce(try_cast(is_suspended AS BOOLEAN), false) AS is_suspended
        FROM read_parquet(?, union_by_name=true)
      ), gap_status AS (
        SELECT g.symbol,g.prior_trade_date,g.trade_date,g.observation_number,
               g.first_trade_date,g.list_date,g.raw_open_ratio,
               g.factor_ratio,g.adjusted_open_ratio,
               count(d.trade_date) AS intervening_open_days,
               count(s.trade_date) AS covered_status_days,
               count(s.trade_date) FILTER (WHERE s.is_suspended) AS suspended_status_days,
               count(s.trade_date) FILTER (WHERE NOT s.is_suspended) AS tradable_status_days
        FROM gaps g
        LEFT JOIN open_dates d
          ON d.trade_date>g.prior_trade_date AND d.trade_date<g.trade_date
        LEFT JOIN status_rows s
          ON s.symbol=g.symbol AND s.trade_date=d.trade_date
        GROUP BY g.symbol,g.prior_trade_date,g.trade_date,g.observation_number,
                 g.first_trade_date,g.list_date,g.raw_open_ratio,
                 g.factor_ratio,g.adjusted_open_ratio
      ), classified AS (
        SELECT *,
          CASE
            WHEN first_trade_date=list_date AND observation_number<=5
              THEN 'listing_price_discovery'
            WHEN intervening_open_days>0
             AND covered_status_days=intervening_open_days
             AND suspended_status_days=intervening_open_days
              THEN 'reopen_discontinuity'
            WHEN factor_ratio IS NOT NULL
             AND (factor_ratio<0.99 OR factor_ratio>1.01)
             AND adjusted_open_ratio BETWEEN 0.70 AND 1.30
              THEN 'factor_compensated_corporate_action'
            ELSE 'unexpected_discontinuity'
          END AS classification
        FROM gap_status
      )
    """


def _manifest_paths(root: Path, manifest: DatasetManifest) -> list[str]:
    return [str(resolve_manifest_path(item.path, root=root)) for item in manifest.shards]


def _reopen_discontinuity_check(
    con: Any,
    *,
    root: Path,
    daily: DatasetManifest,
    factor: DatasetManifest,
    status: DatasetManifest,
    calendar: DatasetManifest,
    identity: DatasetManifest,
    sample_limit: int,
) -> dict[str, Any]:
    base = _reopen_discontinuity_sql()
    params = [
        _manifest_paths(root, daily),
        _manifest_paths(root, factor),
        _manifest_paths(root, identity),
        _manifest_paths(root, calendar),
        _manifest_paths(root, status),
    ]
    row = con.execute(
        base
        + """
        SELECT count(*),
               count(*) FILTER (WHERE classification='reopen_discontinuity'),
               count(*) FILTER (WHERE classification='factor_compensated_corporate_action'),
               count(*) FILTER (WHERE classification='listing_price_discovery'),
               count(*) FILTER (WHERE classification='unexpected_discontinuity'),
               count(*) FILTER (WHERE intervening_open_days>covered_status_days),
               count(*) FILTER (WHERE tradable_status_days>0)
        FROM classified
        """,
        params,
    ).fetchone()
    examples = (
        con.execute(
            base
            + f"""
        SELECT symbol,prior_trade_date,trade_date,raw_open_ratio,factor_ratio,
               adjusted_open_ratio,intervening_open_days,covered_status_days,
               suspended_status_days,classification
        FROM classified
        ORDER BY CASE WHEN classification='unexpected_discontinuity' THEN 0 ELSE 1 END,
                 symbol,trade_date
        LIMIT {int(sample_limit)}
        """,
            params,
        )
        .fetchdf()
        .to_dict("records")
    )
    return {
        "status": "ok" if not int(row[4] or 0) else "error",
        "raw_discontinuity_count": int(row[0] or 0),
        "reopen_discontinuity_count": int(row[1] or 0),
        "factor_compensated_corporate_action_count": int(row[2] or 0),
        "listing_price_discovery_count": int(row[3] or 0),
        "unexpected_discontinuity_count": int(row[4] or 0),
        "status_missing_interval_count": int(row[5] or 0),
        "tradable_day_inside_gap_count": int(row[6] or 0),
        "examples": examples,
    }
