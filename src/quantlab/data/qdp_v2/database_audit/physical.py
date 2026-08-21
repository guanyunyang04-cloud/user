"""Database audit physical checks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.qdp_v2.manifest import (
    EXPECTED_BAR_TIMES,
    ShardManifestEntry,
)
from quantlab.data.qdp_v2.repair import _sql_literal

from .common import (
    _path_texts,
    _q,
)


def _primary_key_check(
    con: Any,
    paths: Sequence[Path],
    primary_key: Sequence[str],
    sample_limit: int,
) -> dict[str, Any]:
    keys = ",".join(_q(item) for item in primary_key)
    nulls = " OR ".join(f"{_q(item)} IS NULL" for item in primary_key)
    null_count = int(
        con.execute(
            f"SELECT count(*) FROM read_parquet(?, union_by_name=true) WHERE {nulls}",
            [_path_texts(paths)],
        ).fetchone()[0]
    )
    duplicate_rows = int(
        con.execute(
            f"""
            SELECT coalesce(sum(n - 1), 0) FROM (
              SELECT count(*) AS n
              FROM read_parquet(?, union_by_name=true)
              GROUP BY {keys}
              HAVING count(*) > 1
            )
            """,
            [_path_texts(paths)],
        ).fetchone()[0]
    )
    examples = (
        con.execute(
            f"""
        SELECT {keys}, count(*) AS rows
        FROM read_parquet(?, union_by_name=true)
        GROUP BY {keys}
        HAVING count(*) > 1
        LIMIT {int(sample_limit)}
        """,
            [_path_texts(paths)],
        )
        .fetchdf()
        .to_dict("records")
        if duplicate_rows
        else []
    )
    return {
        "status": "ok" if not null_count and not duplicate_rows else "error",
        "null_key_rows": null_count,
        "duplicate_rows": duplicate_rows,
        "examples": examples,
    }


def _invalid_order_predicate(
    order: Sequence[str],
    prior_names: dict[str, str],
) -> str:
    comparisons: list[str] = []
    for index, key in enumerate(order):
        prefix = " AND ".join(f"{earlier}={prior_names[earlier]}" for earlier in order[:index])
        operator = "<=" if index == len(order) - 1 else "<"
        comparison = f"{key}{operator}{prior_names[key]}"
        comparisons.append(f"({prefix} AND {comparison})" if prefix else f"({comparison})")
    return f"{prior_names[order[0]]} IS NOT NULL AND (" + " OR ".join(comparisons) + ")"


def _scan_shard_physical_order(
    con: Any,
    *,
    paths: Sequence[Path],
    key_order: list[str],
    sample_limit: int,
) -> tuple[int, int, list[dict[str, Any]], dict[str, int]]:
    prior_names = {
        "symbol": "prior_symbol",
        "trade_date": "prior_date",
        "bar_time": "prior_time",
    }
    candidate_orders = [key_order]
    legacy_order = ["symbol", "trade_date", "bar_time"]
    if legacy_order != key_order:
        candidate_orders.append(legacy_order)
    predicates = [_invalid_order_predicate(order, prior_names) for order in candidate_orders]
    selected = ",\n               ".join(f"cast({key} AS VARCHAR) AS {key}" for key in key_order)
    lagged = ",\n               ".join(
        f"lag(cast({key} AS VARCHAR)) OVER () AS {prior_names[key]}" for key in key_order
    )
    ordered_cte = f"""
      WITH ordered AS (
        SELECT {selected},
               {lagged}
        FROM read_parquet(?, union_by_name=true)
      )
    """
    null_predicate = " OR ".join(f"{key} IS NULL" for key in key_order)
    null_count = 0
    invalid_count = 0
    examples: list[dict[str, Any]] = []
    observed_orders: dict[str, int] = {}
    for path in paths:
        count_expressions = ",\n                     ".join(
            f"coalesce(sum(CASE WHEN {predicate} THEN 1 ELSE 0 END),0)" for predicate in predicates
        )
        row = con.execute(
            ordered_cte
            + f"""
              SELECT coalesce(sum(CASE WHEN {null_predicate} THEN 1 ELSE 0 END),0),
                     {count_expressions}
              FROM ordered
            """,
            [[str(path)]],
        ).fetchone()
        null_count += int(row[0] or 0)
        candidate_counts = [int(item or 0) for item in row[1:]]
        chosen_index = min(
            range(len(candidate_counts)),
            key=lambda item: (candidate_counts[item], item),
        )
        chosen_order = candidate_orders[chosen_index]
        chosen_key = ">".join(chosen_order)
        observed_orders[chosen_key] = observed_orders.get(chosen_key, 0) + 1
        shard_invalid = candidate_counts[chosen_index]
        invalid_count += shard_invalid
        remaining = max(0, int(sample_limit) - len(examples))
        if not shard_invalid or not remaining:
            continue
        rows = (
            con.execute(
                ordered_cte
                + f"""
                  SELECT symbol,trade_date,bar_time,prior_symbol,prior_date,prior_time
                  FROM ordered WHERE {predicates[chosen_index]}
                  LIMIT {remaining}
                """,
                [[str(path)]],
            )
            .fetchdf()
            .to_dict("records")
        )
        for item in rows:
            item["shard"] = str(path)
        examples.extend(rows)
    return null_count, invalid_count, examples, observed_orders


def _overlapping_shard_pairs(
    paths: Sequence[Path],
    entries: Sequence[ShardManifestEntry],
) -> list[tuple[tuple[int, str, str, Path], tuple[int, str, str, Path]]]:
    ranged = [
        (index, str(entry.start_date), str(entry.end_date), path)
        for index, (path, entry) in enumerate(zip(paths, entries, strict=True))
        if str(entry.start_date) and str(entry.end_date)
    ]
    return [
        (left, right)
        for position, left in enumerate(ranged)
        for right in ranged[position + 1 :]
        if left[1] <= right[2] and right[1] <= left[2]
    ]


def _cross_shard_duplicate_check(
    con: Any,
    *,
    pairs: Sequence[tuple[tuple[int, str, str, Path], tuple[int, str, str, Path]]],
    examples: list[dict[str, Any]],
    sample_limit: int,
) -> tuple[int, int]:
    symbol_sets: dict[int, set[str]] = {}
    for item in {part[0]: part for pair in pairs for part in pair}.values():
        symbol_sets[item[0]] = {
            str(row[0])
            for row in con.execute(
                "SELECT DISTINCT cast(symbol AS VARCHAR) FROM read_parquet(?, union_by_name=true)",
                [[str(item[3])]],
            ).fetchall()
        }
    duplicate_total = 0
    verified_pairs = 0
    for left, right in pairs:
        common_symbols = sorted(symbol_sets[left[0]].intersection(symbol_sets[right[0]]))
        if not common_symbols:
            continue
        verified_pairs += 1
        overlap_start = max(left[1], right[1])
        overlap_end = min(left[2], right[2])
        con.register(
            "qdp_intraday_overlap_symbols",
            pd.DataFrame({"symbol": common_symbols}),
        )
        try:
            params = [
                [str(left[3])],
                [str(right[3])],
                overlap_start,
                overlap_end,
                overlap_start,
                overlap_end,
            ]
            join_sql = """
                FROM read_parquet(?, union_by_name=true) l
                JOIN qdp_intraday_overlap_symbols s
                  ON cast(l.symbol AS VARCHAR)=s.symbol
                JOIN read_parquet(?, union_by_name=true) r
                  ON cast(l.symbol AS VARCHAR)=cast(r.symbol AS VARCHAR)
                 AND cast(l.trade_date AS VARCHAR)=cast(r.trade_date AS VARCHAR)
                 AND cast(l.bar_time AS VARCHAR)=cast(r.bar_time AS VARCHAR)
                WHERE cast(l.trade_date AS VARCHAR) BETWEEN ? AND ?
                  AND cast(r.trade_date AS VARCHAR) BETWEEN ? AND ?
            """
            duplicate_count = int(con.execute("SELECT count(*) " + join_sql, params).fetchone()[0])
            duplicate_total += duplicate_count
            remaining = max(0, int(sample_limit) - len(examples))
            if duplicate_count and remaining:
                rows = (
                    con.execute(
                        """
                        SELECT cast(l.symbol AS VARCHAR) AS symbol,
                               cast(l.trade_date AS VARCHAR) AS trade_date,
                               cast(l.bar_time AS VARCHAR) AS bar_time
                        """
                        + join_sql
                        + " LIMIT ?",
                        [*params, remaining],
                    )
                    .fetchdf()
                    .to_dict("records")
                )
                for row in rows:
                    row["left_shard"] = str(left[3])
                    row["right_shard"] = str(right[3])
                examples.extend(rows)
        finally:
            con.unregister("qdp_intraday_overlap_symbols")
    return duplicate_total, verified_pairs


def _ordered_intraday_primary_key_check(
    con: Any,
    paths: Sequence[Path],
    entries: Sequence[ShardManifestEntry],
    primary_key: Sequence[str],
    sample_limit: int,
) -> dict[str, Any]:
    if set(primary_key) != {"symbol", "trade_date", "bar_time"}:
        return {
            "status": "error",
            "null_key_rows": 0,
            "duplicate_rows": 1,
            "range_overlap_count": 0,
            "examples": [
                {
                    "reason": "unexpected_intraday_primary_key",
                    "primary_key": list(primary_key),
                }
            ],
            "method": "partitioned_physical_order",
        }
    key_order = [str(item) for item in primary_key]
    null_count, invalid_count, examples, observed_orders = _scan_shard_physical_order(
        con,
        paths=paths,
        key_order=key_order,
        sample_limit=sample_limit,
    )
    overlapping_pairs = _overlapping_shard_pairs(paths, entries)
    cross_shard_duplicates, verified_pair_count = _cross_shard_duplicate_check(
        con,
        pairs=overlapping_pairs,
        examples=examples,
        sample_limit=sample_limit,
    )
    range_overlap_count = len(overlapping_pairs)
    duplicate_rows = invalid_count + cross_shard_duplicates
    return {
        "status": "ok" if not null_count and not duplicate_rows else "error",
        "null_key_rows": null_count,
        "duplicate_rows": duplicate_rows,
        "range_overlap_count": range_overlap_count,
        "verified_overlap_pair_count": verified_pair_count,
        "cross_shard_duplicate_rows": cross_shard_duplicates,
        "physical_order_violation_rows": invalid_count,
        "declared_primary_key_order": key_order,
        "observed_physical_key_orders": observed_orders,
        "examples": examples,
        "method": "partitioned_physical_order+verified_cross_shard_keys",
    }


def _ohlc_check(con: Any, paths: Sequence[Path], sample_limit: int) -> dict[str, Any]:
    predicate = """
      NOT isfinite(try_cast(open AS DOUBLE))
      OR NOT isfinite(try_cast(high AS DOUBLE))
      OR NOT isfinite(try_cast(low AS DOUBLE))
      OR NOT isfinite(try_cast(close AS DOUBLE))
      OR try_cast(open AS DOUBLE) <= 0
      OR try_cast(high AS DOUBLE) <= 0
      OR try_cast(low AS DOUBLE) <= 0
      OR try_cast(close AS DOUBLE) <= 0
      OR try_cast(high AS DOUBLE) < greatest(try_cast(open AS DOUBLE), try_cast(low AS DOUBLE), try_cast(close AS DOUBLE))
      OR try_cast(low AS DOUBLE) > least(try_cast(open AS DOUBLE), try_cast(high AS DOUBLE), try_cast(close AS DOUBLE))
      OR try_cast(volume AS DOUBLE) < 0
      OR try_cast(amount AS DOUBLE) < 0
    """
    invalid = int(
        con.execute(
            f"SELECT count(*) FROM read_parquet(?, union_by_name=true) WHERE {predicate}",
            [_path_texts(paths)],
        ).fetchone()[0]
    )
    examples = (
        con.execute(
            f"""
        SELECT symbol, trade_date, open, high, low, close, volume, amount
        FROM read_parquet(?, union_by_name=true)
        WHERE {predicate}
        LIMIT {int(sample_limit)}
        """,
            [_path_texts(paths)],
        )
        .fetchdf()
        .to_dict("records")
        if invalid
        else []
    )
    return {
        "status": "ok" if not invalid else "error",
        "invalid_rows": invalid,
        "examples": examples,
    }


def _bar_day_check(con: Any, paths: Sequence[Path], sample_limit: int) -> dict[str, Any]:
    times = ",".join(_sql_literal(item) for item in EXPECTED_BAR_TIMES)
    query = f"""
      SELECT symbol, trade_date,
             count(*) AS rows,
             count(DISTINCT cast(bar_time AS VARCHAR)) AS distinct_times,
             sum(CASE WHEN cast(bar_time AS VARCHAR) IN ({times}) THEN 0 ELSE 1 END) AS unexpected_times
      FROM read_parquet(?, union_by_name=true)
      GROUP BY symbol, trade_date
      HAVING rows <> 48 OR distinct_times <> 48 OR unexpected_times <> 0
    """
    invalid = 0
    examples: list[dict[str, Any]] = []
    # Complete stock-days are stored in one date-partitioned shard. Checking
    # each shard keeps the 48-bar aggregation bounded by one year instead of
    # creating an avoidable all-history hash table for 472M rows.
    for path in paths:
        params = [[str(path)]]
        shard_invalid = int(con.execute(f"SELECT count(*) FROM ({query})", params).fetchone()[0])
        invalid += shard_invalid
        remaining = max(0, int(sample_limit) - len(examples))
        if shard_invalid and remaining:
            rows = (
                con.execute(
                    f"SELECT * FROM ({query}) LIMIT {remaining}",
                    params,
                )
                .fetchdf()
                .to_dict("records")
            )
            for row in rows:
                row["shard"] = str(path)
            examples.extend(rows)
    return {
        "status": "ok" if not invalid else "error",
        "invalid_day_count": invalid,
        "examples": examples,
    }


def _factor_check(con: Any, paths: Sequence[Path], sample_limit: int) -> dict[str, Any]:
    predicate = "NOT isfinite(try_cast(adjust_factor AS DOUBLE)) OR try_cast(adjust_factor AS DOUBLE) <= 0"
    invalid = int(
        con.execute(
            f"SELECT count(*) FROM read_parquet(?, union_by_name=true) WHERE {predicate}",
            [_path_texts(paths)],
        ).fetchone()[0]
    )
    examples = (
        con.execute(
            f"SELECT symbol, trade_date, adjust_factor FROM read_parquet(?, union_by_name=true) WHERE {predicate} LIMIT {int(sample_limit)}",
            [_path_texts(paths)],
        )
        .fetchdf()
        .to_dict("records")
        if invalid
        else []
    )
    return {
        "status": "ok" if not invalid else "error",
        "invalid_rows": invalid,
        "examples": examples,
    }


def _symbol_history_check(con: Any, paths: Sequence[Path], sample_limit: int) -> dict[str, Any]:
    overlap_query = """
      SELECT a.security_id, a.symbol AS left_symbol, b.symbol AS right_symbol,
             a.effective_from AS left_from, a.effective_to AS left_to,
             b.effective_from AS right_from, b.effective_to AS right_to
      FROM read_parquet(?, union_by_name=true) a
      JOIN read_parquet(?, union_by_name=true) b
        ON cast(a.security_id AS VARCHAR)=cast(b.security_id AS VARCHAR)
       AND (cast(a.symbol AS VARCHAR), cast(a.effective_from AS VARCHAR))
           < (cast(b.symbol AS VARCHAR), cast(b.effective_from AS VARCHAR))
       AND cast(a.effective_from AS VARCHAR) <= cast(b.effective_to AS VARCHAR)
       AND cast(b.effective_from AS VARCHAR) <= cast(a.effective_to AS VARCHAR)
    """
    params = [_path_texts(paths), _path_texts(paths)]
    count = int(con.execute(f"SELECT count(*) FROM ({overlap_query})", params).fetchone()[0])
    examples = (
        con.execute(f"SELECT * FROM ({overlap_query}) LIMIT {int(sample_limit)}", params).fetchdf().to_dict("records")
        if count
        else []
    )
    return {
        "status": "ok" if not count else "error",
        "overlap_count": count,
        "examples": examples,
    }
