"""PIT history lifecycle_prepare operations."""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb
from quantlab.data.qdp_v2.repair import (
    _sql_literal,
    resolve_active_domain,
)

from .context import (
    PitHistoryContext,
    PitHistoryError,
)
from .lifecycle_audit import (
    _parquet_scan,
    _quoted_identifier,
)


def _copy_lifecycle_query(con: Any, query: str, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        con.execute(
            f"COPY ({query}) TO {_sql_literal(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(target).metadata.num_rows)


def _normalized_lifecycle_projection(
    *,
    context: Any,
) -> tuple[str, str]:
    columns = [str(item.get("name", "")) for item in context.manifest.schema]
    columns = [item for item in columns if item]
    expressions: list[str] = []
    for column in columns:
        quoted = _quoted_identifier(column)
        if column == "symbol":
            expressions.append(f"m.canonical_symbol AS {quoted}")
        elif context.domain == "universe_snapshot" and column == "name":
            expressions.append(
                f"CASE WHEN m.remap_priority=1 THEN "
                f"coalesce(nullif(m.target_name,''),m.{quoted}) "
                f"ELSE m.{quoted} END AS {quoted}"
            )
        elif context.domain == "universe_snapshot" and column == "list_date":
            expressions.append(
                f"CASE WHEN m.remap_priority=1 THEN "
                f"coalesce(nullif(m.target_list_date,''),m.{quoted}) "
                f"ELSE m.{quoted} END AS {quoted}"
            )
        elif context.domain == "universe_snapshot" and column == "delist_date":
            expressions.append(
                f"CASE WHEN m.remap_priority=1 THEN m.target_delist_date ELSE m.{quoted} END AS {quoted}"
            )
        elif context.domain == "adjust_factor" and column in {
            "adjust_factor",
            "fore_adjust_factor",
            "back_adjust_factor",
        }:
            expressions.append(
                f"CASE WHEN m.remap_priority=1 THEN "
                f"try_cast(m.{quoted} AS DOUBLE)*m.factor_scale "
                f"ELSE m.{quoted} END AS {quoted}"
            )
        elif context.domain == "adjust_factor" and column == "source":
            expressions.append(
                f"CASE WHEN m.remap_priority=1 "
                f"AND abs(m.factor_scale-1.0)>1e-12 THEN "
                f"concat(coalesce(cast(m.{quoted} AS VARCHAR),''),"
                f"'+lifecycle_factor_scale_overlap_v1') "
                f"ELSE m.{quoted} END AS {quoted}"
            )
        else:
            expressions.append(f"m.{quoted}")
    projection = ",".join(expressions)
    partition = ",".join(_quoted_identifier(item) for item in context.manifest.primary_key)
    return projection, partition


def _manifest_column_projection(*, context: Any, alias: str) -> str:
    columns = [str(item.get("name", "")) for item in context.manifest.schema]
    columns = [item for item in columns if item]
    if not columns:
        raise PitHistoryError(f"pit_history_symbol_lifecycle_manifest_schema_missing:{context.domain}")
    return ",".join(f"{alias}.{_quoted_identifier(column)}" for column in columns)


def _validate_lifecycle_prepared_schemas(
    *,
    context: Any,
    paths: Sequence[Path],
) -> None:
    import pyarrow.parquet as pq

    reference = pq.ParquetFile(context.shard_paths[0]).schema_arrow
    expected_columns = [str(item.get("name", "")) for item in context.manifest.schema]
    expected_columns = [item for item in expected_columns if item]
    for path in paths:
        actual = pq.ParquetFile(path).schema_arrow
        if actual.names != expected_columns or not actual.equals(
            reference,
            check_metadata=False,
        ):
            raise PitHistoryError(
                "pit_history_symbol_lifecycle_prepared_schema_mismatch:"
                f"{context.domain}:{path.name}:"
                f"expected={reference}:actual={actual}"
            )


def _validate_lifecycle_source_schemas(
    *,
    context: Any,
    paths: Sequence[Path],
) -> None:
    import pyarrow.parquet as pq

    reference = pq.ParquetFile(context.shard_paths[0]).schema_arrow
    for path in paths:
        actual = pq.ParquetFile(path).schema_arrow
        if actual.equals(reference, check_metadata=False):
            continue
        if set(reference.names).issubset(actual.names) and all(
            actual.field(field.name).equals(field, check_metadata=False) for field in reference
        ):
            continue
        raise PitHistoryError(
            "pit_history_symbol_lifecycle_source_schema_incompatible:"
            f"{context.domain}:{path.name}:"
            f"expected={reference}:actual={actual}"
        )


@dataclass
class _RetainedShardPlan:
    replacements: list[tuple[Path, Path]]
    removals: list[Path]
    affected: list[Path]
    kept_rows: int


def _prepare_retained_lifecycle_shards(
    con: Any,
    *,
    context: Any,
    prepared_dir: Path,
) -> _RetainedShardPlan:
    replacements: list[tuple[Path, Path]] = []
    removals: list[Path] = []
    affected: list[Path] = []
    kept_rows = 0
    retained_projection = _manifest_column_projection(context=context, alias="d")
    for index, old_path in enumerate(context.shard_paths):
        row = con.execute(
            f"SELECT count(*) FROM {_parquet_scan([old_path])} d "
            "JOIN lifecycle_symbol_map s "
            "ON upper(cast(d.symbol AS VARCHAR))=s.symbol",
        ).fetchone()
        if not int(row[0] or 0):
            continue
        affected.append(old_path)
        target = prepared_dir / f"retained_{index:04d}.parquet"
        query = (
            f"SELECT {retained_projection} "
            f"FROM {_parquet_scan([old_path])} d "
            "LEFT JOIN lifecycle_symbol_map s "
            "ON upper(cast(d.symbol AS VARCHAR))=s.symbol "
            "WHERE s.symbol IS NULL"
        )
        count = _copy_lifecycle_query(con, query, target)
        if count:
            replacements.append((old_path, target))
            kept_rows += count
        else:
            target.unlink(missing_ok=True)
            removals.append(old_path)
    return _RetainedShardPlan(replacements, removals, affected, kept_rows)


def _missing_lifecycle_mapping_count(con: Any, scans: str) -> int:
    row = con.execute(
        f"WITH source_rows AS ("
        f" SELECT d.*,upper(cast(d.symbol AS VARCHAR)) source_symbol,"
        f" s.security_id FROM {scans} d JOIN lifecycle_symbol_map s "
        " ON upper(cast(d.symbol AS VARCHAR))=s.symbol"
        ") SELECT count(*) FROM source_rows r LEFT JOIN lifecycle_intervals i "
        " ON i.security_id=r.security_id AND cast(r.trade_date AS VARCHAR) "
        " BETWEEN i.effective_from AND i.effective_to WHERE i.symbol IS NULL"
    ).fetchone()
    return int(row[0] or 0)


def _lifecycle_mapping_ctes(domain: str) -> str:
    if domain != "adjust_factor":
        return """
          mapped AS (
            SELECT r.*,i.symbol canonical_symbol,
                   i.name_on_date target_name,
                   i.list_date target_list_date,
                   i.canonical_delist_date target_delist_date,
                   CASE WHEN r.source_symbol=i.symbol THEN 0 ELSE 1 END remap_priority,
                   1.0 AS factor_scale
            FROM source_rows r
            JOIN lifecycle_intervals i
              ON i.security_id=r.security_id
             AND cast(r.trade_date AS VARCHAR)
                 BETWEEN i.effective_from AND i.effective_to
          )
        """
    return """
      mapped_base AS (
        SELECT r.*,i.symbol canonical_symbol,
               i.name_on_date target_name,
               i.list_date target_list_date,
               i.canonical_delist_date target_delist_date,
               CASE WHEN r.source_symbol=i.symbol THEN 0 ELSE 1 END remap_priority
        FROM source_rows r
        JOIN lifecycle_intervals i
          ON i.security_id=r.security_id
         AND cast(r.trade_date AS VARCHAR)
             BETWEEN i.effective_from AND i.effective_to
      ), factor_scales AS (
        SELECT remapped.security_id,remapped.source_symbol,
               remapped.canonical_symbol,
               median(
                 try_cast(official.adjust_factor AS DOUBLE)
                 /try_cast(remapped.adjust_factor AS DOUBLE)
               ) AS factor_scale
        FROM mapped_base remapped
        JOIN mapped_base official
          ON remapped.security_id=official.security_id
         AND remapped.canonical_symbol=official.canonical_symbol
         AND remapped.trade_date=official.trade_date
         AND official.source_symbol=official.canonical_symbol
        WHERE remapped.source_symbol<>remapped.canonical_symbol
          AND isfinite(try_cast(remapped.adjust_factor AS DOUBLE))
          AND try_cast(remapped.adjust_factor AS DOUBLE)>0
          AND isfinite(try_cast(official.adjust_factor AS DOUBLE))
          AND try_cast(official.adjust_factor AS DOUBLE)>0
        GROUP BY remapped.security_id,remapped.source_symbol,
                 remapped.canonical_symbol
      ), mapped AS (
        SELECT b.*,coalesce(s.factor_scale,1.0) AS factor_scale
        FROM mapped_base b
        LEFT JOIN factor_scales s
          ON b.security_id=s.security_id
         AND b.source_symbol=s.source_symbol
         AND b.canonical_symbol=s.canonical_symbol
      )
    """


def _canonical_lifecycle_query(*, context: Any, scans: str) -> str:
    projection, partition = _normalized_lifecycle_projection(context=context)
    mapped_ctes = _lifecycle_mapping_ctes(context.domain)
    return f"""
      WITH source_rows AS (
        SELECT d.*,upper(cast(d.symbol AS VARCHAR)) source_symbol,
               s.security_id
        FROM {scans} d
        JOIN lifecycle_symbol_map s
          ON upper(cast(d.symbol AS VARCHAR))=s.symbol
      ), {mapped_ctes}, normalized AS (
        SELECT {projection},m.remap_priority,m.source_symbol
        FROM mapped m
      )
      SELECT * EXCLUDE(remap_priority,source_symbol)
      FROM normalized
      QUALIFY row_number() OVER(
        PARTITION BY {partition}
        ORDER BY remap_priority,source_symbol
      )=1
      ORDER BY {partition}
    """


def _write_canonical_lifecycle_shard(
    con: Any,
    *,
    context: Any,
    affected: Sequence[Path],
    prepared_dir: Path,
) -> tuple[Path, int, int]:
    scans = _parquet_scan(affected)
    missing_mapping = _missing_lifecycle_mapping_count(con, scans)
    if missing_mapping:
        raise PitHistoryError(f"pit_history_symbol_lifecycle_mapping_missing:{context.domain}:{missing_mapping}")
    canonical = prepared_dir / "canonicalized.parquet"
    canonical_rows = _copy_lifecycle_query(
        con,
        _canonical_lifecycle_query(context=context, scans=scans),
        canonical,
    )
    source_rows = int(
        con.execute(
            f"SELECT count(*) FROM {scans} d JOIN lifecycle_symbol_map s ON upper(cast(d.symbol AS VARCHAR))=s.symbol",
        ).fetchone()[0]
        or 0
    )
    return canonical, canonical_rows, source_rows


def _prepare_lifecycle_domain_mutation(
    ctx: PitHistoryContext,
    *,
    domain: str,
    intervals: pd.DataFrame,
    symbol_map: pd.DataFrame,
) -> tuple[list[tuple[Path, Path]], list[Path], list[Path], dict[str, Any]]:
    context = resolve_active_domain(domain, workspace_root=ctx.workspace)
    prepared_dir = ctx.runtime / "lifecycle_effectivity" / "prepared" / domain
    if prepared_dir.exists():
        shutil.rmtree(prepared_dir.resolve(strict=True))
    prepared_dir.mkdir(parents=True, exist_ok=True)
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "lifecycle_effectivity" / "spill" / domain,
        threads=4,
    ) as con:
        con.register("lifecycle_intervals", intervals)
        con.register("lifecycle_symbol_map", symbol_map)
        plan = _prepare_retained_lifecycle_shards(
            con,
            context=context,
            prepared_dir=prepared_dir,
        )
        if not plan.affected:
            return (
                [],
                [],
                [],
                {
                    "status": "unchanged",
                    "affected_shard_count": 0,
                    "source_row_count": 0,
                },
            )
        _validate_lifecycle_source_schemas(context=context, paths=plan.affected)
        canonical, canonical_rows, source_rows = _write_canonical_lifecycle_shard(
            con,
            context=context,
            affected=plan.affected,
            prepared_dir=prepared_dir,
        )
    shutil.rmtree(
        ctx.runtime / "lifecycle_effectivity" / "spill" / domain,
        ignore_errors=True,
    )
    _validate_lifecycle_prepared_schemas(
        context=context,
        paths=[item[1] for item in plan.replacements] + [canonical],
    )
    return (
        plan.replacements,
        plan.removals,
        [canonical],
        {
            "status": "prepared",
            "affected_shard_count": len(plan.affected),
            "replacement_count": len(plan.replacements),
            "removal_count": len(plan.removals),
            "source_row_count": source_rows,
            "canonical_row_count": canonical_rows,
            "deduplicated_row_count": source_rows - canonical_rows,
            "retained_row_count": plan.kept_rows,
            "manifest_row_count_before": int(context.manifest.row_count),
            "projected_manifest_row_count": int(context.manifest.row_count - source_rows + canonical_rows),
        },
    )


def _cleanup_lifecycle_prepared_domain(
    ctx: PitHistoryContext,
    domain: str,
) -> None:
    prepared_dir = ctx.runtime / "lifecycle_effectivity" / "prepared" / domain
    if prepared_dir.exists():
        shutil.rmtree(prepared_dir.resolve(strict=True))
