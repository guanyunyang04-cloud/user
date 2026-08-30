"""PIT history lifecycle_factor operations."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.data.qdp_v2.active import resolve_active_domain
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb
from quantlab.data.qdp_v2.manifest import (
    utc_now,
)
from quantlab.data.qdp_v2.repair.common import _sql_literal
from quantlab.data.qdp_v2.repair.mutation import (
    mutate_active_shards_from_parquet,
    update_active_manifest_metadata,
)

from .context import (
    PitHistoryContext,
    PitHistoryError,
)
from .lifecycle_audit import (
    _parquet_scan,
    _quoted_identifier,
)
from .lifecycle_prepare import (
    _copy_lifecycle_query,
    _validate_lifecycle_prepared_schemas,
)


def _factor_transition_rows(intervals: pd.DataFrame) -> pd.DataFrame:
    ordered = intervals.sort_values(["security_id", "effective_from", "effective_to"]).copy()
    ordered["next_symbol"] = ordered.groupby("security_id")["symbol"].shift(-1)
    ordered["next_effective_from"] = ordered.groupby("security_id")["effective_from"].shift(-1)
    return ordered.loc[ordered["next_symbol"].notna()].copy()


def _factor_boundary_row(con: Any, paths: list[str], transition: Any) -> tuple[Any, ...] | None:
    return con.execute(
        """
          WITH factors AS (
            SELECT upper(cast(symbol AS VARCHAR)) AS symbol,
                   cast(trade_date AS VARCHAR) AS trade_date,
                   try_cast(adjust_factor AS DOUBLE) AS factor,
                   cast(source AS VARCHAR) AS source
            FROM read_parquet(?, union_by_name=true)
          ), target AS (
            SELECT * FROM factors
            WHERE symbol=? AND trade_date BETWEEN ? AND ?
              AND source LIKE '%pit_history_restore%'
              AND source NOT LIKE '%lifecycle_factor_scale_%'
          ), bounds AS (
            SELECT min(trade_date) AS first_date,
                   max(trade_date) AS last_date,
                   arg_min(factor,trade_date) AS first_factor,
                   arg_max(factor,trade_date) AS last_factor,
                   count(*) AS target_rows
            FROM target
          )
          SELECT b.*,
                 (SELECT arg_max(factor,trade_date)
                  FROM factors,bounds
                  WHERE symbol=? AND trade_date<bounds.first_date
                    AND factor>0 AND isfinite(factor)) AS prior_factor,
                 (SELECT arg_min(factor,trade_date)
                  FROM factors
                  WHERE symbol=? AND trade_date>=?
                    AND factor>0 AND isfinite(factor)) AS next_factor
          FROM bounds b
        """,
        [
            paths,
            str(transition.symbol),
            str(transition.effective_from),
            str(transition.effective_to),
            str(transition.symbol),
            str(transition.next_symbol),
            str(transition.next_effective_from),
        ],
    ).fetchone()


def _factor_scale_plan(transition: Any, result: tuple[Any, ...] | None) -> dict[str, Any] | None:
    if result is None or int(result[4] or 0) == 0:
        return None
    first_date, last_date, first_factor, last_factor, target_rows, prior_factor, next_factor = result
    values = [first_factor, last_factor, prior_factor, next_factor]
    if any(value is None or not np.isfinite(float(value)) for value in values):
        raise PitHistoryError(
            f"pit_history_lifecycle_factor_scale_boundary_missing:{transition.symbol}:{first_date}:{last_date}"
        )
    if any(float(value) <= 0 for value in values):
        raise PitHistoryError(
            f"pit_history_lifecycle_factor_scale_boundary_nonpositive:{transition.symbol}:{first_date}:{last_date}"
        )
    left_scale = float(prior_factor) / float(first_factor)
    right_scale = float(next_factor) / float(last_factor)
    relative_difference = abs(left_scale / right_scale - 1.0)
    if relative_difference > 0.01:
        raise PitHistoryError(
            "pit_history_lifecycle_factor_scale_evidence_conflict:"
            f"{transition.symbol}:left={left_scale}:right={right_scale}"
        )
    if abs(left_scale - 1.0) <= 1e-12:
        return None
    return {
        "security_id": str(transition.security_id),
        "symbol": str(transition.symbol),
        "next_symbol": str(transition.next_symbol),
        "first_date": str(first_date),
        "last_date": str(last_date),
        "target_row_count": int(target_rows),
        "scale": left_scale,
        "right_boundary_scale": right_scale,
        "boundary_scale_relative_difference": relative_difference,
    }


def _lifecycle_factor_scale_stitch_plan(
    ctx: PitHistoryContext,
    *,
    intervals: pd.DataFrame,
) -> list[dict[str, Any]]:
    factor = resolve_active_domain("adjust_factor", workspace_root=ctx.workspace)
    transitions = _factor_transition_rows(intervals)
    if transitions.empty:
        return []
    paths = [str(item) for item in factor.shard_paths]
    plans: list[dict[str, Any]] = []
    spill = ctx.runtime / "lifecycle_effectivity" / "factor_scale_spill"
    try:
        with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
            for transition in transitions.itertuples(index=False):
                plan = _factor_scale_plan(transition, _factor_boundary_row(con, paths, transition))
                if plan is not None:
                    plans.append(plan)
    finally:
        shutil.rmtree(spill, ignore_errors=True)
    return plans


def _factor_scale_predicates(plans: list[dict[str, Any]]) -> list[str]:
    return [
        "("
        + " AND ".join(
            (
                f"upper(cast(d.symbol AS VARCHAR))={_sql_literal(item['symbol'])}",
                f"cast(d.trade_date AS VARCHAR) BETWEEN "
                f"{_sql_literal(item['first_date'])} AND {_sql_literal(item['last_date'])}",
                "cast(d.source AS VARCHAR) LIKE '%pit_history_restore%'",
                "cast(d.source AS VARCHAR) NOT LIKE '%lifecycle_factor_scale_%'",
            )
        )
        + ")"
        for item in plans
    ]


def _factor_scale_expressions(
    columns: list[str],
    *,
    predicates: list[str],
    plans: list[dict[str, Any]],
) -> list[str]:
    combined = " OR ".join(predicates)
    expressions: list[str] = []
    for column in columns:
        quoted = _quoted_identifier(column)
        if column in {"adjust_factor", "fore_adjust_factor", "back_adjust_factor"}:
            branches = " ".join(
                f"WHEN {predicate} THEN try_cast(d.{quoted} AS DOUBLE)*{repr(float(plan['scale']))}"
                for predicate, plan in zip(predicates, plans, strict=True)
            )
            expressions.append(f"CASE {branches} ELSE d.{quoted} END AS {quoted}")
        elif column == "source":
            expressions.append(
                f"CASE WHEN {combined} THEN concat(coalesce(cast(d.{quoted} AS VARCHAR),''),"
                f"'+lifecycle_factor_scale_stitch_v1') ELSE d.{quoted} END AS {quoted}"
            )
        else:
            expressions.append(f"d.{quoted}")
    return expressions


def _prepare_factor_scale_replacements(
    ctx: PitHistoryContext,
    *,
    context: Any,
    prepared_dir: Path,
    plans: list[dict[str, Any]],
) -> list[tuple[Path, Path]]:
    predicates = _factor_scale_predicates(plans)
    combined = " OR ".join(predicates)
    columns = [str(item.get("name", "")) for item in context.manifest.schema]
    columns = [item for item in columns if item]
    expressions = _factor_scale_expressions(columns, predicates=predicates, plans=plans)
    order = ",".join(_quoted_identifier(item) for item in context.manifest.primary_key)
    replacements: list[tuple[Path, Path]] = []
    spill = ctx.runtime / "lifecycle_effectivity" / "factor_scale_apply_spill"
    try:
        with open_guarded_duckdb(temp_directory=spill, threads=4) as con:
            for index, old_path in enumerate(context.shard_paths):
                count = int(
                    con.execute(
                        f"SELECT count(*) FROM {_parquet_scan([old_path])} d WHERE {combined}"
                    ).fetchone()[0]
                    or 0
                )
                if not count:
                    continue
                target = prepared_dir / f"factor_scale_{index:04d}.parquet"
                _copy_lifecycle_query(
                    con,
                    f"SELECT {','.join(expressions)} FROM {_parquet_scan([old_path])} d ORDER BY {order}",
                    target,
                )
                replacements.append((old_path, target))
    finally:
        shutil.rmtree(spill, ignore_errors=True)
    return replacements


def _normalize_lifecycle_factor_scale_stitches(
    ctx: PitHistoryContext,
    *,
    intervals: pd.DataFrame,
    apply: bool,
) -> dict[str, Any]:
    plans = _lifecycle_factor_scale_stitch_plan(ctx, intervals=intervals)
    if not plans:
        return {"status": "already_aligned", "plans": []}
    if not apply:
        return {"status": "planned", "plans": plans}
    context = resolve_active_domain("adjust_factor", workspace_root=ctx.workspace)
    prepared_dir = ctx.runtime / "lifecycle_effectivity" / "factor_scale_prepared"
    if prepared_dir.exists():
        shutil.rmtree(prepared_dir.resolve(strict=True))
    prepared_dir.mkdir(parents=True, exist_ok=True)
    replacements = _prepare_factor_scale_replacements(
        ctx,
        context=context,
        prepared_dir=prepared_dir,
        plans=plans,
    )
    if not replacements:
        raise PitHistoryError("pit_history_lifecycle_factor_scale_target_missing")
    _validate_lifecycle_prepared_schemas(
        context=context,
        paths=[item[1] for item in replacements],
    )
    mutation = mutate_active_shards_from_parquet(
        "adjust_factor",
        replacements=replacements,
        reason="align lifecycle-restored adjustment-factor absolute scales",
        workspace_root=ctx.workspace,
    )
    update_active_manifest_metadata(
        "adjust_factor",
        reason="record lifecycle adjustment-factor scale alignment",
        workspace_root=ctx.workspace,
        source_updates={
            "lifecycle_factor_scale_semantics": "constant_scale_stitched_across_provider_and_ticker_boundaries",
            "lifecycle_factor_scale_normalized_at": utc_now(),
            "lifecycle_factor_scale_plans": plans,
        },
        quality_updates={"lifecycle_factor_scale_continuous": True},
    )
    shutil.rmtree(prepared_dir.resolve(strict=True))
    after = _lifecycle_factor_scale_stitch_plan(ctx, intervals=intervals)
    if after:
        raise PitHistoryError(f"pit_history_lifecycle_factor_scale_incomplete:{len(after)}")
    return {
        "status": str(mutation.get("status", "")),
        "plans": plans,
    }
