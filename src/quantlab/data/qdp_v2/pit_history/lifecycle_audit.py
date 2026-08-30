"""PIT history lifecycle_audit operations."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.qdp_v2.active import resolve_active_domain
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb
from quantlab.data.qdp_v2.manifest import (
    qdp_v2_root,
    read_active_manifest,
)
from quantlab.data.qdp_v2.repair.common import _sql_literal

from .config import (
    DEFAULT_START_DATE,
    LIFECYCLE_NORMALIZE_DOMAINS,
)
from .context import (
    PitHistoryContext,
    PitHistoryError,
    _context,
)


def _quoted_identifier(value: object) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _parquet_scan(paths: Sequence[str | Path]) -> str:
    values = ",".join(_sql_literal(Path(item).resolve()) for item in paths)
    return f"read_parquet([{values}], union_by_name=true)"


def _symbol_lifecycle_tables(
    ctx: PitHistoryContext,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    history = resolve_active_domain("symbol_history", workspace_root=ctx.workspace)
    identity = resolve_active_domain("security_identity", workspace_root=ctx.workspace)
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "lifecycle_inventory_spill",
        threads=2,
    ) as con:
        intervals = con.execute(
            "SELECT cast(security_id AS VARCHAR) security_id, "
            "upper(cast(symbol AS VARCHAR)) symbol, "
            "cast(effective_from AS VARCHAR) effective_from, "
            "cast(effective_to AS VARCHAR) effective_to, "
            "cast(name_on_date AS VARCHAR) name_on_date "
            "FROM read_parquet(?, union_by_name=true)",
            [[str(item) for item in history.shard_paths]],
        ).fetchdf()
        identities = con.execute(
            "SELECT cast(security_id AS VARCHAR) security_id, "
            "cast(list_date AS VARCHAR) list_date "
            "FROM read_parquet(?, union_by_name=true)",
            [[str(item) for item in identity.shard_paths]],
        ).fetchdf()
    shutil.rmtree(ctx.runtime / "lifecycle_inventory_spill", ignore_errors=True)
    intervals = intervals.drop_duplicates(["security_id", "symbol", "effective_from", "effective_to"]).copy()
    intervals["effective_from"] = intervals["effective_from"].astype(str).str[:10]
    intervals["effective_to"] = intervals["effective_to"].astype(str).str[:10]
    multi_ids = intervals.groupby("security_id")["symbol"].nunique().loc[lambda x: x > 1].index
    intervals = intervals.loc[intervals["security_id"].isin(multi_ids)].copy()
    if intervals.empty:
        return intervals, pd.DataFrame(columns=["security_id", "symbol"]), ""
    intervals = intervals.merge(
        identities.drop_duplicates("security_id", keep="last"),
        on="security_id",
        how="left",
        validate="many_to_one",
    )
    intervals["list_date"] = intervals["list_date"].fillna("").astype(str).str[:10]
    intervals["canonical_delist_date"] = ""
    finite = intervals["effective_to"].ne("9999-12-31")
    intervals.loc[finite, "canonical_delist_date"] = (
        pd.to_datetime(intervals.loc[finite, "effective_to"], errors="raise") + pd.Timedelta(days=1)
    ).dt.strftime("%Y-%m-%d")
    ordered = intervals.sort_values(["security_id", "effective_from", "effective_to"])
    prior_end = ordered.groupby("security_id")["effective_to"].shift(1)
    overlap = prior_end.notna() & ordered["effective_from"].le(prior_end.fillna(""))
    if overlap.any():
        examples = ordered.loc[overlap, ["security_id", "symbol", "effective_from"]]
        raise PitHistoryError(f"pit_history_symbol_lifecycle_interval_overlap:{examples.head(10).to_dict('records')}")
    symbol_map = intervals[["security_id", "symbol"]].drop_duplicates()
    conflicts = symbol_map.groupby("symbol")["security_id"].nunique()
    if (conflicts > 1).any():
        raise PitHistoryError(
            f"pit_history_symbol_lifecycle_symbol_identity_conflict:{conflicts.loc[conflicts > 1].index.tolist()[:10]}"
        )
    transitions = ordered.groupby("security_id").tail(-1)
    cutoff = str(transitions["effective_from"].max()) if not transitions.empty else ""
    return ordered.reset_index(drop=True), symbol_map.reset_index(drop=True), cutoff


def _lifecycle_domain_findings(
    con: Any,
    *,
    context: Any,
    sample_limit: int = 20,
) -> dict[str, Any]:
    if "symbol" not in context.manifest.primary_key:
        return {
            "status": "not_applicable",
            "domain": context.domain,
            "outside_effective_interval_rows": 0,
            "examples": [],
        }
    scan = _parquet_scan(context.shard_paths)
    base = f"""
      WITH source_rows AS (
        SELECT upper(cast(d.symbol AS VARCHAR)) source_symbol,
               cast(d.trade_date AS VARCHAR) trade_date,
               m.security_id
        FROM {scan} d
        JOIN lifecycle_symbol_map m
          ON upper(cast(d.symbol AS VARCHAR))=m.symbol
      ), resolved AS (
        SELECT s.*,i.symbol canonical_symbol
        FROM source_rows s
        LEFT JOIN lifecycle_intervals i
          ON i.security_id=s.security_id
         AND s.trade_date BETWEEN i.effective_from AND i.effective_to
      )
    """
    row = con.execute(
        base + "SELECT count(*) FILTER (WHERE canonical_symbol IS NULL), "
        "count(*) FILTER (WHERE canonical_symbol IS NOT NULL "
        "AND canonical_symbol<>source_symbol) FROM resolved"
    ).fetchone()
    unmapped = int(row[0] or 0)
    outside = int(row[1] or 0)
    examples: list[dict[str, Any]] = []
    if unmapped or outside:
        examples = (
            con.execute(
                base + "SELECT security_id,source_symbol,trade_date,canonical_symbol "
                "FROM resolved WHERE canonical_symbol IS NULL "
                "OR canonical_symbol<>source_symbol "
                "ORDER BY trade_date,source_symbol LIMIT ?",
                [int(sample_limit)],
            )
            .fetchdf()
            .to_dict("records")
        )
    return {
        "status": "ok" if not unmapped and not outside else "needs_repair",
        "domain": context.domain,
        "outside_effective_interval_rows": outside,
        "unmapped_lifecycle_rows": unmapped,
        "examples": examples,
    }


def audit_symbol_lifecycle_effectivity(
    *,
    workspace_root: str | Path | None = None,
    domains: Sequence[str] = LIFECYCLE_NORMALIZE_DOMAINS,
    sample_limit: int = 20,
) -> dict[str, Any]:
    ctx = _context(
        start_date=DEFAULT_START_DATE,
        end_date=str(read_active_manifest(qdp_v2_root(workspace_root)).get("active_as_of_date", DEFAULT_START_DATE)),
        workspace_root=workspace_root,
    )
    intervals, symbol_map, cutoff = _symbol_lifecycle_tables(ctx)
    if intervals.empty:
        return {
            "status": "ok",
            "multi_symbol_identity_count": 0,
            "transition_count": 0,
            "domains": {},
        }
    reports: dict[str, Any] = {}
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "lifecycle_audit_spill",
        threads=4,
    ) as con:
        con.register("lifecycle_intervals", intervals)
        con.register("lifecycle_symbol_map", symbol_map)
        for domain in domains:
            context = resolve_active_domain(domain, workspace_root=ctx.workspace)
            reports[domain] = _lifecycle_domain_findings(
                con,
                context=context,
                sample_limit=sample_limit,
            )
    shutil.rmtree(ctx.runtime / "lifecycle_audit_spill", ignore_errors=True)
    outside = sum(
        int(item.get("outside_effective_interval_rows", 0) or 0) + int(item.get("unmapped_lifecycle_rows", 0) or 0)
        for item in reports.values()
    )
    return {
        "status": "ok" if outside == 0 else "needs_repair",
        "multi_symbol_identity_count": int(intervals["security_id"].nunique()),
        "transition_count": int(len(intervals) - intervals["security_id"].nunique()),
        "effective_cutoff": cutoff,
        "outside_effective_interval_rows": int(outside),
        "domains": reports,
    }
