from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.environment import runtime_environment
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    schema_hash,
    stable_hash,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)
from quant_data_platform.qdp_v2.runtime import resolve_runtime_profile
from quant_data_platform.qdp_v2.status import active_dataset_map


ACTIVE_SCOPE_NAME = "mainboard_hs_a_ex_current_st_name_delisted_v2"
MAINBOARD_SCOPE_SQL = (
    "regexp_matches(symbol, '^(600|601|603|605)[0-9]{3}[.]SH$') "
    "or regexp_matches(symbol, '^(000|001|002|003)[0-9]{3}[.]SZ$')"
)


def rebuild_scope_active(
    *,
    workspace_root: str | Path | None = None,
    runtime: str = "fast",
    duckdb_memory_limit: str = "",
    threads: int = 0,
    as_of_date: str = "",
    domains: str = "",
    max_shards: int = 0,
    workers: int = 1,
    resume: bool = False,
    trust_existing: bool = False,
    activate: bool = False,
) -> dict[str, Any]:
    root, active, datasets, memory_limit, thread_count = _runtime_context(
        workspace_root=workspace_root, runtime=runtime, duckdb_memory_limit=duckdb_memory_limit, threads=threads
    )
    active_as_of = str(as_of_date or active.get("active_as_of_date", "") or "").strip()
    if not active_as_of:
        raise ValueError("active_as_of_date_required")
    scope_symbols_path, scope_stats = _write_active_scope_symbols(root=root, datasets=datasets, active_as_of=active_as_of)
    selected_domains = _domain_list(domains, active)

    from quant_data_platform.qdp_v2.cleaning import _scope_dataset_by_symbols  # local import avoids a hard module cycle

    processed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for domain in selected_domains:
        dataset_id = str(datasets.get(domain, "") or "")
        if not dataset_id:
            skipped.append({"domain": domain, "reason": "domain_not_active"})
            continue
        manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
        if manifest_path is None:
            errors.append({"domain": domain, "dataset_id": dataset_id, "error": "manifest_not_found"})
            continue
        source = read_dataset_manifest(manifest_path)
        if not _manifest_has_column(source, "symbol"):
            skipped.append({"domain": domain, "dataset_id": dataset_id, "reason": "no_symbol_column"})
            continue
        result = _scope_dataset_by_symbols(
            root=root,
            source=source,
            scope_symbols_path=scope_symbols_path,
            scope_name=ACTIVE_SCOPE_NAME,
            active_as_of=active_as_of,
            max_shards=max_shards,
            memory_limit=memory_limit,
            duckdb_threads=thread_count,
            workers=workers,
            resume=resume,
            trust_existing=trust_existing,
            scope_stats=scope_stats,
        )
        if result.get("status") == "ok":
            processed.append(result)
        else:
            errors.append({"domain": domain, "dataset_id": dataset_id, "error": result})

    active_path = ""
    if activate and not errors:
        dataset_map = dict(active.get("datasets", {}) or datasets)
        for item in processed:
            dataset_map[str(item["domain"])] = str(item["target_dataset_id"])
        active_payload = dict(active)
        active_payload["datasets"] = dict(sorted(dataset_map.items()))
        scope = dict(active_payload.get("scope", {}) or {})
        scope.update(
            {
                "name": ACTIVE_SCOPE_NAME,
                "end_date": active_as_of,
                "symbol_count": int(scope_stats.get("symbol_count", 0) or 0),
                "exclude_current_name_contains": "退市",
                "universe": "Shanghai and Shenzhen A-share main board; excludes ChiNext, STAR Market, ST stocks, current ST stocks, and names containing 退市.",
            }
        )
        active_payload["scope"] = scope
        active_payload["updated_at"] = utc_now()
        active_path = str(write_active_manifest(root, active_payload).resolve())

    payload = {
        "status": "error" if errors else "ok",
        "active_as_of_date": active_as_of,
        "scope": ACTIVE_SCOPE_NAME,
        "scope_symbols_path": str(scope_symbols_path.resolve()),
        "scope_stats": scope_stats,
        "processed": processed,
        "skipped": skipped,
        "errors": errors[:50],
        "error_count": len(errors),
        "active_manifest": active_path,
        "runtime_environment": runtime_environment(),
    }
    atomic_write_json(root / "runs" / f"rebuild_scope_active_{_stamp()}.json", payload)
    return payload


def rebuild_industry_concept_filled(
    *,
    workspace_root: str | Path | None = None,
    runtime: str = "fast",
    duckdb_memory_limit: str = "",
    threads: int = 0,
    activate: bool = False,
) -> dict[str, Any]:
    root, active, datasets, memory_limit, thread_count = _runtime_context(
        workspace_root=workspace_root, runtime=runtime, duckdb_memory_limit=duckdb_memory_limit, threads=threads
    )
    source = _require_manifest(root, datasets, "industry_concept")
    target_dataset_id = "industry_concept__" + stable_hash(
        {"source": source.dataset_id, "contract": "qdp_v2_industry_v5"}
    )
    paths_sql = _path_list_sql(_shard_paths(root, source))
    target_dir, staging, target_path = _prepare_target(root, "industry_concept", target_dataset_id, "part_000000_industry_concept_filled.parquet")

    import duckdb  # type: ignore

    profile_overrides: list[dict[str, str]] = []
    profile_errors: list[dict[str, str]] = []
    con = duckdb.connect(":memory:")
    try:
        _configure(con, memory_limit=memory_limit, threads=thread_count)
        con.execute(
            f"""
            copy (
              with src as (
                select
                  symbol,
                  cast(trade_date as date) as trade_date,
                  case
                    when industry is null or trim(cast(industry as varchar)) = '' or industry = 'UNKNOWN' then null
                    else cast(industry as varchar)
                  end as known_industry,
                  coalesce(cast(industry as varchar), '') as original_industry,
                  coalesce(cast(source as varchar), '') as original_source
                from read_parquet({paths_sql}, union_by_name=true)
              ),
              filled as (
                select
                  *,
                  last_value(known_industry ignore nulls) over w_prior as prior_industry,
                  last_value(case when known_industry is not null then trade_date end ignore nulls) over w_prior as prior_industry_date,
                  first_value(known_industry ignore nulls) over w_next as next_industry,
                  first_value(case when known_industry is not null then trade_date end ignore nulls) over w_next as next_industry_date
                from src
                window
                  w_prior as (
                    partition by symbol
                    order by trade_date
                    rows between unbounded preceding and current row
                  ),
                  w_next as (
                    partition by symbol
                    order by trade_date
                    rows between current row and unbounded following
                  )
              )
              select
                symbol,
                strftime(trade_date, '%Y-%m-%d') as trade_date,
                coalesce(known_industry, prior_industry, next_industry, 'UNKNOWN') as industry,
                case
                  when known_industry is not null then original_source
                  when prior_industry is not null then 'qdp_v2_industry_prior_ffill'
                  when next_industry is not null then 'qdp_v2_industry_initial_bfill'
                  else 'qdp_v2_industry_unknown_unresolved'
                end as source,
                original_industry,
                original_source,
                case
                  when known_industry is not null then 'original'
                  when prior_industry is not null then 'prior_ffill'
                  when next_industry is not null then 'initial_bfill'
                  else 'unknown_unresolved'
                end as industry_fill_method,
                strftime(coalesce(
                  case when known_industry is not null then trade_date end,
                  prior_industry_date,
                  next_industry_date
                ), '%Y-%m-%d') as industry_source_date
              from filled
            ) to {_sql_literal(str(target_path))} (format parquet, compression zstd)
            """
        )
        unresolved_symbols = [
            str(row[0])
            for row in con.execute(
                f"""
                select distinct symbol
                from read_parquet({_sql_literal(str(target_path))}, union_by_name=true)
                where industry = 'UNKNOWN' or industry is null or trim(cast(industry as varchar)) = ''
                order by symbol
                """
            ).fetchall()
        ]
        profile_overrides, profile_errors = _fetch_akshare_industry_profiles(unresolved_symbols)
        if profile_overrides:
            import pandas as pd  # type: ignore

            override_path = target_path.with_name("part_000000_industry_concept_filled_profile.parquet")
            con.register("industry_profile_overrides", pd.DataFrame(profile_overrides))
            con.execute(
                f"""
                copy (
                  with src as (
                    select *
                    from read_parquet({_sql_literal(str(target_path))}, union_by_name=true)
                  ),
                  overrides as (
                    select
                      upper(cast(symbol as varchar)) as symbol,
                      cast(industry as varchar) as profile_industry,
                      coalesce(cast(source_date as varchar), '') as profile_source_date
                    from industry_profile_overrides
                    where industry is not null and trim(cast(industry as varchar)) <> ''
                  )
                  select
                    src.symbol,
                    src.trade_date,
                    case
                      when o.profile_industry is not null
                       and (src.industry = 'UNKNOWN' or src.industry is null or trim(cast(src.industry as varchar)) = '')
                      then o.profile_industry
                      else src.industry
                    end as industry,
                    case
                      when o.profile_industry is not null
                       and (src.industry = 'UNKNOWN' or src.industry is null or trim(cast(src.industry as varchar)) = '')
                      then 'akshare_stock_profile_cninfo'
                      else src.source
                    end as source,
                    src.original_industry,
                    src.original_source,
                    case
                      when o.profile_industry is not null
                       and (src.industry = 'UNKNOWN' or src.industry is null or trim(cast(src.industry as varchar)) = '')
                      then 'akshare_profile'
                      else src.industry_fill_method
                    end as industry_fill_method,
                    case
                      when o.profile_industry is not null
                       and (src.industry = 'UNKNOWN' or src.industry is null or trim(cast(src.industry as varchar)) = '')
                      then o.profile_source_date
                      else src.industry_source_date
                    end as industry_source_date
                  from src
                  left join overrides o on upper(cast(src.symbol as varchar)) = o.symbol
                ) to {_sql_literal(str(override_path))} (format parquet, compression zstd)
                """
            )
            target_path.unlink(missing_ok=True)
            override_path.replace(target_path)
        stats = _one(
            con,
            f"""
            select
              count(*) as row_count,
              count(distinct concat_ws(chr(31), cast(trade_date as varchar), symbol)) as distinct_keys,
              min(trade_date) as start_date,
              max(trade_date) as end_date,
              sum(case when industry = 'UNKNOWN' then 1 else 0 end) as unknown_industry_rows,
              sum(case when industry_fill_method = 'prior_ffill' then 1 else 0 end) as prior_ffill_rows,
              sum(case when industry_fill_method = 'initial_bfill' then 1 else 0 end) as initial_bfill_rows,
              sum(case when industry_fill_method = 'akshare_profile' then 1 else 0 end) as akshare_profile_rows,
              sum(case when industry_fill_method = 'unknown_unresolved' then 1 else 0 end) as unresolved_rows
            from read_parquet({_sql_literal(str(target_path))}, union_by_name=true)
            """,
        )
        schema = _duckdb_schema(con, target_path)
    finally:
        con.close()

    blockers = _key_blockers(stats)
    final_path = _commit_single_shard(staging, target_dir, target_path)
    manifest = _single_shard_manifest(
        root=root,
        domain="industry_concept",
        dataset_id=target_dataset_id,
        layer="raw",
        frequency="1d",
        contract_version="qdp_v2_industry_v5",
        primary_key=["trade_date", "symbol"],
        stats=stats,
        schema=schema,
        final_path=final_path,
        content_key="filled_industry",
        source={"provider": "qdp_v2", "created_by": "rebuild_industry_concept_filled", "created_at": utc_now(), "source_dataset_id": source.dataset_id},
        quality={
            "path_refs_exist": True,
            "primary_key_unique": not blockers,
            "industry_unknown_rows": int(stats.get("unknown_industry_rows", 0) or 0),
            "industry_prior_ffill_rows": int(stats.get("prior_ffill_rows", 0) or 0),
            "industry_initial_bfill_rows": int(stats.get("initial_bfill_rows", 0) or 0),
            "industry_akshare_profile_rows": int(stats.get("akshare_profile_rows", 0) or 0),
            "industry_akshare_profile_symbols": sorted({str(item.get("symbol", "")) for item in profile_overrides if item.get("symbol")}),
            "industry_profile_fetch_errors": profile_errors[:50],
        },
        notes=["UNKNOWN industry values are filled from the same symbol's nearest known industry, then unresolved current symbols use AkShare/CNInfo profile industry when available."],
    )
    manifest_path = write_dataset_manifest(root, manifest)
    active_path = _activate(root, active, "industry_concept", target_dataset_id) if activate and not blockers else ""
    return _payload(
        status="ok" if not blockers else "blocked",
        blockers=blockers,
        target_dataset_id=target_dataset_id,
        manifest_path=manifest_path,
        active_path=active_path,
        stats=stats,
        run_name="rebuild_industry_concept_filled",
        root=root,
    )


def rebuild_share_capital_daily(
    *,
    workspace_root: str | Path | None = None,
    runtime: str = "fast",
    duckdb_memory_limit: str = "",
    threads: int = 0,
    activate: bool = False,
) -> dict[str, Any]:
    root, active, datasets, memory_limit, thread_count = _runtime_context(
        workspace_root=workspace_root, runtime=runtime, duckdb_memory_limit=duckdb_memory_limit, threads=threads
    )
    source = _require_manifest(root, datasets, "share_capital")
    universe = _require_manifest(root, datasets, "universe_snapshot")
    target_dataset_id = "share_capital__" + stable_hash(
        {"source": source.dataset_id, "universe": universe.dataset_id, "contract": "qdp_v2_share_capital_daily_v1"}
    )
    source_sql = _path_list_sql(_shard_paths(root, source))
    universe_sql = _path_list_sql(_shard_paths(root, universe))
    target_dir, staging, target_path = _prepare_target(root, "share_capital", target_dataset_id, "part_000000_share_capital_daily.parquet")
    change_reason_expr = "change_reason" if _manifest_has_column(source, "change_reason") else "''"
    event_source_expr = "source" if _manifest_has_column(source, "source") else "'unknown_share_capital_source'"
    total_source_date_expr = "cast(total_share_source_date as date)" if _manifest_has_column(source, "total_share_source_date") else "cast(trade_date as date)"
    float_source_date_expr = "cast(float_share_source_date as date)" if _manifest_has_column(source, "float_share_source_date") else "cast(trade_date as date)"
    restricted_source_date_expr = "cast(restricted_share_source_date as date)" if _manifest_has_column(source, "restricted_share_source_date") else "cast(trade_date as date)"

    import duckdb  # type: ignore

    con = duckdb.connect(":memory:")
    try:
        _configure(con, memory_limit=memory_limit, threads=thread_count)
        con.execute(
            f"""
            copy (
              with events as (
                select
                  symbol,
                  cast(trade_date as date) as event_date,
                  max(cast(total_share as double)) as total_share,
                  max(cast(float_share as double)) as float_share,
                  max(cast(restricted_share as double)) as restricted_share,
                  max({total_source_date_expr}) as total_source_date,
                  max({float_source_date_expr}) as float_source_date,
                  max({restricted_source_date_expr}) as restricted_source_date,
                  string_agg(distinct coalesce(cast({change_reason_expr} as varchar), ''), '|') as change_reason,
                  string_agg(distinct coalesce(cast({event_source_expr} as varchar), ''), '|') as event_source
                from read_parquet({source_sql}, union_by_name=true)
                group by symbol, cast(trade_date as date)
              ),
              combined as (
                select
                  symbol,
                  event_date as trade_date,
                  total_share,
                  float_share,
                  restricted_share,
                  total_source_date,
                  float_source_date,
                  restricted_source_date,
                  change_reason,
                  event_source,
                  0 as row_type
                from events
                union all
                select
                  symbol,
                  cast(trade_date as date) as trade_date,
                  null::double as total_share,
                  null::double as float_share,
                  null::double as restricted_share,
                  null::date as total_source_date,
                  null::date as float_source_date,
                  null::date as restricted_source_date,
                  null::varchar as change_reason,
                  null::varchar as event_source,
                  1 as row_type
                from read_parquet({universe_sql}, union_by_name=true)
              ),
              filled as (
                select
                  *,
                  last_value(total_share ignore nulls) over w_prior as prior_total_share,
                  last_value(float_share ignore nulls) over w_prior as prior_float_share,
                  last_value(restricted_share ignore nulls) over w_prior as prior_restricted_share,
                  last_value(total_source_date ignore nulls) over w_prior as prior_total_source_date,
                  last_value(float_source_date ignore nulls) over w_prior as prior_float_source_date,
                  last_value(restricted_source_date ignore nulls) over w_prior as prior_restricted_source_date,
                  first_value(total_share ignore nulls) over w_next as next_total_share,
                  first_value(float_share ignore nulls) over w_next as next_float_share,
                  first_value(restricted_share ignore nulls) over w_next as next_restricted_share,
                  first_value(total_source_date ignore nulls) over w_next as next_total_source_date,
                  first_value(float_source_date ignore nulls) over w_next as next_float_source_date,
                  first_value(restricted_source_date ignore nulls) over w_next as next_restricted_source_date
                from combined
                window
                  w_prior as (
                    partition by symbol
                    order by trade_date, row_type
                    rows between unbounded preceding and current row
                  ),
                  w_next as (
                    partition by symbol
                    order by trade_date, row_type
                    rows between current row and unbounded following
                  )
              ),
              daily as (
                select
                  symbol,
                  trade_date,
                  coalesce(prior_total_share, next_total_share) as total_raw,
                  coalesce(prior_float_share, next_float_share) as float_raw,
                  coalesce(prior_restricted_share, next_restricted_share) as restricted_raw,
                  coalesce(prior_total_source_date, next_total_source_date) as total_source_date,
                  coalesce(prior_float_source_date, next_float_source_date) as float_source_date,
                  coalesce(prior_restricted_source_date, next_restricted_source_date) as restricted_source_date,
                  case
                    when prior_total_share is not null or prior_float_share is not null or prior_restricted_share is not null then 'prior_ffill'
                    when next_total_share is not null or next_float_share is not null or next_restricted_share is not null then 'initial_bfill'
                    else 'source_missing'
                  end as share_fill_method
                from filled
                where row_type = 1
              )
              select
                symbol,
                strftime(trade_date, '%Y-%m-%d') as trade_date,
                cast(total_raw as double) as total_share,
                cast(coalesce(float_raw, case when total_raw is not null and restricted_raw is not null then total_raw - restricted_raw end) as double) as float_share,
                cast(coalesce(restricted_raw, case when total_raw is not null and float_raw is not null then greatest(total_raw - float_raw, 0) end) as double) as restricted_share,
                strftime(total_source_date, '%Y-%m-%d') as total_share_source_date,
                strftime(float_source_date, '%Y-%m-%d') as float_share_source_date,
                strftime(restricted_source_date, '%Y-%m-%d') as restricted_share_source_date,
                share_fill_method,
                'qdp_v2_share_capital_daily_from_events' as source
              from daily
            ) to {_sql_literal(str(target_path))} (format parquet, compression zstd)
            """
        )
        stats = _one(
            con,
            f"""
            select
              count(*) as row_count,
              count(distinct concat_ws(chr(31), cast(trade_date as varchar), symbol)) as distinct_keys,
              min(trade_date) as start_date,
              max(trade_date) as end_date,
              sum(case when total_share is null then 1 else 0 end) as total_share_null_rows,
              sum(case when float_share is null then 1 else 0 end) as float_share_null_rows,
              sum(case when restricted_share is null then 1 else 0 end) as restricted_share_null_rows,
              sum(case when share_fill_method = 'prior_ffill' then 1 else 0 end) as prior_ffill_rows,
              sum(case when share_fill_method = 'initial_bfill' then 1 else 0 end) as initial_bfill_rows,
              sum(case when share_fill_method = 'source_missing' then 1 else 0 end) as source_missing_rows
            from read_parquet({_sql_literal(str(target_path))}, union_by_name=true)
            """,
        )
        schema = _duckdb_schema(con, target_path)
    finally:
        con.close()

    blockers = _key_blockers(stats)
    final_path = _commit_single_shard(staging, target_dir, target_path)
    manifest = _single_shard_manifest(
        root=root,
        domain="share_capital",
        dataset_id=target_dataset_id,
        layer="raw",
        frequency="1d",
        contract_version="qdp_v2_share_capital_daily_v1",
        primary_key=["trade_date", "symbol"],
        stats=stats,
        schema=schema,
        final_path=final_path,
        content_key="daily_pit_share_capital",
        source={"provider": "qdp_v2", "created_by": "rebuild_share_capital_daily", "created_at": utc_now(), "source_dataset_id": source.dataset_id, "universe_dataset_id": universe.dataset_id},
        quality={
            "path_refs_exist": True,
            "primary_key_unique": not blockers,
            "aligns_to_universe_snapshot": True,
            "total_share_null_rows": int(stats.get("total_share_null_rows", 0) or 0),
            "float_share_null_rows": int(stats.get("float_share_null_rows", 0) or 0),
            "restricted_share_null_rows": int(stats.get("restricted_share_null_rows", 0) or 0),
            "initial_bfill_rows": int(stats.get("initial_bfill_rows", 0) or 0),
            "source_missing_rows": int(stats.get("source_missing_rows", 0) or 0),
        },
        notes=["Daily PIT-style share capital table expanded from share-change events."],
    )
    manifest_path = write_dataset_manifest(root, manifest)
    active_path = _activate(root, active, "share_capital", target_dataset_id) if activate and not blockers else ""
    return _payload(
        status="ok" if not blockers else "blocked",
        blockers=blockers,
        target_dataset_id=target_dataset_id,
        manifest_path=manifest_path,
        active_path=active_path,
        stats=stats,
        run_name="rebuild_share_capital_daily",
        root=root,
    )


def rebuild_valuation_market_cap(
    *,
    workspace_root: str | Path | None = None,
    runtime: str = "fast",
    duckdb_memory_limit: str = "",
    threads: int = 0,
    activate: bool = False,
) -> dict[str, Any]:
    root, active, datasets, memory_limit, thread_count = _runtime_context(
        workspace_root=workspace_root, runtime=runtime, duckdb_memory_limit=duckdb_memory_limit, threads=threads
    )
    source = _require_manifest(root, datasets, "valuation")
    daily = _require_manifest(root, datasets, "market_daily_raw")
    share = _require_manifest(root, datasets, "share_capital")
    target_dataset_id = "valuation__" + stable_hash(
        {
            "source": source.dataset_id,
            "daily": daily.dataset_id,
            "share": share.dataset_id,
            "contract": "qdp_v2_valuation_v2_market_cap",
        }
    )
    source_sql = _path_list_sql(_shard_paths(root, source))
    daily_sql = _path_list_sql(_shard_paths(root, daily))
    share_sql = _path_list_sql(_shard_paths(root, share))
    target_dir, staging, target_path = _prepare_target(root, "valuation", target_dataset_id, "part_000000_valuation_market_cap.parquet")

    import duckdb  # type: ignore

    con = duckdb.connect(":memory:")
    try:
        _configure(con, memory_limit=memory_limit, threads=thread_count)
        con.execute(
            f"""
            copy (
              with base as (
                select
                  symbol,
                  cast(trade_date as date) as trade_date,
                  pe,
                  pb,
                  turnover_rate,
                  source as valuation_source
                from read_parquet({source_sql}, union_by_name=true)
              ),
              bars as (
                select symbol, cast(trade_date as date) as trade_date, close
                from read_parquet({daily_sql}, union_by_name=true)
              ),
              combined as (
                select symbol, trade_date, close, trade_date as close_source_date, 0 as row_type from bars
                union all
                select symbol, trade_date, null::double as close, null::date as close_source_date, 1 as row_type from base
              ),
              close_filled as (
                select
                  symbol,
                  trade_date,
                  row_type,
                  last_value(close ignore nulls) over w_prior as prior_close,
                  last_value(close_source_date ignore nulls) over w_prior as prior_close_source_date,
                  first_value(close ignore nulls) over w_next as next_close,
                  first_value(close_source_date ignore nulls) over w_next as next_close_source_date
                from combined
                window
                  w_prior as (
                    partition by symbol
                    order by trade_date, row_type
                    rows between unbounded preceding and current row
                  ),
                  w_next as (
                    partition by symbol
                    order by trade_date, row_type
                    rows between current row and unbounded following
                  )
              ),
              close_daily as (
                select
                  symbol,
                  trade_date,
                  coalesce(prior_close, next_close) as pit_close,
                  coalesce(prior_close_source_date, next_close_source_date) as close_source_date
                from close_filled
                where row_type = 1
              ),
              share as (
                select symbol, cast(trade_date as date) as trade_date, total_share, float_share
                from read_parquet({share_sql}, union_by_name=true)
              )
              select
                b.symbol,
                strftime(b.trade_date, '%Y-%m-%d') as trade_date,
                cast(c.pit_close * s.total_share as double) as total_mv,
                cast(c.pit_close * s.float_share as double) as circ_mv,
                cast(b.pe as double) as pe,
                cast(b.pb as double) as pb,
                cast(b.turnover_rate as double) as turnover_rate,
                case
                  when c.pit_close is not null and s.total_share is not null then 'qdp_v2_close_x_share_capital'
                  else coalesce(cast(b.valuation_source as varchar), 'qdp_v2_valuation_source')
                end as source
              from base b
              left join close_daily c using (trade_date, symbol)
              left join share s using (trade_date, symbol)
            ) to {_sql_literal(str(target_path))} (format parquet, compression zstd)
            """
        )
        stats = _one(
            con,
            f"""
            select
              count(*) as row_count,
              count(distinct concat_ws(chr(31), cast(trade_date as varchar), symbol)) as distinct_keys,
              min(trade_date) as start_date,
              max(trade_date) as end_date,
              sum(case when total_mv is null then 1 else 0 end) as total_mv_null_rows,
              sum(case when circ_mv is null then 1 else 0 end) as circ_mv_null_rows,
              sum(case when pe is null then 1 else 0 end) as pe_null_rows,
              sum(case when pb is null then 1 else 0 end) as pb_null_rows,
              sum(case when turnover_rate is null then 1 else 0 end) as turnover_rate_null_rows
            from read_parquet({_sql_literal(str(target_path))}, union_by_name=true)
            """,
        )
        schema = _duckdb_schema(con, target_path)
    finally:
        con.close()

    blockers = _key_blockers(stats)
    final_path = _commit_single_shard(staging, target_dir, target_path)
    manifest = _single_shard_manifest(
        root=root,
        domain="valuation",
        dataset_id=target_dataset_id,
        layer="raw",
        frequency="1d",
        contract_version="qdp_v2_valuation_v2",
        primary_key=["trade_date", "symbol"],
        stats=stats,
        schema=schema,
        final_path=final_path,
        content_key="valuation_with_derived_market_cap",
        source={
            "provider": "qdp_v2",
            "created_by": "rebuild_valuation_market_cap",
            "created_at": utc_now(),
            "source_dataset_id": source.dataset_id,
            "daily_dataset_id": daily.dataset_id,
            "share_capital_dataset_id": share.dataset_id,
        },
        quality={
            "path_refs_exist": True,
            "primary_key_unique": not blockers,
            "schema_contract": "qdp_v2_valuation_v2",
            "market_cap_method": "total_mv=pit_close*total_share; circ_mv=pit_close*float_share",
            "total_mv_null_rows": int(stats.get("total_mv_null_rows", 0) or 0),
            "circ_mv_null_rows": int(stats.get("circ_mv_null_rows", 0) or 0),
        },
        notes=["Market-cap fields are derived from active daily close and active daily share-capital facts."],
    )
    manifest_path = write_dataset_manifest(root, manifest)
    active_path = _activate(root, active, "valuation", target_dataset_id) if activate and not blockers else ""
    return _payload(
        status="ok" if not blockers else "blocked",
        blockers=blockers,
        target_dataset_id=target_dataset_id,
        manifest_path=manifest_path,
        active_path=active_path,
        stats=stats,
        run_name="rebuild_valuation_market_cap",
        root=root,
    )


def rebuild_index_constituents_daily(
    *,
    workspace_root: str | Path | None = None,
    runtime: str = "fast",
    duckdb_memory_limit: str = "",
    threads: int = 0,
    activate: bool = False,
) -> dict[str, Any]:
    root, active, datasets, memory_limit, thread_count = _runtime_context(
        workspace_root=workspace_root, runtime=runtime, duckdb_memory_limit=duckdb_memory_limit, threads=threads
    )
    source = _require_manifest(root, datasets, "index_constituents")
    calendar = _require_manifest(root, datasets, "trading_calendar")
    target_dataset_id = "index_constituents__" + stable_hash(
        {"source": source.dataset_id, "calendar": calendar.dataset_id, "contract": "qdp_v2_index_constituents_daily_v1"}
    )
    source_sql = _path_list_sql(_shard_paths(root, source))
    calendar_sql = _path_list_sql(_shard_paths(root, calendar))
    active_as_of = str(active.get("active_as_of_date", "") or source.end_date)
    target_dir, staging, target_path = _prepare_target(root, "index_constituents", target_dataset_id, "part_000000_index_constituents_daily.parquet")

    import duckdb  # type: ignore

    con = duckdb.connect(":memory:")
    try:
        _configure(con, memory_limit=memory_limit, threads=thread_count)
        con.execute(
            f"""
            copy (
              with src as (
                select distinct
                  index_symbol,
                  symbol,
                  cast(trade_date as date) as snapshot_date,
                  coalesce(cast(index_name as varchar), '') as index_name,
                  coalesce(cast(source as varchar), '') as source
                from read_parquet({source_sql}, union_by_name=true)
              ),
              snapshot_dates as (
                select distinct index_symbol, snapshot_date
                from src
              ),
              intervals as (
                select
                  index_symbol,
                  snapshot_date,
                  lead(snapshot_date) over (partition by index_symbol order by snapshot_date) as next_snapshot_date
                from snapshot_dates
              ),
              cal as (
                select cast(trade_date as date) as trade_date
                from read_parquet({calendar_sql}, union_by_name=true)
                where is_open = true and cast(trade_date as date) <= date '{active_as_of}'
              )
              select
                s.index_symbol,
                s.symbol,
                strftime(c.trade_date, '%Y-%m-%d') as trade_date,
                s.index_name,
                s.source,
                strftime(s.snapshot_date, '%Y-%m-%d') as source_snapshot_date
              from src s
              inner join intervals i
                on s.index_symbol = i.index_symbol and s.snapshot_date = i.snapshot_date
              inner join cal c
                on c.trade_date >= s.snapshot_date
               and (i.next_snapshot_date is null or c.trade_date < i.next_snapshot_date)
            ) to {_sql_literal(str(target_path))} (format parquet, compression zstd)
            """
        )
        stats = _one(
            con,
            f"""
            select
              count(*) as row_count,
              count(distinct concat_ws(chr(31), cast(trade_date as varchar), index_symbol, symbol)) as distinct_keys,
              min(trade_date) as start_date,
              max(trade_date) as end_date,
              count(distinct trade_date) as daily_date_count,
              count(distinct source_snapshot_date) as source_snapshot_date_count,
              count(distinct index_symbol) as index_count
            from read_parquet({_sql_literal(str(target_path))}, union_by_name=true)
            """,
        )
        schema = _duckdb_schema(con, target_path)
    finally:
        con.close()

    blockers = _key_blockers(stats)
    final_path = _commit_single_shard(staging, target_dir, target_path)
    manifest = _single_shard_manifest(
        root=root,
        domain="index_constituents",
        dataset_id=target_dataset_id,
        layer="raw",
        frequency="1d",
        contract_version="qdp_v2_index_constituents_daily_v1",
        primary_key=["trade_date", "index_symbol", "symbol"],
        stats=stats,
        schema=schema,
        final_path=final_path,
        content_key="daily_pit_index_constituents",
        source={"provider": "qdp_v2", "created_by": "rebuild_index_constituents_daily", "created_at": utc_now(), "source_dataset_id": source.dataset_id, "calendar_dataset_id": calendar.dataset_id},
        quality={
            "path_refs_exist": True,
            "primary_key_unique": not blockers,
            "daily_pit_expanded": True,
            "source_snapshot_date_count": int(stats.get("source_snapshot_date_count", 0) or 0),
            "index_count": int(stats.get("index_count", 0) or 0),
        },
        notes=["Daily PIT-style index membership expanded from source snapshot dates."],
    )
    manifest_path = write_dataset_manifest(root, manifest)
    active_path = _activate(root, active, "index_constituents", target_dataset_id) if activate and not blockers else ""
    return _payload(
        status="ok" if not blockers else "blocked",
        blockers=blockers,
        target_dataset_id=target_dataset_id,
        manifest_path=manifest_path,
        active_path=active_path,
        stats=stats,
        run_name="rebuild_index_constituents_daily",
        root=root,
    )


def _fetch_akshare_industry_profiles(symbols: list[str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not symbols:
        return [], []
    try:
        import akshare as ak  # type: ignore
    except Exception as exc:
        return [], [{"symbol": "*", "error": f"akshare_import_failed:{exc}"}]

    overrides: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for symbol in symbols:
        code = str(symbol).split(".")[0]
        try:
            df = ak.stock_profile_cninfo(symbol=code)
            if df is None or df.empty or "所属行业" not in df.columns:
                errors.append({"symbol": symbol, "error": "profile_industry_missing"})
                continue
            industry = str(df.iloc[0].get("所属行业") or "").strip()
            if not industry:
                errors.append({"symbol": symbol, "error": "profile_industry_blank"})
                continue
            source_date = str(df.iloc[0].get("上市日期") or "").strip()
            overrides.append({"symbol": symbol, "industry": industry, "source_date": source_date})
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)})
    return overrides, errors


def _runtime_context(
    *,
    workspace_root: str | Path | None,
    runtime: str,
    duckdb_memory_limit: str,
    threads: int,
) -> tuple[Path, dict[str, Any], dict[str, str], str, int]:
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    datasets = active_dataset_map(active)
    profile = resolve_runtime_profile(runtime)
    memory_limit = str(duckdb_memory_limit or "").strip() or profile.duckdb_memory_limit
    thread_count = int(threads or profile.duckdb_threads or 1)
    return root, active, datasets, memory_limit, thread_count


def _require_manifest(root: Path, datasets: Mapping[str, str], domain: str) -> DatasetManifest:
    dataset_id = str(datasets.get(domain, "") or "")
    manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
    if manifest_path is None:
        raise FileNotFoundError(f"active dataset manifest not found: {domain}:{dataset_id}")
    return read_dataset_manifest(manifest_path)


def _domain_list(domains: str, active: Mapping[str, Any]) -> list[str]:
    text = str(domains or "").strip()
    if text:
        return [item.strip() for item in text.split(",") if item.strip()]
    dataset_map = dict(active.get("datasets", {}) or active_dataset_map(dict(active)))
    return [domain for domain in dataset_map if domain != "trading_calendar"]


def _manifest_has_column(manifest: DatasetManifest, column: str) -> bool:
    wanted = str(column).lower()
    return any(str(item.get("name", "")).lower() == wanted for item in manifest.schema)


def _write_active_scope_symbols(*, root: Path, datasets: Mapping[str, str], active_as_of: str) -> tuple[Path, dict[str, Any]]:
    universe = _require_manifest(root, datasets, "universe_snapshot")
    status = _require_manifest(root, datasets, "security_status")
    universe_sql = _path_list_sql(_shard_paths(root, universe))
    status_sql = _path_list_sql(_shard_paths(root, status))
    target = root / "runs" / f"active_scope_symbols_{stable_hash({'as_of': active_as_of, 'universe': universe.dataset_id, 'status': status.dataset_id, 'scope': ACTIVE_SCOPE_NAME})}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)

    import duckdb  # type: ignore

    con = duckdb.connect(":memory:")
    try:
        _configure(con, memory_limit="2GB", threads=2)
        base_cte = f"""
          with universe as (
            select
              upper(cast(symbol as varchar)) as symbol,
              max(coalesce(cast(name as varchar), '')) as name
            from read_parquet({universe_sql}, union_by_name=true)
            where cast(trade_date as varchar)[:10] = {_sql_literal(active_as_of)}
            group by 1
          ),
          status as (
            select
              upper(cast(symbol as varchar)) as symbol,
              bool_or(coalesce(cast(is_st as boolean), false)) as is_st,
              bool_or(coalesce(cast(is_delisted as boolean), false)) as is_delisted
            from read_parquet({status_sql}, union_by_name=true)
            where cast(trade_date as varchar)[:10] = {_sql_literal(active_as_of)}
            group by 1
          ),
          joined as (
            select
              u.symbol,
              u.name,
              ({MAINBOARD_SCOPE_SQL}) as is_mainboard,
              coalesce(s.is_st, false) as is_st,
              coalesce(s.is_delisted, false) as is_delisted,
              (u.name like '%退市%') as name_contains_delisted
            from universe u
            left join status s using(symbol)
          )
        """
        con.execute(
            f"""
            copy (
              {base_cte}
              select symbol
              from joined
              where is_mainboard
                and not is_st
                and not is_delisted
                and not name_contains_delisted
              order by symbol
            ) to {_sql_literal(str(target))} (format parquet, compression zstd)
            """
        )
        stats = _one(
            con,
            f"""
            {base_cte}
            select
              count(*) as active_date_universe_symbols,
              sum(case when is_mainboard then 1 else 0 end) as mainboard_candidate_symbols,
              sum(case when is_mainboard and is_st then 1 else 0 end) as st_symbols,
              sum(case when is_mainboard and is_delisted then 1 else 0 end) as status_delisted_symbols,
              sum(case when is_mainboard and name_contains_delisted then 1 else 0 end) as name_delisted_symbols,
              (select count(*) from read_parquet({_sql_literal(str(target))})) as symbol_count,
              (select sum(case when symbol like '%.SH' then 1 else 0 end) from read_parquet({_sql_literal(str(target))})) as sh_count,
              (select sum(case when symbol like '%.SZ' then 1 else 0 end) from read_parquet({_sql_literal(str(target))})) as sz_count
            from joined
            """
        )
        excluded = con.execute(
            f"""
            {base_cte}
            select symbol, name
            from joined
            where is_mainboard and name_contains_delisted
            order by symbol
            """
        ).fetchall()
    finally:
        con.close()
    stats["name_delisted_examples"] = [{"symbol": str(row[0]), "name": str(row[1])} for row in excluded[:50]]
    stats["scope_name"] = ACTIVE_SCOPE_NAME
    return target, stats


def _shard_paths(root: Path, manifest: DatasetManifest) -> list[Path]:
    paths: list[Path] = []
    for shard in manifest.shards:
        path = Path(shard.path)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            paths.append(path.resolve())
    if not paths:
        raise FileNotFoundError(f"no readable shards for {manifest.domain}:{manifest.dataset_id}")
    return paths


def _prepare_target(root: Path, domain: str, dataset_id: str, filename: str) -> tuple[Path, Path, Path]:
    target_dir = root / "datasets" / domain / dataset_id
    staging = target_dir / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging_shards = staging / "shards"
    staging_shards.mkdir(parents=True, exist_ok=True)
    return target_dir, staging, staging_shards / filename


def _commit_single_shard(staging: Path, target_dir: Path, target_path: Path) -> Path:
    final_shards = target_dir / "shards"
    target_dir.mkdir(parents=True, exist_ok=True)
    if final_shards.exists():
        shutil.rmtree(final_shards)
    (staging / "shards").replace(final_shards)
    shutil.rmtree(staging, ignore_errors=True)
    return final_shards / target_path.name


def _single_shard_manifest(
    *,
    root: Path,
    domain: str,
    dataset_id: str,
    layer: str,
    frequency: str,
    contract_version: str,
    primary_key: list[str],
    stats: Mapping[str, Any],
    schema: list[dict[str, str]],
    final_path: Path,
    content_key: str,
    source: dict[str, Any],
    quality: dict[str, Any],
    notes: list[str],
) -> DatasetManifest:
    entry = ShardManifestEntry(
        path=f"datasets/{domain}/{dataset_id}/shards/{final_path.name}",
        row_count=int(stats.get("row_count", 0) or 0),
        start_date=str(stats.get("start_date", "") or ""),
        end_date=str(stats.get("end_date", "") or ""),
        status="stored",
        file_size=int(final_path.stat().st_size),
        schema_hash=schema_hash(schema),
        content_key=content_key,
    )
    return DatasetManifest(
        dataset_id=dataset_id,
        domain=domain,
        layer=layer,
        frequency=frequency,
        contract_version=contract_version,
        primary_key=primary_key,
        start_date=entry.start_date,
        end_date=entry.end_date,
        row_count=entry.row_count,
        schema_hash=schema_hash(schema),
        schema=schema,
        shards=[entry],
        source=source,
        quality=quality,
        notes=notes,
    )


def _activate(root: Path, active: Mapping[str, Any], domain: str, dataset_id: str) -> str:
    active_payload = dict(active)
    dataset_map = dict(active_payload.get("datasets", {}) or active_dataset_map(active_payload))
    dataset_map[domain] = dataset_id
    active_payload["datasets"] = dataset_map
    return str(write_active_manifest(root, active_payload).resolve())


def _payload(
    *,
    status: str,
    blockers: list[str],
    target_dataset_id: str,
    manifest_path: Path,
    active_path: str,
    stats: Mapping[str, Any],
    run_name: str,
    root: Path,
) -> dict[str, Any]:
    payload = {
        "status": status,
        "blockers": blockers,
        "target_dataset_id": target_dataset_id,
        "manifest_path": str(manifest_path.resolve()),
        "active_manifest": active_path,
        "stats": dict(stats),
        "runtime_environment": runtime_environment(),
    }
    atomic_write_json(root / "runs" / f"{run_name}_{_stamp()}.json", payload)
    return payload


def _configure(con: Any, *, memory_limit: str, threads: int) -> None:
    safe_memory = str(memory_limit or "8GB").replace("'", "")
    con.execute(f"set memory_limit='{safe_memory}'")
    con.execute(f"set threads={max(1, int(threads or 1))}")
    con.execute("set preserve_insertion_order=false")


def _duckdb_schema(con: Any, path: Path) -> list[dict[str, str]]:
    rows = con.execute(f"describe select * from read_parquet({_sql_literal(str(path))})").fetchall()
    return [{"name": str(row[0]), "type": str(row[1])} for row in rows]


def _one(con: Any, sql: str) -> dict[str, Any]:
    df = con.execute(sql).fetchdf()
    if df.empty:
        return {}
    return {str(key): _json_scalar(value) for key, value in df.iloc[0].to_dict().items()}


def _json_scalar(value: Any) -> Any:
    try:
        import pandas as pd  # type: ignore

        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _key_blockers(stats: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if int(stats.get("row_count", 0) or 0) != int(stats.get("distinct_keys", 0) or 0):
        blockers.append("primary_key_not_unique")
    return blockers


def _path_list_sql(paths: list[Path]) -> str:
    return "[" + ", ".join(_sql_literal(str(path)) for path in paths) + "]"


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


def _stamp() -> str:
    return utc_now().replace(":", "").replace("-", "")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp completion")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("industry-concept-filled", "share-capital-daily", "valuation-market-cap", "index-constituents-daily"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--workspace-root", default="")
        cmd.add_argument("--runtime", default="fast", choices=("safe", "balanced", "fast"))
        cmd.add_argument("--duckdb-memory-limit", default="")
        cmd.add_argument("--threads", type=int, default=0)
        cmd.add_argument("--activate", action="store_true")
        cmd.add_argument("--json", action="store_true")
    scope = sub.add_parser("scope-active")
    scope.add_argument("--workspace-root", default="")
    scope.add_argument("--runtime", default="fast", choices=("safe", "balanced", "fast"))
    scope.add_argument("--duckdb-memory-limit", default="")
    scope.add_argument("--threads", type=int, default=0)
    scope.add_argument("--as-of-date", default="")
    scope.add_argument("--domains", default="")
    scope.add_argument("--max-shards", type=int, default=0)
    scope.add_argument("--workers", type=int, default=1)
    scope.add_argument("--resume", action="store_true")
    scope.add_argument("--trust-existing", action="store_true")
    scope.add_argument("--activate", action="store_true")
    scope.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    common = {
        "workspace_root": str(args.workspace_root or "") or None,
        "runtime": str(args.runtime or "fast"),
        "duckdb_memory_limit": str(args.duckdb_memory_limit or ""),
        "threads": int(args.threads or 0),
        "activate": bool(args.activate),
    }
    if args.command == "industry-concept-filled":
        payload = rebuild_industry_concept_filled(**common)
    elif args.command == "share-capital-daily":
        payload = rebuild_share_capital_daily(**common)
    elif args.command == "valuation-market-cap":
        payload = rebuild_valuation_market_cap(**common)
    elif args.command == "index-constituents-daily":
        payload = rebuild_index_constituents_daily(**common)
    elif args.command == "scope-active":
        payload = rebuild_scope_active(
            workspace_root=str(args.workspace_root or "") or None,
            runtime=str(args.runtime or "fast"),
            duckdb_memory_limit=str(args.duckdb_memory_limit or ""),
            threads=int(args.threads or 0),
            as_of_date=str(args.as_of_date or ""),
            domains=str(args.domains or ""),
            max_shards=int(args.max_shards or 0),
            workers=int(args.workers or 1),
            resume=bool(args.resume),
            trust_existing=bool(args.trust_existing),
            activate=bool(args.activate),
        )
    else:  # pragma: no cover
        raise ValueError(f"unsupported_completion_command:{args.command}")
    if bool(args.json):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print("\n".join(f"{key}: {value}" for key, value in payload.items() if key in {"status", "target_dataset_id", "manifest_path", "active_manifest"}))
    return 0 if str(payload.get("status", "")) == "ok" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
