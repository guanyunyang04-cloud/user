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
    write_active_manifest,
    write_dataset_manifest,
    utc_now,
)
from quant_data_platform.qdp_v2.runtime import resolve_runtime_profile
from quant_data_platform.qdp_v2.status import active_dataset_map


META_DOMAINS = [
    "trading_calendar",
    "universe_snapshot",
    "security_status",
    "adjust_factor",
    "industry_concept",
    "index_constituents",
]

PRIMARY_KEYS: dict[str, list[str]] = {
    "trading_calendar": ["trade_date"],
    "universe_snapshot": ["trade_date", "symbol"],
    "security_status": ["trade_date", "symbol"],
    "adjust_factor": ["trade_date", "symbol"],
    "industry_concept": ["trade_date", "symbol"],
    "index_constituents": ["trade_date", "index_symbol", "symbol"],
}

MAINBOARD_PATTERN = (
    "regexp_matches(symbol, '^(600|601|603|605)[0-9]{3}[.]SH$') "
    "or regexp_matches(symbol, '^(000|001|002|003)[0-9]{3}[.]SZ$')"
)


def rebuild_adjust_factor_standard(
    *,
    workspace_root: str | Path | None = None,
    runtime: str = "fast",
    duckdb_memory_limit: str = "",
    threads: int = 0,
    activate: bool = False,
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    datasets = active_dataset_map(active)
    source = _require_manifest(root, datasets, "adjust_factor")
    daily = _require_manifest(root, datasets, "market_daily_raw")
    profile = resolve_runtime_profile(runtime)
    memory_limit = str(duckdb_memory_limit or "").strip() or profile.duckdb_memory_limit
    thread_count = int(threads or profile.duckdb_threads or 1)
    target_dataset_id = "adjust_factor__" + stable_hash(
        {
            "source": source.dataset_id,
            "daily": daily.dataset_id,
            "contract": "qdp_v2_adjust_factor_standard_v2",
        }
    )
    target_dir = root / "datasets" / "adjust_factor" / target_dataset_id
    staging = target_dir / ".staging"
    staging_shards = staging / "shards"
    if staging.exists():
        shutil.rmtree(staging)
    staging_shards.mkdir(parents=True, exist_ok=True)
    target_path = staging_shards / "part_000000_adjust_factor_standard.parquet"
    temp_dir = root / "audits" / ".duckdb_tmp" / f"adjust_standard_{_stamp()}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    import duckdb  # type: ignore

    con = duckdb.connect(":memory:")
    try:
        _configure(con, memory_limit=memory_limit, threads=thread_count, temp_dir=temp_dir)
        source_sql = _path_list_sql(_shard_paths(root, source))
        daily_sql = _path_list_sql(_shard_paths(root, daily))
        con.execute(
            f"""
            create temp table selected_symbol_source as
            with membership as (
              select
                filename,
                symbol,
                min(cast(trade_date as date)) as start_date,
                max(cast(trade_date as date)) as end_date,
                count(*) as row_count
              from read_parquet({source_sql}, union_by_name=true, filename=true)
              where factor_provider = 'tonghuashun'
                and back_adjust_factor is not null and back_adjust_factor > 0
              group by filename, symbol
            ),
            ranked as (
              select
                *,
                row_number() over (
                  partition by symbol
                  order by end_date desc, start_date asc, row_count desc, filename desc
                ) as rn
              from membership
            )
            select filename, symbol, start_date, end_date, row_count
            from ranked
            where rn = 1
            """
        )
        selected_source_quality = _one(
            con,
            """
            select
              count(*) as selected_symbol_sources,
              min(start_date) as min_selected_start_date,
              max(end_date) as max_selected_end_date,
              sum(row_count) as selected_source_rows
            from selected_symbol_source
            """,
        )
        con.execute(
            f"""
            create temp table factor_unique as
            select
              r.symbol,
              cast(r.trade_date as date) as factor_date,
              arg_max(cast(r.fore_adjust_factor as double), r.filename) as fore_adjust_factor,
              arg_max(cast(r.back_adjust_factor as double), r.filename) as back_adjust_factor,
              count(*) as source_row_count,
              count(distinct cast(r.back_adjust_factor as varchar)) as back_factor_variant_count,
              max(r.filename) as selected_source_file
            from read_parquet({source_sql}, union_by_name=true, filename=true) r
            inner join selected_symbol_source s
              on r.filename = s.filename and r.symbol = s.symbol
            where r.factor_provider = 'tonghuashun'
              and r.back_adjust_factor is not null and r.back_adjust_factor > 0
            group by r.symbol, cast(r.trade_date as date)
            """
        )
        source_quality = _one(
            con,
            """
            select
              count(*) as factor_symbol_days,
              sum(case when back_factor_variant_count > 1 then 1 else 0 end) as back_factor_variant_symbol_days,
              max(back_factor_variant_count) as max_back_factor_variant_count,
              max(source_row_count) as max_duplicate_source_rows
            from factor_unique
            """,
        )
        source_quality.update(selected_source_quality)
        con.execute(
            f"""
            copy (
              with daily as (
                select symbol, cast(trade_date as date) as trade_date
                from read_parquet({daily_sql}, union_by_name=true)
              ),
              combined as (
                select
                  symbol,
                  factor_date as trade_date,
                  fore_adjust_factor,
                  back_adjust_factor,
                  back_adjust_factor as adjust_factor,
                  factor_date as factor_source_date,
                  source_row_count,
                  back_factor_variant_count,
                  0 as row_type
                from factor_unique
                union all
                select
                  symbol,
                  trade_date,
                  null::double as fore_adjust_factor,
                  null::double as back_adjust_factor,
                  null::double as adjust_factor,
                  null::date as factor_source_date,
                  null::bigint as source_row_count,
                  null::bigint as back_factor_variant_count,
                  1 as row_type
                from daily
              ),
              filled as (
                select
                  symbol,
                  trade_date,
                  row_type,
                  last_value(fore_adjust_factor ignore nulls) over w as fore_adjust_factor,
                  last_value(back_adjust_factor ignore nulls) over w as back_adjust_factor,
                  last_value(adjust_factor ignore nulls) over w as adjust_factor,
                  last_value(factor_source_date ignore nulls) over w as factor_source_date,
                  last_value(source_row_count ignore nulls) over w as source_row_count,
                  last_value(back_factor_variant_count ignore nulls) over w as back_factor_variant_count
                from combined
                window w as (
                  partition by symbol
                  order by trade_date, row_type
                  rows between unbounded preceding and current row
                )
              )
              select
                symbol,
                strftime(trade_date, '%Y-%m-%d') as trade_date,
                cast(coalesce(fore_adjust_factor, 1.0) as double) as fore_adjust_factor,
                cast(coalesce(back_adjust_factor, 1.0) as double) as back_adjust_factor,
                cast(coalesce(adjust_factor, back_adjust_factor, 1.0) as double) as adjust_factor,
                'tonghuashun' as factor_provider,
                'standard_qdp_v2_dense_hfq_factor_ffill' as factor_semantics,
                case
                  when factor_source_date is null then 'qdp_v2_default_no_prior_factor'
                  else 'external_quant_zip_dedup_ffill'
                end as source,
                strftime(coalesce(factor_source_date, trade_date), '%Y-%m-%d') as factor_source_date,
                case
                  when factor_source_date is null then 0
                  else date_diff('day', factor_source_date, trade_date)
                end as ffill_days
              from filled
              where row_type = 1
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
              sum(case when back_adjust_factor is null or back_adjust_factor <= 0 then 1 else 0 end) as bad_back_rows,
              sum(case when adjust_factor is null or adjust_factor <= 0 then 1 else 0 end) as bad_adjust_rows,
              sum(case when fore_adjust_factor is null then 1 else 0 end) as null_fore_rows,
              sum(case when fore_adjust_factor <= 0 then 1 else 0 end) as non_positive_fore_rows,
              sum(case when ffill_days > 0 then 1 else 0 end) as ffilled_rows,
              sum(case when source = 'qdp_v2_default_no_prior_factor' then 1 else 0 end) as default_factor_rows,
              max(ffill_days) as max_ffill_days
            from read_parquet({_sql_literal(str(target_path))}, union_by_name=true)
            """,
        )
        schema = _duckdb_schema(con, target_path)
    finally:
        try:
            con.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    status = "ok"
    blockers: list[str] = []
    if int(stats.get("row_count", 0) or 0) != int(daily.row_count or 0):
        blockers.append("row_count_does_not_match_market_daily_raw")
    if int(stats.get("row_count", 0) or 0) != int(stats.get("distinct_keys", 0) or 0):
        blockers.append("primary_key_not_unique")
    if int(stats.get("bad_back_rows", 0) or 0) or int(stats.get("bad_adjust_rows", 0) or 0):
        blockers.append("non_positive_or_null_factor")
    if blockers:
        status = "blocked"

    final_shards = target_dir / "shards"
    target_dir.mkdir(parents=True, exist_ok=True)
    if final_shards.exists():
        shutil.rmtree(final_shards)
    staging_shards.replace(final_shards)
    shutil.rmtree(staging, ignore_errors=True)
    final_path = final_shards / target_path.name
    entry = ShardManifestEntry(
        path=f"datasets/adjust_factor/{target_dataset_id}/shards/{final_path.name}",
        row_count=int(stats.get("row_count", 0) or 0),
        start_date=str(stats.get("start_date", "") or ""),
        end_date=str(stats.get("end_date", "") or ""),
        status="stored",
        file_size=int(final_path.stat().st_size),
        schema_hash=schema_hash(schema),
        source_path="",
        content_key="standard_adjust_factor",
        metadata={"source_dataset_id": source.dataset_id, "daily_dataset_id": daily.dataset_id},
    )
    manifest = DatasetManifest(
        dataset_id=target_dataset_id,
        domain="adjust_factor",
        layer="raw",
        frequency="1d",
        contract_version="qdp_v2_adjust_factor_standard_v2",
        primary_key=["trade_date", "symbol"],
        start_date=entry.start_date,
        end_date=entry.end_date,
        row_count=entry.row_count,
        schema_hash=schema_hash(schema),
        schema=schema,
        shards=[entry],
        source={
            "provider": "qdp_v2",
            "created_by": "rebuild_adjust_factor_standard",
            "created_at": utc_now(),
            "source_dataset_id": source.dataset_id,
            "daily_dataset_id": daily.dataset_id,
        },
        quality={
            "path_refs_exist": True,
            "primary_key_unique": status == "ok",
            "date_coverage_ok": status == "ok",
            "aligns_to_market_daily_raw": status == "ok",
            "factor_positive": status == "ok",
            "adjust_factor_semantics": "back_adjust_factor",
            "fore_adjust_factor_note": "source fore-adjusted value may be non-positive; do not use it as a positive multiplicative factor",
            "source_factor_quality": source_quality,
            "ffilled_rows": int(stats.get("ffilled_rows", 0) or 0),
            "default_factor_rows": int(stats.get("default_factor_rows", 0) or 0),
            "max_ffill_days": int(stats.get("max_ffill_days", 0) or 0),
        },
        notes=[
            "standard daily adjust factor derived from deduplicated tonghuashun back-adjust factors",
            "one row per active market_daily_raw symbol-day",
        ],
    )
    manifest_path = write_dataset_manifest(root, manifest)
    active_path = ""
    if activate and status == "ok":
        active_payload = dict(active)
        dataset_map = dict(active_payload.get("datasets", {}) or active_dataset_map(active_payload))
        dataset_map["adjust_factor"] = target_dataset_id
        active_payload["datasets"] = dataset_map
        active_path = str(write_active_manifest(root, active_payload).resolve())
    payload = {
        "status": status,
        "blockers": blockers,
        "target_dataset_id": target_dataset_id,
        "manifest_path": str(manifest_path.resolve()),
        "active_manifest": active_path,
        "source_dataset_id": source.dataset_id,
        "daily_dataset_id": daily.dataset_id,
        "stats": stats,
        "source_quality": source_quality,
        "runtime_environment": runtime_environment(),
    }
    atomic_write_json(root / "runs" / f"rebuild_adjust_factor_standard_{_stamp()}.json", payload)
    return payload


def rebuild_industry_concept_complete(
    *,
    workspace_root: str | Path | None = None,
    runtime: str = "fast",
    duckdb_memory_limit: str = "",
    threads: int = 0,
    activate: bool = False,
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    datasets = active_dataset_map(active)
    source = _require_manifest(root, datasets, "industry_concept")
    universe = _require_manifest(root, datasets, "universe_snapshot")
    profile = resolve_runtime_profile(runtime)
    memory_limit = str(duckdb_memory_limit or "").strip() or profile.duckdb_memory_limit
    thread_count = int(threads or profile.duckdb_threads or 1)
    target_dataset_id = "industry_concept__" + stable_hash(
        {"source": source.dataset_id, "universe": universe.dataset_id, "contract": "qdp_v2_industry_concept_complete_v2"}
    )
    target_dir = root / "datasets" / "industry_concept" / target_dataset_id
    staging = target_dir / ".staging"
    staging_shards = staging / "shards"
    if staging.exists():
        shutil.rmtree(staging)
    staging_shards.mkdir(parents=True, exist_ok=True)
    target_path = staging_shards / "part_000000_industry_concept_complete.parquet"
    temp_dir = root / "audits" / ".duckdb_tmp" / f"industry_complete_{_stamp()}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    import duckdb  # type: ignore

    con = duckdb.connect(":memory:")
    try:
        _configure(con, memory_limit=memory_limit, threads=thread_count, temp_dir=temp_dir)
        source_sql = _path_list_sql(_shard_paths(root, source))
        universe_sql = _path_list_sql(_shard_paths(root, universe))
        before = _one(
            con,
            f"""
            with u as (
              select trade_date, symbol from read_parquet({universe_sql}, union_by_name=true)
            ),
            i as (
              select trade_date, symbol, industry, concept_tags, source
              from read_parquet({source_sql}, union_by_name=true)
            )
            select
              (select count(*) from u) as universe_rows,
              (select count(*) from i) as source_rows,
              (select count(*) from u left join i using (trade_date, symbol) where i.symbol is null) as missing_source_rows,
              (select sum(case when industry is null or trim(cast(industry as varchar)) = '' then 1 else 0 end) from i) as blank_industry_rows
            from u
            limit 1
            """,
        )
        con.execute(
            f"""
            copy (
              with u as (
                select symbol, trade_date
                from read_parquet({universe_sql}, union_by_name=true)
              ),
              i as (
                select symbol, trade_date, industry, concept_tags, source
                from read_parquet({source_sql}, union_by_name=true)
              )
              select
                u.symbol,
                u.trade_date,
                case
                  when i.symbol is null then 'UNKNOWN'
                  when i.industry is null or trim(cast(i.industry as varchar)) = '' then 'UNKNOWN'
                  else cast(i.industry as varchar)
                end as industry,
                coalesce(cast(i.concept_tags as varchar), '') as concept_tags,
                case
                  when i.symbol is null then 'qdp_v2_explicit_unknown'
                  when i.industry is null or trim(cast(i.industry as varchar)) = '' then 'qdp_v2_blank_industry_marked_unknown'
                  else coalesce(cast(i.source as varchar), 'unknown')
                end as source
              from u
              left join i using (trade_date, symbol)
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
              sum(case when industry = 'UNKNOWN' then 1 else 0 end) as unknown_industry_rows
            from read_parquet({_sql_literal(str(target_path))}, union_by_name=true)
            """,
        )
        schema = _duckdb_schema(con, target_path)
    finally:
        try:
            con.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    status = "ok"
    blockers: list[str] = []
    if int(stats.get("row_count", 0) or 0) != int(universe.row_count or 0):
        blockers.append("row_count_does_not_match_universe_snapshot")
    if int(stats.get("row_count", 0) or 0) != int(stats.get("distinct_keys", 0) or 0):
        blockers.append("primary_key_not_unique")
    if blockers:
        status = "blocked"

    final_shards = target_dir / "shards"
    target_dir.mkdir(parents=True, exist_ok=True)
    if final_shards.exists():
        shutil.rmtree(final_shards)
    staging_shards.replace(final_shards)
    shutil.rmtree(staging, ignore_errors=True)
    final_path = final_shards / target_path.name
    entry = ShardManifestEntry(
        path=f"datasets/industry_concept/{target_dataset_id}/shards/{final_path.name}",
        row_count=int(stats.get("row_count", 0) or 0),
        start_date=str(stats.get("start_date", "") or ""),
        end_date=str(stats.get("end_date", "") or ""),
        status="stored",
        file_size=int(final_path.stat().st_size),
        schema_hash=schema_hash(schema),
        source_path="",
        content_key="complete_industry_concept",
        metadata={"source_dataset_id": source.dataset_id, "universe_dataset_id": universe.dataset_id},
    )
    manifest = DatasetManifest(
        dataset_id=target_dataset_id,
        domain="industry_concept",
        layer="raw",
        frequency="1d",
        contract_version="qdp_v2_industry_concept_v2",
        primary_key=["trade_date", "symbol"],
        start_date=entry.start_date,
        end_date=entry.end_date,
        row_count=entry.row_count,
        schema_hash=schema_hash(schema),
        schema=schema,
        shards=[entry],
        source={
            "provider": "qdp_v2",
            "created_by": "rebuild_industry_concept_complete",
            "created_at": utc_now(),
            "source_dataset_id": source.dataset_id,
            "universe_dataset_id": universe.dataset_id,
        },
        quality={
            "path_refs_exist": True,
            "primary_key_unique": status == "ok",
            "date_coverage_ok": status == "ok",
            "aligns_to_universe_snapshot": status == "ok",
            "unknown_industry_rows": int(stats.get("unknown_industry_rows", 0) or 0),
            "source_missing_rows_marked_unknown": int(before.get("missing_source_rows", 0) or 0),
            "source_blank_industry_rows_marked_unknown": int(before.get("blank_industry_rows", 0) or 0),
        },
        notes=[
            "complete daily industry/concept table aligned to universe_snapshot",
            "missing or blank source industry is explicitly represented as UNKNOWN",
        ],
    )
    manifest_path = write_dataset_manifest(root, manifest)
    active_path = ""
    if activate and status == "ok":
        active_payload = dict(active)
        dataset_map = dict(active_payload.get("datasets", {}) or active_dataset_map(active_payload))
        dataset_map["industry_concept"] = target_dataset_id
        active_payload["datasets"] = dataset_map
        active_path = str(write_active_manifest(root, active_payload).resolve())
    payload = {
        "status": status,
        "blockers": blockers,
        "target_dataset_id": target_dataset_id,
        "manifest_path": str(manifest_path.resolve()),
        "active_manifest": active_path,
        "source_dataset_id": source.dataset_id,
        "universe_dataset_id": universe.dataset_id,
        "before": before,
        "stats": stats,
        "runtime_environment": runtime_environment(),
    }
    atomic_write_json(root / "runs" / f"rebuild_industry_concept_complete_{_stamp()}.json", payload)
    return payload


def audit_meta_domains(
    *,
    workspace_root: str | Path | None = None,
    runtime: str = "fast",
    duckdb_memory_limit: str = "",
    threads: int = 0,
    writeback: bool = False,
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    datasets = active_dataset_map(active)
    profile = resolve_runtime_profile(runtime)
    memory_limit = str(duckdb_memory_limit or "").strip() or profile.duckdb_memory_limit
    thread_count = int(threads or profile.duckdb_threads or 1)
    reports: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    import duckdb  # type: ignore

    temp_dir = root / "audits" / ".duckdb_tmp" / f"meta_quality_{_stamp()}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    try:
        _configure(con, memory_limit=memory_limit, threads=thread_count, temp_dir=temp_dir)
        context = _context_sql(root, datasets)
        for domain in META_DOMAINS:
            manifest = _require_manifest(root, datasets, domain)
            pk = PRIMARY_KEYS[domain]
            report = _audit_one_domain(con=con, root=root, manifest=manifest, primary_key=pk, context=context)
            reports.append(report)
            if report["status"] != "ok":
                findings.append({"severity": "high", "domain": domain, "code": "meta_domain_quality_check_failed", "evidence": report})
            if writeback:
                _write_quality_back(root=root, manifest=manifest, primary_key=pk, report=report)
    finally:
        try:
            con.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    status = "ok" if not findings else "needs_attention"
    payload = {
        "status": status,
        "audit_type": "meta_domain_deep_quality",
        "audited_at": utc_now(),
        "active_as_of_date": str(active.get("active_as_of_date", "") or ""),
        "runtime": profile.name,
        "duckdb_memory_limit": memory_limit,
        "threads": thread_count,
        "domains": reports,
        "findings": findings,
        "finding_count": len(findings),
        "writeback": bool(writeback),
        "runtime_environment": runtime_environment(),
    }
    audit_path = root / "audits" / f"meta_domain_quality_{_stamp()}.json"
    md_path = root / "audits" / f"meta_domain_quality_{_stamp()}.md"
    atomic_write_json(audit_path, payload)
    md_path.write_text(_format_meta_markdown(payload), encoding="utf-8")
    payload["audit_path"] = str(audit_path.resolve())
    payload["markdown_path"] = str(md_path.resolve())
    return payload


def _audit_one_domain(*, con: Any, root: Path, manifest: DatasetManifest, primary_key: list[str], context: Mapping[str, str]) -> dict[str, Any]:
    domain = manifest.domain
    paths_sql = _path_list_sql(_shard_paths(root, manifest))
    pk_report = _primary_key_report(con, paths_sql=paths_sql, primary_key=primary_key)
    domain_report: dict[str, Any] = {
        "domain": domain,
        "dataset_id": manifest.dataset_id,
        "primary_key": primary_key,
        "row_count": manifest.row_count,
        "start_date": manifest.start_date,
        "end_date": manifest.end_date,
        "primary_key_report": pk_report,
        "checks": {},
    }
    if domain == "trading_calendar":
        domain_report["checks"]["calendar_continuity"] = _calendar_continuity(con, paths_sql)
        domain_report["checks"]["daily_dates_are_open"] = _daily_dates_are_open(con, calendar_sql=paths_sql, daily_sql=context["market_daily_raw"])
    elif domain == "universe_snapshot":
        domain_report["checks"]["open_date_coverage"] = _open_date_coverage(con, target_sql=paths_sql, calendar_sql=context["trading_calendar"])
        domain_report["checks"]["scope"] = _scope_report(con, paths_sql=paths_sql)
        domain_report["checks"]["symbol_start_overrides"] = _override_report(con, paths_sql=paths_sql)
    elif domain == "security_status":
        domain_report["checks"]["open_date_coverage"] = _open_date_coverage(con, target_sql=paths_sql, calendar_sql=context["trading_calendar"])
        domain_report["checks"]["universe_key_alignment"] = _key_alignment(con, left_sql=context["universe_snapshot"], right_sql=paths_sql)
        domain_report["checks"]["status_flags"] = _security_status_flags(con, paths_sql=paths_sql)
    elif domain == "adjust_factor":
        domain_report["checks"]["daily_key_alignment"] = _key_alignment(con, left_sql=context["market_daily_raw"], right_sql=paths_sql)
        domain_report["checks"]["factor_validity"] = _factor_validity(con, paths_sql=paths_sql)
    elif domain == "industry_concept":
        domain_report["checks"]["universe_key_alignment"] = _key_alignment(con, left_sql=context["universe_snapshot"], right_sql=paths_sql)
        domain_report["checks"]["industry_validity"] = _industry_validity(con, paths_sql=paths_sql)
    elif domain == "index_constituents":
        domain_report["checks"]["index_dates"] = _index_date_report(con, index_sql=paths_sql, calendar_sql=context["trading_calendar"])
        domain_report["checks"]["scope"] = _scope_report(con, paths_sql=paths_sql)

    failed = [name for name, item in domain_report["checks"].items() if item.get("status") != "ok"]
    domain_report["status"] = "ok" if pk_report["status"] == "ok" and not failed else "needs_attention"
    domain_report["failed_checks"] = failed
    return domain_report


def _write_quality_back(*, root: Path, manifest: DatasetManifest, primary_key: list[str], report: Mapping[str, Any]) -> None:
    payload = manifest.to_dict()
    quality = dict(payload.get("quality", {}) or {})
    pk_ok = dict(report.get("primary_key_report", {}) or {}).get("status") == "ok"
    checks = dict(report.get("checks", {}) or {})
    coverage_ok = _coverage_ok(manifest.domain, checks)
    quality.update(
        {
            "path_refs_exist": True,
            "primary_key_unique": bool(pk_ok),
            "primary_key_audit": {
                "audit_type": "meta_domain_deep_quality",
                "audited_at": utc_now(),
                "method": "global_exact_count_distinct",
                "status": "passed" if pk_ok else "failed",
            },
            "date_coverage_ok": bool(coverage_ok),
            "date_coverage_audit": {
                "audit_type": "meta_domain_deep_quality",
                "audited_at": utc_now(),
                "status": "passed" if coverage_ok else "needs_attention",
            },
            "meta_domain_quality": {
                "status": report.get("status"),
                "checks": checks,
            },
        }
    )
    if manifest.domain == "security_status":
        quality["historical_st_rows"] = int(dict(checks.get("status_flags", {}) or {}).get("st_rows", 0) or 0)
    if manifest.domain == "industry_concept":
        quality["unknown_industry_rows"] = int(dict(checks.get("industry_validity", {}) or {}).get("unknown_industry_rows", 0) or 0)
    if manifest.domain == "adjust_factor":
        validity = dict(checks.get("factor_validity", {}) or {})
        quality["factor_positive"] = validity.get("status") == "ok"
        quality["aligns_to_market_daily_raw"] = dict(checks.get("daily_key_alignment", {}) or {}).get("status") == "ok"
    payload["primary_key"] = primary_key
    payload["quality"] = quality
    manifest_path = dataset_manifest_for_id(root, manifest.dataset_id, manifest.domain)
    if manifest_path is None:
        raise FileNotFoundError(f"dataset manifest not found for writeback: {manifest.domain}:{manifest.dataset_id}")
    atomic_write_json(manifest_path, payload)


def _coverage_ok(domain: str, checks: Mapping[str, Any]) -> bool:
    if domain == "trading_calendar":
        return all(dict(checks.get(name, {}) or {}).get("status") == "ok" for name in ("calendar_continuity", "daily_dates_are_open"))
    if domain in {"universe_snapshot", "security_status"}:
        return dict(checks.get("open_date_coverage", {}) or {}).get("status") == "ok"
    if domain == "adjust_factor":
        return dict(checks.get("daily_key_alignment", {}) or {}).get("status") == "ok"
    if domain == "industry_concept":
        return dict(checks.get("universe_key_alignment", {}) or {}).get("status") == "ok"
    if domain == "index_constituents":
        return dict(checks.get("index_dates", {}) or {}).get("status") == "ok"
    return False


def _primary_key_report(con: Any, *, paths_sql: str, primary_key: list[str]) -> dict[str, Any]:
    null_expr = " or ".join(f"{_q(col)} is null" for col in primary_key)
    row = _one(
        con,
        f"""
        select
          count(*) as row_count,
          count(distinct {_key_expr(primary_key)}) as distinct_key_count,
          sum(case when {null_expr} then 1 else 0 end) as key_null_rows
        from read_parquet({paths_sql}, union_by_name=true)
        """,
    )
    duplicate_rows = int(row["row_count"] or 0) - int(row["distinct_key_count"] or 0)
    return {
        "status": "ok" if duplicate_rows == 0 and int(row["key_null_rows"] or 0) == 0 else "failed",
        "row_count": int(row["row_count"] or 0),
        "distinct_key_count": int(row["distinct_key_count"] or 0),
        "duplicate_rows": max(0, duplicate_rows),
        "key_null_rows": int(row["key_null_rows"] or 0),
    }


def _calendar_continuity(con: Any, paths_sql: str) -> dict[str, Any]:
    row = _one(
        con,
        f"""
        select
          count(*) as row_count,
          count(distinct trade_date) as date_count,
          min(cast(trade_date as date)) as start_date,
          max(cast(trade_date as date)) as end_date,
          date_diff('day', min(cast(trade_date as date)), max(cast(trade_date as date))) + 1 as expected_calendar_days,
          sum(case when is_open then 1 else 0 end) as open_days
        from read_parquet({paths_sql}, union_by_name=true)
        """,
    )
    return {
        **row,
        "status": "ok" if int(row["date_count"] or 0) == int(row["expected_calendar_days"] or 0) == int(row["row_count"] or 0) else "failed",
    }


def _daily_dates_are_open(con: Any, *, calendar_sql: str, daily_sql: str) -> dict[str, Any]:
    row = _one(
        con,
        f"""
        with daily_dates as (
          select distinct trade_date from read_parquet({daily_sql}, union_by_name=true)
        ),
        cal as (
          select trade_date, is_open from read_parquet({calendar_sql}, union_by_name=true)
        )
        select
          count(*) as daily_date_count,
          sum(case when cal.trade_date is null then 1 else 0 end) as missing_calendar_dates,
          sum(case when cal.is_open = false then 1 else 0 end) as non_open_daily_dates
        from daily_dates
        left join cal using (trade_date)
        """,
    )
    return {
        **row,
        "status": "ok" if int(row["missing_calendar_dates"] or 0) == 0 and int(row["non_open_daily_dates"] or 0) == 0 else "failed",
    }


def _open_date_coverage(con: Any, *, target_sql: str, calendar_sql: str) -> dict[str, Any]:
    row = _one(
        con,
        f"""
        with open_dates as (
          select trade_date from read_parquet({calendar_sql}, union_by_name=true) where is_open = true
        ),
        target_dates as (
          select distinct trade_date from read_parquet({target_sql}, union_by_name=true)
        )
        select
          (select count(*) from open_dates) as open_date_count,
          (select count(*) from target_dates) as target_date_count,
          (select count(*) from open_dates left join target_dates using (trade_date) where target_dates.trade_date is null) as missing_open_dates,
          (select count(*) from target_dates left join open_dates using (trade_date) where open_dates.trade_date is null) as extra_non_open_dates
        from open_dates
        limit 1
        """,
    )
    return {
        **row,
        "status": "ok" if int(row["missing_open_dates"] or 0) == 0 and int(row["extra_non_open_dates"] or 0) == 0 else "failed",
    }


def _key_alignment(con: Any, *, left_sql: str, right_sql: str) -> dict[str, Any]:
    row = _one(
        con,
        f"""
        with l as (
          select trade_date, symbol from read_parquet({left_sql}, union_by_name=true)
        ),
        r as (
          select trade_date, symbol from read_parquet({right_sql}, union_by_name=true)
        )
        select
          (select count(*) from l) as left_rows,
          (select count(*) from r) as right_rows,
          (select count(*) from l left join r using (trade_date, symbol) where r.symbol is null) as missing_in_right,
          (select count(*) from r left join l using (trade_date, symbol) where l.symbol is null) as extra_in_right
        from l
        limit 1
        """,
    )
    return {
        **row,
        "status": "ok" if int(row["missing_in_right"] or 0) == 0 and int(row["extra_in_right"] or 0) == 0 else "failed",
    }


def _scope_report(con: Any, *, paths_sql: str) -> dict[str, Any]:
    columns = {str(row[0]).lower() for row in con.execute(f"describe select * from read_parquet({paths_sql}, union_by_name=true)").fetchall()}
    name_delisted_expr = "sum(case when cast(name as varchar) like '%退市%' then 1 else 0 end)" if "name" in columns else "0"
    row = _one(
        con,
        f"""
        select
          count(*) as row_count,
          sum(case when not ({MAINBOARD_PATTERN}) then 1 else 0 end) as non_mainboard_rows,
          {name_delisted_expr} as name_delisted_rows,
          count(distinct symbol) as symbol_count
        from read_parquet({paths_sql}, union_by_name=true)
        """,
    )
    return {
        **row,
        "status": "ok"
        if int(row["non_mainboard_rows"] or 0) == 0 and int(row["name_delisted_rows"] or 0) == 0
        else "failed",
    }


def _override_report(con: Any, *, paths_sql: str) -> dict[str, Any]:
    row = _one(
        con,
        f"""
        select count(*) as rows_before_override
        from read_parquet({paths_sql}, union_by_name=true)
        where symbol = '600036.SH' and cast(trade_date as date) < date '2016-07-25'
        """,
    )
    return {**row, "status": "ok" if int(row["rows_before_override"] or 0) == 0 else "failed"}


def _security_status_flags(con: Any, *, paths_sql: str) -> dict[str, Any]:
    row = _one(
        con,
        f"""
        select
          sum(case when is_st then 1 else 0 end) as st_rows,
          sum(case when is_delisted then 1 else 0 end) as delisted_rows,
          sum(case when is_suspended then 1 else 0 end) as suspended_rows
        from read_parquet({paths_sql}, union_by_name=true)
        """,
    )
    return {
        **row,
        "status": "ok" if int(row["delisted_rows"] or 0) == 0 else "failed",
        "interpretation": "Historical ST rows are retained as PIT status flags; use is_st=false when a study excludes ST days.",
    }


def _factor_validity(con: Any, *, paths_sql: str) -> dict[str, Any]:
    row = _one(
        con,
        f"""
        select
          sum(case when back_adjust_factor is null or back_adjust_factor <= 0 then 1 else 0 end) as bad_back_rows,
          sum(case when adjust_factor is null or adjust_factor <= 0 then 1 else 0 end) as bad_adjust_rows,
          sum(case when fore_adjust_factor is null then 1 else 0 end) as null_fore_rows,
          sum(case when fore_adjust_factor <= 0 then 1 else 0 end) as non_positive_fore_rows,
          sum(case when ffill_days > 0 then 1 else 0 end) as ffilled_rows,
          sum(case when source = 'qdp_v2_default_no_prior_factor' then 1 else 0 end) as default_factor_rows,
          max(ffill_days) as max_ffill_days
        from read_parquet({paths_sql}, union_by_name=true)
        """,
    )
    return {
        **row,
        "status": "ok"
        if int(row["bad_back_rows"] or 0) == 0 and int(row["bad_adjust_rows"] or 0) == 0
        else "failed",
    }


def _industry_validity(con: Any, *, paths_sql: str) -> dict[str, Any]:
    row = _one(
        con,
        f"""
        select
          sum(case when industry is null or trim(cast(industry as varchar)) = '' then 1 else 0 end) as blank_industry_rows,
          sum(case when industry = 'UNKNOWN' then 1 else 0 end) as unknown_industry_rows
        from read_parquet({paths_sql}, union_by_name=true)
        """,
    )
    return {**row, "status": "ok" if int(row["blank_industry_rows"] or 0) == 0 else "failed"}


def _index_date_report(con: Any, *, index_sql: str, calendar_sql: str) -> dict[str, Any]:
    row = _one(
        con,
        f"""
        with idx as (
          select distinct trade_date from read_parquet({index_sql}, union_by_name=true)
        ),
        cal as (
          select trade_date, is_open from read_parquet({calendar_sql}, union_by_name=true)
        )
        select
          count(*) as index_date_count,
          sum(case when cal.trade_date is null then 1 else 0 end) as dates_not_in_calendar,
          sum(case when cal.is_open = false then 1 else 0 end) as dates_not_open
        from idx
        left join cal using (trade_date)
        """,
    )
    return {
        **row,
        "status": "ok" if int(row["dates_not_in_calendar"] or 0) == 0 and int(row["dates_not_open"] or 0) == 0 else "failed",
    }


def _context_sql(root: Path, datasets: Mapping[str, str]) -> dict[str, str]:
    return {
        domain: _path_list_sql(_shard_paths(root, _require_manifest(root, datasets, domain)))
        for domain in ("market_daily_raw", "trading_calendar", "universe_snapshot")
    }


def _require_manifest(root: Path, datasets: Mapping[str, str], domain: str) -> DatasetManifest:
    dataset_id = str(datasets.get(domain, "") or "")
    manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
    if manifest_path is None:
        raise FileNotFoundError(f"active dataset manifest not found: {domain}:{dataset_id}")
    return read_dataset_manifest(manifest_path)


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


def _configure(con: Any, *, memory_limit: str, threads: int, temp_dir: Path) -> None:
    safe_memory = str(memory_limit or "8GB").replace("'", "")
    con.execute(f"set memory_limit='{safe_memory}'")
    con.execute(f"set threads={max(1, int(threads or 1))}")
    con.execute("set preserve_insertion_order=false")
    con.execute(f"set temp_directory={_sql_literal(str(temp_dir))}")


def _duckdb_schema(con: Any, path: Path) -> list[dict[str, str]]:
    rows = con.execute(f"describe select * from read_parquet({_sql_literal(str(path))})").fetchall()
    return [{"name": str(row[0]), "type": str(row[1])} for row in rows]


def _one(con: Any, sql: str) -> dict[str, Any]:
    df = con.execute(sql).fetchdf()
    if df.empty:
        return {}
    return {str(k): _json_scalar(v) for k, v in df.iloc[0].to_dict().items()}


def _json_scalar(value: Any) -> Any:
    try:
        import pandas as pd  # type: ignore

        if pd.isna(value):
            return None
        if isinstance(value, (pd.Timestamp,)):
            return value.strftime("%Y-%m-%d")
    except Exception:
        pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _path_list_sql(paths: list[Path]) -> str:
    return "[" + ", ".join(_sql_literal(str(path)) for path in paths) + "]"


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


def _q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _key_expr(cols: list[str]) -> str:
    parts = [f"coalesce(cast({_q(col)} as varchar), '<NULL>')" for col in cols]
    return "concat_ws(chr(31), " + ", ".join(parts) + ")"


def _stamp() -> str:
    return utc_now().replace(":", "").replace("-", "")


def _format_meta_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# QDP v2 Meta Domain Quality Audit",
        "",
        f"- Status: `{payload.get('status', '')}`",
        f"- Active as of: `{payload.get('active_as_of_date', '')}`",
        f"- Writeback: `{payload.get('writeback', False)}`",
        "",
        "| Domain | Status | Rows | Primary key | Failed checks |",
        "|---|---|---:|---|---|",
    ]
    for item in list(payload.get("domains", []) or []):
        lines.append(
            "| {domain} | `{status}` | {rows} | `{pk}` | {failed} |".format(
                domain=item.get("domain", ""),
                status=item.get("status", ""),
                rows=item.get("row_count", 0),
                pk=",".join(list(item.get("primary_key", []) or [])),
                failed=", ".join(list(item.get("failed_checks", []) or [])) or "-",
            )
        )
    lines.extend(["", "## Findings", ""])
    findings = list(payload.get("findings", []) or [])
    if not findings:
        lines.append("No findings.")
    for finding in findings:
        lines.append(f"- `{finding.get('domain', '')}` {finding.get('code', '')}")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp meta-quality")
    sub = parser.add_subparsers(dest="command", required=True)
    rebuild_adjust = sub.add_parser("rebuild-adjust-factor")
    rebuild_adjust.add_argument("--workspace-root", default="")
    rebuild_adjust.add_argument("--runtime", default="fast", choices=("safe", "balanced", "fast"))
    rebuild_adjust.add_argument("--duckdb-memory-limit", default="")
    rebuild_adjust.add_argument("--threads", type=int, default=0)
    rebuild_adjust.add_argument("--activate", action="store_true")
    rebuild_adjust.add_argument("--json", action="store_true")
    rebuild_industry = sub.add_parser("rebuild-industry-concept")
    rebuild_industry.add_argument("--workspace-root", default="")
    rebuild_industry.add_argument("--runtime", default="fast", choices=("safe", "balanced", "fast"))
    rebuild_industry.add_argument("--duckdb-memory-limit", default="")
    rebuild_industry.add_argument("--threads", type=int, default=0)
    rebuild_industry.add_argument("--activate", action="store_true")
    rebuild_industry.add_argument("--json", action="store_true")
    audit = sub.add_parser("audit")
    audit.add_argument("--workspace-root", default="")
    audit.add_argument("--runtime", default="fast", choices=("safe", "balanced", "fast"))
    audit.add_argument("--duckdb-memory-limit", default="")
    audit.add_argument("--threads", type=int, default=0)
    audit.add_argument("--writeback", action="store_true")
    audit.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "rebuild-adjust-factor":
        payload = rebuild_adjust_factor_standard(
            workspace_root=str(args.workspace_root or "") or None,
            runtime=str(args.runtime or "fast"),
            duckdb_memory_limit=str(args.duckdb_memory_limit or ""),
            threads=int(args.threads or 0),
            activate=bool(args.activate),
        )
    elif args.command == "rebuild-industry-concept":
        payload = rebuild_industry_concept_complete(
            workspace_root=str(args.workspace_root or "") or None,
            runtime=str(args.runtime or "fast"),
            duckdb_memory_limit=str(args.duckdb_memory_limit or ""),
            threads=int(args.threads or 0),
            activate=bool(args.activate),
        )
    else:
        payload = audit_meta_domains(
            workspace_root=str(args.workspace_root or "") or None,
            runtime=str(args.runtime or "fast"),
            duckdb_memory_limit=str(args.duckdb_memory_limit or ""),
            threads=int(args.threads or 0),
            writeback=bool(args.writeback),
        )
    if bool(getattr(args, "json", False)):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print("\n".join(f"{key}: {value}" for key, value in payload.items() if key in {"status", "target_dataset_id", "audit_path", "markdown_path", "active_manifest"}))
    return 0 if str(payload.get("status", "")) in {"ok", "needs_attention"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
