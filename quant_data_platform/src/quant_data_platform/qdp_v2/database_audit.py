from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.audit import _manifest_contract_findings, _parquet_row_count
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
)
from quant_data_platform.qdp_v2.runtime import resolve_runtime_profile
from quant_data_platform.qdp_v2.status import _active_dataset_refs


CORE_DOMAINS = {
    "market_daily_raw",
    "market_daily_panel",
    "market_intraday_1m",
    "market_intraday_5m",
    "trading_calendar",
    "universe_snapshot",
    "security_status",
    "valuation",
    "adjust_factor",
    "industry_concept",
    "index_constituents",
    "intraday_daily_features",
    "limit_intraday_features",
}

RECOMMENDED_RAW_DOMAINS = {
    "limit_status",
    "corporate_actions",
    "share_capital",
    "name_change",
}

OPTIONAL_LONG_HORIZON_DOMAINS = {
    "announcement",
    "financial_quarterly",
    "performance_forecast",
    "performance_express",
}

PRIMARY_KEYS: dict[str, list[str]] = {
    "market_daily_raw": ["trade_date", "symbol"],
    "market_daily_panel": ["trade_date", "symbol"],
    "market_intraday_1m": ["trade_date", "symbol", "bar_time"],
    "market_intraday_5m": ["trade_date", "symbol", "bar_time"],
    "trading_calendar": ["trade_date", "exchange"],
    "universe_snapshot": ["trade_date", "symbol"],
    "security_status": ["trade_date", "symbol"],
    "valuation": ["trade_date", "symbol"],
    "adjust_factor": ["trade_date", "symbol"],
    "industry_concept": ["trade_date", "symbol"],
    "index_constituents": ["trade_date", "index_symbol", "symbol"],
    "intraday_daily_features": ["trade_date", "symbol"],
    "limit_intraday_features": ["trade_date", "symbol"],
    "announcement": ["trade_date", "symbol", "title", "url"],
    "limit_status": ["trade_date", "symbol"],
    "financial_quarterly": ["symbol", "report_date", "source"],
    "performance_forecast": ["symbol", "report_date", "publish_date", "source"],
    "performance_express": ["symbol", "report_date", "publish_date", "source"],
    "corporate_actions": ["symbol", "trade_date", "action_type", "description", "source"],
    "share_capital": ["trade_date", "symbol", "source"],
    "name_change": ["trade_date", "symbol", "change_type", "source"],
}

PRICE_COLUMNS = ("open", "high", "low", "close")


def audit_database(
    *,
    workspace_root: str | Path | None = None,
    deep: bool = False,
    runtime: str = "balanced",
    duckdb_memory_limit: str = "",
    threads: int = 0,
    domains: Iterable[str] | None = None,
    max_shards: int = 0,
    batch_shards: int = 64,
    sample_limit: int = 20,
    full_global_uniqueness: bool = False,
    global_uniqueness_max_rows: int = 50_000_000,
    global_uniqueness_max_shards: int = 2_000,
    skip_cross: bool = False,
    skip_cross_frequency: bool = False,
    progress_path: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    if not active:
        return {
            "status": "error",
            "qdp_v2_root": str(root.resolve()),
            "errors": ["active_manifest_missing"],
            "findings": [_finding("critical", "structure", "", "active_manifest_missing", {}, "Run qdp migrate-v2/activate-v2 first.")],
        }

    profile = resolve_runtime_profile(runtime)
    memory_limit = str(duckdb_memory_limit or "").strip() or profile.duckdb_memory_limit
    thread_count = int(threads or profile.duckdb_threads)
    selected_domains = {str(item).strip() for item in list(domains or []) if str(item).strip()}
    active_refs = _active_dataset_refs(active)
    active_domains = {domain for _, domain, _ in active_refs}
    findings: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    structural = _audit_selected_active(root=root, active=active, selected_domains=selected_domains, verify_footers=deep)
    for err in list(structural.get("errors", []) or []):
        errors.append(str(err))
        findings.append(_finding("critical", "structure", "", "active_audit_error", {"error": str(err)}, "Fix manifest or shard file integrity before trusting data."))
    for warning in list(structural.get("warnings", []) or []):
        warnings.append(str(warning))

    missing_core = sorted(CORE_DOMAINS.difference(active_domains))
    missing_recommended = sorted(RECOMMENDED_RAW_DOMAINS.difference(active_domains))
    if missing_core:
        findings.append(
            _finding(
                "high",
                "coverage",
                "",
                "core_domains_missing",
                {"missing_domains": missing_core},
                "Core domains should be present before treating qdp_v2 active as a complete research database.",
            )
        )
    if missing_recommended:
        findings.append(
            _finding(
                "medium",
                "coverage",
                "",
                "recommended_domains_missing",
                {"missing_domains": missing_recommended},
                "Add these raw domains when expanding from market/PIT data into a fuller research database.",
            )
        )

    dataset_reports: list[dict[str, Any]] = []
    for section, domain, dataset_id in active_refs:
        if selected_domains and domain not in selected_domains:
            continue
        manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
        if manifest_path is None:
            errors.append(f"dataset_manifest_missing:{section}.{domain}:{dataset_id}")
            continue
        manifest = read_dataset_manifest(manifest_path)
        paths = _existing_shard_paths(manifest, root=root, max_shards=max_shards)
        columns = _read_columns(paths[0]) if paths else [str(item.get("name", "")) for item in manifest.schema]
        byte_count = _manifest_byte_count(manifest, root=root)
        report: dict[str, Any] = {
            "section": section,
            "domain": domain,
            "dataset_id": dataset_id,
            "layer": manifest.layer,
            "frequency": manifest.frequency,
            "contract_version": manifest.contract_version,
            "primary_key": _primary_key_for(manifest),
            "start_date": manifest.start_date,
            "end_date": manifest.end_date,
            "row_count": manifest.row_count,
            "shard_count": len(manifest.shards),
            "scanned_shard_count": len(paths),
            "scan_limited": bool(max_shards and len(paths) < len([s for s in manifest.shards if resolve_manifest_path(s.path, root=root).exists()])),
            "byte_count": byte_count,
            "columns": columns,
            "quality": manifest.quality,
            "checks": {},
        }
        contract_findings = _contract_checks(domain=domain, manifest=manifest, columns=columns)
        findings.extend(contract_findings)
        if deep and paths:
            deep_report, deep_findings = _deep_dataset_checks(
                root=root,
                active=active,
                domain=domain,
                manifest=manifest,
                paths=paths,
                columns=columns,
                memory_limit=memory_limit,
                threads=thread_count,
                batch_shards=batch_shards,
                sample_limit=sample_limit,
                full_global_uniqueness=full_global_uniqueness,
                global_uniqueness_max_rows=global_uniqueness_max_rows,
                global_uniqueness_max_shards=global_uniqueness_max_shards,
                scan_limited=bool(max_shards),
                progress_path=Path(progress_path) if progress_path else None,
            )
            report["checks"].update(deep_report)
            findings.extend(deep_findings)
        dataset_reports.append(report)

    if deep and not skip_cross and not selected_domains:
        cross_report, cross_findings = _cross_dataset_checks(
            root=root,
            active=active,
            memory_limit=memory_limit,
            threads=thread_count,
            sample_limit=sample_limit,
            max_shards=max_shards,
            skip_cross_frequency=skip_cross_frequency,
            progress_path=Path(progress_path) if progress_path else None,
        )
        findings.extend(cross_findings)
    elif deep and selected_domains:
        cross_report = {"status": "skipped", "reason": "domain_filter_active"}
    elif deep and skip_cross:
        cross_report = {"status": "skipped", "reason": "--skip-cross"}
    else:
        cross_report = {"status": "skipped", "reason": "pass --deep to run cross-dataset checks"}

    severity_rank = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    max_severity = max((severity_rank.get(str(item.get("severity", "")), 0) for item in findings), default=0)
    status = "error" if errors or max_severity >= 4 else "needs_attention" if max_severity >= 2 else "ok"
    payload = {
        "status": status,
        "qdp_v2_root": str(root.resolve()),
        "active_manifest": str((root / "active" / "active.json").resolve()),
        "active_as_of_date": str(active.get("active_as_of_date", "") or ""),
        "audited_at": utc_now(),
        "deep": bool(deep),
        "runtime": profile.name,
        "duckdb_memory_limit": memory_limit,
        "duckdb_threads": thread_count,
        "duckdb_catalog_required": False,
        "dataset_count": len(dataset_reports),
        "datasets": dataset_reports,
        "coverage": {
            "active_domains": sorted(active_domains),
            "core_domains": sorted(CORE_DOMAINS),
            "missing_core_domains": missing_core,
            "recommended_domains": sorted(RECOMMENDED_RAW_DOMAINS),
            "missing_recommended_domains": missing_recommended,
            "optional_long_horizon_domains": sorted(OPTIONAL_LONG_HORIZON_DOMAINS),
        },
        "structural_audit": {
            "status": structural.get("status", ""),
            "warning_count": len(structural.get("warnings", []) or []),
            "error_count": len(structural.get("errors", []) or []),
        },
        "cross_dataset_checks": cross_report,
        "findings": findings,
        "finding_count": len(findings),
        "warnings": warnings,
        "errors": errors,
    }
    if write:
        audit_id = f"database_audit_{utc_now().replace(':', '').replace('-', '')}"
        json_path = root / "audits" / f"{audit_id}.json"
        md_path = root / "audits" / f"{audit_id}.md"
        atomic_write_json(json_path, payload)
        md_path.write_text(_format_markdown(payload), encoding="utf-8")
        payload["audit_path"] = str(json_path.resolve())
        payload["markdown_path"] = str(md_path.resolve())
    return payload


def _deep_dataset_checks(
    *,
    root: Path,
    active: Mapping[str, Any],
    domain: str,
    manifest: DatasetManifest,
    paths: list[Path],
    columns: list[str],
    memory_limit: str,
    threads: int,
    batch_shards: int,
    sample_limit: int,
    full_global_uniqueness: bool,
    global_uniqueness_max_rows: int,
    global_uniqueness_max_shards: int,
    scan_limited: bool,
    progress_path: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    pk = _primary_key_for(manifest)
    missing_pk = [col for col in pk if col not in columns]
    if missing_pk:
        findings.append(_finding("critical", "schema", domain, "primary_key_columns_missing", {"columns": missing_pk}, "Fix dataset schema or primary-key contract."))
        checks["primary_key"] = {"status": "error", "missing_columns": missing_pk}
        return checks, findings

    per_shard = _per_shard_quality_scan(
        domain=domain,
        paths=paths,
        columns=columns,
        primary_key=pk,
        memory_limit=memory_limit,
        threads=threads,
        batch_shards=batch_shards,
        sample_limit=sample_limit,
        progress_path=progress_path,
    )
    checks.update(per_shard)
    pk_report = checks.get("primary_key", {})
    duplicate_rows = int(pk_report.get("duplicate_rows", 0) or 0)
    key_null_rows = int(pk_report.get("key_null_rows", 0) or 0)
    if duplicate_rows:
        findings.append(
            _finding(
                "high",
                "uniqueness",
                domain,
                "primary_key_duplicate_rows",
                {"duplicate_rows": duplicate_rows, "examples": pk_report.get("duplicate_examples", [])},
                "Rewrite this dataset with deterministic de-duplication before using it as trusted active data.",
            )
        )
    if key_null_rows:
        findings.append(
            _finding(
                "high",
                "completeness",
                domain,
                "primary_key_null_rows",
                {"key_null_rows": key_null_rows},
                "Primary-key columns must be non-null in raw and derived active datasets.",
            )
        )

    validity = checks.get("validity", {})
    bad_price_rows = int(validity.get("bad_price_rows", 0) or 0)
    bad_ohlc_rows = int(validity.get("bad_ohlc_rows", 0) or 0)
    if bad_price_rows or bad_ohlc_rows:
        findings.append(
            _finding(
                "high",
                "validity",
                domain,
                "invalid_ohlc_rows",
                {"bad_price_rows": bad_price_rows, "bad_ohlc_rows": bad_ohlc_rows, "examples": validity.get("examples", [])},
                "Keep raw evidence, then create a cleaned replacement dataset or exclusion mask for research panels.",
            )
        )
    if domain in {"market_intraday_1m", "market_intraday_5m"} and bad_price_rows:
        zero_classification = _classify_intraday_zero_price_rows(
            root=root,
            active=active,
            paths=paths,
            memory_limit=memory_limit,
            threads=threads,
        )
        checks["intraday_zero_price_classification"] = zero_classification
    else:
        zero_classification = {}
    classification = zero_classification
    traded_zero_rows = int(classification.get("traded_day_partial_zero_rows", 0) or 0)
    suspended_zero_rows = int(classification.get("suspended_zero_rows", 0) or 0)
    if traded_zero_rows:
        findings.append(
            _finding(
                "high",
                "validity",
                domain,
                "traded_day_zero_price_bars",
                classification,
                "Repair these symbol-days from mootdx or remove the affected zero-price bars before using intraday data for research.",
            )
        )
    if suspended_zero_rows:
        findings.append(
            _finding(
                "medium",
                "validity",
                domain,
                "suspended_day_zero_price_pseudo_bars",
                classification,
                "Do not store suspended days as zero-price intraday bars; rebuild intraday datasets excluding these pseudo bars and represent suspension in panel/status data.",
            )
        )
    bar_report = checks.get("bar_count", {})
    bad_symbol_days = int(bar_report.get("bad_symbol_days", 0) or 0)
    if bad_symbol_days:
        severity = "medium" if domain == "market_intraday_1m" else "high"
        findings.append(
            _finding(
                severity,
                "completeness",
                domain,
                "bad_intraday_bar_count",
                {"bad_symbol_days": bad_symbol_days, "expected_bars": bar_report.get("expected_bars"), "examples": bar_report.get("examples", [])},
                "Classify bad symbol-days as suspended/listing edge cases or source gaps; repair source gaps before model training.",
            )
        )

    global_needed = bool(full_global_uniqueness) or (
        int(manifest.row_count or 0) <= int(global_uniqueness_max_rows)
        and len(paths) <= int(global_uniqueness_max_shards)
    )
    if len(paths) > 1 and pk and not scan_limited and global_needed:
        global_report = _global_primary_key_check(
            paths=paths,
            primary_key=pk,
            memory_limit=memory_limit,
            threads=threads,
            sample_limit=sample_limit,
        )
        checks["global_primary_key"] = global_report
        if int(global_report.get("duplicate_rows", 0) or 0):
            findings.append(
                _finding(
                    "high",
                    "uniqueness",
                    domain,
                    "global_primary_key_duplicate_rows",
                    {"duplicate_rows": global_report.get("duplicate_rows"), "examples": global_report.get("duplicate_examples", [])},
                    "Cross-shard duplicates require rebuilding the dataset from de-duplicated source shards.",
                )
            )
    elif len(paths) > 1 and pk:
        checks["global_primary_key"] = {
            "status": "not_checked",
            "reason": "large_or_many_shards_requires_--full-global-uniqueness" if not scan_limited else "scan_limited_by_max_shards",
            "row_count": manifest.row_count,
            "shard_count": len(paths),
            "global_uniqueness_max_rows": global_uniqueness_max_rows,
            "global_uniqueness_max_shards": global_uniqueness_max_shards,
        }
        findings.append(
            _finding(
                "low",
                "uniqueness",
                domain,
                "global_primary_key_not_checked",
                {"row_count": manifest.row_count, "shard_count": len(paths)},
                "Run with --full-global-uniqueness for an expensive absolute cross-shard duplicate proof.",
            )
        )
    return checks, findings


def _per_shard_quality_scan(
    *,
    domain: str,
    paths: list[Path],
    columns: list[str],
    primary_key: list[str],
    memory_limit: str,
    threads: int,
    batch_shards: int,
    sample_limit: int,
    progress_path: Path | None,
) -> dict[str, Any]:
    import duckdb  # type: ignore

    row_count = 0
    distinct_key_count = 0
    duplicate_rows = 0
    key_null_rows = 0
    bad_price_rows = 0
    bad_ohlc_rows = 0
    bad_panel_has_bar_rows = 0
    symbol_days = 0
    good_symbol_days = 0
    bad_symbol_days = 0
    min_bars: int | None = None
    max_bars: int | None = None
    duplicate_examples: list[dict[str, Any]] = []
    validity_examples: list[dict[str, Any]] = []
    bar_examples: list[dict[str, Any]] = []
    expected_bars = 240 if domain == "market_intraday_1m" else 48 if domain == "market_intraday_5m" else 0

    with duckdb.connect(":memory:") as con:
        _configure_duckdb(con, memory_limit=memory_limit, threads=threads)
        batches = list(_batches(paths, max(1, int(batch_shards or 1))))
        for batch_index, batch in enumerate(batches, start=1):
            paths_sql = _path_list_sql(batch)
            key_expr = _key_expr(primary_key)
            null_expr = " or ".join(f"{_q(col)} is null" for col in primary_key) or "false"
            row = con.execute(
                f"""
                select
                  count(*) as row_count,
                  count(distinct {key_expr}) as distinct_key_count,
                  sum(case when {null_expr} then 1 else 0 end) as key_null_rows
                from read_parquet({paths_sql}, union_by_name=true)
                """
            ).fetchone()
            n = int(row[0] or 0)
            d = int(row[1] or 0)
            row_count += n
            distinct_key_count += d
            duplicate_rows += max(0, n - d)
            key_null_rows += int(row[2] or 0)
            if n > d and len(duplicate_examples) < sample_limit:
                duplicate_examples.extend(
                    _fetch_examples(
                        con,
                        f"""
                        select {_select_cols(primary_key)}, count(*) as duplicate_count
                        from read_parquet({paths_sql}, union_by_name=true)
                        group by {_select_cols(primary_key)}
                        having count(*) > 1
                        limit {max(1, sample_limit - len(duplicate_examples))}
                        """,
                    )
                )
            if _has_columns(columns, PRICE_COLUMNS):
                price = con.execute(
                    f"""
                    select
                      sum(case when open <= 0 or high <= 0 or low <= 0 or close <= 0 then 1 else 0 end) as bad_price_rows,
                      sum(case when high < low or high < open or high < close or low > open or low > close then 1 else 0 end) as bad_ohlc_rows
                    from read_parquet({paths_sql}, union_by_name=true)
                    """
                ).fetchone()
                bp = int(price[0] or 0)
                bo = int(price[1] or 0)
                bad_price_rows += bp
                bad_ohlc_rows += bo
                if (bp or bo) and len(validity_examples) < sample_limit:
                    validity_examples.extend(
                        _fetch_examples(
                            con,
                            f"""
                            select *
                            from read_parquet({paths_sql}, union_by_name=true)
                            where open <= 0 or high <= 0 or low <= 0 or close <= 0
                               or high < low or high < open or high < close or low > open or low > close
                            limit {max(1, sample_limit - len(validity_examples))}
                            """,
                        )
                    )
            if domain == "market_daily_panel" and "has_bar" in columns and _has_columns(columns, PRICE_COLUMNS):
                panel = con.execute(
                    f"""
                    select count(*) as bad_rows
                    from read_parquet({paths_sql}, union_by_name=true)
                    where has_bar = false
                      and (open is not null or high is not null or low is not null or close is not null or volume is not null or amount is not null)
                    """
                ).fetchone()
                bad_panel_has_bar_rows += int(panel[0] or 0)
            if expected_bars and {"symbol", "trade_date", "bar_time"}.issubset(set(columns)):
                bars = con.execute(
                    f"""
                    with counts as (
                      select symbol, trade_date, count(*) as bars, count(distinct bar_time) as distinct_bars
                      from read_parquet({paths_sql}, union_by_name=true)
                      group by symbol, trade_date
                    )
                    select
                      count(*) as symbol_days,
                      sum(case when bars = {expected_bars} and distinct_bars = {expected_bars} then 1 else 0 end) as good_symbol_days,
                      sum(case when bars != {expected_bars} or distinct_bars != {expected_bars} then 1 else 0 end) as bad_symbol_days,
                      min(bars) as min_bars,
                      max(bars) as max_bars
                    from counts
                    """
                ).fetchone()
                sd = int(bars[0] or 0)
                gd = int(bars[1] or 0)
                bd = int(bars[2] or 0)
                symbol_days += sd
                good_symbol_days += gd
                bad_symbol_days += bd
                if bars[3] is not None:
                    min_bars = int(bars[3]) if min_bars is None else min(min_bars, int(bars[3]))
                if bars[4] is not None:
                    max_bars = int(bars[4]) if max_bars is None else max(max_bars, int(bars[4]))
                if bd and len(bar_examples) < sample_limit:
                    bar_examples.extend(
                        _fetch_examples(
                            con,
                            f"""
                            with counts as (
                              select symbol, trade_date, count(*) as bars, count(distinct bar_time) as distinct_bars
                              from read_parquet({paths_sql}, union_by_name=true)
                              group by symbol, trade_date
                            )
                            select *
                            from counts
                            where bars != {expected_bars} or distinct_bars != {expected_bars}
                            limit {max(1, sample_limit - len(bar_examples))}
                            """,
                        )
                    )
            _append_progress(
                progress_path,
                {
                    "event": "dataset_batch_scanned",
                    "domain": domain,
                    "batch_index": batch_index,
                    "batch_count": len(batches),
                    "batch_shards": len(batch),
                    "row_count_scanned_so_far": row_count,
                    "duplicate_rows_so_far": duplicate_rows,
                    "bad_symbol_days_so_far": bad_symbol_days,
                    "timestamp": utc_now(),
                },
            )
    checks: dict[str, Any] = {
        "primary_key": {
            "method": "batched_exact",
            "batch_shards": int(batch_shards or 1),
            "status": "ok" if duplicate_rows == 0 and key_null_rows == 0 else "failed",
            "row_count_scanned": row_count,
            "distinct_key_count_sum": distinct_key_count,
            "duplicate_rows": duplicate_rows,
            "key_null_rows": key_null_rows,
            "duplicate_examples": duplicate_examples[:sample_limit],
        }
    }
    if _has_columns(columns, PRICE_COLUMNS):
        checks["validity"] = {
            "status": "ok" if bad_price_rows == 0 and bad_ohlc_rows == 0 and bad_panel_has_bar_rows == 0 else "failed",
            "bad_price_rows": bad_price_rows,
            "bad_ohlc_rows": bad_ohlc_rows,
            "bad_panel_has_bar_rows": bad_panel_has_bar_rows,
            "examples": validity_examples[:sample_limit],
        }
    if expected_bars:
        checks["bar_count"] = {
            "status": "ok" if bad_symbol_days == 0 else "needs_review",
            "expected_bars": expected_bars,
            "symbol_days": symbol_days,
            "good_symbol_days": good_symbol_days,
            "bad_symbol_days": bad_symbol_days,
            "min_bars": min_bars,
            "max_bars": max_bars,
            "examples": bar_examples[:sample_limit],
        }
    return checks


def _global_primary_key_check(
    *,
    paths: list[Path],
    primary_key: list[str],
    memory_limit: str,
    threads: int,
    sample_limit: int,
) -> dict[str, Any]:
    import duckdb  # type: ignore

    with duckdb.connect(":memory:") as con:
        _configure_duckdb(con, memory_limit=memory_limit, threads=threads)
        paths_sql = _path_list_sql(paths)
        key_expr = _key_expr(primary_key)
        row = con.execute(
            f"""
            select count(*) as row_count, count(distinct {key_expr}) as distinct_key_count
            from read_parquet({paths_sql}, union_by_name=true)
            """
        ).fetchone()
        n = int(row[0] or 0)
        d = int(row[1] or 0)
        examples: list[dict[str, Any]] = []
        if n > d:
            examples = _fetch_examples(
                con,
                f"""
                select {_select_cols(primary_key)}, count(*) as duplicate_count
                from read_parquet({paths_sql}, union_by_name=true)
                group by {_select_cols(primary_key)}
                having count(*) > 1
                limit {sample_limit}
                """,
            )
    return {
        "method": "global_exact",
        "status": "ok" if n == d else "failed",
        "row_count_scanned": n,
        "distinct_key_count": d,
        "duplicate_rows": max(0, n - d),
        "duplicate_examples": examples,
    }


def _cross_dataset_checks(
    *,
    root: Path,
    active: Mapping[str, Any],
    memory_limit: str,
    threads: int,
    sample_limit: int,
    max_shards: int = 0,
    skip_cross_frequency: bool = False,
    progress_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    raw = dict(active.get("raw", {}) or {})
    panels = dict(active.get("research_panels", {}) or {})
    try:
        daily = _daily_raw_panel_check(root=root, raw_id=str(raw.get("market_daily_raw", "") or ""), panel_id=str(panels.get("market_daily_panel", "") or ""), memory_limit=memory_limit, threads=threads)
        report["market_daily_raw_vs_panel"] = daily
        if daily.get("status") != "ok":
            findings.append(
                _finding(
                    "high",
                    "consistency",
                    "market_daily_raw/market_daily_panel",
                    "daily_raw_panel_mismatch",
                    daily,
                    "Rebuild the research panel from raw + trading calendar + status sidecars.",
                )
            )
    except Exception as exc:
        report["market_daily_raw_vs_panel"] = {"status": "error", "error": str(exc)}
        findings.append(_finding("medium", "consistency", "market_daily_raw/market_daily_panel", "daily_raw_panel_check_failed", {"error": str(exc)}, "Inspect daily raw/panel schemas and rerun the audit."))
    try:
        pit = _pit_coverage_check(root=root, active=active, memory_limit=memory_limit, threads=threads, sample_limit=sample_limit)
        report["pit_coverage"] = pit
        if pit.get("status") != "ok":
            findings.append(
                _finding(
                    "high",
                    "coverage",
                    "universe_snapshot/security_status",
                    "pit_coverage_gap",
                    pit,
                    "Backfill PIT snapshots for missing open trading dates before relying on historical universe/status filters.",
                )
            )
    except Exception as exc:
        report["pit_coverage"] = {"status": "error", "error": str(exc)}
        findings.append(_finding("medium", "coverage", "universe_snapshot/security_status", "pit_coverage_check_failed", {"error": str(exc)}, "Inspect trading_calendar, universe_snapshot, and security_status schemas."))
    try:
        scope = _active_scope_check(root=root, active=active, memory_limit=memory_limit, threads=threads)
        report["active_scope"] = scope
        if scope.get("status") != "ok":
            findings.append(
                _finding(
                    "high",
                    "scope",
                    "universe_snapshot/security_status",
                    "active_scope_violation",
                    scope,
                    "Rebuild scope-mainboard active datasets or inspect active-date status source.",
                )
            )
    except Exception as exc:
        report["active_scope"] = {"status": "error", "error": str(exc)}
        findings.append(_finding("medium", "scope", "universe_snapshot/security_status", "active_scope_check_failed", {"error": str(exc)}, "Inspect active-date scope metadata and status sidecar."))
    if skip_cross_frequency:
        report["intraday_5m_from_1m"] = {"status": "skipped", "reason": "--skip-cross-frequency"}
        report["intraday_vs_daily"] = {"status": "skipped", "reason": "--skip-cross-frequency"}
        return report, findings
    try:
        five_from_one = _intraday_5m_from_1m_check(
            root=root,
            active=active,
            memory_limit=memory_limit,
            threads=threads,
            sample_limit=sample_limit,
            max_shards=max_shards,
            progress_path=progress_path,
        )
        report["intraday_5m_from_1m"] = five_from_one
        if five_from_one.get("status") not in {"ok", "skipped"}:
            findings.append(
                _finding(
                    "high",
                    "consistency",
                    "market_intraday_1m/market_intraday_5m",
                    "intraday_5m_not_consistent_with_1m",
                    five_from_one,
                    "Rebuild 5m from the active 1m dataset before using 5m bars or 5m-derived features.",
                )
            )
    except Exception as exc:
        report["intraday_5m_from_1m"] = {"status": "error", "error": str(exc)}
        findings.append(_finding("medium", "consistency", "market_intraday_1m/market_intraday_5m", "intraday_5m_from_1m_check_failed", {"error": str(exc)}, "Inspect 1m/5m schemas and rerun the quality audit."))
    try:
        intraday_daily = _intraday_vs_daily_check(
            root=root,
            active=active,
            memory_limit=memory_limit,
            threads=threads,
            sample_limit=sample_limit,
            max_shards=max_shards,
        )
        report["intraday_vs_daily"] = intraday_daily
        intraday_status = str(intraday_daily.get("status", "") or "")
        if intraday_status == "failed":
            findings.append(
                _finding(
                    "high",
                    "consistency",
                    "market_intraday_5m/market_daily_raw",
                    "intraday_daily_mismatch",
                    intraday_daily,
                    "Classify missing symbol-days and reconcile price/volume/amount mismatches before treating cross-frequency features as fully trusted.",
                )
            )
        elif intraday_status == "needs_review":
            findings.append(
                _finding(
                    "medium",
                    "consistency",
                    "market_intraday_5m/market_daily_raw",
                    "intraday_daily_source_conflict",
                    intraday_daily,
                    "Keep daily and intraday as independent source facts; inspect major source conflicts before using cross-frequency labels.",
                )
            )
        elif intraday_status not in {"ok", "skipped"}:
            findings.append(
                _finding(
                    "medium",
                    "consistency",
                    "market_intraday_5m/market_daily_raw",
                    "intraday_daily_check_unexpected_status",
                    intraday_daily,
                    "Inspect the intraday-vs-daily audit output.",
                )
            )
    except Exception as exc:
        report["intraday_vs_daily"] = {"status": "error", "error": str(exc)}
        findings.append(_finding("medium", "consistency", "market_intraday_5m/market_daily_raw", "intraday_vs_daily_check_failed", {"error": str(exc)}, "Inspect daily and intraday schemas and rerun the quality audit."))
    return report, findings


def _daily_raw_panel_check(*, root: Path, raw_id: str, panel_id: str, memory_limit: str, threads: int) -> dict[str, Any]:
    if not raw_id or not panel_id:
        return {"status": "skipped", "reason": "daily raw or panel missing"}
    raw_manifest = read_dataset_manifest(dataset_manifest_for_id(root, raw_id, "market_daily_raw") or "")
    panel_manifest = read_dataset_manifest(dataset_manifest_for_id(root, panel_id, "market_daily_panel") or "")
    raw_paths = _existing_shard_paths(raw_manifest, root=root, max_shards=0)
    panel_paths = _existing_shard_paths(panel_manifest, root=root, max_shards=0)
    if not raw_paths or not panel_paths:
        return {"status": "error", "reason": "daily raw or panel has no readable shards"}
    import duckdb  # type: ignore

    with duckdb.connect(":memory:") as con:
        _configure_duckdb(con, memory_limit=memory_limit, threads=threads)
        row = con.execute(
            f"""
            with raw_keys as (
              select trade_date, symbol from read_parquet({_path_list_sql(raw_paths)}, union_by_name=true)
            ),
            panel_keys as (
              select trade_date, symbol, has_bar from read_parquet({_path_list_sql(panel_paths)}, union_by_name=true)
            )
            select
              (select count(*) from raw_keys) as raw_rows,
              (select count(*) from panel_keys) as panel_rows,
              (select count(*) from panel_keys where has_bar = true) as panel_has_bar_rows,
              (select count(*) from raw_keys r left join panel_keys p using (trade_date, symbol) where p.symbol is null) as raw_missing_in_panel,
              (select count(*) from panel_keys p left join raw_keys r using (trade_date, symbol) where p.has_bar = true and r.symbol is null) as panel_has_bar_missing_in_raw
            """
        ).fetchone()
    raw_rows, panel_rows, panel_has_bar_rows, raw_missing, panel_missing = [int(item or 0) for item in row]
    return {
        "status": "ok" if raw_missing == 0 and panel_missing == 0 and raw_rows == panel_has_bar_rows else "failed",
        "raw_rows": raw_rows,
        "panel_rows": panel_rows,
        "panel_has_bar_rows": panel_has_bar_rows,
        "raw_missing_in_panel": raw_missing,
        "panel_has_bar_missing_in_raw": panel_missing,
    }


def _intraday_5m_from_1m_check(
    *,
    root: Path,
    active: Mapping[str, Any],
    memory_limit: str,
    threads: int,
    sample_limit: int,
    max_shards: int = 0,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    raw = dict(active.get("raw", {}) or {})
    one_id = str(raw.get("market_intraday_1m", "") or "")
    five_id = str(raw.get("market_intraday_5m", "") or "")
    if not one_id or not five_id:
        return {"status": "skipped", "reason": "market_intraday_1m_or_5m_missing"}
    one_path = dataset_manifest_for_id(root, one_id, "market_intraday_1m")
    five_path = dataset_manifest_for_id(root, five_id, "market_intraday_5m")
    if one_path is None or five_path is None:
        return {"status": "skipped", "reason": "market_intraday_1m_or_5m_manifest_missing"}
    one = read_dataset_manifest(one_path)
    five = read_dataset_manifest(five_path)
    pairs = _paired_intraday_shards(one, five, root=root)
    if max_shards:
        pairs = pairs[: int(max_shards)]
    if not pairs:
        return {"status": "error", "reason": "no_matching_1m_5m_shard_pairs"}
    import duckdb  # type: ignore

    checked_pairs = 0
    one_agg_rows = 0
    five_rows = 0
    missing_from_5m = 0
    extra_5m_rows = 0
    mismatched_rows = 0
    bad_examples: list[dict[str, Any]] = []
    with duckdb.connect(":memory:") as con:
        _configure_duckdb(con, memory_limit=memory_limit, threads=threads)
        for pair_index, (one_shard, five_shard) in enumerate(pairs, start=1):
            one_sql = _sql_literal(str(one_shard))
            five_sql = _sql_literal(str(five_shard))
            row = con.execute(
                f"""
                with one_raw as (
                  select
                    symbol,
                    trade_date,
                    regexp_replace(cast(bar_time as varchar), '[^0-9]', '', 'g') as bt,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    amount
                  from read_parquet({one_sql})
                ),
                one_parsed as (
                  select
                    *,
                    cast(substr(bt, 1, 2) as integer) * 60 + cast(substr(bt, 3, 2) as integer) as minute_of_day
                  from one_raw
                  where length(bt) >= 4
                ),
                one_bucketed as (
                  select
                    *,
                    case
                      when minute_of_day between 571 and 690 then 570 + cast(ceil((minute_of_day - 570) / 5.0) * 5 as integer)
                      when minute_of_day between 781 and 900 then 780 + cast(ceil((minute_of_day - 780) / 5.0) * 5 as integer)
                      else null
                    end as bucket_end
                  from one_parsed
                ),
                one_agg as (
                  select
                    symbol,
                    trade_date,
                    printf('%02d%02d00000', cast(floor(bucket_end / 60) as integer), cast(bucket_end % 60 as integer)) as bar_time,
                    arg_min(open, minute_of_day) as open,
                    max(high) as high,
                    min(low) as low,
                    arg_max(close, minute_of_day) as close,
                    sum(volume) as volume,
                    sum(amount) as amount
                  from one_bucketed
                  where bucket_end is not null
                  group by symbol, trade_date, bucket_end
                ),
                five as (
                  select symbol, trade_date, regexp_replace(cast(bar_time as varchar), '[^0-9]', '', 'g') as bar_time, open, high, low, close, volume, amount
                  from read_parquet({five_sql})
                ),
                joined as (
                  select
                    coalesce(o.symbol, f.symbol) as symbol,
                    coalesce(o.trade_date, f.trade_date) as trade_date,
                    coalesce(o.bar_time, f.bar_time) as bar_time,
                    o.symbol is null as extra_5m,
                    f.symbol is null as missing_5m,
                    { _numeric_mismatch_expr("o.open", "f.open", abs_tolerance=0.0001) } or
                    { _numeric_mismatch_expr("o.high", "f.high", abs_tolerance=0.0001) } or
                    { _numeric_mismatch_expr("o.low", "f.low", abs_tolerance=0.0001) } or
                    { _numeric_mismatch_expr("o.close", "f.close", abs_tolerance=0.0001) } or
                    { _numeric_mismatch_expr("o.volume", "f.volume", abs_tolerance=0.001) } or
                    { _numeric_mismatch_expr("o.amount", "f.amount", abs_tolerance=1.0, rel_tolerance=0.0001) } as value_mismatch
                  from one_agg o
                  full outer join five f using (symbol, trade_date, bar_time)
                )
                select
                  (select count(*) from one_agg) as one_agg_rows,
                  (select count(*) from five) as five_rows,
                  sum(case when missing_5m then 1 else 0 end) as missing_from_5m,
                  sum(case when extra_5m then 1 else 0 end) as extra_5m_rows,
                  sum(case when (not missing_5m) and (not extra_5m) and value_mismatch then 1 else 0 end) as mismatched_rows
                from joined
                """
            ).fetchone()
            checked_pairs += 1
            one_agg_rows += int(row[0] or 0)
            five_rows += int(row[1] or 0)
            missing_from_5m += int(row[2] or 0)
            extra_5m_rows += int(row[3] or 0)
            mismatched_rows += int(row[4] or 0)
            if (int(row[2] or 0) or int(row[3] or 0) or int(row[4] or 0)) and len(bad_examples) < sample_limit:
                bad_examples.extend(
                    _fetch_examples(
                        con,
                        f"""
                        with one_raw as (
                          select symbol, trade_date, regexp_replace(cast(bar_time as varchar), '[^0-9]', '', 'g') as bt, open, high, low, close, volume, amount
                          from read_parquet({one_sql})
                        ),
                        one_parsed as (
                          select *, cast(substr(bt, 1, 2) as integer) * 60 + cast(substr(bt, 3, 2) as integer) as minute_of_day
                          from one_raw
                          where length(bt) >= 4
                        ),
                        one_bucketed as (
                          select *,
                            case
                              when minute_of_day between 571 and 690 then 570 + cast(ceil((minute_of_day - 570) / 5.0) * 5 as integer)
                              when minute_of_day between 781 and 900 then 780 + cast(ceil((minute_of_day - 780) / 5.0) * 5 as integer)
                              else null
                            end as bucket_end
                          from one_parsed
                        ),
                        one_agg as (
                          select symbol, trade_date, printf('%02d%02d00000', cast(floor(bucket_end / 60) as integer), cast(bucket_end % 60 as integer)) as bar_time,
                            arg_min(open, minute_of_day) as one_open, max(high) as one_high, min(low) as one_low, arg_max(close, minute_of_day) as one_close,
                            sum(volume) as one_volume, sum(amount) as one_amount
                          from one_bucketed
                          where bucket_end is not null
                          group by symbol, trade_date, bucket_end
                        ),
                        five as (
                          select symbol, trade_date, regexp_replace(cast(bar_time as varchar), '[^0-9]', '', 'g') as bar_time,
                            open as five_open, high as five_high, low as five_low, close as five_close, volume as five_volume, amount as five_amount
                          from read_parquet({five_sql})
                        )
                        select *
                        from one_agg o
                        full outer join five f using (symbol, trade_date, bar_time)
                        where o.symbol is null or f.symbol is null
                          or { _numeric_mismatch_expr("o.one_open", "f.five_open", abs_tolerance=0.0001) }
                          or { _numeric_mismatch_expr("o.one_high", "f.five_high", abs_tolerance=0.0001) }
                          or { _numeric_mismatch_expr("o.one_low", "f.five_low", abs_tolerance=0.0001) }
                          or { _numeric_mismatch_expr("o.one_close", "f.five_close", abs_tolerance=0.0001) }
                          or { _numeric_mismatch_expr("o.one_volume", "f.five_volume", abs_tolerance=0.001) }
                          or { _numeric_mismatch_expr("o.one_amount", "f.five_amount", abs_tolerance=1.0, rel_tolerance=0.0001) }
                        limit {max(1, sample_limit - len(bad_examples))}
                        """,
                    )
                )
            _append_progress(
                progress_path,
                {
                    "event": "cross_frequency_5m_from_1m_pair_checked",
                    "pair_index": pair_index,
                    "pair_count": len(pairs),
                    "one_shard": str(one_shard),
                    "five_shard": str(five_shard),
                    "missing_from_5m_so_far": missing_from_5m,
                    "extra_5m_rows_so_far": extra_5m_rows,
                    "mismatched_rows_so_far": mismatched_rows,
                    "timestamp": utc_now(),
                },
            )
    failed = missing_from_5m or extra_5m_rows or mismatched_rows
    return {
        "status": "failed" if failed else "ok",
        "method": "paired_shard_exact_aggregation",
        "scan_limited": bool(max_shards),
        "paired_shard_count": len(pairs),
        "checked_pair_count": checked_pairs,
        "one_minute_aggregated_rows": one_agg_rows,
        "five_minute_rows": five_rows,
        "missing_from_5m": missing_from_5m,
        "extra_5m_rows": extra_5m_rows,
        "mismatched_rows": mismatched_rows,
        "examples": bad_examples[:sample_limit],
    }


def _intraday_vs_daily_check(
    *,
    root: Path,
    active: Mapping[str, Any],
    memory_limit: str,
    threads: int,
    sample_limit: int,
    max_shards: int = 0,
) -> dict[str, Any]:
    raw = dict(active.get("raw", {}) or {})
    daily_id = str(raw.get("market_daily_raw", "") or "")
    five_id = str(raw.get("market_intraday_5m", "") or "")
    if not daily_id or not five_id:
        return {"status": "skipped", "reason": "market_daily_raw_or_market_intraday_5m_missing"}
    daily_path = dataset_manifest_for_id(root, daily_id, "market_daily_raw")
    five_path = dataset_manifest_for_id(root, five_id, "market_intraday_5m")
    if daily_path is None or five_path is None:
        return {"status": "skipped", "reason": "daily_or_5m_manifest_missing"}
    daily = read_dataset_manifest(daily_path)
    five = read_dataset_manifest(five_path)
    daily_paths = _existing_shard_paths(daily, root=root, max_shards=0)
    five_paths = _existing_shard_paths(five, root=root, max_shards=max_shards)
    if not daily_paths or not five_paths:
        return {"status": "error", "reason": "daily_or_5m_has_no_readable_shards"}
    import duckdb  # type: ignore

    scan_limited = bool(max_shards)
    daily_missing_expr = "false" if scan_limited else "i.symbol is null"
    example_daily_missing_clause = "" if scan_limited else " or i.symbol is null"
    with duckdb.connect(":memory:") as con:
        _configure_duckdb(con, memory_limit=memory_limit, threads=threads)
        paths_sql = _path_list_sql(five_paths)
        daily_sql = _path_list_sql(daily_paths)
        row = con.execute(
            f"""
            with intra_raw as (
              select
                symbol,
                trade_date,
                regexp_replace(cast(bar_time as varchar), '[^0-9]', '', 'g') as bt,
                open,
                high,
                low,
                close,
                volume,
                amount
              from read_parquet({paths_sql}, union_by_name=true)
            ),
            intra_parsed as (
              select
                *,
                cast(substr(bt, 1, 2) as integer) * 60 + cast(substr(bt, 3, 2) as integer) as minute_of_day
              from intra_raw
              where length(bt) >= 4
            ),
            intra as (
              select
                symbol,
                trade_date,
                arg_min(open, minute_of_day) as open,
                max(high) as high,
                min(low) as low,
                arg_max(close, minute_of_day) as close,
                sum(volume) as volume,
                sum(amount) as amount,
                count(*) as bars
              from intra_parsed
              group by symbol, trade_date
            ),
            daily as (
              select symbol, trade_date, open, high, low, close, volume, amount
              from read_parquet({daily_sql}, union_by_name=true)
            ),
            joined as (
              select
                coalesce(d.symbol, i.symbol) as symbol,
                coalesce(d.trade_date, i.trade_date) as trade_date,
                d.symbol is null as intraday_missing_in_daily,
                {daily_missing_expr} as daily_missing_in_intraday,
                d.symbol is not null and i.symbol is not null as both_present,
                abs(cast(d.open as double) - cast(i.open as double)) as open_diff,
                abs(cast(d.high as double) - cast(i.high as double)) as high_diff,
                abs(cast(d.low as double) - cast(i.low as double)) as low_diff,
                abs(cast(d.close as double) - cast(i.close as double)) as close_diff,
                abs(cast(d.volume as double) - cast(i.volume as double)) as volume_diff,
                abs(cast(d.amount as double) - cast(i.amount as double)) as amount_diff,
                case
                  when greatest(abs(cast(d.amount as double)), abs(cast(i.amount as double))) > 0
                  then abs(cast(d.amount as double) - cast(i.amount as double)) / greatest(abs(cast(d.amount as double)), abs(cast(i.amount as double)))
                  else 0
                end as amount_rel_diff
              from daily d
              full outer join intra i using (symbol, trade_date)
            ),
            classified as (
              select
                *,
                case
                  when both_present and (open_diff is null or high_diff is null or low_diff is null or close_diff is null) then 1000000000.0
                  when both_present then greatest(open_diff, high_diff, low_diff, close_diff)
                  else null
                end as max_price_diff
              from joined
            )
            select
              (select count(*) from daily) as daily_rows,
              (select count(*) from intra) as intraday_symbol_days,
              sum(case when daily_missing_in_intraday then 1 else 0 end) as daily_missing_in_intraday,
              sum(case when intraday_missing_in_daily then 1 else 0 end) as intraday_missing_in_daily,
              sum(case when both_present and max_price_diff <= 0.001 then 1 else 0 end) as price_exact_rows,
              sum(case when both_present and max_price_diff > 0.001 then 1 else 0 end) as price_any_diff_rows,
              sum(case when both_present and max_price_diff > 0.001 and max_price_diff <= 0.0101 then 1 else 0 end) as price_diff_le_1tick_rows,
              sum(case when both_present and max_price_diff > 0.0101 then 1 else 0 end) as price_diff_gt_1tick_rows,
              sum(case when both_present and max_price_diff > 0.0501 then 1 else 0 end) as price_diff_gt_5tick_rows,
              sum(case when both_present and open_diff > 0.0101 then 1 else 0 end) as open_diff_gt_1tick_rows,
              sum(case when both_present and high_diff > 0.0101 then 1 else 0 end) as high_diff_gt_1tick_rows,
              sum(case when both_present and low_diff > 0.0101 then 1 else 0 end) as low_diff_gt_1tick_rows,
              sum(case when both_present and close_diff > 0.0101 then 1 else 0 end) as close_diff_gt_1tick_rows,
              sum(case when both_present and volume_diff > 1.0 then 1 else 0 end) as volume_any_diff_rows,
              sum(case when both_present and volume_diff > 100.0 then 1 else 0 end) as volume_diff_gt_100_rows,
              sum(case when both_present and amount_diff > 10.0 then 1 else 0 end) as amount_any_diff_rows,
              sum(case when both_present and amount_rel_diff > 0.001 then 1 else 0 end) as amount_rel_diff_gt_10bp_rows,
              sum(case when both_present and amount_rel_diff > 0.01 then 1 else 0 end) as amount_rel_diff_gt_1pct_rows
            from classified
            """
        ).fetchone()
        examples = _fetch_examples(
            con,
            f"""
            with intra_raw as (
              select symbol, trade_date, regexp_replace(cast(bar_time as varchar), '[^0-9]', '', 'g') as bt, open, high, low, close, volume, amount
              from read_parquet({paths_sql}, union_by_name=true)
            ),
            intra_parsed as (
              select *, cast(substr(bt, 1, 2) as integer) * 60 + cast(substr(bt, 3, 2) as integer) as minute_of_day
              from intra_raw
              where length(bt) >= 4
            ),
            intra as (
              select symbol, trade_date, arg_min(open, minute_of_day) as intraday_open, max(high) as intraday_high, min(low) as intraday_low,
                arg_max(close, minute_of_day) as intraday_close, sum(volume) as intraday_volume, sum(amount) as intraday_amount, count(*) as intraday_bars
              from intra_parsed
              group by symbol, trade_date
            ),
            daily as (
              select symbol, trade_date, open as daily_open, high as daily_high, low as daily_low, close as daily_close, volume as daily_volume, amount as daily_amount
              from read_parquet({daily_sql}, union_by_name=true)
            )
            select
              *,
              case
                when d.symbol is not null and i.symbol is not null then abs(cast(d.daily_open as double) - cast(i.intraday_open as double))
                else null
              end as open_diff,
              case
                when d.symbol is not null and i.symbol is not null then abs(cast(d.daily_high as double) - cast(i.intraday_high as double))
                else null
              end as high_diff,
              case
                when d.symbol is not null and i.symbol is not null then abs(cast(d.daily_low as double) - cast(i.intraday_low as double))
                else null
              end as low_diff,
              case
                when d.symbol is not null and i.symbol is not null then abs(cast(d.daily_close as double) - cast(i.intraday_close as double))
                else null
              end as close_diff,
              case
                when d.symbol is not null and i.symbol is not null then abs(cast(d.daily_volume as double) - cast(i.intraday_volume as double))
                else null
              end as volume_diff,
              case
                when d.symbol is not null and i.symbol is not null and greatest(abs(cast(d.daily_amount as double)), abs(cast(i.intraday_amount as double))) > 0
                then abs(cast(d.daily_amount as double) - cast(i.intraday_amount as double)) / greatest(abs(cast(d.daily_amount as double)), abs(cast(i.intraday_amount as double)))
                else null
              end as amount_rel_diff
            from daily d
            full outer join intra i using (symbol, trade_date)
            where d.symbol is null{example_daily_missing_clause}
              or (
                d.symbol is not null and i.symbol is not null and (
                  greatest(
                    abs(cast(d.daily_open as double) - cast(i.intraday_open as double)),
                    abs(cast(d.daily_high as double) - cast(i.intraday_high as double)),
                    abs(cast(d.daily_low as double) - cast(i.intraday_low as double)),
                    abs(cast(d.daily_close as double) - cast(i.intraday_close as double))
                  ) > 0.0501
                  or abs(cast(d.daily_open as double) - cast(i.intraday_open as double)) > 0.0101
                  or abs(cast(d.daily_close as double) - cast(i.intraday_close as double)) > 0.0101
                  or abs(cast(d.daily_volume as double) - cast(i.intraday_volume as double)) > 100.0
                  or (
                    greatest(abs(cast(d.daily_amount as double)), abs(cast(i.intraday_amount as double))) > 0
                    and abs(cast(d.daily_amount as double) - cast(i.intraday_amount as double)) / greatest(abs(cast(d.daily_amount as double)), abs(cast(i.intraday_amount as double))) > 0.01
                  )
                )
              )
            order by greatest(
              abs(cast(daily_open as double) - cast(intraday_open as double)),
              abs(cast(daily_high as double) - cast(intraday_high as double)),
              abs(cast(daily_low as double) - cast(intraday_low as double)),
              abs(cast(daily_close as double) - cast(intraday_close as double))
            ) desc nulls last
            limit {sample_limit}
            """,
        )
    (
        daily_rows,
        intraday_symbol_days,
        daily_missing,
        intraday_missing,
        price_exact,
        price_any_diff,
        price_le_1tick,
        price_gt_1tick,
        price_gt_5tick,
        open_gt_1tick,
        high_gt_1tick,
        low_gt_1tick,
        close_gt_1tick,
        volume_any_diff,
        volume_gt_100,
        amount_any_diff,
        amount_rel_gt_10bp,
        amount_rel_gt_1pct,
    ) = [int(item or 0) for item in row]
    coverage_failed = bool(daily_missing or intraday_missing)
    needs_review = bool(price_gt_1tick or volume_gt_100 or amount_rel_gt_10bp)
    status = "failed" if coverage_failed else "needs_review" if needs_review else "ok"
    return {
        "status": status,
        "method": "aggregate_5m_to_daily_classified",
        "scan_limited": scan_limited,
        "scanned_5m_shards": len(five_paths),
        "daily_rows": daily_rows,
        "intraday_symbol_days": intraday_symbol_days,
        "daily_missing_in_intraday": daily_missing,
        "intraday_missing_in_daily": intraday_missing,
        "price_exact_rows": price_exact,
        "price_any_diff_rows": price_any_diff,
        "price_diff_le_1tick_rows": price_le_1tick,
        "price_diff_gt_1tick_rows": price_gt_1tick,
        "price_diff_gt_5tick_rows": price_gt_5tick,
        "open_diff_gt_1tick_rows": open_gt_1tick,
        "high_diff_gt_1tick_rows": high_gt_1tick,
        "low_diff_gt_1tick_rows": low_gt_1tick,
        "close_diff_gt_1tick_rows": close_gt_1tick,
        "volume_any_diff_rows": volume_any_diff,
        "volume_diff_gt_100_rows": volume_gt_100,
        "amount_any_diff_rows": amount_any_diff,
        "amount_rel_diff_gt_10bp_rows": amount_rel_gt_10bp,
        "amount_rel_diff_gt_1pct_rows": amount_rel_gt_1pct,
        "price_mismatch_rows": price_any_diff,
        "volume_mismatch_rows": volume_any_diff,
        "amount_mismatch_rows": amount_any_diff,
        "examples": examples[:sample_limit],
    }


def _classify_intraday_zero_price_rows(
    *,
    root: Path,
    active: Mapping[str, Any],
    paths: list[Path],
    memory_limit: str,
    threads: int,
) -> dict[str, Any]:
    raw = dict(active.get("raw", {}) or {})
    daily_id = str(raw.get("market_daily_raw", "") or "")
    status_id = str(raw.get("security_status", "") or "")
    if not daily_id or not status_id:
        return {"status": "skipped", "reason": "market_daily_raw_or_security_status_missing"}
    daily_path = dataset_manifest_for_id(root, daily_id, "market_daily_raw")
    status_path = dataset_manifest_for_id(root, status_id, "security_status")
    if daily_path is None or status_path is None:
        return {"status": "skipped", "reason": "daily_or_status_manifest_missing"}
    daily = read_dataset_manifest(daily_path)
    status = read_dataset_manifest(status_path)
    daily_paths = _existing_shard_paths(daily, root=root, max_shards=0)
    status_paths = _existing_shard_paths(status, root=root, max_shards=0)
    if not daily_paths or not status_paths:
        return {"status": "skipped", "reason": "daily_or_status_shards_missing"}
    import duckdb  # type: ignore

    with duckdb.connect(":memory:") as con:
        _configure_duckdb(con, memory_limit=memory_limit, threads=threads)
        rows = con.execute(
            f"""
            with bad as (
              select symbol, trade_date, count(*) as bad_rows
              from read_parquet({_path_list_sql(paths)}, union_by_name=true)
              where open <= 0 or high <= 0 or low <= 0 or close <= 0
              group by symbol, trade_date
            ),
            daily as (
              select symbol, trade_date, open, volume
              from read_parquet({_path_list_sql(daily_paths)}, union_by_name=true)
            ),
            st as (
              select symbol, trade_date, is_suspended, is_st, is_delisted
              from read_parquet({_path_list_sql(status_paths)}, union_by_name=true)
            ),
            classified as (
              select
                case
                  when coalesce(st.is_suspended, false) then 'suspended_zero'
                  when d.symbol is not null and d.open > 0 and d.volume > 0 then 'traded_day_partial_zero'
                  when d.symbol is null then 'no_daily_row_not_suspended'
                  else 'other'
                end as class,
                b.bad_rows
              from bad b
              left join daily d using(symbol, trade_date)
              left join st using(symbol, trade_date)
            )
            select
              class,
              count(*) as symbol_days,
              sum(bad_rows) as bad_rows
            from classified
            group by class
            """
        ).fetchdf().to_dict(orient="records")
    by_class = {str(row["class"]): {"symbol_days": int(row["symbol_days"] or 0), "bad_rows": int(row["bad_rows"] or 0)} for row in rows}
    return {
        "status": "classified",
        "by_class": by_class,
        "suspended_zero_symbol_days": int(by_class.get("suspended_zero", {}).get("symbol_days", 0)),
        "suspended_zero_rows": int(by_class.get("suspended_zero", {}).get("bad_rows", 0)),
        "traded_day_partial_zero_symbol_days": int(by_class.get("traded_day_partial_zero", {}).get("symbol_days", 0)),
        "traded_day_partial_zero_rows": int(by_class.get("traded_day_partial_zero", {}).get("bad_rows", 0)),
        "no_daily_row_not_suspended_symbol_days": int(by_class.get("no_daily_row_not_suspended", {}).get("symbol_days", 0)),
        "no_daily_row_not_suspended_rows": int(by_class.get("no_daily_row_not_suspended", {}).get("bad_rows", 0)),
    }


def _pit_coverage_check(*, root: Path, active: Mapping[str, Any], memory_limit: str, threads: int, sample_limit: int) -> dict[str, Any]:
    raw = dict(active.get("raw", {}) or {})
    calendar_id = str(raw.get("trading_calendar", "") or "")
    universe_id = str(raw.get("universe_snapshot", "") or "")
    status_id = str(raw.get("security_status", "") or "")
    if not calendar_id or not universe_id or not status_id:
        return {"status": "skipped", "reason": "calendar/universe/status missing"}
    calendar = read_dataset_manifest(dataset_manifest_for_id(root, calendar_id, "trading_calendar") or "")
    universe = read_dataset_manifest(dataset_manifest_for_id(root, universe_id, "universe_snapshot") or "")
    status = read_dataset_manifest(dataset_manifest_for_id(root, status_id, "security_status") or "")
    import duckdb  # type: ignore

    with duckdb.connect(":memory:") as con:
        _configure_duckdb(con, memory_limit=memory_limit, threads=threads)
        row = con.execute(
            f"""
            with open_dates as (
              select distinct trade_date
              from read_parquet({_path_list_sql(_existing_shard_paths(calendar, root=root, max_shards=0))}, union_by_name=true)
              where is_open = 1 or is_open = true
            ),
            universe_dates as (
              select distinct trade_date
              from read_parquet({_path_list_sql(_existing_shard_paths(universe, root=root, max_shards=0))}, union_by_name=true)
            ),
            status_dates as (
              select distinct trade_date
              from read_parquet({_path_list_sql(_existing_shard_paths(status, root=root, max_shards=0))}, union_by_name=true)
            )
            select
              (select count(*) from open_dates) as open_date_count,
              (select count(*) from open_dates o left join universe_dates u using (trade_date) where u.trade_date is null) as universe_missing_open_dates,
              (select count(*) from open_dates o left join status_dates s using (trade_date) where s.trade_date is null) as status_missing_open_dates
            """
        ).fetchone()
        universe_missing = _fetch_examples(
            con,
            f"""
            with open_dates as (
              select distinct trade_date
              from read_parquet({_path_list_sql(_existing_shard_paths(calendar, root=root, max_shards=0))}, union_by_name=true)
              where is_open = 1 or is_open = true
            ),
            universe_dates as (
              select distinct trade_date
              from read_parquet({_path_list_sql(_existing_shard_paths(universe, root=root, max_shards=0))}, union_by_name=true)
            )
            select o.trade_date
            from open_dates o left join universe_dates u using (trade_date)
            where u.trade_date is null
            order by o.trade_date
            limit {sample_limit}
            """,
        )
        status_missing = _fetch_examples(
            con,
            f"""
            with open_dates as (
              select distinct trade_date
              from read_parquet({_path_list_sql(_existing_shard_paths(calendar, root=root, max_shards=0))}, union_by_name=true)
              where is_open = 1 or is_open = true
            ),
            status_dates as (
              select distinct trade_date
              from read_parquet({_path_list_sql(_existing_shard_paths(status, root=root, max_shards=0))}, union_by_name=true)
            )
            select o.trade_date
            from open_dates o left join status_dates s using (trade_date)
            where s.trade_date is null
            order by o.trade_date
            limit {sample_limit}
            """,
        )
    return {
        "status": "ok" if int(row[1] or 0) == 0 and int(row[2] or 0) == 0 else "failed",
        "open_date_count": int(row[0] or 0),
        "universe_missing_open_dates": int(row[1] or 0),
        "security_status_missing_open_dates": int(row[2] or 0),
        "universe_missing_examples": universe_missing,
        "security_status_missing_examples": status_missing,
    }


def _active_scope_check(*, root: Path, active: Mapping[str, Any], memory_limit: str, threads: int) -> dict[str, Any]:
    raw = dict(active.get("raw", {}) or {})
    universe_id = str(raw.get("universe_snapshot", "") or "")
    status_id = str(raw.get("security_status", "") or "")
    active_date = str(active.get("active_as_of_date", "") or "")
    if not universe_id or not status_id or not active_date:
        return {"status": "skipped", "reason": "universe/status/as_of missing"}
    universe = read_dataset_manifest(dataset_manifest_for_id(root, universe_id, "universe_snapshot") or "")
    status = read_dataset_manifest(dataset_manifest_for_id(root, status_id, "security_status") or "")
    import duckdb  # type: ignore

    with duckdb.connect(":memory:") as con:
        _configure_duckdb(con, memory_limit=memory_limit, threads=threads)
        row = con.execute(
            f"""
            with universe as (
              select symbol
              from read_parquet({_path_list_sql(_existing_shard_paths(universe, root=root, max_shards=0))}, union_by_name=true)
              where cast(trade_date as varchar) = {_sql_literal(active_date)}
            ),
            status as (
              select symbol, is_st, is_delisted
              from read_parquet({_path_list_sql(_existing_shard_paths(status, root=root, max_shards=0))}, union_by_name=true)
              where cast(trade_date as varchar) = {_sql_literal(active_date)}
            ),
            joined as (
              select universe.symbol, coalesce(status.is_st, false) as is_st, coalesce(status.is_delisted, false) as is_delisted
              from universe left join status using (symbol)
            )
            select
              count(*) as active_symbol_count,
              sum(case when regexp_matches(symbol, '^(600|601|603|605)[0-9]{{3}}\\.SH$') then 1 else 0 end) as sh_main,
              sum(case when regexp_matches(symbol, '^(000|001|002|003)[0-9]{{3}}\\.SZ$') then 1 else 0 end) as sz_main,
              sum(case when regexp_matches(symbol, '^(300|301)[0-9]{{3}}\\.SZ$') or regexp_matches(symbol, '^(688|689)[0-9]{{3}}\\.SH$') or regexp_matches(symbol, '^[48][0-9]{{5}}\\.(BJ|SZ|SH)$') then 1 else 0 end) as excluded_board_prefix,
              sum(case when is_st then 1 else 0 end) as st_count,
              sum(case when is_delisted then 1 else 0 end) as delisted_count
            from joined
            """
        ).fetchone()
    excluded = int(row[3] or 0)
    st_count = int(row[4] or 0)
    delisted_count = int(row[5] or 0)
    return {
        "status": "ok" if excluded == 0 and st_count == 0 and delisted_count == 0 else "failed",
        "active_as_of_date": active_date,
        "active_symbol_count": int(row[0] or 0),
        "sh_main": int(row[1] or 0),
        "sz_main": int(row[2] or 0),
        "excluded_board_prefix": excluded,
        "st_count": st_count,
        "delisted_count": delisted_count,
    }


def _contract_checks(*, domain: str, manifest: DatasetManifest, columns: list[str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    required = PRIMARY_KEYS.get(domain, manifest.primary_key)
    missing_pk = [col for col in required if col not in columns]
    if missing_pk:
        findings.append(_finding("critical", "schema", domain, "primary_key_columns_missing", {"columns": missing_pk}, "Fix schema before using this dataset."))
    if domain == "valuation":
        required_cols = {"symbol", "trade_date", "total_mv", "circ_mv", "pe", "pb", "turnover_rate", "source"}
        missing = sorted(required_cols.difference(columns))
        if missing:
            findings.append(_finding("critical", "schema", domain, "valuation_required_columns_missing", {"columns": missing}, "Re-run qdp clean valuation."))
    if domain == "market_intraday_1m" and manifest.contract_version != "mootdx_1m_240_v1":
        findings.append(_finding("high", "contract", domain, "wrong_intraday_1m_contract", {"contract_version": manifest.contract_version}, "Normalize 1m to mootdx_1m_240_v1."))
    if domain == "market_intraday_5m" and manifest.contract_version != "mootdx_5m_48_v1":
        findings.append(_finding("high", "contract", domain, "wrong_intraday_5m_contract", {"contract_version": manifest.contract_version}, "Rebuild 5m from 1m 240."))
    return findings


def _audit_selected_active(*, root: Path, active: Mapping[str, Any], selected_domains: set[str], verify_footers: bool) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    dataset_reports: list[dict[str, Any]] = []
    for section, domain, dataset_id in _active_dataset_refs(dict(active)):
        if selected_domains and domain not in selected_domains:
            continue
        manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
        if manifest_path is None:
            errors.append(f"dataset_manifest_missing:{section}.{domain}:{dataset_id}")
            continue
        manifest = read_dataset_manifest(manifest_path)
        missing_shards: list[str] = []
        footer_errors: list[str] = []
        footer_rows = 0
        for shard in manifest.shards:
            shard_path = resolve_manifest_path(shard.path, root=root)
            if not shard_path.exists():
                missing_shards.append(str(shard_path))
                continue
            if not verify_footers:
                continue
            try:
                rows = _parquet_row_count(shard_path)
                footer_rows += int(rows)
                if shard.row_count > 0 and rows > 0 and int(rows) != int(shard.row_count):
                    footer_errors.append(f"row_count_mismatch:{shard.path}:manifest={shard.row_count}:footer={rows}")
            except Exception as exc:
                footer_errors.append(f"footer_unreadable:{shard.path}:{exc}")
        if missing_shards:
            errors.append(f"missing_shards:{section}.{domain}:{len(missing_shards)}")
        if footer_errors:
            errors.extend(footer_errors[:20])
            if len(footer_errors) > 20:
                warnings.append(f"footer_errors_truncated:{section}.{domain}:{len(footer_errors)}")
        if verify_footers and manifest.row_count and footer_rows and int(manifest.row_count) != int(footer_rows):
            errors.append(f"dataset_row_count_mismatch:{section}.{domain}:manifest={manifest.row_count}:footer={footer_rows}")
        contract_errors, contract_warnings = _manifest_contract_findings(domain, manifest.to_dict())
        errors.extend(contract_errors)
        warnings.extend(contract_warnings)
        dataset_reports.append(
            {
                "section": section,
                "domain": domain,
                "dataset_id": dataset_id,
                "row_count": manifest.row_count,
                "footer_row_count": footer_rows,
                "shard_count": len(manifest.shards),
                "missing_shard_count": len(missing_shards),
                "footer_error_count": len(footer_errors),
            }
        )
    return {
        "status": "ok" if not errors else "error",
        "dataset_count": len(dataset_reports),
        "datasets": dataset_reports,
        "warnings": warnings,
        "errors": errors,
    }


def _existing_shard_paths(manifest: DatasetManifest, *, root: Path, max_shards: int) -> list[Path]:
    paths: list[Path] = []
    for shard in manifest.shards:
        path = resolve_manifest_path(shard.path, root=root)
        if path.exists():
            paths.append(path)
            if max_shards and len(paths) >= max_shards:
                break
    return paths


def _paired_intraday_shards(one: DatasetManifest, five: DatasetManifest, *, root: Path) -> list[tuple[Path, Path]]:
    five_by_name: dict[str, Path] = {}
    five_by_index: dict[str, Path] = {}
    for shard in five.shards:
        path = resolve_manifest_path(shard.path, root=root)
        if not path.exists():
            continue
        name = path.name
        five_by_name[name] = path
        index = _part_index(name)
        if index:
            five_by_index[index] = path
    pairs: list[tuple[Path, Path]] = []
    used: set[Path] = set()
    for shard in one.shards:
        one_path = resolve_manifest_path(shard.path, root=root)
        if not one_path.exists():
            continue
        expected_name = one_path.name.replace("market_intraday_1m", "market_intraday_5m")
        five_path = five_by_name.get(expected_name)
        if five_path is None:
            index = _part_index(one_path.name)
            five_path = five_by_index.get(index) if index else None
        if five_path is None or five_path in used:
            continue
        pairs.append((one_path, five_path))
        used.add(five_path)
    return pairs


def _part_index(name: str) -> str:
    text = str(name)
    if not text.startswith("part_"):
        return ""
    pieces = text.split("_", 2)
    return pieces[1] if len(pieces) >= 2 and pieces[1].isdigit() else ""


def _manifest_byte_count(manifest: DatasetManifest, *, root: Path) -> int:
    total = 0
    for shard in manifest.shards:
        path = resolve_manifest_path(shard.path, root=root)
        try:
            total += int(path.stat().st_size)
        except OSError:
            total += int(shard.file_size or 0)
    return total


def _read_columns(path: Path) -> list[str]:
    import duckdb  # type: ignore

    with duckdb.connect(":memory:") as con:
        rows = con.execute("describe select * from read_parquet(?)", [str(path)]).fetchall()
    return [str(row[0]) for row in rows]


def _primary_key_for(manifest: DatasetManifest) -> list[str]:
    return list(manifest.primary_key or PRIMARY_KEYS.get(manifest.domain, []))


def _has_columns(columns: Iterable[str], required: Iterable[str]) -> bool:
    existing = set(columns)
    return all(col in existing for col in required)


def _configure_duckdb(con: Any, *, memory_limit: str, threads: int) -> None:
    safe_memory = str(memory_limit or "4GB").replace("'", "")
    con.execute(f"set memory_limit='{safe_memory}'")
    con.execute(f"set threads={max(1, int(threads or 1))}")
    con.execute("set preserve_insertion_order=false")


def _q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _select_cols(cols: Iterable[str]) -> str:
    return ", ".join(_q(col) for col in cols)


def _key_expr(cols: Iterable[str]) -> str:
    parts = [f"coalesce(cast({_q(col)} as varchar), '<NULL>')" for col in cols]
    if not parts:
        return "''"
    return "concat_ws(chr(31), " + ", ".join(parts) + ")"


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''").replace("\\", "/") + "'"


def _numeric_mismatch_expr(left: str, right: str, *, abs_tolerance: float, rel_tolerance: float = 0.0) -> str:
    abs_tol = float(abs_tolerance)
    rel_tol = float(rel_tolerance)
    return (
        f"(({left} is null) <> ({right} is null) or "
        f"(not ({left} is null) and not ({right} is null) and "
        f"abs(cast({left} as double) - cast({right} as double)) > "
        f"greatest({abs_tol}, {rel_tol} * greatest(abs(cast({left} as double)), abs(cast({right} as double))))))"
    )


def _path_list_sql(paths: list[Path]) -> str:
    return "[" + ", ".join(_sql_literal(str(path)) for path in paths) + "]"


def _batches(items: list[Path], size: int) -> Iterable[list[Path]]:
    step = max(1, int(size or 1))
    for index in range(0, len(items), step):
        yield items[index : index + step]


def _fetch_examples(con: Any, sql: str) -> list[dict[str, Any]]:
    rows = con.execute(sql).fetchdf()
    return [dict(item) for item in rows.to_dict(orient="records")]


def _append_progress(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(dict(payload)), ensure_ascii=False) + "\n")


def _finding(severity: str, category: str, domain: str, code: str, evidence: Mapping[str, Any], recommendation: str) -> dict[str, Any]:
    return {
        "severity": severity,
        "category": category,
        "domain": domain,
        "code": code,
        "evidence": dict(evidence),
        "recommendation": recommendation,
    }


def _format_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# QDP v2 Database Audit",
        "",
        f"- Status: `{payload.get('status', '')}`",
        f"- Active as of: `{payload.get('active_as_of_date', '')}`",
        f"- Deep audit: `{payload.get('deep', False)}`",
        f"- Dataset count: `{payload.get('dataset_count', 0)}`",
        f"- DuckDB catalog required: `{payload.get('duckdb_catalog_required', False)}`",
        "",
        "## Datasets",
        "",
        "| Domain | Dataset | Rows | Shards | Range | Contract | Bytes |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for item in list(payload.get("datasets", []) or []):
        lines.append(
            "| {domain} | `{dataset}` | {rows} | {shards} | {start}..{end} | `{contract}` | {bytes} |".format(
                domain=item.get("domain", ""),
                dataset=item.get("dataset_id", ""),
                rows=item.get("row_count", 0),
                shards=item.get("shard_count", 0),
                start=item.get("start_date", ""),
                end=item.get("end_date", ""),
                contract=item.get("contract_version", ""),
                bytes=item.get("byte_count", 0),
            )
        )
    lines.extend(["", "## Findings", ""])
    findings = list(payload.get("findings", []) or [])
    if not findings:
        lines.append("No findings.")
    for item in findings:
        lines.append(f"- **{item.get('severity', '')}** `{item.get('code', '')}` {item.get('domain', '')}: {json.dumps(json_safe(item.get('evidence', {})), ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp audit database", description="Run qdp_v2 manifest-first database quality audit.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--runtime", default="balanced", choices=("safe", "balanced", "fast"))
    parser.add_argument("--duckdb-memory-limit", default="")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--domains", default="", help="Comma-separated active domains to scan.")
    parser.add_argument("--deep", action="store_true", help="Run row-level uniqueness, validity, coverage, and consistency checks.")
    parser.add_argument("--max-shards", type=int, default=0, help="Limit scanned shards per dataset for smoke tests.")
    parser.add_argument("--batch-shards", type=int, default=64, help="Number of parquet shards to scan per DuckDB batch.")
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--full-global-uniqueness", action="store_true", help="Run expensive cross-shard global primary-key uniqueness checks for all datasets.")
    parser.add_argument("--global-uniqueness-max-rows", type=int, default=50_000_000)
    parser.add_argument("--global-uniqueness-max-shards", type=int, default=2_000)
    parser.add_argument("--skip-cross", action="store_true", help="Skip cross-dataset consistency checks.")
    parser.add_argument("--skip-cross-frequency", action="store_true", help="Skip 1m/5m/daily consistency checks inside cross-dataset checks.")
    parser.add_argument("--progress-path", default="", help="Optional JSONL progress path for long audits.")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    domains = [item.strip() for item in str(args.domains or "").split(",") if item.strip()]
    payload = audit_database(
        workspace_root=str(args.workspace_root or "") or None,
        deep=bool(args.deep),
        runtime=str(args.runtime or "balanced"),
        duckdb_memory_limit=str(args.duckdb_memory_limit or ""),
        threads=int(args.threads or 0),
        domains=domains,
        max_shards=int(args.max_shards or 0),
        batch_shards=int(args.batch_shards or 64),
        sample_limit=int(args.sample_limit or 20),
        full_global_uniqueness=bool(args.full_global_uniqueness),
        global_uniqueness_max_rows=int(args.global_uniqueness_max_rows or 0),
        global_uniqueness_max_shards=int(args.global_uniqueness_max_shards or 0),
        skip_cross=bool(args.skip_cross),
        skip_cross_frequency=bool(args.skip_cross_frequency),
        progress_path=str(args.progress_path or "") or None,
        write=not bool(args.no_write),
    )
    if bool(args.json):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(_format(payload))
    return 0 if str(payload.get("status", "")) in {"ok", "needs_attention"} else 2


def _format(payload: Mapping[str, Any]) -> str:
    lines = [
        f"status: {payload.get('status', '')}",
        f"active_as_of_date: {payload.get('active_as_of_date', '')}",
        f"dataset_count: {payload.get('dataset_count', 0)}",
        f"finding_count: {payload.get('finding_count', 0)}",
    ]
    if payload.get("audit_path"):
        lines.append(f"audit_path: {payload.get('audit_path')}")
    if payload.get("markdown_path"):
        lines.append(f"markdown_path: {payload.get('markdown_path')}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
