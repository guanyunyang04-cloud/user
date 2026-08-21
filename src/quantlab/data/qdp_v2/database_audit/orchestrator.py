"""Database audit orchestrator checks."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from quantlab.data.qdp_v2.audit import audit_active
from quantlab.data.qdp_v2.auxiliary_update import AUXILIARY_DOMAINS
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    atomic_write_json,
    qdp_v2_root,
    read_active_manifest,
    resolve_manifest_path,
    utc_now,
)
from quantlab.data.qdp_v2.pit_history import (
    LIFECYCLE_NORMALIZE_DOMAINS,
    PitHistoryError,
    audit_symbol_lifecycle_effectivity,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .auxiliary import (
    _auxiliary_semantics_check,
)
from .common import (
    _audit_temp_directory,
    _finding,
)
from .config import (
    REQUIRED_COLUMNS,
    REQUIRED_DOMAINS,
)
from .identity import (
    _active_manifest,
    _identity_history_consistency_check,
    _long_suspension_check,
    _reopen_discontinuity_check,
)
from .latest import (
    audit_latest_keys,
)
from .market import (
    _daily_intraday_consistency_check,
    _factor_semantic_check,
    _status_daily_consistency_check,
)
from .physical import (
    _bar_day_check,
    _factor_check,
    _ohlc_check,
    _ordered_intraday_primary_key_check,
    _primary_key_check,
    _symbol_history_check,
)


def _active_manifest_findings(
    active_audit: Mapping[str, Any],
    datasets: Mapping[str, str],
) -> tuple[list[str], list[dict[str, Any]]]:
    missing_required = sorted(REQUIRED_DOMAINS.difference(datasets))
    findings = [_finding("high", "coverage", domain, "required_domain_missing", {}) for domain in missing_required]
    findings.extend(
        _finding(
            "high",
            "manifest",
            "active",
            "active_manifest_or_footer_error",
            {"error": str(error)},
        )
        for error in list(active_audit.get("errors", []) or [])
    )
    return missing_required, findings


def _audit_active_datasets(
    con: Any,
    *,
    root: Path,
    datasets: Mapping[str, str],
    deep: bool,
    max_shards: int,
    sample_limit: int,
    skip_global_primary_key: bool,
) -> tuple[list[dict[str, Any]], dict[str, DatasetManifest], list[dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    manifests: dict[str, DatasetManifest] = {}
    findings: list[dict[str, Any]] = []
    for domain, dataset_id in sorted(datasets.items()):
        manifest = _active_manifest(root, dataset_id, domain)
        manifests[domain] = manifest
        report, domain_findings = _audit_dataset(
            con,
            root=root,
            manifest=manifest,
            deep=deep,
            max_shards=max_shards,
            sample_limit=sample_limit,
            skip_primary_key=skip_global_primary_key,
        )
        reports.append(report)
        findings.extend(domain_findings)
    return reports, manifests, findings


def _market_cross_checks(
    con: Any,
    *,
    root: Path,
    active: Mapping[str, Any],
    manifests: Mapping[str, DatasetManifest],
    sample_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checks: dict[str, Any] = {}
    findings: list[dict[str, Any]] = []
    if {"market_daily_raw", "market_intraday_5m"}.issubset(manifests):
        result = _daily_intraday_consistency_check(
            con,
            root=root,
            daily=manifests["market_daily_raw"],
            intraday=manifests["market_intraday_5m"],
            history=manifests.get("symbol_history"),
            sample_limit=sample_limit,
        )
        checks["daily_intraday_5m"] = result
        if int(result["invalid_day_count"]) > 0:
            findings.append(
                _finding(
                    "high",
                    "cross_frequency",
                    "market_intraday_5m",
                    "daily_intraday_5m_semantic_mismatch",
                    result,
                )
            )
        missing_days = int(result["missing_positive_daily_count"])
        if missing_days:
            history_complete = bool(
                dict(active.get("scope", {}) or {}).get("intraday_5m_restored_for_historical_symbols", True)
            )
            result["coverage_contract"] = (
                "full_daily_intraday_history" if history_complete else "daily_pit_history_with_explicit_intraday_gaps"
            )
            findings.append(
                _finding(
                    "high" if history_complete else "medium",
                    "coverage",
                    "market_intraday_5m",
                    (
                        "historical_5m_coverage_not_100_percent"
                        if history_complete
                        else "historical_5m_unavailable_for_restored_daily_symbols"
                    ),
                    result,
                )
            )
    if {"market_daily_raw", "adjust_factor"}.issubset(manifests):
        result = _factor_semantic_check(
            con,
            root=root,
            daily=manifests["market_daily_raw"],
            factor=manifests["adjust_factor"],
            sample_limit=sample_limit,
        )
        checks["factor_semantics"] = result
        if int(result["uncompensated_change_count"]) > 0 or int(result["excessive_change_symbol_year_count"]) > 0:
            findings.append(
                _finding(
                    "high",
                    "factor",
                    "adjust_factor",
                    "adjust_factor_change_not_explained_by_raw_price",
                    result,
                )
            )
    if {"market_daily_raw", "security_status"}.issubset(manifests):
        result = _status_daily_consistency_check(
            con,
            root=root,
            daily=manifests["market_daily_raw"],
            status=manifests["security_status"],
            sample_limit=sample_limit,
        )
        checks["status_daily_semantics"] = result
        if int(result["semantic_mismatch_count"]) > 0 or int(result["null_suspension_flag_count"]) > 0:
            findings.append(
                _finding(
                    "high",
                    "status",
                    "security_status",
                    "security_status_daily_semantic_mismatch",
                    result,
                )
            )
    return checks, findings


def _identity_cross_checks(
    con: Any,
    *,
    root: Path,
    manifests: Mapping[str, DatasetManifest],
    sample_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checks: dict[str, Any] = {}
    findings: list[dict[str, Any]] = []
    if {"security_identity", "symbol_history"}.issubset(manifests):
        result = _identity_history_consistency_check(
            con,
            root=root,
            identity=manifests["security_identity"],
            history=manifests["symbol_history"],
            sample_limit=sample_limit,
        )
        checks["identity_symbol_history"] = result
        if result["status"] != "ok":
            findings.append(
                _finding(
                    "high",
                    "identity",
                    "symbol_history",
                    "identity_symbol_history_inconsistent",
                    result,
                )
            )
    if {"security_status", "trading_calendar"}.issubset(manifests):
        checks["long_suspension_continuity"] = _long_suspension_check(
            con,
            root=root,
            status=manifests["security_status"],
            calendar=manifests["trading_calendar"],
            sample_limit=sample_limit,
        )
    required = {
        "market_daily_raw",
        "adjust_factor",
        "security_status",
        "trading_calendar",
        "security_identity",
    }
    if required.issubset(manifests):
        result = _reopen_discontinuity_check(
            con,
            root=root,
            daily=manifests["market_daily_raw"],
            factor=manifests["adjust_factor"],
            status=manifests["security_status"],
            calendar=manifests["trading_calendar"],
            identity=manifests["security_identity"],
            sample_limit=sample_limit,
        )
        checks["reopen_discontinuities"] = result
        if int(result["unexpected_discontinuity_count"]) > 0:
            findings.append(
                _finding(
                    "high",
                    "price_continuity",
                    "market_daily_raw",
                    "unexplained_adjacent_trade_price_discontinuity",
                    result,
                )
            )
    return checks, findings


def _auxiliary_cross_check(
    con: Any,
    *,
    root: Path,
    manifests: Mapping[str, DatasetManifest],
    sample_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = {"market_daily_raw", "universe_snapshot", "adjust_factor", *AUXILIARY_DOMAINS}
    if not required.issubset(manifests):
        return {}, []
    result = _auxiliary_semantics_check(
        con,
        root=root,
        manifests=manifests,
        sample_limit=sample_limit,
    )
    findings = []
    if result["status"] != "ok":
        findings.append(
            _finding(
                "high",
                "auxiliary",
                "auxiliary_domains",
                "auxiliary_strict_pit_or_completeness_error",
                result,
            )
        )
    return {"auxiliary_domains": result}, findings


def _lifecycle_cross_check(
    *,
    workspace: Path,
    datasets: Mapping[str, str],
    sample_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    domains = tuple(domain for domain in LIFECYCLE_NORMALIZE_DOMAINS if domain in datasets)
    if not domains or not {"security_identity", "symbol_history"}.issubset(datasets):
        return {}, []
    try:
        result = audit_symbol_lifecycle_effectivity(
            workspace_root=workspace,
            domains=domains,
            sample_limit=sample_limit,
        )
    except PitHistoryError as exc:
        result = {"status": "error", "error": str(exc), "domains": list(domains)}
    findings = []
    if result["status"] != "ok":
        findings.append(
            _finding(
                "high",
                "identity",
                "daily_research_domains",
                "symbol_lifecycle_effective_interval_violation",
                result,
            )
        )
    return {"symbol_lifecycle_effectivity": result}, findings


def _latest_key_findings(latest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if latest.get("status") == "needs_attention":
        return [
            _finding(
                "high",
                "latest_keys",
                "market_core",
                "latest_key_check_failed",
                {"error": error},
            )
            for error in list(latest.get("errors", []) or [])
        ]
    if latest.get("status") == "warning":
        return [
            _finding(
                "medium",
                "latest_keys",
                "market_intraday_5m",
                "latest_5m_coverage_warning",
                {"five_minute": latest.get("five_minute", {})},
            )
        ]
    return []


def _database_audit_payload(
    *,
    root: Path,
    deep: bool,
    reports: list[dict[str, Any]],
    missing_required: list[str],
    latest: Mapping[str, Any],
    cross_checks: Mapping[str, Any],
    findings: list[dict[str, Any]],
    selected_threads: int,
) -> dict[str, Any]:
    blocking = [item for item in findings if item["severity"] == "high"]
    return {
        "status": "needs_attention" if blocking else "ok",
        "mode": "full",
        "scope": "physical_contract_and_selected_market_identity_semantics",
        "all_active_domains_semantically_certified": False,
        "deep": deep,
        "qdp_v2_root": str(root.resolve()),
        "audited_at": utc_now(),
        "dataset_count": len(reports),
        "required_domains": sorted(REQUIRED_DOMAINS),
        "missing_required_domains": missing_required,
        "datasets": reports,
        "latest_keys": latest,
        "cross_dataset_checks": {"latest_keys": latest, **cross_checks},
        "finding_count": len(findings),
        "findings": findings,
        "warnings": [item for item in findings if item["severity"] == "medium"],
        "errors": blocking,
        "resource_policy": {
            "threads": selected_threads,
            "memory": "dynamic_available_ram_minus_0.5_gib_with_2_second_watchdog",
        },
    }


def _database_cross_checks(
    *,
    workspace: Path,
    root: Path,
    active: Mapping[str, Any],
    datasets: Mapping[str, str],
    deep: bool,
    max_shards: int,
    sample_limit: int,
    skip_global_primary_key: bool,
    threads: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    cross_checks: dict[str, Any] = {}
    with (
        _audit_temp_directory(workspace) as temp,
        open_guarded_duckdb(temp_directory=temp, threads=threads) as con,
    ):
        reports, manifests, findings = _audit_active_datasets(
            con,
            root=root,
            datasets=datasets,
            deep=deep,
            max_shards=max_shards,
            sample_limit=sample_limit,
            skip_global_primary_key=skip_global_primary_key,
        )
        if deep:
            result, result_findings = _market_cross_checks(
                con,
                root=root,
                active=active,
                manifests=manifests,
                sample_limit=sample_limit,
            )
            cross_checks.update(result)
            findings.extend(result_findings)
            for check in (_identity_cross_checks, _auxiliary_cross_check):
                result, result_findings = check(
                    con, root=root, manifests=manifests, sample_limit=sample_limit
                )
                cross_checks.update(result)
                findings.extend(result_findings)
    if deep:
        lifecycle, lifecycle_findings = _lifecycle_cross_check(
            workspace=workspace,
            datasets=datasets,
            sample_limit=sample_limit,
        )
        cross_checks.update(lifecycle)
        findings.extend(lifecycle_findings)
    return reports, cross_checks, findings


def audit_database(
    *,
    workspace_root: str | Path | None = None,
    deep: bool = True,
    runtime: str = "balanced",
    duckdb_memory_limit: str = "",
    threads: int | None = None,
    max_shards: int = 0,
    sample_limit: int = 20,
    skip_global_primary_key: bool = False,
    global_uniqueness_max_shards: int = 0,
    skip_cross_frequency: bool = False,
    write: bool = False,
) -> dict[str, Any]:
    """Audit current tables without provider arbitration or retired 1m dependencies."""
    del duckdb_memory_limit, global_uniqueness_max_shards, skip_cross_frequency
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    if not active:
        return {
            "status": "needs_attention",
            "mode": "full",
            "deep": bool(deep),
            "dataset_count": 0,
            "finding_count": 1,
            "findings": [_finding("high", "manifest", "active", "active_manifest_missing", {})],
        }
    datasets = active_dataset_map(active)
    active_audit = audit_active(workspace_root=workspace_root, write=False, verify_footers=True)
    missing_required, findings = _active_manifest_findings(active_audit, datasets)
    selected_threads = max(
        1,
        min(int(threads or {"safe": 2, "balanced": 4, "fast": 8}.get(runtime, 4)), os.cpu_count() or 1),
    )
    workspace = Path(workspace_root or Path.cwd()).resolve()
    limit = max(1, int(sample_limit))
    reports, cross_checks, domain_findings = _database_cross_checks(
        workspace=workspace,
        root=root,
        active=active,
        datasets=datasets,
        deep=bool(deep),
        max_shards=max(0, int(max_shards)),
        sample_limit=limit,
        skip_global_primary_key=bool(skip_global_primary_key),
        threads=selected_threads,
    )
    findings.extend(domain_findings)
    latest = audit_latest_keys(workspace_root=workspace_root)
    findings.extend(_latest_key_findings(latest))
    payload = _database_audit_payload(
        root=root,
        deep=bool(deep),
        reports=reports,
        missing_required=missing_required,
        latest=latest,
        cross_checks=cross_checks,
        findings=findings,
        selected_threads=selected_threads,
    )
    if write:
        audit_id = f"database_audit_{utc_now().replace(':', '').replace('-', '')}"
        audit_path = root / "audits" / f"{audit_id}.json"
        atomic_write_json(audit_path, payload)
        payload["audit_path"] = str(audit_path.resolve())
    return payload


def _dataset_schema_audit(
    manifest: DatasetManifest,
    paths: list[Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    columns = list(pq.read_schema(paths[0]).names) if paths else []
    missing_columns = sorted(set(REQUIRED_COLUMNS.get(manifest.domain, ())).difference(columns))
    if missing_columns:
        findings.append(
            _finding(
                "high",
                "schema",
                manifest.domain,
                "required_columns_missing",
                {"columns": missing_columns},
            )
        )
    return {
        "status": "ok" if not missing_columns else "error",
        "columns": columns,
        "missing_columns": missing_columns,
    }, findings


def _deep_dataset_audit(
    con: Any,
    *,
    manifest: DatasetManifest,
    paths: list[Path],
    entries: list[Any],
    sample_limit: int,
    skip_primary_key: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checks: dict[str, Any] = {}
    findings: list[dict[str, Any]] = []
    if manifest.primary_key and not skip_primary_key:
        pk = (
            _ordered_intraday_primary_key_check(
                con,
                paths,
                entries,
                manifest.primary_key,
                sample_limit,
            )
            if manifest.domain == "market_intraday_5m"
            else _primary_key_check(con, paths, manifest.primary_key, sample_limit)
        )
        checks["primary_key"] = pk
        if int(pk["null_key_rows"]) > 0:
            findings.append(
                _finding(
                    "high",
                    "primary_key",
                    manifest.domain,
                    "primary_key_null_rows",
                    pk,
                )
            )
        if int(pk["duplicate_rows"]) > 0:
            findings.append(
                _finding(
                    "high",
                    "primary_key",
                    manifest.domain,
                    "primary_key_duplicate_rows",
                    pk,
                )
            )
    if manifest.domain in {"market_daily_raw", "market_intraday_5m"}:
        ohlc = _ohlc_check(con, paths, sample_limit)
        checks["ohlc"] = ohlc
        if int(ohlc["invalid_rows"]) > 0:
            findings.append(_finding("high", "ohlc", manifest.domain, "invalid_ohlcv_rows", ohlc))
    if manifest.domain == "market_intraday_5m":
        bars = _bar_day_check(con, paths, sample_limit)
        checks["bar_days"] = bars
        if int(bars["invalid_day_count"]) > 0:
            findings.append(
                _finding(
                    "high",
                    "bar_contract",
                    manifest.domain,
                    "invalid_48_bar_days",
                    bars,
                )
            )
    if manifest.domain == "adjust_factor":
        factor = _factor_check(con, paths, sample_limit)
        checks["factor"] = factor
        if int(factor["invalid_rows"]) > 0:
            findings.append(
                _finding(
                    "high",
                    "factor",
                    manifest.domain,
                    "invalid_adjust_factor_rows",
                    factor,
                )
            )
    if manifest.domain == "symbol_history":
        identity = _symbol_history_check(con, paths, sample_limit)
        checks["identity_intervals"] = identity
        if int(identity["overlap_count"]) > 0:
            findings.append(
                _finding(
                    "high",
                    "identity",
                    manifest.domain,
                    "symbol_history_interval_overlap",
                    identity,
                )
            )
    return checks, findings


def _audit_dataset(
    con: Any,
    *,
    root: Path,
    manifest: DatasetManifest,
    deep: bool,
    max_shards: int,
    sample_limit: int,
    skip_primary_key: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_paths = [resolve_manifest_path(item.path, root=root) for item in manifest.shards]
    paths = all_paths[:max_shards] if max_shards else all_paths
    entries = list(manifest.shards[:max_shards] if max_shards else manifest.shards)
    schema, findings = _dataset_schema_audit(manifest, paths)

    checks: dict[str, Any] = {
        "schema": schema,
        "scanned_shards": len(paths),
        "total_shards": len(all_paths),
    }
    if deep and paths and not schema["missing_columns"]:
        deep_checks, deep_findings = _deep_dataset_audit(
            con,
            manifest=manifest,
            paths=paths,
            entries=entries,
            sample_limit=sample_limit,
            skip_primary_key=skip_primary_key,
        )
        checks.update(deep_checks)
        findings.extend(deep_findings)

    return (
        {
            "domain": manifest.domain,
            "dataset_id": manifest.dataset_id,
            "row_count": manifest.row_count,
            "shard_count": len(manifest.shards),
            "start_date": manifest.start_date,
            "end_date": manifest.end_date,
            "checks": checks,
        },
        findings,
    )
