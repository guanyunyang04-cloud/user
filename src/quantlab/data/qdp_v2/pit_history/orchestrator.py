"""PIT history orchestrator operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from quantlab.core.io import json_safe
from quantlab.data.providers.baostock.provider import BaostockProvider
from quantlab.data.qdp_v2.active import resolve_active_domain
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
    qdp_v2_root,
    read_active_manifest,
    utc_now,
    write_active_manifest,
)
from quantlab.data.qdp_v2.repair.mutation import (
    mutate_active_shards_from_parquet,
    update_active_manifest_metadata,
)

from .config import (
    DEFAULT_START_DATE,
    LIFECYCLE_NORMALIZE_DOMAINS,
    RESTORE_DOMAINS,
    SSE_FACTBOOK_EVIDENCE,
    SSE_ST_TRANSITIONS,
)
from .context import (
    PitHistoryError,
    _archive_paths,
    _context,
    _read_symbol_history_symbols,
    _stock_basic,
    _workspace_relative_path,
    inventory_pit_history,
)
from .download import (
    _build_archive_cache,
    _download_market_supplements,
    _download_reference_parts,
    _download_sina_factors,
    _fill_missing_turnover_from_tencent,
    _materialize_history_parts,
)
from .lifecycle_audit import (
    _symbol_lifecycle_tables,
    audit_symbol_lifecycle_effectivity,
)
from .lifecycle_factor import (
    _normalize_lifecycle_factor_scale_stitches,
)
from .lifecycle_prepare import (
    _cleanup_lifecycle_prepared_domain,
    _prepare_lifecycle_domain_mutation,
)
from .prepare import (
    _combine_domain_parts,
    _create_composite_dataset,
    _prepare_symbol_parts,
    _validate_prepared,
)


def _lifecycle_factor_scale(
    ctx: Any,
    *,
    intervals: Any,
    domains: Sequence[str],
    apply: bool,
) -> dict[str, Any]:
    if "adjust_factor" not in domains or intervals.empty:
        return {"status": "not_applicable", "plans": []}
    return _normalize_lifecycle_factor_scale_stitches(
        ctx,
        intervals=intervals,
        apply=apply,
    )


def _normalize_lifecycle_domain(
    ctx: Any,
    *,
    domain: str,
    finding: dict[str, Any],
    intervals: Any,
    symbol_map: Any,
    apply: bool,
) -> dict[str, Any]:
    if finding.get("status") == "ok":
        context = resolve_active_domain(domain, workspace_root=ctx.workspace)
        return {
            "status": "already_normalized",
            "manifest_row_count_before": int(context.manifest.row_count),
            "projected_manifest_row_count": int(context.manifest.row_count),
        }
    replacements, removals, appends, prepared = _prepare_lifecycle_domain_mutation(
        ctx,
        domain=domain,
        intervals=intervals,
        symbol_map=symbol_map,
    )
    if not apply:
        _cleanup_lifecycle_prepared_domain(ctx, domain)
        return {**prepared, "status": "planned"}
    mutation = mutate_active_shards_from_parquet(
        domain,
        replacements=replacements,
        removals=removals,
        appends=appends,
        reason="canonicalize ticker rows to symbol_history effective intervals",
        workspace_root=ctx.workspace,
        allow_selected_schema_superset=True,
    )
    update_active_manifest_metadata(
        domain,
        reason="record symbol_history effective-interval normalization",
        workspace_root=ctx.workspace,
        source_updates={
            "symbol_lifecycle_semantics": "date_effective_symbol_history",
            "symbol_lifecycle_normalized_at": utc_now(),
        },
        quality_updates={
            "symbol_history_effective_intervals": True,
            "multi_symbol_identity_count": int(intervals["security_id"].nunique()),
        },
    )
    _cleanup_lifecycle_prepared_domain(ctx, domain)
    return {
        **prepared,
        "status": str(mutation.get("status", "")),
        "manifest_row_count": int(mutation.get("row_count", 0) or 0),
    }


def _write_lifecycle_normalization(ctx: Any, payload: dict[str, Any]) -> dict[str, Any]:
    atomic_write_json(
        ctx.runtime / "lifecycle_effectivity" / "normalization.json",
        payload,
    )
    return json_safe(payload)


def normalize_symbol_lifecycle_effectivity(
    *,
    workspace_root: str | Path | None = None,
    domains: Sequence[str] = LIFECYCLE_NORMALIZE_DOMAINS,
    apply: bool = True,
) -> dict[str, Any]:
    active = read_active_manifest(qdp_v2_root(workspace_root))
    ctx = _context(
        start_date=DEFAULT_START_DATE,
        end_date=str(active.get("active_as_of_date", DEFAULT_START_DATE)),
        workspace_root=workspace_root,
    )
    before = audit_symbol_lifecycle_effectivity(
        workspace_root=ctx.workspace,
        domains=domains,
    )
    intervals, symbol_map, cutoff = _symbol_lifecycle_tables(ctx)
    if before["status"] == "ok":
        factor_scale = _lifecycle_factor_scale(
            ctx,
            intervals=intervals,
            domains=domains,
            apply=apply,
        )
        if factor_scale["status"] in {"already_aligned", "not_applicable"}:
            return {
                "status": "already_normalized",
                "before": before,
                "domains": {},
                "factor_scale": factor_scale,
            }
        payload = {
            "status": "normalized" if apply else "planned",
            "before": before,
            "after": before if apply else {},
            "domains": {},
            "factor_scale": factor_scale,
            "intraday_5m_changed": False,
            "completed_at": utc_now() if apply else "",
        }
        if apply:
            return _write_lifecycle_normalization(ctx, payload)
        return json_safe(payload)
    results = {
        domain: _normalize_lifecycle_domain(
            ctx,
            domain=domain,
            finding=dict(before["domains"].get(domain, {}) or {}),
            intervals=intervals,
            symbol_map=symbol_map,
            apply=apply,
        )
        for domain in domains
    }
    if not apply:
        return json_safe(
            {
                "status": "planned",
                "before": before,
                "effective_cutoff": cutoff,
                "domains": results,
                "factor_scale": {"status": "pending_symbol_normalization"},
                "intraday_5m_changed": False,
            }
        )
    factor_scale = _lifecycle_factor_scale(
        ctx,
        intervals=intervals,
        domains=domains,
        apply=True,
    )
    after = audit_symbol_lifecycle_effectivity(
        workspace_root=ctx.workspace,
        domains=domains,
    )
    if after["status"] != "ok":
        raise PitHistoryError(
            f"pit_history_symbol_lifecycle_normalization_incomplete:{after['outside_effective_interval_rows']}"
        )
    payload = {
        "status": "normalized",
        "before": before,
        "after": after,
        "domains": results,
        "factor_scale": factor_scale,
        "intraday_5m_changed": False,
        "completed_at": utc_now(),
    }
    return _write_lifecycle_normalization(ctx, payload)


def _download_restore_sources(
    ctx: Any,
    *,
    provider: BaostockProvider,
    symbols: Sequence[str],
    workers: int,
) -> Any:
    basic = _stock_basic(ctx, provider).set_index("symbol", drop=False)
    archive_cache = _build_archive_cache(ctx, symbols=symbols)
    _download_market_supplements(ctx, symbols=symbols, workers=workers)
    _download_sina_factors(ctx, symbols=symbols, workers=workers)
    _materialize_history_parts(
        ctx,
        symbols=symbols,
        basic=basic.reset_index(drop=True),
        archive_cache=archive_cache,
    )
    _fill_missing_turnover_from_tencent(ctx, symbols=symbols)
    return basic


def _prepare_restore_domains(
    ctx: Any,
    *,
    symbols: Sequence[str],
    basic: Any,
    workers: int,
) -> dict[str, Path]:
    _download_reference_parts(ctx, symbols=symbols, workers=workers)
    known_history_symbols = _read_symbol_history_symbols(ctx)
    for index, symbol in enumerate(symbols, start=1):
        _prepare_symbol_parts(
            ctx,
            row=basic.loc[symbol].to_dict(),
            identity_already_known=symbol in known_history_symbols,
        )
        if index == 1 or index % 25 == 0 or index == len(symbols):
            print(f"pit_history_prepare={index}/{len(symbols)}", flush=True)
    return {domain: _combine_domain_parts(ctx, domain) for domain in RESTORE_DOMAINS}


def _restore_source_evidence(ctx: Any) -> dict[str, Any]:
    archive_paths = _archive_paths(ctx)
    return {
        "path_base": "workspace_root",
        "archive_paths": {key: _workspace_relative_path(value, ctx.workspace) for key, value in archive_paths.items()},
        "sse_factbooks": SSE_FACTBOOK_EVIDENCE,
        "sse_transition_resource": _workspace_relative_path(SSE_ST_TRANSITIONS, ctx.workspace),
        "factor_provider": "sina_via_akshare_hfq_factor_event",
        "boundary_market_provider": "eastmoney_via_akshare_unadjusted_daily",
    }


def _restore_scope(ctx: Any, inventory: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "point_in_time_historical_mainboard",
        "universe": (
            "Shanghai/Shenzhen main-board securities are included on dates when "
            "listed; same-day ST, suspension, and delisting state controls eligibility."
        ),
        "start_date": ctx.start_date,
        "end_date": ctx.end_date,
        "symbol_count": int(inventory["pit_mainboard_symbol_count"]),
        "restored_symbol_count": int(inventory["missing_symbol_count"]),
        "restored_delisted_symbol_count": int(inventory["missing_delisted_symbol_count"]),
        "survivorship_policy": "point_in_time_no_future_exclusion",
        "intraday_5m_restored_for_historical_symbols": False,
    }


def _activate_restore(
    ctx: Any,
    *,
    before: dict[str, Any],
    inventory: dict[str, Any],
    validation: dict[str, Any],
    prepared: Mapping[str, Path],
) -> tuple[dict[str, Any], Path]:
    datasets: dict[str, str] = dict(before.get("datasets", {}) or {})
    commits: dict[str, Any] = {}
    for domain in RESTORE_DOMAINS:
        dataset_id, commit = _create_composite_dataset(
            ctx,
            domain=domain,
            prepared=prepared[domain],
        )
        datasets[domain] = dataset_id
        commits[domain] = commit
    source_evidence = _restore_source_evidence(ctx)
    after = {
        **before,
        "datasets": datasets,
        "scope": _restore_scope(ctx, inventory),
        "source": {
            **dict(before.get("source", {}) or {}),
            "pit_history_restore": (
                "protected_baostock_archive+akshare_eastmoney+sina_factor+cninfo+szse+sse_factbook"
            ),
            "pit_history_restored_at": utc_now(),
            "pit_history_source_evidence": source_evidence,
        },
        "updated_at": utc_now(),
    }
    audit = {
        "schema_version": 1,
        "audit_type": "qdp_pit_historical_mainboard_activation",
        "created_at": utc_now(),
        "before_active": before,
        "after_active": after,
        "inventory": inventory,
        "validation": validation,
        "commits": commits,
        "source_evidence": source_evidence,
    }
    audit_id = utc_now().replace("-", "").replace(":", "").replace("+00:00", "Z")
    audit_path = ctx.root / "audits" / f"pit_history_restore_{audit_id}.json"
    atomic_write_json(audit_path, audit)
    write_active_manifest(ctx.root, after)
    return commits, audit_path


def _write_restore_result(ctx: Any, payload: dict[str, Any]) -> dict[str, Any]:
    atomic_write_json(ctx.runtime / "result.json", payload)
    return payload


def run_pit_history_restore(
    *,
    start_date: str = DEFAULT_START_DATE,
    end_date: str,
    workspace_root: str | Path | None = None,
    workers: int = 3,
    chunk_size: int = 20,
    apply: bool = True,
) -> dict[str, Any]:
    ctx = _context(
        start_date=start_date,
        end_date=end_date,
        workspace_root=workspace_root,
    )
    before = read_active_manifest(ctx.root)
    provider = BaostockProvider(
        _market_daily_max_workers=1,
        _reuse_symbol_range_session=True,
    )
    try:
        inventory = inventory_pit_history(
            start_date=ctx.start_date,
            end_date=ctx.end_date,
            workspace_root=ctx.workspace,
            provider=provider,
        )
        symbols = list(inventory["missing_symbols"])
        if not symbols:
            lifecycle = normalize_symbol_lifecycle_effectivity(
                workspace_root=ctx.workspace,
                apply=apply,
            )
            return {
                **inventory,
                "status": ("updated" if lifecycle.get("status") == "normalized" else "already_complete"),
                "symbol_lifecycle": lifecycle,
            }
        basic = _download_restore_sources(
            ctx,
            provider=provider,
            symbols=symbols,
            workers=workers,
        )
    finally:
        provider.close()
    prepared = _prepare_restore_domains(
        ctx,
        symbols=symbols,
        basic=basic,
        workers=workers,
    )
    validation = _validate_prepared(ctx, symbols=symbols, prepared=prepared)
    if validation["status"] != "ok":
        return _write_restore_result(
            ctx,
            {
                "status": "blocked",
                "inventory": inventory,
                "validation": validation,
                "active_unchanged": True,
            },
        )
    if not apply:
        return _write_restore_result(
            ctx,
            {
                "status": "prepared",
                "inventory": inventory,
                "validation": validation,
                "active_unchanged": True,
            },
        )
    commits, audit_path = _activate_restore(
        ctx,
        before=before,
        inventory=inventory,
        validation=validation,
        prepared=prepared,
    )
    lifecycle = normalize_symbol_lifecycle_effectivity(
        workspace_root=ctx.workspace,
        apply=True,
    )
    payload = {
        "status": "updated",
        "inventory": inventory,
        "validation": validation,
        "commits": commits,
        "symbol_lifecycle": lifecycle,
        "audit_path": str(audit_path),
    }
    _write_restore_result(ctx, payload)
    return json_safe(payload)
