from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from quant_data_platform.core.json_io import json_safe, read_json, utc_now, write_json
from quant_data_platform.core.paths import QdpPaths, qdp_paths
from quant_data_platform.core.registry import load_root_manifest, write_root_manifest_update
from quant_data_platform.domains.contracts import DataDomain, build_intraday_daily_feature_frame, normalize_domain, normalize_domain_frame
from quant_data_platform.ingest.baostock_backfill import BackfillConfig, run_backfill
from quant_data_platform.ingest.combine_domain_datasets import CombineDomainDatasetsConfig, combine_domain_datasets
from quant_data_platform.ingest.refresh_daily import RefreshConfig, run_refresh
from quant_data_platform.lake.canonical import (
    DEFAULT_CANONICAL_ALIAS,
    DEFAULT_CANONICAL_START_DATE,
    build_coverage_report,
    load_canonical_manifest,
    write_canonical_manifest,
)
from quant_data_platform.lake.catalog import ResearchDataLake
from quant_data_platform.lake.v2_status_sidecar import build_v2_status_sidecar


DEFAULT_DAILY_UPDATE_DOMAINS: tuple[str, ...] = (
    DataDomain.MARKET_DAILY,
    DataDomain.TRADING_CALENDAR,
    DataDomain.UNIVERSE_SNAPSHOT,
    DataDomain.SECURITY_STATUS,
    DataDomain.ADJUST_FACTOR,
    DataDomain.VALUATION,
    DataDomain.INTRADAY_DAILY_FEATURES,
)
DEFAULT_REQUIRED_PIT_DOMAINS: tuple[str, ...] = (
    DataDomain.TRADING_CALENDAR,
    DataDomain.UNIVERSE_SNAPSHOT,
    DataDomain.SECURITY_STATUS,
)
DEFAULT_LAG_TOLERANT_DOMAINS: tuple[str, ...] = (
    DataDomain.ADJUST_FACTOR,
    DataDomain.VALUATION,
    DataDomain.INTRADAY_DAILY_FEATURES,
)
POINT_IN_TIME_DAILY_DOMAINS = {
    DataDomain.UNIVERSE_SNAPSHOT,
    DataDomain.SECURITY_STATUS,
}
SYMBOL_RANGE_DOMAINS = {
    DataDomain.ADJUST_FACTOR,
    DataDomain.VALUATION,
    DataDomain.INTRADAY_DAILY_FEATURES,
}


@dataclass(frozen=True)
class DailyUpdateConfig:
    workspace_root: str | Path | None = None
    lake_root: str | Path | None = None
    run_id: str = ""
    as_of_date: str = ""
    start_date: str = DEFAULT_CANONICAL_START_DATE
    provider_plan: str = "qdp_production_v1"
    readiness_mode: str = "strict"
    domains: tuple[str, ...] = DEFAULT_DAILY_UPDATE_DOMAINS
    required_pit_domains: tuple[str, ...] = DEFAULT_REQUIRED_PIT_DOMAINS
    lag_tolerant_domains: tuple[str, ...] = DEFAULT_LAG_TOLERANT_DOMAINS
    strict_enhanced_domains: tuple[str, ...] = ()
    symbols: str = "all_lake"
    refresh_symbols: tuple[str, ...] = ()
    universe: str = "all_a"
    benchmark: str = "000300.SH"
    adjusted_flag: str = "none"
    snapshot_frequency: str = "daily"
    intraday_features_mode: str = "auto"
    build_policy_bundle: bool = False
    build_v2_status_sidecar: bool = False
    build_memmap: bool = False
    activate: bool = False
    dry_run: bool = False
    resume: bool = True
    chunk_size_symbols: int = 200
    chunk_size_months: int = 1
    task_workers: int = 2
    snapshot_workers: int = 2
    valuation_workers: int = 2
    adjust_factor_workers: int = 2
    market_daily_symbol_workers: int = 4
    intraday_symbol_workers: int = 1
    failed_chunk_retries: int = 1
    failed_chunk_sweeps: int = 1
    retry_backoff_seconds: float = 1.0
    retry_jitter_seconds: float = 0.5
    min_coverage_ratio: float = 0.80
    conflict_tolerance_pct: float = 0.005
    severe_conflict_limit: int = 0
    memmap_profile: str = ""
    memmap_rebuild_start_year: int = 0
    memmap_rebuild_end_year: int = 0
    memmap_workers: int = 1
    memmap_symbol_block_size: int = 300
    memmap_max_universe_size: int = 0
    memmap_max_shards: int = 0
    memmap_year_input_cache: bool = False
    max_duplicate_check_rows: int = 2_000_000

    def normalized(self) -> "DailyUpdateConfig":
        as_of = pd.Timestamp(self.as_of_date or pd.Timestamp.today()).strftime("%Y-%m-%d")
        domains = _normalize_domains(self.domains or DEFAULT_DAILY_UPDATE_DOMAINS)
        required = _normalize_domains(self.required_pit_domains or DEFAULT_REQUIRED_PIT_DOMAINS)
        lag_tolerant = _normalize_domains(self.lag_tolerant_domains or DEFAULT_LAG_TOLERANT_DOMAINS)
        strict_enhanced = _normalize_domains(self.strict_enhanced_domains)
        readiness_mode = str(self.readiness_mode or "strict").strip().lower()
        if readiness_mode not in {"strict", "staged"}:
            raise ValueError(f"unsupported_readiness_mode: {readiness_mode}")
        intraday_features_mode = str(self.intraday_features_mode or "auto").strip().lower()
        if intraday_features_mode not in {"auto", "derive", "skip", "provider"}:
            raise ValueError(f"unsupported_intraday_features_mode: {intraday_features_mode}")
        return DailyUpdateConfig(
            workspace_root=self.workspace_root,
            lake_root=Path(self.lake_root) if self.lake_root else None,
            run_id=str(self.run_id or "").strip(),
            as_of_date=as_of,
            start_date=pd.Timestamp(self.start_date or DEFAULT_CANONICAL_START_DATE).strftime("%Y-%m-%d"),
            provider_plan=str(self.provider_plan or "qdp_production_v1").strip(),
            readiness_mode=readiness_mode,
            domains=domains,
            required_pit_domains=required,
            lag_tolerant_domains=lag_tolerant,
            strict_enhanced_domains=strict_enhanced,
            symbols=str(self.symbols or "all_lake").strip(),
            refresh_symbols=tuple(str(item).strip().upper() for item in self.refresh_symbols if str(item).strip()),
            universe=str(self.universe or "all_a").strip(),
            benchmark=str(self.benchmark or "000300.SH").strip().upper(),
            adjusted_flag=str(self.adjusted_flag or "none").strip(),
            snapshot_frequency=str(self.snapshot_frequency or "daily").strip().lower(),
            intraday_features_mode=intraday_features_mode,
            build_policy_bundle=bool(self.build_policy_bundle),
            build_v2_status_sidecar=bool(self.build_v2_status_sidecar),
            build_memmap=bool(self.build_memmap),
            activate=bool(self.activate),
            dry_run=bool(self.dry_run),
            resume=bool(self.resume),
            chunk_size_symbols=max(1, int(self.chunk_size_symbols or 1)),
            chunk_size_months=max(1, int(self.chunk_size_months or 1)),
            task_workers=max(1, int(self.task_workers or 1)),
            snapshot_workers=max(1, int(self.snapshot_workers or 1)),
            valuation_workers=max(1, int(self.valuation_workers or 1)),
            adjust_factor_workers=max(1, int(self.adjust_factor_workers or 1)),
            market_daily_symbol_workers=max(1, int(self.market_daily_symbol_workers or 1)),
            intraday_symbol_workers=max(1, int(self.intraday_symbol_workers or 1)),
            failed_chunk_retries=max(0, int(self.failed_chunk_retries or 0)),
            failed_chunk_sweeps=max(0, int(self.failed_chunk_sweeps or 0)),
            retry_backoff_seconds=max(0.0, float(self.retry_backoff_seconds or 0.0)),
            retry_jitter_seconds=max(0.0, float(self.retry_jitter_seconds or 0.0)),
            min_coverage_ratio=float(self.min_coverage_ratio),
            conflict_tolerance_pct=float(self.conflict_tolerance_pct),
            severe_conflict_limit=int(self.severe_conflict_limit),
            memmap_profile=str(self.memmap_profile or "").strip(),
            memmap_rebuild_start_year=max(0, int(self.memmap_rebuild_start_year or 0)),
            memmap_rebuild_end_year=max(0, int(self.memmap_rebuild_end_year or 0)),
            memmap_workers=max(1, int(self.memmap_workers or 1)),
            memmap_symbol_block_size=max(1, int(self.memmap_symbol_block_size or 1)),
            memmap_max_universe_size=max(0, int(self.memmap_max_universe_size or 0)),
            memmap_max_shards=max(0, int(self.memmap_max_shards or 0)),
            memmap_year_input_cache=bool(self.memmap_year_input_cache),
            max_duplicate_check_rows=max(0, int(self.max_duplicate_check_rows or 0)),
        )


@dataclass(frozen=True)
class DomainUpdatePlan:
    domain: str
    inherited_dataset_id: str = ""
    inherited_start_date: str = ""
    inherited_end_date: str = ""
    gap_start_date: str = ""
    gap_end_date: str = ""
    needs_backfill: bool = False
    planned_task_count: int = 0
    expected_trade_dates: tuple[str, ...] = ()
    status: str = "current"


@dataclass(frozen=True)
class DailyUpdateResult:
    status: str
    run_id: str
    manifest_path: Path
    as_of_date: str
    target_date: str
    domain_plans: dict[str, DomainUpdatePlan] = field(default_factory=dict)
    before_coverage: dict[str, dict[str, Any]] = field(default_factory=dict)
    after_coverage: dict[str, dict[str, Any]] = field(default_factory=dict)
    backfill_dataset_ids: dict[str, str] = field(default_factory=dict)
    backfill_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    effective_sidecar_dataset_ids: dict[str, str] = field(default_factory=dict)
    combined_sidecar_dataset_ids: dict[str, str] = field(default_factory=dict)
    lagging_domains: dict[str, dict[str, Any]] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    refresh_result: dict[str, Any] = field(default_factory=dict)
    canonical_dataset_id: str = ""
    canonical_manifest_path: str = ""
    v2_status_sidecar_dataset_id: str = ""
    memmap_manifest_path: str = ""
    memmap_validation: dict[str, Any] = field(default_factory=dict)


def run_daily_update(
    config: DailyUpdateConfig,
    *,
    paths: QdpPaths | None = None,
    refresh_providers: Iterable[Any] | None = None,
    backfill_provider: Any | None = None,
) -> DailyUpdateResult:
    cfg = config.normalized()
    resolved_paths = paths or qdp_paths(cfg.workspace_root)
    lake_root = Path(cfg.lake_root or resolved_paths.lake_root)
    lake = ResearchDataLake(lake_root)
    active_context = _active_context(lake=lake, paths=resolved_paths)
    calendar_frame = _read_dataset_frame(lake=lake, dataset_id=active_context["sidecars"].get(DataDomain.TRADING_CALENDAR, ""))
    target_date = _resolve_target_date(cfg.as_of_date, calendar_frame)
    run_id = cfg.run_id or f"daily_update_{target_date.replace('-', '')}_{pd.Timestamp.now().strftime('%H%M%S')}"
    run_dir = lake.root / "data_platform" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "daily_update_manifest.json"

    before_coverage = {
        domain: _dataset_summary(lake=lake, dataset_id=dataset_id)
        for domain, dataset_id in active_context["sidecars"].items()
    }
    expected_trade_dates = _expected_trade_dates(calendar_frame=calendar_frame, start_date=cfg.start_date, target_date=target_date)
    domain_plans = _build_domain_plans(
        cfg=cfg,
        lake=lake,
        active_sidecars=active_context["sidecars"],
        before_coverage=before_coverage,
        target_date=target_date,
        expected_trade_dates=expected_trade_dates,
    )

    if cfg.dry_run:
        dry_run_backfill_results = _dry_run_backfill_results(
            cfg=cfg,
            lake=lake,
            active_sidecars=active_context["sidecars"],
            domain_plans=domain_plans,
        )
        result = DailyUpdateResult(
            status="planned",
            run_id=run_id,
            manifest_path=manifest_path,
            as_of_date=cfg.as_of_date,
            target_date=target_date,
            domain_plans=domain_plans,
            before_coverage=before_coverage,
            backfill_results=dry_run_backfill_results,
            effective_sidecar_dataset_ids=dict(active_context["sidecars"]),
            blockers=[],
        )
        _write_run_manifest(result=result, cfg=cfg, run_dir=run_dir)
        return result

    blockers: list[str] = []
    refresh_payload: dict[str, Any] = {}
    market_dataset_id = str(active_context.get("canonical_dataset_id", "") or "")
    if DataDomain.MARKET_DAILY in cfg.domains:
        refresh_result = run_refresh(
            RefreshConfig(
                lake_root=lake.root,
                as_of_date=target_date,
                start_date=cfg.start_date,
                symbols=cfg.refresh_symbols,
                universe=cfg.universe,
                domains=(DataDomain.MARKET_DAILY,),
                required_domains=(DataDomain.MARKET_DAILY,),
                benchmark=cfg.benchmark,
                provider_plan=cfg.provider_plan,
                adjusted_flag=cfg.adjusted_flag,
                min_coverage_ratio=cfg.min_coverage_ratio,
                conflict_tolerance_pct=cfg.conflict_tolerance_pct,
                severe_conflict_limit=cfg.severe_conflict_limit,
                run_id=f"{run_id}_refresh",
            ),
            providers=refresh_providers,
        )
        refresh_payload = {
            "status": refresh_result.status,
            "refresh_run_id": refresh_result.refresh_run_id,
            "manifest_path": refresh_result.manifest_path,
            "registered_market_dataset_id": refresh_result.registered_market_dataset_id,
            "blockers": list(refresh_result.blockers),
            "coverage_report": dict(refresh_result.coverage_report),
        }
        if refresh_result.status not in {"ok", "skipped"}:
            blockers.append(f"market_refresh_{refresh_result.status}")
            blockers.extend([f"market_refresh:{item}" for item in refresh_result.blockers])
        if refresh_result.registered_market_dataset_id:
            market_dataset_id = refresh_result.registered_market_dataset_id
    if not market_dataset_id:
        blockers.append("market_daily_dataset_id_required")

    backfill_dataset_ids: dict[str, str] = {}
    backfill_results: dict[str, Any] = {}
    lagging_annotations: dict[str, dict[str, Any]] = {}
    for domain in cfg.domains:
        if domain == DataDomain.MARKET_DAILY:
            continue
        plan = domain_plans.get(domain)
        if plan is None:
            continue
        if domain == DataDomain.MARKET_DAILY or not plan.needs_backfill:
            continue
        if domain == DataDomain.INTRADAY_DAILY_FEATURES:
            action = _resolve_intraday_features_action(
                cfg=cfg,
                lake=lake,
                active_sidecars=active_context["sidecars"],
                plan=plan,
            )
            if action["action"] == "skip":
                backfill_results[domain] = action
                lagging_annotations[domain] = {
                    "reason": str(action.get("reason", "") or "skipped"),
                    "action": "skipped_backfill",
                    "mode": cfg.intraday_features_mode,
                    "source_domain": DataDomain.MARKET_INTRADAY_5M,
                    "source_dataset_id": str(action.get("source_dataset_id", "") or ""),
                }
                continue
            if action["action"] == "derive":
                try:
                    derived = _derive_intraday_features_from_existing_raw(
                        cfg=cfg,
                        lake=lake,
                        run_dir=run_dir,
                        run_id=run_id,
                        plan=plan,
                        source_dataset_id=str(action.get("source_dataset_id", "") or ""),
                        raw_shards=list(action.get("raw_shards", []) or []),
                    )
                except Exception as exc:
                    blockers.append(f"backfill_failed:{domain}:{type(exc).__name__}:{exc}")
                    backfill_results[domain] = {
                        **{key: value for key, value in action.items() if key != "raw_shards"},
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    continue
                backfill_results[domain] = {
                    **{key: value for key, value in action.items() if key != "raw_shards"},
                    "status": "ok",
                    "run_id": run_id,
                    "manifest_path": str(manifest_path),
                    "planned_task_count": int(derived["planned_task_count"]),
                    "row_counts": {domain: int(derived["row_count"])},
                    "error_counts": {domain: int(derived["error_count"])},
                    "dataset_ids": {domain: str(derived["dataset_id"])},
                    "timing": dict(derived.get("timing", {}) or {}),
                }
                if str(derived.get("dataset_id", "") or ""):
                    backfill_dataset_ids[domain] = str(derived["dataset_id"])
                if int(derived.get("error_count", 0) or 0) > 0:
                    blockers.append(f"backfill_errors:{domain}:{derived['error_count']}")
                continue
        backfill_cfg = BackfillConfig(
            lake_root=lake.root,
            run_id=f"{run_id}_backfill_{_safe_name(domain)}",
            domains=(domain,),
            start_date=plan.gap_start_date,
            end_date=plan.gap_end_date,
            symbols=cfg.symbols,
            benchmark=cfg.benchmark,
            adjusted_flag=cfg.adjusted_flag,
            resume=cfg.resume,
            snapshot_frequency=cfg.snapshot_frequency,
            chunk_size_symbols=cfg.chunk_size_symbols,
            chunk_size_months=cfg.chunk_size_months,
            task_workers=cfg.task_workers,
            snapshot_workers=cfg.snapshot_workers,
            valuation_workers=cfg.valuation_workers,
            adjust_factor_workers=cfg.adjust_factor_workers,
            market_daily_symbol_workers=cfg.market_daily_symbol_workers,
            intraday_symbol_workers=cfg.intraday_symbol_workers,
            failed_chunk_retries=cfg.failed_chunk_retries,
            failed_chunk_sweeps=cfg.failed_chunk_sweeps,
            retry_backoff_seconds=cfg.retry_backoff_seconds,
            retry_jitter_seconds=cfg.retry_jitter_seconds,
            reuse_existing_market_daily=True,
            trade_dates=tuple(plan.expected_trade_dates),
            build_policy_bundle=False,
            write_canonical_manifest=False,
        )
        try:
            backfill_result = run_backfill(backfill_cfg, provider=backfill_provider)
        except Exception as exc:
            blockers.append(f"backfill_failed:{domain}:{type(exc).__name__}:{exc}")
            continue
        backfill_results[domain] = {
            "status": backfill_result.status,
            "run_id": backfill_result.run_id,
            "manifest_path": str(backfill_result.manifest_path),
            "planned_task_count": int(backfill_result.planned_task_count),
            "row_counts": dict(backfill_result.row_counts),
            "error_counts": dict(backfill_result.error_counts),
            "dataset_ids": dict(backfill_result.dataset_ids),
        }
        dataset_id = str(backfill_result.dataset_ids.get(domain, "") or "")
        if dataset_id:
            backfill_dataset_ids[domain] = dataset_id
        errors = int(backfill_result.error_counts.get(domain, 0) or 0)
        rows = int(backfill_result.row_counts.get(domain, 0) or 0)
        if errors > 0:
            blockers.append(f"backfill_errors:{domain}:{errors}")
        if domain in cfg.required_pit_domains and rows <= 0:
            blockers.append(f"required_pit_backfill_empty:{domain}")
        if domain == DataDomain.TRADING_CALENDAR and dataset_id:
            refreshed_calendar = _effective_calendar_frame(
                lake=lake,
                inherited_dataset_id=active_context["sidecars"].get(DataDomain.TRADING_CALENDAR, ""),
                tail_dataset_id=dataset_id,
            )
            refreshed_expected_trade_dates = _expected_trade_dates(
                calendar_frame=refreshed_calendar,
                start_date=cfg.start_date,
                target_date=target_date,
            )
            if refreshed_expected_trade_dates:
                expected_trade_dates = refreshed_expected_trade_dates
                domain_plans = _build_domain_plans(
                    cfg=cfg,
                    lake=lake,
                    active_sidecars=active_context["sidecars"],
                    before_coverage=before_coverage,
                    target_date=target_date,
                    expected_trade_dates=expected_trade_dates,
                )

    effective_sidecars, combined_sidecars = _combine_effective_sidecars(
        cfg=cfg,
        lake=lake,
        active_sidecars=active_context["sidecars"],
        domain_plans=domain_plans,
        backfill_dataset_ids=backfill_dataset_ids,
        target_date=target_date,
    )
    after_coverage = {
        domain: _dataset_summary(lake=lake, dataset_id=dataset_id)
        for domain, dataset_id in effective_sidecars.items()
    }
    readiness = _validate_readiness(
        cfg=cfg,
        lake=lake,
        target_date=target_date,
        expected_trade_dates=expected_trade_dates,
        domain_plans=domain_plans,
        effective_sidecars=effective_sidecars,
        before_coverage=before_coverage,
        after_coverage=after_coverage,
    )
    blockers.extend(readiness["blockers"])
    lagging_domains = dict(readiness["lagging_domains"])
    for domain, annotation in lagging_annotations.items():
        current = dict(lagging_domains.get(domain, {}) or {})
        current.update(annotation)
        lagging_domains[domain] = current

    canonical_dataset_id = ""
    canonical_manifest_path = ""
    v2_status_sidecar_dataset_id = ""
    memmap_manifest_path = ""
    memmap_validation: dict[str, Any] = {}
    activation_requested = (
        cfg.activate
        and cfg.readiness_mode == "strict"
        and cfg.build_policy_bundle
        and bool(market_dataset_id)
    )
    if cfg.activate and cfg.readiness_mode != "strict":
        blockers.append("activation_requires_strict_readiness_mode")
    if cfg.activate and not cfg.build_policy_bundle:
        blockers.append("activation_requires_build_policy_bundle")
    can_build_bundle = cfg.build_policy_bundle and not blockers and bool(market_dataset_id)
    if can_build_bundle:
        bundle_result = _build_canonical_bundle(
            lake_root=lake.root,
            market_dataset_id=market_dataset_id,
            sidecars=effective_sidecars,
            start_date=cfg.start_date,
            end_date=target_date,
            benchmark=cfg.benchmark,
            write_manifest=False,
        )
        canonical_dataset_id = bundle_result.dataset_id

    if cfg.build_v2_status_sidecar and canonical_dataset_id and not blockers:
        record, _frame, summary = build_v2_status_sidecar(
            lake=lake,
            source_market_dataset_id=canonical_dataset_id,
            start_date=cfg.start_date,
            end_date=target_date,
            universe_snapshot_dataset_id=effective_sidecars.get(DataDomain.UNIVERSE_SNAPSHOT, ""),
            security_status_dataset_id=effective_sidecars.get(DataDomain.SECURITY_STATUS, ""),
            reuse=True,
        )
        v2_status_sidecar_dataset_id = record.dataset_id
        if str(summary.get("status", "") or "") != "ok":
            blockers.append(f"v2_status_sidecar_{summary.get('status')}")
        lake.write_catalog_manifest()

    if cfg.build_memmap and canonical_dataset_id and not blockers:
        try:
            memmap_payload = _build_and_validate_memmap(
                cfg=cfg,
                paths=resolved_paths,
                target_date=target_date,
                canonical_dataset_id=canonical_dataset_id,
            )
            memmap_manifest_path = str(memmap_payload.get("manifest_json", "") or "")
            memmap_validation = dict(memmap_payload.get("validation", {}) or {})
        except Exception as exc:
            memmap_validation = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            blockers.append(f"memmap_build_failed:{type(exc).__name__}:{exc}")
        if memmap_validation and str(memmap_validation.get("status", "") or "") != "ok":
            blockers.append("memmap_validation_failed")

    activation_allowed = activation_requested and canonical_dataset_id and not blockers
    if activation_allowed:
        canonical_manifest_path = str(
            write_canonical_manifest(
                lake,
                dataset_id=canonical_dataset_id,
                alias=DEFAULT_CANONICAL_ALIAS,
                start_date=cfg.start_date,
                end_date=target_date,
                sidecar_dataset_ids=effective_sidecars,
                coverage_report=build_coverage_report(lake, start_date=cfg.start_date),
                notes="canonical policy bundle activated by daily_update after PIT readiness gates",
            )
        )
        _activate_root_manifest(
            paths=resolved_paths,
            cfg=cfg,
            run_id=run_id,
            target_date=target_date,
            canonical_dataset_id=canonical_dataset_id,
            canonical_manifest_path=canonical_manifest_path,
            market_dataset_id=market_dataset_id,
            sidecars=effective_sidecars,
        )
        if cfg.build_memmap and memmap_manifest_path:
            from quant_data_platform.memmap.incremental import register_sharded_memmap_manifest

            register_sharded_memmap_manifest(
                resolved_paths,
                manifest=memmap_manifest_path,
                activate=True,
            )

    if blockers:
        status = "blocked"
    elif cfg.readiness_mode == "staged":
        status = "staged"
    elif activation_allowed:
        status = "activated"
    else:
        status = "ok"

    result = DailyUpdateResult(
        status=status,
        run_id=run_id,
        manifest_path=manifest_path,
        as_of_date=cfg.as_of_date,
        target_date=target_date,
        domain_plans=domain_plans,
        before_coverage=before_coverage,
        after_coverage=after_coverage,
        backfill_dataset_ids=backfill_dataset_ids,
        backfill_results={key: dict(value) for key, value in backfill_results.items()},
        effective_sidecar_dataset_ids=effective_sidecars,
        combined_sidecar_dataset_ids=combined_sidecars,
        lagging_domains=lagging_domains,
        blockers=blockers,
        refresh_result=refresh_payload,
        canonical_dataset_id=canonical_dataset_id,
        canonical_manifest_path=canonical_manifest_path,
        v2_status_sidecar_dataset_id=v2_status_sidecar_dataset_id,
        memmap_manifest_path=memmap_manifest_path,
        memmap_validation=memmap_validation,
    )
    _write_run_manifest(result=result, cfg=cfg, run_dir=run_dir)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run QDP daily market refresh with PIT tail backfill and promotion gates.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--data-lake-root", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--start-date", default=DEFAULT_CANONICAL_START_DATE)
    parser.add_argument("--provider-plan", default="qdp_production_v1")
    parser.add_argument("--readiness-mode", choices=("strict", "staged"), default="strict")
    parser.add_argument("--domains", default=",".join(DEFAULT_DAILY_UPDATE_DOMAINS))
    parser.add_argument("--required-pit-domains", default=",".join(DEFAULT_REQUIRED_PIT_DOMAINS))
    parser.add_argument("--lag-tolerant-domains", default=",".join(DEFAULT_LAG_TOLERANT_DOMAINS))
    parser.add_argument("--strict-enhanced-domains", default="")
    parser.add_argument("--symbols", default="all_lake")
    parser.add_argument("--refresh-symbols", default="")
    parser.add_argument("--universe", default="all_a")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--adjusted-flag", default="none")
    parser.add_argument("--snapshot-frequency", choices=("daily", "monthly", "quarterly", "end"), default="daily")
    parser.add_argument(
        "--intraday-features-mode",
        choices=("auto", "derive", "skip", "provider"),
        default="auto",
        help="auto derives intraday_daily_features from existing market_intraday_5m tail when available and otherwise marks it lagging; provider explicitly allows slow BaoStock 5m downloads.",
    )
    parser.add_argument("--chunk-size-symbols", type=int, default=200)
    parser.add_argument("--chunk-size-months", type=int, default=1)
    parser.add_argument("--task-workers", type=int, default=2)
    parser.add_argument("--snapshot-workers", type=int, default=2)
    parser.add_argument("--valuation-workers", type=int, default=2)
    parser.add_argument("--adjust-factor-workers", type=int, default=2)
    parser.add_argument("--market-daily-symbol-workers", type=int, default=4)
    parser.add_argument("--intraday-symbol-workers", type=int, default=1)
    parser.add_argument("--failed-chunk-retries", type=int, default=1)
    parser.add_argument("--failed-chunk-sweeps", type=int, default=1)
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--retry-jitter-seconds", type=float, default=0.5)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.80)
    parser.add_argument("--conflict-tolerance-pct", type=float, default=0.005)
    parser.add_argument("--severe-conflict-limit", type=int, default=0)
    parser.add_argument("--build-policy-bundle", action="store_true")
    parser.add_argument("--build-v2-status-sidecar", action="store_true")
    parser.add_argument("--build-memmap", action="store_true")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true", default=True)
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--memmap-profile", default="")
    parser.add_argument("--memmap-rebuild-start-year", type=int, default=0)
    parser.add_argument("--memmap-rebuild-end-year", type=int, default=0)
    parser.add_argument("--memmap-workers", type=int, default=1)
    parser.add_argument("--memmap-symbol-block-size", type=int, default=300)
    parser.add_argument("--memmap-max-universe-size", type=int, default=0)
    parser.add_argument("--memmap-max-shards", type=int, default=0)
    parser.add_argument("--memmap-year-input-cache", action="store_true")
    parser.add_argument("--max-duplicate-check-rows", type=int, default=2_000_000)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = run_daily_update(
        DailyUpdateConfig(
            workspace_root=Path(args.workspace_root) if str(args.workspace_root or "").strip() else None,
            lake_root=Path(args.data_lake_root) if str(args.data_lake_root or "").strip() else None,
            run_id=str(args.run_id or ""),
            as_of_date=str(args.as_of_date),
            start_date=str(args.start_date or DEFAULT_CANONICAL_START_DATE),
            provider_plan=str(args.provider_plan or "qdp_production_v1"),
            readiness_mode=str(args.readiness_mode or "strict"),
            domains=_parse_csv_domains(args.domains),
            required_pit_domains=_parse_csv_domains(args.required_pit_domains),
            lag_tolerant_domains=_parse_csv_domains(args.lag_tolerant_domains),
            strict_enhanced_domains=_parse_csv_domains(args.strict_enhanced_domains),
            symbols=str(args.symbols or "all_lake"),
            refresh_symbols=_parse_csv_symbols(args.refresh_symbols),
            universe=str(args.universe or "all_a"),
            benchmark=str(args.benchmark or "000300.SH"),
            adjusted_flag=str(args.adjusted_flag or "none"),
            snapshot_frequency=str(args.snapshot_frequency or "daily"),
            intraday_features_mode=str(args.intraday_features_mode or "auto"),
            build_policy_bundle=bool(args.build_policy_bundle),
            build_v2_status_sidecar=bool(args.build_v2_status_sidecar),
            build_memmap=bool(args.build_memmap),
            activate=bool(args.activate),
            dry_run=bool(args.dry_run),
            resume=bool(args.resume),
            chunk_size_symbols=int(args.chunk_size_symbols),
            chunk_size_months=int(args.chunk_size_months),
            task_workers=int(args.task_workers),
            snapshot_workers=int(args.snapshot_workers),
            valuation_workers=int(args.valuation_workers),
            adjust_factor_workers=int(args.adjust_factor_workers),
            market_daily_symbol_workers=int(args.market_daily_symbol_workers),
            intraday_symbol_workers=int(args.intraday_symbol_workers),
            failed_chunk_retries=int(args.failed_chunk_retries),
            failed_chunk_sweeps=int(args.failed_chunk_sweeps),
            retry_backoff_seconds=float(args.retry_backoff_seconds),
            retry_jitter_seconds=float(args.retry_jitter_seconds),
            min_coverage_ratio=float(args.min_coverage_ratio),
            conflict_tolerance_pct=float(args.conflict_tolerance_pct),
            severe_conflict_limit=int(args.severe_conflict_limit),
            memmap_profile=str(args.memmap_profile or ""),
            memmap_rebuild_start_year=int(args.memmap_rebuild_start_year),
            memmap_rebuild_end_year=int(args.memmap_rebuild_end_year),
            memmap_workers=int(args.memmap_workers),
            memmap_symbol_block_size=int(args.memmap_symbol_block_size),
            memmap_max_universe_size=int(args.memmap_max_universe_size),
            memmap_max_shards=int(args.memmap_max_shards),
            memmap_year_input_cache=bool(args.memmap_year_input_cache),
            max_duplicate_check_rows=int(args.max_duplicate_check_rows),
        )
    )
    payload = _result_payload(result)
    if bool(args.json):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(f"{result.status}: {result.manifest_path}")
        if result.blockers:
            print("blockers=" + ",".join(result.blockers))
    return 0 if result.status in {"planned", "ok", "staged", "activated"} else 2


def _dry_run_backfill_results(
    *,
    cfg: DailyUpdateConfig,
    lake: ResearchDataLake,
    active_sidecars: Mapping[str, str],
    domain_plans: Mapping[str, DomainUpdatePlan],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for domain, plan in domain_plans.items():
        if domain == DataDomain.MARKET_DAILY or not plan.needs_backfill:
            continue
        if domain == DataDomain.INTRADAY_DAILY_FEATURES:
            action = _resolve_intraday_features_action(cfg=cfg, lake=lake, active_sidecars=active_sidecars, plan=plan)
            results[domain] = {key: value for key, value in action.items() if key != "raw_shards"}
            if "raw_shards" in action:
                results[domain]["raw_shard_count"] = len(list(action.get("raw_shards", []) or []))
            continue
        results[domain] = {
            "status": "planned",
            "action": "provider_backfill",
            "planned_task_count": int(plan.planned_task_count),
            "gap_start_date": plan.gap_start_date,
            "gap_end_date": plan.gap_end_date,
        }
    return results


def _resolve_intraday_features_action(
    *,
    cfg: DailyUpdateConfig,
    lake: ResearchDataLake,
    active_sidecars: Mapping[str, str],
    plan: DomainUpdatePlan,
) -> dict[str, Any]:
    mode = cfg.intraday_features_mode
    raw_dataset_id = str(active_sidecars.get(DataDomain.MARKET_INTRADAY_5M, "") or "")
    raw_summary = _dataset_summary(lake=lake, dataset_id=raw_dataset_id) if raw_dataset_id else {"status": "missing"}
    raw_end_date = str(raw_summary.get("end_date", "") or "")
    raw_shards = (
        _raw_5m_tail_shards(
            lake=lake,
            dataset_id=raw_dataset_id,
            start_date=plan.gap_start_date,
            end_date=plan.gap_end_date,
        )
        if raw_dataset_id
        else []
    )
    raw_covers_target = bool(raw_end_date and pd.Timestamp(raw_end_date) >= pd.Timestamp(plan.gap_end_date))
    can_derive = bool(raw_dataset_id and raw_covers_target and raw_shards)
    base = {
        "domain": DataDomain.INTRADAY_DAILY_FEATURES,
        "mode": mode,
        "gap_start_date": plan.gap_start_date,
        "gap_end_date": plan.gap_end_date,
        "source_domain": DataDomain.MARKET_INTRADAY_5M,
        "source_dataset_id": raw_dataset_id,
        "source_end_date": raw_end_date,
    }
    if mode == "skip":
        return {**base, "status": "skipped", "action": "skip", "reason": "intraday_features_mode_skip"}
    if mode in {"auto", "derive"} and can_derive:
        return {
            **base,
            "status": "planned",
            "action": "derive",
            "reason": "derive_from_existing_market_intraday_5m_tail",
            "raw_shards": raw_shards,
            "raw_shard_count": len(raw_shards),
        }
    if mode == "provider":
        return {**base, "status": "planned", "action": "provider", "reason": "explicit_provider_mode"}
    reason = "market_intraday_5m_tail_missing"
    if raw_dataset_id and raw_end_date and not raw_covers_target:
        reason = f"market_intraday_5m_lagging:{raw_end_date}<{plan.gap_end_date}"
    elif raw_dataset_id and raw_covers_target and not raw_shards:
        reason = "market_intraday_5m_tail_shards_missing"
    elif not raw_dataset_id:
        reason = "market_intraday_5m_dataset_missing"
    return {**base, "status": "skipped", "action": "skip", "reason": reason}


def _raw_5m_tail_shards(*, lake: ResearchDataLake, dataset_id: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    try:
        metadata = lake.describe_dataset(dataset_id)
    except Exception:
        return []
    manifest_text = str(dict(metadata.get("content_paths", {}) or {}).get("shard_manifest", "") or "")
    manifest_path = Path(manifest_text) if manifest_text else Path()
    if not manifest_text or not manifest_path.exists():
        return []
    manifest = read_json(manifest_path)
    out: list[dict[str, Any]] = []
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    for item in list(manifest.get("shards", []) or []):
        record = dict(item)
        path = Path(str(record.get("path", "") or ""))
        if str(record.get("status", "stored") or "stored") != "stored":
            continue
        if not path.exists():
            continue
        if int(record.get("row_count", 0) or 0) <= 0:
            continue
        shard_start = str(record.get("start_date", "") or "")
        shard_end = str(record.get("end_date", "") or "")
        if shard_start and shard_end:
            if pd.Timestamp(shard_end) < start or pd.Timestamp(shard_start) > end:
                continue
        out.append(record)
    return out


def _derive_intraday_features_from_existing_raw(
    *,
    cfg: DailyUpdateConfig,
    lake: ResearchDataLake,
    run_dir: Path,
    run_id: str,
    plan: DomainUpdatePlan,
    source_dataset_id: str,
    raw_shards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    domain = DataDomain.INTRADAY_DAILY_FEATURES
    spec = {
        "dataset": f"data_platform_{domain}",
        "source": "daily_update_intraday_raw_derivation",
        "provider_plan": cfg.provider_plan,
        "daily_update_run_id": run_id,
        "domain": domain,
        "start_date": plan.gap_start_date,
        "end_date": plan.gap_end_date,
        "symbols_spec": cfg.symbols,
        "adjusted_flag": cfg.adjusted_flag,
        "source_domain": DataDomain.MARKET_INTRADAY_5M,
        "source_dataset_id": source_dataset_id,
        "derivation": "market_intraday_5m_to_intraday_daily_features",
        "sharded": True,
    }
    identity = lake.build_domain_dataset_identity(domain=domain, spec=spec)
    shard_dir = Path(identity["dataset_dir"]) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    status_dir = run_dir / "chunks" / domain
    status_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    start_ts = pd.Timestamp(plan.gap_start_date)
    end_ts = pd.Timestamp(plan.gap_end_date)
    for idx, raw_record in enumerate(raw_shards, start=1):
        t0 = time.perf_counter()
        raw_path = Path(str(dict(raw_record).get("path", "") or ""))
        raw_chunk_id = str(dict(raw_record).get("chunk_id", "") or f"raw_{idx:06d}")
        chunk_id = _safe_name(raw_chunk_id.replace(DataDomain.MARKET_INTRADAY_5M, domain))
        status_path = status_dir / f"{chunk_id}.json"
        cached = read_json(status_path) if cfg.resume and status_path.exists() else {}
        if cached:
            records.append(cached)
            continue
        errors: list[dict[str, Any]] = []
        try:
            raw = pd.read_parquet(raw_path)
            if "trade_date" in raw.columns:
                dates_ts = pd.to_datetime(raw["trade_date"], errors="coerce")
                mask = (dates_ts >= start_ts) & (dates_ts <= end_ts)
                raw = raw.loc[mask].copy()
                raw["trade_date"] = dates_ts.loc[mask].dt.strftime("%Y-%m-%d").to_numpy()
            features = build_intraday_daily_feature_frame(raw, source="daily_update_intraday_raw_derivation", adjusted_flag=cfg.adjusted_flag)
            if "trade_date" in features.columns:
                feature_dates_ts = pd.to_datetime(features["trade_date"], errors="coerce")
                feature_mask = (feature_dates_ts >= start_ts) & (feature_dates_ts <= end_ts)
                features = features.loc[feature_mask].copy()
                features["trade_date"] = feature_dates_ts.loc[feature_mask].dt.strftime("%Y-%m-%d").to_numpy()
        except Exception as exc:
            features = normalize_domain_frame(
                pd.DataFrame(),
                domain=domain,
                source="daily_update_intraday_raw_derivation",
                as_of_date=plan.gap_end_date,
                adjusted_flag=cfg.adjusted_flag,
                require_columns=False,
            )
            errors.append(
                {
                    "provider": "daily_update",
                    "domain": domain,
                    "code": "intraday_feature_derivation_failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "source_raw_path": str(raw_path),
                }
            )
        shard_path = shard_dir / f"{chunk_id}.parquet"
        store_t0 = time.perf_counter()
        features.to_parquet(shard_path, index=False)
        store_duration = time.perf_counter() - store_t0
        elapsed = time.perf_counter() - t0
        record = {
            "status": "stored",
            "domain": domain,
            "chunk_id": chunk_id,
            "task_kind": "derived_from_existing_market_intraday_5m",
            "start_date": plan.gap_start_date,
            "end_date": plan.gap_end_date,
            "symbol_count": int(features["symbol"].nunique()) if "symbol" in features.columns and not features.empty else 0,
            "row_count": int(len(features)),
            "path": str(shard_path.resolve()),
            "source_raw_chunk_id": raw_chunk_id,
            "source_raw_path": str(raw_path.resolve()) if raw_path.exists() else str(raw_path),
            "error_count": int(len(errors)),
            "error_report": errors,
            "audit": {"status": "ok" if len(features) else "empty", "row_count": int(len(features))},
            "store_duration_sec": round(store_duration, 3),
            "elapsed_sec": round(elapsed, 3),
            "stored_at": utc_now(),
        }
        write_json(status_path, record)
        records.append(record)
    dataset = lake.save_sharded_domain_dataset(domain=domain, spec=spec, shard_records=records, source="daily_update_intraday_raw_derivation", reuse=False)
    lake.write_catalog_manifest()
    row_count = int(sum(int(item.get("row_count", 0) or 0) for item in records))
    error_count = int(sum(int(item.get("error_count", 0) or 0) for item in records))
    return {
        "dataset_id": dataset.dataset_id,
        "planned_task_count": len(raw_shards),
        "row_count": row_count,
        "error_count": error_count,
        "timing": _shard_timing_summary(records),
    }


def _shard_timing_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    elapsed = [float(dict(item).get("elapsed_sec", 0.0) or 0.0) for item in records]
    stored_at = [str(dict(item).get("stored_at", "") or "") for item in records if str(dict(item).get("stored_at", "") or "")]
    return {
        "chunk_count": int(len(records)),
        "elapsed_sec_sum": round(sum(elapsed), 3),
        "elapsed_sec_max": round(max(elapsed), 3) if elapsed else 0.0,
        "first_stored_at": min(stored_at) if stored_at else "",
        "last_stored_at": max(stored_at) if stored_at else "",
    }


def _active_context(*, lake: ResearchDataLake, paths: QdpPaths) -> dict[str, Any]:
    root = load_root_manifest(paths)
    canonical_manifest = load_canonical_manifest(lake)
    canonical_dataset_id = str(
        canonical_manifest.get("canonical_dataset_id", "")
        or canonical_manifest.get("dataset_id", "")
        or root.get("canonical_dataset_id", "")
        or ""
    )
    sidecars: dict[str, str] = {}
    for source in (
        canonical_manifest.get("sidecar_dataset_ids", {}),
        root.get("canonical_bundle_sidecar_dataset_ids", {}),
        root.get("canonical_component_dataset_ids", {}),
    ):
        for key, value in dict(source or {}).items():
            domain = str(key)
            if domain == DataDomain.MARKET_DAILY:
                continue
            if str(value or "").strip():
                sidecars[domain] = str(value)
    if canonical_dataset_id:
        try:
            metadata = lake.describe_dataset(canonical_dataset_id)
            params = dict(metadata.get("parameters", {}) or {})
            for key, value in dict(params.get("sidecar_dataset_ids", {}) or {}).items():
                if str(value or "").strip():
                    sidecars.setdefault(str(key), str(value))
        except Exception:
            pass
    return {
        "root_manifest": root,
        "canonical_manifest": canonical_manifest,
        "canonical_dataset_id": canonical_dataset_id,
        "sidecars": sidecars,
    }


def _build_domain_plans(
    *,
    cfg: DailyUpdateConfig,
    lake: ResearchDataLake,
    active_sidecars: Mapping[str, str],
    before_coverage: Mapping[str, Mapping[str, Any]],
    target_date: str,
    expected_trade_dates: Sequence[str],
) -> dict[str, DomainUpdatePlan]:
    plans: dict[str, DomainUpdatePlan] = {}
    active_symbols = _active_symbol_count(lake=lake, cfg=cfg)
    for domain in cfg.domains:
        if domain == DataDomain.MARKET_DAILY:
            continue
        inherited_id = str(active_sidecars.get(domain, "") or "")
        summary = dict(before_coverage.get(domain, {}) or {})
        inherited_end = str(summary.get("end_date", "") or "")
        inherited_start = str(summary.get("start_date", "") or "")
        needs_backfill = not inherited_end or pd.Timestamp(inherited_end) < pd.Timestamp(target_date)
        gap_start = _gap_start_date(inherited_end, cfg.start_date)
        if domain in POINT_IN_TIME_DAILY_DOMAINS:
            expected = tuple(date for date in expected_trade_dates if gap_start <= date <= target_date)
        else:
            expected = tuple(date for date in expected_trade_dates if gap_start <= date <= target_date)
        planned_tasks = _estimate_task_count(
            domain=domain,
            start_date=gap_start,
            end_date=target_date,
            expected_trade_dates=expected,
            symbol_count=active_symbols,
            chunk_size_symbols=cfg.chunk_size_symbols,
            chunk_size_months=cfg.chunk_size_months,
            snapshot_frequency=cfg.snapshot_frequency,
        ) if needs_backfill else 0
        plans[domain] = DomainUpdatePlan(
            domain=domain,
            inherited_dataset_id=inherited_id,
            inherited_start_date=inherited_start,
            inherited_end_date=inherited_end,
            gap_start_date=gap_start if needs_backfill else "",
            gap_end_date=target_date if needs_backfill else "",
            needs_backfill=needs_backfill,
            planned_task_count=planned_tasks,
            expected_trade_dates=expected,
            status="needs_backfill" if needs_backfill else "current",
        )
    return plans


def _combine_effective_sidecars(
    *,
    cfg: DailyUpdateConfig,
    lake: ResearchDataLake,
    active_sidecars: Mapping[str, str],
    domain_plans: Mapping[str, DomainUpdatePlan],
    backfill_dataset_ids: Mapping[str, str],
    target_date: str,
) -> tuple[dict[str, str], dict[str, str]]:
    effective = {str(key): str(value) for key, value in dict(active_sidecars).items() if str(value or "").strip()}
    combined: dict[str, str] = {}
    for domain, tail_dataset_id in sorted(dict(backfill_dataset_ids).items()):
        inherited = str(active_sidecars.get(domain, "") or "")
        if inherited:
            plan = domain_plans.get(domain)
            start_date = str(plan.inherited_start_date or cfg.start_date) if plan else cfg.start_date
            result = combine_domain_datasets(
                CombineDomainDatasetsConfig(
                    lake_root=lake.root,
                    domain=domain,
                    source_dataset_ids=(inherited, str(tail_dataset_id)),
                    start_date=start_date,
                    end_date=target_date,
                    source="daily_update_pit_tail_combined",
                    merge_policy="inherited_history_plus_daily_tail",
                    dry_run=False,
                    reuse=True,
                )
            )
            if result.dataset_id:
                effective[domain] = result.dataset_id
                combined[domain] = result.dataset_id
            continue
        effective[domain] = str(tail_dataset_id)
    return effective, combined


def _validate_readiness(
    *,
    cfg: DailyUpdateConfig,
    lake: ResearchDataLake,
    target_date: str,
    expected_trade_dates: Sequence[str],
    domain_plans: Mapping[str, DomainUpdatePlan],
    effective_sidecars: Mapping[str, str],
    before_coverage: Mapping[str, Mapping[str, Any]],
    after_coverage: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    blockers: list[str] = []
    lagging: dict[str, dict[str, Any]] = {}
    for domain in cfg.required_pit_domains:
        dataset_id = str(effective_sidecars.get(domain, "") or "")
        summary = dict(after_coverage.get(domain, {}) or {})
        if not dataset_id:
            blockers.append(f"required_pit_missing:{domain}")
            continue
        if str(summary.get("status", "") or "") != "ok":
            blockers.append(f"required_pit_unavailable:{domain}")
            continue
        end_date = str(summary.get("end_date", "") or "")
        if not end_date or pd.Timestamp(end_date) < pd.Timestamp(target_date):
            blockers.append(f"required_pit_lagging:{domain}:{end_date or 'missing'}<{target_date}")
        if domain in POINT_IN_TIME_DAILY_DOMAINS:
            trade_date_count = int(summary.get("trade_date_count", 0) or 0)
            before_count = int(dict(before_coverage.get(domain, {}) or {}).get("trade_date_count", 0) or 0)
            if before_count > 1 and trade_date_count <= 1:
                blockers.append(f"required_pit_single_date_not_history:{domain}")
            plan = domain_plans.get(domain)
            missing_tail = sorted(set(plan.expected_trade_dates if plan else expected_trade_dates) - set(summary.get("trade_dates", []) or []))
            if missing_tail:
                blockers.append(f"required_pit_tail_gap:{domain}:{missing_tail[0]}..{missing_tail[-1]}")
        duplicate_report = _duplicate_key_report(lake=lake, dataset_id=dataset_id, max_rows=cfg.max_duplicate_check_rows)
        if duplicate_report.get("duplicate_count", 0):
            blockers.append(f"required_pit_duplicate_keys:{domain}:{duplicate_report['duplicate_count']}")
    for domain in cfg.lag_tolerant_domains:
        if domain not in cfg.domains:
            continue
        summary = dict(after_coverage.get(domain, {}) or {})
        end_date = str(summary.get("end_date", "") or "")
        if not end_date or pd.Timestamp(end_date) < pd.Timestamp(target_date):
            lagging[domain] = {
                "dataset_id": str(effective_sidecars.get(domain, "") or ""),
                "end_date": end_date,
                "target_date": target_date,
                "status": "lagging",
            }
            if domain in cfg.strict_enhanced_domains:
                blockers.append(f"strict_enhanced_domain_lagging:{domain}")
    return {"blockers": blockers, "lagging_domains": lagging}


def _build_and_validate_memmap(
    *,
    cfg: DailyUpdateConfig,
    paths: QdpPaths,
    target_date: str,
    canonical_dataset_id: str,
) -> dict[str, Any]:
    from quant_data_platform.memmap.incremental import compose_sharded_memmap
    from quant_data_platform.memmap.sharded import ShardedMemmapConfig, build_sharded_memmap, validate_sharded_memmap_manifest

    root = load_root_manifest(paths)
    sharded_registry = read_json(paths.registry_dir / "sharded_memmap_registry.json")
    base_manifest = str(sharded_registry.get("active_manifest_json", "") or "")
    base_payload = read_json(Path(base_manifest)) if base_manifest and Path(base_manifest).exists() else {}
    target_year = int(pd.Timestamp(target_date).year)
    rebuild_start = cfg.memmap_rebuild_start_year or target_year
    rebuild_end = cfg.memmap_rebuild_end_year or target_year
    base_schema_tag = str(base_payload.get("feature_schema_hash", "") or "schema")[:8]
    memmap_tag = f"{cfg.provider_plan}_{target_date.replace('-', '')}_tail_{rebuild_start}_{rebuild_end}_{base_schema_tag}"
    cumulative_horizons = base_payload.get("cumulative_horizons", "")
    if isinstance(cumulative_horizons, list):
        cumulative_horizons = ",".join(str(item) for item in cumulative_horizons)
    overlay = build_sharded_memmap(
        paths,
        config=ShardedMemmapConfig(
            canonical_dataset_id=canonical_dataset_id,
            profile=cfg.memmap_profile or str(base_payload.get("profile", "") or root.get("canonical_sharded_memmap_status", {}).get("latest_profile", "") or ""),
            start_year=rebuild_start,
            end_year=rebuild_end,
            symbol_block_size=cfg.memmap_symbol_block_size,
            max_universe_size=cfg.memmap_max_universe_size,
            max_shards=cfg.memmap_max_shards,
            lookback_days=int(base_payload.get("lookback_days", 0) or 60),
            horizon=int(base_payload.get("horizon", 0) or 20),
            cumulative_horizons=str(cumulative_horizons or "1,3,5,10,20"),
            execution_mode=str(base_payload.get("execution_mode", "") or "next_open"),
            max_feature_columns=int(base_payload.get("max_feature_columns", 0) or base_payload.get("feature_count", 0) or 192),
            min_lookback_valid_ratio=float(base_payload.get("min_lookback_valid_ratio", 0) or 0.80),
            tag=memmap_tag,
            workers=cfg.memmap_workers,
            year_input_cache=cfg.memmap_year_input_cache,
            force_years=",".join(str(year) for year in range(rebuild_start, rebuild_end + 1)),
            pool_view_id=str(base_payload.get("pool_view_id", "") or ""),
            sector_board_view_id=str(base_payload.get("sector_board_view_id", "") or ""),
            include_static_context=bool(base_payload.get("include_static_context", False) or False),
            static_context_fields=str(base_payload.get("static_context_fields", "") or "symbol,exchange,industry"),
        ),
    )
    manifest_json = str(overlay.get("manifest_json", "") or "")
    composite = overlay
    if base_manifest:
        composite = compose_sharded_memmap(
            paths,
            base_manifest=Path(base_manifest),
            overlay_manifests=[Path(manifest_json)],
            target_canonical_dataset_id=canonical_dataset_id,
            tag=f"{memmap_tag}_composite",
            activate=False,
            write=True,
        )
        manifest_json = str(composite.get("manifest_json", "") or "")
    validation = validate_sharded_memmap_manifest(Path(manifest_json))
    if str(validation.get("canonical_dataset_id", "") or "") != str(canonical_dataset_id):
        validation = {
            **dict(validation),
            "status": "blocked",
            "blockers": sorted(set(list(validation.get("blockers", []) or []) + ["source_market_dataset_mismatch"])),
        }
    return {**dict(composite), "manifest_json": manifest_json, "validation": validation}


def _build_canonical_bundle(
    *,
    lake_root: Path,
    market_dataset_id: str,
    sidecars: Mapping[str, str],
    start_date: str,
    end_date: str,
    benchmark: str,
    write_manifest: bool,
) -> Any:
    from quant_data_platform.lake.build_canonical_policy_bundle import (
        BuildCanonicalPolicyBundleConfig,
        build_canonical_policy_bundle,
    )

    return build_canonical_policy_bundle(
        BuildCanonicalPolicyBundleConfig(
            lake_root=lake_root,
            market_daily_dataset_id=market_dataset_id,
            sidecar_dataset_ids=sidecars,
            start_date=start_date,
            end_date=end_date,
            benchmark=benchmark,
            benchmark_source_dataset_id=market_dataset_id,
            source="daily_update",
            alias=DEFAULT_CANONICAL_ALIAS,
            write_manifest=write_manifest,
            reuse=True,
        )
    )


def _activate_root_manifest(
    *,
    paths: QdpPaths,
    cfg: DailyUpdateConfig,
    run_id: str,
    target_date: str,
    canonical_dataset_id: str,
    canonical_manifest_path: str,
    market_dataset_id: str,
    sidecars: Mapping[str, str],
) -> None:
    root = load_root_manifest(paths)
    components = dict(root.get("canonical_component_dataset_ids", {}) or {})
    components[DataDomain.MARKET_DAILY] = str(market_dataset_id)
    components.update({str(key): str(value) for key, value in dict(sidecars).items() if str(value or "").strip()})
    write_root_manifest_update(
        {
            "status": "daily_update_canonical_ready",
            "canonical_dataset_id": str(canonical_dataset_id),
            "canonical_manifest": str(canonical_manifest_path),
            "canonical_start_date": cfg.start_date,
            "canonical_component_dataset_ids": components,
            "canonical_bundle_sidecar_dataset_ids": dict(sidecars),
            "latest_daily_update": {
                "status": "activated",
                "run_id": run_id,
                "target_date": target_date,
                "canonical_dataset_id": str(canonical_dataset_id),
                "canonical_manifest": str(canonical_manifest_path),
                "updated_at": utc_now(),
            },
        },
        paths=paths,
    )


def _write_run_manifest(*, result: DailyUpdateResult, cfg: DailyUpdateConfig, run_dir: Path) -> dict[str, Any]:
    payload = _result_payload(result)
    payload["config"] = {
        "provider_plan": cfg.provider_plan,
        "readiness_mode": cfg.readiness_mode,
        "domains": list(cfg.domains),
        "required_pit_domains": list(cfg.required_pit_domains),
        "lag_tolerant_domains": list(cfg.lag_tolerant_domains),
        "strict_enhanced_domains": list(cfg.strict_enhanced_domains),
        "intraday_features_mode": cfg.intraday_features_mode,
        "task_workers": int(cfg.task_workers),
        "snapshot_workers": int(cfg.snapshot_workers),
        "valuation_workers": int(cfg.valuation_workers),
        "adjust_factor_workers": int(cfg.adjust_factor_workers),
        "market_daily_symbol_workers": int(cfg.market_daily_symbol_workers),
        "intraday_symbol_workers": int(cfg.intraday_symbol_workers),
        "failed_chunk_retries": int(cfg.failed_chunk_retries),
        "failed_chunk_sweeps": int(cfg.failed_chunk_sweeps),
        "retry_backoff_seconds": float(cfg.retry_backoff_seconds),
        "retry_jitter_seconds": float(cfg.retry_jitter_seconds),
        "build_policy_bundle": bool(cfg.build_policy_bundle),
        "build_v2_status_sidecar": bool(cfg.build_v2_status_sidecar),
        "build_memmap": bool(cfg.build_memmap),
        "activate": bool(cfg.activate),
        "dry_run": bool(cfg.dry_run),
    }
    payload["run_dir"] = str(run_dir.resolve())
    write_json(result.manifest_path, payload)
    return payload


def _result_payload(result: DailyUpdateResult) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": result.status,
        "run_id": result.run_id,
        "manifest_path": str(result.manifest_path),
        "as_of_date": result.as_of_date,
        "target_date": result.target_date,
        "domain_plans": {key: _domain_plan_payload(value) for key, value in result.domain_plans.items()},
        "before_coverage": result.before_coverage,
        "after_coverage": result.after_coverage,
        "backfill_dataset_ids": result.backfill_dataset_ids,
        "backfill_results": result.backfill_results,
        "effective_sidecar_dataset_ids": result.effective_sidecar_dataset_ids,
        "combined_sidecar_dataset_ids": result.combined_sidecar_dataset_ids,
        "lagging_domains": result.lagging_domains,
        "blockers": list(result.blockers),
        "refresh_result": result.refresh_result,
        "canonical_dataset_id": result.canonical_dataset_id,
        "canonical_manifest_path": result.canonical_manifest_path,
        "v2_status_sidecar_dataset_id": result.v2_status_sidecar_dataset_id,
        "memmap_manifest_path": result.memmap_manifest_path,
        "memmap_validation": result.memmap_validation,
        "generated_at": utc_now(),
    }


def _domain_plan_payload(plan: DomainUpdatePlan) -> dict[str, Any]:
    return {
        "domain": plan.domain,
        "inherited_dataset_id": plan.inherited_dataset_id,
        "inherited_start_date": plan.inherited_start_date,
        "inherited_end_date": plan.inherited_end_date,
        "gap_start_date": plan.gap_start_date,
        "gap_end_date": plan.gap_end_date,
        "needs_backfill": bool(plan.needs_backfill),
        "planned_task_count": int(plan.planned_task_count),
        "expected_trade_dates": list(plan.expected_trade_dates),
        "status": plan.status,
    }


def _dataset_summary(*, lake: ResearchDataLake, dataset_id: str) -> dict[str, Any]:
    dataset_id = str(dataset_id or "").strip()
    if not dataset_id:
        return {"status": "missing", "dataset_id": ""}
    try:
        metadata = lake.describe_dataset(dataset_id)
    except Exception as exc:
        return {"status": "missing", "dataset_id": dataset_id, "error": str(exc)}
    row_counts = dict(metadata.get("row_counts", {}) or {})
    trade_dates = _dataset_trade_dates(lake=lake, dataset_id=dataset_id, metadata=metadata)
    return {
        "status": "ok",
        "dataset_id": dataset_id,
        "dataset_kind": str(metadata.get("dataset_kind", "") or ""),
        "domain": str(metadata.get("domain", "") or ""),
        "source": str(metadata.get("source", "") or ""),
        "start_date": str(metadata.get("start_date", "") or (trade_dates[0] if trade_dates else "")),
        "end_date": str(metadata.get("end_date", "") or (trade_dates[-1] if trade_dates else "")),
        "row_count": int(row_counts.get("silver_domain_data", 0) or row_counts.get("bronze_market_data", 0) or 0),
        "trade_date_count": int(len(trade_dates)),
        "trade_dates": list(trade_dates),
    }


def _dataset_trade_dates(*, lake: ResearchDataLake, dataset_id: str, metadata: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    try:
        meta = dict(metadata or lake.describe_dataset(dataset_id))
    except Exception:
        return ()
    paths = dict(meta.get("content_paths", {}) or {})
    manifest_text = str(paths.get("shard_manifest", "") or "").strip()
    manifest_path = Path(manifest_text) if manifest_text else Path()
    dates: set[str] = set()
    if manifest_text and manifest_path.exists():
        manifest = read_json(manifest_path)
        for item in list(manifest.get("shards", []) or []):
            row = dict(item)
            start = str(row.get("start_date", "") or "")
            end = str(row.get("end_date", "") or "")
            if start and end and start == end:
                dates.add(start)
            else:
                for value in (start, end):
                    if value:
                        dates.add(value)
    if not dates:
        frame = _read_dataset_frame(lake=lake, dataset_id=dataset_id, columns=["trade_date"])
        if "trade_date" in frame.columns:
            dates.update(pd.to_datetime(frame["trade_date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").tolist())
    return tuple(sorted(dates))


def _read_dataset_frame(*, lake: ResearchDataLake, dataset_id: str, columns: list[str] | None = None) -> pd.DataFrame:
    dataset_id = str(dataset_id or "").strip()
    if not dataset_id:
        return pd.DataFrame()
    try:
        metadata = lake.describe_dataset(dataset_id)
    except Exception:
        return pd.DataFrame()
    paths = dict(metadata.get("content_paths", {}) or {})
    path_text = str(paths.get("silver_domain_data", "") or paths.get("bronze_market_data", "") or "")
    manifest_text = str(paths.get("shard_manifest", "") or "")
    candidates: list[Path] = []
    if manifest_text and Path(manifest_text).exists():
        manifest = read_json(Path(manifest_text))
        candidates = [
            Path(str(dict(item).get("path", "") or ""))
            for item in list(manifest.get("shards", []) or [])
            if str(dict(item).get("path", "") or "")
        ]
    elif "*" in path_text:
        candidates = sorted(Path(path_text).parent.glob(Path(path_text).name))
    elif path_text:
        candidates = [Path(path_text)]
    frames: list[pd.DataFrame] = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            frames.append(pd.read_parquet(path, columns=columns))
        except Exception:
            try:
                frames.append(pd.read_parquet(path))
            except Exception:
                continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _effective_calendar_frame(
    *,
    lake: ResearchDataLake,
    inherited_dataset_id: str,
    tail_dataset_id: str,
) -> pd.DataFrame:
    frames = [
        _read_dataset_frame(lake=lake, dataset_id=str(inherited_dataset_id or "")),
        _read_dataset_frame(lake=lake, dataset_id=str(tail_dataset_id or "")),
    ]
    frames = [frame for frame in frames if not frame.empty and "trade_date" in frame.columns]
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date"]).sort_values("trade_date")
    if data.empty:
        return pd.DataFrame()
    data = data.drop_duplicates(subset=["trade_date"], keep="last")
    data["trade_date"] = data["trade_date"].dt.strftime("%Y-%m-%d")
    return data.reset_index(drop=True)


def _duplicate_key_report(*, lake: ResearchDataLake, dataset_id: str, max_rows: int) -> dict[str, Any]:
    summary = _dataset_summary(lake=lake, dataset_id=dataset_id)
    row_count = int(summary.get("row_count", 0) or 0)
    if max_rows > 0 and row_count > max_rows:
        return {"status": "skipped_large_dataset", "row_count": row_count, "duplicate_count": 0}
    frame = _read_dataset_frame(lake=lake, dataset_id=dataset_id, columns=["trade_date", "symbol"])
    if frame.empty or "trade_date" not in frame.columns or "symbol" not in frame.columns:
        return {"status": "not_applicable", "row_count": row_count, "duplicate_count": 0}
    duplicate_count = int(frame.duplicated(subset=["trade_date", "symbol"]).sum())
    return {"status": "ok", "row_count": int(len(frame)), "duplicate_count": duplicate_count}


def _resolve_target_date(as_of_date: str, calendar_frame: pd.DataFrame) -> str:
    as_of = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
    if calendar_frame.empty or "trade_date" not in calendar_frame.columns:
        return as_of
    data = calendar_frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date"])
    if data.empty:
        return as_of
    if "is_open" in data.columns:
        open_text = data["is_open"].astype(str).str.lower()
        data = data.loc[data["is_open"].eq(True) | open_text.isin({"1", "true", "t", "open"})]
    if data.empty:
        return as_of
    max_known = data["trade_date"].max()
    as_of_ts = pd.Timestamp(as_of)
    if max_known < as_of_ts:
        return as_of
    eligible = data.loc[data["trade_date"] <= as_of_ts]
    return eligible["trade_date"].max().strftime("%Y-%m-%d") if not eligible.empty else as_of


def _expected_trade_dates(*, calendar_frame: pd.DataFrame, start_date: str, target_date: str) -> tuple[str, ...]:
    if not calendar_frame.empty and "trade_date" in calendar_frame.columns:
        data = calendar_frame.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
        data = data.dropna(subset=["trade_date"])
        if "is_open" in data.columns:
            open_text = data["is_open"].astype(str).str.lower()
            data = data.loc[data["is_open"].eq(True) | open_text.isin({"1", "true", "t", "open"})]
        data = data.loc[(data["trade_date"] >= pd.Timestamp(start_date)) & (data["trade_date"] <= pd.Timestamp(target_date))]
        dates = data["trade_date"].dt.strftime("%Y-%m-%d").drop_duplicates().sort_values().tolist()
        if dates and dates[-1] >= target_date:
            return tuple(dates)
    return tuple(pd.bdate_range(start_date, target_date).strftime("%Y-%m-%d").tolist())


def _gap_start_date(inherited_end: str, fallback_start: str) -> str:
    if not inherited_end:
        return pd.Timestamp(fallback_start).strftime("%Y-%m-%d")
    return (pd.Timestamp(inherited_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _estimate_task_count(
    *,
    domain: str,
    start_date: str,
    end_date: str,
    expected_trade_dates: Sequence[str],
    symbol_count: int,
    chunk_size_symbols: int,
    chunk_size_months: int,
    snapshot_frequency: str,
) -> int:
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        return 0
    if domain == DataDomain.TRADING_CALENDAR:
        return 1
    if domain in POINT_IN_TIME_DAILY_DOMAINS:
        if snapshot_frequency == "daily":
            return len(expected_trade_dates) or len(pd.bdate_range(start_date, end_date))
        return 1
    windows = _month_windows(start_date, end_date, chunk_size_months)
    if domain in SYMBOL_RANGE_DOMAINS:
        chunks = max(1, math.ceil(max(1, symbol_count) / max(1, chunk_size_symbols)))
        return len(windows) * chunks
    return len(windows)


def _month_windows(start_date: str, end_date: str, chunk_size_months: int) -> list[tuple[str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start > end:
        return []
    windows: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        window_end = min(cursor + pd.offsets.MonthEnd(int(chunk_size_months)), end)
        windows.append((cursor.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")))
        cursor = window_end + pd.Timedelta(days=1)
    return windows


def _active_symbol_count(*, lake: ResearchDataLake, cfg: DailyUpdateConfig) -> int:
    if cfg.symbols.lower().startswith("csv:"):
        return len(_parse_csv_symbols(cfg.symbols.split(":", 1)[1]))
    if "," in cfg.symbols:
        return len(_parse_csv_symbols(cfg.symbols))
    try:
        rows = lake.list_datasets(dataset_kind="policy_input_bundle")
        if rows.empty:
            return 1
        latest = rows.iloc[-1].to_dict()
        metadata = lake.describe_dataset(str(latest.get("dataset_id", "") or ""))
        frame = _read_market_symbols_frame(metadata)
        if "symbol" in frame.columns:
            return int(frame["symbol"].astype(str).str.upper().nunique())
    except Exception:
        return 1
    return 1


def _read_market_symbols_frame(metadata: Mapping[str, Any]) -> pd.DataFrame:
    dataset_id = str(metadata.get("dataset_id", "") or "")
    paths = dict(metadata.get("content_paths", {}) or {})
    path_text = str(paths.get("bronze_market_data", "") or paths.get("silver_domain_data", "") or "")
    manifest_text = str(paths.get("bronze_market_shard_manifest", "") or paths.get("shard_manifest", "") or "")
    candidates: list[Path] = []
    if manifest_text and Path(manifest_text).exists():
        manifest = read_json(Path(manifest_text))
        candidates = [
            Path(str(dict(item).get("path", "") or ""))
            for item in list(manifest.get("shards", []) or [])
            if str(dict(item).get("path", "") or "")
        ]
    elif "*" in path_text:
        candidates = sorted(Path(path_text).parent.glob(Path(path_text).name))
    elif path_text:
        candidates = [Path(path_text)]
    frames: list[pd.DataFrame] = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            frames.append(pd.read_parquet(path, columns=["symbol"]))
        except Exception:
            continue
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["symbol", "dataset_id"]).assign(dataset_id=dataset_id)


def _normalize_domains(values: Iterable[str]) -> tuple[str, ...]:
    raw: Iterable[str]
    if isinstance(values, str):
        raw = (item.strip() for item in values.split(",") if item.strip())
    else:
        raw = values
    return tuple(dict.fromkeys(normalize_domain(item) for item in raw if str(item or "").strip()))


def _parse_csv_domains(value: str) -> tuple[str, ...]:
    return _normalize_domains(item.strip() for item in str(value or "").split(",") if item.strip())


def _parse_csv_symbols(value: str) -> tuple[str, ...]:
    return tuple(item.strip().upper() for item in str(value or "").split(",") if item.strip())


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or "").strip()) or "daily_update"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
