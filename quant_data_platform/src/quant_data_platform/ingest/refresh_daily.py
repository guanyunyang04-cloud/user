from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant_data_platform.lake.catalog import ResearchDataLake
from quant_data_platform.domains.contracts import (
    DataDomain,
    DomainFetchRequest,
    FetchRequest,
    NUMERIC_MARKET_COLUMNS,
    STANDARD_MARKET_COLUMNS,
    coverage_report_for_domain,
    market_business_dates,
    next_business_date,
    normalize_domain,
    normalize_domain_frame,
    ProviderResult,
    trading_dates_from_calendar,
)
from quant_data_platform.provider_manager import ProviderManager
from quant_data_platform.providers import build_default_providers, provider_capability_matrix
from quant_data_platform.progress import StageProgress, progress_write


@dataclass(frozen=True)
class RefreshConfig:
    lake_root: Path | str = Path("daily_research/output/research_data_lake")
    as_of_date: str = ""
    start_date: str = ""
    start_date_explicit: bool = False
    symbols: tuple[str, ...] = ()
    universe: str = ""
    domains: tuple[str, ...] = (DataDomain.MARKET_DAILY,)
    required_domains: tuple[str, ...] = (DataDomain.MARKET_DAILY,)
    benchmark: str = "000300.SH"
    provider_plan: str = "default_free"
    adjusted_flag: str = "none"
    min_coverage_ratio: float = 0.80
    conflict_tolerance_pct: float = 0.005
    severe_conflict_limit: int = 0
    allow_tdx_family: bool = False
    run_id: str = ""


@dataclass(frozen=True)
class RefreshResult:
    status: str
    refresh_run_id: str
    manifest_path: str
    bronze_paths: dict[str, str]
    silver_market_path: str
    conflict_report_path: str
    registered_market_dataset_id: str = ""
    registered_domain_dataset_ids: dict[str, str] = field(default_factory=dict)
    coverage_report: dict[str, Any] = field(default_factory=dict)
    conflict_summary: dict[str, Any] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _PolicyInputBundleFrames:
    market_frames: dict[str, pd.DataFrame]
    benchmark_close: pd.Series
    benchmark_open: pd.Series
    membership_frame: pd.DataFrame
    feature_frames: dict[str, pd.DataFrame]
    start_date: str
    end_date: str
    source_market_dataset_id: str = ""
    source_market_dataset_end_date: str = ""


@dataclass(frozen=True)
class _UniverseResolution:
    symbols: tuple[str, ...]
    frame: pd.DataFrame
    status: str
    source: str
    source_dataset_id: str = ""
    source_end_date: str = ""
    provider_error_summary: str = ""

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": self.source,
            "source_dataset_id": self.source_dataset_id,
            "source_end_date": self.source_end_date,
            "symbol_count": int(len(self.symbols)),
            "provider_error_summary": self.provider_error_summary,
        }


def run_refresh(config: RefreshConfig, *, providers: Iterable[Any] | None = None) -> RefreshResult:
    resolved = _resolve_config(config)
    lake = ResearchDataLake(resolved.lake_root)
    refresh_run_id = resolved.run_id or _build_run_id(resolved)
    run_root = Path(resolved.lake_root) / "data_platform" / "runs" / refresh_run_id
    bronze_root = run_root / "bronze"
    silver_root = run_root / "silver"
    bronze_root.mkdir(parents=True, exist_ok=True)
    silver_root.mkdir(parents=True, exist_ok=True)
    domains = tuple(dict.fromkeys(normalize_domain(item) for item in resolved.domains))
    required_domains = set(normalize_domain(item) for item in resolved.required_domains)
    resolved_providers = list(providers) if providers is not None else build_default_providers(resolved.provider_plan)
    provider_chain = [str(getattr(item, "name", "")) for item in resolved_providers]
    manager = ProviderManager(
        resolved_providers,
        allow_tdx_family=resolved.allow_tdx_family,
    )
    progress = StageProgress(total=8, label="DataRefresh")
    progress.__enter__()

    try:
        prefetch_results: dict[str, Any] = {}
        calendar_result = None
        calendar_start = resolved.start_date
        base_for_calendar = _latest_policy_input_bundle_before(lake=lake, as_of_date=resolved.as_of_date)
        if base_for_calendar is not None:
            base_end = str(base_for_calendar.get("end_date", "") or "")
            if base_end and pd.Timestamp(base_end) < pd.Timestamp(calendar_start):
                calendar_start = base_end
        if DataDomain.TRADING_CALENDAR in domains or _needs_calendar(resolved):
            with progress.stage("Fetch trading calendar", f"{calendar_start} -> {resolved.as_of_date}"):
                calendar_result = manager.fetch_domain(
                    DomainFetchRequest(
                        domain=DataDomain.TRADING_CALENDAR,
                        start_date=calendar_start,
                        end_date=resolved.as_of_date,
                    )
                )
            prefetch_results[DataDomain.TRADING_CALENDAR] = calendar_result
        else:
            progress.start_stage(1, "Skip trading calendar", "not requested")
            progress.complete_stage()
        calendar_frame = calendar_result.data if calendar_result is not None else pd.DataFrame()

        universe_result = None
        if not resolved.symbols and _universe_mode(resolved) == "all_a":
            with progress.stage("Fetch universe snapshot", resolved.as_of_date):
                universe_result = manager.fetch_domain(
                    DomainFetchRequest(
                        domain=DataDomain.UNIVERSE_SNAPSHOT,
                        start_date=resolved.start_date,
                        end_date=resolved.as_of_date,
                    )
                )
            prefetch_results[DataDomain.UNIVERSE_SNAPSHOT] = universe_result
        else:
            progress.start_stage(2, "Resolve explicit symbols", f"symbols={len(resolved.symbols)}")
            progress.complete_stage()
        universe_resolution = _resolve_request_symbols(
            lake=lake,
            config=resolved,
            universe_frame=universe_result.data if universe_result is not None else pd.DataFrame(),
            universe_error_report=universe_result.error_report if universe_result is not None else [],
        )
        request_symbols = universe_resolution.symbols
        if (
            universe_result is not None
            and universe_resolution.status == "fallback"
            and not universe_resolution.frame.empty
        ):
            universe_result = ProviderResult(
                provider=universe_resolution.source,
                data=universe_resolution.frame,
                coverage_report={
                    "status": "ok",
                    "domain": DataDomain.UNIVERSE_SNAPSHOT,
                    "row_count": int(len(universe_resolution.frame)),
                    "symbol_count": int(universe_resolution.frame["symbol"].nunique()) if "symbol" in universe_resolution.frame.columns else 0,
                    "trade_date_count": int(universe_resolution.frame["trade_date"].nunique()) if "trade_date" in universe_resolution.frame.columns else 0,
                    "expected_rows": int(len(universe_resolution.frame)),
                    "coverage_ratio": 1.0,
                    "fallback_source": universe_resolution.source,
                    "fallback_source_dataset_id": universe_resolution.source_dataset_id,
                    "fallback_source_date": universe_resolution.source_end_date,
                    "fallback_reason": "provider_universe_snapshot_empty",
                },
                error_report=list(universe_result.error_report or []),
            )
            prefetch_results[DataDomain.UNIVERSE_SNAPSHOT] = universe_result
        if universe_result is not None and DataDomain.SECURITY_STATUS in domains:
            prefetch_results[DataDomain.SECURITY_STATUS] = _derive_security_status_from_universe_result(
                universe_result=universe_result,
                as_of_date=resolved.as_of_date,
            )
        progress_write(f"resolved_symbols={len(request_symbols)}")
        covering_bundle = _find_base_policy_input_bundle(
            lake=lake,
            config=resolved,
            request_symbols=request_symbols,
            min_trading_days=1,
            allow_covering=True,
        )
        covering_end_date = ""
        if covering_bundle is not None:
            covering_end_date = str(
                covering_bundle.get("end_date", "")
                or dict(covering_bundle.get("parameters", {}) or {}).get("end_date", "")
                or ""
            )
        if covering_end_date and pd.Timestamp(covering_end_date) >= pd.Timestamp(resolved.as_of_date):
            with progress.stage("Write skipped manifest", f"already covers {resolved.as_of_date}"):
                manifest_path = run_root / "refresh_manifest.json"
                reused_dataset_id = str(covering_bundle.get("dataset_id", "") or "")
                payload = {
                    "status": "skipped",
                    "refresh_run_id": refresh_run_id,
                    "reason": "lake already covers requested as_of_date",
                    "as_of_date": resolved.as_of_date,
                    "request_start_date": next_business_date(covering_end_date),
                    "domains": list(domains),
                    "provider_chain": provider_chain,
                    "reused_policy_input_dataset_id": reused_dataset_id,
                    "reused_policy_input_dataset_end_date": pd.Timestamp(covering_end_date).strftime("%Y-%m-%d"),
                    "registered_market_dataset_id": reused_dataset_id,
                    "policy_input_dataset_id": reused_dataset_id,
                }
                _write_json(manifest_path, payload)
            return RefreshResult(
                status="skipped",
                refresh_run_id=refresh_run_id,
                manifest_path=str(manifest_path.resolve()),
                bronze_paths={},
                silver_market_path="",
                conflict_report_path="",
                registered_market_dataset_id=reused_dataset_id,
                blockers=[],
            )
        request_start = _resolve_incremental_start(lake, resolved, request_symbols=request_symbols, calendar=calendar_frame)
        if pd.Timestamp(request_start) > pd.Timestamp(resolved.as_of_date):
            with progress.stage("Write skipped manifest", f"already covers {resolved.as_of_date}"):
                manifest_path = run_root / "refresh_manifest.json"
                payload = {
                    "status": "skipped",
                    "refresh_run_id": refresh_run_id,
                    "reason": "lake already covers requested as_of_date",
                    "as_of_date": resolved.as_of_date,
                    "request_start_date": request_start,
                    "domains": list(domains),
                    "provider_chain": provider_chain,
                }
                _write_json(manifest_path, payload)
            return RefreshResult(
                status="skipped",
                refresh_run_id=refresh_run_id,
                manifest_path=str(manifest_path.resolve()),
                bronze_paths={},
                silver_market_path="",
                conflict_report_path="",
                blockers=[],
            )

        request = FetchRequest(
            symbols=request_symbols,
            start_date=request_start,
            end_date=resolved.as_of_date,
            adjusted_flag=resolved.adjusted_flag,
        ).normalized()
        domain_outputs: dict[str, dict[str, Any]] = {}
        provider_error_report: list[dict[str, Any]] = []
        provider_coverage_report: dict[str, Any] = {}
        with progress.stage("Fetch provider domains", f"{request.start_date} -> {request.end_date} domains={len(domains)}"):
            for idx, domain in enumerate(domains, start=1):
                progress_write(f"fetch_domain={domain} ({idx}/{len(domains)})")
                domain_request = _domain_request(
                    domain=domain,
                    symbols=request_symbols,
                    start_date=request_start,
                    end_date=resolved.as_of_date,
                    adjusted_flag=resolved.adjusted_flag,
                )
                provider_result = prefetch_results.get(domain)
                if provider_result is None:
                    provider_result = manager.fetch_domain(domain_request)
                provider_error_report.extend(list(provider_result.error_report or []))
                provider_coverage_report[domain] = provider_result.coverage_report
                domain_bronze_paths = _write_bronze_domain(provider_result.data, bronze_root / domain, provider_chain, domain=domain)
                canonical, conflict_report, conflict_summary = _build_silver_domain(
                    provider_result.data,
                    domain=domain,
                    provider_priority=provider_chain,
                    conflict_tolerance_pct=resolved.conflict_tolerance_pct,
                )
                domain_silver_path = silver_root / f"silver_{domain}.parquet"
                domain_conflict_path = silver_root / f"source_conflict_report_{domain}.parquet"
                canonical.to_parquet(domain_silver_path, index=False)
                conflict_report.to_parquet(domain_conflict_path, index=False)
                coverage = _domain_coverage_report(
                    canonical,
                    domain_request,
                    calendar=calendar_frame,
                    market_coverage_basis="observed_symbol_lifecycle" if _universe_mode(resolved) == "all_a" else "raw_full_grid",
                )
                coverage.update(_domain_coverage_metadata(provider_result.coverage_report))
                domain_outputs[domain] = {
                    "provider_result": provider_result,
                    "bronze_paths": domain_bronze_paths,
                    "canonical": canonical,
                    "conflict_report": conflict_report,
                    "conflict_summary": conflict_summary,
                    "coverage_report": coverage,
                    "silver_path": str(domain_silver_path.resolve()),
                    "conflict_report_path": str(domain_conflict_path.resolve()),
                }

        with progress.stage("Validate refresh outputs", "coverage and required domains"):
            market_output = domain_outputs.get(DataDomain.MARKET_DAILY, {})
            canonical = market_output.get("canonical", pd.DataFrame(columns=STANDARD_MARKET_COLUMNS))
            conflict_summary = dict(market_output.get("conflict_summary", {}))
            coverage_report = dict(market_output.get("coverage_report", {}))
            bronze_paths = dict(market_output.get("bronze_paths", {}))
            silver_market_path = Path(str(market_output.get("silver_path", silver_root / "silver_market_daily.parquet")))
            conflict_report_path = Path(str(market_output.get("conflict_report_path", silver_root / "source_conflict_report_market_daily.parquet")))
            blockers = _refresh_blockers(
                coverage_report=coverage_report,
                conflict_summary=conflict_summary,
                min_coverage_ratio=resolved.min_coverage_ratio,
                severe_conflict_limit=resolved.severe_conflict_limit,
            )
            for domain in required_domains:
                output = domain_outputs.get(domain)
                if output is None or int(output.get("coverage_report", {}).get("row_count", 0) or 0) <= 0:
                    blockers.append(f"required_domain_blocked:{domain}")
            if resolved.benchmark and (
                canonical.empty
                or not canonical["symbol"].astype(str).str.upper().eq(str(resolved.benchmark).upper()).any()
            ):
                blockers.append("missing_benchmark")
        registered_dataset_id = ""
        registered_domain_dataset_ids: dict[str, str] = {}
        status = "blocked" if blockers else "ok"
        if not blockers:
            with progress.stage("Register sidecar datasets", ",".join(domain for domain in domains if domain != DataDomain.MARKET_DAILY)):
                registered_domain_dataset_ids = _register_sidecar_domain_datasets(
                    lake=lake,
                    config=resolved,
                    refresh_run_id=refresh_run_id,
                    provider_chain=provider_chain,
                    domain_outputs=domain_outputs,
                )
            with progress.stage("Extend policy input bundle", f"{request.start_date} -> {resolved.as_of_date}"):
                registered_dataset_id = _register_policy_input_bundle(
                    lake=lake,
                    config=resolved,
                    refresh_run_id=refresh_run_id,
                    request_symbols=request_symbols,
                    canonical=canonical,
                    provider_chain=provider_chain,
                    coverage_report=coverage_report,
                    conflict_summary=conflict_summary,
                    domain_outputs=domain_outputs,
                    registered_domain_dataset_ids=registered_domain_dataset_ids,
                )
                progress_write(f"registered_market_dataset_id={registered_dataset_id}")
        else:
            progress.start_stage(6, "Skip lake registration", ",".join(blockers))
            progress.complete_stage()
            progress.start_stage(7, "Skip policy input bundle", "refresh blocked")
            progress.complete_stage()
        with progress.stage("Write refresh manifest", status):
            manifest_path = run_root / "refresh_manifest.json"
            domain_quality_status = _domain_quality_status(domain_outputs=domain_outputs, required_domains=required_domains)
            source_provenance = _source_provenance(domain_outputs)
            provider_health_summary = _provider_health_summary_from_refresh(
                provider_chain=provider_chain,
                provider_coverage_report=provider_coverage_report,
                provider_error_report=provider_error_report,
            )
            manifest = {
                "schema_version": 1,
                "status": status,
                "refresh_run_id": refresh_run_id,
                "provider_plan": resolved.provider_plan,
                "provider_chain": provider_chain,
                "domains": list(domains),
                "required_domains": sorted(required_domains),
                "universe": resolved.universe,
                "as_of_date": resolved.as_of_date,
                "request_start_date": request.start_date,
                "request_end_date": request.end_date,
                "symbols": list(request.symbols),
                "benchmark": resolved.benchmark,
                "adjusted_flag": resolved.adjusted_flag,
                "requested_symbols": list(request.symbols),
                "refresh_universe_key": _universe_key(request.symbols),
                "universe_resolution": universe_resolution.manifest_payload(),
                "calendar_source": "provider" if not calendar_frame.empty else "business_day_fallback",
                "bronze_paths": bronze_paths,
                "domain_outputs": {
                    domain: {
                        "bronze_paths": dict(output["bronze_paths"]),
                        "silver_path": str(output["silver_path"]),
                        "conflict_report_path": str(output["conflict_report_path"]),
                        "coverage_report": dict(output["coverage_report"]),
                        "conflict_summary": dict(output["conflict_summary"]),
                    }
                    for domain, output in domain_outputs.items()
                },
                "silver_market_path": str(silver_market_path),
                "conflict_report_path": str(conflict_report_path),
                "coverage_report": coverage_report,
                "conflict_summary": conflict_summary,
                "provider_coverage_report": provider_coverage_report,
                "provider_error_report": provider_error_report,
                "provider_health_summary": provider_health_summary,
                "domain_quality_status": domain_quality_status,
                "source_provenance": source_provenance,
                "domain_matrix": provider_capability_matrix(resolved.provider_plan),
                "blockers": blockers,
                "registered_market_dataset_id": registered_dataset_id,
                "registered_domain_dataset_ids": registered_domain_dataset_ids,
            }
            _write_json(manifest_path, manifest)
            lake.write_catalog_manifest()
        progress.complete()
        return RefreshResult(
            status=status,
            refresh_run_id=refresh_run_id,
            manifest_path=str(manifest_path.resolve()),
            bronze_paths=bronze_paths,
            silver_market_path=str(silver_market_path.resolve()),
            conflict_report_path=str(conflict_report_path.resolve()),
            registered_market_dataset_id=registered_dataset_id,
            registered_domain_dataset_ids=registered_domain_dataset_ids,
            coverage_report=coverage_report,
            conflict_summary=conflict_summary,
            blockers=blockers,
        )
    finally:
        progress.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh QDP market data lake from configured providers.")
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--provider-plan", default="default_free")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols. Optional when --universe is set.")
    parser.add_argument("--universe", default="", help="all_a, liquid500, file:<path>, or symbols:<csv>.")
    parser.add_argument("--domains", default=DataDomain.MARKET_DAILY)
    parser.add_argument("--required-domains", default=DataDomain.MARKET_DAILY)
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--data-lake-root", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--adjusted-flag", default="none")
    parser.add_argument("--min-coverage-ratio", type=float, default=0.80)
    parser.add_argument("--conflict-tolerance-pct", type=float, default=0.005)
    parser.add_argument("--severe-conflict-limit", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = tuple(item.strip().upper() for item in str(args.symbols or "").split(",") if item.strip())
    if not symbols and not str(args.universe or "").strip():
        raise SystemExit("Either --symbols or --universe is required for TDX-free refresh.")
    result = run_refresh(
        RefreshConfig(
            lake_root=Path(args.data_lake_root) if str(args.data_lake_root or "").strip() else Path("daily_research/output/research_data_lake"),
            as_of_date=args.as_of_date,
            start_date=args.start_date,
            symbols=symbols,
            universe=args.universe,
            domains=_parse_domains(args.domains),
            required_domains=_parse_domains(args.required_domains),
            benchmark=args.benchmark,
            provider_plan=args.provider_plan,
            adjusted_flag=args.adjusted_flag,
            min_coverage_ratio=float(args.min_coverage_ratio),
            conflict_tolerance_pct=float(args.conflict_tolerance_pct),
            severe_conflict_limit=int(args.severe_conflict_limit),
            run_id=str(args.run_id or ""),
        )
    )
    payload = {
        "status": result.status,
        "refresh_run_id": result.refresh_run_id,
        "manifest_path": result.manifest_path,
        "registered_market_dataset_id": result.registered_market_dataset_id,
        "blockers": result.blockers,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{result.status}: {result.manifest_path}")
    return 0 if result.status in {"ok", "skipped"} else 2


def _resolve_config(config: RefreshConfig) -> RefreshConfig:
    as_of = pd.Timestamp(config.as_of_date).strftime("%Y-%m-%d")
    start = pd.Timestamp(config.start_date or as_of).strftime("%Y-%m-%d")
    domains = tuple(dict.fromkeys(normalize_domain(item) for item in (config.domains or (DataDomain.MARKET_DAILY,))))
    required_domains = tuple(dict.fromkeys(normalize_domain(item) for item in (config.required_domains or (DataDomain.MARKET_DAILY,))))
    return RefreshConfig(
        lake_root=Path(config.lake_root),
        as_of_date=as_of,
        start_date=start,
        start_date_explicit=bool(str(config.start_date or "").strip()) or bool(config.start_date_explicit),
        symbols=tuple(str(item).strip().upper() for item in config.symbols if str(item).strip()),
        universe=str(config.universe or "").strip(),
        domains=domains,
        required_domains=required_domains,
        benchmark=str(config.benchmark or "000300.SH").strip().upper(),
        provider_plan=str(config.provider_plan or "default_free"),
        adjusted_flag=str(config.adjusted_flag or "none"),
        min_coverage_ratio=float(config.min_coverage_ratio),
        conflict_tolerance_pct=float(config.conflict_tolerance_pct),
        severe_conflict_limit=int(config.severe_conflict_limit),
        allow_tdx_family=bool(config.allow_tdx_family),
        run_id=str(config.run_id or ""),
    )


def _build_run_id(config: RefreshConfig) -> str:
    return f"refresh_daily_{config.provider_plan}_{config.as_of_date.replace('-', '')}_{pd.Timestamp.now().strftime('%H%M%S')}"


def _resolve_incremental_start(
    lake: ResearchDataLake,
    config: RefreshConfig,
    *,
    request_symbols: tuple[str, ...],
    calendar: pd.DataFrame,
) -> str:
    latest = ""
    base = _find_base_policy_input_bundle(
        lake=lake,
        config=config,
        request_symbols=request_symbols,
        min_trading_days=1,
    )
    if base is not None:
        latest = str(base.get("end_date", "") or dict(base.get("parameters", {}) or {}).get("end_date", "") or "")
        latest = pd.Timestamp(latest).strftime("%Y-%m-%d") if latest else ""
    if latest:
        if calendar is not None and not calendar.empty:
            dates = trading_dates_from_calendar(calendar, latest, config.as_of_date)
            future_dates = [item for item in dates if pd.Timestamp(item) > pd.Timestamp(latest)]
            if future_dates:
                next_start = future_dates[0]
            else:
                next_start = next_business_date(config.as_of_date)
        else:
            next_start = next_business_date(latest)
        if config.start_date_explicit and pd.Timestamp(config.start_date) > pd.Timestamp(next_start):
            return config.start_date
        return next_start
    return config.start_date


def _find_base_policy_input_bundle(
    *,
    lake: ResearchDataLake,
    config: RefreshConfig,
    request_symbols: tuple[str, ...],
    min_trading_days: int,
    allow_covering: bool = False,
) -> dict[str, Any] | None:
    requested = {str(item).strip().upper() for item in request_symbols if str(item).strip()}
    benchmark = str(config.benchmark or "").strip().upper()
    if benchmark:
        requested.add(benchmark)
    try:
        rows = lake.list_datasets(dataset_kind="policy_input_bundle")
    except Exception:
        return None
    if rows.empty:
        return None
    candidates: list[tuple[pd.Timestamp, int, dict[str, Any]]] = []
    for _, row in rows.iterrows():
        try:
            metadata = lake.describe_dataset(str(row["dataset_id"]))
        except Exception:
            continue
        end_date = str(metadata.get("end_date", "") or dict(metadata.get("parameters", {}) or {}).get("end_date", "") or "")
        if not end_date:
            continue
        if not allow_covering and pd.Timestamp(end_date) >= pd.Timestamp(config.as_of_date):
            continue
        paths = dict(metadata.get("content_paths", {}) or {})
        market_path = str(paths.get("bronze_market_data", "") or "")
        if not market_path or not Path(market_path).exists():
            continue
        try:
            market = pd.read_parquet(market_path, columns=["trade_date", "symbol"])
        except Exception:
            continue
        if market.empty:
            continue
        market["symbol"] = market["symbol"].astype(str).str.strip().str.upper()
        available = set(market["symbol"].dropna())
        non_benchmark_requested = {item for item in requested if item != benchmark}
        if non_benchmark_requested and not non_benchmark_requested.issubset(available):
            if not _can_extend_partial_base_for_universe(config=config, requested=non_benchmark_requested, available=available):
                continue
        dates = pd.Index(pd.to_datetime(market["trade_date"], errors="coerce").dropna().unique())
        if len(dates) < int(min_trading_days or 1):
            continue
        candidates.append((pd.Timestamp(end_date), int(len(dates)), metadata))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2].get("created_at", ""))))
    return candidates[-1][2]


def _latest_policy_input_bundle_before(*, lake: ResearchDataLake, as_of_date: str) -> dict[str, Any] | None:
    try:
        rows = lake.list_datasets(dataset_kind="policy_input_bundle")
    except Exception:
        return None
    if rows.empty:
        return None
    candidates: list[tuple[pd.Timestamp, dict[str, Any]]] = []
    for _, row in rows.iterrows():
        try:
            metadata = lake.describe_dataset(str(row["dataset_id"]))
        except Exception:
            continue
        end_date = str(metadata.get("end_date", "") or dict(metadata.get("parameters", {}) or {}).get("end_date", "") or "")
        if not end_date:
            continue
        if pd.Timestamp(end_date) < pd.Timestamp(as_of_date):
            candidates.append((pd.Timestamp(end_date), metadata))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], str(item[1].get("created_at", ""))))
    return candidates[-1][1]


def _can_extend_partial_base_for_universe(*, config: RefreshConfig, requested: set[str], available: set[str]) -> bool:
    if _universe_mode(config) != "all_a":
        return False
    if not requested or not available:
        return False
    return bool(requested & available)


def _parse_domains(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",") if item.strip()]
    else:
        raw = [str(item).strip() for item in value if str(item).strip()]
    return tuple(dict.fromkeys(normalize_domain(item) for item in (raw or [DataDomain.MARKET_DAILY])))


def _needs_calendar(config: RefreshConfig) -> bool:
    return bool(config.universe) or DataDomain.TRADING_CALENDAR in set(config.domains or ())


def _universe_mode(config: RefreshConfig) -> str:
    raw = str(config.universe or "").strip()
    if not raw:
        return "symbols"
    if raw.lower().startswith("file:"):
        return "file"
    if raw.lower().startswith("symbols:"):
        return "symbols_inline"
    return raw.lower()


def _resolve_request_symbols(
    *,
    lake: ResearchDataLake,
    config: RefreshConfig,
    universe_frame: pd.DataFrame,
    universe_error_report: Iterable[dict[str, Any]] | None = None,
) -> _UniverseResolution:
    if config.symbols:
        symbols = _request_symbols(config)
        return _UniverseResolution(
            symbols=symbols,
            frame=pd.DataFrame(),
            status="explicit",
            source="explicit_symbols",
        )
    mode = _universe_mode(config)
    benchmark = str(config.benchmark or "").strip().upper()
    symbols: list[str] = []
    frame = pd.DataFrame()
    status = "provider"
    source = mode
    source_dataset_id = ""
    source_end_date = ""
    provider_error_summary = ""
    if mode == "all_a":
        if universe_frame is None or universe_frame.empty or "symbol" not in universe_frame.columns:
            provider_error_summary = _format_provider_errors(universe_error_report or [])
            fallback = _latest_universe_snapshot_before(lake=lake, as_of_date=config.as_of_date)
            if fallback is None:
                suffix = f"; provider_errors={provider_error_summary}" if provider_error_summary else ""
                raise ValueError(f"universe=all_a requires a non-empty universe_snapshot provider result{suffix}")
            frame = fallback["frame"]
            status = "fallback"
            source = "lake_universe_snapshot"
            source_dataset_id = str(fallback.get("dataset_id", "") or "")
            source_end_date = str(fallback.get("end_date", "") or "")
        else:
            frame = universe_frame
            source = "provider_universe_snapshot"
            source_end_date = str(frame["trade_date"].max()) if "trade_date" in frame.columns and not frame.empty else config.as_of_date
        statuses = frame["list_status"].fillna("").astype(str).str.upper() if "list_status" in frame.columns else pd.Series(["L"] * len(frame))
        symbols = [str(item).strip().upper() for item in frame.loc[statuses.isin({"", "L", "LIST", "上市"}), "symbol"] if str(item).strip()]
    elif mode == "liquid500":
        symbols = _resolve_liquid_universe_from_lake(lake=lake, benchmark=benchmark, limit=500)
        status = "lake"
        source = "lake_liquid500"
    elif mode == "file":
        path = Path(str(config.universe).split(":", 1)[1]).expanduser()
        if not path.exists():
            raise ValueError(f"universe file does not exist: {path}")
        text = path.read_text(encoding="utf-8")
        symbols = [item.strip().upper() for item in text.replace("\n", ",").split(",") if item.strip()]
        status = "file"
        source = str(path)
    elif mode == "symbols_inline":
        symbols = [item.strip().upper() for item in str(config.universe).split(":", 1)[1].split(",") if item.strip()]
        status = "explicit"
        source = "inline_symbols"
    else:
        raise ValueError(f"unsupported universe: {config.universe}")
    normalized_symbols = FetchRequest(
        symbols=tuple(dict.fromkeys([*symbols, benchmark])),
        start_date=config.start_date,
        end_date=config.as_of_date,
        adjusted_flag=config.adjusted_flag,
    ).normalized().symbols
    return _UniverseResolution(
        symbols=normalized_symbols,
        frame=frame,
        status=status,
        source=source,
        source_dataset_id=source_dataset_id,
        source_end_date=source_end_date,
        provider_error_summary=provider_error_summary,
    )


def _latest_universe_snapshot_before(*, lake: ResearchDataLake, as_of_date: str) -> dict[str, Any] | None:
    try:
        rows = lake.list_datasets(dataset_kind=f"data_platform_{DataDomain.UNIVERSE_SNAPSHOT}")
    except Exception:
        return None
    if rows.empty:
        return None
    candidates: list[tuple[pd.Timestamp, str, pd.DataFrame]] = []
    for _, row in rows.iterrows():
        dataset_id = str(row.get("dataset_id", "") or "")
        try:
            metadata = lake.describe_dataset(dataset_id)
        except Exception:
            continue
        end_date = str(metadata.get("end_date", "") or dict(metadata.get("parameters", {}) or {}).get("end_date", "") or "")
        if not end_date or pd.Timestamp(end_date) >= pd.Timestamp(as_of_date):
            continue
        path = str(dict(metadata.get("content_paths", {}) or {}).get("silver_domain_data", "") or "")
        if not path or not Path(path).exists():
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        frame = normalize_domain_frame(
            frame,
            domain=DataDomain.UNIVERSE_SNAPSHOT,
            source="lake_universe_snapshot",
            as_of_date=end_date,
            require_columns=False,
        )
        if frame.empty or "symbol" not in frame.columns:
            continue
        candidates.append((pd.Timestamp(end_date), dataset_id, frame))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]))
        end_ts, dataset_id, frame = candidates[-1]
        return {
            "dataset_id": dataset_id,
            "end_date": end_ts.strftime("%Y-%m-%d"),
            "frame": frame,
        }
    return _latest_universe_from_policy_input_before(lake=lake, as_of_date=as_of_date)


def _latest_universe_from_policy_input_before(*, lake: ResearchDataLake, as_of_date: str) -> dict[str, Any] | None:
    metadata = _latest_policy_input_bundle_before(lake=lake, as_of_date=as_of_date)
    if metadata is None:
        return None
    end_date = str(metadata.get("end_date", "") or dict(metadata.get("parameters", {}) or {}).get("end_date", "") or "")
    market_path = str(dict(metadata.get("content_paths", {}) or {}).get("bronze_market_data", "") or "")
    if not end_date or not market_path or not Path(market_path).exists():
        return None
    try:
        market = pd.read_parquet(market_path, columns=["symbol"])
    except Exception:
        return None
    symbols = [str(item).strip().upper() for item in market["symbol"].dropna().unique() if str(item).strip()]
    if not symbols:
        return None
    frame = pd.DataFrame(
        {
            "symbol": symbols,
            "trade_date": [end_date] * len(symbols),
            "name": [""] * len(symbols),
            "exchange": [item.rsplit(".", 1)[-1] if "." in item else "" for item in symbols],
            "board": [""] * len(symbols),
            "list_status": ["L"] * len(symbols),
            "list_date": [""] * len(symbols),
            "delist_date": [""] * len(symbols),
            "source": ["lake_policy_input_bundle"] * len(symbols),
        }
    )
    return {
        "dataset_id": str(metadata.get("dataset_id", "") or ""),
        "end_date": end_date,
        "frame": normalize_domain_frame(
            frame,
            domain=DataDomain.UNIVERSE_SNAPSHOT,
            source="lake_policy_input_bundle",
            as_of_date=end_date,
            require_columns=False,
        ),
    }


def _format_provider_errors(error_report: Iterable[dict[str, Any]]) -> str:
    items: list[str] = []
    for item in error_report:
        provider = str(item.get("provider", "") or "provider")
        code = str(item.get("code", "") or "error")
        error_type = str(item.get("error_type", "") or "").strip()
        message = str(item.get("message", "") or "").strip()
        label = f"{provider}:{code}"
        if error_type:
            label = f"{label}:{error_type}"
        if message:
            label = f"{label}:{message}"
        items.append(label)
    return " | ".join(items[:8])


def _resolve_liquid_universe_from_lake(*, lake: ResearchDataLake, benchmark: str, limit: int) -> list[str]:
    try:
        rows = lake.list_datasets(dataset_kind="policy_input_bundle")
    except Exception:
        rows = pd.DataFrame()
    if rows.empty:
        raise ValueError("universe=liquid500 requires an existing policy_input_bundle lake dataset")
    latest = rows.iloc[-1]
    metadata = lake.describe_dataset(str(latest["dataset_id"]))
    market_path = str(dict(metadata.get("content_paths", {}) or {}).get("bronze_market_data", ""))
    if not market_path or not Path(market_path).exists():
        raise ValueError("universe=liquid500 could not find bronze_market_data in latest policy_input_bundle")
    market = pd.read_parquet(market_path)
    if market.empty or "field" not in market.columns or "value" not in market.columns:
        raise ValueError("universe=liquid500 market bundle has no long market field/value data")
    amount = market.loc[market["field"].astype(str).str.lower() == "amount"].copy()
    if amount.empty:
        raise ValueError("universe=liquid500 requires amount field in lake market bundle")
    ranked = amount.groupby("stock")["value"].mean().sort_values(ascending=False)
    symbols = [str(item).upper() for item in ranked.index if str(item).upper() != benchmark][: int(limit)]
    if len(symbols) < min(int(limit), 50):
        raise ValueError("universe=liquid500 has insufficient liquid symbols")
    return symbols


def _domain_request(
    *,
    domain: str,
    symbols: tuple[str, ...],
    start_date: str,
    end_date: str,
    adjusted_flag: str,
) -> DomainFetchRequest:
    domain = normalize_domain(domain)
    if domain in {DataDomain.TRADING_CALENDAR, DataDomain.UNIVERSE_SNAPSHOT, DataDomain.INDEX_CONSTITUENTS}:
        request_symbols: tuple[str, ...] = ()
    else:
        request_symbols = symbols
    return DomainFetchRequest(
        domain=domain,
        symbols=request_symbols,
        start_date=start_date,
        end_date=end_date,
        adjusted_flag=adjusted_flag,
    ).normalized()


def _derive_security_status_from_universe_result(*, universe_result: ProviderResult, as_of_date: str) -> ProviderResult:
    universe = normalize_domain_frame(
        universe_result.data,
        domain=DataDomain.UNIVERSE_SNAPSHOT,
        source=str(universe_result.provider or "provider_manager"),
        as_of_date=as_of_date,
        require_columns=False,
    )
    if universe.empty:
        status_frame = pd.DataFrame()
    else:
        name_upper = universe["name"].fillna("").astype(str).str.upper() if "name" in universe.columns else pd.Series("", index=universe.index)
        list_status = universe["list_status"].fillna("").astype(str).str.upper() if "list_status" in universe.columns else pd.Series("", index=universe.index)
        status_frame = pd.DataFrame(
            {
                "symbol": universe["symbol"].astype(str),
                "trade_date": as_of_date,
                "is_st": name_upper.str.startswith(("ST", "*ST")),
                "is_suspended": False,
                "is_delisted": list_status.isin({"D", "DELIST", "0", "退市"}),
                "status_reason": universe["list_status"].fillna("").astype(str) if "list_status" in universe.columns else "",
                "source": "universe_snapshot_derived",
            }
        )
    normalized = normalize_domain_frame(
        status_frame,
        domain=DataDomain.SECURITY_STATUS,
        source="universe_snapshot_derived",
        as_of_date=as_of_date,
        require_columns=False,
    )
    report_request = DomainFetchRequest(
        domain=DataDomain.SECURITY_STATUS,
        symbols=tuple(normalized["symbol"].dropna().astype(str).unique()) if not normalized.empty else (),
        start_date=as_of_date,
        end_date=as_of_date,
    )
    coverage = coverage_report_for_domain(normalized, report_request, provider="universe_snapshot_derived")
    return ProviderResult(
        provider="universe_snapshot_derived",
        data=normalized,
        coverage_report={
            **coverage,
            "derivation_source_domain": DataDomain.UNIVERSE_SNAPSHOT,
            "derivation_note": "derived from the same stock-basic universe snapshot to avoid duplicate baostock query_stock_basic calls",
        },
        error_report=list(universe_result.error_report or []),
    )


def _write_bronze(data: pd.DataFrame, bronze_root: Path, provider_chain: list[str]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for provider in provider_chain:
        safe = _safe_name(provider)
        path = bronze_root / f"{safe}.parquet"
        subset = data.loc[data["source"].astype(str).str.lower() == str(provider).lower()].copy() if not data.empty else pd.DataFrame(columns=STANDARD_MARKET_COLUMNS)
        subset.to_parquet(path, index=False)
        paths[str(provider)] = str(path.resolve())
    return paths


def _write_bronze_domain(data: pd.DataFrame, bronze_root: Path, provider_chain: list[str], *, domain: str) -> dict[str, str]:
    bronze_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    columns = data.columns if not data.empty else []
    providers = list(dict.fromkeys([*provider_chain, *_data_sources(data)]))
    for provider in providers:
        safe = _safe_name(provider)
        path = bronze_root / f"{safe}.parquet"
        if not data.empty and "source" in data.columns:
            subset = data.loc[data["source"].astype(str).str.lower() == str(provider).lower()].copy()
        else:
            subset = pd.DataFrame(columns=columns)
        subset.to_parquet(path, index=False)
        paths[str(provider)] = str(path.resolve())
    return paths


def _data_sources(data: pd.DataFrame) -> list[str]:
    if data is None or data.empty or "source" not in data.columns:
        return []
    return [str(item).strip() for item in data["source"].dropna().astype(str).unique() if str(item).strip()]


def _build_silver_domain(
    data: pd.DataFrame,
    *,
    domain: str,
    provider_priority: list[str],
    conflict_tolerance_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    domain = normalize_domain(domain)
    if domain == DataDomain.MARKET_DAILY:
        return _build_silver_market(
            data,
            provider_priority=provider_priority,
            conflict_tolerance_pct=conflict_tolerance_pct,
        )
    canonical = normalize_domain_frame(
        data,
        domain=domain,
        source=str(data["source"].iloc[0] if "source" in data.columns and len(data) else "unknown"),
        as_of_date="",
        require_columns=False,
    )
    if canonical.empty:
        return canonical, pd.DataFrame(columns=["trade_date", "symbol", "field", "min_value", "max_value", "relative_diff", "providers"]), {
            "source_conflict_count": 0,
            "severe_conflict_count": 0,
            "conflict_tolerance_pct": float(conflict_tolerance_pct),
        }
    priority = {str(provider).lower(): idx for idx, provider in enumerate(provider_priority)}
    working = canonical.copy()
    working["_priority"] = working["source"].astype(str).str.lower().map(priority).fillna(9999).astype(int)
    if domain == DataDomain.TRADING_CALENDAR:
        keys = ["trade_date", "exchange"]
    elif domain == DataDomain.MARKET_INTRADAY_5M:
        keys = ["trade_date", "symbol", "bar_time"]
    elif domain == DataDomain.INDEX_CONSTITUENTS:
        keys = ["trade_date", "index_symbol", "symbol"]
    elif domain in {DataDomain.FINANCIAL_QUARTERLY, DataDomain.PERFORMANCE_FORECAST, DataDomain.PERFORMANCE_EXPRESS}:
        keys = ["trade_date", "symbol", "fiscal_year", "fiscal_quarter"]
    else:
        keys = ["trade_date", "symbol"]
    keys = [key for key in keys if key in working.columns]
    ordered = working.sort_values([*keys, "_priority"], ascending=True)
    deduped = ordered.drop_duplicates(subset=keys, keep="first").drop(columns=["_priority"], errors="ignore")
    conflicts = _domain_conflicts(working.drop(columns=["_priority"], errors="ignore"), keys=keys, tolerance=conflict_tolerance_pct)
    conflict_report = pd.DataFrame(conflicts)
    severe = int(len(conflict_report))
    return deduped.reset_index(drop=True), conflict_report, {
        "source_conflict_count": severe,
        "severe_conflict_count": severe,
        "conflict_tolerance_pct": float(conflict_tolerance_pct),
    }


def _domain_conflicts(data: pd.DataFrame, *, keys: list[str], tolerance: float) -> list[dict[str, Any]]:
    if data.empty or len(keys) == 0 or "source" not in data.columns:
        return []
    numeric_columns = [column for column in data.columns if column not in {*keys, "source"} and pd.api.types.is_numeric_dtype(pd.to_numeric(data[column], errors="coerce"))]
    conflicts: list[dict[str, Any]] = []
    for key_values, group in data.groupby(keys):
        if len(group) <= 1:
            continue
        providers = ",".join(sorted(group["source"].astype(str).unique()))
        key_tuple = key_values if isinstance(key_values, tuple) else (key_values,)
        base = {key: str(value) for key, value in zip(keys, key_tuple)}
        for field in numeric_columns:
            values = pd.to_numeric(group[field], errors="coerce").dropna()
            if len(values) <= 1:
                continue
            min_value = float(values.min())
            max_value = float(values.max())
            denom = max(abs(min_value), 1e-12)
            relative_diff = abs(max_value - min_value) / denom
            if relative_diff > float(tolerance):
                conflicts.append(
                    {
                        **base,
                        "field": field,
                        "min_value": min_value,
                        "max_value": max_value,
                        "relative_diff": float(relative_diff),
                        "providers": providers,
                    }
                )
    return conflicts


def _build_silver_market(
    data: pd.DataFrame,
    *,
    provider_priority: list[str],
    conflict_tolerance_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if data.empty:
        empty = pd.DataFrame(columns=STANDARD_MARKET_COLUMNS)
        return empty, pd.DataFrame(columns=["trade_date", "symbol", "field", "min_value", "max_value", "relative_diff", "providers"]), {
            "source_conflict_count": 0,
            "severe_conflict_count": 0,
        }
    working = data.copy()
    priority = {str(provider).lower(): idx for idx, provider in enumerate(provider_priority)}
    working["_priority"] = working["source"].astype(str).str.lower().map(priority).fillna(9999).astype(int)
    working["_valid"] = _valid_mask(working)
    ordered = working.sort_values(["trade_date", "symbol", "_valid", "_priority"], ascending=[True, True, False, True])
    canonical = ordered.drop_duplicates(subset=["trade_date", "symbol"], keep="first").copy()
    canonical = canonical.drop(columns=["_priority", "_valid"], errors="ignore")
    conflicts: list[dict[str, Any]] = []
    for (trade_date, symbol), group in working.groupby(["trade_date", "symbol"]):
        if len(group) <= 1:
            continue
        providers = ",".join(sorted(group["source"].astype(str).unique()))
        for field in ["open", "high", "low", "close", "volume", "amount"]:
            values = pd.to_numeric(group[field], errors="coerce").dropna()
            if len(values) <= 1:
                continue
            min_value = float(values.min())
            max_value = float(values.max())
            denom = max(abs(min_value), 1e-12)
            relative_diff = abs(max_value - min_value) / denom
            if relative_diff > float(conflict_tolerance_pct):
                conflicts.append(
                    {
                        "trade_date": str(trade_date),
                        "symbol": str(symbol),
                        "field": field,
                        "min_value": min_value,
                        "max_value": max_value,
                        "relative_diff": float(relative_diff),
                        "providers": providers,
                    }
                )
    conflict_report = pd.DataFrame(conflicts)
    severe = int(len(conflict_report))
    return canonical[STANDARD_MARKET_COLUMNS].sort_values(["trade_date", "symbol"]).reset_index(drop=True), conflict_report, {
        "source_conflict_count": severe,
        "severe_conflict_count": severe,
        "conflict_tolerance_pct": float(conflict_tolerance_pct),
    }


def _coverage_report(
    canonical: pd.DataFrame,
    request: FetchRequest,
    *,
    calendar: pd.DataFrame | None = None,
    coverage_basis: str = "raw_full_grid",
) -> dict[str, Any]:
    expected_dates = trading_dates_from_calendar(calendar, request.start_date, request.end_date) if calendar is not None and not calendar.empty else market_business_dates(request.start_date, request.end_date)
    row_count = int(len(canonical))
    raw_expected_rows = int(len(request.symbols) * len(expected_dates))
    resolved_basis = str(coverage_basis or "raw_full_grid")
    if resolved_basis == "observed_symbol_lifecycle":
        expected_rows = _observed_symbol_lifecycle_expected_rows(canonical, expected_dates)
    else:
        expected_rows = raw_expected_rows
    if expected_rows <= 0:
        expected_rows = raw_expected_rows
        resolved_basis = "raw_full_grid"
    row_coverage_ratio = float(row_count / expected_rows) if expected_rows else 0.0
    observed_symbols = int(canonical["symbol"].nunique()) if not canonical.empty else 0
    requested_symbol_count = int(len(set(str(item).strip().upper() for item in request.symbols if str(item).strip())))
    symbol_coverage_ratio = float(observed_symbols / requested_symbol_count) if requested_symbol_count else 0.0
    coverage_ratio = min(row_coverage_ratio, symbol_coverage_ratio) if resolved_basis == "observed_symbol_lifecycle" else row_coverage_ratio
    raw_coverage_ratio = float(row_count / raw_expected_rows) if raw_expected_rows else 0.0
    return {
        "status": "ok" if row_count and coverage_ratio >= 1.0 else "partial" if row_count else "empty",
        "start_date": request.start_date,
        "end_date": request.end_date,
        "expected_rows": expected_rows,
        "row_count": row_count,
        "coverage_ratio": coverage_ratio,
        "row_coverage_ratio": row_coverage_ratio,
        "requested_symbol_count": requested_symbol_count,
        "symbol_coverage_ratio": symbol_coverage_ratio,
        "coverage_basis": resolved_basis,
        "raw_full_grid_expected_rows": raw_expected_rows,
        "raw_full_grid_coverage_ratio": raw_coverage_ratio,
        "symbol_count": observed_symbols,
        "trade_date_count": int(canonical["trade_date"].nunique()) if not canonical.empty else 0,
    }


def _observed_symbol_lifecycle_expected_rows(canonical: pd.DataFrame, expected_dates: list[str]) -> int:
    if canonical is None or canonical.empty or not expected_dates:
        return 0
    if "symbol" not in canonical.columns or "trade_date" not in canonical.columns:
        return 0
    date_index = pd.Index(pd.to_datetime(expected_dates, errors="coerce").dropna().unique()).sort_values()
    if len(date_index) == 0:
        return 0
    positions = {pd.Timestamp(value).strftime("%Y-%m-%d"): idx for idx, value in enumerate(date_index)}
    data = canonical[["symbol", "trade_date"]].copy()
    data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data = data.loc[data["symbol"].astype(bool) & data["trade_date"].isin(positions)]
    if data.empty:
        return 0
    first_dates = data.groupby("symbol", sort=False)["trade_date"].min()
    total = 0
    for first_date in first_dates:
        total += len(date_index) - int(positions.get(str(first_date), len(date_index)))
    return int(total)


def _domain_coverage_report(
    canonical: pd.DataFrame,
    request: DomainFetchRequest,
    *,
    calendar: pd.DataFrame | None = None,
    market_coverage_basis: str = "raw_full_grid",
) -> dict[str, Any]:
    if request.domain == DataDomain.MARKET_DAILY:
        return _coverage_report(
            canonical,
            FetchRequest(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
                adjusted_flag=request.adjusted_flag,
            ),
            calendar=calendar,
            coverage_basis=market_coverage_basis,
        )
    report = coverage_report_for_domain(canonical, request, provider="silver")
    if request.domain != DataDomain.TRADING_CALENDAR and calendar is not None and not calendar.empty and request.symbols:
        expected_dates = trading_dates_from_calendar(calendar, request.start_date, request.end_date)
        expected_rows = int(len(request.symbols) * len(expected_dates))
        report["expected_rows"] = expected_rows
        report["coverage_ratio"] = float(int(report.get("row_count", 0) or 0) / expected_rows) if expected_rows else 0.0
    return report


def _domain_coverage_metadata(coverage_report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(coverage_report, dict):
        return {}
    return {
        str(key): value
        for key, value in coverage_report.items()
        if str(key).startswith(("derivation_", "fallback_"))
    }


def _refresh_blockers(
    *,
    coverage_report: dict[str, Any],
    conflict_summary: dict[str, Any],
    min_coverage_ratio: float,
    severe_conflict_limit: int,
) -> list[str]:
    blockers: list[str] = []
    if float(coverage_report.get("coverage_ratio", 0.0) or 0.0) < float(min_coverage_ratio):
        blockers.append("coverage_below_threshold")
    if int(conflict_summary.get("severe_conflict_count", 0) or 0) > int(severe_conflict_limit):
        blockers.append("severe_source_conflict")
    if int(coverage_report.get("row_count", 0) or 0) <= 0:
        blockers.append("empty_canonical_market")
    return [item for item in blockers if not item.startswith("_")]


def _domain_quality_status(*, domain_outputs: dict[str, dict[str, Any]], required_domains: set[str]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for domain, output in domain_outputs.items():
        coverage = dict(output.get("coverage_report", {}) or {})
        conflict = dict(output.get("conflict_summary", {}) or {})
        row_count = int(coverage.get("row_count", 0) or 0)
        requirement = "required" if domain in required_domains else "optional"
        if row_count > 0:
            status = "ok"
        elif requirement == "required":
            status = "blocked"
        else:
            status = "degraded"
        if requirement == "required" and int(conflict.get("severe_conflict_count", 0) or 0) > 0:
            status = "blocked"
        payload[domain] = {
            "status": status,
            "requirement": requirement,
            "row_count": row_count,
            "coverage_ratio": float(coverage.get("coverage_ratio", 0.0) or 0.0),
            "severe_conflict_count": int(conflict.get("severe_conflict_count", 0) or 0),
        }
    return payload


def _source_provenance(domain_outputs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for domain, output in domain_outputs.items():
        provider_result = output.get("provider_result")
        data = getattr(provider_result, "data", pd.DataFrame())
        providers: list[str] = []
        if isinstance(data, pd.DataFrame) and not data.empty and "source" in data.columns:
            providers = sorted(str(item) for item in data["source"].dropna().astype(str).unique() if str(item).strip())
        payload[domain] = {
            "providers": providers,
            "silver_path": str(output.get("silver_path", "")),
            "conflict_report_path": str(output.get("conflict_report_path", "")),
        }
    return payload


def _provider_health_summary_from_refresh(
    *,
    provider_chain: list[str],
    provider_coverage_report: dict[str, Any],
    provider_error_report: list[dict[str, Any]],
) -> dict[str, Any]:
    domain_status: dict[str, str] = {}
    ok_domain_count = 0
    for domain, coverage in provider_coverage_report.items():
        row_count = int(dict(coverage or {}).get("row_count", 0) or 0)
        status = "ok" if row_count > 0 else "no_data"
        domain_status[str(domain)] = status
        ok_domain_count += int(status == "ok")
    return {
        "provider_chain": list(provider_chain),
        "domain_status": domain_status,
        "checked_domain_count": len(provider_coverage_report),
        "ok_domain_count": ok_domain_count,
        "error_count": len(provider_error_report),
        "errors": list(provider_error_report[:20]),
    }


def _register_policy_input_bundle(
    *,
    lake: ResearchDataLake,
    config: RefreshConfig,
    refresh_run_id: str,
    request_symbols: tuple[str, ...] | None = None,
    canonical: pd.DataFrame,
    provider_chain: list[str],
    coverage_report: dict[str, Any],
    conflict_summary: dict[str, Any],
    domain_outputs: dict[str, dict[str, Any]] | None = None,
    registered_domain_dataset_ids: dict[str, str] | None = None,
) -> str:
    benchmark = str(config.benchmark).upper()
    request_symbols = tuple(request_symbols or _request_symbols(config))
    bundle_frames = _build_policy_input_bundle_frames(
        lake=lake,
        config=config,
        request_symbols=request_symbols,
        canonical=canonical,
        benchmark=benchmark,
        domain_outputs=domain_outputs or {},
    )
    sidecar_dataset_ids = dict(registered_domain_dataset_ids or {})
    coverage_start = bundle_frames.start_date or str(coverage_report.get("start_date", ""))
    coverage_end = bundle_frames.end_date or str(coverage_report.get("end_date", ""))
    bundle_coverage_report = {
        **dict(coverage_report),
        "start_date": coverage_start,
        "end_date": coverage_end,
        "incremental_start_date": str(coverage_report.get("start_date", "")),
        "incremental_end_date": str(coverage_report.get("end_date", "")),
        "source_market_dataset_id": bundle_frames.source_market_dataset_id,
        "source_market_dataset_end_date": bundle_frames.source_market_dataset_end_date,
    }
    spec = {
        "dataset": "policy_input_bundle",
        "source": "data_platform_refresh",
        "provider_plan": config.provider_plan,
        "provider_chain": list(provider_chain),
        "refresh_run_id": refresh_run_id,
        "adjusted_flag": config.adjusted_flag,
        "requested_symbols": list(request_symbols),
        "refresh_universe_key": _universe_key(request_symbols),
        "sidecar_domains": sorted(sidecar_dataset_ids),
        "sidecar_dataset_ids": sidecar_dataset_ids,
        "calendar_dataset_id": sidecar_dataset_ids.get(DataDomain.TRADING_CALENDAR, ""),
        "universe_snapshot_dataset_id": sidecar_dataset_ids.get(DataDomain.UNIVERSE_SNAPSHOT, ""),
        "data_platform_refresh_run_id": refresh_run_id,
        "start_date": coverage_start,
        "end_date": coverage_end,
        "benchmark": benchmark,
        "coverage_report": bundle_coverage_report,
        "conflict_summary": dict(conflict_summary),
    }
    if bundle_frames.source_market_dataset_id:
        spec["source_market_dataset_id"] = bundle_frames.source_market_dataset_id
        spec["source_market_dataset_end_date"] = bundle_frames.source_market_dataset_end_date
        spec["refresh_semantics"] = "extend_existing_policy_input_bundle"
    else:
        spec["refresh_semantics"] = "new_policy_input_bundle"
    record = lake.save_market_data_bundle(
        spec=spec,
        market_frames=bundle_frames.market_frames,
        benchmark_close=bundle_frames.benchmark_close,
        benchmark_open=bundle_frames.benchmark_open,
        membership_frame=bundle_frames.membership_frame,
        feature_frames=bundle_frames.feature_frames,
        source="data_platform_refresh",
        reuse=False,
    )
    return record.dataset_id


def _build_policy_input_bundle_frames(
    *,
    lake: ResearchDataLake,
    config: RefreshConfig,
    request_symbols: tuple[str, ...],
    canonical: pd.DataFrame,
    benchmark: str,
    domain_outputs: dict[str, dict[str, Any]],
) -> _PolicyInputBundleFrames:
    market = canonical.loc[canonical["symbol"] != benchmark].copy()
    benchmark_frame = canonical.loc[canonical["symbol"] == benchmark].copy()
    if market.empty:
        raise ValueError("refresh_blocked: canonical market has no non-benchmark rows")
    if benchmark_frame.empty:
        raise ValueError(f"refresh_blocked: canonical market is missing benchmark {benchmark}")
    incremental = _frames_from_canonical_market(
        market=market,
        benchmark_frame=benchmark_frame,
        benchmark=benchmark,
        feature_frames=_sidecar_feature_frames(
            domain_outputs=domain_outputs,
            market_columns=sorted(str(item).strip().upper() for item in market["symbol"].dropna().unique()),
        ),
        source_market_dataset_id="",
        source_market_dataset_end_date="",
    )
    base_metadata = _find_base_policy_input_bundle(
        lake=lake,
        config=config,
        request_symbols=request_symbols,
        min_trading_days=1,
    )
    if base_metadata is None:
        return incremental
    base = _load_policy_input_bundle_frames(
        metadata=base_metadata,
        benchmark=benchmark,
    )
    return _merge_policy_input_bundle_frames(base, incremental)


def _frames_from_canonical_market(
    *,
    market: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
    benchmark: str,
    feature_frames: dict[str, pd.DataFrame] | None = None,
    source_market_dataset_id: str,
    source_market_dataset_end_date: str,
) -> _PolicyInputBundleFrames:
    market_frames = {field.title() if field != "amount" else "Amount": _pivot(market, field) for field in ["open", "high", "low", "close", "volume", "amount"]}
    market_frames = {
        "Open": market_frames["Open"],
        "High": market_frames["High"],
        "Low": market_frames["Low"],
        "Close": market_frames["Close"],
        "Volume": market_frames["Volume"],
        "Amount": market_frames["Amount"],
    }
    close = market_frames["Close"]
    membership = close.notna().astype(bool)
    return _PolicyInputBundleFrames(
        market_frames=market_frames,
        benchmark_close=_series(benchmark_frame, "close", name=benchmark),
        benchmark_open=_series(benchmark_frame, "open", name=benchmark),
        membership_frame=membership,
        feature_frames=_default_feature_frames(close, feature_frames or {}),
        start_date=_frame_start_date(close),
        end_date=_frame_end_date(close),
        source_market_dataset_id=source_market_dataset_id,
        source_market_dataset_end_date=source_market_dataset_end_date,
    )


def _load_policy_input_bundle_frames(*, metadata: dict[str, Any], benchmark: str) -> _PolicyInputBundleFrames:
    paths = dict(metadata.get("content_paths", {}) or {})
    market_path = str(paths.get("bronze_market_data", "") or "")
    benchmark_path = str(paths.get("silver_benchmark", "") or "")
    membership_path = str(paths.get("silver_membership", "") or "")
    if not market_path or not Path(market_path).exists():
        raise ValueError(f"policy_input_bundle_extend_blocked: missing bronze_market_data for {metadata.get('dataset_id')}")
    if not benchmark_path or not Path(benchmark_path).exists():
        raise ValueError(f"policy_input_bundle_extend_blocked: missing silver_benchmark for {metadata.get('dataset_id')}")
    if not membership_path or not Path(membership_path).exists():
        raise ValueError(f"policy_input_bundle_extend_blocked: missing silver_membership for {metadata.get('dataset_id')}")
    market = pd.read_parquet(market_path).copy()
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce")
    market["symbol"] = market["symbol"].astype(str).str.strip().str.upper()
    market = market.dropna(subset=["trade_date"])
    market_frames = {
        "Open": _pivot(market, "open"),
        "High": _pivot(market, "high"),
        "Low": _pivot(market, "low"),
        "Close": _pivot(market, "close"),
        "Volume": _pivot(market, "volume"),
        "Amount": _pivot(market, "amount"),
    }
    benchmark_frame = pd.read_parquet(benchmark_path).copy()
    benchmark_frame["trade_date"] = pd.to_datetime(benchmark_frame["trade_date"], errors="coerce")
    if "benchmark" in benchmark_frame.columns:
        matched = benchmark_frame.loc[benchmark_frame["benchmark"].astype(str).str.upper().eq(benchmark)].copy()
        if not matched.empty:
            benchmark_frame = matched
    benchmark_close = _series(benchmark_frame, "close", name=benchmark) if "close" in benchmark_frame.columns else pd.Series(dtype=float, name=benchmark)
    benchmark_open = _series(benchmark_frame, "open", name=benchmark) if "open" in benchmark_frame.columns else benchmark_close.copy()
    membership_frame = _read_membership_frame(membership_path, index_fallback=market_frames["Close"].index)
    close = market_frames["Close"]
    return _PolicyInputBundleFrames(
        market_frames=market_frames,
        benchmark_close=benchmark_close,
        benchmark_open=benchmark_open,
        membership_frame=membership_frame,
        feature_frames=_default_feature_frames(close, {}),
        start_date=str(metadata.get("start_date", "") or _frame_start_date(close)),
        end_date=str(metadata.get("end_date", "") or _frame_end_date(close)),
        source_market_dataset_id=str(metadata.get("dataset_id", "") or ""),
        source_market_dataset_end_date=str(metadata.get("end_date", "") or _frame_end_date(close)),
    )


def _merge_policy_input_bundle_frames(
    base: _PolicyInputBundleFrames,
    incremental: _PolicyInputBundleFrames,
) -> _PolicyInputBundleFrames:
    market_columns = _ordered_union_columns(
        *list(base.market_frames.values()),
        *list(incremental.market_frames.values()),
    )
    market_frames = {
        field: _combine_wide_panel(base.market_frames[field], incremental.market_frames[field], columns=market_columns)
        for field in ["Open", "High", "Low", "Close", "Volume", "Amount"]
    }
    close = market_frames["Close"]
    membership = _combine_wide_panel(base.membership_frame, incremental.membership_frame, columns=market_columns).reindex(index=close.index, columns=market_columns)
    membership = membership.fillna(False).astype(bool)
    feature_names = sorted(set(base.feature_frames) | set(incremental.feature_frames) | {"score_none", "score_v2", "score_blend"})
    features: dict[str, pd.DataFrame] = {}
    for name in feature_names:
        left = base.feature_frames.get(name, pd.DataFrame())
        right = incremental.feature_frames.get(name, pd.DataFrame())
        features[name] = _combine_wide_panel(left, right, columns=market_columns).reindex(index=close.index, columns=market_columns)
    features = _default_feature_frames(close, features)
    return _PolicyInputBundleFrames(
        market_frames=market_frames,
        benchmark_close=_combine_series(base.benchmark_close, incremental.benchmark_close, name=incremental.benchmark_close.name or base.benchmark_close.name),
        benchmark_open=_combine_series(base.benchmark_open, incremental.benchmark_open, name=incremental.benchmark_open.name or base.benchmark_open.name),
        membership_frame=membership,
        feature_frames=features,
        start_date=_frame_start_date(close),
        end_date=_frame_end_date(close),
        source_market_dataset_id=base.source_market_dataset_id,
        source_market_dataset_end_date=base.source_market_dataset_end_date,
    )


def _read_membership_frame(path: str, *, index_fallback: pd.Index) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    if "trade_date" in frame.columns:
        frame = frame.rename(columns={"trade_date": "date"})
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        out = frame.dropna(subset=["date"]).set_index("date").sort_index()
    else:
        out = frame.copy()
        if len(out) == len(index_fallback):
            out.index = pd.to_datetime(index_fallback)
    out.index = pd.to_datetime(out.index)
    out.index.name = None
    out.columns = [str(column).strip().upper() for column in out.columns]
    return out.sort_index()


def _default_feature_frames(close: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {str(name): frame.copy() for name, frame in frames.items()}
    zero = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for name in ["score_none", "score_v2", "score_blend"]:
        frame = out.get(name)
        if frame is None or frame.empty:
            out[name] = zero.copy()
        else:
            out[name] = frame.reindex(index=close.index, columns=close.columns).fillna(0.0)
    return out


def _combine_wide_panel(left: pd.DataFrame, right: pd.DataFrame, *, columns: list[str]) -> pd.DataFrame:
    if left is None or left.empty:
        out = right.copy() if right is not None else pd.DataFrame()
    elif right is None or right.empty:
        out = left.copy()
    else:
        out = pd.concat([left.copy(), right.copy()], axis=0)
    if out.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=columns)
    out.index = pd.to_datetime(out.index)
    out.columns = [str(column).strip().upper() for column in out.columns]
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = None
    out.columns.name = None
    return out.reindex(columns=columns).sort_index()


def _combine_series(left: pd.Series, right: pd.Series, *, name: Any) -> pd.Series:
    if left is None or left.empty:
        out = right.copy() if right is not None else pd.Series(dtype=float)
    elif right is None or right.empty:
        out = left.copy()
    else:
        out = pd.concat([left.copy(), right.copy()], axis=0)
    if out.empty:
        out = pd.Series(dtype=float)
    out.index = pd.to_datetime(out.index)
    out = pd.to_numeric(out, errors="coerce").sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.name = str(name or "")
    out.index.name = None
    return out


def _ordered_union_columns(*frames: pd.DataFrame) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for column in frame.columns:
            value = str(column).strip().upper()
            if value and value not in seen:
                seen.add(value)
                ordered.append(value)
    return ordered


def _frame_start_date(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return ""
    return pd.Timestamp(pd.to_datetime(frame.index).min()).strftime("%Y-%m-%d")


def _frame_end_date(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return ""
    return pd.Timestamp(pd.to_datetime(frame.index).max()).strftime("%Y-%m-%d")


def _register_sidecar_domain_datasets(
    *,
    lake: ResearchDataLake,
    config: RefreshConfig,
    refresh_run_id: str,
    provider_chain: list[str],
    domain_outputs: dict[str, dict[str, Any]],
) -> dict[str, str]:
    dataset_ids: dict[str, str] = {}
    for domain, output in domain_outputs.items():
        if domain == DataDomain.MARKET_DAILY:
            continue
        canonical = output.get("canonical", pd.DataFrame())
        if canonical is None or canonical.empty:
            continue
        record = lake.save_domain_dataset(
            domain=domain,
            frame=canonical,
            spec={
                "dataset": f"data_platform_{domain}",
                "source": "data_platform_refresh",
                "provider_plan": config.provider_plan,
                "provider_chain": list(provider_chain),
                "refresh_run_id": refresh_run_id,
                "start_date": str(output.get("coverage_report", {}).get("start_date", config.start_date)),
                "end_date": str(output.get("coverage_report", {}).get("end_date", config.as_of_date)),
                "benchmark": config.benchmark,
                "coverage_report": dict(output.get("coverage_report", {})),
                "conflict_summary": dict(output.get("conflict_summary", {})),
            },
            source="data_platform_refresh",
            reuse=False,
        )
        dataset_ids[domain] = record.dataset_id
    return dataset_ids


def _sidecar_feature_frames(
    *,
    domain_outputs: dict[str, dict[str, Any]],
    market_columns: list[str],
) -> dict[str, pd.DataFrame]:
    features: dict[str, pd.DataFrame] = {}
    for domain, columns in {
        DataDomain.SECURITY_STATUS: ["is_st", "is_suspended", "is_delisted"],
        DataDomain.LIMIT_STATUS: ["is_limit_up", "is_limit_down"],
        DataDomain.VALUATION: ["total_mv", "circ_mv", "pe", "pb", "turnover_rate"],
        DataDomain.INTRADAY_DAILY_FEATURES: [
            "first_5m_ret",
            "first_15m_ret",
            "first_30m_ret",
            "first_30m_amount_share",
            "open_gap",
            "open_gap_first_30m_follow_through",
            "open_gap_first_30m_reversal",
            "last_5m_ret",
            "last_30m_ret",
            "last_30m_amount_share",
            "intraday_ret",
            "intraday_vwap",
            "close_to_vwap",
            "intraday_range",
            "close_position",
            "intraday_realized_vol",
            "intraday_price_volume_corr",
            "am_ret",
            "pm_ret",
            "am_pm_ret_spread",
            "am_pm_vol_spread",
            "am_amount_share",
            "am_pm_amount_spread",
            "early_strength_late_weak",
            "close_pressure_30m",
        ],
        DataDomain.FINANCIAL_QUARTERLY: [
            "roe_avg",
            "net_profit_margin",
            "gross_profit_margin",
            "net_profit_yoy",
            "revenue_yoy",
            "eps",
            "net_profit",
            "revenue",
            "asset_turnover",
            "debt_to_asset",
            "current_ratio",
            "cash_flow_ps",
        ],
        DataDomain.PERFORMANCE_FORECAST: [
            "profit_min",
            "profit_max",
            "profit_change_min",
            "profit_change_max",
        ],
        DataDomain.PERFORMANCE_EXPRESS: [
            "eps",
            "roe",
            "net_profit",
            "revenue",
            "total_assets",
        ],
    }.items():
        frame = domain_outputs.get(domain, {}).get("canonical", pd.DataFrame())
        if frame is None or frame.empty:
            continue
        for column in columns:
            if column not in frame.columns:
                continue
            panel = _pivot(frame.loc[frame["symbol"].isin(market_columns)], column).reindex(columns=market_columns)
            features[f"{domain}_{column}"] = panel
    index_constituents = domain_outputs.get(DataDomain.INDEX_CONSTITUENTS, {}).get("canonical", pd.DataFrame())
    if index_constituents is not None and not index_constituents.empty and {"symbol", "trade_date", "index_symbol"}.issubset(index_constituents.columns):
        data = index_constituents.loc[index_constituents["symbol"].isin(market_columns)].copy()
        data["member"] = 1.0
        for index_symbol, group in data.groupby("index_symbol"):
            suffix = _feature_suffix(index_symbol)
            features[f"index_constituents_{suffix}_member"] = _pivot(group, "member").reindex(columns=market_columns).fillna(0.0)
    industry = domain_outputs.get(DataDomain.INDUSTRY_CONCEPT, {}).get("canonical", pd.DataFrame())
    if industry is not None and not industry.empty and {"symbol", "trade_date", "industry"}.issubset(industry.columns):
        coded = industry.copy()
        coded["industry_code"] = coded["industry"].map(_stable_small_code)
        features["industry_concept_industry_code"] = _pivot(coded.loc[coded["symbol"].isin(market_columns)], "industry_code").reindex(columns=market_columns)
    return features


def _stable_small_code(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    return (int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 10000) / 10000.0


def _pivot(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.pivot_table(index="trade_date", columns="symbol", values=field, aggfunc="last").sort_index()
    out.index = pd.to_datetime(out.index)
    out.index.name = None
    out.columns.name = None
    return out


def _feature_suffix(value: Any) -> str:
    text = str(value or "").strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "unknown"


def _series(frame: pd.DataFrame, field: str, *, name: str) -> pd.Series:
    ordered = frame.sort_values("trade_date")
    out = pd.Series(pd.to_numeric(ordered[field], errors="coerce").to_numpy(dtype=float), index=pd.to_datetime(ordered["trade_date"]), name=name)
    out.index.name = None
    return out


def _valid_mask(frame: pd.DataFrame) -> pd.Series:
    numeric = frame[NUMERIC_MARKET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    positive_prices = numeric[["open", "high", "low", "close"]].gt(0).all(axis=1)
    range_valid = numeric["high"].ge(numeric[["open", "close", "low"]].max(axis=1)) & numeric["low"].le(numeric[["open", "close", "high"]].min(axis=1))
    activity_valid = numeric[["volume", "amount"]].ge(0).all(axis=1)
    return positive_prices & range_valid & activity_valid


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or "provider"))


def _request_symbols(config: RefreshConfig) -> tuple[str, ...]:
    return FetchRequest(
        symbols=tuple(dict.fromkeys([*config.symbols, config.benchmark])),
        start_date=config.start_date or config.as_of_date,
        end_date=config.as_of_date,
        adjusted_flag=config.adjusted_flag,
    ).normalized().symbols


def _universe_key(symbols: Iterable[str]) -> str:
    normalized = sorted(dict.fromkeys(str(item).strip().upper() for item in symbols if str(item).strip()))
    payload = "\n".join(normalized)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
