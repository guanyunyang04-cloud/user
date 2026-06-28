from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from quant_data_platform.lake.canonical import (
    DEFAULT_CANONICAL_ALIAS,
    DEFAULT_CANONICAL_START_DATE,
    build_coverage_report,
    write_canonical_manifest,
)
from quant_data_platform.lake.catalog import DEFAULT_DATA_LAKE_ROOT, ResearchDataLake
from quant_data_platform.domains.contracts import (
    DOMAIN_STANDARD_COLUMNS,
    DataDomain,
    DomainFetchRequest,
    build_intraday_daily_feature_frame,
    coverage_report_for_domain,
    normalize_domain,
    normalize_domain_frame,
)
from quant_data_platform.providers import BaostockProvider
from quant_data_platform.ingest.refresh_daily import _sidecar_feature_frames


BAOSTOCK_FULL_DOMAINS: tuple[str, ...] = (
    DataDomain.TRADING_CALENDAR,
    DataDomain.UNIVERSE_SNAPSHOT,
    DataDomain.SECURITY_STATUS,
    DataDomain.INDUSTRY_CONCEPT,
    DataDomain.VALUATION,
    DataDomain.INDEX_CONSTITUENTS,
    DataDomain.ADJUST_FACTOR,
    DataDomain.MARKET_DAILY,
    DataDomain.MARKET_INTRADAY_5M,
    DataDomain.INTRADAY_DAILY_FEATURES,
    DataDomain.FINANCIAL_QUARTERLY,
    DataDomain.PERFORMANCE_FORECAST,
    DataDomain.PERFORMANCE_EXPRESS,
)
BAOSTOCK_SIDECAR_DOMAINS: tuple[str, ...] = tuple(domain for domain in BAOSTOCK_FULL_DOMAINS if domain != DataDomain.MARKET_DAILY)
SYMBOL_RANGE_DOMAINS = {
    DataDomain.MARKET_DAILY,
    DataDomain.MARKET_INTRADAY_1M,
    DataDomain.MARKET_INTRADAY_5M,
    DataDomain.INTRADAY_DAILY_FEATURES,
    DataDomain.ADJUST_FACTOR,
    DataDomain.VALUATION,
    DataDomain.FINANCIAL_QUARTERLY,
    DataDomain.PERFORMANCE_FORECAST,
    DataDomain.PERFORMANCE_EXPRESS,
}
POINT_IN_TIME_DOMAINS = {
    DataDomain.UNIVERSE_SNAPSHOT,
    DataDomain.SECURITY_STATUS,
    DataDomain.INDUSTRY_CONCEPT,
    DataDomain.INDEX_CONSTITUENTS,
}
REPORT_DOMAINS = {
    DataDomain.FINANCIAL_QUARTERLY,
    DataDomain.PERFORMANCE_FORECAST,
    DataDomain.PERFORMANCE_EXPRESS,
}
DEFAULT_BENCHMARK = "000300.SH"
DEFAULT_EXPECTED_1M_BARS_PER_DAY = 240
DEFAULT_EXPECTED_5M_BARS_PER_DAY = 48
DEFAULT_FAILED_CHUNK_RETRIES = 1


@dataclass(frozen=True)
class BackfillChunk:
    domain: str
    start_date: str
    end_date: str
    symbols: tuple[str, ...] = ()
    chunk_id: str = ""
    task_kind: str = "range"


@dataclass(frozen=True)
class BackfillConfig:
    lake_root: Path = DEFAULT_DATA_LAKE_ROOT
    run_id: str = ""
    domains: tuple[str, ...] = BAOSTOCK_FULL_DOMAINS
    start_date: str = "2010-01-01"
    end_date: str = ""
    symbols: str = "all_lake"
    benchmark: str = DEFAULT_BENCHMARK
    adjusted_flag: str = "none"
    exchange: str = "SSE"
    resume: bool = True
    dry_run: bool = False
    build_policy_bundle: bool = False
    write_canonical_manifest: bool = False
    canonical_alias: str = DEFAULT_CANONICAL_ALIAS
    canonical_start_date: str = DEFAULT_CANONICAL_START_DATE
    exclude_index_symbols: bool = True
    derive_intraday_features_from_raw: bool = True
    chunk_size_symbols: int = 200
    chunk_size_months: int = 1
    report_chunk_size_months: int = 12
    snapshot_frequency: str = "daily"
    index_snapshot_frequency: str = "monthly"
    raw_5m_start_date: str = ""
    expected_1m_bars_per_day: int = DEFAULT_EXPECTED_1M_BARS_PER_DAY
    expected_5m_bars_per_day: int = DEFAULT_EXPECTED_5M_BARS_PER_DAY
    failed_chunk_retries: int = DEFAULT_FAILED_CHUNK_RETRIES
    task_workers: int = 1
    snapshot_workers: int = 4
    industry_concept_workers: int = 4
    valuation_workers: int = 4
    adjust_factor_workers: int = 2
    market_daily_symbol_workers: int = 4
    intraday_symbol_workers: int = 1
    failed_chunk_sweeps: int = 0
    retry_backoff_seconds: float = 0.0
    retry_jitter_seconds: float = 0.0
    reuse_existing_market_daily: bool = False
    extra_sidecar_dataset_ids: tuple[str, ...] = ()
    trade_dates: tuple[str, ...] = ()

    def normalized(self) -> "BackfillConfig":
        end_date = _normalize_date(self.end_date) if str(self.end_date or "").strip() else _today_date()
        domains = _normalize_domains(self.domains)
        if self.reuse_existing_market_daily:
            domains = tuple(domain for domain in domains if domain != DataDomain.MARKET_DAILY)
        if self.derive_intraday_features_from_raw and DataDomain.INTRADAY_DAILY_FEATURES in domains and DataDomain.MARKET_INTRADAY_5M in domains:
            domains = tuple(
                [
                    domain
                    for domain in domains
                    if domain != DataDomain.INTRADAY_DAILY_FEATURES
                ]
                + [DataDomain.INTRADAY_DAILY_FEATURES]
            )
        run_id = str(self.run_id or "").strip() or f"baostock_backfill_{_utc_now_compact()}"
        return BackfillConfig(
            lake_root=Path(self.lake_root),
            run_id=run_id,
            domains=domains,
            start_date=_normalize_date(self.start_date),
            end_date=end_date,
            symbols=str(self.symbols or "all_lake").strip(),
            benchmark=_normalize_symbol(self.benchmark) or DEFAULT_BENCHMARK,
            adjusted_flag=str(self.adjusted_flag or "none").strip().lower(),
            exchange=str(self.exchange or "SSE").strip().upper(),
            resume=bool(self.resume),
            dry_run=bool(self.dry_run),
            build_policy_bundle=bool(self.build_policy_bundle),
            write_canonical_manifest=bool(self.write_canonical_manifest),
            canonical_alias=str(self.canonical_alias or DEFAULT_CANONICAL_ALIAS).strip(),
            canonical_start_date=_normalize_date(self.canonical_start_date or DEFAULT_CANONICAL_START_DATE),
            exclude_index_symbols=bool(self.exclude_index_symbols),
            derive_intraday_features_from_raw=bool(self.derive_intraday_features_from_raw),
            chunk_size_symbols=max(1, int(self.chunk_size_symbols or 1)),
            chunk_size_months=max(1, int(self.chunk_size_months or 1)),
            report_chunk_size_months=max(1, int(self.report_chunk_size_months or self.chunk_size_months or 1)),
            snapshot_frequency=str(self.snapshot_frequency or "daily").strip().lower(),
            index_snapshot_frequency=str(self.index_snapshot_frequency or "monthly").strip().lower(),
            raw_5m_start_date=_normalize_date(self.raw_5m_start_date) if str(self.raw_5m_start_date or "").strip() else "",
            expected_1m_bars_per_day=max(1, int(self.expected_1m_bars_per_day or DEFAULT_EXPECTED_1M_BARS_PER_DAY)),
            expected_5m_bars_per_day=max(1, int(self.expected_5m_bars_per_day or DEFAULT_EXPECTED_5M_BARS_PER_DAY)),
            failed_chunk_retries=max(0, int(self.failed_chunk_retries or 0)),
            task_workers=max(1, int(self.task_workers or 1)),
            snapshot_workers=max(1, int(self.snapshot_workers or 1)),
            industry_concept_workers=max(1, int(self.industry_concept_workers or 1)),
            valuation_workers=max(1, int(self.valuation_workers or 1)),
            adjust_factor_workers=max(1, int(self.adjust_factor_workers or 1)),
            market_daily_symbol_workers=max(1, int(self.market_daily_symbol_workers or 1)),
            intraday_symbol_workers=max(1, int(self.intraday_symbol_workers or 1)),
            failed_chunk_sweeps=max(0, int(self.failed_chunk_sweeps or 0)),
            retry_backoff_seconds=max(0.0, float(self.retry_backoff_seconds or 0.0)),
            retry_jitter_seconds=max(0.0, float(self.retry_jitter_seconds or 0.0)),
            reuse_existing_market_daily=bool(self.reuse_existing_market_daily),
            extra_sidecar_dataset_ids=tuple(str(item).strip() for item in self.extra_sidecar_dataset_ids if str(item).strip()),
            trade_dates=tuple(
                _normalize_date(str(item).strip())
                for item in self.trade_dates
                if str(item).strip()
            ),
        )


@dataclass(frozen=True)
class BackfillResult:
    status: str
    run_id: str
    lake_root: Path
    run_dir: Path
    manifest_path: Path
    dataset_ids: dict[str, str] = field(default_factory=dict)
    policy_bundle_dataset_id: str = ""
    canonical_manifest_path: Path | None = None
    row_counts: dict[str, int] = field(default_factory=dict)
    shard_counts: dict[str, int] = field(default_factory=dict)
    error_counts: dict[str, int] = field(default_factory=dict)
    planned_task_count: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable BaoStock-only data lake backfill")
    parser.add_argument("--lake-root", default=str(DEFAULT_DATA_LAKE_ROOT))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--domains", default="baostock_full")
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--symbols", default="all_lake")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--adjusted-flag", default="none")
    parser.add_argument("--exchange", default="SSE")
    parser.add_argument("--chunk-size-symbols", type=int, default=200)
    parser.add_argument("--chunk-size-months", type=int, default=1)
    parser.add_argument("--report-chunk-size-months", type=int, default=12)
    parser.add_argument("--snapshot-frequency", choices=["daily", "monthly", "quarterly", "end"], default="daily")
    parser.add_argument("--index-snapshot-frequency", choices=["daily", "monthly", "quarterly", "end"], default="monthly")
    parser.add_argument("--raw-5m-start-date", default="")
    parser.add_argument("--expected-1m-bars-per-day", type=int, default=DEFAULT_EXPECTED_1M_BARS_PER_DAY)
    parser.add_argument("--expected-5m-bars-per-day", type=int, default=DEFAULT_EXPECTED_5M_BARS_PER_DAY)
    parser.add_argument("--failed-chunk-retries", type=int, default=DEFAULT_FAILED_CHUNK_RETRIES)
    parser.add_argument("--task-workers", type=int, default=1)
    parser.add_argument("--snapshot-workers", type=int, default=4)
    parser.add_argument("--industry-concept-workers", type=int, default=4)
    parser.add_argument("--valuation-workers", type=int, default=4)
    parser.add_argument("--adjust-factor-workers", type=int, default=2)
    parser.add_argument("--intraday-symbol-workers", type=int, default=1)
    parser.add_argument(
        "--market-daily-symbol-workers",
        type=int,
        default=4,
        help="Per market_daily chunk BaoStock symbol concurrency; total symbol pressure is roughly task-workers times this value.",
    )
    parser.add_argument("--failed-chunk-sweeps", type=int, default=0)
    parser.add_argument("--retry-backoff-seconds", type=float, default=0.0)
    parser.add_argument("--retry-jitter-seconds", type=float, default=0.0)
    parser.add_argument(
        "--reuse-existing-market-daily",
        action="store_true",
        help="Skip BaoStock market_daily backfill and reuse the latest existing policy_input_bundle market table when building the bundle.",
    )
    parser.add_argument(
        "--extra-sidecar-dataset-ids",
        default="",
        help="Comma separated domain=dataset_id entries to attach to the generated policy bundle.",
    )
    parser.add_argument("--build-policy-bundle", action="store_true")
    parser.add_argument(
        "--write-canonical-manifest",
        action="store_true",
        help="After building a policy bundle, point canonical_data_v1 (or --canonical-alias) at it.",
    )
    parser.add_argument("--canonical-alias", default=DEFAULT_CANONICAL_ALIAS)
    parser.add_argument("--canonical-start-date", default=DEFAULT_CANONICAL_START_DATE)
    parser.add_argument("--dry-run", action="store_true")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true", default=True)
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    index_filter = parser.add_mutually_exclusive_group()
    index_filter.add_argument("--exclude-index-symbols", dest="exclude_index_symbols", action="store_true", default=True)
    index_filter.add_argument("--include-index-symbols", dest="exclude_index_symbols", action="store_false")
    feature_derivation = parser.add_mutually_exclusive_group()
    feature_derivation.add_argument(
        "--derive-intraday-features-from-raw",
        dest="derive_intraday_features_from_raw",
        action="store_true",
        default=True,
    )
    feature_derivation.add_argument(
        "--fetch-intraday-features-directly",
        dest="derive_intraday_features_from_raw",
        action="store_false",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> BackfillConfig:
    return BackfillConfig(
        lake_root=Path(args.lake_root),
        run_id=args.run_id,
        domains=_parse_domain_spec(args.domains),
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=args.symbols,
        benchmark=args.benchmark,
        adjusted_flag=args.adjusted_flag,
        exchange=args.exchange,
        resume=args.resume,
        dry_run=args.dry_run,
        build_policy_bundle=args.build_policy_bundle,
        write_canonical_manifest=args.write_canonical_manifest,
        canonical_alias=args.canonical_alias,
        canonical_start_date=args.canonical_start_date,
        exclude_index_symbols=args.exclude_index_symbols,
        derive_intraday_features_from_raw=args.derive_intraday_features_from_raw,
        chunk_size_symbols=args.chunk_size_symbols,
        chunk_size_months=args.chunk_size_months,
        report_chunk_size_months=args.report_chunk_size_months,
        snapshot_frequency=args.snapshot_frequency,
        index_snapshot_frequency=args.index_snapshot_frequency,
        raw_5m_start_date=args.raw_5m_start_date,
        expected_1m_bars_per_day=args.expected_1m_bars_per_day,
        expected_5m_bars_per_day=args.expected_5m_bars_per_day,
        failed_chunk_retries=args.failed_chunk_retries,
        task_workers=args.task_workers,
        snapshot_workers=args.snapshot_workers,
        industry_concept_workers=args.industry_concept_workers,
        valuation_workers=args.valuation_workers,
        adjust_factor_workers=args.adjust_factor_workers,
        market_daily_symbol_workers=args.market_daily_symbol_workers,
        intraday_symbol_workers=args.intraday_symbol_workers,
        failed_chunk_sweeps=args.failed_chunk_sweeps,
        retry_backoff_seconds=args.retry_backoff_seconds,
        retry_jitter_seconds=args.retry_jitter_seconds,
        reuse_existing_market_daily=args.reuse_existing_market_daily,
        extra_sidecar_dataset_ids=_parse_extra_sidecar_dataset_ids(args.extra_sidecar_dataset_ids),
    ).normalized()


def run_backfill(config: BackfillConfig, *, provider: Any | None = None) -> BackfillResult:
    config = config.normalized()
    run_started_at = _utc_now_iso()
    run_t0 = time.perf_counter()
    lake = ResearchDataLake(config.lake_root)
    provider = provider or BaostockProvider(
        _market_daily_max_workers=config.market_daily_symbol_workers,
        _intraday_max_workers=config.intraday_symbol_workers,
    )
    run_dir = lake.root / "backfill_runs" / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    domains = _normalize_domains(config.domains)
    symbols = _resolve_symbols(config, lake=lake, run_dir=run_dir) if _domains_need_symbols(domains) else ()
    trade_dates = tuple(config.trade_dates) or _resolve_run_trade_dates(run_dir=run_dir, config=config) or _resolve_trade_dates(lake=lake, config=config)
    dataset_ids: dict[str, str] = {}
    domain_shards: dict[str, list[dict[str, Any]]] = {}
    row_counts: dict[str, int] = {}
    shard_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    planned_task_count = 0
    raw_5m_shards: list[dict[str, Any]] = []
    dry_run_plan: dict[str, list[dict[str, Any]]] = {}

    for domain in domains:
        if (
            domain == DataDomain.INTRADAY_DAILY_FEATURES
            and config.derive_intraday_features_from_raw
            and raw_5m_shards
        ):
            spec = _domain_spec(config=config, domain=domain, symbols=symbols, extra={"derivation_source_domain": DataDomain.MARKET_INTRADAY_5M})
            if config.dry_run:
                dry_run_plan[domain] = [
                    {
                        "chunk_id": str(item.get("chunk_id", "")),
                        "start_date": str(item.get("start_date", "")),
                        "end_date": str(item.get("end_date", "")),
                        "source_raw_path": str(item.get("path", "")),
                    }
                    for item in raw_5m_shards
                ]
                planned_task_count += len(raw_5m_shards)
                continue
            feature_shards = _derive_intraday_feature_shards(
                lake=lake,
                config=config,
                run_dir=run_dir,
                raw_shards=raw_5m_shards,
                spec=spec,
            )
            record = lake.save_sharded_domain_dataset(
                domain=domain,
                spec=spec,
                shard_records=feature_shards,
                source="baostock_backfill",
                reuse=False,
            )
            dataset_ids[domain] = record.dataset_id
            domain_shards[domain] = feature_shards
            row_counts[domain] = int(sum(int(item.get("row_count", 0) or 0) for item in feature_shards))
            shard_counts[domain] = len(feature_shards)
            error_counts[domain] = int(sum(int(item.get("error_count", 0) or 0) for item in feature_shards))
            planned_task_count += len(feature_shards)
            continue

        tasks = build_backfill_tasks(domain=domain, config=config, symbols=symbols, trade_dates=trade_dates)
        planned_task_count += len(tasks)
        spec = _domain_spec(config=config, domain=domain, symbols=symbols)
        if config.dry_run:
            dry_run_plan[domain] = [_chunk_to_dict(task) for task in tasks]
            continue
        shards = _run_domain_tasks(
            lake=lake,
            config=config,
            run_dir=run_dir,
            provider=provider,
            domain=domain,
            tasks=tasks,
            trade_dates=trade_dates,
            spec=spec,
        )
        record = lake.save_sharded_domain_dataset(
            domain=domain,
            spec=spec,
            shard_records=shards,
            source="baostock_backfill",
            reuse=False,
        )
        dataset_ids[domain] = record.dataset_id
        domain_shards[domain] = shards
        row_counts[domain] = int(sum(int(item.get("row_count", 0) or 0) for item in shards))
        shard_counts[domain] = len(shards)
        error_counts[domain] = int(sum(int(item.get("error_count", 0) or 0) for item in shards))
        if domain == DataDomain.MARKET_INTRADAY_5M:
            raw_5m_shards = list(shards)
        if domain == DataDomain.TRADING_CALENDAR:
            calendar_frame = _read_domain_dataset(lake, record.dataset_id)
            refreshed_dates = _trade_dates_from_calendar(calendar_frame, config.start_date, config.end_date)
            if not refreshed_dates:
                refreshed_dates = _resolve_run_trade_dates(run_dir=run_dir, config=config)
            if refreshed_dates:
                trade_dates = refreshed_dates

    policy_bundle_dataset_id = ""
    if config.build_policy_bundle and not config.dry_run:
        policy_bundle_dataset_id = _build_policy_bundle_from_backfill(lake=lake, config=config, dataset_ids=dataset_ids)
    canonical_manifest_output: Path | None = None
    if config.write_canonical_manifest and policy_bundle_dataset_id and not config.dry_run:
        sidecar_dataset_ids = _sidecar_dataset_ids_for_bundle(config=config, dataset_ids=dataset_ids)
        canonical_manifest_output = write_canonical_manifest(
            lake,
            dataset_id=policy_bundle_dataset_id,
            alias=config.canonical_alias,
            start_date=config.canonical_start_date,
            end_date=config.end_date,
            sidecar_dataset_ids={key: value for key, value in sidecar_dataset_ids.items() if key != DataDomain.MARKET_DAILY},
            coverage_report=build_coverage_report(lake, start_date=config.canonical_start_date),
            notes="Created by baostock_backfill after policy bundle build.",
        )

    manifest_path = run_dir / "manifest.json"
    status = "planned" if config.dry_run else "ok"
    run_finished_at = _utc_now_iso()
    elapsed_sec = round(time.perf_counter() - run_t0, 3)
    _write_json(
        manifest_path,
        {
            "status": status,
            "run_id": config.run_id,
            "started_or_resumed_at": run_started_at,
            "run_started_at": run_started_at,
            "run_finished_at": run_finished_at,
            "elapsed_sec": elapsed_sec,
            "lake_root": str(lake.root.resolve()),
            "domains": list(domains),
            "symbols_spec": config.symbols,
            "resolved_symbol_count": len(symbols),
            "start_date": config.start_date,
            "end_date": config.end_date,
            "raw_5m_start_date": config.raw_5m_start_date,
            "dataset_ids": dataset_ids,
            "policy_bundle_dataset_id": policy_bundle_dataset_id,
            "canonical_alias": config.canonical_alias,
            "canonical_start_date": config.canonical_start_date,
            "canonical_manifest_path": str(canonical_manifest_output.resolve()) if canonical_manifest_output else "",
            "extra_sidecar_dataset_ids": list(config.extra_sidecar_dataset_ids),
            "row_counts": row_counts,
            "shard_counts": shard_counts,
            "error_counts": error_counts,
            "planned_task_count": planned_task_count,
            "task_workers": config.task_workers,
            "snapshot_workers": config.snapshot_workers,
            "industry_concept_workers": config.industry_concept_workers,
            "valuation_workers": config.valuation_workers,
            "adjust_factor_workers": config.adjust_factor_workers,
            "market_daily_symbol_workers": config.market_daily_symbol_workers,
            "intraday_symbol_workers": config.intraday_symbol_workers,
            "failed_chunk_retries": config.failed_chunk_retries,
            "failed_chunk_sweeps": config.failed_chunk_sweeps,
            "retry_backoff_seconds": config.retry_backoff_seconds,
            "retry_jitter_seconds": config.retry_jitter_seconds,
            "reuse_existing_market_daily": config.reuse_existing_market_daily,
            "domain_timing": {domain: _shard_timing_summary(shards) for domain, shards in domain_shards.items()},
            "dry_run_plan": dry_run_plan,
        },
    )
    if not config.dry_run:
        lake.write_catalog_manifest()
    return BackfillResult(
        status=status,
        run_id=config.run_id,
        lake_root=lake.root,
        run_dir=run_dir,
        manifest_path=manifest_path,
        dataset_ids=dataset_ids,
        policy_bundle_dataset_id=policy_bundle_dataset_id,
        canonical_manifest_path=canonical_manifest_output,
        row_counts=row_counts,
        shard_counts=shard_counts,
        error_counts=error_counts,
        planned_task_count=planned_task_count,
    )


def build_backfill_tasks(
    *,
    domain: str,
    config: BackfillConfig,
    symbols: Sequence[str],
    trade_dates: Sequence[str],
) -> list[BackfillChunk]:
    domain = normalize_domain(domain)
    if domain == DataDomain.TRADING_CALENDAR:
        return [
            BackfillChunk(
                domain=domain,
                start_date=config.start_date,
                end_date=config.end_date,
                chunk_id=_chunk_id(domain, config.start_date, config.end_date, (), 0),
                task_kind="calendar_range",
            )
        ]
    if domain in POINT_IN_TIME_DOMAINS:
        frequency = config.index_snapshot_frequency if domain == DataDomain.INDEX_CONSTITUENTS else config.snapshot_frequency
        return [
            BackfillChunk(
                domain=domain,
                start_date=date,
                end_date=date,
                chunk_id=_chunk_id(domain, date, date, (), idx),
                task_kind=f"snapshot_{frequency}",
            )
            for idx, date in enumerate(_snapshot_dates(trade_dates, config.start_date, config.end_date, frequency), start=1)
        ]
    start_date = config.start_date
    if domain == DataDomain.MARKET_INTRADAY_5M and config.raw_5m_start_date:
        start_date = max(config.start_date, config.raw_5m_start_date)
    month_size = config.report_chunk_size_months if domain in REPORT_DOMAINS else config.chunk_size_months
    windows = _month_windows(start_date, config.end_date, month_size)
    if domain in SYMBOL_RANGE_DOMAINS:
        symbol_chunks = _symbol_chunks(_symbols_for_domain(domain, symbols, config), config.chunk_size_symbols)
    else:
        symbol_chunks = [()]
    tasks: list[BackfillChunk] = []
    ordinal = 0
    for start, end in windows:
        for chunk_symbols in symbol_chunks:
            ordinal += 1
            tasks.append(
                BackfillChunk(
                    domain=domain,
                    start_date=start,
                    end_date=end,
                    symbols=tuple(chunk_symbols),
                    chunk_id=_chunk_id(domain, start, end, chunk_symbols, ordinal),
                    task_kind="symbol_range" if chunk_symbols else "range",
                )
            )
    return tasks


def _run_domain_tasks(
    *,
    lake: ResearchDataLake,
    config: BackfillConfig,
    run_dir: Path,
    provider: Any,
    domain: str,
    tasks: Sequence[BackfillChunk],
    trade_dates: Sequence[str],
    spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    identity = lake.build_domain_dataset_identity(domain=domain, spec=spec)
    shard_dir = Path(identity["dataset_dir"]) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_records_by_chunk: dict[str, dict[str, Any]] = {}
    pending: list[tuple[BackfillChunk, DomainFetchRequest, Path]] = []
    for task in tasks:
        status_path = _chunk_status_path(run_dir, task)
        cached = _load_cached_chunk(status_path)
        if config.resume and cached is not None:
            shard_records_by_chunk[task.chunk_id] = cached
            continue
        request = DomainFetchRequest(
            domain=domain,
            symbols=task.symbols,
            start_date=task.start_date,
            end_date=task.end_date,
            adjusted_flag=config.adjusted_flag,
            exchange=config.exchange,
        )
        pending.append((task, request, status_path))

    _run_pending_domain_chunks(
        provider=provider,
        pending=pending,
        shard_records_by_chunk=shard_records_by_chunk,
        shard_dir=shard_dir,
        trade_dates=trade_dates,
        config=config,
        domain=domain,
    )
    for _sweep in range(int(config.failed_chunk_sweeps or 0)):
        failed = [
            task
            for task in tasks
            if int((shard_records_by_chunk.get(task.chunk_id) or {}).get("error_count", 0) or 0) > 0
        ]
        if not failed:
            break
        sweep_pending = []
        for task in failed:
            sweep_pending.append(
                (
                    task,
                    DomainFetchRequest(
                        domain=domain,
                        symbols=task.symbols,
                        start_date=task.start_date,
                        end_date=task.end_date,
                        adjusted_flag=config.adjusted_flag,
                        exchange=config.exchange,
                    ),
                    _chunk_status_path(run_dir, task),
                )
            )
        _run_pending_domain_chunks(
            provider=provider,
            pending=sweep_pending,
            shard_records_by_chunk=shard_records_by_chunk,
            shard_dir=shard_dir,
            trade_dates=trade_dates,
            config=config,
            domain=domain,
        )
    return [shard_records_by_chunk[task.chunk_id] for task in tasks if task.chunk_id in shard_records_by_chunk]


def _run_pending_domain_chunks(
    *,
    provider: Any,
    pending: Sequence[tuple[BackfillChunk, DomainFetchRequest, Path]],
    shard_records_by_chunk: dict[str, dict[str, Any]],
    shard_dir: Path,
    trade_dates: Sequence[str],
    config: BackfillConfig,
    domain: str,
) -> None:
    worker_count = _worker_count_for_domain(domain=domain, config=config, pending_count=len(pending))
    if worker_count <= 1:
        for task, request, status_path in pending:
            record = _fetch_and_store_domain_chunk(
                provider=provider,
                request=request,
                task=task,
                status_path=status_path,
                shard_dir=shard_dir,
                trade_dates=trade_dates,
                config=config,
            )
            shard_records_by_chunk[task.chunk_id] = record
        return
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _fetch_and_store_domain_chunk,
                provider=provider,
                request=request,
                task=task,
                status_path=status_path,
                shard_dir=shard_dir,
                trade_dates=trade_dates,
                config=config,
            ): task.chunk_id
            for task, request, status_path in pending
        }
        for future in as_completed(futures):
            record = future.result()
            shard_records_by_chunk[str(record["chunk_id"])] = record


def _worker_count_for_domain(*, domain: str, config: BackfillConfig, pending_count: int) -> int:
    if pending_count <= 0:
        return 1
    limit = max(1, int(config.task_workers or 1))
    if normalize_domain(domain) in {DataDomain.UNIVERSE_SNAPSHOT, DataDomain.SECURITY_STATUS, DataDomain.INDEX_CONSTITUENTS}:
        limit = min(limit, max(1, int(config.snapshot_workers or 1)))
    if normalize_domain(domain) == DataDomain.INDUSTRY_CONCEPT:
        limit = min(limit, max(1, int(config.industry_concept_workers or 1)))
    if normalize_domain(domain) == DataDomain.VALUATION:
        limit = min(limit, max(1, int(config.valuation_workers or 1)))
    if normalize_domain(domain) == DataDomain.ADJUST_FACTOR:
        limit = min(limit, max(1, int(config.adjust_factor_workers or 1)))
    return min(limit, pending_count)


def _fetch_and_store_domain_chunk(
    *,
    provider: Any,
    request: DomainFetchRequest,
    task: BackfillChunk,
    status_path: Path,
    shard_dir: Path,
    trade_dates: Sequence[str],
    config: BackfillConfig,
) -> dict[str, Any]:
    started_at = _utc_now_iso()
    t0 = time.perf_counter()
    data, error_report, attempt_count = _fetch_domain_chunk_with_retries(
        provider=provider,
        request=request,
        task=task,
        config=config,
    )
    fetch_duration = time.perf_counter() - t0
    shard_path = shard_dir / f"{task.chunk_id}.parquet"
    store_t0 = time.perf_counter()
    data.to_parquet(shard_path, index=False)
    store_duration = time.perf_counter() - store_t0
    audit = _chunk_audit(data=data, request=request, task=task, trade_dates=trade_dates, config=config)
    audit_error_report = _audit_error_report(audit=audit, request=request, task=task)
    combined_error_report = [*list(error_report), *audit_error_report]
    elapsed = time.perf_counter() - t0
    record = {
        "status": "stored",
        "domain": request.domain,
        "chunk_id": task.chunk_id,
        "task_kind": task.task_kind,
        "start_date": task.start_date,
        "end_date": task.end_date,
        "symbol_count": int(len(task.symbols)),
        "row_count": int(len(data)),
        "path": str(shard_path.resolve()),
        "error_count": int(len(combined_error_report)),
        "error_report": combined_error_report,
        "audit": audit,
        "attempt_count": int(attempt_count),
        "fetch_started_at": started_at,
        "fetch_duration_sec": round(fetch_duration, 3),
        "store_duration_sec": round(store_duration, 3),
        "elapsed_sec": round(elapsed, 3),
        "stored_at": _utc_now_iso(),
    }
    _write_json(status_path, record)
    return record


def _audit_error_report(*, audit: Mapping[str, Any], request: DomainFetchRequest, task: BackfillChunk) -> list[dict[str, Any]]:
    if request.domain not in {DataDomain.MARKET_INTRADAY_1M, DataDomain.MARKET_INTRADAY_5M}:
        return []
    expected_symbol_days = int(audit.get("expected_symbol_days", 0) or 0)
    missing_symbol_days = int(audit.get("missing_symbol_days", 0) or 0)
    bad_bar_count_symbol_days = int(audit.get("bad_bar_count_symbol_days", 0) or 0)
    row_count = int(audit.get("row_count", 0) or 0)
    expected_rows = int(audit.get("expected_rows_full_grid", 0) or 0)
    status = str(audit.get("status", "") or "")
    if expected_symbol_days <= 0:
        return []
    coverage_ratio = (float(row_count) / float(expected_rows)) if expected_rows > 0 else 1.0
    # Missing or short symbol-days can be legitimate for suspended, newly listed,
    # or otherwise non-tradeable names. Treat catastrophic coverage loss as an
    # error, but keep ordinary symbol-day gaps in the audit payload for later
    # PIT-aware validation against security_status.
    if row_count > 0 and coverage_ratio >= 0.50:
        return []
    return [
        {
            "provider": "backfill_audit",
            "domain": request.domain,
            "chunk_id": task.chunk_id,
            "code": "intraday_coverage_audit_failed",
            "message": (
                f"intraday coverage incomplete: status={status or 'unknown'} "
                f"row_count={row_count} missing_symbol_days={missing_symbol_days} "
                f"bad_bar_count_symbol_days={bad_bar_count_symbol_days}"
            ),
            "expected_symbol_days": expected_symbol_days,
            "observed_symbol_days": int(audit.get("observed_symbol_days", 0) or 0),
            "missing_symbol_days": missing_symbol_days,
            "bad_bar_count_symbol_days": bad_bar_count_symbol_days,
            "expected_bars_per_symbol_day": int(audit.get("expected_bars_per_symbol_day", 0) or 0),
            "bad_bar_count_examples": list(audit.get("bad_bar_count_examples", []) or [])[:20],
        }
    ]


def _fetch_domain_chunk_with_retries(
    *,
    provider: Any,
    request: DomainFetchRequest,
    task: BackfillChunk,
    config: BackfillConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    attempts = max(1, int(config.failed_chunk_retries or 0) + 1)
    last_data = normalize_domain_frame(
        pd.DataFrame(),
        domain=request.domain,
        source=str(getattr(provider, "name", "baostock") or "baostock"),
        as_of_date=task.end_date,
        require_columns=False,
    )
    last_errors: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        _sleep_before_retry(attempt=attempt, config=config)
        try:
            result = provider.fetch_domain(request)
            provider_name = str(getattr(result, "provider", "") or getattr(provider, "name", "") or "baostock")
            last_data = normalize_domain_frame(
                result.data,
                domain=request.domain,
                source=provider_name,
                as_of_date=task.end_date,
                require_columns=False,
            )
            last_errors = [
                {
                    **dict(item),
                    "attempt": int(attempt),
                    "max_attempts": int(attempts),
                }
                for item in list(result.error_report or [])
            ]
        except Exception as exc:
            provider_name = str(getattr(provider, "name", "") or "baostock")
            last_data = normalize_domain_frame(
                pd.DataFrame(),
                domain=request.domain,
                source=provider_name,
                as_of_date=task.end_date,
                require_columns=False,
            )
            last_errors = [
                {
                    "provider": provider_name,
                    "domain": request.domain,
                    "chunk_id": task.chunk_id,
                    "code": "chunk_fetch_exception",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "attempt": int(attempt),
                    "max_attempts": int(attempts),
                }
            ]
        if not last_errors:
            return last_data, [], int(attempt)
    return last_data, last_errors, int(attempts)


def _sleep_before_retry(*, attempt: int, config: BackfillConfig) -> None:
    if attempt <= 1:
        return
    base = float(config.retry_backoff_seconds or 0.0)
    jitter = float(config.retry_jitter_seconds or 0.0)
    if base <= 0 and jitter <= 0:
        return
    delay = base * (2 ** max(0, int(attempt) - 2))
    if jitter > 0:
        delay += random.uniform(0.0, jitter)
    if delay > 0:
        time.sleep(delay)


def _shard_timing_summary(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    elapsed = [float(dict(item).get("elapsed_sec", 0.0) or 0.0) for item in shards]
    fetch = [float(dict(item).get("fetch_duration_sec", 0.0) or 0.0) for item in shards]
    store = [float(dict(item).get("store_duration_sec", 0.0) or 0.0) for item in shards]
    stored_at = [str(dict(item).get("stored_at", "") or "") for item in shards if str(dict(item).get("stored_at", "") or "")]
    return {
        "chunk_count": int(len(shards)),
        "elapsed_sec_sum": round(sum(elapsed), 3),
        "elapsed_sec_max": round(max(elapsed), 3) if elapsed else 0.0,
        "fetch_sec_sum": round(sum(fetch), 3),
        "fetch_sec_max": round(max(fetch), 3) if fetch else 0.0,
        "store_sec_sum": round(sum(store), 3),
        "store_sec_max": round(max(store), 3) if store else 0.0,
        "first_stored_at": min(stored_at) if stored_at else "",
        "last_stored_at": max(stored_at) if stored_at else "",
    }


def _derive_intraday_feature_shards(
    *,
    lake: ResearchDataLake,
    config: BackfillConfig,
    run_dir: Path,
    raw_shards: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    domain = DataDomain.INTRADAY_DAILY_FEATURES
    identity = lake.build_domain_dataset_identity(domain=domain, spec=spec)
    shard_dir = Path(identity["dataset_dir"]) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    feature_records: list[dict[str, Any]] = []
    for idx, raw_record in enumerate(raw_shards, start=1):
        started_at = _utc_now_iso()
        t0 = time.perf_counter()
        raw_chunk_id = str(raw_record.get("chunk_id", "") or f"raw_{idx:06d}")
        chunk = BackfillChunk(
            domain=domain,
            start_date=str(raw_record.get("start_date", "") or config.start_date),
            end_date=str(raw_record.get("end_date", "") or config.end_date),
            symbols=(),
            chunk_id=_safe_name(raw_chunk_id.replace(DataDomain.MARKET_INTRADAY_5M, domain)),
            task_kind="derived_from_market_intraday_5m",
        )
        status_path = _chunk_status_path(run_dir, chunk)
        cached = _load_cached_chunk(status_path)
        if config.resume and cached is not None:
            feature_records.append(cached)
            continue
        raw_path = Path(str(raw_record.get("path", "") or ""))
        errors: list[dict[str, Any]] = []
        if raw_path.exists():
            fetch_t0 = time.perf_counter()
            raw = pd.read_parquet(raw_path)
            fetch_duration = time.perf_counter() - fetch_t0
            features = build_intraday_daily_feature_frame(raw, source="baostock", adjusted_flag=config.adjusted_flag)
        else:
            fetch_duration = 0.0
            features = normalize_domain_frame(
                pd.DataFrame(),
                domain=domain,
                source="baostock",
                as_of_date=chunk.end_date,
                adjusted_flag=config.adjusted_flag,
                require_columns=False,
            )
            errors.append(
                {
                    "provider": "baostock_backfill",
                    "domain": domain,
                    "code": "missing_raw_5m_shard",
                    "message": str(raw_path),
                }
            )
        shard_path = shard_dir / f"{chunk.chunk_id}.parquet"
        store_t0 = time.perf_counter()
        features.to_parquet(shard_path, index=False)
        store_duration = time.perf_counter() - store_t0
        elapsed = time.perf_counter() - t0
        record = {
            "status": "stored",
            "domain": domain,
            "chunk_id": chunk.chunk_id,
            "task_kind": chunk.task_kind,
            "start_date": chunk.start_date,
            "end_date": chunk.end_date,
            "symbol_count": int(features["symbol"].nunique()) if "symbol" in features.columns and not features.empty else 0,
            "row_count": int(len(features)),
            "path": str(shard_path.resolve()),
            "source_raw_path": str(raw_path.resolve()) if raw_path.exists() else str(raw_path),
            "error_count": int(len(errors)),
            "error_report": errors,
            "audit": {"status": "ok" if len(features) else "empty", "row_count": int(len(features))},
            "attempt_count": 1,
            "fetch_started_at": started_at,
            "fetch_duration_sec": round(fetch_duration, 3),
            "store_duration_sec": round(store_duration, 3),
            "elapsed_sec": round(elapsed, 3),
            "stored_at": _utc_now_iso(),
        }
        _write_json(status_path, record)
        feature_records.append(record)
    return feature_records


def _build_policy_bundle_from_backfill(
    *,
    lake: ResearchDataLake,
    config: BackfillConfig,
    dataset_ids: Mapping[str, str],
) -> str:
    market_daily_source = _market_daily_source_for_bundle(lake=lake, dataset_ids=dataset_ids)
    market = _load_market_daily_for_bundle(lake=lake, dataset_ids=dataset_ids, source=market_daily_source)
    if market.empty:
        return ""
    market = market.copy()
    market["symbol"] = market["symbol"].map(_normalize_symbol)
    market_symbols = sorted(
        symbol
        for symbol in market["symbol"].dropna().astype(str).unique()
        if symbol and symbol != config.benchmark and not _is_likely_index_symbol(symbol)
    )
    if not market_symbols:
        return ""
    market_frames = {
        "Open": _pivot_market(market, "open", market_symbols),
        "High": _pivot_market(market, "high", market_symbols),
        "Low": _pivot_market(market, "low", market_symbols),
        "Close": _pivot_market(market, "close", market_symbols),
        "Volume": _pivot_market(market, "volume", market_symbols),
        "Amount": _pivot_market(market, "amount", market_symbols),
    }
    benchmark_rows = market.loc[market["symbol"] == config.benchmark]
    if benchmark_rows.empty:
        return ""
    benchmark_close = _series_from_long(benchmark_rows, "close", name=config.benchmark)
    benchmark_open = _series_from_long(benchmark_rows, "open", name=config.benchmark)
    membership_frame = market_frames["Close"].notna()
    domain_outputs: dict[str, dict[str, Any]] = {}
    sidecar_dataset_ids = _sidecar_dataset_ids_for_bundle(config=config, dataset_ids=dataset_ids)
    for domain in BAOSTOCK_SIDECAR_DOMAINS:
        if domain == DataDomain.MARKET_INTRADAY_5M:
            continue
        dataset_id = sidecar_dataset_ids.get(domain) or _latest_dataset_id(lake, f"data_platform_{domain}")
        if not dataset_id:
            continue
        frame = _read_domain_dataset(lake, dataset_id)
        if not frame.empty:
            domain_outputs[domain] = {"canonical": frame}
    feature_frames = _sidecar_feature_frames(domain_outputs=domain_outputs, market_columns=market_symbols)
    record = lake.save_market_data_bundle(
        spec={
            "dataset": "policy_input_bundle",
            "source": "baostock_backfill",
            "provider_plan": "baostock_only",
            "backfill_run_id": config.run_id,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "benchmark": config.benchmark,
            "sidecar_dataset_ids": dict(sidecar_dataset_ids),
            "extra_sidecar_dataset_ids": _extra_sidecar_dataset_id_map(config),
            "market_daily_source_dataset_id": str(market_daily_source.get("dataset_id", "") or ""),
            "market_daily_source_kind": str(market_daily_source.get("dataset_kind", "") or ""),
            "market_daily_reuse_mode": str(market_daily_source.get("reuse_mode", "") or ""),
        },
        market_frames=market_frames,
        benchmark_close=benchmark_close,
        benchmark_open=benchmark_open,
        membership_frame=membership_frame,
        feature_frames=feature_frames,
        source="baostock_backfill",
        reuse=False,
    )
    return record.dataset_id


def _domain_spec(
    *,
    config: BackfillConfig,
    domain: str,
    symbols: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    domain = normalize_domain(domain)
    start_date = config.start_date
    if domain == DataDomain.MARKET_INTRADAY_5M and config.raw_5m_start_date:
        start_date = max(config.start_date, config.raw_5m_start_date)
    return {
        "dataset": f"data_platform_{domain}",
        "source": "baostock_backfill",
        "provider_plan": "baostock_only",
        "backfill_run_id": config.run_id,
        "start_date": start_date,
        "end_date": config.end_date,
        "symbols_spec": config.symbols,
        "symbol_count": int(len(symbols)),
        "benchmark": config.benchmark,
        "adjusted_flag": config.adjusted_flag,
        "chunk_size_symbols": config.chunk_size_symbols,
        "chunk_size_months": config.chunk_size_months,
        "report_chunk_size_months": config.report_chunk_size_months,
        "snapshot_frequency": config.snapshot_frequency,
        "index_snapshot_frequency": config.index_snapshot_frequency,
        "raw_5m_start_date": config.raw_5m_start_date,
        "expected_1m_bars_per_day": int(config.expected_1m_bars_per_day),
        "expected_5m_bars_per_day": int(config.expected_5m_bars_per_day),
        "intraday_bar_count_contract": _intraday_bar_count_contract(domain),
        "failed_chunk_retries": config.failed_chunk_retries,
        "task_workers": config.task_workers,
        "industry_concept_workers": config.industry_concept_workers,
        "valuation_workers": config.valuation_workers,
        "adjust_factor_workers": config.adjust_factor_workers,
        "market_daily_symbol_workers": config.market_daily_symbol_workers,
        "intraday_symbol_workers": config.intraday_symbol_workers,
        "failed_chunk_sweeps": config.failed_chunk_sweeps,
        "retry_backoff_seconds": config.retry_backoff_seconds,
        "retry_jitter_seconds": config.retry_jitter_seconds,
        "sharded": True,
        **dict(extra or {}),
    }


def _intraday_bar_count_contract(domain: str) -> str:
    if domain == DataDomain.MARKET_INTRADAY_1M:
        return "mootdx_1m_240_without_0930_by_default; external_csv_1m_may_preserve_241_with_0930"
    if domain == DataDomain.MARKET_INTRADAY_5M:
        return "mootdx_or_baostock_5m_48_full_trading_day"
    return ""


def _chunk_audit(
    *,
    data: pd.DataFrame,
    request: DomainFetchRequest,
    task: BackfillChunk,
    trade_dates: Sequence[str],
    config: BackfillConfig,
) -> dict[str, Any]:
    if request.domain not in {DataDomain.MARKET_INTRADAY_1M, DataDomain.MARKET_INTRADAY_5M}:
        return coverage_report_for_domain(data, request, provider="baostock")
    expected_bars = (
        config.expected_1m_bars_per_day
        if request.domain == DataDomain.MARKET_INTRADAY_1M
        else config.expected_5m_bars_per_day
    )
    window_dates = [date for date in trade_dates if task.start_date <= date <= task.end_date]
    expected_symbol_days = int(len(task.symbols) * len(window_dates))
    expected_rows = int(expected_symbol_days * expected_bars)
    if data.empty or not {"symbol", "trade_date"}.issubset(data.columns):
        return {
            "status": "empty",
            "expected_rows_full_grid": expected_rows,
            "expected_bars_per_symbol_day": int(expected_bars),
            "row_count": 0,
            "expected_symbol_days": expected_symbol_days,
            "observed_symbol_days": 0,
            "missing_symbol_days": expected_symbol_days,
            "bad_bar_count_symbol_days": 0,
        }
    grouped = data.groupby(["symbol", "trade_date"], dropna=False).size()
    bad = grouped.loc[grouped.ne(expected_bars)]
    observed_symbol_days = int(len(grouped))
    expected_pairs = {(symbol, date) for symbol in task.symbols for date in window_dates}
    observed_pairs = {(str(symbol), str(date)) for symbol, date in grouped.index}
    missing = int(len(expected_pairs - observed_pairs)) if expected_pairs else 0
    return {
        "status": "ok" if len(data) else "empty",
        "expected_rows_full_grid": expected_rows,
        "expected_bars_per_symbol_day": int(expected_bars),
        "row_count": int(len(data)),
        "expected_symbol_days": expected_symbol_days,
        "observed_symbol_days": observed_symbol_days,
        "missing_symbol_days": missing,
        "bad_bar_count_symbol_days": int(len(bad)),
        "bad_bar_count_examples": [
            {"symbol": str(symbol), "trade_date": str(trade_date), "bar_count": int(count)}
            for (symbol, trade_date), count in bad.head(20).items()
        ],
    }


def _resolve_symbols(config: BackfillConfig, *, lake: ResearchDataLake, run_dir: Path | None = None) -> tuple[str, ...]:
    raw = str(config.symbols or "all_lake").strip()
    lock_path = run_dir / "resolved_symbols.json" if run_dir is not None else None
    if config.resume and lock_path is not None:
        locked = _load_locked_symbols(lock_path, config=config)
        if locked:
            return locked
    if raw.lower() == "all_lake":
        symbols = _symbols_from_latest_lake_universe(lake)
    elif raw.lower().startswith("file:"):
        symbols = _symbols_from_file(Path(raw.split(":", 1)[1]))
    elif raw.lower().startswith("csv:"):
        symbols = _symbols_from_csv_text(raw.split(":", 1)[1])
    elif "," in raw:
        symbols = _symbols_from_csv_text(raw)
    else:
        symbols = (_normalize_symbol(raw),)
    normalized = tuple(dict.fromkeys(symbol for symbol in (_normalize_symbol(item) for item in symbols) if symbol))
    if config.exclude_index_symbols:
        normalized = tuple(symbol for symbol in normalized if not _is_likely_index_symbol(symbol))
    if lock_path is not None and normalized and not config.dry_run:
        _write_json(
            lock_path,
            {
                "symbols_spec": raw,
                "exclude_index_symbols": bool(config.exclude_index_symbols),
                "symbol_count": len(normalized),
                "symbols": list(normalized),
                "created_or_refreshed_at": _utc_now_iso(),
            },
        )
    return normalized


def _load_locked_symbols(lock_path: Path, *, config: BackfillConfig) -> tuple[str, ...]:
    if not lock_path.exists():
        return ()
    try:
        payload = _read_json(lock_path)
    except Exception:
        return ()
    if str(payload.get("symbols_spec", "") or "").strip() != str(config.symbols or "all_lake").strip():
        return ()
    if bool(payload.get("exclude_index_symbols", True)) != bool(config.exclude_index_symbols):
        return ()
    symbols = payload.get("symbols", [])
    if not isinstance(symbols, list):
        return ()
    normalized = tuple(dict.fromkeys(symbol for symbol in (_normalize_symbol(item) for item in symbols) if symbol))
    return normalized


def _symbols_from_latest_lake_universe(lake: ResearchDataLake) -> tuple[str, ...]:
    for dataset_kind in ("data_platform_universe_snapshot", "policy_input_bundle"):
        dataset_id = _latest_dataset_id(lake, dataset_kind)
        if not dataset_id:
            continue
        metadata = lake.describe_dataset(dataset_id)
        paths = dict(metadata.get("content_paths", {}) or {})
        if dataset_kind == "data_platform_universe_snapshot":
            data = _read_table_path(paths.get("silver_domain_data", ""))
        else:
            data = _read_table_path(paths.get("bronze_market_data", ""))
        if not data.empty and "symbol" in data.columns:
            if "board" in data.columns:
                data = data.loc[data["board"].astype(str).str.lower() != "index"]
            return tuple(data["symbol"].dropna().astype(str).map(_normalize_symbol).drop_duplicates().tolist())
    raise ValueError("symbols=all_lake could not resolve a latest universe or policy bundle from the data lake")


def _symbols_from_file(path: Path) -> tuple[str, ...]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
        for column in ("symbol", "stock", "code", "ts_code"):
            if column in data.columns:
                return tuple(data[column].dropna().astype(str).tolist())
    text = path.read_text(encoding="utf-8")
    return _symbols_from_csv_text(text.replace("\n", ","))


def _symbols_from_csv_text(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(text or "").split(",") if item.strip())


def _resolve_trade_dates(*, lake: ResearchDataLake, config: BackfillConfig) -> tuple[str, ...]:
    dataset_id = _latest_dataset_id(lake, f"data_platform_{DataDomain.TRADING_CALENDAR}")
    if dataset_id:
        frame = _read_domain_dataset(lake, dataset_id)
        dates = _trade_dates_from_calendar(frame, config.start_date, config.end_date)
        if dates:
            return tuple(dates)
    return tuple(pd.bdate_range(config.start_date, config.end_date).strftime("%Y-%m-%d").tolist())


def _resolve_run_trade_dates(*, run_dir: Path, config: BackfillConfig) -> tuple[str, ...]:
    chunk_dir = run_dir / "chunks" / _safe_name(DataDomain.TRADING_CALENDAR)
    if not chunk_dir.exists():
        return ()
    status_paths = sorted(chunk_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for status_path in status_paths:
        payload = _load_cached_chunk(status_path)
        if payload is None:
            continue
        data_path = Path(str(payload.get("path", "") or ""))
        if not data_path.exists():
            continue
        try:
            frame = pd.read_parquet(data_path)
        except Exception:
            continue
        dates = _trade_dates_from_calendar(frame, config.start_date, config.end_date)
        if dates:
            return tuple(dates)
    return ()


def _trade_dates_from_calendar(frame: pd.DataFrame, start_date: str, end_date: str) -> tuple[str, ...]:
    if frame.empty or "trade_date" not in frame.columns:
        return ()
    data = frame.copy()
    if "is_open" in data.columns:
        open_text = data["is_open"].astype(str).str.lower()
        data = data.loc[data["is_open"].eq(True) | open_text.isin({"1", "true", "t", "open"})]
    dates = pd.to_datetime(data["trade_date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d")
    return tuple(date for date in sorted(dates.unique()) if start_date <= date <= end_date)


def _snapshot_dates(trade_dates: Sequence[str], start_date: str, end_date: str, frequency: str) -> tuple[str, ...]:
    dates = [date for date in trade_dates if start_date <= date <= end_date]
    if not dates:
        dates = pd.bdate_range(start_date, end_date).strftime("%Y-%m-%d").tolist()
    frequency = str(frequency or "daily").strip().lower()
    if frequency == "daily":
        return tuple(dates)
    if frequency == "end":
        return (dates[-1],) if dates else ()
    frame = pd.DataFrame({"trade_date": pd.to_datetime(dates)})
    period_freq = "Q" if frequency == "quarterly" else "M"
    frame["period"] = frame["trade_date"].dt.to_period(period_freq)
    selected = frame.groupby("period", sort=True)["trade_date"].max().dt.strftime("%Y-%m-%d").tolist()
    if dates and dates[-1] not in selected:
        selected.append(dates[-1])
    return tuple(dict.fromkeys(selected))


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


def _symbol_chunks(symbols: Sequence[str], chunk_size: int) -> list[tuple[str, ...]]:
    normalized = tuple(dict.fromkeys(_normalize_symbol(item) for item in symbols if _normalize_symbol(item)))
    return [normalized[idx : idx + chunk_size] for idx in range(0, len(normalized), chunk_size)] or [()]


def _symbols_for_domain(domain: str, symbols: Sequence[str], config: BackfillConfig) -> tuple[str, ...]:
    normalized = list(dict.fromkeys(_normalize_symbol(item) for item in symbols if _normalize_symbol(item)))
    if normalize_domain(domain) == DataDomain.MARKET_DAILY and config.benchmark:
        benchmark = _normalize_symbol(config.benchmark)
        if benchmark and benchmark not in normalized:
            normalized.append(benchmark)
    return tuple(normalized)


def _normalize_domains(domains: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for item in domains:
        for domain in _parse_domain_spec(str(item)):
            if domain not in normalized:
                normalized.append(domain)
    return tuple(normalized)


def _domains_need_symbols(domains: Iterable[str]) -> bool:
    return bool(set(_normalize_domains(domains)) & SYMBOL_RANGE_DOMAINS)


def _parse_domain_spec(raw: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in raw if str(item).strip()]
    domains: list[str] = []
    for item in items:
        lower = item.lower()
        if lower == "baostock_full":
            domains.extend(BAOSTOCK_FULL_DOMAINS)
        elif lower == "baostock_sidecars":
            domains.extend(BAOSTOCK_SIDECAR_DOMAINS)
        elif lower == "intraday":
            domains.extend((DataDomain.MARKET_INTRADAY_5M, DataDomain.INTRADAY_DAILY_FEATURES))
        elif lower in {"intraday_full", "mootdx_intraday"}:
            domains.extend((DataDomain.MARKET_INTRADAY_1M, DataDomain.MARKET_INTRADAY_5M, DataDomain.INTRADAY_DAILY_FEATURES))
        else:
            domains.append(normalize_domain(item))
    return tuple(dict.fromkeys(domains))


def _parse_extra_sidecar_dataset_ids(raw: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        return tuple(item.strip() for item in raw.split(",") if item.strip())
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _extra_sidecar_dataset_id_map(config: BackfillConfig) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in config.extra_sidecar_dataset_ids:
        text = str(item or "").strip()
        if not text:
            continue
        if "=" in text:
            domain, dataset_id = text.split("=", 1)
        elif ":" in text:
            domain, dataset_id = text.split(":", 1)
        else:
            raise ValueError(f"invalid_extra_sidecar_dataset_id: {text}; expected domain=dataset_id")
        normalized_domain = normalize_domain(domain)
        dataset_id = str(dataset_id or "").strip()
        if not dataset_id:
            raise ValueError(f"empty_extra_sidecar_dataset_id: {text}")
        out[normalized_domain] = dataset_id
    return out


def _sidecar_dataset_ids_for_bundle(*, config: BackfillConfig, dataset_ids: Mapping[str, str]) -> dict[str, str]:
    out = {normalize_domain(key): str(value) for key, value in dict(dataset_ids).items() if str(value or "").strip()}
    out.update(_extra_sidecar_dataset_id_map(config))
    return out


def _latest_dataset_id(lake: ResearchDataLake, dataset_kind: str) -> str:
    rows = lake.list_datasets(dataset_kind=dataset_kind)
    if rows.empty:
        return ""
    return str(rows.iloc[-1]["dataset_id"])


def _read_domain_dataset(lake: ResearchDataLake, dataset_id: str) -> pd.DataFrame:
    metadata = lake.describe_dataset(dataset_id)
    paths = dict(metadata.get("content_paths", {}) or {})
    data = _read_table_path(str(paths.get("silver_domain_data", "")))
    if not data.empty:
        return data
    return _read_shard_manifest_table(str(paths.get("shard_manifest", "")))


def _read_table_path(path: str) -> pd.DataFrame:
    raw = str(path or "")
    if not raw:
        return pd.DataFrame()
    paths = sorted(glob.glob(raw)) if "*" in raw else [raw]
    existing = [item for item in paths if Path(item).exists()]
    if not existing:
        return pd.DataFrame()
    frames = [pd.read_parquet(item) for item in existing]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _read_shard_manifest_table(path: str) -> pd.DataFrame:
    manifest_path = Path(str(path or ""))
    if not manifest_path.exists():
        return pd.DataFrame()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_paths = [
        str(dict(item).get("path", "") or "")
        for item in list(manifest.get("shards", []) or [])
        if str(dict(item).get("status", "") or "") == "stored" and int(dict(item).get("row_count", 0) or 0) > 0
    ]
    existing = [item for item in shard_paths if Path(item).exists()]
    if not existing:
        return pd.DataFrame()
    frames = [pd.read_parquet(item) for item in existing]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _market_daily_source_for_bundle(*, lake: ResearchDataLake, dataset_ids: Mapping[str, str]) -> dict[str, str]:
    market_dataset_id = dataset_ids.get(DataDomain.MARKET_DAILY)
    if market_dataset_id:
        return {
            "dataset_id": str(market_dataset_id),
            "dataset_kind": f"data_platform_{DataDomain.MARKET_DAILY}",
            "reuse_mode": "current_backfill_run",
        }
    bundle_id = _latest_dataset_id(lake, "policy_input_bundle")
    if bundle_id:
        return {
            "dataset_id": str(bundle_id),
            "dataset_kind": "policy_input_bundle",
            "reuse_mode": "latest_existing_policy_input_bundle",
        }
    return {}


def _load_market_daily_for_bundle(
    *,
    lake: ResearchDataLake,
    dataset_ids: Mapping[str, str],
    source: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    source_info = dict(source or _market_daily_source_for_bundle(lake=lake, dataset_ids=dataset_ids))
    market_dataset_id = str(source_info.get("dataset_id", "") or "")
    if source_info.get("dataset_kind") == f"data_platform_{DataDomain.MARKET_DAILY}" and market_dataset_id:
        return _read_domain_dataset(lake, market_dataset_id)
    bundle_id = market_dataset_id if source_info.get("dataset_kind") == "policy_input_bundle" else ""
    if not bundle_id:
        return pd.DataFrame(columns=DOMAIN_STANDARD_COLUMNS[DataDomain.MARKET_DAILY])
    metadata = lake.describe_dataset(bundle_id)
    data = _read_table_path(str(dict(metadata.get("content_paths", {}) or {}).get("bronze_market_data", "")))
    if "adjusted_flag" not in data.columns:
        data["adjusted_flag"] = "none"
    return normalize_domain_frame(
        data,
        domain=DataDomain.MARKET_DAILY,
        source="baostock_backfill",
        as_of_date="",
        require_columns=False,
    )


def _pivot_market(frame: pd.DataFrame, field: str, symbols: Sequence[str]) -> pd.DataFrame:
    data = frame.loc[frame["symbol"].isin(set(symbols))].copy()
    out = data.pivot_table(index="trade_date", columns="symbol", values=field, aggfunc="last").sort_index()
    out.index = pd.to_datetime(out.index)
    out.index.name = None
    out.columns.name = None
    return out.reindex(columns=list(symbols))


def _series_from_long(frame: pd.DataFrame, field: str, *, name: str) -> pd.Series:
    ordered = frame.sort_values("trade_date")
    out = pd.Series(pd.to_numeric(ordered[field], errors="coerce").to_numpy(dtype=float), index=pd.to_datetime(ordered["trade_date"]), name=name)
    out.index.name = None
    return out


def _load_cached_chunk(status_path: Path) -> dict[str, Any] | None:
    if not status_path.exists():
        return None
    payload = _read_json(status_path)
    if str(payload.get("status", "") or "") not in {"stored", "skipped"}:
        return None
    if int(payload.get("error_count", 0) or 0) > 0:
        return None
    path = Path(str(payload.get("path", "") or ""))
    if not path.exists():
        return None
    return payload


def _chunk_status_path(run_dir: Path, task: BackfillChunk) -> Path:
    return run_dir / "chunks" / _safe_name(task.domain) / f"{_safe_name(task.chunk_id)}.json"


def _chunk_id(domain: str, start_date: str, end_date: str, symbols: Sequence[str], ordinal: int) -> str:
    symbol_payload = ",".join(symbols)
    symbol_hash = hashlib.sha256(symbol_payload.encode("utf-8")).hexdigest()[:10] if symbol_payload else "nosymbols"
    if symbols:
        token = f"s{ordinal:06d}_{len(symbols):04d}_{symbol_hash}"
    else:
        token = f"n{ordinal:06d}_{symbol_hash}"
    return _safe_name(f"{domain}__{start_date}_{end_date}__{token}")


def _chunk_to_dict(task: BackfillChunk) -> dict[str, Any]:
    return {
        "domain": task.domain,
        "start_date": task.start_date,
        "end_date": task.end_date,
        "symbol_count": len(task.symbols),
        "chunk_id": task.chunk_id,
        "task_kind": task.task_kind,
    }


def _normalize_symbol(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    raw = raw.replace("SH.", "").replace("SZ.", "").replace("BJ.", "")
    if "." in raw:
        base, suffix = raw.rsplit(".", 1)
        if suffix == "SS":
            suffix = "SH"
        if suffix in {"SH", "SZ", "BJ"}:
            return f"{base.zfill(6) if base.isdigit() and len(base) < 6 else base}.{suffix}"
        return raw
    if raw.startswith(("SH", "SZ", "BJ")) and len(raw) >= 8 and raw[2:].isdigit():
        return f"{raw[2:]}.{raw[:2]}"
    if raw.isdigit() and len(raw) == 6:
        if raw.startswith(("5", "6", "9")):
            return f"{raw}.SH"
        if raw.startswith(("4", "8")):
            return f"{raw}.BJ"
        return f"{raw}.SZ"
    return raw


def _is_likely_index_symbol(symbol: str) -> bool:
    raw = _normalize_symbol(symbol)
    code = raw.split(".", 1)[0]
    suffix = raw.rsplit(".", 1)[-1] if "." in raw else ""
    return (suffix == "SH" and code.startswith("000")) or (suffix == "SZ" and code.startswith("399"))


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or "item")).strip("_") or "item"


def _normalize_date(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _today_date() -> str:
    return pd.Timestamp.today().strftime("%Y-%m-%d")


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(payload)), ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_backfill(config_from_args(args))
    print(
        json.dumps(
            {
                "status": result.status,
                "run_id": result.run_id,
                "manifest_path": str(result.manifest_path),
                "dataset_ids": result.dataset_ids,
                "policy_bundle_dataset_id": result.policy_bundle_dataset_id,
                "planned_task_count": result.planned_task_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.status in {"ok", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
