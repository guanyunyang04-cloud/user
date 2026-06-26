from __future__ import annotations

import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from quant_data_platform.lake import ResearchDataLake
from quant_data_platform.ingest.baostock_backfill import (
    BackfillConfig,
    _resolve_symbols,
    _resolve_run_trade_dates,
    _sidecar_dataset_ids_for_bundle,
    build_backfill_tasks,
    build_parser,
    config_from_args,
    run_backfill,
)
from quant_data_platform.domains.contracts import DataDomain, DomainFetchRequest, ProviderResult


class FakeBackfillProvider:
    name = "baostock"

    def __init__(self) -> None:
        self.requests: list[DomainFetchRequest] = []

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        self.requests.append(request)
        if request.domain != DataDomain.MARKET_INTRADAY_5M:
            raise AssertionError(f"unexpected domain={request.domain}")
        rows = []
        bar_times = ["093500000", "094000000", "094500000", "095000000", "095500000", "100000000", "100500000", "101000000"]
        for symbol in request.symbols:
            for idx, bar_time in enumerate(bar_times, start=1):
                close = 10.0 + idx / 100.0
                rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": request.start_date,
                        "bar_time": bar_time,
                        "open": close - 0.01,
                        "high": close + 0.02,
                        "low": close - 0.02,
                        "close": close,
                        "volume": 1000 + idx,
                        "amount": (1000 + idx) * close,
                        "source": "baostock",
                        "adjusted_flag": "none",
                    }
                )
        return ProviderResult(provider=self.name, data=pd.DataFrame(rows))


class FailingBackfillProvider:
    name = "baostock"

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        raise AssertionError(f"resume should not refetch {request.domain}")


class FlakyBackfillProvider:
    name = "baostock"

    def __init__(self) -> None:
        self.calls = 0

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("unit timeout")
        return FakeBackfillProvider().fetch_domain(request)


class SlowIndustryProvider:
    name = "baostock"

    def __init__(self, domain: str = DataDomain.INDUSTRY_CONCEPT) -> None:
        self.domain = domain
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.requests: list[DomainFetchRequest] = []

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        if request.domain != self.domain:
            raise AssertionError(f"unexpected domain={request.domain}")
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.requests.append(request)
        try:
            time.sleep(0.05)
            if request.domain == DataDomain.VALUATION:
                row = {
                    "symbol": request.symbols[0] if request.symbols else "600000.SH",
                    "trade_date": request.end_date,
                    "total_mv": float("nan"),
                    "circ_mv": float("nan"),
                    "pe": "10.0",
                    "pb": "1.0",
                    "turnover_rate": "0.5",
                    "source": "baostock",
                }
            else:
                row = {
                    "symbol": "600000.SH",
                    "trade_date": request.end_date,
                    "industry": "bank",
                    "concept_tags": "",
                    "source": "baostock",
                }
            return ProviderResult(
                provider=self.name,
                data=pd.DataFrame([row]),
            )
        finally:
            with self.lock:
                self.active -= 1


class FirstPassFailingIndustryProvider:
    name = "baostock"

    def __init__(self) -> None:
        self.calls_by_date: dict[str, int] = {}

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        self.calls_by_date[request.end_date] = self.calls_by_date.get(request.end_date, 0) + 1
        if self.calls_by_date[request.end_date] == 1:
            return ProviderResult(
                provider=self.name,
                data=pd.DataFrame(),
                error_report=[
                    {
                        "provider": self.name,
                        "domain": request.domain,
                        "code": "unit_transient_error",
                        "error_type": "TimeoutError",
                        "message": "first pass timeout",
                    }
                ],
            )
        return ProviderResult(
            provider=self.name,
            data=pd.DataFrame(
                [
                    {
                        "symbol": "600000.SH",
                        "trade_date": request.end_date,
                        "industry": "bank",
                        "concept_tags": "",
                        "source": "baostock",
                    }
                ]
            ),
        )


def _read_sharded_dataset(lake: ResearchDataLake, dataset_id: str) -> pd.DataFrame:
    metadata = lake.describe_dataset(dataset_id)
    pattern = metadata["content_paths"]["silver_domain_data"]
    paths = sorted(Path(pattern).parent.glob(Path(pattern).name))
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def test_backfill_parser_expands_baostock_full_and_raw_5m_start_date() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--domains",
            "baostock_full",
            "--start-date",
            "2016-01-01",
            "--end-date",
            "2026-06-05",
            "--raw-5m-start-date",
            "2020-01-01",
            "--task-workers",
            "3",
            "--snapshot-workers",
            "2",
            "--industry-concept-workers",
            "2",
            "--valuation-workers",
            "4",
            "--market-daily-symbol-workers",
            "1",
            "--failed-chunk-sweeps",
            "1",
            "--retry-backoff-seconds",
            "1.5",
            "--retry-jitter-seconds",
            "0.25",
            "--reuse-existing-market-daily",
            "--extra-sidecar-dataset-ids",
            "market_intraday_1m=data_platform_market_intraday_1m__abc,intraday_daily_features=data_platform_intraday_daily_features__def",
        ]
    )
    config = config_from_args(args)

    assert DataDomain.MARKET_INTRADAY_5M in config.domains
    assert DataDomain.MARKET_DAILY not in config.domains
    assert DataDomain.FINANCIAL_QUARTERLY in config.domains
    assert config.raw_5m_start_date == "2020-01-01"
    assert config.task_workers == 3
    assert config.snapshot_workers == 2
    assert config.industry_concept_workers == 2
    assert config.valuation_workers == 4
    assert config.market_daily_symbol_workers == 1
    assert config.failed_chunk_sweeps == 1
    assert config.retry_backoff_seconds == 1.5
    assert config.retry_jitter_seconds == 0.25
    assert config.reuse_existing_market_daily is True
    assert config.extra_sidecar_dataset_ids == (
        "market_intraday_1m=data_platform_market_intraday_1m__abc",
        "intraday_daily_features=data_platform_intraday_daily_features__def",
    )
    sidecars = _sidecar_dataset_ids_for_bundle(
        config=config,
        dataset_ids={DataDomain.INTRADAY_DAILY_FEATURES: "run_local_intraday_features"},
    )
    assert sidecars[DataDomain.MARKET_INTRADAY_1M] == "data_platform_market_intraday_1m__abc"
    assert sidecars[DataDomain.INTRADAY_DAILY_FEATURES] == "data_platform_intraday_daily_features__def"


def test_backfill_calendar_only_dry_run_does_not_require_lake_universe() -> None:
    with TemporaryDirectory() as temp_dir:
        result = run_backfill(
            BackfillConfig(
                lake_root=Path(temp_dir),
                run_id="unit_calendar_plan",
                domains=(DataDomain.TRADING_CALENDAR,),
                start_date="2026-01-05",
                end_date="2026-01-07",
                dry_run=True,
            )
        )

    assert result.status == "planned"
    assert result.planned_task_count == 1


def test_backfill_writes_raw_5m_shards_and_derives_daily_features_from_raw() -> None:
    provider = FakeBackfillProvider()
    with TemporaryDirectory() as temp_dir:
        result = run_backfill(
            BackfillConfig(
                lake_root=Path(temp_dir),
                run_id="unit_raw_5m",
                domains=(DataDomain.MARKET_INTRADAY_5M, DataDomain.INTRADAY_DAILY_FEATURES),
                start_date="2026-01-05",
                end_date="2026-01-05",
                symbols="csv:600000.SH,000001.SZ",
                chunk_size_symbols=2,
                expected_5m_bars_per_day=8,
            ),
            provider=provider,
        )
        lake = ResearchDataLake(Path(temp_dir))
        raw = _read_sharded_dataset(lake, result.dataset_ids[DataDomain.MARKET_INTRADAY_5M])
        features = _read_sharded_dataset(lake, result.dataset_ids[DataDomain.INTRADAY_DAILY_FEATURES])
        manifest_exists = result.manifest_path.exists()

    assert result.status == "ok"
    assert manifest_exists
    assert len(provider.requests) == 1
    assert provider.requests[0].domain == DataDomain.MARKET_INTRADAY_5M
    assert len(raw) == 16
    assert set(features["symbol"]) == {"600000.SH", "000001.SZ"}
    assert "first_5m_ret" in features.columns


def test_backfill_parallel_task_workers_write_independent_shards() -> None:
    provider = FakeBackfillProvider()
    with TemporaryDirectory() as temp_dir:
        result = run_backfill(
            BackfillConfig(
                lake_root=Path(temp_dir),
                run_id="unit_parallel_chunks",
                domains=(DataDomain.MARKET_INTRADAY_5M,),
                start_date="2026-01-05",
                end_date="2026-01-05",
                symbols="csv:600000.SH,000001.SZ,600001.SH,000002.SZ",
                chunk_size_symbols=1,
                expected_5m_bars_per_day=8,
                task_workers=2,
            ),
            provider=provider,
        )
        lake = ResearchDataLake(Path(temp_dir))
        raw = _read_sharded_dataset(lake, result.dataset_ids[DataDomain.MARKET_INTRADAY_5M])

    assert result.status == "ok"
    assert result.shard_counts[DataDomain.MARKET_INTRADAY_5M] == 4
    assert result.row_counts[DataDomain.MARKET_INTRADAY_5M] == 32
    assert result.error_counts[DataDomain.MARKET_INTRADAY_5M] == 0
    assert len(provider.requests) == 4
    assert set(raw["symbol"]) == {"600000.SH", "000001.SZ", "600001.SH", "000002.SZ"}


def test_backfill_limits_industry_concept_domain_concurrency() -> None:
    provider = SlowIndustryProvider()
    with TemporaryDirectory() as temp_dir:
        result = run_backfill(
            BackfillConfig(
                lake_root=Path(temp_dir),
                run_id="unit_industry_limited",
                domains=(DataDomain.INDUSTRY_CONCEPT,),
                start_date="2026-01-05",
                end_date="2026-01-08",
                task_workers=4,
                industry_concept_workers=2,
            ),
            provider=provider,
        )

    assert result.status == "ok"
    assert result.shard_counts[DataDomain.INDUSTRY_CONCEPT] == 4
    assert result.row_counts[DataDomain.INDUSTRY_CONCEPT] == 4
    assert result.error_counts[DataDomain.INDUSTRY_CONCEPT] == 0
    assert provider.max_active <= 2
    assert len(provider.requests) == 4


def test_backfill_limits_valuation_domain_concurrency() -> None:
    provider = SlowIndustryProvider(domain=DataDomain.VALUATION)
    with TemporaryDirectory() as temp_dir:
        result = run_backfill(
            BackfillConfig(
                lake_root=Path(temp_dir),
                run_id="unit_valuation_limited",
                domains=(DataDomain.VALUATION,),
                start_date="2026-01-05",
                end_date="2026-01-05",
                symbols="csv:600000.SH,000001.SZ,600001.SH,000002.SZ",
                chunk_size_symbols=1,
                task_workers=4,
                valuation_workers=2,
            ),
            provider=provider,
        )

    assert result.status == "ok"
    assert result.shard_counts[DataDomain.VALUATION] == 4
    assert result.row_counts[DataDomain.VALUATION] == 4
    assert result.error_counts[DataDomain.VALUATION] == 0
    assert provider.max_active <= 2
    assert len(provider.requests) == 4


def test_backfill_sweeps_failed_chunks_before_domain_manifest() -> None:
    provider = FirstPassFailingIndustryProvider()
    with TemporaryDirectory() as temp_dir:
        result = run_backfill(
            BackfillConfig(
                lake_root=Path(temp_dir),
                run_id="unit_failed_sweep",
                domains=(DataDomain.INDUSTRY_CONCEPT,),
                start_date="2026-01-05",
                end_date="2026-01-06",
                task_workers=2,
                industry_concept_workers=2,
                failed_chunk_retries=0,
                failed_chunk_sweeps=1,
            ),
            provider=provider,
        )

    assert result.status == "ok"
    assert result.shard_counts[DataDomain.INDUSTRY_CONCEPT] == 2
    assert result.row_counts[DataDomain.INDUSTRY_CONCEPT] == 2
    assert result.error_counts[DataDomain.INDUSTRY_CONCEPT] == 0
    assert set(provider.calls_by_date.values()) == {2}


def test_backfill_resume_reuses_completed_chunk_without_refetching() -> None:
    with TemporaryDirectory() as temp_dir:
        config = BackfillConfig(
            lake_root=Path(temp_dir),
            run_id="unit_resume",
            domains=(DataDomain.MARKET_INTRADAY_5M,),
            start_date="2026-01-05",
            end_date="2026-01-05",
            symbols="csv:600000.SH",
            chunk_size_symbols=1,
            expected_5m_bars_per_day=8,
        )
        first = run_backfill(config, provider=FakeBackfillProvider())
        second = run_backfill(config, provider=FailingBackfillProvider())

    assert first.status == "ok"
    assert second.status == "ok"
    assert second.row_counts[DataDomain.MARKET_INTRADAY_5M] == 8


def test_backfill_resume_retries_failed_chunk() -> None:
    provider = FlakyBackfillProvider()
    with TemporaryDirectory() as temp_dir:
        config = BackfillConfig(
            lake_root=Path(temp_dir),
            run_id="unit_retry_failed",
            domains=(DataDomain.MARKET_INTRADAY_5M,),
            start_date="2026-01-05",
            end_date="2026-01-05",
            symbols="csv:600000.SH",
            chunk_size_symbols=1,
            expected_5m_bars_per_day=8,
            failed_chunk_retries=0,
        )
        first = run_backfill(config, provider=provider)
        second = run_backfill(config, provider=provider)

    assert first.error_counts[DataDomain.MARKET_INTRADAY_5M] == 1
    assert second.error_counts[DataDomain.MARKET_INTRADAY_5M] == 0
    assert second.row_counts[DataDomain.MARKET_INTRADAY_5M] == 8
    assert provider.calls == 2


def test_backfill_retries_transient_chunk_error_before_storing() -> None:
    provider = FlakyBackfillProvider()
    with TemporaryDirectory() as temp_dir:
        result = run_backfill(
            BackfillConfig(
                lake_root=Path(temp_dir),
                run_id="unit_transient_retry",
                domains=(DataDomain.MARKET_INTRADAY_5M,),
                start_date="2026-01-05",
                end_date="2026-01-05",
                symbols="csv:600000.SH",
                chunk_size_symbols=1,
                expected_5m_bars_per_day=8,
                failed_chunk_retries=1,
            ),
            provider=provider,
        )

    assert result.error_counts[DataDomain.MARKET_INTRADAY_5M] == 0
    assert result.row_counts[DataDomain.MARKET_INTRADAY_5M] == 8
    assert provider.calls == 2


def test_backfill_resume_prefers_run_calendar_over_catalog_fallback() -> None:
    with TemporaryDirectory() as temp_dir:
        config = BackfillConfig(
            lake_root=Path(temp_dir),
            run_id="unit_run_calendar",
            domains=(DataDomain.UNIVERSE_SNAPSHOT,),
            start_date="2026-01-01",
            end_date="2026-01-06",
        ).normalized()
        run_dir = Path(temp_dir) / "backfill_runs" / config.run_id
        chunk_dir = run_dir / "chunks" / DataDomain.TRADING_CALENDAR
        chunk_dir.mkdir(parents=True)
        shard_path = Path(temp_dir) / "calendar.parquet"
        pd.DataFrame(
            [
                {"trade_date": "2026-01-01", "is_open": False, "exchange": "SSE", "source": "baostock"},
                {"trade_date": "2026-01-02", "is_open": True, "exchange": "SSE", "source": "baostock"},
                {"trade_date": "2026-01-05", "is_open": True, "exchange": "SSE", "source": "baostock"},
                {"trade_date": "2026-01-06", "is_open": True, "exchange": "SSE", "source": "baostock"},
            ]
        ).to_parquet(shard_path, index=False)
        (chunk_dir / "trading_calendar__2026-01-01_2026-01-06__n000000_nosymbols.json").write_text(
            (
                "{"
                '"status":"stored",'
                '"domain":"trading_calendar",'
                '"path":"' + str(shard_path).replace("\\", "\\\\") + '",'
                '"error_count":0'
                "}"
            ),
            encoding="utf-8",
        )

        dates = _resolve_run_trade_dates(run_dir=run_dir, config=config)
        tasks = build_backfill_tasks(domain=DataDomain.UNIVERSE_SNAPSHOT, config=config, symbols=(), trade_dates=dates)

    assert dates == ("2026-01-02", "2026-01-05", "2026-01-06")
    assert tasks[0].chunk_id == "universe_snapshot__2026-01-02_2026-01-02__n000001_nosymbols"


def test_backfill_resume_prefers_run_symbol_lock_over_catalog_fallback() -> None:
    with TemporaryDirectory() as temp_dir:
        config = BackfillConfig(
            lake_root=Path(temp_dir),
            run_id="unit_run_symbols",
            domains=(DataDomain.MARKET_DAILY,),
            start_date="2026-01-05",
            end_date="2026-01-05",
            symbols="all_lake",
        ).normalized()
        run_dir = Path(temp_dir) / "backfill_runs" / config.run_id
        run_dir.mkdir(parents=True)
        (run_dir / "resolved_symbols.json").write_text(
            (
                "{"
                '"symbols_spec":"all_lake",'
                '"exclude_index_symbols":true,'
                '"symbols":["600000.SH","000001.SZ"]'
                "}"
            ),
            encoding="utf-8",
        )

        symbols = _resolve_symbols(config, lake=ResearchDataLake(Path(temp_dir)), run_dir=run_dir)

    assert symbols == ("600000.SH", "000001.SZ")


def test_backfill_task_builder_honors_raw_5m_start_and_monthly_index_snapshots() -> None:
    config = BackfillConfig(
        domains=(DataDomain.MARKET_INTRADAY_5M,),
        start_date="2016-01-01",
        end_date="2020-02-05",
        symbols="csv:600000.SH",
        raw_5m_start_date="2020-01-01",
        chunk_size_months=1,
    ).normalized()
    raw_tasks = build_backfill_tasks(
        domain=DataDomain.MARKET_INTRADAY_5M,
        config=config,
        symbols=("600000.SH",),
        trade_dates=pd.bdate_range("2020-01-01", "2020-02-05").strftime("%Y-%m-%d").tolist(),
    )
    index_tasks = build_backfill_tasks(
        domain=DataDomain.INDEX_CONSTITUENTS,
        config=config,
        symbols=("600000.SH",),
        trade_dates=pd.bdate_range("2020-01-01", "2020-02-05").strftime("%Y-%m-%d").tolist(),
    )

    assert raw_tasks[0].start_date == "2020-01-01"
    assert [task.end_date for task in index_tasks] == ["2020-01-31", "2020-02-05"]


def test_backfill_market_daily_tasks_keep_benchmark_when_index_symbols_are_excluded() -> None:
    config = BackfillConfig(
        domains=(DataDomain.MARKET_DAILY,),
        start_date="2026-01-05",
        end_date="2026-01-05",
        symbols="csv:600000.SH",
        benchmark="000300.SH",
    ).normalized()

    tasks = build_backfill_tasks(
        domain=DataDomain.MARKET_DAILY,
        config=config,
        symbols=("600000.SH",),
        trade_dates=("2026-01-05",),
    )

    assert tasks[0].symbols == ("600000.SH", "000300.SH")
