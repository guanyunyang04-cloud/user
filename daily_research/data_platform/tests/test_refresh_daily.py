import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from daily_research.data_lake import ResearchDataLake, load_policy_inputs_from_lake
from daily_research.data_platform.contracts import DataDomain, DomainFetchRequest, FetchRequest, ProviderResult
from daily_research.data_platform.manager import InMemoryDomainProvider, InMemoryMarketProvider
from daily_research.data_platform.refresh_daily import RefreshConfig, build_parser, run_refresh


def _market_frame(*, dates: list[str], provider: str, close_offset: float = 0.0, symbols: list[str] | None = None) -> pd.DataFrame:
    requested_symbols = symbols or ["000001.SZ", "600000.SH", "000300.SH"]
    values = {
        "000001.SZ": (10.0, 10.5, 9.8, 10.2 + close_offset, 1000, 10200),
        "600000.SH": (20.0, 20.5, 19.8, 20.2 + close_offset, 2000, 40400),
        "000300.SH": (4000.0, 4010.0, 3990.0, 4005.0 + close_offset, 3000, 12015000),
        "300001.SZ": (30.0, 30.5, 29.8, 30.2 + close_offset, 3000, 90600),
    }
    rows = []
    for trade_date in dates:
        for symbol in requested_symbols:
            open_, high, low, close, volume, amount = values[symbol]
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "amount": amount,
                    "source": provider,
                    "adjusted_flag": "none",
                }
            )
    return pd.DataFrame(rows)


def _calendar_frame(*, dates: list[str], provider: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": dates,
            "is_open": [True] * len(dates),
            "exchange": ["SSE"] * len(dates),
            "source": [provider] * len(dates),
        }
    )


def _universe_frame(*, provider: str, trade_date: str = "2026-01-05", symbols: list[str] | None = None) -> pd.DataFrame:
    requested_symbols = symbols or ["000001.SZ", "600000.SH", "000300.SH"]
    names = {
        "000001.SZ": "平安银行",
        "600000.SH": "浦发银行",
        "000300.SH": "沪深300",
        "300001.SZ": "特锐德",
    }
    return pd.DataFrame(
        {
            "symbol": requested_symbols,
            "trade_date": [trade_date] * len(requested_symbols),
            "name": [names.get(symbol, symbol) for symbol in requested_symbols],
            "exchange": ["SH" if symbol.endswith(".SH") else "SZ" for symbol in requested_symbols],
            "board": ["index" if symbol == "000300.SH" else "main" for symbol in requested_symbols],
            "list_status": ["L"] * len(requested_symbols),
            "list_date": ["2005-04-08" if symbol == "000300.SH" else "1991-04-03" for symbol in requested_symbols],
            "delist_date": [""] * len(requested_symbols),
            "source": [provider] * len(requested_symbols),
        }
    )


def _status_frame(*, provider: str, dates: list[str]) -> pd.DataFrame:
    rows = []
    for trade_date in dates:
        rows.extend(
            [
                {
                    "symbol": "000001.SZ",
                    "trade_date": trade_date,
                    "is_st": False,
                    "is_suspended": False,
                    "is_delisted": False,
                    "status_reason": "",
                    "source": provider,
                },
                {
                    "symbol": "600000.SH",
                    "trade_date": trade_date,
                    "is_st": False,
                    "is_suspended": False,
                    "is_delisted": False,
                    "status_reason": "",
                    "source": provider,
                },
            ]
        )
    return pd.DataFrame(rows)


def _limit_frame(*, provider: str, dates: list[str]) -> pd.DataFrame:
    rows = []
    for trade_date in dates:
        rows.extend(
            [
                {
                    "symbol": "000001.SZ",
                    "trade_date": trade_date,
                    "up_limit": 11.22,
                    "down_limit": 9.18,
                    "is_limit_up": False,
                    "is_limit_down": False,
                    "source": provider,
                },
                {
                    "symbol": "600000.SH",
                    "trade_date": trade_date,
                    "up_limit": 22.22,
                    "down_limit": 18.18,
                    "is_limit_up": False,
                    "is_limit_down": False,
                    "source": provider,
                },
            ]
        )
    return pd.DataFrame(rows)


def _valuation_frame(*, provider: str, dates: list[str]) -> pd.DataFrame:
    rows = []
    for trade_date in dates:
        rows.extend(
            [
                {
                    "symbol": "000001.SZ",
                    "trade_date": trade_date,
                    "total_mv": 1000.0,
                    "circ_mv": 800.0,
                    "pe": 6.0,
                    "pb": 0.8,
                    "turnover_rate": 1.1,
                    "source": provider,
                },
                {
                    "symbol": "600000.SH",
                    "trade_date": trade_date,
                    "total_mv": 2000.0,
                    "circ_mv": 1600.0,
                    "pe": 5.0,
                    "pb": 0.7,
                    "turnover_rate": 0.9,
                    "source": provider,
                },
            ]
        )
    return pd.DataFrame(rows)


class DataPlatformRefreshDailyTest(unittest.TestCase):
    def test_refresh_cli_does_not_expose_execution_timeout_flag(self) -> None:
        parser = build_parser()
        self.assertNotIn("--timeout-seconds", parser.format_help())

        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--as-of-date",
                    "2026-01-05",
                    "--symbols",
                    "000001.SZ,000300.SH",
                    "--timeout-seconds",
                    "1",
                ]
            )

    def test_first_refresh_writes_bronze_silver_manifest_and_lake_bundle(self) -> None:
        with TemporaryDirectory() as temp_dir:
            provider = InMemoryMarketProvider(
                "eastmoney_efinance",
                _market_frame(dates=["2026-01-05", "2026-01-06"], provider="eastmoney_efinance"),
            )
            result = run_refresh(
                RefreshConfig(
                    lake_root=Path(temp_dir),
                    as_of_date="2026-01-06",
                    start_date="2026-01-05",
                    symbols=("000001.SZ", "600000.SH", "000300.SH"),
                    benchmark="000300.SH",
                ),
                providers=[provider],
            )

            manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
            lake = ResearchDataLake(Path(temp_dir))
            prepared = load_policy_inputs_from_lake(
                lake=lake,
                dataset_id=result.registered_market_dataset_id,
                start_date="2026-01-05",
                end_date="2026-01-06",
                benchmark="000300.SH",
                min_trading_days=2,
            )
            bronze_exists = Path(result.bronze_paths["eastmoney_efinance"]).exists()
            silver_exists = Path(result.silver_market_path).exists()

        self.assertEqual(result.status, "ok")
        self.assertTrue(bronze_exists)
        self.assertTrue(silver_exists)
        self.assertEqual(manifest["refresh_run_id"], result.refresh_run_id)
        self.assertEqual(manifest["provider_chain"], ["eastmoney_efinance"])
        self.assertTrue(result.registered_market_dataset_id.startswith("policy_input_bundle__"))
        self.assertEqual(prepared.universe, ("000001.SZ", "600000.SH"))

    def test_second_refresh_only_fetches_missing_dates(self) -> None:
        with TemporaryDirectory() as temp_dir:
            provider = InMemoryMarketProvider(
                "eastmoney_efinance",
                _market_frame(
                    dates=["2026-01-05", "2026-01-06", "2026-01-07"],
                    provider="eastmoney_efinance",
                ),
            )
            config = RefreshConfig(
                lake_root=Path(temp_dir),
                as_of_date="2026-01-06",
                start_date="2026-01-05",
                symbols=("000001.SZ", "600000.SH", "000300.SH"),
                benchmark="000300.SH",
            )
            first = run_refresh(config, providers=[provider])
            self.assertEqual(first.status, "ok")

            second = run_refresh(
                RefreshConfig(
                    lake_root=Path(temp_dir),
                    as_of_date="2026-01-07",
                    start_date="2026-01-05",
                    symbols=("000001.SZ", "600000.SH", "000300.SH"),
                    benchmark="000300.SH",
                ),
                providers=[provider],
            )

        self.assertEqual(second.status, "ok")
        self.assertGreaterEqual(len(provider.requests), 2)
        last_request: FetchRequest = provider.requests[-1]
        self.assertEqual(last_request.start_date, "2026-01-07")
        self.assertEqual(last_request.end_date, "2026-01-07")

    def test_refresh_skips_when_existing_bundle_already_covers_as_of_date(self) -> None:
        with TemporaryDirectory() as temp_dir:
            provider = InMemoryMarketProvider(
                "eastmoney_efinance",
                _market_frame(
                    dates=["2026-01-05", "2026-01-06"],
                    provider="eastmoney_efinance",
                ),
            )
            config = RefreshConfig(
                lake_root=Path(temp_dir),
                as_of_date="2026-01-06",
                start_date="2026-01-05",
                symbols=("000001.SZ", "600000.SH", "000300.SH"),
                benchmark="000300.SH",
            )
            first = run_refresh(config, providers=[provider])
            self.assertEqual(first.status, "ok")
            request_count_after_first = len(provider.requests)

            second = run_refresh(config, providers=[provider])
            manifest = json.loads(Path(second.manifest_path).read_text(encoding="utf-8"))

        self.assertEqual(second.status, "skipped")
        self.assertEqual(second.registered_market_dataset_id, first.registered_market_dataset_id)
        self.assertEqual(manifest["reused_policy_input_dataset_id"], first.registered_market_dataset_id)
        self.assertEqual(manifest["reused_policy_input_dataset_end_date"], "2026-01-06")
        self.assertEqual(manifest["reason"], "lake already covers requested as_of_date")
        self.assertEqual(len(provider.requests), request_count_after_first)

    def test_incremental_refresh_registers_extended_long_history_bundle(self) -> None:
        with TemporaryDirectory() as temp_dir:
            provider = InMemoryMarketProvider(
                "eastmoney_efinance",
                _market_frame(
                    dates=["2026-01-05", "2026-01-06", "2026-01-07"],
                    provider="eastmoney_efinance",
                ),
            )
            first = run_refresh(
                RefreshConfig(
                    lake_root=Path(temp_dir),
                    as_of_date="2026-01-06",
                    start_date="2026-01-05",
                    symbols=("000001.SZ", "600000.SH", "000300.SH"),
                    benchmark="000300.SH",
                ),
                providers=[provider],
            )
            self.assertEqual(first.status, "ok")

            second = run_refresh(
                RefreshConfig(
                    lake_root=Path(temp_dir),
                    as_of_date="2026-01-07",
                    start_date="2026-01-05",
                    symbols=("000001.SZ", "600000.SH", "000300.SH"),
                    benchmark="000300.SH",
                ),
                providers=[provider],
            )

            lake = ResearchDataLake(Path(temp_dir))
            metadata = lake.describe_dataset(second.registered_market_dataset_id)
            prepared = load_policy_inputs_from_lake(
                lake=lake,
                dataset_id=second.registered_market_dataset_id,
                start_date="2026-01-05",
                end_date="2026-01-07",
                benchmark="000300.SH",
                min_trading_days=3,
            )

        self.assertEqual(second.status, "ok")
        self.assertEqual(metadata["start_date"], "2026-01-05")
        self.assertEqual(metadata["end_date"], "2026-01-07")
        self.assertEqual(len(prepared.close), 3)
        self.assertEqual(prepared.close.index.max().strftime("%Y-%m-%d"), "2026-01-07")

    def test_incremental_refresh_does_not_reuse_history_for_different_universe(self) -> None:
        with TemporaryDirectory() as temp_dir:
            provider = InMemoryMarketProvider(
                "eastmoney_efinance",
                _market_frame(
                    dates=["2026-01-05", "2026-01-06", "2026-01-07"],
                    provider="eastmoney_efinance",
                ),
            )
            first = run_refresh(
                RefreshConfig(
                    lake_root=Path(temp_dir),
                    as_of_date="2026-01-06",
                    start_date="2026-01-05",
                    symbols=("000001.SZ", "000300.SH"),
                    benchmark="000300.SH",
                ),
                providers=[provider],
            )
            self.assertEqual(first.status, "ok")

            second = run_refresh(
                RefreshConfig(
                    lake_root=Path(temp_dir),
                    as_of_date="2026-01-07",
                    start_date="2026-01-05",
                    symbols=("000001.SZ", "600000.SH", "000300.SH"),
                    benchmark="000300.SH",
                ),
                providers=[provider],
            )

        self.assertEqual(second.status, "ok")
        last_request: FetchRequest = provider.requests[-1]
        self.assertEqual(last_request.start_date, "2026-01-05")
        self.assertEqual(last_request.end_date, "2026-01-07")

    def test_all_a_incremental_refresh_extends_base_when_provider_adds_new_symbol(self) -> None:
        with TemporaryDirectory() as temp_dir:
            first_provider = InMemoryDomainProvider(
                "akshare_eastmoney",
                payloads={
                    DataDomain.TRADING_CALENDAR: _calendar_frame(dates=["2026-01-05", "2026-01-06"], provider="akshare_eastmoney"),
                    DataDomain.UNIVERSE_SNAPSHOT: _universe_frame(provider="akshare_eastmoney", symbols=["000001.SZ", "600000.SH", "000300.SH"]),
                    DataDomain.MARKET_DAILY: _market_frame(
                        dates=["2026-01-05", "2026-01-06"],
                        provider="akshare_eastmoney",
                        symbols=["000001.SZ", "600000.SH", "000300.SH"],
                    ),
                },
            )
            first = run_refresh(
                RefreshConfig(
                    lake_root=Path(temp_dir),
                    as_of_date="2026-01-06",
                    start_date="2026-01-05",
                    universe="all_a",
                    domains=(DataDomain.MARKET_DAILY, DataDomain.TRADING_CALENDAR, DataDomain.UNIVERSE_SNAPSHOT),
                    benchmark="000300.SH",
                    min_coverage_ratio=0.70,
                ),
                providers=[first_provider],
            )
            self.assertEqual(first.status, "ok")

            second_provider = InMemoryDomainProvider(
                "akshare_eastmoney",
                payloads={
                    DataDomain.TRADING_CALENDAR: _calendar_frame(dates=["2026-01-05", "2026-01-06", "2026-01-07"], provider="akshare_eastmoney"),
                    DataDomain.UNIVERSE_SNAPSHOT: _universe_frame(
                        provider="akshare_eastmoney",
                        trade_date="2026-01-07",
                        symbols=["000001.SZ", "600000.SH", "300001.SZ", "000300.SH"],
                    ),
                    DataDomain.MARKET_DAILY: _market_frame(
                        dates=["2026-01-07"],
                        provider="akshare_eastmoney",
                        symbols=["000001.SZ", "600000.SH", "300001.SZ", "000300.SH"],
                    ),
                },
            )
            second = run_refresh(
                RefreshConfig(
                    lake_root=Path(temp_dir),
                    as_of_date="2026-01-07",
                    start_date="2026-01-05",
                    universe="all_a",
                    domains=(DataDomain.MARKET_DAILY, DataDomain.TRADING_CALENDAR, DataDomain.UNIVERSE_SNAPSHOT),
                    benchmark="000300.SH",
                    min_coverage_ratio=0.70,
                ),
                providers=[second_provider],
            )
            lake = ResearchDataLake(Path(temp_dir))
            metadata = lake.describe_dataset(second.registered_market_dataset_id)
            prepared = load_policy_inputs_from_lake(
                lake=lake,
                dataset_id=second.registered_market_dataset_id,
                start_date="2026-01-05",
                end_date="2026-01-07",
                benchmark="000300.SH",
                min_trading_days=3,
            )

        market_requests = [req for req in second_provider.domain_requests if req.domain == DataDomain.MARKET_DAILY]
        self.assertEqual(second.status, "ok")
        self.assertEqual(market_requests[-1].start_date, "2026-01-07")
        self.assertEqual(metadata["parameters"]["refresh_semantics"], "extend_existing_policy_input_bundle")
        self.assertEqual(metadata["parameters"]["source_market_dataset_id"], first.registered_market_dataset_id)
        self.assertEqual(metadata["start_date"], "2026-01-05")
        self.assertEqual(metadata["end_date"], "2026-01-07")
        self.assertIn("300001.SZ", prepared.close.columns)
        self.assertEqual(prepared.close.loc[pd.Timestamp("2026-01-05"), "000001.SZ"], 10.2)

    def test_conflict_blocks_gold_registration_but_keeps_bronze_and_report(self) -> None:
        with TemporaryDirectory() as temp_dir:
            result = run_refresh(
                RefreshConfig(
                    lake_root=Path(temp_dir),
                    as_of_date="2026-01-05",
                    start_date="2026-01-05",
                    symbols=("000001.SZ", "600000.SH", "000300.SH"),
                    benchmark="000300.SH",
                    severe_conflict_limit=0,
                    conflict_tolerance_pct=0.001,
                ),
                providers=[
                    InMemoryMarketProvider(
                        "eastmoney_efinance",
                        _market_frame(dates=["2026-01-05"], provider="eastmoney_efinance"),
                    ),
                    InMemoryMarketProvider(
                        "baostock",
                        _market_frame(dates=["2026-01-05"], provider="baostock", close_offset=1.0),
                    ),
                ],
            )
            conflict_report = pd.read_parquet(result.conflict_report_path)
            bronze_exists = Path(result.bronze_paths["eastmoney_efinance"]).exists()

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.registered_market_dataset_id, "")
        self.assertTrue(bronze_exists)
        self.assertGreater(len(conflict_report), 0)
        self.assertGreater(result.conflict_summary["severe_conflict_count"], 0)

    def test_coverage_below_threshold_blocks_gold_registration(self) -> None:
        partial = _market_frame(dates=["2026-01-05"], provider="eastmoney_efinance")
        partial = partial.loc[partial["symbol"] == "000001.SZ"].copy()
        with TemporaryDirectory() as temp_dir:
            result = run_refresh(
                RefreshConfig(
                    lake_root=Path(temp_dir),
                    as_of_date="2026-01-05",
                    start_date="2026-01-05",
                    symbols=("000001.SZ", "600000.SH", "000300.SH"),
                    benchmark="000300.SH",
                    min_coverage_ratio=0.90,
                ),
                providers=[InMemoryMarketProvider("eastmoney_efinance", partial)],
            )

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.registered_market_dataset_id, "")
        self.assertIn("coverage_below_threshold", result.blockers)

    def test_universe_all_a_refresh_uses_calendar_and_writes_sidecar_metadata(self) -> None:
        provider = InMemoryDomainProvider(
            "akshare_eastmoney",
            payloads={
                DataDomain.TRADING_CALENDAR: _calendar_frame(dates=["2026-01-05", "2026-01-07"], provider="akshare_eastmoney"),
                DataDomain.UNIVERSE_SNAPSHOT: _universe_frame(provider="akshare_eastmoney"),
                DataDomain.MARKET_DAILY: _market_frame(dates=["2026-01-05", "2026-01-07"], provider="akshare_eastmoney"),
                DataDomain.SECURITY_STATUS: _status_frame(provider="akshare_eastmoney", dates=["2026-01-05", "2026-01-07"]),
                DataDomain.LIMIT_STATUS: _limit_frame(provider="akshare_eastmoney", dates=["2026-01-05", "2026-01-07"]),
                DataDomain.VALUATION: _valuation_frame(provider="akshare_eastmoney", dates=["2026-01-05", "2026-01-07"]),
            },
        )
        with TemporaryDirectory() as temp_dir:
            result = run_refresh(
                RefreshConfig(
                    lake_root=Path(temp_dir),
                    as_of_date="2026-01-07",
                    start_date="2026-01-05",
                    universe="all_a",
                    domains=(
                        DataDomain.MARKET_DAILY,
                        DataDomain.TRADING_CALENDAR,
                        DataDomain.UNIVERSE_SNAPSHOT,
                        DataDomain.SECURITY_STATUS,
                        DataDomain.LIMIT_STATUS,
                        DataDomain.VALUATION,
                    ),
                    required_domains=(DataDomain.MARKET_DAILY, DataDomain.TRADING_CALENDAR, DataDomain.UNIVERSE_SNAPSHOT),
                    benchmark="000300.SH",
                    min_coverage_ratio=0.70,
                ),
                providers=[provider],
            )
            manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
            metadata = ResearchDataLake(Path(temp_dir)).describe_dataset(result.registered_market_dataset_id)

        self.assertEqual(result.status, "ok")
        self.assertEqual(manifest["calendar_source"], "provider")
        self.assertEqual(manifest["domains"], [
            DataDomain.MARKET_DAILY,
            DataDomain.TRADING_CALENDAR,
            DataDomain.UNIVERSE_SNAPSHOT,
            DataDomain.SECURITY_STATUS,
            DataDomain.LIMIT_STATUS,
            DataDomain.VALUATION,
        ])
        self.assertIn(DataDomain.SECURITY_STATUS, metadata["parameters"]["sidecar_domains"])
        self.assertIn("sidecar_dataset_ids", metadata["parameters"])
        self.assertIn("calendar_dataset_id", metadata["parameters"])

    def test_refresh_manifest_records_v3_domain_quality_and_optional_degradation(self) -> None:
        provider = InMemoryDomainProvider(
            "akshare_eastmoney",
            payloads={
                DataDomain.TRADING_CALENDAR: _calendar_frame(dates=["2026-01-05"], provider="akshare_eastmoney"),
                DataDomain.UNIVERSE_SNAPSHOT: _universe_frame(provider="akshare_eastmoney"),
                DataDomain.MARKET_DAILY: _market_frame(dates=["2026-01-05"], provider="akshare_eastmoney"),
                DataDomain.SECURITY_STATUS: _status_frame(provider="akshare_eastmoney", dates=["2026-01-05"]),
                DataDomain.LIMIT_STATUS: _limit_frame(provider="akshare_eastmoney", dates=["2026-01-05"]),
            },
        )
        with TemporaryDirectory() as temp_dir:
            result = run_refresh(
                RefreshConfig(
                    lake_root=Path(temp_dir),
                    as_of_date="2026-01-05",
                    start_date="2026-01-05",
                    universe="all_a",
                    domains=(
                        DataDomain.MARKET_DAILY,
                        DataDomain.TRADING_CALENDAR,
                        DataDomain.UNIVERSE_SNAPSHOT,
                        DataDomain.SECURITY_STATUS,
                        DataDomain.LIMIT_STATUS,
                        DataDomain.VALUATION,
                    ),
                    required_domains=(
                        DataDomain.MARKET_DAILY,
                        DataDomain.TRADING_CALENDAR,
                        DataDomain.UNIVERSE_SNAPSHOT,
                        DataDomain.SECURITY_STATUS,
                        DataDomain.LIMIT_STATUS,
                    ),
                    benchmark="000300.SH",
                    provider_plan="formal_free_v3",
                    min_coverage_ratio=0.70,
                ),
                providers=[provider],
            )
            manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))

        self.assertEqual(result.status, "ok")
        self.assertEqual(manifest["provider_plan"], "formal_free_v3")
        self.assertEqual(manifest["domain_quality_status"][DataDomain.MARKET_DAILY]["status"], "ok")
        self.assertEqual(manifest["domain_quality_status"][DataDomain.VALUATION]["status"], "degraded")
        self.assertEqual(manifest["domain_quality_status"][DataDomain.VALUATION]["requirement"], "optional")
        self.assertIn("provider_health_summary", manifest)
        self.assertIn("source_provenance", manifest)
        self.assertIn("akshare_eastmoney", manifest["source_provenance"][DataDomain.MARKET_DAILY]["providers"])

    def test_calendar_incremental_skips_closed_business_day(self) -> None:
        provider = InMemoryDomainProvider(
            "akshare_eastmoney",
            payloads={
                DataDomain.TRADING_CALENDAR: _calendar_frame(dates=["2026-01-05", "2026-01-07"], provider="akshare_eastmoney"),
                DataDomain.UNIVERSE_SNAPSHOT: _universe_frame(provider="akshare_eastmoney"),
                DataDomain.MARKET_DAILY: _market_frame(dates=["2026-01-05", "2026-01-07"], provider="akshare_eastmoney"),
            },
        )
        with TemporaryDirectory() as temp_dir:
            first = run_refresh(
                RefreshConfig(
                    lake_root=Path(temp_dir),
                    as_of_date="2026-01-05",
                    start_date="2026-01-05",
                    universe="all_a",
                    domains=(DataDomain.MARKET_DAILY, DataDomain.TRADING_CALENDAR, DataDomain.UNIVERSE_SNAPSHOT),
                    benchmark="000300.SH",
                    min_coverage_ratio=0.70,
                ),
                providers=[provider],
            )
            self.assertEqual(first.status, "ok")
            second = run_refresh(
                RefreshConfig(
                    lake_root=Path(temp_dir),
                    as_of_date="2026-01-07",
                    start_date="2026-01-05",
                    universe="all_a",
                    domains=(DataDomain.MARKET_DAILY, DataDomain.TRADING_CALENDAR, DataDomain.UNIVERSE_SNAPSHOT),
                    benchmark="000300.SH",
                    min_coverage_ratio=0.70,
                ),
                providers=[provider],
            )

        self.assertEqual(second.status, "ok")
        market_requests = [req for req in provider.domain_requests if isinstance(req, DomainFetchRequest) and req.domain == DataDomain.MARKET_DAILY]
        self.assertEqual(market_requests[-1].start_date, "2026-01-07")
        self.assertEqual(market_requests[-1].end_date, "2026-01-07")

    def test_required_sidecar_domain_blocks_registration_when_empty(self) -> None:
        provider = InMemoryDomainProvider(
            "akshare_eastmoney",
            payloads={
                DataDomain.TRADING_CALENDAR: _calendar_frame(dates=["2026-01-05"], provider="akshare_eastmoney"),
                DataDomain.UNIVERSE_SNAPSHOT: _universe_frame(provider="akshare_eastmoney"),
                DataDomain.MARKET_DAILY: _market_frame(dates=["2026-01-05"], provider="akshare_eastmoney"),
                DataDomain.VALUATION: pd.DataFrame(),
            },
        )
        with TemporaryDirectory() as temp_dir:
            result = run_refresh(
                RefreshConfig(
                    lake_root=Path(temp_dir),
                    as_of_date="2026-01-05",
                    start_date="2026-01-05",
                    universe="all_a",
                    domains=(DataDomain.MARKET_DAILY, DataDomain.TRADING_CALENDAR, DataDomain.UNIVERSE_SNAPSHOT, DataDomain.VALUATION),
                    required_domains=(DataDomain.MARKET_DAILY, DataDomain.TRADING_CALENDAR, DataDomain.UNIVERSE_SNAPSHOT, DataDomain.VALUATION),
                    benchmark="000300.SH",
                    min_coverage_ratio=0.70,
                ),
                providers=[provider],
            )

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.registered_market_dataset_id, "")
        self.assertIn("required_domain_blocked:valuation", result.blockers)

    def test_universe_all_a_empty_provider_result_explains_provider_errors(self) -> None:
        def fail_universe(_: DomainFetchRequest) -> ProviderResult:
            return ProviderResult(
                provider="akshare_eastmoney",
                data=pd.DataFrame(),
                error_report=[
                    {
                        "provider": "akshare_eastmoney",
                        "domain": DataDomain.UNIVERSE_SNAPSHOT,
                        "code": "provider_exception",
                        "error_type": "RuntimeError",
                        "message": "akshare is not installed in the yolos environment",
                    }
                ],
            )

        provider = InMemoryDomainProvider(
            "akshare_eastmoney",
            payloads={
                DataDomain.TRADING_CALENDAR: _calendar_frame(dates=["2026-01-05"], provider="akshare_eastmoney"),
                DataDomain.UNIVERSE_SNAPSHOT: fail_universe,
            },
        )
        with TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "akshare is not installed"):
                run_refresh(
                    RefreshConfig(
                        lake_root=Path(temp_dir),
                        as_of_date="2026-01-05",
                        start_date="2026-01-05",
                        universe="all_a",
                        domains=(DataDomain.MARKET_DAILY, DataDomain.TRADING_CALENDAR, DataDomain.UNIVERSE_SNAPSHOT),
                        benchmark="000300.SH",
                    ),
                    providers=[provider],
                )


if __name__ == "__main__":
    unittest.main()
