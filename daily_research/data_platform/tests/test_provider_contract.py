import unittest
import sys
import types
from unittest import mock

import pandas as pd

from daily_research.data_platform.contracts import (
    DataDomain,
    DomainFetchRequest,
    STANDARD_MARKET_COLUMNS,
    FetchRequest,
    ProviderResult,
    normalize_market_frame,
    validate_provider_name,
)
from daily_research.data_platform.manager import InMemoryMarketProvider, ProviderManager
from daily_research.data_platform.providers import (
    BaostockProvider,
    EastmoneyEfinanceProvider,
    ResearchRebuildMinimalFreeProvider,
    TushareHttpOptionalProvider,
    build_default_providers,
    provider_capability_matrix,
)
from daily_research.data_platform.providers import (
    _baostock_all_stock_frame,
    _baostock_industry_frame,
    _baostock_status_frame_from_all_stock,
    _baostock_stock_basic_frame,
    _baostock_valuation_frame_from_history,
)


class DataPlatformProviderContractTest(unittest.TestCase):
    def test_normalize_market_frame_enforces_standard_schema(self) -> None:
        raw = pd.DataFrame(
            {
                "stock": ["000001.SZ"],
                "date": ["2026-01-05"],
                "Open": [10.0],
                "High": [10.5],
                "Low": [9.8],
                "Close": [10.2],
                "Volume": [1000],
                "Amount": [10200],
            }
        )

        normalized = normalize_market_frame(raw, source="eastmoney_efinance")

        self.assertEqual(list(normalized.columns), STANDARD_MARKET_COLUMNS)
        self.assertEqual(normalized["symbol"].iloc[0], "000001.SZ")
        self.assertEqual(normalized["trade_date"].iloc[0], "2026-01-05")
        self.assertEqual(normalized["source"].iloc[0], "eastmoney_efinance")
        self.assertEqual(normalized["adjusted_flag"].iloc[0], "none")

    def test_tdx_family_provider_names_are_rejected_by_default(self) -> None:
        for provider_name in ["tqcenter", "tdx", "pytdx", "mootdx"]:
            with self.subTest(provider_name=provider_name):
                with self.assertRaisesRegex(ValueError, "TDX-family"):
                    validate_provider_name(provider_name)

    def test_provider_manager_records_empty_and_error_reports(self) -> None:
        request = FetchRequest(
            symbols=("000001.SZ",),
            start_date="2026-01-05",
            end_date="2026-01-05",
        )
        empty = InMemoryMarketProvider("eastmoney_efinance", pd.DataFrame())

        def fail_provider(_: FetchRequest) -> ProviderResult:
            raise RuntimeError("provider failed")

        manager = ProviderManager(
            providers=[
                empty,
                InMemoryMarketProvider("akshare_eastmoney", fail_provider),
            ]
        )

        result = manager.fetch_market_bars(request)

        self.assertTrue(result.data.empty)
        self.assertEqual(result.coverage_report["provider_count"], 2)
        codes = {item["code"] for item in result.error_report}
        self.assertIn("empty_return", codes)
        self.assertIn("provider_exception", codes)
        self.assertEqual(result.coverage_report["status"], "no_data")

    def test_provider_manager_does_not_accept_execution_timeout(self) -> None:
        request = FetchRequest(
            symbols=("000001.SZ",),
            start_date="2026-01-05",
            end_date="2026-01-05",
        )
        with self.assertRaises(TypeError):
            ProviderManager(
                providers=[InMemoryMarketProvider("eastmoney_efinance", pd.DataFrame())],
                timeout_seconds=0.05,
            ).fetch_market_bars(request)

    def test_efinance_provider_preserves_requested_symbol_suffix(self) -> None:
        raw = pd.DataFrame(
            {
                "date": ["2026-01-05"],
                "open": [4000.0],
                "high": [4010.0],
                "low": [3990.0],
                "close": [4005.0],
                "volume": [3000],
                "amount": [12015000],
            }
        )
        fake_efinance = types.SimpleNamespace(
            stock=types.SimpleNamespace(get_quote_history=lambda **_: raw.copy())
        )
        request = FetchRequest(symbols=("000300.SH",), start_date="2026-01-05", end_date="2026-01-05")

        with mock.patch.dict(sys.modules, {"efinance": fake_efinance}):
            result = EastmoneyEfinanceProvider().fetch_market_bars(request)

        self.assertEqual(result.data["symbol"].iloc[0], "000300.SH")

    def test_default_free_provider_plan_excludes_unimplemented_realtime_adapter(self) -> None:
        default_names = [provider.name for provider in build_default_providers("default_free")]
        realtime_names = [provider.name for provider in build_default_providers("default_free_with_realtime")]
        baostock_only_names = [provider.name for provider in build_default_providers("baostock_only")]

        self.assertNotIn("sina_tencent_realtime", default_names)
        self.assertIn("sina_tencent_realtime", realtime_names)
        self.assertEqual(baostock_only_names, ["baostock"])

    def test_formal_free_v3_provider_plan_has_required_free_domains_without_tdx(self) -> None:
        provider_names = [provider.name for provider in build_default_providers("formal_free_v3")]
        matrix = provider_capability_matrix("formal_free_v3")
        required_domains = {
            item["domain"]
            for item in matrix
            if item.get("formal_default") is True and item.get("requirement") == "required"
        }
        formal_refresh_domains = {
            item["domain"]
            for item in matrix
            if item.get("formal_refresh") is True and item.get("requirement") == "required"
        }
        optional_domains = {
            item["domain"]
            for item in matrix
            if item.get("formal_default") is True and item.get("requirement") == "optional"
        }

        self.assertIn("baostock", provider_names)
        self.assertIn("eastmoney_efinance", provider_names)
        self.assertIn("akshare_eastmoney", provider_names)
        self.assertIn("tencent_finance", provider_names)
        self.assertIn("tonghuashun_hotspot", provider_names)
        self.assertFalse({"tq", "tdx", "pytdx", "mootdx"} & set(provider_names))
        self.assertEqual(
            required_domains,
            {"market_daily", "trading_calendar", "universe_snapshot", "security_status", "limit_status"},
        )
        self.assertEqual(formal_refresh_domains, required_domains)
        self.assertGreaterEqual(optional_domains, {"valuation", "industry_concept", "money_flow_hotspot"})

    def test_research_rebuild_minimal_free_plan_isolates_weak_providers_from_critical_path(self) -> None:
        providers = build_default_providers("research_rebuild_minimal_free")
        provider_names = [provider.name for provider in providers]
        matrix = provider_capability_matrix("research_rebuild_minimal_free")
        formal_refresh_domains = {
            item["domain"]
            for item in matrix
            if item.get("formal_refresh") is True and item.get("requirement") == "required"
        }

        self.assertEqual(provider_names, ["research_rebuild_minimal_free"])
        self.assertEqual(
            formal_refresh_domains,
            {"market_daily", "trading_calendar", "universe_snapshot"},
        )
        self.assertFalse({"akshare_eastmoney", "sina_tencent_realtime", "tushare_http_optional"} & set(provider_names))

    def test_research_rebuild_minimal_free_provider_routes_only_rebuild_domains(self) -> None:
        provider = ResearchRebuildMinimalFreeProvider()
        calls: list[tuple[str, str]] = []

        def fake_baostock(request: DomainFetchRequest) -> ProviderResult:
            calls.append(("baostock", request.domain))
            return ProviderResult(provider="baostock", data=pd.DataFrame({"symbol": ["000001.SZ"], "trade_date": ["2026-01-05"]}))

        def fake_eastmoney(request: DomainFetchRequest) -> ProviderResult:
            calls.append(("eastmoney_efinance", request.domain))
            return ProviderResult(provider="eastmoney_efinance", data=pd.DataFrame({"symbol": ["000001.SZ"], "trade_date": ["2026-01-05"]}))

        provider._baostock.fetch_domain = fake_baostock  # type: ignore[method-assign]
        provider._eastmoney.fetch_domain = fake_eastmoney  # type: ignore[method-assign]

        provider.fetch_domain(DomainFetchRequest(domain=DataDomain.MARKET_DAILY, start_date="2026-01-05", end_date="2026-01-05"))
        provider.fetch_domain(DomainFetchRequest(domain=DataDomain.TRADING_CALENDAR, start_date="2026-01-05", end_date="2026-01-05"))
        provider.fetch_domain(DomainFetchRequest(domain=DataDomain.UNIVERSE_SNAPSHOT, start_date="2026-01-05", end_date="2026-01-05"))

        self.assertEqual(
            calls,
            [
                ("baostock", DataDomain.MARKET_DAILY),
                ("baostock", DataDomain.TRADING_CALENDAR),
                ("baostock", DataDomain.UNIVERSE_SNAPSHOT),
            ],
        )
        with self.assertRaisesRegex(RuntimeError, "unsupported_domain"):
            provider.fetch_domain(DomainFetchRequest(domain=DataDomain.LIMIT_STATUS, start_date="2026-01-05", end_date="2026-01-05"))

    def test_research_rebuild_minimal_free_routes_universe_to_baostock_when_eastmoney_fails(self) -> None:
        provider = ResearchRebuildMinimalFreeProvider()

        def fake_baostock(request: DomainFetchRequest) -> ProviderResult:
            self.assertEqual(request.domain, DataDomain.UNIVERSE_SNAPSHOT)
            return ProviderResult(
                provider="baostock",
                data=pd.DataFrame(
                    {
                        "symbol": ["600000.SH"],
                        "trade_date": ["2026-01-05"],
                        "name": ["浦发银行"],
                        "exchange": ["SH"],
                        "board": ["main"],
                        "list_status": ["L"],
                        "list_date": [""],
                        "delist_date": [""],
                        "source": ["baostock"],
                    }
                ),
            )

        def fail_eastmoney(_: DomainFetchRequest) -> ProviderResult:
            raise AssertionError("Eastmoney must not be used for rebuild universe")

        provider._baostock.fetch_domain = fake_baostock  # type: ignore[method-assign]
        provider._eastmoney.fetch_domain = fail_eastmoney  # type: ignore[method-assign]

        result = provider.fetch_domain(
            DomainFetchRequest(domain=DataDomain.UNIVERSE_SNAPSHOT, start_date="2026-01-05", end_date="2026-01-05")
        )

        self.assertEqual(result.provider, "baostock")
        self.assertEqual(result.data["source"].tolist(), ["baostock"])

    def test_tushare_http_provider_does_not_set_execution_timeout(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeResponse:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict[str, object]:
                return {"code": 0, "data": {"fields": ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"], "items": []}}

        def fake_post(*args, **kwargs):
            calls.append(dict(kwargs))
            return FakeResponse()

        request = FetchRequest(symbols=("000001.SZ",), start_date="2026-01-05", end_date="2026-01-05")
        with mock.patch("daily_research.data_platform.providers.requests.post", fake_post):
            TushareHttpOptionalProvider(token="token").fetch_market_bars(request)

        self.assertEqual(len(calls), 1)
        self.assertNotIn("timeout", calls[0])

    def test_baostock_stock_basic_frame_maps_stock_rows(self) -> None:
        class FakeQuery:
            fields = ["code", "code_name", "ipoDate", "outDate", "type", "status"]
            error_code = "0"
            error_msg = "success"

            def __init__(self) -> None:
                self._rows = [
                    ["sh.000001", "上证综合指数", "1991-07-15", "", "2", "1"],
                    ["sh.600000", "浦发银行", "1999-11-10", "", "1", "1"],
                    ["sz.000001", "平安银行", "1991-04-03", "", "1", "1"],
                ]
                self._index = -1

            def next(self) -> bool:
                self._index += 1
                return self._index < len(self._rows)

            def get_row_data(self) -> list[str]:
                return self._rows[self._index]

        frame = _baostock_stock_basic_frame(FakeQuery(), trade_date="2026-05-22")

        self.assertEqual(frame["symbol"].tolist(), ["600000.SH", "000001.SZ"])
        self.assertEqual(frame["name"].tolist(), ["浦发银行", "平安银行"])
        self.assertEqual(frame["list_status"].tolist(), ["L", "L"])
        self.assertEqual(frame["list_date"].tolist(), ["1999-11-10", "1991-04-03"])

    def test_baostock_all_stock_frame_normalizes_daily_universe(self) -> None:
        class FakeQuery:
            fields = ["code", "tradeStatus", "code_name"]
            error_code = "0"
            error_msg = "success"

            def __init__(self) -> None:
                self._rows = [
                    ["sh.600000", "1", "浦发银行"],
                    ["sz.000001", "1", "平安银行"],
                    ["bj.430047", "1", "诺思兰德"],
                    ["sh.000001", "1", "上证综合指数"],
                    ["sh.510300", "1", "沪深300ETF"],
                ]
                self._index = -1

            def next(self) -> bool:
                self._index += 1
                return self._index < len(self._rows)

            def get_row_data(self) -> list[str]:
                return self._rows[self._index]

        frame = _baostock_all_stock_frame(FakeQuery(), trade_date="2026-05-22")

        self.assertEqual(frame["symbol"].tolist(), ["600000.SH", "000001.SZ", "430047.BJ"])
        self.assertEqual(frame["exchange"].tolist(), ["SH", "SZ", "BJ"])
        self.assertEqual(frame["board"].tolist(), ["main", "main", "beijing"])
        self.assertEqual(frame["list_status"].tolist(), ["L", "L", "L"])
        self.assertEqual(frame["source"].tolist(), ["baostock", "baostock", "baostock"])

    def test_baostock_security_status_from_all_stock_trade_status(self) -> None:
        class FakeQuery:
            fields = ["code", "tradeStatus", "code_name"]
            error_code = "0"
            error_msg = "success"

            def __init__(self) -> None:
                self._rows = [
                    ["sh.600000", "1", "浦发银行"],
                    ["sz.000001", "0", "*ST平安"],
                ]
                self._index = -1

            def next(self) -> bool:
                self._index += 1
                return self._index < len(self._rows)

            def get_row_data(self) -> list[str]:
                return self._rows[self._index]

        frame = _baostock_status_frame_from_all_stock(FakeQuery(), trade_date="2026-05-22")

        self.assertEqual(frame["symbol"].tolist(), ["600000.SH", "000001.SZ"])
        self.assertEqual(frame["is_suspended"].tolist(), [False, True])
        self.assertEqual(frame["is_st"].tolist(), [False, True])
        self.assertEqual(frame["is_delisted"].tolist(), [False, False])
        self.assertEqual(frame["status_reason"].tolist(), ["tradeStatus=1", "tradeStatus=0"])

    def test_baostock_industry_frame_does_not_emit_concepts(self) -> None:
        class FakeQuery:
            fields = ["updateDate", "code", "code_name", "industry", "industryClassification"]
            error_code = "0"
            error_msg = "success"

            def __init__(self) -> None:
                self._rows = [["2026-05-18", "sh.600000", "浦发银行", "J66货币金融服务", "证监会行业分类"]]
                self._index = -1

            def next(self) -> bool:
                self._index += 1
                return self._index < len(self._rows)

            def get_row_data(self) -> list[str]:
                return self._rows[self._index]

        frame = _baostock_industry_frame(FakeQuery(), trade_date="2026-05-22")

        self.assertEqual(frame["symbol"].tolist(), ["600000.SH"])
        self.assertEqual(frame["trade_date"].tolist(), ["2026-05-22"])
        self.assertEqual(frame["industry"].tolist(), ["J66货币金融服务"])
        self.assertEqual(frame["concept_tags"].tolist(), [""])

    def test_baostock_valuation_frame_maps_supported_fields_only(self) -> None:
        class FakeQuery:
            fields = ["date", "code", "turn", "peTTM", "pbMRQ"]
            error_code = "0"
            error_msg = "success"

            def __init__(self) -> None:
                self._rows = [["2026-05-22", "sh.600000", "1.23", "6.5", "0.8"]]
                self._index = -1

            def next(self) -> bool:
                self._index += 1
                return self._index < len(self._rows)

            def get_row_data(self) -> list[str]:
                return self._rows[self._index]

        fake_bs = types.SimpleNamespace(
            query_history_k_data_plus=mock.Mock(return_value=FakeQuery()),
        )
        request = DomainFetchRequest(domain=DataDomain.VALUATION, symbols=("600000.SH",), start_date="2026-05-22", end_date="2026-05-22")

        frame = _baostock_valuation_frame_from_history(fake_bs, request)

        fake_bs.query_history_k_data_plus.assert_called_once_with(
            "sh.600000",
            "date,code,turn,peTTM,pbMRQ",
            start_date="2026-05-22",
            end_date="2026-05-22",
            frequency="d",
            adjustflag="3",
        )
        self.assertEqual(frame["symbol"].tolist(), ["600000.SH"])
        self.assertEqual(frame["pe"].tolist(), ["6.5"])
        self.assertEqual(frame["pb"].tolist(), ["0.8"])
        self.assertEqual(frame["turnover_rate"].tolist(), ["1.23"])
        self.assertTrue(pd.isna(frame["total_mv"].iloc[0]))
        self.assertTrue(pd.isna(frame["circ_mv"].iloc[0]))

    def test_baostock_security_status_uses_guarded_stock_basic_fetch(self) -> None:
        guarded_frame = pd.DataFrame(
            {
                "symbol": ["000001.SZ"],
                "trade_date": ["2026-05-22"],
                "is_st": [False],
                "is_suspended": [False],
                "is_delisted": [False],
                "status_reason": ["1"],
                "source": ["baostock"],
            }
        )

        with mock.patch(
            "daily_research.data_platform.providers._fetch_baostock_all_stock_frame_with_timeout",
            return_value=guarded_frame,
        ) as guarded:
            result = BaostockProvider().fetch_domain(
                DomainFetchRequest(domain=DataDomain.SECURITY_STATUS, start_date="2026-05-22", end_date="2026-05-22")
            )

        guarded.assert_called_once_with(domain=DataDomain.SECURITY_STATUS, trade_date="2026-05-22")
        self.assertEqual(result.data["symbol"].tolist(), ["000001.SZ"])
        self.assertEqual(result.data["source"].tolist(), ["baostock"])

    def test_baostock_calendar_uses_guarded_fetch(self) -> None:
        guarded_frame = pd.DataFrame(
            {
                "trade_date": ["2026-05-22"],
                "is_open": [True],
                "exchange": ["SSE"],
                "source": ["baostock"],
            }
        )

        with mock.patch(
            "daily_research.data_platform.providers._fetch_baostock_trade_calendar_frame_with_timeout",
            return_value=guarded_frame,
        ) as guarded:
            result = BaostockProvider().fetch_domain(
                DomainFetchRequest(
                    domain=DataDomain.TRADING_CALENDAR,
                    start_date="2026-05-22",
                    end_date="2026-05-22",
                    exchange="SSE",
                )
            )

        guarded.assert_called_once_with(start_date="2026-05-22", end_date="2026-05-22", exchange="SSE")
        self.assertEqual(result.data["trade_date"].tolist(), ["2026-05-22"])
        self.assertEqual(result.data["source"].tolist(), ["baostock"])

    def test_baostock_stock_basic_guard_times_out_and_terminates_child(self) -> None:
        from daily_research.data_platform import providers

        class FakeQueue:
            def get(self, timeout: float | None = None) -> object:
                raise providers.queue_module.Empty

        class FakeProcess:
            exitcode = None

            def __init__(self) -> None:
                self.started = False
                self.terminated = False

            def start(self) -> None:
                self.started = True

            def join(self, timeout: float | None = None) -> None:
                return None

            def is_alive(self) -> bool:
                return not self.terminated

            def terminate(self) -> None:
                self.terminated = True

        fake_process = FakeProcess()

        class FakeContext:
            def Queue(self) -> FakeQueue:
                return FakeQueue()

            def Process(self, **_: object) -> FakeProcess:
                return fake_process

        with mock.patch("daily_research.data_platform.providers.multiprocessing.get_context", return_value=FakeContext()):
            with self.assertRaisesRegex(TimeoutError, "baostock_stock_basic_timeout"):
                providers._fetch_baostock_stock_basic_frame_with_timeout(trade_date="2026-05-22", timeout_seconds=1)

        self.assertTrue(fake_process.started)
        self.assertTrue(fake_process.terminated)

    def test_baostock_stock_basic_guard_uses_blocking_queue_get(self) -> None:
        from daily_research.data_platform import providers

        expected = pd.DataFrame({"symbol": ["000001.SZ"], "trade_date": ["2026-05-22"]})

        class FakeQueue:
            def empty(self) -> bool:
                return True

            def get(self, timeout: float | None = None) -> dict[str, object]:
                return {"status": "ok", "data": expected}

        class FakeProcess:
            exitcode = 0

            def start(self) -> None:
                return None

            def join(self, timeout: float | None = None) -> None:
                return None

            def is_alive(self) -> bool:
                return False

        class FakeContext:
            def Queue(self) -> FakeQueue:
                return FakeQueue()

            def Process(self, **_: object) -> FakeProcess:
                return FakeProcess()

        with mock.patch("daily_research.data_platform.providers.multiprocessing.get_context", return_value=FakeContext()):
            frame = providers._fetch_baostock_stock_basic_frame_with_timeout(trade_date="2026-05-22", timeout_seconds=1)

        pd.testing.assert_frame_equal(frame, expected)

    def test_baostock_payload_guard_reads_queue_before_join_deadlock(self) -> None:
        from daily_research.data_platform import providers

        expected = pd.DataFrame({"symbol": ["600000.SH"], "trade_date": ["2026-05-22"]})
        events: list[str] = []

        class FakeQueue:
            def get(self, timeout: float | None = None) -> dict[str, object]:
                events.append("get")
                fake_process.drained = True
                return {"status": "ok", "data": expected}

        class FakeProcess:
            exitcode = None

            def __init__(self) -> None:
                self.drained = False

            def start(self) -> None:
                events.append("start")

            def join(self, timeout: float | None = None) -> None:
                events.append("join")

            def is_alive(self) -> bool:
                return not self.drained

            def terminate(self) -> None:
                events.append("terminate")

        fake_process = FakeProcess()

        class FakeContext:
            def Queue(self) -> FakeQueue:
                return FakeQueue()

            def Process(self, **_: object) -> FakeProcess:
                return fake_process

        with mock.patch("daily_research.data_platform.providers.multiprocessing.get_context", return_value=FakeContext()):
            frame = providers._fetch_baostock_all_stock_frame_with_timeout(
                domain=DataDomain.UNIVERSE_SNAPSHOT,
                trade_date="2026-05-22",
                timeout_seconds=1,
            )

        pd.testing.assert_frame_equal(frame, expected)
        self.assertLess(events.index("get"), events.index("join"))
        self.assertNotIn("terminate", events)

    def test_baostock_all_stock_guard_times_out_and_terminates_child(self) -> None:
        from daily_research.data_platform import providers

        class FakeQueue:
            def get(self, timeout: float | None = None) -> object:
                raise providers.queue_module.Empty

        class FakeProcess:
            exitcode = None

            def __init__(self) -> None:
                self.started = False
                self.terminated = False

            def start(self) -> None:
                self.started = True

            def join(self, timeout: float | None = None) -> None:
                return None

            def is_alive(self) -> bool:
                return not self.terminated

            def terminate(self) -> None:
                self.terminated = True

        fake_process = FakeProcess()

        class FakeContext:
            def Queue(self) -> FakeQueue:
                return FakeQueue()

            def Process(self, **_: object) -> FakeProcess:
                return fake_process

        with mock.patch("daily_research.data_platform.providers.multiprocessing.get_context", return_value=FakeContext()):
            with self.assertRaisesRegex(TimeoutError, "baostock_all_stock_timeout"):
                providers._fetch_baostock_all_stock_frame_with_timeout(
                    domain=DataDomain.UNIVERSE_SNAPSHOT,
                    trade_date="2026-05-22",
                    timeout_seconds=1,
                )

        self.assertTrue(fake_process.started)
        self.assertTrue(fake_process.terminated)

    def test_baostock_trade_calendar_guard_times_out_and_terminates_child(self) -> None:
        from daily_research.data_platform import providers

        class FakeQueue:
            def get(self, timeout: float | None = None) -> object:
                raise providers.queue_module.Empty

        class FakeProcess:
            exitcode = None

            def __init__(self) -> None:
                self.started = False
                self.terminated = False

            def start(self) -> None:
                self.started = True

            def join(self, timeout: float | None = None) -> None:
                return None

            def is_alive(self) -> bool:
                return not self.terminated

            def terminate(self) -> None:
                self.terminated = True

        fake_process = FakeProcess()

        class FakeContext:
            def Queue(self) -> FakeQueue:
                return FakeQueue()

            def Process(self, **_: object) -> FakeProcess:
                return fake_process

        with mock.patch("daily_research.data_platform.providers.multiprocessing.get_context", return_value=FakeContext()):
            with self.assertRaisesRegex(TimeoutError, "baostock_trade_calendar_timeout"):
                providers._fetch_baostock_trade_calendar_frame_with_timeout(
                    start_date="2026-05-22",
                    end_date="2026-05-22",
                    exchange="SSE",
                    timeout_seconds=1,
                )

        self.assertTrue(fake_process.started)
        self.assertTrue(fake_process.terminated)


if __name__ == "__main__":
    unittest.main()
