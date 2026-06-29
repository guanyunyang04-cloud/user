import unittest
import sys
import types
from unittest import mock

import pandas as pd

from quant_data_platform.domains.contracts import (
    DataDomain,
    DomainFetchRequest,
    STANDARD_MARKET_COLUMNS,
    FetchRequest,
    ProviderResult,
    normalize_market_frame,
    validate_provider_name,
)
from quant_data_platform.provider_manager import InMemoryMarketProvider, ProviderManager
from quant_data_platform.providers import (
    BaostockProvider,
    MootdxOnlineProvider,
    QdpProductionV1Provider,
    EastmoneyEfinanceProvider,
    ResearchRebuildMinimalFreeProvider,
    TushareHttpOptionalProvider,
    build_default_providers,
    provider_capability_matrix,
)
from quant_data_platform.providers import (
    _baostock_all_stock_frame,
    _baostock_industry_frame,
    _baostock_status_frame_from_all_stock,
    _baostock_stock_basic_frame,
    _baostock_valuation_frame_from_history,
    _baostock_adjust_factor_frame_from_bs,
    _baostock_history_frame,
    _baostock_intraday_5m_frame,
    _baostock_index_constituents_frame,
    _to_baostock_code,
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
        production_names = [provider.name for provider in build_default_providers("qdp_production_v1")]

        self.assertNotIn("sina_tencent_realtime", default_names)
        self.assertIn("sina_tencent_realtime", realtime_names)
        self.assertEqual(baostock_only_names, ["baostock"])
        self.assertEqual(production_names, ["qdp_production_v1"])

    def test_qdp_production_v1_matrix_records_source_split(self) -> None:
        matrix = provider_capability_matrix("qdp_production_v1")
        defaults = {
            (item["provider"], item["domain"])
            for item in matrix
            if item.get("formal_default") is True
        }

        self.assertIn(("mootdx_online", DataDomain.MARKET_DAILY), defaults)
        self.assertIn(("mootdx_online", DataDomain.MARKET_INTRADAY_5M), defaults)
        self.assertIn(("baostock", DataDomain.TRADING_CALENDAR), defaults)
        self.assertIn(("baostock", DataDomain.UNIVERSE_SNAPSHOT), defaults)
        self.assertNotIn(("cninfo", DataDomain.ANNOUNCEMENT), defaults)

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

    def test_qdp_production_v1_provider_routes_domains_to_selected_sources(self) -> None:
        provider = QdpProductionV1Provider()
        calls: list[tuple[str, str]] = []

        def fake_mootdx(request: DomainFetchRequest) -> ProviderResult:
            calls.append(("mootdx_online", request.domain))
            return ProviderResult(provider="mootdx_online", data=pd.DataFrame({"symbol": ["000001.SZ"], "trade_date": ["2026-01-05"], "source": ["mootdx_online"]}))

        def fake_baostock(request: DomainFetchRequest) -> ProviderResult:
            calls.append(("baostock", request.domain))
            return ProviderResult(provider="baostock", data=pd.DataFrame({"symbol": ["000001.SZ"], "trade_date": ["2026-01-05"], "source": ["baostock"]}))

        def fake_cninfo(request: DomainFetchRequest) -> ProviderResult:
            calls.append(("cninfo", request.domain))
            return ProviderResult(provider="cninfo", data=pd.DataFrame({"symbol": ["000001.SZ"], "trade_date": ["2026-01-05"], "source": ["cninfo"]}))

        provider._mootdx.fetch_domain = fake_mootdx  # type: ignore[method-assign]
        provider._baostock.fetch_domain = fake_baostock  # type: ignore[method-assign]
        provider._cninfo.fetch_domain = fake_cninfo  # type: ignore[method-assign]

        provider.fetch_domain(DomainFetchRequest(domain=DataDomain.MARKET_DAILY, start_date="2026-01-05", end_date="2026-01-05"))
        provider.fetch_domain(DomainFetchRequest(domain=DataDomain.MARKET_INTRADAY_5M, start_date="2026-01-05", end_date="2026-01-05"))
        provider.fetch_domain(DomainFetchRequest(domain=DataDomain.INTRADAY_DAILY_FEATURES, start_date="2026-01-05", end_date="2026-01-05"))
        provider.fetch_domain(DomainFetchRequest(domain=DataDomain.TRADING_CALENDAR, start_date="2026-01-05", end_date="2026-01-05"))
        provider.fetch_domain(DomainFetchRequest(domain=DataDomain.ANNOUNCEMENT, start_date="2026-01-05", end_date="2026-01-05"))

        self.assertEqual(
            calls,
            [
                ("mootdx_online", DataDomain.MARKET_DAILY),
                ("mootdx_online", DataDomain.MARKET_INTRADAY_5M),
                ("mootdx_online", DataDomain.INTRADAY_DAILY_FEATURES),
                ("baostock", DataDomain.TRADING_CALENDAR),
                ("cninfo", DataDomain.ANNOUNCEMENT),
            ],
        )

    def test_mootdx_online_daily_bars_normalize_volume_from_hands_to_shares(self) -> None:
        class FakeClient:
            def bars(self, **_: object) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "open": [10.0],
                        "high": [10.5],
                        "low": [9.8],
                        "close": [10.2],
                        "vol": [1234.0],
                        "amount": [1258680.0],
                        "datetime": ["2026-01-05 15:00:00"],
                    }
                )

            def close(self) -> None:
                return None

        provider = MootdxOnlineProvider(_client_factory=FakeClient, page_size=10, max_pages=1)
        result = provider.fetch_market_bars(FetchRequest(symbols=("000001.SZ",), start_date="2026-01-05", end_date="2026-01-05"))

        self.assertEqual(result.data["volume"].tolist(), [123400.0])
        self.assertEqual(result.data["source"].tolist(), ["mootdx_online"])
        self.assertEqual(result.coverage_report["volume_factor"], 100.0)
        self.assertEqual(result.coverage_report["raw_volume_unit"], "hands")

    def test_mootdx_online_5m_bars_keep_share_volume_unit(self) -> None:
        class FakeClient:
            def bars(self, **_: object) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "open": [10.0],
                        "high": [10.5],
                        "low": [9.8],
                        "close": [10.2],
                        "vol": [123400.0],
                        "amount": [1258680.0],
                        "datetime": ["2026-01-05 09:35:00"],
                    }
                )

            def close(self) -> None:
                return None

        provider = MootdxOnlineProvider(_client_factory=FakeClient, page_size=10, max_pages=1)
        result = provider.fetch_domain(
            DomainFetchRequest(
                domain=DataDomain.MARKET_INTRADAY_5M,
                symbols=("000001.SZ",),
                start_date="2026-01-05",
                end_date="2026-01-05",
            )
        )

        self.assertEqual(result.data["volume"].tolist(), [123400.0])
        self.assertEqual(result.data["bar_time"].tolist(), ["093500000"])
        self.assertEqual(result.coverage_report["volume_factor"], 1.0)
        self.assertEqual(result.coverage_report["raw_volume_unit"], "shares")

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
        with mock.patch("quant_data_platform.providers.requests.post", fake_post):
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

    def test_baostock_valuation_frame_relogs_in_when_session_drops(self) -> None:
        class ErrorQuery:
            fields: list[str] = []
            error_code = "10001001"
            error_msg = "用户未登录"

        class GoodQuery:
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
            query_history_k_data_plus=mock.Mock(side_effect=[ErrorQuery(), GoodQuery()]),
            logout=mock.Mock(),
            login=mock.Mock(return_value=types.SimpleNamespace(error_code="0", error_msg="success")),
        )
        request = DomainFetchRequest(domain=DataDomain.VALUATION, symbols=("600000.SH",), start_date="2026-05-22", end_date="2026-05-22")

        with mock.patch("quant_data_platform.providers.time.sleep") as sleep:
            frame = _baostock_valuation_frame_from_history(fake_bs, request)

        self.assertEqual(fake_bs.query_history_k_data_plus.call_count, 2)
        fake_bs.logout.assert_called_once()
        fake_bs.login.assert_called_once()
        sleep.assert_called_once()
        self.assertEqual(frame["symbol"].tolist(), ["600000.SH"])
        self.assertEqual(frame["pe"].tolist(), ["6.5"])

    def test_baostock_history_frame_maps_daily_market_rows(self) -> None:
        class FakeQuery:
            fields = ["date", "code", "open", "high", "low", "close", "volume", "amount"]
            error_code = "0"
            error_msg = "success"

            def __init__(self) -> None:
                self._rows = [["2026-05-22", "sh.600000", "10", "11", "9", "10.5", "100", "1050"]]
                self._index = -1

            def next(self) -> bool:
                self._index += 1
                return self._index < len(self._rows)

            def get_row_data(self) -> list[str]:
                return self._rows[self._index]

        frame = _baostock_history_frame(FakeQuery(), symbol="600000.SH")

        self.assertEqual(frame["trade_date"].tolist(), ["2026-05-22"])
        self.assertEqual(frame["symbol"].tolist(), ["600000.SH"])
        self.assertEqual(frame["close"].tolist(), ["10.5"])

    def test_baostock_intraday_5m_frame_maps_minute_rows(self) -> None:
        class FakeQuery:
            fields = ["date", "time", "code", "open", "high", "low", "close", "volume", "amount", "adjustflag"]
            error_code = "0"
            error_msg = "success"

            def __init__(self) -> None:
                self._rows = [["2026-05-22", "20260522093500000", "sh.600000", "10", "10.2", "9.9", "10.1", "100", "1010", "3"]]
                self._index = -1

            def next(self) -> bool:
                self._index += 1
                return self._index < len(self._rows)

            def get_row_data(self) -> list[str]:
                return self._rows[self._index]

        frame = _baostock_intraday_5m_frame(FakeQuery(), symbol="600000.SH")

        self.assertEqual(frame["trade_date"].tolist(), ["2026-05-22"])
        self.assertEqual(frame["symbol"].tolist(), ["600000.SH"])
        self.assertEqual(frame["bar_time"].tolist(), ["20260522093500000"])
        self.assertEqual(frame["adjusted_flag"].tolist(), ["3"])

    def test_baostock_code_converter_preserves_beijing_exchange_suffix(self) -> None:
        self.assertEqual(_to_baostock_code("430047.BJ"), "bj.430047")
        self.assertEqual(_to_baostock_code("830799"), "bj.830799")

    def test_baostock_intraday_daily_features_aggregate_5m_without_auction_process(self) -> None:
        rows = []
        for idx, close in enumerate([10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.4, 10.2], start=1):
            rows.append(
                {
                    "symbol": "600000.SH",
                    "trade_date": "2026-05-22",
                    "bar_time": f"{93000 + idx * 500:06d}000",
                    "open": 10.0 if idx == 1 else close - 0.05,
                    "high": close + 0.1,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1000 + idx,
                    "amount": (1000 + idx) * close,
                    "source": "baostock",
                    "adjusted_flag": "none",
                }
            )

        from quant_data_platform.domains.contracts import build_intraday_daily_feature_frame

        features = build_intraday_daily_feature_frame(pd.DataFrame(rows), source="baostock")

        self.assertEqual(features["symbol"].tolist(), ["600000.SH"])
        self.assertGreater(features["first_30m_ret"].iloc[0], 0.0)
        self.assertLess(features["last_30m_ret"].iloc[0], 0.0)
        self.assertIn("close_pressure_30m", features.columns)
        self.assertIn("close_position", features.columns)
        self.assertIn("early_strength_late_weak", features.columns)
        self.assertIn("open_gap_first_30m_reversal", features.columns)
        self.assertIn("high_time_frac", features.columns)
        self.assertIn("low_time_frac", features.columns)
        self.assertIn("amount_concentration_hhi", features.columns)
        self.assertIn("intraday_max_drawdown", features.columns)
        self.assertGreaterEqual(features["high_time_frac"].iloc[0], 0.0)
        self.assertLessEqual(features["high_time_frac"].iloc[0], 1.0)
        self.assertLess(features["intraday_max_drawdown"].iloc[0], 0.0)
        self.assertIn("opening_auction_pressure", features.columns)
        self.assertIn("closing_auction_pressure", features.columns)
        self.assertGreater(features["opening_auction_amount"].iloc[0], 0.0)
        self.assertGreater(features["closing_auction_amount"].iloc[0], 0.0)

    def test_baostock_index_constituents_frame_combines_supported_indices(self) -> None:
        class FakeQuery:
            fields = ["date", "code", "code_name"]
            error_code = "0"
            error_msg = "success"

            def __init__(self, code: str) -> None:
                self._rows = [["2026-05-22", code, "成分股"]]
                self._index = -1

            def next(self) -> bool:
                self._index += 1
                return self._index < len(self._rows)

            def get_row_data(self) -> list[str]:
                return self._rows[self._index]

        fake_bs = types.SimpleNamespace(
            query_sz50_stocks=mock.Mock(return_value=FakeQuery("sh.600000")),
            query_hs300_stocks=mock.Mock(return_value=FakeQuery("sz.000001")),
            query_zz500_stocks=mock.Mock(return_value=FakeQuery("sz.000002")),
        )

        frame = _baostock_index_constituents_frame(fake_bs, trade_date="2026-05-22")

        self.assertEqual(set(frame["index_symbol"]), {"000016.SH", "000300.SH", "000905.SH"})
        self.assertEqual(set(frame["symbol"]), {"600000.SH", "000001.SZ", "000002.SZ"})

    def test_baostock_market_daily_uses_guarded_per_symbol_fetch(self) -> None:
        guarded_frames = [
            pd.DataFrame(
                {
                    "trade_date": ["2026-05-22"],
                    "symbol": ["600000.SH"],
                    "open": ["10"],
                    "high": ["11"],
                    "low": ["9"],
                    "close": ["10.5"],
                    "volume": ["100"],
                    "amount": ["1050"],
                }
            ),
            pd.DataFrame(
                {
                    "trade_date": ["2026-05-22"],
                    "symbol": ["000001.SZ"],
                    "open": ["20"],
                    "high": ["21"],
                    "low": ["19"],
                    "close": ["20.5"],
                    "volume": ["200"],
                    "amount": ["4100"],
                }
            ),
        ]

        with mock.patch(
            "quant_data_platform.providers._fetch_baostock_history_frame_with_timeout",
            side_effect=guarded_frames,
        ) as guarded:
            result = BaostockProvider().fetch_market_bars(
                FetchRequest(
                    symbols=("600000.SH", "000001.SZ"),
                    start_date="2026-05-22",
                    end_date="2026-05-22",
                )
            )

        self.assertEqual(
            {call.kwargs["symbol"] for call in guarded.call_args_list},
            {"600000.SH", "000001.SZ"},
        )
        self.assertEqual(set(result.data["symbol"].tolist()), {"600000.SH", "000001.SZ"})
        self.assertEqual(result.error_report, [])

    def test_baostock_market_daily_records_timed_out_symbols_without_hanging_batch(self) -> None:
        def guarded_fetch(*, symbol: str, **_: object) -> pd.DataFrame:
            if symbol == "600001.SH":
                raise TimeoutError("baostock_history_timeout: exceeded 1 seconds")
            return pd.DataFrame(
                {
                    "trade_date": ["2026-05-22"],
                    "symbol": [symbol],
                    "open": ["10"],
                    "high": ["11"],
                    "low": ["9"],
                    "close": ["10.5"],
                    "volume": ["100"],
                    "amount": ["1050"],
                }
            )

        with mock.patch(
            "quant_data_platform.providers._fetch_baostock_history_frame_with_timeout",
            side_effect=guarded_fetch,
        ):
            result = BaostockProvider().fetch_market_bars(
                FetchRequest(
                    symbols=("600000.SH", "600001.SH"),
                    start_date="2026-05-22",
                    end_date="2026-05-22",
                )
            )

        self.assertEqual(result.data["symbol"].tolist(), ["600000.SH"])
        self.assertEqual(len(result.error_report), 1)
        self.assertEqual(result.error_report[0]["symbol"], "600001.SH")
        self.assertEqual(result.error_report[0]["code"], "symbol_fetch_timeout")

    def test_baostock_market_daily_retries_failed_parallel_symbols_sequentially(self) -> None:
        attempts: dict[str, int] = {}

        def guarded_fetch(*, symbol: str, **_: object) -> pd.DataFrame:
            attempts[symbol] = attempts.get(symbol, 0) + 1
            if symbol == "000300.SH" and attempts[symbol] == 1:
                raise RuntimeError("baostock_history_worker_error:RuntimeError: baostock_history_query_error:10001001: 用户未登录")
            return pd.DataFrame(
                {
                    "trade_date": ["2026-05-22"],
                    "symbol": [symbol],
                    "open": ["10"],
                    "high": ["11"],
                    "low": ["9"],
                    "close": ["10.5"],
                    "volume": ["100"],
                    "amount": ["1050"],
                }
            )

        with mock.patch(
            "quant_data_platform.providers._fetch_baostock_history_frame_with_timeout",
            side_effect=guarded_fetch,
        ):
            result = BaostockProvider(_market_daily_max_workers=2).fetch_market_bars(
                FetchRequest(
                    symbols=("600000.SH", "000300.SH"),
                    start_date="2026-05-22",
                    end_date="2026-05-22",
                )
            )

        self.assertEqual(attempts["000300.SH"], 2)
        self.assertEqual(set(result.data["symbol"].tolist()), {"600000.SH", "000300.SH"})
        self.assertEqual(result.error_report, [])

    def test_baostock_market_daily_retries_failed_serial_symbols(self) -> None:
        attempts: dict[str, int] = {}

        def guarded_fetch(*, symbol: str, **_: object) -> pd.DataFrame:
            attempts[symbol] = attempts.get(symbol, 0) + 1
            if symbol == "000300.SH" and attempts[symbol] == 1:
                raise RuntimeError("baostock_history_worker_error:RuntimeError: baostock_history_query_error:10001001: 用户未登录")
            return pd.DataFrame(
                {
                    "trade_date": ["2026-05-22"],
                    "symbol": [symbol],
                    "open": ["10"],
                    "high": ["11"],
                    "low": ["9"],
                    "close": ["10.5"],
                    "volume": ["100"],
                    "amount": ["1050"],
                }
            )

        with mock.patch(
            "quant_data_platform.providers._fetch_baostock_history_frame_with_timeout",
            side_effect=guarded_fetch,
        ):
            result = BaostockProvider(_market_daily_max_workers=1).fetch_market_bars(
                FetchRequest(
                    symbols=("600000.SH", "000300.SH"),
                    start_date="2026-05-22",
                    end_date="2026-05-22",
                )
            )

        self.assertEqual(attempts["000300.SH"], 2)
        self.assertEqual(set(result.data["symbol"].tolist()), {"600000.SH", "000300.SH"})
        self.assertEqual(result.error_report, [])

    def test_baostock_history_guard_times_out_and_terminates_child(self) -> None:
        from quant_data_platform import providers

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

        with mock.patch("quant_data_platform.providers.multiprocessing.get_context", return_value=FakeContext()):
            with self.assertRaisesRegex(TimeoutError, "baostock_history_timeout"):
                providers._fetch_baostock_history_frame_with_timeout(
                    symbol="600000.SH",
                    start_date="2026-05-22",
                    end_date="2026-05-22",
                    adjusted_flag="none",
                    timeout_seconds=1,
                )

        self.assertTrue(fake_process.started)
        self.assertTrue(fake_process.terminated)

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
            "quant_data_platform.providers._fetch_baostock_all_stock_frame_with_timeout",
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
            "quant_data_platform.providers._fetch_baostock_trade_calendar_frame_with_timeout",
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

    def test_baostock_valuation_uses_guarded_fetch(self) -> None:
        guarded_frame = pd.DataFrame(
            {
                "symbol": ["600000.SH"],
                "trade_date": ["2026-05-22"],
                "total_mv": [float("nan")],
                "circ_mv": [float("nan")],
                "pe": ["10.5"],
                "pb": ["1.1"],
                "turnover_rate": ["0.8"],
                "source": ["baostock"],
            }
        )

        with mock.patch(
            "quant_data_platform.providers._fetch_baostock_valuation_frame_with_timeout",
            return_value=guarded_frame,
        ) as guarded:
            result = BaostockProvider().fetch_domain(
                DomainFetchRequest(
                    domain=DataDomain.VALUATION,
                    symbols=("600000.SH",),
                    start_date="2026-05-22",
                    end_date="2026-05-22",
                )
            )

        guarded.assert_called_once_with(symbols=("600000.SH",), start_date="2026-05-22", end_date="2026-05-22")
        self.assertEqual(result.data["symbol"].tolist(), ["600000.SH"])
        self.assertEqual(result.data["source"].tolist(), ["baostock"])

    def test_baostock_financial_query_relogs_in_after_not_logged_in(self) -> None:
        from quant_data_platform import providers

        class FakeQuery:
            def __init__(self, *, error_code: str = "0", error_msg: str = "", fields: list[str] | None = None, rows: list[list[str]] | None = None) -> None:
                self.error_code = error_code
                self.error_msg = error_msg
                self.fields = fields or []
                self.rows = rows or []
                self.index = 0

            def next(self) -> bool:
                self.index += 1
                return self.index <= len(self.rows)

            def get_row_data(self) -> list[str]:
                return self.rows[self.index - 1]

        class FakeBaoStock:
            def __init__(self) -> None:
                self.login_calls = 0
                self.logout_calls = 0
                self.profit_calls = 0

            def login(self) -> object:
                self.login_calls += 1
                return types.SimpleNamespace(error_code="0", error_msg="")

            def logout(self) -> None:
                self.logout_calls += 1

            def query_profit_data(self, **_: object) -> FakeQuery:
                self.profit_calls += 1
                if self.profit_calls == 1:
                    return FakeQuery(error_code="10001001", error_msg="用户未登录")
                return FakeQuery(fields=["roeAvg"], rows=[["1.25"]])

            def query_operation_data(self, **_: object) -> FakeQuery:
                return FakeQuery(fields=["NRTurnRatio"], rows=[])

            def query_growth_data(self, **_: object) -> FakeQuery:
                return FakeQuery(fields=["YOYEquity"], rows=[])

            def query_balance_data(self, **_: object) -> FakeQuery:
                return FakeQuery(fields=["totalShare"], rows=[])

            def query_cash_flow_data(self, **_: object) -> FakeQuery:
                return FakeQuery(fields=["CAToAsset"], rows=[])

        fake_bs = FakeBaoStock()
        with mock.patch("quant_data_platform.providers.time.sleep", return_value=None):
            frame = providers._baostock_financial_quarterly_frame_from_bs(
                fake_bs,
                DomainFetchRequest(
                    domain=DataDomain.FINANCIAL_QUARTERLY,
                    symbols=("600000.SH",),
                    start_date="2026-01-01",
                    end_date="2026-03-31",
                ),
            )

        self.assertEqual(fake_bs.profit_calls, 2)
        self.assertEqual(fake_bs.login_calls, 1)
        self.assertEqual(fake_bs.logout_calls, 1)
        self.assertEqual(frame["symbol"].tolist(), ["600000.SH"])
        self.assertEqual(frame["roeAvg"].tolist(), ["1.25"])

    def test_baostock_adjust_factor_query_relogs_in_after_not_logged_in(self) -> None:
        from quant_data_platform import providers

        class FakeQuery:
            def __init__(self, *, error_code: str = "0", error_msg: str = "", fields: list[str] | None = None, rows: list[list[str]] | None = None) -> None:
                self.error_code = error_code
                self.error_msg = error_msg
                self.fields = fields or []
                self.rows = rows or []
                self.index = 0

            def next(self) -> bool:
                self.index += 1
                return self.index <= len(self.rows)

            def get_row_data(self) -> list[str]:
                return self.rows[self.index - 1]

        class FakeBaoStock:
            def __init__(self) -> None:
                self.login_calls = 0
                self.logout_calls = 0
                self.adjust_calls = 0

            def login(self) -> object:
                self.login_calls += 1
                return types.SimpleNamespace(error_code="0", error_msg="")

            def logout(self) -> None:
                self.logout_calls += 1

            def query_adjust_factor(self, **_: object) -> FakeQuery:
                self.adjust_calls += 1
                if self.adjust_calls == 1:
                    return FakeQuery(error_code="10001001", error_msg="用户未登录")
                return FakeQuery(
                    fields=["code", "dividOperateDate", "foreAdjustFactor", "backAdjustFactor", "adjustFactor"],
                    rows=[["sh.600000", "2026-01-05", "1.01", "0.99", "1.0"]],
                )

        fake_bs = FakeBaoStock()
        with mock.patch("quant_data_platform.providers.time.sleep", return_value=None):
            frame = _baostock_adjust_factor_frame_from_bs(
                fake_bs,
                DomainFetchRequest(
                    domain=DataDomain.ADJUST_FACTOR,
                    symbols=("600000.SH",),
                    start_date="2026-01-01",
                    end_date="2026-01-31",
                ),
            )

        self.assertEqual(fake_bs.adjust_calls, 2)
        self.assertEqual(fake_bs.login_calls, 1)
        self.assertEqual(fake_bs.logout_calls, 1)
        self.assertEqual(frame["symbol"].tolist(), ["600000.SH"])
        self.assertEqual(frame["factor_provider"].tolist(), ["baostock"])

    def test_baostock_stock_basic_guard_times_out_and_terminates_child(self) -> None:
        from quant_data_platform import providers

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

        with mock.patch("quant_data_platform.providers.multiprocessing.get_context", return_value=FakeContext()):
            with self.assertRaisesRegex(TimeoutError, "baostock_stock_basic_timeout"):
                providers._fetch_baostock_stock_basic_frame_with_timeout(trade_date="2026-05-22", timeout_seconds=1)

        self.assertTrue(fake_process.started)
        self.assertTrue(fake_process.terminated)

    def test_baostock_stock_basic_guard_uses_blocking_queue_get(self) -> None:
        from quant_data_platform import providers

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

        with mock.patch("quant_data_platform.providers.multiprocessing.get_context", return_value=FakeContext()):
            frame = providers._fetch_baostock_stock_basic_frame_with_timeout(trade_date="2026-05-22", timeout_seconds=1)

        pd.testing.assert_frame_equal(frame, expected)

    def test_baostock_payload_guard_reads_queue_before_join_deadlock(self) -> None:
        from quant_data_platform import providers

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

        with mock.patch("quant_data_platform.providers.multiprocessing.get_context", return_value=FakeContext()):
            frame = providers._fetch_baostock_all_stock_frame_with_timeout(
                domain=DataDomain.UNIVERSE_SNAPSHOT,
                trade_date="2026-05-22",
                timeout_seconds=1,
            )

        pd.testing.assert_frame_equal(frame, expected)
        self.assertLess(events.index("get"), events.index("join"))
        self.assertNotIn("terminate", events)

    def test_baostock_all_stock_guard_times_out_and_terminates_child(self) -> None:
        from quant_data_platform import providers

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

        with mock.patch("quant_data_platform.providers.multiprocessing.get_context", return_value=FakeContext()):
            with self.assertRaisesRegex(TimeoutError, "baostock_all_stock_timeout"):
                providers._fetch_baostock_all_stock_frame_with_timeout(
                    domain=DataDomain.UNIVERSE_SNAPSHOT,
                    trade_date="2026-05-22",
                    timeout_seconds=1,
                )

        self.assertTrue(fake_process.started)
        self.assertTrue(fake_process.terminated)

    def test_baostock_trade_calendar_guard_times_out_and_terminates_child(self) -> None:
        from quant_data_platform import providers

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

        with mock.patch("quant_data_platform.providers.multiprocessing.get_context", return_value=FakeContext()):
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
