import unittest
import sys
import types
from unittest import mock

import pandas as pd

from daily_research.data_platform.contracts import (
    STANDARD_MARKET_COLUMNS,
    FetchRequest,
    ProviderResult,
    normalize_market_frame,
    validate_provider_name,
)
from daily_research.data_platform.manager import InMemoryMarketProvider, ProviderManager
from daily_research.data_platform.providers import (
    EastmoneyEfinanceProvider,
    TushareHttpOptionalProvider,
    build_default_providers,
    provider_capability_matrix,
)
from daily_research.data_platform.providers import _baostock_stock_basic_frame


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


if __name__ == "__main__":
    unittest.main()
