import unittest
import sys
import time
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
from daily_research.data_platform.providers import EastmoneyEfinanceProvider, build_default_providers


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
            raise TimeoutError("provider timed out")

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

    def test_provider_manager_timeout_returns_without_waiting_for_worker(self) -> None:
        request = FetchRequest(
            symbols=("000001.SZ",),
            start_date="2026-01-05",
            end_date="2026-01-05",
        )

        def slow_provider(_: FetchRequest) -> ProviderResult:
            time.sleep(2.0)
            return ProviderResult(provider="eastmoney_efinance", data=pd.DataFrame())

        manager = ProviderManager(
            providers=[InMemoryMarketProvider("eastmoney_efinance", slow_provider)],
            timeout_seconds=0.05,
        )

        started = time.perf_counter()
        result = manager.fetch_market_bars(request)
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 1.0)
        self.assertTrue(result.data.empty)
        self.assertIn("provider_exception", {item["code"] for item in result.error_report})

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

        self.assertNotIn("sina_tencent_realtime", default_names)
        self.assertIn("sina_tencent_realtime", realtime_names)


if __name__ == "__main__":
    unittest.main()
