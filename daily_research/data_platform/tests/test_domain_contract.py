import unittest

import pandas as pd

from daily_research.data_platform.contracts import (
    DataDomain,
    DomainFetchRequest,
    DOMAIN_STANDARD_COLUMNS,
    normalize_domain_frame,
)
from daily_research.data_platform.manager import InMemoryDomainProvider, ProviderManager


class DataPlatformDomainContractTest(unittest.TestCase):
    def test_domain_normalizers_emit_standard_columns(self) -> None:
        samples = {
            DataDomain.TRADING_CALENDAR: pd.DataFrame({"date": ["2026-01-05"], "is_open": [1], "exchange": ["SSE"]}),
            DataDomain.UNIVERSE_SNAPSHOT: pd.DataFrame(
                {
                    "code": ["000001"],
                    "date": ["2026-01-05"],
                    "name": ["平安银行"],
                    "exchange": ["SZ"],
                    "board": ["main"],
                    "list_status": ["L"],
                    "list_date": ["1991-04-03"],
                    "delist_date": [""],
                }
            ),
            DataDomain.SECURITY_STATUS: pd.DataFrame(
                {
                    "symbol": ["000001.SZ"],
                    "trade_date": ["2026-01-05"],
                    "is_st": [False],
                    "is_suspended": [False],
                    "is_delisted": [False],
                    "status_reason": [""],
                }
            ),
            DataDomain.LIMIT_STATUS: pd.DataFrame(
                {
                    "symbol": ["000001.SZ"],
                    "trade_date": ["2026-01-05"],
                    "up_limit": [11.0],
                    "down_limit": [9.0],
                    "is_limit_up": [False],
                    "is_limit_down": [False],
                }
            ),
            DataDomain.INDUSTRY_CONCEPT: pd.DataFrame(
                {
                    "symbol": ["000001.SZ"],
                    "trade_date": ["2026-01-05"],
                    "industry": ["银行"],
                    "concept_tags": [["金融", "低估值"]],
                }
            ),
            DataDomain.VALUATION: pd.DataFrame(
                {
                    "symbol": ["000001.SZ"],
                    "trade_date": ["2026-01-05"],
                    "total_mv": [1000.0],
                    "circ_mv": [900.0],
                    "pe": [6.0],
                    "pb": [0.8],
                    "turnover_rate": [1.2],
                }
            ),
            DataDomain.MONEY_FLOW_HOTSPOT: pd.DataFrame(
                {
                    "symbol": ["000001.SZ"],
                    "trade_date": ["2026-01-05"],
                    "main_net_inflow": [100.0],
                    "sector_rank": [3],
                    "hotspot_tags": ["金融,低估值"],
                }
            ),
        }

        for domain, frame in samples.items():
            with self.subTest(domain=domain):
                normalized = normalize_domain_frame(frame, domain=domain, source="akshare_eastmoney", as_of_date="2026-01-05")
                self.assertEqual(list(normalized.columns), DOMAIN_STANDARD_COLUMNS[domain])
                self.assertEqual(normalized["source"].iloc[0], "akshare_eastmoney")

    def test_domain_fetch_manager_records_unsupported_domain_without_exception(self) -> None:
        provider = InMemoryDomainProvider(
            "eastmoney_efinance",
            payloads={DataDomain.MARKET_DAILY: pd.DataFrame()},
        )
        manager = ProviderManager([provider])
        result = manager.fetch_domain(
            DomainFetchRequest(
                domain=DataDomain.VALUATION,
                symbols=("000001.SZ",),
                start_date="2026-01-05",
                end_date="2026-01-05",
            )
        )

        self.assertTrue(result.data.empty)
        self.assertIn("unsupported_domain", {item["code"] for item in result.error_report})


if __name__ == "__main__":
    unittest.main()
