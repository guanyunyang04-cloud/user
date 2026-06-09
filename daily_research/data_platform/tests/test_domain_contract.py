import unittest

import pandas as pd

from daily_research.data_platform.contracts import (
    DataDomain,
    DomainFetchRequest,
    DOMAIN_STANDARD_COLUMNS,
    aggregate_intraday_1m_to_5m_frame,
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
            DataDomain.MARKET_INTRADAY_5M: pd.DataFrame(
                {
                    "code": ["sz.000001"],
                    "date": ["2026-01-05"],
                    "time": ["20260105093500000"],
                    "open": [10.0],
                    "high": [10.2],
                    "low": [9.9],
                    "close": [10.1],
                    "volume": [1000],
                    "amount": [10100],
                }
            ),
            DataDomain.MARKET_INTRADAY_1M: pd.DataFrame(
                {
                    "股票代码": ["000001.SZ"],
                    "日期": ["2026-01-05 09:30:00"],
                    "开盘": [10.0],
                    "最高": [10.1],
                    "最低": [9.9],
                    "收盘": [10.05],
                    "成交量(股)": [100],
                    "成交额(元)": [1005],
                    "换手率(%)": [0.01],
                    "流通股本(股)": [1000000],
                    "总股本(股)": [1200000],
                }
            ),
            DataDomain.INTRADAY_DAILY_FEATURES: pd.DataFrame(
                {
                    "symbol": ["000001.SZ"],
                    "trade_date": ["2026-01-05"],
                    "first_5m_ret": [0.01],
                    "last_30m_ret": [0.02],
                    "close_pressure_30m": [0.005],
                }
            ),
            DataDomain.INDEX_CONSTITUENTS: pd.DataFrame(
                {
                    "index_code": ["000300.SH"],
                    "code": ["000001.SZ"],
                    "date": ["2026-01-05"],
                    "index": ["CSI 300"],
                }
            ),
            DataDomain.FINANCIAL_QUARTERLY: pd.DataFrame(
                {
                    "code": ["sz.000001"],
                    "statDate": ["2025-12-31"],
                    "year": [2025],
                    "quarter": [4],
                    "roeAvg": [0.12],
                    "YOYPNI": [0.08],
                }
            ),
            DataDomain.PERFORMANCE_FORECAST: pd.DataFrame(
                {
                    "code": ["sz.000001"],
                    "profitForcastExpPubDate": ["2026-01-20"],
                    "profitForcastExpStatDate": ["2025-12-31"],
                    "profitForcastType": ["预增"],
                    "profitForcastChgPctDwn": [10.0],
                    "profitForcastChgPctUp": [30.0],
                }
            ),
            DataDomain.PERFORMANCE_EXPRESS: pd.DataFrame(
                {
                    "code": ["sz.000001"],
                    "performanceExpPubDate": ["2026-02-20"],
                    "performanceExpStatDate": ["2025-12-31"],
                    "performanceExpressEPSDiluted": [1.2],
                    "performanceExpressROEWa": [12.5],
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
            DataDomain.ADJUST_FACTOR: pd.DataFrame(
                {
                    "code": ["sz.000001"],
                    "dividOperateDate": ["2026-01-05"],
                    "foreAdjustFactor": [1.2],
                    "backAdjustFactor": [0.8],
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

    def test_financial_quarterly_without_publish_date_uses_conservative_availability(self) -> None:
        normalized = normalize_domain_frame(
            pd.DataFrame(
                {
                    "code": ["sz.000001"],
                    "statDate": ["2025-12-31"],
                    "year": [2025],
                    "quarter": [4],
                    "roeAvg": [0.12],
                }
            ),
            domain=DataDomain.FINANCIAL_QUARTERLY,
            source="baostock",
            require_columns=False,
        )

        self.assertEqual(normalized["symbol"].iloc[0], "000001.SZ")
        self.assertGreater(pd.Timestamp(normalized["trade_date"].iloc[0]), pd.Timestamp("2025-12-31"))
        self.assertEqual(normalized["lag_policy"].iloc[0], "conservative_report_date_plus_90bd_plus_1d_in_features")

    def test_intraday_1m_normalizer_splits_external_datetime_column(self) -> None:
        normalized = normalize_domain_frame(
            pd.DataFrame(
                {
                    "股票代码": ["000001.SZ"],
                    "日期": ["2026-01-05 09:30:00"],
                    "开盘": [10.0],
                    "最高": [10.1],
                    "最低": [9.9],
                    "收盘": [10.05],
                    "成交量(股)": [100],
                    "成交额(元)": [1005],
                    "换手率(%)": [0.01],
                    "流通股本(股)": [1000000],
                    "总股本(股)": [1200000],
                }
            ),
            domain=DataDomain.MARKET_INTRADAY_1M,
            source="external_1m",
            require_columns=False,
        )

        self.assertEqual(normalized["symbol"].iloc[0], "000001.SZ")
        self.assertEqual(normalized["trade_date"].iloc[0], "2026-01-05")
        self.assertEqual(normalized["bar_time"].iloc[0], "093000000")
        self.assertEqual(float(normalized["amount"].iloc[0]), 1005.0)

    def test_aggregate_intraday_1m_to_5m_frame_uses_ohlcv_semantics(self) -> None:
        frame = pd.DataFrame(
            {
                "symbol": ["000001.SZ"] * 5,
                "trade_date": ["2026-01-05"] * 5,
                "bar_time": ["09:30:00", "09:31:00", "09:32:00", "09:33:00", "09:34:00"],
                "open": [10.0, 10.1, 10.2, 10.3, 10.4],
                "high": [10.2, 10.3, 10.4, 10.5, 10.6],
                "low": [9.9, 10.0, 10.1, 10.2, 10.3],
                "close": [10.1, 10.2, 10.3, 10.4, 10.5],
                "volume": [100, 200, 300, 400, 500],
                "amount": [1010, 2040, 3090, 4160, 5250],
            }
        )

        bars = aggregate_intraday_1m_to_5m_frame(frame, source="external_1m")

        self.assertEqual(len(bars), 1)
        self.assertEqual(bars["bar_time"].iloc[0], "093000000")
        self.assertEqual(float(bars["open"].iloc[0]), 10.0)
        self.assertEqual(float(bars["high"].iloc[0]), 10.6)
        self.assertEqual(float(bars["low"].iloc[0]), 9.9)
        self.assertEqual(float(bars["close"].iloc[0]), 10.5)
        self.assertEqual(float(bars["volume"].iloc[0]), 1500.0)

    def test_adjust_factor_normalizer_keeps_provider_semantics(self) -> None:
        normalized = normalize_domain_frame(
            pd.DataFrame(
                {
                    "证券代码": ["000001.SZ"],
                    "除权除息日": ["2026-01-05"],
                    "复权因子": [1.234],
                    "factor_provider": ["sina"],
                    "factor_semantics": ["external_sina_event_factor"],
                }
            ),
            domain=DataDomain.ADJUST_FACTOR,
            source="external_adjust_factor",
            require_columns=False,
        )

        self.assertEqual(normalized["factor_provider"].iloc[0], "sina")
        self.assertEqual(normalized["factor_semantics"].iloc[0], "external_sina_event_factor")
        self.assertAlmostEqual(float(normalized["adjust_factor"].iloc[0]), 1.234)


if __name__ == "__main__":
    unittest.main()
