import unittest

import pandas as pd

from daily_research.baseline import data_provider


class _FakeTq:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def get_market_data(self, *, field_list, stock_list, start_time, end_time, count, dividend_type, period):
        del field_list, start_time, end_time, count, dividend_type, period
        stocks = tuple(stock_list)
        self.calls.append(stocks)
        if stocks == ("AAA", "BBB"):
            return {}
        index = pd.to_datetime(["2026-01-05"])
        return {
            field: pd.DataFrame({stock: [1.0] for stock in stocks}, index=index)
            for field in data_provider.MARKET_DATA_FIELDS
        }


class DataProviderTqFetchTest(unittest.TestCase):
    def test_empty_batch_is_split_and_recombined(self) -> None:
        tq = _FakeTq()

        payload = data_provider._fetch_tq_market_data_batch(
            tq,
            stock_list=["AAA", "BBB"],
            start_date="20260105",
            end_date="20260105",
            count=0,
            dividend_type="front",
            batch_start=128,
        )

        self.assertIn("Close", payload)
        self.assertEqual(list(payload["Close"].columns), ["AAA", "BBB"])
        self.assertIn(("AAA", "BBB"), tq.calls)
        self.assertIn(("AAA",), tq.calls)
        self.assertIn(("BBB",), tq.calls)

    def test_singleton_empty_batch_reports_stock_and_date(self) -> None:
        class EmptyTq:
            def get_market_data(self, **kwargs):
                del kwargs
                return {}

        with self.assertRaisesRegex(RuntimeError, "stock=ZZZ.*20260105.*batch_start=7"):
            data_provider._fetch_tq_market_data_batch(
                EmptyTq(),
                stock_list=["ZZZ"],
                start_date="20260105",
                end_date="20260105",
                count=0,
                dividend_type="front",
                batch_start=7,
            )

    def test_empty_singleton_reinitializes_before_retry(self) -> None:
        class RecoveredTq:
            def __init__(self) -> None:
                self.calls = 0
                self.reinitialized = 0

            def get_market_data(self, *, field_list, stock_list, start_time, end_time, count, dividend_type, period):
                del start_time, end_time, count, dividend_type, period
                self.calls += 1
                if self.reinitialized <= 0:
                    return {}
                index = pd.to_datetime(["2026-01-05"])
                return {
                    field: pd.DataFrame({stock: [1.0] for stock in stock_list}, index=index)
                    for field in field_list
                }

        tq = RecoveredTq()

        payload = data_provider._fetch_tq_market_data_batch(
            tq,
            stock_list=["ZZZ"],
            start_date="20260105",
            end_date="20260105",
            count=0,
            dividend_type="front",
            batch_start=7,
            reinitialize_callback=lambda client: setattr(client, "reinitialized", client.reinitialized + 1),
        )

        self.assertIn("Close", payload)
        self.assertEqual(tq.calls, 2)
        self.assertEqual(tq.reinitialized, 1)


if __name__ == "__main__":
    unittest.main()
