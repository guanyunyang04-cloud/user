from __future__ import annotations

import pandas as pd

from traditional_quant_research.dataset_builder import _fetch_daily_with_retry, _normalize_daily_payload, build_universe


class FakeSource:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def get_stock_list(self) -> list[str]:
        return ["600000.SH", "000001.SZ", "300750.SZ", "688001.SH", "000004.SZ"]

    def get_stock_info(self, code: str) -> dict[str, str]:
        info = {
            "600000.SH": {"Name": "浦发银行", "J_start": "19991110", "HSStockKind": "1", "IsZS": "0"},
            "000001.SZ": {"Name": "平安银行", "J_start": "19910403", "HSStockKind": "1", "IsZS": "0"},
            "000004.SZ": {"Name": "*ST国华", "J_start": "19901201", "HSStockKind": "1", "IsZS": "0"},
        }
        return info[code]

    def get_daily_bars(self, codes: list[str], *, start_time: str, end_time: str) -> dict[str, pd.DataFrame]:
        self.calls.append(codes)
        if len(codes) > 1:
            return {}
        code = codes[0]
        index = pd.to_datetime(["2026-01-02", "2026-01-05"])
        return {
            "Open": pd.DataFrame({code: [1.0, 1.1]}, index=index),
            "High": pd.DataFrame({code: [1.2, 1.3]}, index=index),
            "Low": pd.DataFrame({code: [0.9, 1.0]}, index=index),
            "Close": pd.DataFrame({code: [1.1, 1.2]}, index=index),
            "Volume": pd.DataFrame({code: [100.0, 0.0]}, index=index),
            "Amount": pd.DataFrame({code: [1000.0, 0.0]}, index=index),
            "ForwardFactor": pd.DataFrame({code: [1.0, 1.0]}, index=index),
        }

    def get_trading_dates(self, *, market: str = "SH", start_time: str, end_time: str) -> list[str]:
        return ["20260102", "20260105", "20260106"]


def test_build_universe_filters_mainboard_metadata() -> None:
    universe, selected, summary = build_universe(FakeSource())

    assert selected == ["600000.SH", "000001.SZ"]
    assert summary["raw_count"] == 5
    assert summary["mainboard_code_count"] == 3
    assert summary["selected_count"] == 2
    rejected = universe.loc[universe["code"] == "000004.SZ"].iloc[0]
    assert not bool(rejected["selected"])
    assert rejected["reject_reason"] == "st"


def test_normalize_daily_payload_marks_zero_volume_as_untradeable() -> None:
    source = FakeSource()
    payload = source.get_daily_bars(["600000.SH"], start_time="20260101", end_time="20260105")

    daily, status, missing = _normalize_daily_payload(payload, ["600000.SH"])

    assert missing == []
    assert len(daily) == 2
    assert status["is_tradeable"].tolist() == [True, False]
    assert status["is_suspended_like"].tolist() == [False, True]


def test_normalize_daily_payload_reindexes_full_trading_calendar() -> None:
    source = FakeSource()
    payload = source.get_daily_bars(["600000.SH"], start_time="20260101", end_time="20260106")
    trading_dates = pd.to_datetime(source.get_trading_dates(start_time="20260101", end_time="20260106"))

    daily, status, missing = _normalize_daily_payload(payload, ["600000.SH"], pd.DatetimeIndex(trading_dates))

    assert missing == []
    assert len(daily) == 2
    assert status["date"].dt.strftime("%Y%m%d").tolist() == ["20260102", "20260105", "20260106"]
    assert status["has_bar"].tolist() == [True, True, False]
    assert status["is_suspended_like"].tolist() == [False, True, True]
    assert status["is_tradeable"].tolist() == [True, False, False]


def test_fetch_daily_with_retry_splits_empty_batch_and_records_failures() -> None:
    source = FakeSource()

    daily, status, failures = _fetch_daily_with_retry(source, ["600000.SH", "000001.SZ"], start_date="20260101", end_date="20260105")

    assert source.calls == [["600000.SH", "000001.SZ"], ["600000.SH"], ["000001.SZ"]]
    assert failures == []
    assert set(daily["code"]) == {"600000.SH", "000001.SZ"}
    assert len(status) == 4
