from __future__ import annotations

import pandas as pd

from traditional_quant_research.dataset_builder_v2 import (
    _baostock_to_std_code,
    _std_to_baostock_code,
    build_daily_universe,
)


def test_baostock_code_conversion_round_trip() -> None:
    assert _baostock_to_std_code("sh.600000") == "600000.SH"
    assert _baostock_to_std_code("sz.000001") == "000001.SZ"
    assert _std_to_baostock_code("600000.SH") == "sh.600000"
    assert _std_to_baostock_code("000001.SZ") == "sz.000001"


def test_build_daily_universe_marks_pit_statuses() -> None:
    stock_lists = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "name_on_date": "浦发银行", "query_all_trade_status": "1"},
            {"date": pd.Timestamp("2026-01-02"), "code": "000001.SZ", "name_on_date": "平安银行", "query_all_trade_status": "1"},
            {"date": pd.Timestamp("2026-01-02"), "code": "000004.SZ", "name_on_date": "*ST国华", "query_all_trade_status": "1"},
            {"date": pd.Timestamp("2026-01-02"), "code": "600001.SH", "name_on_date": "退市样例", "query_all_trade_status": "1"},
        ]
    )
    security_master = pd.DataFrame(
        [
            {"code": "600000.SH", "name": "浦发银行", "ipo_date": "1999-11-10", "out_date": "", "security_type": "1", "status": "1"},
            {"code": "000001.SZ", "name": "平安银行", "ipo_date": "1991-04-03", "out_date": "", "security_type": "1", "status": "1"},
            {"code": "000004.SZ", "name": "*ST国华", "ipo_date": "1990-12-01", "out_date": "", "security_type": "1", "status": "1"},
            {"code": "600001.SH", "name": "退市样例", "ipo_date": "1990-12-01", "out_date": "2025-12-31", "security_type": "1", "status": "0"},
        ]
    )
    daily_bars = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 100.0, "amount": 1000.0, "tradestatus": "1", "isST": "0"},
            {"date": pd.Timestamp("2026-01-02"), "code": "000001.SZ", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 0.0, "amount": 0.0, "tradestatus": "0", "isST": "0"},
            {"date": pd.Timestamp("2026-01-02"), "code": "000004.SZ", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 100.0, "amount": 1000.0, "tradestatus": "1", "isST": "1"},
            {"date": pd.Timestamp("2026-01-02"), "code": "600001.SH", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 100.0, "amount": 1000.0, "tradestatus": "1", "isST": "0"},
        ]
    )

    universe, status = build_daily_universe(stock_lists, security_master, daily_bars)

    by_code = universe.set_index("code")
    assert bool(by_code.loc["600000.SH", "is_tradeable"])
    assert by_code.loc["000001.SZ", "reject_reason"] == "suspended_on_date"
    assert by_code.loc["000004.SZ", "reject_reason"] == "st_on_date"
    assert by_code.loc["600001.SH", "reject_reason"] == "not_listed_on_date"
    assert status.set_index("code").loc["000001.SZ", "is_suspended_like"]
