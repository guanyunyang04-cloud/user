from __future__ import annotations

import inspect

import pandas as pd

from quantlab.data.qdp_v2.manifest import EXPECTED_BAR_TIMES
from quantlab.data.qdp_v2.recent_market_repair.baostock import _baostock_tasks
from quantlab.data.qdp_v2.recent_market_repair.config import DEFAULT_WORKERS
from quantlab.data.qdp_v2.recent_market_repair.entrypoints import run_baostock_intraday_repair
from quantlab.data.qdp_v2.recent_market_repair.inventory import _calendar_year_windows
from quantlab.data.qdp_v2.recent_market_repair.session import _normalize_baostock_raw_5m
from quantlab.data.qdp_v2.recent_market_repair.state import _stable_bucket
from quantlab.data.qdp_v2.recent_market_repair.validation import _validate_daily_frame, _validate_intraday_frame


def test_baostock_repair_defaults_to_four_independent_connections() -> None:
    assert DEFAULT_WORKERS == 4
    assert inspect.signature(run_baostock_intraday_repair).parameters[
        "workers"
    ].default == 4


def test_intraday_inventory_splits_multi_year_range_by_calendar_year() -> None:
    assert _calendar_year_windows("2020-06-01", "2022-02-03") == (
        ("2020-06-01", "2020-12-31"),
        ("2021-01-01", "2021-12-31"),
        ("2022-01-01", "2022-02-03"),
    )


def test_baostock_raw_time_is_normalized_before_common_validation() -> None:
    raw = pd.DataFrame(
        [
            {
                "date": "2020-01-02",
                "time": "20200102093500000",
                "code": "sh.600000",
                "open": "10",
                "high": "10",
                "low": "10",
                "close": "10",
                "volume": "100",
                "amount": "1000",
                "adjustflag": "3",
            }
        ]
    )
    frame = _normalize_baostock_raw_5m(raw, symbol="600000.SH")
    assert frame.loc[0, "symbol"] == "600000.SH"
    assert frame.loc[0, "trade_date"] == "2020-01-02"
    assert frame.loc[0, "bar_time"] == "09:35"


def test_recent_daily_validation_accepts_only_legal_requested_row() -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "600000.SH",
                "trade_date": "2026-07-13",
                "open": 10.0,
                "high": 10.5,
                "low": 9.9,
                "close": 10.2,
                "volume": 100.0,
                "amount": 1000.0,
                "source": "mootdx_online",
                "adjusted_flag": "none",
            }
        ]
    )
    valid, unresolved = _validate_daily_frame(
        frame,
        {"600000.SH": ("2026-07-13",), "000001.SZ": ("2026-07-13",)},
    )
    assert len(valid) == 1
    assert ("000001.SZ", "2026-07-13") in unresolved


def test_recent_intraday_validation_requires_exact_48_times() -> None:
    rows = []
    for bar_time in EXPECTED_BAR_TIMES:
        rows.append(
            {
                "symbol": "600000.SH",
                "trade_date": "2026-07-13",
                "bar_time": bar_time,
                "open": 10.0,
                "high": 10.5,
                "low": 9.9,
                "close": 10.2,
                "volume": 100.0,
                "amount": 1000.0,
                "source": "mootdx_online",
                "adjusted_flag": "none",
            }
        )
    valid, unresolved = _validate_intraday_frame(
        pd.DataFrame(rows), {"600000.SH": ("2026-07-13",)}
    )
    assert len(valid) == 48
    assert not unresolved

    invalid, unresolved = _validate_intraday_frame(
        pd.DataFrame(rows[:-1]), {"600000.SH": ("2026-07-13",)}
    )
    assert invalid.empty
    assert unresolved[("600000.SH", "2026-07-13")].endswith("48_bar_day")


def test_intraday_validation_records_selected_provider() -> None:
    rows = [
        {
            "symbol": "600837.SH",
            "trade_date": "2020-01-02",
            "bar_time": bar_time,
            "open": 10.0,
            "high": 10.5,
            "low": 9.9,
            "close": 10.2,
            "volume": 100.0,
            "amount": 1000.0,
        }
        for bar_time in EXPECTED_BAR_TIMES
    ]
    valid, unresolved = _validate_intraday_frame(
        pd.DataFrame(rows),
        {"600837.SH": ("2020-01-02",)},
        source_name="baostock",
    )
    assert not unresolved
    assert set(valid["source"]) == {"baostock"}


def test_baostock_tasks_split_long_ranges_by_symbol_year() -> None:
    tasks = _baostock_tasks(
        {
            "600837.SH": ("2020-01-02", "2020-12-31", "2021-01-04"),
            "000005.SZ": ("2024-03-05",),
        }
    )
    assert [(item.symbol, item.dates) for item in tasks] == [
        ("000005.SZ", ("2024-03-05",)),
        ("600837.SH", ("2020-01-02", "2020-12-31")),
        ("600837.SH", ("2021-01-04",)),
    ]


def test_recent_repair_bucket_is_stable_and_bounded() -> None:
    first = _stable_bucket("600000.SH")
    assert first == _stable_bucket("600000.SH")
    assert 0 <= first < 16
