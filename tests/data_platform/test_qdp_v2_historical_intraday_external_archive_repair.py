from __future__ import annotations

import pandas as pd

from quantlab.data.qdp_v2.historical_intraday_external_archive_repair import (
    _NUMERIC_COLUMNS,
    _normalize_day,
    _validate_daily,
)
from quantlab.data.qdp_v2.manifest import EXPECTED_BAR_TIMES


def _day(*, include_auction: bool) -> pd.DataFrame:
    clocks = [f"{item[:2]}:{item[2:4]}" for item in EXPECTED_BAR_TIMES]
    if include_auction:
        clocks.insert(0, "09:30")
    rows = len(clocks)
    return pd.DataFrame(
        {
            "trade_date": ["2020-01-02"] * rows,
            "bar_time": clocks,
            "open": [10.0] * rows,
            "high": [10.2] * rows,
            "low": [9.8] * rows,
            "close": [10.0] * rows,
            "volume": [100.0] * rows,
            "amount": [1_000.0] * rows,
        }
    )


def test_exact_49_bar_day_merges_auction_into_0935() -> None:
    source = _day(include_auction=True)
    source.loc[0, ["open", "high", "low", "volume", "amount"]] = [
        9.9,
        10.3,
        9.7,
        50.0,
        500.0,
    ]

    normalized, reason = _normalize_day(source)

    assert reason == ""
    assert len(normalized) == 48
    assert normalized.iloc[0]["bar_time"] == EXPECTED_BAR_TIMES[0]
    assert normalized.iloc[0]["open"] == 9.9
    assert normalized.iloc[0]["high"] == 10.3
    assert normalized.iloc[0]["low"] == 9.7
    assert normalized.iloc[0]["volume"] == 150.0
    assert normalized.iloc[0]["amount"] == 1_500.0


def test_noncanonical_bar_set_is_rejected_without_interpolation() -> None:
    source = _day(include_auction=False).drop(index=5)

    normalized, reason = _normalize_day(source)

    assert normalized.empty
    assert reason == "invalid_bar_time_set"


def test_daily_gate_requires_prices_and_one_flow_measure() -> None:
    normalized, reason = _normalize_day(_day(include_auction=False))
    assert reason == ""
    reference = {
        "open": 10.0,
        "high": 10.2,
        "low": 9.8,
        "close": 10.0,
        "volume": 4_800.0,
        "amount": 999_999.0,
    }

    accepted, reject_reason, *_ = _validate_daily(normalized, reference)

    assert accepted
    assert reject_reason == ""
    assert set(_NUMERIC_COLUMNS).issubset(normalized.columns)
