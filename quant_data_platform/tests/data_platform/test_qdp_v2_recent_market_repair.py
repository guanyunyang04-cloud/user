from __future__ import annotations

import pandas as pd

from quant_data_platform.qdp_v2.recent_market_repair import (
    EXPECTED_BAR_TIMES,
    _stable_bucket,
    _unresolved_pairs_from_state,
    _validate_daily_frame,
    _validate_intraday_frame,
)


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


def test_recent_repair_bucket_is_stable_and_bounded() -> None:
    first = _stable_bucket("600000.SH")
    assert first == _stable_bucket("600000.SH")
    assert 0 <= first < 16


def test_unresolved_pairs_are_deduplicated_from_bucket_state() -> None:
    state = {
        "buckets": {
            "00": {"unresolved": [{"symbol": "600000.sh", "trade_date": "2026-07-13"}]},
            "01": {"unresolved": [{"symbol": "600000.SH", "trade_date": "2026-07-13"}]},
        }
    }
    assert _unresolved_pairs_from_state(state) == (("600000.SH", "2026-07-13"),)
