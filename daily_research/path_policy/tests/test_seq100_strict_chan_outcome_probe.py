from __future__ import annotations

import pandas as pd

from daily_research.path_policy import seq100_strict_chan_outcome_probe as probe


def test_cost_multiplier_includes_minimum_commission_and_stress_slippage() -> None:
    costs = {
        "lot_size": 100,
        "commission_bps": 3.0,
        "minimum_commission_cny": 5.0,
        "transfer_fee_bps": 0.1,
        "slippage_bps": 7.0,
        "stress_slippage_multiplier": 2.0,
        "stamp_tax_bps_before_2023_08_28": 10.0,
        "stamp_tax_bps_from_2023_08_28": 5.0,
    }
    base = probe._cost_multipliers(10.0, "2022-01-01", stress=False, costs=costs)
    stress = probe._cost_multipliers(10.0, "2022-01-01", stress=True, costs=costs)
    assert base[0] > 1.004
    assert stress[0] > base[0]
    assert base[1] / base[0] < 1.0


def test_event_candidates_deduplicate_same_entry_date() -> None:
    events = pd.DataFrame(
        {
            "case_id": ["case", "case"],
            "profile": ["primary", "primary"],
            "event_type": ["trade_point", "trade_point"],
            "symbol": ["000001.SZ", "000001.SZ"],
            "id": ["later", "earlier"],
            "confirmed_time": [
                "2020-01-02T15:00:00",
                "2020-01-02T10:00:00",
            ],
            "payload_json": [
                '{"side":"buy","point_type":3}',
                '{"side":"buy","point_type":3}',
            ],
        }
    )
    result = probe._event_candidates(events)
    assert len(result) == 1
    assert result.iloc[0]["id"] == "earlier"


def test_daily_frame_uses_first_open_last_close() -> None:
    bars = pd.DataFrame(
        {
            "trade_date": ["2020-01-02", "2020-01-02"],
            "timestamp": pd.to_datetime(["2020-01-02 09:35", "2020-01-02 15:00"]),
            "open": [10.0, 10.5],
            "high": [10.5, 11.0],
            "low": [9.9, 10.2],
            "close": [10.4, 10.8],
        }
    )
    result = probe._daily_frame(bars)
    assert result.iloc[0]["open"] == 10.0
    assert result.iloc[0]["close"] == 10.8


def test_daily_frame_preserves_raw_prices_for_minimum_commission() -> None:
    bars = pd.DataFrame(
        {
            "trade_date": ["2020-01-02", "2020-01-02"],
            "timestamp": pd.to_datetime(["2020-01-02 09:35", "2020-01-02 15:00"]),
            "open": [5.0, 5.2],
            "high": [5.3, 5.4],
            "low": [4.9, 5.1],
            "close": [5.2, 5.3],
            "raw_open": [10.0, 10.4],
            "raw_high": [10.6, 10.8],
            "raw_low": [9.8, 10.2],
            "raw_close": [10.4, 10.6],
        }
    )
    result = probe._daily_frame(bars)
    assert result.iloc[0]["raw_open"] == 10.0
    assert result.iloc[0]["raw_close"] == 10.6
