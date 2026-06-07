from __future__ import annotations

import pandas as pd
import pytest

from traditional_quant_research.experiments.two_day_kama_atr_breakout_analysis import (
    build_two_day_breakout_events_from_feature_panel,
)


def _feature_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", periods=8)
    rows = []
    closes = [10.0, 11.0, 12.1, 13.0, 12.0, 14.0, 13.5, 15.0]
    for idx, (date, close) in enumerate(zip(dates, closes)):
        is_day1 = idx == 1
        is_day2 = idx == 2
        rows.append(
            {
                "date": date,
                "code": "000001.SZ",
                "date_index": idx,
                "name_on_date": "A",
                "industry": "test",
                "open": close * 0.98,
                "high": close,
                "low": close * 0.95,
                "close": close,
                "pctChg": 10.0 if is_day1 or is_day2 else 0.5,
                "amount": 100_000_000.0,
                "turn": 3.0,
                "kama": 10.5 if idx == 0 else 10.8,
                "atr_upper": 12.2 if idx == 1 else (12.0 if idx == 2 else close * 1.2),
                "limit_up_like": bool(is_day1 or is_day2),
                "one_word_limit_like": False,
                "near_one_word_limit_like": False,
                "limit_up_run_ending_today": 1 if is_day1 else (2 if is_day2 else 0),
                "board_stage": "first_board" if is_day1 else ("second_board" if is_day2 else "none"),
            }
        )
    return pd.DataFrame(rows)


def test_two_day_breakout_uses_previous_kama_and_day2_future_window() -> None:
    events, path = build_two_day_breakout_events_from_feature_panel(_feature_panel(), horizons=(1, 3))

    assert len(events) == 1
    event = events.iloc[0]
    assert event["day1_date"] == pd.Timestamp("2026-01-05")
    assert event["day2_date"] == pd.Timestamp("2026-01-06")
    assert bool(event["day2_atr_break_from_below"]) is True
    assert event["close_ret_1d_from_day2_close_pct"] == pytest.approx(((13.0 / 12.1) - 1.0) * 100.0)
    assert event["max_high_ret_3d_from_day2_close_pct"] == pytest.approx(((14.0 / 12.1) - 1.0) * 100.0)
    assert event["min_low_ret_3d_from_day2_close_pct"] == pytest.approx(((12.0 * 0.95 / 12.1) - 1.0) * 100.0)
    assert path["future_offset"].tolist()[:3] == [1, 2, 3]
