from __future__ import annotations

import pandas as pd
import pytest

from quantlab.research.minute_ma import (
    MinuteMAConfig,
    MinuteMAError,
    build_ma_event_table,
    build_minute_ma_states,
    prepare_minute_bars,
    session_minute_ordinal,
    sixty_minute_bucket,
)


def _session_times() -> list[str]:
    values: list[str] = []
    for ordinal in range(240):
        if ordinal < 120:
            minute = ordinal
            hour = 9 + (31 + minute) // 60
            minute_value = (31 + minute) % 60
        else:
            minute = ordinal - 120
            hour = 13 + (1 + minute) // 60
            minute_value = (1 + minute) % 60
        values.append(f"{hour:02d}{minute_value:02d}00000")
    return values


def _bars(
    *,
    hour_closes: tuple[float, ...] = (10.0, 10.0, 10.0, 10.0),
    target_pattern: dict[int, tuple[float, float, float]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pattern = target_pattern or {}
    for ordinal, bar_time in enumerate(_session_times()):
        bucket = ordinal // 60 + 1
        close = hour_closes[bucket - 1]
        if bucket == 4 and ordinal - 180 in pattern:
            close, low, high = pattern[ordinal - 180]
        else:
            low, high = close - 0.001, close + 0.001
        rows.append(
            {
                "symbol": "A",
                "trade_date": "2022-01-03",
                "bar_time": bar_time,
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1.0,
                "amount": 10.0,
            }
        )
    return pd.DataFrame(rows)


def test_session_buckets_exclude_lunch_and_use_expected_boundaries() -> None:
    assert session_minute_ordinal("093100000") == 0
    assert session_minute_ordinal("103000000") == 59
    assert session_minute_ordinal("103100000") == 60
    assert session_minute_ordinal("113000000") == 119
    assert session_minute_ordinal("113100000") is None
    assert session_minute_ordinal("130000000") is None
    assert session_minute_ordinal("130100000") == 120
    assert session_minute_ordinal("150000000") == 239
    assert sixty_minute_bucket("130100000") == 3
    assert sixty_minute_bucket("140100000") == 4


def test_incomplete_hour_is_removed_without_merging_across_lunch() -> None:
    frame = _bars().iloc[:-1].copy()
    prepared = prepare_minute_bars(frame)
    assert len(prepared) == 180
    assert prepared["sixty_minute_bucket"].tolist() == [1] * 60 + [2] * 60 + [3] * 60
    assert prepared.attrs["incomplete_hour_count"] == 1


def test_live_ma_uses_current_price_but_causal_intersection_is_fixed() -> None:
    bars = _bars(
        hour_closes=(10.0, 10.0, 10.2, 10.2),
        target_pattern={
            0: (10.2, 10.199, 10.201),
            59: (10.3, 10.299, 10.301),
        },
    )
    states = build_minute_ma_states(
        bars,
        config=MinuteMAConfig(periods=(3,)),
        target_dates=["2022-01-03"],
    )
    target = states.loc[(states["sixty_minute_bucket"] == 4) & (states["ma_period"] == 3)].reset_index(drop=True)
    assert len(target) == 60
    assert target["causal_intersection"].iloc[0] == pytest.approx(10.1)
    assert target["causal_intersection"].nunique() == 1
    assert target["live_ma"].iloc[0] == pytest.approx((20.2 + 10.2) / 3.0)
    assert target["live_ma"].iloc[0] != target["live_ma"].iloc[-1]
    assert "final_ma" not in states.columns
    assert all(name.startswith("diagnostic_") for name in states.attrs["diagnostic_columns"])
    assert "live_ma" in states.attrs["causal_columns"]
    assert all(not name.startswith("diagnostic_") for name in states.attrs["causal_columns"])


def test_posthoc_catchup_is_not_recorded_as_a_true_touch() -> None:
    bars = _bars(
        hour_closes=(10.0, 10.0, 10.2, 10.2),
        target_pattern={
            0: (10.11, 10.105, 10.115),
            59: (10.2, 10.199, 10.201),
        },
    )
    events = build_ma_event_table(
        bars,
        config=MinuteMAConfig(periods=(3,), posthoc_catchup_bps=30.0),
        target_dates=["2022-01-03"],
    )
    target = events.loc[events["sixty_minute_bucket"] == 4].iloc[0]
    assert target["event_kind"] == "posthoc_catchup"
    assert bool(target["true_touch"]) is False
    assert bool(target["posthoc_catchup"]) is True
    assert pd.isna(target["causal_confirmation_time"])


def test_break_reclaim_and_failed_break_are_distinct() -> None:
    reclaim = _bars(
        hour_closes=(10.0, 10.0, 10.2, 10.2),
        target_pattern={
            10: (10.05, 9.98, 10.12),
            30: (10.2, 10.19, 10.21),
        },
    )
    events = build_ma_event_table(
        reclaim,
        config=MinuteMAConfig(periods=(3,)),
        target_dates=["2022-01-03"],
    )
    target = events.loc[events["sixty_minute_bucket"] == 4].iloc[0]
    assert target["event_kind"] == "break_reclaim"
    assert target["first_below_close_time"] is not None
    assert target["first_reclaim_time"] is not None

    failed = _bars(
        hour_closes=(10.0, 10.0, 10.2, 9.9),
        target_pattern={
            10: (10.05, 9.98, 10.12),
            59: (9.9, 9.89, 9.91),
        },
    )
    failed_events = build_ma_event_table(
        failed,
        config=MinuteMAConfig(periods=(3,)),
        target_dates=["2022-01-03"],
    )
    failed_target = failed_events.loc[failed_events["sixty_minute_bucket"] == 4].iloc[0]
    assert failed_target["event_kind"] == "failed_break"
    assert bool(failed_target["final_above_intersection"]) is False


def test_future_minute_mutation_does_not_change_prior_causal_state() -> None:
    original = _bars(
        hour_closes=(10.0, 10.0, 10.2, 10.2),
        target_pattern={
            10: (10.05, 9.98, 10.12),
            30: (10.2, 10.19, 10.21),
        },
    )
    mutated = original.copy()
    row = mutated.index[mutated["bar_time"].eq("150000000")][0]
    mutated.loc[row, ["open", "high", "low", "close"]] = [20.0, 20.1, 19.9, 20.0]
    columns = [
        "bar_time",
        "causal_intersection",
        "live_ma",
        "close_to_intersection_bps",
        "close_to_live_ma_bps",
        "touched_now",
        "minutes_below_so_far",
        "partial_hour_high_adjusted",
        "partial_hour_low_adjusted",
        "rebound_from_partial_low_bps",
    ]
    before = build_minute_ma_states(
        original,
        config=MinuteMAConfig(periods=(3,)),
        target_dates=["2022-01-03"],
    )
    after = build_minute_ma_states(
        mutated,
        config=MinuteMAConfig(periods=(3,)),
        target_dates=["2022-01-03"],
    )
    before = before.loc[before["bar_time"] < "150000000", columns].reset_index(drop=True)
    after = after.loc[after["bar_time"] < "150000000", columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)


def test_invalid_price_and_period_contracts_fail_loudly() -> None:
    malformed = _bars()
    malformed.loc[0, "low"] = 0.0
    with pytest.raises(MinuteMAError, match="invalid_ohlc"):
        prepare_minute_bars(malformed)
    with pytest.raises(MinuteMAError, match="periods"):
        MinuteMAConfig(periods=(20, 10)).validate()
    with pytest.raises(MinuteMAError, match="period"):
        MinuteMAConfig(periods=(10.5,)).validate()
    with pytest.raises(MinuteMAError, match="prior_touch_window"):
        MinuteMAConfig(prior_touch_window_hours=2.5).validate()
