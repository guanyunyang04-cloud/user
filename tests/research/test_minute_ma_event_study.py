from __future__ import annotations

import pandas as pd
import pytest

from quantlab.research.minute_ma_event_study import (
    EventStudyConfig,
    EventStudyError,
    build_representative_regime_frame,
    compare_to_control,
    compare_to_reference_control,
    compute_event_outcomes,
    summarize_control_comparison,
    summarize_event_study,
)


def _bars(*, gap: bool = False) -> pd.DataFrame:
    rows = []
    first_times = []
    for ordinal in range(240):
        if ordinal < 120:
            minute = ordinal
            hour = 9 + (31 + minute) // 60
            minute_value = (31 + minute) % 60
        else:
            minute = ordinal - 120
            hour = 13 + (1 + minute) // 60
            minute_value = (1 + minute) % 60
        first_times.append(f"{hour:02d}{minute_value:02d}00000")
    if gap:
        first_times = [value for value in first_times if value != "093300000"]
    for trade_date, opens in (
        ("2022-01-03", [10.0, 10.2, 10.3, 10.4, 10.5]),
        ("2022-01-04", [11.0, 11.1, 11.2, 11.3, 11.4]),
    ):
        for index, bar_time in enumerate(first_times):
            price = opens[min(index, len(opens) - 1)] + max(0, index - 4) * 0.001
            rows.append(
                {
                    "symbol": "A",
                    "trade_date": trade_date,
                    "bar_time": bar_time,
                    "open": price,
                    "high": price + 0.1,
                    "low": price - 0.1,
                    "close": price,
                    "volume": 1000.0,
                    "amount": 10_000.0,
                }
            )
    return pd.DataFrame(rows)


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_id": "s|A|2022-01-03|1|3|093100000",
                "strategy_id": "s1_touch_reclaim",
                "strategy_family": "S1",
                "symbol": "A",
                "signal_date": "2022-01-03",
                "signal_time": "093100000",
                "sixty_minute_bucket": 1,
                "ma_period": 3,
                "event_trigger": "test",
                "causal_only": True,
                "diagnostic_only": False,
                "signal_executable": True,
                "signal_adjusted_close": 10.0,
            },
            {
                "signal_id": "c|A|2022-01-03|1|3|093100000",
                "strategy_id": "s0_random_matched",
                "strategy_family": "S0",
                "symbol": "A",
                "signal_date": "2022-01-03",
                "signal_time": "093100000",
                "sixty_minute_bucket": 1,
                "ma_period": 3,
                "event_trigger": "test",
                "causal_only": True,
                "diagnostic_only": False,
                "signal_executable": True,
                "signal_adjusted_close": 10.0,
            },
        ]
    )


def test_next_minute_entry_and_t1_outcome_are_observed() -> None:
    outcomes = compute_event_outcomes(
        _bars(),
        _signals(),
        config=EventStudyConfig(minute_horizons=(1,), day_horizons=(1,)),
    )
    row = outcomes.loc[outcomes["strategy_id"].eq("s1_touch_reclaim")].iloc[0]
    assert row["entry_time"] == "093200000"
    assert row["entry_price"] == pytest.approx(10.2)
    assert bool(row["entry_observed"]) is True
    assert bool(row["entry_executable"]) is True
    assert row["t1_exit_date"] == "2022-01-04"
    assert row["t1_exit_adjusted_price"] == pytest.approx(11.0)
    assert pd.notna(row["net_return_1m"])
    assert pd.notna(row["t1_net_return"])


def test_same_day_mfe_and_mae_start_at_entry_not_before_it() -> None:
    bars = _bars()
    bars.loc[(bars["trade_date"] == "2022-01-03") & (bars["bar_time"] == "093100000"), "high"] = 99.0
    bars.loc[(bars["trade_date"] == "2022-01-03") & (bars["bar_time"] == "093100000"), "low"] = 0.1
    outcomes = compute_event_outcomes(
        bars,
        _signals().iloc[[0]],
        config=EventStudyConfig(minute_horizons=(1,), day_horizons=(1,)),
    )
    row = outcomes.iloc[0]
    assert row["entry_time"] == "093200000"
    assert row["mfe_same_day"] < 1.0
    assert row["mae_same_day"] > -1.0


def test_missing_minute_is_not_scored_as_zero() -> None:
    outcomes = compute_event_outcomes(
        _bars(gap=True),
        _signals().iloc[[0]],
        config=EventStudyConfig(minute_horizons=(2,), day_horizons=(1,)),
    )
    row = outcomes.iloc[0]
    assert bool(row["entry_observed"]) is True
    assert row["entry_reason"] == "ok"
    assert pd.isna(row["net_return_2m"])


def test_explicit_calendar_does_not_skip_a_suspended_t1_day() -> None:
    bars = _bars()
    bars = bars.loc[bars["trade_date"] != "2022-01-04"].reset_index(drop=True)
    outcomes = compute_event_outcomes(
        bars,
        _signals().iloc[[0]],
        config=EventStudyConfig(minute_horizons=(1,), day_horizons=(1,)),
        trading_dates=["2022-01-03", "2022-01-04", "2022-01-05"],
    )
    row = outcomes.iloc[0]
    assert row["t1_exit_date"] == "2022-01-04"
    assert row["t1_exit_reason"] == "next_trading_day_bar_missing"
    assert bool(row["t1_exit_observed"]) is False
    assert pd.isna(row["net_return_1d"])


def test_summary_and_paired_control_are_finite_and_explicit() -> None:
    outcomes = compute_event_outcomes(
        _bars(),
        _signals(),
        config=EventStudyConfig(minute_horizons=(1,), day_horizons=(1,)),
    )
    summary = summarize_event_study(outcomes, group_by=("strategy_id", "ma_period"))
    assert set(summary["strategy_id"]) == {"s0_random_matched", "s1_touch_reclaim"}
    paired = compare_to_control(
        outcomes,
        strategy_id="s1_touch_reclaim",
        control_id="s0_random_matched",
        metric="net_return_1m",
    )
    assert len(paired) == 1
    assert summarize_control_comparison(paired)["matched_count"] == 1


def test_invalid_horizon_is_rejected() -> None:
    with pytest.raises(EventStudyError, match="minute_horizons"):
        EventStudyConfig(minute_horizons=(0,)).validate()


def test_minute_horizon_is_measured_from_the_next_open_fill_bar() -> None:
    bars = _bars().loc[lambda frame: frame["trade_date"].eq("2022-01-03")].copy()
    bars.loc[bars["bar_time"].eq("093100000"), ["open", "high", "low", "close"]] = [10.0, 10.0, 10.0, 10.0]
    bars.loc[bars["bar_time"].eq("093200000"), ["open", "high", "low", "close"]] = [11.0, 12.0, 10.5, 12.0]
    outcome = compute_event_outcomes(
        bars,
        _signals().iloc[[0]],
        config=EventStudyConfig(minute_horizons=(1,), day_horizons=(1,)),
    ).iloc[0]
    assert outcome["entry_time"] == "093200000"
    assert outcome["gross_return_1m"] == pytest.approx(12.0 / 11.0 - 1.0)


def test_reference_control_pairs_different_symbols_by_reference_signal_id() -> None:
    outcomes = pd.DataFrame(
        [
            {
                "signal_id": "ref-1",
                "strategy_id": "strategy",
                "symbol": "A",
                "signal_date": "2022-01-03",
                "signal_time": "093100000",
                "net_return_60m": 0.02,
                "reference_signal_id": None,
            },
            {
                "signal_id": "control-1",
                "strategy_id": "s0_liquidity_matched",
                "symbol": "B",
                "signal_date": "2022-01-03",
                "signal_time": "093100000",
                "net_return_60m": 0.01,
                "reference_signal_id": "ref-1",
            },
        ]
    )
    paired = compare_to_reference_control(outcomes, strategy_id="strategy")
    assert len(paired) == 1
    assert paired["strategy_symbol"].iloc[0] == "A"
    assert paired["control_symbol"].iloc[0] == "B"
    assert paired["paired_difference"].iloc[0] == pytest.approx(0.01)


def test_representative_regime_frame_is_fixed_and_unique() -> None:
    frame = build_representative_regime_frame()
    assert len(frame) == 12
    assert not frame.duplicated("trade_date").any()
