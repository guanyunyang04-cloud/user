from __future__ import annotations

import pandas as pd
import pytest

from quantlab.research.minute_ma import (
    MinuteMAConfig,
    build_hourly_bars,
    build_minute_ma_states,
    build_minute_ma_states_from_history,
    session_minute_ordinal,
)
from quantlab.research.minute_strategy_data import (
    MinuteStrategyDataError,
    StrategyDataConfig,
    build_enriched_minute_ma_states,
    build_live_ma_alignment,
    build_minute_market_context,
)


def _bars(symbol: str = "A", dates: tuple[str, ...] = ("2022-01-03",)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day_index, trade_date in enumerate(dates):
        for ordinal in range(240):
            if ordinal < 120:
                minute = ordinal
                hour = 9 + (31 + minute) // 60
                minute_value = (31 + minute) % 60
            else:
                minute = ordinal - 120
                hour = 13 + (1 + minute) // 60
                minute_value = (1 + minute) % 60
            bar_time = f"{hour:02d}{minute_value:02d}00000"
            close = 10.0 + day_index * 0.1 + ordinal * 0.0001
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "bar_time": bar_time,
                    "open": close,
                    "high": close + 0.01,
                    "low": close - 0.01,
                    "close": close,
                    "volume": 100.0,
                    "amount": 1000.0,
                    "adjust_factor": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_aggregated_history_path_matches_contiguous_path() -> None:
    bars = _bars(dates=("2022-01-03", "2022-01-04"))
    config = MinuteMAConfig(periods=(3,))
    direct = build_minute_ma_states(bars, config=config, target_dates=["2022-01-04"])
    hourly = build_hourly_bars(bars)
    reduced = build_minute_ma_states_from_history(
        bars.loc[bars["trade_date"].eq("2022-01-04")], hourly, config=config
    )
    columns = [
        "causal_intersection",
        "live_ma",
        "touched_now",
        "close_below_intersection",
        "prior_true_touch_count_window",
    ]
    pd.testing.assert_frame_equal(
        direct[columns].reset_index(drop=True),
        reduced[columns].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_market_context_exposes_causal_rule_flags() -> None:
    bars = _bars(dates=("2022-01-03",))
    bars["session_minute_ordinal"] = bars["bar_time"].map(session_minute_ordinal)
    bars["sixty_minute_bucket"] = bars["session_minute_ordinal"].floordiv(60).add(1)
    for column in ("open", "high", "low", "close"):
        bars[f"adjusted_{column}"] = bars[column]
    daily = pd.DataFrame(
        [
            {
                "symbol": "A",
                "trade_date": "2022-01-03",
                "industry_name": "I",
                "previous_close_adjusted": 10.0,
                "prior_20d_median_amount": 1000.0,
                "breakout_recent": True,
                "prior_acceleration": True,
                "daily_trend_positive": True,
                "up_limit": 11.0,
                "down_limit": 9.0,
                "auction_gap": 0.0,
                "auction_amount": 100.0,
                "daily_trend_position": 0.1,
            }
        ]
    )
    context = build_minute_market_context(bars, daily_context=daily)
    required = {
        "market_regime",
        "market_supportive",
        "sector_strength_rank",
        "sector_strong",
        "leader_sync",
        "auction_confirmed",
        "vwap_supportive",
        "amount_acceleration_positive",
        "volume_normal",
    }
    assert required.issubset(context.columns)
    assert len(context) == 240


def test_enriched_states_keep_author_filter_inputs_and_alignment() -> None:
    bars = _bars(dates=("2022-01-03", "2022-01-04"))
    hourly = build_hourly_bars(bars)
    target = bars.loc[bars["trade_date"].eq("2022-01-04")].copy()
    daily = pd.DataFrame(
        [
            {
                "symbol": "A",
                "trade_date": "2022-01-04",
                "industry_name": "I",
                "previous_close_adjusted": 10.1,
                "prior_20d_median_amount": 1000.0,
                "breakout_recent": True,
                "prior_acceleration": True,
                "daily_trend_positive": True,
                "up_limit": 11.1,
                "down_limit": 9.1,
                "auction_gap": 0.0,
                "auction_amount": 100.0,
                "daily_trend_position": 0.1,
            }
        ]
    )
    states = build_enriched_minute_ma_states(
        target,
        hourly,
        daily_context=daily,
        config=MinuteMAConfig(periods=(3,)),
    )
    alignment = build_live_ma_alignment(target, hourly, config=MinuteMAConfig(periods=(3,)))
    assert len(states) == 240
    assert {"market_supportive", "sector_strong", "not_repeated_cross"}.issubset(states.columns)
    assert len(alignment) == 240


def test_strategy_data_config_rejects_invalid_matching_band() -> None:
    with pytest.raises(MinuteStrategyDataError, match="liquidity_match_band"):
        StrategyDataConfig(liquidity_match_band=0.5).validate()
