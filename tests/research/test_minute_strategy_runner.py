from __future__ import annotations

import pandas as pd

import quantlab.research.minute_strategy_runner as runner
from quantlab.research.minute_ma_event_study import EventStudyConfig, compute_event_outcomes


def _bar_time(ordinal: int) -> str:
    if ordinal < 120:
        minute = ordinal
        hour = 9 + (31 + minute) // 60
        minute_value = (31 + minute) % 60
    else:
        minute = ordinal - 120
        hour = 13 + (1 + minute) // 60
        minute_value = (1 + minute) % 60
    return f"{hour:02d}{minute_value:02d}00000"


def _bars() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(("A", "B")):
        for day_index, trade_date in enumerate(("2022-01-03", "2022-01-04")):
            for ordinal in range(240):
                close = 10.0 + symbol_index + day_index * 0.1 + ordinal * 0.001
                rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": trade_date,
                        "bar_time": _bar_time(ordinal),
                        "open": close,
                        "high": close + 0.01,
                        "low": close - 0.01,
                        "close": close,
                        "volume": 100.0,
                        "amount": 1000.0,
                        "adjust_factor": 1.0,
                        "is_suspended": False,
                        "is_delisted": False,
                    }
                )
    return pd.DataFrame(rows)


def test_chunked_outcomes_match_single_batch_metrics(monkeypatch) -> None:
    bars = _bars()
    signals = pd.DataFrame(
        [
            {
                "signal_id": "sig-A",
                "strategy_id": "s1",
                "strategy_family": "S1",
                "symbol": "A",
                "signal_date": "2022-01-03",
                "signal_time": "093100000",
                "sixty_minute_bucket": 1,
                "ma_period": 10,
                "event_trigger": "touch_reclaim",
                "causal_only": True,
                "diagnostic_only": False,
                "signal_executable": True,
            },
            {
                "signal_id": "sig-B",
                "strategy_id": "s1",
                "strategy_family": "S1",
                "symbol": "B",
                "signal_date": "2022-01-03",
                "signal_time": "093100000",
                "sixty_minute_bucket": 1,
                "ma_period": 10,
                "event_trigger": "touch_reclaim",
                "causal_only": True,
                "diagnostic_only": False,
                "signal_executable": True,
            },
        ]
    )

    def fake_loader(*_args, symbols, trade_dates, **_kwargs):
        return bars.loc[
            bars["symbol"].isin(symbols) & bars["trade_date"].isin(trade_dates)
        ].copy()

    monkeypatch.setattr(runner, "load_target_bars", fake_loader)
    config = runner.DevelopmentStudyConfig(
        signal_symbol_chunk_size=1,
        memory_floor_gib=0.5,
        soft_memory_floor_gib=0.5,
    )
    event_config = EventStudyConfig(
        minute_horizons=(5,),
        day_horizons=(1,),
    )
    chunked = runner._compute_outcomes_chunked(
        runner.Path("."),
        signals,
        future_dates=("2022-01-03", "2022-01-04"),
        trading_dates=("2022-01-03", "2022-01-04"),
        data_config=runner.StrategyDataConfig(),
        event_config=event_config,
        requested_chunk_size=1,
        memory_config=config,
    )
    expected = runner._lean_outcomes(
        compute_event_outcomes(
            bars,
            signals,
            config=event_config,
            trading_dates=("2022-01-03", "2022-01-04"),
        )
    )
    key = ["signal_id"]
    metric_columns = ["gross_return_5m", "net_return_5m", "t1_gross_return", "t1_net_return"]
    left = chunked.sort_values(key).reset_index(drop=True)
    right = expected.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        left.loc[:, key + metric_columns],
        right.loc[:, key + metric_columns],
        check_dtype=False,
    )
    assert left["entry_bar_index"].is_unique
    assert left["entry_bar_index"].min() >= 0
