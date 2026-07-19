from __future__ import annotations

import numpy as np

from daily_research.path_policy.seq100_candidate_execution import (
    parse_execution_cost_contract,
)
from daily_research.path_policy.seq100_finite_capital_backtest import (
    BacktestMarket,
    ForecastDay,
    ForecastBook,
    PolicySpec,
    _all_jobs,
    detect_stop_plan,
    rolling_plan_update,
    simulate_portfolio,
)


def _contract():
    return parse_execution_cost_contract(
        {
            "execution_cost_contract": {
                "contract": "a_share_round_trip_cashflow_v1",
                "lot_size": 100,
                "commission_bps": 3.0,
                "minimum_commission_cny": 5.0,
                "transfer_fee_bps": 0.1,
                "slippage_bps": 7.0,
                "stress_slippage_multiplier": 2.0,
                "stamp_tax_schedule": [
                    {"effective_date": "1900-01-01", "stamp_tax_bps": 10.0}
                ],
            }
        }
    )


def _market(*, symbol_count: int = 3, date_count: int = 100) -> BacktestMarket:
    dates = np.asarray(
        [str(value) for value in np.datetime64("2023-01-02") + np.arange(date_count)],
        dtype=object,
    )
    open_price = np.full((date_count, symbol_count), 10.0, dtype=np.float32)
    close_price = np.full((date_count, symbol_count), 10.0, dtype=np.float32)
    return BacktestMarket(
        date_values=dates,
        symbol_values=np.asarray([f"S{idx}" for idx in range(symbol_count)], dtype=object),
        entry_open_raw=open_price,
        exit_close_raw=close_price,
        exit_sellable=np.ones((date_count, symbol_count), dtype=bool),
        entry_filled=np.ones((date_count, symbol_count), dtype=bool),
        contract=_contract(),
        terminal_recovery_fraction=0.25,
        forward_days=60,
        execution_days=80,
    )


def _book() -> ForecastBook:
    book = ForecastBook("legal_flat_baseline")
    book.add_day(
        date_idx=0,
        symbol_idx=np.asarray([0, 1, 2]),
        score=np.asarray([0.3, 0.2, 0.1]),
        planned_day=np.asarray([10, 10, 10]),
    )
    book.add_day(
        date_idx=1,
        symbol_idx=np.asarray([0, 1, 2]),
        score=np.asarray([-0.1, 0.2, 0.1]),
        planned_day=np.asarray([10, 10, 10]),
    )
    return book


def test_stop_plan_ignores_d1_and_uses_conservative_same_bar_order() -> None:
    path = np.asarray(
        [
            [12.0, 12.0, 8.0, 12.0],  # D1 is deliberately ignored.
            [10.0, 11.5, 9.0, 10.5],  # D2 hits both +10% and -5%.
        ]
        + [[10.0, 10.0, 10.0, 10.0]] * 58,
        dtype=np.float64,
    )
    result = detect_stop_plan(
        signal_date_idx=7,
        entry_price=10.0,
        raw_ohlc=path,
        take_profit=0.10,
        stop_loss=0.05,
        mode="intraday_conservative",
    )
    assert result.trigger_date_idx == 9
    assert result.requested_date_idx == 9
    assert result.event == "stop_loss"
    assert result.ambiguous is True
    assert result.trigger_price == 9.5


def test_close_confirmed_stop_uses_close_and_falls_back_to_d60() -> None:
    path = np.full((60, 4), 10.0, dtype=np.float64)
    path[:, 1] = 12.0
    path[:, 2] = 8.0
    no_close_trigger = detect_stop_plan(
        signal_date_idx=5,
        entry_price=10.0,
        raw_ohlc=path,
        take_profit=0.10,
        stop_loss=0.05,
        mode="close_confirmed",
    )
    assert no_close_trigger.event == "time_exit"
    assert no_close_trigger.requested_date_idx == 65
    path[4, 3] = 11.1
    close_trigger = detect_stop_plan(
        signal_date_idx=5,
        entry_price=10.0,
        raw_ohlc=path,
        take_profit=0.10,
        stop_loss=0.05,
        mode="close_confirmed",
    )
    assert close_trigger.event == "take_profit"
    assert close_trigger.trigger_date_idx == 10
    assert close_trigger.trigger_price == 11.1


def test_rolling_update_is_causal_accelerating_and_hard_capped() -> None:
    updated, reason = rolling_plan_update(
        current_date_idx=20,
        current_requested_date_idx=40,
        hard_cap_date_idx=60,
        score=-0.01,
        planned_day=2,
    )
    assert updated == 21
    assert reason == "nonpositive_value"
    unchanged, reason = rolling_plan_update(
        current_date_idx=20,
        current_requested_date_idx=30,
        hard_cap_date_idx=60,
        score=0.2,
        planned_day=20,
    )
    assert unchanged == 30
    assert reason == "unchanged"


def test_stateful_portfolio_does_not_replace_top3_or_pyramid() -> None:
    metric, _equity, trades, _annual = simulate_portfolio(
        market=_market(),
        book=_book(),
        raw_top3_paths={},
        policy=PolicySpec(name="fixed_d2", kind="fixed", fixed_day=2),
        slots=1,
        cost_scenario="base",
        first_signal_date_idx=0,
        last_signal_date_idx=1,
    )
    assert metric["buy_count"] == 1
    assert metric["sell_count"] == 1
    assert metric["closed_trade_count"] == 1
    assert metric["maximum_position_count"] == 1
    assert metric["skipped_duplicate_signal_count"] == 1
    assert metric["skipped_no_slot_signal_count"] == 4
    assert trades.iloc[0]["symbol"] == "S0"
    assert int(trades.iloc[0]["occupied_sessions"]) == 2


def test_rolling_portfolio_uses_next_close_after_nonpositive_forecast() -> None:
    metric, _equity, trades, _annual = simulate_portfolio(
        market=_market(),
        book=_book(),
        raw_top3_paths={},
        policy=PolicySpec(name="rolling_reforecast", kind="rolling"),
        slots=1,
        cost_scenario="base",
        first_signal_date_idx=0,
        last_signal_date_idx=1,
    )
    first = trades.iloc[0]
    assert int(first["entry_date_idx"]) == 1
    assert int(first["exit_date_idx"]) == 2
    assert metric["rolling_nonpositive_exit_request_count"] == 1
    assert metric["rolling_acceleration_count"] == 1


def test_stop_portfolio_uses_top3_raw_path_without_future_selection() -> None:
    path = np.full((60, 4), 10.0, dtype=np.float32)
    path[1] = np.asarray([10.0, 11.5, 9.8, 10.5], dtype=np.float32)
    metric, _equity, trades, _annual = simulate_portfolio(
        market=_market(),
        book=_book(),
        raw_top3_paths={(0, 0): path},
        policy=PolicySpec(
            name="stop_intraday_tp10_sl05",
            kind="stop",
            stop_mode="intraday_conservative",
            take_profit=0.10,
            stop_loss=0.05,
        ),
        slots=1,
        cost_scenario="base",
        first_signal_date_idx=0,
        last_signal_date_idx=1,
    )
    first = trades.iloc[0]
    assert int(first["exit_date_idx"]) == 2
    assert first["exit_reason"] == "take_profit"
    assert float(first["exit_price_raw"]) == 11.0
    assert metric["stop_take_profit_count"] == 1


def test_full_grid_has_expected_resume_job_count() -> None:
    jobs = _all_jobs()
    assert len(jobs) == 430
    assert sum(profile == "legal_flat_baseline" for profile, *_ in jobs) == 210
    assert sum(profile == "structured_joint_turnover" for profile, *_ in jobs) == 220


def test_single_daily_selection_is_reported_as_top1() -> None:
    book = _book()
    for date_idx, day in tuple(book.days.items()):
        book.days[date_idx] = ForecastDay(
            symbol_idx=day.symbol_idx,
            score=day.score,
            planned_day=day.planned_day,
            top3_symbol_idx=day.top3_symbol_idx[:1],
        )
    metric, _equity, _trades, _annual = simulate_portfolio(
        market=_market(),
        book=book,
        raw_top3_paths={},
        policy=PolicySpec(name="fixed_d2", kind="fixed", fixed_day=2),
        slots=1,
        cost_scenario="base",
        first_signal_date_idx=0,
        last_signal_date_idx=1,
    )
    assert metric["daily_selection_count"] == 1
    assert "daily_top1" in metric["portfolio_contract"]
    assert "no_top2" in metric["portfolio_contract"]
