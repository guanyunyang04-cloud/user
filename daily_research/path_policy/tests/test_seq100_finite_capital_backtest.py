from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from daily_research.path_policy.seq100_candidate_execution import (
    parse_execution_costs,
)
from daily_research.path_policy.seq100_finite_capital_backtest import (
    BacktestMarket,
    ForecastBook,
    ForecastDay,
    PolicySpec,
    StudyEvaluationSpec,
    _all_jobs,
    _study_jobs,
    detect_stop_plan,
    rolling_plan_update,
    select_study_winner,
    simulate_portfolio,
)


def _contract():
    return parse_execution_costs(
        {
            "execution_costs": {
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
        symbol_values=np.asarray(
            [f"S{idx}" for idx in range(symbol_count)], dtype=object
        ),
        entry_open_raw=open_price,
        exit_close_raw=close_price,
        exit_sellable=np.ones((date_count, symbol_count), dtype=bool),
        entry_filled=np.ones((date_count, symbol_count), dtype=bool),
        costs=_contract(),
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


def test_adjust_factor_preserves_total_return_across_ex_date() -> None:
    raw_market = _market(symbol_count=1)
    raw_market.exit_close_raw[2:, 0] = 5.0
    raw_market.entry_open_raw[2:, 0] = 5.0
    factor = np.ones((100, 1), dtype=np.float32)
    factor[2:, 0] = 2.0
    adjusted_market = BacktestMarket(
        date_values=raw_market.date_values,
        symbol_values=raw_market.symbol_values,
        entry_open_raw=raw_market.entry_open_raw,
        exit_close_raw=raw_market.exit_close_raw,
        exit_sellable=raw_market.exit_sellable,
        entry_filled=raw_market.entry_filled,
        costs=raw_market.costs,
        terminal_recovery_fraction=raw_market.terminal_recovery_fraction,
        forward_days=raw_market.forward_days,
        execution_days=raw_market.execution_days,
        adjust_factor=factor,
    )
    book = ForecastBook("factor", top_k=1)
    book.add_day(
        date_idx=0,
        symbol_idx=np.asarray([0]),
        score=np.asarray([1.0]),
        planned_day=np.asarray([2]),
    )
    raw_metric, _, raw_trades, _ = simulate_portfolio(
        market=raw_market,
        book=book,
        raw_top3_paths={},
        policy=PolicySpec(name="fixed_d2", kind="fixed", fixed_day=2),
        slots=1,
        cost_scenario="base",
        first_signal_date_idx=0,
        last_signal_date_idx=0,
    )
    adjusted_metric, _, adjusted_trades, _ = simulate_portfolio(
        market=adjusted_market,
        book=book,
        raw_top3_paths={},
        policy=PolicySpec(name="fixed_d2", kind="fixed", fixed_day=2),
        slots=1,
        cost_scenario="base",
        first_signal_date_idx=0,
        last_signal_date_idx=0,
    )

    assert raw_metric["liquidated_total_return"] < -0.49
    assert adjusted_metric["liquidated_total_return"] > -0.01
    assert adjusted_metric["corporate_action_adjusted_equivalent"] is True
    assert raw_metric["corporate_action_adjusted_equivalent"] is False
    assert adjusted_trades.iloc[0]["corporate_action_value_multiplier"] == 2.0
    assert adjusted_trades.iloc[0]["exit_price_economic_equivalent"] == 10.0
    assert raw_trades.iloc[0]["exit_price_economic_equivalent"] == 5.0


def test_target_gross_fraction_scales_entry_cash_without_leverage() -> None:
    market = _market(symbol_count=1)
    book = ForecastBook("risk_budget", top_k=1)
    book.add_day(
        date_idx=0,
        symbol_idx=np.asarray([0]),
        score=np.asarray([1.0]),
        planned_day=np.asarray([2]),
    )
    full, _, full_trades, _ = simulate_portfolio(
        market=market,
        book=book,
        raw_top3_paths={},
        policy=PolicySpec(name="fixed_d2", kind="fixed", fixed_day=2),
        slots=1,
        cost_scenario="base",
        first_signal_date_idx=0,
        last_signal_date_idx=0,
    )
    half, _, half_trades, _ = simulate_portfolio(
        market=market,
        book=book,
        raw_top3_paths={},
        policy=PolicySpec(name="fixed_d2", kind="fixed", fixed_day=2),
        slots=1,
        cost_scenario="base",
        first_signal_date_idx=0,
        last_signal_date_idx=0,
        target_gross_fraction=0.5,
    )

    ratio = float(
        half_trades.iloc[0]["buy_notional_cny"]
        / full_trades.iloc[0]["buy_notional_cny"]
    )
    assert 0.49 < ratio < 0.51
    assert half["target_gross_fraction"] == 0.5
    assert full["target_gross_fraction"] == 1.0


def test_stateful_portfolio_can_scan_ranked_candidates_for_replacement() -> None:
    book = ForecastBook("replacement", top_k=1, candidate_scan_k=3)
    for date_idx in (0, 1):
        book.add_day(
            date_idx=date_idx,
            symbol_idx=np.asarray([0, 1, 2]),
            score=np.asarray([0.3, 0.2, 0.1]),
            planned_day=np.asarray([3, 3, 3]),
        )
    metric, _equity, trades, _annual = simulate_portfolio(
        market=_market(),
        book=book,
        raw_top3_paths={},
        policy=PolicySpec(name="fixed_d3", kind="fixed", fixed_day=3),
        slots=2,
        cost_scenario="base",
        first_signal_date_idx=0,
        last_signal_date_idx=1,
        replace_rejected_from_ranked_candidates=True,
    )
    assert metric["buy_count"] == 2
    assert metric["replacement_order_count"] == 1
    assert metric["candidate_scan_k"] == 3
    assert set(trades["symbol"]) == {"S0", "S1"}


def test_low_price_terminal_writeoff_never_overdraws_cash() -> None:
    base = _market(symbol_count=1, date_count=100)
    market = BacktestMarket(
        date_values=base.date_values,
        symbol_values=base.symbol_values,
        entry_open_raw=np.full((100, 1), 0.36, dtype=np.float32),
        exit_close_raw=np.full((100, 1), 0.36, dtype=np.float32),
        exit_sellable=np.zeros((100, 1), dtype=bool),
        entry_filled=base.entry_filled,
        costs=base.costs,
        terminal_recovery_fraction=0.0,
        forward_days=60,
        execution_days=80,
    )
    book = ForecastBook("terminal_writeoff", top_k=1)
    book.add_day(
        date_idx=0,
        symbol_idx=np.asarray([0]),
        score=np.asarray([1.0]),
        planned_day=np.asarray([2]),
    )
    metric, equity, trades, _annual = simulate_portfolio(
        market=market,
        book=book,
        raw_top3_paths={},
        policy=PolicySpec(name="fixed_d2", kind="fixed", fixed_day=2),
        slots=1,
        cost_scenario="double_slippage",
        first_signal_date_idx=0,
        last_signal_date_idx=0,
    )

    trade = trades.iloc[0]
    assert float(trade["buy_cash_cny"]) <= 1_000_000.0
    assert float(trade["sell_proceeds_cny"]) == 0.0
    assert trade["exit_reason"] == "terminal_recovery"
    assert float(equity["cash"].min()) >= 0.0
    assert metric["minimum_cash_cny"] >= 0.0
    assert metric["minimum_equity_cny"] >= 0.0
    assert metric["liquidated_ending_equity_cny"] >= 0.0


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


def test_portfolio_enforces_signal_amount_capacity() -> None:
    market = _market(symbol_count=1)
    book = ForecastBook("capacity", top_k=1)
    book.add_day(
        date_idx=0,
        symbol_idx=np.asarray([0]),
        score=np.asarray([0.2]),
        planned_day=np.asarray([2]),
    )
    signal_amount = np.full((100, 1), 100_000.0, dtype=np.float32)
    metric, _equity, trades, _annual = simulate_portfolio(
        market=market,
        book=book,
        raw_top3_paths={},
        policy=PolicySpec(name="fixed_d2", kind="fixed", fixed_day=2),
        slots=1,
        cost_scenario="base",
        first_signal_date_idx=0,
        last_signal_date_idx=0,
        signal_amount_panel=signal_amount,
        maximum_signal_amount_fraction=0.10,
    )
    trade = trades.iloc[0]
    assert float(trade["buy_notional_cny"]) <= 10_000.0
    assert float(trade["signal_amount_participation"]) <= 0.10
    assert metric["capacity_capped_order_count"] == 1


def test_rolling_portfolio_exits_after_belief_disappears() -> None:
    market = _market(symbol_count=1)
    book = ForecastBook("missing_belief", top_k=1)
    book.add_day(
        date_idx=0,
        symbol_idx=np.asarray([0]),
        score=np.asarray([0.2]),
        planned_day=np.asarray([60]),
    )
    metric, _equity, trades, _annual = simulate_portfolio(
        market=market,
        book=book,
        raw_top3_paths={},
        policy=PolicySpec(name="rolling_reforecast", kind="rolling"),
        slots=1,
        cost_scenario="base",
        first_signal_date_idx=0,
        last_signal_date_idx=0,
        exit_on_missing_forecast=True,
    )
    trade = trades.iloc[0]
    assert int(trade["entry_date_idx"]) == 1
    assert int(trade["exit_date_idx"]) == 2
    assert trade["exit_reason"] == "missing_belief"
    assert metric["rolling_missing_exit_request_count"] == 1


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


def test_portfolio_accepts_varying_positive_score_selection_with_cash_days() -> None:
    book = ForecastBook("v4", top_k=3)
    book.add_day(
        date_idx=0,
        symbol_idx=np.asarray([0, 1, 2]),
        score=np.asarray([0.3, 0.2, 0.0]),
        planned_day=np.asarray([2, 3, 2]),
        selection_mask=np.asarray([True, True, False]),
    )
    book.add_day(
        date_idx=1,
        symbol_idx=np.asarray([0, 1, 2]),
        score=np.asarray([0.0, 0.0, 0.0]),
        planned_day=np.asarray([2, 2, 2]),
        selection_mask=np.asarray([False, False, False]),
    )

    metric, _equity, _trades, _annual = simulate_portfolio(
        market=_market(),
        book=book,
        raw_top3_paths={},
        policy=PolicySpec(name="fixed_d2", kind="fixed", fixed_day=2),
        slots=3,
        cost_scenario="double_slippage",
        first_signal_date_idx=0,
        last_signal_date_idx=1,
    )

    assert metric["configured_top_k"] == 3
    assert metric["daily_selection_count"] == 3
    assert metric["selected_name_count_min"] == 0
    assert metric["selected_name_count_max"] == 2
    assert metric["selected_name_count_mean"] == 1.0
    assert metric["cash_filtered_signal_days"] == 2
    assert metric["buy_count"] == 2
    assert book.lookup(1, 0) == (0.0, 2)


def test_forecast_book_retains_the_requested_top_k() -> None:
    book = ForecastBook("structured_joint_turnover", top_k=2)
    book.add_day(
        date_idx=0,
        symbol_idx=np.asarray([0, 1, 2]),
        score=np.asarray([0.1, 0.3, 0.2]),
        planned_day=np.asarray([7, 8, 9]),
    )
    assert book.days[0].top3_symbol_idx == (1, 2)


def test_forecast_book_rejects_nonpositive_top_k() -> None:
    try:
        ForecastBook("structured_joint_turnover", top_k=0)
    except ValueError as exc:
        assert "top_k must be positive" in str(exc)
    else:
        raise AssertionError("nonpositive top_k was accepted")


def test_simulation_accepts_an_explicit_calendar_year_set() -> None:
    _metric, _equity, _trades, annual = simulate_portfolio(
        market=_market(),
        book=_book(),
        raw_top3_paths={},
        policy=PolicySpec(name="fixed_d2", kind="fixed", fixed_day=2),
        slots=1,
        cost_scenario="base",
        first_signal_date_idx=0,
        last_signal_date_idx=1,
        calendar_years=(2023,),
    )
    assert [int(row["year"]) for row in annual] == [2023]


def test_signal_close_study_grid_has_exactly_4148_jobs() -> None:
    profiles = ("V2C-P0", "V2C-P1", "V4-P0", "V4-P1")
    years = (2023, 2024, 2025)
    policies = tuple(
        [
            PolicySpec(name=f"fixed_d{day}", kind="fixed", fixed_day=day)
            for day in range(2, 61)
        ]
        + [
            PolicySpec(name="model_plan", kind="model_plan"),
            PolicySpec(name="rolling_path", kind="rolling"),
        ]
    )
    spec = StudyEvaluationSpec(
        study_id="tiny_signal_close_2x2",
        profiles=profiles,
        years=years,
        fold_views={year: Path(f"view_{year}.json") for year in years},
        prediction_paths={
            profile: {year: Path(f"{profile}_{year}.csv") for year in years}
            for profile in profiles
        },
        top_k_slot_grid={
            1: (1, 3, 6, 12, 24, 48),
            2: (2, 4, 6, 12, 24, 48),
            3: (3, 6, 12, 24, 48),
        },
        policies=policies,
        expected_job_count=4148,
    )

    spec.validate()
    jobs = _study_jobs(spec)
    assert len(jobs) == 4148
    assert {job.cost_scenario for job in jobs} == {"double_slippage"}


def test_explicit_top_k_uses_one_name_without_rebuilding_forecast_book() -> None:
    book = ForecastBook("shared_book", top_k=3, candidate_scan_k=10)
    book.add_day(
        date_idx=0,
        symbol_idx=np.asarray([0, 1, 2]),
        score=np.asarray([0.3, 0.2, 0.1]),
        planned_day=np.asarray([2, 2, 2]),
    )
    metric, _equity, trades, _annual = simulate_portfolio(
        market=_market(),
        book=book,
        raw_top3_paths={},
        policy=PolicySpec(name="fixed_d2", kind="fixed", fixed_day=2),
        slots=1,
        cost_scenario="double_slippage",
        first_signal_date_idx=0,
        last_signal_date_idx=0,
        top_k=1,
        replace_rejected_from_ranked_candidates=True,
    )

    assert metric["configured_top_k"] == 1
    assert metric["buy_count"] == 1
    assert list(trades["symbol"]) == ["S0"]


def test_next_open_failure_does_not_trigger_ranked_replacement() -> None:
    base = _market()
    entry_filled = base.entry_filled.copy()
    entry_filled[0, 0] = False
    market = BacktestMarket(
        date_values=base.date_values,
        symbol_values=base.symbol_values,
        entry_open_raw=base.entry_open_raw,
        exit_close_raw=base.exit_close_raw,
        exit_sellable=base.exit_sellable,
        entry_filled=entry_filled,
        costs=base.costs,
        terminal_recovery_fraction=base.terminal_recovery_fraction,
        forward_days=base.forward_days,
        execution_days=base.execution_days,
    )
    book = ForecastBook("no_next_open_replacement", top_k=1, candidate_scan_k=3)
    book.add_day(
        date_idx=0,
        symbol_idx=np.asarray([0, 1, 2]),
        score=np.asarray([0.3, 0.2, 0.1]),
        planned_day=np.asarray([2, 2, 2]),
    )
    metric, _equity, trades, _annual = simulate_portfolio(
        market=market,
        book=book,
        raw_top3_paths={},
        policy=PolicySpec(name="fixed_d2", kind="fixed", fixed_day=2),
        slots=1,
        cost_scenario="double_slippage",
        first_signal_date_idx=0,
        last_signal_date_idx=0,
        top_k=1,
        replace_rejected_from_ranked_candidates=True,
    )

    assert metric["failed_entry_count"] == 1
    assert metric["replacement_order_count"] == 0
    assert metric["buy_count"] == 0
    assert trades.empty
    assert metric["liquidated_ending_equity_cny"] == 1_000_000.0
    assert metric["annualized_log_growth"] == 0.0


def test_winner_selection_applies_strict_autonomous_fallback_and_budget_gate(
    tmp_path: Path,
) -> None:
    profiles = ("V2C-P0", "V2C-P1", "V4-P0", "V4-P1")
    top_k_slot_grid = {
        1: (1, 3, 6, 12, 24, 48),
        2: (2, 4, 6, 12, 24, 48),
        3: (3, 6, 12, 24, 48),
    }
    root = tmp_path / "account"
    jobs_root = root / "jobs"
    jobs_root.mkdir(parents=True)
    count = 0
    configured_jobs = []
    for profile in profiles:
        profile_base = {
            "V2C-P0": 0.10,
            "V2C-P1": 0.30,
            "V4-P0": 0.20,
            "V4-P1": 0.40,
        }[profile]
        for top_k, slot_values in top_k_slot_grid.items():
            for slots in slot_values:
                group_bonus = 0.01 if top_k == 1 and slots == 1 else 0.0
                best_fixed = profile_base + group_bonus
                policies = [
                    (f"fixed_d{day}", "fixed", day, best_fixed - abs(day - 10) * 0.001)
                    for day in range(2, 61)
                ]
                model_growth = best_fixed + 0.5e-12
                if profile == "V2C-P0" and top_k == 1 and slots == 1:
                    model_growth = best_fixed + 2.0e-12
                policies.extend(
                    [
                        ("model_plan", "model_plan", None, model_growth),
                        ("rolling_path", "rolling", None, best_fixed - 0.01),
                    ]
                )
                for name, kind, fixed_day, growth in policies:
                    count += 1
                    annual = [
                        {
                            "year": year,
                            "log_growth": growth / 3.0,
                            "net_return": float(np.expm1(growth / 3.0)),
                        }
                        for year in (2023, 2024, 2025)
                    ]
                    payload = {
                        "status": "completed",
                        "job_id": f"job_{count}",
                        "job": {
                            "profile": profile,
                            "top_k": top_k,
                            "slots": slots,
                            "cost_scenario": "double_slippage",
                            "policy": {
                                "name": name,
                                "kind": kind,
                                "fixed_day": fixed_day,
                            },
                        },
                        "metric": {
                            "profile": profile,
                            "policy": name,
                            "policy_kind": kind,
                            "top_k": top_k,
                            "slot_count": slots,
                            "liquidated_log_growth": growth,
                            "annualized_log_growth": growth,
                            "full_path_maximum_drawdown": -0.20,
                        },
                        "annual_metrics": annual,
                    }
                    configured_jobs.append({"job_id": f"job_{count}", **payload["job"]})
                    (jobs_root / f"job_{count}.json").write_text(
                        json.dumps(payload), encoding="utf-8"
                    )
    assert count == 4148
    (root / "config.json").write_text(
        json.dumps({"job_count": 4148, "jobs": configured_jobs}),
        encoding="utf-8",
    )

    result = select_study_winner(
        account_output_root=root,
        profiles=profiles,
        p1_budget_evidence={"V2C-P1": False, "V4-P1": True},
        training_complete=True,
    )

    assert result["winner"]["profile"] == "V4-P1"
    assert result["winner"]["policy"] == "fixed_d10"
    arms = {row["profile"]: row for row in result["arms"]}
    assert arms["V2C-P0"]["policy"] == "model_plan"
    assert arms["V2C-P1"]["qualified"] is False
    assert arms["V2C-P1"]["p1_budget_evidence_passed"] is False
