from __future__ import annotations

import pandas as pd

from quantlab.research.minute_strategy_portfolio import (
    ExitPolicy,
    MinutePortfolioError,
    PortfolioConfig,
    simulate_portfolio_accounts,
    summarize_selection_seed_results,
)


def _time(ordinal: int) -> str:
    if ordinal < 120:
        minute = ordinal
        hour = 9 + (31 + minute) // 60
        minute_value = (31 + minute) % 60
    else:
        minute = ordinal - 120
        hour = 13 + (1 + minute) // 60
        minute_value = (1 + minute) % 60
    return f"{hour:02d}{minute_value:02d}00000"


def _bars(prices: dict[str, list[float]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = ["2022-01-03", "2022-01-04", "2022-01-05"]
    for symbol, day_prices in prices.items():
        for date, base in zip(dates, day_prices, strict=True):
            for ordinal in range(240):
                value = float(base)
                rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": date,
                        "bar_time": _time(ordinal),
                        "open": value,
                        "high": value,
                        "low": value,
                        "close": value,
                        "adjusted_open": value,
                        "adjusted_high": value,
                        "adjusted_low": value,
                        "adjusted_close": value,
                        "session_minute_ordinal": ordinal,
                        "status_known": True,
                        "is_suspended": False,
                        "is_delisted": False,
                    }
                )
    return pd.DataFrame(rows)


def _signal(
    symbol: str,
    signal_id: str,
    *,
    signal_date: str = "2022-01-03",
    signal_time: str = "093100000",
    entry_time: str = "093200000",
) -> dict[str, object]:
    return {
        "signal_id": signal_id,
        "strategy_id": "s1_touch_reclaim",
        "strategy_family": "S1",
        "symbol": symbol,
        "signal_date": signal_date,
        "signal_time": signal_time,
        "entry_date": signal_date,
        "entry_time": entry_time,
        "entry_price": 10.0,
        "entry_adjusted_price": 10.0,
        "entry_observed": True,
        "entry_executable": True,
        "signal_executable": True,
        "diagnostic_only": False,
        "sixty_minute_bucket": 1,
        "ma_period": 10,
        "event_trigger": "touch_reclaim",
    }


def test_time_exit_obeys_t1_and_closes_at_next_day_open() -> None:
    bars = _bars({"A": [10.0, 10.4, 10.5]})
    signals = {"2022-01-03": pd.DataFrame([_signal("A", "sig-A")])}

    def loader(symbols, date):
        return bars.loc[
            bars["symbol"].isin(symbols) & bars["trade_date"].eq(date)
        ].copy()

    results, _equity, trades = simulate_portfolio_accounts(
        signals,
        bars_loader=loader,
        calendar=["2022-01-03", "2022-01-04", "2022-01-05"],
        strategy_ids=("s1_touch_reclaim",),
        policies=(ExitPolicy("one_day", 1),),
        config=PortfolioConfig(starting_cash=100_000, max_positions=5),
    )
    assert results[0]["filled_entry_count"] == 1
    assert len(trades) == 1
    assert trades.iloc[0]["exit_date"] == "2022-01-04"
    assert trades.iloc[0]["exit_time"] == "093100000"
    assert trades.iloc[0]["exit_reason"] == "time_stop"


def test_protective_stop_uses_next_available_minute_open() -> None:
    bars = _bars({"A": [10.0, 10.0, 10.0]})
    day_two = bars["trade_date"].eq("2022-01-04") & bars["symbol"].eq("A")
    second_bar = day_two & bars["session_minute_ordinal"].eq(1)
    bars.loc[second_bar, ["low", "adjusted_low"]] = 9.5
    bars.loc[second_bar, ["high", "close", "adjusted_high", "adjusted_close"]] = 9.6
    signals = {"2022-01-03": pd.DataFrame([_signal("A", "sig-A")])}

    def loader(symbols, date):
        return bars.loc[
            bars["symbol"].isin(symbols) & bars["trade_date"].eq(date)
        ].copy()

    policy = ExitPolicy("stop", 3, stop_loss_bps=300)
    _results, _equity, trades = simulate_portfolio_accounts(
        signals,
        bars_loader=loader,
        calendar=["2022-01-03", "2022-01-04", "2022-01-05"],
        strategy_ids=("s1_touch_reclaim",),
        policies=(policy,),
        config=PortfolioConfig(starting_cash=100_000, max_positions=5),
    )
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "stop_loss"
    assert trades.iloc[0]["exit_time"] == "093300000"


def test_max_positions_is_the_only_entry_capacity_limit() -> None:
    bars = _bars({"A": [10.0, 10.0, 10.0], "B": [10.0, 10.0, 10.0]})
    signals = {
        "2022-01-03": pd.DataFrame(
            [
                _signal("A", "sig-A"),
                _signal("B", "sig-B", signal_time="093200000", entry_time="093300000"),
            ]
        )
    }

    def loader(symbols, date):
        return bars.loc[
            bars["symbol"].isin(symbols) & bars["trade_date"].eq(date)
        ].copy()

    result, _equity, _trades = simulate_portfolio_accounts(
        signals,
        bars_loader=loader,
        calendar=["2022-01-03", "2022-01-04", "2022-01-05"],
        strategy_ids=("s1_touch_reclaim",),
        policies=(ExitPolicy("one_day", 1),),
        config=PortfolioConfig(starting_cash=100_000, max_positions=1),
    )
    assert result[0]["filled_entry_count"] == 1
    assert result[0]["skipped_slot_count"] >= 1


def test_policy_rejects_invalid_trailing_parameters() -> None:
    try:
        ExitPolicy("bad", 2, trail_activation_bps=200, trail_drawdown_bps=200).validate()
    except MinutePortfolioError as exc:
        assert "trail_order" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid trailing policy was accepted")


def test_same_day_mark_and_peak_are_observed_but_exit_waits_for_t1() -> None:
    bars = _bars({"A": [10.0, 10.0, 10.0]})
    day_one = bars["trade_date"].eq("2022-01-03") & bars["symbol"].eq("A")
    after_entry = day_one & bars["session_minute_ordinal"].ge(1)
    bars.loc[after_entry, ["close", "high", "adjusted_close", "adjusted_high"]] = 10.6
    day_two = bars["trade_date"].eq("2022-01-04") & bars["symbol"].eq("A")
    bars.loc[day_two, ["open", "high", "low", "close", "adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"]] = 10.6
    signals = {"2022-01-03": pd.DataFrame([_signal("A", "sig-A")])}

    def loader(symbols, date):
        return bars.loc[
            bars["symbol"].isin(symbols) & bars["trade_date"].eq(date)
        ].copy()

    results, equity, trades = simulate_portfolio_accounts(
        signals,
        bars_loader=loader,
        calendar=["2022-01-03", "2022-01-04", "2022-01-05"],
        strategy_ids=("s1_touch_reclaim",),
        policies=(ExitPolicy("target", 3, take_profit_bps=500),),
        config=PortfolioConfig(starting_cash=100_000, max_positions=5),
    )
    day_one_equity = float(equity.loc[equity["trade_date"].eq("2022-01-03"), "equity"].iloc[0])
    assert day_one_equity > 100_000.0
    assert len(trades) == 1
    assert trades.iloc[0]["exit_date"] == "2022-01-04"
    assert trades.iloc[0]["exit_time"] == "093100000"
    assert trades.iloc[0]["exit_reason"] == "take_profit"
    assert results[0]["filled_entry_count"] == 1


def test_simultaneous_entries_share_cash_across_available_slots() -> None:
    bars = _bars({"A": [10.0, 10.0, 10.0], "B": [10.0, 10.0, 10.0]})
    signals = {
        "2022-01-03": pd.DataFrame(
            [
                _signal("A", "sig-A"),
                _signal("B", "sig-B", signal_time="093100000", entry_time="093200000"),
            ]
        )
    }

    def loader(symbols, date):
        return bars.loc[
            bars["symbol"].isin(symbols) & bars["trade_date"].eq(date)
        ].copy()

    _results, _equity, trades = simulate_portfolio_accounts(
        signals,
        bars_loader=loader,
        calendar=["2022-01-03", "2022-01-04", "2022-01-05"],
        strategy_ids=("s1_touch_reclaim",),
        policies=(ExitPolicy("one_day", 1),),
        config=PortfolioConfig(starting_cash=100_000, max_positions=2),
    )
    assert len(trades) == 2
    assert trades["shares"].nunique() == 1


def test_later_signal_can_use_newly_freed_slot_budget() -> None:
    bars = _bars({"A": [10.0, 10.0, 10.0], "B": [10.0, 10.0, 10.0]})
    signals = {
        "2022-01-03": pd.DataFrame([_signal("A", "sig-A")]),
        "2022-01-04": pd.DataFrame(
            [_signal("B", "sig-B", signal_date="2022-01-04")]
        ),
    }

    def loader(symbols, date):
        return bars.loc[
            bars["symbol"].isin(symbols) & bars["trade_date"].eq(date)
        ].copy()

    result, _equity, trades = simulate_portfolio_accounts(
        signals,
        bars_loader=loader,
        calendar=["2022-01-03", "2022-01-04", "2022-01-05"],
        strategy_ids=("s1_touch_reclaim",),
        policies=(ExitPolicy("one_day", 1),),
        config=PortfolioConfig(starting_cash=100_000, max_positions=1),
    )
    assert result[0]["filled_entry_count"] == 2
    assert len(trades) == 2


def test_same_minute_selection_does_not_depend_on_input_order() -> None:
    bars = _bars({"A": [10.0, 10.0, 10.0], "B": [10.0, 10.0, 10.0]})
    rows = [_signal("A", "sig-A"), _signal("B", "sig-B")]

    def loader(symbols, date):
        return bars.loc[
            bars["symbol"].isin(symbols) & bars["trade_date"].eq(date)
        ].copy()

    selected_symbols = []
    for ordered in (rows, list(reversed(rows))):
        _result, _equity, trades = simulate_portfolio_accounts(
            {"2022-01-03": pd.DataFrame(ordered)},
            bars_loader=loader,
            calendar=["2022-01-03", "2022-01-04"],
            strategy_ids=("s1_touch_reclaim",),
            policies=(ExitPolicy("one_day", 1),),
            config=PortfolioConfig(
                starting_cash=100_000,
                max_positions=1,
                selection_seed=7,
            ),
        )
        selected_symbols.append(str(trades.iloc[0]["symbol"]))
    assert selected_symbols[0] == selected_symbols[1]


def test_multiple_selection_seeds_share_one_replay_and_remain_identifiable() -> None:
    symbols = tuple("ABCDEFGHIJ")
    bars = _bars({symbol: [10.0, 10.0, 10.0] for symbol in symbols})
    signals = {
        "2022-01-03": pd.DataFrame(
            [_signal(symbol, f"sig-{symbol}") for symbol in symbols]
        )
    }

    def loader(selected_symbols, date):
        return bars.loc[
            bars["symbol"].isin(selected_symbols) & bars["trade_date"].eq(date)
        ].copy()

    seeds = tuple(range(10))
    results, _equity, trades = simulate_portfolio_accounts(
        signals,
        bars_loader=loader,
        calendar=["2022-01-03", "2022-01-04"],
        strategy_ids=("s1_touch_reclaim",),
        policies=(ExitPolicy("one_day", 1),),
        config=PortfolioConfig(starting_cash=100_000, max_positions=1),
        selection_seeds=seeds,
    )
    assert {result["selection_seed"] for result in results} == set(seeds)
    assert set(trades["selection_seed"]) == set(seeds)
    assert trades["symbol"].nunique() > 1
    summary = summarize_selection_seed_results(results)
    assert summary["selection_seeds"] == list(seeds)
    assert summary["account_variant_count"] == 1
    assert summary["accounts"][0]["seed_count"] == len(seeds)


def test_explicitly_noncausal_signal_is_not_executable() -> None:
    bars = _bars({"A": [10.0, 10.0, 10.0]})
    row = _signal("A", "sig-A")
    row["causal_only"] = False
    signals = {"2022-01-03": pd.DataFrame([row])}

    def loader(symbols, date):
        return bars.loc[
            bars["symbol"].isin(symbols) & bars["trade_date"].eq(date)
        ].copy()

    results, _equity, trades = simulate_portfolio_accounts(
        signals,
        bars_loader=loader,
        calendar=["2022-01-03", "2022-01-04"],
        strategy_ids=("s1_touch_reclaim",),
        policies=(ExitPolicy("one_day", 1),),
        config=PortfolioConfig(starting_cash=100_000, max_positions=5),
    )
    assert results[0]["filled_entry_count"] == 0
    assert trades.empty
