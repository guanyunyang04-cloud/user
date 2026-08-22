from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.research.minute import _read_bar_frames, build_minute_panel, simulate_minute_account
from quantlab.research.minute_panel import build_minute_dataset


def _bars(
    *,
    dates: tuple[str, ...] = ("2025-01-02", "2025-01-03", "2025-01-06"),
    symbols: tuple[str, ...] = ("600000.SH", "000001.SZ"),
    volumes: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        for date_index, date in enumerate(dates):
            for offset in range(40):
                total_minutes = 9 * 60 + 31 + offset
                hour, minute = divmod(total_minutes, 60)
                price = 10.0 + symbol_index + date_index * 0.1 + offset * 0.001
                volume = float((volumes or {}).get(date, 1000.0 + offset))
                rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": date,
                        "bar_time": f"{hour:02d}{minute:02d}00000",
                        "open": price,
                        "high": price + 0.01,
                        "low": price - 0.01,
                        "close": price + 0.005,
                        "volume": volume,
                        "amount": volume * price,
                    }
                )
    return pd.DataFrame(rows)


def _reference(bars: pd.DataFrame, factors: dict[str, float] | None = None) -> pd.DataFrame:
    result = bars[["symbol", "trade_date"]].drop_duplicates().copy()
    result["adjust_factor"] = result["trade_date"].map(factors or {}).fillna(1.0)
    closes = bars.groupby(["symbol", "trade_date"], sort=False)["close"].last().rename("daily_close")
    result = result.merge(closes.reset_index(), on=["symbol", "trade_date"], validate="one_to_one")
    result["is_st"] = False
    result["is_suspended"] = False
    result["is_delisted"] = False
    return result


def test_minute_panel_uses_next_market_day_and_adjusted_return() -> None:
    bars = _bars()
    panel = build_minute_panel(bars, _reference(bars))
    assert len(panel) == 4
    assert set(panel["signal_date"]) == {"2025-01-02", "2025-01-03"}
    assert (panel["planned_exit_date"] > panel["signal_date"]).all()
    assert panel["label_observed"].all()
    assert np.isfinite(panel["label_gross_return"]).all()


def test_cross_day_label_uses_adjustment_factor() -> None:
    bars = _bars(symbols=("600000.SH",))
    second = bars["trade_date"] == "2025-01-03"
    bars.loc[second, ["open", "high", "low", "close"]] /= 2.0
    bars.loc[second, "amount"] /= 2.0
    factors = {"2025-01-03": 2.0, "2025-01-06": 2.0}
    panel = build_minute_panel(bars, _reference(bars, factors))
    first = panel.loc[panel["signal_date"] == "2025-01-02"].iloc[0]
    expected = first["label_exit_price"] * 2.0 / first["entry_price"] - 1.0
    assert np.isclose(first["label_gross_return"], expected)
    assert first["label_gross_return"] > -0.1


def test_future_one_price_window_does_not_remove_candidate() -> None:
    bars = _bars(symbols=("600000.SH",))
    entry_window = (bars["trade_date"] == "2025-01-02") & (bars["bar_time"] >= "100100000")
    locked = bars.loc[entry_window, "close"].iloc[0]
    bars.loc[entry_window, ["open", "high", "low", "close"]] = locked
    bars.loc[entry_window, "amount"] = bars.loc[entry_window, "volume"] * locked
    panel = build_minute_panel(bars, _reference(bars))
    candidate = panel.loc[panel["signal_date"] == "2025-01-02"].iloc[0]
    assert not candidate["entry_filled"]
    assert candidate["entry_unfilled_reason"] == "one_price"


def test_one_price_exit_is_directional() -> None:
    dates = ("2025-01-02", "2025-01-03", "2025-01-06")
    for price_multiplier, expected_exit_date, expected_reason in (
        (1.05, dates[1], ""),
        (0.95, dates[2], "one_price_down"),
    ):
        bars = _bars(dates=dates, symbols=("600000.SH",))
        previous_close = bars.loc[bars["trade_date"] == dates[0], "close"].iloc[-1]
        exit_window = (
            (bars["trade_date"] == dates[1])
            & (bars["bar_time"] >= "100100000")
            & (bars["bar_time"] <= "101000000")
        )
        locked_price = float(previous_close * price_multiplier)
        bars.loc[exit_window, ["open", "high", "low", "close"]] = locked_price
        bars.loc[exit_window, "amount"] = bars.loc[exit_window, "volume"] * locked_price

        dataset = build_minute_dataset(bars, _reference(bars))
        locked_window = dataset.windows.loc[
            (dataset.windows["symbol"] == "600000.SH")
            & (dataset.windows["trade_date"] == dates[1])
        ].iloc[0]
        assert locked_window["one_price_window"]
        assert locked_window["exit_reason"] == expected_reason
        assert bool(locked_window["exit_executable"]) == (expected_reason == "")

        panel = dataset.panel.copy()
        panel["prediction"] = np.nan
        panel.loc[panel["signal_date"] == dates[0], "prediction"] = 1.0
        result, _, trades, _ = simulate_minute_account(
            panel,
            dataset.windows,
            evaluation_start_date=dates[0],
            evaluation_end_date=dates[0],
            top_k=1,
        )
        assert result["closed_trade_count"] == 1
        assert trades.iloc[0]["exit_date"] == expected_exit_date
        assert bool(trades.iloc[0]["delayed_exit"]) == (expected_exit_date != dates[1])


def test_parquet_reader_enforces_outcome_cutoff(tmp_path) -> None:
    bars = _bars(dates=("2025-12-31", "2026-01-05"), symbols=("600000.SH",))
    parquet = tmp_path / "minutes.parquet"
    bars.to_parquet(parquet, index=False)

    result = _read_bar_frames([parquet], maximum_trade_date="2025-12-31")

    assert set(result["trade_date"]) == {"2025-12-31"}
    assert len(result) == 40


def test_minute_account_is_finite_and_t1() -> None:
    bars = _bars()
    dataset = build_minute_dataset(bars, _reference(bars))
    panel = dataset.panel.copy()
    panel["prediction"] = panel["morning_return"]
    result, equity, trades, fills = simulate_minute_account(
        panel,
        dataset.windows,
        evaluation_start_date="2025-01-02",
        top_k=1,
    )
    assert result["closed_trade_count"] == 2
    assert result["unresolved_position_count"] == 0
    assert np.isfinite(result["ending_equity"])
    assert len(equity) == 3
    assert len(trades) == 2
    assert len(fills) == 2


def test_exit_capacity_causes_partial_delayed_sales() -> None:
    dates = ("2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07")
    bars = _bars(
        dates=dates,
        symbols=("600000.SH",),
        volumes={dates[0]: 3000.0, dates[1]: 1000.0, dates[2]: 1000.0, dates[3]: 1000.0},
    )
    dataset = build_minute_dataset(bars, _reference(bars))
    panel = dataset.panel.copy()
    panel["prediction"] = np.nan
    panel.loc[panel["signal_date"] == dates[0], "prediction"] = 1.0
    result, _, trades, fills = simulate_minute_account(
        panel,
        dataset.windows,
        evaluation_start_date=dates[0],
        evaluation_end_date=dates[0],
        top_k=1,
        maximum_participation_rate=0.01,
    )
    assert result["closed_trade_count"] == 1
    assert result["partial_exit_episode_count"] == 1
    assert result["delayed_exit_episode_count"] == 1
    assert result["unresolved_position_count"] == 0
    assert len(trades) == 1
    assert len(fills) == 3
    assert fills["participation_rate"].max() <= 0.01 + 1.0e-9
