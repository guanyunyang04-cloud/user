from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from quantlab.research.minute_v2.contracts import (
    EXPECTED_DECISION_BARS,
    MODEL_FEATURE_COLUMNS,
    is_decision_bar,
)
from quantlab.research.minute_v2.features import build_feature_frame
from quantlab.research.minute_v2.labels import build_label_frame
from quantlab.research.minute_v2.mining import mine_formula_features
from quantlab.research.minute_v2.models import fit_ridge, rule_score
from quantlab.research.minute_v2.replay import EventReplayConfig, replay_events
from quantlab.research.minute_v2.sampling import build_event_frame
from quantlab.research.minute_v2.training import _bounded_group_sample


def _times() -> list[str]:
    morning = pd.date_range("2000-01-01 09:31", "2000-01-01 11:30", freq="min")
    afternoon = pd.date_range("2000-01-01 13:01", "2000-01-01 15:00", freq="min")
    return [*morning.strftime("%H%M00000"), *afternoon.strftime("%H%M00000")]


def _bars(
    *,
    dates: tuple[str, ...] = ("2022-06-01",),
    symbols: tuple[str, ...] = ("600000.SH", "000001.SZ"),
) -> pd.DataFrame:
    rows = []
    for date_index, trade_date in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            for minute_index, bar_time in enumerate(_times()):
                price = 10.0 + symbol_index + date_index * 0.1 + minute_index * 0.001
                volume = 1000.0 + minute_index + symbol_index * 10
                rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": trade_date,
                        "bar_time": bar_time,
                        "open": price,
                        "high": price + 0.01,
                        "low": price - 0.01,
                        "close": price + 0.005,
                        "volume": volume,
                        "amount": volume * (price + 0.002),
                    }
                )
    return pd.DataFrame(rows)


def _stock_days(
    *,
    trade_date: str = "2022-06-01",
    symbols: tuple[str, ...] = ("600000.SH", "000001.SZ"),
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "industry_name": f"industry_{symbol_index}",
                "adjust_factor": 1.0,
                "previous_close": 9.9 + symbol_index,
                "auction_price": 10.0 + symbol_index,
                "auction_amount": 1_000_000.0,
                "previous_return_1d": 0.01,
                "previous_return_5d": 0.02,
                "previous_return_20d": 0.03,
                "previous_volatility_20d": 0.02,
                "previous_amount_20d": 200_000_000.0,
                "daily_liquidity_rank": 0.8,
                "exclude_open": False,
                "exclude_high": symbol_index == 1,
                "exclude_low": False,
                "exclude_close": False,
            }
            for symbol_index, symbol in enumerate(symbols)
        ]
    )


def test_decision_grid_excludes_lunch_and_closing_auction() -> None:
    selected = [value for value in _times() if is_decision_bar(value)]
    assert len(selected) == EXPECTED_DECISION_BARS
    assert "112900000" in selected
    assert "113000000" not in selected
    assert "145500000" in selected
    assert "145600000" not in selected


def test_features_are_causal_and_field_masks_are_specific() -> None:
    bars = _bars()
    stock_days = _stock_days()
    original = build_feature_frame(bars, stock_days)
    mutated_bars = bars.copy()
    future = mutated_bars["bar_time"] > "100000000"
    mutated_bars.loc[future, ["open", "high", "low", "close"]] *= 8.0
    mutated_bars.loc[future, ["volume", "amount"]] *= 5.0
    mutated = build_feature_frame(mutated_bars, stock_days)
    columns = ["symbol", "trade_date", "bar_time", *MODEL_FEATURE_COLUMNS]
    before = original.loc[original["bar_time"] <= "100000000", columns].reset_index(drop=True)
    after = mutated.loc[mutated["bar_time"] <= "100000000", columns].reset_index(drop=True)
    assert_frame_equal(before, after, check_exact=True)
    masked = original.loc[original["symbol"] == "000001.SZ"]
    assert masked["cumulative_range"].isna().all()
    assert masked["breakout_20m"].isna().all()
    assert masked["return_5m"].notna().sum() > 0


def test_event_sampling_is_deterministic_and_contains_no_outcomes() -> None:
    features = build_feature_frame(_bars(), _stock_days())
    first = build_event_frame(features)
    second = build_event_frame(features)
    assert_frame_equal(first, second)
    assert len(first) > 0
    assert first["event_mask"].gt(0).all()
    assert not any(column.startswith(("label_", "entry_", "actual_exit")) for column in first.columns)


def test_labels_use_next_bar_and_next_market_day() -> None:
    bars = _bars(dates=("2022-06-01", "2022-06-02"), symbols=("600000.SH",))
    features = build_feature_frame(
        bars.loc[bars["trade_date"] == "2022-06-01"],
        _stock_days(symbols=("600000.SH",)),
    )
    events = features.loc[features["bar_time"] == "093500000"].copy()
    label_stock_days = pd.DataFrame(
        [
            {
                "symbol": "600000.SH",
                "trade_date": "2022-06-02",
                "adjust_factor": 1.0,
                "previous_close": 10.2,
                "is_suspended": False,
                "is_delisted": False,
                "exclude_high": False,
                "exclude_low": False,
            }
        ]
    )
    calendar = pd.DataFrame(
        [
            {"trade_date": "2022-06-01", "calendar_index": 1, "next_trade_date": "2022-06-02"},
            {"trade_date": "2022-06-02", "calendar_index": 2, "next_trade_date": None},
        ]
    )
    labels = build_label_frame(
        events,
        bars.loc[bars["trade_date"] == "2022-06-01"],
        bars,
        label_stock_days,
        calendar,
    )
    row = labels.iloc[0]
    assert row["entry_bar_time"] == "093600000"
    assert row["planned_exit_date"] == "2022-06-02"
    assert row["actual_exit_date"] == "2022-06-02"
    assert bool(row["entry_executable"])
    assert bool(row["label_observed"])
    assert np.isfinite(row["label_net_return"])
    assert row["label_net_return"] < row["label_gross_return"]


def test_replay_does_not_replace_an_unfilled_top_rank() -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "600000.SH",
                "trade_date": "2022-06-01",
                "bar_time": "093500000",
                "score": 2.0,
                "planned_exit_date": "2022-06-02",
                "entry_bar_time": "093600000",
                "entry_price": 10.0,
                "entry_amount": 10_000_000.0,
                "entry_executable": False,
                "actual_exit_date": None,
                "exit_amount": np.nan,
                "label_net_return": np.nan,
                "label_observed": False,
            },
            {
                "symbol": "000001.SZ",
                "trade_date": "2022-06-01",
                "bar_time": "093500000",
                "score": 1.0,
                "planned_exit_date": "2022-06-02",
                "entry_bar_time": "093600000",
                "entry_price": 11.0,
                "entry_amount": 10_000_000.0,
                "entry_executable": True,
                "actual_exit_date": "2022-06-02",
                "exit_amount": 10_000_000.0,
                "label_net_return": 0.02,
                "label_observed": True,
            },
        ]
    )
    result, _, trades = replay_events(
        frame,
        replay_config=EventReplayConfig(top_k_per_minute=1),
    )
    assert result["selected_event_count"] == 1
    assert result["unfilled_selection_count"] == 1
    assert trades.empty


def test_rule_ridge_and_formula_mining_have_small_deterministic_contracts() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for day in range(20):
        period = "train" if day < 14 else "validation"
        for minute in ("093500000", "100000000"):
            for symbol_index in range(10):
                signal = rng.normal()
                row = {
                    "symbol": f"{600000 + symbol_index}.SH",
                    "trade_date": f"2022-06-{day + 1:02d}",
                    "bar_time": minute,
                    "period": period,
                    "return_5m": signal,
                    "return_1m": signal * 0.2,
                    "label_net_return": signal * 0.01 + rng.normal(scale=0.001),
                }
                for feature in MODEL_FEATURE_COLUMNS:
                    row.setdefault(feature, rng.normal())
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame["score"] = rule_score(frame)
    assert frame["score"].notna().all()
    ridge = fit_ridge(
        frame.loc[frame["period"] == "train"],
        feature_names=("return_1m", "return_5m"),
        alpha=1.0,
    )
    prediction = ridge.predict(frame.loc[frame["period"] == "validation"])
    assert np.corrcoef(prediction, frame.loc[frame["period"] == "validation", "label_net_return"])[0, 1] > 0.8
    mining = mine_formula_features(
        frame.loc[frame["period"] == "train"],
        frame.loc[frame["period"] == "validation"],
        seed_features=("return_1m", "return_5m"),
        maximum_candidates=12,
        maximum_selected=3,
        minimum_coverage=0.9,
    )
    assert mining["candidate_count"] == 9
    assert mining["selected_count"] > 0


def test_bounded_training_sample_keeps_complete_cross_sections() -> None:
    rows = []
    for day in range(10):
        for minute in ("093500000", "100000000"):
            for symbol in range(20):
                rows.append(
                    {
                        "trade_date": f"2022-06-{day + 1:02d}",
                        "bar_time": minute,
                        "symbol": f"{600000 + symbol}.SH",
                    }
                )
    frame = pd.DataFrame(rows)
    selected = _bounded_group_sample(frame, maximum_rows=125)
    sizes = selected.groupby(["trade_date", "bar_time"]).size()
    assert not selected.empty
    assert sizes.eq(20).all()
    assert len(selected) <= 125
