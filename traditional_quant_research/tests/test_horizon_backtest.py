from __future__ import annotations

import pandas as pd
import pytest

from traditional_quant_research.horizon_backtest import (
    horizon_aligned_top_n_backtest,
    horizon_periods_per_year,
    period_horizon_summary,
    select_buffered_top_n,
    yearly_horizon_summary,
)


def _price_panel() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=12, freq="B")
    rows = []
    prices = {
        "A": [10.0, 10.2, 10.5, 10.4, 10.8, 11.0, 11.1, 11.2, 11.4, 11.3, 11.6, 11.7],
        "B": [20.0, 20.1, 20.0, 20.3, 20.2, 20.5, 20.7, 20.6, 20.8, 21.0, 21.1, 21.2],
        "C": [30.0, 29.8, 30.2, 30.4, 30.5, 30.7, 30.6, 30.9, 31.0, 31.2, 31.1, 31.3],
    }
    for code, closes in prices.items():
        for index, (date, close) in enumerate(zip(dates, closes)):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open": close - 0.1,
                    "close": close,
                    "score": 10.0 - index if code == "A" else index * 3.0 if code == "B" else 1.0,
                    "is_tradeable": True,
                }
            )
    return pd.DataFrame(rows)


def test_horizon_aligned_top_n_matches_1d_label_semantics() -> None:
    frame = _price_panel()

    result = horizon_aligned_top_n_backtest(
        frame,
        "score",
        horizon=1,
        top_n=1,
        fee_bps=0,
        rebalance_frequency="daily",
    )

    first = result.trades.iloc[0]
    expected = 10.2 / (10.2 - 0.1) - 1.0
    assert first["signal_date"] == pd.Timestamp("2026-01-02")
    assert first["entry_date"] == pd.Timestamp("2026-01-05")
    assert first["exit_date"] == pd.Timestamp("2026-01-05")
    assert first["codes"] == "A"
    assert first["gross_return"] == pytest.approx(expected)
    assert result.summary["periods_per_year"] == pytest.approx(252.0)


def test_horizon_aligned_top_n_skips_overlapping_multi_day_baskets() -> None:
    frame = _price_panel()

    result = horizon_aligned_top_n_backtest(
        frame,
        "score",
        horizon=3,
        top_n=1,
        fee_bps=0,
        rebalance_frequency="daily",
    )

    signal_dates = result.trades["signal_date"].tolist()
    assert signal_dates == [
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-07"),
        pd.Timestamp("2026-01-12"),
    ]
    assert result.summary["periods_per_year"] == pytest.approx(84.0)


def test_horizon_aligned_top_n_accounts_for_turnover_costs() -> None:
    frame = _price_panel()

    result = horizon_aligned_top_n_backtest(
        frame,
        "score",
        horizon=3,
        top_n=1,
        fee_bps=10,
        rebalance_frequency="daily",
    )

    assert result.trades["turnover"].tolist()[:2] == [1.0, 2.0]
    assert result.trades["net_return"].iloc[0] == pytest.approx(result.trades["gross_return"].iloc[0] - 0.001)
    assert result.trades["net_return"].iloc[1] == pytest.approx(result.trades["gross_return"].iloc[1] - 0.002)


def test_horizon_backtest_scales_exposure_with_capital_column() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "open": 10.0, "close": 10.0, "score": 3.0, "capital": 0.5, "is_tradeable": True},
            {"date": "2026-01-02", "code": "B", "open": 20.0, "close": 20.0, "score": 2.0, "capital": 0.5, "is_tradeable": True},
            {"date": "2026-01-05", "code": "A", "open": 10.0, "close": 11.0, "score": 1.0, "capital": 1.0, "is_tradeable": True},
            {"date": "2026-01-05", "code": "B", "open": 20.0, "close": 22.0, "score": 1.0, "capital": 1.0, "is_tradeable": True},
        ]
    )

    result = horizon_aligned_top_n_backtest(
        frame,
        "score",
        horizon=1,
        top_n=2,
        fee_bps=10,
        rebalance_frequency="daily",
        capital_col="capital",
    )

    first = result.trades.iloc[0]
    assert first["capital_scale"] == pytest.approx(0.5)
    assert first["gross_return"] == pytest.approx(0.05)
    assert first["turnover"] == pytest.approx(0.5)
    assert first["cost"] == pytest.approx(0.0005)
    assert first["net_return"] == pytest.approx(0.0495)
    assert result.summary["mean_capital_scale"] == pytest.approx(0.5)


def test_horizon_backtest_rejects_invalid_capital_column() -> None:
    frame = _price_panel()
    frame["capital"] = 1.2

    with pytest.raises(ValueError, match="capital column must be between 0 and 1"):
        horizon_aligned_top_n_backtest(
            frame,
            "score",
            horizon=1,
            top_n=1,
            fee_bps=0,
            capital_col="capital",
        )


def test_horizon_backtest_requires_constant_capital_within_signal_date() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "open": 10.0, "close": 10.0, "score": 3.0, "capital": 1.0, "is_tradeable": True},
            {"date": "2026-01-02", "code": "B", "open": 20.0, "close": 20.0, "score": 2.0, "capital": 0.5, "is_tradeable": True},
            {"date": "2026-01-05", "code": "A", "open": 10.0, "close": 11.0, "score": 1.0, "capital": 1.0, "is_tradeable": True},
            {"date": "2026-01-05", "code": "B", "open": 20.0, "close": 22.0, "score": 1.0, "capital": 1.0, "is_tradeable": True},
        ]
    )

    with pytest.raises(ValueError, match="capital column must be constant within each signal date"):
        horizon_aligned_top_n_backtest(
            frame,
            "score",
            horizon=1,
            top_n=2,
            fee_bps=0,
            rebalance_frequency="daily",
            capital_col="capital",
        )


def test_buffered_selection_keeps_prior_holdings_inside_rank_buffer() -> None:
    group = pd.DataFrame(
        [
            {"code": "A", "score": 10.0},
            {"code": "B", "score": 9.0},
            {"code": "C", "score": 8.0},
            {"code": "D", "score": 7.0},
        ]
    )

    selected = select_buffered_top_n(
        group,
        "score",
        code_col="code",
        top_n=2,
        previous_codes={"C"},
        buffer_multiplier=1.5,
    )

    assert selected["code"].tolist() == ["C", "A"]


def test_buffered_selection_respects_group_cap() -> None:
    group = pd.DataFrame(
        [
            {"code": "A", "score": 10.0, "industry": "Bank"},
            {"code": "B", "score": 9.0, "industry": "Bank"},
            {"code": "C", "score": 8.0, "industry": "Tech"},
            {"code": "D", "score": 7.0, "industry": "Tech"},
            {"code": "E", "score": 6.0, "industry": "RealEstate"},
        ]
    )

    selected = select_buffered_top_n(
        group,
        "score",
        code_col="code",
        top_n=3,
        previous_codes={"B"},
        buffer_multiplier=2.0,
        group_col="industry",
        max_group_weight=0.34,
    )

    assert selected["code"].tolist() == ["B", "C", "E"]
    assert selected["industry"].value_counts().max() == 1


def test_buffered_selection_uses_portfolio_exposure_penalty() -> None:
    group = pd.DataFrame(
        [
            {"code": "A", "score": 10.0, "style_z": 5.0},
            {"code": "B", "score": 9.0, "style_z": 5.0},
            {"code": "C", "score": 8.0, "style_z": -5.0},
            {"code": "D", "score": 7.0, "style_z": -5.0},
        ]
    )

    unconstrained = select_buffered_top_n(
        group,
        "score",
        code_col="code",
        top_n=2,
        previous_codes=set(),
        buffer_multiplier=1.0,
    )
    penalized = select_buffered_top_n(
        group,
        "score",
        code_col="code",
        top_n=2,
        previous_codes=set(),
        buffer_multiplier=1.0,
        exposure_penalty_cols=("style_z",),
        exposure_penalty_strength=1.0,
    )

    assert unconstrained["code"].tolist() == ["A", "B"]
    assert penalized["code"].tolist() == ["A", "C"]
    assert penalized["style_z"].mean() == pytest.approx(0.0)


def test_horizon_backtest_respects_group_cap() -> None:
    dates = pd.date_range("2026-01-02", periods=3, freq="B")
    rows = []
    specs = [
        ("A", "Bank", 10.0),
        ("B", "Bank", 9.0),
        ("C", "Tech", 8.0),
        ("D", "RealEstate", 7.0),
    ]
    for date_index, date in enumerate(dates):
        for code, industry, score in specs:
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "industry": industry,
                    "open": 10.0 + date_index,
                    "close": 10.5 + date_index,
                    "score": score,
                    "is_tradeable": True,
                }
            )
    frame = pd.DataFrame(rows)

    result = horizon_aligned_top_n_backtest(
        frame,
        "score",
        horizon=1,
        top_n=3,
        fee_bps=0,
        rebalance_frequency="daily",
        group_col="industry",
        max_group_weight=0.34,
    )

    assert result.trades["codes"].iloc[0] == "A,C,D"
    assert result.summary["group_col"] == "industry"
    assert result.summary["max_group_weight"] == pytest.approx(0.34)


def test_horizon_backtest_applies_portfolio_exposure_penalty() -> None:
    dates = pd.date_range("2026-01-02", periods=3, freq="B")
    rows = []
    specs = [
        ("A", 10.0, 5.0),
        ("B", 9.0, 5.0),
        ("C", 8.0, -5.0),
        ("D", 7.0, -5.0),
    ]
    for date_index, date in enumerate(dates):
        for code, score, style_z in specs:
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open": 10.0 + date_index,
                    "close": 10.5 + date_index,
                    "score": score,
                    "style_z": style_z,
                    "is_tradeable": True,
                }
            )
    frame = pd.DataFrame(rows)

    result = horizon_aligned_top_n_backtest(
        frame,
        "score",
        horizon=1,
        top_n=2,
        fee_bps=0,
        rebalance_frequency="daily",
        exposure_penalty_cols=("style_z",),
        exposure_penalty_strength=1.0,
    )

    assert result.trades["codes"].iloc[0] == "A,C"
    assert result.summary["exposure_penalty_cols"] == "style_z"
    assert result.summary["exposure_penalty_strength"] == pytest.approx(1.0)


def test_horizon_backtest_buffer_reduces_turnover_when_prior_holding_survives() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "open": 10.0, "close": 10.0, "score": 3.0, "is_tradeable": True},
            {"date": "2026-01-02", "code": "B", "open": 20.0, "close": 20.0, "score": 2.0, "is_tradeable": True},
            {"date": "2026-01-02", "code": "C", "open": 30.0, "close": 30.0, "score": 1.0, "is_tradeable": True},
            {"date": "2026-01-05", "code": "A", "open": 10.0, "close": 10.2, "score": 1.0, "is_tradeable": True},
            {"date": "2026-01-05", "code": "B", "open": 20.0, "close": 20.2, "score": 3.0, "is_tradeable": True},
            {"date": "2026-01-05", "code": "C", "open": 30.0, "close": 30.2, "score": 2.0, "is_tradeable": True},
            {"date": "2026-01-06", "code": "A", "open": 10.2, "close": 10.4, "score": 1.0, "is_tradeable": True},
            {"date": "2026-01-06", "code": "B", "open": 20.2, "close": 20.4, "score": 3.0, "is_tradeable": True},
            {"date": "2026-01-06", "code": "C", "open": 30.2, "close": 30.4, "score": 2.0, "is_tradeable": True},
            {"date": "2026-01-07", "code": "A", "open": 10.4, "close": 10.6, "score": 1.0, "is_tradeable": True},
            {"date": "2026-01-07", "code": "B", "open": 20.4, "close": 20.6, "score": 3.0, "is_tradeable": True},
            {"date": "2026-01-07", "code": "C", "open": 30.4, "close": 30.6, "score": 2.0, "is_tradeable": True},
        ]
    )

    unbuffered = horizon_aligned_top_n_backtest(
        frame,
        "score",
        horizon=1,
        top_n=1,
        fee_bps=0,
        rebalance_frequency="daily",
        buffer_multiplier=1.0,
    )
    buffered = horizon_aligned_top_n_backtest(
        frame,
        "score",
        horizon=1,
        top_n=1,
        fee_bps=0,
        rebalance_frequency="daily",
        buffer_multiplier=3.0,
    )

    assert unbuffered.trades["codes"].tolist()[:2] == ["A", "B"]
    assert buffered.trades["codes"].tolist()[:2] == ["A", "A"]
    assert buffered.trades["turnover"].iloc[1] < unbuffered.trades["turnover"].iloc[1]


def test_horizon_backtest_filters_non_tradeable_rows_before_entry_exit() -> None:
    frame = _price_panel()
    frame.loc[(frame["code"] == "A") & (frame["date"] == pd.Timestamp("2026-01-05")), "is_tradeable"] = False

    result = horizon_aligned_top_n_backtest(
        frame,
        "score",
        horizon=1,
        top_n=1,
        fee_bps=0,
        rebalance_frequency="daily",
    )

    first = result.trades.iloc[0]
    assert first["entry_date"] == pd.Timestamp("2026-01-06")
    assert first["exit_date"] == pd.Timestamp("2026-01-06")


def test_execution_constraints_block_limit_up_entry_without_replacement_cash() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "open": 10.0, "close": 10.0, "score": 3.0, "is_tradeable": True},
            {"date": "2026-01-02", "code": "B", "open": 20.0, "close": 20.0, "score": 2.0, "is_tradeable": True},
            {"date": "2026-01-05", "code": "A", "open": 11.0, "close": 11.0, "score": 1.0, "is_tradeable": True},
            {"date": "2026-01-05", "code": "B", "open": 20.0, "close": 20.2, "score": 1.0, "is_tradeable": True},
        ]
    )

    result = horizon_aligned_top_n_backtest(
        frame,
        "score",
        horizon=1,
        top_n=2,
        fee_bps=0,
        rebalance_frequency="daily",
        execution_constraints=True,
        limit_threshold=0.095,
    )

    first = result.trades.iloc[0]
    assert first["codes"] == "B"
    assert first["holdings"] == 1
    assert first["requested_holdings"] == 2
    assert first["blocked_entry_count"] == 1
    assert first["entry_limit_up_count"] == 1
    assert first["gross_return"] == pytest.approx((20.2 / 20.0 - 1.0) / 2.0)
    assert result.summary["execution_constraints"] is True
    assert result.summary["blocked_entry_count"] == 1


def test_execution_constraints_delay_limit_down_exit() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "open": 10.0, "close": 10.0, "score": 3.0, "is_tradeable": True},
            {"date": "2026-01-05", "code": "A", "open": 10.0, "close": 10.0, "score": 2.0, "is_tradeable": True},
            {"date": "2026-01-06", "code": "A", "open": 9.2, "close": 9.0, "score": 1.0, "is_tradeable": True},
            {"date": "2026-01-07", "code": "A", "open": 9.4, "close": 9.5, "score": 1.0, "is_tradeable": True},
        ]
    )

    result = horizon_aligned_top_n_backtest(
        frame,
        "score",
        horizon=2,
        top_n=1,
        fee_bps=0,
        rebalance_frequency="daily",
        execution_constraints=True,
        limit_threshold=0.095,
    )

    first = result.trades.iloc[0]
    assert first["entry_date"] == pd.Timestamp("2026-01-05")
    assert first["exit_date"] == pd.Timestamp("2026-01-07")
    assert first["exit_delayed_count"] == 1
    assert first["exit_limit_down_count"] == 1
    assert first["gross_return"] == pytest.approx(9.5 / 10.0 - 1.0)


def test_horizon_backtest_rejects_invalid_inputs() -> None:
    frame = _price_panel()

    with pytest.raises(ValueError, match="horizon must be positive"):
        horizon_aligned_top_n_backtest(frame, "score", horizon=0, top_n=1, fee_bps=0)
    with pytest.raises(ValueError, match="top_n must be positive"):
        horizon_aligned_top_n_backtest(frame, "score", horizon=1, top_n=0, fee_bps=0)
    with pytest.raises(ValueError, match="buffer_multiplier must be at least 1.0"):
        horizon_aligned_top_n_backtest(frame, "score", horizon=1, top_n=1, fee_bps=0, buffer_multiplier=0.5)
    with pytest.raises(ValueError, match="limit_threshold must be positive"):
        horizon_aligned_top_n_backtest(frame, "score", horizon=1, top_n=1, fee_bps=0, limit_threshold=0)
    with pytest.raises(ValueError, match="group_col is required"):
        horizon_aligned_top_n_backtest(frame, "score", horizon=1, top_n=1, fee_bps=0, max_group_weight=0.5)
    with pytest.raises(ValueError, match="max_group_weight must be between 0 and 1"):
        horizon_aligned_top_n_backtest(frame.assign(industry="A"), "score", horizon=1, top_n=1, fee_bps=0, group_col="industry", max_group_weight=1.5)
    with pytest.raises(ValueError, match="exposure_penalty_strength must be non-negative"):
        horizon_aligned_top_n_backtest(frame.assign(style_z=0.0), "score", horizon=1, top_n=1, fee_bps=0, exposure_penalty_cols=("style_z",), exposure_penalty_strength=-1.0)
    with pytest.raises(ValueError, match="missing required columns"):
        horizon_aligned_top_n_backtest(frame, "score", horizon=1, top_n=1, fee_bps=0, exposure_penalty_cols=("style_z",), exposure_penalty_strength=1.0)
    with pytest.raises(ValueError, match="missing required columns"):
        horizon_aligned_top_n_backtest(frame.drop(columns=["open"]), "score", horizon=1, top_n=1, fee_bps=0)


def test_horizon_periods_per_year_uses_holding_period_for_daily_non_overlap() -> None:
    assert horizon_periods_per_year(horizon=5, rebalance_frequency="daily", non_overlapping=True) == pytest.approx(50.4)
    assert horizon_periods_per_year(horizon=5, rebalance_frequency="weekly", non_overlapping=True) == pytest.approx(50.4)
    assert horizon_periods_per_year(horizon=5, rebalance_frequency="weekly", non_overlapping=False) == pytest.approx(52.0)


def test_yearly_horizon_summary_groups_by_exit_year() -> None:
    trades = pd.DataFrame(
        [
            {"exit_date": "2025-12-31", "net_return": 0.02, "gross_return": 0.021, "turnover": 1.0},
            {"exit_date": "2026-01-08", "net_return": -0.01, "gross_return": -0.009, "turnover": 1.5},
            {"exit_date": "2026-01-15", "net_return": 0.03, "gross_return": 0.031, "turnover": 0.5},
        ]
    )

    summary = yearly_horizon_summary(
        trades,
        horizon=5,
        rebalance_frequency="daily",
        top_n=100,
        fee_bps=10,
        non_overlapping=True,
        buffer_multiplier=1.5,
        signal_col="score",
    )

    assert summary["year"].tolist() == [2025, 2026]
    assert summary["signal"].tolist() == ["score", "score"]
    assert summary["buffer_multiplier"].tolist() == [1.5, 1.5]
    assert summary.loc[summary["year"] == 2025, "periods"].iloc[0] == 1
    assert summary.loc[summary["year"] == 2026, "periods"].iloc[0] == 2
    assert summary.loc[summary["year"] == 2026, "mean_turnover"].iloc[0] == pytest.approx(1.0)


def test_period_horizon_summary_groups_by_exit_month_and_quarter() -> None:
    trades = pd.DataFrame(
        [
            {"exit_date": "2026-01-08", "net_return": 0.02, "gross_return": 0.021, "turnover": 1.0},
            {"exit_date": "2026-01-15", "net_return": -0.01, "gross_return": -0.009, "turnover": 1.5},
            {"exit_date": "2026-04-08", "net_return": 0.03, "gross_return": 0.031, "turnover": 0.5},
        ]
    )

    monthly = period_horizon_summary(
        trades,
        period="M",
        horizon=5,
        rebalance_frequency="weekly",
        top_n=100,
        fee_bps=10,
        non_overlapping=True,
        buffer_multiplier=2.0,
        signal_col="score",
    )
    quarterly = period_horizon_summary(
        trades,
        period="Q",
        horizon=5,
        rebalance_frequency="weekly",
        top_n=100,
        fee_bps=10,
        non_overlapping=True,
        buffer_multiplier=2.0,
        signal_col="score",
    )

    assert monthly["period"].tolist() == ["2026-01", "2026-04"]
    assert monthly["period_type"].tolist() == ["monthly", "monthly"]
    assert monthly.loc[monthly["period"] == "2026-01", "periods"].iloc[0] == 2
    assert monthly.loc[monthly["period"] == "2026-01", "mean_turnover"].iloc[0] == pytest.approx(1.25)
    assert quarterly["period"].tolist() == ["2026Q1", "2026Q2"]
    assert quarterly["period_type"].tolist() == ["quarterly", "quarterly"]
