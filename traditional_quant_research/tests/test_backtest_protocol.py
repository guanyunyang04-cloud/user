from __future__ import annotations

import pandas as pd
import pytest

from traditional_quant_research.backtest_protocol import select_rebalance_dates, top_n_rebalance_backtest


def test_select_rebalance_dates_daily_weekly_monthly() -> None:
    dates = pd.date_range("2026-01-01", "2026-01-20", freq="B")

    daily = select_rebalance_dates(dates, "daily")
    weekly = select_rebalance_dates(dates, "weekly")
    monthly = select_rebalance_dates(dates, "monthly")

    assert len(daily) == len(dates)
    assert weekly.tolist() == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-09"), pd.Timestamp("2026-01-16"), pd.Timestamp("2026-01-20")]
    assert monthly.tolist() == [pd.Timestamp("2026-01-20")]


def test_top_n_rebalance_backtest_accounts_for_turnover_costs() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "score": 3.0, "fwd_ret_1d": 0.03},
            {"date": "2026-01-02", "code": "B", "score": 2.0, "fwd_ret_1d": 0.02},
            {"date": "2026-01-02", "code": "C", "score": 1.0, "fwd_ret_1d": 0.01},
            {"date": "2026-01-05", "code": "A", "score": 1.0, "fwd_ret_1d": 0.00},
            {"date": "2026-01-05", "code": "B", "score": 2.0, "fwd_ret_1d": 0.01},
            {"date": "2026-01-05", "code": "C", "score": 3.0, "fwd_ret_1d": 0.02},
        ]
    )

    result = top_n_rebalance_backtest(frame, "score", "fwd_ret_1d", top_n=1, fee_bps=10)

    assert result.daily_returns["holdings"].tolist() == [1, 1]
    assert result.daily_returns["turnover"].tolist() == [1.0, 2.0]
    assert result.daily_returns["net_return"].iloc[0] == pytest.approx(0.03 - 0.001)
    assert result.daily_returns["net_return"].iloc[1] == pytest.approx(0.02 - 0.002)
