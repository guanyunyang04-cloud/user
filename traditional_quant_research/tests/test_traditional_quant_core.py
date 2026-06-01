from __future__ import annotations

import pytest

from traditional_quant_research.backtest import long_only_backtest
from traditional_quant_research.factors import momentum, moving_average, simple_returns, zscore
from traditional_quant_research.metrics import max_drawdown
from traditional_quant_research.portfolio import equal_weight, normalize_long_only, rank_long_short


def test_simple_returns_and_momentum_align_to_prices() -> None:
    prices = [10, 11, 12.1, 10.89]

    assert simple_returns(prices) == pytest.approx([0.1, 0.1, -0.1])
    assert momentum(prices, 2) == [None, None, pytest.approx(0.21), pytest.approx(-0.01)]


def test_moving_average_and_zscore_are_stable() -> None:
    assert moving_average([1, 2, 3, 4], 3) == [None, None, 2.0, 3.0]
    assert zscore([5, 5, 5]) == [0.0, 0.0, 0.0]


def test_portfolio_baselines_sum_to_expected_exposure() -> None:
    assert equal_weight(["A", "B", "A"]) == {"A": 0.5, "B": 0.5}

    long_only = normalize_long_only({"A": 2, "B": -1, "C": 2})
    assert long_only == {"A": 0.5, "B": 0.0, "C": 0.5}

    long_short = rank_long_short({"A": 1, "B": 2, "C": 3, "D": 4}, leg_size=1)
    assert long_short == {"A": -0.5, "D": 0.5}
    assert sum(long_short.values()) == 0.0


def test_long_only_backtest_accounts_for_cash_and_trade_costs() -> None:
    result = long_only_backtest([0.01, -0.02, 0.03], [True, False, True], fee_bps=10)

    assert result.returns == pytest.approx([0.009, -0.001, 0.029])
    assert len(result.equity_curve) == 3
    assert result.equity_curve[-1] > 1.0
    assert result.max_drawdown == max_drawdown(result.returns)
