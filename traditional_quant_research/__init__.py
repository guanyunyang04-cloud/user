"""Lightweight primitives for traditional quantitative research."""

from .backtest import BacktestResult, long_only_backtest
from .factors import annualized_volatility, momentum, moving_average, simple_returns, zscore
from .metrics import annualized_return, max_drawdown, sharpe_ratio, volatility
from .portfolio import equal_weight, normalize_long_only, rank_long_short

__all__ = [
    "BacktestResult",
    "annualized_return",
    "annualized_volatility",
    "equal_weight",
    "long_only_backtest",
    "max_drawdown",
    "momentum",
    "moving_average",
    "normalize_long_only",
    "rank_long_short",
    "sharpe_ratio",
    "simple_returns",
    "volatility",
    "zscore",
]
