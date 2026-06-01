"""Risk and performance metrics for research-grade sanity checks."""

from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Iterable


def _as_float_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def annualized_return(returns: Iterable[float], periods_per_year: int = 252) -> float:
    """Return geometric annualized return from periodic returns."""
    series = _as_float_list(returns)
    if not series:
        return 0.0
    equity = 1.0
    for value in series:
        equity *= 1.0 + value
    if equity <= 0:
        return -1.0
    return equity ** (periods_per_year / len(series)) - 1.0


def volatility(returns: Iterable[float], periods_per_year: int = 252) -> float:
    """Return annualized population volatility."""
    series = _as_float_list(returns)
    if not series:
        return 0.0
    return pstdev(series) * sqrt(periods_per_year)


def sharpe_ratio(returns: Iterable[float], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Return annualized Sharpe ratio using periodic returns."""
    series = _as_float_list(returns)
    if not series:
        return 0.0
    periodic_rf = risk_free_rate / periods_per_year
    excess = [value - periodic_rf for value in series]
    scale = pstdev(excess)
    if scale == 0:
        return 0.0
    return mean(excess) / scale * sqrt(periods_per_year)


def max_drawdown(returns: Iterable[float]) -> float:
    """Return the worst peak-to-trough drawdown as a negative number."""
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for value in _as_float_list(returns):
        equity *= 1.0 + value
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0
        worst = min(worst, drawdown)
    return worst
