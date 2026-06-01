"""Small dependency-free factor helpers for first-pass research."""

from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Iterable


def _as_float_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def simple_returns(prices: Iterable[float]) -> list[float]:
    """Return arithmetic period returns from a price sequence."""
    series = _as_float_list(prices)
    if len(series) < 2:
        return []
    returns: list[float] = []
    for previous, current in zip(series, series[1:]):
        if previous == 0:
            raise ValueError("price series contains zero before a return calculation")
        returns.append(current / previous - 1.0)
    return returns


def momentum(prices: Iterable[float], window: int) -> list[float | None]:
    """Return trailing window price momentum aligned to the input series."""
    if window <= 0:
        raise ValueError("window must be positive")
    series = _as_float_list(prices)
    values: list[float | None] = [None] * len(series)
    for index in range(window, len(series)):
        base = series[index - window]
        if base == 0:
            raise ValueError("price series contains zero before a momentum calculation")
        values[index] = series[index] / base - 1.0
    return values


def moving_average(values: Iterable[float], window: int) -> list[float | None]:
    """Return trailing simple moving average aligned to the input series."""
    if window <= 0:
        raise ValueError("window must be positive")
    series = _as_float_list(values)
    averages: list[float | None] = [None] * len(series)
    rolling_sum = 0.0
    for index, value in enumerate(series):
        rolling_sum += value
        if index >= window:
            rolling_sum -= series[index - window]
        if index + 1 >= window:
            averages[index] = rolling_sum / window
    return averages


def zscore(values: Iterable[float]) -> list[float]:
    """Return population z-scores for a cross-section or time slice."""
    series = _as_float_list(values)
    if not series:
        return []
    center = mean(series)
    scale = pstdev(series)
    if scale == 0:
        return [0.0 for _ in series]
    return [(value - center) / scale for value in series]


def annualized_volatility(returns: Iterable[float], periods_per_year: int = 252) -> float:
    """Convenience volatility helper kept with factor probes."""
    series = _as_float_list(returns)
    if not series:
        return 0.0
    return pstdev(series) * sqrt(periods_per_year)
