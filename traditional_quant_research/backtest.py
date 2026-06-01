"""Minimal backtest utilities for validating research hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .metrics import annualized_return, max_drawdown, sharpe_ratio, volatility


@dataclass(frozen=True)
class BacktestResult:
    returns: list[float]
    equity_curve: list[float]
    annualized_return: float
    volatility: float
    sharpe: float
    max_drawdown: float


def long_only_backtest(
    asset_returns: Iterable[float],
    signals: Iterable[bool],
    *,
    fee_bps: float = 0.0,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Apply a binary long/cash signal to one return stream."""
    returns = [float(value) for value in asset_returns]
    signal_values = [bool(value) for value in signals]
    if len(returns) != len(signal_values):
        raise ValueError("asset_returns and signals must have the same length")

    strategy_returns: list[float] = []
    equity_curve: list[float] = []
    equity = 1.0
    previous_signal = False
    fee = fee_bps / 10000.0

    for asset_return, signal in zip(returns, signal_values):
        trade_cost = fee if signal != previous_signal else 0.0
        period_return = asset_return if signal else 0.0
        net_return = period_return - trade_cost
        equity *= 1.0 + net_return
        strategy_returns.append(net_return)
        equity_curve.append(equity)
        previous_signal = signal

    return BacktestResult(
        returns=strategy_returns,
        equity_curve=equity_curve,
        annualized_return=annualized_return(strategy_returns, periods_per_year),
        volatility=volatility(strategy_returns, periods_per_year),
        sharpe=sharpe_ratio(strategy_returns, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(strategy_returns),
    )
