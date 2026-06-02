"""Portfolio backtest protocol helpers for first-stage strategy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .metrics import annualized_return, max_drawdown, sharpe_ratio, volatility


REBALANCE_PERIODS_PER_YEAR = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
}


@dataclass(frozen=True)
class RebalanceBacktestResult:
    daily_returns: pd.DataFrame
    summary: dict[str, Any]


def select_rebalance_dates(dates: Sequence[pd.Timestamp] | pd.Series, frequency: str) -> pd.DatetimeIndex:
    """Select rebalance signal dates for daily, weekly, or monthly schedules."""

    if frequency not in REBALANCE_PERIODS_PER_YEAR:
        raise ValueError(f"unsupported frequency: {frequency}")
    date_index = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).dropna().unique())).sort_values()
    if frequency == "daily":
        return date_index
    frame = pd.DataFrame({"date": date_index})
    if frequency == "weekly":
        return pd.DatetimeIndex(frame.groupby(frame["date"].dt.to_period("W-FRI"))["date"].max().tolist())
    return pd.DatetimeIndex(frame.groupby(frame["date"].dt.to_period("M"))["date"].max().tolist())


def top_n_rebalance_backtest(
    frame: pd.DataFrame,
    signal_col: str,
    return_col: str,
    *,
    top_n: int,
    fee_bps: float,
    rebalance_frequency: str = "daily",
) -> RebalanceBacktestResult:
    """Equal-weight Top-N long-only backtest on selected rebalance dates."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    required = ["date", "code", signal_col, return_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    clean = frame.loc[:, required].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return RebalanceBacktestResult(daily_returns=pd.DataFrame(), summary=summarize_rebalance_returns([], rebalance_frequency, top_n, fee_bps))

    clean = clean.copy()
    clean["date"] = pd.to_datetime(clean["date"])
    selected_dates = set(select_rebalance_dates(clean["date"], rebalance_frequency))
    clean = clean.loc[clean["date"].isin(selected_dates)]
    previous_weights: dict[str, float] = {}
    fee = fee_bps / 10000.0
    rows: list[dict[str, Any]] = []

    for date, group in clean.groupby("date", sort=True):
        picks = group.nlargest(top_n, signal_col, keep="first")
        if picks.empty:
            continue
        weight = 1.0 / len(picks)
        weights = {str(code): weight for code in picks["code"]}
        turnover = sum(abs(weights.get(code, 0.0) - previous_weights.get(code, 0.0)) for code in set(weights) | set(previous_weights))
        gross_return = float(picks[return_col].mean())
        cost = turnover * fee
        rows.append(
            {
                "date": pd.Timestamp(date),
                "gross_return": gross_return,
                "cost": cost,
                "net_return": gross_return - cost,
                "turnover": float(turnover),
                "holdings": int(len(picks)),
            }
        )
        previous_weights = weights

    daily_returns = pd.DataFrame(rows)
    summary = summarize_rebalance_returns(
        daily_returns["net_return"].tolist() if not daily_returns.empty else [],
        rebalance_frequency,
        top_n,
        fee_bps,
    )
    summary.update(
        {
            "periods": int(len(daily_returns)),
            "mean_turnover": float(daily_returns["turnover"].mean()) if not daily_returns.empty else np.nan,
            "mean_gross_return": float(daily_returns["gross_return"].mean()) if not daily_returns.empty else np.nan,
            "mean_net_return": float(daily_returns["net_return"].mean()) if not daily_returns.empty else np.nan,
        }
    )
    return RebalanceBacktestResult(daily_returns=daily_returns, summary=summary)


def summarize_rebalance_returns(
    returns: Sequence[float],
    rebalance_frequency: str,
    top_n: int,
    fee_bps: float,
) -> dict[str, Any]:
    periods_per_year = REBALANCE_PERIODS_PER_YEAR[rebalance_frequency]
    values = [float(value) for value in returns if pd.notna(value)]
    return {
        "rebalance_frequency": rebalance_frequency,
        "top_n": int(top_n),
        "fee_bps": float(fee_bps),
        "annualized_return": float(annualized_return(values, periods_per_year=periods_per_year)),
        "volatility": float(volatility(values, periods_per_year=periods_per_year)),
        "sharpe": float(sharpe_ratio(values, periods_per_year=periods_per_year)),
        "max_drawdown": float(max_drawdown(values)),
    }
