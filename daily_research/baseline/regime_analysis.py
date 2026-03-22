from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def classify_market_quadrants(
    benchmark_close: pd.Series,
    ma_window: int = 60,
    vol_window: int = 20,
    vol_threshold: float = 0.32,
) -> pd.DataFrame:
    benchmark_close = benchmark_close.astype(float).sort_index().dropna()
    ma = benchmark_close.rolling(ma_window).mean()
    daily_ret = benchmark_close.pct_change(fill_method=None)
    annual_vol = daily_ret.rolling(vol_window).std() * np.sqrt(252)

    trend_up = (benchmark_close > ma).fillna(False)
    low_vol = (annual_vol <= float(vol_threshold)).fillna(False)

    quadrant = pd.Series("unknown", index=benchmark_close.index, dtype="object")
    quadrant.loc[trend_up & low_vol] = "trend_up_low_vol"
    quadrant.loc[trend_up & (~low_vol)] = "trend_up_high_vol"
    quadrant.loc[(~trend_up) & low_vol] = "trend_down_low_vol"
    quadrant.loc[(~trend_up) & (~low_vol)] = "trend_down_high_vol"

    return pd.DataFrame(
        {
            "benchmark_close": benchmark_close,
            "benchmark_ma": ma,
            "benchmark_annual_vol": annual_vol,
            "trend_up": trend_up.astype(bool),
            "low_vol": low_vol.astype(bool),
            "quadrant": quadrant,
        }
    )


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    returns = returns.dropna().astype(float)
    if returns.empty:
        return 0.0
    equity = (1.0 + returns).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def summarize_strategy_by_quadrant(
    equity_df: pd.DataFrame,
    quadrant_state: pd.DataFrame,
) -> pd.DataFrame:
    aligned = equity_df.join(quadrant_state[["quadrant"]], how="left")
    rows = []
    for quadrant, group in aligned.groupby("quadrant", dropna=False):
        portfolio_ret = group["portfolio_return"].dropna()
        excess_ret = group["excess_return"].dropna()
        holding_count = group["holding_count"].fillna(0.0)
        active_mask = holding_count > 0
        active_excess = group.loc[active_mask, "excess_return"].dropna()
        active_portfolio = group.loc[active_mask, "portfolio_return"].dropna()

        rows.append(
            {
                "quadrant": quadrant,
                "days": int(len(group)),
                "active_days": int(active_mask.sum()),
                "active_ratio": float(active_mask.mean()) if len(group) > 0 else 0.0,
                "avg_holding_count": float(holding_count.mean()) if len(group) > 0 else 0.0,
                "avg_turnover": float(group["turnover"].fillna(0.0).mean()) if len(group) > 0 else 0.0,
                "portfolio_total_return": float((1.0 + portfolio_ret).prod() - 1.0) if not portfolio_ret.empty else 0.0,
                "excess_total_return": float((1.0 + excess_ret).prod() - 1.0) if not excess_ret.empty else 0.0,
                "portfolio_mean_daily": float(portfolio_ret.mean()) if not portfolio_ret.empty else 0.0,
                "excess_mean_daily": float(excess_ret.mean()) if not excess_ret.empty else 0.0,
                "portfolio_win_rate": float((portfolio_ret > 0).mean()) if not portfolio_ret.empty else 0.0,
                "excess_win_rate": float((excess_ret > 0).mean()) if not excess_ret.empty else 0.0,
                "active_portfolio_mean_daily": float(active_portfolio.mean()) if not active_portfolio.empty else 0.0,
                "active_excess_mean_daily": float(active_excess.mean()) if not active_excess.empty else 0.0,
                "active_excess_win_rate": float((active_excess > 0).mean()) if not active_excess.empty else 0.0,
                "portfolio_sharpe_like": float(portfolio_ret.mean() / portfolio_ret.std() * np.sqrt(252)) if len(portfolio_ret) > 1 and portfolio_ret.std() > 0 else 0.0,
                "excess_sharpe_like": float(excess_ret.mean() / excess_ret.std() * np.sqrt(252)) if len(excess_ret) > 1 and excess_ret.std() > 0 else 0.0,
                "active_excess_sharpe_like": float(active_excess.mean() / active_excess.std() * np.sqrt(252)) if len(active_excess) > 1 and active_excess.std() > 0 else 0.0,
                "portfolio_max_drawdown": _max_drawdown_from_returns(portfolio_ret),
                "excess_max_drawdown": _max_drawdown_from_returns(excess_ret),
            }
        )
    return pd.DataFrame(rows).sort_values("quadrant")


def summarize_year_quadrants(
    equity_df: pd.DataFrame,
    quadrant_state: pd.DataFrame,
) -> pd.DataFrame:
    aligned = equity_df.join(quadrant_state[["quadrant"]], how="left")
    aligned = aligned.copy()
    aligned["year"] = aligned.index.year.astype(str)
    rows = []
    for (year, quadrant), group in aligned.groupby(["year", "quadrant"], dropna=False):
        excess_ret = group["excess_return"].dropna()
        rows.append(
            {
                "year": year,
                "quadrant": quadrant,
                "days": int(len(group)),
                "active_days": int((group["holding_count"].fillna(0.0) > 0).sum()),
                "excess_total_return": float((1.0 + excess_ret).prod() - 1.0) if not excess_ret.empty else 0.0,
                "excess_mean_daily": float(excess_ret.mean()) if not excess_ret.empty else 0.0,
                "excess_win_rate": float((excess_ret > 0).mean()) if not excess_ret.empty else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["year", "quadrant"])
