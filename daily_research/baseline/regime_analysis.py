from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.regime import compute_market_regime_state, resolve_regime_label_series


def classify_market_quadrants(
    benchmark_close: pd.Series,
    ma_window: int = 60,
    vol_window: int = 20,
    vol_threshold: float = 0.32,
    trend_flat_band: float = 0.01,
    vol_transition_band: float = 0.10,
    state_selector: str = "quadrant",
) -> pd.DataFrame:
    cfg = ResearchConfig(
        benchmark="000300.SH",
        regime_ma_window=ma_window,
        regime_vol_window=vol_window,
        regime_max_annual_vol=vol_threshold,
        regime_trend_flat_band=trend_flat_band,
        regime_vol_transition_band=vol_transition_band,
        regime_state_selector=state_selector,
        enable_market_regime_filter=False,
    )
    regime_state = compute_market_regime_state(benchmark_close, cfg)
    out = regime_state[
        [
            "benchmark_close",
            "benchmark_ma",
            "benchmark_trend_gap",
            "benchmark_annual_vol",
            "benchmark_vol_gap",
            "benchmark_vol_ratio",
            "trend_bucket",
            "vol_bucket",
            "market_state",
            "quadrant",
        ]
    ].copy()
    out["state_label"] = resolve_regime_label_series(regime_state, state_selector)
    out["state_selector"] = str(state_selector)
    out["trend_up"] = regime_state["trend_pass"].astype(bool)
    out["low_vol"] = regime_state["vol_pass"].astype(bool)
    return out


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
    return summarize_strategy_by_state(equity_df, quadrant_state, label_column="quadrant")


def summarize_strategy_by_state(
    equity_df: pd.DataFrame,
    regime_state: pd.DataFrame,
    label_column: str = "state_label",
) -> pd.DataFrame:
    aligned = equity_df.join(regime_state[[label_column]], how="left")
    rows = []
    for state_label, group in aligned.groupby(label_column, dropna=False):
        portfolio_ret = group["portfolio_return"].dropna()
        excess_ret = group["excess_return"].dropna()
        holding_count = group["holding_count"].fillna(0.0)
        active_mask = holding_count > 0
        active_excess = group.loc[active_mask, "excess_return"].dropna()
        active_portfolio = group.loc[active_mask, "portfolio_return"].dropna()

        rows.append(
            {
                "state_label": state_label,
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
    return pd.DataFrame(rows).sort_values("state_label")


def summarize_year_quadrants(
    equity_df: pd.DataFrame,
    quadrant_state: pd.DataFrame,
) -> pd.DataFrame:
    return summarize_year_states(equity_df, quadrant_state, label_column="quadrant")


def summarize_year_states(
    equity_df: pd.DataFrame,
    regime_state: pd.DataFrame,
    label_column: str = "state_label",
) -> pd.DataFrame:
    aligned = equity_df.join(regime_state[[label_column]], how="left")
    aligned = aligned.copy()
    aligned["year"] = aligned.index.year.astype(str)
    rows = []
    for (year, state_label), group in aligned.groupby(["year", label_column], dropna=False):
        excess_ret = group["excess_return"].dropna()
        rows.append(
            {
                "year": year,
                "state_label": state_label,
                "days": int(len(group)),
                "active_days": int((group["holding_count"].fillna(0.0) > 0).sum()),
                "excess_total_return": float((1.0 + excess_ret).prod() - 1.0) if not excess_ret.empty else 0.0,
                "excess_mean_daily": float(excess_ret.mean()) if not excess_ret.empty else 0.0,
                "excess_win_rate": float((excess_ret > 0).mean()) if not excess_ret.empty else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["year", "state_label"])
