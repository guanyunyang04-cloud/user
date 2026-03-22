from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.position_manager import PositionManager


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def _annualized_return(equity: pd.Series) -> float:
    daily_ret = equity.pct_change().dropna()
    if daily_ret.empty:
        return 0.0
    return float(equity.iloc[-1] ** (252 / len(daily_ret)) - 1.0)


def _annualized_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0


def backtest(
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    target_weights: pd.DataFrame,
    target_scores: pd.DataFrame,
    config: ResearchConfig,
    regime_on: pd.Series | None = None,
    open_df: pd.DataFrame | None = None,
    benchmark_open: pd.Series | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    execution_mode = str(config.execution_mode).lower()
    close = close.dropna(how="all")

    common_index = close.index.intersection(benchmark_close.index)
    if execution_mode == "next_open":
        if open_df is None or benchmark_open is None:
            raise ValueError("next_open backtest requires open_df and benchmark_open.")
        common_index = common_index.intersection(open_df.index).intersection(benchmark_open.index)
        open_df = open_df.loc[common_index]
        benchmark_open = benchmark_open.reindex(common_index).ffill().dropna()

    close = close.loc[common_index]
    benchmark_close = benchmark_close.reindex(common_index).ffill().dropna()
    target_weights = target_weights.reindex(common_index).fillna(0.0)
    target_scores = target_scores.reindex(common_index).fillna(0.0)
    if regime_on is not None:
        regime_on = regime_on.reindex(common_index).fillna(False)

    pm = PositionManager(config)
    weights = pd.Series(0.0, index=close.columns)

    portfolio_equity = 1.0
    benchmark_equity = 1.0
    equity_curve: List[Dict] = []
    action_logs: List[Dict] = []

    dates = close.index
    if execution_mode == "next_open":
        if len(dates) < 3:
            raise ValueError("next_open backtest needs at least 3 dates.")
        initial_dt = dates[0]
        equity_curve.append(
            {
                "date": initial_dt,
                "signal_date": pd.NaT,
                "execution_date": initial_dt,
                "portfolio_equity": portfolio_equity,
                "benchmark_equity": benchmark_equity,
                "excess_equity": portfolio_equity / benchmark_equity if benchmark_equity > 0 else np.nan,
                "portfolio_return": 0.0,
                "benchmark_return": 0.0,
                "excess_return": 0.0,
                "holding_count": 0,
                "turnover": 0.0,
                "regime_on": bool(regime_on.loc[initial_dt]) if regime_on is not None else True,
            }
        )
        for i in range(1, len(dates) - 1):
            execution_dt = dates[i]
            signal_dt = dates[i - 1]
            next_dt = dates[i + 1]

            weights, actions, diagnostics = pm.step(
                date=execution_dt,
                prices=open_df.loc[execution_dt],
                target_weights=target_weights.loc[signal_dt],
                target_scores=target_scores.loc[signal_dt],
            )
            for action in actions:
                action["signal_date"] = signal_dt
                action["execution_date"] = execution_dt
            action_logs.extend(actions)

            ret_vec = open_df.loc[next_dt] / open_df.loc[execution_dt] - 1.0
            portfolio_return = float((weights.reindex(ret_vec.index).fillna(0.0) * ret_vec).sum())
            benchmark_return = float(benchmark_open.loc[next_dt] / benchmark_open.loc[execution_dt] - 1.0)
            portfolio_equity *= 1.0 + portfolio_return
            benchmark_equity *= 1.0 + benchmark_return

            equity_curve.append(
                {
                    "date": execution_dt,
                    "signal_date": signal_dt,
                    "execution_date": execution_dt,
                    "portfolio_equity": portfolio_equity,
                    "benchmark_equity": benchmark_equity,
                    "excess_equity": portfolio_equity / benchmark_equity if benchmark_equity > 0 else np.nan,
                    "portfolio_return": portfolio_return,
                    "benchmark_return": benchmark_return,
                    "excess_return": portfolio_return - benchmark_return,
                    "holding_count": diagnostics["holding_count"],
                    "turnover": diagnostics["turnover"],
                    "regime_on": bool(regime_on.loc[signal_dt]) if regime_on is not None else True,
                }
            )
    else:
        for i, dt in enumerate(dates):
            portfolio_return = 0.0
            benchmark_return = 0.0
            if i > 0:
                prev_dt = dates[i - 1]
                ret_vec = close.loc[dt] / close.loc[prev_dt] - 1.0
                portfolio_return = float((weights.reindex(ret_vec.index).fillna(0.0) * ret_vec).sum())
                benchmark_return = float(benchmark_close.loc[dt] / benchmark_close.loc[prev_dt] - 1.0)
                portfolio_equity *= 1.0 + portfolio_return
                benchmark_equity *= 1.0 + benchmark_return

            weights, actions, diagnostics = pm.step(
                date=dt,
                prices=close.loc[dt],
                target_weights=target_weights.loc[dt],
                target_scores=target_scores.loc[dt],
            )
            action_logs.extend(actions)

            equity_curve.append(
                {
                    "date": dt,
                    "signal_date": dt,
                    "execution_date": dt,
                    "portfolio_equity": portfolio_equity,
                    "benchmark_equity": benchmark_equity,
                    "excess_equity": portfolio_equity / benchmark_equity if benchmark_equity > 0 else np.nan,
                    "portfolio_return": portfolio_return,
                    "benchmark_return": benchmark_return,
                    "excess_return": portfolio_return - benchmark_return,
                    "holding_count": diagnostics["holding_count"],
                    "turnover": diagnostics["turnover"],
                    "regime_on": bool(regime_on.loc[dt]) if regime_on is not None else True,
                }
            )

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    action_df = pd.DataFrame(action_logs)

    portfolio_returns = equity_df["portfolio_return"].dropna()
    benchmark_returns = equity_df["benchmark_return"].dropna()
    excess_returns = equity_df["excess_return"].dropna()

    portfolio_ann_ret = _annualized_return(equity_df["portfolio_equity"])
    benchmark_ann_ret = _annualized_return(equity_df["benchmark_equity"])
    excess_equity = equity_df["excess_equity"].replace([np.inf, -np.inf], np.nan).ffill().dropna()
    excess_ann_ret = _annualized_return(excess_equity) if not excess_equity.empty else 0.0

    portfolio_ann_vol = _annualized_vol(portfolio_returns)
    excess_ann_vol = _annualized_vol(excess_returns)

    metrics = {
        "total_return": float(equity_df["portfolio_equity"].iloc[-1] - 1.0),
        "annual_return": float(portfolio_ann_ret),
        "annual_vol": float(portfolio_ann_vol),
        "sharpe": float(portfolio_ann_ret / portfolio_ann_vol) if portfolio_ann_vol > 0 else 0.0,
        "max_drawdown": float(_max_drawdown(equity_df["portfolio_equity"])),
        "benchmark_total_return": float(equity_df["benchmark_equity"].iloc[-1] - 1.0),
        "benchmark_annual_return": float(benchmark_ann_ret),
        "excess_total_return": float(excess_equity.iloc[-1] - 1.0) if not excess_equity.empty else 0.0,
        "excess_annual_return": float(excess_ann_ret),
        "excess_sharpe": float(excess_ann_ret / excess_ann_vol) if excess_ann_vol > 0 else 0.0,
        "excess_max_drawdown": float(_max_drawdown(excess_equity)) if not excess_equity.empty else 0.0,
        "avg_holding_count": float(equity_df["holding_count"].mean()),
        "avg_turnover": float(equity_df["turnover"].mean()),
        "hit_rate": float((portfolio_returns > 0).mean()) if not portfolio_returns.empty else 0.0,
    }
    if regime_on is not None and len(regime_on) > 0:
        metrics["regime_active_ratio"] = float(regime_on.mean())

    return equity_df, action_df, metrics
