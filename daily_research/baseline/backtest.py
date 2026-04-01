from __future__ import annotations

from typing import Any, Dict, List, Tuple

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


def _trading_cost_return(
    *,
    turnover: float,
    buy_turnover: float,
    sell_turnover: float,
    config: ResearchConfig,
) -> float:
    transaction_cost = max(float(config.transaction_cost_bps), 0.0) / 10000.0
    slippage_cost = max(float(config.slippage_bps), 0.0) / 10000.0
    sell_tax = max(float(config.sell_tax_bps), 0.0) / 10000.0
    return float(turnover * (transaction_cost + slippage_cost) + sell_turnover * sell_tax)


def _runtime_override_row(
    daily_position_overrides: pd.DataFrame | None,
    dt: pd.Timestamp,
) -> dict[str, Any]:
    if daily_position_overrides is None or dt not in daily_position_overrides.index:
        return {}
    row = daily_position_overrides.loc[dt]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return {
        str(key): value
        for key, value in row.items()
        if pd.notna(value)
    }


def backtest(
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    target_weights: pd.DataFrame,
    target_scores: pd.DataFrame,
    config: ResearchConfig,
    regime_on: pd.Series | None = None,
    open_df: pd.DataFrame | None = None,
    benchmark_open: pd.Series | None = None,
    daily_position_overrides: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    execution_mode = str(config.execution_mode).lower()
    close = close.dropna(how="all")

    common_index = close.index.intersection(benchmark_close.index)
    if execution_mode == "next_open":
        if open_df is None or benchmark_open is None:
            raise ValueError("next_open backtest requires open_df and benchmark_open.")
        valid_benchmark_open = benchmark_open.dropna()
        common_index = common_index.intersection(open_df.index).intersection(valid_benchmark_open.index)
        open_df = open_df.loc[common_index]
        benchmark_open = valid_benchmark_open.reindex(common_index).ffill().dropna()

    close = close.loc[common_index]
    benchmark_close = benchmark_close.reindex(common_index).ffill().dropna()
    target_weights = target_weights.reindex(common_index).fillna(0.0)
    target_scores = target_scores.reindex(common_index).fillna(0.0)
    if regime_on is not None:
        regime_on = regime_on.reindex(common_index).fillna(False)
    if daily_position_overrides is not None:
        daily_position_overrides = daily_position_overrides.reindex(common_index)

    pm = PositionManager(config)
    weights = pd.Series(0.0, index=close.columns)

    portfolio_equity = 1.0
    gross_portfolio_equity = 1.0
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
                "gross_portfolio_equity": gross_portfolio_equity,
                "benchmark_equity": benchmark_equity,
                "excess_equity": portfolio_equity / benchmark_equity if benchmark_equity > 0 else np.nan,
                "portfolio_return": 0.0,
                "gross_portfolio_return": 0.0,
                "benchmark_return": 0.0,
                "excess_return": 0.0,
                "holding_count": 0,
                "turnover": 0.0,
                "buy_turnover": 0.0,
                "sell_turnover": 0.0,
                "trading_cost_return": 0.0,
                "regime_on": bool(regime_on.loc[initial_dt]) if regime_on is not None else True,
            }
        )
        for i in range(1, len(dates) - 1):
            execution_dt = dates[i]
            signal_dt = dates[i - 1]
            next_dt = dates[i + 1]
            runtime_overrides = _runtime_override_row(daily_position_overrides, signal_dt)

            weights, actions, diagnostics = pm.step(
                date=execution_dt,
                prices=open_df.loc[execution_dt],
                target_weights=target_weights.loc[signal_dt],
                target_scores=target_scores.loc[signal_dt],
                runtime_overrides=runtime_overrides,
            )
            for action in actions:
                action["signal_date"] = signal_dt
                action["execution_date"] = execution_dt
            action_logs.extend(actions)

            ret_vec = open_df.loc[next_dt] / open_df.loc[execution_dt] - 1.0
            gross_portfolio_return = float((weights.reindex(ret_vec.index).fillna(0.0) * ret_vec).sum())
            trading_cost_return = _trading_cost_return(
                turnover=float(diagnostics.get("turnover", 0.0)),
                buy_turnover=float(diagnostics.get("buy_turnover", 0.0)),
                sell_turnover=float(diagnostics.get("sell_turnover", 0.0)),
                config=config,
            )
            portfolio_return = float((1.0 - trading_cost_return) * (1.0 + gross_portfolio_return) - 1.0)
            benchmark_return = float(benchmark_open.loc[next_dt] / benchmark_open.loc[execution_dt] - 1.0)
            portfolio_equity *= 1.0 + portfolio_return
            gross_portfolio_equity *= 1.0 + gross_portfolio_return
            benchmark_equity *= 1.0 + benchmark_return

            equity_curve.append(
                {
                    "date": execution_dt,
                    "signal_date": signal_dt,
                    "execution_date": execution_dt,
                    "portfolio_equity": portfolio_equity,
                    "gross_portfolio_equity": gross_portfolio_equity,
                    "benchmark_equity": benchmark_equity,
                    "excess_equity": portfolio_equity / benchmark_equity if benchmark_equity > 0 else np.nan,
                    "portfolio_return": portfolio_return,
                    "gross_portfolio_return": gross_portfolio_return,
                    "benchmark_return": benchmark_return,
                    "excess_return": portfolio_return - benchmark_return,
                    "holding_count": diagnostics["holding_count"],
                    "turnover": diagnostics["turnover"],
                    "buy_turnover": diagnostics.get("buy_turnover", 0.0),
                    "sell_turnover": diagnostics.get("sell_turnover", 0.0),
                    "trading_cost_return": trading_cost_return,
                    "regime_on": bool(regime_on.loc[signal_dt]) if regime_on is not None else True,
                    "soft_override_active": diagnostics.get("runtime_override_active", False),
                }
            )
    else:
        for i, dt in enumerate(dates):
            portfolio_return = 0.0
            gross_portfolio_return = 0.0
            benchmark_return = 0.0
            trading_cost_return = 0.0
            if i > 0:
                prev_dt = dates[i - 1]
                ret_vec = close.loc[dt] / close.loc[prev_dt] - 1.0
                gross_portfolio_return = float((weights.reindex(ret_vec.index).fillna(0.0) * ret_vec).sum())
                benchmark_return = float(benchmark_close.loc[dt] / benchmark_close.loc[prev_dt] - 1.0)
                trading_cost_return = _trading_cost_return(
                    turnover=float(diagnostics.get("turnover", 0.0)) if i > 0 else 0.0,
                    buy_turnover=float(diagnostics.get("buy_turnover", 0.0)) if i > 0 else 0.0,
                    sell_turnover=float(diagnostics.get("sell_turnover", 0.0)) if i > 0 else 0.0,
                    config=config,
                )
                portfolio_return = float((1.0 - trading_cost_return) * (1.0 + gross_portfolio_return) - 1.0)
                portfolio_equity *= 1.0 + portfolio_return
                gross_portfolio_equity *= 1.0 + gross_portfolio_return
                benchmark_equity *= 1.0 + benchmark_return

            runtime_overrides = _runtime_override_row(daily_position_overrides, dt)
            weights, actions, diagnostics = pm.step(
                date=dt,
                prices=close.loc[dt],
                target_weights=target_weights.loc[dt],
                target_scores=target_scores.loc[dt],
                runtime_overrides=runtime_overrides,
            )
            action_logs.extend(actions)

            equity_curve.append(
                {
                    "date": dt,
                    "signal_date": dt,
                    "execution_date": dt,
                    "portfolio_equity": portfolio_equity,
                    "gross_portfolio_equity": gross_portfolio_equity,
                    "benchmark_equity": benchmark_equity,
                    "excess_equity": portfolio_equity / benchmark_equity if benchmark_equity > 0 else np.nan,
                    "portfolio_return": portfolio_return,
                    "gross_portfolio_return": gross_portfolio_return,
                    "benchmark_return": benchmark_return,
                    "excess_return": portfolio_return - benchmark_return,
                    "holding_count": diagnostics["holding_count"],
                    "turnover": diagnostics["turnover"],
                    "buy_turnover": diagnostics.get("buy_turnover", 0.0),
                    "sell_turnover": diagnostics.get("sell_turnover", 0.0),
                    "trading_cost_return": trading_cost_return,
                    "regime_on": bool(regime_on.loc[dt]) if regime_on is not None else True,
                    "soft_override_active": diagnostics.get("runtime_override_active", False),
                }
            )

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    action_df = pd.DataFrame(action_logs)

    portfolio_returns = equity_df["portfolio_return"].dropna()
    gross_portfolio_returns = equity_df["gross_portfolio_return"].dropna()
    benchmark_returns = equity_df["benchmark_return"].dropna()
    excess_returns = equity_df["excess_return"].dropna()

    portfolio_ann_ret = _annualized_return(equity_df["portfolio_equity"])
    gross_portfolio_ann_ret = _annualized_return(equity_df["gross_portfolio_equity"])
    benchmark_ann_ret = _annualized_return(equity_df["benchmark_equity"])
    excess_equity = equity_df["excess_equity"].replace([np.inf, -np.inf], np.nan).ffill().dropna()
    excess_ann_ret = _annualized_return(excess_equity) if not excess_equity.empty else 0.0
    gross_excess_equity = (equity_df["gross_portfolio_equity"] / equity_df["benchmark_equity"]).replace([np.inf, -np.inf], np.nan).ffill().dropna()
    gross_excess_ann_ret = _annualized_return(gross_excess_equity) if not gross_excess_equity.empty else 0.0

    portfolio_ann_vol = _annualized_vol(portfolio_returns)
    gross_portfolio_ann_vol = _annualized_vol(gross_portfolio_returns)
    excess_ann_vol = _annualized_vol(excess_returns)

    metrics = {
        "total_return": float(equity_df["portfolio_equity"].iloc[-1] - 1.0),
        "annual_return": float(portfolio_ann_ret),
        "annual_vol": float(portfolio_ann_vol),
        "sharpe": float(portfolio_ann_ret / portfolio_ann_vol) if portfolio_ann_vol > 0 else 0.0,
        "max_drawdown": float(_max_drawdown(equity_df["portfolio_equity"])),
        "gross_total_return": float(equity_df["gross_portfolio_equity"].iloc[-1] - 1.0),
        "gross_annual_return": float(gross_portfolio_ann_ret),
        "gross_annual_vol": float(gross_portfolio_ann_vol),
        "gross_sharpe": float(gross_portfolio_ann_ret / gross_portfolio_ann_vol) if gross_portfolio_ann_vol > 0 else 0.0,
        "gross_max_drawdown": float(_max_drawdown(equity_df["gross_portfolio_equity"])),
        "benchmark_total_return": float(equity_df["benchmark_equity"].iloc[-1] - 1.0),
        "benchmark_annual_return": float(benchmark_ann_ret),
        "excess_total_return": float(excess_equity.iloc[-1] - 1.0) if not excess_equity.empty else 0.0,
        "excess_annual_return": float(excess_ann_ret),
        "excess_sharpe": float(excess_ann_ret / excess_ann_vol) if excess_ann_vol > 0 else 0.0,
        "excess_max_drawdown": float(_max_drawdown(excess_equity)) if not excess_equity.empty else 0.0,
        "gross_excess_total_return": float(gross_excess_equity.iloc[-1] - 1.0) if not gross_excess_equity.empty else 0.0,
        "gross_excess_annual_return": float(gross_excess_ann_ret),
        "avg_holding_count": float(equity_df["holding_count"].mean()),
        "avg_turnover": float(equity_df["turnover"].mean()),
        "avg_buy_turnover": float(equity_df["buy_turnover"].mean()),
        "avg_sell_turnover": float(equity_df["sell_turnover"].mean()),
        "avg_trading_cost_return": float(equity_df["trading_cost_return"].mean()),
        "total_trading_cost_return": float(equity_df["trading_cost_return"].sum()),
        "annual_return_cost_drag": float(gross_portfolio_ann_ret - portfolio_ann_ret),
        "excess_annual_return_cost_drag": float(gross_excess_ann_ret - excess_ann_ret),
        "transaction_cost_bps": float(config.transaction_cost_bps),
        "slippage_bps": float(config.slippage_bps),
        "sell_tax_bps": float(config.sell_tax_bps),
        "hit_rate": float((portfolio_returns > 0).mean()) if not portfolio_returns.empty else 0.0,
    }
    if regime_on is not None and len(regime_on) > 0:
        metrics["regime_active_ratio"] = float(regime_on.mean())

    return equity_df, action_df, metrics
