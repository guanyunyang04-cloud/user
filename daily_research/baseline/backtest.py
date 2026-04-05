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


def summarize_backtest_by_month(
    equity_df: pd.DataFrame,
    action_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    columns = [
        "month",
        "start_date",
        "end_date",
        "portfolio_return",
        "gross_portfolio_return",
        "benchmark_return",
        "excess_return",
        "gross_excess_return",
        "avg_holding_count",
        "avg_turnover",
        "avg_buy_turnover",
        "avg_sell_turnover",
        "total_trading_cost_return",
        "regime_active_ratio",
        "action_count",
        "portfolio_equity_end",
        "benchmark_equity_end",
        "excess_equity_end",
    ]
    if equity_df is None or equity_df.empty:
        return pd.DataFrame(columns=columns)

    work = equity_df.reset_index().copy()
    date_col = "date" if "date" in work.columns else work.columns[0]
    work["date"] = pd.to_datetime(work[date_col])
    work["month"] = work["date"].dt.to_period("M").astype(str)

    action_counts: dict[str, int] = {}
    if action_df is not None and not action_df.empty and "execution_date" in action_df.columns:
        action_work = action_df.copy()
        action_work["execution_date"] = pd.to_datetime(action_work["execution_date"])
        action_counts = action_work.groupby(action_work["execution_date"].dt.to_period("M").astype(str)).size().to_dict()

    rows: List[Dict[str, Any]] = []
    for month, group in work.groupby("month", sort=True):
        portfolio_growth = float((1.0 + group["portfolio_return"].fillna(0.0)).prod())
        gross_growth = float((1.0 + group["gross_portfolio_return"].fillna(0.0)).prod())
        benchmark_growth = float((1.0 + group["benchmark_return"].fillna(0.0)).prod())
        excess_growth = portfolio_growth / benchmark_growth if benchmark_growth > 0 else np.nan
        gross_excess_growth = gross_growth / benchmark_growth if benchmark_growth > 0 else np.nan
        rows.append(
            {
                "month": str(month),
                "start_date": str(pd.Timestamp(group["date"].iloc[0]).date()),
                "end_date": str(pd.Timestamp(group["date"].iloc[-1]).date()),
                "portfolio_return": portfolio_growth - 1.0,
                "gross_portfolio_return": gross_growth - 1.0,
                "benchmark_return": benchmark_growth - 1.0,
                "excess_return": excess_growth - 1.0 if np.isfinite(excess_growth) else np.nan,
                "gross_excess_return": gross_excess_growth - 1.0 if np.isfinite(gross_excess_growth) else np.nan,
                "avg_holding_count": float(group["holding_count"].fillna(0.0).mean()),
                "avg_turnover": float(group["turnover"].fillna(0.0).mean()),
                "avg_buy_turnover": float(group["buy_turnover"].fillna(0.0).mean()),
                "avg_sell_turnover": float(group["sell_turnover"].fillna(0.0).mean()),
                "total_trading_cost_return": float(group["trading_cost_return"].fillna(0.0).sum()),
                "regime_active_ratio": float(group["regime_on"].fillna(False).mean()) if "regime_on" in group.columns else np.nan,
                "action_count": int(action_counts.get(str(month), 0)),
                "portfolio_equity_end": float(group["portfolio_equity"].iloc[-1]),
                "benchmark_equity_end": float(group["benchmark_equity"].iloc[-1]),
                "excess_equity_end": float(group["excess_equity"].iloc[-1]) if "excess_equity" in group.columns else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _longest_sign_streak(values: pd.Series, *, positive: bool) -> int:
    if values.empty:
        return 0
    longest = 0
    current = 0
    for value in values.astype(float):
        is_match = value > 0.0 if positive else value < 0.0
        if is_match:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def summarize_monthly_diagnostics(
    monthly_summary: pd.DataFrame,
    *,
    return_column: str = "excess_return",
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "return_column": str(return_column),
        "month_count": 0,
        "start_month": "",
        "end_month": "",
        "positive_month_count": 0,
        "nonnegative_month_count": 0,
        "negative_month_count": 0,
        "positive_month_ratio": 0.0,
        "nonnegative_month_ratio": 0.0,
        "negative_month_ratio": 0.0,
        "mean_monthly_return": 0.0,
        "median_monthly_return": 0.0,
        "std_monthly_return": 0.0,
        "upside_mean_monthly_return": 0.0,
        "downside_mean_monthly_return": 0.0,
        "best_monthly_return": 0.0,
        "worst_monthly_return": 0.0,
        "top3_positive_month_share": 0.0,
        "bottom3_negative_month_share": 0.0,
        "longest_positive_streak": 0,
        "longest_negative_streak": 0,
        "issue_flags": [],
    }
    if monthly_summary is None or monthly_summary.empty or return_column not in monthly_summary.columns:
        return diagnostics

    work = monthly_summary.copy()
    values = pd.to_numeric(work[return_column], errors="coerce").dropna()
    if values.empty:
        return diagnostics

    positive = values[values > 0.0]
    nonnegative = values[values >= 0.0]
    negative = values[values < 0.0]
    month_count = int(len(values))
    positive_sum = float(positive.sum()) if not positive.empty else 0.0
    negative_abs_sum = float(np.abs(negative).sum()) if not negative.empty else 0.0
    top3_positive_sum = float(positive.nlargest(min(3, len(positive))).sum()) if not positive.empty else 0.0
    bottom3_negative_abs_sum = float(np.abs(negative.nsmallest(min(3, len(negative)))).sum()) if not negative.empty else 0.0

    diagnostics.update(
        {
            "month_count": month_count,
            "start_month": str(work["month"].iloc[0]) if "month" in work.columns and not work.empty else "",
            "end_month": str(work["month"].iloc[-1]) if "month" in work.columns and not work.empty else "",
            "positive_month_count": int(len(positive)),
            "nonnegative_month_count": int(len(nonnegative)),
            "negative_month_count": int(len(negative)),
            "positive_month_ratio": float(len(positive) / month_count),
            "nonnegative_month_ratio": float(len(nonnegative) / month_count),
            "negative_month_ratio": float(len(negative) / month_count),
            "mean_monthly_return": float(values.mean()),
            "median_monthly_return": float(values.median()),
            "std_monthly_return": float(values.std(ddof=0)) if month_count > 1 else 0.0,
            "upside_mean_monthly_return": float(positive.mean()) if not positive.empty else 0.0,
            "downside_mean_monthly_return": float(negative.mean()) if not negative.empty else 0.0,
            "best_monthly_return": float(values.max()),
            "worst_monthly_return": float(values.min()),
            "top3_positive_month_share": float(top3_positive_sum / positive_sum) if positive_sum > 0 else 0.0,
            "bottom3_negative_month_share": float(bottom3_negative_abs_sum / negative_abs_sum)
            if negative_abs_sum > 0
            else 0.0,
            "longest_positive_streak": _longest_sign_streak(values, positive=True),
            "longest_negative_streak": _longest_sign_streak(values, positive=False),
        }
    )

    issue_flags: list[str] = []
    if diagnostics["positive_month_ratio"] < 0.55:
        issue_flags.append("low_positive_month_ratio")
    if diagnostics["median_monthly_return"] <= 0.0:
        issue_flags.append("negative_monthly_median")
    if diagnostics["worst_monthly_return"] <= -0.05:
        issue_flags.append("deep_bad_month")
    if diagnostics["top3_positive_month_share"] >= 0.65 and diagnostics["positive_month_count"] >= 3:
        issue_flags.append("concentrated_positive_months")
    if diagnostics["mean_monthly_return"] > 0.0 and diagnostics["median_monthly_return"] < 0.0:
        issue_flags.append("mean_median_gap")
    if diagnostics["longest_negative_streak"] >= 3:
        issue_flags.append("multi_month_drawdown_streak")
    diagnostics["issue_flags"] = issue_flags
    return diagnostics


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
