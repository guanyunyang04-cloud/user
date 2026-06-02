"""Diagnostics for factor research and simple portfolio baselines."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .metrics import annualized_return, max_drawdown, sharpe_ratio, volatility


def assign_time_split(
    frame: pd.DataFrame,
    *,
    split_date: str | pd.Timestamp | None,
    date_col: str = "date",
    split_col: str = "sample_split",
) -> pd.DataFrame:
    """Add an in-sample/out-of-sample split column by date."""

    if date_col not in frame.columns:
        raise ValueError(f"missing required column: {date_col}")
    output = frame.copy()
    if split_date is None:
        output[split_col] = "full"
        return output
    dates = pd.to_datetime(output[date_col])
    boundary = pd.Timestamp(split_date)
    output[split_col] = np.where(dates < boundary, "in_sample", "out_of_sample")
    return output


def ic_by_date(
    frame: pd.DataFrame,
    signal_col: str,
    label_col: str,
    *,
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute daily IC between a signal and a forward-return label."""

    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    required = ["date", signal_col, label_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    clean = frame.loc[:, required].replace([np.inf, -np.inf], np.nan).dropna()
    rows: list[dict[str, Any]] = []
    for date, group in clean.groupby("date", sort=True):
        if len(group) < 2 or group[signal_col].nunique() < 2 or group[label_col].nunique() < 2:
            continue
        rows.append(
            {
                "date": pd.Timestamp(date),
                "ic": float(group[signal_col].corr(group[label_col], method=method)),
                "observations": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def summarize_factor_ic(frame: pd.DataFrame, signal_cols: Sequence[str], label_col: str) -> pd.DataFrame:
    """Return Pearson IC and RankIC summaries for one or more signals."""

    rows: list[dict[str, Any]] = []
    for signal_col in signal_cols:
        pearson = ic_by_date(frame, signal_col, label_col, method="pearson")
        rank = ic_by_date(frame, signal_col, label_col, method="spearman")
        row: dict[str, Any] = {
            "signal": signal_col,
            "label": label_col,
            "dates": int(len(rank)),
            "observations": int(frame[[signal_col, label_col]].dropna().shape[0]),
            "mean_ic": _mean_or_nan(pearson["ic"]) if not pearson.empty else np.nan,
            "mean_rank_ic": _mean_or_nan(rank["ic"]) if not rank.empty else np.nan,
            "std_rank_ic": _std_or_nan(rank["ic"]) if not rank.empty else np.nan,
        }
        if pd.notna(row["std_rank_ic"]) and row["std_rank_ic"] != 0:
            row["rank_icir"] = row["mean_rank_ic"] / row["std_rank_ic"] * sqrt(252)
        else:
            row["rank_icir"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_factor_ic_by_split(
    frame: pd.DataFrame,
    signal_cols: Sequence[str],
    label_col: str,
    *,
    split_col: str = "sample_split",
) -> pd.DataFrame:
    """Return IC summaries for each sample split."""

    if split_col not in frame.columns:
        raise ValueError(f"missing required column: {split_col}")
    rows: list[pd.DataFrame] = []
    for split_name, group in frame.groupby(split_col, sort=True):
        summary = summarize_factor_ic(group, signal_cols, label_col)
        if summary.empty:
            continue
        summary.insert(0, "sample_split", str(split_name))
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def quantile_returns(
    frame: pd.DataFrame,
    signal_col: str,
    label_col: str,
    *,
    quantiles: int = 5,
) -> pd.DataFrame:
    """Compute average daily returns by signal quantile."""

    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    required = ["date", signal_col, label_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    clean = frame.loc[:, required].replace([np.inf, -np.inf], np.nan).dropna()
    rows: list[dict[str, Any]] = []
    for date, group in clean.groupby("date", sort=True):
        if len(group) < quantiles or group[signal_col].nunique() < quantiles:
            continue
        ranked = group[signal_col].rank(method="first")
        assigned = group.copy()
        assigned["quantile"] = pd.qcut(ranked, q=quantiles, labels=False) + 1
        for quantile, quantile_group in assigned.groupby("quantile", sort=True):
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "quantile": int(quantile),
                    "mean_return": float(quantile_group[label_col].mean()),
                    "observations": int(len(quantile_group)),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["quantile", "mean_return", "periods", "observations"])

    daily = pd.DataFrame(rows)
    return (
        daily.groupby("quantile", sort=True)
        .agg(mean_return=("mean_return", "mean"), periods=("date", "nunique"), observations=("observations", "sum"))
        .reset_index()
    )


def quantile_returns_by_split(
    frame: pd.DataFrame,
    signal_col: str,
    label_col: str,
    *,
    quantiles: int = 5,
    split_col: str = "sample_split",
) -> pd.DataFrame:
    """Compute quantile return summaries for each sample split."""

    if split_col not in frame.columns:
        raise ValueError(f"missing required column: {split_col}")
    rows: list[pd.DataFrame] = []
    for split_name, group in frame.groupby(split_col, sort=True):
        summary = quantile_returns(group, signal_col, label_col, quantiles=quantiles)
        if summary.empty:
            continue
        summary.insert(0, "signal", signal_col)
        summary.insert(0, "sample_split", str(split_name))
        rows.append(summary)
    if not rows:
        return pd.DataFrame(columns=["sample_split", "signal", "quantile", "mean_return", "periods", "observations"])
    return pd.concat(rows, ignore_index=True)


@dataclass(frozen=True)
class TopNBacktestResult:
    daily_returns: pd.DataFrame
    summary: dict[str, Any]


def top_n_backtest(
    frame: pd.DataFrame,
    signal_col: str,
    return_col: str,
    *,
    top_n: int = 100,
    fee_bps: float = 10.0,
) -> TopNBacktestResult:
    """Daily equal-weight Top-N long-only baseline with turnover costs."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    required = ["date", "code", signal_col, return_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    clean = frame.loc[:, required].replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean.sort_values(["date", signal_col, "code"], ascending=[True, False, True])
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
        net_return = gross_return - cost
        rows.append(
            {
                "date": pd.Timestamp(date),
                "gross_return": gross_return,
                "cost": cost,
                "net_return": net_return,
                "turnover": float(turnover),
                "holdings": int(len(picks)),
            }
        )
        previous_weights = weights

    daily_returns = pd.DataFrame(rows)
    summary = summarize_strategy_returns(daily_returns["net_return"].tolist() if not daily_returns.empty else [])
    summary.update(
        {
            "periods": int(len(daily_returns)),
            "top_n": int(top_n),
            "fee_bps": float(fee_bps),
            "mean_turnover": float(daily_returns["turnover"].mean()) if not daily_returns.empty else np.nan,
            "mean_gross_return": float(daily_returns["gross_return"].mean()) if not daily_returns.empty else np.nan,
            "mean_net_return": float(daily_returns["net_return"].mean()) if not daily_returns.empty else np.nan,
        }
    )
    return TopNBacktestResult(daily_returns=daily_returns, summary=summary)


def top_n_backtest_by_split(
    frame: pd.DataFrame,
    signal_cols: Sequence[str],
    return_col: str,
    *,
    top_n: int = 100,
    fee_bps: float = 10.0,
    split_col: str = "sample_split",
) -> pd.DataFrame:
    """Run Top-N diagnostics per signal and sample split."""

    if split_col not in frame.columns:
        raise ValueError(f"missing required column: {split_col}")
    rows: list[dict[str, Any]] = []
    for split_name, group in frame.groupby(split_col, sort=True):
        for signal_col in signal_cols:
            result = top_n_backtest(group, signal_col, return_col, top_n=top_n, fee_bps=fee_bps)
            row = {"sample_split": str(split_name), "signal": signal_col}
            row.update(result.summary)
            rows.append(row)
    return pd.DataFrame(rows)


def single_factor_diagnostics(
    frame: pd.DataFrame,
    signal_cols: Sequence[str],
    label_col: str,
    *,
    split_date: str | pd.Timestamp | None = None,
    top_n: int = 100,
    fee_bps: float = 10.0,
    quantiles: int = 5,
) -> dict[str, pd.DataFrame]:
    """Generate IC, quantile, and Top-N diagnostics for signals."""

    split_frame = assign_time_split(frame, split_date=split_date)
    return {
        "panel": split_frame,
        "ic": summarize_factor_ic_by_split(split_frame, signal_cols, label_col),
        "quantile": pd.concat(
            [
                quantile_returns_by_split(split_frame, signal_col, label_col, quantiles=quantiles)
                for signal_col in signal_cols
            ],
            ignore_index=True,
        ),
        "top_n": top_n_backtest_by_split(split_frame, signal_cols, label_col, top_n=top_n, fee_bps=fee_bps),
    }


def summarize_strategy_returns(returns: Sequence[float]) -> dict[str, float]:
    values = [float(value) for value in returns if pd.notna(value)]
    return {
        "annualized_return": float(annualized_return(values)),
        "volatility": float(volatility(values)),
        "sharpe": float(sharpe_ratio(values)),
        "max_drawdown": float(max_drawdown(values)),
    }


def _mean_or_nan(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.mean())


def _std_or_nan(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.std(ddof=0))
