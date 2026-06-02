"""Research panel construction for the first traditional quant loop."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .dataset_v2 import load_tradeable_panel


DEFAULT_HORIZONS = (1, 5, 20)
DEFAULT_SHORT_WINDOW = 5
DEFAULT_MEDIUM_WINDOW = 20
DEFAULT_VOLATILITY_WINDOW = 20


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _rolling_mean_by_code(frame: pd.DataFrame, column: str, window: int, *, min_periods: int) -> pd.Series:
    return (
        frame.groupby("code", sort=False)[column]
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )


def _rolling_std_by_code(frame: pd.DataFrame, column: str, window: int, *, min_periods: int) -> pd.Series:
    return (
        frame.groupby("code", sort=False)[column]
        .rolling(window=window, min_periods=min_periods)
        .std(ddof=0)
        .reset_index(level=0, drop=True)
    )


def _replace_infinite(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)


def build_factor_label_panel(
    panel: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    short_window: int = DEFAULT_SHORT_WINDOW,
    medium_window: int = DEFAULT_MEDIUM_WINDOW,
    volatility_window: int = DEFAULT_VOLATILITY_WINDOW,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Add baseline factors and future open-to-close return labels.

    Features at date `t` only use information available by the close of `t`.
    Forward labels use the next tradable open as the entry price and the
    horizon close as the exit price, making the default label executable after
    a close-of-day signal.
    """

    _require_columns(panel, ["date", "code", "open", "high", "low", "close", "volume", "amount"])
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must be positive")
    if short_window <= 0 or medium_window <= 0 or volatility_window <= 0:
        raise ValueError("windows must be positive")

    output = panel.copy()
    output["date"] = pd.to_datetime(output["date"])
    output = output.sort_values(["code", "date"]).reset_index(drop=True)

    numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    min_periods = min_periods or min(short_window, medium_window, volatility_window)

    grouped = output.groupby("code", sort=False)
    output["ret_1d"] = grouped["close"].pct_change(1)
    output[f"ret_{short_window}d"] = grouped["close"].pct_change(short_window)
    output[f"ret_{medium_window}d"] = grouped["close"].pct_change(medium_window)
    output[f"reversal_{short_window}d"] = -output[f"ret_{short_window}d"]
    output[f"momentum_{medium_window}d"] = output[f"ret_{medium_window}d"]

    moving_average = _rolling_mean_by_code(output, "close", medium_window, min_periods=min_periods)
    output[f"ma{medium_window}_gap"] = output["close"] / moving_average - 1.0

    output[f"volatility_{volatility_window}d"] = _rolling_std_by_code(
        output,
        "ret_1d",
        volatility_window,
        min_periods=min_periods,
    )
    output[f"neg_volatility_{volatility_window}d"] = -output[f"volatility_{volatility_window}d"]

    amount_mean = _rolling_mean_by_code(output, "amount", medium_window, min_periods=min_periods)
    output[f"log_amount_mean_{medium_window}d"] = np.log1p(amount_mean.clip(lower=0))

    intraday_range = (output["high"] - output["low"]) / output["close"]
    output[f"amplitude_{medium_window}d"] = (
        intraday_range.groupby(output["code"], sort=False)
        .rolling(window=medium_window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )
    output[f"neg_amplitude_{medium_window}d"] = -output[f"amplitude_{medium_window}d"]

    next_open = grouped["open"].shift(-1)
    for horizon in horizons:
        future_close = grouped["close"].shift(-horizon)
        output[f"fwd_ret_{horizon}d"] = future_close / next_open - 1.0

    factor_columns = default_factor_columns(
        short_window=short_window,
        medium_window=medium_window,
        volatility_window=volatility_window,
    )
    label_columns = [f"fwd_ret_{horizon}d" for horizon in horizons]
    _replace_infinite(output, [*factor_columns, *label_columns, "ret_1d"])
    return output.sort_values(["date", "code"]).reset_index(drop=True)


def default_factor_columns(
    *,
    short_window: int = DEFAULT_SHORT_WINDOW,
    medium_window: int = DEFAULT_MEDIUM_WINDOW,
    volatility_window: int = DEFAULT_VOLATILITY_WINDOW,
) -> list[str]:
    return [
        f"reversal_{short_window}d",
        f"momentum_{medium_window}d",
        f"ma{medium_window}_gap",
        f"neg_volatility_{volatility_window}d",
        f"log_amount_mean_{medium_window}d",
        f"neg_amplitude_{medium_window}d",
    ]


def add_cross_sectional_zscores(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    date_col: str = "date",
    suffix: str = "_z",
) -> pd.DataFrame:
    """Add per-date z-scores for factor columns."""

    _require_columns(frame, [date_col, *columns])
    output = frame.copy()

    def zscore(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        scale = values.std(ddof=0)
        if pd.isna(scale) or scale == 0:
            return pd.Series(np.where(values.notna(), 0.0, np.nan), index=series.index)
        return (values - values.mean()) / scale

    for column in columns:
        output[f"{column}{suffix}"] = output.groupby(date_col, sort=False)[column].transform(zscore)
    return output


def add_baseline_score(
    frame: pd.DataFrame,
    *,
    score_columns: Sequence[str] | None = None,
    score_col: str = "baseline_score",
    min_factors: int = 3,
) -> pd.DataFrame:
    """Average z-scored baseline factors into a simple alpha score."""

    output = frame.copy()
    if score_columns is None:
        raw_columns = default_factor_columns()
        score_columns = [f"{column}_z" for column in raw_columns if f"{column}_z" in output.columns]
    _require_columns(output, list(score_columns))
    if min_factors <= 0:
        raise ValueError("min_factors must be positive")

    values = output.loc[:, list(score_columns)].replace([np.inf, -np.inf], np.nan)
    counts = values.notna().sum(axis=1)
    output[score_col] = values.mean(axis=1)
    output.loc[counts < min_factors, score_col] = np.nan
    return output


def load_baseline_factor_panel(
    root: str | None = None,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Load the v2 tradeable panel and add first-loop factors and labels."""

    panel = load_tradeable_panel(root, start_date=start_date, end_date=end_date)
    if panel.empty:
        return panel
    factor_panel = build_factor_label_panel(panel, horizons=horizons)
    factor_panel = add_cross_sectional_zscores(factor_panel, default_factor_columns())
    return add_baseline_score(factor_panel)


def panel_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Return compact row/date/security counts for experiment metadata."""

    if frame.empty:
        return {"rows": 0, "date_count": 0, "security_count": 0, "date_min": None, "date_max": None}
    dates = pd.to_datetime(frame["date"])
    return {
        "rows": int(len(frame)),
        "date_count": int(dates.nunique()),
        "security_count": int(frame["code"].nunique()),
        "date_min": dates.min().date().isoformat(),
        "date_max": dates.max().date().isoformat(),
    }
