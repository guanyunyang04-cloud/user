"""Multi-factor ranking helpers for traditional quant research."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def add_equal_rank_score(
    frame: pd.DataFrame,
    factor_cols: Sequence[str],
    *,
    directions: Mapping[str, int | float] | None = None,
    score_col: str = "multifactor_equal_rank_score",
    date_col: str = "date",
    min_factors: int = 1,
) -> pd.DataFrame:
    """Add an equal-weight cross-sectional percentile-rank factor score."""

    ranks = cross_sectional_factor_ranks(frame, factor_cols, directions=directions, date_col=date_col)
    return _add_weighted_score(frame, ranks, factor_cols, {factor: 1.0 for factor in factor_cols}, score_col, min_factors)


def add_ic_weighted_rank_score(
    frame: pd.DataFrame,
    factor_cols: Sequence[str],
    ic_summary: pd.DataFrame,
    *,
    directions: Mapping[str, int | float] | None = None,
    score_col: str = "multifactor_ic_weighted_score",
    date_col: str = "date",
    min_factors: int = 1,
) -> pd.DataFrame:
    """Add a rank score weighted by absolute mean RankIC from diagnostics."""

    weights = weights_from_ic_summary(ic_summary, factor_cols)
    ranks = cross_sectional_factor_ranks(frame, factor_cols, directions=directions, date_col=date_col)
    return _add_weighted_score(frame, ranks, factor_cols, weights, score_col, min_factors)


def add_rolling_ic_weighted_rank_score(
    frame: pd.DataFrame,
    factor_cols: Sequence[str],
    label_col: str,
    *,
    window: int = 252,
    min_periods: int = 20,
    score_col: str = "multifactor_rolling_ic_weighted_score",
    date_col: str = "date",
    min_factors: int = 1,
) -> pd.DataFrame:
    """Add a rolling IC-weighted rank score using only prior dates."""

    if min_factors <= 0:
        raise ValueError("min_factors must be positive")
    weights = rolling_ic_weights_by_date(
        frame,
        factor_cols,
        label_col,
        window=window,
        min_periods=min_periods,
        date_col=date_col,
    )
    return add_rank_score_from_weight_table(
        frame,
        factor_cols,
        weights,
        score_col=score_col,
        date_col=date_col,
        min_factors=min_factors,
    )


def add_rank_score_from_weight_table(
    frame: pd.DataFrame,
    factor_cols: Sequence[str],
    weights: pd.DataFrame,
    *,
    score_col: str,
    date_col: str = "date",
    min_factors: int = 1,
) -> pd.DataFrame:
    """Add a weighted rank score from a date/factor weight table."""

    _require_columns(frame, [date_col, *factor_cols])
    _require_columns(weights, [date_col, "factor", "weight", "direction"])
    if min_factors <= 0:
        raise ValueError("min_factors must be positive")

    output = frame.copy()
    working = frame.copy()
    working[date_col] = pd.to_datetime(working[date_col])
    weight_table = weights.copy()
    weight_table[date_col] = pd.to_datetime(weight_table[date_col])
    weights_by_date = {date: group.set_index("factor") for date, group in weight_table.groupby(date_col, sort=False)}
    scores = pd.Series(np.nan, index=working.index, dtype=float)

    for date, group in working.groupby(date_col, sort=True):
        date_weights = weights_by_date.get(pd.Timestamp(date))
        if date_weights is None:
            continue
        weighted_sum = pd.Series(0.0, index=group.index)
        weight_sum = pd.Series(0.0, index=group.index)
        available = pd.Series(0, index=group.index)
        for factor in factor_cols:
            if factor not in date_weights.index:
                continue
            weight = float(date_weights.loc[factor, "weight"])
            direction = _factor_direction(factor, {factor: date_weights.loc[factor, "direction"]})
            values = pd.to_numeric(group[factor], errors="coerce") * direction
            ranks = values.rank(pct=True, method="average")
            mask = ranks.notna()
            weighted_sum.loc[group.index] += ranks.fillna(0.0) * weight
            weight_sum.loc[group.index] += mask.astype(float) * weight
            available.loc[group.index] += mask.astype(int)
        date_scores = weighted_sum / weight_sum.replace(0.0, np.nan)
        date_scores.loc[available < min_factors] = np.nan
        scores.loc[group.index] = date_scores

    output[score_col] = scores
    return output


def add_prior_fit_pruned_rank_score(
    frame: pd.DataFrame,
    pruning_plan: pd.DataFrame,
    *,
    score_col: str = "factor_pruned_rank_score_h20_prior_fit",
    date_col: str = "date",
    min_factors: int = 1,
) -> pd.DataFrame:
    """Add an equal-rank score from eval-year prior-fit factor pruning rules."""

    _require_columns(frame, [date_col])
    _require_columns(pruning_plan, ["eval_year", "selected_factors", "selected_factor_directions"])
    if min_factors <= 0:
        raise ValueError("min_factors must be positive")

    output = frame.copy()
    output[score_col] = np.nan
    if pruning_plan.empty:
        return output

    plan = _normalize_pruning_plan(pruning_plan)
    missing_factors = sorted({factor for rule in plan.values() for factor in rule["factors"] if factor not in frame.columns})
    if missing_factors:
        raise ValueError(f"pruned factor columns not found: {missing_factors}")

    working = frame.copy()
    working[date_col] = pd.to_datetime(working[date_col])
    scores = pd.Series(np.nan, index=working.index, dtype=float)
    for eval_year, rule in plan.items():
        mask = working[date_col].dt.year.eq(int(eval_year))
        if not mask.any():
            continue
        factors = rule["factors"]
        ranks = cross_sectional_factor_ranks(
            working.loc[mask, [date_col, *factors]].copy(),
            factors,
            directions=rule["directions"],
            date_col=date_col,
        )
        year_scores = _equal_rank_scores_from_ranks(ranks, factors, min_factors=min(min_factors, len(factors)))
        scores.loc[working.index[mask]] = year_scores

    output[score_col] = scores
    return output


def rolling_ic_weights_by_date(
    frame: pd.DataFrame,
    factor_cols: Sequence[str],
    label_col: str,
    *,
    window: int = 252,
    min_periods: int = 20,
    date_col: str = "date",
    method: str = "spearman",
) -> pd.DataFrame:
    """Return rolling IC weights for each score date and factor.

    The weight row for date `t` is calculated from dates strictly before `t`.
    """

    if window <= 0:
        raise ValueError("window must be positive")
    if min_periods <= 0:
        raise ValueError("min_periods must be positive")
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    _require_columns(frame, [date_col, label_col, *factor_cols])
    if not factor_cols:
        raise ValueError("factor_cols must not be empty")

    working = frame.copy()
    working[date_col] = pd.to_datetime(working[date_col])
    dates = pd.DatetimeIndex(working[date_col].dropna().unique()).sort_values()
    daily_ic = _daily_factor_ic(working, factor_cols, label_col, date_col=date_col, method=method)
    rows: list[dict[str, float | int | bool | pd.Timestamp | str]] = []

    for date in dates:
        history = daily_ic.loc[daily_ic.index < date].tail(window)
        valid_history_days = int(history.notna().any(axis=1).sum()) if not history.empty else 0
        if valid_history_days < min_periods:
            rows.extend(_fallback_weight_rows(date, factor_cols, valid_history_days, reason="insufficient_history"))
            continue

        mean_ic = history.mean(skipna=True)
        raw_weights = mean_ic.abs().replace([np.inf, -np.inf], np.nan)
        total = raw_weights.dropna().sum()
        if total <= 0 or pd.isna(total):
            rows.extend(_fallback_weight_rows(date, factor_cols, valid_history_days, reason="zero_ic"))
            continue

        for factor in factor_cols:
            factor_ic = mean_ic.get(factor, np.nan)
            weight = raw_weights.get(factor, np.nan)
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "factor": factor,
                    "mean_rank_ic": float(factor_ic) if pd.notna(factor_ic) else np.nan,
                    "weight": float(weight / total) if pd.notna(weight) else 0.0,
                    "direction": -1 if pd.notna(factor_ic) and float(factor_ic) < 0 else 1,
                    "history_days": valid_history_days,
                    "is_fallback": False,
                    "fallback_reason": "",
                }
            )
    return pd.DataFrame(rows)


def cross_sectional_factor_ranks(
    frame: pd.DataFrame,
    factor_cols: Sequence[str],
    *,
    directions: Mapping[str, int | float] | None = None,
    date_col: str = "date",
    suffix: str = "_rank",
) -> pd.DataFrame:
    """Return per-date percentile ranks for factor columns after applying directions."""

    _require_columns(frame, [date_col, *factor_cols])
    if not factor_cols:
        raise ValueError("factor_cols must not be empty")
    directions = directions or {}
    ranks = pd.DataFrame(index=frame.index)
    for factor in factor_cols:
        direction = _factor_direction(factor, directions)
        values = pd.to_numeric(frame[factor], errors="coerce") * direction
        ranks[f"{factor}{suffix}"] = values.groupby(frame[date_col], sort=False).rank(pct=True, method="average")
    return ranks


def weights_from_ic_summary(
    ic_summary: pd.DataFrame,
    factor_cols: Sequence[str],
    *,
    signal_col: str = "signal",
    ic_col: str = "mean_rank_ic",
) -> dict[str, float]:
    """Build normalized non-negative factor weights from absolute mean RankIC."""

    if not factor_cols:
        raise ValueError("factor_cols must not be empty")
    _require_columns(ic_summary, [signal_col, ic_col])
    lookup = ic_summary.drop_duplicates(signal_col, keep="first").set_index(signal_col)[ic_col]
    raw = {factor: abs(float(pd.to_numeric(lookup.get(factor, np.nan), errors="coerce"))) for factor in factor_cols}
    total = sum(value for value in raw.values() if pd.notna(value))
    if total <= 0 or pd.isna(total):
        equal = 1.0 / len(factor_cols)
        return {factor: equal for factor in factor_cols}
    return {factor: (0.0 if pd.isna(value) else value / total) for factor, value in raw.items()}


def select_low_correlation_factors(
    correlation: pd.DataFrame,
    factor_cols: Sequence[str],
    *,
    ic_summary: pd.DataFrame | None = None,
    max_abs_corr: float = 0.75,
    signal_col: str = "signal",
    ic_col: str = "mean_rank_ic",
) -> list[str]:
    """Greedily select factors while capping absolute pairwise correlation."""

    if not factor_cols:
        raise ValueError("factor_cols must not be empty")
    if max_abs_corr < 0 or max_abs_corr > 1:
        raise ValueError("max_abs_corr must be between 0 and 1")
    missing = [factor for factor in factor_cols if factor not in correlation.index or factor not in correlation.columns]
    if missing:
        raise ValueError(f"missing correlation rows/columns: {missing}")

    scores = _factor_selection_scores(factor_cols, ic_summary, signal_col=signal_col, ic_col=ic_col)
    ordered = sorted(factor_cols, key=lambda factor: (-scores.get(factor, 0.0), factor))
    selected: list[str] = []
    for factor in ordered:
        if all(abs(float(correlation.loc[factor, existing])) <= max_abs_corr for existing in selected):
            selected.append(factor)
    return selected


def neutralize_factors_by_date(
    frame: pd.DataFrame,
    factor_cols: Sequence[str],
    neutralizer_cols: Sequence[str],
    *,
    date_col: str = "date",
    suffix: str = "_neutral",
    min_obs: int | None = None,
) -> pd.DataFrame:
    """Residualize factor columns against neutralizers within each date."""

    if not factor_cols:
        raise ValueError("factor_cols must not be empty")
    if not neutralizer_cols:
        raise ValueError("neutralizer_cols must not be empty")
    _require_columns(frame, [date_col, *factor_cols, *neutralizer_cols])

    output = frame.copy()
    working = frame.copy()
    working[date_col] = pd.to_datetime(working[date_col])
    min_obs = min_obs or len(neutralizer_cols) + 2
    if min_obs <= len(neutralizer_cols) + 1:
        raise ValueError("min_obs must exceed number of regression parameters")

    for factor in factor_cols:
        output[f"{factor}{suffix}"] = np.nan

    for _, group in working.groupby(date_col, sort=True):
        neutralizers = group.loc[:, neutralizer_cols].replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce")
        for factor in factor_cols:
            y = pd.to_numeric(group[factor], errors="coerce").replace([np.inf, -np.inf], np.nan)
            clean = pd.concat([y.rename(factor), neutralizers], axis=1).dropna()
            if len(clean) < min_obs:
                continue
            x_values = clean.loc[:, neutralizer_cols].to_numpy(dtype=float)
            x_matrix = np.column_stack([np.ones(len(clean)), x_values])
            y_values = clean[factor].to_numpy(dtype=float)
            beta, *_ = np.linalg.lstsq(x_matrix, y_values, rcond=None)
            residuals = y_values - x_matrix @ beta
            output.loc[clean.index, f"{factor}{suffix}"] = residuals
    return output


def factor_coverage(frame: pd.DataFrame, factor_cols: Sequence[str]) -> pd.DataFrame:
    """Summarize non-null coverage for factor columns."""

    _require_columns(frame, factor_cols)
    rows: list[dict[str, float | int | str]] = []
    total_rows = len(frame)
    for factor in factor_cols:
        available = int(pd.to_numeric(frame[factor], errors="coerce").replace([np.inf, -np.inf], np.nan).notna().sum())
        rows.append(
            {
                "factor": factor,
                "rows": int(total_rows),
                "available_rows": available,
                "missing_rows": int(total_rows - available),
                "coverage_rate": float(available / total_rows) if total_rows else np.nan,
            }
        )
    return pd.DataFrame(rows)


def mean_daily_factor_correlation(
    frame: pd.DataFrame,
    factor_cols: Sequence[str],
    *,
    date_col: str = "date",
    method: str = "spearman",
) -> pd.DataFrame:
    """Return the mean daily cross-sectional factor correlation matrix."""

    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    _require_columns(frame, [date_col, *factor_cols])
    if not factor_cols:
        raise ValueError("factor_cols must not be empty")

    matrices: list[pd.DataFrame] = []
    for _, group in frame.groupby(date_col, sort=True):
        clean = group.loc[:, factor_cols].replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce")
        if clean.dropna(how="all").shape[0] < 2:
            continue
        corr = clean.corr(method=method)
        if not corr.empty:
            matrices.append(corr)
    if not matrices:
        return pd.DataFrame(np.nan, index=list(factor_cols), columns=list(factor_cols))
    stacked = pd.concat(matrices, keys=range(len(matrices)))
    return stacked.groupby(level=1).mean().loc[list(factor_cols), list(factor_cols)]


def _factor_selection_scores(
    factor_cols: Sequence[str],
    ic_summary: pd.DataFrame | None,
    *,
    signal_col: str,
    ic_col: str,
) -> dict[str, float]:
    if ic_summary is None or ic_summary.empty:
        return {factor: float(len(factor_cols) - index) for index, factor in enumerate(factor_cols)}
    _require_columns(ic_summary, [signal_col, ic_col])
    lookup = ic_summary.drop_duplicates(signal_col, keep="first").set_index(signal_col)[ic_col]
    return {
        factor: abs(float(pd.to_numeric(lookup.get(factor, np.nan), errors="coerce")))
        if pd.notna(pd.to_numeric(lookup.get(factor, np.nan), errors="coerce"))
        else 0.0
        for factor in factor_cols
    }


def _daily_factor_ic(
    frame: pd.DataFrame,
    factor_cols: Sequence[str],
    label_col: str,
    *,
    date_col: str,
    method: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | pd.Timestamp]] = []
    for date, group in frame.groupby(date_col, sort=True):
        row: dict[str, float | pd.Timestamp] = {"date": pd.Timestamp(date)}
        for factor in factor_cols:
            clean = group[[factor, label_col]].replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce").dropna()
            if len(clean) < 2 or clean[factor].nunique() < 2 or clean[label_col].nunique() < 2:
                row[factor] = np.nan
            else:
                row[factor] = float(clean[factor].corr(clean[label_col], method=method))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=factor_cols)
    return pd.DataFrame(rows).set_index("date").loc[:, list(factor_cols)]


def _fallback_weight_rows(
    date: pd.Timestamp,
    factor_cols: Sequence[str],
    history_days: int,
    *,
    reason: str,
) -> list[dict[str, float | int | bool | pd.Timestamp | str]]:
    equal = 1.0 / len(factor_cols)
    return [
        {
            "date": pd.Timestamp(date),
            "factor": factor,
            "mean_rank_ic": np.nan,
            "weight": equal,
            "direction": 1,
            "history_days": int(history_days),
            "is_fallback": True,
            "fallback_reason": reason,
        }
        for factor in factor_cols
    ]


def _add_weighted_score(
    frame: pd.DataFrame,
    ranks: pd.DataFrame,
    factor_cols: Sequence[str],
    weights: Mapping[str, float],
    score_col: str,
    min_factors: int,
) -> pd.DataFrame:
    if min_factors <= 0:
        raise ValueError("min_factors must be positive")
    output = frame.copy()
    rank_cols = [f"{factor}_rank" for factor in factor_cols]
    weighted = pd.DataFrame(index=frame.index)
    for factor, rank_col in zip(factor_cols, rank_cols):
        weighted[factor] = ranks[rank_col] * float(weights.get(factor, 0.0))
    available = ranks.loc[:, rank_cols].notna().sum(axis=1)
    weight_sum = ranks.loc[:, rank_cols].notna().mul([float(weights.get(factor, 0.0)) for factor in factor_cols], axis=1).sum(axis=1)
    output[score_col] = weighted.sum(axis=1, min_count=1) / weight_sum.replace(0.0, np.nan)
    output.loc[available < min_factors, score_col] = np.nan
    return output


def _equal_rank_scores_from_ranks(ranks: pd.DataFrame, factor_cols: Sequence[str], *, min_factors: int) -> pd.Series:
    rank_cols = [f"{factor}_rank" for factor in factor_cols]
    available = ranks.loc[:, rank_cols].notna().sum(axis=1)
    scores = ranks.loc[:, rank_cols].mean(axis=1)
    scores.loc[available < min_factors] = np.nan
    return scores


def _normalize_pruning_plan(pruning_plan: pd.DataFrame) -> dict[int, dict[str, Any]]:
    if "fit_uses_eval_year" in pruning_plan.columns and _truthy(pruning_plan["fit_uses_eval_year"]).any():
        raise ValueError("pruning plan must be prior-fit; fit_uses_eval_year must be false")
    plan = pruning_plan.copy()
    plan["eval_year"] = pd.to_numeric(plan["eval_year"], errors="coerce")
    plan = plan.dropna(subset=["eval_year"]).copy()
    rules: dict[int, dict[str, Any]] = {}
    for eval_year, group in plan.groupby(plan["eval_year"].astype(int), sort=True):
        row = group.iloc[0]
        factors = _parse_factor_list(row.get("selected_factors", ""))
        if not factors:
            continue
        rules[int(eval_year)] = {
            "factors": factors,
            "directions": _parse_direction_map(row.get("selected_factor_directions", ""), factors),
        }
    return rules


def _parse_factor_list(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _parse_direction_map(value: object, factors: Sequence[str]) -> dict[str, int]:
    if isinstance(value, Mapping):
        raw = value
    else:
        text = str(value or "").strip()
        raw = json.loads(text) if text else {}
    return {factor: (-1 if float(raw.get(factor, 1)) < 0 else 1) for factor in factors}


def _factor_direction(factor: str, directions: Mapping[str, int | float]) -> float:
    direction = float(directions.get(factor, 1.0))
    if direction == 0:
        raise ValueError(f"factor direction must be non-zero: {factor}")
    return 1.0 if direction > 0 else -1.0


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
