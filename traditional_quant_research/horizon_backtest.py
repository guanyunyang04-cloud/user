"""Horizon-aligned portfolio simulation helpers.

The first protocol backtests use a forward-return column directly on each
rebalance date. That is useful for ranking diagnostics, but it can overstate
PnL when a multi-day label is evaluated on overlapping daily schedules. This
module builds explicit entry/exit baskets so multi-day horizons can be tested
with a non-overlapping holding-period interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .backtest_protocol import REBALANCE_PERIODS_PER_YEAR, select_rebalance_dates
from .metrics import annualized_return, max_drawdown, sharpe_ratio, volatility

PERIOD_SUMMARY_FREQ_LABELS = {
    "M": "monthly",
    "Q": "quarterly",
}


@dataclass(frozen=True)
class HorizonBacktestResult:
    """Trade-level result for a horizon-aligned Top-N simulation."""

    trades: pd.DataFrame
    summary: dict[str, Any]


def horizon_aligned_top_n_backtest(
    frame: pd.DataFrame,
    signal_col: str,
    *,
    horizon: int,
    top_n: int,
    fee_bps: float,
    rebalance_frequency: str = "daily",
    date_col: str = "date",
    code_col: str = "code",
    open_col: str = "open",
    close_col: str = "close",
    tradeable_col: str | None = "is_tradeable",
    non_overlapping: bool = True,
    buffer_multiplier: float = 1.0,
    execution_constraints: bool = False,
    limit_threshold: float = 0.095,
    capital_col: str | None = None,
    group_col: str | None = None,
    max_group_weight: float | None = None,
    exposure_penalty_cols: Sequence[str] | str | None = None,
    exposure_penalty_strength: float = 0.0,
) -> HorizonBacktestResult:
    """Run an equal-weight Top-N long-only simulation aligned to a forward horizon.

    Signals are observed at date ``t`` after close. Entry uses the next
    tradeable open for each selected security. Exit uses the close after
    ``horizon`` tradeable observations, matching
    :func:`research_panel.build_factor_label_panel` label semantics.

    With ``non_overlapping=True`` a new basket is opened only after the prior
    basket's exit date, preventing a daily 5-day or 20-day label from being
    counted as independent daily PnL.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if buffer_multiplier < 1.0:
        raise ValueError("buffer_multiplier must be at least 1.0")
    if limit_threshold <= 0:
        raise ValueError("limit_threshold must be positive")
    if max_group_weight is not None and (max_group_weight <= 0 or max_group_weight > 1):
        raise ValueError("max_group_weight must be between 0 and 1")
    if group_col is None and max_group_weight is not None:
        raise ValueError("group_col is required when max_group_weight is set")
    exposure_cols = _normalize_optional_columns(exposure_penalty_cols)
    if exposure_penalty_strength < 0:
        raise ValueError("exposure_penalty_strength must be non-negative")
    if rebalance_frequency not in REBALANCE_PERIODS_PER_YEAR:
        raise ValueError(f"unsupported frequency: {rebalance_frequency}")

    required = [date_col, code_col, signal_col, open_col, close_col]
    if capital_col is not None:
        required.append(capital_col)
    if group_col is not None:
        required.append(group_col)
    required.extend(exposure_cols)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if capital_col is not None:
        _validate_capital_column(frame[capital_col], capital_col=capital_col)

    candidates = _build_candidate_returns(
        frame,
        signal_col,
        horizon=horizon,
        date_col=date_col,
        code_col=code_col,
        open_col=open_col,
        close_col=close_col,
        tradeable_col=tradeable_col if tradeable_col in frame.columns else None,
        execution_constraints=execution_constraints,
        limit_threshold=limit_threshold,
        capital_col=capital_col,
        group_col=group_col,
        exposure_cols=exposure_cols,
    )
    if candidates.empty:
        return HorizonBacktestResult(
            trades=pd.DataFrame(),
            summary=summarize_horizon_returns(
                [],
                horizon=horizon,
                rebalance_frequency=rebalance_frequency,
                top_n=top_n,
                fee_bps=fee_bps,
                non_overlapping=non_overlapping,
                buffer_multiplier=buffer_multiplier,
                execution_constraints=execution_constraints,
                limit_threshold=limit_threshold,
                capital_col=capital_col,
                group_col=group_col,
                max_group_weight=max_group_weight,
                exposure_penalty_cols=exposure_cols,
                exposure_penalty_strength=exposure_penalty_strength,
            ),
        )

    selected_dates = set(select_rebalance_dates(candidates[date_col], rebalance_frequency))
    candidates = candidates.loc[candidates[date_col].isin(selected_dates)]
    if candidates.empty:
        return HorizonBacktestResult(
            trades=pd.DataFrame(),
            summary=summarize_horizon_returns(
                [],
                horizon=horizon,
                rebalance_frequency=rebalance_frequency,
                top_n=top_n,
                fee_bps=fee_bps,
                non_overlapping=non_overlapping,
                buffer_multiplier=buffer_multiplier,
                execution_constraints=execution_constraints,
                limit_threshold=limit_threshold,
                capital_col=capital_col,
                group_col=group_col,
                max_group_weight=max_group_weight,
                exposure_penalty_cols=exposure_cols,
                exposure_penalty_strength=exposure_penalty_strength,
            ),
        )

    fee = fee_bps / 10000.0
    previous_weights: dict[str, float] = {}
    last_exit_date: pd.Timestamp | None = None
    rows: list[dict[str, Any]] = []

    for signal_date, group in candidates.groupby(date_col, sort=True):
        signal_date = pd.Timestamp(signal_date)
        if non_overlapping and last_exit_date is not None and signal_date < last_exit_date:
            continue
        capital_scale = _signal_date_capital_scale(group, capital_col=capital_col)

        picks = select_buffered_top_n(
            group,
            signal_col,
            code_col=code_col,
            top_n=top_n,
            previous_codes=set(previous_weights),
            buffer_multiplier=buffer_multiplier,
            group_col=group_col,
            max_group_weight=max_group_weight,
            exposure_penalty_cols=exposure_cols,
            exposure_penalty_strength=exposure_penalty_strength,
        )
        requested_holdings = int(len(picks))
        filled = picks.dropna(subset=["horizon_return", "entry_date", "exit_date"])
        if capital_scale == 0.0:
            cash_path = picks.dropna(subset=["entry_date", "exit_date"])
            if cash_path.empty:
                previous_weights = {}
                continue
            entry_dates = pd.to_datetime(cash_path["entry_date"])
            exit_dates = pd.to_datetime(cash_path["exit_date"])
            exit_date = pd.Timestamp(exit_dates.max())
            turnover = sum(abs(weight) for weight in previous_weights.values())
            cost = turnover * fee
            rows.append(
                {
                    "signal_date": signal_date,
                    "entry_date": pd.Timestamp(entry_dates.min()),
                    "exit_date": exit_date,
                    "gross_return": 0.0,
                    "cost": float(cost),
                    "net_return": -float(cost),
                    "turnover": float(turnover),
                    "holdings": 0,
                    "requested_holdings": requested_holdings,
                    "capital_scale": 0.0,
                    "blocked_entry_count": 0,
                    "entry_limit_up_count": 0,
                    "exit_delayed_count": 0,
                    "exit_limit_down_count": 0,
                    "buffer_multiplier": float(buffer_multiplier),
                    "execution_constraints": bool(execution_constraints),
                    "limit_threshold": float(limit_threshold),
                    "holding_days": int(max(1, len(pd.date_range(entry_dates.min(), exit_dates.max(), freq="B")))),
                    "codes": "",
                }
            )
            previous_weights = {}
            last_exit_date = exit_date
            continue
        if filled.empty:
            continue

        capital_slots = requested_holdings if execution_constraints else len(filled)
        weight = capital_scale / capital_slots
        weights = {str(code): weight for code in filled[code_col]}
        turnover = sum(
            abs(weights.get(code, 0.0) - previous_weights.get(code, 0.0))
            for code in set(weights) | set(previous_weights)
        )
        gross_return = float(pd.to_numeric(filled["horizon_return"], errors="coerce").sum() * weight)
        cost = turnover * fee
        entry_dates = pd.to_datetime(filled["entry_date"])
        exit_dates = pd.to_datetime(filled["exit_date"])
        exit_date = pd.Timestamp(exit_dates.max())
        rows.append(
            {
                "signal_date": signal_date,
                "entry_date": pd.Timestamp(entry_dates.min()),
                "exit_date": exit_date,
                "gross_return": gross_return,
                "cost": float(cost),
                "net_return": gross_return - cost,
                "turnover": float(turnover),
                "holdings": int(len(filled)),
                "requested_holdings": requested_holdings,
                "capital_scale": float(capital_scale),
                "blocked_entry_count": _sum_bool_column(picks, "entry_blocked"),
                "entry_limit_up_count": _sum_bool_column(picks, "entry_blocked_by_limit_up"),
                "exit_delayed_count": _sum_bool_column(filled, "exit_delayed"),
                "exit_limit_down_count": _sum_bool_column(filled, "exit_blocked_by_limit_down"),
                "buffer_multiplier": float(buffer_multiplier),
                "execution_constraints": bool(execution_constraints),
                "limit_threshold": float(limit_threshold),
                "holding_days": int(max(1, len(pd.date_range(entry_dates.min(), exit_dates.max(), freq="B")))),
                "codes": ",".join(str(code) for code in filled[code_col].tolist()),
            }
        )
        previous_weights = weights
        last_exit_date = exit_date

    trades = pd.DataFrame(rows)
    summary = summarize_horizon_returns(
        trades["net_return"].tolist() if not trades.empty else [],
        horizon=horizon,
        rebalance_frequency=rebalance_frequency,
        top_n=top_n,
        fee_bps=fee_bps,
        non_overlapping=non_overlapping,
        buffer_multiplier=buffer_multiplier,
        execution_constraints=execution_constraints,
        limit_threshold=limit_threshold,
        capital_col=capital_col,
        group_col=group_col,
        max_group_weight=max_group_weight,
        exposure_penalty_cols=exposure_cols,
        exposure_penalty_strength=exposure_penalty_strength,
    )
    summary.update(
        {
            "periods": int(len(trades)),
            "mean_turnover": float(trades["turnover"].mean()) if not trades.empty else np.nan,
            "mean_gross_return": float(trades["gross_return"].mean()) if not trades.empty else np.nan,
            "mean_net_return": float(trades["net_return"].mean()) if not trades.empty else np.nan,
            "mean_holdings": float(trades["holdings"].mean()) if not trades.empty else np.nan,
            "mean_requested_holdings": float(trades["requested_holdings"].mean()) if not trades.empty and "requested_holdings" in trades.columns else np.nan,
            "mean_capital_scale": float(trades["capital_scale"].mean()) if not trades.empty and "capital_scale" in trades.columns else np.nan,
            "blocked_entry_count": int(trades["blocked_entry_count"].sum()) if not trades.empty and "blocked_entry_count" in trades.columns else 0,
            "entry_limit_up_count": int(trades["entry_limit_up_count"].sum()) if not trades.empty and "entry_limit_up_count" in trades.columns else 0,
            "exit_delayed_count": int(trades["exit_delayed_count"].sum()) if not trades.empty and "exit_delayed_count" in trades.columns else 0,
            "exit_limit_down_count": int(trades["exit_limit_down_count"].sum()) if not trades.empty and "exit_limit_down_count" in trades.columns else 0,
            "group_col": group_col or "",
            "max_group_weight": float(max_group_weight) if max_group_weight is not None else np.nan,
        }
    )
    return HorizonBacktestResult(trades=trades, summary=summary)


def select_buffered_top_n(
    group: pd.DataFrame,
    signal_col: str,
    *,
    code_col: str,
    top_n: int,
    previous_codes: set[str],
    buffer_multiplier: float,
    group_col: str | None = None,
    max_group_weight: float | None = None,
    exposure_penalty_cols: Sequence[str] | str | None = None,
    exposure_penalty_strength: float = 0.0,
) -> pd.DataFrame:
    """Select Top-N while allowing prior holdings to survive inside a rank buffer."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if buffer_multiplier < 1.0:
        raise ValueError("buffer_multiplier must be at least 1.0")
    if max_group_weight is not None and (max_group_weight <= 0 or max_group_weight > 1):
        raise ValueError("max_group_weight must be between 0 and 1")
    if group_col is None and max_group_weight is not None:
        raise ValueError("group_col is required when max_group_weight is set")
    if group_col is not None and group_col not in group.columns:
        raise ValueError(f"group column not found: {group_col}")
    exposure_cols = _normalize_optional_columns(exposure_penalty_cols)
    if exposure_penalty_strength < 0:
        raise ValueError("exposure_penalty_strength must be non-negative")
    missing_exposure_cols = [column for column in exposure_cols if column not in group.columns]
    if missing_exposure_cols:
        raise ValueError(f"exposure penalty columns not found: {missing_exposure_cols}")
    ranked = group.sort_values(signal_col, ascending=False, kind="mergesort").copy()
    if ranked.empty:
        return ranked
    if exposure_cols and exposure_penalty_strength > 0:
        return _select_buffered_top_n_exposure_penalized(
            ranked,
            signal_col=signal_col,
            code_col=code_col,
            top_n=top_n,
            previous_codes=previous_codes,
            buffer_multiplier=buffer_multiplier,
            group_col=group_col,
            max_group_weight=max_group_weight,
            exposure_cols=exposure_cols,
            exposure_penalty_strength=exposure_penalty_strength,
        )
    if group_col is None or max_group_weight is None:
        if buffer_multiplier == 1.0 or not previous_codes:
            return ranked.head(top_n)
        return _select_buffered_top_n_unconstrained(
            ranked,
            code_col=code_col,
            top_n=top_n,
            previous_codes=previous_codes,
            buffer_multiplier=buffer_multiplier,
        )

    return _select_buffered_top_n_group_constrained(
        ranked,
        code_col=code_col,
        top_n=top_n,
        previous_codes=previous_codes,
        buffer_multiplier=buffer_multiplier,
        group_col=group_col,
        max_group_weight=max_group_weight,
    )


def _select_buffered_top_n_unconstrained(
    ranked: pd.DataFrame,
    *,
    code_col: str,
    top_n: int,
    previous_codes: set[str],
    buffer_multiplier: float,
) -> pd.DataFrame:
    if buffer_multiplier == 1.0 or not previous_codes:
        return ranked.head(top_n)

    buffer_size = max(top_n, int(np.ceil(top_n * buffer_multiplier)))
    buffered = ranked.head(buffer_size)
    keep_prior = buffered.loc[buffered[code_col].astype(str).isin(previous_codes)]
    selected_codes: list[str] = []
    selected_indices: list[Any] = []

    for index, row in keep_prior.iterrows():
        code = str(row[code_col])
        if code not in selected_codes:
            selected_codes.append(code)
            selected_indices.append(index)
        if len(selected_indices) >= top_n:
            return ranked.loc[selected_indices]

    for index, row in ranked.iterrows():
        code = str(row[code_col])
        if code in selected_codes:
            continue
        selected_codes.append(code)
        selected_indices.append(index)
        if len(selected_indices) >= top_n:
            break
    return ranked.loc[selected_indices]


def _select_buffered_top_n_exposure_penalized(
    ranked: pd.DataFrame,
    *,
    signal_col: str,
    code_col: str,
    top_n: int,
    previous_codes: set[str],
    buffer_multiplier: float,
    group_col: str | None,
    max_group_weight: float | None,
    exposure_cols: Sequence[str],
    exposure_penalty_strength: float,
) -> pd.DataFrame:
    work = ranked.copy()
    for column in exposure_cols:
        work[column] = pd.to_numeric(work[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    work[signal_col] = pd.to_numeric(work[signal_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[signal_col]).reset_index(drop=False).rename(columns={"index": "_original_index"})
    if work.empty:
        return ranked.iloc[0:0]

    max_group_count: int | None = None
    if group_col is not None and max_group_weight is not None:
        max_group_count = max(1, int(np.floor(top_n * max_group_weight + 1e-12)))

    codes = work[code_col].astype(str).to_numpy()
    scores = work[signal_col].to_numpy(dtype=float)
    exposures = work.loc[:, list(exposure_cols)].to_numpy(dtype=float)
    original_indices = work["_original_index"].to_numpy()
    groups = np.array([_group_key(value) for value in work[group_col]], dtype=object) if group_col is not None else None

    selected_codes: list[str] = []
    selected_positions: list[int] = []
    selected_mask = np.zeros(len(work), dtype=bool)
    exposure_sums = np.zeros(len(exposure_cols), dtype=float)
    group_counts: dict[str, int] = {}

    def allowed_mask(base_mask: np.ndarray) -> np.ndarray:
        mask = base_mask & ~selected_mask
        if max_group_count is None or groups is None:
            return mask
        allowed = np.array([group_counts.get(str(group), 0) < max_group_count for group in groups], dtype=bool)
        return mask & allowed

    def select_from(base_mask: np.ndarray) -> bool:
        mask = allowed_mask(base_mask)
        positions = np.flatnonzero(mask)
        if len(positions) == 0:
            return False
        next_count = len(selected_positions) + 1
        candidate_means = (exposure_sums + exposures[positions]) / next_count
        penalties = np.abs(candidate_means).sum(axis=1)
        adjusted = scores[positions] - exposure_penalty_strength * penalties
        best_position = int(positions[int(np.argmax(adjusted))])
        selected_mask[best_position] = True
        selected_positions.append(best_position)
        selected_codes.append(str(codes[best_position]))
        exposure_sums[:] = exposure_sums + exposures[best_position]
        if max_group_count is not None and groups is not None:
            group_key = str(groups[best_position])
            group_counts[group_key] = group_counts.get(group_key, 0) + 1
        return True

    if previous_codes and buffer_multiplier > 1.0:
        buffer_size = max(top_n, int(np.ceil(top_n * buffer_multiplier)))
        prior_mask = np.zeros(len(work), dtype=bool)
        prior_mask[: min(buffer_size, len(work))] = True
        prior_mask &= np.isin(codes, list(previous_codes))
        while len(selected_positions) < top_n and select_from(prior_mask):
            pass

    all_mask = np.ones(len(work), dtype=bool)
    while len(selected_positions) < top_n:
        if not select_from(all_mask):
            break
    selected_original_indices = [original_indices[position] for position in selected_positions]
    return ranked.loc[selected_original_indices]


def _select_buffered_top_n_group_constrained(
    ranked: pd.DataFrame,
    *,
    code_col: str,
    top_n: int,
    previous_codes: set[str],
    buffer_multiplier: float,
    group_col: str,
    max_group_weight: float,
) -> pd.DataFrame:
    max_group_count = max(1, int(np.floor(top_n * max_group_weight + 1e-12)))
    group_counts: dict[str, int] = {}
    selected_codes: list[str] = []
    selected_indices: list[Any] = []

    def try_add(index: Any, row: pd.Series) -> bool:
        code = str(row[code_col])
        if code in selected_codes:
            return False
        group_key = _group_key(row[group_col])
        if group_counts.get(group_key, 0) >= max_group_count:
            return False
        selected_codes.append(code)
        selected_indices.append(index)
        group_counts[group_key] = group_counts.get(group_key, 0) + 1
        return True

    buffer_size = max(top_n, int(np.ceil(top_n * buffer_multiplier)))
    buffered = ranked.head(buffer_size)
    if previous_codes and buffer_multiplier > 1.0:
        keep_prior = buffered.loc[buffered[code_col].astype(str).isin(previous_codes)]
        for index, row in keep_prior.iterrows():
            try_add(index, row)
            if len(selected_indices) >= top_n:
                return ranked.loc[selected_indices]

    for index, row in ranked.iterrows():
        try_add(index, row)
        if len(selected_indices) >= top_n:
            break
    return ranked.loc[selected_indices]


def _build_candidate_returns(
    frame: pd.DataFrame,
    signal_col: str,
    *,
    horizon: int,
    date_col: str,
    code_col: str,
    open_col: str,
    close_col: str,
    tradeable_col: str | None,
    execution_constraints: bool,
    limit_threshold: float,
    capital_col: str | None,
    group_col: str | None,
    exposure_cols: Sequence[str],
) -> pd.DataFrame:
    if execution_constraints:
        return _build_candidate_returns_with_execution_constraints(
            frame,
            signal_col,
            horizon=horizon,
            date_col=date_col,
            code_col=code_col,
            open_col=open_col,
            close_col=close_col,
            tradeable_col=tradeable_col,
            limit_threshold=limit_threshold,
            capital_col=capital_col,
            group_col=group_col,
            exposure_cols=exposure_cols,
        )

    path = frame.copy()
    path[date_col] = pd.to_datetime(path[date_col])
    numeric_columns = [signal_col, open_col, close_col]
    if capital_col is not None:
        numeric_columns.append(capital_col)
    for column in numeric_columns:
        path[column] = pd.to_numeric(path[column], errors="coerce")
    path = path.replace([np.inf, -np.inf], np.nan)
    if tradeable_col is not None:
        path = path.loc[path[tradeable_col].astype(bool)]
    path = path.sort_values([code_col, date_col]).reset_index(drop=True)

    grouped = path.groupby(code_col, sort=False)
    path["entry_date"] = grouped[date_col].shift(-1)
    path["entry_open"] = grouped[open_col].shift(-1)
    path["exit_date"] = grouped[date_col].shift(-horizon)
    path["exit_close"] = grouped[close_col].shift(-horizon)
    path["horizon_return"] = path["exit_close"] / path["entry_open"] - 1.0

    required = [date_col, code_col, signal_col, "entry_date", "entry_open", "exit_date", "exit_close", "horizon_return"]
    if capital_col is not None:
        required.append(capital_col)
    if group_col is not None:
        required.append(group_col)
    required.extend(exposure_cols)
    return path.loc[:, required].dropna(subset=[signal_col, "entry_open", "exit_close", "horizon_return"]).sort_values([date_col, code_col]).reset_index(drop=True)


def _build_candidate_returns_with_execution_constraints(
    frame: pd.DataFrame,
    signal_col: str,
    *,
    horizon: int,
    date_col: str,
    code_col: str,
    open_col: str,
    close_col: str,
    tradeable_col: str | None,
    limit_threshold: float,
    capital_col: str | None,
    group_col: str | None,
    exposure_cols: Sequence[str],
) -> pd.DataFrame:
    path = frame.copy()
    path[date_col] = pd.to_datetime(path[date_col])
    numeric_columns = [signal_col, open_col, close_col]
    if capital_col is not None:
        numeric_columns.append(capital_col)
    numeric_columns.extend(exposure_cols)
    for column in numeric_columns:
        path[column] = pd.to_numeric(path[column], errors="coerce")
    path = path.replace([np.inf, -np.inf], np.nan).sort_values([code_col, date_col]).reset_index(drop=True)
    if tradeable_col is None:
        path["_is_tradeable_for_execution"] = True
        tradeable_col = "_is_tradeable_for_execution"
    else:
        path[tradeable_col] = path[tradeable_col].astype(bool)

    global_dates = pd.DatetimeIndex(path[date_col].dropna().unique()).sort_values()
    if len(global_dates) <= horizon:
        return pd.DataFrame(columns=_candidate_return_columns(date_col, code_col, signal_col, capital_col=capital_col, group_col=group_col))
    date_to_idx = {pd.Timestamp(date): index for index, date in enumerate(global_dates)}
    next_date = {
        pd.Timestamp(global_dates[index]): pd.Timestamp(global_dates[index + 1])
        for index in range(len(global_dates) - 1)
    }
    idx_to_date = {index: pd.Timestamp(date) for index, date in enumerate(global_dates)}

    grouped = path.groupby(code_col, sort=False)
    path["prev_close"] = grouped[close_col].shift(1)
    path["limit_up_at_open"] = _limit_move(path[open_col], path["prev_close"]) >= limit_threshold
    path["limit_down_at_close"] = _limit_move(path[close_col], path["prev_close"]) <= -limit_threshold
    path["date_index"] = path[date_col].map(date_to_idx)

    base_columns = [date_col, code_col, signal_col]
    if capital_col is not None:
        base_columns.append(capital_col)
    if group_col is not None:
        base_columns.append(group_col)
    base_columns.extend(exposure_cols)
    base = path.loc[path[tradeable_col] & path[signal_col].notna(), base_columns].copy()
    base["entry_date"] = base[date_col].map(next_date)
    entry = path.loc[
        :,
        [date_col, code_col, open_col, tradeable_col, "limit_up_at_open", "date_index"],
    ].rename(
        columns={
            date_col: "entry_date",
            open_col: "entry_open",
            tradeable_col: "entry_tradeable",
            "limit_up_at_open": "entry_limit_up",
            "date_index": "entry_date_index",
        }
    )
    candidates = base.merge(entry, on=[code_col, "entry_date"], how="left")
    candidates["entry_has_row"] = candidates["entry_open"].notna()
    candidates["entry_tradeable"] = candidates["entry_tradeable"].eq(True)
    candidates["entry_limit_up"] = candidates["entry_limit_up"].eq(True)
    candidates["entry_blocked_by_limit_up"] = candidates["entry_has_row"] & candidates["entry_tradeable"] & candidates["entry_limit_up"]
    candidates["entry_blocked"] = ~(
        candidates["entry_has_row"]
        & candidates["entry_tradeable"]
        & candidates["entry_open"].notna()
        & ~candidates["entry_limit_up"]
    )
    candidates["exit_target_index"] = candidates["entry_date_index"] + horizon - 1
    candidates["exit_target_date"] = candidates["exit_target_index"].map(idx_to_date)
    candidates = _attach_next_sellable_exit(
        candidates,
        path,
        date_col=date_col,
        code_col=code_col,
        close_col=close_col,
        tradeable_col=tradeable_col,
        date_to_idx=date_to_idx,
    )
    candidates["horizon_return"] = candidates["exit_close"] / candidates["entry_open"] - 1.0
    candidates.loc[candidates["entry_blocked"], ["entry_open", "exit_date", "exit_close", "horizon_return"]] = np.nan

    return (
        candidates.loc[:, _candidate_return_columns(date_col, code_col, signal_col, capital_col=capital_col, group_col=group_col, exposure_cols=exposure_cols)]
        .sort_values([date_col, code_col])
        .reset_index(drop=True)
    )


def _attach_next_sellable_exit(
    candidates: pd.DataFrame,
    path: pd.DataFrame,
    *,
    date_col: str,
    code_col: str,
    close_col: str,
    tradeable_col: str,
    date_to_idx: dict[pd.Timestamp, int],
) -> pd.DataFrame:
    output = candidates.copy()
    output["exit_date"] = pd.NaT
    output["exit_close"] = np.nan
    output["exit_date_index"] = np.nan
    output["exit_delayed"] = False
    output["exit_blocked_by_limit_down"] = False
    sellable = path.loc[path[tradeable_col] & path[close_col].notna() & ~path["limit_down_at_close"]].copy()
    target_limit_down = path.loc[:, [code_col, date_col, "limit_down_at_close"]].rename(
        columns={date_col: "exit_target_date", "limit_down_at_close": "target_limit_down"}
    )
    output = output.merge(target_limit_down, on=[code_col, "exit_target_date"], how="left")
    output["target_limit_down"] = output["target_limit_down"].eq(True)
    sellable_by_code = {
        code: group.sort_values(date_col)
        for code, group in sellable.groupby(code_col, sort=False)
    }

    for code, index in output.groupby(code_col, sort=False).groups.items():
        code_sellable = sellable_by_code.get(code)
        if code_sellable is None or code_sellable.empty:
            continue
        sell_dates = pd.to_datetime(code_sellable[date_col]).to_numpy(dtype="datetime64[ns]")
        sell_closes = pd.to_numeric(code_sellable[close_col], errors="coerce").to_numpy(dtype=float)
        target_dates = pd.to_datetime(output.loc[index, "exit_target_date"]).to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(sell_dates, target_dates, side="left")
        valid = positions < len(sell_dates)
        if not valid.any():
            continue
        valid_index = pd.Index(index)[valid]
        valid_positions = positions[valid]
        resolved_dates = pd.to_datetime(sell_dates[valid_positions])
        output.loc[valid_index, "exit_date"] = resolved_dates
        output.loc[valid_index, "exit_close"] = sell_closes[valid_positions]
        output.loc[valid_index, "exit_date_index"] = [date_to_idx[pd.Timestamp(date)] for date in resolved_dates]
    output["exit_delayed"] = pd.to_numeric(output["exit_date_index"], errors="coerce") > pd.to_numeric(output["exit_target_index"], errors="coerce")
    output["exit_blocked_by_limit_down"] = output["target_limit_down"] & output["exit_delayed"]
    return output


def _candidate_return_columns(
    date_col: str,
    code_col: str,
    signal_col: str,
    *,
    capital_col: str | None = None,
    group_col: str | None = None,
    exposure_cols: Sequence[str] = (),
) -> list[str]:
    columns = [
        date_col,
        code_col,
        signal_col,
        "entry_date",
        "entry_open",
        "exit_date",
        "exit_close",
        "horizon_return",
        "entry_blocked",
        "entry_blocked_by_limit_up",
        "exit_delayed",
        "exit_blocked_by_limit_down",
    ]
    if capital_col is not None:
        columns.insert(3, capital_col)
    if group_col is not None:
        columns.insert(3, group_col)
    for column in reversed(tuple(exposure_cols)):
        if column not in columns:
            columns.insert(3, column)
    return columns


def _group_key(value: Any) -> str:
    if pd.isna(value):
        return "__missing__"
    text = str(value).strip()
    return text if text else "__missing__"


def _limit_move(price: pd.Series, prev_close: pd.Series) -> pd.Series:
    move = price / prev_close - 1.0
    return move.where(prev_close > 0)


def _sum_bool_column(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].fillna(False).astype(bool).sum())


def _validate_capital_column(values: pd.Series, *, capital_col: str) -> None:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.isna().any():
        raise ValueError(f"capital column contains missing or non-numeric values: {capital_col}")
    if numeric.empty:
        raise ValueError(f"capital column has no valid values: {capital_col}")
    if (numeric < 0).any() or (numeric > 1).any():
        raise ValueError(f"capital column must be between 0 and 1: {capital_col}")


def _signal_date_capital_scale(group: pd.DataFrame, *, capital_col: str | None) -> float:
    if capital_col is None:
        return 1.0
    values = pd.to_numeric(group[capital_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return 1.0
    if values.nunique() > 1:
        raise ValueError(f"capital column must be constant within each signal date: {capital_col}")
    return float(values.iloc[0])


def _normalize_optional_columns(values: Sequence[str] | str | None) -> tuple[str, ...]:
    if values is None:
        return ()
    raw_values = values.split(",") if isinstance(values, str) else values
    output: list[str] = []
    for value in raw_values:
        text = str(value).strip()
        if text and text not in output:
            output.append(text)
    return tuple(output)


def summarize_horizon_returns(
    returns: list[float] | tuple[float, ...],
    *,
    horizon: int,
    rebalance_frequency: str,
    top_n: int,
    fee_bps: float,
    non_overlapping: bool,
    buffer_multiplier: float = 1.0,
    execution_constraints: bool = False,
    limit_threshold: float = 0.095,
    capital_col: str | None = None,
    group_col: str | None = None,
    max_group_weight: float | None = None,
    exposure_penalty_cols: Sequence[str] | str | None = None,
    exposure_penalty_strength: float = 0.0,
) -> dict[str, Any]:
    periods_per_year = horizon_periods_per_year(
        horizon=horizon,
        rebalance_frequency=rebalance_frequency,
        non_overlapping=non_overlapping,
    )
    values = [float(value) for value in returns if pd.notna(value)]
    return {
        "horizon": int(horizon),
        "rebalance_frequency": rebalance_frequency,
        "top_n": int(top_n),
        "fee_bps": float(fee_bps),
        "non_overlapping": bool(non_overlapping),
        "buffer_multiplier": float(buffer_multiplier),
        "execution_constraints": bool(execution_constraints),
        "limit_threshold": float(limit_threshold),
        "capital_col": capital_col or "",
        "group_col": group_col or "",
        "max_group_weight": float(max_group_weight) if max_group_weight is not None else np.nan,
        "exposure_penalty_cols": ",".join(_normalize_optional_columns(exposure_penalty_cols)),
        "exposure_penalty_strength": float(exposure_penalty_strength),
        "periods_per_year": float(periods_per_year),
        "annualized_return": float(annualized_return(values, periods_per_year=periods_per_year)),
        "volatility": float(volatility(values, periods_per_year=periods_per_year)),
        "sharpe": float(sharpe_ratio(values, periods_per_year=periods_per_year)),
        "max_drawdown": float(max_drawdown(values)),
    }


def yearly_horizon_summary(
    trades: pd.DataFrame,
    *,
    horizon: int,
    rebalance_frequency: str,
    top_n: int,
    fee_bps: float,
    non_overlapping: bool,
    buffer_multiplier: float = 1.0,
    execution_constraints: bool = False,
    limit_threshold: float = 0.095,
    signal_col: str | None = None,
) -> pd.DataFrame:
    """Summarize horizon-aligned trade returns by exit year."""

    columns = [
        "year",
        "horizon",
        "rebalance_frequency",
        "top_n",
        "fee_bps",
        "non_overlapping",
        "buffer_multiplier",
        "execution_constraints",
        "limit_threshold",
        "periods_per_year",
        "annualized_return",
        "volatility",
        "sharpe",
        "max_drawdown",
        "periods",
        "mean_turnover",
        "mean_gross_return",
        "mean_net_return",
    ]
    if signal_col is not None:
        columns.insert(0, "signal")
    if trades.empty:
        return pd.DataFrame(columns=columns)
    if "exit_date" not in trades.columns or "net_return" not in trades.columns:
        raise ValueError("trades must contain exit_date and net_return columns")

    output_rows: list[dict[str, Any]] = []
    work = trades.copy()
    work["exit_date"] = pd.to_datetime(work["exit_date"])
    work["year"] = work["exit_date"].dt.year
    for year, group in work.groupby("year", sort=True):
        summary = summarize_horizon_returns(
            group["net_return"].tolist(),
            horizon=horizon,
            rebalance_frequency=rebalance_frequency,
            top_n=top_n,
            fee_bps=fee_bps,
            non_overlapping=non_overlapping,
            buffer_multiplier=buffer_multiplier,
            execution_constraints=execution_constraints,
            limit_threshold=limit_threshold,
        )
        summary.update(
            {
                "year": int(year),
                "periods": int(len(group)),
                "mean_turnover": float(group["turnover"].mean()) if "turnover" in group.columns else np.nan,
                "mean_gross_return": float(group["gross_return"].mean()) if "gross_return" in group.columns else np.nan,
                "mean_net_return": float(group["net_return"].mean()),
            }
        )
        if signal_col is not None:
            summary["signal"] = signal_col
        output_rows.append(summary)
    return pd.DataFrame(output_rows).loc[:, columns]


def period_horizon_summary(
    trades: pd.DataFrame,
    *,
    period: str,
    horizon: int,
    rebalance_frequency: str,
    top_n: int,
    fee_bps: float,
    non_overlapping: bool,
    buffer_multiplier: float = 1.0,
    execution_constraints: bool = False,
    limit_threshold: float = 0.095,
    signal_col: str | None = None,
) -> pd.DataFrame:
    """Summarize horizon-aligned trade returns by exit month or quarter."""

    if period not in PERIOD_SUMMARY_FREQ_LABELS:
        raise ValueError(f"unsupported period: {period}")
    columns = [
        "period",
        "period_type",
        "horizon",
        "rebalance_frequency",
        "top_n",
        "fee_bps",
        "non_overlapping",
        "buffer_multiplier",
        "execution_constraints",
        "limit_threshold",
        "periods_per_year",
        "annualized_return",
        "volatility",
        "sharpe",
        "max_drawdown",
        "periods",
        "mean_turnover",
        "mean_gross_return",
        "mean_net_return",
    ]
    if signal_col is not None:
        columns.insert(0, "signal")
    if trades.empty:
        return pd.DataFrame(columns=columns)
    if "exit_date" not in trades.columns or "net_return" not in trades.columns:
        raise ValueError("trades must contain exit_date and net_return columns")

    output_rows: list[dict[str, Any]] = []
    work = trades.copy()
    work["exit_date"] = pd.to_datetime(work["exit_date"])
    work["_period"] = work["exit_date"].dt.to_period(period)
    period_type = PERIOD_SUMMARY_FREQ_LABELS[period]
    for period_value, group in work.groupby("_period", sort=True):
        summary = summarize_horizon_returns(
            group["net_return"].tolist(),
            horizon=horizon,
            rebalance_frequency=rebalance_frequency,
            top_n=top_n,
            fee_bps=fee_bps,
            non_overlapping=non_overlapping,
            buffer_multiplier=buffer_multiplier,
            execution_constraints=execution_constraints,
            limit_threshold=limit_threshold,
        )
        summary.update(
            {
                "period": str(period_value),
                "period_type": period_type,
                "periods": int(len(group)),
                "mean_turnover": float(group["turnover"].mean()) if "turnover" in group.columns else np.nan,
                "mean_gross_return": float(group["gross_return"].mean()) if "gross_return" in group.columns else np.nan,
                "mean_net_return": float(group["net_return"].mean()),
            }
        )
        if signal_col is not None:
            summary["signal"] = signal_col
        output_rows.append(summary)
    return pd.DataFrame(output_rows).loc[:, columns]


def horizon_periods_per_year(
    *,
    horizon: int,
    rebalance_frequency: str,
    non_overlapping: bool,
) -> float:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if rebalance_frequency not in REBALANCE_PERIODS_PER_YEAR:
        raise ValueError(f"unsupported frequency: {rebalance_frequency}")
    base = float(REBALANCE_PERIODS_PER_YEAR[rebalance_frequency])
    if not non_overlapping:
        return base
    return min(base, 252.0 / float(horizon))
