from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy.models import normalize_path20_cumulative_horizons


DEFAULT_PROXY_COST_BPS = 20.0
DEFAULT_PROXY_HIT_THRESHOLD_BPS = 10.0
DEFAULT_PROXY_DRAWDOWN_PENALTY = 0.10


def infer_decision_horizons(frame: pd.DataFrame) -> tuple[int, ...]:
    patterns = (
        r"^pred_decision_utility_(\d+)d$",
        r"^pred_cum_mu_(\d+)d$",
        r"^future_cum_excess_return_(\d+)d$",
    )
    horizons: list[int] = []
    for column in frame.columns:
        text = str(column)
        for pattern in patterns:
            match = re.match(pattern, text)
            if match is not None:
                horizons.append(int(match.group(1)))
                break
    if not horizons:
        return ()
    max_horizon = max(horizons)
    return normalize_path20_cumulative_horizons(tuple(sorted(set(horizons))), horizon=max_horizon)


def has_decision_score_columns(frame: pd.DataFrame) -> bool:
    horizons = tuple(
        sorted(
            int(match.group(1))
            for column in frame.columns
            for match in [re.match(r"^pred_decision_utility_(\d+)d$", str(column))]
            if match is not None
        )
    )
    required = {
        "pred_decision_score",
        "future_decision_score",
        "trade_utility_score",
        "pred_best_horizon",
        "future_best_horizon",
    }
    required.update(f"future_decision_utility_{horizon}d" for horizon in horizons)
    required.update(f"pred_hit_prob_{horizon}d" for horizon in horizons)
    required.update(f"future_hit_label_{horizon}d" for horizon in horizons)
    return bool(horizons) and required.issubset(frame.columns)


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").astype("float64").fillna(float(default))


def _drawdown_proxy(frame: pd.DataFrame, horizon: int) -> pd.Series:
    for column in (
        f"pred_aux_downside_floor_{int(horizon)}d",
        f"future_path_max_drawdown_{int(horizon)}d",
        "future_path_max_drawdown_20d",
    ):
        if column in frame.columns:
            return _numeric(frame, column, default=0.0)
    return pd.Series(0.0, index=frame.index, dtype="float64")


def _future_drawdown(frame: pd.DataFrame, horizon: int) -> pd.Series:
    for column in (f"future_path_max_drawdown_{int(horizon)}d", "future_path_max_drawdown_20d"):
        if column in frame.columns:
            return _numeric(frame, column, default=0.0)
    return pd.Series(0.0, index=frame.index, dtype="float64")


def _sigmoid(values: pd.Series) -> pd.Series:
    clipped = values.clip(lower=-60.0, upper=60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def add_path_proxy_decision_scores(
    frame: pd.DataFrame,
    *,
    cost_bps: float = DEFAULT_PROXY_COST_BPS,
    hit_threshold_bps: float = DEFAULT_PROXY_HIT_THRESHOLD_BPS,
    drawdown_penalty: float = DEFAULT_PROXY_DRAWDOWN_PENALTY,
    cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
) -> pd.DataFrame:
    """Return a copy with decision-score columns derived from path predictions when needed."""

    if has_decision_score_columns(frame):
        out = frame.copy()
        out["decision_score_source"] = out.get("decision_score_source", "model_decision_utility")
        return out

    inferred = infer_decision_horizons(frame)
    if cumulative_horizons is None:
        horizons = inferred
    else:
        max_hint = max([*inferred, *[int(item) for item in cumulative_horizons]] or [1])
        horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=max_hint)
    if not horizons:
        return frame.copy()

    missing = [
        column
        for horizon in horizons
        for column in (f"pred_cum_mu_{int(horizon)}d", f"future_cum_excess_return_{int(horizon)}d")
        if column not in frame.columns
    ]
    if missing:
        return frame.copy()

    out = frame.copy()
    max_horizon = max(int(item) for item in horizons)
    hit_threshold = float(hit_threshold_bps) / 10000.0
    cost = float(cost_bps) / 10000.0
    pred_utilities: list[pd.Series] = []
    future_utilities: list[pd.Series] = []

    for horizon in horizons:
        horizon_int = int(horizon)
        horizon_scale = math.sqrt(horizon_int / max(float(max_horizon), 1.0))
        pred_drawdown = _drawdown_proxy(out, horizon_int)
        future_drawdown = _future_drawdown(out, horizon_int)
        pred_utility = (
            _numeric(out, f"pred_cum_mu_{horizon_int}d")
            - cost
            - float(drawdown_penalty) * pred_drawdown.mul(-1.0).clip(lower=0.0) * horizon_scale
        )
        future_utility = (
            _numeric(out, f"future_cum_excess_return_{horizon_int}d")
            - cost
            - float(drawdown_penalty) * future_drawdown.mul(-1.0).clip(lower=0.0) * horizon_scale
        )
        out[f"pred_decision_utility_{horizon_int}d"] = pred_utility
        out[f"future_decision_utility_{horizon_int}d"] = future_utility
        out[f"pred_hit_prob_{horizon_int}d"] = _sigmoid((pred_utility - hit_threshold) * 100.0)
        out[f"future_hit_label_{horizon_int}d"] = (future_utility > hit_threshold).astype(int)
        pred_utilities.append(pred_utility)
        future_utilities.append(future_utility)

    pred_matrix = np.column_stack([series.to_numpy(dtype=float) for series in pred_utilities])
    future_matrix = np.column_stack([series.to_numpy(dtype=float) for series in future_utilities])
    horizon_values = np.asarray(horizons, dtype=int)
    pred_idx = np.nanargmax(np.where(np.isfinite(pred_matrix), pred_matrix, -np.inf), axis=1)
    future_idx = np.nanargmax(np.where(np.isfinite(future_matrix), future_matrix, -np.inf), axis=1)
    out["pred_best_horizon"] = horizon_values[pred_idx]
    out["future_best_horizon"] = horizon_values[future_idx]
    out["pred_decision_score"] = np.nanmax(pred_matrix, axis=1)
    out["trade_utility_score"] = out["pred_decision_score"]
    out["future_decision_score"] = np.nanmax(future_matrix, axis=1)
    out["decision_score_source"] = "path_proxy"
    return out
