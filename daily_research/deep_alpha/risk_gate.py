from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class RiskGateArtifact:
    mode: str
    global_threshold: float
    state_thresholds: Dict[str, float]
    candidate_thresholds: List[float]
    objective_rows: List[dict]
    group_thresholds: Dict[str, float] = field(default_factory=dict)


def _cross_sectional_signal(pivot: pd.DataFrame, rank_blend: float) -> pd.DataFrame:
    z = pivot.sub(pivot.mean(axis=1), axis=0).div(pivot.std(axis=1).replace(0, np.nan), axis=0)
    rank = pivot.rank(axis=1, pct=True)
    rank_centered = (rank - 0.5) * 2.0
    return z.fillna(0.0) * (1.0 - rank_blend) + rank_centered.fillna(0.0) * rank_blend


def build_return_score_frame(
    pred_df: pd.DataFrame,
    target_names: list[str],
    horizon_weights: dict[int, float],
    score_rank_blend: float,
    all_dates: pd.Index,
    all_stocks: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name in target_names:
        horizon = int(name.replace("fwd_excess_", "")) if name.startswith("fwd_excess_") else None
        if horizon is None:
            continue
        weight = float(horizon_weights.get(horizon, 0.0))
        if weight <= 0:
            continue
        pivot = pred_df.pivot(index="date", columns="stock", values=f"pred_{name}")
        signal = _cross_sectional_signal(pivot, score_rank_blend)
        frames.append(signal * weight)
    if not frames:
        return pd.DataFrame(index=all_dates, columns=all_stocks, dtype=float)
    score = sum(frame.reindex(index=all_dates, columns=all_stocks).fillna(0.0) for frame in frames)
    valid = pd.concat([frame.reindex(index=all_dates, columns=all_stocks).notna() for frame in frames]).groupby(level=0).max()
    return score.where(valid)


def _build_safe_rank_frame(
    pred_df: pd.DataFrame,
    all_dates: pd.Index,
    all_stocks: list[str],
) -> pd.DataFrame:
    downside = pred_df.pivot(index="date", columns="stock", values="pred_risk_downside_20")
    return downside.rank(axis=1, pct=True).reindex(index=all_dates, columns=all_stocks).fillna(0.5)


def _evaluate_threshold(
    merged: pd.DataFrame,
    holding_count: int,
    threshold: float,
) -> tuple[float, float, float]:
    merged = merged[merged["safe_rank"] >= threshold].copy()
    if merged.empty:
        return 0.0, 0.0, 0.0

    daily_returns = []
    daily_downside = []
    eligible_days = 0
    for _, group in merged.groupby("date"):
        sample = group.dropna(subset=["return_score", "true_fwd_excess_20", "true_risk_downside_20"])
        if sample.empty:
            daily_returns.append(0.0)
            daily_downside.append(0.0)
            continue
        chosen = sample.nlargest(holding_count, "return_score")
        eligible_days += 1
        daily_returns.append(float(chosen["true_fwd_excess_20"].mean()))
        daily_downside.append(float(chosen["true_risk_downside_20"].mean()))

    total_days = merged["date"].nunique()
    coverage = eligible_days / total_days if total_days > 0 else 0.0
    avg_return = float(np.mean(daily_returns)) if daily_returns else 0.0
    avg_downside = float(np.mean(daily_downside)) if daily_downside else 0.0
    return avg_return, avg_downside, coverage


def fit_state_risk_gate(
    train_pred_df: pd.DataFrame,
    state_frame: pd.DataFrame,
    target_names: list[str],
    horizon_weights: dict[int, float],
    score_rank_blend: float,
    holding_count: int,
    candidate_thresholds: list[float] | None = None,
    liquidity_bucket_frame: pd.DataFrame | None = None,
) -> RiskGateArtifact:
    if candidate_thresholds is None:
        candidate_thresholds = [0.0, 0.2, 0.35, 0.5, 0.65]
    candidate_thresholds = sorted(set(float(x) for x in candidate_thresholds))

    all_dates = pd.Index(sorted(pd.to_datetime(train_pred_df["date"]).unique()))
    all_stocks = sorted(train_pred_df["stock"].unique())
    return_score = build_return_score_frame(
        pred_df=train_pred_df,
        target_names=target_names,
        horizon_weights=horizon_weights,
        score_rank_blend=score_rank_blend,
        all_dates=all_dates,
        all_stocks=all_stocks,
    )
    safe_rank = _build_safe_rank_frame(train_pred_df, all_dates, all_stocks)
    state_df = state_frame.copy()
    if "Date" in state_df.columns:
        state_df = state_df.rename(columns={"Date": "date"})
    elif "date" not in state_df.columns:
        state_df = state_df.reset_index().rename(columns={state_df.index.name or "index": "date"})
    state_df["date"] = pd.to_datetime(state_df["date"])

    return_long = return_score.stack(future_stack=True).rename("return_score").reset_index()
    return_long.columns = ["date", "stock", "return_score"]
    safe_long = safe_rank.stack(future_stack=True).rename("safe_rank").reset_index()
    safe_long.columns = ["date", "stock", "safe_rank"]
    truth_cols = train_pred_df[["date", "stock", "true_fwd_excess_20", "true_risk_downside_20"]].copy()
    truth_cols["date"] = pd.to_datetime(truth_cols["date"])
    base = return_long.merge(safe_long, on=["date", "stock"], how="left")
    base = base.merge(truth_cols, on=["date", "stock"], how="left")
    base = base.merge(state_df[["date", "state_name"]], on="date", how="left")
    base["state_name"] = base["state_name"].fillna("unknown")
    if liquidity_bucket_frame is not None:
        bucket_long = (
            liquidity_bucket_frame.reindex(index=all_dates, columns=all_stocks)
            .stack(future_stack=True)
            .rename("liquidity_bucket")
            .reset_index()
        )
        bucket_long.columns = ["date", "stock", "liquidity_bucket"]
        base = base.merge(bucket_long, on=["date", "stock"], how="left")

    objective_rows: list[dict] = []
    best_global_threshold = 0.0
    best_global_objective = -1e18
    for threshold in candidate_thresholds:
        avg_return, avg_downside, coverage = _evaluate_threshold(base, holding_count, threshold)
        objective = avg_return
        objective_rows.append(
            {
                "scope": "global",
                "state_name": "global",
                "threshold": threshold,
                "avg_true_fwd_excess_20": avg_return,
                "avg_true_risk_downside_20": avg_downside,
                "coverage": coverage,
                "objective": objective,
            }
        )
        if objective > best_global_objective + 1e-12 or (
            abs(objective - best_global_objective) <= 1e-12 and threshold < best_global_threshold
        ):
            best_global_objective = objective
            best_global_threshold = threshold

    state_thresholds: dict[str, float] = {}
    for state_name, state_group in base.groupby("state_name"):
        if state_group["date"].nunique() < 10:
            state_thresholds[str(state_name)] = best_global_threshold
            continue
        best_threshold = best_global_threshold
        best_objective = -1e18
        for threshold in candidate_thresholds:
            avg_return, avg_downside, coverage = _evaluate_threshold(state_group, holding_count, threshold)
            objective = avg_return
            objective_rows.append(
                {
                    "scope": "state",
                    "state_name": str(state_name),
                    "threshold": threshold,
                    "avg_true_fwd_excess_20": avg_return,
                    "avg_true_risk_downside_20": avg_downside,
                    "coverage": coverage,
                    "objective": objective,
                }
            )
            if objective > best_objective + 1e-12 or (
                abs(objective - best_objective) <= 1e-12 and threshold < best_threshold
            ):
                best_objective = objective
                best_threshold = threshold
        state_thresholds[str(state_name)] = best_threshold

    group_thresholds: dict[str, float] = {}
    if liquidity_bucket_frame is not None:
        for (state_name, bucket), group in base.groupby(["state_name", "liquidity_bucket"], dropna=False):
            if pd.isna(bucket):
                continue
            state_name = str(state_name)
            bucket_int = int(bucket)
            fallback_threshold = float(state_thresholds.get(state_name, best_global_threshold))
            if group["date"].nunique() < 10:
                group_thresholds[f"{state_name}|{bucket_int}"] = fallback_threshold
                continue
            best_threshold = fallback_threshold
            best_objective = -1e18
            for threshold in candidate_thresholds:
                avg_return, avg_downside, coverage = _evaluate_threshold(group, holding_count, threshold)
                objective = avg_return
                objective_rows.append(
                    {
                        "scope": "state_liquidity",
                        "state_name": state_name,
                        "liquidity_bucket": bucket_int,
                        "threshold": threshold,
                        "avg_true_fwd_excess_20": avg_return,
                        "avg_true_risk_downside_20": avg_downside,
                        "coverage": coverage,
                        "objective": objective,
                    }
                )
                if objective > best_objective + 1e-12 or (
                    abs(objective - best_objective) <= 1e-12 and threshold < best_threshold
                ):
                    best_objective = objective
                    best_threshold = threshold
            group_thresholds[f"{state_name}|{bucket_int}"] = best_threshold

    return RiskGateArtifact(
        mode="state_liquidity_gate" if liquidity_bucket_frame is not None else "state_gate",
        global_threshold=best_global_threshold,
        state_thresholds=state_thresholds,
        group_thresholds=group_thresholds,
        candidate_thresholds=candidate_thresholds,
        objective_rows=objective_rows,
    )


def apply_state_risk_gate(
    score_frame: pd.DataFrame,
    pred_df: pd.DataFrame,
    state_frame: pd.DataFrame,
    artifact: RiskGateArtifact,
    liquidity_bucket_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    gated = score_frame.copy()
    safe_rank = _build_safe_rank_frame(
        pred_df=pred_df,
        all_dates=score_frame.index,
        all_stocks=list(score_frame.columns),
    )
    state_df = state_frame.copy()
    if "Date" in state_df.columns:
        state_df = state_df.rename(columns={"Date": "date"})
    elif "date" not in state_df.columns:
        state_df = state_df.reset_index().rename(columns={state_df.index.name or "index": "date"})
    state_df["date"] = pd.to_datetime(state_df["date"])
    state_map = state_df.set_index("date")["state_name"].to_dict()
    state_names = np.array([str(state_map.get(pd.Timestamp(dt), "unknown")) for dt in score_frame.index], dtype=object)
    threshold_arr = np.full(score_frame.shape, float(artifact.global_threshold), dtype=float)
    for state_name, threshold in artifact.state_thresholds.items():
        row_mask = state_names == str(state_name)
        if row_mask.any():
            threshold_arr[row_mask, :] = float(threshold)

    if artifact.mode == "state_liquidity_gate" and liquidity_bucket_frame is not None and artifact.group_thresholds:
        bucket_arr = liquidity_bucket_frame.reindex(index=score_frame.index, columns=score_frame.columns).to_numpy(dtype=float)
        for key, threshold in artifact.group_thresholds.items():
            state_name, bucket_text = key.rsplit("|", 1)
            row_mask = state_names == state_name
            if not row_mask.any():
                continue
            bucket_val = float(int(bucket_text))
            cell_mask = row_mask[:, None] & np.isfinite(bucket_arr) & (bucket_arr == bucket_val)
            threshold_arr[cell_mask] = float(threshold)

    threshold_frame = pd.DataFrame(threshold_arr, index=score_frame.index, columns=score_frame.columns)
    return gated.where(safe_rank >= threshold_frame)
