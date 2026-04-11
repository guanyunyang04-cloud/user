from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from daily_research.deep_alpha.pipeline_utils import (
    RESEARCH_TIME_UNIT_CALENDAR_MONTHS,
    resolve_window_dates,
)

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None


RETURN_TARGET_PREFIX = "fwd_excess_"
RISK_TARGET_NAME = "risk_downside_20"

POLICY_V1_GATE_MULTIPLIER = 2.0
POLICY_V1_MIN_GROSS_EXPOSURE = 0.35
POLICY_V1_MAX_GROSS_EXPOSURE = 1.0
POLICY_V1_WEIGHT_POWER = 1.35
POLICY_V1_HOLD_BOOST = 0.30
POLICY_V1_CANDIDATE_MULTIPLIER = 2.5

POLICY_V2_GATE_MULTIPLIER = 2.2
POLICY_V2_MIN_GROSS_EXPOSURE = 0.88
POLICY_V2_MAX_GROSS_EXPOSURE = 0.98
POLICY_V2_WEIGHT_POWER = 1.18
POLICY_V2_HOLD_BOOST = 0.18
POLICY_V2_CANDIDATE_MULTIPLIER = 2.2

POLICY_V2_FAMILY_CONFIGS: Dict[str, Dict[str, float]] = {
    "policy_v2": {
        "gate_multiplier": POLICY_V2_GATE_MULTIPLIER,
        "min_gross_exposure": POLICY_V2_MIN_GROSS_EXPOSURE,
        "max_gross_exposure": POLICY_V2_MAX_GROSS_EXPOSURE,
        "weight_power": POLICY_V2_WEIGHT_POWER,
        "hold_boost": POLICY_V2_HOLD_BOOST,
        "candidate_multiplier": POLICY_V2_CANDIDATE_MULTIPLIER,
        "selection_rank_blend": 0.45,
        "cash_model_alpha": 1.6,
        "build_weight_mix": 0.65,
        "build_gate_mix": 0.25,
        "build_hold_mix": 0.10,
    },
    "policy_v2a": {
        "gate_multiplier": 1.90,
        "min_gross_exposure": 0.90,
        "max_gross_exposure": 0.98,
        "weight_power": 1.28,
        "hold_boost": 0.20,
        "candidate_multiplier": 1.80,
        "selection_rank_blend": 0.40,
        "cash_model_alpha": 1.5,
        "build_weight_mix": 0.72,
        "build_gate_mix": 0.18,
        "build_hold_mix": 0.10,
    },
    "policy_v2b": {
        "gate_multiplier": 2.00,
        "min_gross_exposure": 0.90,
        "max_gross_exposure": 0.96,
        "weight_power": 1.24,
        "hold_boost": 0.22,
        "candidate_multiplier": 2.00,
        "selection_rank_blend": 0.42,
        "cash_model_alpha": 1.4,
        "build_weight_mix": 0.68,
        "build_gate_mix": 0.18,
        "build_hold_mix": 0.14,
    },
    "policy_v2c": {
        "gate_multiplier": 1.90,
        "min_gross_exposure": 0.90,
        "max_gross_exposure": 0.98,
        "weight_power": 1.22,
        "hold_boost": 0.30,
        "candidate_multiplier": 1.80,
        "selection_rank_blend": 0.48,
        "cash_model_alpha": 1.5,
        "build_weight_mix": 0.55,
        "build_gate_mix": 0.15,
        "build_hold_mix": 0.30,
    },
    "policy_v4a": {
        "gate_multiplier": 2.00,
        "min_gross_exposure": 0.90,
        "max_gross_exposure": 0.96,
        "weight_power": 1.16,
        "hold_boost": 0.20,
        "candidate_multiplier": 2.20,
        "selection_rank_blend": 0.44,
        "cash_model_alpha": 2.0,
        "build_weight_mix": 0.60,
        "build_gate_mix": 0.24,
        "build_hold_mix": 0.16,
    },
    "policy_v4b": {
        "gate_multiplier": 2.10,
        "min_gross_exposure": 0.90,
        "max_gross_exposure": 0.98,
        "weight_power": 1.10,
        "hold_boost": 0.16,
        "candidate_multiplier": 2.60,
        "selection_rank_blend": 0.46,
        "cash_model_alpha": 1.8,
        "build_weight_mix": 0.48,
        "build_gate_mix": 0.30,
        "build_hold_mix": 0.22,
    },
}
POLICY_V2_METHODS = frozenset(POLICY_V2_FAMILY_CONFIGS)

POLICY_V3_GATE_MULTIPLIER = 1.8
POLICY_V3_MIN_GROSS_EXPOSURE = 0.90
POLICY_V3_MAX_GROSS_EXPOSURE = 0.98
POLICY_V3_WEIGHT_POWER = 1.28
POLICY_V3_HOLD_BOOST = 0.16
POLICY_V3_CANDIDATE_MULTIPLIER = 1.6


@dataclass
class ScoreHeadArtifact:
    method: str
    feature_columns: List[str]
    task_weights: Dict[str, float]
    model: object | None
    extra: Dict[str, Any] = field(default_factory=dict)


def _cross_sectional_standardize(series: pd.Series) -> pd.Series:
    grouped = series.groupby(level=0)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return ((series - mean) / std).fillna(0.0)


def _cross_sectional_rank(series: pd.Series) -> pd.Series:
    return series.groupby(level=0).rank(pct=True).fillna(0.5)


def _build_feature_frame(pred_df: pd.DataFrame, target_names: List[str]) -> pd.DataFrame:
    indexed = pred_df.set_index(["date", "stock"]).sort_index()
    feature_dict: Dict[str, pd.Series] = {}
    for target_name in target_names:
        pred_col = f"pred_{target_name}"
        if pred_col not in indexed.columns:
            continue
        pred_series = indexed[pred_col].astype(float)
        feature_dict[f"{pred_col}_z"] = _cross_sectional_standardize(pred_series)
        feature_dict[f"{pred_col}_rank"] = _cross_sectional_rank(pred_series)
    return pd.DataFrame(feature_dict).reset_index()


def compute_task_rankic_summary(pred_df: pd.DataFrame, target_names: List[str]) -> pd.DataFrame:
    rows = []
    for target_name in target_names:
        pred_col = f"pred_{target_name}"
        true_col = f"true_{target_name}"
        for dt, g in pred_df.groupby("date"):
            if len(g) < 5:
                continue
            corr = spearmanr(g[pred_col], g[true_col], nan_policy="omit").correlation
            rows.append(
                {
                    "date": dt,
                    "target": target_name,
                    "rankic": float(corr) if corr is not None and np.isfinite(corr) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["target", "rankic_mean", "rankic_std", "rankic_ir"])
    summary = out.groupby("target")["rankic"].agg(["mean", "std"]).reset_index()
    summary["rankic_ir"] = summary["mean"] / summary["std"].replace(0, np.nan)
    return summary.rename(columns={"mean": "rankic_mean", "std": "rankic_std"})


def _slice_recent_window(
    pred_df: pd.DataFrame,
    window_days: int | None,
    *,
    research_time_unit: str = RESEARCH_TIME_UNIT_CALENDAR_MONTHS,
    window_months: int | None = None,
) -> pd.DataFrame:
    if (not window_days or window_days <= 0) and (not window_months or window_months <= 0):
        return pred_df
    dates = sorted(pd.to_datetime(pred_df["date"]).unique())
    if not dates:
        return pred_df
    selected = set(
        resolve_window_dates(
            pd.Index(dates),
            pd.Timestamp(dates[-1]),
            int(window_days or 0),
            research_time_unit=research_time_unit,
            window_months=window_months,
        )
    )
    out = pred_df[pd.to_datetime(pred_df["date"]).isin(selected)].copy()
    return out if not out.empty else pred_df


def derive_adaptive_task_weights(
    pred_df: pd.DataFrame,
    target_names: List[str],
    recent_window_days: int | None = None,
    *,
    research_time_unit: str = RESEARCH_TIME_UNIT_CALENDAR_MONTHS,
    recent_window_months: int | None = None,
) -> Dict[str, float]:
    summary = compute_task_rankic_summary(
        _slice_recent_window(
            pred_df,
            recent_window_days,
            research_time_unit=research_time_unit,
            window_months=recent_window_months,
        ),
        target_names,
    )
    weights: Dict[str, float] = {}
    for target_name in target_names:
        row = summary.loc[summary["target"] == target_name]
        rankic = float(row["rankic_mean"].iloc[0]) if not row.empty else 0.0
        if target_name.startswith(RETURN_TARGET_PREFIX):
            weights[target_name] = max(rankic, 0.01)
        elif target_name == RISK_TARGET_NAME:
            weights[target_name] = max(rankic, 0.05)
        else:
            weights[target_name] = max(abs(rankic), 0.01)
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


def _build_utility_series(pred_df: pd.DataFrame, task_weights: Dict[str, float]) -> pd.Series:
    indexed = pred_df.set_index(["date", "stock"]).sort_index()
    utility = pd.Series(0.0, index=indexed.index, dtype=float)
    for target_name, weight in task_weights.items():
        true_col = f"true_{target_name}"
        if true_col not in indexed.columns:
            continue
        utility = utility.add(indexed[true_col].astype(float) * float(weight), fill_value=0.0)
    utility.name = "utility_target"
    return utility


def _build_training_target(pred_df: pd.DataFrame, task_weights: Dict[str, float]) -> pd.Series:
    target = _cross_sectional_rank(_build_utility_series(pred_df, task_weights))
    target.name = "score_target"
    return target


def _build_confidence_target(pred_df: pd.DataFrame, task_weights: Dict[str, float]) -> pd.Series:
    indexed = pred_df.set_index(["date", "stock"]).sort_index()
    confidence = pd.Series(0.0, index=indexed.index, dtype=float)
    breakout_series: pd.Series | None = None
    clean_breakout_series: pd.Series | None = None
    for target_name, weight in task_weights.items():
        true_col = f"true_{target_name}"
        if true_col not in indexed.columns:
            continue
        series = indexed[true_col].astype(float)
        if target_name.startswith("fwd_excess_"):
            confidence = confidence.add(series.clip(lower=0.0) * float(weight), fill_value=0.0)
        elif target_name == RISK_TARGET_NAME:
            confidence = confidence.add(series.clip(upper=0.0) * float(weight), fill_value=0.0)
        elif target_name.startswith("event_breakout_"):
            breakout_series = series.clip(lower=0.0, upper=1.0)
            confidence = confidence.add(breakout_series * max(float(weight), 0.05), fill_value=0.0)
        elif target_name.startswith("event_clean_breakout_"):
            clean_breakout_series = series.clip(lower=0.0, upper=1.0)
            confidence = confidence.add(clean_breakout_series * max(float(weight), 0.10) * 1.5, fill_value=0.0)
    if breakout_series is not None and clean_breakout_series is not None:
        false_breakout = (breakout_series - clean_breakout_series).clip(lower=0.0)
        confidence = confidence.sub(false_breakout * 0.25, fill_value=0.0)
    target = _cross_sectional_rank(confidence)
    target.name = "confidence_target"
    return target


def _build_topk_binary_target(signal: pd.Series, top_k: int) -> pd.Series:
    top_k = max(int(top_k), 1)
    out_parts: list[pd.Series] = []
    for _, group in signal.groupby(level=0):
        ordered = group.sort_values(ascending=False)
        selected = ordered.iloc[: min(len(ordered), top_k)]
        part = pd.Series(0.0, index=group.index, dtype=float)
        part.loc[selected.index] = 1.0
        out_parts.append(part)
    out = pd.concat(out_parts).sort_index() if out_parts else pd.Series(dtype=float)
    out.name = "gate_target"
    return out


def _build_weight_target(utility_target: pd.Series, gate_target: pd.Series) -> pd.Series:
    out_parts: list[pd.Series] = []
    for dt, utility_group in utility_target.groupby(level=0):
        gate_group = gate_target.loc[utility_group.index].fillna(0.0)
        selected_mask = gate_group > 0.5
        selected_utility = utility_group.loc[selected_mask].clip(lower=0.0)
        part = pd.Series(0.0, index=utility_group.index, dtype=float)
        if selected_utility.empty or float(selected_utility.sum()) <= 0.0:
            selected_index = gate_group.loc[selected_mask].index
            if len(selected_index) > 0:
                part.loc[selected_index] = 1.0 / float(len(selected_index))
        else:
            normalized = selected_utility / float(selected_utility.sum())
            part.loc[normalized.index] = normalized.astype(float)
        part.name = dt
        out_parts.append(part)
    out = pd.concat(out_parts).sort_index() if out_parts else pd.Series(dtype=float)
    out.name = "weight_target"
    return out


def _build_hold_target(pred_df: pd.DataFrame, task_weights: Dict[str, float]) -> pd.Series:
    indexed = pred_df.set_index(["date", "stock"]).sort_index()
    persistence = pd.Series(0.0, index=indexed.index, dtype=float)
    for target_name, weight in task_weights.items():
        true_col = f"true_{target_name}"
        if true_col not in indexed.columns:
            continue
        series = indexed[true_col].astype(float)
        if target_name.startswith("fwd_excess_"):
            try:
                horizon = int(target_name.replace("fwd_excess_", ""))
            except Exception:
                horizon = 0
            horizon_scale = 1.15 if horizon >= 5 else 0.75
            persistence = persistence.add(series.clip(lower=0.0) * float(weight) * horizon_scale, fill_value=0.0)
        elif target_name == RISK_TARGET_NAME:
            persistence = persistence.add(series.clip(upper=0.0) * float(weight) * 0.75, fill_value=0.0)
        elif target_name.startswith("event_clean_breakout_"):
            persistence = persistence.add(series.clip(lower=0.0, upper=1.0) * max(float(weight), 0.10) * 1.25, fill_value=0.0)
        elif target_name.startswith("event_breakout_"):
            persistence = persistence.add(series.clip(lower=0.0, upper=1.0) * max(float(weight), 0.05) * 0.50, fill_value=0.0)
    target = _cross_sectional_rank(persistence)
    target.name = "hold_target"
    return target


def _group_top_mean(series: pd.Series, top_n: int) -> pd.Series:
    top_n = max(int(top_n), 1)
    return series.groupby(level=0).apply(lambda group: float(group.sort_values(ascending=False).head(min(len(group), top_n)).mean()))


def _group_top_std(series: pd.Series, top_n: int) -> pd.Series:
    top_n = max(int(top_n), 1)
    return series.groupby(level=0).apply(
        lambda group: float(group.sort_values(ascending=False).head(min(len(group), top_n)).std(ddof=0) or 0.0)
    )


def _group_positive_share(series: pd.Series) -> pd.Series:
    return series.groupby(level=0).apply(lambda group: float((group > 0.0).mean()))


def _group_above_threshold_share(series: pd.Series, threshold: float) -> pd.Series:
    return series.groupby(level=0).apply(lambda group: float((group >= float(threshold)).mean()))


def _build_policy_cash_target(
    pred_df: pd.DataFrame,
    utility_target: pd.Series,
    *,
    holding_count: int,
    min_gross_exposure: float,
    max_gross_exposure: float,
) -> pd.Series:
    indexed = pred_df.set_index(["date", "stock"]).sort_index()
    downside = indexed.get(f"true_{RISK_TARGET_NAME}", pd.Series(0.0, index=indexed.index, dtype=float)).astype(float)
    clean_breakout = pd.Series(0.0, index=indexed.index, dtype=float)
    for column in indexed.columns:
        if str(column).startswith("true_event_clean_breakout_"):
            clean_breakout = clean_breakout.add(indexed[column].astype(float), fill_value=0.0)
    date_signal: dict[pd.Timestamp, float] = {}
    top_n = max(int(holding_count), 3)
    for dt, utility_group in utility_target.groupby(level=0):
        ordered = utility_group.sort_values(ascending=False).head(min(len(utility_group), top_n))
        if ordered.empty:
            date_signal[pd.Timestamp(dt)] = float(min_gross_exposure)
            continue
        opportunity = float(ordered.clip(lower=0.0).mean())
        downside_penalty = float((-downside.loc[ordered.index].clip(upper=0.0)).mean()) if len(ordered.index) > 0 else 0.0
        breakout_bonus = float(clean_breakout.loc[ordered.index].clip(lower=0.0).mean()) if len(ordered.index) > 0 else 0.0
        date_signal[pd.Timestamp(dt)] = opportunity + 0.35 * breakout_bonus - 0.50 * downside_penalty
    signal_series = pd.Series(date_signal, dtype=float).sort_index()
    ranked = signal_series.rank(pct=True).fillna(0.5)
    gross = float(min_gross_exposure) + (float(max_gross_exposure) - float(min_gross_exposure)) * ranked
    gross.name = "cash_target"
    return gross.astype(float)


def _build_policy_cash_target_v2(
    pred_df: pd.DataFrame,
    utility_target: pd.Series,
    *,
    holding_count: int,
    min_gross_exposure: float,
    max_gross_exposure: float,
) -> pd.Series:
    indexed = pred_df.set_index(["date", "stock"]).sort_index()
    downside = indexed.get(f"true_{RISK_TARGET_NAME}", pd.Series(0.0, index=indexed.index, dtype=float)).astype(float)
    breakout = pd.Series(0.0, index=indexed.index, dtype=float)
    clean_breakout = pd.Series(0.0, index=indexed.index, dtype=float)
    for column in indexed.columns:
        if str(column).startswith("true_event_breakout_"):
            breakout = breakout.add(indexed[column].astype(float), fill_value=0.0)
        if str(column).startswith("true_event_clean_breakout_"):
            clean_breakout = clean_breakout.add(indexed[column].astype(float), fill_value=0.0)

    top_n = max(int(holding_count), 3)
    signal_rows: dict[pd.Timestamp, float] = {}
    downside_rows: dict[pd.Timestamp, float] = {}
    for dt, utility_group in utility_target.groupby(level=0):
        ordered = utility_group.sort_values(ascending=False).head(min(len(utility_group), top_n))
        if ordered.empty:
            signal_rows[pd.Timestamp(dt)] = 0.0
            downside_rows[pd.Timestamp(dt)] = 1.0
            continue
        top1 = float(ordered.head(1).clip(lower=0.0).mean())
        top2 = float(ordered.head(min(len(ordered), 2)).clip(lower=0.0).mean())
        top5 = float(ordered.clip(lower=0.0).mean())
        breadth = float((ordered > 0.0).mean())
        downside_penalty = float((-downside.loc[ordered.index].clip(upper=0.0)).mean()) if len(ordered.index) > 0 else 0.0
        breakout_bonus = float(clean_breakout.loc[ordered.index].clip(lower=0.0).mean()) if len(ordered.index) > 0 else 0.0
        false_breakout = float((breakout.loc[ordered.index] - clean_breakout.loc[ordered.index]).clip(lower=0.0).mean()) if len(ordered.index) > 0 else 0.0
        concentration_penalty = max(top1 - top5, 0.0)
        signal_rows[pd.Timestamp(dt)] = (
            1.20 * top1
            + 0.80 * top2
            + 0.45 * top5
            + 0.10 * breadth
            + 0.25 * breakout_bonus
            - 0.85 * downside_penalty
            - 0.20 * false_breakout
            - 0.20 * concentration_penalty
        )
        downside_rows[pd.Timestamp(dt)] = downside_penalty

    signal_series = pd.Series(signal_rows, dtype=float).sort_index()
    downside_series = pd.Series(downside_rows, dtype=float).sort_index()
    signal_rank = signal_series.rank(pct=True).fillna(0.5)
    downside_rank = downside_series.rank(pct=True).fillna(0.5)
    composite = (0.70 * signal_rank + 0.30 * (1.0 - downside_rank)).clip(0.0, 1.0)
    gross = float(min_gross_exposure) + (float(max_gross_exposure) - float(min_gross_exposure)) * composite
    aggressive_mask = signal_rank >= 0.80
    defensive_mask = (signal_rank <= 0.35) | (downside_rank >= 0.70)
    gross.loc[aggressive_mask] = float(max_gross_exposure)
    gross.loc[defensive_mask] = float(min_gross_exposure)
    gross.name = "cash_target"
    return gross.astype(float)


def _build_policy_cash_target_v3(
    pred_df: pd.DataFrame,
    utility_target: pd.Series,
    *,
    holding_count: int,
    min_gross_exposure: float,
    max_gross_exposure: float,
) -> pd.Series:
    indexed = pred_df.set_index(["date", "stock"]).sort_index()
    downside = indexed.get(f"true_{RISK_TARGET_NAME}", pd.Series(0.0, index=indexed.index, dtype=float)).astype(float)
    breakout = pd.Series(0.0, index=indexed.index, dtype=float)
    clean_breakout = pd.Series(0.0, index=indexed.index, dtype=float)
    for column in indexed.columns:
        if str(column).startswith("true_event_breakout_"):
            breakout = breakout.add(indexed[column].astype(float), fill_value=0.0)
        if str(column).startswith("true_event_clean_breakout_"):
            clean_breakout = clean_breakout.add(indexed[column].astype(float), fill_value=0.0)

    top_n = max(int(holding_count), 3)
    signal_rows: dict[pd.Timestamp, float] = {}
    downside_rows: dict[pd.Timestamp, float] = {}
    for dt, utility_group in utility_target.groupby(level=0):
        ordered = utility_group.sort_values(ascending=False).head(min(len(utility_group), top_n))
        if ordered.empty:
            signal_rows[pd.Timestamp(dt)] = 0.0
            downside_rows[pd.Timestamp(dt)] = 1.0
            continue
        top1 = float(ordered.head(1).clip(lower=0.0).mean())
        top2 = float(ordered.head(min(len(ordered), 2)).clip(lower=0.0).mean())
        topn_mean = float(ordered.clip(lower=0.0).mean())
        breadth = float((ordered > 0.0).mean())
        concentration = max(top1 - topn_mean, 0.0)
        downside_penalty = float((-downside.loc[ordered.index].clip(upper=0.0)).mean()) if len(ordered.index) > 0 else 0.0
        breakout_bonus = float(clean_breakout.loc[ordered.index].clip(lower=0.0).mean()) if len(ordered.index) > 0 else 0.0
        false_breakout = float((breakout.loc[ordered.index] - clean_breakout.loc[ordered.index]).clip(lower=0.0).mean()) if len(ordered.index) > 0 else 0.0
        signal_rows[pd.Timestamp(dt)] = (
            1.35 * top1
            + 0.85 * top2
            + 0.35 * topn_mean
            + 0.12 * breadth
            + 0.30 * breakout_bonus
            + 0.18 * concentration
            - 0.95 * downside_penalty
            - 0.30 * false_breakout
        )
        downside_rows[pd.Timestamp(dt)] = downside_penalty

    signal_series = pd.Series(signal_rows, dtype=float).sort_index()
    downside_series = pd.Series(downside_rows, dtype=float).sort_index()
    signal_rank = signal_series.rank(pct=True).fillna(0.5)
    downside_rank = downside_series.rank(pct=True).fillna(0.5)
    composite = (0.78 * signal_rank + 0.22 * (1.0 - downside_rank)).clip(0.0, 1.0)
    gross = float(min_gross_exposure) + (float(max_gross_exposure) - float(min_gross_exposure)) * composite
    aggressive_mask = signal_rank >= 0.82
    defensive_mask = (signal_rank <= 0.30) | (downside_rank >= 0.72)
    gross.loc[aggressive_mask] = float(max_gross_exposure)
    gross.loc[defensive_mask] = float(min_gross_exposure)
    gross.name = "cash_target"
    return gross.astype(float)


def _build_policy_cash_feature_frame(
    selection_signal: pd.Series,
    gate_signal: pd.Series,
    weight_signal: pd.Series,
    hold_signal: pd.Series,
    *,
    top_n: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "selection_top_mean": _group_top_mean(selection_signal, top_n),
            "selection_top_std": _group_top_std(selection_signal, top_n),
            "gate_top_mean": _group_top_mean(gate_signal, top_n),
            "gate_positive_share": _group_positive_share(gate_signal),
            "weight_top_mean": _group_top_mean(weight_signal, top_n),
            "weight_top_std": _group_top_std(weight_signal, top_n),
            "hold_top_mean": _group_top_mean(hold_signal, top_n),
            "hold_top_std": _group_top_std(hold_signal, top_n),
        }
    ).fillna(0.0)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    return frame.sort_index()


def _build_policy_cash_feature_frame_v2(
    selection_signal: pd.Series,
    gate_signal: pd.Series,
    weight_signal: pd.Series,
    hold_signal: pd.Series,
    *,
    top_n: int,
) -> pd.DataFrame:
    selection_top1 = _group_top_mean(selection_signal, 1)
    selection_top2 = _group_top_mean(selection_signal, 2)
    selection_topn = _group_top_mean(selection_signal, max(top_n, 3))
    frame = pd.DataFrame(
        {
            "selection_top1": selection_top1,
            "selection_top2": selection_top2,
            "selection_topn": selection_topn,
            "selection_top_gap": selection_top1 - selection_topn,
            "selection_positive_share": _group_positive_share(selection_signal),
            "gate_top_mean": _group_top_mean(gate_signal, top_n),
            "gate_top_std": _group_top_std(gate_signal, top_n),
            "gate_high_share": _group_above_threshold_share(gate_signal, 0.70),
            "weight_top_mean": _group_top_mean(weight_signal, top_n),
            "weight_top_std": _group_top_std(weight_signal, top_n),
            "weight_high_share": _group_above_threshold_share(weight_signal, 0.70),
            "hold_top_mean": _group_top_mean(hold_signal, top_n),
            "hold_top_std": _group_top_std(hold_signal, top_n),
            "hold_high_share": _group_above_threshold_share(hold_signal, 0.70),
        }
    ).fillna(0.0)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    return frame.sort_index()


def _build_policy_cash_feature_frame_v3(
    selection_signal: pd.Series,
    gate_signal: pd.Series,
    weight_signal: pd.Series,
    hold_signal: pd.Series,
    *,
    top_n: int,
) -> pd.DataFrame:
    selection_top1 = _group_top_mean(selection_signal, 1)
    selection_top2 = _group_top_mean(selection_signal, 2)
    selection_topn = _group_top_mean(selection_signal, max(top_n, 3))
    weight_top1 = _group_top_mean(weight_signal, 1)
    weight_topn = _group_top_mean(weight_signal, max(top_n, 3))
    frame = pd.DataFrame(
        {
            "selection_top1": selection_top1,
            "selection_top2": selection_top2,
            "selection_topn": selection_topn,
            "selection_gap": selection_top1 - selection_topn,
            "selection_positive_share": _group_positive_share(selection_signal),
            "gate_top_mean": _group_top_mean(gate_signal, top_n),
            "gate_top_std": _group_top_std(gate_signal, top_n),
            "gate_high_share": _group_above_threshold_share(gate_signal, 0.72),
            "weight_top1": weight_top1,
            "weight_topn": weight_topn,
            "weight_gap": weight_top1 - weight_topn,
            "weight_high_share": _group_above_threshold_share(weight_signal, 0.72),
            "hold_top_mean": _group_top_mean(hold_signal, top_n),
            "hold_top_std": _group_top_std(hold_signal, top_n),
            "hold_high_share": _group_above_threshold_share(hold_signal, 0.68),
        }
    ).fillna(0.0)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    return frame.sort_index()


def _build_policy_candidate_target_v3(
    pred_df: pd.DataFrame,
    utility_target: pd.Series,
    *,
    holding_count: int,
    max_candidates: int,
) -> pd.Series:
    indexed = pred_df.set_index(["date", "stock"]).sort_index()
    downside = indexed.get(f"true_{RISK_TARGET_NAME}", pd.Series(0.0, index=indexed.index, dtype=float)).astype(float)
    top_n = max(int(max_candidates), int(holding_count), 3)
    breadth_rows: dict[pd.Timestamp, float] = {}
    concentration_rows: dict[pd.Timestamp, float] = {}
    downside_rows: dict[pd.Timestamp, float] = {}
    for dt, utility_group in utility_target.groupby(level=0):
        ordered = utility_group.sort_values(ascending=False).head(min(len(utility_group), top_n))
        if ordered.empty:
            breadth_rows[pd.Timestamp(dt)] = 0.0
            concentration_rows[pd.Timestamp(dt)] = 1.0
            downside_rows[pd.Timestamp(dt)] = 1.0
            continue
        top1 = float(ordered.head(1).clip(lower=0.0).mean())
        topn_mean = float(ordered.clip(lower=0.0).mean())
        breadth_rows[pd.Timestamp(dt)] = float((ordered > 0.0).mean())
        concentration_rows[pd.Timestamp(dt)] = max(top1 - topn_mean, 0.0)
        downside_rows[pd.Timestamp(dt)] = float((-downside.loc[ordered.index].clip(upper=0.0)).mean()) if len(ordered.index) > 0 else 0.0

    breadth_series = pd.Series(breadth_rows, dtype=float).sort_index()
    concentration_series = pd.Series(concentration_rows, dtype=float).sort_index()
    downside_series = pd.Series(downside_rows, dtype=float).sort_index()
    breadth_rank = breadth_series.rank(pct=True).fillna(0.5)
    concentration_rank = concentration_series.rank(pct=True).fillna(0.5)
    downside_rank = downside_series.rank(pct=True).fillna(0.5)
    spread = (0.60 * breadth_rank + 0.25 * (1.0 - concentration_rank) + 0.15 * (1.0 - downside_rank)).clip(0.0, 1.0)
    candidate_float = float(holding_count) + (float(max_candidates) - float(holding_count)) * spread
    candidate_target = candidate_float.round().clip(lower=float(holding_count), upper=float(max_candidates))
    candidate_target.name = "candidate_target"
    return candidate_target.astype(float)


def _blend_selection_signal(selection_raw: pd.Series, *, rank_blend: float) -> pd.Series:
    return (
        _cross_sectional_standardize(selection_raw) * (1.0 - float(rank_blend))
        + (_cross_sectional_rank(selection_raw) - 0.5) * 2.0 * float(rank_blend)
    )


def _get_policy_v2_config(method: str) -> Dict[str, float]:
    normalized = str(method).strip().lower()
    if normalized not in POLICY_V2_FAMILY_CONFIGS:
        raise KeyError(f"Unsupported policy_v2-family method: {method}")
    return dict(POLICY_V2_FAMILY_CONFIGS[normalized])


def fit_score_head(
    train_pred_df: pd.DataFrame,
    target_names: List[str],
    method: str = "ridge",
    adaptive_task_weights: bool = True,
    adaptive_window_days: int | None = None,
    *,
    research_time_unit: str = RESEARCH_TIME_UNIT_CALENDAR_MONTHS,
    adaptive_window_months: int | None = None,
    holding_count: int = 5,
) -> ScoreHeadArtifact:
    if method == "manual":
        task_weights = (
            derive_adaptive_task_weights(
                train_pred_df,
                target_names,
                recent_window_days=adaptive_window_days,
                research_time_unit=research_time_unit,
                recent_window_months=adaptive_window_months,
            )
            if adaptive_task_weights
            else {}
        )
        return ScoreHeadArtifact(method="manual", feature_columns=[], task_weights=task_weights, model=None)

    features = _build_feature_frame(train_pred_df, target_names)
    indexed_features = features.set_index(["date", "stock"]).sort_index()
    task_weights = (
        derive_adaptive_task_weights(
            train_pred_df,
            target_names,
            recent_window_days=adaptive_window_days,
            research_time_unit=research_time_unit,
            recent_window_months=adaptive_window_months,
        )
        if adaptive_task_weights
        else {name: 1.0 / len(target_names) for name in target_names}
    )
    selection_target = _build_training_target(train_pred_df, task_weights)
    if method == "short_expert":
        confidence_target = _build_confidence_target(train_pred_df, task_weights)
        aligned = indexed_features.join(selection_target, how="inner").join(confidence_target, how="inner").dropna()
    elif method in {"policy_v1", "policy_v3"} or method in POLICY_V2_METHODS:
        utility_target = _build_utility_series(train_pred_df, task_weights)
        if method == "policy_v1":
            gate_multiplier = POLICY_V1_GATE_MULTIPLIER
        elif method in POLICY_V2_METHODS:
            gate_multiplier = float(_get_policy_v2_config(method)["gate_multiplier"])
        else:
            gate_multiplier = POLICY_V3_GATE_MULTIPLIER
        gate_target_count = max(int(holding_count), int(round(float(holding_count) * gate_multiplier)))
        gate_target = _build_topk_binary_target(selection_target, gate_target_count)
        weight_target = _build_weight_target(utility_target, gate_target)
        hold_target = _build_hold_target(train_pred_df, task_weights)
        aligned = (
            indexed_features.join(selection_target, how="inner")
            .join(gate_target, how="inner")
            .join(weight_target, how="inner")
            .join(hold_target, how="inner")
            .dropna()
        )
    else:
        aligned = indexed_features.join(selection_target, how="inner").dropna()
    if aligned.empty:
        raise RuntimeError("Score head training data is empty.")
    X = aligned.drop(columns=[col for col in ["score_target", "confidence_target", "gate_target", "weight_target", "hold_target"] if col in aligned.columns])
    if method == "ridge":
        y = aligned["score_target"].values
        model = Ridge(alpha=1.0, random_state=7)
        model.fit(X, y)
        return ScoreHeadArtifact(
            method=method,
            feature_columns=list(X.columns),
            task_weights=task_weights,
            model=model,
        )
    if method == "lgbm":
        y = aligned["score_target"].values
        if LGBMRegressor is None:
            raise RuntimeError("LightGBM is not available in current environment.")
        model = LGBMRegressor(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=7,
        )
        model.fit(X, y)
        return ScoreHeadArtifact(
            method=method,
            feature_columns=list(X.columns),
            task_weights=task_weights,
            model=model,
        )
    if method == "short_expert":
        selection_model = Ridge(alpha=1.0, random_state=7)
        confidence_model = Ridge(alpha=2.0, random_state=7)
        selection_model.fit(X, aligned["score_target"].values)
        confidence_model.fit(X, aligned["confidence_target"].values)
        return ScoreHeadArtifact(
            method=method,
            feature_columns=list(X.columns),
            task_weights=task_weights,
            model={
                "selection_model": selection_model,
                "confidence_model": confidence_model,
            },
            extra={
                "selection_rank_blend": 0.35,
                "confidence_floor": 0.50,
                "confidence_scale": 0.50,
            },
        )
    if method == "policy_v1":
        selection_model = Ridge(alpha=1.0, random_state=7)
        gate_model = Ridge(alpha=1.5, random_state=7)
        weight_model = Ridge(alpha=2.0, random_state=7)
        hold_model = Ridge(alpha=2.0, random_state=7)
        selection_model.fit(X, aligned["score_target"].values)
        gate_model.fit(X, aligned["gate_target"].values)
        weight_model.fit(X, aligned["weight_target"].values)
        hold_model.fit(X, aligned["hold_target"].values)

        selection_raw = pd.Series(selection_model.predict(X), index=aligned.index, dtype=float)
        selection_signal = _blend_selection_signal(selection_raw, rank_blend=0.35)
        gate_signal = _cross_sectional_rank(pd.Series(gate_model.predict(X), index=aligned.index, dtype=float))
        weight_signal = _cross_sectional_rank(pd.Series(weight_model.predict(X), index=aligned.index, dtype=float))
        hold_signal = _cross_sectional_rank(pd.Series(hold_model.predict(X), index=aligned.index, dtype=float))
        cash_target = _build_policy_cash_target(
            train_pred_df,
            utility_target,
            holding_count=holding_count,
            min_gross_exposure=POLICY_V1_MIN_GROSS_EXPOSURE,
            max_gross_exposure=POLICY_V1_MAX_GROSS_EXPOSURE,
        )
        cash_feature_frame = _build_policy_cash_feature_frame(
            selection_signal,
            gate_signal,
            weight_signal,
            hold_signal,
            top_n=gate_target_count,
        )
        cash_feature_aligned = cash_feature_frame.join(cash_target, how="inner").dropna()
        cash_columns = [col for col in cash_feature_aligned.columns if col != "cash_target"]
        cash_model = Ridge(alpha=2.5, random_state=7)
        cash_model.fit(cash_feature_aligned[cash_columns], cash_feature_aligned["cash_target"].values)
        return ScoreHeadArtifact(
            method=method,
            feature_columns=list(X.columns),
            task_weights=task_weights,
            model={
                "selection_model": selection_model,
                "gate_model": gate_model,
                "weight_model": weight_model,
                "hold_model": hold_model,
                "cash_model": cash_model,
            },
            extra={
                "selection_rank_blend": 0.35,
                "holding_count": int(holding_count),
                "gate_target_count": int(gate_target_count),
                "policy_min_gross_exposure": float(POLICY_V1_MIN_GROSS_EXPOSURE),
                "policy_max_gross_exposure": float(POLICY_V1_MAX_GROSS_EXPOSURE),
                "policy_weight_power": float(POLICY_V1_WEIGHT_POWER),
                "policy_hold_boost": float(POLICY_V1_HOLD_BOOST),
                "policy_candidate_multiplier": float(POLICY_V1_CANDIDATE_MULTIPLIER),
                "cash_feature_columns": list(cash_columns),
            },
        )
    if method in POLICY_V2_METHODS:
        config = _get_policy_v2_config(method)
        selection_model = Ridge(alpha=1.0, random_state=7)
        gate_model = Ridge(alpha=1.2, random_state=7)
        weight_model = Ridge(alpha=1.8, random_state=7)
        hold_model = Ridge(alpha=1.8, random_state=7)
        selection_model.fit(X, aligned["score_target"].values)
        gate_model.fit(X, aligned["gate_target"].values)
        weight_model.fit(X, aligned["weight_target"].values)
        hold_model.fit(X, aligned["hold_target"].values)

        selection_raw = pd.Series(selection_model.predict(X), index=aligned.index, dtype=float)
        selection_signal = _blend_selection_signal(selection_raw, rank_blend=float(config["selection_rank_blend"]))
        gate_signal = _cross_sectional_rank(pd.Series(gate_model.predict(X), index=aligned.index, dtype=float))
        weight_signal = _cross_sectional_rank(pd.Series(weight_model.predict(X), index=aligned.index, dtype=float))
        hold_signal = _cross_sectional_rank(pd.Series(hold_model.predict(X), index=aligned.index, dtype=float))
        cash_target = _build_policy_cash_target_v2(
            train_pred_df,
            utility_target,
            holding_count=holding_count,
            min_gross_exposure=float(config["min_gross_exposure"]),
            max_gross_exposure=float(config["max_gross_exposure"]),
        )
        cash_feature_frame = _build_policy_cash_feature_frame_v2(
            selection_signal,
            gate_signal,
            weight_signal,
            hold_signal,
            top_n=gate_target_count,
        )
        cash_feature_aligned = cash_feature_frame.join(cash_target, how="inner").dropna()
        cash_columns = [col for col in cash_feature_aligned.columns if col != "cash_target"]
        cash_model = Ridge(alpha=float(config["cash_model_alpha"]), random_state=7)
        cash_model.fit(cash_feature_aligned[cash_columns], cash_feature_aligned["cash_target"].values)
        return ScoreHeadArtifact(
            method=method,
            feature_columns=list(X.columns),
            task_weights=task_weights,
            model={
                "selection_model": selection_model,
                "gate_model": gate_model,
                "weight_model": weight_model,
                "hold_model": hold_model,
                "cash_model": cash_model,
            },
            extra={
                "selection_rank_blend": float(config["selection_rank_blend"]),
                "holding_count": int(holding_count),
                "gate_target_count": int(gate_target_count),
                "policy_min_gross_exposure": float(config["min_gross_exposure"]),
                "policy_max_gross_exposure": float(config["max_gross_exposure"]),
                "policy_weight_power": float(config["weight_power"]),
                "policy_hold_boost": float(config["hold_boost"]),
                "policy_candidate_multiplier": float(config["candidate_multiplier"]),
                "cash_feature_columns": list(cash_columns),
                "policy_build_weight_mix": float(config["build_weight_mix"]),
                "policy_build_gate_mix": float(config["build_gate_mix"]),
                "policy_build_hold_mix": float(config["build_hold_mix"]),
            },
        )
    if method == "policy_v3":
        selection_model = Ridge(alpha=0.9, random_state=7)
        gate_model = Ridge(alpha=1.1, random_state=7)
        weight_model = Ridge(alpha=1.6, random_state=7)
        hold_model = Ridge(alpha=1.6, random_state=7)
        selection_model.fit(X, aligned["score_target"].values)
        gate_model.fit(X, aligned["gate_target"].values)
        weight_model.fit(X, aligned["weight_target"].values)
        hold_model.fit(X, aligned["hold_target"].values)

        selection_raw = pd.Series(selection_model.predict(X), index=aligned.index, dtype=float)
        selection_signal = _blend_selection_signal(selection_raw, rank_blend=0.50)
        gate_signal = _cross_sectional_rank(pd.Series(gate_model.predict(X), index=aligned.index, dtype=float))
        weight_signal = _cross_sectional_rank(pd.Series(weight_model.predict(X), index=aligned.index, dtype=float))
        hold_signal = _cross_sectional_rank(pd.Series(hold_model.predict(X), index=aligned.index, dtype=float))
        max_candidates = max(int(holding_count), int(round(float(holding_count) * POLICY_V3_CANDIDATE_MULTIPLIER)))
        cash_target = _build_policy_cash_target_v3(
            train_pred_df,
            utility_target,
            holding_count=holding_count,
            min_gross_exposure=POLICY_V3_MIN_GROSS_EXPOSURE,
            max_gross_exposure=POLICY_V3_MAX_GROSS_EXPOSURE,
        )
        candidate_target = _build_policy_candidate_target_v3(
            train_pred_df,
            utility_target,
            holding_count=holding_count,
            max_candidates=max_candidates,
        )
        cash_feature_frame = _build_policy_cash_feature_frame_v3(
            selection_signal,
            gate_signal,
            weight_signal,
            hold_signal,
            top_n=gate_target_count,
        )
        cash_feature_aligned = cash_feature_frame.join(cash_target, how="inner").join(candidate_target, how="inner").dropna()
        cash_columns = [col for col in cash_feature_aligned.columns if col not in {"cash_target", "candidate_target"}]
        cash_model = Ridge(alpha=1.4, random_state=7)
        candidate_model = Ridge(alpha=1.2, random_state=7)
        cash_model.fit(cash_feature_aligned[cash_columns], cash_feature_aligned["cash_target"].values)
        candidate_model.fit(cash_feature_aligned[cash_columns], cash_feature_aligned["candidate_target"].values)
        return ScoreHeadArtifact(
            method=method,
            feature_columns=list(X.columns),
            task_weights=task_weights,
            model={
                "selection_model": selection_model,
                "gate_model": gate_model,
                "weight_model": weight_model,
                "hold_model": hold_model,
                "cash_model": cash_model,
                "candidate_model": candidate_model,
            },
            extra={
                "selection_rank_blend": 0.50,
                "holding_count": int(holding_count),
                "gate_target_count": int(gate_target_count),
                "policy_min_gross_exposure": float(POLICY_V3_MIN_GROSS_EXPOSURE),
                "policy_max_gross_exposure": float(POLICY_V3_MAX_GROSS_EXPOSURE),
                "policy_weight_power": float(POLICY_V3_WEIGHT_POWER),
                "policy_hold_boost": float(POLICY_V3_HOLD_BOOST),
                "policy_candidate_multiplier": float(POLICY_V3_CANDIDATE_MULTIPLIER),
                "policy_candidate_min_count": int(holding_count),
                "policy_candidate_max_count": int(max_candidates),
                "cash_feature_columns": list(cash_columns),
                "policy_build_score_mix": 0.40,
                "policy_build_weight_mix": 0.35,
                "policy_build_gate_mix": 0.15,
                "policy_build_hold_mix": 0.10,
            },
        )
    raise ValueError(f"Unsupported score head method: {method}")


def apply_score_head(
    artifact: ScoreHeadArtifact,
    pred_df: pd.DataFrame,
    target_names: List[str],
) -> pd.DataFrame:
    if artifact.method == "manual" or artifact.model is None:
        raise ValueError("apply_score_head requires a trained non-manual score head.")
    features = _build_feature_frame(pred_df, target_names)
    indexed = features.set_index(["date", "stock"]).sort_index()
    aligned = indexed.reindex(columns=artifact.feature_columns).fillna(0.0)
    out = aligned.reset_index()[["date", "stock"]].copy()
    if artifact.method == "short_expert":
        model_dict = artifact.model if isinstance(artifact.model, dict) else {}
        selection_model = model_dict.get("selection_model")
        confidence_model = model_dict.get("confidence_model")
        if selection_model is None or confidence_model is None:
            raise ValueError("short_expert score head requires selection_model and confidence_model.")
        selection_raw = pd.Series(selection_model.predict(aligned), index=aligned.index, dtype=float)
        confidence_raw = pd.Series(confidence_model.predict(aligned), index=aligned.index, dtype=float)
        selection_rank_blend = float(artifact.extra.get("selection_rank_blend", 0.35))
        confidence_floor = float(artifact.extra.get("confidence_floor", 0.50))
        confidence_scale = float(artifact.extra.get("confidence_scale", 0.50))
        selection_signal = _blend_selection_signal(selection_raw, rank_blend=selection_rank_blend)
        confidence_rank = _cross_sectional_rank(confidence_raw)
        sizing_score = confidence_floor + confidence_scale * confidence_rank
        learned_score = selection_signal * sizing_score
        out["selection_score"] = selection_signal.values
        out["confidence_score"] = confidence_rank.values
        out["sizing_score"] = sizing_score.values
        out["learned_score"] = learned_score.values
        return out
    if artifact.method == "policy_v1":
        model_dict = artifact.model if isinstance(artifact.model, dict) else {}
        selection_model = model_dict.get("selection_model")
        gate_model = model_dict.get("gate_model")
        weight_model = model_dict.get("weight_model")
        hold_model = model_dict.get("hold_model")
        cash_model = model_dict.get("cash_model")
        if any(model is None for model in (selection_model, gate_model, weight_model, hold_model, cash_model)):
            raise ValueError("policy_v1 score head requires selection/gate/weight/hold/cash models.")
        selection_raw = pd.Series(selection_model.predict(aligned), index=aligned.index, dtype=float)
        gate_raw = pd.Series(gate_model.predict(aligned), index=aligned.index, dtype=float)
        weight_raw = pd.Series(weight_model.predict(aligned), index=aligned.index, dtype=float)
        hold_raw = pd.Series(hold_model.predict(aligned), index=aligned.index, dtype=float)

        selection_signal = _blend_selection_signal(
            selection_raw,
            rank_blend=float(artifact.extra.get("selection_rank_blend", 0.35)),
        )
        gate_rank = _cross_sectional_rank(gate_raw)
        weight_rank = _cross_sectional_rank(weight_raw)
        hold_rank = _cross_sectional_rank(hold_raw)
        learned_score = selection_signal * (0.65 + 0.35 * gate_rank) * (0.55 + 0.45 * weight_rank)

        cash_feature_frame = _build_policy_cash_feature_frame(
            selection_signal,
            gate_rank,
            weight_rank,
            hold_rank,
            top_n=int(artifact.extra.get("gate_target_count", artifact.extra.get("holding_count", 5))),
        )
        cash_feature_columns = [
            str(col)
            for col in artifact.extra.get("cash_feature_columns", list(cash_feature_frame.columns))
        ]
        cash_aligned = cash_feature_frame.reindex(columns=cash_feature_columns).fillna(0.0)
        cash_pred = pd.Series(cash_model.predict(cash_aligned), index=cash_aligned.index, dtype=float)
        min_gross = float(artifact.extra.get("policy_min_gross_exposure", POLICY_V1_MIN_GROSS_EXPOSURE))
        max_gross = float(artifact.extra.get("policy_max_gross_exposure", POLICY_V1_MAX_GROSS_EXPOSURE))
        cash_pred = cash_pred.clip(lower=min_gross, upper=max_gross)

        out["selection_score"] = selection_signal.values
        out["gate_score"] = gate_rank.values
        out["weight_score"] = weight_rank.values
        out["hold_score"] = hold_rank.values
        out["learned_score"] = learned_score.values
        out["cash_score"] = pd.to_datetime(out["date"]).map(cash_pred).astype(float)
        out["gross_exposure_target"] = out["cash_score"].astype(float)
        return out
    if artifact.method in POLICY_V2_METHODS:
        config = _get_policy_v2_config(artifact.method)
        model_dict = artifact.model if isinstance(artifact.model, dict) else {}
        selection_model = model_dict.get("selection_model")
        gate_model = model_dict.get("gate_model")
        weight_model = model_dict.get("weight_model")
        hold_model = model_dict.get("hold_model")
        cash_model = model_dict.get("cash_model")
        if any(model is None for model in (selection_model, gate_model, weight_model, hold_model, cash_model)):
            raise ValueError(f"{artifact.method} score head requires selection/gate/weight/hold/cash models.")
        selection_raw = pd.Series(selection_model.predict(aligned), index=aligned.index, dtype=float)
        gate_raw = pd.Series(gate_model.predict(aligned), index=aligned.index, dtype=float)
        weight_raw = pd.Series(weight_model.predict(aligned), index=aligned.index, dtype=float)
        hold_raw = pd.Series(hold_model.predict(aligned), index=aligned.index, dtype=float)

        selection_signal = _blend_selection_signal(
            selection_raw,
            rank_blend=float(artifact.extra.get("selection_rank_blend", config["selection_rank_blend"])),
        )
        gate_rank = _cross_sectional_rank(gate_raw)
        weight_rank = _cross_sectional_rank(weight_raw)
        hold_rank = _cross_sectional_rank(hold_raw)
        learned_score = selection_signal * (0.60 + 0.25 * gate_rank + 0.15 * hold_rank) * (0.60 + 0.40 * weight_rank)

        cash_feature_frame = _build_policy_cash_feature_frame_v2(
            selection_signal,
            gate_rank,
            weight_rank,
            hold_rank,
            top_n=int(artifact.extra.get("gate_target_count", artifact.extra.get("holding_count", 5))),
        )
        cash_feature_columns = [
            str(col)
            for col in artifact.extra.get("cash_feature_columns", list(cash_feature_frame.columns))
        ]
        cash_aligned = cash_feature_frame.reindex(columns=cash_feature_columns).fillna(0.0)
        cash_pred = pd.Series(cash_model.predict(cash_aligned), index=cash_aligned.index, dtype=float)
        min_gross = float(artifact.extra.get("policy_min_gross_exposure", config["min_gross_exposure"]))
        max_gross = float(artifact.extra.get("policy_max_gross_exposure", config["max_gross_exposure"]))
        cash_pred = cash_pred.clip(lower=min_gross, upper=max_gross)

        out["selection_score"] = selection_signal.values
        out["gate_score"] = gate_rank.values
        out["weight_score"] = weight_rank.values
        out["hold_score"] = hold_rank.values
        out["learned_score"] = learned_score.values
        out["cash_score"] = pd.to_datetime(out["date"]).map(cash_pred).astype(float)
        out["gross_exposure_target"] = out["cash_score"].astype(float)
        return out
    if artifact.method == "policy_v3":
        model_dict = artifact.model if isinstance(artifact.model, dict) else {}
        selection_model = model_dict.get("selection_model")
        gate_model = model_dict.get("gate_model")
        weight_model = model_dict.get("weight_model")
        hold_model = model_dict.get("hold_model")
        cash_model = model_dict.get("cash_model")
        candidate_model = model_dict.get("candidate_model")
        if any(model is None for model in (selection_model, gate_model, weight_model, hold_model, cash_model, candidate_model)):
            raise ValueError("policy_v3 score head requires selection/gate/weight/hold/cash/candidate models.")
        selection_raw = pd.Series(selection_model.predict(aligned), index=aligned.index, dtype=float)
        gate_raw = pd.Series(gate_model.predict(aligned), index=aligned.index, dtype=float)
        weight_raw = pd.Series(weight_model.predict(aligned), index=aligned.index, dtype=float)
        hold_raw = pd.Series(hold_model.predict(aligned), index=aligned.index, dtype=float)

        selection_signal = _blend_selection_signal(
            selection_raw,
            rank_blend=float(artifact.extra.get("selection_rank_blend", 0.50)),
        )
        gate_rank = _cross_sectional_rank(gate_raw)
        weight_rank = _cross_sectional_rank(weight_raw)
        hold_rank = _cross_sectional_rank(hold_raw)
        learned_score = selection_signal * (0.65 + 0.20 * gate_rank + 0.15 * hold_rank) * (0.65 + 0.35 * weight_rank)

        cash_feature_frame = _build_policy_cash_feature_frame_v3(
            selection_signal,
            gate_rank,
            weight_rank,
            hold_rank,
            top_n=int(artifact.extra.get("gate_target_count", artifact.extra.get("holding_count", 5))),
        )
        cash_feature_columns = [
            str(col)
            for col in artifact.extra.get("cash_feature_columns", list(cash_feature_frame.columns))
        ]
        cash_aligned = cash_feature_frame.reindex(columns=cash_feature_columns).fillna(0.0)
        cash_pred = pd.Series(cash_model.predict(cash_aligned), index=cash_aligned.index, dtype=float)
        min_gross = float(artifact.extra.get("policy_min_gross_exposure", POLICY_V3_MIN_GROSS_EXPOSURE))
        max_gross = float(artifact.extra.get("policy_max_gross_exposure", POLICY_V3_MAX_GROSS_EXPOSURE))
        cash_pred = cash_pred.clip(lower=min_gross, upper=max_gross)
        candidate_pred = pd.Series(candidate_model.predict(cash_aligned), index=cash_aligned.index, dtype=float)
        candidate_min = int(artifact.extra.get("policy_candidate_min_count", artifact.extra.get("holding_count", 5)))
        candidate_max = int(
            artifact.extra.get(
                "policy_candidate_max_count",
                max(candidate_min, int(round(candidate_min * POLICY_V3_CANDIDATE_MULTIPLIER))),
            )
        )
        candidate_pred = candidate_pred.clip(lower=float(candidate_min), upper=float(candidate_max))

        out["selection_score"] = selection_signal.values
        out["gate_score"] = gate_rank.values
        out["weight_score"] = weight_rank.values
        out["hold_score"] = hold_rank.values
        out["learned_score"] = learned_score.values
        out["cash_score"] = pd.to_datetime(out["date"]).map(cash_pred).astype(float)
        out["gross_exposure_target"] = out["cash_score"].astype(float)
        out["candidate_count_target"] = pd.to_datetime(out["date"]).map(candidate_pred).astype(float)
        return out
    score = artifact.model.predict(aligned)
    out["learned_score"] = score
    return out


def build_policy_target_weight_frame(
    policy_output_df: pd.DataFrame,
    *,
    all_dates: pd.Index,
    all_stocks: list[str],
    holding_count: int,
    artifact: ScoreHeadArtifact,
) -> pd.DataFrame:
    if policy_output_df.empty:
        return pd.DataFrame(0.0, index=pd.Index(all_dates), columns=all_stocks, dtype=float)
    required = {"date", "stock", "learned_score", "gate_score", "weight_score", "hold_score"}
    missing = required.difference(policy_output_df.columns)
    if missing:
        raise KeyError(f"{artifact.method} output is missing required columns: {sorted(missing)}")

    if artifact.method == "policy_v1":
        default_min_gross = POLICY_V1_MIN_GROSS_EXPOSURE
        default_max_gross = POLICY_V1_MAX_GROSS_EXPOSURE
        default_weight_power = POLICY_V1_WEIGHT_POWER
        default_hold_boost = POLICY_V1_HOLD_BOOST
        default_candidate_multiplier = POLICY_V1_CANDIDATE_MULTIPLIER
    elif artifact.method in POLICY_V2_METHODS:
        config = _get_policy_v2_config(artifact.method)
        default_min_gross = float(config["min_gross_exposure"])
        default_max_gross = float(config["max_gross_exposure"])
        default_weight_power = float(config["weight_power"])
        default_hold_boost = float(config["hold_boost"])
        default_candidate_multiplier = float(config["candidate_multiplier"])
    else:
        default_min_gross = POLICY_V3_MIN_GROSS_EXPOSURE
        default_max_gross = POLICY_V3_MAX_GROSS_EXPOSURE
        default_weight_power = POLICY_V3_WEIGHT_POWER
        default_hold_boost = POLICY_V3_HOLD_BOOST
        default_candidate_multiplier = POLICY_V3_CANDIDATE_MULTIPLIER

    min_gross = float(artifact.extra.get("policy_min_gross_exposure", default_min_gross))
    max_gross = float(artifact.extra.get("policy_max_gross_exposure", default_max_gross))
    weight_power = float(artifact.extra.get("policy_weight_power", default_weight_power))
    hold_boost = float(artifact.extra.get("policy_hold_boost", default_hold_boost))
    candidate_multiplier = float(artifact.extra.get("policy_candidate_multiplier", default_candidate_multiplier))

    indexed = policy_output_df.copy()
    indexed["date"] = pd.to_datetime(indexed["date"])
    indexed["stock"] = indexed["stock"].astype(str).str.upper().str.strip()
    indexed = indexed.set_index(["date", "stock"]).sort_index()
    result = pd.DataFrame(0.0, index=pd.Index(pd.to_datetime(all_dates)), columns=list(all_stocks), dtype=float)

    base_holding_count = max(int(holding_count), 1)
    max_candidates = max(base_holding_count, int(round(base_holding_count * candidate_multiplier)))

    for dt, group in indexed.groupby(level=0):
        frame = group.reset_index(level=0, drop=True).copy()
        frame = frame.sort_values(["gate_score", "learned_score"], ascending=False)
        if "gross_exposure_target" in frame.columns and not frame["gross_exposure_target"].empty:
            gross = float(frame["gross_exposure_target"].iloc[0])
        else:
            gross = float(max_gross)
        gross = float(np.clip(gross, min_gross, max_gross))
        if artifact.method == "policy_v3" and "candidate_count_target" in frame.columns and not frame["candidate_count_target"].empty:
            candidate_min = int(artifact.extra.get("policy_candidate_min_count", base_holding_count))
            candidate_max = int(artifact.extra.get("policy_candidate_max_count", max_candidates))
            candidate_count = int(round(float(frame["candidate_count_target"].iloc[0])))
            candidate_count = max(candidate_min, min(candidate_count, candidate_max, len(frame)))
        else:
            candidate_count = int(round(base_holding_count + (max_candidates - base_holding_count) * gross))
            candidate_count = max(base_holding_count, min(candidate_count, len(frame)))
        if artifact.method in POLICY_V2_METHODS or artifact.method == "policy_v3":
            frame = frame.sort_values(["gate_score", "hold_score", "learned_score"], ascending=False)
        selected = frame.head(candidate_count).copy()
        if artifact.method in POLICY_V2_METHODS:
            raw = (
                selected["weight_score"].astype(float).clip(lower=0.0) * float(artifact.extra.get("policy_build_weight_mix", 0.65))
                + selected["gate_score"].astype(float).clip(lower=0.0) * float(artifact.extra.get("policy_build_gate_mix", 0.25))
                + selected["hold_score"].astype(float).clip(lower=0.0) * float(artifact.extra.get("policy_build_hold_mix", 0.10))
            )
        elif artifact.method == "policy_v3":
            learned_rank = selected["learned_score"].rank(method="first", pct=True).astype(float).clip(lower=0.0)
            raw = (
                learned_rank * float(artifact.extra.get("policy_build_score_mix", 0.40))
                + selected["weight_score"].astype(float).clip(lower=0.0) * float(artifact.extra.get("policy_build_weight_mix", 0.35))
                + selected["gate_score"].astype(float).clip(lower=0.0) * float(artifact.extra.get("policy_build_gate_mix", 0.15))
                + selected["hold_score"].astype(float).clip(lower=0.0) * float(artifact.extra.get("policy_build_hold_mix", 0.10))
            )
        else:
            raw = selected["weight_score"].astype(float).clip(lower=0.0)
        if float(raw.sum()) <= 0.0:
            raw = selected["learned_score"].rank(method="first", pct=True).astype(float).clip(lower=0.0)
        hold_scale = 1.0 + hold_boost * (selected["hold_score"].astype(float).fillna(0.5) - 0.5) * 2.0
        raw = raw.mul(hold_scale.clip(lower=0.25), fill_value=0.0)
        raw = raw.pow(max(weight_power, 1e-6)).clip(lower=0.0)
        if float(raw.sum()) <= 0.0:
            raw = pd.Series(1.0, index=selected.index, dtype=float)
        weights = raw / float(raw.sum()) * gross
        result.loc[pd.Timestamp(dt), weights.index] = weights.astype(float).values
    return result.fillna(0.0)
