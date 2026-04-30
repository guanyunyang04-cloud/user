from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _series(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in frame.columns:
        values = pd.to_numeric(frame[name], errors="coerce")
    else:
        values = pd.Series(default, index=frame.index, dtype=float)
    return values.replace([np.inf, -np.inf], np.nan).fillna(float(default)).astype(float)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(np.clip(float(value), float(low), float(high)))


def _masked_mean(values: pd.Series, mask: pd.Series) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(values.loc[mask].mean())


def build_allocation_teacher_summary(label_frame: pd.DataFrame) -> dict[str, float]:
    """Summarize one daily source/receiver/cash allocation teacher surface.

    This is intentionally a summary contract, not a new live decision path. It exposes
    whether the daily teacher contains enough source, receiver and cash structure for
    future listwise allocation training.
    """
    if label_frame.empty:
        return {
            "allocation_receiver_candidate_count": 0.0,
            "allocation_source_candidate_count": 0.0,
            "allocation_receiver_demand_target": 0.0,
            "allocation_source_release_target": 0.0,
            "allocation_cash_reserve_target": 0.0,
            "allocation_transfer_intensity_target": 0.0,
            "source_receiver_cash_allocation_teacher_ready": 0.0,
        }

    action = label_frame.get("action_label", pd.Series("", index=label_frame.index)).astype(str).str.lower()
    receiver_score = _series(label_frame, "portfolio_daily_receiver_score")
    receiver_mask = (_series(label_frame, "portfolio_daily_receiver_candidate_mask") > 0.5) | action.isin({"open", "add"})
    source_score = _series(label_frame, "portfolio_daily_source_score")
    source_mask = (_series(label_frame, "portfolio_daily_source_candidate_mask") > 0.5) | action.isin({"reduce", "exit"})
    cash_score = _series(label_frame, "portfolio_daily_cash_score")
    transfer_score = _series(label_frame, "portfolio_daily_allocation_transfer_score")

    receiver_count = float(receiver_mask.sum())
    source_count = float(source_mask.sum())
    receiver_demand = _masked_mean(receiver_score, receiver_mask)
    source_release = _masked_mean(source_score, source_mask)
    paired_transfer = _masked_mean(transfer_score, receiver_mask | source_mask)
    cash_reserve = _clip(
        float(cash_score.mean()) * 0.72
        + max(0.0, 1.0 - min(receiver_count, 3.0) / 3.0) * 0.16
        + max(0.0, 1.0 - min(source_count, 3.0) / 3.0) * 0.12
    )
    transfer_intensity = _clip(
        paired_transfer * 0.62
        + min(receiver_count, 4.0) / 4.0 * 0.18
        + min(source_count, 4.0) / 4.0 * 0.20
    )
    ready = float(receiver_count >= 1.0 and source_count >= 1.0 and transfer_intensity > 0.0)

    return {
        "allocation_receiver_candidate_count": receiver_count,
        "allocation_source_candidate_count": source_count,
        "allocation_receiver_demand_target": _clip(receiver_demand),
        "allocation_source_release_target": _clip(source_release),
        "allocation_cash_reserve_target": cash_reserve,
        "allocation_transfer_intensity_target": transfer_intensity,
        "source_receiver_cash_allocation_teacher_ready": ready,
    }


def summarize_allocation_teacher_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return build_allocation_teacher_summary(pd.DataFrame())
    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return {
        str(column): float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).mean())
        for column in frame.columns
    }
