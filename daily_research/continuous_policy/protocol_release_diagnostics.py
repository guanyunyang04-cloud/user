from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_DIRECT_MEAN_COLUMNS: tuple[str, ...] = (
    "release_first_allocation_v3_mode_used",
    "release_first_allocation_v3_used",
    "release_first_intent_target_count",
    "release_first_source_intent_count",
    "release_first_source_realized_count",
    "release_first_rotation_amount",
    "release_first_cash_buffer_amount",
)

_REQUIRED_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    *_DIRECT_MEAN_COLUMNS,
    "portfolio_daily_source_target_count",
    "allocation_layer_target_sum_gap",
    "cash_weight",
    "intent_translation_conflict_rate",
)


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default))


def _safe_mean(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if frame.empty:
        return float(default)
    return float(_numeric(frame, column, default).mean())


def summarize_release_first_turnover_diagnostics(turnover_csv: str | Path) -> dict[str, Any]:
    path = Path(turnover_csv)
    missing_columns: list[str] = []
    if not path.exists():
        return {
            "metrics": {
                **{column: 0.0 for column in _DIRECT_MEAN_COLUMNS},
                "portfolio_daily_source_target_count": 0.0,
                "portfolio_daily_source_realized_sell_rate": 0.0,
                "portfolio_daily_target_sum_gap": 0.0,
                "portfolio_daily_actual_cash_weight_mean": 0.0,
                "intent_translation_conflict_rate": 1.0,
            },
            "missing_columns": list(_REQUIRED_DIAGNOSTIC_COLUMNS),
            "turnover_csv": str(path),
            "status": "missing_turnover_csv",
        }
    frame = pd.read_csv(path)
    for column in _REQUIRED_DIAGNOSTIC_COLUMNS:
        if column not in frame.columns:
            missing_columns.append(column)
    metrics = {column: _safe_mean(frame, column) for column in _DIRECT_MEAN_COLUMNS}
    source_target = _numeric(frame, "portfolio_daily_source_target_count", 0.0)
    source_target_sum = float(source_target.sum())
    if "portfolio_daily_source_realized_sell_rate" in frame.columns:
        source_sell_rate = _safe_mean(frame, "portfolio_daily_source_realized_sell_rate")
    elif source_target_sum > 0.0:
        realized = _numeric(frame, "release_first_source_realized_count", 0.0)
        if float(realized.sum()) <= 0.0 and "sell_intent_realized_count" in frame.columns:
            realized = _numeric(frame, "sell_intent_realized_count", 0.0)
        source_sell_rate = float(np.clip(float(realized.sum()) / source_target_sum, 0.0, 1.0))
    elif "sell_intent_realized_rate" in frame.columns:
        source_sell_rate = _safe_mean(frame, "sell_intent_realized_rate")
    else:
        source_sell_rate = 0.0
    metrics.update(
        {
            "portfolio_daily_source_target_count": float(source_target.mean()) if not frame.empty else 0.0,
            "portfolio_daily_source_realized_sell_rate": float(source_sell_rate),
            "portfolio_daily_target_sum_gap": _safe_mean(frame, "allocation_layer_target_sum_gap"),
            "portfolio_daily_actual_cash_weight_mean": _safe_mean(frame, "cash_weight"),
            "intent_translation_conflict_rate": _safe_mean(frame, "intent_translation_conflict_rate", 1.0),
        }
    )
    return {
        "metrics": metrics,
        "missing_columns": missing_columns,
        "turnover_csv": str(path),
        "status": "ok" if not missing_columns else "partial",
    }


def enrich_continuity_metrics_with_release_first_diagnostics(
    continuity_metrics: dict[str, Any] | None,
    turnover_csv: str | Path,
) -> dict[str, Any]:
    continuity = dict(continuity_metrics or {})
    diagnostics = summarize_release_first_turnover_diagnostics(turnover_csv)
    continuity.update(diagnostics["metrics"])
    continuity["release_first_diagnostic_missing_columns"] = list(diagnostics["missing_columns"])
    continuity["release_first_diagnostic_status"] = str(diagnostics["status"])
    return continuity


def enrich_summary_with_release_first_diagnostics(summary: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(summary or {})
    turnover_csv = str(enriched.get("turnover_csv", "") or "")
    if not turnover_csv:
        return enriched
    diagnostics = summarize_release_first_turnover_diagnostics(turnover_csv)
    continuity = dict(enriched.get("continuity_metrics", {}) or {})
    continuity.update(diagnostics["metrics"])
    continuity["release_first_diagnostic_missing_columns"] = list(diagnostics["missing_columns"])
    continuity["release_first_diagnostic_status"] = str(diagnostics["status"])
    enriched["continuity_metrics"] = continuity
    enriched["release_first_diagnostics"] = diagnostics
    return enriched
