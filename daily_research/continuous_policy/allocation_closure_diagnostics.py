from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.continuous_policy.runtime import (
    STUDIES_ROOT,
    read_json,
    safe_print_json,
    write_json,
)


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    value = float(series.replace([np.inf, -np.inf], np.nan).dropna().mean())
    return value if np.isfinite(value) else 0.0


def _numeric(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _first_numeric(frame: pd.DataFrame, names: tuple[str, ...], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return _numeric(frame, name, default)
    return pd.Series(default, index=frame.index, dtype=float)


def summarize_allocation_closure_from_turnover(turnover_frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize cash/exposure closure from a shadow turnover panel."""
    if turnover_frame.empty:
        return {
            "row_count": 0,
            "day_count": 0,
            "actual_cash_weight_mean": 0.0,
            "actual_gross_exposure_mean": 0.0,
            "avg_gross_exposure_target": 0.0,
            "exposure_utilization": 0.0,
            "deployable_idle_cash_mean": 0.0,
            "cash_reserve_signal_mean": 0.0,
            "cash_semantics_mismatch": False,
            "receiver_candidate_without_target_day_share": 0.0,
            "source_dead_day_share": 0.0,
            "allocation_objective_mean": 0.0,
            "native_fallback_mean": 0.0,
        }

    working = turnover_frame.copy()
    cash_weight = _first_numeric(working, ("cash_weight", "allocation_layer_cash_after"), 0.0).clip(0.0, 1.0)
    gross_exposure = _first_numeric(working, ("gross_exposure",), np.nan)
    if gross_exposure.isna().all() or float(gross_exposure.fillna(0.0).abs().sum()) <= 1.0e-12:
        gross_exposure = 1.0 - cash_weight
    gross_exposure = gross_exposure.fillna(0.0).clip(0.0, 1.0)
    gross_target = _first_numeric(working, ("gross_exposure_target", "avg_gross_exposure_target"), 0.0).clip(0.0, 1.0)
    target_weight_sum = _first_numeric(
        working,
        ("allocation_layer_target_weight_sum", "target_weight_sum", "gross_exposure"),
        np.nan,
    )
    target_weight_sum = target_weight_sum.where(target_weight_sum.notna(), gross_exposure).clip(0.0, 1.0)
    stock_budget = _first_numeric(
        working,
        ("allocation_layer_stock_budget", "stock_budget", "gross_exposure_target"),
        0.0,
    ).clip(0.0, 1.0)
    available_cash = _first_numeric(
        working,
        ("allocation_layer_available_cash_to_deploy", "available_cash_to_deploy"),
        0.0,
    ).clip(0.0, 1.0)
    deployable_idle_cash = pd.concat(
        [
            available_cash,
            (stock_budget - target_weight_sum).clip(lower=0.0),
        ],
        axis=1,
    ).min(axis=1).clip(lower=0.0, upper=1.0)
    cash_reserve_signal = _first_numeric(
        working,
        (
            "portfolio_daily_cash_reserve_signal",
            "portfolio_daily_cash_reserve_rate",
            "cash_reserve_signal",
            "cash_reserve_target",
            "cash_reserve_rate",
        ),
        0.0,
    ).clip(0.0, 1.0)
    receiver_candidates = _first_numeric(
        working,
        (
            "allocation_layer_receiver_candidate_count",
            "allocation_layer_receiver_executable_candidate_count",
            "portfolio_daily_receiver_candidate_count",
            "portfolio_daily_unified_receiver_candidate_count",
        ),
        0.0,
    )
    receiver_targets = _first_numeric(
        working,
        ("allocation_layer_receiver_target_count", "portfolio_daily_receiver_target_count"),
        0.0,
    )
    source_targets = _first_numeric(
        working,
        (
            "allocation_layer_source_target_count",
            "portfolio_daily_source_target_count",
            "native_source_target_count",
        ),
        0.0,
    )
    allocation_objective = _first_numeric(
        working,
        ("allocation_layer_objective_value", "portfolio_daily_unified_allocation_objective"),
        0.0,
    )
    native_fallback = _first_numeric(working, ("allocation_layer_native_fallback_used",), 0.0).clip(0.0, 1.0)

    day_count = int(working["date"].nunique()) if "date" in working.columns else int(len(working))
    avg_gross_target = _safe_mean(gross_target)
    actual_gross = _safe_mean(gross_exposure)
    exposure_utilization = float(actual_gross / max(avg_gross_target, 1.0e-8)) if avg_gross_target > 0.0 else 0.0
    actual_cash = _safe_mean(cash_weight)
    signal_mean = _safe_mean(cash_reserve_signal)
    idle_mean = _safe_mean(deployable_idle_cash)
    mismatch = bool(actual_cash >= 0.55 and signal_mean <= 0.05 and idle_mean >= 0.20)
    receiver_gap_share = _safe_mean((receiver_candidates > receiver_targets).astype(float))
    source_dead_share = _safe_mean(((receiver_targets > 0.0) & (source_targets <= 0.0)).astype(float))

    return {
        "row_count": int(len(working)),
        "day_count": day_count,
        "actual_cash_weight_mean": actual_cash,
        "actual_gross_exposure_mean": actual_gross,
        "avg_gross_exposure_target": avg_gross_target,
        "exposure_utilization": exposure_utilization,
        "deployable_idle_cash_mean": idle_mean,
        "cash_reserve_signal_mean": signal_mean,
        "cash_semantics_mismatch": mismatch,
        "receiver_candidate_without_target_day_share": receiver_gap_share,
        "source_dead_day_share": source_dead_share,
        "allocation_objective_mean": _safe_mean(allocation_objective),
        "native_fallback_mean": _safe_mean(native_fallback),
    }


def _read_csv(path_value: str) -> pd.DataFrame:
    path = Path(str(path_value or "")).expanduser()
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _trial_turnover_path(trial: dict[str, Any]) -> Path | None:
    protocol_path = Path(str(trial.get("protocol_summary_json", "") or "")).expanduser()
    if not protocol_path.exists():
        return None
    protocol_summary = read_json(protocol_path)
    evaluation = dict(protocol_summary.get("evaluation", {}) or {})
    for key in ("turnover_csv", "shadow_turnover_csv"):
        value = str(evaluation.get(key, "") or "")
        if value:
            path = Path(value).expanduser()
            if path.exists():
                return path
    fallback = protocol_path.parent / "shadow_daily_turnover.csv"
    return fallback if fallback.exists() else None


def build_allocation_closure_report(*, study_tag: str = "", study_summary_path: str = "", turnover_csv: str = "") -> dict[str, Any]:
    if turnover_csv:
        turnover_path = Path(turnover_csv).expanduser().resolve()
        return {
            "report_type": "allocation_closure",
            "source_turnover_csv": str(turnover_path),
            "summary": summarize_allocation_closure_from_turnover(_read_csv(str(turnover_path))),
        }

    summary_path = (
        Path(study_summary_path).expanduser().resolve()
        if str(study_summary_path or "").strip()
        else STUDIES_ROOT / str(study_tag).strip() / "study_summary.json"
    )
    study_summary = read_json(summary_path)
    trials: list[dict[str, Any]] = []
    for trial in study_summary.get("trials", []) or []:
        turnover_path = _trial_turnover_path(dict(trial or {}))
        closure = summarize_allocation_closure_from_turnover(
            _read_csv(str(turnover_path)) if turnover_path is not None else pd.DataFrame()
        )
        trials.append(
            {
                "trial_id": trial.get("trial_id", ""),
                "trial_tag": trial.get("trial_tag", ""),
                "status": trial.get("status", ""),
                "protocol_summary_json": trial.get("protocol_summary_json", ""),
                "turnover_csv": str(turnover_path) if turnover_path is not None else "",
                "allocation_closure": closure,
            }
        )
    return {
        "report_type": "allocation_closure",
        "study_tag": str(study_tag or study_summary.get("study_tag", study_summary.get("run_tag", "")) or ""),
        "study_summary_json": str(summary_path),
        "trial_count": len(trials),
        "trials": trials,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose portfolio-daily allocation cash/exposure closure.")
    parser.add_argument("--study-tag", default="")
    parser.add_argument("--study-summary", default="")
    parser.add_argument("--turnover-csv", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_allocation_closure_report(
        study_tag=str(args.study_tag or ""),
        study_summary_path=str(args.study_summary or ""),
        turnover_csv=str(args.turnover_csv or ""),
    )
    if str(args.output_json or "").strip():
        write_json(Path(str(args.output_json)).expanduser(), report)
    if args.json or not str(args.output_json or "").strip():
        safe_print_json(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
