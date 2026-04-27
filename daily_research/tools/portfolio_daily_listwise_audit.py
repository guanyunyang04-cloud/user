from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ACTION_OUTCOME_NAMES = (
    "shadow_daily_action_outcomes.csv",
    "daily_action_outcomes.csv",
    "shadow_daily_action_panel.csv",
    "daily_action_panel.csv",
)

PROTOCOL_SUMMARY_NAMES = (
    "protocol_summary.json",
    "shadow_window_summary.json",
)

PROTOCOL_METRIC_KEYS = (
    "portfolio_daily_receiver_target_count",
    "portfolio_daily_receiver_realized_deploy_rate",
    "portfolio_daily_receiver_unrealized_deploy_share",
    "portfolio_daily_source_candidate_count",
    "portfolio_daily_source_target_count",
    "portfolio_daily_source_sell_count",
    "portfolio_daily_source_realized_sell_rate",
    "portfolio_daily_source_target_not_sold_share",
    "portfolio_daily_effective_capital_transfer_count",
    "portfolio_daily_cash_reserve_rate",
    "portfolio_daily_source_score_mean",
    "portfolio_daily_source_release_capacity_mean",
    "portfolio_daily_source_release_quality_mean",
    "portfolio_daily_source_executability_mean",
    "portfolio_daily_receiver_forward_excess_5d",
    "portfolio_daily_source_forward_excess_5d",
    "portfolio_daily_receiver_minus_source_forward_excess_5d",
)


def _read_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _read_num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def _read_text(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=object)
    return frame[column].fillna(default).astype(str)


def _rate(mask: pd.Series, denom: pd.Series | None = None) -> float:
    if denom is None:
        return float(mask.mean()) if len(mask) else 0.0
    total = int(denom.sum())
    return float((mask & denom).sum() / total) if total else 0.0


def _mean(series: pd.Series, mask: pd.Series | None = None) -> float:
    if mask is not None:
        series = series.loc[mask]
    return float(series.mean()) if len(series.dropna()) else 0.0


def _find_action_file(protocol_dir: Path) -> Path:
    for name in ACTION_OUTCOME_NAMES:
        path = protocol_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No action outcome/panel CSV found under {protocol_dir}")


def _find_protocol_summary(protocol_dir: Path) -> Path | None:
    for name in PROTOCOL_SUMMARY_NAMES:
        path = protocol_dir / name
        if path.exists():
            return path
    return None


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _nested_dict(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, dict) else {}


def _select_metric_block(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in PROTOCOL_METRIC_KEYS:
        value = metrics.get(key)
        if value is None:
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _protocol_metric_blocks(protocol_summary: dict[str, Any]) -> dict[str, Any]:
    evaluation_continuity = _nested_dict(protocol_summary, "evaluation", "continuity_metrics")
    shadow_continuity = _nested_dict(protocol_summary, "shadow", "continuity_metrics")
    model_continuity = _nested_dict(protocol_summary, "latest_behavior_audit", "model_continuity_metrics")
    semantic_conflicts = _nested_dict(protocol_summary, "latest_behavior_audit", "semantic_conflicts")
    return {
        "evaluation_continuity_metrics": _select_metric_block(evaluation_continuity),
        "shadow_continuity_metrics": _select_metric_block(shadow_continuity),
        "latest_model_continuity_metrics": _select_metric_block(model_continuity),
        "latest_semantic_conflict_metrics": _select_metric_block(semantic_conflicts),
    }


def _source_disappearance(frame: pd.DataFrame, protocol_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    held = _read_num(frame, "current_weight") > 1.0e-8
    current = _read_num(frame, "current_weight")
    source_score = _read_num(frame, "portfolio_daily_source_score")
    source_capacity = _read_num(frame, "portfolio_daily_source_release_capacity")
    source_release_quality = _read_num(frame, "portfolio_daily_source_release_quality")
    source_exec = _read_num(frame, "portfolio_daily_source_executability")
    receiver_target = _read_bool(frame, "portfolio_daily_receiver_target")
    cash_signal = _read_bool(frame, "portfolio_daily_cash_reserve_signal")
    source_candidate = _read_bool(frame, "portfolio_daily_source_candidate")
    source_target = _read_bool(frame, "portfolio_daily_source_target")
    funding_protected = _read_bool(frame, "direct_action_funding_protected")
    floor_guarded = _read_bool(frame, "sell_source_floor_guarded")
    realized_sell = _read_text(frame, "weight_change_action").str.lower().isin({"reduce", "exit"})

    source_target_not_sold_reason = _read_text(frame, "portfolio_daily_source_target_not_sold_reason", "none")
    reason_counts = (
        source_target_not_sold_reason.loc[source_target & ~realized_sell]
        .replace("", "none")
        .value_counts()
        .to_dict()
    )
    eligible_context = held & (current >= 0.012) & (receiver_target.groupby(frame["date"]).transform("any") | cash_signal)
    high_source_score = source_score >= 0.135
    high_source_exec = (source_exec >= 0.30) & (source_capacity >= 0.50)
    protection_pass = (~funding_protected) | (source_score >= 0.235) | (source_exec >= 0.36)

    evaluation_source_target_count = 0.0
    evaluation_source_realized_sell_rate = 0.0
    evaluation_effective_transfer_count = 0.0
    if protocol_metrics:
        evaluation_metrics = protocol_metrics.get("evaluation_continuity_metrics", {})
        semantic_metrics = protocol_metrics.get("latest_semantic_conflict_metrics", {})
        evaluation_source_target_count = float(evaluation_metrics.get("portfolio_daily_source_target_count", 0.0))
        evaluation_source_realized_sell_rate = float(
            evaluation_metrics.get("portfolio_daily_source_realized_sell_rate", 0.0)
        )
        evaluation_effective_transfer_count = float(
            evaluation_metrics.get("portfolio_daily_effective_capital_transfer_count", 0.0)
        )
        if evaluation_effective_transfer_count <= 0.0:
            evaluation_effective_transfer_count = float(
                semantic_metrics.get("portfolio_daily_effective_capital_transfer_count", 0.0)
            )

    action_diagnosis = _diagnose_source(
        source_candidate_count=int(source_candidate.sum()),
        source_target_count=int(source_target.sum()),
        source_realized_count=int((source_target & realized_sell).sum()),
        eligible_context_count=int(eligible_context.sum()),
        high_score_count=int((eligible_context & high_source_score).sum()),
        exec_capacity_count=int((eligible_context & high_source_exec).sum()),
        floor_guard_count=int(floor_guarded.sum()),
        protected_block_count=int((eligible_context & ~protection_pass).sum()),
    )
    if (
        action_diagnosis.startswith("no_source_context")
        and evaluation_source_target_count > 0
        and (evaluation_source_realized_sell_rate > 0 or evaluation_effective_transfer_count > 0)
    ):
        action_diagnosis = (
            "latest_window_inactive_but_evaluation_source_active: "
            "action CSV covers the latest shadow window while evaluation continuity metrics contain realized source transfers"
        )

    return {
        "held_rows": int(held.sum()),
        "eligible_current_floor_rows": int((held & (current >= 0.012)).sum()),
        "eligible_receiver_or_cash_context_rows": int(eligible_context.sum()),
        "source_candidate_rows": int(source_candidate.sum()),
        "source_target_rows": int(source_target.sum()),
        "source_realized_sell_rows": int((source_target & realized_sell).sum()),
        "source_score_ge_0135_rows": int((eligible_context & high_source_score).sum()),
        "source_exec_capacity_rows": int((eligible_context & high_source_exec).sum()),
        "protected_block_rows": int((eligible_context & ~protection_pass).sum()),
        "sell_source_floor_guarded_rows": int(floor_guarded.sum()),
        "source_target_not_sold_reasons": {str(k): int(v) for k, v in reason_counts.items()},
        "top_source_score": float(source_score.max()) if len(source_score) else 0.0,
        "top_source_executability": float(source_exec.max()) if len(source_exec) else 0.0,
        "top_source_release_capacity": float(source_capacity.max()) if len(source_capacity) else 0.0,
        "top_source_release_quality": float(source_release_quality.max()) if len(source_release_quality) else 0.0,
        "evaluation_source_target_count": evaluation_source_target_count,
        "evaluation_source_realized_sell_rate": evaluation_source_realized_sell_rate,
        "evaluation_effective_capital_transfer_count": evaluation_effective_transfer_count,
        "diagnosis": action_diagnosis,
    }


def _diagnose_source(
    *,
    source_candidate_count: int,
    source_target_count: int,
    source_realized_count: int,
    eligible_context_count: int,
    high_score_count: int,
    exec_capacity_count: int,
    floor_guard_count: int,
    protected_block_count: int,
) -> str:
    if source_candidate_count == 0:
        if eligible_context_count == 0:
            return "no_source_context: no held rows met current floor plus receiver/cash pressure"
        if high_score_count == 0 and exec_capacity_count == 0:
            return "score_signal_absent: held rows exist but source score/executability never crossed candidate gates"
        if protected_block_count > 0:
            return "protection_blocked: source-like rows were mostly protected by funding/keep guards"
        return "candidate_filter_blocked: source context exists but final candidate mask removed all rows"
    if source_target_count == 0:
        return "ranking_not_selected: candidates existed but source limit/rank selected none"
    if source_realized_count == 0:
        if floor_guard_count > 0:
            return "execution_floor_blocked: source targets existed but sell-source floor/translation guards prevented sells"
        return "translation_blocked: source targets existed but no reduce/exit reached final weights"
    return "source_path_active"


def build_audit(protocol_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, Path]:
    action_path = _find_action_file(protocol_dir)
    protocol_summary_path = _find_protocol_summary(protocol_dir)
    protocol_summary = _load_json(protocol_summary_path)
    protocol_metrics = _protocol_metric_blocks(protocol_summary)
    frame = pd.read_csv(action_path)
    if "date" not in frame.columns:
        raise ValueError(f"{action_path} is missing required date column")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    receiver_target = _read_bool(frame, "portfolio_daily_receiver_target")
    source_candidate = _read_bool(frame, "portfolio_daily_source_candidate")
    source_target = _read_bool(frame, "portfolio_daily_source_target")
    cash_signal = _read_bool(frame, "portfolio_daily_cash_reserve_signal")
    floor_guarded = _read_bool(frame, "sell_source_floor_guarded")
    realized_sell = _read_text(frame, "weight_change_action").str.lower().isin({"reduce", "exit"})
    realized_buy = _read_text(frame, "weight_change_action").str.lower().isin({"open", "add"})
    receiver_realized_buy = receiver_target & realized_buy
    source_realized_sell = source_target & realized_sell

    source_score = _read_num(frame, "portfolio_daily_source_score")
    source_gap = _read_num(frame, "portfolio_daily_source_gap")
    source_capacity = _read_num(frame, "portfolio_daily_source_release_capacity")
    source_release_quality = _read_num(frame, "portfolio_daily_source_release_quality")
    source_exec = _read_num(frame, "portfolio_daily_source_executability")
    receiver_score = _read_num(frame, "portfolio_daily_receiver_score")
    receiver_capacity = _read_num(frame, "portfolio_daily_receiver_add_capacity")
    cash_score = _read_num(frame, "portfolio_daily_cash_score")

    daily = (
        frame.assign(
            receiver_target=receiver_target,
            source_candidate=source_candidate,
            source_target=source_target,
            cash_signal=cash_signal,
            floor_guarded=floor_guarded,
            source_realized_sell=source_realized_sell,
            receiver_realized_buy=receiver_realized_buy,
            source_score=source_score,
            source_gap=source_gap,
            source_capacity=source_capacity,
            source_release_quality=source_release_quality,
            source_exec=source_exec,
            receiver_score=receiver_score,
            receiver_capacity=receiver_capacity,
            cash_score=cash_score,
        )
        .groupby("date", dropna=False)
        .agg(
            rows=("stock", "count"),
            receiver_target_count=("receiver_target", "sum"),
            receiver_realized_buy_count=("receiver_realized_buy", "sum"),
            source_candidate_count=("source_candidate", "sum"),
            source_target_count=("source_target", "sum"),
            source_realized_sell_count=("source_realized_sell", "sum"),
            sell_source_floor_guard_count=("floor_guarded", "sum"),
            cash_reserve_signal=("cash_signal", "max"),
            top_receiver_score=("receiver_score", "max"),
            top_receiver_capacity=("receiver_capacity", "max"),
            top_source_score=("source_score", "max"),
            top_source_gap=("source_gap", "max"),
            top_source_release_capacity=("source_capacity", "max"),
            top_source_release_quality=("source_release_quality", "max"),
            top_source_executability=("source_exec", "max"),
            cash_score=("cash_score", "mean"),
        )
        .reset_index()
    )
    daily["source_realized_sell_rate"] = np.where(
        daily["source_target_count"] > 0,
        daily["source_realized_sell_count"] / daily["source_target_count"],
        0.0,
    )
    daily["receiver_realized_buy_rate"] = np.where(
        daily["receiver_target_count"] > 0,
        daily["receiver_realized_buy_count"] / daily["receiver_target_count"],
        0.0,
    )

    summary = {
        "protocol_dir": str(protocol_dir),
        "action_file": str(action_path),
        "protocol_summary_json": str(protocol_summary_path) if protocol_summary_path else "",
        "protocol_metric_scope_note": (
            "action_file metrics describe the exported/latest action window; "
            "protocol evaluation_continuity_metrics describe the full evaluation window when present"
        ),
        "protocol_metrics": protocol_metrics,
        "row_count": int(len(frame)),
        "date_count": int(frame["date"].nunique(dropna=True)),
        "receiver_target_count": int(receiver_target.sum()),
        "receiver_realized_buy_rate": _rate(realized_buy, receiver_target),
        "receiver_add_capacity_mean": _mean(receiver_capacity, receiver_target),
        "source_candidate_count": int(source_candidate.sum()),
        "source_target_count": int(source_target.sum()),
        "source_realized_sell_rate": _rate(realized_sell, source_target),
        "source_score_mean_on_targets": _mean(source_score, source_target),
        "source_gap_mean_on_targets": _mean(source_gap, source_target),
        "source_release_capacity_mean_on_targets": _mean(source_capacity, source_target),
        "source_release_quality_mean_on_targets": _mean(source_release_quality, source_target),
        "source_executability_mean_on_targets": _mean(source_exec, source_target),
        "cash_reserve_rate": float(cash_signal.groupby(frame["date"]).max().mean()) if len(frame) else 0.0,
        "cash_score_mean": _mean(cash_score),
        "sell_source_floor_guarded_count": int(floor_guarded.sum()),
        "source_disappearance": _source_disappearance(frame, protocol_metrics),
    }
    return summary, daily, action_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit portfolio-daily listwise receiver/source/cash execution paths.")
    parser.add_argument("--protocol-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    protocol_dir = args.protocol_dir
    output_dir = args.output_dir or (protocol_dir / "listwise_audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, daily, _ = build_audit(protocol_dir)
    summary_path = output_dir / "portfolio_daily_listwise_audit_summary.json"
    daily_path = output_dir / "portfolio_daily_listwise_audit_by_date.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    print(json.dumps({"summary": str(summary_path), "daily": str(daily_path), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
