from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from daily_research.path_policy.labels import PATH20_CUMULATIVE_HORIZONS, PATH20_HORIZON


STUDIES_ROOT = Path("daily_research/output/path_policy/studies")
DEFAULT_TARGET_GRID = "20:10:0.10"
HIT_BASE_RATE_MIN = 0.02
HIT_BASE_RATE_MAX = 0.95
MONTHLY_SPREAD_POSITIVE_RATE_MIN = 0.60
CONTEXT_COLUMNS = (
    "history_bucket",
    "stock_seen_in_train",
    "model_family",
    "pred_best_horizon",
    "future_best_horizon",
    "industry_name",
    "industry",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return float(default)
    return resolved if np.isfinite(resolved) else float(default)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"target_calibration_audit missing required columns: {missing}")


def parse_target_grid(value: str | None) -> list[dict[str, float]]:
    text = str(value or DEFAULT_TARGET_GRID).strip()
    out: list[dict[str, float]] = []
    for item in [part.strip() for part in text.split(",") if part.strip()]:
        pieces = [part.strip() for part in item.split(":")]
        if len(pieces) != 3:
            raise ValueError("target grid entries must use cost_bps:hit_threshold_bps:drawdown_penalty")
        try:
            cost_bps, hit_threshold_bps, drawdown_penalty = (float(part) for part in pieces)
        except ValueError as exc:
            raise ValueError("target grid entries must be numeric cost_bps:hit_threshold_bps:drawdown_penalty") from exc
        if cost_bps < 0.0:
            raise ValueError("target grid cost_bps must be non-negative")
        if drawdown_penalty < 0.0:
            raise ValueError("target grid drawdown_penalty must be non-negative")
        out.append(
            {
                "cost_bps": float(cost_bps),
                "hit_threshold_bps": float(hit_threshold_bps),
                "drawdown_penalty": float(drawdown_penalty),
            }
        )
    if not out:
        raise ValueError("target grid must contain at least one entry")
    return out


def _target_name(target: dict[str, float]) -> str:
    return (
        f"cost{_finite_float(target.get('cost_bps')):g}_"
        f"hit{_finite_float(target.get('hit_threshold_bps')):g}_"
        f"dd{_finite_float(target.get('drawdown_penalty')):g}"
    )


def _decision_utility_columns(frame: pd.DataFrame, target: dict[str, float]) -> pd.DataFrame:
    horizons = list(PATH20_CUMULATIVE_HORIZONS)
    required = [f"future_cum_excess_return_{int(horizon)}d" for horizon in horizons]
    required.append("future_path_max_drawdown_20d")
    _require_columns(frame, required)

    y_cum = frame[[f"future_cum_excess_return_{int(horizon)}d" for horizon in horizons]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    max_dd = pd.to_numeric(frame["future_path_max_drawdown_20d"], errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
    horizon_arr = np.asarray(horizons, dtype=float).reshape(1, -1)
    downside = np.maximum(0.0, -max_dd)
    utility = (
        y_cum.to_numpy(dtype=float)
        - _finite_float(target.get("cost_bps")) / 10000.0
        - _finite_float(target.get("drawdown_penalty")) * downside * np.sqrt(horizon_arr / float(PATH20_HORIZON))
    )
    audit_columns = [column for column in frame.columns if str(column).startswith("audit_")]
    result = frame.drop(columns=audit_columns).copy() if audit_columns else frame.copy()
    for pos, horizon in enumerate(horizons):
        result[f"audit_future_utility_{int(horizon)}d"] = utility[:, pos]
        result[f"audit_future_hit_label_{int(horizon)}d"] = utility[:, pos] > (
            _finite_float(target.get("hit_threshold_bps")) / 10000.0
        )
    finite_utility = np.where(np.isfinite(utility), utility, -np.inf)
    decision_utility = np.nanmax(utility, axis=1)
    result["audit_future_utility"] = pd.Series(decision_utility, index=result.index, dtype=float)
    result["audit_best_horizon"] = [horizons[int(pos)] for pos in np.nanargmax(finite_utility, axis=1)]
    result["audit_hit_label"] = decision_utility > (_finite_float(target.get("hit_threshold_bps")) / 10000.0)
    return result


def _rank_ic_by_date(frame: pd.DataFrame, score_column: str, target_column: str) -> float:
    values: list[float] = []
    for _, group in frame[["date", score_column, target_column]].dropna().groupby("date", sort=True):
        if len(group) < 3:
            continue
        corr = group[score_column].rank().corr(group[target_column].rank())
        if pd.notna(corr) and np.isfinite(float(corr)):
            values.append(float(corr))
    return float(np.mean(values)) if values else 0.0


def _daily_spreads(frame: pd.DataFrame, score_column: str, target_column: str, frac: float = 0.20) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, group in frame[["date", score_column, target_column]].dropna().groupby("date", sort=True):
        if len(group) < 2:
            continue
        k = max(int(len(group) * float(frac)), 1)
        top = float(group.nlargest(k, score_column)[target_column].mean())
        bottom = float(group.nsmallest(k, score_column)[target_column].mean())
        rows.append({"date": pd.Timestamp(date), "spread": top - bottom})
    return pd.DataFrame(rows)


def _monthly_spread_summary(frame: pd.DataFrame, score_column: str, target_column: str) -> dict[str, Any]:
    daily = _daily_spreads(frame, score_column, target_column)
    if daily.empty:
        return {
            "monthly_spread": {},
            "monthly_spread_positive_rate": 0.0,
            "negative_months": [],
            "worst_month": "",
            "worst_month_spread": 0.0,
        }
    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    monthly = daily.groupby("month", sort=True)["spread"].mean()
    monthly_dict = {str(key): float(value) for key, value in monthly.items()}
    negative_months = [month for month, value in monthly_dict.items() if float(value) <= 0.0]
    worst_month = str(monthly.idxmin()) if len(monthly) else ""
    return {
        "monthly_spread": monthly_dict,
        "monthly_spread_positive_rate": float((monthly > 0.0).mean()) if len(monthly) else 0.0,
        "negative_months": negative_months,
        "worst_month": worst_month,
        "worst_month_spread": float(monthly.loc[worst_month]) if worst_month else 0.0,
    }


def _top_bottom_spread_by_date(frame: pd.DataFrame, score_column: str, target_column: str) -> float:
    daily = _daily_spreads(frame, score_column, target_column)
    return float(daily["spread"].mean()) if len(daily) else 0.0


def _hit_lift_top20(frame: pd.DataFrame, score_column: str) -> float:
    work = frame[["date", score_column, "audit_hit_label"]].copy()
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    work["audit_hit_label"] = work["audit_hit_label"].astype(float)
    lifts: list[float] = []
    for _, group in work.dropna(subset=[score_column, "audit_hit_label"]).groupby("date", sort=True):
        if len(group) < 2:
            continue
        k = max(int(len(group) * 0.20), 1)
        top_hit = float(group.nlargest(k, score_column)["audit_hit_label"].mean())
        all_hit = float(group["audit_hit_label"].mean())
        lifts.append(top_hit - all_hit)
    return float(np.mean(lifts)) if lifts else 0.0


def _decile_index(series: pd.Series, deciles: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    ranked = numeric.rank(method="first", pct=True)
    return np.ceil(ranked * int(deciles)).clip(1, int(deciles)).astype("Int64")


def _decile_payload(frame: pd.DataFrame, by_column: str, *, deciles: int) -> list[dict[str, Any]]:
    columns = list(dict.fromkeys(["date", by_column, "audit_future_utility", "audit_hit_label"]))
    work = frame[columns].copy()
    work["_decile"] = _decile_index(work[by_column], deciles)
    rows: list[dict[str, Any]] = []
    for decile, group in work.dropna(subset=["_decile"]).groupby("_decile", sort=True):
        rows.append(
            {
                "decile": int(decile),
                "row_count": int(len(group)),
                "future_utility_mean": _finite_float(pd.to_numeric(group["audit_future_utility"], errors="coerce").mean()),
                "hit_rate": _finite_float(group["audit_hit_label"].astype(float).mean()),
            }
        )
    return rows


def _monotonicity(rows: list[dict[str, Any]], value_key: str = "future_utility_mean") -> dict[str, Any]:
    values = [float(row.get(value_key, 0.0) or 0.0) for row in rows]
    decreases = sum(1 for prev, current in zip(values, values[1:]) if current + 1.0e-12 < prev)
    if len(values) >= 2:
        corr = pd.Series(range(1, len(values) + 1), dtype=float).corr(pd.Series(values, dtype=float), method="spearman")
    else:
        corr = 0.0
    corr_value = _finite_float(corr)
    return {
        "is_monotonic": decreases == 0,
        "adjacent_decrease_count": int(decreases),
        "spearman": corr_value,
        "status": "passed" if decreases == 0 and corr_value > 0.0 else "failed",
    }


def _value_counts_payload(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.astype(str).value_counts(dropna=False).sort_index().items()}


def _negative_month_context(frame: pd.DataFrame, score_column: str, negative_months: list[str]) -> dict[str, Any]:
    if not negative_months:
        return {}
    work = frame.copy()
    work["month"] = pd.to_datetime(work["date"]).dt.to_period("M").astype(str)
    work = work[work["month"].isin(negative_months)].dropna(subset=[score_column, "audit_future_utility"])
    if work.empty:
        return {}
    rows: dict[str, Any] = {}
    for month, month_frame in work.groupby("month", sort=True):
        k = max(int(len(month_frame) * 0.20), 1)
        top = month_frame.nlargest(k, score_column)
        context: dict[str, Any] = {"top20_count": int(len(top))}
        for column in CONTEXT_COLUMNS:
            if column in top.columns:
                context[column] = _value_counts_payload(top[column])
        alpha_cols = [column for column in top.columns if str(column).startswith("alpha_prior_")]
        if alpha_cols:
            context["alpha_prior_means"] = {
                column: _finite_float(pd.to_numeric(top[column], errors="coerce").mean())
                for column in alpha_cols[:12]
            }
        rows[str(month)] = context
    return rows


def _hit_base_rate_status(rate: float) -> str:
    if float(rate) < HIT_BASE_RATE_MIN:
        return "too_sparse"
    if float(rate) > HIT_BASE_RATE_MAX:
        return "too_dense"
    return "ok"


def _conclusion_hint(*, score_calibration: dict[str, Any], hit_base_rate_status: str, gate_passed: bool) -> str:
    if (
        _finite_float(score_calibration.get("future_utility_top_bottom_spread")) > 0.0
        and _finite_float(score_calibration.get("monthly_spread_positive_rate")) >= MONTHLY_SPREAD_POSITIVE_RATE_MIN
        and _finite_float(score_calibration.get("hit_lift_top20_mean")) <= 0.0
    ):
        return "ranking_signal_exists_but_hit_target_is_miscalibrated"
    if gate_passed:
        return "target_calibration_gate_passed"
    if hit_base_rate_status != "ok":
        return "hit_label_base_rate_out_of_range"
    if _finite_float(score_calibration.get("monthly_spread_positive_rate")) < MONTHLY_SPREAD_POSITIVE_RATE_MIN:
        return "score_selection_or_regime_instability"
    return "target_calibration_inconclusive"


def summarize_target_frame(
    frame: pd.DataFrame,
    target: dict[str, float],
    *,
    score_column: str = "pred_decision_score",
    deciles: int = 10,
) -> dict[str, Any]:
    _require_columns(frame, ["date", score_column])
    work = _decision_utility_columns(frame, target)
    work["date"] = pd.to_datetime(work["date"])
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    utility_deciles = _decile_payload(work, "audit_future_utility", deciles=deciles)
    score_deciles = _decile_payload(work, score_column, deciles=deciles)
    monthly = _monthly_spread_summary(work, score_column, "audit_future_utility")
    hit_base_rate = _finite_float(work["audit_hit_label"].astype(float).mean())
    hit_status = _hit_base_rate_status(hit_base_rate)
    score_calibration = {
        "score_deciles": score_deciles,
        "rank_ic": _rank_ic_by_date(work, score_column, "audit_future_utility"),
        "future_utility_top_bottom_spread": _top_bottom_spread_by_date(work, score_column, "audit_future_utility"),
        "hit_lift_top20_mean": _hit_lift_top20(work, score_column),
        **monthly,
    }
    score_calibration["monthly_negative_context"] = _negative_month_context(
        work,
        score_column,
        list(monthly.get("negative_months", []) or []),
    )
    monotonicity = _monotonicity(utility_deciles)
    score_alignment_checks = {
        "score_rank_ic_positive": _finite_float(score_calibration["rank_ic"]) > 0.0,
        "score_spread_positive": _finite_float(score_calibration["future_utility_top_bottom_spread"]) > 0.0,
        "score_hit_lift_positive": _finite_float(score_calibration["hit_lift_top20_mean"]) > 0.0,
        "monthly_spread_positive_rate_ge_60pct": _finite_float(score_calibration["monthly_spread_positive_rate"]) >= MONTHLY_SPREAD_POSITIVE_RATE_MIN,
    }
    score_calibration["alignment_checks"] = score_alignment_checks
    checks = {
        "utility_decile_monotonic": bool(monotonicity["is_monotonic"]),
        "hit_base_rate_ok": hit_status == "ok",
    }
    gate = {"passed": all(checks.values()), "checks": checks}
    return {
        "target": {**target, "name": _target_name(target)},
        "row_count": int(len(work)),
        "date_count": int(work["date"].nunique()),
        "hit_base_rate": hit_base_rate,
        "hit_base_rate_status": hit_status,
        "utility_deciles": utility_deciles,
        "utility_decile_monotonicity": monotonicity,
        "score_decile_calibration": score_calibration,
        "monthly_negative_context": score_calibration["monthly_negative_context"],
        "gate_a": gate,
        "conclusion_hint": _conclusion_hint(
            score_calibration=score_calibration,
            hit_base_rate_status=hit_status,
            gate_passed=bool(gate["passed"]),
        ),
    }


def summarize_study_target_calibration(
    tag: str,
    *,
    studies_root: Path = STUDIES_ROOT,
    target_grid: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    study_root = Path(studies_root) / str(tag)
    if not study_root.exists():
        raise FileNotFoundError(f"study tag not found: {tag}")
    summary_path = study_root / "study_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"study summary not found for tag={tag}: {summary_path}")
    study_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(study_summary.get("status", "")).lower() != "completed":
        raise ValueError(f"study tag is not completed: {tag}")
    validation_path = study_root / "forecast_predictions_validation.csv"
    test_path = study_root / "forecast_predictions_test.csv"
    if not validation_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"study tag missing forecast prediction CSVs: {tag}")

    targets = []
    for target in list(target_grid or parse_target_grid(DEFAULT_TARGET_GRID)):
        validation = summarize_target_frame(pd.read_csv(validation_path), target)
        test = summarize_target_frame(pd.read_csv(test_path), target)
        drift = abs(_finite_float(validation.get("hit_base_rate")) - _finite_float(test.get("hit_base_rate")))
        checks = {
            "validation_gate_a_passed": bool(validation["gate_a"]["passed"]),
            "test_gate_a_passed": bool(test["gate_a"]["passed"]),
            "hit_base_rate_drift_le_25pct": drift <= 0.25,
        }
        targets.append(
            {
                "target": {**target, "name": _target_name(target)},
                "validation": validation,
                "test": test,
                "validation_test_hit_base_rate_abs_diff": float(drift),
                "gate_a": {"passed": all(checks.values()), "checks": checks},
            }
        )
    return {
        "tag": str(tag),
        "status": study_summary.get("status"),
        "evidence_verdict": study_summary.get("evidence_verdict"),
        "model_families": study_summary.get("model_families", []),
        "dataset_ids": [
            item
            for item in [
                study_summary.get("lake_dataset_id"),
                study_summary.get("dataset_manifest", {}).get("source_pool_view_id"),
                study_summary.get("dataset_manifest", {}).get("source_sector_board_view_id"),
            ]
            if item
        ],
        "role_years": study_summary.get("dataset_manifest", {}).get("role_years", {}),
        "targets": targets,
    }


def build_target_calibration_audit(
    tags: list[str],
    *,
    studies_root: Path = STUDIES_ROOT,
    target_grid: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    if not tags:
        raise ValueError("target calibration audit requires at least one study tag")
    resolved_grid = list(target_grid or parse_target_grid(DEFAULT_TARGET_GRID))
    studies = [
        summarize_study_target_calibration(tag, studies_root=Path(studies_root), target_grid=resolved_grid)
        for tag in tags
    ]
    passed = [
        {"tag": study["tag"], "target": target["target"]["name"]}
        for study in studies
        for target in study["targets"]
        if target["gate_a"]["passed"]
    ]
    hints = sorted(
        {
            target[role]["conclusion_hint"]
            for study in studies
            for target in study["targets"]
            for role in ("validation", "test")
        }
    )
    return {
        "schema_version": 1,
        "purpose": "Path20 target calibration audit before new training.",
        "target_grid": [{**target, "name": _target_name(target)} for target in resolved_grid],
        "studies": studies,
        "passed_target_candidates": passed,
        "gate_a": {"passed": bool(passed), "passed_target_count": int(len(passed))},
        "conclusion_hints": hints,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Path20 target calibration before new training.")
    parser.add_argument("--tags", required=True, help="Comma-separated completed study tags.")
    parser.add_argument("--studies-root", default=str(STUDIES_ROOT))
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-grid", default=DEFAULT_TARGET_GRID, help="cost_bps:hit_threshold_bps:drawdown_penalty,...")
    args = parser.parse_args(argv)

    tags = [item.strip() for item in str(args.tags).split(",") if item.strip()]
    payload = build_target_calibration_audit(
        tags,
        studies_root=Path(args.studies_root),
        target_grid=parse_target_grid(str(args.target_grid)),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "gate_a_passed": payload["gate_a"]["passed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
