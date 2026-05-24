from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OUTPUT_AUX_VERDICTS = {
    "utility_baseline_keep",
    "path_aux_continue",
    "hit_risk_aux_continue",
    "rank_aux_continue",
    "daily_grid_rejected",
    "daily_grid_continue_feasibility_only",
    "architecture_compare_allowed",
    "needs_input_redesign",
}
REQUIRED_PREDICTION_COLUMNS = (
    "date",
    "pred_decision_score",
    "future_decision_score",
    "trade_utility_score",
    "pred_best_horizon",
    "future_best_horizon",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _prediction_horizons(frame: pd.DataFrame) -> tuple[int, ...]:
    horizons = sorted(
        int(match.group(1))
        for column in frame.columns
        for match in [re.match(r"^pred_decision_utility_(\d+)d$", str(column))]
        if match is not None
    )
    return tuple(dict.fromkeys(horizons))


def _require_prediction_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in frame.columns]
    horizons = _prediction_horizons(frame)
    if not horizons:
        missing.append("pred_decision_utility_<horizon>d")
    for horizon in horizons:
        for column in (
            f"future_decision_utility_{horizon}d",
            f"pred_hit_prob_{horizon}d",
            f"future_hit_label_{horizon}d",
        ):
            if column not in frame.columns:
                missing.append(column)
    if missing:
        raise ValueError(f"missing required prediction columns: {missing}")


def _rank_ic_by_date(frame: pd.DataFrame, score_column: str, target_column: str) -> float:
    values: list[float] = []
    for _, group in frame.groupby("date", sort=True):
        work = group[[score_column, target_column]].copy()
        work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
        work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
        work = work.dropna()
        if len(work) < 2:
            continue
        corr = work[score_column].rank(method="average").corr(work[target_column].rank(method="average"))
        if pd.notna(corr) and math.isfinite(float(corr)):
            values.append(float(corr))
    return _finite_float(np.mean(values) if values else 0.0)


def _top_bottom_spread_by_date(frame: pd.DataFrame, score_column: str, target_column: str) -> float:
    values: list[float] = []
    for _, group in frame.groupby("date", sort=True):
        work = group[[score_column, target_column]].copy()
        work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
        work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
        work = work.dropna()
        if len(work) < 2:
            continue
        k = max(int(len(work) * 0.20), 1)
        top = work.nlargest(k, score_column)[target_column].mean()
        bottom = work.nsmallest(k, score_column)[target_column].mean()
        if pd.notna(top) and pd.notna(bottom):
            values.append(float(top - bottom))
    return _finite_float(np.mean(values) if values else 0.0)


def _monthly_positive_rate(frame: pd.DataFrame, score_column: str, target_column: str) -> tuple[float, list[str], float]:
    daily_rows: list[dict[str, Any]] = []
    for date, group in frame.groupby("date", sort=True):
        work = group[[score_column, target_column]].copy()
        work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
        work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
        work = work.dropna()
        if len(work) < 2:
            continue
        k = max(int(len(work) * 0.20), 1)
        daily_rows.append(
            {
                "date": pd.Timestamp(date),
                "spread": float(work.nlargest(k, score_column)[target_column].mean() - work.nsmallest(k, score_column)[target_column].mean()),
            }
        )
    if not daily_rows:
        return 0.0, [], 0.0
    daily = pd.DataFrame(daily_rows)
    daily["month"] = pd.to_datetime(daily["date"], errors="coerce").dt.strftime("%Y-%m")
    monthly = daily.groupby("month", sort=True)["spread"].mean()
    negative = [str(month) for month, value in monthly.items() if float(value) < 0.0]
    positive_rate = float((monthly > 0.0).mean()) if len(monthly) else 0.0
    worst_month = float(monthly.min()) if len(monthly) else 0.0
    return _finite_float(positive_rate), negative, _finite_float(worst_month)


def _hit_lift(frame: pd.DataFrame, score_column: str, horizons: tuple[int, ...]) -> float:
    hit_cols = [f"future_hit_label_{horizon}d" for horizon in horizons if f"future_hit_label_{horizon}d" in frame.columns]
    if not hit_cols:
        return 0.0
    work = frame[["date", score_column, *hit_cols]].copy()
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    for column in hit_cols:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["_hit_any"] = work[hit_cols].max(axis=1)
    lifts: list[float] = []
    for _, group in work.dropna(subset=[score_column, "_hit_any"]).groupby("date", sort=True):
        if len(group) < 2:
            continue
        k = max(int(len(group) * 0.20), 1)
        lifts.append(float(group.nlargest(k, score_column)["_hit_any"].mean() - group["_hit_any"].mean()))
    return _finite_float(np.mean(lifts) if lifts else 0.0)


def _summarize_predictions(frame: pd.DataFrame, *, role: str) -> dict[str, Any]:
    _require_prediction_columns(frame)
    horizons = _prediction_horizons(frame)
    positive_rate, negative_months, worst_month = _monthly_positive_rate(
        frame,
        "pred_decision_score",
        "future_decision_score",
    )
    return {
        "role": role,
        "status": "completed",
        "row_count": int(len(frame)),
        "horizons": [int(item) for item in horizons],
        "decision_score_rank_ic": _rank_ic_by_date(frame, "pred_decision_score", "future_decision_score"),
        "decision_score_top_bottom_spread": _top_bottom_spread_by_date(
            frame,
            "pred_decision_score",
            "future_decision_score",
        ),
        "decision_hit_lift_top20_mean": _hit_lift(frame, "pred_decision_score", horizons),
        "monthly_spread_positive_rate": positive_rate,
        "negative_months": negative_months,
        "negative_month_count": len(negative_months),
        "worst_month_spread": worst_month,
        "horizon_concentration": {
            str(int(key)): int(value)
            for key, value in pd.to_numeric(frame["pred_best_horizon"], errors="coerce")
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "long_horizon_share": float(pd.to_numeric(frame["pred_best_horizon"], errors="coerce").isin([15, 20, 30, 45]).mean()),
    }


def _study_tag(path: Path, training: dict[str, Any]) -> str:
    return str(training.get("study_tag") or training.get("tag") or path.name)


def _study_summary(study_dir: str | Path) -> dict[str, Any]:
    root = Path(study_dir)
    training = _read_json(root / "forecast_training_summary.json")
    config = dict(training.get("training_config", {}) or {})
    try:
        validation = pd.read_csv(root / "forecast_predictions_validation.csv")
        test = pd.read_csv(root / "forecast_predictions_test.csv")
        validation_metrics = _summarize_predictions(validation, role="validation")
        test_metrics = _summarize_predictions(test, role="test")
    except Exception as exc:
        return {
            "study_tag": _study_tag(root, training),
            "study_dir": str(root),
            "status": "blocked",
            "reason": str(exc),
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
    horizons = tuple(int(item) for item in test_metrics.get("horizons", []) or [])
    return {
        "study_tag": _study_tag(root, training),
        "study_dir": str(root),
        "status": "completed",
        "loss_profile": str(config.get("loss_profile", training.get("loss_profile", ""))),
        "output_profile": str(config.get("output_profile", "")),
        "model_family": str(training.get("selected_model_family", "")),
        "seed": int(training.get("selected_seed", 0) or 0),
        "feature_profile": str(config.get("feature_profile", training.get("feature_profile", ""))),
        "forecast_horizon": int(config.get("forecast_horizon", max(horizons) if horizons else 0) or 0),
        "cumulative_horizons": [int(item) for item in horizons],
        "horizon_grid_name": str(config.get("horizon_grid_name", "")),
        "daily_grid_feasibility": bool(len(horizons) >= 40 and max(horizons or (0,)) >= 45),
        "validation": validation_metrics,
        "test": test_metrics,
        "evidence_verdict": str(training.get("evidence_verdict", "")),
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }


def _passes_role(metrics: dict[str, Any], *, strong_test: bool = False) -> bool:
    return bool(
        metrics.get("status") == "completed"
        and _finite_float(metrics.get("decision_score_rank_ic")) > 0.0
        and _finite_float(metrics.get("decision_score_top_bottom_spread")) > 0.0
        and _finite_float(metrics.get("decision_hit_lift_top20_mean")) > 0.0
        and _finite_float(metrics.get("monthly_spread_positive_rate")) >= (0.90 if strong_test else 0.75)
        and int(metrics.get("negative_month_count", 99) or 99) <= (1 if strong_test else 2)
    )


def _derive_verdicts(studies: list[dict[str, Any]]) -> list[str]:
    completed = [study for study in studies if study.get("status") == "completed"]
    blocked = [study for study in studies if study.get("status") != "completed"]
    if not completed:
        return ["needs_input_redesign"]
    verdicts: set[str] = set()
    baseline = [
        study
        for study in completed
        if str(study.get("loss_profile")) in {"decision_utility_v1", "decision_utility_v1_baseline"}
    ]
    if baseline:
        verdicts.add("utility_baseline_keep")
    for study in completed:
        if bool(study.get("daily_grid_feasibility")):
            validation = dict(study.get("validation", {}) or {})
            test = dict(study.get("test", {}) or {})
            if (
                _finite_float(validation.get("decision_score_rank_ic")) > 0.0
                and _finite_float(validation.get("decision_score_top_bottom_spread")) > 0.0
                and _finite_float(test.get("decision_score_rank_ic")) > 0.0
                and _finite_float(test.get("decision_score_top_bottom_spread")) > 0.0
                and _finite_float(test.get("monthly_spread_positive_rate")) >= 0.75
            ):
                verdicts.add("daily_grid_continue_feasibility_only")
            else:
                verdicts.add("daily_grid_rejected")
            continue
        if not (_passes_role(dict(study.get("validation", {}))) and _passes_role(dict(study.get("test", {})), strong_test=True)):
            continue
        profile = str(study.get("loss_profile"))
        if profile == "decision_utility_path_aux_v1":
            verdicts.add("path_aux_continue")
        elif profile == "decision_utility_hit_risk_aux_v1":
            verdicts.add("hit_risk_aux_continue")
        elif profile == "decision_utility_rank_aux_v1":
            verdicts.add("rank_aux_continue")
    if len([study for study in completed if _passes_role(dict(study.get("validation", {}))) and _passes_role(dict(study.get("test", {})), strong_test=True)]) >= 2:
        verdicts.add("architecture_compare_allowed")
    if blocked and not verdicts:
        verdicts.add("needs_input_redesign")
    if not verdicts:
        verdicts.add("needs_input_redesign")
    return sorted(verdicts)


def build_output_aux_profile_comparison(study_dirs: list[str | Path], *, run_tag: str = "") -> dict[str, Any]:
    studies = [_study_summary(path) for path in study_dirs]
    status = "completed" if all(study.get("status") == "completed" for study in studies) else "completed_with_blocked_studies"
    return _jsonable(
        {
            "schema_version": 1,
            "run_tag": str(run_tag or ""),
            "status": status,
            "purpose": "research-only comparison of output/loss profiles, auxiliary tasks, and horizon grids",
            "studies": studies,
            "research_verdicts": _derive_verdicts(studies),
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
    )


def output_profile_comparison_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for study in report.get("studies", []) or []:
        for role in ("validation", "test"):
            metrics = dict(study.get(role, {}) or {})
            rows.append(
                {
                    "study_tag": study.get("study_tag", ""),
                    "study_status": study.get("status", ""),
                    "loss_profile": study.get("loss_profile", ""),
                    "output_profile": study.get("output_profile", ""),
                    "model_family": study.get("model_family", ""),
                    "seed": study.get("seed", 0),
                    "feature_profile": study.get("feature_profile", ""),
                    "role": role,
                    "decision_score_rank_ic": metrics.get("decision_score_rank_ic", 0.0),
                    "decision_score_top_bottom_spread": metrics.get("decision_score_top_bottom_spread", 0.0),
                    "decision_hit_lift_top20_mean": metrics.get("decision_hit_lift_top20_mean", 0.0),
                    "monthly_spread_positive_rate": metrics.get("monthly_spread_positive_rate", 0.0),
                    "negative_month_count": metrics.get("negative_month_count", 0),
                    "worst_month_spread": metrics.get("worst_month_spread", 0.0),
                    "long_horizon_share": metrics.get("long_horizon_share", 0.0),
                }
            )
    return rows


def horizon_grid_comparison_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for study in report.get("studies", []) or []:
        horizons = [int(item) for item in study.get("cumulative_horizons", []) or []]
        rows.append(
            {
                "study_tag": study.get("study_tag", ""),
                "status": study.get("status", ""),
                "loss_profile": study.get("loss_profile", ""),
                "forecast_horizon": study.get("forecast_horizon", 0),
                "horizon_count": len(horizons),
                "horizon_min": min(horizons) if horizons else 0,
                "horizon_max": max(horizons) if horizons else 0,
                "horizon_grid": ",".join(str(item) for item in horizons),
                "daily_grid_feasibility": bool(study.get("daily_grid_feasibility", False)),
                "validation_rank_ic": dict(study.get("validation", {}) or {}).get("decision_score_rank_ic", 0.0),
                "test_rank_ic": dict(study.get("test", {}) or {}).get("decision_score_rank_ic", 0.0),
                "test_spread": dict(study.get("test", {}) or {}).get("decision_score_top_bottom_spread", 0.0),
            }
        )
    return rows


def write_output_aux_profile_comparison(report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    output_profile_csv = root / "output_profile_comparison.csv"
    horizon_grid_csv = root / "horizon_grid_comparison.csv"
    interaction_json = root / "architecture_input_interaction_report.json"
    verdict_md = root / "research_verdict.md"
    pd.DataFrame(output_profile_comparison_rows(report)).to_csv(output_profile_csv, index=False, encoding="utf-8")
    pd.DataFrame(horizon_grid_comparison_rows(report)).to_csv(horizon_grid_csv, index=False, encoding="utf-8")
    interaction_json.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    verdict_lines = [
        "# Alpha Multi-Horizon Output/Aux/Grid Comparison",
        "",
        f"- status: `{report.get('status', '')}`",
        f"- verdicts: `{', '.join(report.get('research_verdicts', []) or [])}`",
        "- boundary: research-only / shadow-only; no active manifest, production root, trade-plan, paper account, or broker integration.",
        "",
        "## Studies",
    ]
    for study in report.get("studies", []) or []:
        verdict_lines.append(
            f"- `{study.get('study_tag', '')}`: status=`{study.get('status', '')}`, "
            f"loss=`{study.get('loss_profile', '')}`, horizons=`{','.join(str(item) for item in study.get('cumulative_horizons', []) or [])}`"
        )
    verdict_md.write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")
    return {
        "output_profile_comparison_csv": output_profile_csv,
        "horizon_grid_comparison_csv": horizon_grid_csv,
        "architecture_input_interaction_report_json": interaction_json,
        "research_verdict_md": verdict_md,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare multi-horizon output, auxiliary loss, and horizon grid studies.")
    parser.add_argument("--study-dir", action="append", required=True, help="Study directory containing forecast artifacts. Repeatable.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = build_output_aux_profile_comparison(args.study_dir, run_tag=args.run_tag)
    paths = write_output_aux_profile_comparison(report, args.output_dir)
    report["artifact_paths"] = {key: str(path.resolve()) for key, path in paths.items()}
    if args.json:
        print(json.dumps(_jsonable(report), ensure_ascii=False, indent=2))
    else:
        print(f"status={report.get('status')} verdicts={','.join(report.get('research_verdicts', []) or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
