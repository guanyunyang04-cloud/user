from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy.decision_score_proxy import add_path_proxy_decision_scores


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
SCORE_VARIANT_COLUMNS = (
    "pred_decision_score",
    "trade_utility_score",
    "max_pred_utility",
    "horizon_selected_utility",
    "hit_weighted_utility",
    "short_horizon_blend",
    "forecast_5d_mu",
    "forecast_20d_mu",
    "decision_forecast_blend",
)
STAGE2_SELECTED_LOSS_PROFILES = (
    "decision_utility_path_aux_v1",
    "decision_utility_v1_baseline",
)
PATH_PROXY_LOSS_PROFILES = {
    "default",
    "rank_aux",
    "multitask_v1",
    "forecast_path_v1_baseline",
}


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


def _decision_score_source(frame: pd.DataFrame) -> str:
    if "decision_score_source" not in frame.columns:
        return "unknown"
    values = sorted({str(item) for item in frame["decision_score_source"].dropna().unique() if str(item)})
    if not values:
        return "unknown"
    return values[0] if len(values) == 1 else "mixed"


def _prepare_prediction_frame(frame: pd.DataFrame, *, loss_profile: str) -> pd.DataFrame:
    profile = str(loss_profile or "").strip().lower()
    if profile in PATH_PROXY_LOSS_PROFILES:
        return _add_score_variant_columns(add_path_proxy_decision_scores(frame))
    out = frame.copy()
    if "decision_score_source" not in out.columns:
        out["decision_score_source"] = "model_decision_utility"
    return _add_score_variant_columns(out)


def _add_score_variant_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    horizons = _prediction_horizons(out)
    if "pred_decision_score" in out.columns and "max_pred_utility" not in out.columns:
        out["max_pred_utility"] = pd.to_numeric(out["pred_decision_score"], errors="coerce")
    utility_cols = [f"pred_decision_utility_{int(horizon)}d" for horizon in horizons if f"pred_decision_utility_{int(horizon)}d" in out.columns]
    hit_cols = [f"pred_hit_prob_{int(horizon)}d" for horizon in horizons if f"pred_hit_prob_{int(horizon)}d" in out.columns]
    if utility_cols and "trade_utility_score" not in out.columns:
        out["trade_utility_score"] = out[utility_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
    if utility_cols and hit_cols and len(utility_cols) == len(hit_cols) and "hit_weighted_utility" not in out.columns:
        out[utility_cols + hit_cols] = out[utility_cols + hit_cols].apply(pd.to_numeric, errors="coerce")
        out["hit_weighted_utility"] = (
            out[utility_cols].to_numpy(dtype=float) * out[hit_cols].to_numpy(dtype=float)
        ).max(axis=1)
    if utility_cols and "horizon_selected_utility" not in out.columns and "pred_best_horizon" in out.columns:
        horizon_to_col = {int(horizon): f"pred_decision_utility_{int(horizon)}d" for horizon in horizons}
        out["horizon_selected_utility"] = [
            _finite_float(row.get(horizon_to_col.get(int(_finite_float(row.get("pred_best_horizon"), 1)), ""), np.nan), np.nan)
            for _, row in out.iterrows()
        ]
    if utility_cols and "short_horizon_blend" not in out.columns:
        short_weights = (0.50, 0.30, 0.20)
        short_horizons = [item for item in (1, 3, 5) if f"pred_decision_utility_{item}d" in out.columns]
        if len(short_horizons) < 3:
            short_horizons = list(horizons[: min(3, len(horizons))])
        weight_sum = sum(short_weights[: len(short_horizons)]) or 1.0
        out["short_horizon_blend"] = sum(
            (short_weights[pos] / weight_sum) * pd.to_numeric(out[f"pred_decision_utility_{int(horizon)}d"], errors="coerce")
            for pos, horizon in enumerate(short_horizons)
        )
    if "forecast_5d_mu" not in out.columns:
        if "pred_cum_mu_5d" in out.columns:
            out["forecast_5d_mu"] = pd.to_numeric(out["pred_cum_mu_5d"], errors="coerce")
        elif "short_horizon_blend" in out.columns:
            out["forecast_5d_mu"] = out["short_horizon_blend"]
    if "forecast_20d_mu" not in out.columns:
        if "pred_cum_mu_20d" in out.columns:
            out["forecast_20d_mu"] = pd.to_numeric(out["pred_cum_mu_20d"], errors="coerce")
        elif "max_pred_utility" in out.columns:
            out["forecast_20d_mu"] = out["max_pred_utility"]
    if (
        "decision_forecast_blend" not in out.columns
        and "max_pred_utility" in out.columns
        and "forecast_5d_mu" in out.columns
        and "forecast_20d_mu" in out.columns
    ):
        out["decision_forecast_blend"] = (
            0.50 * pd.to_numeric(out["max_pred_utility"], errors="coerce")
            + 0.30 * pd.to_numeric(out["forecast_5d_mu"], errors="coerce")
            + 0.20 * pd.to_numeric(out["forecast_20d_mu"], errors="coerce")
        )
    return out


def _score_variant_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(column for column in SCORE_VARIANT_COLUMNS if column in frame.columns)


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


def _long_horizon_share(frame: pd.DataFrame) -> float:
    horizons = pd.to_numeric(frame["pred_best_horizon"], errors="coerce").dropna()
    if horizons.empty:
        return 0.0
    return float((horizons >= 15).mean())


def _thirty_d_concentration(frame: pd.DataFrame) -> float:
    horizons = pd.to_numeric(frame["pred_best_horizon"], errors="coerce").dropna()
    if horizons.empty:
        return 0.0
    return float((horizons == 30).mean())


def _future_long_horizon_share(frame: pd.DataFrame) -> float:
    horizons = pd.to_numeric(frame["future_best_horizon"], errors="coerce").dropna()
    if horizons.empty:
        return 0.0
    return float((horizons >= 15).mean())


def _pred_future_horizon_gap(frame: pd.DataFrame) -> float:
    pred = pd.to_numeric(frame["pred_best_horizon"], errors="coerce")
    future = pd.to_numeric(frame["future_best_horizon"], errors="coerce")
    valid = pred.notna() & future.notna()
    if int(valid.sum()) == 0:
        return 0.0
    return float((pred.loc[valid] - future.loc[valid]).abs().mean())


def _utility_distribution_summary(frame: pd.DataFrame, *, prefix: str) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)d$")
    for column in sorted(frame.columns):
        match = pattern.match(str(column))
        if match is None:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            rows[str(int(match.group(1)))] = {"mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
            continue
        rows[str(int(match.group(1)))] = {
            "mean": _finite_float(values.mean()),
            "std": _finite_float(values.std(ddof=0)),
            "p10": _finite_float(values.quantile(0.10)),
            "p50": _finite_float(values.quantile(0.50)),
            "p90": _finite_float(values.quantile(0.90)),
        }
    return rows


def _future_target_normalized_best_horizon_distribution(frame: pd.DataFrame) -> dict[str, int]:
    horizons = _prediction_horizons(frame)
    utility_cols = [f"future_decision_utility_{int(horizon)}d" for horizon in horizons]
    if not utility_cols:
        return {}
    utility = frame[utility_cols].apply(pd.to_numeric, errors="coerce")
    if utility.empty:
        return {}
    std = utility.std(axis=0, ddof=0).clip(lower=1.0e-8)
    normalized = (utility - utility.mean(axis=0)) / std
    best_idx = normalized.to_numpy(dtype=float).argmax(axis=1)
    horizon_values = np.asarray(horizons, dtype=int)
    best_horizons = pd.Series(horizon_values[best_idx])
    return {
        str(int(key)): int(value)
        for key, value in best_horizons.value_counts().sort_index().to_dict().items()
    }


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
        "decision_score_source": _decision_score_source(frame),
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
        "future_horizon_concentration": {
            str(int(key)): int(value)
            for key, value in pd.to_numeric(frame["future_best_horizon"], errors="coerce")
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "future_target_normalized_best_horizon_concentration": _future_target_normalized_best_horizon_distribution(frame),
        "future_utility_distribution": _utility_distribution_summary(frame, prefix="future_decision_utility"),
        "long_horizon_share": _long_horizon_share(frame),
        "thirty_d_concentration": _thirty_d_concentration(frame),
        "future_long_horizon_share": _future_long_horizon_share(frame),
        "pred_future_horizon_gap": _pred_future_horizon_gap(frame),
    }


def _score_variant_metrics(frame: pd.DataFrame, *, role: str) -> list[dict[str, Any]]:
    _require_prediction_columns(frame)
    horizons = _prediction_horizons(frame)
    rows: list[dict[str, Any]] = []
    for score_name in _score_variant_columns(frame):
        positive_rate, negative_months, worst_month = _monthly_positive_rate(
            frame,
            score_name,
            "future_decision_score",
        )
        rows.append(
            {
                "role": role,
                "score_name": score_name,
                "decision_score_source": _decision_score_source(frame),
                "decision_score_rank_ic": _rank_ic_by_date(frame, score_name, "future_decision_score"),
                "decision_score_top_bottom_spread": _top_bottom_spread_by_date(
                    frame,
                    score_name,
                    "future_decision_score",
                ),
                "decision_hit_lift_top20_mean": _hit_lift(frame, score_name, horizons),
                "monthly_spread_positive_rate": positive_rate,
                "negative_month_count": len(negative_months),
                "worst_month_spread": worst_month,
                "long_horizon_share": _long_horizon_share(frame),
                "thirty_d_concentration": _thirty_d_concentration(frame),
                "future_long_horizon_share": _future_long_horizon_share(frame),
                "pred_future_horizon_gap": _pred_future_horizon_gap(frame),
            }
        )
    return rows


def _study_tag(path: Path, training: dict[str, Any]) -> str:
    return str(training.get("study_tag") or training.get("tag") or path.name)


def _study_summary(study_dir: str | Path) -> dict[str, Any]:
    root = Path(study_dir)
    training = _read_json(root / "forecast_training_summary.json")
    config = dict(training.get("training_config", {}) or {})
    loss_profile = str(config.get("loss_profile", training.get("loss_profile", "")))
    try:
        validation = _prepare_prediction_frame(
            pd.read_csv(root / "forecast_predictions_validation.csv"),
            loss_profile=loss_profile,
        )
        test = _prepare_prediction_frame(
            pd.read_csv(root / "forecast_predictions_test.csv"),
            loss_profile=loss_profile,
        )
        validation_metrics = _summarize_predictions(validation, role="validation")
        test_metrics = _summarize_predictions(test, role="test")
        validation_score_variants = _score_variant_metrics(validation, role="validation")
        test_score_variants = _score_variant_metrics(test, role="test")
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
    configured_grid = str(config.get("horizon_grid_name", "") or "")
    horizon_grid = ",".join(str(int(item)) for item in horizons)
    horizon_grid_key = configured_grid or horizon_grid
    return {
        "study_tag": _study_tag(root, training),
        "study_dir": str(root),
        "status": "completed",
        "loss_profile": loss_profile,
        "output_profile": str(config.get("output_profile", "")),
        "model_family": str(training.get("selected_model_family", "")),
        "seed": int(training.get("selected_seed", 0) or 0),
        "feature_profile": str(config.get("feature_profile", training.get("feature_profile", ""))),
        "forecast_horizon": int(config.get("forecast_horizon", max(horizons) if horizons else 0) or 0),
        "cumulative_horizons": [int(item) for item in horizons],
        "horizon_grid_name": configured_grid,
        "horizon_grid": horizon_grid,
        "horizon_grid_key": horizon_grid_key,
        "daily_grid_feasibility": bool(len(horizons) >= 40 and max(horizons or (0,)) >= 45),
        "validation": validation_metrics,
        "test": test_metrics,
        "score_variants": [*validation_score_variants, *test_score_variants],
        "decision_score_source": validation_metrics.get("decision_score_source")
        if validation_metrics.get("decision_score_source") == test_metrics.get("decision_score_source")
        else "mixed",
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
    architecture_ready = [
        study
        for study in completed
        if not bool(study.get("daily_grid_feasibility"))
        and _passes_role(dict(study.get("validation", {})))
        and _passes_role(dict(study.get("test", {})), strong_test=True)
    ]
    if len(architecture_ready) >= 2:
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
                    "horizon_grid_key": study.get("horizon_grid_key", ""),
                    "horizon_grid": study.get("horizon_grid", ",".join(str(item) for item in study.get("cumulative_horizons", []) or [])),
                    "role": role,
                    "decision_score_source": study.get("decision_score_source", metrics.get("decision_score_source", "")),
                    "decision_score_rank_ic": metrics.get("decision_score_rank_ic", 0.0),
                    "decision_score_top_bottom_spread": metrics.get("decision_score_top_bottom_spread", 0.0),
                    "decision_hit_lift_top20_mean": metrics.get("decision_hit_lift_top20_mean", 0.0),
                    "monthly_spread_positive_rate": metrics.get("monthly_spread_positive_rate", 0.0),
                    "negative_month_count": metrics.get("negative_month_count", 0),
                    "worst_month_spread": metrics.get("worst_month_spread", 0.0),
                    "long_horizon_share": metrics.get("long_horizon_share", 0.0),
                    "thirty_d_concentration": metrics.get("thirty_d_concentration", 0.0),
                    "future_long_horizon_share": metrics.get("future_long_horizon_share", 0.0),
                    "pred_future_horizon_gap": metrics.get("pred_future_horizon_gap", 0.0),
                }
            )
    return rows


def score_variant_comparison_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for study in report.get("studies", []) or []:
        for metrics in study.get("score_variants", []) or []:
            row = {
                "study_tag": study.get("study_tag", ""),
                "study_status": study.get("status", ""),
                "loss_profile": study.get("loss_profile", ""),
                "output_profile": study.get("output_profile", ""),
                "model_family": study.get("model_family", ""),
                "seed": study.get("seed", 0),
                "feature_profile": study.get("feature_profile", ""),
                "horizon_grid_key": study.get("horizon_grid_key", ""),
                "horizon_grid": study.get("horizon_grid", ",".join(str(item) for item in study.get("cumulative_horizons", []) or [])),
                "role": metrics.get("role", ""),
                "score_name": metrics.get("score_name", ""),
                "decision_score_source": metrics.get("decision_score_source", study.get("decision_score_source", "")),
                "decision_score_rank_ic": metrics.get("decision_score_rank_ic", 0.0),
                "decision_score_top_bottom_spread": metrics.get("decision_score_top_bottom_spread", 0.0),
                "decision_hit_lift_top20_mean": metrics.get("decision_hit_lift_top20_mean", 0.0),
                "monthly_spread_positive_rate": metrics.get("monthly_spread_positive_rate", 0.0),
                "negative_month_count": metrics.get("negative_month_count", 0),
                "worst_month_spread": metrics.get("worst_month_spread", 0.0),
                "long_horizon_share": metrics.get("long_horizon_share", 0.0),
                "thirty_d_concentration": metrics.get("thirty_d_concentration", 0.0),
                "future_long_horizon_share": metrics.get("future_long_horizon_share", 0.0),
                "pred_future_horizon_gap": metrics.get("pred_future_horizon_gap", 0.0),
            }
            rows.append(row)
    return rows


def _mean(values: list[float]) -> float:
    return _finite_float(np.mean(values) if values else 0.0)


def _std(values: list[float]) -> float:
    return _finite_float(np.std(values, ddof=0) if values else 0.0)


def profile_aggregate_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = score_variant_comparison_rows(report)
    groups: dict[tuple[str, str, str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("study_status") != "completed":
            continue
        key = (
            str(row.get("loss_profile", "")),
            str(row.get("output_profile", "")),
            str(row.get("model_family", "")),
            str(row.get("feature_profile", "")),
            str(row.get("horizon_grid_key", "")),
            str(row.get("horizon_grid", "")),
            str(row.get("role", "")),
            str(row.get("score_name", "")),
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (
        loss_profile,
        output_profile,
        model_family,
        feature_profile,
        horizon_grid_key,
        horizon_grid,
        role,
        score_name,
    ), items in sorted(groups.items()):
        rank_ic = [_finite_float(item.get("decision_score_rank_ic")) for item in items]
        spread = [_finite_float(item.get("decision_score_top_bottom_spread")) for item in items]
        hit_lift = [_finite_float(item.get("decision_hit_lift_top20_mean")) for item in items]
        monthly = [_finite_float(item.get("monthly_spread_positive_rate")) for item in items]
        negative = [int(item.get("negative_month_count", 0) or 0) for item in items]
        worst = [_finite_float(item.get("worst_month_spread")) for item in items]
        long_share = [_finite_float(item.get("long_horizon_share")) for item in items]
        thirty_d = [_finite_float(item.get("thirty_d_concentration")) for item in items]
        future_long = [_finite_float(item.get("future_long_horizon_share")) for item in items]
        pred_future_gap = [_finite_float(item.get("pred_future_horizon_gap")) for item in items]
        seed_count = len({int(item.get("seed", 0) or 0) for item in items})
        horizon_count = len([item for item in horizon_grid.split(",") if item.strip()])
        daily_grid_feasibility = bool(
            horizon_count >= 40
            or "daily" in horizon_grid_key.lower()
            or "daily" in horizon_grid.lower()
        )
        all_positive = all(value > 0.0 for value in rank_ic) and all(value > 0.0 for value in spread) and all(
            value > 0.0 for value in hit_lift
        )
        stage3_weak_gate = bool(
            role == "test"
            and seed_count >= 3
            and not daily_grid_feasibility
            and all_positive
            and _mean(monthly) >= 0.75
            and max(negative or [99]) <= 2
        )
        out.append(
            {
                "loss_profile": loss_profile,
                "output_profile": output_profile,
                "model_family": model_family,
                "feature_profile": feature_profile,
                "horizon_grid_key": horizon_grid_key,
                "horizon_grid": horizon_grid,
                "role": role,
                "score_name": score_name,
                "seed_count": seed_count,
                "daily_grid_feasibility": daily_grid_feasibility,
                "rank_ic_mean": _mean(rank_ic),
                "rank_ic_min": min(rank_ic) if rank_ic else 0.0,
                "rank_ic_std": _std(rank_ic),
                "spread_mean": _mean(spread),
                "spread_min": min(spread) if spread else 0.0,
                "spread_std": _std(spread),
                "hit_lift_mean": _mean(hit_lift),
                "hit_lift_min": min(hit_lift) if hit_lift else 0.0,
                "hit_lift_std": _std(hit_lift),
                "monthly_positive_rate_mean": _mean(monthly),
                "monthly_positive_rate_min": min(monthly) if monthly else 0.0,
                "negative_month_count_mean": _mean([float(item) for item in negative]),
                "negative_month_count_max": max(negative) if negative else 0,
                "worst_month_spread_min": min(worst) if worst else 0.0,
                "long_horizon_share_mean": _mean(long_share),
                "thirty_d_concentration_mean": _mean(thirty_d),
                "future_long_horizon_share_mean": _mean(future_long),
                "pred_future_horizon_gap_mean": _mean(pred_future_gap),
                "all_seed_rank_ic_spread_hit_positive": all_positive,
                "stage3_weak_gate_pass": stage3_weak_gate,
            }
        )
    validation_lookup = {
        (
            row["loss_profile"],
            row["output_profile"],
            row["model_family"],
            row["feature_profile"],
            row["horizon_grid_key"],
            row["horizon_grid"],
            row["score_name"],
        ): row
        for row in out
        if row.get("role") == "validation"
    }
    for row in out:
        validation = validation_lookup.get(
            (
                row["loss_profile"],
                row["output_profile"],
                row["model_family"],
                row["feature_profile"],
                row["horizon_grid_key"],
                row["horizon_grid"],
                row["score_name"],
            ),
            {},
        )
        validation_rank = _finite_float(validation.get("rank_ic_mean"))
        validation_spread = _finite_float(validation.get("spread_mean"))
        validation_hit = _finite_float(validation.get("hit_lift_mean"))
        validation_monthly = _finite_float(validation.get("monthly_positive_rate_mean"))
        row["validation_rank_ic_mean"] = validation_rank
        row["validation_spread_mean"] = validation_spread
        row["validation_hit_lift_mean"] = validation_hit
        row["validation_monthly_positive_rate_mean"] = validation_monthly
        if row.get("role") == "test" and validation:
            row["validation_test_rank_ic_gap"] = _finite_float(row.get("rank_ic_mean")) - validation_rank
            row["validation_test_spread_gap"] = _finite_float(row.get("spread_mean")) - validation_spread
            row["validation_test_hit_lift_gap"] = _finite_float(row.get("hit_lift_mean")) - validation_hit
            row["validation_test_monthly_positive_rate_gap"] = _finite_float(row.get("monthly_positive_rate_mean")) - validation_monthly
            row["validation_test_metric_gap"] = row["validation_test_rank_ic_gap"]
        else:
            row["validation_test_rank_ic_gap"] = 0.0
            row["validation_test_spread_gap"] = 0.0
            row["validation_test_hit_lift_gap"] = 0.0
            row["validation_test_monthly_positive_rate_gap"] = 0.0
            row["validation_test_metric_gap"] = 0.0
    return out


def next_stage_decision(report: dict[str, Any]) -> dict[str, Any]:
    aggregates = profile_aggregate_rows(report)
    stage3_candidates = [
        row
        for row in aggregates
        if row.get("role") == "test"
        and row.get("score_name") == "pred_decision_score"
        and bool(row.get("stage3_weak_gate_pass"))
    ]
    return {
        "schema_version": 1,
        "run_tag": report.get("run_tag", ""),
        "stage2_selected_loss_profiles": list(STAGE2_SELECTED_LOSS_PROFILES),
        "stage2_horizon_grids": {
            "reuse_full": "1,2,3,5,8,10,15,20,30",
            "sparse_long": "1,3,5,10,15,20,30,45",
            "dense_short_mid": "1,2,3,4,5,8,10,15,20,30",
            "daily1_45_feas": ",".join(str(item) for item in range(1, 46)),
        },
        "rejected_loss_profiles": [
            "forecast_path_v1_baseline",
            "decision_utility_hit_risk_aux_v1",
            "decision_utility_rank_aux_v1",
        ],
        "stage3_architecture_allowed": bool(stage3_candidates),
        "stage3_architecture_gate": "test rank IC/spread/hit lift all positive across seeds, mean monthly positive rate >= 0.75, max test negative months <= 2",
        "stage3_candidates": stage3_candidates,
        "default_next_action": "run_stage2_horizon_grid_calibration",
        "boundary": "research-only / shadow-only; no active promotion, production root, trade-plan, paper account, or broker integration",
    }


def calibration_review_markdown(report: dict[str, Any]) -> str:
    decision = next_stage_decision(report)
    aggregates = [
        row
        for row in profile_aggregate_rows(report)
        if row.get("role") == "test" and row.get("score_name") == "pred_decision_score"
    ]
    lines = [
        "# Alpha Multi-Horizon Calibration Review",
        "",
        f"- status: `{report.get('status', '')}`",
        "- path-only baseline rejected: `forecast_path_v1_baseline` remains a negative control, not a Stage 2 candidate.",
        "- utility family kept: model-native decision utility remains the main research direction.",
        "- path-aux continue: `decision_utility_path_aux_v1` is the current primary Stage 2 candidate.",
        "- horizon chooser not solved: long-horizon concentration remains a calibration issue, not promotion evidence.",
        "- boundary: research-only / shadow-only; no active manifest, production root, trade-plan, paper account, or broker integration.",
        "",
        "## Stage 2 Decision",
        "",
        f"- selected loss profiles: `{', '.join(decision['stage2_selected_loss_profiles'])}`",
        f"- architecture allowed now: `{decision['stage3_architecture_allowed']}`",
        "",
        "## Test Aggregates",
    ]
    for row in sorted(aggregates, key=lambda item: _finite_float(item.get("rank_ic_mean")), reverse=True):
        lines.append(
            f"- `{row.get('loss_profile')}`: rank_ic_mean=`{_finite_float(row.get('rank_ic_mean')):.6f}`, "
            f"spread_mean=`{_finite_float(row.get('spread_mean')):.6f}`, "
            f"monthly_positive_rate_mean=`{_finite_float(row.get('monthly_positive_rate_mean')):.6f}`, "
            f"stage3_weak_gate_pass=`{bool(row.get('stage3_weak_gate_pass'))}`"
        )
    return "\n".join(lines) + "\n"


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
    profile_aggregate_csv = root / "profile_aggregate.csv"
    score_variant_csv = root / "score_variant_comparison.csv"
    next_stage_json = root / "next_stage_decision.json"
    calibration_review_md = root / "calibration_review.md"
    interaction_json = root / "architecture_input_interaction_report.json"
    verdict_md = root / "research_verdict.md"
    pd.DataFrame(output_profile_comparison_rows(report)).to_csv(output_profile_csv, index=False, encoding="utf-8")
    pd.DataFrame(horizon_grid_comparison_rows(report)).to_csv(horizon_grid_csv, index=False, encoding="utf-8")
    pd.DataFrame(profile_aggregate_rows(report)).to_csv(profile_aggregate_csv, index=False, encoding="utf-8")
    pd.DataFrame(score_variant_comparison_rows(report)).to_csv(score_variant_csv, index=False, encoding="utf-8")
    next_stage_json.write_text(json.dumps(_jsonable(next_stage_decision(report)), ensure_ascii=False, indent=2), encoding="utf-8")
    calibration_review_md.write_text(calibration_review_markdown(report), encoding="utf-8")
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
        "profile_aggregate_csv": profile_aggregate_csv,
        "score_variant_comparison_csv": score_variant_csv,
        "next_stage_decision_json": next_stage_json,
        "calibration_review_md": calibration_review_md,
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
