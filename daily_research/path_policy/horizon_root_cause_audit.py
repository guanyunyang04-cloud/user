from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT_CAUSE_ENUMS = (
    "true_long_horizon_edge",
    "target_bias_to_long",
    "horizon_head_collapse",
    "short_feature_insufficient",
    "training_instability",
)
SHORT_HORIZONS = (1, 2, 3, 5)
MID_HORIZONS = (8, 10, 15)
LONG_HORIZONS = (20, 30)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    return value


def infer_horizons(frame: pd.DataFrame) -> tuple[int, ...]:
    horizons: list[int] = []
    for column in frame.columns:
        match = re.match(r"^pred_decision_utility_(\d+)d$", str(column))
        if match:
            horizons.append(int(match.group(1)))
    if not horizons:
        raise ValueError("No pred_decision_utility_<horizon>d columns found.")
    return tuple(sorted(dict.fromkeys(horizons)))


def _missing_required_columns(frame: pd.DataFrame, horizons: tuple[int, ...]) -> list[str]:
    required = ["date", "pred_best_horizon", "future_best_horizon"]
    for horizon in horizons:
        required.extend(
            [
                f"pred_decision_utility_{horizon}d",
                f"future_decision_utility_{horizon}d",
                f"future_hit_label_{horizon}d",
            ]
        )
    return [column for column in required if column not in frame.columns]


def _validate_prediction_frames(validation: pd.DataFrame, test: pd.DataFrame) -> tuple[tuple[int, ...] | None, str]:
    if validation.empty or test.empty:
        return None, "validation/test predictions must both be non-empty"
    try:
        validation_horizons = infer_horizons(validation)
        test_horizons = infer_horizons(test)
    except ValueError as exc:
        return None, str(exc)
    if validation_horizons != test_horizons:
        return None, f"validation/test horizon mismatch: {validation_horizons} != {test_horizons}"
    missing = sorted(set(_missing_required_columns(validation, validation_horizons) + _missing_required_columns(test, test_horizons)))
    if missing:
        return None, f"missing required horizon root-cause columns: {missing}"
    return validation_horizons, ""


def _rank_ic_by_date(frame: pd.DataFrame, score_column: str, target_column: str) -> float:
    work = frame[["date", score_column, target_column]].copy()
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
    values: list[float] = []
    for _, group in work.dropna().groupby("date", sort=True):
        if len(group) < 3:
            continue
        corr = group[score_column].rank(method="average").corr(group[target_column].rank(method="average"))
        if pd.notna(corr) and math.isfinite(float(corr)):
            values.append(float(corr))
    return _finite_float(np.mean(values) if values else 0.0)


def _daily_spreads(frame: pd.DataFrame, score_column: str, target_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = frame[["date", score_column, target_column]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
    for date, group in work.dropna().groupby("date", sort=True):
        if len(group) < 2:
            continue
        k = max(int(len(group) * 0.20), 1)
        top = float(group.nlargest(k, score_column)[target_column].mean())
        bottom = float(group.nsmallest(k, score_column)[target_column].mean())
        rows.append({"date": pd.Timestamp(date), "spread": top - bottom, "top": top, "bottom": bottom})
    return pd.DataFrame(rows)


def _monthly_spread_summary(frame: pd.DataFrame, score_column: str, target_column: str) -> dict[str, Any]:
    daily = _daily_spreads(frame, score_column, target_column)
    if daily.empty:
        return {"monthly_spread": {}, "monthly_spread_positive_rate": 0.0, "negative_months": []}
    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    monthly = daily.groupby("month", sort=True)["spread"].mean()
    monthly_dict = {str(month): _finite_float(value) for month, value in monthly.items()}
    return {
        "monthly_spread": monthly_dict,
        "monthly_spread_positive_rate": _finite_float((monthly > 0.0).mean() if len(monthly) else 0.0),
        "negative_months": [str(month) for month, value in monthly_dict.items() if value <= 0.0],
    }


def _hit_lift_top20(frame: pd.DataFrame, score_column: str, hit_column: str) -> float:
    work = frame[["date", score_column, hit_column]].copy()
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    work[hit_column] = pd.to_numeric(work[hit_column], errors="coerce")
    lifts: list[float] = []
    for _, group in work.dropna().groupby("date", sort=True):
        if len(group) < 2:
            continue
        k = max(int(len(group) * 0.20), 1)
        lifts.append(float(group.nlargest(k, score_column)[hit_column].mean() - group[hit_column].mean()))
    return _finite_float(np.mean(lifts) if lifts else 0.0)


def _value_counts(series: pd.Series) -> dict[str, int]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return {str(int(key)): int(value) for key, value in numeric.astype(int).value_counts().sort_index().items()}


def _per_horizon_metrics(frame: pd.DataFrame, horizons: tuple[int, ...]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for horizon in horizons:
        score_column = f"pred_decision_utility_{horizon}d"
        target_column = f"future_decision_utility_{horizon}d"
        hit_column = f"future_hit_label_{horizon}d"
        target = pd.to_numeric(frame[target_column], errors="coerce")
        score = pd.to_numeric(frame[score_column], errors="coerce")
        hit = pd.to_numeric(frame[hit_column], errors="coerce")
        spreads = _daily_spreads(frame, score_column, target_column)
        monthly = _monthly_spread_summary(frame, score_column, target_column)
        rows[str(horizon)] = {
            "rank_ic": _rank_ic_by_date(frame, score_column, target_column),
            "top_bottom_spread": _finite_float(spreads["spread"].mean() if not spreads.empty else 0.0),
            "top20_target_mean": _finite_float(spreads["top"].mean() if not spreads.empty else 0.0),
            "bottom20_target_mean": _finite_float(spreads["bottom"].mean() if not spreads.empty else 0.0),
            "hit_lift_top20_mean": _hit_lift_top20(frame, score_column, hit_column),
            "target_mean": _finite_float(target.mean()),
            "target_std": _finite_float(target.std(ddof=0)),
            "prediction_mean": _finite_float(score.mean()),
            "prediction_std": _finite_float(score.std(ddof=0)),
            "hit_base_rate": _finite_float(hit.mean()),
            **monthly,
        }
    return rows


def _horizon_alignment(frame: pd.DataFrame) -> dict[str, Any]:
    pred = pd.to_numeric(frame["pred_best_horizon"], errors="coerce").dropna()
    future = pd.to_numeric(frame["future_best_horizon"], errors="coerce").dropna()
    pred_long = pred.isin([15, 20, 30])
    future_long = future.isin([15, 20, 30])
    return {
        "pred_best_horizon_distribution": _value_counts(pred),
        "future_best_horizon_distribution": _value_counts(future),
        "pred_30d_share": _finite_float((pred == 30).mean() if len(pred) else 0.0),
        "pred_long_share": _finite_float(pred_long.mean() if len(pred_long) else 0.0),
        "future_30d_share": _finite_float((future == 30).mean() if len(future) else 0.0),
        "future_long_share": _finite_float(future_long.mean() if len(future_long) else 0.0),
    }


def _role_summary(frame: pd.DataFrame, horizons: tuple[int, ...]) -> dict[str, Any]:
    return {
        "row_count": int(len(frame)),
        "date_count": int(pd.to_datetime(frame["date"], errors="coerce").nunique()),
        "per_horizon": _per_horizon_metrics(frame, horizons),
        "horizon_alignment": _horizon_alignment(frame),
    }


def _nested_get(payload: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _load_metadata(study_summary: dict[str, Any] | None, training_summary: dict[str, Any] | None) -> dict[str, Any]:
    study = study_summary or {}
    training = training_summary or {}
    nested_training = study.get("training_summary", {}) if isinstance(study.get("training_summary"), dict) else {}
    dataset_manifest = study.get("dataset_manifest", {}) if isinstance(study.get("dataset_manifest"), dict) else {}
    metadata = {
        "study_tag": study.get("study_tag", study.get("run_tag", "")),
        "dataset_id": study.get("lake_dataset_id", dataset_manifest.get("source_market_dataset_id", "")),
        "feature_profile": training.get("feature_profile", nested_training.get("feature_profile", "")),
        "model_family": training.get(
            "selected_model_family",
            nested_training.get(
                "selected_model_family",
                (study.get("forecast_model_families", study.get("model_families", [""])) or [""])[0],
            ),
        ),
        "seed": training.get("selected_seed", nested_training.get("selected_seed", study.get("seed", ""))),
        "role_years": dataset_manifest.get("role_years", {}),
        "training_config": training.get("training_config", nested_training.get("training_config", {})),
    }
    return _jsonable(metadata)


def _iter_dicts(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        rows.append(value)
        for child in value.values():
            rows.extend(_iter_dicts(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_iter_dicts(child))
    return rows


def _training_audit(training_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not training_summary:
        return {
            "status": "missing",
            "warnings": ["missing_training_summary"],
            "seed_runs": [],
        }
    seed_runs: list[dict[str, Any]] = []
    for row in _iter_dicts(training_summary):
        if "epochs_ran" not in row or "best_epoch" not in row:
            continue
        epochs_ran = int(_finite_float(row.get("epochs_ran")))
        best_epoch = int(_finite_float(row.get("best_epoch")))
        if epochs_ran <= 0:
            continue
        seed_runs.append(
            {
                "model_family": row.get("model_family", training_summary.get("selected_model_family", "")),
                "seed": row.get("seed", training_summary.get("selected_seed", "")),
                "epochs_ran": epochs_ran,
                "best_epoch": best_epoch,
                "stopped_reason": row.get("stopped_reason", ""),
                "best_validation_loss": row.get("best_validation_loss", ""),
                "best_score": row.get("best_score", ""),
            }
        )
    warnings: list[str] = []
    for run in seed_runs:
        epochs_ran = int(run["epochs_ran"])
        best_epoch = int(run["best_epoch"])
        stopped_reason = str(run.get("stopped_reason", ""))
        if best_epoch <= max(2, int(math.ceil(epochs_ran * 0.33))) and "early_stopping" in stopped_reason:
            warnings.append("early_best_epoch_warning")
            break
    return {
        "status": "ok" if seed_runs else "missing_seed_run_details",
        "warnings": sorted(dict.fromkeys(warnings)),
        "seed_runs": seed_runs,
    }


def _target_audit_context(target_audit: dict[str, Any] | None) -> dict[str, Any]:
    if not target_audit:
        return {"status": "missing"}
    rows: list[dict[str, Any]] = []
    for study in target_audit.get("studies", []) or []:
        if not isinstance(study, dict):
            continue
        for target in study.get("targets", []) or []:
            if not isinstance(target, dict):
                continue
            validation = target.get("validation", {}) if isinstance(target.get("validation"), dict) else {}
            test = target.get("test", {}) if isinstance(target.get("test"), dict) else {}
            rows.append(
                {
                    "tag": study.get("tag", ""),
                    "validation_hit_base_rate": validation.get("hit_base_rate", ""),
                    "test_hit_base_rate": test.get("hit_base_rate", ""),
                    "validation_monotonicity_status": _nested_get(validation, ("utility_decile_monotonicity", "status"), ""),
                    "test_monotonicity_status": _nested_get(test, ("utility_decile_monotonicity", "status"), ""),
                }
            )
    return {"status": "ok", "target_rows": rows}


def _horizon_calibration_context(horizon_calibration_report: dict[str, Any] | None) -> dict[str, Any]:
    if not horizon_calibration_report:
        return {"status": "missing"}
    gate = horizon_calibration_report.get("gate", {}) if isinstance(horizon_calibration_report.get("gate"), dict) else {}
    profiles = horizon_calibration_report.get("profiles", {}) if isinstance(horizon_calibration_report.get("profiles"), dict) else {}
    return {
        "status": horizon_calibration_report.get("status", ""),
        "gate": gate,
        "baseline": profiles.get("baseline_trade_utility", {}),
        "top2_blend": profiles.get("top2_blend", {}),
    }


def _mean_by_horizon(per_horizon: dict[str, dict[str, Any]], key: str, horizons: tuple[int, ...]) -> float:
    values = [_finite_float(per_horizon.get(str(horizon), {}).get(key)) for horizon in horizons if str(horizon) in per_horizon]
    return _finite_float(np.mean(values) if values else 0.0)


def _conclusion_enums(roles: dict[str, Any], training_audit: dict[str, Any]) -> list[str]:
    validation = roles.get("validation", {})
    test = roles.get("test", {})
    val_alignment = validation.get("horizon_alignment", {})
    test_alignment = test.get("horizon_alignment", {})
    test_per_horizon = test.get("per_horizon", {})
    val_per_horizon = validation.get("per_horizon", {})

    conclusions: list[str] = []
    pred_30d = max(_finite_float(val_alignment.get("pred_30d_share")), _finite_float(test_alignment.get("pred_30d_share")))
    future_long = max(_finite_float(val_alignment.get("future_long_share")), _finite_float(test_alignment.get("future_long_share")))
    if pred_30d >= 0.80 and future_long < 0.80:
        conclusions.append("horizon_head_collapse")
    short_rank = max(
        _mean_by_horizon(val_per_horizon, "rank_ic", SHORT_HORIZONS),
        _mean_by_horizon(test_per_horizon, "rank_ic", SHORT_HORIZONS),
    )
    long_rank = min(
        _mean_by_horizon(val_per_horizon, "rank_ic", LONG_HORIZONS),
        _mean_by_horizon(test_per_horizon, "rank_ic", LONG_HORIZONS),
    )
    short_spread = max(
        _mean_by_horizon(val_per_horizon, "top_bottom_spread", SHORT_HORIZONS),
        _mean_by_horizon(test_per_horizon, "top_bottom_spread", SHORT_HORIZONS),
    )
    long_spread = min(
        _mean_by_horizon(val_per_horizon, "top_bottom_spread", LONG_HORIZONS),
        _mean_by_horizon(test_per_horizon, "top_bottom_spread", LONG_HORIZONS),
    )
    if long_rank > max(short_rank * 1.5, 0.02) and long_spread > max(short_spread * 1.5, 0.002):
        conclusions.append("short_feature_insufficient")
    if future_long >= 0.70 or (
        long_spread > max(short_spread * 1.25, 0.002)
        and long_rank >= max(short_rank * 0.80, 0.02)
    ):
        conclusions.append("true_long_horizon_edge")

    short_target_std = max(
        _mean_by_horizon(val_per_horizon, "target_std", SHORT_HORIZONS),
        _mean_by_horizon(test_per_horizon, "target_std", SHORT_HORIZONS),
    )
    long_target_std = min(
        _mean_by_horizon(val_per_horizon, "target_std", LONG_HORIZONS),
        _mean_by_horizon(test_per_horizon, "target_std", LONG_HORIZONS),
    )
    short_target_mean = max(
        abs(_mean_by_horizon(val_per_horizon, "target_mean", SHORT_HORIZONS)),
        abs(_mean_by_horizon(test_per_horizon, "target_mean", SHORT_HORIZONS)),
    )
    long_target_mean = min(
        abs(_mean_by_horizon(val_per_horizon, "target_mean", LONG_HORIZONS)),
        abs(_mean_by_horizon(test_per_horizon, "target_mean", LONG_HORIZONS)),
    )
    if long_target_std > max(short_target_std * 1.5, 1.0e-9) or long_target_mean > max(short_target_mean * 1.5, 1.0e-9):
        conclusions.append("target_bias_to_long")

    if "early_best_epoch_warning" in (training_audit.get("warnings") or []):
        conclusions.append("training_instability")
    return [item for item in ROOT_CAUSE_ENUMS if item in set(conclusions)]


def _root_cause_matrix(conclusions: list[str]) -> dict[str, dict[str, Any]]:
    supported = set(conclusions)
    labels = {
        "true_long_horizon_edge": "future_best_horizon or realized horizon metrics also favor long horizons",
        "target_bias_to_long": "target scale/variance is materially larger at long horizons",
        "horizon_head_collapse": "pred_best_horizon is concentrated in 30d beyond realized best-horizon distribution",
        "short_feature_insufficient": "short-horizon rank/spread is materially weaker than long-horizon rank/spread",
        "training_instability": "selected checkpoint appears very early relative to epochs ran",
    }
    return {
        key: {
            "status": "supported" if key in supported else "not_supported_by_current_audit",
            "evidence_rule": labels[key],
        }
        for key in ROOT_CAUSE_ENUMS
    }


def _next_experiment_plan() -> dict[str, Any]:
    return {
        "phase_1_read_only_audit": "completed_by_this_report",
        "phase_2_target_function_experiments": target_experiment_configs(
            dataset_id="policy_input_bundle__7c8f58d851bce8179e1e9e2d",
            pool="liquid500",
            seed=7,
        ),
        "phase_3_architecture_candidates_after_target_gate": [
            "gru_sequence_static_context",
            "patch_transformer",
            "stock_mixer_sequence",
        ],
        "phase_3_input_candidates_after_target_gate": [
            "raw_kline_context_no_alpha_prior_v1",
            "raw_kline_context_v1",
            "raw_kline_context_short_signal_v1",
        ],
        "decision_gate": [
            "prediction metrics must pass validation/test",
            "execution simulation after costs must beat production baselines before paper shadow",
            "single seed can only support continue/stop research, not production",
        ],
    }


def target_experiment_configs(*, dataset_id: str, pool: str = "liquid500", seed: int = 7) -> list[dict[str, Any]]:
    base = {
        "dataset_id": dataset_id,
        "pool": pool,
        "train_years": [2019, 2020, 2021, 2022],
        "validation_year": 2023,
        "test_year": 2024,
        "feature_profile": "raw_kline_context_no_alpha_prior_v1",
        "model_family": "gru_sequence_static_context",
        "seed": int(seed),
        "scope": "shadow_only",
        "may_touch_active_manifest": False,
        "may_touch_production_root": False,
        "may_start_broker_path": False,
    }
    return [
        {**base, "tag": "mh_short_utility_1_3_5d_v1", "horizons": [1, 3, 5]},
        {**base, "tag": "mh_mid_utility_5_10_20d_v1", "horizons": [5, 10, 20]},
        {**base, "tag": "mh_long_utility_15_20_30d_v1", "horizons": [15, 20, 30]},
    ]


def evaluate_next_experiment_gate(experiment_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not experiment_results:
        return {
            "status": "evidence_insufficient",
            "selected_tag": "",
            "reason": "no target-function experiment results were provided",
        }
    short = experiment_results.get("mh_short_utility_1_3_5d_v1", {})
    if short.get("prediction_gate") == "passed" and short.get("execution_gate") != "passed":
        return {
            "status": "short_prediction_signal_execution_failed",
            "selected_tag": "mh_short_utility_1_3_5d_v1",
            "reason": "short target passed prediction checks but failed cost-adjusted execution checks",
        }
    if (
        short.get("prediction_gate") == "passed"
        and short.get("execution_gate") == "passed"
        and bool(short.get("beats_production_baseline"))
    ):
        return {
            "status": "shortline_candidate_ready_for_multiseed",
            "selected_tag": "mh_short_utility_1_3_5d_v1",
            "reason": "short target passed prediction and execution checks against the current production baseline",
        }
    for tag in ("mh_mid_utility_5_10_20d_v1", "mh_long_utility_15_20_30d_v1"):
        result = experiment_results.get(tag, {})
        if result.get("prediction_gate") == "passed" and result.get("execution_gate") == "passed":
            return {
                "status": "long_or_mid_family_wins_research_only",
                "selected_tag": tag,
                "reason": "mid/long target passed research gates; compare execution and multi-seed before any integration",
            }
    if any(result.get("training_status") in {"unstable", "failed"} for result in experiment_results.values()):
        return {
            "status": "training_instability",
            "selected_tag": "",
            "reason": "one or more target-function experiments reported unstable or failed training",
        }
    if all(result.get("prediction_gate") == "failed" for result in experiment_results.values()):
        return {
            "status": "short_feature_or_objective_not_supported",
            "selected_tag": "",
            "reason": "no target family passed prediction gates under the current dataset/input/model controls",
        }
    return {
        "status": "evidence_insufficient",
        "selected_tag": "",
        "reason": "experiment results are incomplete or mixed without a valid gate outcome",
    }


def build_horizon_root_cause_audit(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    *,
    study_summary: dict[str, Any] | None = None,
    training_summary: dict[str, Any] | None = None,
    target_audit: dict[str, Any] | None = None,
    horizon_calibration_report: dict[str, Any] | None = None,
    baseline_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    horizons, blocker = _validate_prediction_frames(validation_predictions, test_predictions)
    if blocker:
        return {"status": "blocked", "reason": blocker}
    assert horizons is not None
    roles = {
        "validation": _role_summary(validation_predictions, horizons),
        "test": _role_summary(test_predictions, horizons),
    }
    training = _training_audit(training_summary)
    conclusions = _conclusion_enums(roles, training)
    report = {
        "status": "completed",
        "schema_version": 1,
        "purpose": "research-only root-cause audit for alpha_multi_horizon_utility_policy_v1 horizon behavior",
        "horizons": [int(item) for item in horizons],
        "metadata": _load_metadata(study_summary, training_summary),
        "roles": roles,
        "training_audit": training,
        "target_audit_context": _target_audit_context(target_audit),
        "horizon_calibration_context": _horizon_calibration_context(horizon_calibration_report),
        "conclusion_enums": conclusions,
        "root_cause_matrix": _root_cause_matrix(conclusions),
        "baseline_context": baseline_context or {},
        "next_experiment_plan": _next_experiment_plan(),
        "production_boundaries": {
            "active_manifest_touched": False,
            "live_default_touched": False,
            "production_root_touched": False,
            "broker_touched": False,
            "training_started_by_audit": False,
        },
    }
    return _jsonable(report)


def artifact_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in ("validation", "test"):
        role_payload = (report.get("roles", {}) or {}).get(role, {})
        for horizon, payload in (role_payload.get("per_horizon", {}) or {}).items():
            rows.append(
                {
                    "role": role,
                    "horizon": int(horizon),
                    "rank_ic": payload.get("rank_ic", 0.0),
                    "top_bottom_spread": payload.get("top_bottom_spread", 0.0),
                    "hit_lift_top20_mean": payload.get("hit_lift_top20_mean", 0.0),
                    "monthly_spread_positive_rate": payload.get("monthly_spread_positive_rate", 0.0),
                    "negative_month_count": len(payload.get("negative_months", []) or []),
                    "target_mean": payload.get("target_mean", 0.0),
                    "target_std": payload.get("target_std", 0.0),
                    "hit_base_rate": payload.get("hit_base_rate", 0.0),
                }
            )
    return rows


def _markdown_report(report: dict[str, Any]) -> str:
    metadata = report.get("metadata", {}) or {}
    conclusions = ", ".join(report.get("conclusion_enums", []) or []) or "none"
    validation_alignment = (report.get("roles", {}) or {}).get("validation", {}).get("horizon_alignment", {})
    test_alignment = (report.get("roles", {}) or {}).get("test", {}).get("horizon_alignment", {})
    training_warnings = ", ".join(report.get("training_audit", {}).get("warnings", []) or []) or "none"
    gate = (report.get("horizon_calibration_context", {}) or {}).get("gate", {})
    lines = [
        "# Alpha Multi-Horizon Horizon Root-Cause Audit",
        "",
        "## Facts",
        f"- study_tag: `{metadata.get('study_tag', '')}`",
        f"- dataset_id: `{metadata.get('dataset_id', '')}`",
        f"- feature_profile: `{metadata.get('feature_profile', '')}`",
        f"- model_family: `{metadata.get('model_family', '')}`",
        f"- seed: `{metadata.get('seed', '')}`",
        f"- validation pred 30d share: `{validation_alignment.get('pred_30d_share', 0.0)}`",
        f"- test pred 30d share: `{test_alignment.get('pred_30d_share', 0.0)}`",
        f"- validation future long share: `{validation_alignment.get('future_long_share', 0.0)}`",
        f"- test future long share: `{test_alignment.get('future_long_share', 0.0)}`",
        f"- training warnings: `{training_warnings}`",
        f"- posthoc calibration gate: `{gate.get('status', '')}`",
        "",
        "## Inference",
        f"- conclusion_enums: `{conclusions}`",
        "- 当前报告只能说明现有 prediction 与训练摘要下的证据方向，不能证明模型能力边界已经到头。",
        "",
        "## Boundaries",
        "- research-only artifact.",
        "- 无 live/default、无 active manifest、无 production root、无 broker 下单。",
        "- 本审计不启动训练；后续短/中/长目标函数实验需要单独授权运行。",
    ]
    return "\n".join(lines) + "\n"


def write_horizon_root_cause_artifacts(report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "horizon_root_cause_audit.json"
    csv_path = root / "horizon_root_cause_audit.csv"
    markdown_path = root / "horizon_root_cause_audit.md"
    json_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(artifact_rows(report)).to_csv(csv_path, index=False, encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def _read_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _read_prediction_subset(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    header = pd.read_csv(csv_path, nrows=0)
    columns = list(header.columns)
    horizons = sorted(
        int(match.group(1))
        for column in columns
        for match in [re.match(r"^pred_decision_utility_(\d+)d$", str(column))]
        if match is not None
    )
    wanted = {"date", "stock", "role", "trade_utility_score", "future_decision_score", "pred_best_horizon", "future_best_horizon"}
    for horizon in horizons:
        wanted.update(
            {
                f"pred_decision_utility_{horizon}d",
                f"future_decision_utility_{horizon}d",
                f"future_hit_label_{horizon}d",
            }
        )
    return pd.read_csv(csv_path, usecols=[column for column in columns if column in wanted])


def _baseline_context_from_scoreboard(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    scoreboard = Path(path)
    if not scoreboard.exists():
        return {}
    frame = pd.read_csv(scoreboard)
    rows: dict[str, Any] = {"source_path": str(scoreboard.resolve())}
    if "profile" in frame.columns:
        key_column = "profile"
    elif "candidate" in frame.columns:
        key_column = "candidate"
    elif "label" in frame.columns:
        key_column = "label"
    else:
        key_column = frame.columns[0]
    for _, row in frame.head(20).iterrows():
        key = str(row.get(key_column, "")).strip()
        if not key:
            continue
        rows[key] = {
            column: _jsonable(row[column])
            for column in frame.columns
            if column != key_column and isinstance(row.get(column), (int, float, str, np.integer, np.floating))
        }
    return rows


def run_audit_from_files(
    *,
    validation_predictions_csv: str | Path,
    test_predictions_csv: str | Path,
    output_dir: str | Path,
    study_summary_json: str | Path | None = None,
    training_summary_json: str | Path | None = None,
    target_audit_json: str | Path | None = None,
    horizon_calibration_report_json: str | Path | None = None,
    baseline_scoreboard_csv: str | Path | None = None,
) -> dict[str, Any]:
    report = build_horizon_root_cause_audit(
        _read_prediction_subset(validation_predictions_csv),
        _read_prediction_subset(test_predictions_csv),
        study_summary=_read_json(study_summary_json),
        training_summary=_read_json(training_summary_json),
        target_audit=_read_json(target_audit_json),
        horizon_calibration_report=_read_json(horizon_calibration_report_json),
        baseline_context=_baseline_context_from_scoreboard(baseline_scoreboard_csv),
    )
    paths = write_horizon_root_cause_artifacts(report, output_dir)
    report["artifact_paths"] = {key: str(path.resolve()) for key, path in paths.items()}
    paths["json"].write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research-only root-cause audit for multi-horizon utility behavior.")
    parser.add_argument("--validation-predictions-csv", required=True)
    parser.add_argument("--test-predictions-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--study-summary-json", default="")
    parser.add_argument("--training-summary-json", default="")
    parser.add_argument("--target-audit-json", default="")
    parser.add_argument("--horizon-calibration-report-json", default="")
    parser.add_argument("--baseline-scoreboard-csv", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = run_audit_from_files(
        validation_predictions_csv=args.validation_predictions_csv,
        test_predictions_csv=args.test_predictions_csv,
        output_dir=args.output_dir,
        study_summary_json=args.study_summary_json or None,
        training_summary_json=args.training_summary_json or None,
        target_audit_json=args.target_audit_json or None,
        horizon_calibration_report_json=args.horizon_calibration_report_json or None,
        baseline_scoreboard_csv=args.baseline_scoreboard_csv or None,
    )
    if args.json:
        print(json.dumps(_jsonable(report), ensure_ascii=False, indent=2))
    else:
        print(
            "status={status} conclusions={conclusions}".format(
                status=report.get("status"),
                conclusions=",".join(report.get("conclusion_enums", []) or []),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
