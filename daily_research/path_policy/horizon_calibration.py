from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROFILE_COLUMNS = (
    "baseline_trade_utility",
    "softmax_expected_utility_t015",
    "long_blend_15_20_30",
    "long_blend_10_15_20_30",
    "top2_blend",
)
LONG_BLEND_REQUIRED_HORIZONS = (10, 15, 20, 30)


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
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def infer_decision_utility_horizons(frame: pd.DataFrame) -> tuple[int, ...]:
    horizons: list[int] = []
    for column in frame.columns:
        match = re.match(r"^pred_decision_utility_(\d+)d$", str(column))
        if match:
            horizons.append(int(match.group(1)))
    if not horizons:
        raise ValueError("No pred_decision_utility_<horizon>d columns found.")
    return tuple(sorted(dict.fromkeys(horizons)))


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required decision utility columns: {missing}")


def _softmax_expected(values: np.ndarray, horizons: np.ndarray, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    scaled = values / max(float(temperature), 1.0e-9)
    scaled = scaled - np.nanmax(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    weights = exp / np.clip(np.nansum(exp, axis=1, keepdims=True), 1.0e-12, None)
    score = np.nansum(weights * values, axis=1)
    implied_horizon = np.nansum(weights * horizons.reshape(1, -1), axis=1)
    return score, implied_horizon


def _top2_blend(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(values, axis=1)
    if sorted_values.shape[1] == 1:
        return sorted_values[:, -1]
    return 0.70 * sorted_values[:, -1] + 0.30 * sorted_values[:, -2]


def _top_horizon(values: np.ndarray, horizons: tuple[int, ...], *, rank_from_top: int = 1) -> np.ndarray:
    order = np.argsort(values, axis=1)
    rank = max(int(rank_from_top), 1)
    index = -min(rank, order.shape[1])
    return np.asarray(horizons, dtype=float)[order[:, index]]


def score_horizon_profiles(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[int, ...]]:
    horizons = infer_decision_utility_horizons(frame)
    utility_columns = [f"pred_decision_utility_{horizon}d" for horizon in horizons]
    hit_columns = [f"pred_hit_prob_{horizon}d" for horizon in horizons]
    future_columns = [f"future_decision_utility_{horizon}d" for horizon in horizons]
    long_blend_columns = [f"pred_decision_utility_{horizon}d" for horizon in LONG_BLEND_REQUIRED_HORIZONS]
    _require_columns(frame, ["trade_utility_score", *utility_columns, *hit_columns, *future_columns, *long_blend_columns])

    out = frame.copy(deep=True)
    values = out[utility_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    horizon_values = np.asarray(horizons, dtype=float)
    softmax_score, softmax_horizon = _softmax_expected(values, horizon_values, temperature=0.15)
    top1_horizon = _top_horizon(values, horizons, rank_from_top=1)
    top2_horizon = _top_horizon(values, horizons, rank_from_top=2)

    out["baseline_trade_utility"] = pd.to_numeric(out["trade_utility_score"], errors="coerce")
    out["softmax_expected_utility_t015"] = softmax_score
    out["softmax_implied_horizon_t015"] = softmax_horizon
    out["top1_pred_utility_horizon"] = top1_horizon
    out["top2_pred_utility_horizon"] = top2_horizon
    for column in long_blend_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["long_blend_15_20_30"] = (
        0.20 * out["pred_decision_utility_15d"]
        + 0.35 * out["pred_decision_utility_20d"]
        + 0.45 * out["pred_decision_utility_30d"]
    )
    out["long_blend_10_15_20_30"] = (
        0.10 * out["pred_decision_utility_10d"]
        + 0.20 * out["pred_decision_utility_15d"]
        + 0.35 * out["pred_decision_utility_20d"]
        + 0.35 * out["pred_decision_utility_30d"]
    )
    out["top2_blend"] = _top2_blend(values)
    return out, horizons


def _rank_ic_by_date(frame: pd.DataFrame, score_column: str, target_column: str) -> float:
    if "date" not in frame.columns:
        return _rank_ic(frame[score_column], frame[target_column])
    values: list[float] = []
    work = frame[["date", score_column, target_column]].copy()
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
    for _, group in work.dropna().groupby("date", sort=True):
        if len(group) < 3:
            continue
        corr = group[score_column].rank(method="average").corr(group[target_column].rank(method="average"))
        if pd.notna(corr) and math.isfinite(float(corr)):
            values.append(float(corr))
    return _finite_float(np.mean(values) if values else 0.0)


def _rank_ic(x: pd.Series, y: pd.Series) -> float:
    working = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(working) < 3:
        return 0.0
    corr = working["x"].rank(method="average").corr(working["y"].rank(method="average"))
    return _finite_float(corr)


def _daily_spreads(frame: pd.DataFrame, score_column: str, target_column: str) -> pd.DataFrame:
    work = frame[["date", score_column, target_column]].copy() if "date" in frame.columns else frame[[score_column, target_column]].copy()
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
    rows: list[dict[str, Any]] = []
    if "date" not in work.columns:
        work = work.dropna()
        if len(work) >= 2:
            k = max(int(len(work) * 0.20), 1)
            top = float(work.nlargest(k, score_column)[target_column].mean())
            bottom = float(work.nsmallest(k, score_column)[target_column].mean())
            rows.append({"date": pd.Timestamp("1970-01-01"), "spread": top - bottom, "top": top, "bottom": bottom})
        return pd.DataFrame(rows)
    for date, group in work.dropna().groupby("date", sort=True):
        if len(group) < 2:
            continue
        k = max(int(len(group) * 0.20), 1)
        top = float(group.nlargest(k, score_column)[target_column].mean())
        bottom = float(group.nsmallest(k, score_column)[target_column].mean())
        rows.append({"date": pd.Timestamp(date), "spread": top - bottom, "top": top, "bottom": bottom})
    return pd.DataFrame(rows)


def _top_bottom_spread_by_date(frame: pd.DataFrame, score_column: str, target_column: str) -> tuple[float, float, float]:
    daily = _daily_spreads(frame, score_column, target_column)
    if daily.empty:
        return 0.0, 0.0, 0.0
    return (
        _finite_float(daily["spread"].mean()),
        _finite_float(daily["top"].mean()),
        _finite_float(daily["bottom"].mean()),
    )


def _hit_lift(frame: pd.DataFrame, score_column: str, horizons: tuple[int, ...]) -> float:
    hit_cols = [f"future_hit_label_{horizon}d" for horizon in horizons if f"future_hit_label_{horizon}d" in frame.columns]
    if not hit_cols:
        return 0.0
    columns = ["date", score_column, *hit_cols] if "date" in frame.columns else [score_column, *hit_cols]
    working = frame[columns].copy()
    working[score_column] = pd.to_numeric(working[score_column], errors="coerce")
    for column in hit_cols:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["_hit_any"] = working[hit_cols].max(axis=1)
    if "date" not in working.columns:
        working = working.dropna(subset=[score_column, "_hit_any"])
        if len(working) < 2:
            return 0.0
        k = max(int(len(working) * 0.20), 1)
        return _finite_float(working.nlargest(k, score_column)["_hit_any"].mean() - working["_hit_any"].mean())
    lifts: list[float] = []
    for _, group in working.dropna(subset=[score_column, "_hit_any"]).groupby("date", sort=True):
        if len(group) < 2:
            continue
        k = max(int(len(group) * 0.20), 1)
        lifts.append(float(group.nlargest(k, score_column)["_hit_any"].mean() - group["_hit_any"].mean()))
    return _finite_float(np.mean(lifts) if lifts else 0.0)


def _monthly_spread(frame: pd.DataFrame, score_column: str, target_column: str) -> dict[str, float]:
    if "date" not in frame.columns:
        return {}
    working = _daily_spreads(frame, score_column, target_column)
    if working.empty:
        return {}
    working["month"] = pd.to_datetime(working["date"], errors="coerce").dt.strftime("%Y-%m")
    out: dict[str, float] = {}
    for month, group in working.dropna(subset=["month"]).groupby("month", sort=True):
        out[str(month)] = _finite_float(group["spread"].mean())
    return out


def summarize_profile(frame: pd.DataFrame, score_column: str, horizons: tuple[int, ...]) -> dict[str, Any]:
    if len(frame) == 0:
        return {
            "status": "blocked",
            "row_count": 0,
            "rank_ic": 0.0,
            "top_bottom_spread": 0.0,
            "hit_lift_top20_mean": 0.0,
            "monthly_spread_positive_rate": 0.0,
            "negative_months": [],
            "long_horizon_share": 0.0,
        }
    target_column = "future_decision_score" if "future_decision_score" in frame.columns else f"future_decision_utility_{max(horizons)}d"
    spread, top_mean, bottom_mean = _top_bottom_spread_by_date(frame, score_column, target_column)
    monthly = _monthly_spread(frame, score_column, target_column)
    negative_months = [month for month, value in monthly.items() if value < 0.0]
    monthly_positive_rate = (
        sum(1 for value in monthly.values() if value > 0.0) / len(monthly) if monthly else 0.0
    )
    pred_best = pd.to_numeric(frame.get("pred_best_horizon", pd.Series(dtype=float)), errors="coerce")
    long_share = float(pred_best.isin([15, 20, 30]).mean()) if len(pred_best) else 0.0
    thirty_d_concentration = float((pred_best == 30).mean()) if len(pred_best) else 0.0
    if score_column == "softmax_expected_utility_t015" and "softmax_implied_horizon_t015" in frame.columns:
        implied = pd.to_numeric(frame["softmax_implied_horizon_t015"], errors="coerce")
        long_share = float((implied >= 15.0).mean()) if len(implied) else 0.0
        thirty_d_concentration = float((implied >= 25.0).mean()) if len(implied) else 0.0
    elif score_column == "long_blend_15_20_30":
        long_share = 1.0
        thirty_d_concentration = 0.45
    elif score_column == "long_blend_10_15_20_30":
        long_share = 0.90
        thirty_d_concentration = 0.35
    elif score_column == "top2_blend" and "top1_pred_utility_horizon" in frame.columns:
        top1 = pd.to_numeric(frame["top1_pred_utility_horizon"], errors="coerce")
        top2 = pd.to_numeric(frame["top2_pred_utility_horizon"], errors="coerce")
        top_pair_long = (top1.isin([15, 20, 30]) | top2.isin([15, 20, 30]))
        long_share = float(top_pair_long.mean()) if len(top_pair_long) else 0.0
        thirty_d_concentration = float(((top1 == 30) | (top2 == 30)).mean()) if len(top1) else 0.0
    return {
        "status": "ok",
        "row_count": int(len(frame)),
        "target_column": target_column,
        "rank_ic": _rank_ic_by_date(frame, score_column, target_column),
        "top_bottom_spread": spread,
        "top20_future_utility_mean": top_mean,
        "bottom20_future_utility_mean": bottom_mean,
        "hit_lift_top20_mean": _hit_lift(frame, score_column, horizons),
        "monthly_spread": monthly,
        "monthly_spread_positive_rate": _finite_float(monthly_positive_rate),
        "negative_months": negative_months,
        "long_horizon_share": _finite_float(long_share),
        "thirty_d_concentration": _finite_float(thirty_d_concentration),
        "pred_best_horizon_distribution": {
            str(int(key)): int(value)
            for key, value in pred_best.dropna().astype(int).value_counts().sort_index().to_dict().items()
        },
    }


def evaluate_horizon_calibration_gate(report: dict[str, Any]) -> dict[str, Any]:
    profiles = report.get("profiles", {})
    baseline = profiles.get("baseline_trade_utility", {})
    candidates = [name for name in PROFILE_COLUMNS if name != "baseline_trade_utility"]

    def passes_metric_floor(name: str) -> bool:
        payload = profiles.get(name, {})
        for role in ("validation", "test"):
            role_payload = payload.get(role, {})
            base_payload = baseline.get(role, {})
            if _finite_float(role_payload.get("rank_ic")) < 0.90 * _finite_float(base_payload.get("rank_ic")):
                return False
            if _finite_float(role_payload.get("top_bottom_spread")) < 0.90 * _finite_float(base_payload.get("top_bottom_spread")):
                return False
        validation = payload.get("validation", {})
        test = payload.get("test", {})
        if _finite_float(validation.get("monthly_spread_positive_rate")) < 0.75:
            return False
        if _finite_float(test.get("monthly_spread_positive_rate")) < 0.90:
            return False
        if len(validation.get("negative_months", []) or []) > 2:
            return False
        if len(test.get("negative_months", []) or []) > 1:
            return False
        return True

    baseline_long = max(
        _finite_float(baseline.get("validation", {}).get("long_horizon_share")),
        _finite_float(baseline.get("test", {}).get("long_horizon_share")),
    )
    for name in candidates:
        if not passes_metric_floor(name):
            continue
        payload = profiles.get(name, {})
        candidate_long = max(
            _finite_float(payload.get("validation", {}).get("long_horizon_share")),
            _finite_float(payload.get("test", {}).get("long_horizon_share")),
        )
        if candidate_long <= baseline_long - 0.10:
            return {
                "status": "passed_calibrated",
                "selected_profile": name,
                "reason": "profile preserved baseline metrics while reducing long-horizon concentration",
            }

    for name in ("long_blend_15_20_30", "long_blend_10_15_20_30"):
        if passes_metric_floor(name):
            return {
                "status": "passed_long_family",
                "selected_profile": name,
                "reason": "long-horizon blend preserved baseline metrics; treat signal as longer-horizon utility family",
            }

    return {
        "status": "failed_regression",
        "selected_profile": "",
        "reason": "no calibration profile preserved baseline validation/test quality",
    }


def build_horizon_calibration_report(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    *,
    study_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(validation_predictions) == 0 or len(test_predictions) == 0:
        return {
            "status": "blocked",
            "reason": "validation/test predictions must both be non-empty",
            **(study_metadata or {}),
        }
    try:
        validation_scored, validation_horizons = score_horizon_profiles(validation_predictions)
        test_scored, test_horizons = score_horizon_profiles(test_predictions)
    except ValueError as exc:
        return {
            "status": "blocked",
            "reason": str(exc),
            **(study_metadata or {}),
        }
    if validation_horizons != test_horizons:
        raise ValueError(f"validation/test horizon mismatch: {validation_horizons} != {test_horizons}")

    profiles: dict[str, Any] = {}
    for profile in PROFILE_COLUMNS:
        profiles[profile] = {
            "validation": summarize_profile(validation_scored, profile, validation_horizons),
            "test": summarize_profile(test_scored, profile, validation_horizons),
        }

    metadata = dict(study_metadata or {})
    report: dict[str, Any] = {
        "status": "completed",
        "schema_version": 1,
        "purpose": "post-hoc multi-horizon utility score calibration",
        "horizons": [int(item) for item in validation_horizons],
        "profiles": profiles,
        **metadata,
    }
    report["gate"] = evaluate_horizon_calibration_gate(report)
    return _jsonable(report)


def comparison_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile, profile_payload in (report.get("profiles", {}) or {}).items():
        for role in ("validation", "test"):
            payload = profile_payload.get(role, {})
            rows.append(
                {
                    "profile": profile,
                    "role": role,
                    "status": payload.get("status", ""),
                    "row_count": payload.get("row_count", 0),
                    "rank_ic": payload.get("rank_ic", 0.0),
                    "top_bottom_spread": payload.get("top_bottom_spread", 0.0),
                    "hit_lift_top20_mean": payload.get("hit_lift_top20_mean", 0.0),
                    "monthly_spread_positive_rate": payload.get("monthly_spread_positive_rate", 0.0),
                    "negative_month_count": len(payload.get("negative_months", []) or []),
                    "long_horizon_share": payload.get("long_horizon_share", 0.0),
                    "thirty_d_concentration": payload.get("thirty_d_concentration", 0.0),
                }
            )
    return rows


def write_horizon_calibration_artifacts(report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    report_json = root / "horizon_calibration_report.json"
    comparison_csv = root / "horizon_score_comparison.csv"
    report_json.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(comparison_rows(report)).to_csv(comparison_csv, index=False, encoding="utf-8")
    return {"report_json": report_json, "comparison_csv": comparison_csv}


def _load_metadata(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    training = payload.get("training_summary", {}) or {}
    dataset = payload.get("dataset_manifest", {}) or {}
    return {
        "study_tag": payload.get("study_tag", payload.get("run_tag", "")),
        "evidence_verdict": payload.get("evidence_verdict", ""),
        "dataset_id": payload.get("lake_dataset_id", dataset.get("source_market_dataset_id", "")),
        "feature_profile": training.get("feature_profile", dataset.get("feature_profile", "")),
        "model_family": training.get("selected_model_family", (payload.get("model_families", [""]) or [""])[0]),
        "seed": training.get("selected_seed", payload.get("seed", "")),
        "role_years": dataset.get("role_years", {}),
        "training_config": training.get("training_config", {}),
    }


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
    wanted = {
        "date",
        "stock",
        "role",
        "trade_utility_score",
        "future_decision_score",
        "pred_best_horizon",
        "future_best_horizon",
    }
    for horizon in horizons:
        wanted.update(
            {
                f"pred_decision_utility_{horizon}d",
                f"future_decision_utility_{horizon}d",
                f"pred_hit_prob_{horizon}d",
                f"future_hit_label_{horizon}d",
            }
        )
    return pd.read_csv(csv_path, usecols=[column for column in columns if column in wanted])


def run_calibration_from_files(
    *,
    validation_predictions_csv: str | Path,
    test_predictions_csv: str | Path,
    output_dir: str | Path,
    study_summary_json: str | Path | None = None,
) -> dict[str, Any]:
    validation = _read_prediction_subset(validation_predictions_csv)
    test = _read_prediction_subset(test_predictions_csv)
    metadata = _load_metadata(Path(study_summary_json)) if study_summary_json else {}
    report = build_horizon_calibration_report(validation, test, study_metadata=metadata)
    paths = write_horizon_calibration_artifacts(report, output_dir)
    report["artifact_paths"] = {key: str(path.resolve()) for key, path in paths.items()}
    Path(paths["report_json"]).write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Post-hoc calibration for multi-horizon utility prediction scores.")
    parser.add_argument("--validation-predictions-csv", required=True)
    parser.add_argument("--test-predictions-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--study-summary-json", default="")
    parser.add_argument("--json", action="store_true", help="Print the generated report as JSON.")
    args = parser.parse_args(argv)
    report = run_calibration_from_files(
        validation_predictions_csv=args.validation_predictions_csv,
        test_predictions_csv=args.test_predictions_csv,
        output_dir=args.output_dir,
        study_summary_json=args.study_summary_json or None,
    )
    if args.json:
        print(json.dumps(_jsonable(report), ensure_ascii=False, indent=2))
    else:
        gate = report.get("gate", {})
        print(f"status={report.get('status')} gate={gate.get('status')} selected={gate.get('selected_profile')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
