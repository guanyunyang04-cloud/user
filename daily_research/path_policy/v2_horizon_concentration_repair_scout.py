from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import v2_local_state_loss_calibration_scout as loss_scout
from daily_research.path_policy import v2_research_reset_baseline as v2
from daily_research.path_policy.mainboard_rebuild_baseline import OLD_STAGE28_THIRTY_D_CONCENTRATION


PROJECT_ROOT = v2.PROJECT_ROOT
STUDIES_ROOT = v2.STUDIES_ROOT
RUN_TAG = "mh_v2_horizon_concentration_repair_anchor_20260603_01"
RESEARCH_PROGRAM = v2.RESEARCH_PROGRAM
STUDY_FAMILY = "v2_horizon_concentration_repair_scout"
SOURCE_ANCHOR_RUN_TAG = loss_scout.RUN_TAG
SOURCE_LOSS_PROFILE = "horizon_entropy_regularized_v1"
SOURCE_FEATURE_PROFILE = loss_scout.FEATURE_PROFILE
DATASET_ID = v2.DATASET_ID
V2_STRICT_POOL_VIEW_ID = v2.V2_STRICT_POOL_VIEW_ID
MODEL_FAMILY = v2.MODEL_FAMILY
OUTPUT_PROFILE = v2.OUTPUT_PROFILE
HORIZON_GRID = v2.HORIZON_GRID
DEFAULT_SEEDS = (7, 11, 19)
ACTIVE_MANIFEST = v2.ACTIVE_MANIFEST
THIRTY_D_CONCENTRATION_LIMIT = OLD_STAGE28_THIRTY_D_CONCENTRATION
QUALITY_FLOOR_RATIO = 0.90

DEFAULT_VARIANTS = (
    "baseline",
    "penalty_30d_0p001",
    "penalty_30d_0p002",
    "penalty_30d_0p005",
    "penalty_long_linear_0p001",
    "penalty_long_linear_0p002",
    "validation_center_mean",
    "validation_center_median",
    "validation_zscore_rescaled",
    "validation_zscore_blend_0p50",
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return str(value)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _active_artifact_has_diff() -> bool:
    result = subprocess.run(["git", "diff", "--", str(ACTIVE_MANIFEST)], check=False, capture_output=True, text=True)
    return bool(str(result.stdout or "").strip() or str(result.stderr or "").strip())


def _parse_csv(raw: str | tuple[Any, ...] | list[Any] | None, default: tuple[Any, ...]) -> tuple[str, ...]:
    if raw is None:
        values = [str(item) for item in default]
    elif isinstance(raw, (tuple, list)):
        values = [str(item) for item in raw]
    else:
        values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for item in values or [str(value) for value in default]:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def _parse_seeds(raw: str | tuple[int, ...] | list[int] | None, default: tuple[int, ...] = DEFAULT_SEEDS) -> tuple[int, ...]:
    values = _parse_csv(raw, tuple(str(seed) for seed in default))
    return tuple(int(float(value)) for value in values)


def _parse_variants(raw: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    variants = _parse_csv(raw, DEFAULT_VARIANTS)
    supported = set(DEFAULT_VARIANTS)
    unknown = [variant for variant in variants if variant not in supported]
    if unknown:
        raise ValueError(f"unknown_horizon_repair_variants: {unknown}")
    return variants


def _anchor_root(output_root: str | Path | None = None) -> Path:
    return Path(output_root) if output_root is not None else STUDIES_ROOT / RUN_TAG


def source_study_tag(seed: int) -> str:
    return loss_scout._study_tag(loss_profile=SOURCE_LOSS_PROFILE, seed=int(seed))


def source_study_dir(seed: int) -> Path:
    return STUDIES_ROOT / source_study_tag(seed)


def infer_horizons(frame: pd.DataFrame) -> tuple[int, ...]:
    horizons = sorted(
        int(match.group(1))
        for column in frame.columns
        for match in [re.match(r"^pred_decision_utility_(\d+)d$", str(column))]
        if match is not None
    )
    if not horizons:
        raise ValueError("missing_pred_decision_utility_columns")
    return tuple(dict.fromkeys(horizons))


def _required_prediction_columns(frame: pd.DataFrame) -> list[str]:
    horizons = infer_horizons(frame)
    required = ["date", "pred_decision_score", "future_decision_score", "pred_best_horizon", "future_best_horizon"]
    for horizon in horizons:
        required.extend(
            [
                f"pred_decision_utility_{horizon}d",
                f"future_hit_label_{horizon}d",
            ]
        )
    return required


def _require_prediction_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in _required_prediction_columns(frame) if column not in frame.columns]
    if missing:
        raise ValueError(f"missing_required_prediction_columns: {missing}")


def fit_validation_stats(validation_frame: pd.DataFrame) -> dict[str, Any]:
    _require_prediction_columns(validation_frame)
    horizons = infer_horizons(validation_frame)
    utility_cols = [f"pred_decision_utility_{horizon}d" for horizon in horizons]
    utilities = validation_frame[utility_cols].apply(pd.to_numeric, errors="coerce")
    values = utilities.to_numpy(dtype=float)
    means = np.nanmean(values, axis=0)
    medians = np.nanmedian(values, axis=0)
    stds = np.nanstd(values, axis=0)
    global_mean = float(np.nanmean(values)) if np.isfinite(values).any() else 0.0
    global_std = float(np.nanstd(values)) if np.isfinite(values).any() else 1.0
    stds = np.where(np.isfinite(stds) & (stds > 1.0e-8), stds, 1.0)
    return {
        "horizons": [int(item) for item in horizons],
        "mean_by_horizon": {str(int(horizon)): _finite_float(means[pos]) for pos, horizon in enumerate(horizons)},
        "median_by_horizon": {str(int(horizon)): _finite_float(medians[pos]) for pos, horizon in enumerate(horizons)},
        "std_by_horizon": {str(int(horizon)): _finite_float(stds[pos], 1.0) for pos, horizon in enumerate(horizons)},
        "global_mean": _finite_float(global_mean),
        "global_std": _finite_float(global_std, 1.0),
    }


def _stats_array(stats: dict[str, Any], key: str, horizons: tuple[int, ...], default: float) -> np.ndarray:
    values = dict(stats.get(key, {}) or {})
    return np.asarray([_finite_float(values.get(str(int(horizon))), default) for horizon in horizons], dtype=float)


def _adjusted_utilities(values: np.ndarray, horizons: tuple[int, ...], *, variant: str, stats: dict[str, Any]) -> np.ndarray:
    adjusted = np.array(values, dtype=float, copy=True)
    horizon_arr = np.asarray(horizons, dtype=float).reshape(1, -1)
    if variant == "baseline":
        return adjusted
    if variant.startswith("penalty_30d_"):
        penalty = float(variant.rsplit("_", 1)[-1].replace("p", "."))
        adjusted[:, np.asarray(horizons) == 30] -= penalty
        return adjusted
    if variant.startswith("penalty_long_linear_"):
        penalty = float(variant.rsplit("_", 1)[-1].replace("p", "."))
        scale = np.clip((horizon_arr - 10.0) / 20.0, 0.0, 1.0)
        adjusted -= penalty * scale
        return adjusted
    means = _stats_array(stats, "mean_by_horizon", horizons, 0.0).reshape(1, -1)
    medians = _stats_array(stats, "median_by_horizon", horizons, 0.0).reshape(1, -1)
    stds = _stats_array(stats, "std_by_horizon", horizons, 1.0).reshape(1, -1)
    global_mean = _finite_float(stats.get("global_mean"), 0.0)
    global_std = max(_finite_float(stats.get("global_std"), 1.0), 1.0e-8)
    if variant == "validation_center_mean":
        return adjusted - (means - global_mean)
    if variant == "validation_center_median":
        return adjusted - (medians - global_mean)
    zscore_rescaled = ((adjusted - means) / np.clip(stds, 1.0e-8, None)) * global_std + global_mean
    if variant == "validation_zscore_rescaled":
        return zscore_rescaled
    if variant == "validation_zscore_blend_0p50":
        return 0.50 * adjusted + 0.50 * zscore_rescaled
    raise ValueError(f"unknown_horizon_repair_variant: {variant}")


def apply_horizon_repair_variant(frame: pd.DataFrame, *, variant: str, validation_stats: dict[str, Any]) -> pd.DataFrame:
    _require_prediction_columns(frame)
    if variant == "baseline":
        out = frame.copy()
        out["horizon_repair_variant"] = str(variant)
        return out
    horizons = infer_horizons(frame)
    utility_cols = [f"pred_decision_utility_{horizon}d" for horizon in horizons]
    values = frame[utility_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    adjusted = _adjusted_utilities(values, horizons, variant=variant, stats=validation_stats)
    finite = np.where(np.isfinite(adjusted), adjusted, -np.inf)
    best_idx = np.argmax(finite, axis=1)
    best_score = finite[np.arange(finite.shape[0]), best_idx]
    best_score = np.where(np.isfinite(best_score), best_score, np.nan)
    horizon_values = np.asarray(horizons, dtype=int)
    out = frame.copy()
    out["pred_decision_score"] = best_score
    out["trade_utility_score"] = best_score
    out["pred_best_horizon"] = horizon_values[best_idx]
    out["horizon_repair_variant"] = str(variant)
    return out


def _rank_ic_by_date(frame: pd.DataFrame, score_col: str = "pred_decision_score", target_col: str = "future_decision_score") -> float:
    values: list[float] = []
    work = frame[["date", score_col, target_col]].copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    for _, group in work.dropna().groupby("date", sort=True):
        if len(group) < 2:
            continue
        corr = group[score_col].rank(method="average").corr(group[target_col].rank(method="average"))
        if pd.notna(corr) and math.isfinite(float(corr)):
            values.append(float(corr))
    return _finite_float(np.mean(values) if values else 0.0)


def _daily_spreads(frame: pd.DataFrame, score_col: str = "pred_decision_score", target_col: str = "future_decision_score") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = frame[["date", score_col, target_col]].copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    for date, group in work.dropna().groupby("date", sort=True):
        if len(group) < 2:
            continue
        k = max(int(len(group) * 0.20), 1)
        top = float(group.nlargest(k, score_col)[target_col].mean())
        bottom = float(group.nsmallest(k, score_col)[target_col].mean())
        rows.append({"date": pd.Timestamp(date), "spread": top - bottom, "top": top, "bottom": bottom})
    return pd.DataFrame(rows)


def _top_bottom_spread_by_date(frame: pd.DataFrame) -> float:
    daily = _daily_spreads(frame)
    return _finite_float(daily["spread"].mean() if not daily.empty else 0.0)


def _monthly_positive_rate(frame: pd.DataFrame) -> tuple[float, int, float]:
    daily = _daily_spreads(frame)
    if daily.empty:
        return 0.0, 0, 0.0
    daily["month"] = pd.to_datetime(daily["date"], errors="coerce").dt.strftime("%Y-%m")
    monthly = daily.dropna(subset=["month"]).groupby("month", sort=True)["spread"].mean()
    if monthly.empty:
        return 0.0, 0, 0.0
    positive_rate = float((monthly > 0.0).mean())
    negative_count = int((monthly < 0.0).sum())
    worst = float(monthly.min())
    return _finite_float(positive_rate), negative_count, _finite_float(worst)


def _hit_lift(frame: pd.DataFrame, horizons: tuple[int, ...]) -> float:
    hit_cols = [f"future_hit_label_{horizon}d" for horizon in horizons if f"future_hit_label_{horizon}d" in frame.columns]
    if not hit_cols:
        return 0.0
    work = frame[["date", "pred_decision_score", *hit_cols]].copy()
    work["pred_decision_score"] = pd.to_numeric(work["pred_decision_score"], errors="coerce")
    for column in hit_cols:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["_hit_any"] = work[hit_cols].max(axis=1)
    values: list[float] = []
    for _, group in work.dropna(subset=["pred_decision_score", "_hit_any"]).groupby("date", sort=True):
        if len(group) < 2:
            continue
        k = max(int(len(group) * 0.20), 1)
        values.append(float(group.nlargest(k, "pred_decision_score")["_hit_any"].mean() - group["_hit_any"].mean()))
    return _finite_float(np.mean(values) if values else 0.0)


def summarize_variant_frame(frame: pd.DataFrame, *, seed: int, role: str, variant: str) -> dict[str, Any]:
    _require_prediction_columns(frame)
    horizons = infer_horizons(frame)
    monthly_positive_rate, negative_count, worst_month = _monthly_positive_rate(frame)
    pred_horizon = pd.to_numeric(frame["pred_best_horizon"], errors="coerce").dropna()
    future_horizon = pd.to_numeric(frame["future_best_horizon"], errors="coerce").dropna()
    return {
        "seed": int(seed),
        "role": str(role),
        "variant": str(variant),
        "row_count": int(len(frame)),
        "rank_ic": _rank_ic_by_date(frame),
        "top_bottom_spread": _top_bottom_spread_by_date(frame),
        "hit_lift": _hit_lift(frame, horizons),
        "monthly_positive_rate": monthly_positive_rate,
        "negative_month_count": int(negative_count),
        "worst_month_spread": worst_month,
        "long_horizon_share": _finite_float((pred_horizon >= 15).mean() if not pred_horizon.empty else 0.0),
        "thirty_d_concentration": _finite_float((pred_horizon == 30).mean() if not pred_horizon.empty else 0.0),
        "future_long_horizon_share": _finite_float((future_horizon >= 15).mean() if not future_horizon.empty else 0.0),
        "pred_best_horizon_distribution": {
            str(int(key)): int(value)
            for key, value in pred_horizon.astype(int).value_counts().sort_index().to_dict().items()
        },
    }


def aggregate_variant_metrics(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(metric_rows)
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for (variant, role), group in frame.groupby(["variant", "role"], sort=True):
        seeds = sorted({int(value) for value in group["seed"].dropna().tolist()})
        rows.append(
            {
                "variant": str(variant),
                "role": str(role),
                "seed_count": int(len(seeds)),
                "rank_ic_mean": _finite_float(group["rank_ic"].mean()),
                "rank_ic_min": _finite_float(group["rank_ic"].min()),
                "spread_mean": _finite_float(group["top_bottom_spread"].mean()),
                "spread_min": _finite_float(group["top_bottom_spread"].min()),
                "hit_lift_mean": _finite_float(group["hit_lift"].mean()),
                "hit_lift_min": _finite_float(group["hit_lift"].min()),
                "monthly_positive_rate_mean": _finite_float(group["monthly_positive_rate"].mean()),
                "monthly_positive_rate_min": _finite_float(group["monthly_positive_rate"].min()),
                "negative_month_count_max": int(group["negative_month_count"].max()),
                "worst_month_spread_min": _finite_float(group["worst_month_spread"].min()),
                "long_horizon_share_mean": _finite_float(group["long_horizon_share"].mean()),
                "thirty_d_concentration_mean": _finite_float(group["thirty_d_concentration"].mean()),
                "future_long_horizon_share_mean": _finite_float(group["future_long_horizon_share"].mean()),
            }
        )
    validation_lookup = {(row["variant"], row["role"]): row for row in rows}
    baseline_by_role = {row["role"]: row for row in rows if row["variant"] == "baseline"}
    for row in rows:
        baseline = baseline_by_role.get(str(row["role"]), {})
        row["rank_ic_delta_vs_baseline"] = _finite_float(row["rank_ic_mean"]) - _finite_float(baseline.get("rank_ic_mean"))
        row["spread_delta_vs_baseline"] = _finite_float(row["spread_mean"]) - _finite_float(baseline.get("spread_mean"))
        row["hit_lift_delta_vs_baseline"] = _finite_float(row["hit_lift_mean"]) - _finite_float(baseline.get("hit_lift_mean"))
        row["thirty_d_concentration_delta_vs_baseline"] = _finite_float(row["thirty_d_concentration_mean"]) - _finite_float(
            baseline.get("thirty_d_concentration_mean")
        )
        validation = validation_lookup.get((row["variant"], "validation"), {})
        if str(row["role"]) == "test" and validation:
            row["validation_test_rank_ic_gap"] = _finite_float(row["rank_ic_mean"]) - _finite_float(validation.get("rank_ic_mean"))
            row["validation_test_spread_gap"] = _finite_float(row["spread_mean"]) - _finite_float(validation.get("spread_mean"))
            row["validation_test_hit_lift_gap"] = _finite_float(row["hit_lift_mean"]) - _finite_float(validation.get("hit_lift_mean"))
    return rows


def _passes_quality(row: dict[str, Any], *, baseline: dict[str, Any], require_concentration: bool) -> bool:
    baseline_rank = max(_finite_float(baseline.get("rank_ic_mean")), 1.0e-12)
    baseline_spread = max(_finite_float(baseline.get("spread_mean")), 1.0e-12)
    baseline_hit = max(_finite_float(baseline.get("hit_lift_mean")), 1.0e-12)
    checks = [
        int(row.get("seed_count", 0) or 0) >= len(DEFAULT_SEEDS),
        _finite_float(row.get("rank_ic_min")) > 0.0,
        _finite_float(row.get("spread_min")) > 0.0,
        _finite_float(row.get("hit_lift_min")) > 0.0,
        _finite_float(row.get("rank_ic_mean")) >= QUALITY_FLOOR_RATIO * baseline_rank,
        _finite_float(row.get("spread_mean")) >= QUALITY_FLOOR_RATIO * baseline_spread,
        _finite_float(row.get("hit_lift_mean")) >= QUALITY_FLOOR_RATIO * baseline_hit,
        _finite_float(row.get("monthly_positive_rate_mean")) >= 0.75,
        int(row.get("negative_month_count_max", 99) or 99) <= 2,
    ]
    if require_concentration:
        checks.append(_finite_float(row.get("thirty_d_concentration_mean"), 1.0) <= THIRTY_D_CONCENTRATION_LIMIT)
    return all(checks)


def evaluate_repair_gate(aggregate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["variant"], row["role"]): row for row in aggregate_rows}
    validation_baseline = by_key.get(("baseline", "validation"), {})
    test_baseline = by_key.get(("baseline", "test"), {})
    validation_pass: list[dict[str, Any]] = []
    test_confirmed: list[dict[str, Any]] = []
    for row in aggregate_rows:
        if row["variant"] == "baseline":
            continue
        if row["role"] != "validation":
            continue
        if _passes_quality(row, baseline=validation_baseline, require_concentration=True):
            validation_pass.append(row)
            test_row = by_key.get((row["variant"], "test"), {})
            if test_row and _passes_quality(test_row, baseline=test_baseline, require_concentration=True):
                test_confirmed.append(test_row)
    validation_pass = sorted(
        validation_pass,
        key=lambda item: (
            _finite_float(item.get("thirty_d_concentration_mean")),
            -_finite_float(item.get("rank_ic_mean")),
        ),
    )
    test_confirmed = sorted(
        test_confirmed,
        key=lambda item: (
            _finite_float(item.get("thirty_d_concentration_mean")),
            -_finite_float(item.get("rank_ic_mean")),
        ),
    )
    if test_confirmed:
        status = "repair_pass"
        selected = str(test_confirmed[0]["variant"])
    elif validation_pass:
        status = "validation_only_needs_confirmation"
        selected = str(validation_pass[0]["variant"])
    else:
        status = "repair_not_found"
        selected = ""
    return {
        "status": status,
        "selected_variant": selected,
        "validation_pass_variants": validation_pass,
        "test_confirmed_variants": test_confirmed,
        "thresholds": {
            "thirty_d_concentration_limit": THIRTY_D_CONCENTRATION_LIMIT,
            "quality_floor_ratio_vs_baseline": QUALITY_FLOOR_RATIO,
            "monthly_positive_rate_min": 0.75,
            "negative_month_count_max": 2,
        },
    }


def build_repair_report_from_frames(
    seed_frames: dict[int, dict[str, pd.DataFrame]],
    *,
    variants: tuple[str, ...] | list[str] | str | None = DEFAULT_VARIANTS,
) -> dict[str, Any]:
    resolved_variants = _parse_variants(variants)
    metric_rows: list[dict[str, Any]] = []
    for seed, frames in sorted(seed_frames.items()):
        validation = frames.get("validation")
        test = frames.get("test")
        if validation is None or test is None:
            raise ValueError(f"missing_validation_or_test_frame_for_seed: {seed}")
        stats = fit_validation_stats(validation)
        for variant in resolved_variants:
            for role, frame in (("validation", validation), ("test", test)):
                repaired = apply_horizon_repair_variant(frame, variant=variant, validation_stats=stats)
                metric_rows.append(summarize_variant_frame(repaired, seed=int(seed), role=role, variant=variant))
    aggregate_rows = aggregate_variant_metrics(metric_rows)
    return {
        "schema_version": 1,
        "status": "completed",
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "source_anchor_run_tag": SOURCE_ANCHOR_RUN_TAG,
        "source_loss_profile": SOURCE_LOSS_PROFILE,
        "dataset_id": DATASET_ID,
        "pool_view_id": V2_STRICT_POOL_VIEW_ID,
        "feature_profile": SOURCE_FEATURE_PROFILE,
        "model_family": MODEL_FAMILY,
        "output_profile": OUTPUT_PROFILE,
        "horizon_grid": HORIZON_GRID,
        "seeds": [int(seed) for seed in sorted(seed_frames)],
        "variants": list(resolved_variants),
        "metric_rows": metric_rows,
        "aggregate_rows": aggregate_rows,
        "gate": evaluate_repair_gate(aggregate_rows),
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "posthoc_output_calibration_only": True,
            "active_execution_strategy_expected_diff": "none",
        },
        "updated_at": _now(),
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
    wanted = {"date", "stock", "role", "pred_decision_score", "future_decision_score", "pred_best_horizon", "future_best_horizon"}
    for horizon in horizons:
        wanted.update({f"pred_decision_utility_{horizon}d", f"future_hit_label_{horizon}d"})
    return pd.read_csv(csv_path, usecols=[column for column in columns if column in wanted])


def load_source_prediction_frames(seeds: tuple[int, ...] | list[int] | str | None = DEFAULT_SEEDS) -> dict[int, dict[str, pd.DataFrame]]:
    resolved_seeds = _parse_seeds(seeds)
    frames: dict[int, dict[str, pd.DataFrame]] = {}
    for seed in resolved_seeds:
        root = source_study_dir(int(seed))
        validation_path = root / "forecast_predictions_validation.csv"
        test_path = root / "forecast_predictions_test.csv"
        if not validation_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"missing_source_prediction_files_for_seed_{seed}: {root}")
        frames[int(seed)] = {
            "validation": _read_prediction_subset(validation_path),
            "test": _read_prediction_subset(test_path),
        }
    return frames


def research_verdict_markdown(report: dict[str, Any]) -> str:
    gate = dict(report.get("gate", {}) or {})
    rows = [
        row
        for row in report.get("aggregate_rows", []) or []
        if row.get("role") == "test"
    ]
    rows = sorted(rows, key=lambda item: (_finite_float(item.get("thirty_d_concentration_mean")), -_finite_float(item.get("rank_ic_mean"))))
    lines = [
        "# daily_research v2 Horizon Concentration Repair Scout",
        "",
        f"- status: `{report.get('status')}`",
        f"- gate: `{gate.get('status')}`",
        f"- selected_variant: `{gate.get('selected_variant', '')}`",
        f"- source_anchor: `{SOURCE_ANCHOR_RUN_TAG}`",
        f"- source_loss_profile: `{SOURCE_LOSS_PROFILE}`",
        "- boundary: research-only / shadow-only; posthoc output calibration only; no execution candidate or active artifact change.",
        "",
        "## Test Aggregates",
    ]
    for row in rows:
        lines.append(
            f"- `{row.get('variant')}`: rank_ic_mean=`{_finite_float(row.get('rank_ic_mean')):.6f}`, "
            f"spread_mean=`{_finite_float(row.get('spread_mean')):.6f}`, "
            f"hit_lift_mean=`{_finite_float(row.get('hit_lift_mean')):.6f}`, "
            f"monthly_positive_rate_mean=`{_finite_float(row.get('monthly_positive_rate_mean')):.6f}`, "
            f"negative_month_count_max=`{row.get('negative_month_count_max')}`, "
            f"thirty_d_concentration_mean=`{_finite_float(row.get('thirty_d_concentration_mean')):.6f}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- A `repair_pass` means a validation-aware posthoc calibration preserved core research metrics while reducing 30d concentration below the historical concentration limit.",
            "- This is not a model-training pass and does not authorize score-backtest bridge or execution unfreeze by itself.",
            "- A `repair_not_found` means the current loss/output branch likely needs a training-side horizon constraint rather than simple posthoc calibration.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_repair_artifacts(report: dict[str, Any], output_root: str | Path | None = None) -> dict[str, str]:
    root = _anchor_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "v2_horizon_concentration_repair_summary.json"
    metrics_path = root / "variant_metrics.csv"
    aggregate_path = root / "variant_aggregate.csv"
    verdict_path = root / "research_verdict.md"
    _write_json(summary_path, report)
    pd.DataFrame(report.get("metric_rows", []) or []).to_csv(metrics_path, index=False, encoding="utf-8")
    pd.DataFrame(report.get("aggregate_rows", []) or []).to_csv(aggregate_path, index=False, encoding="utf-8")
    verdict_path.write_text(research_verdict_markdown(report), encoding="utf-8")
    return {
        "summary_json": str(summary_path),
        "variant_metrics_csv": str(metrics_path),
        "variant_aggregate_csv": str(aggregate_path),
        "research_verdict_md": str(verdict_path),
    }


def run_repair_scout(
    *,
    output_root: str | Path | None = None,
    seeds: tuple[int, ...] | list[int] | str | None = DEFAULT_SEEDS,
    variants: tuple[str, ...] | list[str] | str | None = DEFAULT_VARIANTS,
) -> dict[str, Any]:
    if _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    frames = load_source_prediction_frames(seeds)
    report = build_repair_report_from_frames(frames, variants=variants)
    paths = write_repair_artifacts(report, output_root)
    report["artifact_paths"] = paths
    _write_json(Path(paths["summary_json"]), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run v2 local-state horizon concentration posthoc repair scout.")
    parser.add_argument("--output-root", default=str(STUDIES_ROOT / RUN_TAG))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--run-scout", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    seeds = _parse_seeds(args.seeds)
    variants = _parse_variants(args.variants)
    if args.run_scout:
        payload = run_repair_scout(output_root=args.output_root, seeds=seeds, variants=variants)
    else:
        payload = {
            "run_tag": RUN_TAG,
            "output_root": args.output_root,
            "seeds": list(seeds),
            "variants": list(variants),
            "source_anchor_run_tag": SOURCE_ANCHOR_RUN_TAG,
            "boundary": {
                "research_only": True,
                "shadow_only": True,
                "promotion_allowed": False,
                "active_execution_strategy_expected_diff": "none",
            },
        }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    else:
        gate = dict(payload.get("gate", {}) or {})
        print(f"status={payload.get('status', 'planned')} gate={gate.get('status', '')} selected={gate.get('selected_variant', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
