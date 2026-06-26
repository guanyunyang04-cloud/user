from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


STUDIES_ROOT = Path("daily_research/output/path_policy/studies")
DATA_LAKE_MANIFEST = Path("quant_data_platform/data/lake/manifest_latest.json")
DEFAULT_ANCHOR_TAG = "mh_rebuild_infra_v2_fullpool_anchor_20260531_01"
DEFAULT_NEW_SEED_TAGS = (
    "mh_rebuild_infra_v2_fullpool_target_norm_head_constraint_raw_seed7_20260531_01",
    "mh_rebuild_infra_v2_fullpool_target_norm_head_constraint_raw_seed11_20260531_01",
    "mh_rebuild_infra_v2_fullpool_target_norm_head_constraint_raw_seed19_20260531_01",
)
DEFAULT_STAGE28_REFERENCE = Path(
    "daily_research/brain/references/alpha_multi_horizon_stage28_full_pool_revalidation_20260528.md"
)
DEFAULT_STAGE36_REFERENCE = Path(
    "daily_research/brain/references/alpha_multi_horizon_stage36_input_cross_section_scout_20260530.md"
)
DEFAULT_OLD_STAGE28_ROOT = STUDIES_ROOT / "mh_stage28_full_pool_revalidation_20260527_01"
DEFAULT_OLD_STAGE36_ROOT = STUDIES_ROOT / "mh_stage36_input_cross_section_scout_20260529_01"
DEFAULT_SHORT_V5B_ROOT = Path("daily_research/output/short_expert_policy_v5b_execalign_production_default")
DEFAULT_ACTIVE_MANIFEST = Path("daily_research/output/active_execution_strategy.json")


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
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _finite_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _csv_columns(path: Path) -> list[str]:
    if not path.exists():
        return []
    return list(pd.read_csv(path, nrows=0).columns)


def _read_csv_subset(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    available = set(_csv_columns(path))
    usecols = [column for column in columns if column in available]
    if not usecols:
        return pd.DataFrame()
    return pd.read_csv(path, usecols=usecols)


def _summary_stats(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(len(numeric)),
        "mean": _finite_float(numeric.mean()),
        "std": _finite_float(numeric.std(ddof=0)),
        "min": _finite_float(numeric.min()),
        "p10": _finite_float(numeric.quantile(0.10)),
        "median": _finite_float(numeric.median()),
        "p90": _finite_float(numeric.quantile(0.90)),
        "max": _finite_float(numeric.max()),
    }


def _value_counts_payload(series: pd.Series, *, normalize: bool = False) -> dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {}
    counts = clean.astype(int).value_counts(dropna=False).sort_index()
    if normalize:
        total = float(counts.sum()) or 1.0
        return {str(int(key)): _finite_float(value / total) for key, value in counts.items()}
    return {str(int(key)): int(value) for key, value in counts.items()}


def _daily_spreads(frame: pd.DataFrame, score_column: str, target_column: str) -> pd.DataFrame:
    if frame.empty or not {"date", score_column, target_column}.issubset(frame.columns):
        return pd.DataFrame(columns=["date", "spread", "top", "bottom"])
    work = frame[["date", score_column, target_column]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
    rows: list[dict[str, Any]] = []
    for date, group in work.dropna().groupby("date", sort=True):
        if len(group) < 2:
            continue
        k = max(int(len(group) * 0.20), 1)
        top = _finite_float(group.nlargest(k, score_column)[target_column].mean())
        bottom = _finite_float(group.nsmallest(k, score_column)[target_column].mean())
        rows.append({"date": pd.Timestamp(date), "spread": top - bottom, "top": top, "bottom": bottom})
    return pd.DataFrame(rows)


def _monthly_spread_payload(frame: pd.DataFrame) -> dict[str, Any]:
    daily = _daily_spreads(frame, "pred_decision_score", "future_decision_score")
    if daily.empty:
        return {
            "monthly_spread": {},
            "monthly_positive_rate": 0.0,
            "negative_months": [],
            "worst_month": "",
            "worst_month_spread": 0.0,
        }
    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    monthly = daily.groupby("month", sort=True)["spread"].mean()
    monthly_dict = {str(month): _finite_float(value) for month, value in monthly.items()}
    negative_months = [month for month, value in monthly_dict.items() if value <= 0.0]
    worst_month = str(monthly.idxmin()) if len(monthly) else ""
    return {
        "monthly_spread": monthly_dict,
        "monthly_positive_rate": _finite_float((monthly > 0.0).mean() if len(monthly) else 0.0),
        "negative_months": negative_months,
        "worst_month": worst_month,
        "worst_month_spread": _finite_float(monthly.loc[worst_month] if worst_month else 0.0),
    }


def _prediction_role_stats(study_root: Path, role: str) -> dict[str, Any]:
    csv_path = study_root / f"forecast_predictions_{role}.csv"
    if not csv_path.exists():
        return {"available": False, "path": str(csv_path)}
    columns = _csv_columns(csv_path)
    hit_columns = [column for column in columns if re.match(r"^future_hit_label_\d+d$", str(column))]
    frame = _read_csv_subset(
        csv_path,
        [
            "date",
            "stock",
            "pred_decision_score",
            "future_decision_score",
            "pred_best_horizon",
            "future_best_horizon",
            "history_bucket",
            "stock_seen_in_train",
            *hit_columns,
        ],
    )
    if frame.empty:
        return {"available": False, "path": str(csv_path), "reason": "no_required_columns"}
    payload: dict[str, Any] = {
        "available": True,
        "path": str(csv_path),
        "row_count": int(len(frame)),
        "date_count": int(pd.to_datetime(frame.get("date"), errors="coerce").nunique()) if "date" in frame else 0,
        "stock_count": int(frame["stock"].astype(str).nunique()) if "stock" in frame else 0,
        "future_decision_score": _summary_stats(frame.get("future_decision_score", pd.Series(dtype=float))),
        "pred_decision_score": _summary_stats(frame.get("pred_decision_score", pd.Series(dtype=float))),
        "pred_best_horizon_distribution": _value_counts_payload(frame.get("pred_best_horizon", pd.Series(dtype=float))),
        "pred_best_horizon_share": _value_counts_payload(frame.get("pred_best_horizon", pd.Series(dtype=float)), normalize=True),
        "future_best_horizon_distribution": _value_counts_payload(frame.get("future_best_horizon", pd.Series(dtype=float))),
        "future_best_horizon_share": _value_counts_payload(frame.get("future_best_horizon", pd.Series(dtype=float)), normalize=True),
        "monthly_spread": _monthly_spread_payload(frame),
        "hit_base_rates": {
            column: _finite_float(pd.to_numeric(frame[column], errors="coerce").mean())
            for column in hit_columns
        },
    }
    if "history_bucket" in frame:
        payload["history_bucket_counts"] = {
            str(key): int(value) for key, value in frame["history_bucket"].astype(str).value_counts(dropna=False).sort_index().items()
        }
    if "stock_seen_in_train" in frame:
        seen = frame["stock_seen_in_train"].astype(str).str.lower().isin({"true", "1", "yes"})
        payload["stock_seen_in_train_share"] = _finite_float(seen.mean())
    return payload


def _seed_prediction_stats(studies_root: Path, tags: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tag in tags:
        study_root = studies_root / str(tag)
        summary = _read_json(study_root / "study_summary.json")
        training = _read_json(study_root / "forecast_training_summary.json")
        seed = summary.get("seed", training.get("selected_seed", ""))
        rows.append(
            {
                "tag": str(tag),
                "study_root": str(study_root),
                "payload_exists": bool(study_root.exists()),
                "status": str(summary.get("status", training.get("status", ""))),
                "evidence_verdict": str(summary.get("evidence_verdict", training.get("evidence_verdict", ""))),
                "seed": _finite_int(seed, default=-1),
                "validation": _prediction_role_stats(study_root, "validation"),
                "test": _prediction_role_stats(study_root, "test"),
            }
        )
    return rows


def _manifest_summary(studies_root: Path, tags: Iterable[str]) -> dict[str, Any]:
    first_tag = next(iter(tags), "")
    manifest_path = studies_root / first_tag / "forecast_dataset_manifest.json"
    validation_path = studies_root / first_tag / "rebuild_memmap_validation.json"
    manifest = _read_json(manifest_path)
    validation = _read_json(validation_path)
    return {
        "manifest_path": str(manifest_path),
        "manifest_exists": bool(manifest_path.exists()),
        "validation_path": str(validation_path),
        "validation_status": str(validation.get("status", "")),
        "validation_blockers": list(validation.get("blockers", []) or []),
        "source_market_dataset_id": str(manifest.get("source_market_dataset_id", "")),
        "source_pool_view_id": str(manifest.get("source_pool_view_id", "")),
        "source_pool_view_kind": str(manifest.get("source_pool_view_kind", "")),
        "source_pool_view_name": str(manifest.get("source_pool_view_name", "")),
        "feature_profile": str(manifest.get("feature_profile", "")),
        "feature_count": _finite_int(manifest.get("feature_count_after_cap", len(manifest.get("feature_columns", []) or []))),
        "feature_count_before_cap": _finite_int(manifest.get("feature_count_before_cap", 0)),
        "feature_store_shape": [int(item) for item in manifest.get("feature_store_shape", []) or []],
        "sample_count": _finite_int(manifest.get("sample_count", 0)),
        "sample_count_by_role": {str(key): int(value) for key, value in (manifest.get("sample_count_by_role", {}) or {}).items()},
        "date_count": len(manifest.get("date_values", []) or []),
        "stock_count": len(manifest.get("stock_values", []) or []),
        "role_years": dict(manifest.get("role_years", {}) or {}),
        "label_semantics": dict(manifest.get("label_semantics", {}) or {}),
        "feature_group_counts": dict(manifest.get("feature_group_counts", {}) or {}),
        "feature_columns": [str(item) for item in manifest.get("feature_columns", []) or []],
        "normalization": {
            "method": str((manifest.get("normalization", {}) or {}).get("method", "")),
            "fit_role": str((manifest.get("normalization", {}) or {}).get("fit_role", "")),
            "raw_feature_nan_ratio": _finite_float((manifest.get("normalization", {}) or {}).get("raw_feature_nan_ratio", 0.0)),
        },
    }


def _dataset_record(data_lake_manifest: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    for record in data_lake_manifest.get("datasets", []) or []:
        if isinstance(record, dict) and str(record.get("dataset_id", "")) == str(dataset_id):
            return record
    return {}


def _lake_summary(data_lake_manifest_path: Path, source_dataset_id: str, pool_view_id: str) -> dict[str, Any]:
    payload = _read_json(data_lake_manifest_path)
    source = _dataset_record(payload, source_dataset_id)
    pool = _dataset_record(payload, pool_view_id)
    source_params = source.get("parameters", {}) if isinstance(source.get("parameters"), dict) else {}
    market_cache = pool.get("source_cache", {}) if isinstance(pool.get("source_cache"), dict) else {}
    quality = market_cache.get("quality_report", {}) if isinstance(market_cache.get("quality_report"), dict) else {}
    return {
        "manifest_path": str(data_lake_manifest_path),
        "manifest_exists": bool(data_lake_manifest_path.exists()),
        "source_dataset": {
            "dataset_id": str(source.get("dataset_id", "")),
            "dataset_kind": str(source.get("dataset_kind", "")),
            "provider_chain": [str(item) for item in source_params.get("provider_chain", []) or []],
            "provider_plan": str(source_params.get("provider_plan", "")),
            "start_date": str(source.get("start_date", source_params.get("start_date", ""))),
            "end_date": str(source.get("end_date", source_params.get("end_date", ""))),
            "row_counts": dict(source.get("row_counts", {}) or {}),
            "coverage_report": dict(source_params.get("coverage_report", {}) or {}),
            "sidecar_dataset_ids": dict(source_params.get("sidecar_dataset_ids", {}) or {}),
        },
        "pool_view": {
            "dataset_id": str(pool.get("dataset_id", "")),
            "view_kind": str(pool.get("parameters", {}).get("view_kind", "") if isinstance(pool.get("parameters"), dict) else ""),
            "view_name": str(pool.get("parameters", {}).get("view_name", "") if isinstance(pool.get("parameters"), dict) else ""),
            "row_counts": dict(pool.get("row_counts", {}) or {}),
            "quality_report": quality,
        },
    }


def _profile_aggregate(anchor_root: Path) -> dict[str, Any]:
    path = anchor_root / "profile_aggregate.csv"
    if not path.exists():
        return {"available": False, "path": str(path)}
    frame = pd.read_csv(path)
    rows = frame[(frame.get("role") == "test") & (frame.get("score_name") == "pred_decision_score")]
    if rows.empty:
        return {"available": False, "path": str(path), "reason": "missing_test_pred_decision_score"}
    row = rows.iloc[0].to_dict()
    fields = (
        "seed_count",
        "rank_ic_mean",
        "rank_ic_min",
        "spread_mean",
        "spread_min",
        "hit_lift_mean",
        "hit_lift_min",
        "monthly_positive_rate_mean",
        "monthly_positive_rate_min",
        "negative_month_count_max",
        "long_horizon_share_mean",
        "thirty_d_concentration_mean",
        "future_long_horizon_share_mean",
        "pred_future_horizon_gap_mean",
    )
    return {
        "available": True,
        "path": str(path),
        **{field: _finite_float(row.get(field, 0.0)) for field in fields if field in row},
        "all_seed_rank_ic_spread_hit_positive": str(row.get("all_seed_rank_ic_spread_hit_positive", "")).lower() == "true",
        "stage3_weak_gate_pass": str(row.get("stage3_weak_gate_pass", "")).lower() == "true",
    }


def _seed_score_rows(anchor_root: Path) -> list[dict[str, Any]]:
    path = anchor_root / "score_variant_comparison.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    rows = frame[(frame.get("role") == "test") & (frame.get("score_name") == "pred_decision_score")]
    out: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        out.append(
            {
                "tag": str(row.get("study_tag", "")),
                "seed": _finite_int(row.get("seed", -1), default=-1),
                "rank_ic": _finite_float(row.get("decision_score_rank_ic", 0.0)),
                "spread": _finite_float(row.get("decision_score_top_bottom_spread", 0.0)),
                "hit_lift": _finite_float(row.get("decision_hit_lift_top20_mean", 0.0)),
                "monthly_positive_rate": _finite_float(row.get("monthly_spread_positive_rate", 0.0)),
                "negative_month_count": _finite_int(row.get("negative_month_count", 0)),
                "worst_month_spread": _finite_float(row.get("worst_month_spread", 0.0)),
                "long_horizon_share": _finite_float(row.get("long_horizon_share", 0.0)),
                "thirty_d_concentration": _finite_float(row.get("thirty_d_concentration", 0.0)),
                "pred_future_horizon_gap": _finite_float(row.get("pred_future_horizon_gap", 0.0)),
            }
        )
    return out


def _diagnostic_hints(anchor_root: Path) -> dict[str, Any]:
    decision = _read_json(anchor_root / "rebuild_decision_score_diagnostics.json")
    target = _read_json(anchor_root / "rebuild_target_calibration_audit.json")
    target_gate = target.get("gate_a", {}) if isinstance(target.get("gate_a"), dict) else {}
    return {
        "decision_score_conclusion_hint": str(decision.get("conclusion_hint", "")),
        "score_target_inconsistency_reasons": [str(item) for item in decision.get("score_target_inconsistency_reasons", []) or []],
        "target_gate_a_passed": bool(target_gate.get("passed", False)),
        "target_gate_a_passed_target_count": _finite_int(target_gate.get("passed_target_count", 0)),
        "target_conclusion_hints": [str(item) for item in target.get("conclusion_hints", []) or []],
    }


def _parse_stage28_reference(path: Path, old_payload_root: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    dataset_match = re.search(r"Dataset:\s+`([^`]+)`", text)
    manifest_match = re.search(
        r"universe_size=(\d+)`,\s+`train_rows=(\d+)`,\s+`feature_store_shape=\[([^\]]+)\]",
        text,
    )
    target_row: dict[str, Any] = {}
    for line in text.splitlines():
        if line.startswith("| `target_norm_head_constraint_v1`"):
            cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 11:
                target_row = {
                    "candidate": cells[0],
                    "seed_count": _finite_int(cells[1]),
                    "min_rank_ic": _finite_float(cells[2]),
                    "min_spread": _finite_float(cells[3]),
                    "min_hit_lift": _finite_float(cells[4]),
                    "mean_monthly_positive_rate": _finite_float(cells[5]),
                    "max_negative_months": _finite_int(cells[6]),
                    "mean_long_horizon_share": _finite_float(cells[7]),
                    "mean_thirty_d_concentration": _finite_float(cells[8]),
                    "mean_future_long_share": _finite_float(cells[9]),
                    "mean_pred_future_horizon_gap": _finite_float(cells[10]),
                    "gate": cells[11] if len(cells) > 11 else "",
                }
            break
    return {
        "reference_path": str(path),
        "reference_exists": bool(path.exists()),
        "payload_root": str(old_payload_root),
        "payload_exists": bool(old_payload_root.exists()),
        "evidence_level": "brain_confirmed_text_only" if path.exists() and not old_payload_root.exists() else "file_backed",
        "dataset_id": dataset_match.group(1) if dataset_match else "",
        "universe_size": _finite_int(manifest_match.group(1), 0) if manifest_match else 0,
        "train_rows": _finite_int(manifest_match.group(2), 0) if manifest_match else 0,
        "feature_store_shape": [
            _finite_int(item.strip(), 0)
            for item in (manifest_match.group(3).split(",") if manifest_match else [])
        ],
        "target_norm_head_constraint_v1": target_row,
    }


def _parse_stage36_reference(path: Path, old_payload_root: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    dataset_match = re.search(r"Dataset:\s+`([^`]+)`", text)
    final_match = re.search(r"Final decision:\s+(.+)", text)
    return {
        "reference_path": str(path),
        "reference_exists": bool(path.exists()),
        "payload_root": str(old_payload_root),
        "payload_exists": bool(old_payload_root.exists()),
        "evidence_level": "brain_confirmed_text_only" if path.exists() and not old_payload_root.exists() else "file_backed",
        "dataset_id": dataset_match.group(1) if dataset_match else "",
        "final_decision": final_match.group(1).strip() if final_match else "",
    }


def _availability_matrix(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {"path": str(path), "exists": bool(path.exists()), "is_dir": bool(path.is_dir())}
        for name, path in paths.items()
    }


def _root_cause_rankings(report: dict[str, Any]) -> list[dict[str, str]]:
    new_manifest = report.get("new_lineage", {}).get("manifest", {})
    old_stage28 = report.get("historical_evidence", {}).get("stage28", {})
    diagnostics = report.get("new_lineage", {}).get("diagnostics", {})
    rankings = [
        {
            "rank": "1",
            "cause": "new_lineage_not_payload_restore",
            "evidence": "new source_market_dataset_id differs from old Stage 2.8 dataset and old payload roots are unavailable",
            "confidence": "high",
        },
        {
            "rank": "2",
            "cause": "universe_and_pool_composition_shift",
            "evidence": (
                f"old universe_size={old_stage28.get('universe_size', 0)}, "
                f"new stock_count={new_manifest.get('stock_count', 0)}"
            ),
            "confidence": "high",
        },
        {
            "rank": "3",
            "cause": "feature_schema_shift",
            "evidence": (
                f"old feature_store_shape={old_stage28.get('feature_store_shape', [])}, "
                f"new feature_store_shape={new_manifest.get('feature_store_shape', [])}"
            ),
            "confidence": "high",
        },
        {
            "rank": "4",
            "cause": "seed11_hit_and_month_stability_failure",
            "evidence": "new seed11 test hit lift is negative and max negative months is 3 in score comparison",
            "confidence": "high",
        },
        {
            "rank": "5",
            "cause": "selection_only_fix_unlikely",
            "evidence": str(diagnostics.get("decision_score_conclusion_hint", "")),
            "confidence": "medium_high",
        },
        {
            "rank": "6",
            "cause": "provider_price_label_regime_shift",
            "evidence": "BaoStock-first provider route changed raw market/benchmark/sidecar source semantics; old raw payload is unavailable for direct OHLCV diff",
            "confidence": "medium",
        },
    ]
    return rankings


def build_rebuild_lineage_diff_audit(
    *,
    studies_root: Path = STUDIES_ROOT,
    anchor_tag: str = DEFAULT_ANCHOR_TAG,
    new_seed_tags: Iterable[str] = DEFAULT_NEW_SEED_TAGS,
    data_lake_manifest_path: Path = DATA_LAKE_MANIFEST,
    stage28_reference_path: Path = DEFAULT_STAGE28_REFERENCE,
    stage36_reference_path: Path = DEFAULT_STAGE36_REFERENCE,
    old_stage28_root: Path = DEFAULT_OLD_STAGE28_ROOT,
    old_stage36_root: Path = DEFAULT_OLD_STAGE36_ROOT,
    short_v5b_root: Path = DEFAULT_SHORT_V5B_ROOT,
    active_manifest_path: Path = DEFAULT_ACTIVE_MANIFEST,
) -> dict[str, Any]:
    tags = [str(tag) for tag in new_seed_tags]
    anchor_root = Path(studies_root) / str(anchor_tag)
    manifest = _manifest_summary(Path(studies_root), tags)
    lake = _lake_summary(
        Path(data_lake_manifest_path),
        str(manifest.get("source_market_dataset_id", "")),
        str(manifest.get("source_pool_view_id", "")),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed" if bool(manifest.get("manifest_exists")) else "blocked",
        "purpose": "Compare the file-backed new research rebuild lineage against historical Stage 2.8/Stage 3G/short_v5b evidence boundaries.",
        "boundary": {
            "research_only": True,
            "active_execution_strategy_expected_diff": "none",
            "training_launched_by_audit": False,
            "promotion_allowed": False,
        },
        "new_lineage": {
            "anchor_tag": str(anchor_tag),
            "anchor_root": str(anchor_root),
            "seed_tags": tags,
            "manifest": manifest,
            "lake": lake,
            "aggregate_test_pred_decision_score": _profile_aggregate(anchor_root),
            "seed_test_pred_decision_score": _seed_score_rows(anchor_root),
            "seed_prediction_stats": _seed_prediction_stats(Path(studies_root), tags),
            "diagnostics": _diagnostic_hints(anchor_root),
        },
        "historical_evidence": {
            "stage28": _parse_stage28_reference(Path(stage28_reference_path), Path(old_stage28_root)),
            "stage36": _parse_stage36_reference(Path(stage36_reference_path), Path(old_stage36_root)),
        },
        "payload_availability": _availability_matrix(
            {
                "active_execution_strategy": Path(active_manifest_path),
                "short_v5b_production_root": Path(short_v5b_root),
                "old_stage28_root": Path(old_stage28_root),
                "old_stage36_root": Path(old_stage36_root),
                "new_anchor_root": anchor_root,
            }
        ),
        "next_diagnostics": [
            "Compare feature columns and missing old feature groups once old payload or feature manifest is recovered.",
            "Compare pool membership overlap by date if old Stage 2.8 membership frame is recovered.",
            "Run common-symbol/date OHLCV and next-open label diff if old policy_input_bundle payload is recovered.",
            "On the new lineage, isolate seed11 negative months by regime, history bucket, horizon, and hit base rate.",
            "Before any execution rebuild, define a same-protocol backtest bridge versus short_v5b with common pool, costs, rebalance policy, benchmark, date splits, and score/target-weight panel semantics.",
        ],
    }
    report["root_cause_rankings"] = _root_cause_rankings(report)
    return report


def _seed_month_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_payload in report.get("new_lineage", {}).get("seed_prediction_stats", []) or []:
        test = seed_payload.get("test", {}) if isinstance(seed_payload, dict) else {}
        monthly = (test.get("monthly_spread", {}) or {}).get("monthly_spread", {}) if isinstance(test, dict) else {}
        for month, spread in monthly.items():
            rows.append(
                {
                    "tag": seed_payload.get("tag", ""),
                    "seed": seed_payload.get("seed", ""),
                    "role": "test",
                    "month": month,
                    "spread": _finite_float(spread),
                    "is_negative": _finite_float(spread) <= 0.0,
                }
            )
    return rows


def _markdown(report: dict[str, Any]) -> str:
    new_manifest = report["new_lineage"]["manifest"]
    aggregate = report["new_lineage"]["aggregate_test_pred_decision_score"]
    stage28 = report["historical_evidence"]["stage28"]
    lines = [
        "# Rebuild Lineage Difference Audit",
        "",
        f"- status: `{report['status']}`",
        "- boundary: research-only / shadow-only; no execution promotion or active artifact write.",
        f"- new dataset: `{new_manifest.get('source_market_dataset_id', '')}`",
        f"- new pool: `{new_manifest.get('source_pool_view_id', '')}`",
        f"- old Stage 2.8 dataset: `{stage28.get('dataset_id', '')}`",
        f"- old Stage 2.8 file-backed payload exists: `{stage28.get('payload_exists')}`",
        "",
        "## Key Deltas",
        "",
        f"- Universe/stock count: old `{stage28.get('universe_size', 0)}` vs new `{new_manifest.get('stock_count', 0)}`.",
        f"- Feature store: old `{stage28.get('feature_store_shape', [])}` vs new `{new_manifest.get('feature_store_shape', [])}`.",
        f"- Train rows: old `{stage28.get('train_rows', 0)}` vs new `{new_manifest.get('sample_count_by_role', {}).get('train', 0)}`.",
        f"- New aggregate test rank/spread/hit min: `{aggregate.get('rank_ic_min', 0.0):.6f}` / `{aggregate.get('spread_min', 0.0):.6f}` / `{aggregate.get('hit_lift_min', 0.0):.6f}`.",
        f"- New max negative months: `{aggregate.get('negative_month_count_max', 0.0):.0f}`.",
        "",
        "## Root Cause Ranking",
        "",
    ]
    for item in report.get("root_cause_rankings", []):
        lines.append(f"- {item['rank']}. `{item['cause']}` ({item['confidence']}): {item['evidence']}")
    lines.extend(
        [
            "",
            "## Seed Test Rows",
            "",
            "| Seed | Rank IC | Spread | Hit lift | Monthly positive | Negative months |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["new_lineage"].get("seed_test_pred_decision_score", []) or []:
        lines.append(
            "| {seed} | `{rank_ic:.6f}` | `{spread:.6f}` | `{hit_lift:.6f}` | `{monthly_positive_rate:.6f}` | `{negative_month_count}` |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Next Diagnostics",
            "",
            *[f"- {item}" for item in report.get("next_diagnostics", [])],
            "",
        ]
    )
    return "\n".join(lines)


def write_rebuild_lineage_diff_audit(report: dict[str, Any], output_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "rebuild_lineage_diff_audit.json"
    md_path = output_root / "rebuild_lineage_diff_audit.md"
    seed_month_path = output_root / "rebuild_seed_monthly_spread.csv"
    feature_path = output_root / "rebuild_feature_columns.csv"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    pd.DataFrame(_seed_month_rows(report)).to_csv(seed_month_path, index=False, encoding="utf-8-sig")
    features = report.get("new_lineage", {}).get("manifest", {}).get("feature_columns", []) or []
    pd.DataFrame({"feature": [str(item) for item in features]}).to_csv(feature_path, index=False, encoding="utf-8-sig")
    return {
        "json": json_path,
        "markdown": md_path,
        "seed_month_csv": seed_month_path,
        "feature_columns_csv": feature_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit new rebuild lineage deltas against historical research evidence.")
    parser.add_argument("--studies-root", default=str(STUDIES_ROOT))
    parser.add_argument("--anchor-tag", default=DEFAULT_ANCHOR_TAG)
    parser.add_argument("--new-seed-tags", default=",".join(DEFAULT_NEW_SEED_TAGS))
    parser.add_argument("--data-lake-manifest", default=str(DATA_LAKE_MANIFEST))
    parser.add_argument("--stage28-reference", default=str(DEFAULT_STAGE28_REFERENCE))
    parser.add_argument("--stage36-reference", default=str(DEFAULT_STAGE36_REFERENCE))
    parser.add_argument("--old-stage28-root", default=str(DEFAULT_OLD_STAGE28_ROOT))
    parser.add_argument("--old-stage36-root", default=str(DEFAULT_OLD_STAGE36_ROOT))
    parser.add_argument("--short-v5b-root", default=str(DEFAULT_SHORT_V5B_ROOT))
    parser.add_argument("--active-manifest", default=str(DEFAULT_ACTIVE_MANIFEST))
    parser.add_argument("--output-root", default="", help="Defaults to the anchor study root.")
    args = parser.parse_args(argv)

    studies_root = Path(args.studies_root)
    tags = [item.strip() for item in str(args.new_seed_tags).split(",") if item.strip()]
    output_root = Path(args.output_root) if str(args.output_root).strip() else studies_root / str(args.anchor_tag)
    report = build_rebuild_lineage_diff_audit(
        studies_root=studies_root,
        anchor_tag=str(args.anchor_tag),
        new_seed_tags=tags,
        data_lake_manifest_path=Path(args.data_lake_manifest),
        stage28_reference_path=Path(args.stage28_reference),
        stage36_reference_path=Path(args.stage36_reference),
        old_stage28_root=Path(args.old_stage28_root),
        old_stage36_root=Path(args.old_stage36_root),
        short_v5b_root=Path(args.short_v5b_root),
        active_manifest_path=Path(args.active_manifest),
    )
    paths = write_rebuild_lineage_diff_audit(report, output_root)
    print(
        json.dumps(
            {
                "status": report["status"],
                "json": str(paths["json"]),
                "markdown": str(paths["markdown"]),
                "top_root_cause": report.get("root_cause_rankings", [{}])[0].get("cause", ""),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
