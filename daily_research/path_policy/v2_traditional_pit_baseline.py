from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.data_lake import ResearchDataLake, load_pool_view
from daily_research.path_policy import v2_research_reset_baseline as v2
from daily_research.path_policy.output_aux_profile_comparison import (
    build_output_aux_profile_comparison,
    write_output_aux_profile_comparison,
)


RUN_TAG = "mh_v2_traditional_pit_tradeable_mainboard_anchor_20260602_01"
RESEARCH_PROGRAM = "daily_research_v2_research_reset"
STUDY_FAMILY = "v2_traditional_pit_tradeable_mainboard_baseline"
DATASET_ID = v2.DATASET_ID
TRADITIONAL_PIT_STATUS_SIDECAR_ID = "data_platform_v2_status_sidecar__37dba59cdced261cddfedf11"
TRADITIONAL_PIT_STRICT_POOL_VIEW_ID = "policy_pool_view__0af1d96b413fb8927270b680"
BASELINE_V2_STATUS_SIDECAR_ID = v2.V2_STATUS_SIDECAR_ID
BASELINE_V2_STRICT_POOL_VIEW_ID = v2.V2_STRICT_POOL_VIEW_ID
FEATURE_PROFILE = v2.FEATURE_PROFILE
BASELINE_FEATURE_PROFILE = v2.FEATURE_PROFILE
MODEL_FAMILY = v2.MODEL_FAMILY
LOSS_PROFILE = v2.LOSS_PROFILE
OUTPUT_PROFILE = v2.OUTPUT_PROFILE
SELECTION_PROFILE = v2.SELECTION_PROFILE
HORIZON_GRID = v2.HORIZON_GRID
SEEDS = v2.SEEDS
EXCLUDED_PREFIXES = v2.EXCLUDED_PREFIXES
ACTIVE_MANIFEST = v2.ACTIVE_MANIFEST
STUDIES_ROOT = v2.STUDIES_ROOT
PROJECT_ROOT = v2.PROJECT_ROOT
PYTHON = v2.PYTHON


def _study_tag(seed: int) -> str:
    return f"mh_v2_traditional_pit_tradeable_mainboard_seed{int(seed)}_20260602_01"


def _anchor_root(output_root: str | Path | None = None) -> Path:
    return Path(output_root) if output_root is not None else STUDIES_ROOT / RUN_TAG


def _seed7_manifest_path() -> Path:
    return STUDIES_ROOT / _study_tag(7) / "forecast_dataset_manifest.json"


def _active_artifact_has_diff() -> bool:
    result = subprocess.run(["git", "diff", "--", str(ACTIVE_MANIFEST)], check=False, capture_output=True, text=True)
    return bool(str(result.stdout or "").strip() or str(result.stderr or "").strip())


def _training_command(*, seed: int) -> list[str]:
    command = [
        PYTHON,
        "-m",
        "daily_research.path_policy.run_alpha_path20_protocol",
        "--stage",
        "forecast-walkforward-study",
        "--tag",
        _study_tag(seed),
        "--data-source",
        "lake",
        "--lake-dataset-id",
        DATASET_ID,
        "--pool-name",
        "rolling_liquid500_tradeable_mainboard_traditional_pit_v2",
        "--pool-view-id",
        TRADITIONAL_PIT_STRICT_POOL_VIEW_ID,
        "--benchmark",
        "000300.SH",
        "--start-date",
        "20180101",
        "--end-date",
        "20241231",
        "--max-universe-size",
        "0",
        "--execution-mode",
        "next_open",
        "--forecast-dataset-mode",
        "memmap",
        "--forecast-train-start-year",
        "2019",
        "--forecast-train-end-year",
        "2022",
        "--forecast-validation-year",
        "2023",
        "--forecast-test-year",
        "2024",
        "--forecast-model-families",
        MODEL_FAMILY,
        "--forecast-feature-profile",
        FEATURE_PROFILE,
        "--forecast-include-static-context",
        "--forecast-max-feature-columns",
        "192",
        "--forecast-cumulative-horizons",
        HORIZON_GRID,
        "--forecast-horizon",
        "30",
        "--forecast-output-profile",
        OUTPUT_PROFILE,
        "--forecast-loss-profile",
        LOSS_PROFILE,
        "--forecast-selection-profile",
        SELECTION_PROFILE,
        "--forecast-decision-cost-bps",
        "20",
        "--forecast-decision-hit-threshold-bps",
        "10",
        "--forecast-decision-drawdown-penalty",
        "0.10",
        "--forecast-seeds",
        str(int(seed)),
        "--forecast-epochs",
        "24",
        "--forecast-min-epochs",
        "8",
        "--forecast-early-stop-patience",
        "6",
        "--forecast-checkpoint-every-n-epochs",
        "4",
        "--forecast-device",
        "cuda",
    ]
    if int(seed) != 7:
        command.extend(["--forecast-memmap-manifest", str(_seed7_manifest_path())])
    return command


def build_training_tasks(*, output_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = _anchor_root(output_root)
    tasks: list[dict[str, Any]] = []
    for seed in SEEDS:
        tag = _study_tag(seed)
        tasks.append(
            {
                "tag": tag,
                "seed": int(seed),
                "study_dir": str(STUDIES_ROOT / tag),
                "stdout": str(root / f"{tag}_stdout.log"),
                "stderr": str(root / f"{tag}_stderr.log"),
                "command": _training_command(seed=int(seed)),
                "research_program": RESEARCH_PROGRAM,
                "study_family": STUDY_FAMILY,
                "source_market_dataset_id": DATASET_ID,
                "source_pool_view_id": TRADITIONAL_PIT_STRICT_POOL_VIEW_ID,
                "source_status_sidecar_dataset_id": TRADITIONAL_PIT_STATUS_SIDECAR_ID,
                "comparison_pool_view_id": BASELINE_V2_STRICT_POOL_VIEW_ID,
                "comparison_status_sidecar_dataset_id": BASELINE_V2_STATUS_SIDECAR_ID,
                "feature_profile": FEATURE_PROFILE,
                "baseline_feature_profile": BASELINE_FEATURE_PROFILE,
                "model_family": MODEL_FAMILY,
                "loss_profile": LOSS_PROFILE,
                "horizon_grid": HORIZON_GRID,
                "reuses_seed7_memmap_manifest": int(seed) != 7,
                "seed7_memmap_manifest": str(_seed7_manifest_path()) if int(seed) != 7 else "",
                "shadow_only": True,
                "promotion_allowed": False,
                "active_execution_strategy_expected_diff": "none",
            }
        )
    return tasks


def write_task_list(output_root: str | Path | None = None) -> Path:
    root = _anchor_root(output_root)
    path = root / "v2_traditional_pit_task_list.json"
    payload = {
        "schema_version": 1,
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "created_at": v2._now(),
        "training_task_count": len(SEEDS),
        "training_tasks": build_training_tasks(output_root=root),
        "boundary": "research-only / shadow-only; execution remains frozen_skeleton_only.",
    }
    v2._write_json(path, payload)
    return path


def validate_seed_manifest(manifest_path: str | Path) -> dict[str, Any]:
    manifest = v2._read_json(manifest_path)
    blockers: list[str] = []
    feature_columns = [str(column) for column in manifest.get("feature_columns", [])] if manifest else []
    if not manifest:
        blockers.append("missing_manifest")
    if str(manifest.get("source_market_dataset_id", "")) != DATASET_ID:
        blockers.append("source_market_dataset_mismatch")
    if str(manifest.get("source_pool_view_id", "")) != TRADITIONAL_PIT_STRICT_POOL_VIEW_ID:
        blockers.append("source_pool_view_mismatch")
    if str(manifest.get("feature_profile", "")) != FEATURE_PROFILE:
        blockers.append("feature_profile_mismatch")
    alpha_like = [
        column
        for column in feature_columns
        if column.startswith("alpha_prior_")
        or column in {"score_none", "score_v2", "score_blend", "z_score_none", "z_score_v2"}
        or column.startswith("score_delta")
        or column.startswith("score_blend_lag")
        or column.startswith("score_rank")
        or column.startswith("score_cross")
    ]
    if alpha_like:
        blockers.append("alpha_prior_or_score_columns_present")
    return {
        "status": "ok" if not blockers else "blocked",
        "manifest_path": str(manifest_path),
        "blockers": blockers,
        "feature_count": int(len(feature_columns)),
        "alpha_like_feature_count": int(len(alpha_like)),
        "alpha_like_features": alpha_like,
        "amount_unit": dict(manifest.get("feature_profile_audit", {}).get("amount_unit", {}) or {}) if manifest else {},
    }


def _prefix_counts(symbols: list[str]) -> dict[str, int]:
    counts = {prefix: 0 for prefix in EXCLUDED_PREFIXES}
    for symbol in symbols:
        code = str(symbol).strip().upper().split(".", 1)[0]
        for prefix in counts:
            if code.startswith(prefix):
                counts[prefix] += 1
    return counts


def _pool_membership(lake: ResearchDataLake, pool_view_id: str) -> pd.DataFrame:
    return load_pool_view(lake=lake, pool_view_id=str(pool_view_id)).membership_frame.fillna(False).astype(bool)


def _status_sidecar_frame(lake: ResearchDataLake) -> pd.DataFrame:
    metadata = lake.describe_dataset(TRADITIONAL_PIT_STATUS_SIDECAR_ID)
    path = Path(str(dict(metadata.get("content_paths", {}) or {}).get("silver_domain_data", "") or ""))
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def validate_traditional_pit_strict_pool(*, data_lake_root: str | Path | None = None) -> dict[str, Any]:
    lake = ResearchDataLake(str(data_lake_root or "").strip() or None)
    pool = load_pool_view(lake=lake, pool_view_id=TRADITIONAL_PIT_STRICT_POOL_VIEW_ID)
    membership = pool.membership_frame.fillna(False).astype(bool)
    active_symbols = [str(column) for column in membership.columns if bool(membership[column].any())]
    daily_counts = membership.sum(axis=1).astype(int)
    shortfall = daily_counts[daily_counts < 500]
    status_frame = _status_sidecar_frame(lake)
    status_counts = {
        "active_is_st_rows": 0,
        "active_is_suspended_rows": 0,
        "active_is_delisted_rows": 0,
        "active_not_listed_rows": 0,
        "active_missing_bar_rows": 0,
        "active_not_tradeable_rows": 0,
    }
    if not status_frame.empty:
        active = membership.stack()
        active = active[active.astype(bool)].reset_index()
        active.columns = ["trade_date", "symbol", "in_pool"]
        active["trade_date"] = pd.to_datetime(active["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        active["symbol"] = active["symbol"].astype(str).str.strip().str.upper()
        status_frame = status_frame.copy()
        status_frame["trade_date"] = pd.to_datetime(status_frame["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        status_frame["symbol"] = status_frame["symbol"].astype(str).str.strip().str.upper()
        merged = active.merge(status_frame, on=["trade_date", "symbol"], how="left")
        status_counts = {
            "active_is_st_rows": int(merged.get("is_st", pd.Series(False, index=merged.index)).fillna(False).astype(bool).sum()),
            "active_is_suspended_rows": int(merged.get("is_suspended", pd.Series(False, index=merged.index)).fillna(False).astype(bool).sum()),
            "active_is_delisted_rows": int(merged.get("is_delisted", pd.Series(False, index=merged.index)).fillna(False).astype(bool).sum()),
            "active_not_listed_rows": int((~merged.get("is_listed_on_date", pd.Series(True, index=merged.index)).fillna(False).astype(bool)).sum()),
            "active_missing_bar_rows": int((~merged.get("has_bar", pd.Series(True, index=merged.index)).fillna(False).astype(bool)).sum()),
            "active_not_tradeable_rows": int((~merged.get("is_tradeable", pd.Series(True, index=merged.index)).fillna(False).astype(bool)).sum()),
        }
    blockers: list[str] = []
    if str(pool.metadata.get("parameters", {}).get("source_market_dataset_id", "")) != DATASET_ID:
        blockers.append("source_market_dataset_mismatch")
    if str(pool.metadata.get("parameters", {}).get("status_sidecar_dataset_id", "")) != TRADITIONAL_PIT_STATUS_SIDECAR_ID:
        blockers.append("status_sidecar_dataset_mismatch")
    if sum(_prefix_counts(active_symbols).values()) != 0:
        blockers.append("excluded_prefix_active_symbols_present")
    if any(value > 0 for value in status_counts.values()):
        blockers.append("non_tradeable_active_rows_present")
    diagnostics = []
    if not shortfall.empty:
        diagnostics.append("daily_member_count_below_500_on_some_dates")
    return {
        "schema_version": 1,
        "status": "ok" if not blockers else "blocked",
        "pool_view_id": str(pool.dataset_id),
        "source_market_dataset_id": str(pool.metadata.get("parameters", {}).get("source_market_dataset_id", "")),
        "status_sidecar_dataset_id": str(pool.metadata.get("parameters", {}).get("status_sidecar_dataset_id", "")),
        "require_tradeable": bool(pool.metadata.get("parameters", {}).get("require_tradeable", False)),
        "active_universe_size": int(len(active_symbols)),
        "membership_symbols": int(len(membership.columns)),
        "daily_member_count_min": int(daily_counts.min()) if not daily_counts.empty else 0,
        "daily_member_count_median": float(daily_counts.median()) if not daily_counts.empty else 0.0,
        "daily_member_count_max": int(daily_counts.max()) if not daily_counts.empty else 0,
        "daily_shortfall_count": int(len(shortfall)),
        "daily_shortfall_min_count": int(shortfall.min()) if not shortfall.empty else 500,
        "daily_shortfall_head": [
            {"date": pd.Timestamp(index).strftime("%Y-%m-%d"), "member_count": int(value)}
            for index, value in shortfall.head(20).items()
        ],
        "active_excluded_prefix_counts": _prefix_counts(active_symbols),
        **status_counts,
        "diagnostics": diagnostics,
        "blockers": blockers,
    }


def compare_traditional_pit_pool_to_v2(*, data_lake_root: str | Path | None = None) -> dict[str, Any]:
    lake = ResearchDataLake(str(data_lake_root or "").strip() or None)
    baseline = _pool_membership(lake, BASELINE_V2_STRICT_POOL_VIEW_ID)
    traditional = _pool_membership(lake, TRADITIONAL_PIT_STRICT_POOL_VIEW_ID)
    all_dates = baseline.index.union(traditional.index)
    all_symbols = baseline.columns.union(traditional.columns)
    baseline = baseline.reindex(index=all_dates, columns=all_symbols, fill_value=False).astype(bool)
    traditional = traditional.reindex(index=all_dates, columns=all_symbols, fill_value=False).astype(bool)
    overlap = (baseline & traditional).sum(axis=1).astype(int)
    union = (baseline | traditional).sum(axis=1).astype(int)
    removed = baseline & (~traditional)
    added = traditional & (~baseline)
    overlap_ratio = overlap.div(union.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return {
        "schema_version": 1,
        "status": "completed",
        "baseline_v2_pool_view_id": BASELINE_V2_STRICT_POOL_VIEW_ID,
        "traditional_pit_pool_view_id": TRADITIONAL_PIT_STRICT_POOL_VIEW_ID,
        "date_count": int(len(all_dates)),
        "baseline_true_cells": int(baseline.to_numpy(dtype=bool).sum()),
        "traditional_pit_true_cells": int(traditional.to_numpy(dtype=bool).sum()),
        "overlap_true_cells": int((baseline & traditional).to_numpy(dtype=bool).sum()),
        "removed_from_baseline_true_cells": int(removed.to_numpy(dtype=bool).sum()),
        "added_by_traditional_pit_true_cells": int(added.to_numpy(dtype=bool).sum()),
        "daily_overlap_ratio_mean": float(overlap_ratio.mean()) if overlap_ratio.notna().any() else 0.0,
        "daily_removed_count_mean": float(removed.sum(axis=1).mean()) if len(removed) else 0.0,
        "daily_added_count_mean": float(added.sum(axis=1).mean()) if len(added) else 0.0,
    }


def write_pool_diagnostics(output_root: str | Path | None = None) -> dict[str, Path]:
    root = _anchor_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    validation = validate_traditional_pit_strict_pool()
    comparison = compare_traditional_pit_pool_to_v2()
    validation_path = root / "traditional_pit_strict_pool_validation.json"
    comparison_path = root / "traditional_pit_pool_overlap_vs_v2.json"
    v2._write_json(validation_path, validation)
    v2._write_json(comparison_path, comparison)
    return {"validation": validation_path, "comparison": comparison_path}


def run_training_tasks(*, output_root: str | Path | None = None, skip_existing: bool = True) -> dict[str, Any]:
    if _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    root = _anchor_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    failed: list[str] = []
    results: list[dict[str, Any]] = []
    for task in build_training_tasks(output_root=root):
        summary_path = Path(task["study_dir"]) / "study_summary.json"
        if skip_existing and summary_path.exists():
            completed.append(task["tag"])
            results.append({"tag": task["tag"], "status": "skipped_existing", "study_summary_json": str(summary_path)})
            continue
        if bool(task.get("reuses_seed7_memmap_manifest")) and not _seed7_manifest_path().exists():
            failed.append(task["tag"])
            results.append({"tag": task["tag"], "status": "blocked_missing_seed7_memmap_manifest", "manifest": str(_seed7_manifest_path())})
            break
        if bool(task.get("reuses_seed7_memmap_manifest")):
            seed7_validation = validate_seed_manifest(_seed7_manifest_path())
            if seed7_validation["status"] != "ok":
                failed.append(task["tag"])
                results.append({"tag": task["tag"], "status": "blocked_invalid_seed7_memmap_manifest", "validation": seed7_validation})
                break
        with Path(task["stdout"]).open("w", encoding="utf-8", errors="replace") as stdout, Path(task["stderr"]).open("w", encoding="utf-8", errors="replace") as stderr:
            proc = subprocess.Popen(task["command"], cwd=str(PROJECT_ROOT), stdout=stdout, stderr=stderr, text=True)
            returncode = proc.wait()
        row = {"tag": task["tag"], "returncode": int(returncode), "stdout": task["stdout"], "stderr": task["stderr"]}
        if returncode == 0:
            manifest_path = Path(task["study_dir"]) / "forecast_dataset_manifest.json"
            if int(task["seed"]) == 7 and manifest_path.exists():
                row["seed_manifest_validation"] = validate_seed_manifest(manifest_path)
                if row["seed_manifest_validation"]["status"] != "ok":
                    failed.append(task["tag"])
                    results.append(row)
                    break
            completed.append(task["tag"])
        else:
            failed.append(task["tag"])
        results.append(row)
        if returncode != 0:
            break
    payload = {
        "schema_version": 1,
        "status": "completed" if not failed and len(completed) == len(SEEDS) else "blocked",
        "run_tag": RUN_TAG,
        "completed_tags": completed,
        "failed_tags": failed,
        "results": results,
        "updated_at": v2._now(),
        "active_execution_strategy_expected_diff": "none",
    }
    v2._write_json(root / "traditional_pit_training_summary.json", payload)
    return payload


def run_comparison_and_audits(*, output_root: str | Path | None = None) -> dict[str, Any]:
    root = _anchor_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    tasks = build_training_tasks(output_root=root)
    report = build_output_aux_profile_comparison([task["study_dir"] for task in tasks], run_tag=RUN_TAG)
    comparison_paths = write_output_aux_profile_comparison(report, root)
    pool_paths = write_pool_diagnostics(root)
    gate = v2.v2_gate_status(root)
    summary = {
        "schema_version": 1,
        "status": "completed",
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "dataset_id": DATASET_ID,
        "traditional_pit_strict_pool_view_id": TRADITIONAL_PIT_STRICT_POOL_VIEW_ID,
        "traditional_pit_status_sidecar_dataset_id": TRADITIONAL_PIT_STATUS_SIDECAR_ID,
        "feature_profile": FEATURE_PROFILE,
        "baseline_v2_strict_pool_view_id": BASELINE_V2_STRICT_POOL_VIEW_ID,
        "baseline_v2_status_sidecar_dataset_id": BASELINE_V2_STATUS_SIDECAR_ID,
        "comparison_paths": {key: str(path) for key, path in comparison_paths.items()},
        "pool_diagnostic_paths": {key: str(path) for key, path in pool_paths.items()},
        "traditional_pit_gate": gate,
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        },
        "updated_at": v2._now(),
    }
    v2._write_json(root / "v2_traditional_pit_baseline_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run daily_research v2 traditional-PIT strict tradeable mainboard baseline.")
    parser.add_argument("--output-root", default=str(STUDIES_ROOT / RUN_TAG))
    parser.add_argument("--write-task-list", action="store_true")
    parser.add_argument("--validate-pool", action="store_true")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--run-comparison", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    outputs: dict[str, Any] = {"run_tag": RUN_TAG, "output_root": args.output_root}
    if args.write_task_list or not (args.validate_pool or args.run_training or args.run_comparison):
        outputs["task_list"] = str(write_task_list(args.output_root))
    if args.validate_pool:
        paths = write_pool_diagnostics(args.output_root)
        outputs["pool_diagnostics"] = {key: str(path) for key, path in paths.items()}
        outputs["pool_validation"] = v2._read_json(paths["validation"])
        outputs["pool_overlap"] = v2._read_json(paths["comparison"])
    if args.run_training:
        outputs["training_summary"] = run_training_tasks(output_root=args.output_root, skip_existing=not args.no_skip_existing)
    if args.run_comparison:
        outputs["comparison_summary"] = run_comparison_and_audits(output_root=args.output_root)
    if args.json:
        print(json.dumps(outputs, ensure_ascii=False, indent=2, default=v2._json_default))
    else:
        print(f"status=ok run_tag={RUN_TAG} output_root={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
