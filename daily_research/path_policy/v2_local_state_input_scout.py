from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_data_platform.lake import ResearchDataLake
from quant_data_platform.lake.policy_input_loader import load_policy_inputs_from_lake
from daily_research.path_policy import v2_research_reset_baseline as v2
from daily_research.path_policy.forecast_features import build_forecast_feature_panels
from daily_research.path_policy.mainboard_rebuild_baseline import corrected_gate_status
from daily_research.path_policy.output_aux_profile_comparison import (
    build_output_aux_profile_comparison,
    write_output_aux_profile_comparison,
)


PYTHON = v2.PYTHON
PROJECT_ROOT = v2.PROJECT_ROOT
STUDIES_ROOT = v2.STUDIES_ROOT
DATA_LAKE_ROOT = PROJECT_ROOT / "quant_data_platform/data/lake"
RUN_TAG = "mh_v2_local_state_input_scout_anchor_20260602_01"
RESEARCH_PROGRAM = v2.RESEARCH_PROGRAM
STUDY_FAMILY = "v2_local_state_input_scout"
DATASET_ID = v2.DATASET_ID
V2_STRICT_POOL_VIEW_ID = v2.V2_STRICT_POOL_VIEW_ID
V2_STATUS_SIDECAR_ID = v2.V2_STATUS_SIDECAR_ID
BASELINE_ANCHOR_RUN_TAG = v2.RUN_TAG
BASELINE_FEATURE_PROFILE = v2.FEATURE_PROFILE
FEATURE_PROFILE = "raw_kline_context_v2_tradeable_local_state_v1"
MODEL_FAMILY = v2.MODEL_FAMILY
LOSS_PROFILE = v2.LOSS_PROFILE
OUTPUT_PROFILE = v2.OUTPUT_PROFILE
SELECTION_PROFILE = v2.SELECTION_PROFILE
HORIZON_GRID = v2.HORIZON_GRID
DEFAULT_SEEDS = (7,)
FULL_CONFIRM_SEEDS = v2.SEEDS
ACTIVE_MANIFEST = v2.ACTIVE_MANIFEST


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


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        return {}
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _active_artifact_has_diff() -> bool:
    result = subprocess.run(["git", "diff", "--", str(ACTIVE_MANIFEST)], check=False, capture_output=True, text=True)
    return bool(str(result.stdout or "").strip() or str(result.stderr or "").strip())


def _parse_seeds(raw: str | tuple[int, ...] | list[int] | None, default: tuple[int, ...] = DEFAULT_SEEDS) -> tuple[int, ...]:
    if raw is None:
        values = list(default)
    elif isinstance(raw, (tuple, list)):
        values = [int(item) for item in raw]
    else:
        values = [int(float(item.strip())) for item in str(raw or "").split(",") if item.strip()]
    out: list[int] = []
    seen: set[int] = set()
    for seed in values or list(default):
        value = int(seed)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def _anchor_root(output_root: str | Path | None = None) -> Path:
    return Path(output_root) if output_root is not None else STUDIES_ROOT / RUN_TAG


def _study_tag(seed: int) -> str:
    return f"mh_v2_local_state_input_scout_seed{int(seed)}_20260602_01"


def _seed7_manifest_path() -> Path:
    return STUDIES_ROOT / _study_tag(7) / "forecast_dataset_manifest.json"


def _training_command(*, seed: int) -> list[str]:
    tag = _study_tag(seed)
    command = [
        PYTHON,
        "-m",
        "daily_research.path_policy.run_alpha_path20_protocol",
        "--stage",
        "forecast-walkforward-study",
        "--tag",
        tag,
        "--data-source",
        "lake",
        "--lake-dataset-id",
        DATASET_ID,
        "--pool-name",
        "rolling_liquid500_tradeable_mainboard_v2",
        "--pool-view-id",
        V2_STRICT_POOL_VIEW_ID,
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


def build_training_tasks(
    *,
    output_root: str | Path | None = None,
    seeds: tuple[int, ...] | list[int] | str | None = DEFAULT_SEEDS,
) -> list[dict[str, Any]]:
    root = _anchor_root(output_root)
    resolved_seeds = _parse_seeds(seeds)
    tasks: list[dict[str, Any]] = []
    for seed in resolved_seeds:
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
                "source_pool_view_id": V2_STRICT_POOL_VIEW_ID,
                "source_status_sidecar_dataset_id": V2_STATUS_SIDECAR_ID,
                "feature_profile": FEATURE_PROFILE,
                "baseline_anchor_run_tag": BASELINE_ANCHOR_RUN_TAG,
                "baseline_feature_profile": BASELINE_FEATURE_PROFILE,
                "model_family": MODEL_FAMILY,
                "loss_profile": LOSS_PROFILE,
                "horizon_grid": HORIZON_GRID,
                "reuses_seed7_memmap_manifest": int(seed) != 7,
                "seed7_memmap_manifest": str(_seed7_manifest_path()) if int(seed) != 7 else "",
                "evidence_grade": "scout_only" if tuple(resolved_seeds) == (7,) else "evidence_grade_candidate",
                "shadow_only": True,
                "promotion_allowed": False,
                "active_execution_strategy_expected_diff": "none",
            }
        )
    return tasks


def write_task_list(
    output_root: str | Path | None = None,
    *,
    seeds: tuple[int, ...] | list[int] | str | None = DEFAULT_SEEDS,
) -> Path:
    root = _anchor_root(output_root)
    path = root / "v2_local_state_input_scout_task_list.json"
    resolved_seeds = _parse_seeds(seeds)
    payload = {
        "schema_version": 1,
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "created_at": _now(),
        "training_task_count": len(resolved_seeds),
        "training_tasks": build_training_tasks(output_root=root, seeds=resolved_seeds),
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
            "baseline_anchor_run_tag": BASELINE_ANCHOR_RUN_TAG,
        },
    }
    _write_json(path, payload)
    return path


def validate_local_state_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = _read_json(path)
    feature_columns = [str(column) for column in manifest.get("feature_columns", []) or []]
    shape = [int(item) for item in manifest.get("feature_store_shape", []) or []]
    group_counts = dict(manifest.get("feature_group_counts", {}) or {})
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
    blockers: list[str] = []
    if not path.exists():
        blockers.append("missing_manifest")
    if str(manifest.get("source_market_dataset_id", "")) != DATASET_ID:
        blockers.append("source_market_dataset_mismatch")
    if str(manifest.get("source_pool_view_id", "")) != V2_STRICT_POOL_VIEW_ID:
        blockers.append("source_pool_view_mismatch")
    if str(manifest.get("feature_profile", "")) != FEATURE_PROFILE:
        blockers.append("feature_profile_mismatch")
    if len(shape) == 3 and int(shape[2]) != len(feature_columns):
        blockers.append("feature_shape_column_count_mismatch")
    if int(group_counts.get("local_state_context", 0) or 0) <= 0:
        blockers.append("missing_local_state_context_features")
    if not any(column == "local_vol_20d" for column in feature_columns):
        blockers.append("missing_local_vol_20d")
    if not any(column == "local_high_volatility_x_reversal" for column in feature_columns):
        blockers.append("missing_local_high_volatility_x_reversal")
    if alpha_like:
        blockers.append("alpha_prior_or_score_columns_present")
    amount_audit = dict(manifest.get("feature_profile_audit", {}).get("amount_unit", {}) or {})
    if not amount_audit:
        blockers.append("missing_amount_unit_audit")
    return {
        "schema_version": 1,
        "status": "ok" if not blockers else "blocked",
        "manifest_path": str(path),
        "blockers": blockers,
        "feature_profile": str(manifest.get("feature_profile", "")),
        "feature_count": int(len(feature_columns)),
        "feature_store_shape": shape,
        "feature_group_counts": group_counts,
        "local_state_context_feature_count": int(group_counts.get("local_state_context", 0) or 0),
        "alpha_like_feature_count": int(len(alpha_like)),
        "alpha_like_features": alpha_like,
        "amount_unit": amount_audit,
    }


def run_profile_smoke(
    *,
    output_root: str | Path | None = None,
    data_lake_root: str | Path = DATA_LAKE_ROOT,
    start_date: str = "2024-01-01",
    end_date: str = "2024-03-29",
    max_feature_columns: int = 256,
) -> dict[str, Any]:
    root = _anchor_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    lake = ResearchDataLake(data_lake_root)
    prepared = load_policy_inputs_from_lake(
        lake=lake,
        dataset_id=DATASET_ID,
        pool_view_id=V2_STRICT_POOL_VIEW_ID,
        start_date=start_date,
        end_date=end_date,
        benchmark="000300.SH",
        max_universe_size=0,
        min_trading_days=40,
        alpha_prior_source="none",
        require_benchmark_open=True,
    )
    date = pd.Timestamp(prepared.close.index[-1]).normalize()
    panels, columns, manifest = build_forecast_feature_panels(
        prepared,
        [date],
        feature_profile=FEATURE_PROFILE,
        max_feature_columns=int(max_feature_columns),
    )
    panel = panels[date]
    local_columns = [
        column
        for column in columns
        if column.startswith("local_") or column.startswith("cs_rank_local_") or column.startswith("cs_z_local_")
    ]
    local_values = panel[local_columns].apply(pd.to_numeric, errors="coerce") if local_columns else pd.DataFrame(index=panel.index)
    finite = np.isfinite(local_values.to_numpy(dtype=float)) if local_columns else np.zeros((0, 0), dtype=bool)
    payload = {
        "schema_version": 1,
        "status": "ok" if local_columns and not any(_alpha_like_column(column) for column in columns) else "blocked",
        "run_tag": RUN_TAG,
        "dataset_id": DATASET_ID,
        "pool_view_id": V2_STRICT_POOL_VIEW_ID,
        "feature_profile": FEATURE_PROFILE,
        "smoke_date": date.strftime("%Y-%m-%d"),
        "universe_size": int(len(prepared.universe)),
        "feature_count": int(len(columns)),
        "local_state_feature_count": int(manifest.get("local_state_context_feature_count", 0) or 0),
        "local_columns": local_columns,
        "local_finite_ratio": float(finite.sum() / max(finite.size, 1)) if finite.size else 0.0,
        "has_alpha_like": bool(any(_alpha_like_column(column) for column in columns)),
        "amount_unit": dict(manifest.get("feature_profile_audit", {}).get("amount_unit", {}) or {}),
        "manifest": manifest,
        "updated_at": _now(),
    }
    if payload["status"] != "ok":
        payload["blockers"] = ["missing_local_columns_or_alpha_like_leakage"]
    _write_json(root / "v2_local_state_profile_smoke.json", payload)
    return payload


def _alpha_like_column(column: str) -> bool:
    name = str(column)
    return bool(
        name.startswith("alpha_prior_")
        or name in {"score_none", "score_v2", "score_blend", "z_score_none", "z_score_v2"}
        or name.startswith("score_delta")
        or name.startswith("score_blend_lag")
        or name.startswith("score_rank")
        or name.startswith("score_cross")
    )


def run_training_tasks(
    *,
    output_root: str | Path | None = None,
    seeds: tuple[int, ...] | list[int] | str | None = DEFAULT_SEEDS,
    skip_existing: bool = True,
) -> dict[str, Any]:
    if _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    root = _anchor_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    resolved_seeds = _parse_seeds(seeds)
    completed: list[str] = []
    failed: list[str] = []
    results: list[dict[str, Any]] = []
    for task in build_training_tasks(output_root=root, seeds=resolved_seeds):
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
            seed7_validation = validate_local_state_manifest(_seed7_manifest_path())
            if seed7_validation["status"] != "ok":
                failed.append(task["tag"])
                results.append({"tag": task["tag"], "status": "blocked_invalid_seed7_memmap_manifest", "validation": seed7_validation})
                break
        with Path(task["stdout"]).open("w", encoding="utf-8", errors="replace") as stdout, Path(task["stderr"]).open(
            "w", encoding="utf-8", errors="replace"
        ) as stderr:
            proc = subprocess.Popen(task["command"], cwd=str(PROJECT_ROOT), stdout=stdout, stderr=stderr, text=True)
            returncode = proc.wait()
        row: dict[str, Any] = {"tag": task["tag"], "returncode": int(returncode), "stdout": task["stdout"], "stderr": task["stderr"]}
        if returncode == 0:
            manifest_path = Path(task["study_dir"]) / "forecast_dataset_manifest.json"
            if manifest_path.exists():
                row["seed_manifest_validation"] = validate_local_state_manifest(manifest_path)
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
        "status": "completed" if not failed and len(completed) == len(resolved_seeds) else "blocked",
        "run_tag": RUN_TAG,
        "seeds": list(resolved_seeds),
        "completed_tags": completed,
        "failed_tags": failed,
        "results": results,
        "updated_at": _now(),
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        },
    }
    _write_json(root / "v2_local_state_training_summary.json", payload)
    return payload


def run_comparison_and_audits(
    *,
    output_root: str | Path | None = None,
    seeds: tuple[int, ...] | list[int] | str | None = DEFAULT_SEEDS,
) -> dict[str, Any]:
    root = _anchor_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    tasks = build_training_tasks(output_root=root, seeds=seeds)
    report = build_output_aux_profile_comparison([task["study_dir"] for task in tasks], run_tag=RUN_TAG)
    comparison_paths = write_output_aux_profile_comparison(report, root)
    gate = corrected_gate_status(root)
    manifest_validations = []
    for task in tasks:
        manifest_path = Path(task["study_dir"]) / "forecast_dataset_manifest.json"
        if manifest_path.exists():
            manifest_validations.append({"tag": task["tag"], **validate_local_state_manifest(manifest_path)})
    summary = {
        "schema_version": 1,
        "status": "completed",
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "dataset_id": DATASET_ID,
        "pool_view_id": V2_STRICT_POOL_VIEW_ID,
        "feature_profile": FEATURE_PROFILE,
        "baseline_anchor_run_tag": BASELINE_ANCHOR_RUN_TAG,
        "baseline_feature_profile": BASELINE_FEATURE_PROFILE,
        "comparison_paths": {key: str(path) for key, path in comparison_paths.items()},
        "local_state_manifest_validations": manifest_validations,
        "gate": gate,
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        },
        "updated_at": _now(),
    }
    _write_json(root / "v2_local_state_input_scout_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run daily_research v2 local-state input scout tasks and audits.")
    parser.add_argument("--output-root", default=str(STUDIES_ROOT / RUN_TAG))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--write-task-list", action="store_true")
    parser.add_argument("--validate-profile", action="store_true")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--run-comparison", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    seeds = _parse_seeds(args.seeds)
    outputs: dict[str, Any] = {"run_tag": RUN_TAG, "output_root": args.output_root, "seeds": list(seeds)}
    if args.write_task_list or not (args.validate_profile or args.run_training or args.run_comparison):
        outputs["task_list"] = str(write_task_list(args.output_root, seeds=seeds))
    if args.validate_profile:
        outputs["profile_smoke"] = run_profile_smoke(output_root=args.output_root)
    if args.run_training:
        outputs["training_summary"] = run_training_tasks(output_root=args.output_root, seeds=seeds, skip_existing=not args.no_skip_existing)
    if args.run_comparison:
        outputs["comparison_summary"] = run_comparison_and_audits(output_root=args.output_root, seeds=seeds)
    if args.json:
        print(json.dumps(outputs, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(f"status=ok run_tag={RUN_TAG} output_root={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
