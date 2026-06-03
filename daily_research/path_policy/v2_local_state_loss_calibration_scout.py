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

from daily_research.path_policy import v2_local_state_input_scout as local_state
from daily_research.path_policy import v2_research_reset_baseline as v2
from daily_research.path_policy.forecast_training import forecast_loss_profile_contract
from daily_research.path_policy.mainboard_rebuild_baseline import corrected_gate_status
from daily_research.path_policy.output_aux_profile_comparison import (
    build_output_aux_profile_comparison,
    write_output_aux_profile_comparison,
)


PYTHON = v2.PYTHON
PROJECT_ROOT = v2.PROJECT_ROOT
STUDIES_ROOT = v2.STUDIES_ROOT
RUN_TAG = "mh_v2_local_state_loss_calibration_anchor_20260603_01"
RESEARCH_PROGRAM = v2.RESEARCH_PROGRAM
STUDY_FAMILY = "v2_local_state_loss_calibration_scout"
DATASET_ID = v2.DATASET_ID
V2_STRICT_POOL_VIEW_ID = v2.V2_STRICT_POOL_VIEW_ID
V2_STATUS_SIDECAR_ID = v2.V2_STATUS_SIDECAR_ID
BASELINE_ANCHOR_RUN_TAG = local_state.RUN_TAG
BASELINE_FEATURE_PROFILE = local_state.FEATURE_PROFILE
FEATURE_PROFILE = local_state.FEATURE_PROFILE
MODEL_FAMILY = v2.MODEL_FAMILY
OUTPUT_PROFILE = v2.OUTPUT_PROFILE
SELECTION_PROFILE = v2.SELECTION_PROFILE
HORIZON_GRID = v2.HORIZON_GRID
DEFAULT_SEEDS = (7,)
DEFAULT_LOSS_PROFILES = (
    "score_monthly_robust_v1",
    "horizon_entropy_regularized_v1",
    "risk_drawdown_reweighted_v1",
)
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
    values = _parse_csv(raw, tuple(str(item) for item in default))
    return tuple(int(float(item)) for item in values)


def _parse_loss_profiles(raw: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    profiles = _parse_csv(raw, DEFAULT_LOSS_PROFILES)
    for profile in profiles:
        contract = forecast_loss_profile_contract(profile, cumulative_horizons=HORIZON_GRID, forecast_horizon=30)
        if str(contract.get("required_output_profile", "")) != OUTPUT_PROFILE:
            raise ValueError(f"loss_profile_output_mismatch: {profile} requires {contract.get('required_output_profile')}")
    return profiles


def _anchor_root(output_root: str | Path | None = None) -> Path:
    return Path(output_root) if output_root is not None else STUDIES_ROOT / RUN_TAG


def _slug_loss_profile(loss_profile: str) -> str:
    return str(loss_profile).strip().lower().replace("-", "_")


def _study_tag(*, loss_profile: str, seed: int) -> str:
    return f"mh_v2_local_state_loss_calibration_{_slug_loss_profile(loss_profile)}_seed{int(seed)}_20260603_01"


def _seed7_manifest_path() -> Path:
    return STUDIES_ROOT / local_state._study_tag(7) / "forecast_dataset_manifest.json"


def validate_source_manifest(manifest_path: str | Path | None = None) -> dict[str, Any]:
    return local_state.validate_local_state_manifest(manifest_path or _seed7_manifest_path())


def _training_command(*, loss_profile: str, seed: int) -> list[str]:
    tag = _study_tag(loss_profile=loss_profile, seed=seed)
    return [
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
        "--forecast-memmap-manifest",
        str(_seed7_manifest_path()),
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
        str(loss_profile),
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


def build_training_tasks(
    *,
    output_root: str | Path | None = None,
    seeds: tuple[int, ...] | list[int] | str | None = DEFAULT_SEEDS,
    loss_profiles: tuple[str, ...] | list[str] | str | None = DEFAULT_LOSS_PROFILES,
) -> list[dict[str, Any]]:
    root = _anchor_root(output_root)
    resolved_seeds = _parse_seeds(seeds)
    resolved_profiles = _parse_loss_profiles(loss_profiles)
    tasks: list[dict[str, Any]] = []
    for profile in resolved_profiles:
        contract = forecast_loss_profile_contract(profile, cumulative_horizons=HORIZON_GRID, forecast_horizon=30)
        for seed in resolved_seeds:
            tag = _study_tag(loss_profile=profile, seed=int(seed))
            tasks.append(
                {
                    "tag": tag,
                    "seed": int(seed),
                    "loss_profile": str(profile),
                    "loss_profile_contract": contract,
                    "study_dir": str(STUDIES_ROOT / tag),
                    "stdout": str(root / f"{tag}_stdout.log"),
                    "stderr": str(root / f"{tag}_stderr.log"),
                    "command": _training_command(loss_profile=str(profile), seed=int(seed)),
                    "research_program": RESEARCH_PROGRAM,
                    "study_family": STUDY_FAMILY,
                    "source_market_dataset_id": DATASET_ID,
                    "source_pool_view_id": V2_STRICT_POOL_VIEW_ID,
                    "source_status_sidecar_dataset_id": V2_STATUS_SIDECAR_ID,
                    "feature_profile": FEATURE_PROFILE,
                    "baseline_anchor_run_tag": BASELINE_ANCHOR_RUN_TAG,
                    "baseline_feature_profile": BASELINE_FEATURE_PROFILE,
                    "model_family": MODEL_FAMILY,
                    "output_profile": OUTPUT_PROFILE,
                    "horizon_grid": HORIZON_GRID,
                    "reuses_local_state_seed7_memmap_manifest": True,
                    "local_state_seed7_memmap_manifest": str(_seed7_manifest_path()),
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
    loss_profiles: tuple[str, ...] | list[str] | str | None = DEFAULT_LOSS_PROFILES,
) -> Path:
    root = _anchor_root(output_root)
    resolved_seeds = _parse_seeds(seeds)
    resolved_profiles = _parse_loss_profiles(loss_profiles)
    path = root / "v2_local_state_loss_calibration_task_list.json"
    payload = {
        "schema_version": 1,
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "created_at": _now(),
        "seeds": list(resolved_seeds),
        "loss_profiles": list(resolved_profiles),
        "training_task_count": len(resolved_seeds) * len(resolved_profiles),
        "source_manifest_validation": validate_source_manifest(),
        "training_tasks": build_training_tasks(output_root=root, seeds=resolved_seeds, loss_profiles=resolved_profiles),
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


def run_training_tasks(
    *,
    output_root: str | Path | None = None,
    seeds: tuple[int, ...] | list[int] | str | None = DEFAULT_SEEDS,
    loss_profiles: tuple[str, ...] | list[str] | str | None = DEFAULT_LOSS_PROFILES,
    skip_existing: bool = True,
) -> dict[str, Any]:
    if _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    root = _anchor_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    source_validation = validate_source_manifest()
    if source_validation["status"] != "ok":
        payload = {
            "schema_version": 1,
            "status": "blocked",
            "run_tag": RUN_TAG,
            "blockers": ["invalid_or_missing_local_state_seed7_manifest"],
            "source_manifest_validation": source_validation,
            "updated_at": _now(),
        }
        _write_json(root / "v2_local_state_loss_calibration_training_summary.json", payload)
        return payload
    resolved_seeds = _parse_seeds(seeds)
    resolved_profiles = _parse_loss_profiles(loss_profiles)
    completed: list[str] = []
    failed: list[str] = []
    results: list[dict[str, Any]] = []
    for task in build_training_tasks(output_root=root, seeds=resolved_seeds, loss_profiles=resolved_profiles):
        summary_path = Path(task["study_dir"]) / "study_summary.json"
        if skip_existing and summary_path.exists():
            completed.append(task["tag"])
            results.append({"tag": task["tag"], "status": "skipped_existing", "study_summary_json": str(summary_path)})
            continue
        with Path(task["stdout"]).open("w", encoding="utf-8", errors="replace") as stdout, Path(task["stderr"]).open(
            "w", encoding="utf-8", errors="replace"
        ) as stderr:
            proc = subprocess.Popen(task["command"], cwd=str(PROJECT_ROOT), stdout=stdout, stderr=stderr, text=True)
            returncode = proc.wait()
        row: dict[str, Any] = {
            "tag": task["tag"],
            "loss_profile": task["loss_profile"],
            "returncode": int(returncode),
            "stdout": task["stdout"],
            "stderr": task["stderr"],
        }
        if returncode == 0:
            completed.append(task["tag"])
        else:
            failed.append(task["tag"])
        results.append(row)
        if returncode != 0:
            break
    payload = {
        "schema_version": 1,
        "status": "completed" if not failed and len(completed) == len(resolved_seeds) * len(resolved_profiles) else "blocked",
        "run_tag": RUN_TAG,
        "seeds": list(resolved_seeds),
        "loss_profiles": list(resolved_profiles),
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
    _write_json(root / "v2_local_state_loss_calibration_training_summary.json", payload)
    return payload


def run_comparison_and_audits(
    *,
    output_root: str | Path | None = None,
    seeds: tuple[int, ...] | list[int] | str | None = DEFAULT_SEEDS,
    loss_profiles: tuple[str, ...] | list[str] | str | None = DEFAULT_LOSS_PROFILES,
) -> dict[str, Any]:
    root = _anchor_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    tasks = build_training_tasks(output_root=root, seeds=seeds, loss_profiles=loss_profiles)
    report = build_output_aux_profile_comparison([task["study_dir"] for task in tasks], run_tag=RUN_TAG)
    comparison_paths = write_output_aux_profile_comparison(report, root)
    gate = corrected_gate_status(root)
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
        "loss_profiles": list(_parse_loss_profiles(loss_profiles)),
        "comparison_paths": {key: str(path) for key, path in comparison_paths.items()},
        "source_manifest_validation": validate_source_manifest(),
        "gate": gate,
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        },
        "updated_at": _now(),
    }
    _write_json(root / "v2_local_state_loss_calibration_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run daily_research v2 local-state loss/output calibration scouts.")
    parser.add_argument("--output-root", default=str(STUDIES_ROOT / RUN_TAG))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--loss-profiles", default=",".join(DEFAULT_LOSS_PROFILES))
    parser.add_argument("--write-task-list", action="store_true")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--run-comparison", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    seeds = _parse_seeds(args.seeds)
    loss_profiles = _parse_loss_profiles(args.loss_profiles)
    outputs: dict[str, Any] = {
        "run_tag": RUN_TAG,
        "output_root": args.output_root,
        "seeds": list(seeds),
        "loss_profiles": list(loss_profiles),
    }
    if args.write_task_list or not (args.run_training or args.run_comparison):
        outputs["task_list"] = str(write_task_list(args.output_root, seeds=seeds, loss_profiles=loss_profiles))
    if args.run_training:
        outputs["training_summary"] = run_training_tasks(
            output_root=args.output_root,
            seeds=seeds,
            loss_profiles=loss_profiles,
            skip_existing=not args.no_skip_existing,
        )
    if args.run_comparison:
        outputs["comparison_summary"] = run_comparison_and_audits(output_root=args.output_root, seeds=seeds, loss_profiles=loss_profiles)
    if args.json:
        print(json.dumps(outputs, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(f"status=ok run_tag={RUN_TAG} output_root={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
