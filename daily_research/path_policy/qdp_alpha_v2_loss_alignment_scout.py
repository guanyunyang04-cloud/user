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

from daily_research.path_policy import v2_research_reset_baseline as v2
from daily_research.path_policy.forecast_training import forecast_loss_profile_contract


PYTHON = v2.PYTHON
PROJECT_ROOT = v2.PROJECT_ROOT
STUDIES_ROOT = v2.STUDIES_ROOT
RUN_TAG = "qdp_alpha_v2_loss_alignment_scout_20260617_02"
RESEARCH_PROGRAM = v2.RESEARCH_PROGRAM
STUDY_FAMILY = "qdp_alpha_v2_strong_model_research"
ANCHOR_RUN_TAG = "qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01"
DATASET_ID = "policy_input_bundle__f926a496f69c61f6b92b5faf"
POOL_VIEW_ID = "policy_pool_view__fc63d6b996abc7abde88bae7"
TRAINING_PACK_MANIFEST = (
    PROJECT_ROOT
    / "quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01/qdp_training_pack_manifest.json"
)
FEATURE_PROFILE = "style_structural_alpha_v2"
FEATURE_COUNT = 307
LABEL_SCHEMA = "path20_basic_v2"
LABEL_SCHEMA_VERSION = 2
MODEL_FAMILY = "hybrid_expert_fusion_static_context"
OUTPUT_PROFILE = "decision_utility_v1"
SELECTION_PROFILE = "decision_utility"
HORIZON_GRID = "1,3,5,10,20"
DECISION_COST_BPS = 20.0
DECISION_HIT_THRESHOLD_BPS = 20.0
DECISION_DRAWDOWN_PENALTY = 0.25
DEFAULT_LOSS_PROFILES = (
    "topn_excess_rank_v1",
    "decision_score_topk_alignment_v1",
    "score_monthly_robust_v1",
    "horizon_30d_soft_penalty_v1",
)
DEFAULT_SEEDS = (7,)
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
    return tuple(int(float(item)) for item in values)


def _parse_loss_profiles(raw: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    profiles = _parse_csv(raw, DEFAULT_LOSS_PROFILES)
    for profile in profiles:
        contract = forecast_loss_profile_contract(profile, cumulative_horizons=HORIZON_GRID, forecast_horizon=20)
        if str(contract.get("required_output_profile", "")) != OUTPUT_PROFILE:
            raise ValueError(f"loss_profile_output_mismatch: {profile} requires {contract.get('required_output_profile')}")
    return profiles


def _loss_alias(loss_profile: str) -> str:
    aliases = {
        "topn_excess_rank_v1": "topn",
        "decision_score_topk_alignment_v1": "topk_align",
        "score_monthly_robust_v1": "monthly_robust",
        "horizon_30d_soft_penalty_v1": "h30soft",
        "personal_alpha_scorer_hybrid_v1": "personal_hybrid",
    }
    return aliases.get(str(loss_profile), str(loss_profile).replace("_v1", "").replace("_", ""))


def _anchor_root(output_root: str | Path | None = None) -> Path:
    return Path(output_root) if output_root is not None else STUDIES_ROOT / RUN_TAG


def _study_tag(*, loss_profile: str, seed: int, sampled_smoke: bool = False) -> str:
    suffix = "_smoke" if bool(sampled_smoke) else ""
    return f"qdp_alpha_v2_hybrid_{_loss_alias(loss_profile)}_h256_t4_b512_seed{int(seed)}{suffix}_20260617_02"


def validate_training_pack_manifest(manifest_path: str | Path | None = None) -> dict[str, Any]:
    resolved_manifest_path = Path(manifest_path) if manifest_path is not None else TRAINING_PACK_MANIFEST
    manifest = _read_json(resolved_manifest_path)
    blockers: list[str] = []
    if not manifest:
        blockers.append("missing_training_pack_manifest")
    if manifest:
        if str(manifest.get("artifact_type", "")) != "qdp_training_pack_v1":
            blockers.append("artifact_type_not_qdp_training_pack_v1")
        if str(manifest.get("feature_profile", "")) != FEATURE_PROFILE:
            blockers.append("feature_profile_mismatch")
        if int(manifest.get("feature_count", 0) or 0) != FEATURE_COUNT:
            blockers.append("feature_count_mismatch")
        label_schema = str(manifest.get("label_schema_name", manifest.get("label_schema", "")) or "")
        pool_view_id = str(manifest.get("source_pool_view_id", manifest.get("pool_view_id", "")) or "")
        if label_schema != LABEL_SCHEMA:
            blockers.append("label_schema_mismatch")
        if int(manifest.get("label_schema_version", 0) or 0) != LABEL_SCHEMA_VERSION:
            blockers.append("label_schema_version_mismatch")
        if str(manifest.get("canonical_dataset_id", "")) != DATASET_ID:
            blockers.append("canonical_dataset_id_mismatch")
        if pool_view_id != POOL_VIEW_ID:
            blockers.append("pool_view_id_mismatch")
        if list(manifest.get("cumulative_horizons", []) or []) != [1, 3, 5, 10, 20]:
            blockers.append("cumulative_horizons_mismatch")
        role_years = dict(manifest.get("role_years", {}) or {})
        expected_years = {"train_start_year": 2012, "train_end_year": 2023, "validation_year": 2024, "test_year": 2025}
        for key, value in expected_years.items():
            if int(role_years.get(key, 0) or 0) != int(value):
                blockers.append(f"role_year_{key}_mismatch")
    return {
        "status": "ok" if not blockers else "blocked",
        "blockers": blockers,
        "manifest_path": str(resolved_manifest_path),
        "feature_profile": str(manifest.get("feature_profile", "")) if manifest else "",
        "feature_count": int(manifest.get("feature_count", 0) or 0) if manifest else 0,
        "label_schema": str(manifest.get("label_schema_name", manifest.get("label_schema", "")) or "") if manifest else "",
        "label_schema_version": int(manifest.get("label_schema_version", 0) or 0) if manifest else 0,
        "sample_count": int(manifest.get("sample_count", 0) or 0) if manifest else 0,
        "pool_view_id": str(manifest.get("source_pool_view_id", manifest.get("pool_view_id", "")) or "") if manifest else "",
    }


def _training_command(*, loss_profile: str, seed: int, max_samples_per_role: int = 0, max_samples_per_date_per_role: int = 0) -> list[str]:
    sampled_smoke = int(max_samples_per_role) > 0 or int(max_samples_per_date_per_role) > 0
    tag = _study_tag(loss_profile=loss_profile, seed=seed, sampled_smoke=sampled_smoke)
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
        "--execution-mode",
        "next_open",
        "--forecast-dataset-mode",
        "memmap",
        "--forecast-memmap-manifest",
        str(TRAINING_PACK_MANIFEST),
        "--forecast-train-start-year",
        "2012",
        "--forecast-train-end-year",
        "2023",
        "--forecast-validation-year",
        "2024",
        "--forecast-test-year",
        "2025",
        "--forecast-model-families",
        MODEL_FAMILY,
        "--forecast-max-feature-columns",
        "384",
        "--forecast-include-static-context",
        "--forecast-output-profile",
        OUTPUT_PROFILE,
        "--forecast-loss-profile",
        str(loss_profile),
        "--forecast-selection-profile",
        SELECTION_PROFILE,
        "--forecast-decision-cost-bps",
        f"{DECISION_COST_BPS:g}",
        "--forecast-decision-hit-threshold-bps",
        f"{DECISION_HIT_THRESHOLD_BPS:g}",
        "--forecast-decision-drawdown-penalty",
        f"{DECISION_DRAWDOWN_PENALTY:g}",
        "--forecast-cumulative-horizons",
        HORIZON_GRID,
        "--forecast-horizon",
        "20",
        "--forecast-seeds",
        str(int(seed)),
        "--forecast-epochs",
        "12",
        "--forecast-min-epochs",
        "4",
        "--forecast-early-stop-patience",
        "3",
        "--forecast-batch-size",
        "512",
        "--forecast-lr",
        "0.0003",
        "--forecast-hidden-dim",
        "256",
        "--forecast-dropout",
        "0.15",
        "--forecast-gru-layers",
        "2",
        "--forecast-transformer-layers",
        "4",
        "--forecast-transformer-heads",
        "8",
        "--forecast-patch-sizes",
        "4,20",
        "--forecast-checkpoint-every-n-epochs",
        "1",
        "--forecast-device",
        "cuda",
    ]
    if int(max_samples_per_role) > 0:
        command.extend(["--forecast-max-samples-per-role", str(int(max_samples_per_role))])
    if int(max_samples_per_date_per_role) > 0:
        command.extend(["--forecast-max-samples-per-date-per-role", str(int(max_samples_per_date_per_role))])
    return command


def build_training_tasks(
    *,
    output_root: str | Path | None = None,
    seeds: tuple[int, ...] | list[int] | str | None = DEFAULT_SEEDS,
    loss_profiles: tuple[str, ...] | list[str] | str | None = DEFAULT_LOSS_PROFILES,
    max_samples_per_role: int = 0,
    max_samples_per_date_per_role: int = 0,
) -> list[dict[str, Any]]:
    root = _anchor_root(output_root)
    resolved_seeds = _parse_seeds(seeds)
    resolved_profiles = _parse_loss_profiles(loss_profiles)
    tasks: list[dict[str, Any]] = []
    for loss_profile in resolved_profiles:
        contract = forecast_loss_profile_contract(loss_profile, cumulative_horizons=HORIZON_GRID, forecast_horizon=20)
        for seed in resolved_seeds:
            sampled_smoke = int(max_samples_per_role) > 0 or int(max_samples_per_date_per_role) > 0
            tag = _study_tag(loss_profile=loss_profile, seed=int(seed), sampled_smoke=sampled_smoke)
            tasks.append(
                {
                    "tag": tag,
                    "candidate_id": f"alpha_v2_hybrid_{_loss_alias(loss_profile)}_h256_seed{int(seed)}",
                    "anchor_run_tag": ANCHOR_RUN_TAG,
                    "research_program": RESEARCH_PROGRAM,
                    "study_family": STUDY_FAMILY,
                    "dataset_id": DATASET_ID,
                    "pool_view_id": POOL_VIEW_ID,
                    "training_pack_manifest": str(TRAINING_PACK_MANIFEST),
                    "feature_profile": FEATURE_PROFILE,
                    "feature_count": FEATURE_COUNT,
                    "label_schema": LABEL_SCHEMA,
                    "label_schema_version": LABEL_SCHEMA_VERSION,
                    "model_family": MODEL_FAMILY,
                    "loss_profile": str(loss_profile),
                    "loss_profile_contract": contract,
                    "output_profile": OUTPUT_PROFILE,
                    "selection_profile": SELECTION_PROFILE,
                    "seed": int(seed),
                    "epochs": 12,
                    "min_epochs": 4,
                    "early_stop_patience": 3,
                    "batch_size": 512,
                    "hidden_dim": 256,
                    "gru_layers": 2,
                    "transformer_layers": 4,
                    "transformer_heads": 8,
                    "patch_sizes": "4,20",
                    "decision_cost_bps": DECISION_COST_BPS,
                    "decision_hit_threshold_bps": DECISION_HIT_THRESHOLD_BPS,
                    "decision_drawdown_penalty": DECISION_DRAWDOWN_PENALTY,
                    "study_dir": str(STUDIES_ROOT / tag),
                    "stdout": str(root / "logs" / f"{tag}_stdout.log"),
                    "stderr": str(root / "logs" / f"{tag}_stderr.log"),
                    "command": _training_command(
                        loss_profile=str(loss_profile),
                        seed=int(seed),
                        max_samples_per_role=int(max_samples_per_role),
                        max_samples_per_date_per_role=int(max_samples_per_date_per_role),
                    ),
                    "evidence_grade": "smoke_only" if sampled_smoke else "single_seed_scout",
                    "single_seed_scout_only": len(resolved_seeds) == 1,
                    "max_samples_per_role": int(max_samples_per_role),
                    "max_samples_per_date_per_role": int(max_samples_per_date_per_role),
                    "personal_topk_required_after_completion": True,
                    "same_candidate_test_required": True,
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
    max_samples_per_role: int = 0,
    max_samples_per_date_per_role: int = 0,
    enforce_active_artifact_clean: bool = True,
) -> Path:
    if enforce_active_artifact_clean and _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    root = _anchor_root(output_root)
    tasks = build_training_tasks(
        output_root=root,
        seeds=seeds,
        loss_profiles=loss_profiles,
        max_samples_per_role=max_samples_per_role,
        max_samples_per_date_per_role=max_samples_per_date_per_role,
    )
    manifest_validation = validate_training_pack_manifest(TRAINING_PACK_MANIFEST)
    payload = {
        "schema_version": 1,
        "run_tag": RUN_TAG,
        "created_at": _now(),
        "stage": "alpha_v2_loss_alignment_scout",
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "anchor_run_tag": ANCHOR_RUN_TAG,
        "objective": "fixed alpha_v2 h256 hybrid one-variable loss/output alignment scout before multi-seed or execution bridge",
        "training_pack_manifest": str(TRAINING_PACK_MANIFEST),
        "training_pack_validation": manifest_validation,
        "dataset_id": DATASET_ID,
        "pool_view_id": POOL_VIEW_ID,
        "feature_profile": FEATURE_PROFILE,
        "feature_count": FEATURE_COUNT,
        "label_schema": LABEL_SCHEMA,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "output_profile": OUTPUT_PROFILE,
        "selection_profile": SELECTION_PROFILE,
        "loss_profiles": list(_parse_loss_profiles(loss_profiles)),
        "seeds": list(_parse_seeds(seeds)),
        "single_seed_scout_only": len(_parse_seeds(seeds)) == 1,
        "training_task_count": len(tasks),
        "max_samples_per_role": int(max_samples_per_role),
        "max_samples_per_date_per_role": int(max_samples_per_date_per_role),
        "training_tasks": tasks,
        "evaluation_requirements": {
            "personal_topk_v1_validation_selection": True,
            "same_candidate_test_reporting": True,
            "compare_against_anchor_run_tag": ANCHOR_RUN_TAG,
            "multi_seed_deferred": True,
            "score_backtest_bridge_deferred": True,
            "candidate_matrix_deferred": True,
        },
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "paper_live_broker_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        },
    }
    path = root / "qdp_alpha_v2_loss_alignment_task_list.json"
    _write_json(path, payload)
    return path


def _run_command(command: list[str], *, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr:
        proc = subprocess.run(command, cwd=str(PROJECT_ROOT), stdout=stdout, stderr=stderr, text=True, check=False)
    return {"returncode": int(proc.returncode), "stdout": str(stdout_path), "stderr": str(stderr_path)}


def run_training_tasks(
    *,
    output_root: str | Path | None = None,
    seeds: tuple[int, ...] | list[int] | str | None = DEFAULT_SEEDS,
    loss_profiles: tuple[str, ...] | list[str] | str | None = DEFAULT_LOSS_PROFILES,
    max_tasks: int = 0,
    skip_existing: bool = True,
    max_samples_per_role: int = 0,
    max_samples_per_date_per_role: int = 0,
) -> dict[str, Any]:
    if _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    root = _anchor_root(output_root)
    validation = validate_training_pack_manifest()
    if str(validation.get("status", "")) != "ok":
        payload = {
            "schema_version": 1,
            "status": "blocked",
            "blockers": ["invalid_or_missing_qdp_alpha_v2_training_pack_manifest", *list(validation.get("blockers", []) or [])],
            "training_pack_validation": validation,
            "updated_at": _now(),
        }
        _write_json(root / "qdp_alpha_v2_loss_alignment_run_summary.json", payload)
        return payload
    task_list_path = write_task_list(
        output_root=root,
        seeds=seeds,
        loss_profiles=loss_profiles,
        max_samples_per_role=max_samples_per_role,
        max_samples_per_date_per_role=max_samples_per_date_per_role,
        enforce_active_artifact_clean=False,
    )
    tasks = build_training_tasks(
        output_root=root,
        seeds=seeds,
        loss_profiles=loss_profiles,
        max_samples_per_role=max_samples_per_role,
        max_samples_per_date_per_role=max_samples_per_date_per_role,
    )
    completed: list[str] = []
    failed: list[str] = []
    results: list[dict[str, Any]] = []
    launched = 0
    for task in tasks:
        if int(max_tasks) > 0 and launched >= int(max_tasks):
            results.append({"tag": task["tag"], "status": "deferred_by_max_tasks"})
            continue
        summary_path = Path(str(task["study_dir"])) / "study_summary.json"
        if skip_existing and summary_path.exists():
            completed.append(str(task["tag"]))
            results.append({"tag": task["tag"], "status": "skipped_existing", "study_summary_json": str(summary_path)})
            continue
        launched += 1
        result = _run_command(
            list(task["command"]),
            stdout_path=Path(str(task["stdout"])),
            stderr_path=Path(str(task["stderr"])),
        )
        row = {"tag": task["tag"], "candidate_id": task["candidate_id"], "loss_profile": task["loss_profile"], **result}
        results.append(row)
        if int(result["returncode"]) == 0:
            completed.append(str(task["tag"]))
        else:
            failed.append(str(task["tag"]))
            break
    payload = {
        "schema_version": 1,
        "status": "completed" if not failed else "failed",
        "run_tag": RUN_TAG,
        "task_list_path": str(task_list_path),
        "completed_tags": completed,
        "failed_tags": failed,
        "launched_task_count": int(launched),
        "max_tasks": int(max_tasks),
        "loss_profiles": list(_parse_loss_profiles(loss_profiles)),
        "max_samples_per_role": int(max_samples_per_role),
        "max_samples_per_date_per_role": int(max_samples_per_date_per_role),
        "results": results,
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        },
        "updated_at": _now(),
    }
    _write_json(root / "qdp_alpha_v2_loss_alignment_run_summary.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QDP alpha_v2 fixed-hybrid loss alignment scout.")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--loss-profiles", default=",".join(DEFAULT_LOSS_PROFILES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--forecast-max-samples-per-role", type=int, default=0)
    parser.add_argument("--forecast-max-samples-per-date-per-role", type=int, default=0)
    parser.add_argument("--write-task-list", action="store_true")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    output_root = args.output_root or str(STUDIES_ROOT / RUN_TAG)
    actions: dict[str, Any] = {}
    if args.write_task_list:
        actions["task_list"] = str(
            write_task_list(
                output_root=output_root,
                seeds=args.seeds,
                loss_profiles=args.loss_profiles,
                max_samples_per_role=int(args.forecast_max_samples_per_role),
                max_samples_per_date_per_role=int(args.forecast_max_samples_per_date_per_role),
            )
        )
    if args.run_training:
        actions["training_run"] = run_training_tasks(
            output_root=output_root,
            seeds=args.seeds,
            loss_profiles=args.loss_profiles,
            max_tasks=int(args.max_tasks),
            skip_existing=bool(args.skip_existing),
            max_samples_per_role=int(args.forecast_max_samples_per_role),
            max_samples_per_date_per_role=int(args.forecast_max_samples_per_date_per_role),
        )
    if not actions:
        actions["task_list"] = str(
            write_task_list(
                output_root=output_root,
                seeds=args.seeds,
                loss_profiles=args.loss_profiles,
                max_samples_per_role=int(args.forecast_max_samples_per_role),
                max_samples_per_date_per_role=int(args.forecast_max_samples_per_date_per_role),
            )
        )
    if args.json:
        print(json.dumps(actions, ensure_ascii=False, indent=2, default=_json_default))
    else:
        for key, value in actions.items():
            print(f"{key}: {value}")
    training_run = actions.get("training_run")
    if isinstance(training_run, dict) and str(training_run.get("status", "")) in {"blocked", "failed"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
