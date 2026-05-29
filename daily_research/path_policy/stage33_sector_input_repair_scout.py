from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from daily_research.path_policy.output_aux_profile_comparison import (
    build_output_aux_profile_comparison,
    profile_aggregate_rows,
    write_output_aux_profile_comparison,
)
from daily_research.path_policy.stage_universe_scope import (
    FULL_ROLLING_LIQUID500_MIN_TRAIN_ROWS,
    FULL_ROLLING_LIQUID500_MIN_UNIVERSE_SIZE,
    FULL_ROLLING_LIQUID500_POOL_VIEW_KIND,
    FULL_ROLLING_LIQUID500_POOL_VIEW_NAME,
    FULL_ROLLING_LIQUID500_SCOPE,
    full_pool_task_metadata,
    full_rolling_liquid500_memmap_manifest,
    study_scope_from_summary_path,
)


PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDIES_ROOT = PROJECT_ROOT / "daily_research/output/path_policy/studies"
RUN_TAG = "mh_stage33_sector_input_repair_scout_20260529_01"
STAGE34_RUN_TAG = "mh_stage34_sector_input_confirmation_20260529_01"
STAGE35_RUN_TAG = "mh_stage35_sector_input_final_confirmation_20260529_01"
RESEARCH_PROGRAM = "alpha_multi_horizon_utility_policy_v1"
STUDY_FAMILY = "stage33_sector_input_repair_scout"
STAGE34_STUDY_FAMILY = "stage34_sector_input_confirmation"
STAGE35_STUDY_FAMILY = "stage35_sector_input_final_confirmation"
DATASET_ID = "policy_input_bundle__7c8f58d851bce8179e1e9e2d"
FULLGRID = "1,2,3,5,8,10,15,20,30"
SCOUT_SEED = 7
CONFIRMATION_SEED = 11
FINAL_SEED = 19
BASELINE_STAGE28_SEED7_TAG = "mh28_fullpool_target_norm_head_constraint_v1_fullgrid_seed7_20260527_01"
BASELINE_STAGE28_SEED11_TAG = "mh28_fullpool_target_norm_head_constraint_v1_fullgrid_seed11_20260527_01"
BASELINE_STAGE28_SEED19_TAG = "mh28_fullpool_target_norm_head_constraint_v1_fullgrid_seed19_20260527_01"
LOSS_PROFILE = "target_norm_head_constraint_v1"
OUTPUT_PROFILE = "decision_utility_v1"
RAW_FEATURE_PROFILE = "raw_kline_context_no_alpha_prior_v1"
SECTOR_FEATURE_PROFILE = "raw_kline_context_sector_v1"
FULL_ROLLING_LIQUID500_SECTOR_BOARD_VIEW_ID = "policy_sector_board_view__ed15b2873f544e9e9b24aae5"
SECTOR_BOARD_VIEW_KIND = "latest_static_snapshot"
BOUNDARY = "research-only / shadow-only; no active manifest, live/default, production root, paper, or broker integration"
STAGE33_SCOUT_SPECS = (
    {
        "candidate_id": "sector_gru_sequence_static_context",
        "feature_profile": SECTOR_FEATURE_PROFILE,
        "model_family": "gru_sequence_static_context",
        "sector_anchor": True,
    },
    {
        "candidate_id": "sector_stock_mixer_sequence",
        "feature_profile": SECTOR_FEATURE_PROFILE,
        "model_family": "stock_mixer_sequence",
    },
    {
        "candidate_id": "sector_sector_slot_mixer_sequence",
        "feature_profile": SECTOR_FEATURE_PROFILE,
        "model_family": "sector_slot_mixer_sequence",
    },
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _stage33_output_root(output_root: str | Path | None = None) -> Path:
    return Path(output_root) if output_root is not None else STUDIES_ROOT / RUN_TAG


def _baseline_dir(seed: int) -> Path:
    tag_by_seed = {
        SCOUT_SEED: BASELINE_STAGE28_SEED7_TAG,
        CONFIRMATION_SEED: BASELINE_STAGE28_SEED11_TAG,
        FINAL_SEED: BASELINE_STAGE28_SEED19_TAG,
    }
    return STUDIES_ROOT / tag_by_seed[int(seed)]


def _sector_anchor_tag(seed: int = SCOUT_SEED) -> str:
    return f"mh33_scout_sector_gru_sequence_static_context_seed{int(seed)}_20260529_01"


def _sector_anchor_manifest() -> Path:
    return STUDIES_ROOT / _sector_anchor_tag() / "forecast_dataset_manifest.json"


def _full_pool_base_args() -> list[str]:
    return [
        "--pool-name",
        FULL_ROLLING_LIQUID500_POOL_VIEW_NAME,
        "--pool-view-kind",
        FULL_ROLLING_LIQUID500_POOL_VIEW_KIND,
        "--pool-view-name",
        FULL_ROLLING_LIQUID500_POOL_VIEW_NAME,
        "--max-universe-size",
        "0",
    ]


def _training_command(
    *,
    tag: str,
    model_family: str,
    feature_profile: str,
    seed: int,
    epochs: int,
    min_epochs: int,
    patience: int,
    memmap_manifest: str | Path | None,
    slot_diagnostics: bool,
) -> list[str]:
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
        *_full_pool_base_args(),
        "--sector-board-view-id",
        FULL_ROLLING_LIQUID500_SECTOR_BOARD_VIEW_ID,
        "--sector-board-view-kind",
        SECTOR_BOARD_VIEW_KIND,
        "--forecast-dataset-mode",
        "memmap",
    ]
    if memmap_manifest is not None:
        command.extend(["--forecast-memmap-manifest", str(memmap_manifest)])
    command.extend(
        [
            "--forecast-train-start-year",
            "2019",
            "--forecast-train-end-year",
            "2022",
            "--forecast-validation-year",
            "2023",
            "--forecast-test-year",
            "2024",
            "--forecast-model-families",
            str(model_family),
            "--forecast-feature-profile",
            str(feature_profile),
            "--forecast-include-static-context",
            "--forecast-max-feature-columns",
            "192",
            "--forecast-cumulative-horizons",
            FULLGRID,
            "--forecast-horizon",
            "30",
            "--forecast-output-profile",
            OUTPUT_PROFILE,
            "--forecast-loss-profile",
            LOSS_PROFILE,
            "--forecast-selection-profile",
            "decision_utility",
            "--forecast-decision-cost-bps",
            "20",
            "--forecast-decision-hit-threshold-bps",
            "10",
            "--forecast-decision-drawdown-penalty",
            "0.10",
            "--forecast-seeds",
            str(seed),
            "--forecast-epochs",
            str(epochs),
            "--forecast-min-epochs",
            str(min_epochs),
            "--forecast-early-stop-patience",
            str(patience),
            "--forecast-checkpoint-every-n-epochs",
            "4",
            "--forecast-device",
            "cuda",
        ]
    )
    if slot_diagnostics:
        command.append("--forecast-slot-diagnostics")
    return command


def _task(
    *,
    root: Path,
    tag: str,
    candidate_id: str,
    model_family: str,
    feature_profile: str,
    seed: int,
    epochs: int,
    min_epochs: int,
    patience: int,
    memmap_manifest: str | Path | None,
    reuses_forecast_memmap_manifest: bool,
    slot_diagnostics: bool,
    study_family: str,
    run_tag: str,
) -> dict[str, Any]:
    study_dir = STUDIES_ROOT / tag
    return {
        "tag": tag,
        "candidate_id": candidate_id,
        "model_family": model_family,
        "feature_profile": feature_profile,
        "loss_profile": LOSS_PROFILE,
        "output_profile": OUTPUT_PROFILE,
        "seed": int(seed),
        "horizons": FULLGRID,
        "max_horizon": 30,
        "study_dir": str(study_dir),
        "stdout": str(root / f"{tag}_stdout.log"),
        "stderr": str(root / f"{tag}_stderr.log"),
        "command": _training_command(
            tag=tag,
            model_family=model_family,
            feature_profile=feature_profile,
            seed=int(seed),
            epochs=int(epochs),
            min_epochs=int(min_epochs),
            patience=int(patience),
            memmap_manifest=memmap_manifest,
            slot_diagnostics=slot_diagnostics,
        ),
        "research_program": RESEARCH_PROGRAM,
        "study_family": study_family,
        "run_tag": run_tag,
        "scope": "research_shadow_only",
        **full_pool_task_metadata(STUDIES_ROOT),
        "forecast_memmap_manifest": str(memmap_manifest or study_dir / "forecast_dataset_manifest.json"),
        "reuses_forecast_memmap_manifest": bool(reuses_forecast_memmap_manifest),
        "slot_diagnostics": bool(slot_diagnostics),
        "may_touch_active_manifest": False,
    }


def build_stage33_scout_tasks(*, output_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = _stage33_output_root(output_root)
    sector_manifest = _sector_anchor_manifest()
    tasks: list[dict[str, Any]] = []
    for spec in STAGE33_SCOUT_SPECS:
        candidate_id = str(spec["candidate_id"])
        model_family = str(spec["model_family"])
        tag = f"mh33_scout_{candidate_id}_seed{SCOUT_SEED}_20260529_01"
        sector_anchor = bool(spec.get("sector_anchor", False))
        memmap_manifest: Path | None = None if sector_anchor else sector_manifest
        tasks.append(
            _task(
                root=root,
                tag=tag,
                candidate_id=candidate_id,
                model_family=model_family,
                feature_profile=SECTOR_FEATURE_PROFILE,
                seed=SCOUT_SEED,
                epochs=16,
                min_epochs=6,
                patience=4,
                memmap_manifest=memmap_manifest,
                reuses_forecast_memmap_manifest=memmap_manifest is not None,
                slot_diagnostics=model_family == "sector_slot_mixer_sequence",
                study_family=STUDY_FAMILY,
                run_tag=RUN_TAG,
            )
        )
    return tasks


def write_stage33_task_list(output_root: str | Path | None = None) -> Path:
    root = _stage33_output_root(output_root)
    path = root / "stage33_task_list.json"
    tasks = build_stage33_scout_tasks(output_root=root)
    payload = {
        "schema_version": 1,
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "created_at": _now(),
        "baseline_stage28_seed7_tag": BASELINE_STAGE28_SEED7_TAG,
        "single_seed_scout_only": True,
        "evidence_grade_input_or_architecture_pass": False,
        "stage3e_max_candidates": 2,
        "stage3f_max_candidates": 1,
        "universe_scope": FULL_ROLLING_LIQUID500_SCOPE,
        "stage3d_dataset_gate": {
            "sector_context_feature_count_min": 1,
            "source_sector_board_view_id_required": True,
        },
        "training_task_count": len(tasks),
        "training_tasks": tasks,
        "boundary": BOUNDARY,
    }
    _write_json(path, payload)
    return path


def _training_progress_payload(*, status: str, completed: list[str], failed: list[str], run_tag: str, study_family: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "run_tag": run_tag,
        "research_program": RESEARCH_PROGRAM,
        "study_family": study_family,
        "completed_tags": completed,
        "failed_tags": failed,
        "updated_at": _now(),
    }


def _manifest_sector_context_count(manifest: dict[str, Any]) -> int:
    group_counts = manifest.get("feature_group_counts", {})
    if isinstance(group_counts, dict):
        try:
            return int(group_counts.get("sector_context", 0) or 0)
        except (TypeError, ValueError):
            return 0
    try:
        return int(manifest.get("sector_context_feature_count", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _summary_dataset_manifest(summary_path: Path) -> dict[str, Any]:
    summary = _read_json(summary_path)
    manifest = summary.get("dataset_manifest")
    return manifest if isinstance(manifest, dict) else {}


def _task_scope_mismatch(task: dict[str, Any]) -> dict[str, Any] | None:
    summary_path = Path(task["study_dir"]) / "study_summary.json"
    observed_scope = study_scope_from_summary_path(summary_path)
    manifest = _summary_dataset_manifest(summary_path)
    reasons: list[str] = []
    if observed_scope.get("universe_scope") != FULL_ROLLING_LIQUID500_SCOPE:
        reasons.append("universe_scope_mismatch")
    if int(observed_scope.get("universe_size", 0) or 0) < int(task.get("expected_min_universe_size", FULL_ROLLING_LIQUID500_MIN_UNIVERSE_SIZE) or 0):
        reasons.append("universe_size_too_small")
    if int(observed_scope.get("train_rows", 0) or 0) < int(task.get("expected_min_train_rows", FULL_ROLLING_LIQUID500_MIN_TRAIN_ROWS) or 0):
        reasons.append("train_rows_too_small")
    if str(observed_scope.get("pool_view_kind", "") or "") != FULL_ROLLING_LIQUID500_POOL_VIEW_KIND:
        reasons.append("pool_view_kind_mismatch")
    if str(observed_scope.get("pool_view_name", "") or "") != FULL_ROLLING_LIQUID500_POOL_VIEW_NAME:
        reasons.append("pool_view_name_mismatch")
    if str(manifest.get("feature_profile", "") or "") != str(task.get("feature_profile", "")):
        reasons.append("feature_profile_mismatch")
    if _manifest_sector_context_count(manifest) <= 0:
        reasons.append("missing_sector_context_features")
    if not str(manifest.get("source_sector_board_view_id", "") or "").strip():
        reasons.append("missing_sector_board_view_id")
    if not reasons:
        return None
    return {
        "tag": task.get("tag"),
        "study_summary_json": str(summary_path),
        "reasons": reasons,
        "expected_universe_scope": FULL_ROLLING_LIQUID500_SCOPE,
        "expected_feature_profile": task.get("feature_profile"),
        "observed_scope": observed_scope,
        "observed_feature_profile": manifest.get("feature_profile", ""),
        "observed_sector_context_feature_count": _manifest_sector_context_count(manifest),
        "observed_source_sector_board_view_id": str(manifest.get("source_sector_board_view_id", "") or ""),
    }


def _stage33_scope_mismatches(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in tasks:
        mismatch = _task_scope_mismatch(task)
        if mismatch is not None:
            out.append(mismatch)
    return out


def _validate_sector_anchor_ready(*, output_root: str | Path | None = None) -> dict[str, Any]:
    root = _stage33_output_root(output_root)
    anchor = next(task for task in build_stage33_scout_tasks(output_root=root) if task["candidate_id"] == "sector_gru_sequence_static_context")
    mismatch = _task_scope_mismatch(anchor)
    manifest_path = Path(anchor["study_dir"]) / "forecast_dataset_manifest.json"
    if mismatch is not None:
        payload = {"status": "blocked_sector_anchor_scope_mismatch", "mismatch": mismatch}
    elif not manifest_path.exists():
        payload = {"status": "blocked_missing_sector_anchor_manifest", "manifest": str(manifest_path)}
    else:
        payload = {"status": "ok", "manifest": str(manifest_path)}
    _write_json(root / "stage33_sector_manifest_check.json", payload)
    return payload


def _run_training_tasks(
    *,
    tasks: list[dict[str, Any]],
    output_root: Path,
    summary_name: str,
    progress_name: str,
    run_tag: str,
    study_family: str,
    skip_existing: bool = True,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    failed: list[str] = []
    results: list[dict[str, Any]] = []
    for task in tasks:
        if bool(task.get("reuses_forecast_memmap_manifest", False)):
            sector_status = _validate_sector_anchor_ready(output_root=output_root)
            if sector_status.get("status") != "ok":
                failed.append(task["tag"])
                results.append({"tag": task["tag"], **sector_status})
                break
        summary_path = Path(task["study_dir"]) / "study_summary.json"
        if summary_path.exists():
            mismatch = _task_scope_mismatch(task)
            if mismatch is not None:
                failed.append(task["tag"])
                results.append({"tag": task["tag"], "status": "blocked_existing_scope_mismatch", **mismatch})
                break
        if skip_existing and summary_path.exists():
            completed.append(task["tag"])
            results.append({"tag": task["tag"], "status": "skipped_existing", "study_summary_json": str(summary_path)})
            continue
        with Path(task["stdout"]).open("w", encoding="utf-8", errors="replace") as stdout, Path(task["stderr"]).open("w", encoding="utf-8", errors="replace") as stderr:
            proc = subprocess.Popen(task["command"], cwd=str(PROJECT_ROOT), stdout=stdout, stderr=stderr, text=True)
            returncode = proc.wait()
        row = {"tag": task["tag"], "returncode": int(returncode), "stdout": task["stdout"], "stderr": task["stderr"]}
        if returncode == 0:
            completed.append(task["tag"])
        else:
            failed.append(task["tag"])
        results.append(row)
        _write_json(
            output_root / progress_name,
            _training_progress_payload(
                status="running" if not failed else "blocked",
                completed=completed,
                failed=failed,
                run_tag=run_tag,
                study_family=study_family,
            ),
        )
        if returncode != 0:
            break
    final_status = "completed" if not failed and len(completed) == len(tasks) else "blocked"
    payload = {
        "schema_version": 1,
        "status": final_status,
        "run_tag": run_tag,
        "research_program": RESEARCH_PROGRAM,
        "study_family": study_family,
        "completed_tags": completed,
        "failed_tags": failed,
        "results": results,
        "updated_at": _now(),
    }
    _write_json(
        output_root / progress_name,
        _training_progress_payload(
            status=final_status,
            completed=completed,
            failed=failed,
            run_tag=run_tag,
            study_family=study_family,
        ),
    )
    _write_json(output_root / summary_name, payload)
    return payload


def run_stage33_scout(*, output_root: str | Path | None = None, skip_existing: bool = True) -> dict[str, Any]:
    root = _stage33_output_root(output_root)
    return _run_training_tasks(
        tasks=build_stage33_scout_tasks(output_root=root),
        output_root=root,
        summary_name="stage33_training_summary.json",
        progress_name="stage33_progress.json",
        run_tag=RUN_TAG,
        study_family=STUDY_FAMILY,
        skip_existing=skip_existing,
    )


def _float(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if row is None:
        return float(default)
    value = row.get(key, default)
    if value is None or value == "":
        value = default
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int(row: dict[str, Any] | None, key: str, default: int = 0) -> int:
    if row is None:
        return int(default)
    value = row.get(key, default)
    if value is None or value == "":
        value = default
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), float(lo)), float(hi))


def _ratio(value: float, baseline: float) -> float:
    if baseline <= 0.0:
        return 0.0
    return _clamp(value / baseline, 0.0, 2.0)


def _reduction(value: float, baseline: float) -> float:
    if baseline <= 0.0:
        return 0.0
    return _clamp((baseline - value) / baseline, -0.5, 0.5)


def _gain(value: float, baseline: float) -> float:
    if baseline <= 0.0:
        return 0.0
    return _clamp((value - baseline) / baseline, -0.5, 0.5)


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("model_family", "") or ""), str(row.get("feature_profile", "") or ""))


def _aggregate_maps(report: dict[str, Any]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    validation: dict[tuple[str, str], dict[str, Any]] = {}
    test: dict[tuple[str, str], dict[str, Any]] = {}
    for row in profile_aggregate_rows(report):
        if row.get("score_name") != "pred_decision_score":
            continue
        if row.get("role") == "validation":
            validation[_row_key(row)] = row
        elif row.get("role") == "test":
            test[_row_key(row)] = row
    return validation, test


def _baseline_key() -> tuple[str, str]:
    return ("gru_sequence_static_context", RAW_FEATURE_PROFILE)


def _stage3d_row(*, task: dict[str, Any], validation_row: dict[str, Any], test_row: dict[str, Any] | None, baseline: dict[str, Any]) -> dict[str, Any]:
    rank_ratio = _ratio(_float(validation_row, "rank_ic_mean"), _float(baseline, "rank_ic_mean"))
    spread_ratio = _ratio(_float(validation_row, "spread_mean"), _float(baseline, "spread_mean"))
    hit_ratio = _ratio(_float(validation_row, "hit_lift_mean"), _float(baseline, "hit_lift_mean"))
    concentration_reduction = _reduction(
        _float(validation_row, "thirty_d_concentration_mean"),
        _float(baseline, "thirty_d_concentration_mean"),
    )
    gap_reduction = _reduction(
        _float(validation_row, "pred_future_horizon_gap_mean"),
        _float(baseline, "pred_future_horizon_gap_mean"),
    )
    composite = 0.45 * rank_ratio + 0.25 * spread_ratio + 0.15 * hit_ratio + 0.10 * concentration_reduction + 0.05 * gap_reduction
    validation_gate = bool(
        _float(validation_row, "rank_ic_mean") > 0.0
        and _float(validation_row, "spread_mean") > 0.0
        and _float(validation_row, "hit_lift_mean") > 0.0
        and _float(validation_row, "monthly_positive_rate_mean") >= 0.75
        and _int(validation_row, "negative_month_count_max", 99) <= 2
        and _float(validation_row, "thirty_d_concentration_mean", 1.0) <= 0.85
    )
    test_veto = not bool(
        _float(test_row, "rank_ic_mean") > 0.0
        and _float(test_row, "spread_mean") > 0.0
        and _float(test_row, "hit_lift_mean") > 0.0
    )
    return {
        "tag": task.get("tag"),
        "study_dir": task.get("study_dir"),
        "model_family": task.get("model_family"),
        "feature_profile": task.get("feature_profile"),
        "candidate_id": task.get("candidate_id"),
        "seed": task.get("seed"),
        "stage3d_composite": composite,
        "rank_ic_ratio": rank_ratio,
        "spread_ratio": spread_ratio,
        "hit_lift_ratio": hit_ratio,
        "concentration_reduction_ratio": concentration_reduction,
        "pred_future_gap_reduction_ratio": gap_reduction,
        "stage3d_hard_gate_pass": validation_gate,
        "test_veto": test_veto,
        "stage3e_eligible": bool(validation_gate and not test_veto),
        "validation_rank_ic_mean": _float(validation_row, "rank_ic_mean"),
        "validation_spread_mean": _float(validation_row, "spread_mean"),
        "validation_hit_lift_mean": _float(validation_row, "hit_lift_mean"),
        "validation_monthly_positive_rate_mean": _float(validation_row, "monthly_positive_rate_mean"),
        "validation_negative_month_count_max": _int(validation_row, "negative_month_count_max", 99),
        "validation_thirty_d_concentration_mean": _float(validation_row, "thirty_d_concentration_mean"),
        "validation_pred_future_horizon_gap_mean": _float(validation_row, "pred_future_horizon_gap_mean"),
        "test_rank_ic_mean": _float(test_row, "rank_ic_mean"),
        "test_spread_mean": _float(test_row, "spread_mean"),
        "test_hit_lift_mean": _float(test_row, "hit_lift_mean"),
    }


def _select_stage3e_candidates(leaderboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in leaderboard if bool(row.get("stage3e_eligible"))]
    strong = [row for row in eligible if float(row.get("stage3d_composite", 0.0) or 0.0) >= 1.02]
    if strong:
        return strong[:2]
    fallback = [
        row
        for row in eligible
        if float(row.get("stage3d_composite", 0.0) or 0.0) >= 0.97
        and (
            float(row.get("concentration_reduction_ratio", 0.0) or 0.0) >= 0.05
            or float(row.get("pred_future_gap_reduction_ratio", 0.0) or 0.0) >= 0.05
        )
    ]
    return fallback[:1]


def _blocked_scope_payload(root: Path, *, run_tag: str, study_family: str, summary_name: str, mismatches: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "status": "blocked_scope_mismatch",
        "run_tag": run_tag,
        "research_program": RESEARCH_PROGRAM,
        "study_family": study_family,
        "scope_mismatches": mismatches,
        "evidence_grade_input_or_architecture_pass": False,
        "stage3e_candidates": [],
        "stage3f_candidates": [],
        "updated_at": _now(),
    }
    _write_json(root / summary_name, payload)
    return payload


def _write_leaderboard(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _stage33_verdict(payload: dict[str, Any]) -> str:
    lines = [
        "# Alpha Multi-Horizon Stage 3D Sector Input Repair Scout",
        "",
        f"- status: `{payload.get('status')}`",
        f"- run_tag: `{RUN_TAG}`",
        f"- research_program: `{RESEARCH_PROGRAM}`",
        f"- study_family: `{STUDY_FAMILY}`",
        "- evidence_grade_input_or_architecture_pass: `False`",
        f"- stage3e_candidate_count: `{len(payload.get('stage3e_candidates', []))}`",
        f"- boundary: {BOUNDARY}.",
        "",
        "## Sector Scout Leaderboard",
    ]
    for row in payload.get("sector_scout_leaderboard", []):
        lines.append(
            f"- `{row.get('candidate_id')}`: composite=`{float(row.get('stage3d_composite', 0.0) or 0.0):.6f}`, "
            f"validation_gate=`{bool(row.get('stage3d_hard_gate_pass'))}`, test_veto=`{bool(row.get('test_veto'))}`, "
            f"stage3e_eligible=`{bool(row.get('stage3e_eligible'))}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Stage 3D is single-seed scout evidence only; it cannot upgrade architecture or input evidence by itself.",
            "- Stage 3E seed11 confirmation is allowed only for the selected sector scout candidates.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_stage33_comparison(output_root: str | Path | None = None) -> dict[str, Any]:
    root = _stage33_output_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    tasks = build_stage33_scout_tasks(output_root=root)
    mismatches = _stage33_scope_mismatches(tasks)
    if mismatches:
        return _blocked_scope_payload(root, run_tag=RUN_TAG, study_family=STUDY_FAMILY, summary_name="stage33_comparison_summary.json", mismatches=mismatches)

    study_dirs = [str(_baseline_dir(SCOUT_SEED)), *[task["study_dir"] for task in tasks]]
    report = build_output_aux_profile_comparison(study_dirs, run_tag=RUN_TAG)
    paths = write_output_aux_profile_comparison(report, root / "stage33_comparison")
    validation_rows, test_rows = _aggregate_maps(report)
    baseline = validation_rows.get(_baseline_key())
    if baseline is None:
        return _blocked_scope_payload(
            root,
            run_tag=RUN_TAG,
            study_family=STUDY_FAMILY,
            summary_name="stage33_comparison_summary.json",
            mismatches=[{"tag": BASELINE_STAGE28_SEED7_TAG, "reasons": ["missing_baseline_validation_row"]}],
        )
    leaderboard = [
        _stage3d_row(
            task=task,
            validation_row=validation_rows[_row_key(task)],
            test_row=test_rows.get(_row_key(task)),
            baseline=baseline,
        )
        for task in tasks
        if _row_key(task) in validation_rows
    ]
    leaderboard.sort(key=lambda item: float(item.get("stage3d_composite", 0.0) or 0.0), reverse=True)
    stage3e_candidates = _select_stage3e_candidates(leaderboard)
    leaderboard_path = root / "stage33_sector_scout_leaderboard.csv"
    _write_leaderboard(leaderboard_path, leaderboard)
    if stage3e_candidates:
        write_stage34_confirmation_task_list(stage3e_candidates, output_root=root)
    verdict_path = root / "stage33_research_verdict.md"
    payload = {
        "schema_version": 1,
        "status": report.get("status"),
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "baseline_stage28_seed7_tag": BASELINE_STAGE28_SEED7_TAG,
        "single_seed_scout_only": True,
        "evidence_grade_input_or_architecture_pass": False,
        "scope_mismatches": [],
        "compared_task_tags": [task.get("tag") for task in tasks],
        "sector_scout_leaderboard": leaderboard,
        "stage3e_candidates": stage3e_candidates,
        "comparison_paths": {key: str(path) for key, path in paths.items()},
        "stage33_sector_scout_leaderboard_csv": str(leaderboard_path),
        "stage33_research_verdict_md": str(verdict_path),
        "updated_at": _now(),
    }
    verdict_path.write_text(_stage33_verdict(payload), encoding="utf-8")
    _write_json(root / "stage33_comparison_summary.json", payload)
    return payload


def _candidate_task_id(candidate: dict[str, Any]) -> str:
    return f"sector_{str(candidate.get('model_family', '')).replace('_', '-')}"


def build_stage34_confirmation_tasks(candidates: list[dict[str, Any]], *, output_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = _stage33_output_root(output_root)
    tasks: list[dict[str, Any]] = []
    for candidate in candidates[:2]:
        model_family = str(candidate["model_family"])
        candidate_id = _candidate_task_id(candidate)
        tag = f"mh34_confirm_{candidate_id}_seed{CONFIRMATION_SEED}_20260529_01"
        task = _task(
            root=root,
            tag=tag,
            candidate_id=candidate_id,
            model_family=model_family,
            feature_profile=SECTOR_FEATURE_PROFILE,
            seed=CONFIRMATION_SEED,
            epochs=24,
            min_epochs=8,
            patience=6,
            memmap_manifest=_sector_anchor_manifest(),
            reuses_forecast_memmap_manifest=True,
            slot_diagnostics=model_family == "sector_slot_mixer_sequence",
            study_family=STAGE34_STUDY_FAMILY,
            run_tag=STAGE34_RUN_TAG,
        )
        task["scout_tag"] = candidate.get("tag", "")
        task["scout_study_dir"] = candidate.get("study_dir", "")
        tasks.append(task)
    return tasks


def write_stage34_confirmation_task_list(candidates: list[dict[str, Any]], *, output_root: str | Path | None = None) -> Path:
    root = _stage33_output_root(output_root)
    path = root / "stage34_confirmation_task_list.json"
    tasks = build_stage34_confirmation_tasks(candidates, output_root=root)
    payload = {
        "schema_version": 1,
        "run_tag": STAGE34_RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STAGE34_STUDY_FAMILY,
        "created_at": _now(),
        "stage3e_candidates": candidates[:2],
        "training_task_count": len(tasks),
        "training_tasks": tasks,
        "boundary": BOUNDARY,
    }
    _write_json(path, payload)
    return path


def _load_stage34_candidates(output_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = _stage33_output_root(output_root)
    payload = _read_json(root / "stage34_confirmation_task_list.json")
    candidates = payload.get("stage3e_candidates", [])
    return candidates if isinstance(candidates, list) else []


def run_stage34_confirmation(*, output_root: str | Path | None = None, skip_existing: bool = True) -> dict[str, Any]:
    root = _stage33_output_root(output_root)
    candidates = _load_stage34_candidates(root)
    tasks = build_stage34_confirmation_tasks(candidates, output_root=root)
    return _run_training_tasks(
        tasks=tasks,
        output_root=root,
        summary_name="stage34_confirmation_summary.json",
        progress_name="stage34_progress.json",
        run_tag=STAGE34_RUN_TAG,
        study_family=STAGE34_STUDY_FAMILY,
        skip_existing=skip_existing,
    )


def _stage34_candidate_rows(*, candidates: list[dict[str, Any]], validation_rows: dict[tuple[str, str], dict[str, Any]], test_rows: dict[tuple[str, str], dict[str, Any]], baseline_test: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        key = (str(candidate.get("model_family", "")), str(candidate.get("feature_profile", "")))
        validation = validation_rows.get(key)
        test = test_rows.get(key)
        if validation is None or test is None:
            continue
        validation_positive = bool(
            _float(validation, "rank_ic_min") > 0.0
            and _float(validation, "spread_min") > 0.0
            and _float(validation, "hit_lift_min") > 0.0
        )
        test_positive = bool(
            _float(test, "rank_ic_min") > 0.0
            and _float(test, "spread_min") > 0.0
            and _float(test, "hit_lift_min") > 0.0
        )
        monthly_ok = bool(_float(test, "monthly_positive_rate_mean") >= 0.75 and _int(test, "negative_month_count_max", 99) <= 2)
        concentration_reduction = _reduction(_float(test, "thirty_d_concentration_mean"), _float(baseline_test, "thirty_d_concentration_mean"))
        gap_reduction = _reduction(_float(test, "pred_future_horizon_gap_mean"), _float(baseline_test, "pred_future_horizon_gap_mean"))
        rank_gain = _gain(_float(test, "rank_ic_mean"), _float(baseline_test, "rank_ic_mean"))
        spread_gain = _gain(_float(test, "spread_mean"), _float(baseline_test, "spread_mean"))
        hit_gain = _gain(_float(test, "hit_lift_mean"), _float(baseline_test, "hit_lift_mean"))
        advantage = max(rank_gain, spread_gain, hit_gain, concentration_reduction, gap_reduction)
        concentration_ok = _float(test, "thirty_d_concentration_mean", 1.0) <= _float(baseline_test, "thirty_d_concentration_mean", 0.0)
        gap_ok = _float(test, "pred_future_horizon_gap_mean", 1.0e9) <= _float(baseline_test, "pred_future_horizon_gap_mean", 0.0)
        pass_gate = bool(validation_positive and test_positive and monthly_ok and concentration_ok and gap_ok and advantage >= 0.05)
        rows.append(
            {
                **candidate,
                "stage34_pass": pass_gate,
                "stage34_score": advantage,
                "test_rank_ic_mean": _float(test, "rank_ic_mean"),
                "test_spread_mean": _float(test, "spread_mean"),
                "test_hit_lift_mean": _float(test, "hit_lift_mean"),
                "test_monthly_positive_rate_mean": _float(test, "monthly_positive_rate_mean"),
                "test_negative_month_count_max": _int(test, "negative_month_count_max", 99),
                "rank_gain_ratio": rank_gain,
                "spread_gain_ratio": spread_gain,
                "hit_gain_ratio": hit_gain,
                "concentration_reduction_ratio": concentration_reduction,
                "pred_future_gap_reduction_ratio": gap_reduction,
            }
        )
    rows.sort(key=lambda item: float(item.get("stage34_score", 0.0) or 0.0), reverse=True)
    return rows


def write_stage35_final_task_list(candidates: list[dict[str, Any]], *, output_root: str | Path | None = None) -> Path:
    root = _stage33_output_root(output_root)
    path = root / "stage35_final_task_list.json"
    tasks = build_stage35_final_tasks(candidates, output_root=root)
    payload = {
        "schema_version": 1,
        "run_tag": STAGE35_RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STAGE35_STUDY_FAMILY,
        "created_at": _now(),
        "stage3f_candidates": candidates[:1],
        "training_task_count": len(tasks),
        "training_tasks": tasks,
        "boundary": BOUNDARY,
    }
    _write_json(path, payload)
    return path


def run_stage34_comparison(output_root: str | Path | None = None) -> dict[str, Any]:
    root = _stage33_output_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    candidates = _load_stage34_candidates(root)
    tasks = build_stage34_confirmation_tasks(candidates, output_root=root)
    mismatches = _stage33_scope_mismatches(tasks)
    if mismatches:
        return _blocked_scope_payload(root, run_tag=STAGE34_RUN_TAG, study_family=STAGE34_STUDY_FAMILY, summary_name="stage34_confirmation_summary.json", mismatches=mismatches)
    candidate_seed7_dirs = [candidate.get("study_dir") for candidate in candidates if candidate.get("study_dir")]
    study_dirs = [str(_baseline_dir(SCOUT_SEED)), str(_baseline_dir(CONFIRMATION_SEED)), *candidate_seed7_dirs, *[task["study_dir"] for task in tasks]]
    report = build_output_aux_profile_comparison(study_dirs, run_tag=STAGE34_RUN_TAG)
    paths = write_output_aux_profile_comparison(report, root / "stage34_comparison")
    validation_rows, test_rows = _aggregate_maps(report)
    baseline_test = test_rows.get(_baseline_key())
    if baseline_test is None:
        return _blocked_scope_payload(
            root,
            run_tag=STAGE34_RUN_TAG,
            study_family=STAGE34_STUDY_FAMILY,
            summary_name="stage34_confirmation_summary.json",
            mismatches=[{"tag": "stage28_seed7_seed11_baseline", "reasons": ["missing_baseline_test_row"]}],
        )
    leaderboard = _stage34_candidate_rows(candidates=candidates, validation_rows=validation_rows, test_rows=test_rows, baseline_test=baseline_test)
    stage3f_candidates = [row for row in leaderboard if bool(row.get("stage34_pass"))][:1]
    leaderboard_path = root / "stage34_candidate_leaderboard.csv"
    _write_leaderboard(leaderboard_path, leaderboard)
    if stage3f_candidates:
        write_stage35_final_task_list(stage3f_candidates, output_root=root)
    payload = {
        "schema_version": 1,
        "status": report.get("status"),
        "run_tag": STAGE34_RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STAGE34_STUDY_FAMILY,
        "stage34_leaderboard": leaderboard,
        "stage3f_candidates": stage3f_candidates,
        "comparison_paths": {key: str(path) for key, path in paths.items()},
        "stage34_candidate_leaderboard_csv": str(leaderboard_path),
        "updated_at": _now(),
    }
    _write_json(root / "stage34_confirmation_summary.json", payload)
    verdict_path = root / "stage34_research_verdict.md"
    verdict_path.write_text(
        "# Alpha Multi-Horizon Stage 3E Sector Confirmation\n\n"
        f"- status: `{payload.get('status')}`\n"
        f"- stage3f_candidate_count: `{len(stage3f_candidates)}`\n"
        f"- boundary: {BOUNDARY}.\n",
        encoding="utf-8",
    )
    return payload


def _load_stage35_candidates(output_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = _stage33_output_root(output_root)
    payload = _read_json(root / "stage35_final_task_list.json")
    candidates = payload.get("stage3f_candidates", [])
    return candidates if isinstance(candidates, list) else []


def build_stage35_final_tasks(candidates: list[dict[str, Any]], *, output_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = _stage33_output_root(output_root)
    tasks: list[dict[str, Any]] = []
    for candidate in candidates[:1]:
        model_family = str(candidate["model_family"])
        candidate_id = _candidate_task_id(candidate)
        tag = f"mh35_final_{candidate_id}_seed{FINAL_SEED}_20260529_01"
        tasks.append(
            _task(
                root=root,
                tag=tag,
                candidate_id=candidate_id,
                model_family=model_family,
                feature_profile=SECTOR_FEATURE_PROFILE,
                seed=FINAL_SEED,
                epochs=24,
                min_epochs=8,
                patience=6,
                memmap_manifest=_sector_anchor_manifest(),
                reuses_forecast_memmap_manifest=True,
                slot_diagnostics=model_family == "sector_slot_mixer_sequence",
                study_family=STAGE35_STUDY_FAMILY,
                run_tag=STAGE35_RUN_TAG,
            )
        )
    return tasks


def run_stage35_final(*, output_root: str | Path | None = None, skip_existing: bool = True) -> dict[str, Any]:
    root = _stage33_output_root(output_root)
    candidates = _load_stage35_candidates(root)
    tasks = build_stage35_final_tasks(candidates, output_root=root)
    return _run_training_tasks(
        tasks=tasks,
        output_root=root,
        summary_name="stage35_final_training_summary.json",
        progress_name="stage35_progress.json",
        run_tag=STAGE35_RUN_TAG,
        study_family=STAGE35_STUDY_FAMILY,
        skip_existing=skip_existing,
    )


def _stage35_candidate_rows(*, candidates: list[dict[str, Any]], validation_rows: dict[tuple[str, str], dict[str, Any]], test_rows: dict[tuple[str, str], dict[str, Any]], baseline_test: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates[:1]:
        key = (str(candidate.get("model_family", "")), str(candidate.get("feature_profile", "")))
        validation = validation_rows.get(key)
        test = test_rows.get(key)
        if validation is None or test is None:
            continue
        validation_positive = bool(
            _float(validation, "rank_ic_min") > 0.0
            and _float(validation, "spread_min") > 0.0
            and _float(validation, "hit_lift_min") > 0.0
        )
        test_positive = bool(
            _float(test, "rank_ic_min") > 0.0
            and _float(test, "spread_min") > 0.0
            and _float(test, "hit_lift_min") > 0.0
        )
        monthly_ok = bool(_float(test, "monthly_positive_rate_mean") >= 0.75 and _int(test, "negative_month_count_max", 99) <= 2)
        long_horizon_ok = _float(test, "long_horizon_share_mean", 1.0) <= _float(baseline_test, "long_horizon_share_mean", 0.0)
        concentration_ok = _float(test, "thirty_d_concentration_mean", 1.0) <= _float(baseline_test, "thirty_d_concentration_mean", 0.0)
        gap_ok = _float(test, "pred_future_horizon_gap_mean", 1.0e9) <= _float(baseline_test, "pred_future_horizon_gap_mean", 0.0)
        rank_gain = _gain(_float(test, "rank_ic_mean"), _float(baseline_test, "rank_ic_mean"))
        spread_gain = _gain(_float(test, "spread_mean"), _float(baseline_test, "spread_mean"))
        hit_gain = _gain(_float(test, "hit_lift_mean"), _float(baseline_test, "hit_lift_mean"))
        concentration_reduction = _reduction(_float(test, "thirty_d_concentration_mean"), _float(baseline_test, "thirty_d_concentration_mean"))
        gap_reduction = _reduction(_float(test, "pred_future_horizon_gap_mean"), _float(baseline_test, "pred_future_horizon_gap_mean"))
        clear_advantage = max(rank_gain, spread_gain, hit_gain, concentration_reduction, gap_reduction) >= 0.05
        full_gate = bool(validation_positive and test_positive and monthly_ok and long_horizon_ok and concentration_ok and gap_ok)
        rows.append(
            {
                **candidate,
                "stage35_full_gate_pass": full_gate,
                "clear_advantage": clear_advantage,
                "input_or_architecture_upgrade_allowed": bool(full_gate and clear_advantage),
                "rank_gain_ratio": rank_gain,
                "spread_gain_ratio": spread_gain,
                "hit_gain_ratio": hit_gain,
                "concentration_reduction_ratio": concentration_reduction,
                "pred_future_gap_reduction_ratio": gap_reduction,
                "validation_rank_ic_mean": _float(validation, "rank_ic_mean"),
                "validation_rank_ic_min": _float(validation, "rank_ic_min"),
                "validation_spread_mean": _float(validation, "spread_mean"),
                "validation_spread_min": _float(validation, "spread_min"),
                "validation_hit_lift_mean": _float(validation, "hit_lift_mean"),
                "validation_hit_lift_min": _float(validation, "hit_lift_min"),
                "validation_monthly_positive_rate_mean": _float(validation, "monthly_positive_rate_mean"),
                "validation_negative_month_count_max": _int(validation, "negative_month_count_max", 99),
                "validation_long_horizon_share_mean": _float(validation, "long_horizon_share_mean"),
                "validation_thirty_d_concentration_mean": _float(validation, "thirty_d_concentration_mean"),
                "validation_pred_future_horizon_gap_mean": _float(validation, "pred_future_horizon_gap_mean"),
                "test_rank_ic_mean": _float(test, "rank_ic_mean"),
                "test_rank_ic_min": _float(test, "rank_ic_min"),
                "test_spread_mean": _float(test, "spread_mean"),
                "test_spread_min": _float(test, "spread_min"),
                "test_hit_lift_mean": _float(test, "hit_lift_mean"),
                "test_hit_lift_min": _float(test, "hit_lift_min"),
                "test_monthly_positive_rate_mean": _float(test, "monthly_positive_rate_mean"),
                "test_negative_month_count_max": _int(test, "negative_month_count_max", 99),
                "test_long_horizon_share_mean": _float(test, "long_horizon_share_mean"),
                "test_thirty_d_concentration_mean": _float(test, "thirty_d_concentration_mean"),
                "test_pred_future_horizon_gap_mean": _float(test, "pred_future_horizon_gap_mean"),
            }
        )
    rows.sort(key=lambda item: float(item.get("input_or_architecture_upgrade_allowed", False)), reverse=True)
    return rows


def _aggregate_snapshot(row: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "model_family": str((row or {}).get("model_family", "") or ""),
        "feature_profile": str((row or {}).get("feature_profile", "") or ""),
        "seed_count": _int(row, "seed_count"),
        "rank_ic_mean": _float(row, "rank_ic_mean"),
        "rank_ic_min": _float(row, "rank_ic_min"),
        "spread_mean": _float(row, "spread_mean"),
        "spread_min": _float(row, "spread_min"),
        "hit_lift_mean": _float(row, "hit_lift_mean"),
        "hit_lift_min": _float(row, "hit_lift_min"),
        "monthly_positive_rate_mean": _float(row, "monthly_positive_rate_mean"),
        "negative_month_count_max": _int(row, "negative_month_count_max", 99),
        "long_horizon_share_mean": _float(row, "long_horizon_share_mean"),
        "thirty_d_concentration_mean": _float(row, "thirty_d_concentration_mean"),
        "pred_future_horizon_gap_mean": _float(row, "pred_future_horizon_gap_mean"),
    }


def _metric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "rank_ic_mean_delta": _float(candidate, "rank_ic_mean") - _float(baseline, "rank_ic_mean"),
        "spread_mean_delta": _float(candidate, "spread_mean") - _float(baseline, "spread_mean"),
        "hit_lift_mean_delta": _float(candidate, "hit_lift_mean") - _float(baseline, "hit_lift_mean"),
        "thirty_d_concentration_mean_delta": _float(candidate, "thirty_d_concentration_mean") - _float(baseline, "thirty_d_concentration_mean"),
        "pred_future_horizon_gap_mean_delta": _float(candidate, "pred_future_horizon_gap_mean") - _float(baseline, "pred_future_horizon_gap_mean"),
    }


def run_stage35_comparison(output_root: str | Path | None = None) -> dict[str, Any]:
    root = _stage33_output_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    candidates = _load_stage35_candidates(root)
    stage34_tasks = build_stage34_confirmation_tasks(candidates, output_root=root)
    stage35_tasks = build_stage35_final_tasks(candidates, output_root=root)
    mismatches = _stage33_scope_mismatches([*stage34_tasks, *stage35_tasks])
    if mismatches:
        return _blocked_scope_payload(
            root,
            run_tag=STAGE35_RUN_TAG,
            study_family=STAGE35_STUDY_FAMILY,
            summary_name="stage35_final_sector_confirmation_summary.json",
            mismatches=mismatches,
        )
    candidate_seed7_dirs = [candidate.get("study_dir") for candidate in candidates if candidate.get("study_dir")]
    study_dirs = [
        str(_baseline_dir(SCOUT_SEED)),
        str(_baseline_dir(CONFIRMATION_SEED)),
        str(_baseline_dir(FINAL_SEED)),
        *candidate_seed7_dirs,
        *[task["study_dir"] for task in stage34_tasks],
        *[task["study_dir"] for task in stage35_tasks],
    ]
    report = build_output_aux_profile_comparison(study_dirs, run_tag=STAGE35_RUN_TAG)
    paths = write_output_aux_profile_comparison(report, root / "stage35_comparison")
    validation_rows, test_rows = _aggregate_maps(report)
    baseline_validation = validation_rows.get(_baseline_key())
    baseline_test = test_rows.get(_baseline_key())
    if baseline_test is None:
        return _blocked_scope_payload(
            root,
            run_tag=STAGE35_RUN_TAG,
            study_family=STAGE35_STUDY_FAMILY,
            summary_name="stage35_final_sector_confirmation_summary.json",
            mismatches=[{"tag": "stage28_seed7_seed11_seed19_baseline", "reasons": ["missing_baseline_test_row"]}],
        )
    leaderboard = _stage35_candidate_rows(candidates=candidates, validation_rows=validation_rows, test_rows=test_rows, baseline_test=baseline_test)
    input_or_architecture_upgrade_allowed = any(bool(row.get("input_or_architecture_upgrade_allowed")) for row in leaderboard)
    leaderboard_path = root / "stage35_final_candidate_leaderboard.csv"
    _write_leaderboard(leaderboard_path, leaderboard)
    verdict_path = root / "stage35_research_verdict.md"
    candidate_key = _row_key(leaderboard[0]) if leaderboard else ("", "")
    candidate_validation = validation_rows.get(candidate_key)
    candidate_test = test_rows.get(candidate_key)
    baseline_validation_snapshot = _aggregate_snapshot(baseline_validation)
    baseline_test_snapshot = _aggregate_snapshot(baseline_test)
    candidate_validation_snapshot = _aggregate_snapshot(candidate_validation)
    candidate_test_snapshot = _aggregate_snapshot(candidate_test)
    payload = {
        "schema_version": 1,
        "status": report.get("status"),
        "run_tag": STAGE35_RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STAGE35_STUDY_FAMILY,
        "baseline_stage28_tags": [BASELINE_STAGE28_SEED7_TAG, BASELINE_STAGE28_SEED11_TAG, BASELINE_STAGE28_SEED19_TAG],
        "final_candidate_leaderboard": leaderboard,
        "baseline_validation_aggregate": baseline_validation_snapshot,
        "candidate_validation_aggregate": candidate_validation_snapshot,
        "baseline_test_aggregate": baseline_test_snapshot,
        "candidate_test_aggregate": candidate_test_snapshot,
        "test_metric_delta_candidate_minus_baseline": _metric_delta(candidate_test_snapshot, baseline_test_snapshot),
        "input_or_architecture_upgrade_allowed": input_or_architecture_upgrade_allowed,
        "evidence_grade_input_or_architecture_pass": input_or_architecture_upgrade_allowed,
        "comparison_paths": {key: str(path) for key, path in paths.items()},
        "stage35_final_candidate_leaderboard_csv": str(leaderboard_path),
        "stage35_research_verdict_md": str(verdict_path),
        "boundary": BOUNDARY,
        "updated_at": _now(),
    }
    verdict_lines = [
        "# Alpha Multi-Horizon Stage 3F Final Sector Confirmation",
        "",
        f"- status: `{payload.get('status')}`",
        f"- input_or_architecture_upgrade_allowed: `{input_or_architecture_upgrade_allowed}`",
        f"- boundary: {BOUNDARY}.",
    ]
    for row in leaderboard:
        verdict_lines.append(
            f"- `{row.get('model_family')}` + `{row.get('feature_profile')}`: "
            f"full_gate=`{bool(row.get('stage35_full_gate_pass'))}`, "
            f"clear_advantage=`{bool(row.get('clear_advantage'))}`, "
            f"upgrade_allowed=`{bool(row.get('input_or_architecture_upgrade_allowed'))}`."
        )
    verdict_path.write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")
    _write_json(root / "stage35_final_sector_confirmation_summary.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 3D sector input repair scout driver.")
    parser.add_argument("--output-root", default=str(STUDIES_ROOT / RUN_TAG))
    parser.add_argument("--write-task-list", action="store_true")
    parser.add_argument("--run-scout", action="store_true")
    parser.add_argument("--run-comparison", action="store_true")
    parser.add_argument("--run-confirmation", action="store_true")
    parser.add_argument("--run-confirmation-comparison", action="store_true")
    parser.add_argument("--run-final", action="store_true")
    parser.add_argument("--run-final-comparison", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    outputs: dict[str, Any] = {"run_tag": RUN_TAG, "output_root": args.output_root}
    if args.write_task_list or not any((args.run_scout, args.run_comparison, args.run_confirmation, args.run_confirmation_comparison, args.run_final, args.run_final_comparison)):
        outputs["task_list"] = str(write_stage33_task_list(args.output_root))
    if args.run_scout:
        outputs["stage33_training_summary"] = run_stage33_scout(output_root=args.output_root, skip_existing=not args.no_skip_existing)
    if args.run_comparison:
        outputs["stage33_comparison_summary"] = run_stage33_comparison(args.output_root)
    if args.run_confirmation:
        outputs["stage34_confirmation_summary"] = run_stage34_confirmation(output_root=args.output_root, skip_existing=not args.no_skip_existing)
    if args.run_confirmation_comparison:
        outputs["stage34_comparison_summary"] = run_stage34_comparison(args.output_root)
    if args.run_final:
        outputs["stage35_final_training_summary"] = run_stage35_final(output_root=args.output_root, skip_existing=not args.no_skip_existing)
    if args.run_final_comparison:
        outputs["stage35_final_sector_confirmation_summary"] = run_stage35_comparison(args.output_root)
    if args.json:
        print(json.dumps(outputs, ensure_ascii=False, indent=2))
    else:
        print(f"status=ok run_tag={RUN_TAG} output_root={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
