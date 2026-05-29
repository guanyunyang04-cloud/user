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
    study_scope_from_summary_path,
)


PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDIES_ROOT = PROJECT_ROOT / "daily_research/output/path_policy/studies"
RUN_TAG = "mh_stage36_input_cross_section_scout_20260529_01"
STAGE37_RUN_TAG = "mh_stage37_cross_section_arch_retest_20260529_01"
STAGE38_RUN_TAG = "mh_stage38_input_arch_confirmation_20260529_01"
STAGE39_RUN_TAG = "mh_stage39_final_input_arch_confirmation_20260529_01"
RESEARCH_PROGRAM = "alpha_multi_horizon_utility_policy_v1"
STUDY_FAMILY = "stage36_input_cross_section_scout"
STAGE37_STUDY_FAMILY = "stage37_cross_section_arch_retest"
STAGE38_STUDY_FAMILY = "stage38_input_arch_confirmation"
STAGE39_STUDY_FAMILY = "stage39_final_input_arch_confirmation"
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
RAW_BASELINE_FEATURE_PROFILE = "raw_kline_context_no_alpha_prior_v1"
SECTOR_BOARD_VIEW_ID = "policy_sector_board_view__ed15b2873f544e9e9b24aae5"
SECTOR_BOARD_VIEW_KIND = "latest_static_snapshot"
BOUNDARY = "research-only / shadow-only; no active manifest, live/default, production root, paper, or broker integration"
STAGE36_SCOUT_SPECS = (
    {
        "candidate_id": "raw_kline_context_sector_relative_v1",
        "feature_profile": "raw_kline_context_sector_relative_v1",
        "model_family": "gru_sequence_static_context",
    },
    {
        "candidate_id": "raw_kline_context_regime_v1",
        "feature_profile": "raw_kline_context_regime_v1",
        "model_family": "gru_sequence_static_context",
    },
    {
        "candidate_id": "raw_kline_context_sector_relative_regime_v1",
        "feature_profile": "raw_kline_context_sector_relative_regime_v1",
        "model_family": "gru_sequence_static_context",
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


def _stage36_output_root(output_root: str | Path | None = None) -> Path:
    return Path(output_root) if output_root is not None else STUDIES_ROOT / RUN_TAG


def _baseline_dir(seed: int) -> Path:
    tag_by_seed = {
        SCOUT_SEED: BASELINE_STAGE28_SEED7_TAG,
        CONFIRMATION_SEED: BASELINE_STAGE28_SEED11_TAG,
        FINAL_SEED: BASELINE_STAGE28_SEED19_TAG,
    }
    return STUDIES_ROOT / tag_by_seed[int(seed)]


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
        SECTOR_BOARD_VIEW_ID,
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
            "--forecast-amp",
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


def build_stage36_scout_tasks(*, output_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = _stage36_output_root(output_root)
    tasks: list[dict[str, Any]] = []
    for spec in STAGE36_SCOUT_SPECS:
        feature_profile = str(spec["feature_profile"])
        tag = f"mh36_scout_{feature_profile}_seed{SCOUT_SEED}_20260529_01"
        tasks.append(
            _task(
                root=root,
                tag=tag,
                candidate_id=str(spec["candidate_id"]),
                model_family=str(spec["model_family"]),
                feature_profile=feature_profile,
                seed=SCOUT_SEED,
                epochs=16,
                min_epochs=6,
                patience=4,
                memmap_manifest=None,
                reuses_forecast_memmap_manifest=False,
                slot_diagnostics=False,
                study_family=STUDY_FAMILY,
                run_tag=RUN_TAG,
            )
        )
    return tasks


def write_stage36_task_list(output_root: str | Path | None = None) -> Path:
    root = _stage36_output_root(output_root)
    path = root / "stage36_task_list.json"
    tasks = build_stage36_scout_tasks(output_root=root)
    payload = {
        "schema_version": 1,
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "created_at": _now(),
        "baseline_stage28_seed7_tag": BASELINE_STAGE28_SEED7_TAG,
        "single_seed_scout_only": True,
        "evidence_grade_input_or_architecture_pass": False,
        "training_task_count": len(tasks),
        "training_tasks": tasks,
        "boundary": BOUNDARY,
    }
    _write_json(path, payload)
    return path


def _training_progress_payload(
    *,
    status: str,
    completed: list[str],
    failed: list[str],
    run_tag: str,
    study_family: str,
    current_tag: str = "",
    current_step: int | None = None,
    total_steps: int | None = None,
) -> dict[str, Any]:
    completed_steps = len(completed) if current_step is None else int(current_step)
    step_count = len(completed) + len(failed) if total_steps is None else int(total_steps)
    if str(status) == "completed":
        next_decision = "verify_artifacts"
    elif str(status) == "blocked":
        next_decision = "inspect_failure_artifacts"
    else:
        next_decision = "continue_short_polling"
    return {
        "schema_version": 1,
        "status": status,
        "run_tag": run_tag,
        "research_program": RESEARCH_PROGRAM,
        "study_family": study_family,
        "completed_tags": completed,
        "failed_tags": failed,
        "current_tag": str(current_tag or ""),
        "current_step": completed_steps,
        "completed_steps": completed_steps,
        "total_steps": step_count,
        "step_count": step_count,
        "poll_window_seconds": 7200,
        "monitoring_mode": "background_process_short_poll",
        "long_timeout_is_not_failure": True,
        "next_decision": next_decision,
        "updated_at": _now(),
    }


def _summary_dataset_manifest(summary_path: Path) -> dict[str, Any]:
    summary = _read_json(summary_path)
    manifest = summary.get("dataset_manifest")
    return manifest if isinstance(manifest, dict) else {}


def _manifest_count(manifest: dict[str, Any], key: str, group_key: str) -> int:
    group_counts = manifest.get("feature_group_counts", {})
    if isinstance(group_counts, dict):
        try:
            value = int(group_counts.get(group_key, 0) or 0)
            if value:
                return value
        except (TypeError, ValueError):
            pass
    try:
        return int(manifest.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


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
    feature_profile = str(task.get("feature_profile", "") or "")
    sector_relative_count = _manifest_count(manifest, "sector_relative_context_feature_count", "sector_relative_context")
    regime_count = _manifest_count(manifest, "regime_context_feature_count", "regime_context")
    if "sector_relative" in feature_profile and sector_relative_count <= 0:
        reasons.append("missing_sector_relative_context_features")
    if "sector_relative" in feature_profile and not str(manifest.get("source_sector_board_view_id", "") or "").strip():
        reasons.append("missing_sector_board_view_id")
    if "regime" in feature_profile and regime_count <= 0:
        reasons.append("missing_regime_context_features")
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
        "observed_sector_relative_context_feature_count": sector_relative_count,
        "observed_regime_context_feature_count": regime_count,
        "observed_source_sector_board_view_id": str(manifest.get("source_sector_board_view_id", "") or ""),
    }


def _stage36_scope_mismatches(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in tasks:
        mismatch = _task_scope_mismatch(task)
        if mismatch is not None:
            out.append(mismatch)
    return out


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
    total_steps = int(len(tasks))
    for task in tasks:
        _write_json(
            output_root / progress_name,
            _training_progress_payload(
                status="running",
                completed=completed,
                failed=failed,
                run_tag=run_tag,
                study_family=study_family,
                current_tag=str(task.get("tag", "")),
                current_step=len(completed),
                total_steps=total_steps,
            ),
        )
        summary_path = Path(task["study_dir"]) / "study_summary.json"
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
                current_tag=str(task.get("tag", "")),
                current_step=len(completed),
                total_steps=total_steps,
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
            current_tag="",
            current_step=len(completed),
            total_steps=total_steps,
        ),
    )
    _write_json(output_root / summary_name, payload)
    return payload


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
    return ("gru_sequence_static_context", RAW_BASELINE_FEATURE_PROFILE)


def _blocked_scope_payload(root: Path, *, run_tag: str, study_family: str, summary_name: str, mismatches: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "status": "blocked_scope_or_feature_mismatch",
        "run_tag": run_tag,
        "research_program": RESEARCH_PROGRAM,
        "study_family": study_family,
        "scope_mismatches": mismatches,
        "evidence_grade_input_or_architecture_pass": False,
        "stage38_candidates": [],
        "updated_at": _now(),
    }
    _write_json(root / summary_name, payload)
    return payload


def _write_leaderboard(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _feature_profile_audit_rows(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    audits: list[dict[str, Any]] = []
    for task in tasks:
        manifest = _summary_dataset_manifest(Path(task["study_dir"]) / "study_summary.json")
        audits.append(
            {
                "tag": task.get("tag"),
                "feature_profile": task.get("feature_profile"),
                "feature_group_counts": dict(manifest.get("feature_group_counts", {}) or {}),
                "sector_relative_context_feature_count": _manifest_count(manifest, "sector_relative_context_feature_count", "sector_relative_context"),
                "regime_context_feature_count": _manifest_count(manifest, "regime_context_feature_count", "regime_context"),
                "source_sector_board_view_id": str(manifest.get("source_sector_board_view_id", "") or ""),
                "feature_profile_audit": dict(manifest.get("feature_profile_audit", {}) or {}),
            }
        )
    return {
        "schema_version": 1,
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "generated_at": _now(),
        "profiles": audits,
    }


def _write_stage36_feature_profile_audit(*, output_root: str | Path | None = None, tasks: list[dict[str, Any]] | None = None) -> Path:
    root = _stage36_output_root(output_root)
    path = root / "stage36_feature_profile_audit.json"
    payload = _feature_profile_audit_rows(tasks or build_stage36_scout_tasks(output_root=root))
    _write_json(path, payload)
    return path


def _stage36_candidate_row(*, task: dict[str, Any], validation_row: dict[str, Any], test_row: dict[str, Any] | None, baseline_validation: dict[str, Any], baseline_test: dict[str, Any]) -> dict[str, Any]:
    rank_ratio = _ratio(_float(validation_row, "rank_ic_mean"), _float(baseline_validation, "rank_ic_mean"))
    spread_ratio = _ratio(_float(validation_row, "spread_mean"), _float(baseline_validation, "spread_mean"))
    hit_ratio = _ratio(_float(validation_row, "hit_lift_mean"), _float(baseline_validation, "hit_lift_mean"))
    concentration_reduction = _reduction(_float(validation_row, "thirty_d_concentration_mean"), _float(baseline_validation, "thirty_d_concentration_mean"))
    gap_reduction = _reduction(_float(validation_row, "pred_future_horizon_gap_mean"), _float(baseline_validation, "pred_future_horizon_gap_mean"))
    composite = (
        0.45 * rank_ratio
        + 0.30 * spread_ratio
        + 0.20 * hit_ratio
        + 0.03 * concentration_reduction
        + 0.02 * gap_reduction
    )
    validation_gate = bool(
        _float(validation_row, "rank_ic_mean") > 0.0
        and _float(validation_row, "spread_mean") > 0.0
        and _float(validation_row, "hit_lift_mean") > 0.0
        and _float(validation_row, "monthly_positive_rate_mean") >= 0.75
        and _int(validation_row, "negative_month_count_max", 99) <= 2
        and _float(validation_row, "thirty_d_concentration_mean", 1.0) <= 0.85
        and hit_ratio >= 0.95
    )
    test_negative = not bool(
        _float(test_row, "rank_ic_mean") > 0.0
        and _float(test_row, "spread_mean") > 0.0
        and _float(test_row, "hit_lift_mean") > 0.0
    )
    test_hit_drop_too_large = _float(test_row, "hit_lift_mean") < (_float(baseline_test, "hit_lift_mean") * 0.80)
    return {
        **task,
        "stage36_composite": composite,
        "rank_ic_ratio": rank_ratio,
        "spread_ratio": spread_ratio,
        "hit_lift_ratio": hit_ratio,
        "concentration_reduction_ratio": concentration_reduction,
        "pred_future_gap_reduction_ratio": gap_reduction,
        "stage36_hard_gate_pass": validation_gate,
        "test_veto": bool(test_negative or test_hit_drop_too_large),
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


def _select_stage38_candidates(leaderboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in leaderboard if bool(row.get("stage36_hard_gate_pass")) and not bool(row.get("test_veto"))]
    strong = [row for row in eligible if float(row.get("stage36_composite", 0.0) or 0.0) >= 1.02]
    if strong:
        return strong[:2]
    diagnostic = [
        row
        for row in eligible
        if float(row.get("stage36_composite", 0.0) or 0.0) >= 0.97
        and (
            float(row.get("concentration_reduction_ratio", 0.0) or 0.0) >= 0.10
            or float(row.get("pred_future_gap_reduction_ratio", 0.0) or 0.0) >= 0.10
        )
        and float(row.get("hit_lift_ratio", 0.0) or 0.0) >= 0.90
    ]
    return diagnostic[:1]


def build_stage37_arch_retest_tasks(candidates: list[dict[str, Any]], *, output_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = _stage36_output_root(output_root)
    if not candidates:
        return []
    best = candidates[0]
    feature_profile = str(best["feature_profile"])
    manifest_path = Path(str(best.get("manifest_source_study_dir") or best.get("study_dir"))) / "forecast_dataset_manifest.json"
    tasks: list[dict[str, Any]] = []
    for model_family in ("stock_mixer_sequence", "sector_slot_mixer_sequence"):
        candidate_id = f"{model_family}_{feature_profile}"
        tag = f"mh37_retest_{candidate_id}_seed{SCOUT_SEED}_20260529_01"
        task = _task(
            root=root,
            tag=tag,
            candidate_id=candidate_id,
            model_family=model_family,
            feature_profile=feature_profile,
            seed=SCOUT_SEED,
            epochs=16,
            min_epochs=6,
            patience=4,
            memmap_manifest=manifest_path,
            reuses_forecast_memmap_manifest=True,
            slot_diagnostics=model_family == "sector_slot_mixer_sequence",
            study_family=STAGE37_STUDY_FAMILY,
            run_tag=STAGE37_RUN_TAG,
        )
        task["manifest_source_study_dir"] = str(best.get("study_dir", ""))
        task["input_candidate_tag"] = str(best.get("tag", ""))
        tasks.append(task)
    return tasks


def _arch_wins_against_input_gru(
    *,
    candidate: dict[str, Any],
    validation_row: dict[str, Any] | None,
    test_row: dict[str, Any] | None,
    input_gru_validation: dict[str, Any] | None,
    input_gru_test: dict[str, Any] | None,
) -> bool:
    if validation_row is None or test_row is None or input_gru_validation is None or input_gru_test is None:
        return False
    composite = (
        0.45 * _ratio(_float(validation_row, "rank_ic_mean"), _float(input_gru_validation, "rank_ic_mean"))
        + 0.30 * _ratio(_float(validation_row, "spread_mean"), _float(input_gru_validation, "spread_mean"))
        + 0.20 * _ratio(_float(validation_row, "hit_lift_mean"), _float(input_gru_validation, "hit_lift_mean"))
        + 0.03 * _reduction(_float(validation_row, "thirty_d_concentration_mean"), _float(input_gru_validation, "thirty_d_concentration_mean"))
        + 0.02 * _reduction(_float(validation_row, "pred_future_horizon_gap_mean"), _float(input_gru_validation, "pred_future_horizon_gap_mean"))
    )
    validation_positive = bool(
        _float(validation_row, "rank_ic_mean") > 0.0
        and _float(validation_row, "spread_mean") > 0.0
        and _float(validation_row, "hit_lift_mean") > 0.0
    )
    test_positive = bool(
        _float(test_row, "rank_ic_mean") > 0.0
        and _float(test_row, "spread_mean") > 0.0
        and _float(test_row, "hit_lift_mean") > 0.0
    )
    concentration_ok = _float(test_row, "thirty_d_concentration_mean", 1.0) <= _float(input_gru_test, "thirty_d_concentration_mean", 0.0)
    gap_ok = _float(test_row, "pred_future_horizon_gap_mean", 1.0e9) <= _float(input_gru_test, "pred_future_horizon_gap_mean", 0.0)
    candidate["stage37_composite"] = composite
    candidate["stage37_pass"] = bool(validation_positive and test_positive and composite >= 1.02 and concentration_ok and gap_ok)
    return bool(candidate["stage37_pass"])


def build_stage38_confirmation_tasks(candidates: list[dict[str, Any]], *, output_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = _stage36_output_root(output_root)
    tasks: list[dict[str, Any]] = []
    for candidate in candidates[:2]:
        feature_profile = str(candidate["feature_profile"])
        model_family = str(candidate["model_family"])
        manifest_study_dir = str(candidate.get("manifest_source_study_dir") or candidate.get("study_dir") or "")
        manifest_path = Path(manifest_study_dir) / "forecast_dataset_manifest.json"
        tag = f"mh38_confirm_{model_family}_{feature_profile}_seed{CONFIRMATION_SEED}_20260529_01"
        task = _task(
            root=root,
            tag=tag,
            candidate_id=str(candidate.get("candidate_id", f"{model_family}_{feature_profile}")),
            model_family=model_family,
            feature_profile=feature_profile,
            seed=CONFIRMATION_SEED,
            epochs=24,
            min_epochs=8,
            patience=6,
            memmap_manifest=manifest_path,
            reuses_forecast_memmap_manifest=True,
            slot_diagnostics=model_family == "sector_slot_mixer_sequence",
            study_family=STAGE38_STUDY_FAMILY,
            run_tag=STAGE38_RUN_TAG,
        )
        task["scout_tag"] = str(candidate.get("tag", ""))
        task["manifest_source_study_dir"] = manifest_study_dir
        tasks.append(task)
    return tasks


def _load_stage38_candidates(output_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = _stage36_output_root(output_root)
    payload = _read_json(root / "stage38_confirmation_task_list.json")
    items = payload.get("stage38_candidates", [])
    return items if isinstance(items, list) else []


def _load_stage39_candidates(output_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = _stage36_output_root(output_root)
    payload = _read_json(root / "stage39_final_task_list.json")
    items = payload.get("stage39_candidates", [])
    return items if isinstance(items, list) else []


def _write_stage37_task_list(candidates: list[dict[str, Any]], *, output_root: str | Path | None = None) -> Path:
    root = _stage36_output_root(output_root)
    path = root / "stage37_arch_retest_task_list.json"
    tasks = build_stage37_arch_retest_tasks(candidates, output_root=root)
    payload = {
        "schema_version": 1,
        "run_tag": STAGE37_RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STAGE37_STUDY_FAMILY,
        "created_at": _now(),
        "stage37_input_candidate": candidates[0] if candidates else None,
        "training_task_count": len(tasks),
        "training_tasks": tasks,
        "boundary": BOUNDARY,
    }
    _write_json(path, payload)
    return path


def _write_stage38_task_list(candidates: list[dict[str, Any]], *, output_root: str | Path | None = None) -> Path:
    root = _stage36_output_root(output_root)
    path = root / "stage38_confirmation_task_list.json"
    tasks = build_stage38_confirmation_tasks(candidates, output_root=root)
    payload = {
        "schema_version": 1,
        "run_tag": STAGE38_RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STAGE38_STUDY_FAMILY,
        "created_at": _now(),
        "stage38_candidates": candidates[:2],
        "training_task_count": len(tasks),
        "training_tasks": tasks,
        "boundary": BOUNDARY,
    }
    _write_json(path, payload)
    return path


def _write_stage39_task_list(candidates: list[dict[str, Any]], *, output_root: str | Path | None = None) -> Path:
    root = _stage36_output_root(output_root)
    path = root / "stage39_final_task_list.json"
    tasks: list[dict[str, Any]] = []
    for candidate in candidates[:1]:
        feature_profile = str(candidate["feature_profile"])
        model_family = str(candidate["model_family"])
        manifest_study_dir = str(candidate.get("manifest_source_study_dir") or candidate.get("study_dir") or "")
        manifest_path = Path(manifest_study_dir) / "forecast_dataset_manifest.json"
        tag = f"mh39_final_{model_family}_{feature_profile}_seed{FINAL_SEED}_20260529_01"
        task = _task(
            root=root,
            tag=tag,
            candidate_id=str(candidate.get("candidate_id", f"{model_family}_{feature_profile}")),
            model_family=model_family,
            feature_profile=feature_profile,
            seed=FINAL_SEED,
            epochs=24,
            min_epochs=8,
            patience=6,
            memmap_manifest=manifest_path,
            reuses_forecast_memmap_manifest=True,
            slot_diagnostics=model_family == "sector_slot_mixer_sequence",
            study_family=STAGE39_STUDY_FAMILY,
            run_tag=STAGE39_RUN_TAG,
        )
        task["manifest_source_study_dir"] = manifest_study_dir
        tasks.append(task)
    payload = {
        "schema_version": 1,
        "run_tag": STAGE39_RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STAGE39_STUDY_FAMILY,
        "created_at": _now(),
        "stage39_candidates": candidates[:1],
        "training_task_count": len(tasks),
        "training_tasks": tasks,
        "boundary": BOUNDARY,
    }
    _write_json(path, payload)
    return path


def run_stage36_scout(*, output_root: str | Path | None = None, skip_existing: bool = True) -> dict[str, Any]:
    root = _stage36_output_root(output_root)
    return _run_training_tasks(
        tasks=build_stage36_scout_tasks(output_root=root),
        output_root=root,
        summary_name="stage36_training_summary.json",
        progress_name="stage36_progress.json",
        run_tag=RUN_TAG,
        study_family=STUDY_FAMILY,
        skip_existing=skip_existing,
    )


def run_stage36_comparison(output_root: str | Path | None = None) -> dict[str, Any]:
    root = _stage36_output_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    tasks = build_stage36_scout_tasks(output_root=root)
    mismatches = _stage36_scope_mismatches(tasks)
    if mismatches:
        return _blocked_scope_payload(root, run_tag=RUN_TAG, study_family=STUDY_FAMILY, summary_name="stage36_comparison_summary.json", mismatches=mismatches)
    audit_path = _write_stage36_feature_profile_audit(output_root=root, tasks=tasks)
    study_dirs = [str(_baseline_dir(SCOUT_SEED)), *[task["study_dir"] for task in tasks]]
    report = build_output_aux_profile_comparison(study_dirs, run_tag=RUN_TAG)
    paths = write_output_aux_profile_comparison(report, root / "stage36_comparison")
    validation_rows, test_rows = _aggregate_maps(report)
    baseline_validation = validation_rows.get(_baseline_key())
    baseline_test = test_rows.get(_baseline_key())
    if baseline_validation is None or baseline_test is None:
        return _blocked_scope_payload(
            root,
            run_tag=RUN_TAG,
            study_family=STUDY_FAMILY,
            summary_name="stage36_comparison_summary.json",
            mismatches=[{"tag": BASELINE_STAGE28_SEED7_TAG, "reasons": ["missing_baseline_validation_or_test_row"]}],
        )
    leaderboard = [
        _stage36_candidate_row(
            task=task,
            validation_row=validation_rows[_row_key(task)],
            test_row=test_rows.get(_row_key(task)),
            baseline_validation=baseline_validation,
            baseline_test=baseline_test,
        )
        for task in tasks
        if _row_key(task) in validation_rows
    ]
    leaderboard.sort(key=lambda item: float(item.get("stage36_composite", 0.0) or 0.0), reverse=True)
    stage38_candidates = _select_stage38_candidates(leaderboard)
    stage37_input_candidate = stage38_candidates[:1]
    if stage37_input_candidate:
        _write_stage37_task_list(stage37_input_candidate, output_root=root)
    if stage38_candidates:
        _write_stage38_task_list(stage38_candidates, output_root=root)
    leaderboard_path = root / "stage36_input_leaderboard.csv"
    _write_leaderboard(leaderboard_path, leaderboard)
    verdict_path = root / "stage36_research_verdict.md"
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
        "stage36_feature_profile_audit_json": str(audit_path),
        "stage36_input_leaderboard": leaderboard,
        "stage37_input_candidate": stage37_input_candidate[0] if stage37_input_candidate else None,
        "stage38_candidates": stage38_candidates,
        "comparison_paths": {key: str(path) for key, path in paths.items()},
        "stage36_input_leaderboard_csv": str(leaderboard_path),
        "stage36_research_verdict_md": str(verdict_path),
        "updated_at": _now(),
    }
    verdict_path.write_text(
        "# Alpha Multi-Horizon Stage 36 Input Cross-Section Scout\n\n"
        f"- status: `{payload.get('status')}`\n"
        f"- stage38_candidate_count: `{len(stage38_candidates)}`\n"
        f"- boundary: {BOUNDARY}.\n",
        encoding="utf-8",
    )
    _write_json(root / "stage36_comparison_summary.json", payload)
    return payload


def run_stage37_arch_retest(*, output_root: str | Path | None = None, skip_existing: bool = True) -> dict[str, Any]:
    root = _stage36_output_root(output_root)
    summary = _read_json(root / "stage36_comparison_summary.json")
    candidate = summary.get("stage37_input_candidate")
    candidates = [candidate] if isinstance(candidate, dict) and candidate else []
    tasks = build_stage37_arch_retest_tasks(candidates, output_root=root)
    return _run_training_tasks(
        tasks=tasks,
        output_root=root,
        summary_name="stage37_training_summary.json",
        progress_name="stage37_progress.json",
        run_tag=STAGE37_RUN_TAG,
        study_family=STAGE37_STUDY_FAMILY,
        skip_existing=skip_existing,
    )


def run_stage37_comparison(output_root: str | Path | None = None) -> dict[str, Any]:
    root = _stage36_output_root(output_root)
    stage36_summary = _read_json(root / "stage36_comparison_summary.json")
    input_candidate = stage36_summary.get("stage37_input_candidate")
    if not isinstance(input_candidate, dict) or not input_candidate:
        payload = {
            "schema_version": 1,
            "status": "blocked_no_stage37_input_candidate",
            "run_tag": STAGE37_RUN_TAG,
            "research_program": RESEARCH_PROGRAM,
            "study_family": STAGE37_STUDY_FAMILY,
            "stage38_candidates": [],
            "updated_at": _now(),
        }
        _write_json(root / "stage37_comparison_summary.json", payload)
        return payload
    tasks = build_stage37_arch_retest_tasks([input_candidate], output_root=root)
    mismatches = _stage36_scope_mismatches(tasks)
    if mismatches:
        return _blocked_scope_payload(root, run_tag=STAGE37_RUN_TAG, study_family=STAGE37_STUDY_FAMILY, summary_name="stage37_comparison_summary.json", mismatches=mismatches)
    study_dirs = [str(_baseline_dir(SCOUT_SEED)), str(input_candidate["study_dir"]), *[task["study_dir"] for task in tasks]]
    report = build_output_aux_profile_comparison(study_dirs, run_tag=STAGE37_RUN_TAG)
    paths = write_output_aux_profile_comparison(report, root / "stage37_comparison")
    validation_rows, test_rows = _aggregate_maps(report)
    input_gru_key = (str(input_candidate.get("model_family", "")), str(input_candidate.get("feature_profile", "")))
    arch_candidates: list[dict[str, Any]] = []
    for task in tasks:
        candidate = dict(task)
        _arch_wins_against_input_gru(
            candidate=candidate,
            validation_row=validation_rows.get(_row_key(task)),
            test_row=test_rows.get(_row_key(task)),
            input_gru_validation=validation_rows.get(input_gru_key),
            input_gru_test=test_rows.get(input_gru_key),
        )
        arch_candidates.append(candidate)
    arch_candidates.sort(key=lambda item: float(item.get("stage37_composite", 0.0) or 0.0), reverse=True)
    stage38_arch_candidates = [item for item in arch_candidates if bool(item.get("stage37_pass"))][:2]
    merged_stage38_candidates = [input_candidate, *stage38_arch_candidates][:2]
    if merged_stage38_candidates:
        _write_stage38_task_list(merged_stage38_candidates, output_root=root)
    leaderboard_path = root / "stage37_arch_retest_leaderboard.csv"
    _write_leaderboard(leaderboard_path, arch_candidates)
    verdict_path = root / "stage37_research_verdict.md"
    payload = {
        "schema_version": 1,
        "status": report.get("status"),
        "run_tag": STAGE37_RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STAGE37_STUDY_FAMILY,
        "stage37_input_candidate": input_candidate,
        "stage37_arch_retest_leaderboard": arch_candidates,
        "stage38_candidates": merged_stage38_candidates,
        "comparison_paths": {key: str(path) for key, path in paths.items()},
        "stage37_arch_retest_leaderboard_csv": str(leaderboard_path),
        "stage37_research_verdict_md": str(verdict_path),
        "updated_at": _now(),
    }
    verdict_path.write_text(
        "# Alpha Multi-Horizon Stage 37 Cross-Section Architecture Retest\n\n"
        f"- status: `{payload.get('status')}`\n"
        f"- stage38_candidate_count: `{len(merged_stage38_candidates)}`\n"
        f"- boundary: {BOUNDARY}.\n",
        encoding="utf-8",
    )
    _write_json(root / "stage37_comparison_summary.json", payload)
    return payload


def run_stage38_confirmation(*, output_root: str | Path | None = None, skip_existing: bool = True) -> dict[str, Any]:
    root = _stage36_output_root(output_root)
    candidates = _load_stage38_candidates(root)
    tasks = build_stage38_confirmation_tasks(candidates, output_root=root)
    return _run_training_tasks(
        tasks=tasks,
        output_root=root,
        summary_name="stage38_confirmation_summary.json",
        progress_name="stage38_progress.json",
        run_tag=STAGE38_RUN_TAG,
        study_family=STAGE38_STUDY_FAMILY,
        skip_existing=skip_existing,
    )


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


def run_stage38_comparison(output_root: str | Path | None = None) -> dict[str, Any]:
    root = _stage36_output_root(output_root)
    candidates = _load_stage38_candidates(root)
    tasks = build_stage38_confirmation_tasks(candidates, output_root=root)
    mismatches = _stage36_scope_mismatches(tasks)
    if mismatches:
        return _blocked_scope_payload(root, run_tag=STAGE38_RUN_TAG, study_family=STAGE38_STUDY_FAMILY, summary_name="stage38_confirmation_summary.json", mismatches=mismatches)
    candidate_seed7_dirs = [candidate.get("study_dir") for candidate in candidates if candidate.get("study_dir")]
    study_dirs = [str(_baseline_dir(SCOUT_SEED)), str(_baseline_dir(CONFIRMATION_SEED)), *candidate_seed7_dirs, *[task["study_dir"] for task in tasks]]
    report = build_output_aux_profile_comparison(study_dirs, run_tag=STAGE38_RUN_TAG)
    paths = write_output_aux_profile_comparison(report, root / "stage38_comparison")
    validation_rows, test_rows = _aggregate_maps(report)
    baseline_test = test_rows.get(_baseline_key())
    if baseline_test is None:
        return _blocked_scope_payload(
            root,
            run_tag=STAGE38_RUN_TAG,
            study_family=STAGE38_STUDY_FAMILY,
            summary_name="stage38_confirmation_summary.json",
            mismatches=[{"tag": "stage28_seed7_seed11_baseline", "reasons": ["missing_baseline_test_row"]}],
        )
    leaderboard: list[dict[str, Any]] = []
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
            and _float(test, "hit_lift_min") > 0.0
        )
        monthly_ok = bool(_float(test, "monthly_positive_rate_mean") >= 0.75 and _int(test, "negative_month_count_max", 99) <= 2)
        rank_gain = _gain(_float(test, "rank_ic_mean"), _float(baseline_test, "rank_ic_mean"))
        spread_gain = _gain(_float(test, "spread_mean"), _float(baseline_test, "spread_mean"))
        hit_gain = _gain(_float(test, "hit_lift_mean"), _float(baseline_test, "hit_lift_mean"))
        concentration_reduction = _reduction(_float(test, "thirty_d_concentration_mean"), _float(baseline_test, "thirty_d_concentration_mean"))
        gap_reduction = _reduction(_float(test, "pred_future_horizon_gap_mean"), _float(baseline_test, "pred_future_horizon_gap_mean"))
        at_least_one = max(rank_gain, spread_gain, hit_gain) >= 0.05
        others_not_too_bad = min(rank_gain, spread_gain, hit_gain) >= -0.05
        concentration_or_gap = concentration_reduction >= 0.10 or gap_reduction >= 0.10
        pass_gate = bool(validation_positive and test_positive and monthly_ok and at_least_one and others_not_too_bad and concentration_or_gap)
        leaderboard.append(
            {
                **candidate,
                "stage38_pass": pass_gate,
                "stage38_score": max(rank_gain, spread_gain, hit_gain, concentration_reduction, gap_reduction),
            }
        )
    leaderboard.sort(key=lambda item: float(item.get("stage38_score", 0.0) or 0.0), reverse=True)
    stage39_candidates = [row for row in leaderboard if bool(row.get("stage38_pass"))][:1]
    if stage39_candidates:
        _write_stage39_task_list(stage39_candidates, output_root=root)
    leaderboard_path = root / "stage38_candidate_leaderboard.csv"
    _write_leaderboard(leaderboard_path, leaderboard)
    verdict_path = root / "stage38_research_verdict.md"
    payload = {
        "schema_version": 1,
        "status": report.get("status"),
        "run_tag": STAGE38_RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STAGE38_STUDY_FAMILY,
        "stage38_candidate_leaderboard": leaderboard,
        "stage39_candidates": stage39_candidates,
        "comparison_paths": {key: str(path) for key, path in paths.items()},
        "stage38_candidate_leaderboard_csv": str(leaderboard_path),
        "stage38_research_verdict_md": str(verdict_path),
        "updated_at": _now(),
    }
    verdict_path.write_text(
        "# Alpha Multi-Horizon Stage 38 Input/Architecture Confirmation\n\n"
        f"- status: `{payload.get('status')}`\n"
        f"- stage39_candidate_count: `{len(stage39_candidates)}`\n"
        f"- boundary: {BOUNDARY}.\n",
        encoding="utf-8",
    )
    _write_json(root / "stage38_confirmation_summary.json", payload)
    return payload


def _stage38_task_dirs_for_candidates(candidates: list[dict[str, Any]], *, output_root: str | Path | None = None) -> list[str]:
    return [str(task.get("study_dir")) for task in build_stage38_confirmation_tasks(candidates, output_root=output_root)]


def run_stage39_final(*, output_root: str | Path | None = None, skip_existing: bool = True) -> dict[str, Any]:
    root = _stage36_output_root(output_root)
    candidates = _load_stage39_candidates(root)
    tasks_payload = _read_json(root / "stage39_final_task_list.json")
    tasks = tasks_payload.get("training_tasks", [])
    tasks = tasks if isinstance(tasks, list) else []
    if not tasks:
        _write_stage39_task_list(candidates, output_root=root)
        tasks_payload = _read_json(root / "stage39_final_task_list.json")
        tasks = tasks_payload.get("training_tasks", [])
        tasks = tasks if isinstance(tasks, list) else []
    return _run_training_tasks(
        tasks=tasks,
        output_root=root,
        summary_name="stage39_final_training_summary.json",
        progress_name="stage39_progress.json",
        run_tag=STAGE39_RUN_TAG,
        study_family=STAGE39_STUDY_FAMILY,
        skip_existing=skip_existing,
    )


def run_stage39_comparison(output_root: str | Path | None = None) -> dict[str, Any]:
    root = _stage36_output_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    candidates = _load_stage39_candidates(root)
    tasks_payload = _read_json(root / "stage39_final_task_list.json")
    stage39_tasks = tasks_payload.get("training_tasks", [])
    stage39_tasks = stage39_tasks if isinstance(stage39_tasks, list) else []
    mismatches = _stage36_scope_mismatches(stage39_tasks) if stage39_tasks else []
    if mismatches:
        return _blocked_scope_payload(
            root,
            run_tag=STAGE39_RUN_TAG,
            study_family=STAGE39_STUDY_FAMILY,
            summary_name="stage39_final_input_arch_confirmation_summary.json",
            mismatches=mismatches,
        )
    candidate_seed7_dirs = [candidate.get("study_dir") for candidate in candidates if candidate.get("study_dir")]
    candidate_seed11_dirs = _stage38_task_dirs_for_candidates(candidates, output_root=root)
    study_dirs = [
        str(_baseline_dir(SCOUT_SEED)),
        str(_baseline_dir(CONFIRMATION_SEED)),
        str(_baseline_dir(FINAL_SEED)),
        *candidate_seed7_dirs,
        *candidate_seed11_dirs,
        *[str(task.get("study_dir")) for task in stage39_tasks],
    ]
    report = build_output_aux_profile_comparison(study_dirs, run_tag=STAGE39_RUN_TAG)
    paths = write_output_aux_profile_comparison(report, root / "stage39_comparison")
    validation_rows, test_rows = _aggregate_maps(report)
    baseline_validation = validation_rows.get(_baseline_key())
    baseline_test = test_rows.get(_baseline_key())
    leaderboard: list[dict[str, Any]] = []
    for candidate in candidates[:1]:
        key = (str(candidate.get("model_family", "")), str(candidate.get("feature_profile", "")))
        validation = validation_rows.get(key)
        test = test_rows.get(key)
        if validation is None or test is None or baseline_validation is None or baseline_test is None:
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
        hit_min_ok = _float(test, "hit_lift_min") > 0.0
        rank_gain = _gain(_float(test, "rank_ic_mean"), _float(baseline_test, "rank_ic_mean"))
        spread_gain = _gain(_float(test, "spread_mean"), _float(baseline_test, "spread_mean"))
        hit_gain = _gain(_float(test, "hit_lift_mean"), _float(baseline_test, "hit_lift_mean"))
        concentration_reduction = _reduction(_float(test, "thirty_d_concentration_mean"), _float(baseline_test, "thirty_d_concentration_mean"))
        gap_reduction = _reduction(_float(test, "pred_future_horizon_gap_mean"), _float(baseline_test, "pred_future_horizon_gap_mean"))
        core_metrics = [rank_gain >= 0.0, spread_gain >= 0.0, hit_gain >= 0.0]
        core_advantage = sum(bool(item) for item in core_metrics) >= 2 and max(rank_gain, spread_gain, hit_gain) >= 0.05
        full_gate = bool(validation_positive and test_positive and monthly_ok and long_horizon_ok and concentration_ok and gap_ok and hit_min_ok and core_advantage)
        leaderboard.append(
            {
                **candidate,
                "stage39_full_gate_pass": full_gate,
                "input_or_architecture_upgrade_allowed": bool(full_gate),
            }
        )
    input_or_architecture_upgrade_allowed = any(bool(row.get("input_or_architecture_upgrade_allowed")) for row in leaderboard)
    leaderboard_path = root / "stage39_final_candidate_leaderboard.csv"
    _write_leaderboard(leaderboard_path, leaderboard)
    verdict_path = root / "stage39_research_verdict.md"
    baseline_validation_snapshot = _aggregate_snapshot(baseline_validation)
    baseline_test_snapshot = _aggregate_snapshot(baseline_test)
    candidate_validation_snapshot = _aggregate_snapshot(validation_rows.get(_row_key(leaderboard[0])) if leaderboard else None)
    candidate_test_snapshot = _aggregate_snapshot(test_rows.get(_row_key(leaderboard[0])) if leaderboard else None)
    payload = {
        "schema_version": 1,
        "status": report.get("status"),
        "run_tag": STAGE39_RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STAGE39_STUDY_FAMILY,
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
        "stage39_final_candidate_leaderboard_csv": str(leaderboard_path),
        "stage39_research_verdict_md": str(verdict_path),
        "boundary": BOUNDARY,
        "updated_at": _now(),
    }
    verdict_path.write_text(
        "# Alpha Multi-Horizon Stage 39 Final Input/Architecture Confirmation\n\n"
        f"- status: `{payload.get('status')}`\n"
        f"- input_or_architecture_upgrade_allowed: `{input_or_architecture_upgrade_allowed}`\n"
        f"- boundary: {BOUNDARY}.\n",
        encoding="utf-8",
    )
    _write_json(root / "stage39_final_input_arch_confirmation_summary.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 36 input engineering and cross-section scout driver.")
    parser.add_argument("--output-root", default=str(STUDIES_ROOT / RUN_TAG))
    parser.add_argument("--write-task-list", action="store_true")
    parser.add_argument("--run-scout", action="store_true")
    parser.add_argument("--run-comparison", action="store_true")
    parser.add_argument("--run-arch-retest", action="store_true")
    parser.add_argument("--run-arch-comparison", action="store_true")
    parser.add_argument("--run-confirmation", action="store_true")
    parser.add_argument("--run-confirmation-comparison", action="store_true")
    parser.add_argument("--run-final", action="store_true")
    parser.add_argument("--run-final-comparison", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    outputs: dict[str, Any] = {"run_tag": RUN_TAG, "output_root": args.output_root}
    if args.write_task_list or not any((args.run_scout, args.run_comparison, args.run_arch_retest, args.run_arch_comparison, args.run_confirmation, args.run_confirmation_comparison, args.run_final, args.run_final_comparison)):
        outputs["task_list"] = str(write_stage36_task_list(args.output_root))
    if args.run_scout:
        outputs["stage36_training_summary"] = run_stage36_scout(output_root=args.output_root, skip_existing=not args.no_skip_existing)
    if args.run_comparison:
        outputs["stage36_comparison_summary"] = run_stage36_comparison(args.output_root)
    if args.run_arch_retest:
        outputs["stage37_training_summary"] = run_stage37_arch_retest(output_root=args.output_root, skip_existing=not args.no_skip_existing)
    if args.run_arch_comparison:
        outputs["stage37_comparison_summary"] = run_stage37_comparison(args.output_root)
    if args.run_confirmation:
        outputs["stage38_confirmation_summary"] = run_stage38_confirmation(output_root=args.output_root, skip_existing=not args.no_skip_existing)
    if args.run_confirmation_comparison:
        outputs["stage38_comparison_summary"] = run_stage38_comparison(args.output_root)
    if args.run_final:
        outputs["stage39_final_training_summary"] = run_stage39_final(output_root=args.output_root, skip_existing=not args.no_skip_existing)
    if args.run_final_comparison:
        outputs["stage39_final_input_arch_confirmation_summary"] = run_stage39_comparison(args.output_root)
    if args.json:
        print(json.dumps(outputs, ensure_ascii=False, indent=2))
    else:
        print(f"status=ok run_tag={RUN_TAG} output_root={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
