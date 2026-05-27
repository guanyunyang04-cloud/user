from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from daily_research.path_policy.output_aux_profile_comparison import (
    build_output_aux_profile_comparison,
    profile_aggregate_rows,
    write_output_aux_profile_comparison,
)
from daily_research.path_policy.stage_universe_scope import (
    FULL_ROLLING_LIQUID500_SCOPE,
    full_pool_task_metadata,
    full_rolling_liquid500_command_args,
    study_scope_from_summary_path,
)


PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDIES_ROOT = PROJECT_ROOT / "daily_research/output/path_policy/studies"
RUN_TAG = "mh_stage27_target_head_stability_20260527_01"
RESEARCH_PROGRAM = "alpha_multi_horizon_utility_policy_v1"
STUDY_FAMILY = "stage27_target_head_stability"
DATASET_ID = "policy_input_bundle__7c8f58d851bce8179e1e9e2d"
SEEDS = (7, 11, 19)
FULLGRID = "1,2,3,5,8,10,15,20,30"
STAGE27_CANDIDATES = (
    "horizon_target_normalized_v1",
    "horizon_head_soft_constraint_v1",
    "target_norm_head_constraint_v1",
)
BASELINE_STAGE26_PROFILES = (
    "score_monthly_robust_v1",
    "horizon_entropy_regularized_v1",
    "risk_drawdown_reweighted_v1",
)
BASELINE_LONG_HORIZON_SHARE = 0.930173775671406
MAX_THIRTY_D_CONCENTRATION = 0.85
BOUNDARY = "research-only / shadow-only; no active manifest, live/default, production root, paper, or broker integration"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stage27_output_root(output_root: str | Path | None = None) -> Path:
    return Path(output_root) if output_root is not None else STUDIES_ROOT / RUN_TAG


def _training_command(*, tag: str, loss_profile: str, seed: int) -> list[str]:
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
        *full_rolling_liquid500_command_args(STUDIES_ROOT),
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
        "gru_sequence_static_context",
        "--forecast-feature-profile",
        "raw_kline_context_no_alpha_prior_v1",
        "--forecast-include-static-context",
        "--forecast-max-feature-columns",
        "192",
        "--forecast-cumulative-horizons",
        FULLGRID,
        "--forecast-horizon",
        "30",
        "--forecast-output-profile",
        "decision_utility_v1",
        "--forecast-loss-profile",
        loss_profile,
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


def build_stage27_training_tasks(*, output_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = _stage27_output_root(output_root)
    tasks: list[dict[str, Any]] = []
    for candidate in STAGE27_CANDIDATES:
        for seed in SEEDS:
            tag = f"mh27_{candidate}_fullgrid_seed{seed}_20260527_01"
            study_dir = STUDIES_ROOT / tag
            tasks.append(
                {
                    "tag": tag,
                    "candidate": candidate,
                    "seed": int(seed),
                    "horizons": FULLGRID,
                    "max_horizon": 30,
                    "study_dir": str(study_dir),
                    "stdout": str(root / f"{tag}_stdout.log"),
                    "stderr": str(root / f"{tag}_stderr.log"),
                    "command": _training_command(tag=tag, loss_profile=candidate, seed=int(seed)),
                    "research_program": RESEARCH_PROGRAM,
                    "study_family": STUDY_FAMILY,
                    "scope": "research_shadow_only",
                    **full_pool_task_metadata(STUDIES_ROOT),
                    "may_touch_active_manifest": False,
                }
            )
    return tasks


def write_stage27_task_list(output_root: str | Path | None = None) -> Path:
    root = _stage27_output_root(output_root)
    path = root / "stage27_task_list.json"
    payload = {
        "schema_version": 1,
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "created_at": _now(),
        "baseline_stage26_profiles": list(BASELINE_STAGE26_PROFILES),
        "baseline_long_horizon_share": BASELINE_LONG_HORIZON_SHARE,
        "max_thirty_d_concentration": MAX_THIRTY_D_CONCENTRATION,
        "universe_scope": FULL_ROLLING_LIQUID500_SCOPE,
        "reused_forecast_memmap_manifest": str(full_pool_task_metadata(STUDIES_ROOT)["forecast_memmap_manifest"]),
        "training_task_count": len(build_stage27_training_tasks(output_root=root)),
        "training_tasks": build_stage27_training_tasks(output_root=root),
        "boundary": BOUNDARY,
    }
    _write_json(path, payload)
    return path


def _training_progress_payload(*, status: str, completed: list[str], failed: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "completed_tags": completed,
        "failed_tags": failed,
        "updated_at": _now(),
    }


def run_training_tasks(*, output_root: str | Path | None = None, skip_existing: bool = True) -> dict[str, Any]:
    root = _stage27_output_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    failed: list[str] = []
    results: list[dict[str, Any]] = []
    for task in build_stage27_training_tasks(output_root=root):
        summary_path = Path(task["study_dir"]) / "study_summary.json"
        if summary_path.exists():
            observed_scope = study_scope_from_summary_path(summary_path)
            if observed_scope.get("universe_scope") != task.get("universe_scope"):
                failed.append(task["tag"])
                results.append(
                    {
                        "tag": task["tag"],
                        "status": "blocked_existing_scope_mismatch",
                        "study_summary_json": str(summary_path),
                        "expected_universe_scope": task.get("universe_scope"),
                        "observed_scope": observed_scope,
                    }
                )
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
            root / "stage27_progress.json",
            _training_progress_payload(status="running" if not failed else "blocked", completed=completed, failed=failed),
        )
        if returncode != 0:
            break
    final_status = "completed" if not failed and len(completed) == len(build_stage27_training_tasks(output_root=root)) else "blocked"
    payload = {
        "schema_version": 1,
        "status": final_status,
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "completed_tags": completed,
        "failed_tags": failed,
        "results": results,
        "updated_at": _now(),
    }
    _write_json(root / "stage27_progress.json", _training_progress_payload(status=final_status, completed=completed, failed=failed))
    _write_json(root / "stage27_training_summary.json", payload)
    return payload


def _stage27_gate_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in profile_aggregate_rows(report)
        if row.get("role") == "test" and row.get("score_name") == "pred_decision_score"
    ]


def _stage27_gate_pass(row: dict[str, Any]) -> bool:
    return bool(
        int(row.get("seed_count", 0) or 0) >= 3
        and float(row.get("rank_ic_min", 0.0) or 0.0) > 0.0
        and float(row.get("spread_min", 0.0) or 0.0) > 0.0
        and float(row.get("hit_lift_min", 0.0) or 0.0) > 0.0
        and float(row.get("monthly_positive_rate_mean", 0.0) or 0.0) >= 0.75
        and int(row.get("negative_month_count_max", 99) or 99) <= 2
        and float(row.get("long_horizon_share_mean", 1.0) or 1.0) <= BASELINE_LONG_HORIZON_SHARE
        and float(row.get("thirty_d_concentration_mean", 1.0) or 1.0) <= MAX_THIRTY_D_CONCENTRATION
    )


def _stage27_research_verdict(report: dict[str, Any]) -> str:
    rows = sorted(_stage27_gate_rows(report), key=lambda item: float(item.get("rank_ic_mean", 0.0) or 0.0), reverse=True)
    passing = [row for row in rows if _stage27_gate_pass(row)]
    lines = [
        "# Alpha Multi-Horizon Stage 2.7 Target/Head Stability",
        "",
        f"- status: `{report.get('status', '')}`",
        f"- run_tag: `{report.get('run_tag', RUN_TAG)}`",
        f"- research_program: `{RESEARCH_PROGRAM}`",
        f"- study_family: `{STUDY_FAMILY}`",
        f"- gate: `{'pass' if passing else 'fail'}`",
        f"- baseline_long_horizon_share: `{BASELINE_LONG_HORIZON_SHARE:.6f}`",
        f"- max_thirty_d_concentration: `{MAX_THIRTY_D_CONCENTRATION:.6f}`",
        f"- boundary: {BOUNDARY}.",
        "",
        "## Candidate Results",
    ]
    for row in rows:
        lines.append(
            f"- `{row.get('loss_profile')}`: seeds=`{row.get('seed_count')}`, "
            f"rank_ic_min=`{float(row.get('rank_ic_min', 0.0) or 0.0):.6f}`, "
            f"spread_min=`{float(row.get('spread_min', 0.0) or 0.0):.6f}`, "
            f"hit_lift_min=`{float(row.get('hit_lift_min', 0.0) or 0.0):.6f}`, "
            f"monthly_positive_rate_mean=`{float(row.get('monthly_positive_rate_mean', 0.0) or 0.0):.6f}`, "
            f"negative_month_count_max=`{row.get('negative_month_count_max')}`, "
            f"long_horizon_share_mean=`{float(row.get('long_horizon_share_mean', 0.0) or 0.0):.6f}`, "
            f"thirty_d_concentration_mean=`{float(row.get('thirty_d_concentration_mean', 0.0) or 0.0):.6f}`, "
            f"stage3_gate=`{_stage27_gate_pass(row)}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Stage 2.7 tests target normalization and horizon-head stability under the fixed GRU/raw-input boundary.",
            "- A pass only unlocks Stage 3 architecture review; it does not authorize allocator, replay, paper, live/default, or promotion.",
            "- A fail means keep the line as research-only and consider a narrower 15/20/30d long-horizon utility family.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_stage27_comparison(output_root: str | Path | None = None) -> dict[str, Any]:
    root = _stage27_output_root(output_root)
    tasks = build_stage27_training_tasks(output_root=root)
    report = build_output_aux_profile_comparison([task["study_dir"] for task in tasks], run_tag=RUN_TAG)
    paths = write_output_aux_profile_comparison(report, root)
    verdict_path = root / "stage27_research_verdict.md"
    verdict_path.write_text(_stage27_research_verdict(report), encoding="utf-8")
    rows = _stage27_gate_rows(report)
    passing = [row for row in rows if _stage27_gate_pass(row)]
    payload = {
        "schema_version": 1,
        "status": report.get("status"),
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "baseline_long_horizon_share": BASELINE_LONG_HORIZON_SHARE,
        "max_thirty_d_concentration": MAX_THIRTY_D_CONCENTRATION,
        "stage3_architecture_allowed": bool(passing),
        "stage3_candidates": passing,
        "comparison_paths": {key: str(path) for key, path in paths.items()},
        "stage27_research_verdict_md": str(verdict_path),
        "updated_at": _now(),
    }
    _write_json(root / "stage27_comparison_summary.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2.7 multi-horizon target/head stability driver.")
    parser.add_argument("--output-root", default=str(STUDIES_ROOT / RUN_TAG))
    parser.add_argument("--write-task-list", action="store_true")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--run-comparison", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    outputs: dict[str, Any] = {"run_tag": RUN_TAG, "output_root": args.output_root}
    if args.write_task_list or not (args.run_training or args.run_comparison):
        outputs["task_list"] = str(write_stage27_task_list(args.output_root))
    if args.run_training:
        outputs["training_summary"] = run_training_tasks(output_root=args.output_root, skip_existing=not args.no_skip_existing)
    if args.run_comparison:
        outputs["comparison_summary"] = run_stage27_comparison(args.output_root)
    if args.json:
        print(json.dumps(outputs, ensure_ascii=False, indent=2))
    else:
        print(f"status=ok run_tag={RUN_TAG} output_root={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
