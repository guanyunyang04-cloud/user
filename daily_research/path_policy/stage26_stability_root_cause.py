from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from daily_research.path_policy.horizon_root_cause_audit import (
    build_stage26_root_cause_summary,
    run_audit_from_files,
    write_stage26_root_cause_summary_artifacts,
)
from daily_research.path_policy.output_aux_profile_comparison import (
    build_output_aux_profile_comparison,
    profile_aggregate_rows,
    write_output_aux_profile_comparison,
)


PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDIES_ROOT = PROJECT_ROOT / "daily_research/output/path_policy/studies"
RUN_TAG = "mh_stage26_stability_root_cause_20260527_01"
RESEARCH_PROGRAM = "alpha_multi_horizon_utility_policy_v1"
STUDY_FAMILY = "stage26_stability_root_cause"
DATASET_ID = "policy_input_bundle__7c8f58d851bce8179e1e9e2d"
SEEDS = (7, 11, 19)
FULLGRID = "1,2,3,5,8,10,15,20,30"
STAGE26_CANDIDATES = (
    "score_monthly_robust_v1",
    "horizon_entropy_regularized_v1",
    "risk_drawdown_reweighted_v1",
)
STAGE25_AUDIT_CANDIDATES = (
    ("fullgrid_rebudget", FULLGRID, 30),
    ("daily1_45_multiseed", ",".join(str(item) for item in range(1, 46)), 45),
)
BOUNDARY = "research-only / shadow-only; no active manifest, live/default, production root, paper, or broker integration"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def _stage26_output_root(output_root: str | Path | None = None) -> Path:
    return Path(output_root) if output_root is not None else STUDIES_ROOT / RUN_TAG


def selected_stage25_audit_tasks(*, output_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = _stage26_output_root(output_root)
    tasks: list[dict[str, Any]] = []
    for candidate, horizons, max_horizon in STAGE25_AUDIT_CANDIDATES:
        for seed in SEEDS:
            source_tag = f"mh25_path_aux_{candidate}_seed{seed}_20260526_01"
            source_dir = STUDIES_ROOT / source_tag
            audit_dir = root / "stage25_root_cause_audits" / source_tag
            tasks.append(
                {
                    "source_tag": source_tag,
                    "candidate": candidate,
                    "seed": int(seed),
                    "horizons": horizons,
                    "max_horizon": int(max_horizon),
                    "source_dir": str(source_dir),
                    "validation_predictions_csv": str(source_dir / "forecast_predictions_validation.csv"),
                    "test_predictions_csv": str(source_dir / "forecast_predictions_test.csv"),
                    "study_summary_json": str(source_dir / "study_summary.json"),
                    "training_summary_json": str(source_dir / "forecast_training_summary.json"),
                    "audit_output_dir": str(audit_dir),
                }
            )
    return tasks


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
        "--pool-name",
        "rolling_liquid500",
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


def build_stage26_training_tasks(*, output_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = _stage26_output_root(output_root)
    tasks: list[dict[str, Any]] = []
    for candidate in STAGE26_CANDIDATES:
        for seed in SEEDS:
            tag = f"mh26_{candidate}_fullgrid_seed{seed}_20260527_01"
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
                    "may_touch_active_manifest": False,
                }
            )
    return tasks


def write_stage26_task_list(output_root: str | Path | None = None) -> Path:
    root = _stage26_output_root(output_root)
    path = root / "stage26_task_list.json"
    payload = {
        "schema_version": 1,
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "created_at": _now(),
        "audit_task_count": len(selected_stage25_audit_tasks(output_root=root)),
        "training_task_count": len(build_stage26_training_tasks(output_root=root)),
        "audit_tasks": selected_stage25_audit_tasks(output_root=root),
        "training_tasks": build_stage26_training_tasks(output_root=root),
        "boundary": BOUNDARY,
    }
    _write_json(path, payload)
    return path


def run_stage25_root_cause_audits(output_root: str | Path | None = None) -> dict[str, Any]:
    root = _stage26_output_root(output_root)
    reports: list[dict[str, Any]] = []
    task_results: list[dict[str, Any]] = []
    for task in selected_stage25_audit_tasks(output_root=root):
        report = run_audit_from_files(
            validation_predictions_csv=task["validation_predictions_csv"],
            test_predictions_csv=task["test_predictions_csv"],
            output_dir=task["audit_output_dir"],
            study_summary_json=task["study_summary_json"],
            training_summary_json=task["training_summary_json"],
        )
        reports.append(report)
        task_results.append(
            {
                "source_tag": task["source_tag"],
                "status": report.get("status"),
                "audit_output_dir": task["audit_output_dir"],
                "conclusion_enums": report.get("conclusion_enums", []),
            }
        )
    summary = build_stage26_root_cause_summary(reports, run_tag=RUN_TAG)
    paths = write_stage26_root_cause_summary_artifacts(summary, root)
    payload = {
        "schema_version": 1,
        "status": summary.get("status"),
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "task_results": task_results,
        "summary_paths": {key: str(path) for key, path in paths.items()},
        "updated_at": _now(),
    }
    _write_json(root / "stage26_audit_summary.json", payload)
    return payload


def run_training_tasks(*, output_root: str | Path | None = None, skip_existing: bool = True) -> dict[str, Any]:
    root = _stage26_output_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    failed: list[str] = []
    results: list[dict[str, Any]] = []
    for task in build_stage26_training_tasks(output_root=root):
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
            root / "stage26_progress.json",
            _training_progress_payload(
                status="running" if not failed else "blocked",
                completed=completed,
                failed=failed,
            ),
        )
        if returncode != 0:
            break
    final_status = "completed" if not failed and len(completed) == len(build_stage26_training_tasks(output_root=root)) else "blocked"
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
    _write_json(root / "stage26_progress.json", _training_progress_payload(status=final_status, completed=completed, failed=failed))
    _write_json(root / "stage26_training_summary.json", payload)
    return payload


def _stage26_gate_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in profile_aggregate_rows(report)
        if row.get("role") == "test" and row.get("score_name") == "pred_decision_score"
    ]


def _stage26_research_verdict(report: dict[str, Any]) -> str:
    rows = sorted(_stage26_gate_rows(report), key=lambda item: float(item.get("rank_ic_mean", 0.0) or 0.0), reverse=True)
    passing = [row for row in rows if bool(row.get("stage3_weak_gate_pass"))]
    lines = [
        "# Alpha Multi-Horizon Stage 2.6 Stability Root-Cause Calibration",
        "",
        f"- status: `{report.get('status', '')}`",
        f"- run_tag: `{report.get('run_tag', RUN_TAG)}`",
        f"- research_program: `{RESEARCH_PROGRAM}`",
        f"- study_family: `{STUDY_FAMILY}`",
        f"- gate: `{'pass' if passing else 'fail'}`",
        "- boundary: research-only / shadow-only; no active manifest, live/default, production root, paper, or broker integration.",
        "",
        "## Candidate Results",
    ]
    for row in rows:
        lines.append(
            f"- `{row.get('loss_profile')}`: seeds=`{row.get('seed_count')}`, "
            f"rank_ic_mean=`{float(row.get('rank_ic_mean', 0.0) or 0.0):.6f}`, "
            f"spread_mean=`{float(row.get('spread_mean', 0.0) or 0.0):.6f}`, "
            f"hit_lift_mean=`{float(row.get('hit_lift_mean', 0.0) or 0.0):.6f}`, "
            f"monthly_positive_rate_mean=`{float(row.get('monthly_positive_rate_mean', 0.0) or 0.0):.6f}`, "
            f"negative_month_count_max=`{row.get('negative_month_count_max')}`, "
            f"long_horizon_share_mean=`{float(row.get('long_horizon_share_mean', 0.0) or 0.0):.6f}`, "
            f"stage3_gate=`{bool(row.get('stage3_weak_gate_pass'))}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Stage 2.6 only tests loss/score/root-cause calibration under the fixed GRU/raw-input boundary.",
            "- A pass only unlocks Stage 3 architecture review; it does not authorize allocator, replay, paper, live/default, or promotion.",
            "- A fail means continue target/loss/monthly stability work, not architecture expansion.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_stage26_comparison(output_root: str | Path | None = None) -> dict[str, Any]:
    root = _stage26_output_root(output_root)
    tasks = build_stage26_training_tasks(output_root=root)
    report = build_output_aux_profile_comparison([task["study_dir"] for task in tasks], run_tag=RUN_TAG)
    paths = write_output_aux_profile_comparison(report, root)
    verdict_path = root / "stage26_research_verdict.md"
    verdict_path.write_text(_stage26_research_verdict(report), encoding="utf-8")
    rows = _stage26_gate_rows(report)
    passing = [row for row in rows if bool(row.get("stage3_weak_gate_pass"))]
    payload = {
        "schema_version": 1,
        "status": report.get("status"),
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "stage3_architecture_allowed": bool(passing),
        "stage3_candidates": passing,
        "comparison_paths": {key: str(path) for key, path in paths.items()},
        "stage26_research_verdict_md": str(verdict_path),
        "updated_at": _now(),
    }
    _write_json(root / "stage26_comparison_summary.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2.6 multi-horizon root-cause and calibration driver.")
    parser.add_argument("--output-root", default=str(STUDIES_ROOT / RUN_TAG))
    parser.add_argument("--write-task-list", action="store_true")
    parser.add_argument("--run-audits", action="store_true")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--run-comparison", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    outputs: dict[str, Any] = {"run_tag": RUN_TAG, "output_root": args.output_root}
    if args.write_task_list or not (args.run_audits or args.run_training or args.run_comparison):
        outputs["task_list"] = str(write_stage26_task_list(args.output_root))
    if args.run_audits:
        outputs["audit_summary"] = run_stage25_root_cause_audits(args.output_root)
    if args.run_training:
        outputs["training_summary"] = run_training_tasks(output_root=args.output_root, skip_existing=not args.no_skip_existing)
    if args.run_comparison:
        outputs["comparison_summary"] = run_stage26_comparison(args.output_root)
    if args.json:
        print(json.dumps(outputs, ensure_ascii=False, indent=2))
    else:
        print(f"status=ok run_tag={RUN_TAG} output_root={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
