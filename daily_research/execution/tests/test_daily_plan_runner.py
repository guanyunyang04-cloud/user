from __future__ import annotations

from pathlib import Path
from typing import Any


def test_post_close_blocks_when_system_task_disabled(tmp_path: Path) -> None:
    from daily_research.execution import daily_plan_runner

    result = daily_plan_runner.run_daily_plan(
        mode="post-close",
        run_date="2026-05-26",
        runtime_root=tmp_path,
        scheduler_status_fn=lambda: {"installed": True, "enabled": False, "task_name": "DailyResearchDailyPlan"},
    )

    assert result["status"] == "blocked"
    assert result["blocker_code"] == "scheduler_disabled"
    assert result["stage_results"]["preflight"]["status"] == "blocked"
    verdict_path = Path(result["evidence_paths"]["verdict"])
    assert verdict_path.exists()
    assert tmp_path in verdict_path.parents


def test_market_daily_empty_blocks_before_trade_plan(tmp_path: Path) -> None:
    from daily_research.execution import daily_plan_runner

    launched_tasks: list[str] = []

    def task_runner(*, task_name: str, passthrough_args: list[str], job_label: str, force_unlock: bool) -> dict[str, Any]:
        launched_tasks.append(task_name)
        return {"status": "succeeded", "exit_code": 0, "metadata": {}}

    result = daily_plan_runner.run_daily_plan(
        mode="post-close",
        run_date="2026-05-26",
        runtime_root=tmp_path,
        scheduler_status_fn=lambda: {"installed": True, "enabled": True},
        latest_completed_trading_date_fn=lambda: "2026-05-26",
        readiness_fn=lambda **_: {
            "status": "blocked",
            "blocker_code": "data_not_ready",
            "candidate_date": "2026-05-26",
            "row_count": 0,
            "coverage_ratio": 0.0,
            "provider_error": "empty_canonical_market",
            "refresh_manifest_path": "H:/quant_project/daily_research/output/research_data_lake/data_platform/runs/failed/refresh_manifest.json",
        },
        task_runner=task_runner,
    )

    assert result["status"] == "blocked"
    assert result["blocker_code"] == "data_not_ready"
    assert result["target_trading_date"] == "2026-05-26"
    assert result["stage_results"]["data_readiness"]["row_count"] == 0
    assert launched_tasks == []
    assert result["trade_plan_run_dir"] == ""


def test_data_readiness_prefers_blocked_same_date_refresh_manifest(tmp_path: Path) -> None:
    from daily_research.execution import data_readiness

    runs_root = tmp_path / "runs"
    run_dir = runs_root / "refresh_daily_formal_free_v3_20260526_160417"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "refresh_manifest.json"
    manifest_path.write_text(
        """
{
  "status": "blocked",
  "as_of_date": "2026-05-26",
  "blockers": ["coverage_below_threshold", "empty_canonical_market", "required_domain_blocked:market_daily"],
  "coverage_report": {
    "expected_rows": 5208,
    "row_count": 0,
    "coverage_ratio": 0.0
  }
}
""".strip(),
        encoding="utf-8",
    )

    payload = data_readiness.resolve_provider_ready_trading_date(
        candidate_date="2026-05-26",
        runs_root=runs_root,
        provider_builder=lambda _: [],
    )

    assert payload["status"] == "blocked"
    assert payload["blocker_code"] == "data_not_ready"
    assert payload["row_count"] == 0
    assert payload["coverage_ratio"] == 0.0
    assert payload["refresh_manifest_path"] == str(manifest_path.resolve())


def test_daily_runner_does_not_fallback_complete_previous_ready_day(tmp_path: Path) -> None:
    from daily_research.execution import daily_plan_runner

    result = daily_plan_runner.run_daily_plan(
        mode="post-close",
        run_date="2026-05-26",
        runtime_root=tmp_path,
        scheduler_status_fn=lambda: {"installed": True, "enabled": True},
        latest_completed_trading_date_fn=lambda: "2026-05-26",
        readiness_fn=lambda **_: {
            "status": "blocked",
            "blocker_code": "data_not_ready",
            "candidate_date": "2026-05-26",
            "provider_ready_date": "2026-05-25",
            "row_count": 0,
            "coverage_ratio": 0.0,
        },
    )

    assert result["status"] == "blocked"
    assert result["target_trading_date"] == "2026-05-26"
    assert result["stage_results"]["data_readiness"]["provider_ready_date"] == "2026-05-25"
    assert result["dataset_id"] == ""
    assert result["blocker_code"] == "data_not_ready"


def test_success_verdict_contains_required_evidence(tmp_path: Path) -> None:
    from daily_research.execution import daily_plan_runner

    def task_runner(*, task_name: str, passthrough_args: list[str], job_label: str, force_unlock: bool) -> dict[str, Any]:
        if task_name == "data-platform-refresh":
            return {
                "status": "succeeded",
                "exit_code": 0,
                "refresh_dataset_id": "policy_input_bundle__20260526",
                "metadata": {
                    "refresh_dataset_id": "policy_input_bundle__20260526",
                    "refresh_manifest_path": "H:/refresh_manifest.json",
                    "evidence_paths": {"refresh_manifest": "H:/refresh_manifest.json"},
                },
            }
        if task_name == "refresh-production-live-panels":
            return {
                "status": "succeeded",
                "exit_code": 0,
                "metadata": {
                    "panel_latest_date": "2026-05-26",
                    "signal_refresh_manifest_path": "H:/signal_manifest.json",
                    "evidence_paths": {"signal_manifest": "H:/signal_manifest.json"},
                },
            }
        if task_name == "trade-plan":
            return {
                "status": "succeeded",
                "exit_code": 0,
                "metadata": {
                    "artifact_paths": {"run_dir": "H:/trade_plan_run"},
                    "evidence_paths": {"trade_plan_run_dir": "H:/trade_plan_run"},
                },
            }
        raise AssertionError(f"unexpected task: {task_name}")

    result = daily_plan_runner.run_daily_plan(
        mode="post-close",
        run_date="2026-05-26",
        runtime_root=tmp_path,
        scheduler_status_fn=lambda: {"installed": True, "enabled": True},
        latest_completed_trading_date_fn=lambda: "2026-05-26",
        readiness_fn=lambda **_: {
            "status": "ready",
            "candidate_date": "2026-05-26",
            "row_count": 3,
            "coverage_ratio": 1.0,
        },
        task_runner=task_runner,
        paper_reconcile_fn=lambda *, as_of_date: {"status": "ok", "as_of_date": as_of_date, "evidence_paths": {"paper": "H:/paper.sqlite3"}},
    )

    assert result["status"] == "completed"
    assert result["dataset_id"] == "policy_input_bundle__20260526"
    assert result["signal_panel_date"] == "2026-05-26"
    assert result["trade_plan_run_dir"] == "H:/trade_plan_run"
    assert result["paper_reconcile_status"] == "ok"
    assert result["evidence_paths"]["refresh_manifest"] == "H:/refresh_manifest.json"
    assert result["evidence_paths"]["signal_manifest"] == "H:/signal_manifest.json"
    assert result["evidence_paths"]["trade_plan_run_dir"] == "H:/trade_plan_run"
