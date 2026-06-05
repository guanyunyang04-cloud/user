from __future__ import annotations

import json
import subprocess
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.brain.long_task_monitor import (
    build_status,
    build_template,
    build_trace_event,
    validate_project_namespace,
    validate_resource_lease,
)


ROOT = Path(__file__).resolve().parents[3]
PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"


class LongTaskMonitorTest(unittest.TestCase):
    def test_template_uses_gpu_active_wait_process_window(self) -> None:
        payload = build_template(timeout_seconds=7200)
        script = payload["powershell_template"]

        self.assertEqual(payload["poll_window_seconds"], 7200)
        self.assertEqual(payload["monitoring_mode"], "foreground_wait_process_after_gpu_start")
        self.assertEqual(payload["required_wait_command"], "Wait-Process -Id <pid> -Timeout 7200")
        self.assertIn("Wait-Process -Id $taskPid -Timeout 7200", script)
        self.assertIn("$taskPid = $proc.Id", script)
        self.assertIn("--pid $taskPid", script)
        self.assertIn("--project-id <project_id>", script)
        self.assertIn("--run-id <run_id>", script)
        self.assertNotIn("$pid = $proc.Id", script)
        self.assertNotIn("--pid $pid", script)
        self.assertNotIn("Start-Sleep", script)
        self.assertTrue(payload["eta_required"])

    def test_status_estimates_eta_from_step_progress(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp) / "traditional_quant_research/output/agent_runs/run_01"
            progress = root / "progress.json"
            stdout = root / "stdout.log"
            artifact_dir = root / "artifacts"
            root.mkdir(parents=True)
            artifact_dir.mkdir()
            started_at = datetime.now(timezone.utc) - timedelta(minutes=20)
            progress.write_text(
                json.dumps(
                    {
                        "started_at": started_at.isoformat(),
                        "current_step": 40,
                        "total_steps": 100,
                        "stage": "forecast_train",
                        "latest_metric": {"validation_rank_ic": 0.08},
                    }
                ),
                encoding="utf-8",
            )
            stdout.write_text("epoch 1\nrank_ic=0.08\n", encoding="utf-8")
            (artifact_dir / "checkpoint.pt").write_text("x", encoding="utf-8")

            payload = build_status(
                pid=999999,
                project_id="traditional_quant_research",
                run_id="run_01",
                progress_path=progress,
                stdout_path=stdout,
                artifact_dir=artifact_dir,
                workspace_root=Path(raw_tmp),
            )

        self.assertEqual(payload["progress_percent"], 40.0)
        self.assertEqual(payload["eta_status"], "estimated")
        self.assertGreater(payload["estimated_remaining_seconds"], 0)
        self.assertIn("eta_at", payload)
        self.assertIn("elapsed_seconds", payload)
        self.assertEqual(payload["latest_metric"]["validation_rank_ic"], 0.08)

    def test_status_reports_warming_up_when_progress_is_insufficient(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            run_root = Path(raw_tmp) / "traditional_quant_research/output/agent_runs/run_01"
            run_root.mkdir(parents=True)
            progress = run_root / "progress.json"
            progress.write_text(json.dumps({"current_step": 0, "total_steps": 100}), encoding="utf-8")

            payload = build_status(
                pid=999999,
                project_id="traditional_quant_research",
                run_id="run_01",
                progress_path=progress,
                workspace_root=Path(raw_tmp),
            )

        self.assertEqual(payload["eta_status"], "warming_up")
        self.assertIsNone(payload["estimated_remaining_seconds"])

    def test_status_reports_stalled_when_progress_marker_is_stale(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            run_root = Path(raw_tmp) / "traditional_quant_research/output/agent_runs/run_01"
            run_root.mkdir(parents=True)
            progress = run_root / "progress.json"
            stale_time = datetime.now(timezone.utc) - timedelta(hours=3)
            progress.write_text(
                json.dumps(
                    {
                        "started_at": (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat(),
                        "updated_at": stale_time.isoformat(),
                        "current_step": 10,
                        "total_steps": 100,
                    }
                ),
                encoding="utf-8",
            )

            payload = build_status(
                pid=999999,
                project_id="traditional_quant_research",
                run_id="run_01",
                progress_path=progress,
                stale_after_seconds=3600,
                workspace_root=Path(raw_tmp),
            )

        self.assertEqual(payload["eta_status"], "stalled_or_waiting")
        self.assertEqual(payload["decision"], "inspect_logs_or_resources")

    def test_running_pid_decision_uses_short_polling_language(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            run_root = Path(raw_tmp) / "traditional_quant_research/output/agent_runs/run_01"
            run_root.mkdir(parents=True)
            progress = run_root / "progress.json"
            progress.write_text(json.dumps({"current_step": 0, "total_steps": 100}), encoding="utf-8")
            with unittest.mock.patch("tools.brain.long_task_monitor._pid_alive", return_value=True):
                payload = build_status(
                    pid=1234,
                    project_id="traditional_quant_research",
                    run_id="run_01",
                    progress_path=progress,
                    workspace_root=Path(raw_tmp),
                )

        self.assertEqual(payload["decision"], "continue_short_polling")

    def test_cli_status_outputs_json(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            run_root = Path(raw_tmp) / "traditional_quant_research/output/agent_runs/run_01"
            run_root.mkdir(parents=True)
            progress = run_root / "progress.json"
            started_at = datetime.now(timezone.utc) - timedelta(minutes=10)
            progress.write_text(
                json.dumps({"started_at": started_at.isoformat(), "current_step": 2, "total_steps": 4}),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    PYTHON,
                    "-m",
                    "tools.brain.long_task_monitor",
                    "status",
                    "--pid",
                    "999999",
                    "--project-id",
                    "traditional_quant_research",
                    "--run-id",
                    "run_01",
                    "--progress",
                    str(progress),
                    "--workspace-root",
                    str(raw_tmp),
                    "--json",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(payload["progress_percent"], 50.0)
        self.assertIn("estimated_remaining_seconds", payload)

    def test_cli_status_blocks_anonymous_process_monitoring(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            progress = Path(raw_tmp) / "progress.json"
            progress.write_text(json.dumps({"current_step": 1, "total_steps": 2}), encoding="utf-8")
            result = subprocess.run(
                [
                    PYTHON,
                    "-m",
                    "tools.brain.long_task_monitor",
                    "status",
                    "--pid",
                    "999999",
                    "--progress",
                    str(progress),
                    "--json",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(result.returncode, 2)
        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["reason"], "project_run_identity_required")

    def test_trace_event_records_structured_polling_evidence(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp) / "traditional_quant_research/output/agent_runs/run_01"
            progress = root / "progress.json"
            stdout = root / "stdout.log"
            stderr = root / "stderr.log"
            artifact_dir = root / "artifacts"
            root.mkdir(parents=True)
            artifact_dir.mkdir()
            started_at = datetime.now(timezone.utc) - timedelta(minutes=20)
            progress.write_text(
                json.dumps({"started_at": started_at.isoformat(), "current_step": 2, "total_steps": 4}),
                encoding="utf-8",
            )
            stdout.write_text("epoch 1\n", encoding="utf-8")
            stderr.write_text("", encoding="utf-8")
            (artifact_dir / "checkpoint.pt").write_text("x", encoding="utf-8")

            event = build_trace_event(
                task="train patch seed19",
                project_id="traditional_quant_research",
                run_id="run_01",
                step_id="stage32_final",
                run_tag="mh_stage32_arch_input_final_confirmation_20260528_01",
                pid=999999,
                child_pids=[111, 222],
                poll_window_seconds=7200,
                progress_path=progress,
                stdout_path=stdout,
                stderr_path=stderr,
                artifact_dir=artifact_dir,
                final_verification="pending",
                workspace_root=Path(raw_tmp),
            )

        self.assertEqual(event["type"], "long_task_poll")
        self.assertEqual(event["step_id"], "stage32_final")
        self.assertEqual(event["pid"], 999999)
        self.assertEqual(event["child_pids"], [111, 222])
        self.assertEqual(event["poll_window_seconds"], 7200)
        self.assertEqual(event["eta_status"], "estimated")
        self.assertEqual(event["eta_no_eta_reason"], "")
        self.assertEqual(event["final_verification"], "pending")
        self.assertIn("checkpoint.pt", [item["name"] for item in event["artifact_summary"]])

    def test_cli_trace_poll_appends_review_trace_event(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp) / "traditional_quant_research/output/agent_runs/run_01"
            root.mkdir(parents=True)
            progress = root / "progress.json"
            trace_json = root / "long_task_trace.json"
            started_at = datetime.now(timezone.utc) - timedelta(minutes=10)
            progress.write_text(
                json.dumps({"started_at": started_at.isoformat(), "current_step": 1, "total_steps": 2}),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    PYTHON,
                    "-m",
                    "tools.brain.long_task_monitor",
                    "trace-poll",
                    "--trace-json",
                    str(trace_json),
                    "--task",
                    "long training",
                    "--project-id",
                    "traditional_quant_research",
                    "--run-id",
                    "run_01",
                    "--step-id",
                    "train",
                    "--run-tag",
                    "run_01",
                    "--pid",
                    "999999",
                    "--poll-window-seconds",
                    "7200",
                    "--progress",
                    str(progress),
                    "--workspace-root",
                    str(raw_tmp),
                    "--final-verification",
                    "pending",
                    "--json",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            payload = json.loads(result.stdout)
            trace_payload = json.loads(trace_json.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(trace_payload["task"], "long training")
        self.assertEqual(trace_payload["events"][0]["type"], "long_task_poll")
        self.assertEqual(trace_payload["events"][0]["run_tag"], "run_01")

    def test_cli_trace_poll_blocks_trace_writes_outside_project_namespace(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            run_root = Path(raw_tmp) / "traditional_quant_research/output/agent_runs/run_01"
            run_root.mkdir(parents=True)
            progress = run_root / "progress.json"
            progress.write_text(json.dumps({"current_step": 1, "total_steps": 2}), encoding="utf-8")
            trace_json = Path(raw_tmp) / "external_trace.json"
            result = subprocess.run(
                [
                    PYTHON,
                    "-m",
                    "tools.brain.long_task_monitor",
                    "trace-poll",
                    "--trace-json",
                    str(trace_json),
                    "--project-id",
                    "traditional_quant_research",
                    "--run-id",
                    "run_01",
                    "--pid",
                    "999999",
                    "--progress",
                    str(progress),
                    "--workspace-root",
                    str(raw_tmp),
                    "--json",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(result.returncode, 2)
        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["reason"], "project_namespace_violation")

    def test_project_namespace_accepts_own_agent_run_paths(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            run_root = Path(raw_tmp) / "traditional_quant_research/output/agent_runs/run_01"
            progress = run_root / "progress.json"
            stdout = run_root / "stdout.log"
            run_root.mkdir(parents=True)
            progress.write_text("{}", encoding="utf-8")
            stdout.write_text("", encoding="utf-8")

            payload = validate_project_namespace(
                project_id="traditional_quant_research",
                run_id="run_01",
                paths=[progress, stdout],
                workspace_root=Path(raw_tmp),
            )

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["namespace_root"], "traditional_quant_research/output/agent_runs/run_01")

    def test_project_namespace_blocks_other_project_paths_without_cross_project_lease(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            progress = Path(raw_tmp) / "daily_research/output/agent_runs/run_01/progress.json"
            progress.parent.mkdir(parents=True)
            progress.write_text("{}", encoding="utf-8")

            payload = validate_project_namespace(
                project_id="traditional_quant_research",
                run_id="run_01",
                paths=[progress],
                workspace_root=Path(raw_tmp),
            )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["reason"], "project_namespace_violation")
        self.assertIn("daily_research/output/agent_runs/run_01/progress.json", payload["violating_paths"])

    def test_project_namespace_allows_other_project_paths_with_active_lease(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            lease_root = root / "brain/output/resource_leases"
            lease_root.mkdir(parents=True)
            (lease_root / "lease_01.json").write_text(
                json.dumps(
                    {
                        "status": "active",
                        "requester_project_id": "traditional_quant_research",
                        "target_project_id": "daily_research",
                    }
                ),
                encoding="utf-8",
            )
            progress = root / "daily_research/output/agent_runs/run_01/progress.json"
            progress.parent.mkdir(parents=True)
            progress.write_text("{}", encoding="utf-8")

            payload = validate_project_namespace(
                project_id="traditional_quant_research",
                run_id="run_01",
                paths=[progress],
                workspace_root=root,
                allow_cross_project=True,
                lease_id="lease_01",
            )

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["lease"]["status"], "ok")

    def test_resource_lease_blocks_wrong_requester(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            lease_root = Path(raw_tmp) / "brain/output/resource_leases"
            lease_root.mkdir(parents=True)
            (lease_root / "lease_01.json").write_text(
                json.dumps(
                    {
                        "status": "active",
                        "requester_project_id": "daily_research",
                        "target_project_id": "traditional_quant_research",
                    }
                ),
                encoding="utf-8",
            )

            payload = validate_resource_lease(
                lease_id="lease_01",
                project_id="traditional_quant_research",
                target_project_ids=["daily_research"],
                workspace_root=Path(raw_tmp),
            )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["reason"], "lease_requester_not_authorized")

    def test_status_does_not_read_other_project_namespace(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            run_root = Path(raw_tmp) / "daily_research/output/agent_runs/run_01"
            run_root.mkdir(parents=True)
            progress = run_root / "progress.json"
            stdout = run_root / "stdout.log"
            progress.write_text(json.dumps({"current_step": 1, "total_steps": 2}), encoding="utf-8")
            stdout.write_text("external project log\n", encoding="utf-8")

            payload = build_status(
                project_id="traditional_quant_research",
                run_id="run_01",
                progress_path=progress,
                stdout_path=stdout,
                workspace_root=Path(raw_tmp),
            )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["reason"], "project_namespace_violation")
        self.assertEqual(payload["eta_status"], "namespace_blocked")
        self.assertEqual(payload["decision"], "project_namespace_violation")
        self.assertEqual(payload["last_log_lines"], [])

    def test_wait_once_blocks_other_project_namespace_before_waiting(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            progress = Path(raw_tmp) / "daily_research/output/agent_runs/run_01/progress.json"
            progress.parent.mkdir(parents=True)
            progress.write_text("{}", encoding="utf-8")

            result = subprocess.run(
                [
                    PYTHON,
                    "-m",
                    "tools.brain.long_task_monitor",
                    "wait-once",
                    "--pid",
                    "999999",
                    "--project-id",
                    "traditional_quant_research",
                    "--run-id",
                    "run_01",
                    "--progress",
                    str(progress),
                    "--json",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(result.returncode, 2)
        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["reason"], "project_namespace_violation")

    def test_trace_event_records_project_namespace(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            run_root = Path(raw_tmp) / "traditional_quant_research/output/agent_runs/run_01"
            run_root.mkdir(parents=True)
            progress = run_root / "progress.json"
            progress.write_text(json.dumps({"current_step": 1, "total_steps": 2}), encoding="utf-8")

            event = build_trace_event(
                task="traditional quant run",
                project_id="traditional_quant_research",
                run_id="run_01",
                pid=999999,
                progress_path=progress,
                workspace_root=Path(raw_tmp),
            )

        self.assertEqual(event["project_id"], "traditional_quant_research")
        self.assertEqual(event["run_id"], "run_01")
        self.assertEqual(event["namespace"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
