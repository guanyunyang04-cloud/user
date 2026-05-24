from __future__ import annotations

import json
import subprocess
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.brain.long_task_monitor import build_status, build_template


ROOT = Path(__file__).resolve().parents[3]
PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"


class LongTaskMonitorTest(unittest.TestCase):
    def test_template_uses_wait_process_and_not_fixed_sleep_polling(self) -> None:
        payload = build_template(timeout_seconds=7200)
        script = payload["powershell_template"]

        self.assertIn("Wait-Process -Id <pid> -Timeout 7200", script)
        self.assertNotIn("Start-Sleep", script)
        self.assertTrue(payload["eta_required"])

    def test_status_estimates_eta_from_step_progress(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            progress = root / "forecast_progress.json"
            stdout = root / "train.out"
            artifact_dir = root / "artifacts"
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

            payload = build_status(pid=999999, progress_path=progress, stdout_path=stdout, artifact_dir=artifact_dir)

        self.assertEqual(payload["progress_percent"], 40.0)
        self.assertEqual(payload["eta_status"], "estimated")
        self.assertGreater(payload["estimated_remaining_seconds"], 0)
        self.assertIn("eta_at", payload)
        self.assertIn("elapsed_seconds", payload)
        self.assertEqual(payload["latest_metric"]["validation_rank_ic"], 0.08)

    def test_status_reports_warming_up_when_progress_is_insufficient(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            progress = Path(raw_tmp) / "forecast_progress.json"
            progress.write_text(json.dumps({"current_step": 0, "total_steps": 100}), encoding="utf-8")

            payload = build_status(pid=999999, progress_path=progress)

        self.assertEqual(payload["eta_status"], "warming_up")
        self.assertIsNone(payload["estimated_remaining_seconds"])

    def test_status_reports_stalled_when_progress_marker_is_stale(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            progress = Path(raw_tmp) / "forecast_progress.json"
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

            payload = build_status(pid=999999, progress_path=progress, stale_after_seconds=3600)

        self.assertEqual(payload["eta_status"], "stalled_or_waiting")
        self.assertEqual(payload["decision"], "inspect_logs_or_resources")

    def test_cli_status_outputs_json(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            progress = Path(raw_tmp) / "forecast_progress.json"
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
                    "--progress",
                    str(progress),
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


if __name__ == "__main__":
    unittest.main()
