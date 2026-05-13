import json
import inspect
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from daily_research.continuous_policy.run_self_optimizing_study import (
    _run_protocol_with_progress,
    _run_subprocess_with_watchdog,
)


class ProtocolRuntimeWatchdogTest(unittest.TestCase):
    def test_subprocess_without_progress_is_stopped_as_no_progress_timeout(self) -> None:
        with TemporaryDirectory() as temp_dir:
            study_root = Path(temp_dir) / "study"
            protocol_root = Path(temp_dir) / "protocol"
            result = _run_subprocess_with_watchdog(
                command=[sys.executable, "-c", "import time; print('started', flush=True); time.sleep(5)"],
                cwd=Path.cwd(),
                study_root=study_root,
                trial_tag="study__trial_01",
                protocol_progress_json=protocol_root / "protocol_progress.json",
                stdout_log_path=study_root / "protocol_logs" / "study__trial_01.stdout.log",
                stderr_log_path=study_root / "protocol_logs" / "study__trial_01.stderr.log",
                max_wall_seconds=5.0,
                max_stale_seconds=1.00,
                progress_grace_seconds=0.0,
                poll_interval_seconds=0.02,
            )
            failure = json.loads((protocol_root / "runtime_failure_summary.json").read_text(encoding="utf-8"))
            stdout_text = (study_root / "protocol_logs" / "study__trial_01.stdout.log").read_text(encoding="utf-8")

        self.assertEqual(result["exit_code"], 124)
        self.assertEqual(result["runtime_failure_reason"], "no_progress_timeout")
        self.assertEqual(failure["runtime_failure_reason"], "no_progress_timeout")
        self.assertIn("started", stdout_text)

    def test_subprocess_with_fresh_progress_is_not_stale_stopped(self) -> None:
        script = (
            "import json, pathlib, sys, time\n"
            "path = pathlib.Path(sys.argv[1])\n"
            "path.parent.mkdir(parents=True, exist_ok=True)\n"
            "for idx in range(3):\n"
            "    path.write_text(json.dumps({'event': 'tick', 'epoch': idx}), encoding='utf-8')\n"
            "    print(f'tick {idx}', flush=True)\n"
            "    time.sleep(0.03)\n"
        )
        with TemporaryDirectory() as temp_dir:
            study_root = Path(temp_dir) / "study"
            protocol_root = Path(temp_dir) / "protocol"
            result = _run_subprocess_with_watchdog(
                command=[sys.executable, "-c", script, str(protocol_root / "protocol_progress.json")],
                cwd=Path.cwd(),
                study_root=study_root,
                trial_tag="study__trial_01",
                protocol_progress_json=protocol_root / "protocol_progress.json",
                stdout_log_path=study_root / "protocol_logs" / "study__trial_01.stdout.log",
                stderr_log_path=study_root / "protocol_logs" / "study__trial_01.stderr.log",
                max_wall_seconds=5.0,
                max_stale_seconds=0.20,
                progress_grace_seconds=0.0,
                poll_interval_seconds=0.02,
            )
            stdout_text = (study_root / "protocol_logs" / "study__trial_01.stdout.log").read_text(encoding="utf-8")

        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["runtime_failure_reason"], "")
        self.assertIn("tick 2", stdout_text)

    def test_wall_timeout_writes_runtime_failure_summary(self) -> None:
        with TemporaryDirectory() as temp_dir:
            study_root = Path(temp_dir) / "study"
            protocol_root = Path(temp_dir) / "protocol"
            result = _run_subprocess_with_watchdog(
                command=[sys.executable, "-c", "import time; time.sleep(5)"],
                cwd=Path.cwd(),
                study_root=study_root,
                trial_tag="study__trial_01",
                protocol_progress_json=protocol_root / "protocol_progress.json",
                stdout_log_path=study_root / "protocol_logs" / "study__trial_01.stdout.log",
                stderr_log_path=study_root / "protocol_logs" / "study__trial_01.stderr.log",
                max_wall_seconds=0.05,
                max_stale_seconds=10.0,
                progress_grace_seconds=10.0,
                poll_interval_seconds=0.02,
            )
            failure = json.loads((protocol_root / "runtime_failure_summary.json").read_text(encoding="utf-8"))

        self.assertEqual(result["exit_code"], 124)
        self.assertEqual(result["runtime_failure_reason"], "wall_timeout")
        self.assertEqual(failure["exit_code"], 124)

    def test_protocol_failure_emit_does_not_pass_duplicate_exit_code(self) -> None:
        source = inspect.getsource(_run_protocol_with_progress)

        self.assertNotIn('exit_code=exit_code, **result', source)


if __name__ == "__main__":
    unittest.main()
