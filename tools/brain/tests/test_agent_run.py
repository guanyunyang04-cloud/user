from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.brain.agent_run import build_run_paths, list_runs, register_run, run_status


ROOT = Path(__file__).resolve().parents[3]
PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"


class AgentRunTest(unittest.TestCase):
    def test_workspace_paths_use_brain_agent_runs_namespace(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            paths = build_run_paths(project_id="workspace", run_id="run_01", workspace_root=Path(raw_tmp))

        self.assertEqual(paths["run_root"], "brain/output/agent_runs/run_01")
        self.assertEqual(paths["pid_registry"], "brain/output/agent_runs/run_01/pid.json")
        self.assertNotIn("workspace/output/agent_runs", json.dumps(paths))

    def test_register_run_writes_project_scoped_registry_and_summary(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            payload = register_run(
                project_id="traditional_quant_research",
                run_id="run_01",
                pid=999999,
                command=["python", "-m", "example"],
                run_tag="tag_01",
                task="unit test run",
                workspace_root=root,
            )
            run_root = root / "traditional_quant_research/output/agent_runs/run_01"
            pid_payload = json.loads((run_root / "pid.json").read_text(encoding="utf-8"))
            summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
            listed = list_runs(project_id="traditional_quant_research", workspace_root=root)

            self.assertEqual(payload["status"], "ok")
            self.assertEqual(pid_payload["project_id"], "traditional_quant_research")
            self.assertEqual(summary["run_id"], "run_01")
            self.assertTrue((run_root / "stdout.log").exists())
            self.assertTrue((run_root / "stderr.log").exists())
            self.assertEqual(listed["run_count"], 1)
            self.assertEqual(listed["runs"][0]["run_id"], "run_01")
            self.assertTrue(listed["runs"][0]["has_summary"])

    def test_run_status_reads_only_registered_project_run(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            register_run(
                project_id="traditional_quant_research",
                run_id="run_01",
                pid=999999,
                workspace_root=root,
            )
            progress = root / "traditional_quant_research/output/agent_runs/run_01/progress.json"
            progress.write_text(json.dumps({"current_step": 1, "total_steps": 2}), encoding="utf-8")

            payload = run_status(project_id="traditional_quant_research", run_id="run_01", workspace_root=root)

        self.assertEqual(payload["namespace"]["status"], "ok")
        self.assertEqual(payload["progress_percent"], 50.0)
        self.assertEqual(payload["pid_registry"]["pid"], 999999)

    def test_cli_register_respects_explicit_workspace_root(self) -> None:
        with TemporaryDirectory() as raw_tmp:
            result = subprocess.run(
                [
                    PYTHON,
                    "-m",
                    "tools.brain.agent_run",
                    "register",
                    "--project-id",
                    "traditional_quant_research",
                    "--run-id",
                    "run_01",
                    "--pid",
                    "999999",
                    "--workspace-root",
                    raw_tmp,
                    "--json",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            payload = json.loads(result.stdout)
            pid_path = Path(raw_tmp) / "traditional_quant_research/output/agent_runs/run_01/pid.json"

            self.assertEqual(payload["status"], "ok")
            self.assertTrue(pid_path.exists())


if __name__ == "__main__":
    unittest.main()
