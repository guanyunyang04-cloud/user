from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"


def run_cli(*args: str) -> dict:
    result = subprocess.run(
        [PYTHON, "-m", "daily_research.tools.brain_workflow", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(result.stdout)


def run_script_cli(*args: str) -> dict:
    result = subprocess.run(
        [PYTHON, "daily_research/tools/brain_workflow.py", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(result.stdout)


class BrainWorkflowCliTest(unittest.TestCase):
    def test_handoff_cli_outputs_valid_json_capsule(self) -> None:
        payload = run_cli("handoff", "--child", "daily_research", "--json")

        self.assertEqual(payload["child_brain"], "daily_research")
        self.assertIn("boot_order", payload)
        self.assertIn("artifact_freshness", payload)
        self.assertIn("workflow_hints", payload)

    def test_existing_brain_bootstrap_json_includes_platform_fields(self) -> None:
        result = subprocess.run(
            [PYTHON, "daily_research/tools/brain_bootstrap.py", "--child", "daily_research", "--json"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["child_brain"], "daily_research")
        self.assertIn("artifact_freshness", payload)
        self.assertIn("workflow_hints", payload)
        self.assertIn("encoding_report", payload)

    def test_health_cli_aggregates_read_only_checks(self) -> None:
        payload = run_cli("health", "--json")

        self.assertEqual(payload["status"], "ok")
        self.assertIn("brain_integrity", payload["checks"])
        self.assertIn("doc_guard", payload["checks"])
        self.assertIn("project_consistency", payload["checks"])
        self.assertIn("openmp_strict", payload["checks"])

    def test_preflight_cli_does_not_create_study_directories(self) -> None:
        studies_root = ROOT / "daily_research/output/continuous_policy/studies"
        before = {path.name for path in studies_root.iterdir()} if studies_root.exists() else set()

        payload = run_cli("preflight", "--workflow", "continuous_policy_safe_screening", "--json")

        after = {path.name for path in studies_root.iterdir()} if studies_root.exists() else set()
        self.assertEqual(before, after)
        self.assertEqual(payload["workflow_id"], "continuous_policy_safe_screening")
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["writes_tracked_files"])

    def test_writeback_plan_cli_only_returns_routes_by_default(self) -> None:
        payload = run_cli("writeback-plan", "--source", "latest", "--json")

        self.assertEqual(payload["source"], "latest")
        self.assertFalse(payload["apply_brain_writeback"])
        self.assertIn("routes", payload)
        self.assertIn("state", payload["routes"])

    def test_status_cli_accepts_explicit_study_tag_capsule(self) -> None:
        tag = "self_opt_study_r52_native_source_delta_closure_screening_safe_20260510_02"
        payload = run_cli("status", "--workflow", "continuous_policy", "--study-tag", tag, "--json")

        self.assertEqual(payload["workflow_id"], "continuous_policy_result_review")
        self.assertIn("study_evidence", payload)
        self.assertEqual(payload["study_evidence"]["study_tag"], tag)
        self.assertEqual(payload["study_evidence"]["completed_trial_count"], 3)
        self.assertTrue(payload["artifact_freshness"]["is_stale_risk"])

    def test_writeback_plan_study_source_is_read_only(self) -> None:
        tag = "self_opt_study_r52_native_source_delta_closure_screening_safe_20260510_02"
        payload = run_cli("writeback-plan", "--source", f"study:{tag}", "--json")

        self.assertEqual(payload["source"], f"study:{tag}")
        self.assertFalse(payload["apply_brain_writeback"])
        self.assertEqual(payload["study_evidence"]["study_tag"], tag)
        self.assertTrue(payload["requires_explicit_apply"])

    def test_script_path_cli_imports_package_root(self) -> None:
        payload = run_script_cli("status", "--workflow", "continuous_policy", "--json")

        self.assertEqual(payload["workflow_id"], "continuous_policy_result_review")
        self.assertIn("artifact_freshness", payload)

    def test_write_output_cli_writes_only_brain_workflow_output(self) -> None:
        payload = run_cli("status", "--workflow", "continuous_policy", "--json", "--write-output")
        output_path = Path(payload["output_path"])

        self.assertTrue(output_path.exists())
        self.assertIn("daily_research", output_path.parts)
        self.assertIn("output", output_path.parts)
        self.assertIn("brain_workflow", output_path.parts)


if __name__ == "__main__":
    unittest.main()
