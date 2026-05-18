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
        self.assertIn("elapsed_seconds", payload)
        self.assertIn("brain_integrity", payload["checks"])
        self.assertIn("doc_guard", payload["checks"])
        self.assertIn("project_consistency", payload["checks"])
        self.assertIn("openmp_strict", payload["checks"])
        for check_payload in payload["checks"].values():
            self.assertIn("elapsed_seconds", check_payload)

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

    def test_workflow_registry_contains_brain_native_superpowers(self) -> None:
        registry = json.loads((ROOT / "daily_research/brain/workflow_registry.json").read_text(encoding="utf-8"))
        expected = {
            "brainstorming_design",
            "writing_plan",
            "executing_plan",
            "systematic_debugging",
            "verification_before_completion",
            "brain_writeback_verified",
        }

        self.assertTrue(expected.issubset(registry))
        for workflow_id in expected:
            entry = registry[workflow_id]
            self.assertIn("description", entry)
            self.assertIn("preflight", entry)
            self.assertIn("forbidden_actions", entry)
            self.assertIn("completion", entry)
            self.assertIn("checklist", entry)
            self.assertIn("stop_conditions", entry)

    def test_workflow_guide_cli_outputs_checklist(self) -> None:
        payload = run_cli("workflow-guide", "--workflow", "executing_plan", "--json")

        self.assertEqual(payload["workflow_id"], "executing_plan")
        self.assertIn("description", payload)
        self.assertIn("checklist", payload)
        self.assertIn("stop_conditions", payload)
        self.assertIn("validation_commands", payload)
        self.assertIn("writeback_routes", payload)

    def test_select_workflow_cli_maps_task_intent(self) -> None:
        cases = {
            "继续实施计划": "executing_plan",
            "报错了帮我查": "systematic_debugging",
            "是否全部完成": "verification_before_completion",
            "深入思考详细计划": "writing_plan",
            "更新脑区和 evidence registry": "brain_writeback_verified",
        }
        for task, expected in cases.items():
            with self.subTest(task=task):
                payload = run_cli("select-workflow", "--task", task, "--json")
                self.assertEqual(payload["selected_workflow"], expected)
                self.assertIn("reason", payload)

    def test_capsule_auto_workflow_embeds_guide(self) -> None:
        payload = run_cli(
            "capsule",
            "--child",
            "daily_research",
            "--task",
            "继续实施计划",
            "--workflow",
            "auto",
            "--json",
        )

        self.assertEqual(payload["selected_workflow"], "executing_plan")
        self.assertEqual(payload["workflow"], "executing_plan")
        self.assertIn("workflow_guide", payload)
        self.assertIn("required_checklist", payload)
        self.assertIn("stop_conditions", payload)
        self.assertEqual(payload["workflow_guide"]["workflow_id"], "executing_plan")


if __name__ == "__main__":
    unittest.main()
