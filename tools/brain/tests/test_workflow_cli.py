from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

from tools.brain.platform import load_workflow_registry
from tools.brain.tests.workflow_cli_helpers import PYTHON, ROOT, run_cli, run_script_cli


REMOVED_WORKFLOWS = {
    "brainstorming_" + "design",
    "writing_" + "plan",
    "executing_" + "plan",
    "systematic_" + "debugging",
    "verification_before_" + "completion",
    "long_task",
}
REMOVED_CLI_COMMANDS = (
    "handoff",
    "preflight",
    "workflow-guide",
    "select-workflow",
    "audit-brain",
)


class BrainWorkflowCliTest(unittest.TestCase):
    def test_bootstrap_cli_outputs_valid_json_capsule(self) -> None:
        payload = run_cli("bootstrap", "--brain", "daily_research", "--json")

        self.assertEqual(payload["target_kind"], "child")
        self.assertEqual(payload["workflow_domain"], "daily_research")
        self.assertEqual(payload["child_brain"], "daily_research")
        self.assertIn("boot_order", payload)
        self.assertIn("artifact_freshness", payload)
        self.assertIn("workflow_hints", payload)

    def test_workspace_bootstrap_json_includes_platform_fields(self) -> None:
        payload = run_cli("bootstrap", "--brain", "workspace", "--json")

        self.assertEqual(payload["main_manifest"], "brain/brain_manifest.json")
        self.assertIn("artifact_freshness", payload)
        self.assertIn("workflow_hints", payload)
        self.assertIn("encoding_report", payload)
        self.assertEqual(payload["target_kind"], "workspace")
        self.assertEqual(payload["workflow_domain"], "workspace_governance")

    def test_workspace_governance_bootstrap_alias_outputs_workspace(self) -> None:
        payload = run_cli("bootstrap", "--brain", "workspace_governance", "--json")

        self.assertEqual(payload["target_kind"], "workspace")
        self.assertEqual(payload["workflow_domain"], "workspace_governance")
        self.assertEqual(payload["child_brain"], "")
        self.assertEqual(payload["main_manifest"], "brain/brain_manifest.json")

    def test_old_daily_research_brain_workflow_module_fails(self) -> None:
        result = subprocess.run(
            [PYTHON, "-m", "daily_research.tools.brain_workflow", "capsule", "--child", "daily_research", "--json"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)

    def test_health_cli_defaults_to_compact_checks(self) -> None:
        payload = run_cli("health", "--json")

        self.assertIn(payload["status"], {"ok", "failed"})
        self.assertEqual(payload["mode"], "compact")
        self.assertIn("elapsed_seconds", payload)
        self.assertIn("brain_integrity", payload["checks"])
        self.assertNotIn("doc_guard", payload["checks"])
        self.assertNotIn("project_consistency", payload["checks"])
        self.assertNotIn("openmp_strict", payload["checks"])
        for check_payload in payload["checks"].values():
            self.assertIn("elapsed_seconds", check_payload)

        daily_payload = run_cli("health", "--brain", "daily_research", "--json")
        self.assertIn(daily_payload["status"], {"ok", "failed"})
        self.assertEqual(daily_payload["mode"], "compact")
        self.assertIn("brain_integrity", daily_payload["checks"])
        self.assertNotIn("project_consistency", daily_payload["checks"])
        self.assertNotIn("openmp_strict", daily_payload["checks"])
        for check_payload in daily_payload["checks"].values():
            self.assertIn("elapsed_seconds", check_payload)

    def test_removed_cli_commands_are_not_registered(self) -> None:
        for command in REMOVED_CLI_COMMANDS:
            with self.subTest(command=command):
                result = subprocess.run(
                    [PYTHON, "-m", "tools.brain.workflow", command, "--help"],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=False,
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid choice", result.stderr)

    def test_status_cli_accepts_explicit_run_tag_capsule(self) -> None:
        tag = "missing_continuous_policy_fixture_20990101_01"
        payload = run_cli("status", "--workflow", "continuous_policy", "--run-tag", tag, "--json")

        self.assertEqual(payload["workflow_id"], "continuous_policy_result_review")
        self.assertEqual(payload["status"], "blocked")
        self.assertIn("run_evidence", payload)
        self.assertNotIn("study_evidence", payload)
        self.assertEqual(payload["run_evidence"]["run_tag"], tag)
        self.assertEqual(payload["run_evidence"]["workflow"], "continuous_policy")
        self.assertFalse(payload["run_evidence"]["exists"])
        self.assertIn("study summary not found", payload["run_evidence"]["evidence_gaps"][0])

    def test_status_cli_rejects_removed_study_tag_option(self) -> None:
        result = subprocess.run(
            [
                PYTHON,
                "-m",
                "tools.brain.workflow",
                "status",
                "--workflow",
                "continuous_policy",
                "--study-tag",
                "legacy_tag",
                "--json",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)

    def test_script_path_cli_imports_package_root(self) -> None:
        payload = run_script_cli("status", "--workflow", "continuous_policy", "--json")

        self.assertEqual(payload["workflow_id"], "continuous_policy_result_review")
        self.assertIn("artifact_freshness", payload)

    def test_write_output_cli_writes_only_brain_workflow_output(self) -> None:
        payload = run_cli("status", "--workflow", "continuous_policy", "--json", "--write-output")
        output_path = Path(payload["output_path"])

        self.assertTrue(output_path.exists())
        self.assertIn("brain", output_path.parts)
        self.assertIn("output", output_path.parts)
        self.assertIn("brain_workflow", output_path.parts)

    def test_workflow_registry_contains_workspace_governance_without_generic_adapters(self) -> None:
        registry = load_workflow_registry()
        expected = {
            "brain_handoff",
            "brain_maintenance",
            "brain_system_audit",
            "brain_architecture_refactor",
            "brain_writeback",
            "brain_writeback_verified",
        }
        removed = REMOVED_WORKFLOWS

        self.assertTrue(expected.issubset(registry))
        self.assertFalse(removed.intersection(registry))
        for workflow_id in expected:
            entry = registry[workflow_id]
            self.assertIn("description", entry)
            self.assertIn("preflight", entry)
            self.assertIn("capability_hints", entry)
            self.assertIn("risk_signals", entry)
            self.assertIn("verification_hints", entry)
            self.assertIn("completion", entry)
            self.assertIn("checklist", entry)
            self.assertIn("stop_conditions", entry)
            self.assertNotIn("forbidden_" + "actions", entry)
            self.assertNotIn("start_" + "training", json.dumps(entry, ensure_ascii=False))

    def test_verify_plan_cli_delegates_to_selective_verification(self) -> None:
        payload = run_cli(
            "verify-plan",
            "--paths",
            "daily_research/path_policy/forecast_features.py",
            "--json",
        )

        self.assertEqual(payload["changed_paths"], ["daily_research/path_policy/forecast_features.py"])
        joined = "\n".join(payload["selected_commands"])
        self.assertIn("daily_research/path_policy/tests/test_forecast_features.py", joined)
        self.assertIn("always_commands", payload)

    def test_current_frontier_cli_outputs_stable_json(self) -> None:
        payload = run_cli("current-frontier", "--json")

        self.assertIn("latest_output_runs", payload)
        self.assertIn("latest_brain_reference_time", payload)
        self.assertIn("unregistered_latest_run_tags", payload)
        self.assertIn("brain_may_be_stale", payload)


if __name__ == "__main__":
    unittest.main()
