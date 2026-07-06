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

    def test_preflight_cli_does_not_create_study_directories(self) -> None:
        studies_root = ROOT / "daily_research/output/continuous_policy/studies"
        before = {path.name for path in studies_root.iterdir()} if studies_root.exists() else set()

        payload = run_cli("preflight", "--workflow", "continuous_policy_safe_screening", "--json")

        after = {path.name for path in studies_root.iterdir()} if studies_root.exists() else set()
        self.assertEqual(before, after)
        self.assertEqual(payload["workflow_id"], "continuous_policy_safe_screening")
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["writes_tracked_files"])

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

    def test_workflow_guide_cli_rejects_removed_generic_workflows(self) -> None:
        for workflow_id in ("executing_" + "plan", "writing_" + "plan", "long_task"):
            with self.subTest(workflow_id=workflow_id):
                result = subprocess.run(
                    [PYTHON, "-m", "tools.brain.workflow", "workflow-guide", "--workflow", workflow_id, "--json"],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                payload = json.loads(result.stdout)
                self.assertEqual(payload["status"], "error")
                self.assertIn("Unknown workflow", payload["error"])

    def test_workflow_guide_cli_outputs_brain_native_contract(self) -> None:
        payload = run_cli("workflow-guide", "--workflow", "brain_writeback_verified", "--json")

        self.assertEqual(payload["workflow_id"], "brain_writeback_verified")
        self.assertIn("description", payload)
        self.assertIn("checklist", payload)
        self.assertIn("capability_hints", payload)
        self.assertIn("risk_signals", payload)
        self.assertIn("verification_hints", payload)
        self.assertIn("stop_conditions", payload)
        self.assertIn("validation_commands", payload)
        self.assertIn("writeback_routes", payload)
        self.assertNotIn("forbidden_" + "actions", payload)
        self.assertTrue(payload["completion_review_required"])
        self.assertEqual(payload["workflow_mode"], "brain_native_contract")
        self.assertIn("Brain workflow owns", payload["skill_boundary"])

    def test_workflow_guide_cli_exposes_capabilities_without_training_blocks(self) -> None:
        payload = run_cli("workflow-guide", "--workflow", "brain_maintenance", "--json")
        encoded = json.dumps(
            {
                "capability_hints": payload["capability_hints"],
                "risk_signals": payload["risk_signals"],
                "verification_hints": payload["verification_hints"],
            },
            ensure_ascii=False,
        )

        self.assertEqual(payload["workflow_id"], "brain_maintenance")
        self.assertIn("capability_hints", payload)
        self.assertIn("risk_signals", payload)
        self.assertIn("verification_hints", payload)
        self.assertNotIn("forbidden_" + "actions", payload)
        self.assertNotIn("start_" + "training", encoded)

    def test_workflow_guide_cli_requires_writeback_completion_review(self) -> None:
        payload = run_cli("workflow-guide", "--workflow", "brain_writeback_verified", "--json")

        self.assertEqual(payload["workflow_id"], "brain_writeback_verified")
        self.assertTrue(payload["completion_review_required"])

    def test_select_workflow_cli_maps_task_intent(self) -> None:
        cases = {
            "继续实施计划": "brain_handoff",
            "报错了帮我查": "brain_handoff",
            "CSS selector bug": "brain_handoff",
            "React selector dropdown 报错": "brain_handoff",
            "是否全部完成": "brain_handoff",
            "深入思考详细计划": "brain_handoff",
            "更新脑区和 evidence registry": "brain_writeback_verified",
            "修改脑区规则": "brain_maintenance",
            "更新 workspace-brain skill 并同步本机 skill": "brain_maintenance",
            "更新 workflow registry": "brain_maintenance",
            "workflow selector 漏判": "brain_maintenance",
            "更新 evidence registry": "brain_writeback_verified",
            "现在脑区乱不乱复杂不复杂": "brain_system_audit",
            "强重构脑区结构": "brain_architecture_refactor",
            "启动长训练并估算剩余时间": "brain_handoff",
        }
        for task, expected in cases.items():
            with self.subTest(task=task):
                payload = run_cli("select-workflow", "--task", task, "--json")
                self.assertEqual(payload["selected_workflow"], expected)
                self.assertIn("reason", payload)

    def test_select_workflow_cli_keeps_mutate_plan_titles_in_brain_handoff(self) -> None:
        payload = run_cli(
            "select-workflow",
            "--task",
            "Alpha Multi-Horizon 输出设计、辅助任务与 Horizon Grid 完整对照计划",
            "--intent",
            "mutate",
            "--json",
        )

        self.assertEqual(payload["selected_workflow"], "brain_handoff")
        self.assertFalse(payload["intent_override_applied"])
        self.assertIn("计划", payload["matched_terms"])

    def test_select_workflow_cli_keeps_plan_only_requests_in_brain_handoff(self) -> None:
        payload = run_cli(
            "select-workflow",
            "--task",
            "深入分析，详细计划",
            "--intent",
            "read",
            "--json",
        )

        self.assertEqual(payload["selected_workflow"], "brain_handoff")
        self.assertFalse(payload["intent_override_applied"])
        self.assertIn("external_skill_signal", payload["decision_sources"])

    def test_select_workflow_cli_writeback_intent_selects_verified_writeback(self) -> None:
        payload = run_cli(
            "select-workflow",
            "--task",
            "执行 brain writeback verified 登记证据",
            "--intent",
            "writeback",
            "--json",
        )

        self.assertEqual(payload["selected_workflow"], "brain_writeback_verified")
        self.assertIn("writeback", payload["matched_terms"])

    def test_audit_brain_cli_outputs_catalog_language_and_guards(self) -> None:
        payload = run_cli("audit-brain", "--scope", "all", "--json")

        self.assertIn(payload["verdict"], {"clean", "usable_with_warnings", "needs_refactor", "blocked"})
        self.assertIn("catalog", payload)
        self.assertIn("language", payload)
        self.assertIn("workflow_registry", payload)
        self.assertIn("active_artifact_guard", payload)
        self.assertIn("loose_latest", payload)

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
