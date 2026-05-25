from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

from tools.brain.platform import load_workflow_registry


ROOT = Path(__file__).resolve().parents[3]
PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"


def run_cli(*args: str) -> dict:
    result = subprocess.run(
        [PYTHON, "-m", "tools.brain.workflow", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(result.stdout)


def run_script_cli(*args: str) -> dict:
    result = subprocess.run(
        [PYTHON, "tools/brain/workflow.py", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(result.stdout)


class BrainWorkflowCliTest(unittest.TestCase):
    def test_bootstrap_cli_outputs_valid_json_capsule(self) -> None:
        payload = run_cli("bootstrap", "--brain", "daily_research", "--json")

        self.assertEqual(payload["child_brain"], "daily_research")
        self.assertIn("boot_order", payload)
        self.assertIn("artifact_freshness", payload)
        self.assertIn("workflow_hints", payload)

    def test_workspace_bootstrap_json_includes_platform_fields(self) -> None:
        result = subprocess.run(
            [PYTHON, "-m", "tools.brain.workflow", "bootstrap", "--brain", "workspace", "--json"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["main_manifest"], "brain/brain_manifest.json")
        self.assertIn("artifact_freshness", payload)
        self.assertIn("workflow_hints", payload)
        self.assertIn("encoding_report", payload)

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
        self.assertIn("workspace_state", payload["routes"])
        self.assertIn("daily_research_state", payload["routes"])

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
        self.assertEqual(payload["study_evidence"]["workflow"], "continuous_policy")
        self.assertTrue(payload["requires_explicit_apply"])

    def test_writeback_plan_study_source_infers_path_policy(self) -> None:
        tag = "path20_input_ablation_no_alpha_liquid500_du_cost20_hit10_dd010_20260522_01"
        payload = run_cli("writeback-plan", "--source", f"study:{tag}", "--json")
        evidence = payload["study_evidence"]

        self.assertEqual(evidence["study_tag"], tag)
        self.assertEqual(evidence["workflow"], "path_policy")
        self.assertEqual(
            evidence["study_summary_json"],
            f"daily_research/output/path_policy/studies/{tag}/study_summary.json",
        )
        self.assertTrue(evidence["exists"])
        self.assertEqual(evidence["status"], "completed")
        self.assertEqual(evidence["stage"], "forecast_walkforward_study")
        self.assertEqual(evidence["evidence_verdict"], "forecast_test_confirmed")
        self.assertIn("policy_input_bundle__7c8f58d851bce8179e1e9e2d", evidence["dataset_ids"])
        self.assertIn("gru_sequence_static_context", evidence["model_families"])

    def test_writeback_plan_study_source_accepts_explicit_path_policy(self) -> None:
        tag = "path20_input_ablation_no_alpha_liquid500_du_cost20_hit10_dd010_20260522_01"
        payload = run_cli("writeback-plan", "--source", f"study:path_policy:{tag}", "--json")
        evidence = payload["study_evidence"]

        self.assertEqual(payload["source"], f"study:path_policy:{tag}")
        self.assertEqual(evidence["study_tag"], tag)
        self.assertEqual(evidence["workflow"], "path_policy")
        self.assertTrue(evidence["exists"])

    def test_writeback_plan_missing_study_reports_all_searched_paths(self) -> None:
        tag = "missing_study_for_writeback_plan_regression_20990101_01"
        payload = run_cli("writeback-plan", "--source", f"study:{tag}", "--json")
        evidence = payload["study_evidence"]

        self.assertEqual(evidence["study_tag"], tag)
        self.assertFalse(evidence["exists"])
        self.assertIn(
            f"daily_research/output/path_policy/studies/{tag}/study_summary.json",
            evidence["searched_paths"],
        )
        self.assertIn(
            f"daily_research/output/continuous_policy/studies/{tag}/study_summary.json",
            evidence["searched_paths"],
        )

    def test_writeback_plan_unsupported_study_workflow_returns_evidence_gap(self) -> None:
        payload = run_cli("writeback-plan", "--source", "study:unknown_workflow:any_tag", "--json")
        evidence = payload["study_evidence"]

        self.assertFalse(evidence["exists"])
        self.assertIn("unsupported_study_workflow: unknown_workflow", evidence["evidence_gaps"])

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

    def test_workflow_registry_contains_brain_native_superpowers(self) -> None:
        registry = load_workflow_registry()
        expected = {
            "brainstorming_design",
            "writing_plan",
            "executing_plan",
            "systematic_debugging",
            "verification_before_completion",
            "brain_writeback_verified",
            "long_task",
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
        self.assertTrue(payload["completion_review_required"])
        self.assertIn("project_guard_adapter", payload["workflow_mode"])
        self.assertIn("Generic skills own the execution method", payload["skill_boundary"])

    def test_workflow_guide_cli_knows_long_task(self) -> None:
        payload = run_cli("workflow-guide", "--workflow", "long_task", "--json")

        self.assertEqual(payload["workflow_id"], "long_task")
        self.assertIn("Wait-Process", "\n".join(payload["allowed_commands"]))
        self.assertIn("fixed_sleep_polling", payload["forbidden_actions"])
        self.assertIn("eta_reported_each_poll", payload["completion"])

    def test_workflow_guide_cli_requires_writeback_completion_review(self) -> None:
        payload = run_cli("workflow-guide", "--workflow", "brain_writeback_verified", "--json")

        self.assertEqual(payload["workflow_id"], "brain_writeback_verified")
        self.assertTrue(payload["completion_review_required"])

    def test_select_workflow_cli_maps_task_intent(self) -> None:
        cases = {
            "继续实施计划": "executing_plan",
            "报错了帮我查": "systematic_debugging",
            "是否全部完成": "verification_before_completion",
            "深入思考详细计划": "writing_plan",
            "更新脑区和 evidence registry": "brain_writeback_verified",
            "现在脑区乱不乱复杂不复杂": "brain_system_audit",
            "强重构脑区结构": "brain_architecture_refactor",
            "启动长训练并估算剩余时间": "long_task",
        }
        for task, expected in cases.items():
            with self.subTest(task=task):
                payload = run_cli("select-workflow", "--task", task, "--json")
                self.assertEqual(payload["selected_workflow"], expected)
                self.assertIn("reason", payload)

    def test_select_workflow_cli_uses_mutate_intent_to_execute_plan_titles(self) -> None:
        payload = run_cli(
            "select-workflow",
            "--task",
            "Alpha Multi-Horizon 输出设计、辅助任务与 Horizon Grid 完整对照计划",
            "--intent",
            "mutate",
            "--json",
        )

        self.assertEqual(payload["selected_workflow"], "executing_plan")
        self.assertTrue(payload["intent_override_applied"])
        self.assertIn("intent_mutate_override", payload["decision_sources"])
        self.assertIn("计划", payload["matched_terms"])

    def test_select_workflow_cli_keeps_plan_only_requests_as_writing_plan(self) -> None:
        payload = run_cli(
            "select-workflow",
            "--task",
            "深入分析，详细计划",
            "--intent",
            "read",
            "--json",
        )

        self.assertEqual(payload["selected_workflow"], "writing_plan")
        self.assertFalse(payload["intent_override_applied"])
        self.assertIn("plan_only_signal", payload["decision_sources"])

    def test_audit_brain_cli_outputs_catalog_language_and_guards(self) -> None:
        payload = run_cli("audit-brain", "--scope", "all", "--json")

        self.assertIn(payload["verdict"], {"clean", "usable_with_warnings", "needs_refactor", "blocked"})
        self.assertIn("catalog", payload)
        self.assertIn("language", payload)
        self.assertIn("workflow_registry", payload)
        self.assertIn("active_artifact_guard", payload)
        self.assertIn("loose_latest", payload)

    def test_capsule_auto_workflow_embeds_guide(self) -> None:
        payload = run_cli(
            "capsule",
            "--task",
            "继续实施计划",
            "--workflow",
            "auto",
            "--json",
        )

        self.assertEqual(payload["workflow_selection"]["selected_workflow"], "executing_plan")
        self.assertEqual(payload["workflow"], "executing_plan")
        self.assertIn("workflow_guide", payload)
        self.assertIn("required_checklist", payload)
        self.assertIn("stop_conditions", payload)
        self.assertEqual(payload["workflow_guide"]["workflow_id"], "executing_plan")

    def test_capsule_auto_workflow_uses_mutate_intent_for_plan_title(self) -> None:
        payload = run_cli(
            "capsule",
            "--task",
            "将所有未完成计划结合在一起，全部完成",
            "--workflow",
            "auto",
            "--intent",
            "mutate",
            "--json",
        )

        self.assertEqual(payload["workflow"], "executing_plan")
        self.assertIn(
            "implement_signal",
            payload["workflow_selection"]["decision_sources"],
        )
        self.assertIn("self_evolution_hooks", payload)
        self.assertIn("completion_review_required", payload["self_evolution_hooks"])

    def test_capsule_writeback_workflow_requires_completion_review(self) -> None:
        payload = run_cli(
            "capsule",
            "--task",
            "更新脑区和 evidence registry",
            "--workflow",
            "auto",
            "--intent",
            "writeback",
            "--json",
        )

        self.assertEqual(payload["workflow"], "brain_writeback_verified")
        self.assertTrue(payload["self_evolution_hooks"]["completion_review_required"])

    def test_capsule_cli_defaults_to_lite_context(self) -> None:
        payload = run_cli(
            "capsule",
            "--task",
            "Path20 当前到哪了",
            "--json",
        )

        self.assertEqual(payload["context_profile"], "lite")
        self.assertIn("frontier_report", payload["guards"])
        self.assertNotIn("latest_output_studies", payload["guards"]["frontier_report"])

    def test_capsule_cli_full_context_keeps_frontier_details(self) -> None:
        payload = run_cli(
            "capsule",
            "--task",
            "Path20 当前到哪了",
            "--verbosity",
            "full",
            "--json",
        )

        self.assertEqual(payload["context_profile"], "full")
        self.assertIn("latest_output_studies", payload["guards"]["frontier_report"])

    def test_capsule_intent_mutate_blocks_non_main_branch(self) -> None:
        payload = run_cli(
            "capsule",
            "--task",
            "实现脑区 Runtime Skill 计划",
            "--workflow",
            "auto",
            "--intent",
            "mutate",
        )

        self.assertIn("mutation_allowed", payload)
        if payload["main_context"]["git"]["on_main"]:
            self.assertTrue(payload["mutation_allowed"])
            self.assertNotIn("not_on_main_for_mutation", payload["preflight_blockers"])
        else:
            self.assertFalse(payload["mutation_allowed"])
            self.assertIn("not_on_main_for_mutation", payload["preflight_blockers"])

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

        self.assertIn("latest_output_studies", payload)
        self.assertIn("latest_brain_reference_time", payload)
        self.assertIn("unregistered_latest_tags", payload)
        self.assertIn("brain_may_be_stale", payload)

    def test_capsule_includes_current_frontier_report(self) -> None:
        payload = run_cli(
            "capsule",
            "--task",
            "Path20 当前到哪了",
            "--json",
        )

        self.assertIn("child_context", payload)
        self.assertIn("frontier_report", payload["guards"])
        self.assertIn("brain_may_be_stale", payload["guards"]["frontier_report"])


if __name__ == "__main__":
    unittest.main()
