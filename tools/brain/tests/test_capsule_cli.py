from __future__ import annotations

import json
import unittest

from tools.brain.tests.workflow_cli_helpers import run_cli


class BrainCapsuleCliTest(unittest.TestCase):
    def test_capsule_auto_workflow_keeps_generic_plan_task_in_handoff(self) -> None:
        payload = run_cli(
            "capsule",
            "--task",
            "继续实施计划",
            "--workflow",
            "auto",
            "--json",
        )

        self.assertEqual(payload["schema_version"], 4)
        self.assertEqual(payload["workflow_selection"]["selected_workflow"], "brain_handoff")
        self.assertEqual(payload["workflow"], "brain_handoff")
        self.assertIn("workflow_guide", payload)
        self.assertIn("required_checklist", payload)
        self.assertIn("capability_hints", payload)
        self.assertIn("risk_signals", payload)
        self.assertIn("verification_hints", payload)
        self.assertIn("stop_conditions", payload)
        self.assertNotIn("forbidden_" + "actions", payload)
        self.assertEqual(payload["workflow_guide"]["workflow_id"], "brain_handoff")

    def test_capsule_does_not_treat_training_as_brain_workflow_blocker(self) -> None:
        payload = run_cli(
            "capsule",
            "--task",
            "修改脑区规则并启动训练",
            "--workflow",
            "auto",
            "--intent",
            "mutate",
            "--json",
        )
        encoded = json.dumps(
            {
                "capability_hints": payload["capability_hints"],
                "risk_signals": payload["risk_signals"],
                "verification_hints": payload["verification_hints"],
            },
            ensure_ascii=False,
        )

        self.assertEqual(payload["schema_version"], 4)
        self.assertEqual(payload["workflow"], "brain_maintenance")
        self.assertEqual(payload["target_kind"], "workspace")
        self.assertEqual(payload["workflow_domain"], "workspace_governance")
        self.assertNotIn("forbidden_" + "actions", payload)
        self.assertNotIn("start_" + "training", encoded)
        self.assertIn("capability_hints", payload)
        self.assertIn("risk_signals", payload)
        self.assertIn("verification_hints", payload)
        self.assertIn("polling/async task", encoded)
        self.assertNotIn("long_task_monitor", encoded)
        self.assertNotIn("training", payload["preflight_blockers"])

    def test_capsule_polling_training_plan_stays_handoff_with_observable_capability(self) -> None:
        payload = run_cli(
            "capsule",
            "--task",
            "继续实施计划，到最后启动长训练",
            "--workflow",
            "auto",
            "--intent",
            "mutate",
            "--json",
        )
        encoded = json.dumps(
            {
                "capability_hints": payload["capability_hints"],
                "risk_signals": payload["risk_signals"],
                "verification_hints": payload["verification_hints"],
            },
            ensure_ascii=False,
        )

        self.assertEqual(payload["workflow"], "brain_handoff")
        self.assertIn("polling/async task", encoded)
        self.assertIn("best observable handle", encoded)
        self.assertNotIn("long_task_monitor", encoded)
        self.assertIn("capability_hints", payload)

    def test_capsule_auto_workflow_keeps_mutate_plan_title_in_handoff(self) -> None:
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

        self.assertEqual(payload["workflow"], "brain_handoff")
        self.assertIn("external_skill_signal", payload["workflow_selection"]["decision_sources"])
        self.assertNotIn("self_" + "evolution_hooks", payload)
        self.assertNotIn("runtime_learning_hooks", payload)
        self.assertIn("agent_meta", payload)
        self.assertIn("agent_review", payload)
        self.assertFalse(payload["agent_review"]["before_final_required"])

    def test_capsule_writeback_workflow_does_not_force_completion_review(self) -> None:
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
        self.assertFalse(payload["agent_review"]["before_final_required"])
        self.assertNotIn("workflow_completion_review", payload["agent_review"]["reason_codes"])

    def test_capsule_brain_rule_mutation_uses_maintenance_without_forced_review(self) -> None:
        payload = run_cli(
            "capsule",
            "--task",
            "修改脑区规则",
            "--workflow",
            "auto",
            "--intent",
            "mutate",
            "--json",
        )

        self.assertEqual(payload["workflow"], "brain_maintenance")
        self.assertEqual(payload["schema_version"], 4)
        self.assertEqual(payload["target_kind"], "workspace")
        self.assertEqual(payload["workflow_domain"], "workspace_governance")
        self.assertFalse(payload["agent_review"]["before_final_required"])
        self.assertNotIn("workflow_completion_review", payload["agent_review"]["reason_codes"])

    def test_capsule_cli_defaults_to_lite_context(self) -> None:
        payload = run_cli(
            "capsule",
            "--task",
            "Path20 当前到哪了",
            "--json",
        )

        self.assertEqual(payload["context_profile"], "lite")
        self.assertIn("frontier_report", payload["guards"])
        self.assertNotIn("latest_output_runs", payload["guards"]["frontier_report"])

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
        self.assertIn("latest_output_runs", payload["guards"]["frontier_report"])

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
