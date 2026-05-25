from __future__ import annotations

import unittest

from tools.brain.capsule import build_task_capsule
from tools.brain.routing import route_task_to_brain


class BrainCapsuleTest(unittest.TestCase):
    def test_route_workspace_governance_task(self) -> None:
        payload = route_task_to_brain("审阅主脑和分脑接管治理")

        self.assertEqual(payload["status"], "selected")
        self.assertEqual(payload["selected_brain_id"], "workspace_governance")

    def test_route_path20_task_to_daily_research(self) -> None:
        payload = route_task_to_brain("Path20 多 horizon 训练计划")

        self.assertEqual(payload["status"], "selected")
        self.assertEqual(payload["selected_brain_id"], "daily_research")

    def test_route_multi_horizon_and_path_policy_terms_to_daily_research(self) -> None:
        cases = (
            "alpha_multi_horizon_utility_policy_v1 根因审计",
            "Alpha Multi-Horizon 长短周期根因评估",
            "path_policy horizon root cause audit",
            "daily_research/path_policy/horizon_root_cause_audit.py 修复",
        )
        for task in cases:
            with self.subTest(task=task):
                payload = route_task_to_brain(task)
                self.assertEqual(payload["status"], "selected")
                self.assertEqual(payload["selected_brain_id"], "daily_research")
                self.assertTrue(payload.get("routing_sources"))

    def test_route_daily_research_execution_terms(self) -> None:
        cases = (
            "执行端交易计划没有动作",
            "数据刷新后 signal panel 过期",
            "模型页 active manifest 状态",
            "daily_research/execution/app_service.py 修复",
            "production signal refresh 闭环",
            "当前数据集不完整",
        )
        for task in cases:
            with self.subTest(task=task):
                payload = route_task_to_brain(task)
                self.assertEqual(payload["status"], "selected")
                self.assertEqual(payload["selected_brain_id"], "daily_research")

    def test_route_intraday_task_to_t0_project(self) -> None:
        payload = route_task_to_brain("盘中 RL 原型接管")

        self.assertEqual(payload["status"], "selected")
        self.assertEqual(payload["selected_brain_id"], "t0_project")

    def test_route_daily_stock_analysis_task_to_product_brain(self) -> None:
        payload = route_task_to_brain("daily_stock_analysis 多市场产品修复")

        self.assertEqual(payload["status"], "selected")
        self.assertEqual(payload["selected_brain_id"], "daily_stock_analysis-main")

    def test_route_ambiguous_task_does_not_default_to_daily_research(self) -> None:
        payload = route_task_to_brain("Path20 和盘中 RL 联合接管")

        self.assertEqual(payload["status"], "ambiguous")
        self.assertEqual(payload["selected_brain_id"], "")

    def test_capsule_without_child_returns_schema_v2_main_context(self) -> None:
        payload = build_task_capsule(task="审阅主脑接管规则", workflow="auto")

        self.assertEqual(payload["schema_version"], 2)
        self.assertIn("main_context", payload)
        self.assertEqual(payload["routing"]["selected_brain_id"], "workspace_governance")
        self.assertNotIn("child", payload)
        self.assertNotIn("state_summary", payload)
        self.assertNotIn("hard_rules", payload)
        self.assertNotIn("fast_handoff_paths", payload)

    def test_capsule_path20_attaches_daily_research_child_context(self) -> None:
        payload = build_task_capsule(task="Path20 当前状态", workflow="auto")

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["context_profile"], "lite")
        self.assertEqual(payload["routing"]["selected_brain_id"], "daily_research")
        self.assertIn("child_context", payload)
        self.assertEqual(payload["child_context"]["brain_id"], "daily_research")
        self.assertIn("daily_research/brain/state_center.md", payload["child_context"]["fast_handoff_paths"])
        self.assertIn("active_artifact_guard", payload["guards"])
        self.assertNotIn("state_summary", payload["child_context"])
        self.assertNotIn("hard_rules", payload["child_context"])
        self.assertNotIn("latest_output_studies", payload["guards"].get("frontier_report", {}))

    def test_capsule_multi_horizon_registry_fallback_attaches_daily_research_child_context(self) -> None:
        payload = build_task_capsule(task="alpha_multi_horizon_utility_policy_v1 根因审计", workflow="auto")

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["routing"]["selected_brain_id"], "daily_research")
        self.assertIn("child_context", payload)
        self.assertEqual(payload["child_context"]["brain_id"], "daily_research")
        self.assertIn("active_artifact_guard", payload["guards"])
        self.assertIn("frontier_report", payload["guards"])
        self.assertIn("registry_exact_match", payload["routing"].get("routing_sources", []))

    def test_capsule_lite_exposes_deep_dive_commands(self) -> None:
        payload = build_task_capsule(task="Path20 当前状态", workflow="auto")

        self.assertEqual(payload["context_profile"], "lite")
        self.assertIn("summary_budget", payload)
        commands = "\n".join(payload["available_deep_dive_commands"])
        self.assertIn("current-frontier", commands)
        self.assertIn("query", commands)
        self.assertIn("health --mode full", commands)

    def test_capsule_full_preserves_deep_context(self) -> None:
        payload = build_task_capsule(task="Path20 当前状态", workflow="auto", verbosity="full")

        self.assertEqual(payload["context_profile"], "full")
        self.assertIn("state_summary", payload["child_context"])
        self.assertIn("hard_rules", payload["child_context"])
        self.assertIn("latest_output_studies", payload["guards"]["frontier_report"])

    def test_capsule_workspace_governance_has_no_daily_research_state_summary(self) -> None:
        payload = build_task_capsule(task="主脑 capsule 重构", workflow="auto")

        self.assertEqual(payload["routing"]["selected_brain_id"], "workspace_governance")
        self.assertNotIn("child_context", payload)
        self.assertNotIn("summary", payload["main_context"])
        self.assertNotIn("hard_rules", payload["main_context"])

    def test_capsule_long_task_wording_stays_in_brain_handoff(self) -> None:
        payload = build_task_capsule(
            task="三组 shadow-only seed7 长训练轮询",
            workflow="auto",
            intent="mutate",
            verbosity="lite",
        )

        self.assertEqual(payload["workflow"], "brain_handoff")
        self.assertNotIn("long_task_contract", payload)
        self.assertIn("guards", payload)
        self.assertIn("preflight_blockers", payload)

    def test_capsule_mutate_plan_title_stays_in_brain_handoff(self) -> None:
        payload = build_task_capsule(
            task="将所有未完成计划结合在一起，全部完成",
            workflow="auto",
            intent="mutate",
            verbosity="lite",
        )

        self.assertEqual(payload["workflow"], "brain_handoff")
        self.assertIn("external_skill_signal", payload["workflow_selection"]["decision_sources"])
        hooks = payload["self_evolution_hooks"]
        self.assertFalse(hooks["completion_review_required"])
        self.assertIn("routing_or_workflow_conflict", hooks["review_triggers"])

    def test_capsule_brain_rule_mutation_uses_brain_maintenance(self) -> None:
        payload = build_task_capsule(
            task="修改脑区规则",
            workflow="auto",
            intent="mutate",
            verbosity="lite",
        )

        self.assertEqual(payload["workflow"], "brain_maintenance")
        self.assertEqual(payload["routing"]["selected_brain_id"], "workspace_governance")
        hooks = payload["self_evolution_hooks"]
        self.assertTrue(hooks["completion_review_required"])


if __name__ == "__main__":
    unittest.main()
