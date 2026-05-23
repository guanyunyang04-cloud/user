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

    def test_route_intraday_task_to_t0_project(self) -> None:
        payload = route_task_to_brain("盘中 RL 原型接管")

        self.assertEqual(payload["status"], "selected")
        self.assertEqual(payload["selected_brain_id"], "t0_project")

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
        self.assertEqual(payload["routing"]["selected_brain_id"], "daily_research")
        self.assertIn("child_context", payload)
        self.assertEqual(payload["child_context"]["brain_id"], "daily_research")
        self.assertIn("daily_research/brain/state_center.md", payload["child_context"]["fast_handoff_paths"])
        self.assertIn("active_artifact_guard", payload["guards"])

    def test_capsule_workspace_governance_has_no_daily_research_state_summary(self) -> None:
        payload = build_task_capsule(task="主脑 capsule 重构", workflow="auto")

        self.assertEqual(payload["routing"]["selected_brain_id"], "workspace_governance")
        self.assertNotIn("child_context", payload)
        text = "\n".join(payload["main_context"]["summary"])
        self.assertNotIn("Path20 当前研究主线指针", text)


if __name__ == "__main__":
    unittest.main()
