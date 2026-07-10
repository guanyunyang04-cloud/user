from __future__ import annotations

import unittest

from tools.brain.tests.workflow_cli_helpers import run_cli


class BrainRouteCliTest(unittest.TestCase):
    def test_route_cli_outputs_structured_workspace_target(self) -> None:
        payload = run_cli("route", "--task", "清理脑区治理规则", "--json")

        self.assertEqual(payload["status"], "selected")
        self.assertEqual(payload["selected_brain_id"], "workspace")
        self.assertEqual(payload["target"]["id"], "workspace")
        self.assertEqual(payload["target"]["kind"], "workspace")
        self.assertEqual(payload["target"]["domain"], "workspace_governance")
        self.assertIn("workspace_governance", payload["target"]["bootstrap_aliases"])
        self.assertFalse(payload["decision_required"])
        self.assertEqual(payload["recommended_default"], "workspace")

    def test_route_cli_outputs_structured_child_target(self) -> None:
        payload = run_cli("route", "--task", "修复 daily_research execution web 控制台", "--json")

        self.assertEqual(payload["status"], "selected")
        self.assertEqual(payload["selected_brain_id"], "daily_research")
        self.assertEqual(payload["target"]["id"], "daily_research")
        self.assertEqual(payload["target"]["kind"], "child")
        self.assertEqual(payload["target"]["domain"], "daily_research")

    def test_route_cli_keeps_ambiguous_target(self) -> None:
        payload = run_cli("route", "--task", "Path20 和盘中 RL 联合接管", "--json")

        self.assertEqual(payload["status"], "ambiguous")
        self.assertEqual(payload["selected_brain_id"], "")
        self.assertEqual(payload["target"]["kind"], "ambiguous")
        self.assertTrue(payload["decision_required"])

    def test_route_cli_outputs_needs_agent_decision_for_soft_terms(self) -> None:
        payload = run_cli("route", "--task", "training 复盘", "--json")

        self.assertEqual(payload["status"], "needs_agent_decision")
        self.assertEqual(payload["selected_brain_id"], "")
        self.assertEqual(payload["target"]["kind"], "ambiguous")
        self.assertTrue(payload["decision_required"])
        self.assertEqual(payload["recommended_default"], "workspace")

    def test_route_cli_exposes_active_execution_object(self) -> None:
        payload = run_cli("route", "--task", "切换默认执行策略并写入 active_execution_strategy", "--json")

        by_id = {item["object"]: item for item in payload["object_routes"]}
        self.assertEqual(payload["selected_brain_id"], "daily_research")
        self.assertEqual(by_id["active_execution_artifact"]["mode"], "write")


if __name__ == "__main__":
    unittest.main()
