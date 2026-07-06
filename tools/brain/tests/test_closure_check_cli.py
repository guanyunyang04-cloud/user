from __future__ import annotations

import unittest

from tools.brain.tests.workflow_cli_helpers import run_cli


class BrainClosureCheckCliTest(unittest.TestCase):
    def test_closure_check_cli_reports_cross_project_object_routes(self) -> None:
        payload = run_cli(
            "closure-check",
            "--task",
            "qdp_v2 sequence pack GRU 训练",
            "--paths",
            "brain/object_registry.json",
            "tools/brain/routing.py",
            "--json",
        )

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["routing"]["primary_brain_id"], "daily_research")
        self.assertIn("quant_data_platform", payload["routing"]["supporting_brain_ids"])
        object_ids = {item["object"] for item in payload["object_routes"]}
        self.assertIn("qdp_v2_active_data_base", object_ids)
        self.assertIn("sequence_training_pack", object_ids)
        self.assertTrue(payload["brain_sync_required"])
        self.assertIn("brain_surface_changed", payload["reason_codes"])
        self.assertIn("cross_project_orchestration", payload["reason_codes"])
        joined = "\n".join(payload["validation_commands"])
        self.assertIn("tools/brain/tests -q", joined)
        self.assertIn("doc_guard", joined)
        self.assertIn("integrity_check", joined)

    def test_closure_check_cli_reports_qdp_active_writeback_for_active_path(self) -> None:
        payload = run_cli(
            "closure-check",
            "--paths",
            "quant_data_platform/data/qdp_v2/active/active.json",
            "--json",
        )

        self.assertTrue(payload["brain_sync_required"])
        self.assertIn("qdp_active_data_base_touched", payload["reason_codes"])
        self.assertIn("quant_data_platform/brain/state_center.md", payload["writeback_targets"])
        self.assertTrue(
            any(
                item["object"] == "qdp_v2_active_data_base"
                and item["owner"] == "quant_data_platform"
                and item["mode"] == "write"
                for item in payload["object_routes"]
            )
        )
        self.assertIn("qdp check --quick --json", payload["validation_commands"])

    def test_closure_check_cli_reports_active_artifact_guard(self) -> None:
        payload = run_cli(
            "closure-check",
            "--paths",
            "daily_research/output/active_execution_strategy.json",
            "--json",
        )

        self.assertTrue(payload["active_artifact_guard_required"])
        self.assertIn("active_execution_artifact_touched", payload["reason_codes"])
        self.assertIn("git diff -- daily_research/output/active_execution_strategy.json", payload["validation_commands"])


if __name__ == "__main__":
    unittest.main()
