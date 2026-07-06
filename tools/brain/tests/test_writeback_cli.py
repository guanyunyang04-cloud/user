from __future__ import annotations

import unittest

from tools.brain.tests.workflow_cli_helpers import run_cli


class BrainWritebackCliTest(unittest.TestCase):
    def test_writeback_plan_cli_only_returns_routes_by_default(self) -> None:
        payload = run_cli("writeback-plan", "--source", "latest", "--json")

        self.assertEqual(payload["source"], "latest")
        self.assertFalse(payload["apply_brain_writeback"])
        self.assertIn("routes", payload)
        self.assertIn("workspace_state", payload["routes"])
        self.assertIn("daily_research_state", payload["routes"])

    def test_writeback_plan_cli_uses_task_orchestration_routes(self) -> None:
        payload = run_cli(
            "writeback-plan",
            "--source",
            "latest",
            "--task",
            "qdp_v2 sequence pack GRU 训练",
            "--json",
        )

        self.assertEqual(payload["task"], "qdp_v2 sequence pack GRU 训练")
        self.assertEqual(payload["brain_orchestration"]["primary_brain_id"], "daily_research")
        self.assertIn("quant_data_platform", payload["brain_orchestration"]["supporting_brain_ids"])
        self.assertIn("daily_research_state", payload["routes"])
        self.assertIn("daily_research_references", payload["routes"])
        self.assertIn("workspace_state", payload["routes"])
        self.assertNotIn("quant_data_platform_state", payload["routes"])

    def test_writeback_plan_run_source_is_read_only(self) -> None:
        tag = "missing_continuous_policy_fixture_20990101_01"
        payload = run_cli("writeback-plan", "--source", f"run:continuous_policy:{tag}", "--json")

        self.assertEqual(payload["source"], f"run:continuous_policy:{tag}")
        self.assertFalse(payload["apply_brain_writeback"])
        self.assertIn("run_evidence", payload)
        self.assertNotIn("study_evidence", payload)
        self.assertEqual(payload["run_evidence"]["run_tag"], tag)
        self.assertEqual(payload["run_evidence"]["workflow"], "continuous_policy")
        self.assertFalse(payload["run_evidence"]["exists"])
        self.assertTrue(payload["requires_explicit_apply"])

    def test_writeback_plan_run_source_infers_path_policy(self) -> None:
        tag = "mh_v2_horizon_30d_soft_penalty_seed7_20260603_01"
        payload = run_cli("writeback-plan", "--source", f"run:{tag}", "--json")
        evidence = payload["run_evidence"]

        self.assertEqual(evidence["run_tag"], tag)
        self.assertEqual(evidence["workflow"], "path_policy")
        self.assertEqual(
            evidence["run_summary_json"],
            f"daily_research/output/path_policy/studies/{tag}/study_summary.json",
        )
        self.assertTrue(evidence["exists"])
        self.assertEqual(evidence["status"], "completed")
        self.assertEqual(evidence["stage"], "forecast_walkforward_study")
        self.assertEqual(evidence["evidence_verdict"], "forecast_test_confirmed")
        self.assertIn("policy_input_bundle__45e3d8c059ba718426a9f887", evidence["dataset_ids"])
        self.assertIn("gru_sequence_static_context", evidence["model_families"])

    def test_writeback_plan_run_source_accepts_explicit_path_policy(self) -> None:
        tag = "mh_v2_horizon_30d_soft_penalty_seed7_20260603_01"
        payload = run_cli("writeback-plan", "--source", f"run:path_policy:{tag}", "--json")
        evidence = payload["run_evidence"]

        self.assertEqual(payload["source"], f"run:path_policy:{tag}")
        self.assertEqual(evidence["run_tag"], tag)
        self.assertEqual(evidence["workflow"], "path_policy")
        self.assertTrue(evidence["exists"])

    def test_writeback_plan_missing_run_reports_all_searched_paths(self) -> None:
        tag = "missing_study_for_writeback_plan_regression_20990101_01"
        payload = run_cli("writeback-plan", "--source", f"run:{tag}", "--json")
        evidence = payload["run_evidence"]

        self.assertEqual(evidence["run_tag"], tag)
        self.assertFalse(evidence["exists"])
        self.assertIn(
            f"daily_research/output/path_policy/studies/{tag}/study_summary.json",
            evidence["searched_paths"],
        )
        self.assertIn(
            f"daily_research/output/continuous_policy/studies/{tag}/study_summary.json",
            evidence["searched_paths"],
        )

    def test_writeback_plan_unsupported_run_workflow_returns_evidence_gap(self) -> None:
        payload = run_cli("writeback-plan", "--source", "run:unknown_workflow:any_tag", "--json")
        evidence = payload["run_evidence"]

        self.assertFalse(evidence["exists"])
        self.assertIn("unsupported_run_workflow: unknown_workflow", evidence["evidence_gaps"])


if __name__ == "__main__":
    unittest.main()
