from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.brain.selective_verification import build_verification_plan


ROOT = Path(__file__).resolve().parents[3]
PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"


def run_cli(*args: str) -> dict:
    result = subprocess.run(
        [PYTHON, "-m", "tools.brain.selective_verification", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(result.stdout)


class SelectiveVerificationTest(unittest.TestCase):
    def test_docs_only_changes_keep_only_always_guards(self) -> None:
        payload = build_verification_plan(paths=["daily_research/brain/operations_center.md"])

        self.assertEqual(payload["changed_paths"], ["daily_research/brain/operations_center.md"])
        self.assertEqual(payload["coverage_policy"], "changed_surface_only")
        self.assertEqual(len(payload["always_commands"]), 4)
        self.assertEqual(payload["selected_commands"], [])
        self.assertEqual(
            payload["blocking_commands"],
            [
                "git diff --check",
                f"{PYTHON} -m tools.brain.doc_guard check --files daily_research/brain/operations_center.md",
            ],
        )
        self.assertEqual(payload["deferred_commands"], [])
        self.assertEqual(payload["skipped_reason_by_area"]["daily_research"], "docs_only_minimal_guards")
        self.assertEqual(payload["skipped_reason_by_area"]["tools/brain"], "unchanged_area_not_tested")
        self.assertEqual(payload["risk_level"], "low")
        self.assertFalse(payload["manual_review_required"])

    def test_brain_tool_change_selects_brain_tool_tests(self) -> None:
        payload = build_verification_plan(paths=["tools/brain/capsule.py"])

        joined = "\n".join(payload["selected_commands"])
        self.assertIn("tools/brain/tests/test_capsule.py", joined)
        self.assertIn("tools/brain/tests/test_workflow_cli.py::BrainWorkflowCliTest::test_bootstrap_cli_outputs_valid_json_capsule", joined)
        self.assertNotIn("test_health_cli_aggregates_read_only_checks", joined)
        self.assertEqual(payload["blocking_commands"][0], "git diff --check")
        self.assertIn(payload["selected_commands"][0], payload["blocking_commands"])
        self.assertEqual(payload["risk_level"], "medium")

    def test_selective_verification_change_uses_fast_packet_plus_verify_nodeid(self) -> None:
        payload = build_verification_plan(paths=["tools/brain/selective_verification.py"])
        joined = "\n".join(payload["selected_commands"])

        self.assertIn("tools/brain/tests/test_selective_verification.py", joined)
        self.assertIn("tools/brain/tests/test_project_commit.py", joined)
        self.assertIn("tools/brain/tests/test_platform.py", joined)
        self.assertIn(
            "tools/brain/tests/test_workflow_cli.py::BrainWorkflowCliTest::test_verify_plan_cli_delegates_to_selective_verification",
            joined,
        )
        self.assertNotIn("tools/brain/tests/test_workflow_cli.py -q", joined)
        self.assertNotIn("test_health_cli_aggregates_read_only_checks", joined)

    def test_non_core_brain_tool_change_uses_fast_default_packet(self) -> None:
        payload = build_verification_plan(paths=["tools/brain/project_commit.py"])
        joined = "\n".join(payload["selected_commands"])

        self.assertIn("tools/brain/tests/test_selective_verification.py", joined)
        self.assertIn("tools/brain/tests/test_project_commit.py", joined)
        self.assertIn("tools/brain/tests/test_platform.py", joined)
        self.assertNotIn("pytest tools/brain/tests -q", joined)

    def test_process_tool_change_selects_process_namespace_tests(self) -> None:
        payload = build_verification_plan(paths=["tools/brain/long_task_monitor.py"])
        joined = "\n".join(payload["selected_commands"])

        self.assertIn("tools/brain/tests/test_agent_run.py", joined)
        self.assertIn("tools/brain/tests/test_long_task_monitor.py", joined)
        self.assertNotIn("tools/brain/tests/test_project_commit.py", joined)

    def test_overlapping_brain_tool_packets_keep_only_widest_pytest_command(self) -> None:
        payload = build_verification_plan(
            paths=[
                "tools/brain/selective_verification.py",
                "tools/brain/project_profiles.py",
            ]
        )

        self.assertEqual(len(payload["selected_commands"]), 1)
        joined = payload["selected_commands"][0]
        self.assertIn("tools/brain/tests/test_selective_verification.py", joined)
        self.assertIn("tools/brain/tests/test_project_commit.py", joined)
        self.assertIn("tools/brain/tests/test_platform.py", joined)
        self.assertIn("test_verify_plan_cli_delegates_to_selective_verification", joined)

    def test_data_lake_change_selects_data_lake_tests(self) -> None:
        payload = build_verification_plan(paths=["daily_research/data_lake/catalog.py"])

        self.assertEqual(
            payload["selected_commands"],
            [f"{PYTHON} -m pytest daily_research/data_lake/tests -q"],
        )
        self.assertEqual(payload["risk_level"], "medium")

    def test_data_platform_change_selects_data_platform_tests(self) -> None:
        payload = build_verification_plan(paths=["daily_research/data_platform/refresh_daily.py"])

        self.assertEqual(
            payload["selected_commands"],
            [f"{PYTHON} -m pytest daily_research/data_platform/tests -q"],
        )
        self.assertEqual(payload["blocking_commands"][0], "git diff --check")
        self.assertIn(payload["selected_commands"][0], payload["blocking_commands"])
        self.assertEqual(payload["risk_level"], "medium")

    def test_bridge_change_selects_bridge_and_candidate_matrix_tests(self) -> None:
        payload = build_verification_plan(paths=["daily_research/path_policy/v2_score_backtest_bridge.py"])
        joined = "\n".join(payload["selected_commands"])

        self.assertIn("daily_research/path_policy/tests/test_v2_score_backtest_bridge.py", joined)
        self.assertIn("daily_research/path_policy/tests/test_v2_candidate_review_matrix.py", joined)
        self.assertEqual(payload["deferred_commands"], [])

    def test_candidate_matrix_change_selects_candidate_matrix_test_only(self) -> None:
        payload = build_verification_plan(paths=["daily_research/path_policy/v2_candidate_review_matrix.py"])

        self.assertEqual(
            payload["selected_commands"],
            [f"{PYTHON} -m pytest daily_research/path_policy/tests/test_v2_candidate_review_matrix.py -q"],
        )

    def test_high_return_discovery_change_selects_discovery_bridge_and_matrix_tests(self) -> None:
        payload = build_verification_plan(paths=["daily_research/path_policy/v2_high_return_model_discovery.py"])
        joined = "\n".join(payload["selected_commands"])

        self.assertIn("daily_research/path_policy/tests/test_v2_high_return_model_discovery.py", joined)
        self.assertIn("daily_research/path_policy/tests/test_v2_score_backtest_bridge.py", joined)
        self.assertIn("daily_research/path_policy/tests/test_v2_candidate_review_matrix.py", joined)
        self.assertNotIn("daily_research/path_policy/tests -q", joined)

    def test_forecast_change_selects_forecast_tests_without_rl_protocol_file(self) -> None:
        payload = build_verification_plan(paths=["daily_research/path_policy/forecast_training.py"])

        joined = "\n".join(payload["selected_commands"])
        self.assertIn("daily_research/path_policy/tests/test_forecast_features.py", joined)
        self.assertIn(
            "daily_research/path_policy/tests/test_forecast_dataset.py::test_forecast_sequence_dataset_accepts_custom_horizon_grid_with_horizon_specific_risk",
            joined,
        )
        self.assertIn(
            "daily_research/path_policy/tests/test_forecast_training.py::test_train_forecast_models_accepts_custom_horizon_decision_utility_contract",
            joined,
        )
        self.assertIn("daily_research/path_policy/tests/test_models.py", joined)
        self.assertNotIn("daily_research/path_policy/tests/test_rl_protocol.py -q", joined)
        deferred = "\n".join(payload["deferred_long_commands"])
        self.assertIn("daily_research/path_policy/tests/test_forecast_dataset.py", deferred)
        self.assertIn("daily_research/path_policy/tests/test_forecast_training.py", deferred)
        self.assertEqual(payload["deferred_commands"], payload["deferred_long_commands"])

    def test_rl_protocol_change_uses_nodeid_groups_not_whole_file(self) -> None:
        payload = build_verification_plan(paths=["daily_research/path_policy/run_alpha_path20_protocol.py"])

        commands = payload["selected_commands"]
        self.assertGreaterEqual(len(commands), 3)
        rl_commands = [command for command in commands if "daily_research/path_policy/tests/test_rl_protocol.py" in command]
        self.assertTrue(all("daily_research/path_policy/tests/test_rl_protocol.py::" in command for command in rl_commands))
        self.assertIn(
            f"{PYTHON} -m pytest daily_research/path_policy/tests/test_run_alpha_path20_protocol_entrypoint.py -q",
            commands,
        )
        self.assertFalse(any(command.endswith("daily_research/path_policy/tests/test_rl_protocol.py -q") for command in commands))

    def test_continuous_policy_shared_core_marks_high_risk_and_defers_long_suite(self) -> None:
        payload = build_verification_plan(paths=["daily_research/continuous_policy/portfolio_simulator.py"])

        self.assertEqual(payload["risk_level"], "high")
        self.assertTrue(payload["deferred_long_commands"])
        joined = "\n".join(payload["selected_commands"])
        self.assertIn("daily_research/continuous_policy/tests/test_portfolio_cashflow_decision.py", joined)
        self.assertIn("daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py", "\n".join(payload["deferred_long_commands"]))

    def test_execution_change_requires_manual_review(self) -> None:
        payload = build_verification_plan(paths=["daily_research/execution/run_trade_plan.py"])

        self.assertEqual(payload["risk_level"], "high")
        self.assertTrue(payload["manual_review_required"])
        self.assertTrue(any("execution_or_active_boundary" in warning for warning in payload["warnings"]))

    def test_active_artifact_change_reports_blocker(self) -> None:
        payload = build_verification_plan(paths=["daily_research/output/active_execution_strategy.json"])

        self.assertEqual(payload["risk_level"], "critical")
        self.assertTrue(payload["manual_review_required"])
        self.assertTrue(any("active_artifact_diff_blocker" in warning for warning in payload["warnings"]))
        self.assertEqual(payload["blocking_commands"], [])
        self.assertEqual(payload["skipped_reason_by_area"]["daily_research"], "critical_active_artifact_blocker")

    def test_active_artifact_mixed_change_suppresses_normal_blocking_tests(self) -> None:
        payload = build_verification_plan(
            paths=[
                "daily_research/output/active_execution_strategy.json",
                "daily_research/path_policy/v2_score_backtest_bridge.py",
            ]
        )

        self.assertEqual(payload["risk_level"], "critical")
        self.assertEqual(payload["blocking_commands"], [])
        self.assertTrue(any("active_artifact_diff_blocker" in warning for warning in payload["warnings"]))

    def test_base_diff_active_artifact_change_reports_blocker(self) -> None:
        def fake_run_git(args: list[str]) -> list[str]:
            if args == [
                "diff",
                "--name-only",
                "HEAD~1",
                "--",
                "daily_research/output/active_execution_strategy.json",
            ]:
                return ["daily_research/output/active_execution_strategy.json"]
            return []

        with patch("tools.brain.selective_verification._run_git", side_effect=fake_run_git):
            payload = build_verification_plan(base="HEAD~1")

        self.assertEqual(payload["changed_paths"], ["daily_research/output/active_execution_strategy.json"])
        self.assertEqual(payload["risk_level"], "critical")
        self.assertTrue(payload["manual_review_required"])

    def test_cli_outputs_json_for_explicit_paths(self) -> None:
        payload = run_cli("--paths", "daily_research/path_policy/forecast_features.py", "--json")

        self.assertEqual(payload["changed_paths"], ["daily_research/path_policy/forecast_features.py"])
        self.assertIn("always_commands", payload)
        self.assertIn("selected_commands", payload)
        self.assertIn("blocking_commands", payload)
        self.assertIn("deferred_commands", payload)
        self.assertEqual(payload["coverage_policy"], "changed_surface_only")

    def test_cli_paths_take_priority_over_base(self) -> None:
        payload = run_cli(
            "--base",
            "HEAD",
            "--paths",
            "daily_research/brain/state_center.md",
            "--json",
        )

        self.assertEqual(payload["changed_paths"], ["daily_research/brain/state_center.md"])
        self.assertEqual(payload["selected_commands"], [])

    def test_traditional_quant_change_uses_same_surface_test_without_daily_active_guard(self) -> None:
        payload = build_verification_plan(paths=["traditional_quant_research/experiments/frontier_ml_signal_rebuild.py"])
        encoded = json.dumps(payload, ensure_ascii=False)

        self.assertEqual(payload["project_id"], "traditional_quant_research")
        self.assertNotIn("daily_research/output/active_execution_strategy.json", encoded)
        self.assertEqual(
            payload["selected_commands"],
            [f"{PYTHON} -m pytest traditional_quant_research/tests/test_frontier_ml_signal_rebuild.py -q"],
        )
        self.assertIn("git diff --check", payload["always_commands"])
        self.assertNotIn(f"{PYTHON} -m pytest traditional_quant_research/tests -q", payload["selected_commands"])

    def test_unmapped_project_python_change_requires_manual_review_not_project_suite(self) -> None:
        payload = build_verification_plan(paths=["traditional_quant_research/backtest.py"])

        self.assertEqual(payload["project_id"], "traditional_quant_research")
        self.assertEqual(payload["selected_commands"], [])
        self.assertTrue(payload["manual_review_required"])
        self.assertTrue(any("unmapped_project_python_change" in warning for warning in payload["warnings"]))
        self.assertEqual(payload["skipped_reason_by_area"]["traditional_quant_research"], "manual_review_required_no_direct_test")

    def test_project_test_file_change_runs_that_test_file_only(self) -> None:
        payload = build_verification_plan(paths=["daily_stock_analysis-main/tests/test_config_manager.py"])

        self.assertEqual(payload["project_id"], "daily_stock_analysis-main")
        self.assertEqual(
            payload["selected_commands"],
            [f"{PYTHON} -m pytest daily_stock_analysis-main/tests/test_config_manager.py -q"],
        )
        self.assertFalse(payload["manual_review_required"])

    def test_daily_change_keeps_daily_active_guard(self) -> None:
        payload = build_verification_plan(paths=["daily_research/path_policy/forecast_features.py"])
        encoded = json.dumps(payload, ensure_ascii=False)

        self.assertEqual(payload["project_id"], "daily_research")
        self.assertIn("daily_research/output/active_execution_strategy.json", encoded)

    def test_daily_tool_change_uses_py_compile_smoke(self) -> None:
        payload = build_verification_plan(paths=["daily_research/tools/run_policy_v4_family_pipeline.py"])

        self.assertEqual(payload["project_id"], "daily_research")
        self.assertEqual(
            payload["selected_commands"],
            [f"{PYTHON} -m py_compile daily_research/tools/run_policy_v4_family_pipeline.py"],
        )
        self.assertFalse(payload["manual_review_required"])
        self.assertEqual(payload["skipped_reason_by_area"]["daily_research"], "changed_surface_selected")

    def test_daily_guard_tool_change_selects_health_contract_tests(self) -> None:
        payload = build_verification_plan(paths=["daily_research/tools/openmp_runtime_check.py"])
        joined = "\n".join(payload["selected_commands"])

        self.assertIn(f"{PYTHON} -m py_compile daily_research/tools/openmp_runtime_check.py", joined)
        self.assertIn("test_health_default_skips_daily_strict_lanes", joined)
        self.assertIn("test_health_full_daily_runs_project_and_openmp_strict_lanes", joined)
        self.assertFalse(payload["manual_review_required"])


if __name__ == "__main__":
    unittest.main()
