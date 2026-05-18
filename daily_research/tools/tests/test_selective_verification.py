from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from daily_research.tools.selective_verification import build_verification_plan


ROOT = Path(__file__).resolve().parents[3]
PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"


def run_cli(*args: str) -> dict:
    result = subprocess.run(
        [PYTHON, "-m", "daily_research.tools.selective_verification", *args],
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
        self.assertEqual(len(payload["always_commands"]), 4)
        self.assertEqual(payload["selected_commands"], [])
        self.assertEqual(payload["risk_level"], "low")
        self.assertFalse(payload["manual_review_required"])

    def test_brain_tool_change_selects_brain_tool_tests(self) -> None:
        payload = build_verification_plan(paths=["daily_research/tools/brain_capsule.py"])

        joined = "\n".join(payload["selected_commands"])
        self.assertIn("daily_research/tools/tests/test_brain_capsule.py", joined)
        self.assertIn("daily_research/tools/tests/test_brain_workflow_cli.py", joined)
        self.assertEqual(payload["risk_level"], "medium")

    def test_data_lake_change_selects_data_lake_tests(self) -> None:
        payload = build_verification_plan(paths=["daily_research/data_lake/catalog.py"])

        self.assertEqual(
            payload["selected_commands"],
            [f"{PYTHON} -m pytest daily_research/data_lake/tests -q"],
        )
        self.assertEqual(payload["risk_level"], "medium")

    def test_forecast_change_selects_forecast_tests_without_rl_protocol_file(self) -> None:
        payload = build_verification_plan(paths=["daily_research/path_policy/forecast_training.py"])

        joined = "\n".join(payload["selected_commands"])
        self.assertIn("daily_research/path_policy/tests/test_forecast_features.py", joined)
        self.assertIn("daily_research/path_policy/tests/test_forecast_dataset.py", joined)
        self.assertIn("daily_research/path_policy/tests/test_forecast_training.py", joined)
        self.assertIn("daily_research/path_policy/tests/test_models.py", joined)
        self.assertNotIn("daily_research/path_policy/tests/test_rl_protocol.py -q", joined)

    def test_rl_protocol_change_uses_nodeid_groups_not_whole_file(self) -> None:
        payload = build_verification_plan(paths=["daily_research/path_policy/run_alpha_path20_protocol.py"])

        commands = payload["selected_commands"]
        self.assertGreaterEqual(len(commands), 3)
        self.assertTrue(all("daily_research/path_policy/tests/test_rl_protocol.py::" in command for command in commands))
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

        with patch("daily_research.tools.selective_verification._run_git", side_effect=fake_run_git):
            payload = build_verification_plan(base="HEAD~1")

        self.assertEqual(payload["changed_paths"], ["daily_research/output/active_execution_strategy.json"])
        self.assertEqual(payload["risk_level"], "critical")
        self.assertTrue(payload["manual_review_required"])

    def test_cli_outputs_json_for_explicit_paths(self) -> None:
        payload = run_cli("--paths", "daily_research/path_policy/forecast_features.py", "--json")

        self.assertEqual(payload["changed_paths"], ["daily_research/path_policy/forecast_features.py"])
        self.assertIn("always_commands", payload)
        self.assertIn("selected_commands", payload)

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


if __name__ == "__main__":
    unittest.main()
