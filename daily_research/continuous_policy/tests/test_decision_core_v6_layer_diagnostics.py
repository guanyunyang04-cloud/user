import unittest
from unittest.mock import MagicMock, patch

from daily_research.continuous_policy import diagnose_decision_core_v6_layers as diagnostics
from daily_research.continuous_policy.decision_core_v6 import DECISION_CORE_V6_STRICT_GOLD_DATASET_ID
from daily_research.continuous_policy.portfolio_simulator import (
    BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
    BUDGET_SEMANTICS_ALLOCATION_LAYER,
)


class DecisionCoreV6LayerDiagnosticsTest(unittest.TestCase):
    def test_parser_defaults_to_v6_execution_path(self) -> None:
        args = diagnostics.build_parser().parse_args(["--model-path", "artifact.pt"])

        self.assertEqual(args.budget_semantics, BUDGET_SEMANTICS_ALLOCATION_LAYER)
        self.assertEqual(args.budget_calibration, BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER)

    def test_main_rejects_non_v6_execution_path_without_counterfactual_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "allocation_layer_v1"):
            diagnostics.main(
                [
                    "--model-path",
                    "artifact.pt",
                    "--training-dataset-id",
                    DECISION_CORE_V6_STRICT_GOLD_DATASET_ID,
                    "--budget-semantics",
                    "legacy_total_candidate",
                ]
            )

    def test_main_allows_non_v6_execution_path_when_marked_counterfactual(self) -> None:
        with patch.object(diagnostics, "load_artifact", return_value=MagicMock(train_summary={})), patch.object(
            diagnostics,
            "prepare_policy_inputs",
            side_effect=RuntimeError("stop_after_path_validation"),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop_after_path_validation"):
                diagnostics.main(
                    [
                        "--model-path",
                        "artifact.pt",
                        "--training-dataset-id",
                        DECISION_CORE_V6_STRICT_GOLD_DATASET_ID,
                        "--budget-semantics",
                        "legacy_total_candidate",
                        "--allow-non-v6-execution-path",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
