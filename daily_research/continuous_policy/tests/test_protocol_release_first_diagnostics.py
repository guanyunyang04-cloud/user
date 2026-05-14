import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from daily_research.continuous_policy.protocol_release_diagnostics import (
    enrich_continuity_metrics_with_release_first_diagnostics,
    summarize_release_first_turnover_diagnostics,
)


class ProtocolReleaseFirstDiagnosticsTest(unittest.TestCase):
    def test_summarizes_release_first_source_cash_and_intent_metrics_from_turnover_csv(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "daily_turnover.csv"
            pd.DataFrame(
                {
                    "release_first_allocation_v3_mode_used": [1.0, 1.0],
                    "release_first_allocation_v3_used": [1.0, 0.0],
                    "release_first_intent_target_count": [2.0, 1.0],
                    "release_first_source_intent_count": [2.0, 1.0],
                    "release_first_source_realized_count": [1.0, 0.0],
                    "release_first_rotation_amount": [0.05, 0.01],
                    "release_first_cash_buffer_amount": [0.02, 0.00],
                    "portfolio_daily_source_target_count": [2.0, 0.0],
                    "release_flow_held_count": [3.0, 2.0],
                    "release_flow_source_executable_count": [2.0, 1.0],
                    "release_flow_receiver_executable_count": [2.0, 1.0],
                    "release_flow_held_negative_delta_count": [2.0, 1.0],
                    "release_flow_receiver_positive_delta_count": [2.0, 1.0],
                    "release_flow_release_score_above_threshold_count": [2.0, 1.0],
                    "release_flow_release_action_hint_count": [2.0, 1.0],
                    "release_flow_source_intent_without_realization_count": [1.0, 1.0],
                    "release_flow_receiver_score_dead_count": [0.0, 1.0],
                    "release_flow_target_delta_weight_conflict_count": [0.0, 0.0],
                    "allocation_layer_target_sum_gap": [0.02, 0.04],
                    "cash_weight": [0.20, 0.30],
                    "intent_translation_conflict_rate": [0.00, 0.02],
                }
            ).to_csv(path, index=False)

            diagnostics = summarize_release_first_turnover_diagnostics(path)

        metrics = diagnostics["metrics"]
        self.assertEqual(metrics["release_first_allocation_v3_mode_used"], 1.0)
        self.assertEqual(metrics["release_first_allocation_v3_used"], 0.5)
        self.assertEqual(metrics["release_first_source_intent_count"], 1.5)
        self.assertEqual(metrics["release_first_source_realized_count"], 0.5)
        self.assertAlmostEqual(metrics["release_first_rotation_amount"], 0.03)
        self.assertAlmostEqual(metrics["release_first_cash_buffer_amount"], 0.01)
        self.assertEqual(metrics["portfolio_daily_source_target_count"], 1.0)
        self.assertEqual(metrics["portfolio_daily_source_realized_sell_rate"], 0.5)
        self.assertAlmostEqual(metrics["portfolio_daily_target_sum_gap"], 0.03)
        self.assertAlmostEqual(metrics["portfolio_daily_actual_cash_weight_mean"], 0.25)
        self.assertAlmostEqual(metrics["intent_translation_conflict_rate"], 0.01)
        self.assertEqual(metrics["release_flow_held_negative_delta_count"], 1.5)
        self.assertEqual(metrics["release_flow_receiver_score_dead_count"], 0.5)
        self.assertEqual(metrics["release_flow_target_delta_weight_conflict_count"], 0.0)
        self.assertEqual(diagnostics["missing_columns"], [])

    def test_missing_columns_fill_safe_defaults_and_record_missing_columns(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "daily_turnover.csv"
            pd.DataFrame({"cash_weight": [0.45]}).to_csv(path, index=False)

            diagnostics = summarize_release_first_turnover_diagnostics(path)

        metrics = diagnostics["metrics"]
        self.assertEqual(metrics["release_first_source_intent_count"], 0.0)
        self.assertEqual(metrics["portfolio_daily_source_target_count"], 0.0)
        self.assertEqual(metrics["portfolio_daily_source_realized_sell_rate"], 0.0)
        self.assertEqual(metrics["portfolio_daily_actual_cash_weight_mean"], 0.45)
        self.assertIn("release_first_source_intent_count", diagnostics["missing_columns"])
        self.assertIn("release_flow_held_negative_delta_count", diagnostics["missing_columns"])
        self.assertIn("portfolio_daily_source_target_count", diagnostics["missing_columns"])

    def test_enriches_existing_continuity_metrics_without_dropping_existing_values(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "daily_turnover.csv"
            pd.DataFrame(
                {
                    "release_first_source_intent_count": [1.0],
                    "portfolio_daily_source_target_count": [1.0],
                    "release_first_source_realized_count": [1.0],
                    "allocation_layer_target_sum_gap": [0.01],
                    "cash_weight": [0.22],
                    "intent_translation_conflict_rate": [0.0],
                }
            ).to_csv(path, index=False)

            continuity = enrich_continuity_metrics_with_release_first_diagnostics(
                {"cash_timing_quality_1d": -0.12},
                path,
            )

        self.assertEqual(continuity["cash_timing_quality_1d"], -0.12)
        self.assertEqual(continuity["release_first_source_intent_count"], 1.0)
        self.assertEqual(continuity["portfolio_daily_source_realized_sell_rate"], 1.0)
        self.assertEqual(continuity["portfolio_daily_actual_cash_weight_mean"], 0.22)


if __name__ == "__main__":
    unittest.main()
