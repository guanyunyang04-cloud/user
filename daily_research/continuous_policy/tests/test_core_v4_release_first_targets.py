import unittest

import pandas as pd

from daily_research.continuous_policy.core_v4_release_targets import (
    build_core_v4_release_first_targets,
)


class CoreV4ReleaseFirstTargetsTest(unittest.TestCase):
    def test_held_reduce_action_generates_release_target_and_coherent_delta(self) -> None:
        frame = pd.DataFrame(
            {
                "stock": ["SRC"],
                "current_weight": [0.20],
                "action_label": ["reduce"],
                "portfolio_daily_target_delta_intent": [0.0],
                "portfolio_daily_source_release_preference": [0.80],
                "portfolio_daily_source_forward_proxy_keep_risk": [0.05],
                "portfolio_daily_source_economic_block_risk": [0.02],
            }
        ).set_index("stock")

        targets = build_core_v4_release_first_targets(frame, deadband=0.003)

        self.assertGreater(targets.loc["SRC", "release_intent"], 0.0)
        self.assertLess(targets.loc["SRC", "target_delta"], -0.003)
        self.assertAlmostEqual(
            targets.loc["SRC", "target_delta"],
            targets.loc["SRC", "target_weight"] - 0.20,
            places=8,
        )

    def test_flat_names_never_generate_release(self) -> None:
        frame = pd.DataFrame(
            {
                "stock": ["FLAT"],
                "current_weight": [0.0],
                "action_label": ["exit"],
                "portfolio_daily_target_delta_intent": [-0.08],
                "portfolio_daily_source_release_preference": [1.0],
            }
        ).set_index("stock")

        targets = build_core_v4_release_first_targets(frame)

        self.assertEqual(targets.loc["FLAT", "release_intent"], 0.0)
        self.assertGreaterEqual(targets.loc["FLAT", "target_delta"], 0.0)

    def test_keep_and_block_risk_suppress_release(self) -> None:
        frame = pd.DataFrame(
            {
                "stock": ["KEEP_RISK", "BLOCK_RISK"],
                "current_weight": [0.20, 0.20],
                "action_label": ["reduce", "reduce"],
                "portfolio_daily_target_delta_intent": [-0.06, -0.06],
                "portfolio_daily_source_release_preference": [0.90, 0.90],
                "portfolio_daily_source_forward_proxy_keep_risk": [0.95, 0.05],
                "portfolio_daily_source_economic_block_risk": [0.05, 0.92],
            }
        ).set_index("stock")

        targets = build_core_v4_release_first_targets(frame)

        self.assertEqual(targets.loc["KEEP_RISK", "release_intent"], 0.0)
        self.assertEqual(targets.loc["BLOCK_RISK", "release_intent"], 0.0)
        self.assertGreaterEqual(targets.loc["KEEP_RISK", "target_delta"], -0.003)
        self.assertGreaterEqual(targets.loc["BLOCK_RISK", "target_delta"], -0.003)

    def test_receiver_rows_generate_receiver_support_with_headroom(self) -> None:
        frame = pd.DataFrame(
            {
                "stock": ["RCV", "NO_HEADROOM"],
                "current_weight": [0.00, 0.24],
                "action_label": ["open", "add"],
                "portfolio_daily_target_delta_intent": [0.05, 0.05],
                "portfolio_daily_receiver_score": [0.82, 0.90],
                "portfolio_daily_unified_receiver_score": [0.78, 0.92],
                "result_value_deploy_gate_target": [0.76, 0.95],
                "result_value_alpha_opportunity_value": [0.70, 0.90],
            }
        ).set_index("stock")

        targets = build_core_v4_release_first_targets(frame, position_cap=0.24)

        self.assertGreater(targets.loc["RCV", "receiver_score"], 0.0)
        self.assertEqual(targets.loc["RCV", "receiver_support"], 1.0)
        self.assertEqual(targets.loc["NO_HEADROOM", "receiver_support"], 0.0)

    def test_holding_flag_and_release_capacity_make_tiny_weights_reusable_for_training(self) -> None:
        frame = pd.DataFrame(
            {
                "stock": ["TINY_HELD"],
                "current_weight": [0.0005],
                "holding_flag": [1.0],
                "action_label": ["exit"],
                "target_delta_hint": [-0.50],
                "portfolio_daily_source_release_capacity": [0.12],
                "portfolio_daily_source_min_release_delta": [0.003],
                "portfolio_daily_source_release_preference": [0.85],
                "portfolio_daily_source_forward_proxy_keep_risk": [0.05],
                "portfolio_daily_source_economic_block_risk": [0.02],
            }
        ).set_index("stock")

        targets = build_core_v4_release_first_targets(frame)

        self.assertGreater(targets.loc["TINY_HELD", "release_intent"], 0.0)
        self.assertLess(targets.loc["TINY_HELD", "target_delta"], -0.003)
        self.assertGreaterEqual(targets.loc["TINY_HELD", "source_score"], targets.loc["TINY_HELD", "release_intent"])


if __name__ == "__main__":
    unittest.main()
