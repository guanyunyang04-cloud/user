from __future__ import annotations

import unittest

import pandas as pd

from daily_research.continuous_policy.semantic_budget_intent import (
    derive_cash_timing_intent,
    derive_release_first_intent,
    derive_release_intent_from_target_delta,
)


class SemanticBudgetIntentTest(unittest.TestCase):
    def test_cash_timing_intent_increases_for_downside_and_falls_for_deploy_opportunity(self) -> None:
        defensive = derive_cash_timing_intent(
            risk_score=0.80,
            deploy_score=0.15,
            benchmark_downside=0.70,
            alpha_alignment=0.10,
        )
        opportunistic = derive_cash_timing_intent(
            risk_score=0.20,
            deploy_score=0.85,
            benchmark_downside=0.05,
            alpha_alignment=0.75,
        )

        self.assertGreater(defensive, 0.60)
        self.assertLess(opportunistic, 0.30)

    def test_release_intent_uses_negative_target_delta_for_held_names_only(self) -> None:
        frame = pd.DataFrame(
            {
                "stock": ["KEEP", "REDUCE", "EXIT", "NEW"],
                "current_weight": [0.20, 0.20, 0.12, 0.00],
                "portfolio_daily_target_delta_intent": [0.00, -0.04, -0.12, -0.02],
                "exit_hazard": [0.10, 0.20, 0.80, 0.90],
                "reduce_quality": [0.10, 0.70, 0.60, 0.90],
            }
        ).set_index("stock")

        result = derive_release_intent_from_target_delta(frame, deadband=0.01)

        self.assertEqual(result.loc["KEEP", "release_intent_action"], "hold")
        self.assertEqual(result.loc["REDUCE", "release_intent_action"], "reduce")
        self.assertEqual(result.loc["EXIT", "release_intent_action"], "exit")
        self.assertEqual(result.loc["NEW", "release_intent_action"], "hold")
        self.assertGreater(
            result.loc["EXIT", "release_intent_score"],
            result.loc["REDUCE", "release_intent_score"],
        )

    def test_release_first_intent_only_targets_held_names(self) -> None:
        frame = pd.DataFrame(
            {
                "stock": ["HELD", "FLAT"],
                "current_weight": [0.20, 0.00],
                "portfolio_daily_target_delta_intent": [-0.05, -0.05],
                "portfolio_daily_source_score": [0.80, 0.95],
                "portfolio_daily_source_release_quality": [0.75, 0.90],
                "exit_hazard": [0.50, 0.90],
                "portfolio_daily_source_forward_proxy_keep_risk": [0.10, 0.05],
                "portfolio_daily_source_economic_block_risk": [0.05, 0.05],
                "portfolio_daily_receiver_score": [0.0, 0.0],
            }
        ).set_index("stock")

        result = derive_release_first_intent(frame, deadband=0.01)

        self.assertGreater(result.loc["HELD", "release_first_intent_score"], 0.0)
        self.assertEqual(result.loc["HELD", "release_first_action_hint"], "reduce")
        self.assertEqual(result.loc["FLAT", "release_first_intent_score"], 0.0)
        self.assertEqual(result.loc["FLAT", "release_first_action_hint"], "hold")

    def test_release_first_negative_delta_raises_intent_and_keep_risk_blocks_it(self) -> None:
        frame = pd.DataFrame(
            {
                "stock": ["NEG_DELTA", "NO_DELTA", "KEEP_RISK"],
                "current_weight": [0.20, 0.20, 0.20],
                "portfolio_daily_target_delta_intent": [-0.06, 0.00, -0.06],
                "portfolio_daily_source_score": [0.80, 0.80, 0.80],
                "portfolio_daily_source_release_quality": [0.75, 0.75, 0.75],
                "exit_hazard": [0.40, 0.40, 0.40],
                "portfolio_daily_source_forward_proxy_keep_risk": [0.10, 0.10, 0.95],
                "portfolio_daily_source_economic_block_risk": [0.05, 0.05, 0.05],
                "portfolio_daily_receiver_score": [0.0, 0.0, 0.0],
            }
        ).set_index("stock")

        result = derive_release_first_intent(frame, deadband=0.01)

        self.assertGreater(
            result.loc["NEG_DELTA", "release_first_intent_score"],
            result.loc["NO_DELTA", "release_first_intent_score"],
        )
        self.assertLess(result.loc["KEEP_RISK", "release_first_intent_score"], 0.10)
        self.assertEqual(result.loc["KEEP_RISK", "release_first_block_reason"], "keep_risk")

    def test_release_first_block_risk_suppresses_release(self) -> None:
        frame = pd.DataFrame(
            {
                "stock": ["BLOCKED"],
                "current_weight": [0.24],
                "portfolio_daily_target_delta_intent": [-0.08],
                "portfolio_daily_source_score": [0.90],
                "portfolio_daily_source_release_quality": [0.90],
                "exit_hazard": [0.80],
                "portfolio_daily_source_forward_proxy_keep_risk": [0.10],
                "portfolio_daily_source_economic_block_risk": [0.92],
                "portfolio_daily_receiver_score": [0.0],
            }
        ).set_index("stock")

        result = derive_release_first_intent(frame, deadband=0.01)

        self.assertLess(result.loc["BLOCKED", "release_first_intent_score"], 0.10)
        self.assertEqual(result.loc["BLOCKED", "release_first_block_reason"], "economic_block")


if __name__ == "__main__":
    unittest.main()
