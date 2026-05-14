import unittest

from daily_research.continuous_policy.model_portfolio_set_v5 import (
    resolve_portfolio_set_v5_loss_profile,
)


class PortfolioSetV5ReleaseFirstLossTest(unittest.TestCase):
    def test_v48_and_alias_resolve_to_portfolio_set_release_first_loss(self) -> None:
        for name in ("alpha_result_value_budget_split_v48", "portfolio_set_release_first_decision_v1"):
            resolved, config = resolve_portfolio_set_v5_loss_profile(name)
            weights = config["multi_objective_loss_weights"]

            self.assertEqual(resolved, "alpha_result_value_budget_split_v48")
            self.assertEqual(weights["action_total"], 0.0)
            self.assertEqual(weights["duration_total"], 0.0)
            self.assertGreater(weights["target_weight_closure_total"], 0.0)
            self.assertGreater(weights["source_supply_total"], 0.0)
            self.assertGreater(weights["receiver_demand_total"], 0.0)
            self.assertGreater(weights["cash_buffer_total"], 0.0)
            self.assertGreater(weights["target_delta_weight_coherence_total"], 0.0)
            self.assertGreater(weights["release_flow_balance_total"], 0.0)
            self.assertGreater(weights["underdeployment_high_cash_total"], 0.0)
            self.assertGreater(weights["intent_translation_conflict_total"], 0.0)

    def test_legacy_losses_are_rejected_for_v5(self) -> None:
        for legacy in ("alpha_result_value_budget_split_v47", "alpha_result_value_budget_split_v46", "teacher_imitation"):
            with self.assertRaises(ValueError):
                resolve_portfolio_set_v5_loss_profile(legacy)


if __name__ == "__main__":
    unittest.main()
