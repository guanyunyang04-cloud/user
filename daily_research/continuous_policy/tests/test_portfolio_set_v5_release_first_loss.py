import unittest

import torch

from daily_research.continuous_policy.model_portfolio_set_v5 import (
    PORTFOLIO_SET_V5_INTERNAL_VERSION,
    _portfolio_set_loss,
    portfolio_set_v5_decision_diagnostics,
    project_portfolio_set_v5_cashflow_oracle,
    resolve_portfolio_set_v5_loss_profile,
)


class PortfolioSetV5ReleaseFirstLossTest(unittest.TestCase):
    def test_v48_and_alias_resolve_to_portfolio_set_release_first_loss(self) -> None:
        expected_names = {
            "alpha_result_value_budget_split_v48": "alpha_result_value_budget_split_v48",
            "portfolio_set_release_first_decision_v1": PORTFOLIO_SET_V5_INTERNAL_VERSION,
            PORTFOLIO_SET_V5_INTERNAL_VERSION: PORTFOLIO_SET_V5_INTERNAL_VERSION,
        }
        for name, expected_resolved in expected_names.items():
            resolved, config = resolve_portfolio_set_v5_loss_profile(name)
            weights = config["multi_objective_loss_weights"]

            self.assertEqual(resolved, expected_resolved)
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
            self.assertGreater(weights["decision_oracle_total"], 0.0)
            self.assertGreater(weights["pg_dfl_surrogate_total"], 0.0)
            self.assertGreater(weights["constraint_violation_total"], 0.0)

    def test_legacy_losses_are_rejected_for_v5(self) -> None:
        for legacy in ("alpha_result_value_budget_split_v47", "alpha_result_value_budget_split_v46", "teacher_imitation"):
            with self.assertRaises(ValueError):
                resolve_portfolio_set_v5_loss_profile(legacy)

    def test_torch_projection_oracle_conserves_source_receiver_cash(self) -> None:
        current = torch.tensor([[0.20, 0.00, 0.12]], dtype=torch.float32)
        source_score = torch.tensor([[0.95, 0.00, 0.10]], dtype=torch.float32)
        receiver_score = torch.tensor([[0.00, 0.98, 0.05]], dtype=torch.float32)

        oracle = project_portfolio_set_v5_cashflow_oracle(
            current_weight=current,
            source_score=source_score,
            receiver_score=receiver_score,
            cash_buffer_score=torch.zeros_like(current),
            sample_mask=torch.ones_like(current, dtype=torch.bool),
            turnover_budget=torch.tensor([0.18], dtype=torch.float32),
            risk_budget=torch.tensor([0.10], dtype=torch.float32),
        )

        self.assertGreater(float(oracle["source_supply"][0, 0]), 0.003)
        self.assertGreater(float(oracle["receiver_demand"][0, 1]), 0.003)
        self.assertLessEqual(float(oracle["target_weight"].max()), 0.24)
        self.assertGreaterEqual(float(oracle["target_weight"].min()), 0.0)
        self.assertLess(float(oracle["constraint_violation"][0]), 1.0e-6)
        cash_now = 1.0 - float(current.sum())
        cash_after = float(oracle["cash_buffer"][0])
        source = float(oracle["source_supply"].sum())
        receiver = float(oracle["receiver_demand"].sum())
        self.assertAlmostEqual(cash_after, cash_now + source - receiver, places=6)

    def test_pg_dfl_surrogate_prefers_oracle_aligned_logits(self) -> None:
        weights = resolve_portfolio_set_v5_loss_profile(PORTFOLIO_SET_V5_INTERNAL_VERSION)[1]["multi_objective_loss_weights"]
        current = torch.tensor([[0.20, 0.00]], dtype=torch.float32)
        decision_target = torch.tensor(
            [[[0.08, 0.00, 0.50, 0.12, 0.18, 0.10, 0.00, 0.08], [0.00, 0.08, 0.50, 0.08, 0.18, 0.10, 0.00, 0.08]]],
            dtype=torch.float32,
        )
        target_y = torch.tensor(
            [[[0.12, -0.08, 1.00, 0.00, 0.50, 1.00, 1.00, 0.00], [0.08, 0.08, 0.00, 1.00, 0.50, 0.00, 0.00, 0.00]]],
            dtype=torch.float32,
        )
        batch = {
            "target_y": target_y,
            "decision_target_y": decision_target,
            "sample_mask": torch.ones((1, 2), dtype=torch.bool),
            "current_weight": current,
        }
        aligned = torch.tensor([[[0.0, -0.5, 3.0, -3.0, 0.0, 3.0, 3.0, -3.0], [-1.0, 0.5, -3.0, 3.0, 0.0, -3.0, -3.0, -3.0]]], dtype=torch.float32)
        inverted = torch.tensor([[[0.0, 0.5, -3.0, 3.0, 0.0, -3.0, -3.0, -3.0], [-1.0, -0.5, 3.0, -3.0, 0.0, 3.0, 3.0, -3.0]]], dtype=torch.float32)

        aligned_loss = float(_portfolio_set_loss(aligned, batch, weights))
        inverted_loss = float(_portfolio_set_loss(inverted, batch, weights))
        diagnostics = portfolio_set_v5_decision_diagnostics(aligned, batch, weights)

        self.assertLess(aligned_loss, inverted_loss)
        self.assertGreater(diagnostics["decision_oracle_value_mean"], 0.0)
        self.assertGreater(diagnostics["release_flow_source_target_count"], 0.0)
        self.assertGreater(diagnostics["release_flow_receiver_target_count"], 0.0)


if __name__ == "__main__":
    unittest.main()
