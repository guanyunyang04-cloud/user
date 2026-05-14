import unittest

import pandas as pd

from daily_research.continuous_policy.portfolio_cashflow_decision import (
    normalize_portfolio_cashflow_decision,
)
from daily_research.continuous_policy.portfolio_simulator import HoldingState, PortfolioState
from daily_research.continuous_policy.release_flow_trace import build_release_flow_trace


class PortfolioCashflowDecisionTest(unittest.TestCase):
    def _cashflow_policy(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "stock": ["SRC", "RCV", "CASH_ONLY"],
                "current_weight": [0.30, 0.0, 0.0],
                "portfolio_cashflow_decision_v1_mode": [1.0, 1.0, 1.0],
                "portfolio_daily_target_weight_intent": [0.16, 0.14, 0.0],
                "portfolio_daily_target_delta_intent": [-0.14, 0.14, 0.0],
                "portfolio_set_v5_source_supply": [0.14, 0.0, 0.0],
                "portfolio_set_v5_receiver_demand": [0.0, 0.14, 0.0],
                "portfolio_set_v5_cash_buffer_score": [0.05, 0.05, 0.05],
                "portfolio_set_v5_turnover_budget": [0.50, 0.50, 0.50],
                "portfolio_set_v5_turnover_used": [0.28, 0.28, 0.28],
                "portfolio_set_v5_oracle_feasible": [1.0, 1.0, 1.0],
                "portfolio_daily_source_score": [0.95, 0.0, 0.0],
                "portfolio_daily_receiver_score": [0.0, 0.96, 0.0],
            }
        ).set_index("stock")

    def test_contract_normalizes_source_receiver_cashflow_semantics(self) -> None:
        decision = normalize_portfolio_cashflow_decision(
            self._cashflow_policy(),
            position_cap=0.30,
            turnover_limit=0.50,
        )

        self.assertTrue(decision.valid)
        self.assertEqual(int(decision.source_target.sum()), 1)
        self.assertEqual(int(decision.receiver_target.sum()), 1)
        self.assertEqual(decision.action_label.loc["SRC"], "reduce")
        self.assertEqual(decision.action_label.loc["RCV"], "open")
        self.assertAlmostEqual(
            float(decision.frame.loc["SRC", "portfolio_daily_target_delta_intent"]),
            float(decision.frame.loc["SRC", "portfolio_daily_target_weight_intent"]) - 0.30,
            places=8,
        )
        self.assertEqual(float(decision.diagnostics["cashflow_decision_violation_count"]), 0.0)

    def test_contract_fail_closes_on_direction_conflict(self) -> None:
        policy = self._cashflow_policy()
        policy.loc["SRC", "portfolio_daily_target_delta_intent"] = 0.04
        policy.loc["SRC", "portfolio_daily_target_weight_intent"] = 0.34

        decision = normalize_portfolio_cashflow_decision(
            policy,
            position_cap=0.35,
            turnover_limit=0.50,
            fail_closed=True,
        )

        self.assertFalse(decision.valid)
        self.assertEqual(int(decision.source_target.sum()), 0)
        self.assertEqual(int(decision.receiver_target.sum()), 0)
        self.assertEqual(decision.action_label.loc["SRC"], "hold")
        self.assertIn("source_direction_conflict", decision.diagnostics["cashflow_decision_invalid_reason"])

    def test_simulator_cashflow_mode_uses_v5_target_without_release_first_redecode(self) -> None:
        state = PortfolioState(
            cash_weight=0.70,
            holdings={"SRC": HoldingState(weight=0.30, entry_price=10.0, peak_price=11.0)},
            max_positions=4,
            max_position_weight=0.50,
            turnover_limit=1.00,
        )

        result = state.step(
            date="2026-05-14",
            prices=pd.Series({"SRC": 10.0, "RCV": 10.0, "CASH_ONLY": 10.0}),
            policy_frame=self._cashflow_policy(),
            global_targets={
                "gross_exposure_target": 0.30,
                "turnover_budget": 1.00,
                "max_position_weight_target": 0.50,
                "cash_reserve_target": 0.05,
                "release_first_allocation_v3_mode": 1.0,
                "portfolio_cashflow_decision_v1_mode": 1.0,
            },
            budget_semantics="allocation_layer_v1",
            budget_calibration="end_to_end_allocation_layer_v1",
        )

        self.assertEqual(result.diagnostics["cashflow_decision_used"], 1.0)
        self.assertEqual(result.diagnostics["cashflow_decision_failed_closed"], 0.0)
        self.assertEqual(result.diagnostics["release_first_allocation_v3_used"], 0.0)
        self.assertEqual(result.diagnostics["portfolio_daily_source_target_count"], 1)
        self.assertEqual(result.diagnostics["portfolio_daily_receiver_target_count"], 1)
        self.assertEqual(result.diagnostics["intent_translation_conflict_rate"], 0.0)
        self.assertEqual(result.diagnostics["release_flow_primary_blocker"], "none")
        actions = {item["stock"]: item for item in result.actions}
        self.assertEqual(actions["SRC"]["weight_change_action"], "reduce")
        self.assertEqual(actions["RCV"]["weight_change_action"], "open")
        self.assertEqual(actions["SRC"]["portfolio_cashflow_decision_v1_mode"], 1.0)
        self.assertEqual(actions["RCV"]["portfolio_cashflow_decision_v1_mode"], 1.0)
        self.assertEqual(actions["SRC"]["portfolio_set_v5_value_arbitration_mode"], 0.0)
        self.assertEqual(actions["RCV"]["portfolio_set_v5_value_arbitration_mode"], 0.0)
        self.assertEqual(actions["SRC"]["portfolio_daily_source_target_intent"], True)
        self.assertEqual(actions["RCV"]["portfolio_daily_receiver_target_intent"], True)
        self.assertGreater(actions["SRC"]["portfolio_set_v5_source_supply"], 0.003)
        self.assertGreater(actions["RCV"]["portfolio_set_v5_receiver_demand"], 0.003)
        self.assertEqual(actions["SRC"]["portfolio_cashflow_decision_v1_invalid_reason"], "none")

    def test_simulator_cashflow_mode_uses_v5_turnover_contract_limit(self) -> None:
        policy = self._cashflow_policy()
        policy["portfolio_set_v5_turnover_budget"] = 0.32
        state = PortfolioState(
            cash_weight=0.70,
            holdings={"SRC": HoldingState(weight=0.30, entry_price=10.0, peak_price=11.0)},
            max_positions=4,
            max_position_weight=0.50,
            turnover_limit=1.00,
        )

        result = state.step(
            date="2026-05-14",
            prices=pd.Series({"SRC": 10.0, "RCV": 10.0, "CASH_ONLY": 10.0}),
            policy_frame=policy,
            global_targets={
                "gross_exposure_target": 0.30,
                "turnover_budget": 1.00,
                "max_position_weight_target": 0.50,
                "cash_reserve_target": 0.05,
                "portfolio_cashflow_decision_v1_mode": 1.0,
            },
            budget_semantics="allocation_layer_v1",
            budget_calibration="end_to_end_allocation_layer_v1",
        )

        self.assertEqual(result.diagnostics["cashflow_decision_used"], 1.0)
        self.assertEqual(result.diagnostics["cashflow_decision_failed_closed"], 0.0)
        self.assertLessEqual(result.diagnostics["cashflow_decision_turnover"], 0.32 + 1.0e-8)

    def test_release_trace_reports_cashflow_decision_evidence(self) -> None:
        trace = build_release_flow_trace(self._cashflow_policy())

        self.assertEqual(trace["cashflow_decision_source_intent_count"], 1)
        self.assertEqual(trace["cashflow_decision_receiver_intent_count"], 1)
        self.assertEqual(trace["held_negative_delta_count"], 1)
        self.assertEqual(trace["receiver_positive_delta_count"], 1)
        self.assertEqual(trace["primary_blocker"], "none")


if __name__ == "__main__":
    unittest.main()
