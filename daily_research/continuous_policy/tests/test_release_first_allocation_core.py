import unittest

import pandas as pd

from daily_research.continuous_policy.allocation_core_v3 import (
    ReleaseFirstAllocationConstraints,
    solve_release_first_allocation_v3,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).set_index("stock")
    frame.index = frame.index.astype(str)
    return frame


class ReleaseFirstAllocationCoreTest(unittest.TestCase):
    def test_budget_closed_but_spread_rotates_source_before_cash_deploy(self) -> None:
        frame = _frame(
            [
                {
                    "stock": "SRC",
                    "current_weight": 0.30,
                    "source": 1,
                    "receiver": 0,
                    "release_first_intent_score": 0.92,
                    "source_score": 0.94,
                    "portfolio_daily_source_forward_proxy_keep_risk": 0.05,
                    "portfolio_daily_source_economic_block_risk": 0.05,
                    "receiver_score": 0.0,
                },
                {
                    "stock": "KEEP",
                    "current_weight": 0.30,
                    "source": 1,
                    "receiver": 0,
                    "release_first_intent_score": 0.10,
                    "source_score": 0.20,
                    "portfolio_daily_source_forward_proxy_keep_risk": 0.80,
                    "portfolio_daily_source_economic_block_risk": 0.10,
                    "receiver_score": 0.0,
                },
                {
                    "stock": "RCV",
                    "current_weight": 0.20,
                    "source": 0,
                    "receiver": 1,
                    "release_first_intent_score": 0.0,
                    "source_score": 0.0,
                    "receiver_score": 0.95,
                },
            ]
        )

        result = solve_release_first_allocation_v3(
            frame,
            ReleaseFirstAllocationConstraints(
                gross_target=0.80,
                cash_reserve_target=0.20,
                turnover_limit=0.30,
                position_cap=0.35,
                transaction_cost_rate=0.0,
                cash_defense_intent=0.0,
            ),
        )

        self.assertEqual(result.diagnostics["release_first_source_intent_count"], 1)
        self.assertEqual(result.diagnostics["release_first_source_realized_count"], 1)
        self.assertGreater(result.source_funded_deploy_amount, 0.01)
        self.assertGreater(float(result.target_weight["RCV"]), 0.20)
        self.assertLess(float(result.target_weight["SRC"]), 0.30)

    def test_cash_limited_receiver_demand_uses_source_release(self) -> None:
        frame = _frame(
            [
                {"stock": "SRC1", "current_weight": 0.25, "source": 1, "receiver": 0, "release_first_intent_score": 0.88, "source_score": 0.90, "receiver_score": 0.0},
                {"stock": "SRC2", "current_weight": 0.25, "source": 1, "receiver": 0, "release_first_intent_score": 0.80, "source_score": 0.85, "receiver_score": 0.0},
                {"stock": "RCV1", "current_weight": 0.00, "source": 0, "receiver": 1, "release_first_intent_score": 0.0, "source_score": 0.0, "receiver_score": 0.95},
                {"stock": "RCV2", "current_weight": 0.00, "source": 0, "receiver": 1, "release_first_intent_score": 0.0, "source_score": 0.0, "receiver_score": 0.90},
            ]
        )

        result = solve_release_first_allocation_v3(
            frame,
            ReleaseFirstAllocationConstraints(
                gross_target=0.82,
                cash_reserve_target=0.05,
                turnover_limit=0.70,
                position_cap=0.30,
                transaction_cost_rate=0.0,
                available_cash_to_deploy=0.02,
            ),
        )

        self.assertGreater(result.source_funded_deploy_amount, 0.10)
        self.assertGreaterEqual(result.source_target_count, 1)
        self.assertGreaterEqual(result.receiver_target_count, 1)

    def test_keep_risk_and_block_risk_prevent_false_source(self) -> None:
        frame = _frame(
            [
                {"stock": "PROTECTED", "current_weight": 0.30, "source": 1, "receiver": 0, "release_first_intent_score": 0.95, "source_score": 0.95, "portfolio_daily_source_forward_proxy_keep_risk": 0.90, "portfolio_daily_source_economic_block_risk": 0.85, "receiver_score": 0.0},
                {"stock": "RCV", "current_weight": 0.00, "source": 0, "receiver": 1, "release_first_intent_score": 0.0, "source_score": 0.0, "receiver_score": 0.95},
            ]
        )

        result = solve_release_first_allocation_v3(
            frame,
            ReleaseFirstAllocationConstraints(
                gross_target=0.50,
                cash_reserve_target=0.05,
                turnover_limit=0.50,
                position_cap=0.35,
                transaction_cost_rate=0.0,
                available_cash_to_deploy=0.0,
            ),
        )

        self.assertEqual(result.source_target_count, 0)
        self.assertEqual(result.diagnostics["release_first_source_realized_count"], 0)
        self.assertEqual(result.diagnostics["release_first_block_reason"], "source_blocked_by_keep_or_economic_risk")

    def test_high_cash_defense_buffers_release_as_cash(self) -> None:
        frame = _frame(
            [
                {"stock": "SRC", "current_weight": 0.30, "source": 1, "receiver": 0, "release_first_intent_score": 0.95, "source_score": 0.90, "receiver_score": 0.0},
                {"stock": "RCV", "current_weight": 0.00, "source": 0, "receiver": 1, "release_first_intent_score": 0.0, "source_score": 0.0, "receiver_score": 0.95},
            ]
        )

        result = solve_release_first_allocation_v3(
            frame,
            ReleaseFirstAllocationConstraints(
                gross_target=0.30,
                cash_reserve_target=0.20,
                turnover_limit=0.40,
                position_cap=0.35,
                transaction_cost_rate=0.0,
                cash_defense_intent=0.90,
            ),
        )

        self.assertGreater(result.diagnostics["release_first_cash_buffer_amount"], 0.0)
        self.assertEqual(result.source_target_count, 1)
        self.assertEqual(result.receiver_target_count, 0)

    def test_turnover_cap_position_cap_and_nonheld_source_are_respected(self) -> None:
        frame = _frame(
            [
                {"stock": "SRC_HELD", "current_weight": 0.10, "source": 1, "receiver": 0, "release_first_intent_score": 0.95, "source_score": 0.95, "receiver_score": 0.0},
                {"stock": "SRC_FLAT", "current_weight": 0.00, "source": 1, "receiver": 0, "release_first_intent_score": 1.00, "source_score": 1.00, "receiver_score": 0.0},
                {"stock": "RCV", "current_weight": 0.00, "source": 0, "receiver": 1, "release_first_intent_score": 0.0, "source_score": 0.0, "receiver_score": 0.99},
            ]
        )

        result = solve_release_first_allocation_v3(
            frame,
            ReleaseFirstAllocationConstraints(
                gross_target=0.70,
                cash_reserve_target=0.05,
                turnover_limit=0.10,
                position_cap=0.12,
                transaction_cost_rate=0.0,
                available_cash_to_deploy=0.0,
            ),
        )

        self.assertLessEqual(result.buy_turnover + result.sell_turnover, 0.10 + 1.0e-9)
        self.assertLessEqual(float(result.target_weight.max()), 0.12 + 1.0e-9)
        self.assertEqual(float(result.target_weight["SRC_FLAT"]), 0.0)


if __name__ == "__main__":
    unittest.main()
