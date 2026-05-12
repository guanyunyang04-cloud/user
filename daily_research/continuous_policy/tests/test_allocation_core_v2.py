import unittest

import pandas as pd

from daily_research.continuous_policy.allocation_core_v2 import (
    AllocationCoreV2Constraints,
    solve_cash_funded_allocation_v2,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).set_index("stock")
    frame.index = frame.index.astype(str)
    return frame


class AllocationCoreV2Test(unittest.TestCase):
    def test_cash_rich_receiver_supported_case_deploys_idle_cash_first(self) -> None:
        frame = _frame(
            [
                {"stock": "H1", "current_weight": 0.14, "receiver": 0, "source": 1, "receiver_score": 0.1, "source_score": 0.4},
                {"stock": "H2", "current_weight": 0.14, "receiver": 0, "source": 1, "receiver_score": 0.1, "source_score": 0.3},
                {"stock": "R1", "current_weight": 0.00, "receiver": 1, "source": 0, "receiver_score": 0.95, "source_score": 0.0},
                {"stock": "R2", "current_weight": 0.00, "receiver": 1, "source": 0, "receiver_score": 0.90, "source_score": 0.0},
                {"stock": "R3", "current_weight": 0.00, "receiver": 1, "source": 0, "receiver_score": 0.85, "source_score": 0.0},
                {"stock": "R4", "current_weight": 0.00, "receiver": 1, "source": 0, "receiver_score": 0.80, "source_score": 0.0},
            ]
        )

        result = solve_cash_funded_allocation_v2(
            frame,
            AllocationCoreV2Constraints(
                gross_target=0.85,
                cash_reserve_target=0.05,
                turnover_limit=1.0,
                position_cap=0.20,
                transaction_cost_rate=0.0,
            ),
        )

        self.assertGreaterEqual(float(result.target_weight.sum()), 0.75)
        self.assertGreater(result.cash_funded_deploy_amount, 0.45)
        self.assertFalse(result.source_release_required)
        self.assertEqual(result.source_target_count, 0)
        self.assertEqual(result.underdeployment_reason, "none")

    def test_cash_limited_receiver_demand_uses_source_funded_rotation(self) -> None:
        frame = _frame(
            [
                {"stock": "H1", "current_weight": 0.25, "receiver": 0, "source": 1, "receiver_score": 0.0, "source_score": 0.95},
                {"stock": "H2", "current_weight": 0.25, "receiver": 0, "source": 1, "receiver_score": 0.0, "source_score": 0.90},
                {"stock": "R1", "current_weight": 0.00, "receiver": 1, "source": 0, "receiver_score": 0.95, "source_score": 0.0},
                {"stock": "R2", "current_weight": 0.00, "receiver": 1, "source": 0, "receiver_score": 0.90, "source_score": 0.0},
                {"stock": "R3", "current_weight": 0.00, "receiver": 1, "source": 0, "receiver_score": 0.85, "source_score": 0.0},
            ]
        )

        result = solve_cash_funded_allocation_v2(
            frame,
            AllocationCoreV2Constraints(
                gross_target=0.85,
                cash_reserve_target=0.05,
                turnover_limit=1.0,
                position_cap=0.25,
                transaction_cost_rate=0.0,
                available_cash_to_deploy=0.05,
            ),
        )

        self.assertTrue(result.source_release_required)
        self.assertGreater(result.source_funded_deploy_amount, 0.10)
        self.assertGreaterEqual(result.source_target_count, 1)
        self.assertGreaterEqual(result.receiver_target_count, 1)

    def test_receiver_headroom_insufficient_is_explicit_underdeployment_reason(self) -> None:
        frame = _frame(
            [
                {"stock": "H1", "current_weight": 0.14, "receiver": 0, "source": 1, "receiver_score": 0.0, "source_score": 0.3},
                {"stock": "H2", "current_weight": 0.14, "receiver": 0, "source": 1, "receiver_score": 0.0, "source_score": 0.2},
                {"stock": "R1", "current_weight": 0.18, "receiver": 1, "source": 0, "receiver_score": 0.95, "source_score": 0.0},
                {"stock": "R2", "current_weight": 0.18, "receiver": 1, "source": 0, "receiver_score": 0.90, "source_score": 0.0},
            ]
        )

        result = solve_cash_funded_allocation_v2(
            frame,
            AllocationCoreV2Constraints(
                gross_target=0.85,
                cash_reserve_target=0.05,
                turnover_limit=1.0,
                position_cap=0.20,
                transaction_cost_rate=0.0,
            ),
        )

        self.assertLess(float(result.target_weight.sum()), 0.75)
        self.assertEqual(result.underdeployment_reason, "receiver_headroom_insufficient")
        self.assertGreater(result.diagnostics["target_sum_gap"], 0.10)

    def test_turnover_limit_is_explicit_underdeployment_reason(self) -> None:
        frame = _frame(
            [
                {"stock": "H1", "current_weight": 0.14, "receiver": 0, "source": 1, "receiver_score": 0.0, "source_score": 0.5},
                {"stock": "H2", "current_weight": 0.14, "receiver": 0, "source": 1, "receiver_score": 0.0, "source_score": 0.4},
                {"stock": "R1", "current_weight": 0.00, "receiver": 1, "source": 0, "receiver_score": 0.95, "source_score": 0.0},
                {"stock": "R2", "current_weight": 0.00, "receiver": 1, "source": 0, "receiver_score": 0.90, "source_score": 0.0},
                {"stock": "R3", "current_weight": 0.00, "receiver": 1, "source": 0, "receiver_score": 0.85, "source_score": 0.0},
            ]
        )

        result = solve_cash_funded_allocation_v2(
            frame,
            AllocationCoreV2Constraints(
                gross_target=0.85,
                cash_reserve_target=0.05,
                turnover_limit=0.10,
                position_cap=0.25,
                transaction_cost_rate=0.0,
            ),
        )

        self.assertLess(float(result.target_weight.sum()), 0.50)
        self.assertEqual(result.underdeployment_reason, "turnover_limited")
        self.assertLessEqual(result.buy_turnover + result.sell_turnover, 0.10 + 1.0e-9)

    def test_unsupported_receiver_and_nonheld_source_are_never_traded(self) -> None:
        frame = _frame(
            [
                {"stock": "H1", "current_weight": 0.30, "receiver": 0, "source": 1, "receiver_score": 0.0, "source_score": 0.95},
                {"stock": "R_SUPPORTED", "current_weight": 0.00, "receiver": 1, "source": 0, "receiver_score": 0.80, "source_score": 0.0},
                {"stock": "R_UNSUPPORTED", "current_weight": 0.00, "receiver": 0, "source": 0, "receiver_score": 1.00, "source_score": 0.0},
                {"stock": "S_NONHELD", "current_weight": 0.00, "receiver": 0, "source": 1, "receiver_score": 0.0, "source_score": 1.00},
            ]
        )

        result = solve_cash_funded_allocation_v2(
            frame,
            AllocationCoreV2Constraints(
                gross_target=0.75,
                cash_reserve_target=0.05,
                turnover_limit=1.0,
                position_cap=0.30,
                transaction_cost_rate=0.0,
            ),
        )

        self.assertEqual(float(result.target_weight["R_UNSUPPORTED"]), 0.0)
        self.assertEqual(float(result.target_weight["S_NONHELD"]), 0.0)
        self.assertGreater(float(result.target_weight["R_SUPPORTED"]), 0.0)
        self.assertGreaterEqual(float(result.target_weight["H1"]), 0.0)


if __name__ == "__main__":
    unittest.main()
