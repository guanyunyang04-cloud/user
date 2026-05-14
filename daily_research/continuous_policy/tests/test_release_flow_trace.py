import unittest

import pandas as pd

from daily_research.continuous_policy.release_flow_trace import build_release_flow_trace


class ReleaseFlowTraceTest(unittest.TestCase):
    def test_trace_identifies_receiver_dead_when_source_intent_has_no_realization(self) -> None:
        policy = pd.DataFrame(
            {
                "stock": ["SRC", "RCV"],
                "current_weight": [0.20, 0.0],
                "portfolio_daily_target_weight_intent": [0.12, 0.0],
                "portfolio_daily_target_delta_intent": [-0.08, 0.0],
                "release_first_intent_score": [0.76, 0.0],
                "release_first_action_hint": ["reduce", "hold"],
                "release_first_block_reason": ["none", "not_held"],
                "portfolio_daily_source_executable_candidate": [1.0, 0.0],
                "portfolio_daily_receiver_executable_candidate": [0.0, 1.0],
                "portfolio_daily_receiver_score": [0.0, 0.0],
            }
        ).set_index("stock")

        trace = build_release_flow_trace(
            policy,
            allocation_result={
                "release_first_source_intent_count": 1,
                "release_first_source_realized_count": 0,
            },
        )

        self.assertEqual(trace["held_count"], 1)
        self.assertEqual(trace["held_negative_delta_count"], 1)
        self.assertEqual(trace["release_score_above_threshold_count"], 1)
        self.assertEqual(trace["source_intent_without_realization_count"], 1)
        self.assertEqual(trace["receiver_score_dead_count"], 1)
        self.assertEqual(trace["primary_blocker"], "receiver_score_dead")

    def test_trace_reports_target_delta_weight_conflict(self) -> None:
        policy = pd.DataFrame(
            {
                "stock": ["HELD"],
                "current_weight": [0.20],
                "portfolio_daily_target_weight_intent": [0.26],
                "portfolio_daily_target_delta_intent": [-0.06],
                "release_first_intent_score": [0.80],
                "release_first_action_hint": ["reduce"],
                "portfolio_daily_source_executable_candidate": [1.0],
            }
        ).set_index("stock")

        trace = build_release_flow_trace(policy)

        self.assertEqual(trace["target_delta_weight_conflict_count"], 1)
        self.assertEqual(trace["primary_blocker"], "target_delta_weight_conflict")

    def test_trace_handles_missing_columns_without_marking_clean(self) -> None:
        trace = build_release_flow_trace(pd.DataFrame({"stock": ["A"]}).set_index("stock"))

        self.assertEqual(trace["held_count"], 0)
        self.assertEqual(trace["source_executable_count"], 0)
        self.assertIn("not_held", trace["release_block_reason_counts"])
        self.assertEqual(trace["primary_blocker"], "no_held_source")


if __name__ == "__main__":
    unittest.main()
