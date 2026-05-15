import unittest
from types import SimpleNamespace

import pandas as pd

from daily_research.continuous_policy.portfolio_decision_features import (
    PORTFOLIO_DECISION_FEATURE_BUNDLE_MODE_COLUMN,
    attach_portfolio_decision_features,
    build_lake_feature_utilization_report,
)


class PortfolioDecisionFeatureBundleTest(unittest.TestCase):
    def test_attach_portfolio_decision_features_maps_lake_state_to_oracle_fields(self) -> None:
        state = pd.DataFrame(
            {
                "stock": ["AAA", "BBB", "CCC"],
                "score_blend": [-0.8, 1.7, 0.4],
                "score_rank_pct": [0.1, 1.0, 0.6],
                "ret_1d": [-0.02, 0.03, 0.0],
                "ret_3d": [-0.04, 0.05, 0.01],
                "ret_5d": [-0.06, 0.07, 0.02],
                "ret_accel_5_20": [-0.02, 0.04, 0.01],
                "vol_20d": [0.03, 0.02, 0.04],
                "distance_to_20d_high": [-0.12, -0.01, -0.04],
                "distance_to_20d_low": [0.01, 0.10, 0.03],
                "in_pool": [1.0, 1.0, 1.0],
                "current_weight": [0.12, 0.0, 0.0],
                "holding_flag": [1.0, 0.0, 0.0],
                "market_downside_pressure": [0.05, 0.05, 0.05],
            }
        )

        enriched = attach_portfolio_decision_features(state, mode="predict")

        self.assertEqual(float(enriched[PORTFOLIO_DECISION_FEATURE_BUNDLE_MODE_COLUMN].iloc[0]), 1.0)
        self.assertIn("deploy_value_target", enriched.columns)
        self.assertIn("portfolio_daily_receiver_score", enriched.columns)
        self.assertIn("portfolio_daily_source_release_quality", enriched.columns)
        self.assertGreater(float(enriched.loc[1, "deploy_value_target"]), float(enriched.loc[0, "deploy_value_target"]))
        self.assertGreater(float(enriched.loc[1, "portfolio_daily_receiver_score"]), 0.05)
        self.assertGreater(float(enriched.loc[0, "portfolio_daily_source_release_quality"]), 0.0)
        self.assertGreaterEqual(float(enriched["portfolio_decision_feature_contract_missing_count"].iloc[0]), 0.0)
        self.assertEqual(str(enriched["portfolio_decision_feature_contract_blocker"].iloc[0]), "none")

    def test_lake_feature_utilization_report_flags_unused_high_value_features(self) -> None:
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        stocks = ["AAA", "BBB"]
        score = pd.DataFrame([[0.1, 0.2], [0.2, 0.4], [0.3, 0.6]], index=dates, columns=stocks)
        unused_alpha = pd.DataFrame([[1.0, 0.2], [1.1, 0.3], [1.2, 0.4]], index=dates, columns=stocks)
        membership = pd.DataFrame(True, index=dates, columns=stocks)
        prepared = SimpleNamespace(
            data_source="lake",
            pool_name="learned_all_a",
            start_date="2026-01-05",
            end_date="2026-01-07",
            close=score + 10.0,
            open_=score + 9.9,
            high=score + 10.2,
            low=score + 9.8,
            volume=score + 1000.0,
            amount=score + 10000.0,
            score_none=score,
            score_v2=score,
            score_blend=score,
            feature_frames={"unused_alpha_edge": unused_alpha},
            derived_frames={"ret_1d": score.diff().fillna(0.0)},
            market_features={},
            membership_frame=membership,
        )

        report = build_lake_feature_utilization_report(
            prepared,
            used_feature_names=["score_blend", "ret_1d"],
            daily_feature_names=[],
        )

        self.assertEqual(report["status"], "ok")
        self.assertGreaterEqual(report["feature_panel_count"], 1)
        self.assertIn("score_blend", report["actually_used_features"])
        self.assertIn("unused_alpha_edge", report["unused_high_value_features"])


if __name__ == "__main__":
    unittest.main()
