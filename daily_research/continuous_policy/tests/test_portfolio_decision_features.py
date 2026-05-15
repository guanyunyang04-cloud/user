import unittest
from types import SimpleNamespace

import pandas as pd

from daily_research.continuous_policy.portfolio_decision_features import (
    PORTFOLIO_DECISION_FEATURE_BUNDLE_MODE_COLUMN,
    PORTFOLIO_DECISION_FEATURE_CONTRACT_BLOCKER_COLUMN,
    PORTFOLIO_DECISION_FEATURE_CONTRACT_SEVERITY_COLUMN,
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
                "adv20": [1000.0, 5000.0, 2500.0],
                "amount": [1.0e6, 5.0e6, 2.5e6],
                "volume": [100.0, 500.0, 250.0],
                "z_drawdown_20": [1.2, 0.0, 0.4],
                "z_volatility_20": [0.7, 0.1, 0.3],
                "z_vol_ratio_5_20": [0.6, 0.0, 0.2],
                "z_price_volume_divergence": [0.2, 0.0, 0.1],
                "z_breakout_volume": [0.0, 1.5, 0.2],
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
        self.assertEqual(str(enriched[PORTFOLIO_DECISION_FEATURE_CONTRACT_SEVERITY_COLUMN].iloc[0]), "degraded")
        self.assertIn("portfolio_decision_r74_cash_defense_value", enriched.columns)
        self.assertIn("portfolio_decision_r74_receiver_source_spread_value", enriched.columns)
        self.assertGreater(float(enriched.loc[1, "portfolio_decision_r74_deploy_value"]), float(enriched.loc[0, "portfolio_decision_r74_deploy_value"]))

    def test_feature_contract_severity_blocks_missing_score_or_membership(self) -> None:
        missing_score = pd.DataFrame(
            {
                "stock": ["AAA"],
                "in_pool": [1.0],
                "ret_3d": [0.01],
                "ret_5d": [0.02],
                "vol_20d": [0.03],
            }
        )
        score_blocked = attach_portfolio_decision_features(missing_score, mode="predict")
        self.assertEqual(str(score_blocked[PORTFOLIO_DECISION_FEATURE_CONTRACT_SEVERITY_COLUMN].iloc[0]), "blocker")
        self.assertIn("missing_score_signal", str(score_blocked[PORTFOLIO_DECISION_FEATURE_CONTRACT_BLOCKER_COLUMN].iloc[0]))

        missing_membership = pd.DataFrame(
            {
                "stock": ["AAA"],
                "score_blend": [0.4],
                "ret_3d": [0.01],
                "ret_5d": [0.02],
                "vol_20d": [0.03],
            }
        )
        membership_blocked = attach_portfolio_decision_features(missing_membership, mode="predict")
        self.assertEqual(str(membership_blocked[PORTFOLIO_DECISION_FEATURE_CONTRACT_SEVERITY_COLUMN].iloc[0]), "blocker")
        self.assertIn("missing_membership_signal", str(membership_blocked[PORTFOLIO_DECISION_FEATURE_CONTRACT_BLOCKER_COLUMN].iloc[0]))

    def test_feature_contract_complete_input_is_ok(self) -> None:
        state = pd.DataFrame(
            {
                "stock": ["AAA", "BBB"],
                "score_blend": [0.1, 0.8],
                "score_rank_pct": [0.4, 0.9],
                "ret_1d": [0.0, 0.01],
                "ret_3d": [0.01, 0.02],
                "ret_5d": [0.02, 0.03],
                "vol_20d": [0.03, 0.02],
                "in_pool": [1.0, 1.0],
                "current_weight": [0.0, 0.0],
                "close": [10.0, 20.0],
                "open": [9.9, 19.8],
                "high": [10.2, 20.2],
                "low": [9.7, 19.6],
                "adv20": [1000.0, 2000.0],
                "amount": [1.0e6, 2.0e6],
                "volume": [100.0, 200.0],
                "z_drawdown_20": [0.0, 0.1],
                "z_volatility_20": [0.1, 0.2],
                "z_vol_ratio_5_20": [0.0, 0.1],
                "z_price_volume_divergence": [0.0, 0.1],
                "z_breakout_volume": [0.2, 0.6],
                "z_volatility_contraction": [0.0, 0.1],
                "z_volume_contraction": [0.0, 0.1],
            }
        )

        enriched = attach_portfolio_decision_features(state, mode="predict")

        self.assertEqual(str(enriched[PORTFOLIO_DECISION_FEATURE_CONTRACT_SEVERITY_COLUMN].iloc[0]), "ok")
        self.assertEqual(str(enriched[PORTFOLIO_DECISION_FEATURE_CONTRACT_BLOCKER_COLUMN].iloc[0]), "none")

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
        self.assertGreaterEqual(int(report["decision_feature_high_value_used_count"]), 1)
        self.assertGreaterEqual(int(report["decision_feature_high_value_unused_count"]), 1)
        self.assertIn("decision_feature_degraded_count", report)
        self.assertIn("decision_feature_blocker_count", report)


if __name__ == "__main__":
    unittest.main()
