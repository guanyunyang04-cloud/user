import unittest

from daily_research.continuous_policy.analyze_behavior_gap import (
    _release_translation_deploy_health,
    _resolve_audit_prepare_kwargs,
)


class BehaviorAuditPrepareContractTest(unittest.TestCase):
    def test_behavior_audit_preserves_evaluation_max_universe_size(self) -> None:
        kwargs = _resolve_audit_prepare_kwargs(
            {
                "pool_name": "learned_all_a",
                "start_date": "20260101",
                "end_date": "20260110",
                "benchmark": "000300.SH",
                "data_source": "csv",
                "csv_folder": "H:/fixture",
                "max_universe_size": 80,
                "alpha_prior_source": "panel",
                "alpha_prior_score_panel": "H:/score.csv",
                "alpha_prior_target_weight_panel": "H:/weight.csv",
                "prepared_summary": {"universe_size": 3071},
            }
        )

        self.assertEqual(kwargs["pool_name"], "learned_all_a")
        self.assertEqual(kwargs["data_source"], "csv")
        self.assertEqual(kwargs["csv_folder"], "H:/fixture")
        self.assertEqual(kwargs["max_universe_size"], 80)
        self.assertEqual(kwargs["alpha_prior_source"], "panel")
        self.assertEqual(kwargs["alpha_prior_score_panel"], "H:/score.csv")
        self.assertEqual(kwargs["alpha_prior_target_weight_panel"], "H:/weight.csv")

    def test_behavior_audit_caps_legacy_summary_to_prepared_universe(self) -> None:
        kwargs = _resolve_audit_prepare_kwargs(
            {
                "pool_name": "learned_all_a",
                "start_date": "20260101",
                "end_date": "20260110",
                "benchmark": "000300.SH",
                "prepared_summary": {
                    "universe_size": 80,
                    "data_source": "csv",
                    "csv_folder": "H:/legacy_fixture",
                    "alpha_prior_summary": {
                        "source": "panel",
                        "score_panel_csv": "H:/legacy_score.csv",
                        "target_weight_panel_csv": "H:/legacy_weight.csv",
                    },
                },
            }
        )

        self.assertEqual(kwargs["max_universe_size"], 80)
        self.assertEqual(kwargs["data_source"], "csv")
        self.assertEqual(kwargs["csv_folder"], "H:/legacy_fixture")
        self.assertEqual(kwargs["alpha_prior_source"], "panel")
        self.assertEqual(kwargs["alpha_prior_score_panel"], "H:/legacy_score.csv")
        self.assertEqual(kwargs["alpha_prior_target_weight_panel"], "H:/legacy_weight.csv")

    def test_behavior_audit_preserves_lake_evaluation_source(self) -> None:
        kwargs = _resolve_audit_prepare_kwargs(
            {
                "pool_name": "learned_all_a",
                "start_date": "20190401",
                "end_date": "20190430",
                "benchmark": "000300.SH",
                "data_source": "lake",
                "lake_dataset_id": "policy_input_bundle__fixture",
                "data_lake_root": "H:/lake",
                "prepared_summary": {"universe_size": 3},
            }
        )

        self.assertEqual(kwargs["data_source"], "lake")
        self.assertEqual(kwargs["lake_dataset_id"], "policy_input_bundle__fixture")
        self.assertEqual(kwargs["data_lake_root"], "H:/lake")

    def test_release_translation_health_counts_cashflow_source_contract(self) -> None:
        health = _release_translation_deploy_health(
            deploy_intent_action_count=20.0,
            deploy_intent_realized_rate=1.0,
            order_translation_conflict_rate=0.0,
            add_to_hold_conflict_share=0.0,
            sell_intent_suppressed_share=0.0,
            budget_origin_sell_share=0.0,
            deploy_funding_rebalance_sell_count=0.0,
            deploy_funding_rebalance_sell_share=0.0,
            deploy_funding_rebalance_forward_excess_5d=0.0,
            deploy_funding_against_protected_hold_share=0.0,
            deploy_funding_release_consistent_share=0.0,
            model_release_signal_sell_count=0.0,
            model_release_signal_forward_excess_5d=0.0,
            model_release_against_protected_hold_share=0.0,
            model_release_release_consistent_share=0.0,
            cashflow_source_target_count=8.0,
            cashflow_source_release_consistent_share=0.70,
            cashflow_source_forward_excess_5d=-0.02,
            cashflow_source_positive_forward_sell_share=0.10,
            cashflow_source_realized_sell_rate=1.0,
        )

        self.assertNotEqual(
            health["release_translation_deploy_failure_mode"],
            "funding_release_not_observed",
        )
        self.assertEqual(
            health["release_translation_deploy_components"]["cashflow_source_observed"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
