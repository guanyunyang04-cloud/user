import unittest

from daily_research.continuous_policy.analyze_behavior_gap import _resolve_audit_prepare_kwargs


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


if __name__ == "__main__":
    unittest.main()
