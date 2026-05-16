import unittest

from daily_research.continuous_policy.run_continuous_policy_protocol import (
    _parse_args_with_profile_binding,
)


ACTIVE_V5_PROFILE = "split_heads_portfolio_daily_release_first_portfolio_set_v5_r65"
R69_RESEARCH_PROFILE = "split_heads_portfolio_daily_value_arbitration_portfolio_set_v5_r69"
R71_RESEARCH_PROFILE = "split_heads_portfolio_daily_multistage_regret_portfolio_set_v5_r71"
R74_RESEARCH_PROFILE = "split_heads_portfolio_daily_lake_behavior_quality_portfolio_set_v5_r74"


class ProtocolProfileBindingTest(unittest.TestCase):
    def test_active_v5_search_profile_applies_registry_defaults_without_identity_flags(self) -> None:
        args, binding = _parse_args_with_profile_binding(["--search-profile", ACTIVE_V5_PROFILE])

        self.assertTrue(binding["profile_applied"])
        self.assertTrue(binding["active_profile"])
        self.assertEqual(binding["requested_search_profile"], ACTIVE_V5_PROFILE)
        self.assertEqual(args.trainer_backend, "formal_torch_portfolio_set_v5")
        self.assertEqual(args.loss_profile, "portfolio_set_v5_dfl_pg_v1")
        self.assertEqual(args.budget_semantics, "allocation_layer_v1")
        self.assertEqual(args.budget_calibration, "end_to_end_allocation_layer_v1")
        self.assertEqual(args.budget_objective, "result_value_v10")
        self.assertEqual(args.alpha_prior_source, "active_execution_strategy")
        self.assertEqual(args.daily_head_layout, "split_v2")
        self.assertEqual(args.epochs, 12)
        self.assertEqual(args.min_epochs, 8)
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(binding["effective_base_trial"]["trainer_backend"], "formal_torch_portfolio_set_v5")

    def test_explicit_cli_values_override_profile_defaults_and_are_recorded(self) -> None:
        args, binding = _parse_args_with_profile_binding(
            [
                "--search-profile",
                ACTIVE_V5_PROFILE,
                "--epochs",
                "1",
                "--min-epochs",
                "1",
                "--batch-size",
                "1",
            ]
        )

        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.min_epochs, 1)
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(binding["explicit_overrides"]["epochs"], 1)
        self.assertEqual(binding["explicit_overrides"]["min_epochs"], 1)
        self.assertEqual(binding["explicit_overrides"]["batch_size"], 1)
        self.assertEqual(binding["effective_base_trial"]["epochs"], 1)
        self.assertEqual(binding["effective_base_trial"]["min_epochs"], 1)
        self.assertEqual(binding["effective_base_trial"]["batch_size"], 1)

    def test_non_active_legacy_profile_is_rejected_for_direct_protocol(self) -> None:
        for profile in (
            "split_heads_portfolio_daily_deployment_cash_exposure_closure_r52e",
            "split_heads_portfolio_daily_cash_timing_release_controller_r55",
        ):
            with self.subTest(profile=profile):
                with self.assertRaises(SystemExit):
                    _parse_args_with_profile_binding(["--search-profile", profile])

    def test_registered_r69_r71_r74_research_profiles_are_rejected_for_new_protocol_runs(self) -> None:
        for profile in (R69_RESEARCH_PROFILE, R71_RESEARCH_PROFILE, R74_RESEARCH_PROFILE):
            with self.subTest(profile=profile):
                with self.assertRaises(SystemExit):
                    _parse_args_with_profile_binding(["--search-profile", profile])

    def test_legacy_rxx_loss_profiles_are_rejected_for_new_protocol_runs(self) -> None:
        for loss_profile in (
            "portfolio_set_v5_dfl_pg_v1_r69_value_arbitration",
            "portfolio_set_v5_dfl_pg_v1_r71_multistage_regret",
            "portfolio_set_v5_dfl_pg_v1_r74_lake_behavior_quality",
        ):
            with self.subTest(loss_profile=loss_profile):
                with self.assertRaises(SystemExit):
                    _parse_args_with_profile_binding(["--loss-profile", loss_profile])

    def test_decision_core_v6_protocol_requires_fixed_lake_contract(self) -> None:
        args, binding = _parse_args_with_profile_binding(
            [
                "--decision-core",
                "v6",
                "--data-source",
                "lake",
                "--training-dataset-id",
                "continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1",
                "--lake-dataset-id",
                "policy_input_bundle__fixture",
            ]
        )

        self.assertFalse(binding["profile_applied"])
        self.assertEqual(args.trainer_backend, "formal_torch_decision_core_v6")
        self.assertEqual(args.loss_profile, "portfolio_decision_core_v6_longrun_candidate")
        self.assertEqual(args.decision_core, "v6")

    def test_protocol_parser_accepts_lake_evaluator_arguments(self) -> None:
        args, binding = _parse_args_with_profile_binding(
            [
                "--data-source",
                "lake",
                "--lake-dataset-id",
                "policy_input_bundle__fixture",
                "--data-lake-root",
                "H:/lake",
            ]
        )

        self.assertFalse(binding["profile_applied"])
        self.assertEqual(args.data_source, "lake")
        self.assertEqual(args.lake_dataset_id, "policy_input_bundle__fixture")
        self.assertEqual(args.data_lake_root, "H:/lake")


if __name__ == "__main__":
    unittest.main()
