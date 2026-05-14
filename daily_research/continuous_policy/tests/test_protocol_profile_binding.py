import unittest

from daily_research.continuous_policy.run_continuous_policy_protocol import (
    _parse_args_with_profile_binding,
)


R61_PROFILE = "split_heads_portfolio_daily_release_first_decision_focused_core_v4_r61"


class ProtocolProfileBindingTest(unittest.TestCase):
    def test_r61_search_profile_applies_registry_defaults_without_identity_flags(self) -> None:
        args, binding = _parse_args_with_profile_binding(["--search-profile", R61_PROFILE])

        self.assertTrue(binding["profile_applied"])
        self.assertTrue(binding["active_profile"])
        self.assertEqual(binding["requested_search_profile"], R61_PROFILE)
        self.assertEqual(args.trainer_backend, "formal_torch_core_v4")
        self.assertEqual(args.loss_profile, "alpha_result_value_budget_split_v47")
        self.assertEqual(args.budget_semantics, "allocation_layer_v1")
        self.assertEqual(args.budget_calibration, "end_to_end_allocation_layer_v1")
        self.assertEqual(args.budget_objective, "result_value_v10")
        self.assertEqual(args.alpha_prior_source, "active_execution_strategy")
        self.assertEqual(args.daily_head_layout, "split_v2")
        self.assertEqual(args.epochs, 12)
        self.assertEqual(args.min_epochs, 8)
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(binding["effective_base_trial"]["trainer_backend"], "formal_torch_core_v4")

    def test_explicit_cli_values_override_profile_defaults_and_are_recorded(self) -> None:
        args, binding = _parse_args_with_profile_binding(
            [
                "--search-profile",
                R61_PROFILE,
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
        with self.assertRaises(SystemExit):
            _parse_args_with_profile_binding(
                ["--search-profile", "split_heads_portfolio_daily_release_first_core_v4_r59"]
            )


if __name__ == "__main__":
    unittest.main()
