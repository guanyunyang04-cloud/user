import unittest

from daily_research.continuous_policy import run_self_optimizing_study as study_runner
from daily_research.continuous_policy.research_profile_registry import (
    get_active_search_profiles,
    get_default_objective,
    get_search_profile_config,
    is_resource_gated_profile,
)


class ResearchRegistrySimplificationTest(unittest.TestCase):
    def test_active_registry_is_small_and_current(self) -> None:
        active_profiles = get_active_search_profiles()

        self.assertEqual(
            active_profiles,
            (
                "focused_seq_v1",
                "split_heads_portfolio_daily_release_first_constrained_decoder_r56",
                "split_heads_portfolio_daily_release_first_portfolio_set_v5_r65",
            ),
        )
        self.assertLessEqual(len(active_profiles), 3)
        self.assertNotIn("split_heads_portfolio_daily_end_to_end_allocation_layer_r40", active_profiles)
        self.assertNotIn("split_heads_portfolio_daily_integrated_convex_capital_flow_r50", active_profiles)
        self.assertNotIn("split_heads_portfolio_daily_deployment_cash_exposure_closure_r52e", active_profiles)
        self.assertNotIn("split_heads_portfolio_daily_cash_timing_release_controller_r55", active_profiles)

    def test_active_profile_config_is_copy_safe_and_complete(self) -> None:
        profile_name = "split_heads_portfolio_daily_release_first_constrained_decoder_r56"

        config = get_search_profile_config(profile_name)
        config["base_trial"]["loss_profile"] = "mutated"
        fresh_config = get_search_profile_config(profile_name)

        self.assertEqual(fresh_config["search_space"]["loss_profile"], ["alpha_result_value_budget_split_v46"])
        self.assertEqual(fresh_config["base_trial"]["loss_profile"], "alpha_result_value_budget_split_v46")
        self.assertEqual(get_default_objective(profile_name), "end_to_end_allocation_layer_v1")
        self.assertTrue(is_resource_gated_profile(profile_name))
        self.assertEqual(fresh_config["resource_gate"]["release_first_source_intent_floor"], 1.0)

    def test_r65_portfolio_set_v5_profile_is_active_and_shadow_only(self) -> None:
        profile_name = "split_heads_portfolio_daily_release_first_portfolio_set_v5_r65"

        config = get_search_profile_config(profile_name)

        self.assertEqual(config["base_trial"]["trainer_backend"], "formal_torch_portfolio_set_v5")
        self.assertEqual(config["search_space"]["loss_profile"], ["portfolio_set_v5_dfl_pg_v1"])
        self.assertEqual(config["base_trial"]["loss_profile"], "portfolio_set_v5_dfl_pg_v1")
        self.assertEqual(config["base_trial"]["epochs"], 12)
        self.assertEqual(config["base_trial"]["min_epochs"], 8)
        self.assertEqual(get_default_objective(profile_name), "end_to_end_allocation_layer_v1")
        self.assertTrue(is_resource_gated_profile(profile_name))

    def test_study_runner_resolves_portfolio_set_v5_loss_with_v5_resolver(self) -> None:
        resolved_name, resolved_config = study_runner._resolve_trial_loss_profile(
            {
                "trainer_backend": "formal_torch_portfolio_set_v5",
                "loss_profile": "portfolio_set_v5_dfl_pg_v1",
            }
        )

        self.assertEqual(resolved_name, "portfolio_set_v5_dfl_pg_v1")
        self.assertGreater(
            resolved_config["multi_objective_loss_weights"]["release_flow_balance_total"],
            0.0,
        )

    def test_r59_and_r61_core_v4_profiles_remain_readable_but_not_active(self) -> None:
        for profile_name, loss_profile in (
            ("split_heads_portfolio_daily_release_first_core_v4_r59", "alpha_result_value_budget_split_v46"),
            ("split_heads_portfolio_daily_release_first_decision_focused_core_v4_r61", "alpha_result_value_budget_split_v47"),
        ):
            self.assertNotIn(profile_name, get_active_search_profiles())
            config = get_search_profile_config(profile_name)

            self.assertEqual(config["base_trial"]["trainer_backend"], "formal_torch_core_v4")
            self.assertEqual(config["base_trial"]["loss_profile"], loss_profile)

    def test_r69_and_r71_v5_profiles_remain_explicit_research_only(self) -> None:
        for profile_name, loss_profile in (
            ("split_heads_portfolio_daily_value_arbitration_portfolio_set_v5_r69", "portfolio_set_v5_dfl_pg_v1_r69_value_arbitration"),
            ("split_heads_portfolio_daily_multistage_regret_portfolio_set_v5_r71", "portfolio_set_v5_dfl_pg_v1_r71_multistage_regret"),
        ):
            self.assertNotIn(profile_name, get_active_search_profiles())
            config = get_search_profile_config(profile_name)

            self.assertEqual(config["base_trial"]["trainer_backend"], "formal_torch_portfolio_set_v5")
            self.assertEqual(config["base_trial"]["loss_profile"], loss_profile)
            self.assertTrue(config["resource_gate"]["portfolio_set_v5_shadow_only"])

    def test_legacy_profiles_are_not_new_study_entrypoints(self) -> None:
        parser = study_runner.build_parser()
        search_action = next(action for action in parser._actions if action.dest == "search_profile")

        self.assertEqual(tuple(search_action.choices), get_active_search_profiles())
        with self.assertRaises(KeyError):
            get_search_profile_config("split_heads_portfolio_daily_deployment_cash_exposure_closure_r52e")
        with self.assertRaises(KeyError):
            get_search_profile_config("split_heads_portfolio_daily_cash_timing_release_controller_r55")

    def test_study_runner_no_longer_exposes_subprocess_protocol_options(self) -> None:
        parser = study_runner.build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }

        self.assertNotIn("--protocol-runner", option_strings)
        self.assertNotIn("--protocol-max-wall-seconds", option_strings)
        self.assertNotIn("--protocol-max-stale-seconds", option_strings)
        self.assertNotIn("--protocol-progress-grace-seconds", option_strings)


if __name__ == "__main__":
    unittest.main()
