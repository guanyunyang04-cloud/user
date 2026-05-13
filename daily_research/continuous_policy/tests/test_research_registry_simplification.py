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
                "split_heads_portfolio_daily_cash_funded_allocation_core_r53",
                "split_heads_portfolio_daily_semantic_budget_controller_r54",
                "split_heads_portfolio_daily_cash_timing_release_controller_r55",
                "split_heads_portfolio_daily_release_first_constrained_decoder_r56",
            ),
        )
        self.assertLessEqual(len(active_profiles), 5)
        self.assertNotIn("split_heads_portfolio_daily_end_to_end_allocation_layer_r40", active_profiles)
        self.assertNotIn("split_heads_portfolio_daily_integrated_convex_capital_flow_r50", active_profiles)
        self.assertNotIn("split_heads_portfolio_daily_deployment_cash_exposure_closure_r52e", active_profiles)

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

    def test_legacy_profiles_are_not_new_study_entrypoints(self) -> None:
        parser = study_runner.build_parser()
        search_action = next(action for action in parser._actions if action.dest == "search_profile")

        self.assertEqual(tuple(search_action.choices), get_active_search_profiles())
        with self.assertRaises(KeyError):
            get_search_profile_config("split_heads_portfolio_daily_deployment_cash_exposure_closure_r52e")

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
