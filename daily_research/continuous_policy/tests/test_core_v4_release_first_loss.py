import unittest

from daily_research.continuous_policy.model_core_v4 import (
    CORE_V4_RELEASE_FIRST_LOSS_ALIASES,
    resolve_core_v4_loss_profile,
)


class CoreV4ReleaseFirstLossTest(unittest.TestCase):
    def test_v46_and_alias_resolve_to_release_first_loss(self) -> None:
        for profile_name in ("alpha_result_value_budget_split_v46", "core_v4_release_first_v1"):
            with self.subTest(profile_name=profile_name):
                resolved_name, config = resolve_core_v4_loss_profile(profile_name)

                self.assertIn(resolved_name, CORE_V4_RELEASE_FIRST_LOSS_ALIASES)
                weights = config["multi_objective_loss_weights"]
                self.assertEqual(weights["action_total"], 0.0)
                self.assertEqual(weights["duration_total"], 0.0)
                self.assertGreater(weights["target_weight_closure_total"], 0.0)
                self.assertGreater(weights["cash_timing_directional_total"], 0.0)
                self.assertGreater(weights["source_release_intent_total"], 0.0)
                self.assertGreater(weights["reduce_exit_intent_total"], 0.0)
                self.assertGreater(weights["release_first_allocation_total"], 0.0)

    def test_legacy_loss_profiles_are_not_core_v4_training_entrypoints(self) -> None:
        for profile_name in ("alpha_result_value_budget_split_v45", "default_v1"):
            with self.subTest(profile_name=profile_name):
                with self.assertRaises(ValueError):
                    resolve_core_v4_loss_profile(profile_name)


if __name__ == "__main__":
    unittest.main()
