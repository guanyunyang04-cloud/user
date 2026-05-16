import shutil
import unittest

from daily_research.continuous_policy import run_self_optimizing_study as study_runner
from daily_research.continuous_policy.research_profile_registry import (
    get_active_search_profiles,
    get_default_objective,
    get_search_profile_config,
    is_resource_gated_profile,
)


V6_PROFILE = "portfolio_decision_core_v6_longrun_candidate"
V6_STRICT_GOLD_ID = "continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1"


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

    def test_r69_r71_and_r74_v5_profiles_remain_explicit_research_only(self) -> None:
        for profile_name, loss_profile in (
            ("split_heads_portfolio_daily_value_arbitration_portfolio_set_v5_r69", "portfolio_set_v5_dfl_pg_v1_r69_value_arbitration"),
            ("split_heads_portfolio_daily_multistage_regret_portfolio_set_v5_r71", "portfolio_set_v5_dfl_pg_v1_r71_multistage_regret"),
            ("split_heads_portfolio_daily_lake_behavior_quality_portfolio_set_v5_r74", "portfolio_set_v5_dfl_pg_v1_r74_lake_behavior_quality"),
        ):
            self.assertNotIn(profile_name, get_active_search_profiles())
            config = get_search_profile_config(profile_name)

            self.assertEqual(config["base_trial"]["trainer_backend"], "formal_torch_portfolio_set_v5")
            self.assertEqual(config["base_trial"]["loss_profile"], loss_profile)
            self.assertTrue(config["resource_gate"]["portfolio_set_v5_shadow_only"])

    def test_legacy_profiles_are_not_new_study_entrypoints(self) -> None:
        parser = study_runner.build_parser()
        search_action = next(action for action in parser._actions if action.dest == "search_profile")

        self.assertEqual(tuple(search_action.choices), (*get_active_search_profiles(), V6_PROFILE))
        with self.assertRaises(KeyError):
            get_search_profile_config("split_heads_portfolio_daily_deployment_cash_exposure_closure_r52e")
        with self.assertRaises(KeyError):
            get_search_profile_config("split_heads_portfolio_daily_cash_timing_release_controller_r55")

    def test_v6_profile_is_readable_shadow_only_but_not_active(self) -> None:
        self.assertNotIn(V6_PROFILE, get_active_search_profiles())

        config = get_search_profile_config(V6_PROFILE)

        self.assertFalse(config["active_profile"])
        self.assertEqual(config["base_trial"]["trainer_backend"], "formal_torch_decision_core_v6")
        self.assertEqual(config["base_trial"]["loss_profile"], V6_PROFILE)
        self.assertEqual(config["base_trial"]["epochs"], 32)
        self.assertEqual(config["base_trial"]["min_epochs"], 32)
        self.assertTrue(config["resource_gate"]["decision_core_v6_shadow_only"])
        self.assertEqual(config["default_objective"], "end_to_end_allocation_layer_v1")
        self.assertEqual(config["resource_gate"]["cashflow_decision_contract_valid_rate_floor"], 1.0)
        self.assertEqual(config["resource_gate"]["feature_contract_blocker_rate_cap"], 0.0)
        self.assertLess(config["resource_gate"]["decision_oracle_constraint_violation_mean_cap"], 1.0e-5)

    def test_study_runner_resolves_decision_core_v6_loss_without_seq_v3_fallback(self) -> None:
        resolved_name, resolved_config = study_runner._resolve_trial_loss_profile(
            {
                "trainer_backend": "formal_torch_decision_core_v6",
                "loss_profile": V6_PROFILE,
            }
        )

        self.assertEqual(resolved_name, V6_PROFILE)
        self.assertEqual(resolved_config["trainer_backend"], "formal_torch_decision_core_v6")
        self.assertEqual(resolved_config["decision_core_version"], "v6")
        self.assertTrue(resolved_config["shadow_only"])

    def test_study_runner_decision_core_v6_requires_explicit_lake_contract(self) -> None:
        study_tag = "decision_core_v6_contract_dry_run_unit"
        study_root = study_runner.STUDIES_ROOT / study_tag
        if study_root.exists():
            shutil.rmtree(study_root)
        required = [
            "--decision-core",
            "v6",
            "--search-profile",
            V6_PROFILE,
            "--data-source",
            "lake",
            "--training-dataset-id",
            V6_STRICT_GOLD_ID,
        ]

        with self.assertRaises(SystemExit):
            study_runner.main([*required, "--dry-run"])

        self.assertEqual(
            study_runner.main(
                [
                    *required,
                    "--lake-dataset-id",
                    "policy_input_bundle__fixture",
                    "--study-tag",
                    study_tag,
                    "--dry-run",
                ]
            ),
            0,
        )
        if study_root.exists():
            shutil.rmtree(study_root)

    def test_v6_protocol_args_forward_lake_and_strict_gold_ids(self) -> None:
        args = study_runner.build_parser().parse_args(
            [
                "--decision-core",
                "v6",
                "--search-profile",
                V6_PROFILE,
                "--data-source",
                "lake",
                "--training-dataset-id",
                V6_STRICT_GOLD_ID,
                "--lake-dataset-id",
                "policy_input_bundle__fixture",
            ]
        )
        trial_config = get_search_profile_config(V6_PROFILE)["base_trial"]

        protocol_args = study_runner._build_protocol_args(args, trial_config, "v6_unit_protocol")

        self.assertEqual(protocol_args[protocol_args.index("--decision-core") + 1], "v6")
        self.assertEqual(protocol_args[protocol_args.index("--data-source") + 1], "lake")
        self.assertEqual(protocol_args[protocol_args.index("--training-dataset-id") + 1], V6_STRICT_GOLD_ID)
        self.assertEqual(protocol_args[protocol_args.index("--lake-dataset-id") + 1], "policy_input_bundle__fixture")

    def test_v6_score_and_resource_gate_include_contract_metrics(self) -> None:
        protocol_summary = {
            "trainer_backend": "formal_torch_decision_core_v6",
            "promotion_gate": {"checks": {"decision_core_v6_research_shadow_only": True}},
            "training_evidence": {"status": "sufficient"},
            "evaluation": {
                "continuous_policy_metrics": {"annual_return": 0.01, "sharpe": 0.2},
                "continuity_metrics": {
                    "decision_core_v6_mode_count": 10.0,
                    "decision_core_v6_contract_valid_rate": 1.0,
                    "cashflow_decision_contract_valid_rate": 1.0,
                    "decision_oracle_constraint_violation_mean": 2.0e-6,
                    "feature_contract_blocker_rate": 0.10,
                    "feature_contract_degraded_rate": 0.30,
                    "feature_contract_neutral_fallback_rate": 0.4,
                    "portfolio_daily_source_target_count": 2.0,
                    "portfolio_daily_source_realized_sell_rate": 0.2,
                    "portfolio_daily_source_positive_forward_sell_share": 0.2,
                    "portfolio_daily_receiver_target_count": 4.0,
                    "portfolio_daily_receiver_unrealized_deploy_share": 0.0,
                    "portfolio_daily_receiver_minus_source_forward_excess_5d": 0.01,
                    "cash_timing_quality_1d": 0.0,
                    "immediate_reversal_rate_3d": 0.01,
                },
            },
            "shadow": {
                "continuity_metrics": {
                    "portfolio_daily_receiver_minus_source_forward_excess_5d": -0.001,
                    "immediate_reversal_rate_3d": 0.01,
                }
            },
        }

        score = study_runner._score_protocol_summary(
            protocol_summary,
            objective_profile="end_to_end_allocation_layer_v1",
        )
        metrics = score["primary_metrics"]
        self.assertEqual(metrics["decision_core_v6_contract_valid_rate"], 1.0)
        self.assertEqual(metrics["feature_contract_blocker_rate"], 0.10)
        self.assertEqual(metrics["feature_contract_degraded_rate"], 0.30)
        self.assertEqual(metrics["shadow_portfolio_daily_receiver_minus_source_forward_excess_5d"], -0.001)

        trial = study_runner.TrialResult(
            trial_id=1,
            trial_tag="v6_gate_unit",
            status="completed",
            phase="screening",
            role="",
            source_trial_tag="",
            trial_config=get_search_profile_config(V6_PROFILE)["base_trial"],
            protocol_summary_path="",
            performance_score=0.0,
            stability_score=0.0,
            composite_score=0.0,
            score_breakdown={},
            primary_metrics=metrics,
            promotion_status="shadow_only",
            failed_checks=[],
            gate_pass_ratio=1.0,
            passed_check_count=1,
            total_check_count=1,
        )
        gate = study_runner._resource_gate_after_screening(
            V6_PROFILE,
            [trial],
            selected_trial_count=1,
        )

        self.assertTrue(gate["resource_gate_triggered"])
        self.assertIn("feature_contract_blocker", gate["failed_resource_checks"])
        self.assertIn("feature_contract_degraded", gate["failed_resource_checks"])
        self.assertIn("decision_oracle_constraint_violation", gate["failed_resource_checks"])
        self.assertIn("source_positive_forward_sell_high", gate["failed_resource_checks"])
        self.assertIn("immediate_reversal_high", gate["failed_resource_checks"])
        self.assertIn("shadow_receiver_source_spread_nonpositive", gate["failed_resource_checks"])

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
