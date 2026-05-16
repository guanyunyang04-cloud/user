import inspect
import unittest

from daily_research.continuous_policy import train_policy
from daily_research.continuous_policy.model_portfolio_set_v5 import (
    PORTFOLIO_SET_V5_BEHAVIOR_MODE_DFL_PG_V1,
    PORTFOLIO_SET_V5_BEHAVIOR_MODE_R69_VALUE_ARBITRATION,
    PORTFOLIO_SET_V5_BEHAVIOR_MODE_R74_LAKE_BEHAVIOR_QUALITY,
    PORTFOLIO_SET_V5_INTERNAL_VERSION,
    PORTFOLIO_SET_V5_R69_INTERNAL_VERSION,
    PORTFOLIO_SET_V5_R74_INTERNAL_VERSION,
    resolve_portfolio_set_v5_loss_profile,
)
from daily_research.continuous_policy.training_contracts import (
    TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5,
    TRAINER_BACKENDS,
    build_training_contract,
    normalize_trainer_backend,
)


class PortfolioSetV5TrainingContractTest(unittest.TestCase):
    def test_formal_torch_portfolio_set_v5_backend_is_shadow_only(self) -> None:
        self.assertIn(TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5, TRAINER_BACKENDS)
        self.assertEqual(normalize_trainer_backend("portfolio_set_v5"), TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5)
        self.assertEqual(normalize_trainer_backend("formal_torch_portfolio_set_v5"), TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5)

        contract = build_training_contract(
            trainer_backend=TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5,
            requested_epochs=1,
            min_epochs=1,
        )

        self.assertEqual(contract["trainer_backend"], TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5)
        self.assertTrue(contract["epoch_based"])
        self.assertTrue(contract["resume_capable"])
        self.assertTrue(contract["gpu_required"])
        self.assertFalse(contract["promotable"])

    def test_train_policy_routes_to_portfolio_set_v5_not_core_v4(self) -> None:
        source = inspect.getsource(train_policy.main)

        self.assertIn("fit_policy_models_portfolio_set_v5", source)
        self.assertIn("PORTFOLIO_SET_V5_ARTIFACT_FILENAME", source)
        self.assertIn("TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5", source)
        self.assertIn("PORTFOLIO_SET_V5_DEFAULT_STRICT_GOLD_DATASET_ID", inspect.getsource(train_policy))
        self.assertIn("_apply_backend_default_loss", source)
        self.assertIn("model_dim=max(int(args.hidden_dim), 16)", source)
        self.assertIn("latent_count=max(4, int(args.daily_hidden_dim))", source)

    def test_portfolio_set_v5_backend_defaults_to_dfl_pg_loss_when_not_explicit(self) -> None:
        args = train_policy.build_parser().parse_args(["--trainer-backend", "formal_torch_portfolio_set_v5"])

        backend = train_policy._apply_backend_default_loss(args, ["--trainer-backend", "formal_torch_portfolio_set_v5"])

        self.assertEqual(backend, TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5)
        self.assertEqual(args.loss_profile, "portfolio_set_v5_dfl_pg_v1")

    def test_portfolio_set_v5_backend_preserves_explicit_legacy_loss_alias(self) -> None:
        argv = [
            "--trainer-backend",
            "formal_torch_portfolio_set_v5",
            "--loss-profile",
            "alpha_result_value_budget_split_v48",
        ]
        args = train_policy.build_parser().parse_args(argv)

        backend = train_policy._apply_backend_default_loss(args, argv)

        self.assertEqual(backend, TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5)
        self.assertEqual(args.loss_profile, "alpha_result_value_budget_split_v48")

    def test_portfolio_set_v5_backend_rejects_historical_rxx_profiles_for_new_training(self) -> None:
        argv = [
            "--trainer-backend",
            "formal_torch_portfolio_set_v5",
            "--loss-profile",
            "portfolio_set_v5_dfl_pg_v1_r74_lake_behavior_quality",
        ]
        args = train_policy.build_parser().parse_args(argv)

        with self.assertRaises(ValueError):
            train_policy._apply_backend_default_loss(args, argv)

    def test_dfl_pg_loss_profile_is_recorded_as_internal_version(self) -> None:
        resolved, config = resolve_portfolio_set_v5_loss_profile("portfolio_set_v5_dfl_pg_v1")

        self.assertEqual(resolved, "portfolio_set_v5_dfl_pg_v1")
        self.assertEqual(PORTFOLIO_SET_V5_INTERNAL_VERSION, "portfolio_set_v5_dfl_pg_v1")
        self.assertEqual(config["portfolio_set_v5_behavior_mode"], PORTFOLIO_SET_V5_BEHAVIOR_MODE_DFL_PG_V1)
        self.assertGreater(config["multi_objective_loss_weights"]["pg_dfl_surrogate_total"], 0.0)
        self.assertGreater(config["multi_objective_loss_weights"]["decision_oracle_total"], 0.0)
        self.assertEqual(config["multi_objective_loss_weights"]["value_arbitration_total"], 0.0)

    def test_legacy_loss_alias_resolves_to_base_internal_version(self) -> None:
        resolved, config = resolve_portfolio_set_v5_loss_profile("alpha_result_value_budget_split_v48")

        self.assertEqual(resolved, "portfolio_set_v5_dfl_pg_v1")
        self.assertEqual(config["portfolio_set_v5_behavior_mode"], PORTFOLIO_SET_V5_BEHAVIOR_MODE_DFL_PG_V1)
        self.assertEqual(config["multi_objective_loss_weights"]["value_arbitration_total"], 0.0)

    def test_r69_loss_profile_is_explicit_value_arbitration_mode(self) -> None:
        resolved, config = resolve_portfolio_set_v5_loss_profile(PORTFOLIO_SET_V5_R69_INTERNAL_VERSION)

        self.assertEqual(resolved, PORTFOLIO_SET_V5_R69_INTERNAL_VERSION)
        self.assertEqual(config["portfolio_set_v5_behavior_mode"], PORTFOLIO_SET_V5_BEHAVIOR_MODE_R69_VALUE_ARBITRATION)
        self.assertGreater(config["multi_objective_loss_weights"]["value_arbitration_total"], 0.0)

    def test_r74_loss_profile_is_explicit_lake_behavior_quality_mode(self) -> None:
        resolved, config = resolve_portfolio_set_v5_loss_profile(PORTFOLIO_SET_V5_R74_INTERNAL_VERSION)

        self.assertEqual(resolved, PORTFOLIO_SET_V5_R74_INTERNAL_VERSION)
        self.assertEqual(config["portfolio_set_v5_behavior_mode"], PORTFOLIO_SET_V5_BEHAVIOR_MODE_R74_LAKE_BEHAVIOR_QUALITY)
        self.assertGreater(config["multi_objective_loss_weights"]["value_arbitration_total"], 0.0)
        self.assertGreater(config["multi_objective_loss_weights"]["multistage_regret_total"], 0.0)
        self.assertGreater(config["multi_objective_loss_weights"]["lake_behavior_quality_total"], 0.0)


if __name__ == "__main__":
    unittest.main()
