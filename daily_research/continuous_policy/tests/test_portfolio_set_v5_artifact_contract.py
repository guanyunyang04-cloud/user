import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from daily_research.continuous_policy.model import load_artifact, predict_policy
from daily_research.continuous_policy.model_portfolio_set_v5 import (
    PORTFOLIO_SET_V5_ARTIFACT_FILENAME,
    PORTFOLIO_SET_V5_R69_INTERNAL_VERSION,
    TorchPortfolioSetV5Artifact,
    load_torch_portfolio_set_v5_artifact,
    predict_policy_portfolio_set_v5,
)
from daily_research.continuous_policy.training_contracts import TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5


class PortfolioSetV5ArtifactContractTest(unittest.TestCase):
    def _artifact(self) -> TorchPortfolioSetV5Artifact:
        return TorchPortfolioSetV5Artifact(
            feature_names=["alpha_score", "current_weight", "portfolio_daily_target_delta_intent"],
            daily_feature_names=["market_downside_pressure"],
            static_feature_names=["alpha_score", "current_weight", "portfolio_daily_target_delta_intent"],
            sequence_bases=[],
            feature_fill_values=np.zeros(3, dtype=np.float32),
            feature_means=np.zeros(3, dtype=np.float32),
            feature_stds=np.ones(3, dtype=np.float32),
            sequence_fill_values=np.zeros(1, dtype=np.float32),
            sequence_means=np.zeros(1, dtype=np.float32),
            sequence_stds=np.ones(1, dtype=np.float32),
            daily_fill_values=np.zeros(1, dtype=np.float32),
            daily_means=np.zeros(1, dtype=np.float32),
            daily_stds=np.ones(1, dtype=np.float32),
            train_summary={"run_tag": "portfolio_set_v5_test"},
            training_diagnostics={
                "trainer_backend": TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5,
                "loss_profile": "alpha_result_value_budget_split_v48",
                "portfolio_set_v5_uses_latent_attention": True,
                "portfolio_set_v5_full_self_attention": False,
            },
            training_contract={"trainer_backend": TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5, "promotable": False},
            trained_at="2026-05-14T00:00:00+08:00",
            model_config={
                "model_dim": 16,
                "temporal_layers": 1,
                "cross_layers": 1,
                "latent_count": 4,
                "dropout": 0.0,
                "portfolio_set_v5_internal_version": PORTFOLIO_SET_V5_R69_INTERNAL_VERSION,
            },
            global_target_defaults={"gross_exposure_target": 0.7},
        )

    def _biased_artifact(self) -> TorchPortfolioSetV5Artifact:
        artifact = self._artifact()
        from daily_research.continuous_policy.model_portfolio_set_v5 import _make_model

        state = {key: value.detach().clone() for key, value in _make_model(artifact).state_dict().items()}
        state["head.weight"] = torch.zeros_like(state["head.weight"])
        state["head.bias"] = torch.tensor(
            [0.0, 0.0, 0.0, 2.4, -3.0, 0.0, 0.0, 0.0],
            dtype=state["head.bias"].dtype,
        )
        artifact.model_state_dict = state
        artifact.global_target_defaults = {"gross_exposure_target": 0.7, "turnover_budget": 0.08}
        return artifact

    def test_portfolio_set_v5_artifact_round_trips_through_generic_loader(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / PORTFOLIO_SET_V5_ARTIFACT_FILENAME
            self._artifact().save(path)

            loaded_direct = load_torch_portfolio_set_v5_artifact(path)
            loaded_generic = load_artifact(path)

        self.assertIsInstance(loaded_direct, TorchPortfolioSetV5Artifact)
        self.assertIsInstance(loaded_generic, TorchPortfolioSetV5Artifact)
        self.assertEqual(loaded_direct.training_diagnostics["trainer_backend"], TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5)
        self.assertEqual(loaded_direct.model_config["portfolio_set_v5_internal_version"], PORTFOLIO_SET_V5_R69_INTERNAL_VERSION)

    def test_predict_policy_portfolio_set_v5_outputs_release_first_semantics(self) -> None:
        artifact = self._artifact()
        state_frame = pd.DataFrame(
            {
                "stock": ["SRC", "RCV"],
                "alpha_score": [0.1, 0.9],
                "current_weight": [0.12, 0.0],
                "portfolio_daily_target_delta_intent": [-0.05, 0.06],
            }
        )

        policy, global_targets = predict_policy_portfolio_set_v5(
            artifact,
            state_frame=state_frame,
            daily_features={"market_downside_pressure": 0.1},
        )
        generic_policy, generic_globals = predict_policy(
            artifact,
            state_frame=state_frame,
            daily_features={"market_downside_pressure": 0.1},
        )

        for frame in (policy, generic_policy):
            self.assertIn("portfolio_daily_target_weight_intent", frame.columns)
            self.assertIn("portfolio_daily_target_delta_intent", frame.columns)
            self.assertIn("portfolio_daily_release_first_intent", frame.columns)
            self.assertIn("portfolio_cashflow_decision_v1_mode", frame.columns)
            self.assertIn("portfolio_daily_source_target_intent", frame.columns)
            self.assertIn("portfolio_daily_receiver_target_intent", frame.columns)
            self.assertIn("portfolio_daily_receiver_score", frame.columns)
            self.assertIn("portfolio_daily_source_score", frame.columns)
            self.assertIn("portfolio_daily_receiver_executable_candidate", frame.columns)
            self.assertIn("portfolio_daily_source_release_quality", frame.columns)
            self.assertIn("portfolio_daily_receiver_add_headroom", frame.columns)
            self.assertIn("portfolio_set_v5_cash_buffer_score", frame.columns)
            self.assertIn("portfolio_set_v5_r69_deploy_value", frame.columns)
            self.assertIn("portfolio_set_v5_r69_release_value", frame.columns)
            self.assertIn("portfolio_set_v5_r69_defense_value", frame.columns)
            self.assertIn("portfolio_set_v5_r69_cash_timing_value", frame.columns)
            self.assertIn("portfolio_set_v5_r69_source_opportunity_cost", frame.columns)
            self.assertIn("portfolio_set_v5_r69_receiver_source_spread_value", frame.columns)
            self.assertIn("portfolio_set_v5_r69_reversal_risk_penalty", frame.columns)
            self.assertIn("portfolio_set_v5_r69_source_wrong_side_sell_penalty", frame.columns)
            self.assertIn("portfolio_set_v5_r69_reversal_guarded", frame.columns)
            self.assertIn("portfolio_set_v5_oracle_constraint_violation", frame.columns)
            self.assertIn("portfolio_set_v5_oracle_feasible", frame.columns)
            self.assertIn("release_first_action_hint", frame.columns)
            self.assertAlmostEqual(
                float(frame.loc[0, "portfolio_daily_target_delta_intent"]),
                float(frame.loc[0, "portfolio_daily_target_weight_intent"]) - 0.12,
                places=8,
            )
            self.assertEqual(float(frame["portfolio_set_v5_target_delta_weight_conflict_count"].iloc[0]), 0.0)
            self.assertEqual(float(frame["portfolio_cashflow_decision_v1_mode"].iloc[0]), 1.0)
        self.assertEqual(global_targets["release_first_allocation_v3_mode"], 1.0)
        self.assertEqual(generic_globals["release_first_allocation_v3_mode"], 1.0)
        self.assertEqual(global_targets["portfolio_cashflow_decision_v1_mode"], 1.0)
        self.assertEqual(generic_globals["portfolio_cashflow_decision_v1_mode"], 1.0)

    def test_predict_policy_portfolio_set_v5_opens_receiver_from_cash_in_cashflow_mode(self) -> None:
        artifact = self._biased_artifact()
        state_frame = pd.DataFrame(
            {
                "stock": ["AAA", "BBB", "CCC"],
                "alpha_score": [0.2, 0.9, 0.4],
                "current_weight": [0.0, 0.0, 0.0],
                "portfolio_daily_target_delta_intent": [0.0, 0.0, 0.0],
            }
        )

        policy, global_targets = predict_policy_portfolio_set_v5(
            artifact,
            state_frame=state_frame,
            daily_features={"market_downside_pressure": 0.0},
        )

        self.assertEqual(global_targets["portfolio_cashflow_decision_v1_mode"], 1.0)
        self.assertGreater(float(policy["portfolio_daily_receiver_target_intent"].sum()), 0.0)
        self.assertGreater(float(policy["portfolio_daily_target_delta_intent"].clip(lower=0.0).sum()), 0.003)
        self.assertLessEqual(float(policy["portfolio_daily_target_delta_intent"].abs().sum()), 0.08 + 1.0e-6)
        self.assertEqual(float(policy["portfolio_daily_source_target_intent"].sum()), 0.0)
        self.assertTrue(set(policy.loc[policy["portfolio_daily_receiver_target_intent"] > 0.5, "action_label"]).issubset({"open"}))


if __name__ == "__main__":
    unittest.main()
