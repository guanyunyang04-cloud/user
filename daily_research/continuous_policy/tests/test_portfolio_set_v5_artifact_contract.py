import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from daily_research.continuous_policy.model import load_artifact, predict_policy
from daily_research.continuous_policy.model_portfolio_set_v5 import (
    PORTFOLIO_SET_V5_ARTIFACT_FILENAME,
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
            },
            global_target_defaults={"gross_exposure_target": 0.7},
        )

    def test_portfolio_set_v5_artifact_round_trips_through_generic_loader(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / PORTFOLIO_SET_V5_ARTIFACT_FILENAME
            self._artifact().save(path)

            loaded_direct = load_torch_portfolio_set_v5_artifact(path)
            loaded_generic = load_artifact(path)

        self.assertIsInstance(loaded_direct, TorchPortfolioSetV5Artifact)
        self.assertIsInstance(loaded_generic, TorchPortfolioSetV5Artifact)
        self.assertEqual(loaded_direct.training_diagnostics["trainer_backend"], TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5)

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
            self.assertIn("portfolio_daily_receiver_score", frame.columns)
            self.assertIn("portfolio_daily_source_score", frame.columns)
            self.assertIn("portfolio_daily_receiver_executable_candidate", frame.columns)
            self.assertIn("portfolio_set_v5_cash_buffer_score", frame.columns)
            self.assertIn("release_first_action_hint", frame.columns)
            self.assertAlmostEqual(
                float(frame.loc[0, "portfolio_daily_target_delta_intent"]),
                float(frame.loc[0, "portfolio_daily_target_weight_intent"]) - 0.12,
                places=8,
            )
        self.assertEqual(global_targets["release_first_allocation_v3_mode"], 1.0)
        self.assertEqual(generic_globals["release_first_allocation_v3_mode"], 1.0)


if __name__ == "__main__":
    unittest.main()
