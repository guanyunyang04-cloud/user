import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from daily_research.continuous_policy.model import load_artifact, predict_policy
from daily_research.continuous_policy.model_core_v4 import (
    TorchContinuousPolicyCoreV4Artifact,
    load_torch_core_v4_artifact,
    predict_policy_core_v4,
)
from daily_research.continuous_policy.training_contracts import TRAINER_BACKEND_FORMAL_CORE_V4


class CoreV4ArtifactContractTest(unittest.TestCase):
    def _artifact(self) -> TorchContinuousPolicyCoreV4Artifact:
        return TorchContinuousPolicyCoreV4Artifact(
            feature_names=["alpha_score", "current_weight", "portfolio_daily_target_delta_intent"],
            daily_feature_names=["market_risk"],
            feature_fill_values=np.zeros(3, dtype=np.float32),
            feature_means=np.zeros(3, dtype=np.float32),
            feature_stds=np.ones(3, dtype=np.float32),
            daily_fill_values=np.zeros(1, dtype=np.float32),
            daily_means=np.zeros(1, dtype=np.float32),
            daily_stds=np.ones(1, dtype=np.float32),
            linear_weights={
                "target_weight": {"alpha_score": 0.03, "current_weight": 0.65},
                "target_delta": {"portfolio_daily_target_delta_intent": 1.0, "alpha_score": 0.01},
                "release_intent": {"current_weight": 1.0, "portfolio_daily_target_delta_intent": -4.0},
            },
            linear_biases={
                "target_weight": 0.02,
                "target_delta": 0.0,
                "release_intent": 0.0,
            },
            train_summary={"run_tag": "core_v4_test"},
            training_diagnostics={
                "trainer_backend": TRAINER_BACKEND_FORMAL_CORE_V4,
                "loss_profile": "alpha_result_value_budget_split_v46",
                "supports_release_first_allocation_v3_mode": True,
            },
            training_contract={"trainer_backend": TRAINER_BACKEND_FORMAL_CORE_V4, "promotable": False},
            trained_at="2026-05-13T00:00:00+08:00",
        )

    def test_core_v4_artifact_round_trips_through_generic_loader(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "continuous_policy_core_v4_artifact.pt"
            self._artifact().save(path)

            loaded_direct = load_torch_core_v4_artifact(path)
            loaded_generic = load_artifact(path)

        self.assertIsInstance(loaded_direct, TorchContinuousPolicyCoreV4Artifact)
        self.assertIsInstance(loaded_generic, TorchContinuousPolicyCoreV4Artifact)
        self.assertEqual(loaded_direct.training_diagnostics["trainer_backend"], TRAINER_BACKEND_FORMAL_CORE_V4)

    def test_predict_policy_core_v4_outputs_release_first_semantics(self) -> None:
        artifact = self._artifact()
        state_frame = pd.DataFrame(
            {
                "ts_code": ["AAA", "BBB"],
                "alpha_score": [0.6, -0.2],
                "current_weight": [0.12, 0.0],
                "portfolio_daily_target_delta_intent": [-0.04, 0.03],
            }
        )

        policy, global_targets = predict_policy_core_v4(
            artifact,
            state_frame=state_frame,
            daily_features={"market_risk": 0.4},
        )
        generic_policy, generic_globals = predict_policy(
            artifact,
            state_frame=state_frame,
            daily_features={"market_risk": 0.4},
        )

        for frame in (policy, generic_policy):
            self.assertIn("portfolio_daily_target_weight_intent", frame.columns)
            self.assertIn("portfolio_daily_target_delta_intent", frame.columns)
            self.assertIn("portfolio_daily_release_first_intent", frame.columns)
            self.assertIn("release_first_action_hint", frame.columns)
            self.assertIn("release_first_block_reason", frame.columns)
            self.assertGreater(float(frame.loc[0, "portfolio_daily_release_first_intent"]), 0.0)
            self.assertIn(str(frame.loc[0, "release_first_action_hint"]), {"reduce", "exit"})
            self.assertEqual(str(frame.loc[1, "release_first_block_reason"]), "not_held")
        self.assertEqual(global_targets["release_first_allocation_v3_mode"], 1.0)
        self.assertEqual(generic_globals["release_first_allocation_v3_mode"], 1.0)


if __name__ == "__main__":
    unittest.main()
