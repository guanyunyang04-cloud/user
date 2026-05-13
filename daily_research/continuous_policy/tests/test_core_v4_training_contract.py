import inspect
import unittest

from daily_research.continuous_policy import train_policy
from daily_research.continuous_policy.training_contracts import (
    TRAINER_BACKEND_FORMAL_CORE_V4,
    TRAINER_BACKENDS,
    build_training_contract,
    normalize_trainer_backend,
)


class CoreV4TrainingContractTest(unittest.TestCase):
    def test_formal_torch_core_v4_backend_is_registered_shadow_only_formal(self) -> None:
        self.assertIn(TRAINER_BACKEND_FORMAL_CORE_V4, TRAINER_BACKENDS)
        self.assertEqual(normalize_trainer_backend("core_v4"), TRAINER_BACKEND_FORMAL_CORE_V4)
        self.assertEqual(normalize_trainer_backend("formal_torch_core_v4"), TRAINER_BACKEND_FORMAL_CORE_V4)

        contract = build_training_contract(
            trainer_backend=TRAINER_BACKEND_FORMAL_CORE_V4,
            runtime_env="yolos",
            requested_epochs=12,
            min_epochs=8,
            resume_mode="strict",
        )

        self.assertEqual(contract["trainer_backend"], TRAINER_BACKEND_FORMAL_CORE_V4)
        self.assertTrue(contract["epoch_based"])
        self.assertTrue(contract["resume_capable"])
        self.assertTrue(contract["gpu_required"])
        self.assertFalse(contract["promotable"])
        self.assertEqual(contract["contract_class"], "epoch_resume_shadow_research_candidate")

    def test_train_policy_routes_core_v4_without_calling_seq_v3(self) -> None:
        source = inspect.getsource(train_policy.main)

        self.assertIn("fit_policy_models_core_v4", source)
        self.assertIn("continuous_policy_core_v4_artifact.pt", source)
        self.assertIn("TRAINER_BACKEND_FORMAL_CORE_V4", source)


if __name__ == "__main__":
    unittest.main()
