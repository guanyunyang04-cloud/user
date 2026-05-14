import inspect
import unittest

from daily_research.continuous_policy import train_policy
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


if __name__ == "__main__":
    unittest.main()
