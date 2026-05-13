import unittest
import inspect
from unittest.mock import patch

import torch

import daily_research.continuous_policy.model_seq_v3 as model_seq_v3
from daily_research.continuous_policy.training_runtime_acceleration import (
    configure_torch_training_acceleration,
)


class TrainingRuntimeAccelerationTest(unittest.TestCase):
    def test_cuda_runtime_enables_amp_and_fast_transfers_without_cvxpy_layers(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            runtime = configure_torch_training_acceleration(
                torch.device("cuda"),
                cvxpy_layers_enabled=False,
            )

        self.assertTrue(runtime.amp_enabled)
        self.assertEqual(runtime.amp_dtype, "float16")
        self.assertTrue(runtime.pin_memory)
        self.assertTrue(runtime.non_blocking_transfer)
        self.assertEqual(runtime.matmul_precision, "high")
        self.assertEqual(runtime.disabled_reason, "")

    def test_cvxpy_layer_keeps_fast_transfer_but_disables_amp_for_solver_safety(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            runtime = configure_torch_training_acceleration(
                torch.device("cuda"),
                cvxpy_layers_enabled=True,
            )

        self.assertFalse(runtime.amp_enabled)
        self.assertTrue(runtime.pin_memory)
        self.assertTrue(runtime.non_blocking_transfer)
        self.assertEqual(runtime.disabled_reason, "cvxpy_layer")

    def test_cpu_runtime_keeps_acceleration_disabled(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            runtime = configure_torch_training_acceleration(torch.device("cpu"))

        self.assertFalse(runtime.amp_enabled)
        self.assertFalse(runtime.pin_memory)
        self.assertFalse(runtime.non_blocking_transfer)
        self.assertEqual(runtime.disabled_reason, "non_cuda_device")

    def test_seq_v3_training_diagnostics_include_progress_contract(self) -> None:
        source = inspect.getsource(model_seq_v3.fit_policy_models_v3)

        self.assertIn("train_dataframe_ready", source)
        self.assertIn("train_dataloader_ready", source)
        self.assertIn("train_epoch_complete", source)
        self.assertIn("training_complete", source)
        self.assertIn("progress_event_count", source)
        self.assertNotIn("with torch.no_grad(), autocast_context", source)


if __name__ == "__main__":
    unittest.main()
