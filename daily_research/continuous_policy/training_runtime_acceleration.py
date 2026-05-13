from __future__ import annotations

import contextlib
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TorchTrainingAcceleration:
    device_type: str
    cuda_available: bool
    amp_enabled: bool
    amp_dtype: str
    pin_memory: bool
    non_blocking_transfer: bool
    matmul_precision: str
    cudnn_benchmark: bool
    disabled_reason: str

    def to_diagnostics(self) -> dict[str, Any]:
        return asdict(self)


def configure_torch_training_acceleration(
    device: torch.device | str,
    *,
    cvxpy_layers_enabled: bool = False,
    prefer_amp: bool = True,
) -> TorchTrainingAcceleration:
    resolved_device = torch.device(device)
    cuda_available = bool(torch.cuda.is_available())
    cuda_training = resolved_device.type == "cuda" and cuda_available
    matmul_precision = ""
    cudnn_benchmark = False
    disabled_reason = ""
    amp_enabled = False
    amp_dtype = ""

    if cuda_training:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
            matmul_precision = "high"
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
            cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True

        if str(os.environ.get("CONTINUOUS_POLICY_DISABLE_AMP", "") or "").strip() in {"1", "true", "TRUE", "yes"}:
            disabled_reason = "env_disabled"
        elif cvxpy_layers_enabled:
            disabled_reason = "cvxpy_layer"
        elif not bool(prefer_amp):
            disabled_reason = "amp_disabled_by_config"
        else:
            amp_enabled = True
            amp_dtype = "float16"
    else:
        disabled_reason = "non_cuda_device" if resolved_device.type != "cuda" else "cuda_unavailable"

    return TorchTrainingAcceleration(
        device_type=resolved_device.type,
        cuda_available=cuda_available,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        pin_memory=bool(cuda_training),
        non_blocking_transfer=bool(cuda_training),
        matmul_precision=matmul_precision,
        cudnn_benchmark=cudnn_benchmark,
        disabled_reason=disabled_reason,
    )


def autocast_context(runtime: TorchTrainingAcceleration) -> contextlib.AbstractContextManager[Any]:
    if runtime.amp_enabled:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def move_to_device(value: Any, device: torch.device, *, non_blocking: bool) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=bool(non_blocking))
    if isinstance(value, dict):
        return {key: move_to_device(item, device, non_blocking=non_blocking) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device, non_blocking=non_blocking) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device, non_blocking=non_blocking) for item in value)
    return value
