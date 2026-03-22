from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class RuntimeProfile:
    stage: str
    device: str
    gpu_name: str
    gpu_memory_mb: int
    logical_cpus: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    use_amp: bool
    prefetch_factor: int | None
    applied_notes: list[str]


def configure_torch_runtime(device: torch.device, use_amp: bool) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high" if use_amp else "medium")
    except Exception:
        pass


def resolve_runtime_profile(
    *,
    stage: str,
    encoder_family: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    use_amp: bool,
    safe_profile: bool = True,
) -> RuntimeProfile:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logical_cpus = int(os.cpu_count() or 1)
    gpu_name = ""
    gpu_memory_mb = 0
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_memory_mb = int(props.total_memory / (1024 * 1024))

    effective_batch_size = int(batch_size)
    effective_workers = int(num_workers)
    effective_pin_memory = bool(pin_memory and device.type == "cuda")
    effective_amp = bool(use_amp and device.type == "cuda")
    prefetch_factor: int | None = None
    notes: list[str] = []

    if not safe_profile:
        if effective_workers > 0:
            prefetch_factor = 2
        return RuntimeProfile(
            stage=stage,
            device=str(device),
            gpu_name=gpu_name,
            gpu_memory_mb=gpu_memory_mb,
            logical_cpus=logical_cpus,
            batch_size=effective_batch_size,
            num_workers=effective_workers,
            pin_memory=effective_pin_memory,
            use_amp=effective_amp,
            prefetch_factor=prefetch_factor,
            applied_notes=notes,
        )

    if device.type == "cuda" and gpu_memory_mb > 0:
        if stage == "pretrain" and encoder_family == "patch_transformer":
            safe_batch_cap = 192 if gpu_memory_mb <= 6500 else 256
        elif encoder_family in {"transformer", "patch_transformer"}:
            safe_batch_cap = 192 if gpu_memory_mb <= 6500 else 256
        else:
            safe_batch_cap = 256 if gpu_memory_mb <= 6500 else 320
        if effective_batch_size > safe_batch_cap:
            notes.append(f"batch_size capped to {safe_batch_cap} for {gpu_name or 'current GPU'}")
            effective_batch_size = safe_batch_cap

    if os.name == "nt":
        safe_worker_cap = 2
        if effective_workers > safe_worker_cap:
            effective_workers = safe_worker_cap
            notes.append("num_workers capped at 2 on Windows to avoid host instability")
    else:
        safe_worker_cap = max(1, min(4, logical_cpus // 4))
        if effective_workers > safe_worker_cap:
            effective_workers = safe_worker_cap
            notes.append(f"num_workers capped at {safe_worker_cap}")

    if effective_workers > 0:
        prefetch_factor = 1 if os.name == "nt" else 2

    return RuntimeProfile(
        stage=stage,
        device=str(device),
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_memory_mb,
        logical_cpus=logical_cpus,
        batch_size=effective_batch_size,
        num_workers=effective_workers,
        pin_memory=effective_pin_memory,
        use_amp=effective_amp,
        prefetch_factor=prefetch_factor,
        applied_notes=notes,
    )
