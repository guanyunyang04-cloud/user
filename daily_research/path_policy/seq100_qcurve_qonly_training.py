from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from daily_research.path_policy.seq100_qcurve import (
    QCURVE_RANK_ANCHORS,
    QCurveModel,
    denormalize_qcurve_outputs,
    full_day_topk_rank_loss,
    qcurve_distribution_loss,
)
from daily_research.path_policy.seq100_qcurve_backtest import evaluate_qcurve_fold
from daily_research.path_policy.seq100_qcurve_data import (
    BASE_MASKS,
    QCurvePack,
    _read_json,
    verify_qcurve_development_fold,
)


QONLY_PROFILES = ("qcurve_gru_qonly", "qcurve_multiscale_ma_qonly")
BASE_FEATURE_PROFILE = {
    "qcurve_gru_qonly": "qcurve_gru",
    "qcurve_multiscale_ma_qonly": "qcurve_multiscale_ma",
}
QONLY_LOSS_WEIGHT_MAP = {
    "mean_loss": 0.25 / 0.85,
    "quantile_loss": 0.20 / 0.85,
    "positive_loss": 0.05 / 0.85,
    "soft_exit_loss": 0.10 / 0.85,
    "rank_loss": 0.25 / 0.85,
}
QONLY_OBJECTIVE_VERSION = "qcurve_qonly_proportional_v1"
QONLY_IMPLEMENTATION_VERSION = "qcurve_qonly_training_20260713_v1"
DEFAULT_SEED = 7
DEFAULT_MAX_EPOCHS = 10
DEFAULT_PATIENCE = 2
DEFAULT_MICROBATCH_SIZE = 1024
DEFAULT_PRECISION = "amp_fp16"
SOFT_RECLAIM_AVAILABLE_GIB = 2.0
IN_EPOCH_CHECKPOINT_DATE_INTERVAL = 50
IN_EPOCH_CHECKPOINT_SECONDS = 600.0
PROGRESS_DATE_INTERVAL = 25
PROGRESS_SECONDS = 300.0


if not math.isclose(sum(QONLY_LOSS_WEIGHT_MAP.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
    raise RuntimeError("Q-only loss weights must sum to one")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, target)
    return target


def _set_seed(seed: int, *, fast_cudnn: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = not bool(fast_cudnn)
    torch.backends.cudnn.benchmark = bool(fast_cudnn)


def _dropout_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def _soft_reclaim() -> dict[str, float]:
    try:
        import psutil

        before = float(psutil.virtual_memory().available / 1024**3)
    except ImportError:
        before = float("nan")
    if math.isfinite(before) and before < SOFT_RECLAIM_AVAILABLE_GIB:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _trim_process_working_set()
    try:
        import psutil

        after = float(psutil.virtual_memory().available / 1024**3)
    except ImportError:
        after = before
    return {"available_gib_before": before, "available_gib_after": after}


def _trim_process_working_set() -> None:
    if os.name != "nt":
        return
    try:
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.EmptyWorkingSet(process)
    except (AttributeError, OSError):
        return


def _close_memmap(value: Any) -> None:
    mapping = getattr(value, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _release_source_feature_panels(pack: QCurvePack) -> None:
    for name in ("daily_raw", "daily_state", "ma_state"):
        value = getattr(pack, name, None)
        if value is not None:
            _close_memmap(value)
            setattr(pack, name, None)
    for name in BASE_MASKS:
        value = pack.masks.pop(name, None)
        if value is not None:
            _close_memmap(value)
    gc.collect()
    _trim_process_working_set()


def _model_config(pack: QCurvePack, profile: str) -> dict[str, Any]:
    if profile not in QONLY_PROFILES:
        raise ValueError(f"Q-only profile must be one of {QONLY_PROFILES}")
    base_profile = BASE_FEATURE_PROFILE[profile]
    dims = pack.sequence_group_dims(base_profile)
    multiscale = profile == "qcurve_multiscale_ma_qonly"
    return {
        "input_dim": int(sum(dims)),
        "hidden_dim": 128,
        "layers": 2,
        "dropout": 0.10,
        "encoder_type": "multiscale_tcn" if multiscale else "gru",
        "horizon_embedding_dim": 32,
        "input_group_dims": dims if multiscale else None,
        "enable_auxiliary_heads": False,
    }


def _objective_digest(profile: str) -> str:
    return _canonical_sha256(
        {
            "version": QONLY_OBJECTIVE_VERSION,
            "profile": profile,
            "base_feature_profile": BASE_FEATURE_PROFILE[profile],
            "loss_weights": QONLY_LOSS_WEIGHT_MAP,
            "rank_anchors": list(QCURVE_RANK_ANCHORS),
            "rank_top_count": 32,
            "rank_true_top3_weight": 4.0,
            "auxiliary_heads": False,
        }
    )


class NormalizedInputCache:
    def __init__(
        self,
        *,
        pack: QCurvePack,
        fold: Mapping[str, Any],
        profile: str,
        cache_dir: str | Path,
        build: bool = True,
    ) -> None:
        if profile not in QONLY_PROFILES:
            raise ValueError(f"unsupported Q-only profile: {profile}")
        self.pack = pack
        self.fold = fold
        self.profile = profile
        self.base_profile = BASE_FEATURE_PROFILE[profile]
        self.group_dims = tuple(int(value) for value in pack.sequence_group_dims(self.base_profile))
        self.shape = (
            int(pack.daily_raw.shape[0]),
            int(pack.daily_raw.shape[1]),
            int(sum(self.group_dims)),
        )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = self.cache_dir / "normalized_input.float32.dat"
        self.marker_path = self.cache_dir / "normalized_input.complete.json"
        self.digest_payload = {
            "schema_version": 1,
            "fold_contract_sha256": str(fold["fold_contract_sha256"]),
            "pack_manifest": str(pack.manifest_path),
            "profile": profile,
            "base_feature_profile": self.base_profile,
            "group_dims": list(self.group_dims),
            "shape": list(self.shape),
            "normalization": fold["normalization"]["inputs"],
        }
        self.digest = _canonical_sha256(self.digest_payload)
        if not self._valid() and build:
            self._build()
        if not self._valid():
            raise ValueError(f"invalid or incomplete normalized input cache: {self.cache_dir}")
        self.panel = np.memmap(self.data_path, dtype="float32", mode="r", shape=self.shape)

    def _valid(self) -> bool:
        expected_size = int(np.prod(self.shape)) * np.dtype("float32").itemsize
        if not self.data_path.is_file() or self.data_path.stat().st_size != expected_size:
            return False
        if not self.marker_path.is_file():
            return False
        try:
            marker = json.loads(self.marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            str(marker.get("cache_digest", "")) == self.digest
            and tuple(int(value) for value in marker.get("shape", [])) == self.shape
            and tuple(int(value) for value in marker.get("group_dims", [])) == self.group_dims
        )

    def _build(self) -> None:
        temporary = self.data_path.with_suffix(self.data_path.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        output = np.memmap(temporary, dtype="float32", mode="w+", shape=self.shape)
        normalization = self.fold["normalization"]["inputs"]
        use_ma = self.base_profile == "qcurve_multiscale_ma"
        for date_start in range(0, self.shape[0], 32):
            date_stop = min(date_start + 32, self.shape[0])
            dates = slice(date_start, date_stop)
            daily = np.asarray(self.pack.daily_raw[dates], dtype=np.float32)
            state = np.asarray(self.pack.daily_state[dates], dtype=np.float32)
            price_raw = daily[..., self.pack.price_daily_idx]
            va_raw = daily[..., self.pack.va_daily_idx]
            price_available = np.isfinite(price_raw).all(axis=-1, keepdims=True).astype(np.float32)
            va_available = np.isfinite(va_raw).all(axis=-1, keepdims=True).astype(np.float32)
            price = self.pack._normalize(price_raw, normalization["daily_raw"], self.pack.price_daily_idx)
            va = self.pack._normalize(va_raw, normalization["daily_raw"], self.pack.va_daily_idx)
            state_numeric = self.pack._normalize(
                state,
                normalization["daily_state"],
                np.arange(state.shape[-1], dtype=np.int64),
            )
            masks = [
                np.asarray(self.pack.masks[name][dates], dtype=np.float32)[..., None]
                for name in BASE_MASKS
            ]
            availability = [price_available, va_available]
            if use_ma:
                ma = np.asarray(self.pack.ma_state[dates], dtype=np.float32)
                price_ma_raw = ma[..., self.pack.price_ma_idx]
                va_ma_raw = ma[..., self.pack.va_ma_idx]
                ma_available = np.isfinite(ma).all(axis=-1, keepdims=True).astype(np.float32)
                price = np.concatenate(
                    [price, self.pack._normalize(price_ma_raw, normalization["ma_state"], self.pack.price_ma_idx)],
                    axis=-1,
                )
                va = np.concatenate(
                    [va, self.pack._normalize(va_ma_raw, normalization["ma_state"], self.pack.va_ma_idx)],
                    axis=-1,
                )
                availability.append(ma_available)
            state_group = np.concatenate([state_numeric, *masks, *availability], axis=-1)
            block = np.concatenate([price, va, state_group], axis=-1)
            if block.shape[-1] != self.shape[-1]:
                raise ValueError("normalized input cache feature-width mismatch")
            output[date_start:date_stop] = block.astype(np.float32, copy=False)
            output.flush()
        del output
        os.replace(temporary, self.data_path)
        _write_json(
            self.marker_path,
            {
                "schema_version": 1,
                "status": "completed",
                "cache_digest": self.digest,
                "shape": list(self.shape),
                "group_dims": list(self.group_dims),
                "dtype": "float32",
                "profile": self.profile,
                "fold_contract_sha256": str(self.fold["fold_contract_sha256"]),
                "created_at": _now(),
            },
        )

    def sequence_symbols(self, *, date_idx: int, symbols: np.ndarray) -> np.ndarray:
        window_start = int(date_idx) - 99
        if window_start < 0:
            raise ValueError("candidate does not have a complete 100-day input window")
        symbol_values = np.asarray(symbols, dtype=np.int32).reshape(-1)
        values = np.take(
            self.panel[window_start : int(date_idx) + 1],
            symbol_values,
            axis=1,
        )
        return values.transpose(1, 0, 2)


def _target_stats(
    fold: Mapping[str, Any], action: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    values = dict(fold["normalization"]["targets"][str(action)])
    return (
        torch.as_tensor(values["median"], dtype=torch.float32, device=device),
        torch.as_tensor(values["scale"], dtype=torch.float32, device=device),
    )


def _day_targets(
    pack: QCurvePack,
    *,
    start: int,
    stop: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    symbols = pack.candidate_symbols(start, stop)
    enter, hold = pack.q_targets(start, stop, cost="base")
    return (
        torch.as_tensor(enter, dtype=torch.float32, device=device),
        torch.as_tensor(hold, dtype=torch.float32, device=device),
        symbols,
    )


def _normalized_targets(
    raw_enter: torch.Tensor,
    raw_hold: torch.Tensor,
    *,
    enter_location: torch.Tensor,
    enter_scale: torch.Tensor,
    hold_location: torch.Tensor,
    hold_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        (raw_enter - enter_location) / enter_scale.clamp_min(1.0e-8),
        (raw_hold - hold_location) / hold_scale.clamp_min(1.0e-8),
    )


def _loss_components(
    outputs: Mapping[str, torch.Tensor],
    *,
    raw_enter: torch.Tensor,
    raw_hold: torch.Tensor,
    enter_location: torch.Tensor,
    enter_scale: torch.Tensor,
    hold_location: torch.Tensor,
    hold_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    enter_target, hold_target = _normalized_targets(
        raw_enter,
        raw_hold,
        enter_location=enter_location,
        enter_scale=enter_scale,
        hold_location=hold_location,
        hold_scale=hold_scale,
    )
    _, distribution = qcurve_distribution_loss(
        outputs,
        enter_target=enter_target,
        hold_target=hold_target,
        enter_raw_target=raw_enter,
        hold_raw_target=raw_hold,
        enter_location=enter_location,
        enter_scale=enter_scale,
        hold_location=hold_location,
        hold_scale=hold_scale,
    )
    raw_outputs = denormalize_qcurve_outputs(
        outputs,
        enter_location=enter_location,
        enter_scale=enter_scale,
        hold_location=hold_location,
        hold_scale=hold_scale,
    )
    anchor_indices = torch.as_tensor([value - 2 for value in QCURVE_RANK_ANCHORS], device=raw_enter.device)
    rank = full_day_topk_rank_loss(
        raw_outputs["enter_mean"].index_select(1, anchor_indices),
        raw_enter.index_select(1, anchor_indices),
        top_count=32,
        top3_weight=4.0,
    )
    return {
        "mean_loss": distribution["mean_loss"],
        "quantile_loss": distribution["quantile_loss"],
        "positive_loss": distribution["positive_loss"],
        "soft_exit_loss": distribution["soft_exit_loss"],
        "rank_loss": rank,
    }


def _weighted_components(
    components: Mapping[str, torch.Tensor], scales: Mapping[str, float]
) -> dict[str, torch.Tensor]:
    return {
        name: components[name] * float(QONLY_LOSS_WEIGHT_MAP[name]) * float(scales[name])
        for name in QONLY_LOSS_WEIGHT_MAP
    }


def _loss_total(components: Mapping[str, torch.Tensor], scales: Mapping[str, float]) -> torch.Tensor:
    return torch.stack(tuple(_weighted_components(components, scales).values())).sum()


def _gradient_diagnostics(
    pooled: torch.Tensor,
    components: Mapping[str, torch.Tensor],
    scales: Mapping[str, float],
) -> dict[str, float]:
    weighted = _weighted_components(components, scales)
    q_policy = sum(
        weighted[name]
        for name in ("mean_loss", "quantile_loss", "positive_loss", "soft_exit_loss")
    )
    rank = weighted["rank_loss"]
    q_policy_grad = torch.autograd.grad(q_policy, pooled, retain_graph=True)[0]
    rank_grad = torch.autograd.grad(rank, pooled, retain_graph=True)[0]
    primary_grad = q_policy_grad + rank_grad
    q_norm = torch.linalg.vector_norm(q_policy_grad)
    rank_norm = torch.linalg.vector_norm(rank_grad)
    total_norm = (q_norm + rank_norm).clamp_min(1.0e-12)
    denominator = (q_norm * rank_norm).clamp_min(1.0e-12)
    return {
        "q_policy_gradient_norm": float(q_norm.detach().cpu()),
        "rank_gradient_norm": float(rank_norm.detach().cpu()),
        "q_policy_rank_gradient_norm": float(torch.linalg.vector_norm(primary_grad).detach().cpu()),
        "q_policy_gradient_fraction": float((q_norm / total_norm).detach().cpu()),
        "rank_gradient_fraction": float((rank_norm / total_norm).detach().cpu()),
        "q_policy_rank_gradient_fraction": 1.0,
        "q_policy_rank_gradient_cosine": float(
            (torch.sum(q_policy_grad * rank_grad) / denominator).detach().cpu()
        ),
    }


class _DaySequence:
    def __init__(self, values: np.ndarray, *, device: torch.device) -> None:
        self.host = torch.from_numpy(np.asarray(values, dtype=np.float32))
        self.device = device
        self.device_tensor: torch.Tensor | None = None
        self.transfer_mode = "full_day_gpu"
        try:
            self.device_tensor = self.host.to(
                device,
                non_blocking=device.type == "cuda",
            ).contiguous()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            self.host = self.host.pin_memory()
            self.device_tensor = None
            self.transfer_mode = "pinned_host_microbatch"

    def slice(self, start: int, stop: int) -> torch.Tensor:
        if self.device_tensor is not None:
            return self.device_tensor[int(start) : int(stop)]
        return self.host[int(start) : int(stop)].to(
            self.device,
            non_blocking=self.device.type == "cuda",
        )


def _autocast(device: torch.device, precision: str):
    if precision not in {"fp32", "amp_fp16"}:
        raise ValueError("precision must be fp32 or amp_fp16")
    return torch.amp.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=precision == "amp_fp16" and device.type == "cuda",
    ) if device.type == "cuda" else contextlib.nullcontext()


def _encode_day(
    *,
    model: QCurveModel,
    day: _DaySequence,
    count: int,
    microbatch_size: int,
    device: torch.device,
    precision: str,
    dropout_seeds: Sequence[int] | None,
) -> torch.Tensor:
    pooled: list[torch.Tensor] = []
    with torch.no_grad():
        for micro_idx, start in enumerate(range(0, int(count), int(microbatch_size))):
            stop = min(start + int(microbatch_size), int(count))
            if dropout_seeds is not None:
                _dropout_seed(int(dropout_seeds[micro_idx]))
            with _autocast(device, precision):
                pooled.append(model.encode(day.slice(start, stop)))
    return torch.cat(pooled, dim=0).float()


def _freeze_loss_scales(
    *,
    model: QCurveModel,
    pack: QCurvePack,
    fold: Mapping[str, Any],
    input_cache: NormalizedInputCache,
    device: torch.device,
    precision: str,
    microbatch_size: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    span = dict(fold["train_spans"][0])
    start, stop, date_idx = int(span["candidate_start"]), int(span["candidate_stop"]), int(span["date_idx"])
    raw_enter, raw_hold, symbols = _day_targets(pack, start=start, stop=stop, device=device)
    day = _DaySequence(input_cache.sequence_symbols(date_idx=date_idx, symbols=symbols), device=device)
    model.eval()
    pooled = _encode_day(
        model=model,
        day=day,
        count=symbols.size,
        microbatch_size=microbatch_size,
        device=device,
        precision=precision,
        dropout_seeds=None,
    ).detach().requires_grad_(True)
    outputs = model.decode(pooled)
    enter_location, enter_scale = _target_stats(fold, "enter", device)
    hold_location, hold_scale = _target_stats(fold, "hold", device)
    components = _loss_components(
        outputs,
        raw_enter=raw_enter,
        raw_hold=raw_hold,
        enter_location=enter_location,
        enter_scale=enter_scale,
        hold_location=hold_location,
        hold_scale=hold_scale,
    )
    reference = {name: float(value.detach().cpu()) for name, value in components.items()}
    scales = {
        name: float(np.clip(1.0 / max(abs(value), 1.0e-3), 0.1, 100.0))
        for name, value in reference.items()
    }
    before = _gradient_diagnostics(pooled, components, scales)
    q_norm = max(float(before["q_policy_gradient_norm"]), 1.0e-12)
    rank_norm = max(float(before["rank_gradient_norm"]), 1.0e-12)
    target_q, target_rank = 0.65 / 0.85, 0.20 / 0.85
    rank_multiplier = float(np.clip((target_rank / target_q) * (q_norm / rank_norm), 0.01, 100.0))
    scales["rank_loss"] *= rank_multiplier
    after = _gradient_diagnostics(pooled, components, scales)
    return scales, reference, {
        "fixed_date_idx": date_idx,
        "target_gradient_fractions": {"q_policy": target_q, "rank": target_rank},
        "group_scale_multipliers": {"q_policy": 1.0, "rank": rank_multiplier},
        "before": before,
        "after": after,
    }


def _train_day(
    *,
    model: QCurveModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    pack: QCurvePack,
    fold: Mapping[str, Any],
    input_cache: NormalizedInputCache,
    span: Mapping[str, Any],
    device: torch.device,
    precision: str,
    microbatch_size: int,
    loss_scales: Mapping[str, float],
    epoch: int,
    seed: int,
    record_gradient_diagnostics: bool,
) -> tuple[dict[str, float], dict[str, float] | None, dict[str, float]]:
    wall = time.perf_counter()
    date_idx = int(span["date_idx"])
    start, stop = int(span["candidate_start"]), int(span["candidate_stop"])
    raw_enter, raw_hold, symbols = _day_targets(pack, start=start, stop=stop, device=device)
    build_start = time.perf_counter()
    day = _DaySequence(input_cache.sequence_symbols(date_idx=date_idx, symbols=symbols), device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    build_seconds = time.perf_counter() - build_start
    optimizer.zero_grad(set_to_none=True)
    model.train()
    dropout_seeds = [
        int(seed) * 10_000_019 + int(epoch) * 100_003 + date_idx * 97 + micro_start
        for micro_start in range(0, symbols.size, int(microbatch_size))
    ]
    encode_start = time.perf_counter()
    pooled = _encode_day(
        model=model,
        day=day,
        count=symbols.size,
        microbatch_size=microbatch_size,
        device=device,
        precision=precision,
        dropout_seeds=dropout_seeds,
    ).detach().requires_grad_(True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    first_encode_seconds = time.perf_counter() - encode_start
    outputs = model.decode(pooled)
    enter_location, enter_scale = _target_stats(fold, "enter", device)
    hold_location, hold_scale = _target_stats(fold, "hold", device)
    components = _loss_components(
        outputs,
        raw_enter=raw_enter,
        raw_hold=raw_hold,
        enter_location=enter_location,
        enter_scale=enter_scale,
        hold_location=hold_location,
        hold_scale=hold_scale,
    )
    diagnostics = _gradient_diagnostics(pooled, components, loss_scales) if record_gradient_diagnostics else None
    total = _loss_total(components, loss_scales)
    decode_start = time.perf_counter()
    scaler.scale(total).backward()
    representation_gradient = pooled.grad.detach()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    decode_seconds = time.perf_counter() - decode_start
    replay_start = time.perf_counter()
    scaler_scale_before = float(scaler.get_scale())
    offset = 0
    for micro_idx, micro_start in enumerate(range(0, symbols.size, int(microbatch_size))):
        micro_stop = min(micro_start + int(microbatch_size), symbols.size)
        _dropout_seed(dropout_seeds[micro_idx])
        with _autocast(device, precision):
            replay = model.encode(day.slice(micro_start, micro_stop))
        count = micro_stop - micro_start
        torch.autograd.backward(
            replay,
            representation_gradient[offset : offset + count].to(dtype=replay.dtype),
        )
        offset += count
    scaler.unscale_(optimizer)
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0).detach().cpu())
    scaler.step(optimizer)
    scaler.update()
    scaler_scale_after = float(scaler.get_scale())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    replay_seconds = time.perf_counter() - replay_start
    elapsed = time.perf_counter() - wall
    result = {name: float(value.detach().cpu()) for name, value in components.items()}
    result["total_loss"] = float(total.detach().cpu())
    gradient_norm_finite = math.isfinite(gradient_norm)
    result["model_gradient_norm"] = gradient_norm if gradient_norm_finite else 0.0
    result["model_gradient_norm_nonfinite"] = 0.0 if gradient_norm_finite else 1.0
    result["gradient_overflow"] = 1.0 if scaler_scale_after < scaler_scale_before else 0.0
    timing = {
        "elapsed_seconds": elapsed,
        "input_build_and_transfer_seconds": build_seconds,
        "first_encode_seconds": first_encode_seconds,
        "decode_loss_backward_seconds": decode_seconds,
        "replay_backward_optimizer_seconds": replay_seconds,
        "candidate_count": float(symbols.size),
        "stocks_per_second": float(symbols.size / max(elapsed, 1.0e-9)),
        "transfer_mode_full_day_gpu": 1.0 if day.transfer_mode == "full_day_gpu" else 0.0,
    }
    return result, diagnostics, timing


def _forward_day_no_grad(
    *,
    model: QCurveModel,
    input_cache: NormalizedInputCache,
    date_idx: int,
    symbols: np.ndarray,
    device: torch.device,
    precision: str,
    microbatch_size: int,
) -> dict[str, torch.Tensor]:
    day = _DaySequence(input_cache.sequence_symbols(date_idx=int(date_idx), symbols=symbols), device=device)
    model.eval()
    pooled = _encode_day(
        model=model,
        day=day,
        count=int(symbols.size),
        microbatch_size=int(microbatch_size),
        device=device,
        precision=precision,
        dropout_seeds=None,
    )
    with torch.no_grad():
        return model.decode(pooled)


def _evaluate_development_loss(
    *,
    model: QCurveModel,
    pack: QCurvePack,
    fold: Mapping[str, Any],
    input_cache: NormalizedInputCache,
    device: torch.device,
    precision: str,
    microbatch_size: int,
    loss_scales: Mapping[str, float],
) -> tuple[float, dict[str, float]]:
    totals: list[float] = []
    values: dict[str, list[float]] = {name: [] for name in QONLY_LOSS_WEIGHT_MAP}
    enter_location, enter_scale = _target_stats(fold, "enter", device)
    hold_location, hold_scale = _target_stats(fold, "hold", device)
    for span in fold["development_spans"]:
        date_idx = int(span["date_idx"])
        start, stop = int(span["candidate_start"]), int(span["candidate_stop"])
        raw_enter, raw_hold, symbols = _day_targets(pack, start=start, stop=stop, device=device)
        outputs = _forward_day_no_grad(
            model=model,
            input_cache=input_cache,
            date_idx=date_idx,
            symbols=symbols,
            device=device,
            precision=precision,
            microbatch_size=microbatch_size,
        )
        components = _loss_components(
            outputs,
            raw_enter=raw_enter,
            raw_hold=raw_hold,
            enter_location=enter_location,
            enter_scale=enter_scale,
            hold_location=hold_location,
            hold_scale=hold_scale,
        )
        total = _loss_total(components, loss_scales)
        totals.append(float(total.detach().cpu()))
        for name, value in components.items():
            values[name].append(float(value.detach().cpu()))
    return float(np.mean(totals)), {name: float(np.mean(items)) for name, items in values.items()}


class QOnlyQCurveProvider:
    def __init__(
        self,
        *,
        model: QCurveModel,
        pack: QCurvePack,
        fold: Mapping[str, Any],
        input_cache: NormalizedInputCache,
        device: torch.device,
        precision: str,
        microbatch_size: int,
    ) -> None:
        self.model = model
        self.pack = pack
        self.fold = fold
        self.input_cache = input_cache
        self.device = device
        self.precision = precision
        self.microbatch_size = int(microbatch_size)
        self.enter_location, self.enter_scale = _target_stats(fold, "enter", device)
        self.hold_location, self.hold_scale = _target_stats(fold, "hold", device)

    def predict_symbols(self, date_idx: int, symbols: np.ndarray) -> Mapping[str, np.ndarray]:
        outputs = _forward_day_no_grad(
            model=self.model,
            input_cache=self.input_cache,
            date_idx=int(date_idx),
            symbols=np.asarray(symbols, dtype=np.int32),
            device=self.device,
            precision=self.precision,
            microbatch_size=self.microbatch_size,
        )
        raw = denormalize_qcurve_outputs(
            outputs,
            enter_location=self.enter_location,
            enter_scale=self.enter_scale,
            hold_location=self.hold_location,
            hold_scale=self.hold_scale,
        )
        wanted = {
            f"{action}_{name}"
            for action in ("enter", "hold")
            for name in ("mean", "q20", "q50", "q80", "p_positive")
        }
        return {
            name: raw[name].detach().cpu().numpy().astype(np.float32)
            for name in wanted
        }


def _checkpoint_contract(
    *,
    profile: str,
    fold: Mapping[str, Any],
    input_cache: NormalizedInputCache,
    precision: str,
    microbatch_size: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "implementation_version": QONLY_IMPLEMENTATION_VERSION,
        "profile": profile,
        "fold_contract_sha256": str(fold["fold_contract_sha256"]),
        "objective_digest": _objective_digest(profile),
        "input_cache_digest": input_cache.digest,
        "precision": precision,
        "microbatch_size": int(microbatch_size),
        "seed": int(seed),
        "fast_cudnn": True,
    }


def _validate_checkpoint_contract(checkpoint: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    actual = dict(checkpoint.get("checkpoint_contract", {}) or {})
    if actual != dict(expected):
        raise ValueError(f"Q-only checkpoint contract mismatch: expected={expected}, actual={actual}")


def _scaler_state(scaler: torch.amp.GradScaler) -> dict[str, Any]:
    return dict(scaler.state_dict()) if scaler.is_enabled() else {}


def train_qonly_fold(
    *,
    fold_path: str | Path,
    profile: str,
    output_dir: str | Path,
    cache_dir: str | Path,
    device: str = "cuda",
    precision: str = DEFAULT_PRECISION,
    seed: int = DEFAULT_SEED,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    microbatch_size: int = DEFAULT_MICROBATCH_SIZE,
    learning_rate: float = 3.0e-4,
) -> dict[str, Any]:
    if profile not in QONLY_PROFILES:
        raise ValueError(f"profile must be one of {QONLY_PROFILES}")
    if int(seed) != DEFAULT_SEED or int(patience) != DEFAULT_PATIENCE:
        raise ValueError("frozen Q-only contract requires seed7 and patience2")
    if not 1 <= int(max_epochs) <= 10:
        raise ValueError("maximum epochs must be within 1..10")
    if precision not in {"fp32", "amp_fp16"}:
        raise ValueError("precision must be fp32 or amp_fp16")
    verification = verify_qcurve_development_fold(fold_path)
    if verification["status"] != "ok":
        raise ValueError(f"invalid development fold: {verification['blockers']}")
    fold = _read_json(fold_path)
    pack = QCurvePack(fold["pack_manifest"])
    resolved_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    if precision == "amp_fp16" and resolved_device.type != "cuda":
        raise ValueError("amp_fp16 requires CUDA")
    _set_seed(seed, fast_cudnn=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    history_path = output / "training_history.json"
    best_path = output / "best_checkpoint.pt"
    last_path = output / "last_complete_checkpoint.pt"
    in_epoch_path = output / "in_epoch_checkpoint.pt"
    _write_json(
        progress_path,
        {
            "status": "building_input_cache",
            "profile": profile,
            "development_year": int(fold["development_year"]),
            "updated_at": _now(),
        },
    )
    config = _model_config(pack, profile)
    input_cache = NormalizedInputCache(
        pack=pack,
        fold=fold,
        profile=profile,
        cache_dir=cache_dir,
        build=True,
    )
    _release_source_feature_panels(pack)
    model = QCurveModel(**config).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=precision == "amp_fp16" and resolved_device.type == "cuda",
    )
    checkpoint_contract = _checkpoint_contract(
        profile=profile,
        fold=fold,
        input_cache=input_cache,
        precision=precision,
        microbatch_size=microbatch_size,
        seed=seed,
    )
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    wait = 0
    start_epoch = 1
    resume_epoch_state: dict[str, Any] | None = None
    if in_epoch_path.is_file():
        resume_epoch_state = torch.load(in_epoch_path, map_location=resolved_device, weights_only=False)
        _validate_checkpoint_contract(resume_epoch_state, checkpoint_contract)
        model.load_state_dict(resume_epoch_state["model_state_dict"])
        optimizer.load_state_dict(resume_epoch_state["optimizer_state_dict"])
        if scaler.is_enabled():
            scaler.load_state_dict(resume_epoch_state.get("grad_scaler_state_dict", {}))
        _restore_rng_state(resume_epoch_state["rng_state"])
        history = list(resume_epoch_state.get("history", []) or [])
        best_loss = float(resume_epoch_state["best_development_total_loss"])
        best_epoch = int(resume_epoch_state["best_epoch"])
        wait = int(resume_epoch_state["early_stopping_wait"])
        start_epoch = int(resume_epoch_state["epoch"])
        loss_scales = dict(resume_epoch_state["loss_scales"])
        loss_scale_reference = dict(resume_epoch_state["loss_scale_reference"])
        loss_scale_calibration = dict(resume_epoch_state["loss_scale_calibration"])
    elif last_path.is_file():
        resumed = torch.load(last_path, map_location=resolved_device, weights_only=False)
        _validate_checkpoint_contract(resumed, checkpoint_contract)
        model.load_state_dict(resumed["model_state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        if scaler.is_enabled():
            scaler.load_state_dict(resumed.get("grad_scaler_state_dict", {}))
        history = list(resumed.get("history", []) or [])
        best_loss = float(resumed["best_development_total_loss"])
        best_epoch = int(resumed["best_epoch"])
        wait = int(resumed["early_stopping_wait"])
        start_epoch = int(resumed["epoch"]) + 1
        loss_scales = dict(resumed["loss_scales"])
        loss_scale_reference = dict(resumed["loss_scale_reference"])
        loss_scale_calibration = dict(resumed["loss_scale_calibration"])
    else:
        loss_scales, loss_scale_reference, loss_scale_calibration = _freeze_loss_scales(
            model=model,
            pack=pack,
            fold=fold,
            input_cache=input_cache,
            device=resolved_device,
            precision=precision,
            microbatch_size=int(microbatch_size),
        )
    _write_json(
        progress_path,
        {
            "status": "training",
            "profile": profile,
            "development_year": int(fold["development_year"]),
            "precision": precision,
            "microbatch_size": int(microbatch_size),
            "fast_cudnn": True,
            "input_cache_digest": input_cache.digest,
            "loss_scales": loss_scales,
            "resumed_in_epoch": resume_epoch_state is not None,
            "updated_at": _now(),
        },
    )
    spans = list(fold["train_spans"])
    for epoch in range(start_epoch, int(max_epochs) + 1):
        if resume_epoch_state is not None and int(resume_epoch_state["epoch"]) == epoch:
            order = np.asarray(resume_epoch_state["date_order"], dtype=np.int64)
            order_offset = int(resume_epoch_state["next_order_offset"])
            train_sums = {name: float(value) for name, value in resume_epoch_state["train_sums"].items()}
            if not math.isfinite(train_sums.get("model_gradient_norm", 0.0)):
                train_sums["model_gradient_norm"] = 0.0
                train_sums["model_gradient_norm_nonfinite"] = max(
                    train_sums.get("model_gradient_norm_nonfinite", 0.0),
                    1.0,
                )
            train_count = int(resume_epoch_state["train_count"])
            gradient_diagnostics = resume_epoch_state.get("gradient_diagnostics")
            epoch_elapsed_prior = float(resume_epoch_state.get("epoch_elapsed_seconds", 0.0))
            timing_sums = {
                name: float(value)
                for name, value in dict(resume_epoch_state.get("timing_sums", {}) or {}).items()
            }
        else:
            order = np.random.default_rng(int(seed) + epoch).permutation(len(spans))
            order_offset = 0
            train_sums: dict[str, float] = {}
            train_count = 0
            gradient_diagnostics = None
            epoch_elapsed_prior = 0.0
            timing_sums: dict[str, float] = {}
        resume_epoch_state = None
        epoch_wall = time.perf_counter()
        last_checkpoint_wall = epoch_wall
        last_progress_wall = epoch_wall
        recent_seconds = 0.0
        recent_stocks = 0.0
        for position in range(order_offset, len(order)):
            span_idx = int(order[position])
            row, diagnostics, timing = _train_day(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                pack=pack,
                fold=fold,
                input_cache=input_cache,
                span=spans[span_idx],
                device=resolved_device,
                precision=precision,
                microbatch_size=int(microbatch_size),
                loss_scales=loss_scales,
                epoch=epoch,
                seed=seed,
                record_gradient_diagnostics=span_idx == 0,
            )
            for name, value in row.items():
                train_sums[name] = train_sums.get(name, 0.0) + float(value)
            for name, value in timing.items():
                timing_sums[name] = timing_sums.get(name, 0.0) + float(value)
            train_count += 1
            recent_seconds += float(timing["elapsed_seconds"])
            recent_stocks += float(timing["candidate_count"])
            if diagnostics is not None:
                gradient_diagnostics = diagnostics
            now_wall = time.perf_counter()
            completed_position = position + 1
            epoch_elapsed = epoch_elapsed_prior + (now_wall - epoch_wall)
            checkpoint_due = (
                completed_position % IN_EPOCH_CHECKPOINT_DATE_INTERVAL == 0
                or now_wall - last_checkpoint_wall >= IN_EPOCH_CHECKPOINT_SECONDS
            )
            if checkpoint_due:
                _atomic_torch_save(
                    {
                        "schema_version": 1,
                        "checkpoint_kind": "in_epoch",
                        "checkpoint_contract": checkpoint_contract,
                        "epoch": epoch,
                        "date_order": order.tolist(),
                        "next_order_offset": completed_position,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "grad_scaler_state_dict": _scaler_state(scaler),
                        "rng_state": _rng_state(),
                        "train_sums": train_sums,
                        "train_count": train_count,
                        "gradient_diagnostics": gradient_diagnostics,
                        "timing_sums": timing_sums,
                        "epoch_elapsed_seconds": epoch_elapsed,
                        "loss_scales": loss_scales,
                        "loss_scale_reference": loss_scale_reference,
                        "loss_scale_calibration": loss_scale_calibration,
                        "history": history,
                        "best_epoch": best_epoch,
                        "best_development_total_loss": best_loss,
                        "early_stopping_wait": wait,
                        "saved_at": _now(),
                    },
                    in_epoch_path,
                )
                last_checkpoint_wall = now_wall
            progress_due = (
                completed_position % PROGRESS_DATE_INTERVAL == 0
                or now_wall - last_progress_wall >= PROGRESS_SECONDS
            )
            if progress_due:
                rate = recent_stocks / max(recent_seconds, 1.0e-9)
                remaining = len(order) - completed_position
                dates_per_hour = 3600.0 * (completed_position - order_offset) / max(now_wall - epoch_wall, 1.0e-9)
                _write_json(
                    progress_path,
                    {
                        "status": "training",
                        "profile": profile,
                        "development_year": int(fold["development_year"]),
                        "epoch": epoch,
                        "completed_train_dates": completed_position,
                        "total_train_dates": len(order),
                        "stocks_per_second": rate,
                        "dates_per_hour": dates_per_hour,
                        "epoch_eta_seconds": 3600.0 * remaining / max(dates_per_hour, 1.0e-9),
                        "precision": precision,
                        "microbatch_size": int(microbatch_size),
                        "resource": _soft_reclaim(),
                        "updated_at": _now(),
                    },
                )
                recent_seconds = 0.0
                recent_stocks = 0.0
                last_progress_wall = now_wall
        train_mean = {name: value / max(train_count, 1) for name, value in train_sums.items()}
        development_loss, development_components = _evaluate_development_loss(
            model=model,
            pack=pack,
            fold=fold,
            input_cache=input_cache,
            device=resolved_device,
            precision=precision,
            microbatch_size=int(microbatch_size),
            loss_scales=loss_scales,
        )
        improved = bool(math.isfinite(development_loss) and development_loss < best_loss)
        if improved:
            best_loss = float(development_loss)
            best_epoch = int(epoch)
            wait = 0
            _atomic_torch_save(
                {
                    "schema_version": 1,
                    "checkpoint_kind": "best_epoch",
                    "checkpoint_contract": checkpoint_contract,
                    "epoch": epoch,
                    "development_total_loss": development_loss,
                    "model_config": config,
                    "model_state_dict": model.state_dict(),
                    "fold_contract_sha256": fold["fold_contract_sha256"],
                    "objective_digest": _objective_digest(profile),
                    "seed": seed,
                },
                best_path,
            )
        else:
            wait += 1
        epoch_row = {
            "epoch": epoch,
            "complete_train_date_count": train_count,
            "train_date_equal_mean": train_mean,
            "development_total_loss": development_loss,
            "development_date_equal_components": development_components,
            "gradient_diagnostics": gradient_diagnostics,
            "timing_date_equal_mean": {
                name: value / max(train_count, 1) for name, value in timing_sums.items()
            },
            "precision": precision,
            "microbatch_size": int(microbatch_size),
            "improved": improved,
            "early_stopping_wait": wait,
            "resource": _soft_reclaim(),
            "finished_at": _now(),
        }
        history.append(epoch_row)
        _write_json(history_path, {"history": history})
        _atomic_torch_save(
            {
                "schema_version": 1,
                "checkpoint_kind": "last_complete_epoch",
                "checkpoint_contract": checkpoint_contract,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "grad_scaler_state_dict": _scaler_state(scaler),
                "loss_scales": loss_scales,
                "loss_scale_reference": loss_scale_reference,
                "loss_scale_calibration": loss_scale_calibration,
                "history": history,
                "best_epoch": best_epoch,
                "best_development_total_loss": best_loss,
                "early_stopping_wait": wait,
            },
            last_path,
        )
        if in_epoch_path.exists():
            in_epoch_path.unlink()
        _write_json(
            progress_path,
            {
                "status": "training",
                "profile": profile,
                "development_year": int(fold["development_year"]),
                "last_complete_epoch": epoch,
                "best_epoch": best_epoch,
                "best_development_total_loss": best_loss,
                "early_stopping_wait": wait,
                "precision": precision,
                "microbatch_size": int(microbatch_size),
                "updated_at": _now(),
            },
        )
        if wait >= int(patience):
            break
    if not best_path.is_file():
        raise RuntimeError("Q-only training finished without a finite best checkpoint")
    checkpoint = torch.load(best_path, map_location=resolved_device, weights_only=False)
    _validate_checkpoint_contract(checkpoint, checkpoint_contract)
    model.load_state_dict(checkpoint["model_state_dict"])
    provider = QOnlyQCurveProvider(
        model=model,
        pack=pack,
        fold=fold,
        input_cache=input_cache,
        device=resolved_device,
        precision=precision,
        microbatch_size=int(microbatch_size),
    )
    evaluation = evaluate_qcurve_fold(pack=pack, fold=fold, provider=provider)
    evaluation_path = _write_json(output / "evaluation.json", evaluation)
    summary = {
        "schema_version": 1,
        "status": "completed",
        "profile": profile,
        "development_year": int(fold["development_year"]),
        "seed": int(seed),
        "fold_path": str(Path(fold_path).resolve()),
        "fold_contract_sha256": fold["fold_contract_sha256"],
        "objective_digest": _objective_digest(profile),
        "implementation_version": QONLY_IMPLEMENTATION_VERSION,
        "checkpoint_path": str(best_path.resolve()),
        "best_epoch": best_epoch,
        "best_development_total_loss": best_loss,
        "complete_epochs": len(history),
        "stopped_early": len(history) < int(max_epochs),
        "early_stopping": {
            "metric": "development_total_loss",
            "mode": "min",
            "patience": int(patience),
            "topk_selects_checkpoint": False,
            "restored_best_checkpoint": True,
        },
        "loss_weights": QONLY_LOSS_WEIGHT_MAP,
        "loss_scales": loss_scales,
        "loss_scale_reference": loss_scale_reference,
        "loss_scale_calibration": loss_scale_calibration,
        "model_config": config,
        "precision": precision,
        "fast_cudnn": True,
        "bitwise_deterministic": False,
        "microbatch_size": int(microbatch_size),
        "input_cache_digest": input_cache.digest,
        "evaluation_path": str(evaluation_path.resolve()),
        "history_path": str(history_path.resolve()),
        "completed_at": _now(),
    }
    summary_path = _write_json(output / "summary.json", summary)
    _write_json(
        progress_path,
        {"status": "completed", "summary_path": str(summary_path.resolve()), "updated_at": _now()},
    )
    return summary


def _precision_snapshot(
    *,
    model: QCurveModel,
    day_values: np.ndarray,
    raw_enter: torch.Tensor,
    raw_hold: torch.Tensor,
    fold: Mapping[str, Any],
    device: torch.device,
    precision: str,
    microbatch_size: int,
) -> dict[str, Any]:
    model.eval()
    day = _DaySequence(day_values, device=device)
    pooled = _encode_day(
        model=model,
        day=day,
        count=day_values.shape[0],
        microbatch_size=microbatch_size,
        device=device,
        precision=precision,
        dropout_seeds=None,
    ).detach().requires_grad_(True)
    outputs = model.decode(pooled)
    enter_location, enter_scale = _target_stats(fold, "enter", device)
    hold_location, hold_scale = _target_stats(fold, "hold", device)
    components = _loss_components(
        outputs,
        raw_enter=raw_enter,
        raw_hold=raw_hold,
        enter_location=enter_location,
        enter_scale=enter_scale,
        hold_location=hold_location,
        hold_scale=hold_scale,
    )
    total = _loss_total(components, {name: 1.0 for name in QONLY_LOSS_WEIGHT_MAP})
    gradient = torch.autograd.grad(total, pooled)[0]
    ordered = bool(
        torch.all(outputs["enter_q20"] <= outputs["enter_q50"])
        and torch.all(outputs["enter_q50"] <= outputs["enter_q80"])
        and torch.all(outputs["hold_q20"] <= outputs["hold_q50"])
        and torch.all(outputs["hold_q50"] <= outputs["hold_q80"])
    )
    finite = bool(
        torch.isfinite(total)
        and torch.isfinite(gradient).all()
        and all(torch.isfinite(value).all() for value in outputs.values())
    )
    return {
        "total": float(total.detach().cpu()),
        "components": {name: float(value.detach().cpu()) for name, value in components.items()},
        "gradient": gradient.detach().cpu(),
        "ordered_quantiles": ordered,
        "finite": finite,
    }


def _encoder_microbatch_probe(
    *,
    model_config: Mapping[str, Any],
    day_values: np.ndarray,
    device: torch.device,
    precision: str,
    candidates: Sequence[int] = (512, 768, 1024),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in candidates:
        count = min(int(batch), int(day_values.shape[0]))
        _set_seed(DEFAULT_SEED, fast_cudnn=True)
        model = QCurveModel(**dict(model_config)).to(device).train()
        x = torch.as_tensor(day_values[:count], dtype=torch.float32, device=device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        try:
            with torch.no_grad(), _autocast(device, precision):
                model.encode(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_times: list[float] = []
            backward_times: list[float] = []
            for _ in range(2):
                started = time.perf_counter()
                with torch.no_grad(), _autocast(device, precision):
                    model.encode(x)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                forward_times.append(time.perf_counter() - started)
                model.zero_grad(set_to_none=True)
                started = time.perf_counter()
                with _autocast(device, precision):
                    encoded = model.encode(x)
                torch.autograd.backward(encoded, torch.ones_like(encoded))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                backward_times.append(time.perf_counter() - started)
            combined = float(np.mean(forward_times) + np.mean(backward_times))
            peak = float(torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else 0.0
            rows.append(
                {
                    "microbatch_size": int(batch),
                    "measured_count": count,
                    "status": "ok",
                    "forward_seconds": float(np.mean(forward_times)),
                    "forward_backward_seconds": float(np.mean(backward_times)),
                    "combined_stocks_per_second": float(count / max(combined, 1.0e-9)),
                    "peak_gpu_gib": peak,
                    "memory_accepted": peak <= 5.2,
                }
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            rows.append(
                {
                    "microbatch_size": int(batch),
                    "measured_count": count,
                    "status": "oom",
                    "error": str(exc)[:500],
                    "memory_accepted": False,
                }
            )
        del model, x
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def benchmark_qonly_profile(
    *,
    fold_path: str | Path,
    profile: str,
    cache_dir: str | Path,
    output_path: str | Path,
    baseline_path: str | Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    if profile not in QONLY_PROFILES:
        raise ValueError(f"profile must be one of {QONLY_PROFILES}")
    verification = verify_qcurve_development_fold(fold_path)
    if verification["status"] != "ok":
        raise ValueError(f"invalid development fold: {verification['blockers']}")
    fold = _read_json(fold_path)
    pack = QCurvePack(fold["pack_manifest"])
    resolved_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    if resolved_device.type != "cuda":
        raise ValueError("Q-only performance preflight requires CUDA")
    _set_seed(DEFAULT_SEED, fast_cudnn=True)
    input_cache = NormalizedInputCache(
        pack=pack,
        fold=fold,
        profile=profile,
        cache_dir=cache_dir,
        build=True,
    )
    spans = list(fold["train_spans"])
    counts = np.asarray(
        [int(span["candidate_stop"]) - int(span["candidate_start"]) for span in spans],
        dtype=np.int64,
    )
    max_index = int(np.argmax(counts))
    span = spans[max_index]
    start, stop, date_idx = int(span["candidate_start"]), int(span["candidate_stop"]), int(span["date_idx"])
    raw_enter, raw_hold, symbols = _day_targets(pack, start=start, stop=stop, device=resolved_device)
    day_values = input_cache.sequence_symbols(date_idx=date_idx, symbols=symbols)
    config = _model_config(pack, profile)
    _set_seed(DEFAULT_SEED, fast_cudnn=True)
    comparison_model = QCurveModel(**config).to(resolved_device)
    fp32 = _precision_snapshot(
        model=comparison_model,
        day_values=day_values,
        raw_enter=raw_enter,
        raw_hold=raw_hold,
        fold=fold,
        device=resolved_device,
        precision="fp32",
        microbatch_size=512,
    )
    amp = _precision_snapshot(
        model=comparison_model,
        day_values=day_values,
        raw_enter=raw_enter,
        raw_hold=raw_hold,
        fold=fold,
        device=resolved_device,
        precision="amp_fp16",
        microbatch_size=512,
    )
    total_relative = abs(amp["total"] - fp32["total"]) / max(abs(fp32["total"]), 1.0e-8)
    component_relative = {
        name: abs(amp["components"][name] - fp32["components"][name])
        / max(abs(fp32["components"][name]), 1.0e-8)
        for name in QONLY_LOSS_WEIGHT_MAP
    }
    left = fp32.pop("gradient").float().reshape(-1)
    right = amp.pop("gradient").float().reshape(-1)
    gradient_cosine = float(
        torch.dot(left, right) / (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)).clamp_min(1.0e-12)
    )
    amp_accepted = bool(
        fp32["finite"]
        and amp["finite"]
        and fp32["ordered_quantiles"]
        and amp["ordered_quantiles"]
        and total_relative <= 0.02
        and max(component_relative.values()) <= 0.03
        and gradient_cosine >= 0.98
    )
    precision = "amp_fp16" if amp_accepted else "fp32"
    microbatch_rows = _encoder_microbatch_probe(
        model_config=config,
        day_values=day_values,
        device=resolved_device,
        precision=precision,
    )
    accepted = [
        row for row in microbatch_rows
        if row.get("status") == "ok" and bool(row.get("memory_accepted"))
    ]
    if not accepted:
        raise RuntimeError("no safe Q-only microbatch candidate passed preflight")
    chosen = max(accepted, key=lambda row: float(row["combined_stocks_per_second"]))
    microbatch_size = int(chosen["microbatch_size"])
    _set_seed(DEFAULT_SEED, fast_cudnn=True)
    train_model = QCurveModel(**config).to(resolved_device)
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "amp_fp16")
    scales, reference, calibration = _freeze_loss_scales(
        model=train_model,
        pack=pack,
        fold=fold,
        input_cache=input_cache,
        device=resolved_device,
        precision=precision,
        microbatch_size=microbatch_size,
    )
    row, diagnostics, timing = _train_day(
        model=train_model,
        optimizer=optimizer,
        scaler=scaler,
        pack=pack,
        fold=fold,
        input_cache=input_cache,
        span=span,
        device=resolved_device,
        precision=precision,
        microbatch_size=microbatch_size,
        loss_scales=scales,
        epoch=1,
        seed=DEFAULT_SEED,
        record_gradient_diagnostics=True,
    )
    one_step_finite = bool(
        all(math.isfinite(float(value)) for value in row.values())
        and all(torch.isfinite(parameter).all() for parameter in train_model.parameters())
    )
    projected_seconds = float(counts.sum() / max(timing["stocks_per_second"], 1.0e-9))
    baseline = None
    speedup = None
    if baseline_path is not None and Path(baseline_path).is_file():
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        speedup = float(timing["stocks_per_second"] / float(baseline["median_stocks_per_second"]))
    performance_gate = True
    if profile == "qcurve_multiscale_ma_qonly":
        performance_gate = bool(
            speedup is not None
            and speedup >= 2.5
            and projected_seconds <= 90.0 * 60.0
            and one_step_finite
            and float(chosen["peak_gpu_gib"]) <= 5.2
        )
    result = {
        "schema_version": 1,
        "status": "passed" if performance_gate else "failed",
        "profile": profile,
        "development_year": int(fold["development_year"]),
        "fold_contract_sha256": fold["fold_contract_sha256"],
        "objective_digest": _objective_digest(profile),
        "input_cache_digest": input_cache.digest,
        "candidate_date_idx": date_idx,
        "candidate_count": int(symbols.size),
        "precision_validation": {
            "fp32": fp32,
            "amp_fp16": amp,
            "total_relative_difference": total_relative,
            "component_relative_difference": component_relative,
            "primary_gradient_cosine": gradient_cosine,
            "amp_accepted": amp_accepted,
        },
        "precision": precision,
        "microbatch_candidates": microbatch_rows,
        "microbatch_size": microbatch_size,
        "one_step_finite": one_step_finite,
        "one_step_parts": row,
        "one_step_gradient_diagnostics": diagnostics,
        "one_step_timing": timing,
        "loss_scales": scales,
        "loss_scale_reference": reference,
        "loss_scale_calibration": calibration,
        "baseline_median_stocks_per_second": (
            None if baseline is None else float(baseline["median_stocks_per_second"])
        ),
        "speedup_vs_old_multiscale": speedup,
        "projected_2022_train_epoch_seconds": projected_seconds,
        "performance_gate_passed": performance_gate,
        "fast_cudnn": True,
        "bitwise_deterministic": False,
        "created_at": _now(),
    }
    _write_json(output_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or preflight one Seq100 Q-only development fold.")
    parser.add_argument("--fold", type=Path, required=True)
    parser.add_argument("--profile", choices=QONLY_PROFILES, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "amp_fp16"), default=DEFAULT_PRECISION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--microbatch-size", type=int, default=DEFAULT_MICROBATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--baseline", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.preflight_output is not None:
        result = benchmark_qonly_profile(
            fold_path=args.fold,
            profile=str(args.profile),
            cache_dir=args.cache_dir,
            output_path=args.preflight_output,
            baseline_path=args.baseline,
            device=str(args.device),
        )
    else:
        if args.output_dir is None:
            raise ValueError("--output-dir is required for training")
        result = train_qonly_fold(
            fold_path=args.fold,
            profile=str(args.profile),
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            device=str(args.device),
            precision=str(args.precision),
            seed=int(args.seed),
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            microbatch_size=int(args.microbatch_size),
            learning_rate=float(args.learning_rate),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
