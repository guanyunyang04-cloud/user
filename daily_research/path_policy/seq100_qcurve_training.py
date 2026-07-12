from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from daily_research.path_policy.seq100_qcurve import (
    QCURVE_RANK_ANCHORS,
    QCurveLossWeights,
    QCurveModel,
    denormalize_qcurve_outputs,
    full_day_topk_rank_loss,
    qcurve_distribution_loss,
    structured_qcurve_auxiliary_loss,
)
from daily_research.path_policy.seq100_qcurve_backtest import evaluate_qcurve_fold
from daily_research.path_policy.seq100_qcurve_data import (
    FEATURE_PROFILES,
    QCurvePack,
    _read_json,
    verify_qcurve_development_fold,
)


DEFAULT_SEED = 7
DEFAULT_MAX_EPOCHS = 10
DEFAULT_PATIENCE = 2
DEFAULT_MICROBATCH_SIZE = 512
SOFT_RECLAIM_AVAILABLE_GIB = 2.0
LOSS_WEIGHTS = QCurveLossWeights()
LOSS_WEIGHT_MAP = {
    "mean_loss": LOSS_WEIGHTS.mean,
    "quantile_loss": LOSS_WEIGHTS.quantile,
    "positive_loss": LOSS_WEIGHTS.positive,
    "soft_exit_loss": LOSS_WEIGHTS.soft_exit,
    "rank_loss": LOSS_WEIGHTS.rank,
    "structured_price_trend_aux_loss": LOSS_WEIGHTS.price_aux,
    "va_vwap_aux_loss": LOSS_WEIGHTS.va_aux,
}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _dropout_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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
    try:
        import psutil

        after = float(psutil.virtual_memory().available / 1024**3)
    except ImportError:
        after = before
    return {"available_gib_before": before, "available_gib_after": after}


def _target_stats(fold: Mapping[str, Any], action: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    values = dict(fold["normalization"]["targets"][str(action)])
    return (
        torch.as_tensor(values["median"], dtype=torch.float32, device=device),
        torch.as_tensor(values["scale"], dtype=torch.float32, device=device),
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
    aux_target: torch.Tensor,
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
    price_aux, va_aux, _ = structured_qcurve_auxiliary_loss(outputs, aux_target)
    return {
        "mean_loss": distribution["mean_loss"],
        "quantile_loss": distribution["quantile_loss"],
        "positive_loss": distribution["positive_loss"],
        "soft_exit_loss": distribution["soft_exit_loss"],
        "rank_loss": rank,
        "structured_price_trend_aux_loss": price_aux,
        "va_vwap_aux_loss": va_aux,
    }


def _weighted_components(
    components: Mapping[str, torch.Tensor],
    scales: Mapping[str, float],
) -> dict[str, torch.Tensor]:
    return {
        name: value * float(LOSS_WEIGHT_MAP[name]) * float(scales[name])
        for name, value in components.items()
    }


def _loss_total(components: Mapping[str, torch.Tensor], scales: Mapping[str, float]) -> torch.Tensor:
    weighted = _weighted_components(components, scales)
    return torch.stack(tuple(weighted.values())).sum()


def _day_targets(
    pack: QCurvePack,
    *,
    date_idx: int,
    start: int,
    stop: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    symbols = pack.candidate_symbols(start, stop)
    enter, hold = pack.q_targets(start, stop, cost="base")
    aux = pack.future_ohlcva(date_idx, symbols)
    return (
        torch.as_tensor(enter, dtype=torch.float32, device=device),
        torch.as_tensor(hold, dtype=torch.float32, device=device),
        torch.as_tensor(aux, dtype=torch.float32, device=device),
        symbols,
    )


def _forward_day_no_grad(
    *,
    model: QCurveModel,
    pack: QCurvePack,
    date_idx: int,
    symbols: np.ndarray,
    profile: str,
    normalization: Mapping[str, Any],
    device: torch.device,
    microbatch_size: int,
) -> dict[str, torch.Tensor]:
    values: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for start in range(0, symbols.size, int(microbatch_size)):
            stop = min(start + int(microbatch_size), symbols.size)
            sequence = pack.sequence_symbols(
                date_idx=date_idx,
                symbols=symbols[start:stop],
                profile=profile,
                normalization=normalization,
            )
            outputs = model(torch.as_tensor(sequence, dtype=torch.float32, device=device))
            for name, tensor in outputs.items():
                values.setdefault(name, []).append(tensor)
    return {name: torch.cat(parts, dim=0) for name, parts in values.items()}


def _freeze_loss_scales(
    *,
    model: QCurveModel,
    pack: QCurvePack,
    fold: Mapping[str, Any],
    profile: str,
    device: torch.device,
    microbatch_size: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    span = dict(fold["train_spans"][0])
    start, stop, date_idx = int(span["candidate_start"]), int(span["candidate_stop"]), int(span["date_idx"])
    raw_enter, raw_hold, aux, symbols = _day_targets(
        pack,
        date_idx=date_idx,
        start=start,
        stop=stop,
        device=device,
    )
    model.eval()
    cached: list[torch.Tensor] = []
    with torch.no_grad():
        for micro_start in range(0, symbols.size, int(microbatch_size)):
            micro_stop = min(micro_start + int(microbatch_size), symbols.size)
            sequence = pack.sequence_symbols(
                date_idx=date_idx,
                symbols=symbols[micro_start:micro_stop],
                profile=profile,
                normalization=fold["normalization"]["inputs"],
            )
            cached.append(model.encode(torch.as_tensor(sequence, dtype=torch.float32, device=device)))
    pooled = torch.cat(cached, dim=0).detach().requires_grad_(True)
    outputs = model.decode(pooled)
    enter_location, enter_scale = _target_stats(fold, "enter", device)
    hold_location, hold_scale = _target_stats(fold, "hold", device)
    components = _loss_components(
        outputs,
        raw_enter=raw_enter,
        raw_hold=raw_hold,
        aux_target=aux,
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
    aux_norm = max(float(before["auxiliary_gradient_norm"]), 1.0e-12)
    multipliers = {
        "q_policy": 1.0,
        "rank": float(np.clip((0.20 / 0.65) * (q_norm / rank_norm), 0.01, 100.0)),
        "auxiliary": float(np.clip((0.15 / 0.65) * (q_norm / aux_norm), 0.01, 100.0)),
    }
    for name in ("mean_loss", "quantile_loss", "positive_loss", "soft_exit_loss"):
        scales[name] *= multipliers["q_policy"]
    scales["rank_loss"] *= multipliers["rank"]
    for name in ("structured_price_trend_aux_loss", "va_vwap_aux_loss"):
        scales[name] *= multipliers["auxiliary"]
    after = _gradient_diagnostics(pooled, components, scales)
    calibration = {
        "fixed_date_idx": date_idx,
        "target_gradient_fractions": {"q_policy": 0.65, "rank": 0.20, "auxiliary": 0.15},
        "group_scale_multipliers": multipliers,
        "before": before,
        "after": after,
    }
    return scales, reference, calibration


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
    auxiliary = weighted["structured_price_trend_aux_loss"] + weighted["va_vwap_aux_loss"]
    q_policy_grad = torch.autograd.grad(q_policy, pooled, retain_graph=True)[0]
    rank_grad = torch.autograd.grad(rank, pooled, retain_graph=True)[0]
    auxiliary_grad = torch.autograd.grad(auxiliary, pooled, retain_graph=True)[0]
    primary_grad = q_policy_grad + rank_grad
    q_policy_norm = torch.linalg.vector_norm(q_policy_grad)
    rank_norm = torch.linalg.vector_norm(rank_grad)
    primary_norm = torch.linalg.vector_norm(primary_grad)
    auxiliary_norm = torch.linalg.vector_norm(auxiliary_grad)
    component_norm_sum = q_policy_norm + rank_norm + auxiliary_norm

    def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        denominator = (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)).clamp_min(1.0e-12)
        return torch.sum(left * right) / denominator

    return {
        "q_policy_gradient_norm": float(q_policy_norm.detach().cpu()),
        "rank_gradient_norm": float(rank_norm.detach().cpu()),
        "q_policy_rank_gradient_norm": float(primary_norm.detach().cpu()),
        "auxiliary_gradient_norm": float(auxiliary_norm.detach().cpu()),
        "q_policy_gradient_fraction": float((q_policy_norm / component_norm_sum.clamp_min(1.0e-12)).detach().cpu()),
        "rank_gradient_fraction": float((rank_norm / component_norm_sum.clamp_min(1.0e-12)).detach().cpu()),
        "auxiliary_gradient_fraction": float((auxiliary_norm / component_norm_sum.clamp_min(1.0e-12)).detach().cpu()),
        "q_policy_rank_gradient_fraction": float(
            ((q_policy_norm + rank_norm) / component_norm_sum.clamp_min(1.0e-12)).detach().cpu()
        ),
        "q_policy_rank_gradient_cosine": float(cosine(q_policy_grad, rank_grad).detach().cpu()),
        "q_policy_auxiliary_gradient_cosine": float(cosine(q_policy_grad, auxiliary_grad).detach().cpu()),
        "rank_auxiliary_gradient_cosine": float(cosine(rank_grad, auxiliary_grad).detach().cpu()),
        "primary_auxiliary_gradient_cosine": float(cosine(primary_grad, auxiliary_grad).detach().cpu()),
    }


def _train_gradient_cache_day(
    *,
    model: QCurveModel,
    optimizer: torch.optim.Optimizer,
    pack: QCurvePack,
    fold: Mapping[str, Any],
    span: Mapping[str, Any],
    profile: str,
    device: torch.device,
    microbatch_size: int,
    loss_scales: Mapping[str, float],
    epoch: int,
    seed: int,
    record_gradient_diagnostics: bool,
) -> tuple[dict[str, float], dict[str, float] | None]:
    date_idx = int(span["date_idx"])
    start, stop = int(span["candidate_start"]), int(span["candidate_stop"])
    raw_enter, raw_hold, aux, symbols = _day_targets(
        pack,
        date_idx=date_idx,
        start=start,
        stop=stop,
        device=device,
    )
    normalization = fold["normalization"]["inputs"]
    optimizer.zero_grad(set_to_none=True)
    cached: list[torch.Tensor] = []
    dropout_seeds: list[int] = []
    model.train()
    with torch.no_grad():
        for micro_start in range(0, symbols.size, int(microbatch_size)):
            micro_stop = min(micro_start + int(microbatch_size), symbols.size)
            dropout_seed = int(seed) * 10_000_019 + int(epoch) * 100_003 + date_idx * 97 + micro_start
            dropout_seeds.append(dropout_seed)
            _dropout_seed(dropout_seed)
            sequence = pack.sequence_symbols(
                date_idx=date_idx,
                symbols=symbols[micro_start:micro_stop],
                profile=profile,
                normalization=normalization,
            )
            cached.append(model.encode(torch.as_tensor(sequence, dtype=torch.float32, device=device)))
    pooled = torch.cat(cached, dim=0).detach().requires_grad_(True)
    outputs = model.decode(pooled)
    enter_location, enter_scale = _target_stats(fold, "enter", device)
    hold_location, hold_scale = _target_stats(fold, "hold", device)
    components = _loss_components(
        outputs,
        raw_enter=raw_enter,
        raw_hold=raw_hold,
        aux_target=aux,
        enter_location=enter_location,
        enter_scale=enter_scale,
        hold_location=hold_location,
        hold_scale=hold_scale,
    )
    diagnostics = (
        _gradient_diagnostics(pooled, components, loss_scales)
        if bool(record_gradient_diagnostics)
        else None
    )
    total = _loss_total(components, loss_scales)
    total.backward()
    representation_gradient = pooled.grad.detach()

    offset = 0
    for micro_idx, micro_start in enumerate(range(0, symbols.size, int(microbatch_size))):
        micro_stop = min(micro_start + int(microbatch_size), symbols.size)
        _dropout_seed(dropout_seeds[micro_idx])
        sequence = pack.sequence_symbols(
            date_idx=date_idx,
            symbols=symbols[micro_start:micro_stop],
            profile=profile,
            normalization=normalization,
        )
        replay = model.encode(torch.as_tensor(sequence, dtype=torch.float32, device=device))
        count = micro_stop - micro_start
        torch.autograd.backward(replay, representation_gradient[offset : offset + count])
        offset += count
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0).detach().cpu())
    optimizer.step()
    result = {name: float(value.detach().cpu()) for name, value in components.items()}
    result["total_loss"] = float(total.detach().cpu())
    result["model_gradient_norm"] = gradient_norm
    return result, diagnostics


def _evaluate_development_loss(
    *,
    model: QCurveModel,
    pack: QCurvePack,
    fold: Mapping[str, Any],
    profile: str,
    device: torch.device,
    microbatch_size: int,
    loss_scales: Mapping[str, float],
) -> tuple[float, dict[str, float]]:
    model.eval()
    totals: list[float] = []
    component_values: dict[str, list[float]] = {name: [] for name in LOSS_WEIGHT_MAP}
    enter_location, enter_scale = _target_stats(fold, "enter", device)
    hold_location, hold_scale = _target_stats(fold, "hold", device)
    with torch.no_grad():
        for span in fold["development_spans"]:
            date_idx, start, stop = int(span["date_idx"]), int(span["candidate_start"]), int(span["candidate_stop"])
            raw_enter, raw_hold, aux, symbols = _day_targets(
                pack,
                date_idx=date_idx,
                start=start,
                stop=stop,
                device=device,
            )
            outputs = _forward_day_no_grad(
                model=model,
                pack=pack,
                date_idx=date_idx,
                symbols=symbols,
                profile=profile,
                normalization=fold["normalization"]["inputs"],
                device=device,
                microbatch_size=microbatch_size,
            )
            components = _loss_components(
                outputs,
                raw_enter=raw_enter,
                raw_hold=raw_hold,
                aux_target=aux,
                enter_location=enter_location,
                enter_scale=enter_scale,
                hold_location=hold_location,
                hold_scale=hold_scale,
            )
            total = _loss_total(components, loss_scales)
            totals.append(float(total.cpu()))
            for name, value in components.items():
                component_values[name].append(float(value.cpu()))
    return float(np.mean(totals)), {
        name: float(np.mean(values)) for name, values in component_values.items()
    }


class NeuralQCurveProvider:
    def __init__(
        self,
        *,
        model: QCurveModel,
        pack: QCurvePack,
        fold: Mapping[str, Any],
        profile: str,
        device: torch.device,
        microbatch_size: int,
    ) -> None:
        self.model = model
        self.pack = pack
        self.fold = fold
        self.profile = str(profile)
        self.device = device
        self.microbatch_size = int(microbatch_size)
        self.enter_location, self.enter_scale = _target_stats(fold, "enter", device)
        self.hold_location, self.hold_scale = _target_stats(fold, "hold", device)

    def predict_symbols(self, date_idx: int, symbols: np.ndarray) -> Mapping[str, np.ndarray]:
        self.model.eval()
        values: dict[str, list[np.ndarray]] = {}
        wanted = {
            f"{action}_{name}"
            for action in ("enter", "hold")
            for name in ("mean", "q20", "q50", "q80", "p_positive")
        }
        with torch.no_grad():
            for start in range(0, int(symbols.size), self.microbatch_size):
                stop = min(start + self.microbatch_size, int(symbols.size))
                sequence = self.pack.sequence_symbols(
                    date_idx=int(date_idx),
                    symbols=symbols[start:stop],
                    profile=self.profile,
                    normalization=self.fold["normalization"]["inputs"],
                )
                outputs = self.model(torch.as_tensor(sequence, dtype=torch.float32, device=self.device))
                raw = denormalize_qcurve_outputs(
                    outputs,
                    enter_location=self.enter_location,
                    enter_scale=self.enter_scale,
                    hold_location=self.hold_location,
                    hold_scale=self.hold_scale,
                )
                for name in wanted:
                    values.setdefault(name, []).append(raw[name].detach().cpu().numpy().astype(np.float32))
        return {name: np.concatenate(parts, axis=0) for name, parts in values.items()}


def _model_config(pack: QCurvePack, profile: str) -> dict[str, Any]:
    dims = pack.sequence_group_dims(profile)
    return {
        "input_dim": int(sum(dims)),
        "hidden_dim": 128,
        "layers": 2,
        "dropout": 0.10,
        "encoder_type": "gru" if profile == "qcurve_gru" else "multiscale_tcn",
        "horizon_embedding_dim": 32,
        "input_group_dims": dims if profile == "qcurve_multiscale_ma" else None,
    }


def train_neural_qcurve_fold(
    *,
    fold_path: str | Path,
    profile: str,
    output_dir: str | Path,
    device: str = "cuda",
    seed: int = DEFAULT_SEED,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    microbatch_size: int = DEFAULT_MICROBATCH_SIZE,
    learning_rate: float = 3.0e-4,
) -> dict[str, Any]:
    if profile not in FEATURE_PROFILES:
        raise ValueError(f"neural profile must be one of {tuple(FEATURE_PROFILES)}")
    verification = verify_qcurve_development_fold(fold_path)
    if verification["status"] != "ok":
        raise ValueError(f"invalid development fold: {verification['blockers']}")
    fold = _read_json(fold_path)
    if int(seed) != 7:
        raise ValueError("the frozen Q-curve development contract requires seed 7")
    if int(max_epochs) > 10 or int(max_epochs) < 1:
        raise ValueError("maximum epochs must be within 1..10")
    if int(patience) != 2:
        raise ValueError("the frozen Q-curve development contract requires patience 2")
    resolved_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    _set_seed(seed)
    pack = QCurvePack(fold["pack_manifest"])
    config = _model_config(pack, profile)
    model = QCurveModel(**config).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1.0e-4)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best_checkpoint.pt"
    last_checkpoint_path = output / "last_complete_checkpoint.pt"
    progress_path = output / "progress.json"
    history_path = output / "training_history.json"
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    wait = 0
    start_epoch = 1
    if last_checkpoint_path.is_file():
        resumed = torch.load(last_checkpoint_path, map_location=resolved_device, weights_only=False)
        if (
            str(resumed.get("profile", "")) != profile
            or str(resumed.get("fold_contract_sha256", "")) != str(fold["fold_contract_sha256"])
            or int(resumed.get("seed", -1)) != int(seed)
        ):
            raise ValueError("last complete Q-curve checkpoint provenance mismatch")
        model.load_state_dict(resumed["model_state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        loss_scales = dict(resumed["loss_scales"])
        loss_scale_reference = dict(resumed["loss_scale_reference"])
        loss_scale_calibration = dict(resumed["loss_scale_calibration"])
        history = list(resumed.get("history", []) or [])
        best_loss = float(resumed["best_development_total_loss"])
        best_epoch = int(resumed["best_epoch"])
        wait = int(resumed["early_stopping_wait"])
        start_epoch = int(resumed["epoch"]) + 1
        if wait >= int(patience):
            start_epoch = int(max_epochs) + 1
    else:
        loss_scales, loss_scale_reference, loss_scale_calibration = _freeze_loss_scales(
            model=model,
            pack=pack,
            fold=fold,
            profile=profile,
            device=resolved_device,
            microbatch_size=int(microbatch_size),
        )
    _write_json(
        progress_path,
        {
            "status": "training",
            "profile": profile,
            "development_year": int(fold["development_year"]),
            "loss_scales": loss_scales,
            "started_at": _now(),
        },
    )
    train_spans = list(fold["train_spans"])
    for epoch in range(start_epoch, int(max_epochs) + 1):
        order = np.random.default_rng(int(seed) + epoch).permutation(len(train_spans))
        train_rows: list[dict[str, float]] = []
        gradient_diagnostics: dict[str, float] | None = None
        for order_idx, span_idx in enumerate(order):
            row, diagnostics = _train_gradient_cache_day(
                model=model,
                optimizer=optimizer,
                pack=pack,
                fold=fold,
                span=train_spans[int(span_idx)],
                profile=profile,
                device=resolved_device,
                microbatch_size=int(microbatch_size),
                loss_scales=loss_scales,
                epoch=epoch,
                seed=seed,
                record_gradient_diagnostics=int(span_idx) == 0,
            )
            train_rows.append(row)
            if diagnostics is not None:
                gradient_diagnostics = diagnostics
        development_loss, development_components = _evaluate_development_loss(
            model=model,
            pack=pack,
            fold=fold,
            profile=profile,
            device=resolved_device,
            microbatch_size=int(microbatch_size),
            loss_scales=loss_scales,
        )
        improved = bool(math.isfinite(development_loss) and development_loss < best_loss)
        if improved:
            best_loss = float(development_loss)
            best_epoch = int(epoch)
            wait = 0
            torch.save(
                {
                    "schema_version": 1,
                    "profile": profile,
                    "development_year": int(fold["development_year"]),
                    "epoch": epoch,
                    "development_total_loss": development_loss,
                    "model_config": config,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "fold_contract_sha256": fold["fold_contract_sha256"],
                    "loss_scales": loss_scales,
                    "loss_scale_calibration": loss_scale_calibration,
                    "seed": seed,
                },
                checkpoint_path,
            )
        else:
            wait += 1
        train_mean = {
            name: float(np.mean([row[name] for row in train_rows]))
            for name in train_rows[0]
        }
        epoch_row = {
            "epoch": epoch,
            "complete_train_date_count": len(train_rows),
            "train_date_equal_mean": train_mean,
            "development_total_loss": development_loss,
            "development_date_equal_components": development_components,
            "gradient_diagnostics": gradient_diagnostics,
            "improved": improved,
            "early_stopping_wait": wait,
            "resource": _soft_reclaim(),
            "finished_at": _now(),
        }
        history.append(epoch_row)
        _write_json(history_path, {"history": history})
        torch.save(
            {
                "schema_version": 1,
                "profile": profile,
                "fold_contract_sha256": fold["fold_contract_sha256"],
                "seed": seed,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_scales": loss_scales,
                "loss_scale_reference": loss_scale_reference,
                "loss_scale_calibration": loss_scale_calibration,
                "history": history,
                "best_epoch": best_epoch,
                "best_development_total_loss": best_loss,
                "early_stopping_wait": wait,
            },
            last_checkpoint_path,
        )
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
                "updated_at": _now(),
            },
        )
        if wait >= int(patience):
            break
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    provider = NeuralQCurveProvider(
        model=model,
        pack=pack,
        fold=fold,
        profile=profile,
        device=resolved_device,
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
        "checkpoint_path": str(checkpoint_path.resolve()),
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
        "loss_weights": LOSS_WEIGHT_MAP,
        "loss_scales": loss_scales,
        "loss_scale_reference": loss_scale_reference,
        "loss_scale_calibration": loss_scale_calibration,
        "model_config": config,
        "evaluation_path": str(evaluation_path.resolve()),
        "history_path": str(history_path.resolve()),
        "completed_at": _now(),
    }
    summary_path = _write_json(output / "summary.json", summary)
    _write_json(progress_path, {"status": "completed", "summary_path": str(summary_path.resolve()), "updated_at": _now()})
    return summary


def smoke_neural_gradient_cache(
    *,
    fold_path: str | Path,
    profile: str,
    device: str = "cuda",
    microbatch_size: int = DEFAULT_MICROBATCH_SIZE,
) -> dict[str, Any]:
    verification = verify_qcurve_development_fold(fold_path)
    if verification["status"] != "ok":
        raise ValueError(f"invalid development fold: {verification['blockers']}")
    fold = _read_json(fold_path)
    pack = QCurvePack(fold["pack_manifest"])
    resolved_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    _set_seed(DEFAULT_SEED)
    config = _model_config(pack, profile)
    model = QCurveModel(**config).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    loss_scales, loss_scale_reference, loss_scale_calibration = _freeze_loss_scales(
        model=model,
        pack=pack,
        fold=fold,
        profile=profile,
        device=resolved_device,
        microbatch_size=int(microbatch_size),
    )
    span = dict(fold["train_spans"][0])
    parts, diagnostics = _train_gradient_cache_day(
        model=model,
        optimizer=optimizer,
        pack=pack,
        fold=fold,
        span=span,
        profile=profile,
        device=resolved_device,
        microbatch_size=int(microbatch_size),
        loss_scales=loss_scales,
        epoch=1,
        seed=DEFAULT_SEED,
        record_gradient_diagnostics=True,
    )
    gpu = {}
    if resolved_device.type == "cuda":
        gpu = {
            "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
            "peak_reserved_gib": float(torch.cuda.max_memory_reserved() / 1024**3),
        }
    return {
        "status": "ok",
        "profile": profile,
        "development_year": int(fold["development_year"]),
        "date_idx": int(span["date_idx"]),
        "full_day_candidate_count": int(span["candidate_stop"] - span["candidate_start"]),
        "microbatch_size": int(microbatch_size),
        "device": str(resolved_device),
        "parts": parts,
        "gradient_diagnostics": diagnostics,
        "loss_scales": loss_scales,
        "loss_scale_reference": loss_scale_reference,
        "loss_scale_calibration": loss_scale_calibration,
        "gpu": gpu,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one Seq100 dynamic Q-curve development fold.")
    parser.add_argument("--fold", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(FEATURE_PROFILES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--microbatch-size", type=int, default=DEFAULT_MICROBATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--smoke-one-day", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if bool(args.smoke_one_day):
        result = smoke_neural_gradient_cache(
            fold_path=args.fold,
            profile=str(args.profile),
            device=str(args.device),
            microbatch_size=int(args.microbatch_size),
        )
    else:
        result = train_neural_qcurve_fold(
            fold_path=args.fold,
            profile=str(args.profile),
            output_dir=args.output_dir,
            device=str(args.device),
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
