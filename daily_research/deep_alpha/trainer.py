from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from daily_research.progress import create_progress, progress_write
from daily_research.deep_alpha.sequence_dataset import STRUCTURE_ID_TO_LABEL, STRUCTURE_LABEL_TO_ID


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    valid_loss: float
    train_reg_loss: float
    train_rank_loss: float
    train_listwise_loss: float
    train_aux_loss: float
    train_proto_loss: float
    valid_reg_loss: float
    valid_rank_loss: float
    valid_listwise_loss: float
    valid_aux_loss: float
    valid_proto_loss: float
    selection_metric_name: str = "valid_loss"
    selection_metric_value: float = float("nan")


@dataclass
class PretrainEpochRecord:
    epoch: int
    train_loss: float
    valid_loss: float
    train_mask_ratio: float
    valid_mask_ratio: float


@dataclass
class TrainingDiagnostics:
    epochs_requested: int
    epochs_completed: int
    best_epoch: int
    best_valid_loss: float
    final_valid_loss: float
    selected_epoch: int
    checkpoint_selection_mode: str
    selected_metric_name: str
    selected_metric_value: float
    selection_metric_final_value: float
    selected_epoch_ratio: float
    selected_epoch_gap_to_budget: int
    selected_in_tail: bool
    selected_at_right_boundary: bool
    stopped_early: bool
    still_improving: bool
    objective_aligned_budget_pressure: bool
    learning_rate_final: float
    status: str
    recommendation: str


@dataclass
class TrainingRunResult:
    history: List[EpochRecord]
    diagnostics: TrainingDiagnostics
    selected_model_state_dict: Dict[str, Any]
    last_model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Dict[str, Any]
    scaler_state_dict: Dict[str, Any]
    training_state: Dict[str, Any]


@dataclass
class PretrainRunResult:
    history: List[PretrainEpochRecord]
    diagnostics: TrainingDiagnostics


def _format_training_step_desc(
    *,
    phase_label: str,
    epoch: int,
    epoch_total: int,
    batch_idx: int,
    batch_total: int,
) -> str:
    return f"Finetune epoch {epoch}/{epoch_total} {phase_label} batch {batch_idx}/{batch_total}"


def _format_pretrain_step_desc(
    *,
    phase_label: str,
    epoch: int,
    batch_idx: int,
    batch_total: int,
    epoch_budget: int,
    max_epoch_budget: int,
) -> str:
    budget_text = (
        f"budget={epoch_budget}/{max_epoch_budget}"
        if int(max_epoch_budget) > int(epoch_budget)
        else f"budget={epoch_budget}"
    )
    return f"Pretrain epoch {epoch} {phase_label} batch {batch_idx}/{batch_total} | {budget_text}"


def _build_training_diagnostics(
    *,
    history: List[EpochRecord] | List[PretrainEpochRecord],
    epochs_requested: int,
    stopped_early: bool,
    learning_rate_final: float,
    min_improvement: float,
    selected_epoch: int | None = None,
    checkpoint_selection_mode: str = "valid_loss",
    selected_metric_name: str = "valid_loss",
    selected_metric_value: float = float("nan"),
    checkpoint_selection_min_improvement: float | None = None,
) -> TrainingDiagnostics:
    normalized_selection_mode = str(checkpoint_selection_mode or "valid_loss").strip().lower() or "valid_loss"
    selection_improvement = float(
        min_improvement if checkpoint_selection_min_improvement is None else checkpoint_selection_min_improvement
    )

    if not history:
        return TrainingDiagnostics(
            epochs_requested=int(epochs_requested),
            epochs_completed=0,
            best_epoch=0,
            best_valid_loss=float("nan"),
            final_valid_loss=float("nan"),
            selected_epoch=0,
            checkpoint_selection_mode=str(normalized_selection_mode),
            selected_metric_name=str(selected_metric_name),
            selected_metric_value=float(selected_metric_value),
            selection_metric_final_value=float("nan"),
            selected_epoch_ratio=0.0,
            selected_epoch_gap_to_budget=0,
            selected_in_tail=False,
            selected_at_right_boundary=False,
            stopped_early=bool(stopped_early),
            still_improving=False,
            objective_aligned_budget_pressure=False,
            learning_rate_final=float(learning_rate_final),
            status="empty",
            recommendation="No training history was produced.",
        )

    valid_losses = np.asarray([float(record.valid_loss) for record in history], dtype=float)
    finite_mask = np.isfinite(valid_losses)
    if bool(finite_mask.any()):
        finite_indices = np.flatnonzero(finite_mask)
        finite_losses = valid_losses[finite_mask]
        best_pos = int(finite_indices[int(np.argmin(finite_losses))])
        best_valid = float(valid_losses[best_pos])
    else:
        best_pos = len(history) - 1
        best_valid = float("nan")
    final_valid = float(valid_losses[-1])

    selected_epoch_value = int(selected_epoch or history[best_pos].epoch)
    selected_epoch_ratio = float(selected_epoch_value) / float(max(len(history), 1))
    selected_epoch_gap_to_budget = max(len(history) - int(selected_epoch_value), 0)
    selected_at_right_boundary = selected_epoch_value >= len(history)
    selected_in_tail = selected_epoch_ratio >= 0.80

    if normalized_selection_mode == "valid_loss":
        selection_values = np.asarray([float(record.valid_loss) for record in history], dtype=float)
        final_selection_value = float(selection_values[-1])
        still_improving = False
        if len(selection_values) >= 2 and np.isfinite(final_selection_value):
            prior_best = np.nanmin(selection_values[:-1])
            if np.isfinite(prior_best) and final_selection_value < prior_best - float(selection_improvement):
                still_improving = True
    else:
        selection_values = np.asarray([float(getattr(record, "selection_metric_value", float("nan"))) for record in history], dtype=float)
        final_selection_value = float(selection_values[-1])
        still_improving = False
        if len(selection_values) >= 2 and np.isfinite(final_selection_value):
            prior_best = np.nanmax(selection_values[:-1])
            if np.isfinite(prior_best) and final_selection_value > prior_best + float(selection_improvement):
                still_improving = True

    objective_aligned_budget_pressure = bool(still_improving or selected_at_right_boundary or selected_in_tail)

    if objective_aligned_budget_pressure:
        status = "undertrained"
        recommendation = (
            f"{selected_metric_name} remained budget-sensitive near the right edge. "
            "Extend epochs before freezing this recipe."
        )
    elif stopped_early:
        status = "plateaued"
        recommendation = f"{selected_metric_name} plateaued; the selected checkpoint was restored."
    else:
        status = "stable"
        recommendation = f"{selected_metric_name} stabilized under the current epoch budget."

    return TrainingDiagnostics(
        epochs_requested=int(epochs_requested),
        epochs_completed=len(history),
        best_epoch=int(history[best_pos].epoch),
        best_valid_loss=best_valid,
        final_valid_loss=final_valid,
        selected_epoch=selected_epoch_value,
        checkpoint_selection_mode=str(normalized_selection_mode),
        selected_metric_name=str(selected_metric_name),
        selected_metric_value=float(selected_metric_value),
        selection_metric_final_value=final_selection_value,
        selected_epoch_ratio=float(selected_epoch_ratio),
        selected_epoch_gap_to_budget=int(selected_epoch_gap_to_budget),
        selected_in_tail=bool(selected_in_tail),
        selected_at_right_boundary=bool(selected_at_right_boundary),
        stopped_early=bool(stopped_early),
        still_improving=bool(still_improving),
        objective_aligned_budget_pressure=bool(objective_aligned_budget_pressure),
        learning_rate_final=float(learning_rate_final),
        status=status,
        recommendation=recommendation,
    )


def _build_liquidity_weights(
    liquidity_bucket: torch.Tensor,
    mode: str,
    top_bucket: int,
    bucket_count: int,
    top_weight: float,
    other_weight: float,
) -> torch.Tensor:
    weights = torch.full(
        liquidity_bucket.shape,
        float(other_weight),
        dtype=torch.float32,
        device=liquidity_bucket.device,
    )
    if mode == "none":
        return weights
    if mode == "top_vs_other":
        top_mask = liquidity_bucket == int(top_bucket)
        if bool(top_mask.any().item()):
            weights[top_mask] = float(top_weight)
        return weights
    if mode == "smooth_bucket":
        denom = max(int(bucket_count) - 1, 1)
        valid = liquidity_bucket >= 0
        if bool(valid.any().item()):
            scaled = liquidity_bucket[valid].float() / float(denom)
            weights[valid] = float(other_weight) + scaled * (float(top_weight) - float(other_weight))
        return weights
    return weights


def _build_structure_rank_weights(
    liquidity_bucket: torch.Tensor,
    structure_id: torch.Tensor,
    bucket_count: int,
    attack_structure_ids: List[int],
    protect_structure_ids: List[int],
    top_attack_rank_weight: float,
    other_protect_rank_weight: float,
) -> torch.Tensor:
    weights = torch.ones_like(liquidity_bucket, dtype=torch.float32)
    top_bucket = max(int(bucket_count) - 1, 0)
    if attack_structure_ids:
        top_mask = liquidity_bucket == int(top_bucket)
        attack_ids = torch.as_tensor(attack_structure_ids, dtype=torch.long, device=liquidity_bucket.device)
        attack_mask = top_mask & torch.isin(structure_id, attack_ids)
        if bool(attack_mask.any().item()):
            weights[attack_mask] = float(top_attack_rank_weight)
    if protect_structure_ids:
        other_mask = (liquidity_bucket >= 0) & (liquidity_bucket != int(top_bucket))
        protect_ids = torch.as_tensor(protect_structure_ids, dtype=torch.long, device=liquidity_bucket.device)
        protect_mask = other_mask & torch.isin(structure_id, protect_ids)
        if bool(protect_mask.any().item()):
            weights[protect_mask] = float(other_protect_rank_weight)
    return weights


def _build_binary_labels(
    raw_target: torch.Tensor,
    dates: List[pd.Timestamp],
    mode: str,
    top_frac: float,
    bottom_frac: float,
) -> torch.Tensor:
    labels = torch.full_like(raw_target, fill_value=-1.0)
    grouped: Dict[pd.Timestamp, List[int]] = {}
    for idx, dt in enumerate(dates):
        grouped.setdefault(pd.Timestamp(dt), []).append(idx)

    for indices in grouped.values():
        if len(indices) < 5:
            continue
        idx_tensor = torch.as_tensor(indices, dtype=torch.long, device=raw_target.device)
        values = raw_target.index_select(0, idx_tensor)
        valid = torch.isfinite(values)
        if int(valid.sum().item()) < 5:
            continue
        valid_indices = idx_tensor[valid]
        valid_values = values[valid]
        order = torch.argsort(valid_values)
        n = len(valid_indices)
        top_k = max(1, int(np.ceil(n * top_frac)))
        bottom_k = max(1, int(np.ceil(n * bottom_frac)))

        if mode == "top_rest_bce":
            labels.index_fill_(0, valid_indices, 0.0)
            labels.index_fill_(0, valid_indices[order[-top_k:]], 1.0)
        elif mode == "top_bottom_bce":
            labels.index_fill_(0, valid_indices[order[:bottom_k]], 0.0)
            labels.index_fill_(0, valid_indices[order[-top_k:]], 1.0)
        else:
            raise ValueError(f"Unsupported return_loss_mode: {mode}")
    return labels


def _compute_primary_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    dates: List[pd.Timestamp],
    target_names: List[str],
    target_weights: Dict[str, float],
    return_loss_mode: str,
    return_top_frac: float,
    return_bottom_frac: float,
    liquidity_bucket: torch.Tensor | None = None,
    liquidity_conditioning_mode: str = "none",
    liquidity_bucket_count: int = 5,
    top_liquidity_return_loss_weight: float = 1.0,
    other_liquidity_return_loss_weight: float = 1.0,
    top_liquidity_sample_weight: float = 1.0,
    other_liquidity_sample_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    losses = []
    return_losses = []
    risk_losses = []
    if liquidity_bucket is None:
        liquidity_bucket = torch.full((pred.shape[0],), -1, dtype=torch.long, device=pred.device)
    top_bucket = max(int(liquidity_bucket_count) - 1, 0)
    return_sample_weights = _build_liquidity_weights(
        liquidity_bucket=liquidity_bucket,
        mode=liquidity_conditioning_mode,
        top_bucket=top_bucket,
        bucket_count=liquidity_bucket_count,
        top_weight=top_liquidity_return_loss_weight * top_liquidity_sample_weight,
        other_weight=other_liquidity_return_loss_weight * other_liquidity_sample_weight,
    )

    for idx, name in enumerate(target_names):
        weight = float(target_weights.get(name, 1.0))
        pred_col = pred[:, idx]
        target_col = target[:, idx]
        if name.startswith("fwd_excess_"):
            if return_loss_mode == "regression":
                loss_vec = F.smooth_l1_loss(pred_col, target_col, reduction="none")
                loss_val = (loss_vec * return_sample_weights).sum() / return_sample_weights.sum().clamp_min(1e-8)
            else:
                labels = _build_binary_labels(
                    raw_target=target_col,
                    dates=dates,
                    mode=return_loss_mode,
                    top_frac=return_top_frac,
                    bottom_frac=return_bottom_frac,
                )
                valid = labels >= 0
                if not torch.any(valid):
                    continue
                loss_vec = F.binary_cross_entropy_with_logits(pred_col[valid], labels[valid], reduction="none")
                sample_weights = return_sample_weights[valid]
                loss_val = (loss_vec * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
            losses.append(loss_val * weight)
            return_losses.append(loss_val)
        elif name.startswith("event_"):
            valid = torch.isfinite(target_col)
            if not torch.any(valid):
                continue
            labels = target_col[valid].clamp(0.0, 1.0)
            loss_val = F.binary_cross_entropy_with_logits(pred_col[valid], labels, reduction="mean")
            losses.append(loss_val * weight)
            risk_losses.append(loss_val)
        else:
            loss_val = F.smooth_l1_loss(pred_col, target_col, reduction="mean")
            losses.append(loss_val * weight)
            risk_losses.append(loss_val)

    if not losses:
        zero = pred.new_tensor(0.0)
        return zero, zero

    primary = torch.stack(losses).mean()
    return_component = torch.stack(return_losses).mean() if return_losses else pred.new_tensor(0.0)
    risk_component = torch.stack(risk_losses).mean() if risk_losses else pred.new_tensor(0.0)
    component = 0.5 * (return_component + risk_component) if return_losses and risk_losses else return_component + risk_component
    return primary, component


def collate_batch(batch):
    xs, ys, ys_raw, state_ids, liquidity_buckets, structure_ids, dts, stocks = zip(*batch)
    return (
        torch.stack(xs, dim=0),
        torch.stack(ys, dim=0),
        torch.stack(ys_raw, dim=0),
        torch.as_tensor(state_ids, dtype=torch.long),
        torch.as_tensor(liquidity_buckets, dtype=torch.long),
        torch.as_tensor(structure_ids, dtype=torch.long),
        list(dts),
        list(stocks),
    )


def collate_pretrain_batch(batch):
    xs, dts, stocks = zip(*batch)
    return torch.stack(xs, dim=0), list(dts), list(stocks)


def _structure_aux_loss(
    structure_logits: torch.Tensor | None,
    structure_id: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if structure_logits is None:
        return structure_id.new_tensor(0.0, dtype=torch.float32)
    valid = structure_id >= 0
    if not torch.any(valid):
        return structure_logits.new_tensor(0.0)
    return F.cross_entropy(
        structure_logits[valid],
        structure_id[valid],
        reduction="mean",
        label_smoothing=max(float(label_smoothing), 0.0),
    )


def _structure_prototype_loss(
    embedding: torch.Tensor,
    structure_id: torch.Tensor,
    prototypes: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor:
    if prototypes is None:
        return embedding.new_tensor(0.0)
    valid = structure_id >= 0
    if not torch.any(valid):
        return embedding.new_tensor(0.0)
    safe_temp = max(float(temperature), 1e-3)
    emb = F.normalize(embedding[valid], dim=1)
    proto = F.normalize(prototypes, dim=1)
    logits = emb @ proto.T / safe_temp
    return F.cross_entropy(logits, structure_id[valid], reduction="mean")


def _build_state_structure_rank_weights(
    state_id: torch.Tensor,
    structure_id: torch.Tensor,
    target_state_ids: List[int],
    attack_structure_ids: List[int],
    protect_structure_ids: List[int],
    target_state_rank_weight: float,
    target_state_protect_rank_weight: float,
) -> torch.Tensor:
    weights = torch.ones_like(state_id, dtype=torch.float32)
    if target_state_ids:
        state_ids = torch.as_tensor(target_state_ids, dtype=torch.long, device=state_id.device)
        state_mask = torch.isin(state_id, state_ids)
        if attack_structure_ids and bool(state_mask.any().item()):
            attack_ids = torch.as_tensor(attack_structure_ids, dtype=torch.long, device=state_id.device)
            attack_mask = state_mask & torch.isin(structure_id, attack_ids)
            if bool(attack_mask.any().item()):
                weights[attack_mask] = float(target_state_rank_weight)
        if protect_structure_ids and bool(state_mask.any().item()):
            protect_ids = torch.as_tensor(protect_structure_ids, dtype=torch.long, device=state_id.device)
            protect_mask = state_mask & torch.isin(structure_id, protect_ids)
            if bool(protect_mask.any().item()):
                weights[protect_mask] = float(target_state_protect_rank_weight)
    return weights


def _pairwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    dates: List[pd.Timestamp],
    target_weight_map: Dict[int, float],
    state_id: torch.Tensor | None = None,
    liquidity_bucket: torch.Tensor | None = None,
    structure_id: torch.Tensor | None = None,
    liquidity_conditioning_mode: str = "none",
    liquidity_bucket_count: int = 5,
    top_liquidity_rank_loss_weight: float = 1.0,
    other_liquidity_rank_loss_weight: float = 1.0,
    top_liquidity_sample_weight: float = 1.0,
    other_liquidity_sample_weight: float = 1.0,
    structure_conditioning_mode: str = "none",
    top_attack_structure_ids: List[int] | None = None,
    other_protect_structure_ids: List[int] | None = None,
    top_attack_rank_weight: float = 1.0,
    other_protect_rank_weight: float = 1.0,
    target_state_ids: List[int] | None = None,
    target_state_attack_structure_ids: List[int] | None = None,
    target_state_protect_structure_ids: List[int] | None = None,
    target_state_rank_weight: float = 1.0,
    target_state_protect_rank_weight: float = 1.0,
    max_pairs_per_group: int = 2048,
) -> torch.Tensor:
    losses = []
    grouped: Dict[pd.Timestamp, List[int]] = {}
    for idx, dt in enumerate(dates):
        grouped.setdefault(pd.Timestamp(dt), []).append(idx)
    if liquidity_bucket is None:
        liquidity_bucket = torch.full((pred.shape[0],), -1, dtype=torch.long, device=pred.device)
    if state_id is None:
        state_id = torch.full((pred.shape[0],), -1, dtype=torch.long, device=pred.device)
    if structure_id is None:
        structure_id = torch.full((pred.shape[0],), -1, dtype=torch.long, device=pred.device)
    top_bucket = max(int(liquidity_bucket_count) - 1, 0)
    rank_sample_weights = _build_liquidity_weights(
        liquidity_bucket=liquidity_bucket,
        mode=liquidity_conditioning_mode,
        top_bucket=top_bucket,
        bucket_count=liquidity_bucket_count,
        top_weight=top_liquidity_rank_loss_weight * top_liquidity_sample_weight,
        other_weight=other_liquidity_rank_loss_weight * other_liquidity_sample_weight,
    )
    structure_rank_weights = _build_structure_rank_weights(
        liquidity_bucket=liquidity_bucket,
        structure_id=structure_id,
        bucket_count=liquidity_bucket_count,
        attack_structure_ids=top_attack_structure_ids or [],
        protect_structure_ids=other_protect_structure_ids or [],
        top_attack_rank_weight=top_attack_rank_weight if structure_conditioning_mode == "targeted_liquidity_rank" else 1.0,
        other_protect_rank_weight=other_protect_rank_weight if structure_conditioning_mode == "targeted_liquidity_rank" else 1.0,
    )
    combined_rank_weights = rank_sample_weights * structure_rank_weights
    if structure_conditioning_mode == "state_targeted_rank":
        state_structure_weights = _build_state_structure_rank_weights(
            state_id=state_id,
            structure_id=structure_id,
            target_state_ids=target_state_ids or [],
            attack_structure_ids=target_state_attack_structure_ids or [],
            protect_structure_ids=target_state_protect_structure_ids or [],
            target_state_rank_weight=target_state_rank_weight,
            target_state_protect_rank_weight=target_state_protect_rank_weight,
        )
        combined_rank_weights = combined_rank_weights * state_structure_weights

    for indices in grouped.values():
        if len(indices) < 4:
            continue
        idx_tensor = torch.as_tensor(indices, dtype=torch.long, device=pred.device)
        for target_idx, target_weight in target_weight_map.items():
            pred_slice = pred.index_select(0, idx_tensor)[:, target_idx]
            target_slice = target.index_select(0, idx_tensor)[:, target_idx]
            n = pred_slice.numel()
            if n < 2:
                continue
            total_pairs = n * (n - 1) // 2
            if max_pairs_per_group and total_pairs > max_pairs_per_group:
                pair_i = torch.randint(0, n, (max_pairs_per_group,), device=pred.device)
                pair_j = torch.randint(0, n - 1, (max_pairs_per_group,), device=pred.device)
                pair_j = pair_j + (pair_j >= pair_i).long()
            else:
                pair_idx = torch.triu_indices(n, n, offset=1, device=pred.device)
                pair_i, pair_j = pair_idx[0], pair_idx[1]
            true_diff = target_slice[pair_i] - target_slice[pair_j]
            valid = torch.abs(true_diff) > 1e-8
            if not torch.any(valid):
                continue
            sign = torch.sign(true_diff[valid])
            pred_diff = pred_slice[pair_i[valid]] - pred_slice[pair_j[valid]]
            group_weights = combined_rank_weights.index_select(0, idx_tensor)
            pair_weights = 0.5 * (group_weights[pair_i[valid]] + group_weights[pair_j[valid]])
            pair_loss = F.softplus(-sign * pred_diff)
            losses.append(((pair_loss * pair_weights).sum() / pair_weights.sum().clamp_min(1e-8)) * float(target_weight))

    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def _listwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    dates: List[pd.Timestamp],
    target_weight_map: Dict[int, float],
    state_id: torch.Tensor | None = None,
    liquidity_bucket: torch.Tensor | None = None,
    structure_id: torch.Tensor | None = None,
    liquidity_conditioning_mode: str = "none",
    liquidity_bucket_count: int = 5,
    top_liquidity_rank_loss_weight: float = 1.0,
    other_liquidity_rank_loss_weight: float = 1.0,
    top_liquidity_sample_weight: float = 1.0,
    other_liquidity_sample_weight: float = 1.0,
    structure_conditioning_mode: str = "none",
    top_attack_structure_ids: List[int] | None = None,
    other_protect_structure_ids: List[int] | None = None,
    top_attack_rank_weight: float = 1.0,
    other_protect_rank_weight: float = 1.0,
    target_state_ids: List[int] | None = None,
    target_state_attack_structure_ids: List[int] | None = None,
    target_state_protect_structure_ids: List[int] | None = None,
    target_state_rank_weight: float = 1.0,
    target_state_protect_rank_weight: float = 1.0,
    temperature: float = 0.35,
) -> torch.Tensor:
    losses = []
    grouped: Dict[pd.Timestamp, List[int]] = {}
    for idx, dt in enumerate(dates):
        grouped.setdefault(pd.Timestamp(dt), []).append(idx)
    if liquidity_bucket is None:
        liquidity_bucket = torch.full((pred.shape[0],), -1, dtype=torch.long, device=pred.device)
    if state_id is None:
        state_id = torch.full((pred.shape[0],), -1, dtype=torch.long, device=pred.device)
    if structure_id is None:
        structure_id = torch.full((pred.shape[0],), -1, dtype=torch.long, device=pred.device)
    top_bucket = max(int(liquidity_bucket_count) - 1, 0)
    rank_sample_weights = _build_liquidity_weights(
        liquidity_bucket=liquidity_bucket,
        mode=liquidity_conditioning_mode,
        top_bucket=top_bucket,
        bucket_count=liquidity_bucket_count,
        top_weight=top_liquidity_rank_loss_weight * top_liquidity_sample_weight,
        other_weight=other_liquidity_rank_loss_weight * other_liquidity_sample_weight,
    )
    structure_rank_weights = _build_structure_rank_weights(
        liquidity_bucket=liquidity_bucket,
        structure_id=structure_id,
        bucket_count=liquidity_bucket_count,
        attack_structure_ids=top_attack_structure_ids or [],
        protect_structure_ids=other_protect_structure_ids or [],
        top_attack_rank_weight=top_attack_rank_weight if structure_conditioning_mode == "targeted_liquidity_rank" else 1.0,
        other_protect_rank_weight=other_protect_rank_weight if structure_conditioning_mode == "targeted_liquidity_rank" else 1.0,
    )
    combined_rank_weights = rank_sample_weights * structure_rank_weights
    if structure_conditioning_mode == "state_targeted_rank":
        state_structure_weights = _build_state_structure_rank_weights(
            state_id=state_id,
            structure_id=structure_id,
            target_state_ids=target_state_ids or [],
            attack_structure_ids=target_state_attack_structure_ids or [],
            protect_structure_ids=target_state_protect_structure_ids or [],
            target_state_rank_weight=target_state_rank_weight,
            target_state_protect_rank_weight=target_state_protect_rank_weight,
        )
        combined_rank_weights = combined_rank_weights * state_structure_weights

    safe_temp = max(float(temperature), 1e-3)
    for indices in grouped.values():
        if len(indices) < 4:
            continue
        idx_tensor = torch.as_tensor(indices, dtype=torch.long, device=pred.device)
        for target_idx, target_weight in target_weight_map.items():
            pred_slice = pred.index_select(0, idx_tensor)[:, target_idx]
            target_slice = target.index_select(0, idx_tensor)[:, target_idx]
            valid = torch.isfinite(pred_slice) & torch.isfinite(target_slice)
            if int(valid.sum().item()) < 4:
                continue
            pred_valid = pred_slice[valid]
            target_valid = target_slice[valid]
            sample_weights = combined_rank_weights.index_select(0, idx_tensor)[valid]
            target_centered = target_valid - target_valid.mean()
            target_std = target_valid.std(unbiased=False)
            if torch.isfinite(target_std) and float(target_std.item()) > 1e-6:
                target_centered = target_centered / target_std
            target_probs = torch.softmax(target_centered / safe_temp, dim=0)
            pred_log_probs = F.log_softmax(pred_valid, dim=0)
            per_sample_loss = -(target_probs * pred_log_probs)
            losses.append(((per_sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)) * float(target_weight))

    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def train_multitask_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    target_names: List[str],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    ranking_loss_weight: float,
    listwise_loss_weight: float,
    listwise_temperature: float,
    max_rank_pairs_per_group: int,
    target_loss_weights: Dict[str, float],
    return_loss_mode: str,
    return_top_frac: float,
    return_bottom_frac: float,
    liquidity_conditioning_mode: str,
    liquidity_bucket_count: int,
    top_liquidity_return_loss_weight: float,
    other_liquidity_return_loss_weight: float,
    top_liquidity_rank_loss_weight: float,
    other_liquidity_rank_loss_weight: float,
    top_liquidity_sample_weight: float,
    other_liquidity_sample_weight: float,
    structure_conditioning_mode: str,
    top_attack_structure_names: List[str],
    other_protect_structure_names: List[str],
    top_attack_rank_weight: float,
    other_protect_rank_weight: float,
    target_state_ids: List[int],
    target_state_attack_structure_names: List[str],
    target_state_protect_structure_names: List[str],
    target_state_rank_weight: float,
    target_state_protect_rank_weight: float,
    aux_structure_task: bool,
    aux_structure_loss_weight: float,
    aux_structure_label_smoothing: float,
    structure_prototype_task: bool,
    structure_prototype_loss_weight: float,
    structure_prototype_temperature: float,
    device: torch.device,
    use_amp: bool = True,
    min_epochs: int = 4,
    early_stop_patience: int = 2,
    lr_plateau_patience: int = 1,
    lr_plateau_factor: float = 0.5,
    min_improvement: float = 1e-4,
    checkpoint_selection_mode: str = "valid_loss",
    checkpoint_selection_callback: Callable[[nn.Module, int], dict[str, Any]] | None = None,
    checkpoint_selection_min_improvement: float | None = None,
    resume_payload: Dict[str, Any] | None = None,
) -> TrainingRunResult:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(lr_plateau_factor),
        patience=max(int(lr_plateau_patience), 0),
        threshold=float(min_improvement),
        threshold_mode="abs",
        min_lr=max(float(learning_rate) * 0.1, 1e-6),
    )
    model.to(device)
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)
    best_valid_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    last_state = deepcopy(model.state_dict())
    no_improve_epochs = 0
    stopped_early = False
    normalized_selection_mode = str(checkpoint_selection_mode or "valid_loss").strip().lower() or "valid_loss"
    selection_improvement = float(
        min_improvement if checkpoint_selection_min_improvement is None else checkpoint_selection_min_improvement
    )
    best_selection_metric = float("-inf") if normalized_selection_mode != "valid_loss" else float("inf")
    selected_epoch = 0
    selected_metric_name = "valid_loss"
    selected_metric_value = float("nan")
    normalized_target_loss_weights = {
        name: float(target_loss_weights.get(name, 1.0)) for name in target_names
    }
    mean_weight = np.mean(list(normalized_target_loss_weights.values())) if normalized_target_loss_weights else 1.0
    normalized_target_loss_weights = {
        name: value / max(mean_weight, 1e-8) for name, value in normalized_target_loss_weights.items()
    }
    rank_target_weight_map = {
        idx: float(target_loss_weights.get(name, 1.0))
        for idx, name in enumerate(target_names)
        if name.startswith("fwd_excess_")
    }
    top_attack_structure_ids = [STRUCTURE_LABEL_TO_ID[name] for name in top_attack_structure_names if name in STRUCTURE_LABEL_TO_ID]
    other_protect_structure_ids = [STRUCTURE_LABEL_TO_ID[name] for name in other_protect_structure_names if name in STRUCTURE_LABEL_TO_ID]
    target_state_attack_structure_ids = [
        STRUCTURE_LABEL_TO_ID[name]
        for name in target_state_attack_structure_names
        if name in STRUCTURE_LABEL_TO_ID
    ]
    target_state_protect_structure_ids = [
        STRUCTURE_LABEL_TO_ID[name]
        for name in target_state_protect_structure_names
        if name in STRUCTURE_LABEL_TO_ID
    ]

    train_batch_sampler = getattr(train_loader, "batch_sampler", None)
    history: List[EpochRecord] = []
    epochs_completed = 0
    if resume_payload:
        payload_history = resume_payload.get("history", [])
        history = [
            record
            if isinstance(record, EpochRecord)
            else EpochRecord(
                epoch=int(record["epoch"]),
                train_loss=float(record["train_loss"]),
                valid_loss=float(record["valid_loss"]),
                train_reg_loss=float(record["train_reg_loss"]),
                train_rank_loss=float(record["train_rank_loss"]),
                train_listwise_loss=float(record["train_listwise_loss"]),
                train_aux_loss=float(record.get("train_aux_loss", 0.0)),
                train_proto_loss=float(record.get("train_proto_loss", 0.0)),
                valid_reg_loss=float(record["valid_reg_loss"]),
                valid_rank_loss=float(record["valid_rank_loss"]),
                valid_listwise_loss=float(record["valid_listwise_loss"]),
                valid_aux_loss=float(record.get("valid_aux_loss", 0.0)),
                valid_proto_loss=float(record.get("valid_proto_loss", 0.0)),
                selection_metric_name=str(record.get("selection_metric_name", "valid_loss")),
                selection_metric_value=float(record.get("selection_metric_value", float("nan"))),
            )
            for record in payload_history
        ]
        training_state = dict(resume_payload.get("training_state", {}))
        epochs_completed = int(training_state.get("epochs_completed", len(history)) or len(history))
        selected_state_dict = resume_payload.get("selected_model_state_dict") or resume_payload.get("model_state_dict")
        last_model_state_dict = resume_payload.get("last_model_state_dict")
        if last_model_state_dict:
            model.load_state_dict(last_model_state_dict)
            last_state = deepcopy(last_model_state_dict)
        if selected_state_dict:
            best_state = deepcopy(selected_state_dict)
        optimizer_state_dict = resume_payload.get("optimizer_state_dict")
        if optimizer_state_dict:
            optimizer.load_state_dict(optimizer_state_dict)
        scheduler_state_dict = resume_payload.get("scheduler_state_dict")
        if scheduler_state_dict:
            scheduler.load_state_dict(scheduler_state_dict)
        scaler_state_dict = resume_payload.get("scaler_state_dict")
        if scaler_state_dict:
            scaler.load_state_dict(scaler_state_dict)
        best_valid_loss = float(training_state.get("best_valid_loss", best_valid_loss))
        no_improve_epochs = int(training_state.get("no_improve_epochs", no_improve_epochs) or no_improve_epochs)
        best_selection_metric = float(training_state.get("best_selection_metric", best_selection_metric))
        selected_epoch = int(training_state.get("selected_epoch", selected_epoch) or selected_epoch)
        selected_metric_name = str(training_state.get("selected_metric_name", selected_metric_name))
        selected_metric_value = float(training_state.get("selected_metric_value", selected_metric_value))
        if train_batch_sampler is not None and hasattr(train_batch_sampler, "load_state_dict"):
            train_batch_sampler.load_state_dict(training_state.get("train_sampler_state", {}))

    if int(epochs) <= int(epochs_completed):
        model.load_state_dict(best_state)
        diagnostics = _build_training_diagnostics(
            history=history,
            epochs_requested=epochs,
            stopped_early=stopped_early,
            learning_rate_final=float(optimizer.param_groups[0]["lr"]),
            min_improvement=float(min_improvement),
            selected_epoch=selected_epoch or None,
            checkpoint_selection_mode=normalized_selection_mode,
            selected_metric_name=selected_metric_name,
            selected_metric_value=selected_metric_value,
            checkpoint_selection_min_improvement=selection_improvement,
        )
        training_state = {
            "epochs_completed": int(epochs_completed),
            "best_valid_loss": float(best_valid_loss),
            "best_selection_metric": float(best_selection_metric),
            "selected_epoch": int(selected_epoch),
            "selected_metric_name": str(selected_metric_name),
            "selected_metric_value": float(selected_metric_value),
            "no_improve_epochs": int(no_improve_epochs),
            "checkpoint_selection_mode": str(normalized_selection_mode),
            "selection_improvement": float(selection_improvement),
            "stopped_early": bool(stopped_early),
            "train_sampler_state": {}
            if train_batch_sampler is None or not hasattr(train_batch_sampler, "state_dict")
            else dict(train_batch_sampler.state_dict()),
        }
        return TrainingRunResult(
            history=history,
            diagnostics=diagnostics,
            selected_model_state_dict=deepcopy(best_state),
            last_model_state_dict=deepcopy(last_state),
            optimizer_state_dict=deepcopy(optimizer.state_dict()),
            scheduler_state_dict=deepcopy(scheduler.state_dict()),
            scaler_state_dict=deepcopy(scaler.state_dict()),
            training_state=training_state,
        )

    train_batch_count = max(len(train_loader), 1)
    valid_batch_count = max(len(valid_loader), 1)

    for epoch in range(int(epochs_completed) + 1, int(epochs) + 1):
        epoch_batch_total = train_batch_count + valid_batch_count
        with create_progress(
            total=epoch_batch_total,
            desc=f"Finetune epoch {epoch}/{epochs} setup",
            unit="batch",
            leave=False,
        ) as progress:
            model.train()
            train_losses: List[float] = []
            train_reg_losses: List[float] = []
            train_rank_losses: List[float] = []
            train_listwise_losses: List[float] = []
            train_aux_losses: List[float] = []
            train_proto_losses: List[float] = []
            for batch_idx, (x, y, _y_raw, state_id, liquidity_bucket, structure_id, dts, _) in enumerate(train_loader, start=1):
                progress.set_description_str(
                    _format_training_step_desc(
                        phase_label="train",
                        epoch=epoch,
                        epoch_total=epochs,
                        batch_idx=batch_idx,
                        batch_total=train_batch_count,
                    )
                )
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                state_id = state_id.to(device, non_blocking=True)
                liquidity_bucket = liquidity_bucket.to(device, non_blocking=True)
                structure_id = structure_id.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    pred, emb, structure_logits = model(
                        x,
                        liquidity_bucket=liquidity_bucket,
                        state_id=state_id,
                        structure_id=structure_id,
                    )
                    reg_loss, reg_component = _compute_primary_loss(
                        pred=pred,
                        target=y,
                        dates=dts,
                        target_names=target_names,
                        target_weights=normalized_target_loss_weights,
                        return_loss_mode=return_loss_mode,
                        return_top_frac=return_top_frac,
                        return_bottom_frac=return_bottom_frac,
                        liquidity_bucket=liquidity_bucket,
                        liquidity_conditioning_mode=liquidity_conditioning_mode,
                        liquidity_bucket_count=liquidity_bucket_count,
                        top_liquidity_return_loss_weight=top_liquidity_return_loss_weight,
                        other_liquidity_return_loss_weight=other_liquidity_return_loss_weight,
                        top_liquidity_sample_weight=top_liquidity_sample_weight,
                        other_liquidity_sample_weight=other_liquidity_sample_weight,
                    )
                    rank_loss = _pairwise_rank_loss(
                        pred,
                        y,
                        dts,
                        rank_target_weight_map,
                        state_id=state_id,
                        liquidity_bucket=liquidity_bucket,
                        structure_id=structure_id,
                        liquidity_conditioning_mode=liquidity_conditioning_mode,
                        liquidity_bucket_count=liquidity_bucket_count,
                        top_liquidity_rank_loss_weight=top_liquidity_rank_loss_weight,
                        other_liquidity_rank_loss_weight=other_liquidity_rank_loss_weight,
                        top_liquidity_sample_weight=top_liquidity_sample_weight,
                        other_liquidity_sample_weight=other_liquidity_sample_weight,
                        structure_conditioning_mode=structure_conditioning_mode,
                        top_attack_structure_ids=top_attack_structure_ids,
                        other_protect_structure_ids=other_protect_structure_ids,
                        top_attack_rank_weight=top_attack_rank_weight,
                        other_protect_rank_weight=other_protect_rank_weight,
                        target_state_ids=target_state_ids,
                        target_state_attack_structure_ids=target_state_attack_structure_ids,
                        target_state_protect_structure_ids=target_state_protect_structure_ids,
                        target_state_rank_weight=target_state_rank_weight,
                        target_state_protect_rank_weight=target_state_protect_rank_weight,
                        max_pairs_per_group=max_rank_pairs_per_group,
                    ) if ranking_loss_weight > 0 else pred.new_tensor(0.0)
                    listwise_loss = _listwise_rank_loss(
                        pred,
                        y,
                        dts,
                        rank_target_weight_map,
                        state_id=state_id,
                        liquidity_bucket=liquidity_bucket,
                        structure_id=structure_id,
                        liquidity_conditioning_mode=liquidity_conditioning_mode,
                        liquidity_bucket_count=liquidity_bucket_count,
                        top_liquidity_rank_loss_weight=top_liquidity_rank_loss_weight,
                        other_liquidity_rank_loss_weight=other_liquidity_rank_loss_weight,
                        top_liquidity_sample_weight=top_liquidity_sample_weight,
                        other_liquidity_sample_weight=other_liquidity_sample_weight,
                        structure_conditioning_mode=structure_conditioning_mode,
                        top_attack_structure_ids=top_attack_structure_ids,
                        other_protect_structure_ids=other_protect_structure_ids,
                        top_attack_rank_weight=top_attack_rank_weight,
                        other_protect_rank_weight=other_protect_rank_weight,
                        target_state_ids=target_state_ids,
                        target_state_attack_structure_ids=target_state_attack_structure_ids,
                        target_state_protect_structure_ids=target_state_protect_structure_ids,
                        target_state_rank_weight=target_state_rank_weight,
                        target_state_protect_rank_weight=target_state_protect_rank_weight,
                        temperature=listwise_temperature,
                    ) if listwise_loss_weight > 0 else pred.new_tensor(0.0)
                    aux_loss = _structure_aux_loss(
                        structure_logits=structure_logits,
                        structure_id=structure_id,
                        label_smoothing=aux_structure_label_smoothing,
                    ) if aux_structure_task and aux_structure_loss_weight > 0 else pred.new_tensor(0.0)
                    proto_loss = _structure_prototype_loss(
                        embedding=emb,
                        structure_id=structure_id,
                        prototypes=getattr(model, "structure_prototypes", None),
                        temperature=structure_prototype_temperature,
                    ) if structure_prototype_task and structure_prototype_loss_weight > 0 else pred.new_tensor(0.0)
                    loss = (
                        reg_loss
                        + ranking_loss_weight * rank_loss
                        + listwise_loss_weight * listwise_loss
                        + aux_structure_loss_weight * aux_loss
                        + structure_prototype_loss_weight * proto_loss
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.item()))
                train_reg_losses.append(float(reg_component.item()))
                train_rank_losses.append(float(rank_loss.item()))
                train_listwise_losses.append(float(listwise_loss.item()))
                train_aux_losses.append(float(aux_loss.item()))
                train_proto_losses.append(float(proto_loss.item()))
                progress.update(1)

            model.eval()
            valid_losses: List[float] = []
            valid_reg_losses: List[float] = []
            valid_rank_losses: List[float] = []
            valid_listwise_losses: List[float] = []
            valid_aux_losses: List[float] = []
            valid_proto_losses: List[float] = []
            with torch.no_grad():
                for batch_idx, (x, y, _y_raw, state_id, liquidity_bucket, structure_id, dts, _) in enumerate(valid_loader, start=1):
                    progress.set_description_str(
                        _format_training_step_desc(
                            phase_label="valid",
                            epoch=epoch,
                            epoch_total=epochs,
                            batch_idx=batch_idx,
                            batch_total=valid_batch_count,
                        )
                    )
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    state_id = state_id.to(device, non_blocking=True)
                    liquidity_bucket = liquidity_bucket.to(device, non_blocking=True)
                    structure_id = structure_id.to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                        pred, emb, structure_logits = model(
                            x,
                            liquidity_bucket=liquidity_bucket,
                            state_id=state_id,
                            structure_id=structure_id,
                        )
                        reg_loss, reg_component = _compute_primary_loss(
                            pred=pred,
                            target=y,
                            dates=dts,
                            target_names=target_names,
                            target_weights=normalized_target_loss_weights,
                            return_loss_mode=return_loss_mode,
                            return_top_frac=return_top_frac,
                            return_bottom_frac=return_bottom_frac,
                            liquidity_bucket=liquidity_bucket,
                            liquidity_conditioning_mode=liquidity_conditioning_mode,
                            liquidity_bucket_count=liquidity_bucket_count,
                            top_liquidity_return_loss_weight=top_liquidity_return_loss_weight,
                            other_liquidity_return_loss_weight=other_liquidity_return_loss_weight,
                            top_liquidity_sample_weight=top_liquidity_sample_weight,
                            other_liquidity_sample_weight=other_liquidity_sample_weight,
                        )
                        rank_loss = _pairwise_rank_loss(
                            pred,
                            y,
                            dts,
                            rank_target_weight_map,
                            state_id=state_id,
                            liquidity_bucket=liquidity_bucket,
                            structure_id=structure_id,
                            liquidity_conditioning_mode=liquidity_conditioning_mode,
                            liquidity_bucket_count=liquidity_bucket_count,
                            top_liquidity_rank_loss_weight=top_liquidity_rank_loss_weight,
                            other_liquidity_rank_loss_weight=other_liquidity_rank_loss_weight,
                            top_liquidity_sample_weight=top_liquidity_sample_weight,
                            other_liquidity_sample_weight=other_liquidity_sample_weight,
                            structure_conditioning_mode=structure_conditioning_mode,
                            top_attack_structure_ids=top_attack_structure_ids,
                            other_protect_structure_ids=other_protect_structure_ids,
                            top_attack_rank_weight=top_attack_rank_weight,
                            other_protect_rank_weight=other_protect_rank_weight,
                            target_state_ids=target_state_ids,
                            target_state_attack_structure_ids=target_state_attack_structure_ids,
                            target_state_protect_structure_ids=target_state_protect_structure_ids,
                            target_state_rank_weight=target_state_rank_weight,
                            target_state_protect_rank_weight=target_state_protect_rank_weight,
                            max_pairs_per_group=max_rank_pairs_per_group,
                        ) if ranking_loss_weight > 0 else pred.new_tensor(0.0)
                        listwise_loss = _listwise_rank_loss(
                            pred,
                            y,
                            dts,
                            rank_target_weight_map,
                            state_id=state_id,
                            liquidity_bucket=liquidity_bucket,
                            structure_id=structure_id,
                            liquidity_conditioning_mode=liquidity_conditioning_mode,
                            liquidity_bucket_count=liquidity_bucket_count,
                            top_liquidity_rank_loss_weight=top_liquidity_rank_loss_weight,
                            other_liquidity_rank_loss_weight=other_liquidity_rank_loss_weight,
                            top_liquidity_sample_weight=top_liquidity_sample_weight,
                            other_liquidity_sample_weight=other_liquidity_sample_weight,
                            structure_conditioning_mode=structure_conditioning_mode,
                            top_attack_structure_ids=top_attack_structure_ids,
                            other_protect_structure_ids=other_protect_structure_ids,
                            top_attack_rank_weight=top_attack_rank_weight,
                            other_protect_rank_weight=other_protect_rank_weight,
                            target_state_ids=target_state_ids,
                            target_state_attack_structure_ids=target_state_attack_structure_ids,
                            target_state_protect_structure_ids=target_state_protect_structure_ids,
                            target_state_rank_weight=target_state_rank_weight,
                            target_state_protect_rank_weight=target_state_protect_rank_weight,
                            temperature=listwise_temperature,
                        ) if listwise_loss_weight > 0 else pred.new_tensor(0.0)
                        aux_loss = _structure_aux_loss(
                            structure_logits=structure_logits,
                            structure_id=structure_id,
                            label_smoothing=aux_structure_label_smoothing,
                        ) if aux_structure_task and aux_structure_loss_weight > 0 else pred.new_tensor(0.0)
                        proto_loss = _structure_prototype_loss(
                            embedding=emb,
                            structure_id=structure_id,
                            prototypes=getattr(model, "structure_prototypes", None),
                            temperature=structure_prototype_temperature,
                        ) if structure_prototype_task and structure_prototype_loss_weight > 0 else pred.new_tensor(0.0)
                        valid_losses.append(
                            float(
                                (
                                    reg_loss
                                    + ranking_loss_weight * rank_loss
                                    + listwise_loss_weight * listwise_loss
                                    + aux_structure_loss_weight * aux_loss
                                    + structure_prototype_loss_weight * proto_loss
                                ).item()
                            )
                        )
                        valid_reg_losses.append(float(reg_component.item()))
                        valid_rank_losses.append(float(rank_loss.item()))
                        valid_listwise_losses.append(float(listwise_loss.item()))
                        valid_aux_losses.append(float(aux_loss.item()))
                        valid_proto_losses.append(float(proto_loss.item()))
                    progress.update(1)

            epoch_record = EpochRecord(
                epoch=epoch,
                train_loss=float(np.mean(train_losses)) if train_losses else np.nan,
                valid_loss=float(np.mean(valid_losses)) if valid_losses else np.nan,
                train_reg_loss=float(np.mean(train_reg_losses)) if train_reg_losses else np.nan,
                train_rank_loss=float(np.mean(train_rank_losses)) if train_rank_losses else np.nan,
                train_listwise_loss=float(np.mean(train_listwise_losses)) if train_listwise_losses else np.nan,
                train_aux_loss=float(np.mean(train_aux_losses)) if train_aux_losses else np.nan,
                train_proto_loss=float(np.mean(train_proto_losses)) if train_proto_losses else np.nan,
                valid_reg_loss=float(np.mean(valid_reg_losses)) if valid_reg_losses else np.nan,
                valid_rank_loss=float(np.mean(valid_rank_losses)) if valid_rank_losses else np.nan,
                valid_listwise_loss=float(np.mean(valid_listwise_losses)) if valid_listwise_losses else np.nan,
                valid_aux_loss=float(np.mean(valid_aux_losses)) if valid_aux_losses else np.nan,
                valid_proto_loss=float(np.mean(valid_proto_losses)) if valid_proto_losses else np.nan,
            )
            current_valid = float(epoch_record.valid_loss)
            if normalized_selection_mode != "valid_loss" and checkpoint_selection_callback is not None:
                callback_payload = checkpoint_selection_callback(model, epoch) or {}
                epoch_record.selection_metric_name = str(
                    callback_payload.get("metric_name", normalized_selection_mode)
                )
                try:
                    epoch_record.selection_metric_value = float(
                        callback_payload.get("metric_value", float("nan"))
                    )
                except Exception:
                    epoch_record.selection_metric_value = float("nan")
            else:
                epoch_record.selection_metric_name = "valid_loss"
                epoch_record.selection_metric_value = current_valid
            history.append(epoch_record)
            last_state = deepcopy(model.state_dict())
            progress_write(
                f"epoch {epoch}/{epochs} | train={epoch_record.train_loss:.4f} "
                f"| valid={epoch_record.valid_loss:.4f} | rank={epoch_record.valid_rank_loss:.4f} "
                f"| listwise={epoch_record.valid_listwise_loss:.4f} | "
                f"{epoch_record.selection_metric_name}={epoch_record.selection_metric_value:.4f}"
            )
            if np.isfinite(current_valid):
                scheduler.step(current_valid)
                if current_valid < best_valid_loss - float(min_improvement):
                    best_valid_loss = current_valid
                improved = False
                if normalized_selection_mode == "valid_loss":
                    if current_valid < best_selection_metric - float(selection_improvement):
                        best_selection_metric = current_valid
                        improved = True
                else:
                    metric_value = float(epoch_record.selection_metric_value)
                    if np.isfinite(metric_value) and metric_value > best_selection_metric + float(selection_improvement):
                        best_selection_metric = metric_value
                        improved = True
                if improved:
                    best_state = deepcopy(model.state_dict())
                    selected_epoch = int(epoch)
                    selected_metric_name = str(epoch_record.selection_metric_name)
                    selected_metric_value = float(epoch_record.selection_metric_value)
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                if epoch >= max(int(min_epochs), 1) and no_improve_epochs >= max(int(early_stop_patience), 1):
                    stopped_early = True
                    break

    model.load_state_dict(best_state)
    diagnostics = _build_training_diagnostics(
        history=history,
        epochs_requested=epochs,
        stopped_early=stopped_early,
        learning_rate_final=float(optimizer.param_groups[0]["lr"]),
        min_improvement=float(min_improvement),
        selected_epoch=selected_epoch or None,
        checkpoint_selection_mode=normalized_selection_mode,
        selected_metric_name=selected_metric_name,
        selected_metric_value=selected_metric_value,
        checkpoint_selection_min_improvement=selection_improvement,
    )
    training_state = {
        "epochs_completed": int(len(history)),
        "best_valid_loss": float(best_valid_loss),
        "best_selection_metric": float(best_selection_metric),
        "selected_epoch": int(selected_epoch),
        "selected_metric_name": str(selected_metric_name),
        "selected_metric_value": float(selected_metric_value),
        "no_improve_epochs": int(no_improve_epochs),
        "checkpoint_selection_mode": str(normalized_selection_mode),
        "selection_improvement": float(selection_improvement),
        "stopped_early": bool(stopped_early),
        "train_sampler_state": {}
        if train_batch_sampler is None or not hasattr(train_batch_sampler, "state_dict")
        else dict(train_batch_sampler.state_dict()),
    }
    return TrainingRunResult(
        history=history,
        diagnostics=diagnostics,
        selected_model_state_dict=deepcopy(best_state),
        last_model_state_dict=deepcopy(last_state),
        optimizer_state_dict=deepcopy(optimizer.state_dict()),
        scheduler_state_dict=deepcopy(scheduler.state_dict()),
        scaler_state_dict=deepcopy(scaler.state_dict()),
        training_state=training_state,
    )


def infer_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_names: List[str],
    use_amp: bool = True,
    task_label: str = "Inference",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    amp_enabled = bool(use_amp and device.type == "cuda")
    pred_rows = []
    embed_rows = []
    total_batches = max(len(loader), 1)
    task_name = str(task_label or "Inference").strip() or "Inference"
    with create_progress(total=total_batches, desc=f"{task_name} setup", unit="batch", leave=False) as progress:
        with torch.no_grad():
            for batch_idx, (x, y, y_raw, state_id, liquidity_bucket, structure_id, dts, stocks) in enumerate(loader, start=1):
                progress.set_description_str(f"{task_name} batch {batch_idx}/{total_batches}")
                x = x.to(device, non_blocking=True)
                liquidity_bucket = liquidity_bucket.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    pred, emb, structure_logits = model(
                        x,
                        liquidity_bucket=liquidity_bucket,
                        state_id=state_id.to(device, non_blocking=True),
                        structure_id=structure_id.to(device, non_blocking=True),
                    )
                pred_np = pred.cpu().numpy()
                emb_np = emb.cpu().numpy()
                structure_prob_np = None
                structure_pred_np = None
                if structure_logits is not None:
                    structure_prob = torch.softmax(structure_logits, dim=1)
                    structure_prob_np = structure_prob.cpu().numpy()
                    structure_pred_np = structure_logits.argmax(dim=1).cpu().numpy()
                prototype_prob_np = None
                prototype_pred_np = None
                structure_similarity_fn = getattr(model, "structure_similarity", None)
                if callable(structure_similarity_fn):
                    proto_scores = model.structure_similarity(emb)
                    if proto_scores is not None:
                        proto_prob = torch.softmax(proto_scores, dim=1)
                        prototype_prob_np = proto_prob.cpu().numpy()
                        prototype_pred_np = proto_scores.argmax(dim=1).cpu().numpy()
                y_np = y.numpy()
                y_raw_np = y_raw.numpy()
                state_np = state_id.numpy()
                liquidity_np = liquidity_bucket.cpu().numpy()
                structure_np = structure_id.numpy()
                for i, (dt, stock) in enumerate(zip(dts, stocks)):
                    row = {"date": pd.Timestamp(dt), "stock": stock}
                    row["state_id"] = int(state_np[i]) if int(state_np[i]) >= 0 else np.nan
                    row["liquidity_bucket"] = int(liquidity_np[i]) if int(liquidity_np[i]) >= 0 else np.nan
                    row["structure_id"] = int(structure_np[i]) if int(structure_np[i]) >= 0 else np.nan
                    row["structure_label"] = STRUCTURE_ID_TO_LABEL.get(int(structure_np[i])) if int(structure_np[i]) >= 0 else ""
                    if structure_pred_np is not None and structure_prob_np is not None:
                        row["pred_structure_id"] = int(structure_pred_np[i])
                        row["pred_structure_label"] = STRUCTURE_ID_TO_LABEL.get(int(structure_pred_np[i]), "")
                        row["pred_structure_confidence"] = float(structure_prob_np[i, int(structure_pred_np[i])])
                    if prototype_pred_np is not None and prototype_prob_np is not None:
                        row["pred_prototype_structure_id"] = int(prototype_pred_np[i])
                        row["pred_prototype_structure_label"] = STRUCTURE_ID_TO_LABEL.get(int(prototype_pred_np[i]), "")
                        row["pred_prototype_structure_confidence"] = float(prototype_prob_np[i, int(prototype_pred_np[i])])
                    for j, name in enumerate(target_names):
                        row[f"pred_{name}"] = float(pred_np[i, j])
                        row[f"label_{name}"] = float(y_np[i, j])
                        row[f"true_{name}"] = float(y_raw_np[i, j])
                    pred_rows.append(row)
                    emb_row = {"date": pd.Timestamp(dt), "stock": stock}
                    emb_row["liquidity_bucket"] = int(liquidity_np[i]) if int(liquidity_np[i]) >= 0 else np.nan
                    for j in range(emb_np.shape[1]):
                        emb_row[f"emb_{j}"] = float(emb_np[i, j])
                    embed_rows.append(emb_row)
                progress.update(1)
    return pd.DataFrame(pred_rows), pd.DataFrame(embed_rows)


def train_masked_pretrainer(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    device: torch.device,
    use_amp: bool = True,
    min_epochs: int = 8,
    early_stop_patience: int = 3,
    lr_plateau_patience: int = 2,
    lr_plateau_factor: float = 0.5,
    min_improvement: float = 1e-4,
    auto_extend_undertrained: bool = False,
    epoch_extend_step: int = 4,
    max_total_epochs: int | None = None,
) -> PretrainRunResult:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(lr_plateau_factor),
        patience=max(int(lr_plateau_patience), 0),
        threshold=float(min_improvement),
        threshold_mode="abs",
        min_lr=max(float(learning_rate) * 0.1, 1e-6),
    )
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)
    history: List[PretrainEpochRecord] = []
    best_valid_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    no_improve_epochs = 0
    stopped_early = False
    initial_budget = max(int(epochs), 1)
    epoch_budget = int(initial_budget)
    max_epoch_budget = max(int(max_total_epochs or epoch_budget), epoch_budget)
    extend_step = max(int(epoch_extend_step), 1)
    epoch = 0
    train_batch_count = max(len(train_loader), 1)
    valid_batch_count = max(len(valid_loader), 1)
    while epoch < epoch_budget:
        epoch_batch_total = train_batch_count + valid_batch_count
        with create_progress(
            total=epoch_batch_total,
            desc=f"Pretrain epoch {epoch + 1}/{epoch_budget}",
            unit="batch",
            leave=False,
        ) as progress:
            epoch += 1
            model.train()
            train_losses: List[float] = []
            train_mask_ratios: List[float] = []
            for batch_idx, (x, _dts, _stocks) in enumerate(train_loader, start=1):
                progress.set_description_str(
                    _format_pretrain_step_desc(
                        phase_label="train",
                        epoch=epoch,
                        batch_idx=batch_idx,
                        batch_total=train_batch_count,
                        epoch_budget=epoch_budget,
                        max_epoch_budget=max_epoch_budget,
                    )
                )
                x = x.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    loss, stats = model(x)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.item()))
                train_mask_ratios.append(float(stats.get("mask_ratio", 0.0)))
                progress.update(1)

            model.eval()
            valid_losses: List[float] = []
            valid_mask_ratios: List[float] = []
            with torch.no_grad():
                for batch_idx, (x, _dts, _stocks) in enumerate(valid_loader, start=1):
                    progress.set_description_str(
                        _format_pretrain_step_desc(
                            phase_label="valid",
                            epoch=epoch,
                            batch_idx=batch_idx,
                            batch_total=valid_batch_count,
                            epoch_budget=epoch_budget,
                            max_epoch_budget=max_epoch_budget,
                        )
                    )
                    x = x.to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                        loss, stats = model(x)
                    valid_losses.append(float(loss.item()))
                    valid_mask_ratios.append(float(stats.get("mask_ratio", 0.0)))
                    progress.update(1)

            epoch_record = PretrainEpochRecord(
                epoch=epoch,
                train_loss=float(np.mean(train_losses)) if train_losses else np.nan,
                valid_loss=float(np.mean(valid_losses)) if valid_losses else np.nan,
                train_mask_ratio=float(np.mean(train_mask_ratios)) if train_mask_ratios else np.nan,
                valid_mask_ratio=float(np.mean(valid_mask_ratios)) if valid_mask_ratios else np.nan,
            )
            history.append(epoch_record)
            progress_write(
                f"pretrain epoch {epoch}/{epoch_budget} | train={epoch_record.train_loss:.4f} "
                f"| valid={epoch_record.valid_loss:.4f} | mask={epoch_record.valid_mask_ratio:.3f}"
            )
            current_valid = float(epoch_record.valid_loss)
            if np.isfinite(current_valid):
                scheduler.step(current_valid)
                if current_valid < best_valid_loss - float(min_improvement):
                    best_valid_loss = current_valid
                    best_state = deepcopy(model.state_dict())
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                if epoch >= max(int(min_epochs), 1) and no_improve_epochs >= max(int(early_stop_patience), 1):
                    stopped_early = True
                    break

            if stopped_early:
                break

            if epoch >= epoch_budget and bool(auto_extend_undertrained) and epoch_budget < max_epoch_budget:
                diagnostics_preview = _build_training_diagnostics(
                    history=history,
                    epochs_requested=epoch_budget,
                    stopped_early=False,
                    learning_rate_final=float(optimizer.param_groups[0]["lr"]),
                    min_improvement=float(min_improvement),
                )
                if diagnostics_preview.status == "undertrained":
                    new_budget = min(max_epoch_budget, epoch_budget + extend_step)
                    if new_budget > epoch_budget:
                        progress_write(
                            f"Auto-extending pretraining budget: {epoch_budget} -> {new_budget} "
                            f"(best_epoch={diagnostics_preview.best_epoch}, best_valid_loss={diagnostics_preview.best_valid_loss:.6f})"
                        )
                        epoch_budget = new_budget

    model.load_state_dict(best_state)
    diagnostics = _build_training_diagnostics(
        history=history,
        epochs_requested=epoch_budget,
        stopped_early=stopped_early,
        learning_rate_final=float(optimizer.param_groups[0]["lr"]),
        min_improvement=float(min_improvement),
    )
    return PretrainRunResult(history=history, diagnostics=diagnostics)


def compute_rankic_timeseries(pred_df: pd.DataFrame, target_names: List[str]) -> pd.DataFrame:
    rows = []
    for target_name in target_names:
        pred_col = f"pred_{target_name}"
        true_col = f"true_{target_name}"
        for dt, g in pred_df.groupby("date"):
            if len(g) < 5:
                continue
            ic = spearmanr(g[pred_col], g[true_col], nan_policy="omit").correlation
            rows.append({"date": dt, "target": target_name, "rankic": float(ic) if ic is not None else np.nan})
    return pd.DataFrame(rows, columns=["date", "target", "rankic"])


def compute_rankic(pred_df: pd.DataFrame, target_names: List[str]) -> pd.DataFrame:
    out = compute_rankic_timeseries(pred_df, target_names)
    if out.empty:
        return pd.DataFrame(columns=["target", "rankic_mean", "rankic_std", "rankic_ir"])
    summary = out.groupby("target")["rankic"].agg(["mean", "std"]).reset_index()
    summary["rankic_ir"] = summary["mean"] / summary["std"].replace(0, np.nan)
    return summary.rename(columns={"mean": "rankic_mean", "std": "rankic_std"})
