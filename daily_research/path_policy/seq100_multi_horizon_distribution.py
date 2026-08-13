"""D5/D10 probabilistic sequence challenger with an explicit portfolio boundary.

The study keeps the existing causal source pack, forward folds, legal outcome
contract, and account simulator.  It changes only the sequence feature view and
forecast heads.  Every validation signal row is scored before future fill or
outcome status is consulted.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from daily_research.path_policy import seq100_full_market_multitask_forecast as base
from daily_research.path_policy import seq100_full_market_sequence_challenger as seq

SCHEMA = "seq100_multi_horizon_distribution/1"
AUDIT_SCHEMA = "seq100_multi_horizon_distribution_audit/1"
EVALUATION_SCHEMA = "seq100_multi_horizon_distribution_evaluation/1"
ACCOUNT_SCHEMA = "seq100_multi_horizon_distribution_account_replay/1"
COMPARISON_SCHEMA = "seq100_multi_horizon_distribution_comparison/1"
DEFAULT_STUDY_PATH = (
    base.WORKSPACE_ROOT
    / "daily_research/studies/seq100_multi_horizon_distribution_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    base.WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_multi_horizon_distribution_v1"
)
MODEL_VARIANTS = ("single_d10_point", "multi_d5_d10_distribution")


class MultiHorizonError(RuntimeError):
    """Raised when the frozen study contract is violated."""


def _now() -> str:
    return pd.Timestamp.now(tz="Asia/Shanghai").isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(partial, path)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _file_record(path: Path, **extra: Any) -> dict[str, Any]:
    record = base._file_record(path)
    record.update(extra)
    return record


def _chunks(values: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(values), int(size)):
        yield values[start : start + int(size)]


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = json.loads(path.read_text(encoding="utf-8"))
    if study.get("study_id") != "seq100_multi_horizon_distribution_v1":
        raise MultiHorizonError("study_id_changed")
    if list(study["targets"]["primary_horizons"]) != [5, 10]:
        raise MultiHorizonError("primary_horizons_changed")
    if list(study["targets"]["deferred_auxiliary_horizons"]) != [2, 3]:
        raise MultiHorizonError("deferred_horizons_changed")
    if study["features"]["variant"] != "price_path_core_183":
        raise MultiHorizonError("feature_variant_changed")
    if int(study["features"]["static_feature_count"]) != 183:
        raise MultiHorizonError("feature_count_changed")
    if int(study["features"]["lookback"]) != 16:
        raise MultiHorizonError("lookback_changed")
    if list(study["targets"]["quantiles"]) != [0.1, 0.5, 0.9]:
        raise MultiHorizonError("quantiles_changed")
    if not bool(study["population"]["rank_before_future_fill_or_outcome_filter"]):
        raise MultiHorizonError("full_slate_ranking_disabled")
    if not bool(study["epistemic_contract"]["no_2026_outcome_may_be_read"]):
        raise MultiHorizonError("forbidden_outcome_boundary_changed")
    if bool(study["validation"]["D2_D3_training_allowed_in_v1"]):
        raise MultiHorizonError("deferred_horizon_training_enabled")
    if tuple(study["model_variants"]) != MODEL_VARIANTS:
        raise MultiHorizonError("model_variants_changed")
    return study


def _workspace_path(value: str) -> Path:
    return (base.WORKSPACE_ROOT / value).resolve()


@dataclass(frozen=True)
class TargetNormalization:
    center: dict[int, float]
    scale: dict[int, float]
    valid_count: dict[int, int]

    def payload(self) -> dict[str, Any]:
        return {
            "center": {str(key): float(value) for key, value in self.center.items()},
            "scale": {str(key): float(value) for key, value in self.scale.items()},
            "valid_count": {
                str(key): int(value) for key, value in self.valid_count.items()
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TargetNormalization:
        return cls(
            center={int(key): float(value) for key, value in payload["center"].items()},
            scale={int(key): float(value) for key, value in payload["scale"].items()},
            valid_count={
                int(key): int(value) for key, value in payload["valid_count"].items()
            },
        )


@dataclass
class MultiHorizonSources:
    study: dict[str, Any]
    base_study: dict[str, Any]
    base_output_root: Path
    by_horizon: dict[int, seq.SequenceSources]
    feature_names: tuple[str, ...]
    feature_columns: np.ndarray
    cutoff_date_idx: int

    @property
    def primary(self) -> seq.SequenceSources:
        return self.by_horizon[10]


def _load_sources(study: Mapping[str, Any]) -> MultiHorizonSources:
    base_study_path = _workspace_path(str(study["sources"]["base_study"]))
    base_output_root = _workspace_path(str(study["sources"]["base_output_root"]))
    base_study = base.load_study(base_study_path)
    by_horizon = {
        horizon: seq._load_sources(
            base_study, output_root=base_output_root, horizon=horizon
        )
        for horizon in (5, 10)
    }
    primary = by_horizon[10]
    for horizon, current in by_horizon.items():
        if (
            current.context.model_manifest["input_fingerprint"]
            != primary.context.model_manifest["input_fingerprint"]
            or current.target_manifest["fingerprint"]
            != primary.target_manifest["fingerprint"]
            or len(current.context.row_index) != len(primary.context.row_index)
        ):
            raise MultiHorizonError(f"source_alignment_failed:{horizon}")
    feature_names, feature_columns, _ = base._feature_variant_contract(
        base_study,
        primary.context.model_manifest,
        str(study["features"]["variant"]),
    )
    if len(feature_names) != int(study["features"]["static_feature_count"]):
        raise MultiHorizonError("resolved_feature_count_changed")
    target_contract = dict(primary.path_manifest["contract"])
    cutoff_date_idx = int(target_contract["maximum_source_date_idx_read"])
    if (
        str(target_contract["maximum_outcome_date"])
        != str(study["sources"]["maximum_outcome_date"])
        or int(target_contract["forbidden_2026_read_count"]) != 0
    ):
        raise MultiHorizonError("target_outcome_boundary_changed")
    return MultiHorizonSources(
        study=dict(study),
        base_study=base_study,
        base_output_root=base_output_root,
        by_horizon=by_horizon,
        feature_names=feature_names,
        feature_columns=feature_columns,
        cutoff_date_idx=cutoff_date_idx,
    )


def _variant_contract(
    study: Mapping[str, Any], variant: str
) -> tuple[tuple[int, ...], bool]:
    if variant not in MODEL_VARIANTS:
        raise MultiHorizonError(f"unknown_model_variant:{variant}")
    contract = dict(study["model_variants"][variant])
    horizons = tuple(int(value) for value in contract["horizons"])
    distribution = bool(contract["distribution_heads"])
    if variant == "single_d10_point" and (horizons != (10,) or distribution):
        raise MultiHorizonError("single_d10_contract_changed")
    if variant == "multi_d5_d10_distribution" and (
        horizons != (5, 10) or not distribution
    ):
        raise MultiHorizonError("multi_distribution_contract_changed")
    return horizons, distribution


def _forward_folds(sources: MultiHorizonSources) -> list[dict[str, Any]]:
    validation = sources.base_study["validation"]
    row_index = sources.primary.context.row_index
    return base.build_forward_folds(
        date_idx=row_index["date_idx"].to_numpy(dtype=np.int32),
        trade_date=row_index["trade_date"].astype(str).to_numpy(),
        validation_start_date=str(validation["validation_start_date"]),
        validation_end_date=str(validation["validation_end_date"]),
        fold_count=int(validation["forward_fold_count"]),
        purge_days=int(validation["common_purge_trading_days"]),
    )


def _fold_dates(
    sources: MultiHorizonSources, fold: Mapping[str, Any]
) -> tuple[list[int], list[int]]:
    blocks = sources.primary.blocks
    train = sorted(
        value for value in blocks if value <= int(fold["training_maximum_date_idx"])
    )
    validation = sorted(
        value
        for value in blocks
        if int(fold["validation_start_date_idx"])
        <= value
        <= int(fold["validation_end_date_idx"])
    )
    lookback = int(sources.study["features"]["lookback"])
    train = [value for value in train if value >= lookback - 1]
    validation = [value for value in validation if value >= lookback - 1]
    if not train or not validation:
        raise MultiHorizonError("fold_dates_empty")
    return train, validation


def _fit_target_normalization(
    sources: MultiHorizonSources,
    train_dates: Sequence[int],
    horizons: Sequence[int],
) -> TargetNormalization:
    centers: dict[int, float] = {}
    scales: dict[int, float] = {}
    counts: dict[int, int] = {}
    for horizon in horizons:
        source = sources.by_horizon[int(horizon)]
        parts: list[np.ndarray] = []
        for date_idx in train_dates:
            block = source.blocks[int(date_idx)]
            rows = slice(block.start, block.stop)
            valid = np.asarray(source.exact_valid[rows, source.base_column], dtype=bool)
            if valid.any():
                values = np.asarray(
                    source.exact_values[rows, source.base_column], dtype=np.float32
                )
                parts.append(values[valid & np.isfinite(values)])
        if not parts:
            raise MultiHorizonError(f"target_normalization_empty:{horizon}")
        values = np.concatenate(parts).astype(np.float64, copy=False)
        center = float(np.median(values))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = float(max(q75 - q25, 0.01))
        centers[int(horizon)] = center
        scales[int(horizon)] = scale
        counts[int(horizon)] = len(values)
        del parts, values
    return TargetNormalization(center=centers, scale=scales, valid_count=counts)


@dataclass
class MultiHorizonBatch:
    static: torch.Tensor
    sequence: torch.Tensor
    market: torch.Tensor
    returns: torch.Tensor
    return_valid: torch.Tensor
    entry_fill: torch.Tensor
    entry_valid: torch.Tensor
    exit_block: torch.Tensor
    exit_valid: torch.Tensor
    group_sizes: tuple[int, ...]
    row_positions: np.ndarray
    date_indices: tuple[int, ...]


class MultiHorizonBatchAssembler:
    def __init__(
        self,
        sources: MultiHorizonSources,
        *,
        horizons: Sequence[int],
        sequence_normalization: seq.Normalization,
        target_normalization: TargetNormalization,
        device: torch.device,
    ) -> None:
        self.sources = sources
        self.horizons = tuple(int(value) for value in horizons)
        self.sequence_normalization = sequence_normalization
        self.target_normalization = target_normalization
        self.device = device
        self.lookback = int(sources.study["features"]["lookback"])
        self.sequence_helper = seq.BatchAssembler(
            sources.primary,
            lookback=self.lookback,
            normalization=sequence_normalization,
            device=device,
            require_targets=False,
        )

    def load(self, date_indices: Sequence[int]) -> MultiHorizonBatch | None:
        static_parts: list[np.ndarray] = []
        sequence_parts: list[np.ndarray] = []
        return_parts: list[np.ndarray] = []
        return_valid_parts: list[np.ndarray] = []
        entry_parts: list[np.ndarray] = []
        entry_valid_parts: list[np.ndarray] = []
        exit_parts: list[np.ndarray] = []
        exit_valid_parts: list[np.ndarray] = []
        market_parts: list[np.ndarray] = []
        position_parts: list[np.ndarray] = []
        group_sizes: list[int] = []
        accepted_dates: list[int] = []
        primary = self.sources.primary
        for date_idx in date_indices:
            block = primary.blocks[int(date_idx)]
            rows = np.arange(block.start, block.stop, dtype=np.int64)
            if len(rows) < 2:
                continue
            symbols = primary.context.row_index.iloc[rows]["symbol_idx"].to_numpy(
                dtype=np.int32
            )
            static_parts.append(
                np.asarray(
                    primary.matrix[
                        rows[:, None], self.sources.feature_columns[None, :]
                    ],
                    dtype=np.float32,
                )
            )
            sequence_parts.append(
                self.sequence_helper._sequence(int(date_idx), symbols)
            )
            current_returns = np.zeros(
                (len(rows), len(self.horizons)), dtype=np.float32
            )
            current_return_valid = np.zeros_like(current_returns, dtype=bool)
            current_exit = np.zeros_like(current_returns, dtype=np.float32)
            current_exit_valid = np.zeros_like(current_returns, dtype=bool)
            for column, horizon in enumerate(self.horizons):
                source = self.sources.by_horizon[horizon]
                values = np.asarray(
                    source.exact_values[rows, source.base_column], dtype=np.float32
                )
                valid = np.asarray(
                    source.exact_valid[rows, source.base_column], dtype=bool
                ) & np.isfinite(values)
                current_returns[:, column] = np.where(
                    valid,
                    (values - self.target_normalization.center[horizon])
                    / self.target_normalization.scale[horizon],
                    0.0,
                )
                current_return_valid[:, column] = valid

                path_valid = np.asarray(
                    source.path_valid[rows, source.gross_column], dtype=bool
                )
                fill_days = np.asarray(
                    source.fill_days[rows, source.horizon_column], dtype=np.int16
                )
                market_entry = np.asarray(
                    source.entry_filled[int(date_idx), symbols], dtype=bool
                )
                exit_valid = market_entry & path_valid & (fill_days >= horizon)
                current_exit[:, column] = np.where(
                    exit_valid, fill_days > horizon, False
                ).astype(np.float32)
                current_exit_valid[:, column] = exit_valid
            entry_known = int(date_idx) + 1 <= self.sources.cutoff_date_idx
            entry = np.asarray(
                primary.entry_filled[int(date_idx), symbols], dtype=np.float32
            )
            entry_valid = np.full(len(rows), entry_known, dtype=bool)
            market_raw = np.asarray(
                primary.matrix[block.start, primary.market_columns], dtype=np.float32
            )
            market_parts.append(
                np.nan_to_num(
                    (market_raw - self.sequence_normalization.market_center)
                    / self.sequence_normalization.market_scale,
                    nan=0.0,
                    posinf=10.0,
                    neginf=-10.0,
                ).clip(-10.0, 10.0)
            )
            return_parts.append(current_returns)
            return_valid_parts.append(current_return_valid)
            entry_parts.append(entry)
            entry_valid_parts.append(entry_valid)
            exit_parts.append(current_exit)
            exit_valid_parts.append(current_exit_valid)
            position_parts.append(rows)
            group_sizes.append(len(rows))
            accepted_dates.append(int(date_idx))
        if not group_sizes:
            return None
        return MultiHorizonBatch(
            static=torch.from_numpy(np.concatenate(static_parts)).to(self.device),
            sequence=torch.from_numpy(np.concatenate(sequence_parts)).to(self.device),
            market=torch.from_numpy(np.stack(market_parts)).to(self.device),
            returns=torch.from_numpy(np.concatenate(return_parts)).to(self.device),
            return_valid=torch.from_numpy(np.concatenate(return_valid_parts)).to(
                self.device
            ),
            entry_fill=torch.from_numpy(np.concatenate(entry_parts)).to(self.device),
            entry_valid=torch.from_numpy(np.concatenate(entry_valid_parts)).to(
                self.device
            ),
            exit_block=torch.from_numpy(np.concatenate(exit_parts)).to(self.device),
            exit_valid=torch.from_numpy(np.concatenate(exit_valid_parts)).to(
                self.device
            ),
            group_sizes=tuple(group_sizes),
            row_positions=np.concatenate(position_parts),
            date_indices=tuple(accepted_dates),
        )


def monotone_quantiles(
    lower: torch.Tensor, positive_deltas: torch.Tensor
) -> torch.Tensor:
    """Construct q10/q50/q90 without quantile crossing."""

    if positive_deltas.ndim != 2 or positive_deltas.shape[1] != 2:
        raise MultiHorizonError("quantile_delta_shape_invalid")
    deltas = F.softplus(positive_deltas)
    median = lower + deltas[:, 0]
    upper = median + deltas[:, 1]
    return torch.stack([lower, median, upper], dim=1)


@dataclass
class ModelOutput:
    point: dict[int, torch.Tensor]
    quantiles: dict[int, torch.Tensor]
    entry_logit: torch.Tensor
    exit_logit: dict[int, torch.Tensor]
    market_return: torch.Tensor
    market_logit: torch.Tensor


class MultiHorizonSequenceModel(nn.Module):
    def __init__(
        self,
        *,
        horizons: Sequence[int],
        distribution_heads: bool,
        static_dim: int = 183,
        sequence_dim: int = 32,
        market_dim: int = 54,
        hidden_dim: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.horizons = tuple(int(value) for value in horizons)
        self.distribution_heads = bool(distribution_heads)
        self.static = nn.Sequential(
            nn.Linear(static_dim, 96),
            nn.GELU(),
            nn.LayerNorm(96),
            nn.Dropout(dropout),
        )
        self.missing = nn.Sequential(nn.Linear(static_dim, 16), nn.GELU())
        self.sequence_projection = nn.Sequential(
            nn.Linear(sequence_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.sequence_encoder = nn.GRU(
            hidden_dim, hidden_dim, batch_first=True, num_layers=1
        )
        self.market = nn.Sequential(
            nn.Linear(market_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Dropout(dropout),
        )
        self.stock_trunk = nn.Sequential(
            nn.Linear(96 + 16 + hidden_dim + 32, 96),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.point_heads = nn.ModuleDict(
            {str(horizon): nn.Linear(96, 1) for horizon in self.horizons}
        )
        self.exit_heads = nn.ModuleDict(
            {str(horizon): nn.Linear(96, 1) for horizon in self.horizons}
        )
        self.entry_head = nn.Linear(96, 1)
        self.quantile_lower = nn.ModuleDict()
        self.quantile_deltas = nn.ModuleDict()
        if self.distribution_heads:
            delta_bias = float(math.log(math.expm1(0.5)))
            for horizon in self.horizons:
                lower = nn.Linear(96, 1)
                deltas = nn.Linear(96, 2)
                nn.init.constant_(lower.bias, -0.5)
                nn.init.constant_(deltas.bias, delta_bias)
                self.quantile_lower[str(horizon)] = lower
                self.quantile_deltas[str(horizon)] = deltas
        self.market_return_head = nn.Linear(32, 1)
        self.market_positive_head = nn.Linear(32, 1)

    def forward(
        self,
        static: torch.Tensor,
        missing: torch.Tensor,
        sequence: torch.Tensor,
        market: torch.Tensor,
        group_sizes: Sequence[int],
    ) -> ModelOutput:
        static_embedding = self.static(static)
        missing_embedding = self.missing(missing)
        projected = self.sequence_projection(sequence)
        _, hidden = self.sequence_encoder(projected)
        sequence_embedding = hidden[-1]
        market_embedding = self.market(market)
        repeated_market = torch.repeat_interleave(
            market_embedding,
            torch.as_tensor(group_sizes, device=market.device),
            dim=0,
        )
        stock = self.stock_trunk(
            torch.cat(
                [
                    static_embedding,
                    missing_embedding,
                    sequence_embedding,
                    repeated_market,
                ],
                dim=1,
            )
        )
        point = {
            horizon: self.point_heads[str(horizon)](stock).squeeze(1)
            for horizon in self.horizons
        }
        quantiles = (
            {
                horizon: monotone_quantiles(
                    self.quantile_lower[str(horizon)](stock).squeeze(1),
                    self.quantile_deltas[str(horizon)](stock),
                )
                for horizon in self.horizons
            }
            if self.distribution_heads
            else {}
        )
        return ModelOutput(
            point=point,
            quantiles=quantiles,
            entry_logit=self.entry_head(stock).squeeze(1),
            exit_logit={
                horizon: self.exit_heads[str(horizon)](stock).squeeze(1)
                for horizon in self.horizons
            },
            market_return=self.market_return_head(market_embedding).squeeze(1),
            market_logit=self.market_positive_head(market_embedding).squeeze(1),
        )


def pinball_loss(
    prediction: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    error = target[:, None] - prediction
    return torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean()


def _correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    group_sizes: Sequence[int],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    offset = 0
    for size in group_sizes:
        current = slice(offset, offset + int(size))
        mask = valid[current]
        if int(mask.sum()) >= 2:
            score = prediction[current][mask]
            actual = target[current][mask]
            score = score - score.mean()
            actual = actual - actual.mean()
            denominator = (
                score.square().sum().clamp_min(1.0e-8).sqrt()
                * actual.square().sum().clamp_min(1.0e-8).sqrt()
            )
            losses.append(1.0 - (score * actual).sum() / denominator)
        offset += int(size)
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def payoff_distribution_loss(
    output: ModelOutput,
    batch: MultiHorizonBatch,
    *,
    horizons: Sequence[int],
    distribution_heads: bool,
    weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    rank_losses: list[torch.Tensor] = []
    huber_losses: list[torch.Tensor] = []
    quantile_losses: list[torch.Tensor] = []
    exit_losses: list[torch.Tensor] = []
    quantile_tensor = torch.as_tensor(
        [0.1, 0.5, 0.9], device=batch.returns.device, dtype=batch.returns.dtype
    )
    for column, horizon in enumerate(horizons):
        valid = batch.return_valid[:, column]
        rank_losses.append(
            _correlation_loss(
                output.point[int(horizon)],
                batch.returns[:, column],
                valid,
                batch.group_sizes,
            )
        )
        huber_losses.append(
            F.smooth_l1_loss(
                output.point[int(horizon)][valid], batch.returns[:, column][valid]
            )
            if bool(valid.any())
            else output.point[int(horizon)].sum() * 0.0
        )
        if distribution_heads:
            quantile_losses.append(
                pinball_loss(
                    output.quantiles[int(horizon)][valid],
                    batch.returns[:, column][valid],
                    quantile_tensor,
                )
                if bool(valid.any())
                else output.quantiles[int(horizon)].sum() * 0.0
            )
        exit_valid = batch.exit_valid[:, column]
        exit_losses.append(
            F.binary_cross_entropy_with_logits(
                output.exit_logit[int(horizon)][exit_valid],
                batch.exit_block[:, column][exit_valid],
            )
            if bool(exit_valid.any())
            else output.exit_logit[int(horizon)].sum() * 0.0
        )
    rank_loss = torch.stack(rank_losses).mean()
    huber_loss = torch.stack(huber_losses).mean()
    quantile_loss_value = (
        torch.stack(quantile_losses).mean()
        if quantile_losses
        else rank_loss.detach() * 0.0
    )
    exit_loss = torch.stack(exit_losses).mean()
    entry_loss = (
        F.binary_cross_entropy_with_logits(
            output.entry_logit[batch.entry_valid], batch.entry_fill[batch.entry_valid]
        )
        if bool(batch.entry_valid.any())
        else output.entry_logit.sum() * 0.0
    )

    d10_column = tuple(int(value) for value in horizons).index(10)
    market_targets: list[torch.Tensor] = []
    market_valid: list[bool] = []
    offset = 0
    for size in batch.group_sizes:
        current = slice(offset, offset + int(size))
        valid = batch.return_valid[current, d10_column]
        market_valid.append(bool(valid.any()))
        market_targets.append(
            batch.returns[current, d10_column][valid].mean()
            if bool(valid.any())
            else batch.returns[current, d10_column].sum() * 0.0
        )
        offset += int(size)
    market_target = torch.stack(market_targets)
    market_mask = torch.as_tensor(market_valid, device=market_target.device)
    market_huber = (
        F.smooth_l1_loss(output.market_return[market_mask], market_target[market_mask])
        if bool(market_mask.any())
        else output.market_return.sum() * 0.0
    )
    market_binary = (
        F.binary_cross_entropy_with_logits(
            output.market_logit[market_mask],
            (market_target[market_mask] > 0.0).to(output.market_logit.dtype),
        )
        if bool(market_mask.any())
        else output.market_logit.sum() * 0.0
    )
    total = (
        float(weights["daily_cross_sectional_correlation"]) * rank_loss
        + float(weights["absolute_huber"]) * huber_loss
        + float(weights["quantile_pinball"]) * quantile_loss_value
        + float(weights["d10_market_huber"]) * market_huber
        + float(weights["d10_market_binary"]) * market_binary
        + float(weights["entry_fill_binary"]) * entry_loss
        + float(weights["exit_block_binary"]) * exit_loss
    )
    return total, {
        "rank_loss": float(rank_loss.detach().cpu()),
        "absolute_huber_loss": float(huber_loss.detach().cpu()),
        "quantile_pinball_loss": float(quantile_loss_value.detach().cpu()),
        "market_huber_loss": float(market_huber.detach().cpu()),
        "market_binary_loss": float(market_binary.detach().cpu()),
        "entry_fill_binary_loss": float(entry_loss.detach().cpu()),
        "exit_block_binary_loss": float(exit_loss.detach().cpu()),
    }


@dataclass
class PredictionOutput:
    rows: np.ndarray
    point: dict[int, np.ndarray]
    quantiles: dict[int, np.ndarray]
    entry_probability: np.ndarray
    exit_probability: dict[int, np.ndarray]


@torch.no_grad()
def _predict_dates(
    model: MultiHorizonSequenceModel,
    assembler: MultiHorizonBatchAssembler,
    date_indices: Sequence[int],
    *,
    date_batch_size: int,
) -> PredictionOutput:
    model.eval()
    row_parts: list[np.ndarray] = []
    point_parts = {horizon: [] for horizon in model.horizons}
    quantile_parts = {horizon: [] for horizon in model.horizons}
    exit_parts = {horizon: [] for horizon in model.horizons}
    entry_parts: list[np.ndarray] = []
    for dates in _chunks([int(value) for value in date_indices], date_batch_size):
        batch = assembler.load(dates)
        if batch is None:
            continue
        with torch.amp.autocast(
            device_type=assembler.device.type, enabled=assembler.device.type == "cuda"
        ):
            static, missing = seq.cross_sectional_standardize(
                batch.static, batch.group_sizes
            )
            output = model(
                static, missing, batch.sequence, batch.market, batch.group_sizes
            )
        row_parts.append(batch.row_positions)
        entry_parts.append(torch.sigmoid(output.entry_logit).float().cpu().numpy())
        for horizon in model.horizons:
            center = assembler.target_normalization.center[horizon]
            scale = assembler.target_normalization.scale[horizon]
            point_parts[horizon].append(
                output.point[horizon].float().cpu().numpy() * scale + center
            )
            exit_parts[horizon].append(
                torch.sigmoid(output.exit_logit[horizon]).float().cpu().numpy()
            )
            if model.distribution_heads:
                quantile_parts[horizon].append(
                    output.quantiles[horizon].float().cpu().numpy() * scale + center
                )
        del batch, static, missing, output
    if not row_parts:
        raise MultiHorizonError("prediction_dates_empty")
    return PredictionOutput(
        rows=np.concatenate(row_parts),
        point={key: np.concatenate(value) for key, value in point_parts.items()},
        quantiles={
            key: np.concatenate(value) for key, value in quantile_parts.items() if value
        },
        entry_probability=np.concatenate(entry_parts),
        exit_probability={
            key: np.concatenate(value) for key, value in exit_parts.items()
        },
    )


def _date_equal_binary_metrics(
    dates: np.ndarray,
    actual: np.ndarray,
    probability: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    brier: list[float] = []
    observed: list[float] = []
    predicted: list[float] = []
    for date_idx in np.unique(dates):
        mask = (dates == date_idx) & valid
        if not mask.any():
            continue
        brier.append(float(np.square(probability[mask] - actual[mask]).mean()))
        observed.append(float(actual[mask].mean()))
        predicted.append(float(probability[mask].mean()))
    return {
        "date_count": len(brier),
        "date_equal_brier": float(np.mean(brier)) if brier else math.nan,
        "mean_observed_rate": float(np.mean(observed)) if observed else math.nan,
        "mean_predicted_rate": float(np.mean(predicted)) if predicted else math.nan,
    }


def _prediction_metrics(
    sources: MultiHorizonSources,
    prediction: PredictionOutput,
    horizons: Sequence[int],
) -> dict[str, Any]:
    row_index = sources.primary.context.row_index.iloc[prediction.rows]
    dates = row_index["date_idx"].to_numpy(dtype=np.int32)
    symbols = row_index["symbol_idx"].to_numpy(dtype=np.int32)
    result: dict[str, Any] = {"horizons": {}}
    for horizon in horizons:
        source = sources.by_horizon[int(horizon)]
        actual = np.asarray(
            source.exact_values[prediction.rows, source.base_column], dtype=np.float64
        )
        valid = np.asarray(
            source.exact_valid[prediction.rows, source.base_column], dtype=bool
        ) & np.isfinite(actual)
        rank_values: list[float] = []
        mae_values: list[float] = []
        pinball_values = {0.1: [], 0.5: [], 0.9: []}
        coverage_values = {0.1: [], 0.5: [], 0.9: []}
        widths: list[float] = []
        for date_idx in np.unique(dates):
            mask = (dates == date_idx) & valid
            if int(mask.sum()) < 2:
                continue
            rank_values.append(
                seq._rank_correlation(
                    actual[mask], prediction.point[int(horizon)][mask]
                )
            )
            mae_values.append(
                float(
                    np.abs(actual[mask] - prediction.point[int(horizon)][mask]).mean()
                )
            )
            if int(horizon) in prediction.quantiles:
                current = prediction.quantiles[int(horizon)][mask]
                for column, quantile in enumerate((0.1, 0.5, 0.9)):
                    error = actual[mask] - current[:, column]
                    pinball_values[quantile].append(
                        float(
                            np.maximum(
                                quantile * error, (quantile - 1.0) * error
                            ).mean()
                        )
                    )
                    coverage_values[quantile].append(
                        float((actual[mask] <= current[:, column]).mean())
                    )
                widths.append(float((current[:, 2] - current[:, 0]).mean()))
        horizon_result: dict[str, Any] = {
            "valid_row_count": int(valid.sum()),
            "date_count": len(rank_values),
            "daily_rank_ic_mean": float(np.mean(rank_values)),
            "daily_rank_ic_positive_fraction": float(
                np.mean(np.asarray(rank_values) > 0.0)
            ),
            "date_equal_point_mae": float(np.mean(mae_values)),
        }
        if int(horizon) in prediction.quantiles:
            crossing = np.any(
                np.diff(prediction.quantiles[int(horizon)], axis=1) < -1.0e-7,
                axis=1,
            )
            horizon_result["quantiles"] = {
                "date_equal_pinball": {
                    str(key): float(np.mean(value))
                    for key, value in pinball_values.items()
                },
                "date_equal_coverage": {
                    str(key): float(np.mean(value))
                    for key, value in coverage_values.items()
                },
                "mean_q90_minus_q10": float(np.mean(widths)),
                "crossing_row_count": int(crossing.sum()),
            }
        exit_valid = np.asarray(
            source.entry_filled[dates, symbols], dtype=bool
        ) & np.asarray(
            source.path_valid[prediction.rows, source.gross_column], dtype=bool
        )
        fill_days = np.asarray(
            source.fill_days[prediction.rows, source.horizon_column], dtype=np.int16
        )
        exit_valid &= fill_days >= int(horizon)
        exit_actual = (fill_days > int(horizon)).astype(np.float64)
        horizon_result["exit_block"] = _date_equal_binary_metrics(
            dates,
            exit_actual,
            prediction.exit_probability[int(horizon)],
            exit_valid,
        )
        result["horizons"][str(horizon)] = horizon_result
    entry_valid = dates + 1 <= sources.cutoff_date_idx
    entry_actual = np.asarray(
        sources.primary.entry_filled[dates, symbols], dtype=np.float64
    )
    result["entry_fill"] = _date_equal_binary_metrics(
        dates,
        entry_actual,
        prediction.entry_probability,
        entry_valid,
    )
    result["selection_score"] = float(
        np.mean(
            [
                result["horizons"][str(horizon)]["daily_rank_ic_mean"]
                for horizon in horizons
            ]
        )
    )
    return result


def _train_epoch(
    model: MultiHorizonSequenceModel,
    assembler: MultiHorizonBatchAssembler,
    date_indices: Sequence[int],
    *,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    date_batch_size: int,
    seed: int,
    weights: Mapping[str, float],
) -> dict[str, float]:
    model.train()
    dates = [int(value) for value in date_indices]
    np.random.default_rng(seed).shuffle(dates)
    totals: dict[str, float] = {
        "loss": 0.0,
        "rank_loss": 0.0,
        "absolute_huber_loss": 0.0,
        "quantile_pinball_loss": 0.0,
        "market_huber_loss": 0.0,
        "market_binary_loss": 0.0,
        "entry_fill_binary_loss": 0.0,
        "exit_block_binary_loss": 0.0,
        "gradient_norm": 0.0,
        "gradient_nonfinite_batches": 0.0,
    }
    batch_count = 0
    for current_dates in _chunks(dates, date_batch_size):
        batch = assembler.load(current_dates)
        if batch is None:
            continue
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=assembler.device.type, enabled=assembler.device.type == "cuda"
        ):
            static, missing = seq.cross_sectional_standardize(
                batch.static, batch.group_sizes
            )
            output = model(
                static, missing, batch.sequence, batch.market, batch.group_sizes
            )
            loss, components = payoff_distribution_loss(
                output,
                batch,
                horizons=model.horizons,
                distribution_heads=model.distribution_heads,
                weights=weights,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        totals["loss"] += float(loss.detach().cpu())
        if bool(torch.isfinite(gradient_norm)):
            totals["gradient_norm"] += float(gradient_norm.detach().cpu())
        else:
            totals["gradient_nonfinite_batches"] += 1.0
        for key, value in components.items():
            totals[key] += value
        batch_count += 1
        del batch, static, missing, output, loss
        if (
            batch_count % 4 == 0
            and base.psutil.virtual_memory().available < seq.MEMORY_TRIM_THRESHOLD_BYTES
        ):
            gc.collect()
            base._trim_working_set()
    if not batch_count:
        raise MultiHorizonError("training_epoch_empty")
    return {key: value / batch_count for key, value in totals.items()}


def _model_parameter_norm(model: nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.requires_grad:
            total += float(parameter.detach().float().square().sum().cpu())
    return float(math.sqrt(total))


def _new_model(
    *, horizons: Sequence[int], distribution_heads: bool, seed: int
) -> MultiHorizonSequenceModel:
    _set_determinism(seed)
    return MultiHorizonSequenceModel(
        horizons=horizons, distribution_heads=distribution_heads
    )


def _fit_model(
    sources: MultiHorizonSources,
    *,
    variant: str,
    train_dates: Sequence[int],
    validation_dates: Sequence[int],
    device: torch.device,
    seed: int,
) -> tuple[
    MultiHorizonSequenceModel,
    seq.Normalization,
    TargetNormalization,
    int,
    list[dict[str, Any]],
]:
    study = sources.study
    horizons, distribution_heads = _variant_contract(study, variant)
    model_contract = study["model"]
    sequence_normalization = seq._fit_normalization(sources.primary, train_dates)
    target_normalization = _fit_target_normalization(sources, train_dates, horizons)
    assembler = MultiHorizonBatchAssembler(
        sources,
        horizons=horizons,
        sequence_normalization=sequence_normalization,
        target_normalization=target_normalization,
        device=device,
    )
    model = _new_model(
        horizons=horizons, distribution_heads=distribution_heads, seed=seed
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_contract["learning_rate"]),
        weight_decay=float(model_contract["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    score_history: list[float] = []
    history: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(1, int(model_contract["maximum_epochs"]) + 1):
        started = time.perf_counter()
        train_metrics = _train_epoch(
            model,
            assembler,
            train_dates,
            optimizer=optimizer,
            scaler=scaler,
            date_batch_size=int(model_contract["date_batch_size"]),
            seed=seed + epoch,
            weights=model_contract["loss_weights"],
        )
        validation_prediction = _predict_dates(
            model,
            assembler,
            validation_dates,
            date_batch_size=int(model_contract["date_batch_size"]),
        )
        validation_metrics = _prediction_metrics(
            sources, validation_prediction, horizons
        )
        raw_score = float(validation_metrics["selection_score"])
        score_history.append(raw_score)
        window = min(
            int(model_contract["selection_smoothing_window"]), len(score_history)
        )
        smoothed_score = float(np.mean(score_history[-window:]))
        row = {
            "epoch": epoch,
            "elapsed_seconds": float(time.perf_counter() - started),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "validation_selection_score": raw_score,
            "validation_selection_score_smoothed": smoothed_score,
            "validation_horizons": validation_metrics["horizons"],
            "validation_entry_fill": validation_metrics["entry_fill"],
            "parameter_norm": _model_parameter_norm(model),
        }
        history.append(row)
        base._emit("multi_horizon_epoch_completed", variant=variant, seed=seed, **row)
        del validation_prediction
        if smoothed_score > best_score + 1.0e-5:
            best_score = smoothed_score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(model_contract["patience"]):
                break
    if best_state is None or best_epoch <= 0:
        raise MultiHorizonError("model_selection_failed")
    model.load_state_dict(best_state)
    model.eval()
    del optimizer, scaler, best_state
    gc.collect()
    base._trim_working_set()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        model,
        sequence_normalization,
        target_normalization,
        best_epoch,
        history,
    )


def _task_root(output_root: Path, *, variant: str, fold: int) -> Path:
    return output_root / "models" / str(variant) / f"fold_{int(fold)}"


def _save_array(path: Path, values: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as stream:
        np.save(stream, np.asarray(values), allow_pickle=False)
    os.replace(partial, path)
    return _file_record(path, shape=list(values.shape), dtype=str(values.dtype))


def train_fold(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    variant: str,
    fold_number: int,
) -> dict[str, Any]:
    study = load_study(study_path)
    horizons, distribution_heads = _variant_contract(study, variant)
    if not torch.cuda.is_available():
        raise MultiHorizonError("cuda_is_required_for_formal_run")
    sources = _load_sources(study)
    folds = _forward_folds(sources)
    fold = next((item for item in folds if int(item["fold"]) == int(fold_number)), None)
    if fold is None:
        raise MultiHorizonError(f"unknown_fold:{fold_number}")
    train_dates, validation_dates = _fold_dates(sources, fold)
    fingerprint = _stable_hash(
        {
            "schema": SCHEMA,
            "implementation_sha256": base._sha256(Path(__file__)),
            "study_sha256": base._sha256(study_path),
            "base_study_sha256": base._sha256(
                _workspace_path(str(study["sources"]["base_study"]))
            ),
            "input_fingerprint": sources.primary.context.model_manifest[
                "input_fingerprint"
            ],
            "target_fingerprint": sources.primary.target_manifest["fingerprint"],
            "feature_names": sources.feature_names,
            "feature_columns": sources.feature_columns.tolist(),
            "variant": variant,
            "horizons": horizons,
            "distribution_heads": distribution_heads,
            "fold": fold,
            "model": study["model"],
        }
    )
    root = _task_root(output_root, variant=variant, fold=fold_number)
    manifest_path = root / "task_result.json"
    if manifest_path.is_file():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            current.get("status") == "completed"
            and current.get("fingerprint") == fingerprint
            and current.get("forbidden_2026_read_count") == 0
        ):
            return current
    resource = base.resource_plan(
        reserve_gib=float(sources.base_study["resources"]["reserve_system_gib"]),
        maximum_threads=int(sources.base_study["resources"]["maximum_cpu_threads"]),
        histogram_pool_cap_mb=int(
            sources.base_study["resources"]["histogram_pool_cap_mb"]
        ),
        sequence_batch_cap=int(sources.base_study["resources"]["sequence_batch_cap"]),
    )
    base._assert_resource_capacity(resource, 1 * (1 << 30))
    seed = int(sources.base_study["validation"]["seed"]) + 200_000
    seed += int(fold_number) * 100 + (0 if variant == MODEL_VARIANTS[0] else 50)
    base._emit(
        "multi_horizon_fold_started",
        variant=variant,
        fold=fold_number,
        horizons=list(horizons),
        train_dates=len(train_dates),
        validation_dates=len(validation_dates),
        seed=seed,
        resource_plan=asdict(resource),
    )
    device = torch.device("cuda")
    with base.ResourceMonitor() as monitor:
        (
            model,
            sequence_normalization,
            target_normalization,
            best_epoch,
            history,
        ) = _fit_model(
            sources,
            variant=variant,
            train_dates=train_dates,
            validation_dates=validation_dates,
            device=device,
            seed=seed,
        )
        assembler = MultiHorizonBatchAssembler(
            sources,
            horizons=horizons,
            sequence_normalization=sequence_normalization,
            target_normalization=target_normalization,
            device=device,
        )
        prediction = _predict_dates(
            model,
            assembler,
            validation_dates,
            date_batch_size=int(study["model"]["date_batch_size"]),
        )
        validation_metrics = _prediction_metrics(sources, prediction, horizons)
        resource_metrics = monitor.metrics()
    validation_positions = base._fold_rows(
        sources.primary.context.row_index, fold, "validation"
    )
    if not np.array_equal(prediction.rows, validation_positions):
        raise MultiHorizonError("full_validation_slate_alignment_failed")
    files: dict[str, Any] = {}
    for horizon in horizons:
        files[f"point_d{horizon}"] = _save_array(
            root / f"point_d{horizon}.npy",
            prediction.point[horizon].astype(np.float32),
        )
        files[f"exit_probability_d{horizon}"] = _save_array(
            root / f"exit_probability_d{horizon}.npy",
            prediction.exit_probability[horizon].astype(np.float32),
        )
        if horizon in prediction.quantiles:
            files[f"quantiles_d{horizon}"] = _save_array(
                root / f"quantiles_d{horizon}.npy",
                prediction.quantiles[horizon].astype(np.float32),
            )
    files["entry_probability"] = _save_array(
        root / "entry_probability.npy",
        prediction.entry_probability.astype(np.float32),
    )
    checkpoint_path = root / "model.pt"
    checkpoint_partial = checkpoint_path.with_suffix(".pt.partial")
    torch.save(
        {
            "schema": SCHEMA,
            "variant": variant,
            "horizons": list(horizons),
            "distribution_heads": distribution_heads,
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "sequence_normalization": sequence_normalization.payload(),
            "target_normalization": target_normalization.payload(),
            "feature_names": sources.feature_names,
            "feature_columns": sources.feature_columns.tolist(),
            "lookback": int(study["features"]["lookback"]),
            "seed": seed,
        },
        checkpoint_partial,
    )
    os.replace(checkpoint_partial, checkpoint_path)
    files["model"] = _file_record(checkpoint_path)
    history_path = root / "training_history.json"
    _write_json(history_path, {"selection": history})
    files["training_history"] = _file_record(history_path)
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": study["study_id"],
        "fingerprint": fingerprint,
        "variant": variant,
        "horizons": list(horizons),
        "distribution_heads": distribution_heads,
        "fold": fold,
        "seed": seed,
        "best_epoch": best_epoch,
        "feature_contract": {
            "variant": study["features"]["variant"],
            "feature_count": len(sources.feature_names),
            "feature_names": list(sources.feature_names),
        },
        "target_normalization": target_normalization.payload(),
        "validation_metrics": validation_metrics,
        "full_validation_slate_scored_before_future_status": True,
        "validation_row_count": len(validation_positions),
        "forbidden_2026_read_count": 0,
        "resource_plan": asdict(resource),
        "resource_metrics": resource_metrics,
        "files": files,
    }
    _write_json(manifest_path, result)
    base._emit(
        "multi_horizon_fold_completed",
        variant=variant,
        fold=fold_number,
        best_epoch=best_epoch,
        selection_score=validation_metrics["selection_score"],
    )
    return result


def audit_contract(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    sources = _load_sources(study)
    primary = sources.primary
    row_index = primary.context.row_index
    dates = row_index["date_idx"].to_numpy(dtype=np.int32)
    symbols = row_index["symbol_idx"].to_numpy(dtype=np.int32)
    entry_known = dates + 1 <= sources.cutoff_date_idx
    entry_filled = np.asarray(primary.entry_filled[dates, symbols], dtype=bool)
    horizons: dict[str, Any] = {}
    for horizon in (5, 10):
        source = sources.by_horizon[horizon]
        base_valid = np.asarray(source.exact_valid[:, source.base_column], dtype=bool)
        stress_valid = np.asarray(
            source.exact_valid[:, source.stress_column], dtype=bool
        )
        path_valid = np.asarray(source.path_valid[:, source.gross_column], dtype=bool)
        fill_days = np.asarray(
            source.fill_days[:, source.horizon_column], dtype=np.int16
        )
        exit_valid = entry_filled & path_valid & (fill_days >= horizon)
        horizons[str(horizon)] = {
            "base_exact_valid_count": int(base_valid.sum()),
            "stress_exact_valid_count": int(stress_valid.sum()),
            "base_and_stress_valid_count": int((base_valid & stress_valid).sum()),
            "legal_path_valid_count": int(path_valid.sum()),
            "exit_head_valid_count": int(exit_valid.sum()),
            "planned_exit_count": int((exit_valid & (fill_days == horizon)).sum()),
            "delayed_or_terminal_exit_count": int(
                (exit_valid & (fill_days > horizon)).sum()
            ),
            "right_censored_or_no_entry_count": int((~path_valid).sum()),
        }
    folds = _forward_folds(sources)
    fold_counts = []
    for fold in folds:
        train_dates, validation_dates = _fold_dates(sources, fold)
        fold_counts.append(
            {
                "fold": int(fold["fold"]),
                "training_date_count": len(train_dates),
                "validation_date_count": len(validation_dates),
                "training_maximum_date_idx": int(fold["training_maximum_date_idx"]),
                "validation_start_date_idx": int(fold["validation_start_date_idx"]),
                "validation_end_date_idx": int(fold["validation_end_date_idx"]),
            }
        )
    fingerprint = _stable_hash(
        {
            "schema": AUDIT_SCHEMA,
            "implementation_sha256": base._sha256(Path(__file__)),
            "study_sha256": base._sha256(study_path),
            "input_fingerprint": primary.context.model_manifest["input_fingerprint"],
            "target_fingerprint": primary.target_manifest["fingerprint"],
            "feature_names": sources.feature_names,
            "feature_columns": sources.feature_columns.tolist(),
            "horizon_counts": horizons,
            "fold_counts": fold_counts,
        }
    )
    result = {
        "schema": AUDIT_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": study["study_id"],
        "fingerprint": fingerprint,
        "row_count": len(row_index),
        "feature_contract": {
            "variant": study["features"]["variant"],
            "feature_count": len(sources.feature_names),
            "feature_names": list(sources.feature_names),
            "feature_columns_unique": len(np.unique(sources.feature_columns))
            == len(sources.feature_columns),
        },
        "entry": {
            "known_count": int(entry_known.sum()),
            "right_censored_count": int((~entry_known).sum()),
            "market_filled_count": int((entry_known & entry_filled).sum()),
            "market_unfilled_count": int((entry_known & ~entry_filled).sum()),
        },
        "horizons": horizons,
        "folds": fold_counts,
        "mask_contract": {
            "return_loss_uses_exact_valid_only": True,
            "entry_head_uses_full_known_signal_slate": True,
            "exit_head_conditions_on_market_entry_and_known_legal_path": True,
            "quantiles_are_outputs_not_separate_labels": True,
            "ranking_precedes_future_fill_and_outcome_status": True,
            "D2_D3_materialized_but_not_trained_in_v1": True,
        },
        "maximum_source_date_idx_read": sources.cutoff_date_idx,
        "maximum_source_date_read": study["sources"]["maximum_outcome_date"],
        "forbidden_2026_read_count": 0,
        "sources": {
            "study": _file_record(study_path),
            "base_targets": _file_record(
                sources.base_output_root / "targets" / "manifest.json"
            ),
        },
    }
    manifest_path = output_root / "contract_audit" / "manifest.json"
    _write_json(manifest_path, result)
    return result


def _load_task(
    output_root: Path, *, variant: str, fold: int
) -> tuple[dict[str, Any], Path]:
    path = _task_root(output_root, variant=variant, fold=fold) / "task_result.json"
    if not path.is_file():
        raise MultiHorizonError(f"missing_completed_task:{variant}:{fold}")
    task = json.loads(path.read_text(encoding="utf-8"))
    if (
        task.get("status") != "completed"
        or task.get("variant") != variant
        or int(task["fold"]["fold"]) != int(fold)
        or task.get("forbidden_2026_read_count") != 0
    ):
        raise MultiHorizonError(f"invalid_completed_task:{variant}:{fold}")
    return task, path


def _load_prediction_array(task: Mapping[str, Any], key: str) -> np.ndarray:
    return np.asarray(
        np.load(task["files"][key]["path"], mmap_mode="r", allow_pickle=False)
    )


def evaluate_variant(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    variant: str,
) -> dict[str, Any]:
    study = load_study(study_path)
    horizons, distribution_heads = _variant_contract(study, variant)
    sources = _load_sources(study)
    folds = _forward_folds(sources)
    tasks = [
        _load_task(output_root, variant=variant, fold=int(fold["fold"]))
        for fold in folds
    ]
    fingerprint = _stable_hash(
        {
            "schema": EVALUATION_SCHEMA,
            "implementation_sha256": base._sha256(Path(__file__)),
            "study_sha256": base._sha256(study_path),
            "variant": variant,
            "tasks": {str(path): base._sha256(path) for _, path in tasks},
            "ranking": "all_finite_predictions_before_future_status__v1",
        }
    )
    root = output_root / "evaluation" / variant
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            current.get("status") == "completed"
            and current.get("fingerprint") == fingerprint
            and current.get("forbidden_2026_read_count") == 0
        ):
            return current
    horizon_outputs: dict[str, Any] = {}
    combined_prediction_parts: list[PredictionOutput] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold, (task, _) in zip(folds, tasks, strict=True):
        fold_number = int(fold["fold"])
        positions = base._fold_rows(
            sources.primary.context.row_index, fold, "validation"
        )
        point = {
            horizon: _load_prediction_array(task, f"point_d{horizon}")
            for horizon in horizons
        }
        quantiles = (
            {
                horizon: _load_prediction_array(task, f"quantiles_d{horizon}")
                for horizon in horizons
            }
            if distribution_heads
            else {}
        )
        prediction = PredictionOutput(
            rows=positions,
            point=point,
            quantiles=quantiles,
            entry_probability=_load_prediction_array(task, "entry_probability"),
            exit_probability={
                horizon: _load_prediction_array(task, f"exit_probability_d{horizon}")
                for horizon in horizons
            },
        )
        if any(len(value) != len(positions) for value in prediction.point.values()):
            raise MultiHorizonError("evaluation_prediction_length_mismatch")
        fold_metrics.append(
            {"fold": fold_number, **_prediction_metrics(sources, prediction, horizons)}
        )
        combined_prediction_parts.append(prediction)

    combined_prediction = PredictionOutput(
        rows=np.concatenate([item.rows for item in combined_prediction_parts]),
        point={
            horizon: np.concatenate(
                [item.point[horizon] for item in combined_prediction_parts]
            )
            for horizon in horizons
        },
        quantiles={
            horizon: np.concatenate(
                [item.quantiles[horizon] for item in combined_prediction_parts]
            )
            for horizon in horizons
            if distribution_heads
        },
        entry_probability=np.concatenate(
            [item.entry_probability for item in combined_prediction_parts]
        ),
        exit_probability={
            horizon: np.concatenate(
                [item.exit_probability[horizon] for item in combined_prediction_parts]
            )
            for horizon in horizons
        },
    )
    combined_metrics = _prediction_metrics(sources, combined_prediction, horizons)
    for horizon in horizons:
        source = sources.by_horizon[horizon]
        daily_parts: list[pd.DataFrame] = []
        decile_parts: list[pd.DataFrame] = []
        selection_parts: list[pd.DataFrame] = []
        payoff_fold_summaries: list[dict[str, Any]] = []
        for fold, prediction in zip(folds, combined_prediction_parts, strict=True):
            date_values = sources.primary.context.row_index.iloc[prediction.rows][
                "date_idx"
            ].to_numpy(dtype=np.int32)
            market_return = {int(value): 0.0 for value in np.unique(date_values)}
            market_probability = {int(value): 0.5 for value in np.unique(date_values)}
            score = seq.PredictionOutput(
                rows=prediction.rows,
                stock_score=prediction.point[horizon],
                market_return=market_return,
                market_probability=market_probability,
            )
            daily, deciles, selections, summary = seq._build_evaluation_frames(
                source,
                score,
                fold=int(fold["fold"]),
                unfilled_as_cash=True,
            )
            daily_parts.append(daily)
            decile_parts.append(deciles)
            selection_parts.append(selections)
            payoff_fold_summaries.append({"fold": int(fold["fold"]), **summary})
        horizon_root = root / f"horizon_{horizon}"
        daily = pd.concat(daily_parts, ignore_index=True).sort_values("date_idx")
        deciles = pd.concat(decile_parts, ignore_index=True).sort_values(
            ["date_idx", "decile"]
        )
        selections = pd.concat(selection_parts, ignore_index=True).sort_values(
            ["date_idx", "selection_rank"]
        )
        daily_path = horizon_root / "daily_metrics.parquet"
        decile_path = horizon_root / "decile_daily.parquet"
        selection_path = horizon_root / "top10_selections.parquet"
        base._write_parquet(daily, daily_path)
        base._write_parquet(deciles, decile_path)
        base._write_parquet(selections, selection_path)
        payoff_combined = base._payoff_summary(daily, deciles)
        horizon_manifest = {
            "schema": EVALUATION_SCHEMA,
            "status": "completed",
            "completed_at": _now(),
            "study_id": study["study_id"],
            "fingerprint": fingerprint,
            "variant": variant,
            "horizon": horizon,
            "fold_summaries": payoff_fold_summaries,
            "combined": payoff_combined,
            "forecast_metrics": combined_metrics["horizons"][str(horizon)],
            "ranking_contract": "all_finite_predictions_before_future_status__v1",
            "forbidden_2026_read_count": 0,
            "files": {
                "daily_metrics": _file_record(daily_path, row_count=len(daily)),
                "decile_daily": _file_record(decile_path, row_count=len(deciles)),
                "top10_selections": _file_record(
                    selection_path, row_count=len(selections)
                ),
            },
        }
        _write_json(horizon_root / "manifest.json", horizon_manifest)
        horizon_outputs[str(horizon)] = {
            "manifest": _file_record(horizon_root / "manifest.json"),
            "combined": payoff_combined,
            "forecast_metrics": combined_metrics["horizons"][str(horizon)],
        }
    result = {
        "schema": EVALUATION_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": study["study_id"],
        "fingerprint": fingerprint,
        "variant": variant,
        "horizons": list(horizons),
        "distribution_heads": distribution_heads,
        "fold_metrics": fold_metrics,
        "combined_forecast_metrics": combined_metrics,
        "horizon_outputs": horizon_outputs,
        "ranking_contract": "all_finite_predictions_before_future_status__v1",
        "forbidden_2026_read_count": 0,
        "sources": {
            "tasks": [_file_record(path) for _, path in tasks],
            "contract_audit": _file_record(
                output_root / "contract_audit" / "manifest.json"
            ),
        },
    }
    _write_json(manifest_path, result)
    return result


def replay_variant_accounts(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    variant: str,
    starting_cash: float = 1_000_000.0,
) -> dict[str, Any]:
    study = load_study(study_path)
    horizons, _ = _variant_contract(study, variant)
    sources = _load_sources(study)
    evaluation = evaluate_variant(
        study_path=study_path, output_root=output_root, variant=variant
    )
    outputs: dict[str, Any] = {}
    for horizon in horizons:
        evaluation_path = (
            output_root
            / "evaluation"
            / variant
            / f"horizon_{horizon}"
            / "manifest.json"
        )
        horizon_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        selections = pd.read_parquet(
            horizon_evaluation["files"]["top10_selections"]["path"]
        )
        source = sources.by_horizon[horizon]
        schedule, dropped = seq._prepare_exact_payoff_schedule(
            selections=selections,
            sources=source,
            horizon=horizon,
            variant=f"{variant}_d{horizon}",
        )
        specs = seq._exact_payoff_account_specs(
            variant=f"{variant}_d{horizon}", horizon=horizon
        )
        fingerprint = _stable_hash(
            {
                "schema": ACCOUNT_SCHEMA,
                "implementation_sha256": base._sha256(Path(__file__)),
                "evaluation_fingerprint": evaluation["fingerprint"],
                "selection_sha256": horizon_evaluation["files"]["top10_selections"][
                    "sha256"
                ],
                "variant": variant,
                "horizon": horizon,
                "starting_cash": float(starting_cash),
                "specs": specs,
            }
        )
        root = output_root / "account_replay" / variant / f"horizon_{horizon}"
        manifest_path = root / "manifest.json"
        if manifest_path.is_file():
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                current.get("status") == "completed"
                and current.get("fingerprint") == fingerprint
                and current.get("forbidden_2026_read_count") == 0
            ):
                outputs[str(horizon)] = current
                continue
        tasks = seq._run_exact_payoff_account_tasks(
            root=root,
            schema=ACCOUNT_SCHEMA,
            fingerprint=fingerprint,
            schedule=schedule,
            sources=source,
            specs=specs,
            starting_cash=starting_cash,
        )
        summaries = tasks["summaries"]
        primary = seq._account_result(
            summaries,
            top_k=10,
            cost_scenario="stress",
            allow_overlapping_same_symbol=False,
        )
        cap5 = seq._account_result(
            summaries,
            top_k=10,
            cost_scenario="stress",
            cap=0.05,
            allow_overlapping_same_symbol=False,
        )
        cap10 = seq._account_result(
            summaries,
            top_k=10,
            cost_scenario="stress",
            cap=0.10,
            allow_overlapping_same_symbol=False,
        )
        result = {
            "schema": ACCOUNT_SCHEMA,
            "status": "completed",
            "completed_at": _now(),
            "study_id": study["study_id"],
            "fingerprint": fingerprint,
            "variant": variant,
            "horizon": horizon,
            "starting_cash": float(starting_cash),
            "summaries": summaries,
            "primary_stress_top10_no_overlapping_same_symbol": primary,
            "stress_top10_no_overlap_cap5": cap5,
            "stress_top10_no_overlap_cap10": cap10,
            "dropped_selection_count_after_gross_join": int(dropped),
            "decision_boundary": {
                "fixed_horizon": horizon,
                "model_does_not_choose_horizon_per_stock": True,
                "cohort_equity_fraction": 1.0 / float(horizon),
                "next_open_entry": True,
                "finite_cash": True,
                "no_leverage": True,
                "unfilled_selected_order_is_cash_without_substitution": True,
                "stable_profit_claim_allowed": False,
                "forbidden_2026_read_count": 0,
            },
            "forbidden_2026_read_count": 0,
            "files": {
                "summary": tasks["summary"],
                "selection_schedule": tasks["selection_schedule"],
                "tasks": tasks["tasks"],
            },
            "sources": {
                "evaluation": _file_record(evaluation_path),
            },
        }
        _write_json(manifest_path, result)
        outputs[str(horizon)] = result
    top_manifest = {
        "schema": ACCOUNT_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": study["study_id"],
        "variant": variant,
        "horizons": {
            key: {
                "manifest": _file_record(
                    output_root
                    / "account_replay"
                    / variant
                    / f"horizon_{key}"
                    / "manifest.json"
                ),
                "primary": value["primary_stress_top10_no_overlapping_same_symbol"],
                "cap10": value["stress_top10_no_overlap_cap10"],
                "cap5": value["stress_top10_no_overlap_cap5"],
            }
            for key, value in outputs.items()
        },
        "forbidden_2026_read_count": 0,
    }
    _write_json(
        output_root / "account_replay" / variant / "manifest.json", top_manifest
    )
    return top_manifest


def build_comparison(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    evaluations = {
        variant: evaluate_variant(
            study_path=study_path, output_root=output_root, variant=variant
        )
        for variant in MODEL_VARIANTS
    }
    accounts = {
        variant: replay_variant_accounts(
            study_path=study_path, output_root=output_root, variant=variant
        )
        for variant in MODEL_VARIANTS
    }
    control_ic = float(
        evaluations[MODEL_VARIANTS[0]]["combined_forecast_metrics"]["horizons"]["10"][
            "daily_rank_ic_mean"
        ]
    )
    challenger_d10_ic = float(
        evaluations[MODEL_VARIANTS[1]]["combined_forecast_metrics"]["horizons"]["10"][
            "daily_rank_ic_mean"
        ]
    )
    control_account = accounts[MODEL_VARIANTS[0]]["horizons"]["10"]
    challenger_account = accounts[MODEL_VARIANTS[1]]["horizons"]["10"]
    comparison = {
        "d10_rank_ic_delta": challenger_d10_ic - control_ic,
        "d10_no_overlap_return_delta": float(
            challenger_account["primary"]["total_net_return"]
            - control_account["primary"]["total_net_return"]
        ),
        "d10_no_overlap_drawdown_delta": float(
            challenger_account["primary"]["maximum_drawdown"]
            - control_account["primary"]["maximum_drawdown"]
        ),
        "d10_no_overlap_cap10_return_delta": float(
            challenger_account["cap10"]["total_net_return"]
            - control_account["cap10"]["total_net_return"]
        ),
    }
    result = {
        "schema": COMPARISON_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": study["study_id"],
        "variants": {
            variant: {
                "evaluation": _file_record(
                    output_root / "evaluation" / variant / "manifest.json"
                ),
                "accounts": _file_record(
                    output_root / "account_replay" / variant / "manifest.json"
                ),
            }
            for variant in MODEL_VARIANTS
        },
        "architecture_matched_d10_comparison": comparison,
        "decision_boundary": {
            "D2_D3_not_trained": True,
            "quantiles_not_used_for_ranking_or_risk_overlay": True,
            "fixed_D5_and_D10_policies_only": True,
            "no_topk_threshold_or_horizon_search": True,
            "comparison_is_adaptive_development_evidence": True,
            "stable_profit_claim_allowed": False,
        },
        "forbidden_2026_read_count": 0,
    }
    _write_json(output_root / "comparison" / "manifest.json", result)
    return result


def run_all(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    audit_contract(study_path=study_path, output_root=output_root)
    for variant in MODEL_VARIANTS:
        for fold_number in range(1, 6):
            train_fold(
                study_path=study_path,
                output_root=output_root,
                variant=variant,
                fold_number=fold_number,
            )
        evaluate_variant(
            study_path=study_path, output_root=output_root, variant=variant
        )
        replay_variant_accounts(
            study_path=study_path, output_root=output_root, variant=variant
        )
    return build_comparison(study_path=study_path, output_root=output_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen D5/D10 multi-horizon distribution study."
    )
    parser.add_argument("--study-path", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit-contract")
    train = subparsers.add_parser("train-fold")
    train.add_argument("--variant", choices=MODEL_VARIANTS, required=True)
    train.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--variant", choices=MODEL_VARIANTS, required=True)
    replay = subparsers.add_parser("replay-accounts")
    replay.add_argument("--variant", choices=MODEL_VARIANTS, required=True)
    subparsers.add_parser("compare")
    subparsers.add_parser("run-all")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "audit-contract":
        result = audit_contract(
            study_path=args.study_path, output_root=args.output_root
        )
    elif args.command == "train-fold":
        result = train_fold(
            study_path=args.study_path,
            output_root=args.output_root,
            variant=args.variant,
            fold_number=args.fold,
        )
    elif args.command == "evaluate":
        result = evaluate_variant(
            study_path=args.study_path,
            output_root=args.output_root,
            variant=args.variant,
        )
    elif args.command == "replay-accounts":
        result = replay_variant_accounts(
            study_path=args.study_path,
            output_root=args.output_root,
            variant=args.variant,
        )
    elif args.command == "compare":
        result = build_comparison(
            study_path=args.study_path, output_root=args.output_root
        )
    else:
        result = run_all(study_path=args.study_path, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
