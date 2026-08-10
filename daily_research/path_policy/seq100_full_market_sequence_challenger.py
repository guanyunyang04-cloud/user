"""Payoff-aligned daily sequence challenger for the full-market forecast study.

The model deliberately separates two questions:

* which stocks have the best relative D5 payoff on a signal date; and
* whether the market date itself has positive executable D5 expectancy.

All outer validation predictions are generated from strictly earlier signal dates.
The current-day branch consumes the frozen 557-field causal matrix, while the
temporal branch reads compact daily paths directly from the existing sequence pack.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

from daily_research.path_policy import seq100_full_market_multitask_forecast as base

SCHEMA = "seq100_full_market_sequence_challenger/1"
ENSEMBLE_SCHEMA = "seq100_full_market_sequence_ensemble_evaluation/1"
ACCOUNT_SCHEMA = "seq100_full_market_sequence_ensemble_account_replay/1"
ROBUSTNESS_SCHEMA = "seq100_full_market_sequence_ensemble_account_robustness/1"
DUAL_HORIZON_SCHEMA = "seq100_full_market_dual_gate_horizon_challenge/1"
AVAILABILITY_SCHEMA = "seq100_full_market_source_availability_stress/1"
REPAIRED_RANK_SCHEMA = "seq100_full_market_repaired_cross_sectional_features/1"
REBUILDABLE_ROBUSTNESS_SCHEMA = (
    "seq100_full_market_rebuildable_core_account_robustness/2"
)
FINAL_BUNDLE_SCHEMA = "seq100_full_market_forward_policy_bundle/1"
DEFAULT_STUDY_PATH = base.DEFAULT_STUDY_PATH
DEFAULT_OUTPUT_ROOT = base.DEFAULT_OUTPUT_ROOT
DEFAULT_LOOKBACK = 8
DEFAULT_MAX_EPOCHS = 8
DEFAULT_PATIENCE = 2
DEFAULT_DATE_BATCH_SIZE = 4
DEFAULT_LEGACY_BASE_FEATURE_MANIFEST = (
    base.WORKSPACE_ROOT
    / "tmp/seq100_learnability_inputs/attempt_001/base_feature_manifest.json"
)


def _lookback_artifact_root(
    output_root: Path, artifact_name: str, *, lookback: int
) -> Path:
    """Keep the established lookback-8 paths while isolating new challengers."""

    root = output_root / artifact_name
    if int(lookback) == DEFAULT_LOOKBACK:
        return root
    return root / f"lookback_{int(lookback)}"
MEMORY_TRIM_THRESHOLD_BYTES = int(1.75 * (1 << 30))
PRICE_PATH_SCALE = 0.10
RELATIVE_TARGET_SCALE = 0.05
MARKET_TARGET_SCALE = 0.01
DAILY_RAW_DERIVED_SLICE = slice(6, 13)


class SequenceChallengerError(RuntimeError):
    """Raised when the sequence study contract is violated."""


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


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@dataclass(frozen=True)
class DateBlock:
    date_idx: int
    trade_date: str
    start: int
    stop: int


def _date_blocks(row_index: pd.DataFrame) -> dict[int, DateBlock]:
    dates = row_index["date_idx"].to_numpy(dtype=np.int32)
    labels = row_index["trade_date"].astype(str).to_numpy()
    unique, first, counts = np.unique(dates, return_index=True, return_counts=True)
    blocks = {
        int(date_idx): DateBlock(
            date_idx=int(date_idx),
            trade_date=str(labels[int(start)]),
            start=int(start),
            stop=int(start + count),
        )
        for date_idx, start, count in zip(unique, first, counts, strict=True)
    }
    if sum(block.stop - block.start for block in blocks.values()) != len(row_index):
        raise SequenceChallengerError("date_block_row_count_mismatch")
    return blocks


def _open_memmap(record: Mapping[str, Any], dtype: np.dtype[Any]) -> np.memmap:
    path = Path(record["path"])
    shape = tuple(int(value) for value in record["shape"])
    expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    if not path.is_file() or int(path.stat().st_size) != expected:
        raise SequenceChallengerError(f"invalid_memmap:{path}")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


@dataclass
class SequenceSources:
    context: base.SourceContext
    matrix: np.memmap
    exact_values: np.memmap
    exact_valid: np.memmap
    base_column: int
    stress_column: int
    path_values: np.memmap
    path_valid: np.memmap
    fill_days: np.memmap
    gross_column: int
    horizon_column: int
    daily_raw: np.memmap
    daily_state: np.memmap
    turnover: np.memmap
    entry_filled: np.memmap
    blocks: dict[int, DateBlock]
    market_columns: np.ndarray
    feature_names: tuple[str, ...]
    pack_feature_names: tuple[str, ...]
    target_manifest: dict[str, Any]
    path_manifest: dict[str, Any]


@dataclass(frozen=True)
class StaticFeatureOverride:
    values: np.memmap
    columns: np.ndarray
    feature_names: tuple[str, ...]
    manifest: dict[str, Any]


def _load_sources(study: Mapping[str, Any], *, output_root: Path) -> SequenceSources:
    context = base._source_context(study)
    target_manifest = base.prepare_exact_net_targets(
        study_path=DEFAULT_STUDY_PATH, output_root=output_root
    )
    target_columns = list(target_manifest["target_columns"])
    base_column = target_columns.index("exact_net_return_d5_base")
    stress_column = target_columns.index("exact_net_return_d5_stress")
    exact_values = _open_memmap(target_manifest["files"]["values"], np.float32)
    exact_valid = _open_memmap(target_manifest["files"]["valid"], np.uint8)
    path_manifest = json.loads(
        (output_root / "targets" / "manifest.json").read_text(encoding="utf-8")
    )
    path_values = _open_memmap(path_manifest["files"]["values"], np.float32)
    path_valid = _open_memmap(path_manifest["files"]["valid"], np.uint8)
    fill_days = _open_memmap(path_manifest["files"]["legal_fill_days"], np.int16)
    gross_column = list(path_manifest["target_columns"]).index("legal_exit_return_d5")
    horizon_column = list(path_manifest["files"]["legal_fill_days"]["horizons"]).index(
        5
    )
    matrix = _open_memmap(context.model_manifest["storage"]["compact"], np.float32)
    channels = context.pack["feature_channels"]
    daily_raw = _open_memmap(channels["daily_raw"], np.float32)
    daily_state = _open_memmap(channels["daily_state"], np.float32)
    turnover = _open_memmap(channels["turnover"], np.float32)
    entry_filled = _open_memmap(context.pack["masks"]["entry_filled"], np.bool_)
    feature_names = tuple(context.model_manifest["feature_groups"]["compact_core"])
    records = {
        str(record["feature_name"]): dict(record)
        for record in context.model_manifest["features"]
    }
    market_names = tuple(
        name
        for name in feature_names
        if str(records[name]["analytic_family"]) == "market_state"
    )
    if len(feature_names) != 557 or len(market_names) != 54:
        raise SequenceChallengerError("frozen_feature_contract_changed")
    market_columns = np.asarray(
        [feature_names.index(name) for name in market_names], dtype=np.int32
    )
    raw_names = tuple(channels["daily_raw"]["columns"])
    state_names = tuple(channels["daily_state"]["columns"])
    turnover_names = tuple(channels["turnover"]["columns"])
    pack_feature_names = (
        *(f"relative_{name}" for name in raw_names[:4]),
        *raw_names[DAILY_RAW_DERIVED_SLICE],
        *state_names,
        *turnover_names,
    )
    if len(pack_feature_names) != 32:
        raise SequenceChallengerError("sequence_feature_count_changed")
    return SequenceSources(
        context=context,
        matrix=matrix,
        exact_values=exact_values,
        exact_valid=exact_valid,
        base_column=base_column,
        stress_column=stress_column,
        path_values=path_values,
        path_valid=path_valid,
        fill_days=fill_days,
        gross_column=gross_column,
        horizon_column=horizon_column,
        daily_raw=daily_raw,
        daily_state=daily_state,
        turnover=turnover,
        entry_filled=entry_filled,
        blocks=_date_blocks(context.row_index),
        market_columns=market_columns,
        feature_names=feature_names,
        pack_feature_names=tuple(pack_feature_names),
        target_manifest=target_manifest,
        path_manifest=path_manifest,
    )


@dataclass(frozen=True)
class Normalization:
    sequence_mean: np.ndarray
    sequence_scale: np.ndarray
    market_center: np.ndarray
    market_scale: np.ndarray

    def payload(self) -> dict[str, list[float]]:
        return {
            "sequence_mean": self.sequence_mean.astype(float).tolist(),
            "sequence_scale": self.sequence_scale.astype(float).tolist(),
            "market_center": self.market_center.astype(float).tolist(),
            "market_scale": self.market_scale.astype(float).tolist(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Normalization:
        return cls(
            sequence_mean=np.asarray(payload["sequence_mean"], dtype=np.float32),
            sequence_scale=np.asarray(payload["sequence_scale"], dtype=np.float32),
            market_center=np.asarray(payload["market_center"], dtype=np.float32),
            market_scale=np.asarray(payload["market_scale"], dtype=np.float32),
        )


def _current_sequence_features(
    sources: SequenceSources, date_idx: int, symbols: np.ndarray
) -> np.ndarray:
    raw = np.asarray(sources.daily_raw[date_idx, symbols], dtype=np.float32)
    close = raw[:, 3:4]
    price_relative = np.divide(
        raw[:, :4],
        close,
        out=np.full_like(raw[:, :4], np.nan),
        where=np.isfinite(close) & (close > 0.0),
    ) - np.float32(1.0)
    return np.concatenate(
        [
            price_relative,
            raw[:, DAILY_RAW_DERIVED_SLICE],
            np.asarray(sources.daily_state[date_idx, symbols], dtype=np.float32),
            np.asarray(sources.turnover[date_idx, symbols], dtype=np.float32),
        ],
        axis=1,
    )


def _fit_normalization(
    sources: SequenceSources, date_indices: Sequence[int]
) -> Normalization:
    sequence_sum = np.zeros(32, dtype=np.float64)
    sequence_square_sum = np.zeros(32, dtype=np.float64)
    sequence_count = np.zeros(32, dtype=np.int64)
    market_rows: list[np.ndarray] = []
    for date_idx in date_indices:
        block = sources.blocks[int(date_idx)]
        rows = np.arange(block.start, block.stop, dtype=np.int64)
        valid = np.asarray(sources.exact_valid[rows, sources.base_column], dtype=bool)
        if not valid.any():
            continue
        symbols = sources.context.row_index.iloc[rows[valid]]["symbol_idx"].to_numpy(
            dtype=np.int32
        )
        current = _current_sequence_features(sources, int(date_idx), symbols)
        finite = np.isfinite(current)
        safe = np.where(finite, current, 0.0).astype(np.float64, copy=False)
        sequence_sum += safe.sum(axis=0)
        sequence_square_sum += np.square(safe).sum(axis=0)
        sequence_count += finite.sum(axis=0)
        market_rows.append(
            np.asarray(
                sources.matrix[block.start, sources.market_columns],
                dtype=np.float32,
            )
        )
    if not market_rows or bool((sequence_count == 0).any()):
        raise SequenceChallengerError("normalization_support_empty")
    sequence_mean = sequence_sum / sequence_count
    variance = sequence_square_sum / sequence_count - np.square(sequence_mean)
    sequence_scale = np.sqrt(np.maximum(variance, 1.0e-8))
    # Relative OHLC history spans more than its current-day cross section. A fixed
    # economically interpretable 10% scale avoids fitting future path positions.
    sequence_mean[:4] = 0.0
    sequence_scale[:4] = PRICE_PATH_SCALE
    market = np.stack(market_rows).astype(np.float64)
    market_center = np.nanmedian(market, axis=0)
    q25 = np.nanquantile(market, 0.25, axis=0)
    q75 = np.nanquantile(market, 0.75, axis=0)
    market_scale = q75 - q25
    fallback = np.nanstd(market, axis=0)
    market_scale = np.where(
        np.isfinite(market_scale) & (market_scale > 1.0e-8),
        market_scale,
        np.where(np.isfinite(fallback) & (fallback > 1.0e-8), fallback, 1.0),
    )
    market_center = np.where(np.isfinite(market_center), market_center, 0.0)
    return Normalization(
        sequence_mean=sequence_mean.astype(np.float32),
        sequence_scale=sequence_scale.astype(np.float32),
        market_center=market_center.astype(np.float32),
        market_scale=market_scale.astype(np.float32),
    )


@dataclass
class Batch:
    static: torch.Tensor
    sequence: torch.Tensor
    market: torch.Tensor
    target: torch.Tensor
    group_sizes: tuple[int, ...]
    row_positions: np.ndarray
    date_indices: tuple[int, ...]


def mask_static_columns(
    values: np.ndarray, columns: Sequence[int] | np.ndarray
) -> np.ndarray:
    """Return a float32 batch with selected live-unavailable fields set to NaN."""

    result = np.asarray(values, dtype=np.float32)
    selected = np.asarray(columns, dtype=np.int32)
    if not len(selected):
        return result
    if (
        result.ndim != 2
        or int(selected.min()) < 0
        or int(selected.max()) >= result.shape[1]
    ):
        raise SequenceChallengerError("masked_static_column_out_of_range")
    result = result.copy()
    result[:, selected] = np.nan
    return result


def apply_static_feature_contract(
    values: np.ndarray,
    *,
    row_positions: np.ndarray,
    masked_columns: Sequence[int] | np.ndarray = (),
    override: StaticFeatureOverride | None = None,
) -> np.ndarray:
    """Apply a row-aligned feature repair and then the frozen availability mask."""

    result = np.asarray(values, dtype=np.float32)
    rows = np.asarray(row_positions, dtype=np.int64)
    masked = np.asarray(masked_columns, dtype=np.int32)
    if result.ndim != 2 or len(result) != len(rows):
        raise SequenceChallengerError("static_feature_batch_alignment_failed")
    if override is None and not len(masked):
        return result
    result = result.copy()
    if override is not None:
        columns = np.asarray(override.columns, dtype=np.int32)
        if (
            override.values.ndim != 2
            or override.values.shape[0] <= int(rows.max(initial=-1))
            or len(columns) != override.values.shape[1]
            or int(columns.min(initial=0)) < 0
            or int(columns.max(initial=-1)) >= result.shape[1]
        ):
            raise SequenceChallengerError("static_feature_override_contract_failed")
        result[:, columns] = np.asarray(override.values[rows], dtype=np.float32)
    if len(masked):
        if int(masked.min()) < 0 or int(masked.max()) >= result.shape[1]:
            raise SequenceChallengerError("masked_static_column_out_of_range")
        result[:, masked] = np.nan
    return result


def _rank_percentile_columns(values: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of the frozen per-column average percentile rank."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise SequenceChallengerError("rank_values_must_be_two_dimensional")
    finite = np.isfinite(array)
    counts = finite.sum(axis=0)
    ranks = stats.rankdata(array, method="average", axis=0, nan_policy="omit")
    output = np.full(array.shape, np.nan, dtype=np.float64)
    multiple = counts > 1
    output[:, multiple] = (ranks[:, multiple] - 1.0) / (counts[multiple] - 1.0)
    single = counts == 1
    if bool(single.any()):
        output[:, single] = np.where(finite[:, single], 1.0, np.nan)
    return output.astype(np.float32)


def frozen_median(values: Sequence[int | float]) -> int | float:
    """Choose the deterministic middle OOF setting without a new outcome search."""

    ordered = sorted(values)
    if not ordered:
        raise SequenceChallengerError("frozen_median_support_empty")
    return ordered[len(ordered) // 2]


class BatchAssembler:
    def __init__(
        self,
        sources: SequenceSources,
        *,
        lookback: int,
        normalization: Normalization,
        device: torch.device,
        require_targets: bool = True,
        masked_static_columns: Sequence[int] = (),
        static_feature_override: StaticFeatureOverride | None = None,
    ) -> None:
        self.sources = sources
        self.lookback = int(lookback)
        self.normalization = normalization
        self.device = device
        self.require_targets = bool(require_targets)
        self.masked_static_columns = np.asarray(
            tuple(int(value) for value in masked_static_columns), dtype=np.int32
        )
        self.static_feature_override = static_feature_override
        if len(self.masked_static_columns) and (
            int(self.masked_static_columns.min()) < 0
            or int(self.masked_static_columns.max()) >= len(self.sources.feature_names)
        ):
            raise SequenceChallengerError("masked_static_column_out_of_range")

    def _sequence(self, date_idx: int, symbols: np.ndarray) -> np.ndarray:
        start = int(date_idx) - self.lookback + 1
        if start < 0:
            raise SequenceChallengerError("sequence_lookback_before_pack")
        raw = np.asarray(
            self.sources.daily_raw[start : date_idx + 1, symbols, :],
            dtype=np.float32,
        )
        state = np.asarray(
            self.sources.daily_state[start : date_idx + 1, symbols, :],
            dtype=np.float32,
        )
        turnover = np.asarray(
            self.sources.turnover[start : date_idx + 1, symbols, :],
            dtype=np.float32,
        )
        current_close = raw[-1, :, 3]
        price_relative = np.divide(
            raw[:, :, :4],
            current_close[None, :, None],
            out=np.full_like(raw[:, :, :4], np.nan),
            where=np.isfinite(current_close[None, :, None])
            & (current_close[None, :, None] > 0.0),
        ) - np.float32(1.0)
        sequence = np.concatenate(
            [
                price_relative,
                raw[:, :, DAILY_RAW_DERIVED_SLICE],
                state,
                turnover,
            ],
            axis=2,
        ).transpose(1, 0, 2)
        sequence = (
            sequence - self.normalization.sequence_mean[None, None, :]
        ) / self.normalization.sequence_scale[None, None, :]
        return np.nan_to_num(sequence, nan=0.0, posinf=10.0, neginf=-10.0).clip(
            -10.0, 10.0
        )

    def load(self, date_indices: Sequence[int]) -> Batch | None:
        static_parts: list[np.ndarray] = []
        sequence_parts: list[np.ndarray] = []
        target_parts: list[np.ndarray] = []
        positions_parts: list[np.ndarray] = []
        market_parts: list[np.ndarray] = []
        group_sizes: list[int] = []
        accepted_dates: list[int] = []
        for date_idx in date_indices:
            block = self.sources.blocks[int(date_idx)]
            rows = np.arange(block.start, block.stop, dtype=np.int64)
            valid = (
                np.asarray(
                    self.sources.exact_valid[rows, self.sources.base_column],
                    dtype=bool,
                )
                & np.asarray(
                    self.sources.exact_valid[rows, self.sources.stress_column],
                    dtype=bool,
                )
                if self.require_targets
                else np.ones(len(rows), dtype=bool)
            )
            rows = rows[valid]
            if len(rows) < 2:
                continue
            symbols = self.sources.context.row_index.iloc[rows]["symbol_idx"].to_numpy(
                dtype=np.int32
            )
            static_parts.append(
                apply_static_feature_contract(
                    np.asarray(self.sources.matrix[rows], dtype=np.float32),
                    row_positions=rows,
                    masked_columns=self.masked_static_columns,
                    override=self.static_feature_override,
                )
            )
            sequence_parts.append(self._sequence(int(date_idx), symbols))
            target_parts.append(
                np.nan_to_num(
                    np.asarray(
                        self.sources.exact_values[rows, self.sources.base_column],
                        dtype=np.float32,
                    ),
                    nan=0.0,
                )
            )
            market_raw = np.asarray(
                self.sources.matrix[block.start, self.sources.market_columns],
                dtype=np.float32,
            )
            market_parts.append(
                np.nan_to_num(
                    (market_raw - self.normalization.market_center)
                    / self.normalization.market_scale,
                    nan=0.0,
                    posinf=10.0,
                    neginf=-10.0,
                ).clip(-10.0, 10.0)
            )
            positions_parts.append(rows)
            group_sizes.append(len(rows))
            accepted_dates.append(int(date_idx))
        if not group_sizes:
            return None
        return Batch(
            static=torch.from_numpy(np.concatenate(static_parts)).to(self.device),
            sequence=torch.from_numpy(np.concatenate(sequence_parts)).to(self.device),
            market=torch.from_numpy(np.stack(market_parts)).to(self.device),
            target=torch.from_numpy(np.concatenate(target_parts)).to(self.device),
            group_sizes=tuple(group_sizes),
            row_positions=np.concatenate(positions_parts),
            date_indices=tuple(accepted_dates),
        )


def cross_sectional_standardize(
    values: torch.Tensor, group_sizes: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Robustly bound per-date current features while preserving missingness."""

    output = torch.empty_like(values)
    missing = torch.empty_like(values)
    offset = 0
    for size in group_sizes:
        current = values[offset : offset + int(size)]
        finite = torch.isfinite(current)
        safe = torch.where(finite, current, torch.zeros_like(current))
        count = finite.sum(dim=0).clamp_min(1)
        mean = safe.sum(dim=0) / count
        centered = torch.where(finite, current - mean, torch.zeros_like(current))
        variance = centered.square().sum(dim=0) / count
        scale = variance.clamp_min(1.0e-6).sqrt()
        output[offset : offset + int(size)] = (centered / scale).clamp(-8.0, 8.0)
        missing[offset : offset + int(size)] = (~finite).to(values.dtype)
        offset += int(size)
    return output, missing


class PayoffSequenceModel(nn.Module):
    def __init__(
        self,
        *,
        static_dim: int = 557,
        sequence_dim: int = 32,
        market_dim: int = 54,
        hidden_dim: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.static = nn.Sequential(
            nn.Linear(static_dim, 96),
            nn.GELU(),
            nn.LayerNorm(96),
            nn.Dropout(dropout),
        )
        self.missing = nn.Sequential(
            nn.Linear(static_dim, 16),
            nn.GELU(),
        )
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
        self.stock_head = nn.Sequential(
            nn.Linear(96 + 16 + hidden_dim + 32, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )
        self.market_return_head = nn.Linear(32, 1)
        self.market_positive_head = nn.Linear(32, 1)

    def forward(
        self,
        static: torch.Tensor,
        missing: torch.Tensor,
        sequence: torch.Tensor,
        market: torch.Tensor,
        group_sizes: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        stock = self.stock_head(
            torch.cat(
                [
                    static_embedding,
                    missing_embedding,
                    sequence_embedding,
                    repeated_market,
                ],
                dim=1,
            )
        ).squeeze(1)
        return (
            stock,
            self.market_return_head(market_embedding).squeeze(1),
            self.market_positive_head(market_embedding).squeeze(1),
        )


def payoff_loss(
    stock_score: torch.Tensor,
    market_return: torch.Tensor,
    market_logit: torch.Tensor,
    target: torch.Tensor,
    group_sizes: Sequence[int],
) -> tuple[torch.Tensor, dict[str, float]]:
    rank_losses: list[torch.Tensor] = []
    relative_losses: list[torch.Tensor] = []
    market_targets: list[torch.Tensor] = []
    offset = 0
    for size in group_sizes:
        current_target = target[offset : offset + int(size)]
        current_score = stock_score[offset : offset + int(size)]
        market_target = current_target.mean()
        relative = ((current_target - market_target) / RELATIVE_TARGET_SCALE).clamp(
            -3.0, 3.0
        )
        score_centered = current_score - current_score.mean()
        target_centered = relative - relative.mean()
        denominator = (
            score_centered.square().sum().clamp_min(1.0e-8).sqrt()
            * target_centered.square().sum().clamp_min(1.0e-8).sqrt()
        )
        rank_losses.append(1.0 - (score_centered * target_centered).sum() / denominator)
        relative_losses.append(F.smooth_l1_loss(current_score, relative))
        market_targets.append(market_target / MARKET_TARGET_SCALE)
        offset += int(size)
    market_target_tensor = torch.stack(market_targets)
    rank_loss = torch.stack(rank_losses).mean()
    relative_loss = torch.stack(relative_losses).mean()
    market_regression = F.smooth_l1_loss(market_return, market_target_tensor)
    market_binary = F.binary_cross_entropy_with_logits(
        market_logit, (market_target_tensor > 0.0).to(market_logit.dtype)
    )
    total = (
        rank_loss
        + 0.15 * relative_loss
        + 0.50 * market_regression
        + 0.20 * market_binary
    )
    return total, {
        "rank_loss": float(rank_loss.detach().cpu()),
        "relative_loss": float(relative_loss.detach().cpu()),
        "market_regression_loss": float(market_regression.detach().cpu()),
        "market_binary_loss": float(market_binary.detach().cpu()),
    }


def _chunks(values: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(values), int(size)):
        yield values[start : start + int(size)]


def _train_epoch(
    model: PayoffSequenceModel,
    assembler: BatchAssembler,
    date_indices: Sequence[int],
    *,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    date_batch_size: int,
    seed: int,
) -> dict[str, float]:
    model.train()
    dates = [int(value) for value in date_indices]
    np.random.default_rng(seed).shuffle(dates)
    totals: dict[str, float] = {
        "loss": 0.0,
        "rank_loss": 0.0,
        "relative_loss": 0.0,
        "market_regression_loss": 0.0,
        "market_binary_loss": 0.0,
    }
    batch_count = 0
    use_amp = assembler.device.type == "cuda"
    for current_dates in _chunks(dates, date_batch_size):
        batch = assembler.load(current_dates)
        if batch is None:
            continue
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=assembler.device.type, enabled=use_amp):
            static, missing = cross_sectional_standardize(
                batch.static, batch.group_sizes
            )
            stock_score, market_return, market_logit = model(
                static,
                missing,
                batch.sequence,
                batch.market,
                batch.group_sizes,
            )
            loss, components = payoff_loss(
                stock_score,
                market_return,
                market_logit,
                batch.target,
                batch.group_sizes,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        totals["loss"] += float(loss.detach().cpu())
        for key, value in components.items():
            totals[key] += value
        batch_count += 1
        del (
            batch,
            static,
            missing,
            stock_score,
            market_return,
            market_logit,
            loss,
        )
        if (
            batch_count % 4 == 0
            and base.psutil.virtual_memory().available < MEMORY_TRIM_THRESHOLD_BYTES
        ):
            gc.collect()
            base._trim_working_set()
    if not batch_count:
        raise SequenceChallengerError("training_epoch_empty")
    return {key: value / batch_count for key, value in totals.items()}


@dataclass
class PredictionOutput:
    rows: np.ndarray
    stock_score: np.ndarray
    market_return: dict[int, float]
    market_probability: dict[int, float]


@torch.no_grad()
def _predict_dates(
    model: PayoffSequenceModel,
    assembler: BatchAssembler,
    date_indices: Sequence[int],
    *,
    date_batch_size: int,
) -> PredictionOutput:
    model.eval()
    row_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    market_returns: dict[int, float] = {}
    market_probabilities: dict[int, float] = {}
    use_amp = assembler.device.type == "cuda"
    for current_dates in _chunks(list(date_indices), date_batch_size):
        batch = assembler.load(current_dates)
        if batch is None:
            continue
        with torch.amp.autocast(device_type=assembler.device.type, enabled=use_amp):
            static, missing = cross_sectional_standardize(
                batch.static, batch.group_sizes
            )
            stock_score, market_return, market_logit = model(
                static,
                missing,
                batch.sequence,
                batch.market,
                batch.group_sizes,
            )
        row_parts.append(batch.row_positions)
        score_parts.append(stock_score.float().cpu().numpy())
        for index, date_idx in enumerate(batch.date_indices):
            market_returns[int(date_idx)] = (
                float(market_return[index].float().cpu()) * MARKET_TARGET_SCALE
            )
            market_probabilities[int(date_idx)] = float(
                torch.sigmoid(market_logit[index].float()).cpu()
            )
        del batch, static, missing, stock_score, market_return, market_logit
        if (
            len(row_parts) % 4 == 0
            and base.psutil.virtual_memory().available < MEMORY_TRIM_THRESHOLD_BYTES
        ):
            gc.collect()
            base._trim_working_set()
    if not row_parts:
        raise SequenceChallengerError("prediction_dates_empty")
    return PredictionOutput(
        rows=np.concatenate(row_parts),
        stock_score=np.concatenate(score_parts),
        market_return=market_returns,
        market_probability=market_probabilities,
    )


def _rank_correlation(actual: np.ndarray, prediction: np.ndarray) -> float:
    finite = np.isfinite(actual) & np.isfinite(prediction)
    if int(finite.sum()) < 2:
        return math.nan
    actual_rank = pd.Series(actual[finite]).rank(pct=True).to_numpy(dtype=np.float64)
    prediction_rank = (
        pd.Series(prediction[finite]).rank(pct=True).to_numpy(dtype=np.float64)
    )
    if np.std(actual_rank) <= 0.0 or np.std(prediction_rank) <= 0.0:
        return 0.0
    return float(np.corrcoef(actual_rank, prediction_rank)[0, 1])


def _validation_metrics(
    sources: SequenceSources, prediction: PredictionOutput
) -> dict[str, float | int]:
    dates = sources.context.row_index.iloc[prediction.rows]["date_idx"].to_numpy(
        dtype=np.int32
    )
    actual = np.asarray(
        sources.exact_values[prediction.rows, sources.base_column], dtype=np.float64
    )
    rank_values: list[float] = []
    market_actual: list[float] = []
    market_prediction: list[float] = []
    market_probability: list[float] = []
    top10_stress: list[float] = []
    offset = 0
    for date_idx, size in zip(*np.unique(dates, return_counts=True), strict=True):
        current = slice(offset, offset + int(size))
        rank_values.append(
            _rank_correlation(actual[current], prediction.stock_score[current])
        )
        market_actual.append(float(actual[current].mean()))
        market_prediction.append(prediction.market_return[int(date_idx)])
        market_probability.append(prediction.market_probability[int(date_idx)])
        order = np.argsort(prediction.stock_score[current], kind="stable")[-10:]
        rows = prediction.rows[current][order]
        stress = np.asarray(
            sources.exact_values[rows, sources.stress_column], dtype=np.float64
        )
        gate = bool(
            prediction.market_return[int(date_idx)] > 0.0
            and prediction.market_probability[int(date_idx)] > 0.5
        )
        top10_stress.append(float(stress.mean()) if gate else 0.0)
        offset += int(size)
    binary = np.asarray(market_actual) > 0.0
    probability = np.asarray(market_probability)
    auc = (
        float(roc_auc_score(binary, probability))
        if len(np.unique(binary)) == 2
        else 0.5
    )
    rank_ic = float(np.mean(rank_values))
    market_mae = float(
        np.mean(np.abs(np.asarray(market_prediction) - np.asarray(market_actual)))
    )
    selection_score = rank_ic + 0.05 * (auc - 0.5) - 2.0 * market_mae
    return {
        "date_count": len(rank_values),
        "daily_rank_ic_mean": rank_ic,
        "daily_rank_ic_positive_fraction": float(
            np.mean(np.asarray(rank_values) > 0.0)
        ),
        "market_auc": auc,
        "market_mae": market_mae,
        "consensus_top10_stress_mean": float(np.mean(top10_stress)),
        "selection_score": float(selection_score),
    }


def _build_evaluation_frames(
    sources: SequenceSources,
    prediction: PredictionOutput,
    *,
    fold: int,
    unfilled_as_cash: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = sources.context.row_index.iloc[prediction.rows][
        ["candidate_id", "date_idx", "trade_date", "symbol", "symbol_idx"]
    ].copy()
    frame["model_row_position"] = prediction.rows
    frame["fold"] = int(fold)
    frame["prediction"] = prediction.stock_score
    exact_base = np.asarray(
        sources.exact_values[prediction.rows, sources.base_column], dtype=np.float32
    )
    exact_stress = np.asarray(
        sources.exact_values[prediction.rows, sources.stress_column], dtype=np.float32
    )
    if unfilled_as_cash:
        date_indices = frame["date_idx"].to_numpy(dtype=np.int32)
        symbol_indices = frame["symbol_idx"].to_numpy(dtype=np.int32)
        entry_filled = np.asarray(
            sources.entry_filled[date_indices, symbol_indices], dtype=bool
        )
        exact_valid = np.asarray(
            sources.exact_valid[prediction.rows, sources.base_column], dtype=bool
        ) & np.asarray(
            sources.exact_valid[prediction.rows, sources.stress_column], dtype=bool
        )
        path_valid = np.asarray(
            sources.path_valid[prediction.rows, sources.gross_column], dtype=bool
        )
        frame["market_entry_filled"] = entry_filled
        # Exact-net validity additionally requires that a CNY100k slot can buy
        # at least one board lot. An unaffordable order is cash, not an unknown
        # outcome and not a reason to substitute the next-ranked stock.
        frame["entry_filled"] = exact_valid
        frame["outcome_known"] = ~entry_filled | path_valid
        frame["base"] = np.where(exact_valid, exact_base, 0.0)
        frame["stress"] = np.where(exact_valid, exact_stress, 0.0)
    else:
        frame["entry_filled"] = True
        frame["outcome_known"] = np.isfinite(exact_base) & np.isfinite(exact_stress)
        frame["base"] = exact_base
        frame["stress"] = exact_stress
    frame["market_return_prediction"] = frame["date_idx"].map(prediction.market_return)
    frame["market_positive_probability"] = frame["date_idx"].map(
        prediction.market_probability
    )
    frame = frame.sort_values(
        ["date_idx", "prediction", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    frame["score_position"] = frame.groupby("date_idx", sort=False).cumcount()
    frame["date_size"] = frame.groupby("date_idx", sort=False)[
        "candidate_id"
    ].transform("size")
    frame["decile"] = np.minimum(
        10,
        np.floor(
            frame["score_position"].to_numpy(dtype=np.float64)
            * 10.0
            / frame["date_size"].to_numpy(dtype=np.float64)
        ).astype(np.int8)
        + 1,
    )
    frame["selection_rank"] = (frame["date_size"] - frame["score_position"]).astype(
        np.int32
    )
    if unfilled_as_cash:
        top10_known = (
            frame.loc[frame["selection_rank"] <= 10]
            .groupby("date_idx", sort=False)["outcome_known"]
            .all()
        )
        eligible_dates = top10_known.index[top10_known].to_numpy(dtype=np.int32)
        frame = frame.loc[frame["date_idx"].isin(eligible_dates)].copy()
    baseline = frame.groupby(["date_idx", "trade_date"], sort=False)[
        ["base", "stress"]
    ].mean()
    daily = baseline.rename(
        columns={"base": "baseline_base", "stress": "baseline_stress"}
    ).reset_index()
    rank_ic = (
        frame.groupby("date_idx", sort=False)
        .apply(
            lambda current: _rank_correlation(
                current["base"].to_numpy(), current["prediction"].to_numpy()
            ),
            include_groups=False,
        )
        .rename("rank_ic")
        .reset_index()
    )
    daily = daily.merge(rank_ic, on="date_idx", validate="one_to_one")
    for top_k in (1, 3, 10):
        selected = frame.loc[frame["selection_rank"] <= top_k]
        means = selected.groupby("date_idx", sort=False)[["base", "stress"]].mean()
        daily = daily.merge(
            means.rename(
                columns={
                    "base": f"top{top_k}_base",
                    "stress": f"top{top_k}_stress",
                }
            ).reset_index(),
            on="date_idx",
            how="left",
            validate="one_to_one",
        )
    daily["fold"] = int(fold)
    deciles = (
        frame.groupby(["date_idx", "trade_date", "decile"], sort=False)[
            ["base", "stress"]
        ]
        .mean()
        .reset_index()
    )
    deciles["fold"] = int(fold)
    selections = frame.loc[frame["selection_rank"] <= 10].copy()
    summary = base._payoff_summary(daily, deciles)
    return daily, deciles, selections, summary


def _gated_metrics(daily: pd.DataFrame, selections: pd.DataFrame) -> dict[str, Any]:
    date_predictions = selections.groupby("date_idx", sort=False).agg(
        market_return_prediction=("market_return_prediction", "first"),
        market_positive_probability=("market_positive_probability", "first"),
    )
    outputs: dict[str, Any] = {}
    gates = {
        "regression": date_predictions["market_return_prediction"].gt(0.0),
        "probability": date_predictions["market_positive_probability"].gt(0.5),
        "consensus": date_predictions["market_return_prediction"].gt(0.0)
        & date_predictions["market_positive_probability"].gt(0.5),
    }
    for gate_name, gate in gates.items():
        for top_k in (1, 3, 10):
            current = selections.loc[selections["selection_rank"] <= top_k]
            returns = current.groupby("date_idx", sort=False)[["base", "stress"]].mean()
            result = daily[["date_idx", "trade_date", "fold"]].copy()
            result = result.merge(
                returns.reset_index(), on="date_idx", how="left", validate="one_to_one"
            )
            result[["base", "stress"]] = result[["base", "stress"]].fillna(0.0)
            active = result["date_idx"].map(gate).fillna(False).to_numpy(dtype=bool)
            result.loc[~active, ["base", "stress"]] = 0.0
            if "entry_filled" in current.columns:
                filled = current.groupby("date_idx", sort=False)["entry_filled"].sum()
                filled_count = (
                    result["date_idx"].map(filled).fillna(0).to_numpy(dtype=np.int32)
                )
                result["trade_count"] = np.where(active, filled_count, 0)
            else:
                result["trade_count"] = np.where(active, top_k, 0)
            outputs[f"{gate_name}_top{top_k}"] = base._market_gate_metrics(result)
    return outputs


def _new_model(seed: int) -> PayoffSequenceModel:
    _set_determinism(seed)
    return PayoffSequenceModel()


def _fit_model(
    sources: SequenceSources,
    *,
    train_dates: Sequence[int],
    validation_dates: Sequence[int],
    lookback: int,
    device: torch.device,
    maximum_epochs: int,
    patience: int,
    date_batch_size: int,
    seed: int,
) -> tuple[int, list[dict[str, Any]], dict[str, torch.Tensor]]:
    normalization = _fit_normalization(sources, train_dates)
    assembler = BatchAssembler(
        sources, lookback=lookback, normalization=normalization, device=device
    )
    model = _new_model(seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(1, int(maximum_epochs) + 1):
        started = time.perf_counter()
        train_metrics = _train_epoch(
            model,
            assembler,
            train_dates,
            optimizer=optimizer,
            scaler=scaler,
            date_batch_size=date_batch_size,
            seed=seed + epoch,
        )
        prediction = _predict_dates(
            model,
            assembler,
            validation_dates,
            date_batch_size=date_batch_size,
        )
        validation_metrics = _validation_metrics(sources, prediction)
        row = {
            "epoch": epoch,
            "elapsed_seconds": float(time.perf_counter() - started),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        base._emit("sequence_inner_epoch_completed", **row)
        score = float(validation_metrics["selection_score"])
        if score > best_score + 1.0e-5:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_state is None or best_epoch <= 0:
        raise SequenceChallengerError("inner_model_selection_failed")
    del model, optimizer, scaler
    gc.collect()
    base._trim_working_set()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, history, best_state


def _refit_model(
    sources: SequenceSources,
    *,
    train_dates: Sequence[int],
    lookback: int,
    device: torch.device,
    epochs: int,
    date_batch_size: int,
    seed: int,
) -> tuple[PayoffSequenceModel, Normalization, list[dict[str, float]]]:
    normalization = _fit_normalization(sources, train_dates)
    assembler = BatchAssembler(
        sources, lookback=lookback, normalization=normalization, device=device
    )
    model = _new_model(seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history: list[dict[str, float]] = []
    for epoch in range(1, int(epochs) + 1):
        started = time.perf_counter()
        metrics = _train_epoch(
            model,
            assembler,
            train_dates,
            optimizer=optimizer,
            scaler=scaler,
            date_batch_size=date_batch_size,
            seed=seed + 10_000 + epoch,
        )
        row = {
            "epoch": float(epoch),
            "elapsed_seconds": float(time.perf_counter() - started),
            **metrics,
        }
        history.append(row)
        base._emit("sequence_outer_refit_epoch_completed", **row)
    return model, normalization, history


def train_fold(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    fold_number: int = 1,
    lookback: int = DEFAULT_LOOKBACK,
    maximum_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    date_batch_size: int = DEFAULT_DATE_BATCH_SIZE,
) -> dict[str, Any]:
    study = base.load_study(study_path)
    if int(lookback) not in tuple(study["sequence_challengers"]["daily_lookbacks"]):
        raise SequenceChallengerError(f"unsupported_lookback:{lookback}")
    if not torch.cuda.is_available():
        raise SequenceChallengerError("cuda_is_required_for_formal_sequence_run")
    resource = base.resource_plan(
        reserve_gib=float(study["resources"]["reserve_system_gib"]),
        maximum_threads=int(study["resources"]["maximum_cpu_threads"]),
        histogram_pool_cap_mb=int(study["resources"]["histogram_pool_cap_mb"]),
        sequence_batch_cap=int(study["resources"]["sequence_batch_cap"]),
    )
    base._assert_resource_capacity(resource, 1 * (1 << 30))
    sources = _load_sources(study, output_root=output_root)
    folds = base.build_forward_folds(
        date_idx=sources.context.row_index["date_idx"].to_numpy(dtype=np.int32),
        trade_date=sources.context.row_index["trade_date"].astype(str).to_numpy(),
        validation_start_date=str(study["validation"]["validation_start_date"]),
        validation_end_date=str(study["validation"]["validation_end_date"]),
        fold_count=int(study["validation"]["forward_fold_count"]),
        purge_days=int(study["validation"]["common_purge_trading_days"]),
    )
    matches = [fold for fold in folds if int(fold["fold"]) == int(fold_number)]
    if len(matches) != 1:
        raise SequenceChallengerError(f"unknown_fold:{fold_number}")
    fold = matches[0]
    outer_train_dates = sorted(
        date_idx
        for date_idx in sources.blocks
        if date_idx <= int(fold["training_maximum_date_idx"])
    )
    outer_validation_dates = sorted(
        date_idx
        for date_idx in sources.blocks
        if int(fold["validation_start_date_idx"])
        <= date_idx
        <= int(fold["validation_end_date_idx"])
    )
    expanded_train_dates = np.repeat(
        np.asarray(outer_train_dates, dtype=np.int32),
        [
            sources.blocks[date_idx].stop - sources.blocks[date_idx].start
            for date_idx in outer_train_dates
        ],
    )
    inner_train_rows, inner_validation_rows, inner_metadata = base._nested_inner_rows(
        expanded_train_dates,
        validation_date_count=int(study["validation"]["inner_validation_trading_days"]),
        purge_days=int(study["validation"]["common_purge_trading_days"]),
    )
    inner_train_dates = sorted(
        np.unique(expanded_train_dates[inner_train_rows]).astype(int).tolist()
    )
    inner_validation_dates = sorted(
        np.unique(expanded_train_dates[inner_validation_rows]).astype(int).tolist()
    )
    del expanded_train_dates, inner_train_rows, inner_validation_rows
    fingerprint = _stable_hash(
        {
            "schema": SCHEMA,
            "implementation_sha256": base._sha256(Path(__file__)),
            "study_sha256": base._sha256(study_path),
            "target_fingerprint": sources.target_manifest["fingerprint"],
            "input_fingerprint": sources.context.model_manifest["input_fingerprint"],
            "fold": fold,
            "lookback": int(lookback),
            "maximum_epochs": int(maximum_epochs),
            "patience": int(patience),
            "date_batch_size": int(date_batch_size),
            "seed": int(study["validation"]["seed"]),
            "loss": {
                "rank": 1.0,
                "relative_huber": 0.15,
                "market_huber": 0.50,
                "market_binary": 0.20,
            },
        }
    )
    task_root = (
        output_root
        / "sequence_challenger"
        / f"lookback_{lookback}"
        / f"fold_{fold_number}"
    )
    manifest_path = task_root / "task_result.json"
    if manifest_path.is_file():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            current.get("status") == "completed"
            and current.get("fingerprint") == fingerprint
        ):
            return current
    task_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    seed = int(study["validation"]["seed"]) + int(fold_number) * 100 + int(lookback)
    base._emit(
        "sequence_fold_started",
        fold=fold_number,
        lookback=lookback,
        inner_train_dates=len(inner_train_dates),
        inner_validation_dates=len(inner_validation_dates),
        outer_train_dates=len(outer_train_dates),
        outer_validation_dates=len(outer_validation_dates),
        resource_plan=asdict(resource),
    )
    with base.ResourceMonitor() as monitor:
        best_epoch, selection_history, _ = _fit_model(
            sources,
            train_dates=inner_train_dates,
            validation_dates=inner_validation_dates,
            lookback=lookback,
            device=device,
            maximum_epochs=maximum_epochs,
            patience=patience,
            date_batch_size=date_batch_size,
            seed=seed,
        )
        model, normalization, refit_history = _refit_model(
            sources,
            train_dates=outer_train_dates,
            lookback=lookback,
            device=device,
            epochs=best_epoch,
            date_batch_size=date_batch_size,
            seed=seed + 1,
        )
        validation_assembler = BatchAssembler(
            sources,
            lookback=lookback,
            normalization=normalization,
            device=device,
        )
        prediction = _predict_dates(
            model,
            validation_assembler,
            outer_validation_dates,
            date_batch_size=date_batch_size,
        )
        resource_metrics = monitor.metrics()
    daily, deciles, selections, payoff_summary = _build_evaluation_frames(
        sources, prediction, fold=fold_number
    )
    gated = _gated_metrics(daily, selections)
    prediction_array = np.full(
        len(base._fold_rows(sources.context.row_index, fold, "validation")),
        np.nan,
        dtype=np.float32,
    )
    validation_positions = base._fold_rows(
        sources.context.row_index, fold, "validation"
    )
    local = np.searchsorted(validation_positions, prediction.rows)
    if bool((local >= len(validation_positions)).any()) or not np.array_equal(
        validation_positions[local], prediction.rows
    ):
        raise SequenceChallengerError("validation_prediction_alignment_failed")
    prediction_array[local] = prediction.stock_score
    prediction_path = task_root / "prediction.npy"
    with prediction_path.with_suffix(".npy.partial").open("wb") as stream:
        np.save(stream, prediction_array, allow_pickle=False)
    os.replace(prediction_path.with_suffix(".npy.partial"), prediction_path)
    market_path = task_root / "market_predictions.parquet"
    evaluated_dates = sorted(prediction.market_return)
    market_frame = pd.DataFrame(
        {
            "date_idx": evaluated_dates,
            "trade_date": [
                sources.blocks[value].trade_date for value in evaluated_dates
            ],
            "predicted_return": [
                prediction.market_return[value] for value in evaluated_dates
            ],
            "positive_probability": [
                prediction.market_probability[value] for value in evaluated_dates
            ],
        }
    )
    base._write_parquet(market_frame, market_path)
    daily_path = task_root / "daily_metrics.parquet"
    decile_path = task_root / "decile_daily.parquet"
    selections_path = task_root / "top10_selections.parquet"
    base._write_parquet(daily, daily_path)
    base._write_parquet(deciles, decile_path)
    base._write_parquet(selections, selections_path)
    checkpoint_path = task_root / "model.pt"
    checkpoint_partial = checkpoint_path.with_suffix(".pt.partial")
    torch.save(
        {
            "schema": SCHEMA,
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "normalization": normalization.payload(),
            "lookback": int(lookback),
            "feature_names": sources.feature_names,
            "sequence_feature_names": sources.pack_feature_names,
        },
        checkpoint_partial,
    )
    os.replace(checkpoint_partial, checkpoint_path)
    history_path = task_root / "training_history.json"
    _write_json(
        history_path,
        {"selection": selection_history, "outer_refit": refit_history},
    )
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "fingerprint": fingerprint,
        "study_id": base.STUDY_ID,
        "fold": fold,
        "lookback": int(lookback),
        "best_epoch": int(best_epoch),
        "nested_selection": {
            **inner_metadata,
            "outer_validation_used_for_epoch_selection": False,
        },
        "model": {
            "static_feature_count": 557,
            "sequence_feature_count": 32,
            "market_feature_count": 54,
            "parameter_count": int(sum(value.numel() for value in model.parameters())),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0),
        },
        "payoff_summary": payoff_summary,
        "market_gates": gated,
        "resource_plan": asdict(resource),
        "resource_metrics": resource_metrics,
        "forbidden_2026_read_count": 0,
        "account_replay_performed": False,
        "files": {
            "model": _file_record(checkpoint_path),
            "prediction": _file_record(
                prediction_path, shape=list(prediction_array.shape)
            ),
            "market_predictions": _file_record(
                market_path, row_count=len(market_frame)
            ),
            "daily_metrics": _file_record(daily_path, row_count=len(daily)),
            "decile_daily": _file_record(decile_path, row_count=len(deciles)),
            "top10_selections": _file_record(
                selections_path, row_count=len(selections)
            ),
            "training_history": _file_record(history_path),
        },
    }
    _write_json(manifest_path, result)
    base._emit(
        "sequence_fold_completed",
        fold=fold_number,
        lookback=lookback,
        best_epoch=best_epoch,
        rank_ic=payoff_summary["daily_rank_ic_mean"],
        consensus_top10_stress=gated["consensus_top10"]["stress"]["mean"],
    )
    return result


def _completed_sequence_task(
    output_root: Path, *, fold: int, lookback: int
) -> tuple[dict[str, Any], Path]:
    path = (
        output_root
        / "sequence_challenger"
        / f"lookback_{int(lookback)}"
        / f"fold_{int(fold)}"
        / "task_result.json"
    )
    if not path.is_file():
        raise SequenceChallengerError(f"sequence_task_missing:{path}")
    task = json.loads(path.read_text(encoding="utf-8"))
    if (
        task.get("status") != "completed"
        or int(task.get("fold", {}).get("fold", -1)) != int(fold)
        or int(task.get("lookback", -1)) != int(lookback)
    ):
        raise SequenceChallengerError(f"sequence_task_invalid:{path}")
    return task, path


def expand_fold_predictions(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    fold_number: int,
    lookback: int = DEFAULT_LOOKBACK,
    date_batch_size: int = DEFAULT_DATE_BATCH_SIZE,
) -> dict[str, Any]:
    """Score the complete signal-day universe without future fill filtering."""

    study = base.load_study(study_path)
    sources = _load_sources(study, output_root=output_root)
    task, task_path = _completed_sequence_task(
        output_root, fold=fold_number, lookback=lookback
    )
    folds = base.build_forward_folds(
        date_idx=sources.context.row_index["date_idx"].to_numpy(dtype=np.int32),
        trade_date=sources.context.row_index["trade_date"].astype(str).to_numpy(),
        validation_start_date=str(study["validation"]["validation_start_date"]),
        validation_end_date=str(study["validation"]["validation_end_date"]),
        fold_count=int(study["validation"]["forward_fold_count"]),
        purge_days=int(study["validation"]["common_purge_trading_days"]),
    )
    fold = next((item for item in folds if int(item["fold"]) == int(fold_number)), None)
    if fold is None:
        raise SequenceChallengerError(f"unknown_fold:{fold_number}")
    checkpoint_path = Path(task["files"]["model"]["path"])
    output_path = checkpoint_path.parent / "prediction_full_universe.npy"
    expansion_fingerprint = _stable_hash(
        {
            "schema": SCHEMA,
            "role": "full_signal_day_universe_prediction",
            "checkpoint_sha256": base._sha256(checkpoint_path),
            "row_index_sha256": sources.context.model_manifest["row_index"]["sha256"],
            "fold": fold,
            "lookback": int(lookback),
            "entry_fill_or_outcome_used_for_ranking": False,
        }
    )
    current_expansion = dict(task.get("full_universe_expansion", {}) or {})
    if (
        current_expansion.get("fingerprint") == expansion_fingerprint
        and output_path.is_file()
        and int(output_path.stat().st_size)
        == int(
            task.get("files", {}).get("full_universe_prediction", {}).get("size", -1)
        )
    ):
        return task
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != SCHEMA or int(checkpoint["lookback"]) != int(
        lookback
    ):
        raise SequenceChallengerError("sequence_checkpoint_contract_failed")
    normalization = Normalization.from_payload(checkpoint["normalization"])
    model = PayoffSequenceModel()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    assembler = BatchAssembler(
        sources,
        lookback=lookback,
        normalization=normalization,
        device=device,
        require_targets=False,
    )
    validation_dates = sorted(
        date_idx
        for date_idx in sources.blocks
        if int(fold["validation_start_date_idx"])
        <= date_idx
        <= int(fold["validation_end_date_idx"])
    )
    prediction = _predict_dates(
        model,
        assembler,
        validation_dates,
        date_batch_size=date_batch_size,
    )
    validation_positions = base._fold_rows(
        sources.context.row_index, fold, "validation"
    )
    output = np.full(len(validation_positions), np.nan, dtype=np.float32)
    local = np.searchsorted(validation_positions, prediction.rows)
    if (
        bool((local >= len(validation_positions)).any())
        or not np.array_equal(validation_positions[local], prediction.rows)
        or len(prediction.rows) != len(validation_positions)
    ):
        raise SequenceChallengerError("full_universe_prediction_alignment_failed")
    output[local] = prediction.stock_score
    if not np.isfinite(output).all():
        raise SequenceChallengerError("full_universe_prediction_nonfinite")
    partial = output_path.with_suffix(".npy.partial")
    with partial.open("wb") as stream:
        np.save(stream, output, allow_pickle=False)
    os.replace(partial, output_path)
    task["files"]["full_universe_prediction"] = _file_record(
        output_path, shape=list(output.shape)
    )
    task["full_universe_expansion"] = {
        "fingerprint": expansion_fingerprint,
        "completed_at": _now(),
        "row_count": len(output),
        "entry_fill_or_outcome_used_for_ranking": False,
        "forbidden_2026_read_count": 0,
    }
    _write_json(task_path, task)
    base._emit(
        "sequence_full_universe_prediction_completed",
        fold=fold_number,
        rows=len(output),
    )
    return task


def _formal_market_predictions(output_root: Path) -> tuple[pd.DataFrame, Path]:
    manifest_path = output_root / "market_regime_evaluation" / "manifest.json"
    if not manifest_path.is_file():
        raise SequenceChallengerError("formal_market_regime_evaluation_missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise SequenceChallengerError("formal_market_regime_evaluation_incomplete")
    prediction_path = Path(manifest["files"]["market_predictions"]["path"])
    frame = pd.read_parquet(prediction_path)
    required = {
        "date_idx",
        "ridge_prediction",
        "positive_probability",
        "consensus_gate",
    }
    if not required.issubset(frame.columns) or frame["date_idx"].duplicated().any():
        raise SequenceChallengerError("formal_market_prediction_contract_failed")
    expected_consensus = frame["ridge_prediction"].gt(0.0) & frame[
        "positive_probability"
    ].gt(0.5)
    if not np.array_equal(
        expected_consensus.to_numpy(dtype=bool),
        frame["consensus_gate"].to_numpy(dtype=bool),
    ):
        raise SequenceChallengerError("formal_market_consensus_definition_changed")
    return frame.set_index("date_idx"), manifest_path


def _evaluation_prediction(
    *,
    sources: SequenceSources,
    validation_positions: np.ndarray,
    sequence_prediction: np.ndarray,
    tree_prediction: np.ndarray,
    market: pd.DataFrame,
) -> tuple[PredictionOutput, pd.DataFrame]:
    if len(sequence_prediction) != len(validation_positions) or len(
        tree_prediction
    ) != len(validation_positions):
        raise SequenceChallengerError("ensemble_prediction_row_count_mismatch")
    validation_dates = sources.context.row_index.iloc[validation_positions][
        "date_idx"
    ].to_numpy(dtype=np.int32)
    valid = (
        np.isfinite(sequence_prediction)
        & np.isfinite(tree_prediction)
        & np.isin(validation_dates, market.index.to_numpy(dtype=np.int32))
    )
    rows = validation_positions[valid]
    components = sources.context.row_index.iloc[rows][
        ["candidate_id", "date_idx"]
    ].copy()
    components["model_row_position"] = rows
    components["sequence_prediction"] = np.asarray(
        sequence_prediction[valid], dtype=np.float32
    )
    components["tree_prediction"] = np.asarray(tree_prediction[valid], dtype=np.float32)
    components["sequence_rank"] = components.groupby("date_idx", sort=False)[
        "sequence_prediction"
    ].rank(pct=True)
    components["tree_rank"] = components.groupby("date_idx", sort=False)[
        "tree_prediction"
    ].rank(pct=True)
    components["ensemble_score"] = (
        components["sequence_rank"] + components["tree_rank"]
    ) / 2.0
    dates = sorted(components["date_idx"].astype(int).unique().tolist())
    missing_market = sorted(set(dates).difference(market.index.astype(int)))
    if missing_market:
        raise SequenceChallengerError(
            f"formal_market_dates_missing:{missing_market[:5]}"
        )
    prediction = PredictionOutput(
        rows=rows,
        stock_score=components["ensemble_score"].to_numpy(dtype=np.float32),
        market_return={
            date_idx: float(market.loc[date_idx, "ridge_prediction"])
            for date_idx in dates
        },
        market_probability={
            date_idx: float(market.loc[date_idx, "positive_probability"])
            for date_idx in dates
        },
    )
    return prediction, components


def evaluate_ensemble(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    lookback: int = DEFAULT_LOOKBACK,
) -> dict[str, Any]:
    """Evaluate the frozen equal-rank ensemble and pre-existing market gate."""

    study = base.load_study(study_path)
    sources = _load_sources(study, output_root=output_root)
    folds = base.build_forward_folds(
        date_idx=sources.context.row_index["date_idx"].to_numpy(dtype=np.int32),
        trade_date=sources.context.row_index["trade_date"].astype(str).to_numpy(),
        validation_start_date=str(study["validation"]["validation_start_date"]),
        validation_end_date=str(study["validation"]["validation_end_date"]),
        fold_count=int(study["validation"]["forward_fold_count"]),
        purge_days=int(study["validation"]["common_purge_trading_days"]),
    )
    market, market_manifest_path = _formal_market_predictions(output_root)
    for fold in folds:
        expand_fold_predictions(
            study_path=study_path,
            output_root=output_root,
            fold_number=int(fold["fold"]),
            lookback=lookback,
        )
    sequence_tasks = [
        _completed_sequence_task(output_root, fold=int(fold["fold"]), lookback=lookback)
        for fold in folds
    ]
    tree_tasks = [
        base._oof_task_result(
            output_root,
            fold=int(fold["fold"]),
            target="exact_net_return_d5_rank",
            profile=str(study["lightgbm"]["primary_profile"]),
            feature_variant=base.DEFAULT_FEATURE_VARIANT,
            training_mode="causal_nested",
        )
        for fold in folds
    ]
    fingerprint = _stable_hash(
        {
            "schema": ENSEMBLE_SCHEMA,
            "study_sha256": base._sha256(study_path),
            "target_fingerprint": sources.target_manifest["fingerprint"],
            "lookback": int(lookback),
            "stock_score": "equal_average_of_within_date_sequence_and_tree_percentile_ranks",
            "market_gate": "frozen_ridge_above_zero_and_logistic_probability_above_0p5",
            "execution_evaluation": "rank_full_signal_day_universe_then_unfilled_or_unaffordable_order_is_cash_without_substitution",
            "top_k": [1, 3, 10],
            "sequence_tasks": {
                str(path): base._sha256(path) for _, path in sequence_tasks
            },
            "tree_tasks": {str(path): base._sha256(path) for _, path in tree_tasks},
            "market_manifest_sha256": base._sha256(market_manifest_path),
        }
    )
    evaluation_root = _lookback_artifact_root(
        output_root, "sequence_ensemble_evaluation", lookback=lookback
    )
    manifest_path = evaluation_root / "manifest.json"
    if manifest_path.is_file():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            current.get("status") == "completed"
            and current.get("fingerprint") == fingerprint
        ):
            return current
    daily_parts: list[pd.DataFrame] = []
    decile_parts: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []
    component_parts: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold, (sequence_task, _), (tree_task, _) in zip(
        folds, sequence_tasks, tree_tasks, strict=True
    ):
        fold_number = int(fold["fold"])
        positions = base._fold_rows(sources.context.row_index, fold, "validation")
        sequence_prediction = np.load(
            sequence_task["files"]["full_universe_prediction"]["path"],
            mmap_mode="r",
            allow_pickle=False,
        )
        tree_prediction = np.load(
            tree_task["files"]["prediction"]["path"],
            mmap_mode="r",
            allow_pickle=False,
        )
        prediction, components = _evaluation_prediction(
            sources=sources,
            validation_positions=positions,
            sequence_prediction=sequence_prediction,
            tree_prediction=tree_prediction,
            market=market,
        )
        daily, deciles, selections, summary = _build_evaluation_frames(
            sources,
            prediction,
            fold=fold_number,
            unfilled_as_cash=True,
        )
        component_values = components[
            [
                "model_row_position",
                "sequence_prediction",
                "tree_prediction",
                "sequence_rank",
                "tree_rank",
            ]
        ]
        selections = selections.merge(
            component_values,
            on="model_row_position",
            how="left",
            validate="one_to_one",
        )
        summary["fold"] = fold_number
        summary["market_gates"] = _gated_metrics(daily, selections)
        fold_summaries.append(summary)
        daily_parts.append(daily)
        decile_parts.append(deciles)
        selection_parts.append(selections)
        component_parts.append(components)
        base._emit(
            "sequence_ensemble_fold_evaluated",
            fold=fold_number,
            rank_ic=summary["daily_rank_ic_mean"],
            consensus_top10_stress=summary["market_gates"]["consensus_top10"]["stress"][
                "mean"
            ],
        )
    daily = pd.concat(daily_parts, ignore_index=True).sort_values("date_idx")
    deciles = pd.concat(decile_parts, ignore_index=True).sort_values(
        ["date_idx", "decile"]
    )
    selections = pd.concat(selection_parts, ignore_index=True).sort_values(
        ["date_idx", "selection_rank"]
    )
    components = pd.concat(component_parts, ignore_index=True).sort_values(
        ["date_idx", "candidate_id"]
    )
    combined_payoff = base._payoff_summary(daily, deciles)
    combined_gates = _gated_metrics(daily, selections)
    confirmation_daily = daily.loc[daily["fold"] >= 2].copy()
    confirmation_deciles = deciles.loc[deciles["fold"] >= 2].copy()
    confirmation_selections = selections.loc[selections["fold"] >= 2].copy()
    confirmation = {
        "scope": "folds_2_to_5_after_equal_rank_combination_was_frozen_on_fold_1",
        "payoff": base._payoff_summary(confirmation_daily, confirmation_deciles),
        "market_gates": _gated_metrics(confirmation_daily, confirmation_selections),
    }
    primary = combined_gates["consensus_top10"]
    positive_rank_folds = sum(
        float(item["daily_rank_ic_mean"]) > 0.0 for item in fold_summaries
    )
    account_gate = {
        "primary_variant": "equal_rank_ensemble__frozen_market_consensus__top10__d5",
        "requires_positive_rank_ic_all_folds": True,
        "requires_positive_stress_hac_lower_bound": True,
        "requires_positive_stress_all_folds": True,
        "requires_at_least_five_of_six_positive_years": True,
        "positive_rank_fold_count": int(positive_rank_folds),
        "positive_stress_fold_count": int(primary["positive_stress_fold_count"]),
        "positive_stress_year_count": int(primary["positive_stress_year_count"]),
        "stress_hac_lower_bound": primary["stress"]["lower"],
        "passed": bool(
            positive_rank_folds == len(folds)
            and int(primary["positive_stress_fold_count"]) == len(folds)
            and int(primary["positive_stress_year_count"]) >= 5
            and primary["stress"]["lower"] is not None
            and float(primary["stress"]["lower"]) > 0.0
        ),
    }
    # Attach the gross-return path needed by the no-leverage account simulator.
    selection_rows = selections["model_row_position"].to_numpy(dtype=np.int64)
    selections["legal_gross_return"] = np.asarray(
        sources.path_values[selection_rows, sources.gross_column], dtype=np.float32
    )
    selections["fill_day"] = np.asarray(
        sources.fill_days[selection_rows, sources.horizon_column], dtype=np.int16
    )
    selections["formal_market_consensus_gate"] = selections["date_idx"].map(
        market["consensus_gate"]
    )
    evaluation_root.mkdir(parents=True, exist_ok=True)
    daily_path = evaluation_root / "daily_metrics.parquet"
    decile_path = evaluation_root / "decile_daily.parquet"
    selections_path = evaluation_root / "top10_selections.parquet"
    components_path = evaluation_root / "component_predictions.parquet"
    base._write_parquet(daily, daily_path)
    base._write_parquet(deciles, decile_path)
    base._write_parquet(selections, selections_path)
    base._write_parquet(components, components_path)
    result = {
        "schema": ENSEMBLE_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": base.STUDY_ID,
        "fingerprint": fingerprint,
        "lookback": int(lookback),
        "frozen_definition": {
            "stock_score": "equal_average_of_within_date_sequence_and_tree_percentile_ranks",
            "tree_model": "exact_net_return_d5_rank__strong_127__causal_nested",
            "market_gate": "existing_ridge_prediction_above_zero_and_existing_logistic_probability_above_0p5",
            "holding": "next_open_to_d5_legal_exit",
            "primary_top_k": 10,
            "unfilled_or_unaffordable_order": "cash_without_rank_substitution",
            "weights_or_thresholds_tuned_after_fold_1": False,
        },
        "fold_summaries": fold_summaries,
        "combined": {
            "payoff": combined_payoff,
            "market_gates": combined_gates,
        },
        "confirmation": confirmation,
        "account_replay_gate": account_gate,
        "account_replay_performed": False,
        "forbidden_2026_read_count": 0,
        "files": {
            "daily_metrics": _file_record(daily_path, row_count=len(daily)),
            "decile_daily": _file_record(decile_path, row_count=len(deciles)),
            "top10_selections": _file_record(
                selections_path, row_count=len(selections)
            ),
            "component_predictions": _file_record(
                components_path, row_count=len(components)
            ),
        },
        "sources": {
            "market_regime": _file_record(market_manifest_path),
            "sequence_tasks": [_file_record(path) for _, path in sequence_tasks],
            "tree_tasks": [_file_record(path) for _, path in tree_tasks],
            "exact_net_targets": _file_record(
                output_root / "exact_net_targets" / "manifest.json"
            ),
            "path_targets": _file_record(output_root / "targets" / "manifest.json"),
        },
    }
    _write_json(manifest_path, result)
    base._emit(
        "sequence_ensemble_evaluation_completed",
        rank_ic=combined_payoff["daily_rank_ic_mean"],
        consensus_top10_stress=primary["stress"]["mean"],
        stress_lower=primary["stress"]["lower"],
        account_replay_gate=account_gate["passed"],
    )
    return result


def replay_ensemble_accounts(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    lookback: int = DEFAULT_LOOKBACK,
    starting_cash: float = 1_000_000.0,
) -> dict[str, Any]:
    """Replay the admitted D5 ensemble with real cash and overlapping holdings."""

    evaluation = evaluate_ensemble(
        study_path=study_path, output_root=output_root, lookback=lookback
    )
    if not bool(evaluation["account_replay_gate"]["passed"]):
        raise SequenceChallengerError("ensemble_account_replay_gate_not_passed")
    study = base.load_study(study_path)
    context = base._source_context(study)
    selection_path = Path(evaluation["files"]["top10_selections"]["path"])
    selections = pd.read_parquet(selection_path)
    active = selections.loc[
        selections["formal_market_consensus_gate"].astype(bool)
    ].copy()
    if active.empty:
        raise SequenceChallengerError("ensemble_active_selection_schedule_empty")
    schedules: list[pd.DataFrame] = []
    for top_k in (1, 3, 10):
        current = active.loc[active["selection_rank"] <= top_k].copy()
        current["variant"] = "sequence_tree_equal_rank"
        current["exit_policy"] = "planned_close"
        current["gate"] = "formal_market_consensus"
        current["top_k"] = int(top_k)
        current["cost_scenario"] = "base"
        current["take_profit_hit"] = False
        schedules.append(current)
    schedule = pd.concat(schedules, ignore_index=True)
    specs = tuple(
        {
            "variant": "sequence_tree_equal_rank",
            "exit_policy": "planned_close",
            "gate": "formal_market_consensus",
            "top_k": top_k,
            "cost_scenario": scenario,
            "slippage_multiplier": 1.0 if scenario == "base" else 2.0,
            "cohort_equity_fraction": 0.20,
            "planned_fill_day": 5,
        }
        for top_k in (1, 3, 10)
        for scenario in ("base", "stress")
    )
    fingerprint = _stable_hash(
        {
            "schema": ACCOUNT_SCHEMA,
            "evaluation_fingerprint": evaluation["fingerprint"],
            "selection_sha256": evaluation["files"]["top10_selections"]["sha256"],
            "starting_cash": float(starting_cash),
            "specs": specs,
            "capital_policy": "one_fifth_of_previous_close_equity_per_signal_cohort_capped_by_cash",
            "entry": "next_open_board_lot_no_substitution",
            "exit": "d5_close_or_first_sellable_open_within_20_days",
            "no_leverage": True,
        }
    )
    replay_root = _lookback_artifact_root(
        output_root, "sequence_ensemble_account_replay", lookback=lookback
    )
    manifest_path = replay_root / "manifest.json"
    if manifest_path.is_file():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            current.get("status") == "completed"
            and current.get("fingerprint") == fingerprint
        ):
            return current
    daily_raw = base._open_array(
        context.pack["feature_channels"]["daily_raw"], dtype=np.float32
    )
    raw_open = base._open_array(
        context.pack["execution_arrays"]["entry_open_raw"], dtype=np.float32
    )
    entry_filled = base._open_array(
        context.pack["masks"]["entry_filled"], dtype=np.bool_
    )
    costs = base.parse_execution_costs(context.pack)
    summaries: list[dict[str, Any]] = []
    files: dict[str, Any] = {}
    for spec in specs:
        result, equity, trades = base._simulate_account_spec(
            spec=spec,
            selections=schedule,
            context=context,
            daily_raw=daily_raw,
            raw_open=raw_open,
            entry_filled=entry_filled,
            costs=costs,
            starting_cash=float(starting_cash),
        )
        daily_interval = base._newey_west_interval(
            equity["daily_net_return"].to_numpy(dtype=np.float64), lag=20
        )
        result["daily_return_hac"] = daily_interval
        result["annualized_net_return"] = float(
            (result["ending_equity"] / float(starting_cash))
            ** (242.0 / max(len(equity), 1))
            - 1.0
        )
        result["mean_invested_fraction"] = float(
            np.mean(
                1.0
                - equity["cash"].to_numpy(dtype=np.float64)
                / equity["equity"].to_numpy(dtype=np.float64)
            )
        )
        task_id = base._account_task_id(spec)
        task_root = replay_root / "tasks" / task_id
        equity_path = task_root / "equity.parquet"
        trades_path = task_root / "trades.parquet"
        result_path = task_root / "task_result.json"
        base._write_parquet(equity, equity_path)
        base._write_parquet(trades, trades_path)
        task_payload = {
            "schema": ACCOUNT_SCHEMA,
            "status": "completed",
            "completed_at": _now(),
            "study_id": base.STUDY_ID,
            "fingerprint": fingerprint,
            **result,
            "files": {
                "equity": _file_record(equity_path, row_count=len(equity)),
                "trades": _file_record(trades_path, row_count=len(trades)),
            },
        }
        _write_json(result_path, task_payload)
        files[task_id] = _file_record(result_path)
        summaries.append(result)
        base._emit(
            "sequence_ensemble_account_task_completed",
            task_id=task_id,
            total_return=result["total_net_return"],
            maximum_drawdown=result["maximum_drawdown"],
        )
    primary = next(
        item
        for item in summaries
        if int(item["spec"]["top_k"]) == 10
        and item["spec"]["cost_scenario"] == "stress"
    )
    account_gate = {
        "primary": "top10_stress_cost_one_fifth_cohort",
        "requires_positive_total_return": True,
        "requires_positive_daily_hac_lower_bound": True,
        "requires_at_least_five_positive_years": True,
        "requires_worst_year_above_minus_5pct": True,
        "requires_maximum_drawdown_above_minus_20pct": True,
        "passed": bool(
            float(primary["total_net_return"]) > 0.0
            and primary["daily_return_hac"]["lower"] is not None
            and float(primary["daily_return_hac"]["lower"]) > 0.0
            and int(primary["positive_year_count"]) >= 5
            and float(primary["worst_year_return"]) > -0.05
            and float(primary["maximum_drawdown"]) > -0.20
        ),
    }
    summary_frame = pd.DataFrame(
        [
            {
                "top_k": int(item["spec"]["top_k"]),
                "cost_scenario": item["spec"]["cost_scenario"],
                "total_net_return": item["total_net_return"],
                "annualized_net_return": item["annualized_net_return"],
                "maximum_drawdown": item["maximum_drawdown"],
                "annualized_daily_sharpe": item["annualized_daily_sharpe"],
                "positive_year_count": item["positive_year_count"],
                "worst_year_return": item["worst_year_return"],
                "trade_count": item["trade_count"],
                "mean_invested_fraction": item["mean_invested_fraction"],
                "daily_hac_lower": item["daily_return_hac"]["lower"],
            }
            for item in summaries
        ]
    ).sort_values(["top_k", "cost_scenario"])
    summary_path = replay_root / "summary.parquet"
    base._write_parquet(summary_frame, summary_path)
    result = {
        "schema": ACCOUNT_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": base.STUDY_ID,
        "fingerprint": fingerprint,
        "starting_cash": float(starting_cash),
        "capital_policy": {
            "cohort_equity_fraction": 0.20,
            "reason": "D1_open_to_D5_close_can_overlap_five_intraday_cohorts",
            "no_leverage": True,
            "unused_or_unfilled_allocation": "cash_no_substitution",
        },
        "summaries": summaries,
        "primary_account": primary,
        "account_gate": account_gate,
        "epistemic_status": {
            "historical_result_is_adaptive_development_evidence": True,
            "folds_2_to_5_are_not_pristine_because_market_gate_was_already_selected": True,
            "stable_profit_claim_allowed": False,
            "forbidden_2026_read_count": 0,
        },
        "files": {
            "summary": _file_record(summary_path, row_count=len(summary_frame)),
            "tasks": files,
        },
        "sources": {
            "ensemble_evaluation": _file_record(
                _lookback_artifact_root(
                    output_root, "sequence_ensemble_evaluation", lookback=lookback
                )
                / "manifest.json"
            ),
            "selections": _file_record(selection_path),
        },
    }
    _write_json(manifest_path, result)
    base._emit(
        "sequence_ensemble_account_replay_completed",
        primary_total_return=primary["total_net_return"],
        primary_maximum_drawdown=primary["maximum_drawdown"],
        account_gate=account_gate["passed"],
    )
    return result


def replay_account_robustness(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    lookback: int = DEFAULT_LOOKBACK,
    starting_cash: float = 1_000_000.0,
) -> dict[str, Any]:
    """Stress tail dependence, risk scaling, and post-fold-1 persistence."""

    account = replay_ensemble_accounts(
        study_path=study_path,
        output_root=output_root,
        lookback=lookback,
        starting_cash=starting_cash,
    )
    evaluation = evaluate_ensemble(
        study_path=study_path, output_root=output_root, lookback=lookback
    )
    study = base.load_study(study_path)
    context = base._source_context(study)
    selection_path = Path(evaluation["files"]["top10_selections"]["path"])
    selections = pd.read_parquet(selection_path)
    active = selections.loc[
        selections["formal_market_consensus_gate"].astype(bool)
    ].copy()
    variants: list[dict[str, Any]] = []
    for top_k in (1, 3, 10):
        for cap in (0.05, 0.10):
            variants.append(
                {
                    "name": f"top{top_k}__fraction20__cap{round(cap * 100)}__all_folds",
                    "top_k": top_k,
                    "cohort_equity_fraction": 0.20,
                    "maximum_credited_gross_return": cap,
                    "minimum_fold": 1,
                }
            )
    for top_k in (1, 3, 10):
        variants.append(
            {
                "name": f"top{top_k}__fraction10__uncapped__all_folds",
                "top_k": top_k,
                "cohort_equity_fraction": 0.10,
                "minimum_fold": 1,
            }
        )
        variants.append(
            {
                "name": f"top{top_k}__fraction20__uncapped__folds2to5",
                "top_k": top_k,
                "cohort_equity_fraction": 0.20,
                "minimum_fold": 2,
            }
        )
    fingerprint = _stable_hash(
        {
            "schema": ROBUSTNESS_SCHEMA,
            "account_fingerprint": account["fingerprint"],
            "evaluation_fingerprint": evaluation["fingerprint"],
            "selection_sha256": evaluation["files"]["top10_selections"]["sha256"],
            "starting_cash": float(starting_cash),
            "variants": variants,
            "cost_scenario": "stress_double_slippage",
        }
    )
    output_dir = _lookback_artifact_root(
        output_root, "sequence_ensemble_account_robustness", lookback=lookback
    )
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            current.get("status") == "completed"
            and current.get("fingerprint") == fingerprint
        ):
            return current
    daily_raw = base._open_array(
        context.pack["feature_channels"]["daily_raw"], dtype=np.float32
    )
    raw_open = base._open_array(
        context.pack["execution_arrays"]["entry_open_raw"], dtype=np.float32
    )
    entry_filled = base._open_array(
        context.pack["masks"]["entry_filled"], dtype=np.bool_
    )
    costs = base.parse_execution_costs(context.pack)
    summaries: list[dict[str, Any]] = []
    task_files: dict[str, Any] = {}
    for variant in variants:
        top_k = int(variant["top_k"])
        current = active.loc[
            (active["selection_rank"] <= top_k)
            & (active["fold"] >= int(variant["minimum_fold"]))
        ].copy()
        current["variant"] = "sequence_tree_equal_rank_robustness"
        current["exit_policy"] = "planned_close"
        current["gate"] = "formal_market_consensus"
        current["top_k"] = top_k
        current["cost_scenario"] = "base"
        current["take_profit_hit"] = False
        spec: dict[str, Any] = {
            "variant": "sequence_tree_equal_rank_robustness",
            "exit_policy": "planned_close",
            "gate": "formal_market_consensus",
            "top_k": top_k,
            "cost_scenario": "stress",
            "slippage_multiplier": 2.0,
            "cohort_equity_fraction": float(variant["cohort_equity_fraction"]),
            "planned_fill_day": 5,
        }
        if "maximum_credited_gross_return" in variant:
            spec["maximum_credited_gross_return"] = float(
                variant["maximum_credited_gross_return"]
            )
        result, equity, trades = base._simulate_account_spec(
            spec=spec,
            selections=current,
            context=context,
            daily_raw=daily_raw,
            raw_open=raw_open,
            entry_filled=entry_filled,
            costs=costs,
            starting_cash=float(starting_cash),
        )
        result["variant_name"] = str(variant["name"])
        result["minimum_fold"] = int(variant["minimum_fold"])
        result["annualized_net_return"] = float(
            (result["ending_equity"] / float(starting_cash))
            ** (242.0 / max(len(equity), 1))
            - 1.0
        )
        result["daily_return_hac"] = base._newey_west_interval(
            equity["daily_net_return"].to_numpy(dtype=np.float64), lag=20
        )
        result["mean_invested_fraction"] = float(
            np.mean(
                1.0
                - equity["cash"].to_numpy(dtype=np.float64)
                / equity["equity"].to_numpy(dtype=np.float64)
            )
        )
        total_pnl = float(trades["pnl"].sum()) if len(trades) else 0.0
        result["largest_10_trade_pnl_share"] = (
            float(trades.nlargest(10, "pnl")["pnl"].sum() / total_pnl)
            if total_pnl > 0.0
            else math.nan
        )
        task_dir = output_dir / "tasks" / str(variant["name"])
        equity_path = task_dir / "equity.parquet"
        trades_path = task_dir / "trades.parquet"
        task_path = task_dir / "task_result.json"
        base._write_parquet(equity, equity_path)
        base._write_parquet(trades, trades_path)
        payload = {
            "schema": ROBUSTNESS_SCHEMA,
            "status": "completed",
            "completed_at": _now(),
            "fingerprint": fingerprint,
            **result,
            "files": {
                "equity": _file_record(equity_path, row_count=len(equity)),
                "trades": _file_record(trades_path, row_count=len(trades)),
            },
        }
        _write_json(task_path, payload)
        task_files[str(variant["name"])] = _file_record(task_path)
        summaries.append(result)
        base._emit(
            "sequence_account_robustness_task_completed",
            variant=variant["name"],
            total_return=result["total_net_return"],
            maximum_drawdown=result["maximum_drawdown"],
        )
    indexed = {str(item["variant_name"]): item for item in summaries}
    risk_scaled = indexed["top10__fraction10__uncapped__all_folds"]
    cap5 = indexed["top10__fraction20__cap5__all_folds"]
    cap10 = indexed["top10__fraction20__cap10__all_folds"]
    confirmation = indexed["top10__fraction20__uncapped__folds2to5"]
    robustness_gate = {
        "requires_fraction10_maximum_drawdown_above_minus_15pct": True,
        "requires_fraction10_positive_daily_hac_lower": True,
        "requires_cap5_and_cap10_positive_total_return": True,
        "requires_folds2to5_positive_total_return": True,
        "passed": bool(
            float(risk_scaled["maximum_drawdown"]) > -0.15
            and risk_scaled["daily_return_hac"]["lower"] is not None
            and float(risk_scaled["daily_return_hac"]["lower"]) > 0.0
            and float(cap5["total_net_return"]) > 0.0
            and float(cap10["total_net_return"]) > 0.0
            and float(confirmation["total_net_return"]) > 0.0
        ),
    }
    summary_frame = pd.DataFrame(
        [
            {
                "variant": item["variant_name"],
                "top_k": item["spec"]["top_k"],
                "cohort_equity_fraction": item["spec"]["cohort_equity_fraction"],
                "maximum_credited_gross_return": item["spec"].get(
                    "maximum_credited_gross_return"
                ),
                "minimum_fold": item["minimum_fold"],
                "total_net_return": item["total_net_return"],
                "annualized_net_return": item["annualized_net_return"],
                "maximum_drawdown": item["maximum_drawdown"],
                "annualized_daily_sharpe": item["annualized_daily_sharpe"],
                "positive_year_count": item["positive_year_count"],
                "worst_year_return": item["worst_year_return"],
                "daily_hac_lower": item["daily_return_hac"]["lower"],
                "largest_10_trade_pnl_share": item["largest_10_trade_pnl_share"],
            }
            for item in summaries
        ]
    )
    summary_path = output_dir / "summary.parquet"
    base._write_parquet(summary_frame, summary_path)
    result = {
        "schema": ROBUSTNESS_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": base.STUDY_ID,
        "fingerprint": fingerprint,
        "variant_count": len(variants),
        "summaries": summaries,
        "robustness_gate": robustness_gate,
        "decision_boundary": {
            "caps_only_reduce_winners_and_leave_losses_unchanged": True,
            "folds2to5_combination_was_frozen_after_fold1": True,
            "risk_scaling_was_added_after_observing_the_20pct_drawdown": True,
            "stable_profit_claim_allowed": False,
            "forbidden_2026_read_count": 0,
        },
        "files": {
            "summary": _file_record(summary_path, row_count=len(summary_frame)),
            "tasks": task_files,
        },
        "sources": {
            "account_replay": _file_record(
                _lookback_artifact_root(
                    output_root,
                    "sequence_ensemble_account_replay",
                    lookback=lookback,
                )
                / "manifest.json"
            ),
            "ensemble_evaluation": _file_record(
                _lookback_artifact_root(
                    output_root, "sequence_ensemble_evaluation", lookback=lookback
                )
                / "manifest.json"
            ),
        },
    }
    _write_json(manifest_path, result)
    base._emit(
        "sequence_account_robustness_completed",
        gate=robustness_gate["passed"],
        risk_scaled_return=risk_scaled["total_net_return"],
        risk_scaled_drawdown=risk_scaled["maximum_drawdown"],
    )
    return result


def replay_dual_gate_horizons(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    lookback: int = DEFAULT_LOOKBACK,
    starting_cash: float = 1_000_000.0,
) -> dict[str, Any]:
    """Challenge fixed exits under the intersection of two frozen market heads."""

    evaluation = evaluate_ensemble(
        study_path=study_path, output_root=output_root, lookback=lookback
    )
    study = base.load_study(study_path)
    sources = _load_sources(study, output_root=output_root)
    selection_path = Path(evaluation["files"]["top10_selections"]["path"])
    selections = pd.read_parquet(selection_path)
    daily = pd.read_parquet(evaluation["files"]["daily_metrics"]["path"])
    tree_market, tree_market_manifest = _formal_market_predictions(output_root)
    neural_parts = [
        pd.read_parquet(
            output_root
            / "sequence_challenger"
            / f"lookback_{lookback}"
            / f"fold_{int(fold['fold'])}"
            / "market_predictions.parquet"
        )
        for fold in base.build_forward_folds(
            date_idx=sources.context.row_index["date_idx"].to_numpy(dtype=np.int32),
            trade_date=sources.context.row_index["trade_date"].astype(str).to_numpy(),
            fold_count=int(study["validation"]["forward_fold_count"]),
            purge_days=int(study["validation"]["common_purge_trading_days"]),
        )
    ]
    neural_market = (
        pd.concat(neural_parts, ignore_index=True)
        .drop_duplicates("date_idx", keep="last")
        .set_index("date_idx")
    )
    common_dates = sorted(
        set(tree_market.index.astype(int)).intersection(neural_market.index.astype(int))
    )
    dual_gate = (
        tree_market.loc[common_dates, "ridge_prediction"].gt(0.0)
        & tree_market.loc[common_dates, "positive_probability"].gt(0.5)
        & neural_market.loc[common_dates, "predicted_return"].gt(0.0)
        & neural_market.loc[common_dates, "positive_probability"].gt(0.5)
    )
    dual_gate.index = np.asarray(common_dates, dtype=np.int32)
    event_selections = selections.copy()
    event_selections["market_return_prediction"] = np.minimum(
        event_selections["date_idx"].map(tree_market["ridge_prediction"]),
        event_selections["date_idx"].map(neural_market["predicted_return"]),
    )
    event_selections["market_positive_probability"] = np.minimum(
        event_selections["date_idx"].map(tree_market["positive_probability"]),
        event_selections["date_idx"].map(neural_market["positive_probability"]),
    )
    event_metrics = _gated_metrics(daily, event_selections)
    active = selections.loc[selections["date_idx"].map(dual_gate).fillna(False)].copy()
    variants: list[dict[str, Any]] = []
    for horizon in (2, 3, 5, 10):
        for top_k in (1, 3, 10):
            variants.append(
                {
                    "name": f"h{horizon}__top{top_k}__uncapped__all_folds",
                    "horizon": horizon,
                    "top_k": top_k,
                    "minimum_fold": 1,
                }
            )
    for horizon in (3, 5):
        for top_k in (1, 3, 10):
            for cap in (0.05, 0.10):
                variants.append(
                    {
                        "name": f"h{horizon}__top{top_k}__cap{round(cap * 100)}__all_folds",
                        "horizon": horizon,
                        "top_k": top_k,
                        "minimum_fold": 1,
                        "maximum_credited_gross_return": cap,
                    }
                )
            variants.append(
                {
                    "name": f"h{horizon}__top{top_k}__uncapped__folds2to5",
                    "horizon": horizon,
                    "top_k": top_k,
                    "minimum_fold": 2,
                }
            )
    fingerprint = _stable_hash(
        {
            "schema": DUAL_HORIZON_SCHEMA,
            "evaluation_fingerprint": evaluation["fingerprint"],
            "selection_sha256": evaluation["files"]["top10_selections"]["sha256"],
            "tree_market_sha256": base._sha256(tree_market_manifest),
            "neural_market_files": [
                base._sha256(
                    output_root
                    / "sequence_challenger"
                    / f"lookback_{lookback}"
                    / f"fold_{fold}"
                    / "market_predictions.parquet"
                )
                for fold in range(1, 6)
            ],
            "gate": "tree_consensus_and_neural_consensus",
            "variants": variants,
            "capital_fraction": "one_divided_by_planned_horizon",
            "cost_scenario": "stress_double_slippage",
            "starting_cash": float(starting_cash),
        }
    )
    output_dir = _lookback_artifact_root(
        output_root, "dual_market_horizon_challenge", lookback=lookback
    )
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            current.get("status") == "completed"
            and current.get("fingerprint") == fingerprint
        ):
            return current
    daily_raw = base._open_array(
        sources.context.pack["feature_channels"]["daily_raw"], dtype=np.float32
    )
    raw_open = base._open_array(
        sources.context.pack["execution_arrays"]["entry_open_raw"], dtype=np.float32
    )
    market_entry_filled = base._open_array(
        sources.context.pack["masks"]["entry_filled"], dtype=np.bool_
    )
    costs = base.parse_execution_costs(sources.context.pack)
    summaries: list[dict[str, Any]] = []
    task_files: dict[str, Any] = {}
    target_columns = list(sources.path_manifest["target_columns"])
    horizons = list(sources.path_manifest["files"]["legal_fill_days"]["horizons"])
    for variant in variants:
        horizon = int(variant["horizon"])
        top_k = int(variant["top_k"])
        gross_column = target_columns.index(f"legal_exit_return_d{horizon}")
        horizon_column = horizons.index(horizon)
        current = active.loc[
            (active["selection_rank"] <= top_k)
            & (active["fold"] >= int(variant["minimum_fold"]))
        ].copy()
        rows = current["model_row_position"].to_numpy(dtype=np.int64)
        known = ~current["market_entry_filled"].to_numpy(dtype=bool) | np.asarray(
            sources.path_valid[rows, gross_column], dtype=bool
        )
        date_known = (
            pd.Series(known, index=current.index).groupby(current["date_idx"]).all()
        )
        current = current.loc[current["date_idx"].map(date_known).fillna(False)].copy()
        rows = current["model_row_position"].to_numpy(dtype=np.int64)
        current["legal_gross_return"] = np.asarray(
            sources.path_values[rows, gross_column], dtype=np.float32
        )
        current["fill_day"] = np.asarray(
            sources.fill_days[rows, horizon_column], dtype=np.int16
        )
        current["variant"] = "dual_market_equal_rank"
        current["exit_policy"] = "planned_close"
        current["gate"] = "tree_and_neural_consensus"
        current["top_k"] = top_k
        current["cost_scenario"] = "base"
        current["take_profit_hit"] = False
        spec: dict[str, Any] = {
            "variant": "dual_market_equal_rank",
            "exit_policy": "planned_close",
            "gate": "tree_and_neural_consensus",
            "top_k": top_k,
            "cost_scenario": "stress",
            "slippage_multiplier": 2.0,
            "cohort_equity_fraction": 1.0 / horizon,
            "planned_fill_day": horizon,
        }
        if "maximum_credited_gross_return" in variant:
            spec["maximum_credited_gross_return"] = float(
                variant["maximum_credited_gross_return"]
            )
        result, equity, trades = base._simulate_account_spec(
            spec=spec,
            selections=current,
            context=sources.context,
            daily_raw=daily_raw,
            raw_open=raw_open,
            entry_filled=market_entry_filled,
            costs=costs,
            starting_cash=float(starting_cash),
        )
        result["variant_name"] = str(variant["name"])
        result["horizon"] = horizon
        result["minimum_fold"] = int(variant["minimum_fold"])
        result["signal_date_count"] = int(current["date_idx"].nunique())
        result["annualized_net_return"] = float(
            (result["ending_equity"] / float(starting_cash))
            ** (242.0 / max(len(equity), 1))
            - 1.0
        )
        result["daily_return_hac"] = base._newey_west_interval(
            equity["daily_net_return"].to_numpy(dtype=np.float64), lag=20
        )
        result["mean_invested_fraction"] = float(
            np.mean(
                1.0
                - equity["cash"].to_numpy(dtype=np.float64)
                / equity["equity"].to_numpy(dtype=np.float64)
            )
        )
        total_pnl = float(trades["pnl"].sum()) if len(trades) else 0.0
        result["largest_10_trade_pnl_share"] = (
            float(trades.nlargest(10, "pnl")["pnl"].sum() / total_pnl)
            if total_pnl > 0.0
            else math.nan
        )
        task_dir = output_dir / "tasks" / str(variant["name"])
        equity_path = task_dir / "equity.parquet"
        trades_path = task_dir / "trades.parquet"
        task_path = task_dir / "task_result.json"
        base._write_parquet(equity, equity_path)
        base._write_parquet(trades, trades_path)
        payload = {
            "schema": DUAL_HORIZON_SCHEMA,
            "status": "completed",
            "completed_at": _now(),
            "fingerprint": fingerprint,
            **result,
            "files": {
                "equity": _file_record(equity_path, row_count=len(equity)),
                "trades": _file_record(trades_path, row_count=len(trades)),
            },
        }
        _write_json(task_path, payload)
        task_files[str(variant["name"])] = _file_record(task_path)
        summaries.append(result)
        base._emit(
            "dual_market_horizon_task_completed",
            variant=variant["name"],
            total_return=result["total_net_return"],
            maximum_drawdown=result["maximum_drawdown"],
        )
    indexed = {str(item["variant_name"]): item for item in summaries}
    primary = indexed["h3__top10__uncapped__all_folds"]
    cap10 = indexed["h3__top10__cap10__all_folds"]
    cap5 = indexed["h3__top10__cap5__all_folds"]
    confirmation = indexed["h3__top10__uncapped__folds2to5"]
    candidate_gate = {
        "primary": "h3_top10_stress_cost_dual_market_gate",
        "requires_primary_positive_hac_lower": True,
        "requires_primary_six_positive_years": True,
        "requires_primary_drawdown_above_minus_20pct": True,
        "requires_cap10_positive_hac_lower": True,
        "requires_folds2to5_positive_hac_lower": True,
        "passed": bool(
            primary["daily_return_hac"]["lower"] is not None
            and float(primary["daily_return_hac"]["lower"]) > 0.0
            and int(primary["positive_year_count"]) == 6
            and float(primary["maximum_drawdown"]) > -0.20
            and cap10["daily_return_hac"]["lower"] is not None
            and float(cap10["daily_return_hac"]["lower"]) > 0.0
            and confirmation["daily_return_hac"]["lower"] is not None
            and float(confirmation["daily_return_hac"]["lower"]) > 0.0
        ),
    }
    small_gain_robustness = {
        "definition": "all_gross_winners_above_5pct_are_credited_as_only_5pct_while_losses_are_unchanged",
        "total_net_return": cap5["total_net_return"],
        "daily_hac_lower": cap5["daily_return_hac"]["lower"],
        "passed": bool(
            float(cap5["total_net_return"]) > 0.0
            and cap5["daily_return_hac"]["lower"] is not None
            and float(cap5["daily_return_hac"]["lower"]) > 0.0
        ),
    }
    summary_frame = pd.DataFrame(
        [
            {
                "variant": item["variant_name"],
                "horizon": item["horizon"],
                "top_k": item["spec"]["top_k"],
                "maximum_credited_gross_return": item["spec"].get(
                    "maximum_credited_gross_return"
                ),
                "minimum_fold": item["minimum_fold"],
                "signal_date_count": item["signal_date_count"],
                "total_net_return": item["total_net_return"],
                "annualized_net_return": item["annualized_net_return"],
                "maximum_drawdown": item["maximum_drawdown"],
                "annualized_daily_sharpe": item["annualized_daily_sharpe"],
                "positive_year_count": item["positive_year_count"],
                "worst_year_return": item["worst_year_return"],
                "daily_hac_lower": item["daily_return_hac"]["lower"],
                "largest_10_trade_pnl_share": item["largest_10_trade_pnl_share"],
            }
            for item in summaries
        ]
    )
    summary_path = output_dir / "summary.parquet"
    base._write_parquet(summary_frame, summary_path)
    result = {
        "schema": DUAL_HORIZON_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": base.STUDY_ID,
        "fingerprint": fingerprint,
        "gate_definition": {
            "tree": "ridge_return_above_zero_and_logistic_probability_above_0p5",
            "neural": "return_head_above_zero_and_probability_head_above_0p5",
            "combined": "tree_and_neural_both_true",
            "weights_or_thresholds_fitted_for_this_challenge": False,
            "signal_date_count": int(dual_gate.sum()),
        },
        "event_metrics": event_metrics,
        "summaries": summaries,
        "candidate_gate": candidate_gate,
        "small_gain_robustness": small_gain_robustness,
        "decision_boundary": {
            "dual_gate_was_proposed_after_observing_the_tree_gate_2024_drawdown": True,
            "h3_primary_was_selected_after_comparing_fixed_horizons": True,
            "historical_result_is_adaptive_development_evidence": True,
            "stable_profit_claim_allowed": False,
            "forbidden_2026_read_count": 0,
        },
        "files": {
            "summary": _file_record(summary_path, row_count=len(summary_frame)),
            "tasks": task_files,
        },
        "sources": {
            "ensemble_evaluation": _file_record(
                _lookback_artifact_root(
                    output_root, "sequence_ensemble_evaluation", lookback=lookback
                )
                / "manifest.json"
            ),
            "tree_market": _file_record(tree_market_manifest),
            "selections": _file_record(selection_path),
        },
    }
    _write_json(manifest_path, result)
    base._emit(
        "dual_market_horizon_challenge_completed",
        candidate_gate=candidate_gate["passed"],
        small_gain_robustness=small_gain_robustness["passed"],
        primary_return=primary["total_net_return"],
        primary_drawdown=primary["maximum_drawdown"],
    )
    return result


def _masked_tree_prediction(
    *,
    model_path: Path,
    best_iteration: int,
    matrix: np.ndarray,
    rows: np.ndarray,
    masked_columns: np.ndarray,
    batch_size: int,
    static_feature_override: StaticFeatureOverride | None = None,
) -> np.ndarray:
    booster = lgb.Booster(model_file=str(model_path))
    output = np.empty(len(rows), dtype=np.float32)
    for left in range(0, len(rows), int(batch_size)):
        right = min(left + int(batch_size), len(rows))
        batch_rows = rows[left:right]
        values = apply_static_feature_contract(
            np.asarray(matrix[batch_rows], dtype=np.float32),
            row_positions=batch_rows,
            masked_columns=masked_columns,
            override=static_feature_override,
        )
        output[left:right] = np.asarray(
            booster.predict(values, num_iteration=int(best_iteration)),
            dtype=np.float32,
        )
        if (left // int(batch_size)) % 4 == 3:
            base._trim_working_set()
    return output


def prepare_repaired_cross_sectional_features(
    sources: SequenceSources,
    *,
    output_root: Path,
    base_feature_manifest_path: Path = DEFAULT_LEGACY_BASE_FEATURE_MANIFEST,
) -> StaticFeatureOverride:
    """Re-rank frozen causal F1 values over the repaired current candidate pool.

    The 105 repaired percentiles are written in formal model-row order.  This
    changes only the cross-sectional denominator; it does not read a return,
    fill, or other post-signal label.
    """

    if not base_feature_manifest_path.is_file():
        raise SequenceChallengerError(
            f"legacy_base_feature_manifest_missing:{base_feature_manifest_path}"
        )
    base_manifest = json.loads(base_feature_manifest_path.read_text(encoding="utf-8"))
    catalog = list(base_manifest["continuous_catalog"])
    f1_records = [record for record in catalog if str(record["family"]) == "F1"]
    if len(f1_records) != 105:
        raise SequenceChallengerError("legacy_f1_feature_count_changed")
    f1_columns = np.asarray(
        [int(record["column_index"]) for record in f1_records], dtype=np.int32
    )
    repaired_names = tuple(f"cs_percentile__{record['name']}" for record in f1_records)
    feature_positions = {
        name: index for index, name in enumerate(sources.feature_names)
    }
    if not set(repaired_names).issubset(feature_positions):
        raise SequenceChallengerError("repaired_rank_features_absent_from_model")
    override_columns = np.asarray(
        [feature_positions[name] for name in repaired_names], dtype=np.int32
    )
    if not np.array_equal(override_columns, np.arange(104, 209, dtype=np.int32)):
        raise SequenceChallengerError("repaired_rank_model_columns_changed")

    candidate_path = Path(base_manifest["candidate_alignment"]["candidate_index"])
    row_index_path = Path(sources.context.model_manifest["row_index"]["path"])
    continuous_path = Path(base_manifest["files"]["continuous"]["path"])
    legacy_date_path = Path(base_manifest["files"]["candidate_date_idx"]["path"])
    legacy_symbol_path = Path(base_manifest["files"]["candidate_symbol_idx"]["path"])
    required_paths = (
        candidate_path,
        row_index_path,
        continuous_path,
        legacy_date_path,
        legacy_symbol_path,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise SequenceChallengerError(f"repaired_rank_source_missing:{missing}")
    fingerprint = _stable_hash(
        {
            "schema": REPAIRED_RANK_SCHEMA,
            "builder_version": 1,
            "base_feature_manifest_sha256": base._sha256(base_feature_manifest_path),
            "current_candidate_index_sha256": base._sha256(candidate_path),
            "formal_row_index_sha256": base._sha256(row_index_path),
            "legacy_continuous_size": int(continuous_path.stat().st_size),
            "legacy_candidate_date_size": int(legacy_date_path.stat().st_size),
            "legacy_candidate_symbol_size": int(legacy_symbol_path.stat().st_size),
            "f1_columns": f1_columns.tolist(),
            "repaired_names": repaired_names,
            "row_count": len(sources.context.row_index),
            "rank_method": "scipy_average_(rank_minus_1)/(finite_count_minus_1)",
            "candidate_universe": "current_repaired_candidate_eligible",
        }
    )
    output_dir = output_root / "repaired_cross_sectional_features"
    manifest_path = output_dir / "manifest.json"
    values_path = output_dir / "repaired_f2.float32.dat"
    expected_size = len(sources.context.row_index) * len(repaired_names) * 4
    if manifest_path.is_file() and values_path.is_file():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            current.get("status") == "completed"
            and current.get("fingerprint") == fingerprint
            and int(values_path.stat().st_size) == expected_size
        ):
            return StaticFeatureOverride(
                values=np.memmap(
                    values_path,
                    dtype=np.float32,
                    mode="r",
                    shape=(len(sources.context.row_index), len(repaired_names)),
                ),
                columns=override_columns,
                feature_names=repaired_names,
                manifest=current,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = values_path.with_suffix(values_path.suffix + ".partial")
    progress_path = output_dir / "progress.json"
    date_blocks = [sources.blocks[key] for key in sorted(sources.blocks)]
    if (
        not date_blocks
        or max(block.trade_date for block in date_blocks) >= "2026-01-01"
    ):
        raise SequenceChallengerError("repaired_rank_formal_dates_include_2026")
    start_position = 0
    if partial_path.is_file() and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("fingerprint") == fingerprint
            and int(partial_path.stat().st_size) == expected_size
        ):
            start_position = int(progress.get("completed_date_count", 0))
        else:
            partial_path.unlink(missing_ok=True)
            progress_path.unlink(missing_ok=True)
    if not partial_path.is_file():
        with partial_path.open("wb") as stream:
            stream.truncate(expected_size)
        initializing = np.memmap(
            partial_path,
            dtype=np.float32,
            mode="r+",
            shape=(len(sources.context.row_index), len(repaired_names)),
        )
        for left in range(0, len(initializing), 250_000):
            initializing[left : left + 250_000] = np.nan
        initializing.flush()
        del initializing
        _write_json(
            progress_path,
            {"fingerprint": fingerprint, "completed_date_count": 0},
        )

    current = pd.read_parquet(candidate_path, columns=["date_idx", "symbol_idx"])
    current_dates = current["date_idx"].to_numpy(dtype=np.int32, copy=False)
    current_symbols = current["symbol_idx"].to_numpy(dtype=np.int32, copy=False)
    symbol_count = len(sources.context.pack["symbol_values"])
    current_keys = current_dates.astype(np.int64) * symbol_count + current_symbols
    if len(current_keys) > 1 and bool((current_keys[1:] < current_keys[:-1]).any()):
        raise SequenceChallengerError("current_candidate_index_is_not_sorted")
    legacy_count = int(base_manifest["continuous_shape"][0])
    legacy_dates = np.memmap(
        legacy_date_path, dtype=np.int32, mode="r", shape=(legacy_count,)
    )
    legacy_symbols = np.memmap(
        legacy_symbol_path, dtype=np.int32, mode="r", shape=(legacy_count,)
    )
    continuous = np.memmap(
        continuous_path,
        dtype=np.float32,
        mode="r",
        shape=tuple(int(value) for value in base_manifest["continuous_shape"]),
    )
    repaired = np.memmap(
        partial_path,
        dtype=np.float32,
        mode="r+",
        shape=(len(sources.context.row_index), len(repaired_names)),
    )
    current_universe_counts = [
        int(np.searchsorted(current_dates, int(block.date_idx), side="right"))
        - int(np.searchsorted(current_dates, int(block.date_idx), side="left"))
        for block in date_blocks
    ]
    supplemental_candidate_count = 0
    for position, block in enumerate(
        date_blocks[start_position:], start=start_position
    ):
        date_idx = int(block.date_idx)
        current_left = int(np.searchsorted(current_dates, date_idx, side="left"))
        current_right = int(np.searchsorted(current_dates, date_idx, side="right"))
        if current_right <= current_left:
            raise SequenceChallengerError(
                f"repaired_rank_candidate_date_missing:{date_idx}"
            )
        candidate_symbols = current_symbols[current_left:current_right]
        legacy_left = int(np.searchsorted(legacy_dates, date_idx, side="left"))
        legacy_right = int(np.searchsorted(legacy_dates, date_idx, side="right"))
        legacy_block_symbols = np.asarray(
            legacy_symbols[legacy_left:legacy_right], dtype=np.int32
        )
        legacy_positions = np.searchsorted(legacy_block_symbols, candidate_symbols)
        legacy_found = legacy_positions < len(legacy_block_symbols)
        bounded_positions = np.minimum(
            legacy_positions, max(len(legacy_block_symbols) - 1, 0)
        )
        if len(legacy_block_symbols):
            legacy_found &= legacy_block_symbols[bounded_positions] == candidate_symbols
        legacy_rows = np.full(len(candidate_symbols), -1, dtype=np.int64)
        legacy_rows[legacy_found] = legacy_left + legacy_positions[legacy_found]
        f1_values = np.full(
            (len(candidate_symbols), len(f1_columns)), np.nan, dtype=np.float32
        )
        if bool(legacy_found.any()):
            f1_values[legacy_found] = np.asarray(
                continuous[np.ix_(legacy_rows[legacy_found], f1_columns)],
                dtype=np.float32,
            )
        missing_legacy = ~legacy_found
        if bool(missing_legacy.any()):
            # Status repairs can add a stock-day that was absent from the legacy
            # candidate table. Rebuild only those rare F1 rows from the same
            # causal pack instead of dropping them from the rank denominator.
            from daily_research.path_policy import (
                seq100_full_market_live_inference as live,
            )

            missing_symbols = candidate_symbols[missing_legacy]
            if date_idx < max(live.FEATURE_WINDOWS):
                raise SequenceChallengerError(
                    f"repaired_rank_supplement_lacks_lookback:{date_idx}"
                )
            history = slice(date_idx - max(live.FEATURE_WINDOWS), date_idx + 1)
            supplemental = live._last_daily_features(
                np.asarray(
                    sources.daily_raw[history, missing_symbols, :], dtype=np.float32
                ),
                np.asarray(
                    sources.turnover[history, missing_symbols, :], dtype=np.float32
                ),
            )
            f1_values[missing_legacy] = np.column_stack(
                [
                    np.asarray(supplemental[str(record["name"])], dtype=np.float32)
                    for record in f1_records
                ]
            )
            supplemental_candidate_count += int(missing_legacy.sum())
        ranks = _rank_percentile_columns(f1_values)
        formal_symbols = sources.context.row_index.iloc[block.start : block.stop][
            "symbol_idx"
        ].to_numpy(dtype=np.int32)
        formal_positions = np.searchsorted(candidate_symbols, formal_symbols)
        if bool(
            (formal_positions >= len(candidate_symbols)).any()
        ) or not np.array_equal(candidate_symbols[formal_positions], formal_symbols):
            raise SequenceChallengerError(
                f"repaired_rank_formal_row_not_in_current_candidate:{date_idx}"
            )
        formal_legacy = sources.context.row_index.iloc[block.start : block.stop][
            "legacy_candidate_row"
        ].to_numpy(dtype=np.int64)
        formal_legacy_found = formal_legacy >= 0
        if bool(formal_legacy_found.any()) and not np.array_equal(
            legacy_rows[formal_positions][formal_legacy_found],
            formal_legacy[formal_legacy_found],
        ):
            raise SequenceChallengerError(
                f"repaired_rank_legacy_alignment_changed:{date_idx}"
            )
        repaired[block.start : block.stop] = ranks[formal_positions]
        completed = position + 1
        if completed % 25 == 0 or completed == len(date_blocks):
            repaired.flush()
            _write_json(
                progress_path,
                {"fingerprint": fingerprint, "completed_date_count": completed},
            )
        if completed % 250 == 0:
            base._emit(
                "repaired_cross_sectional_progress",
                completed_date_count=completed,
                total_date_count=len(date_blocks),
            )
    repaired.flush()
    del repaired, continuous, legacy_dates, legacy_symbols, current, current_keys
    gc.collect()
    os.replace(partial_path, values_path)
    progress_path.unlink(missing_ok=True)

    values = np.memmap(
        values_path,
        dtype=np.float32,
        mode="r",
        shape=(len(sources.context.row_index), len(repaired_names)),
    )
    final_block = date_blocks[-1]
    frozen = np.asarray(
        sources.matrix[final_block.start : final_block.stop, override_columns],
        dtype=np.float32,
    )
    current_final = np.asarray(
        values[final_block.start : final_block.stop], dtype=np.float32
    )
    same_finite = np.array_equal(np.isfinite(frozen), np.isfinite(current_final))
    comparable = np.isfinite(frozen) & np.isfinite(current_final)
    drift = np.abs(frozen[comparable] - current_final[comparable])
    result = {
        "schema": REPAIRED_RANK_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "fingerprint": fingerprint,
        "row_count": len(sources.context.row_index),
        "feature_count": len(repaired_names),
        "feature_names": list(repaired_names),
        "model_columns": override_columns.tolist(),
        "formal_date_count": len(date_blocks),
        "formal_minimum_date": date_blocks[0].trade_date,
        "formal_maximum_date": date_blocks[-1].trade_date,
        "current_candidate_universe_min": min(current_universe_counts),
        "current_candidate_universe_max": max(current_universe_counts),
        "supplemental_candidate_day_count": supplemental_candidate_count,
        "last_date_drift_vs_frozen": {
            "trade_date": final_block.trade_date,
            "finite_mask_identical": same_finite,
            "comparable_cell_count": int(comparable.sum()),
            "mean_absolute_difference": float(drift.mean()),
            "p99_absolute_difference": float(np.quantile(drift, 0.99)),
            "maximum_absolute_difference": float(drift.max()),
        },
        "causality": {
            "rank_universe": "current repaired candidate_eligible on the same date",
            "underlying_values": "frozen causal F1 values",
            "future_return_or_fill_read": False,
            "forbidden_2026_row_count": 0,
        },
        "file": _file_record(
            values_path,
            dtype="float32",
            shape=[len(sources.context.row_index), len(repaired_names)],
        ),
        "sources": {
            "base_feature_manifest": _file_record(base_feature_manifest_path),
            "current_candidate_index": _file_record(candidate_path),
            "formal_row_index": _file_record(row_index_path),
        },
    }
    _write_json(manifest_path, result)
    return StaticFeatureOverride(
        values=values,
        columns=override_columns,
        feature_names=repaired_names,
        manifest=result,
    )


def _build_rebuildable_membership_panels(
    sources: SequenceSources,
    *,
    qdp_paths: Mapping[str, Sequence[Path]],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load current date-specific industry and index snapshots into small panels."""

    from daily_research.path_policy import seq100_quality_liquidity_data_prep as prep
    from daily_research.path_policy import seq100_signal_quality as signal

    date_values = tuple(str(value) for value in sources.context.pack["date_values"])
    date_map = {value: index for index, value in enumerate(date_values)}
    symbols = tuple(str(value) for value in sources.context.pack["symbol_values"])
    symbol_map = {value: index for index, value in enumerate(symbols)}
    date_count = len(date_values)
    symbol_count = len(symbols)
    industry_codes = np.zeros((date_count, symbol_count), dtype=np.int64)
    industry_age = np.full((date_count, symbol_count), np.nan, dtype=np.float32)
    index_membership = {
        name: np.zeros((date_count, symbol_count), dtype=bool)
        for name in ("csi300", "csi500", "sse50")
    }
    connection = duckdb.connect()
    try:
        connection.execute("SET threads=8")
        required = {"industry_concept", "index_constituents"}
        missing = sorted(required.difference(qdp_paths))
        if missing:
            raise SequenceChallengerError(
                f"rebuildable_membership_sources_missing:{missing}"
            )
        industry_scan = prep._scan(qdp_paths["industry_concept"])
        label_rows = connection.execute(
            "SELECT DISTINCT coalesce(nullif(industry_name,''),"
            "nullif(industry,''),nullif(original_industry,'')) AS label "
            f"FROM {industry_scan} WHERE trade_date BETWEEN '2010-01-01' AND '2025-12-31'"
        ).fetchdf()
        label_codes = {
            str(row.label): signal._stable_category_hash(row.label)
            for row in label_rows.itertuples(index=False)
            if str(row.label or "").strip()
        }
        connection.execute(
            "SELECT trade_date,symbol,"
            "coalesce(nullif(industry_name,''),nullif(industry,''),"
            "nullif(original_industry,'')) AS label,industry_source_date "
            f"FROM {industry_scan} "
            "WHERE trade_date BETWEEN '2010-01-01' AND '2025-12-31'"
        )
        industry_reader = connection.to_arrow_reader(batch_size=400_000)
        for batch in industry_reader:
            frame = batch.to_pandas()
            date_idx = frame["trade_date"].astype(str).map(date_map)
            symbol_idx = frame["symbol"].astype(str).map(symbol_map)
            label_idx = frame["label"].astype(str).map(label_codes)
            usable = date_idx.notna() & symbol_idx.notna() & label_idx.notna()
            if not bool(usable.any()):
                continue
            dates = date_idx.loc[usable].to_numpy(dtype=np.int32)
            symbols_ = symbol_idx.loc[usable].to_numpy(dtype=np.int32)
            labels = label_idx.loc[usable].to_numpy(dtype=np.int64)
            industry_codes[dates, symbols_] = labels
            source = pd.to_datetime(
                frame.loc[usable, "industry_source_date"], errors="coerce"
            )
            trade = pd.to_datetime(frame.loc[usable, "trade_date"], errors="coerce")
            ages = (trade - source).dt.days.to_numpy(dtype=np.float32)
            industry_age[dates, symbols_] = ages

        index_scan = prep._scan(qdp_paths["index_constituents"])
        index_names = {
            "000300.SH": "csi300",
            "000905.SH": "csi500",
            "000016.SH": "sse50",
        }
        connection.execute(
            "SELECT trade_date,symbol,index_symbol,source_snapshot_date "
            f"FROM {index_scan} "
            "WHERE trade_date BETWEEN '2010-01-01' AND '2025-12-31' "
            "AND index_symbol IN ('000300.SH','000905.SH','000016.SH')"
        )
        index_reader = connection.to_arrow_reader(batch_size=400_000)
        for batch in index_reader:
            frame = batch.to_pandas()
            date_idx = frame["trade_date"].astype(str).map(date_map)
            symbol_idx = frame["symbol"].astype(str).map(symbol_map)
            index_label = frame["index_symbol"].astype(str).map(index_names)
            usable = date_idx.notna() & symbol_idx.notna() & index_label.notna()
            if not bool(usable.any()):
                continue
            source = pd.to_datetime(
                frame.loc[usable, "source_snapshot_date"], errors="coerce"
            )
            trade = pd.to_datetime(frame.loc[usable, "trade_date"], errors="coerce")
            if bool(((source.notna()) & (source > trade)).any()):
                raise SequenceChallengerError("rebuildable_index_source_time_violation")
            dates = date_idx.loc[usable].to_numpy(dtype=np.int32)
            symbols_ = symbol_idx.loc[usable].to_numpy(dtype=np.int32)
            names = index_label.loc[usable].astype(str).to_numpy()
            for name, membership in index_membership.items():
                selected = names == name
                if bool(selected.any()):
                    membership[dates[selected], symbols_[selected]] = True
    finally:
        connection.close()
    return industry_codes, industry_age, index_membership


def prepare_rebuildable_core_features(
    sources: SequenceSources,
    *,
    output_root: Path,
    qdp_paths: Mapping[str, Sequence[Path]],
) -> StaticFeatureOverride:
    """Rebuild F2, industry and index-aware market fields from active PIT sources."""

    repaired_rank = prepare_repaired_cross_sectional_features(
        sources, output_root=output_root
    )
    from daily_research.path_policy import (
        seq100_full_market_live_inference as live,
    )

    feature_names = tuple(sources.feature_names)
    f2_names = tuple(repaired_rank.feature_names)
    market_names = tuple(
        name
        for name in feature_names
        if str(
            sources.context.model_manifest["features"][feature_names.index(name)][
                "analytic_family"
            ]
        )
        == "market_state"
    )
    industry_names = tuple(
        name
        for name in feature_names
        if str(
            sources.context.model_manifest["features"][feature_names.index(name)][
                "analytic_family"
            ]
        )
        == "industry_context"
    )
    expected_industry = tuple(live.INDUSTRY_FEATURES)
    if industry_names != expected_industry or len(market_names) != 54:
        raise SequenceChallengerError("rebuildable_core_feature_catalog_changed")
    override_names = (*f2_names, *market_names, *industry_names)
    override_columns = np.asarray(
        [feature_names.index(name) for name in override_names], dtype=np.int32
    )
    fingerprint = _stable_hash(
        {
            "schema": "seq100_full_market_rebuildable_core_features/1",
            "builder_version": 1,
            "repaired_rank_fingerprint": repaired_rank.manifest["fingerprint"],
            "active_source_boundaries": _active_source_boundaries(),
            "feature_names": override_names,
            "formal_row_count": len(sources.context.row_index),
            "formal_maximum_date": str(
                sources.context.row_index["trade_date"].astype(str).max()
            ),
        }
    )
    output_dir = output_root / "rebuildable_core_features"
    manifest_path = output_dir / "manifest.json"
    values_path = output_dir / "rebuildable_core.float32.dat"
    expected_size = len(sources.context.row_index) * len(override_names) * 4
    if manifest_path.is_file() and values_path.is_file():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            current.get("status") == "completed"
            and current.get("fingerprint") == fingerprint
            and int(values_path.stat().st_size) == expected_size
        ):
            return StaticFeatureOverride(
                values=np.memmap(
                    values_path,
                    dtype=np.float32,
                    mode="r",
                    shape=(len(sources.context.row_index), len(override_names)),
                ),
                columns=override_columns,
                feature_names=override_names,
                manifest=current,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = values_path.with_suffix(values_path.suffix + ".partial")
    progress_path = output_dir / "progress.json"
    date_blocks = [sources.blocks[key] for key in sorted(sources.blocks)]
    if (
        not date_blocks
        or max(block.trade_date for block in date_blocks) >= "2026-01-01"
    ):
        raise SequenceChallengerError("rebuildable_core_formal_dates_include_2026")
    start_position = 0
    if partial_path.is_file() and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("fingerprint") == fingerprint
            and int(partial_path.stat().st_size) == expected_size
        ):
            start_position = int(progress.get("completed_date_count", 0))
        else:
            partial_path.unlink(missing_ok=True)
            progress_path.unlink(missing_ok=True)
    if not partial_path.is_file():
        with partial_path.open("wb") as stream:
            stream.truncate(expected_size)
        initializing = np.memmap(
            partial_path,
            dtype=np.float32,
            mode="r+",
            shape=(len(sources.context.row_index), len(override_names)),
        )
        for left in range(0, len(initializing), 250_000):
            initializing[left : left + 250_000] = np.nan
        initializing.flush()
        del initializing
        _write_json(
            progress_path,
            {"fingerprint": fingerprint, "completed_date_count": 0},
        )

    industry_codes, industry_age, index_membership = (
        _build_rebuildable_membership_panels(sources, qdp_paths=qdp_paths)
    )
    repaired = np.memmap(
        partial_path,
        dtype=np.float32,
        mode="r+",
        shape=(len(sources.context.row_index), len(override_names)),
    )
    pit_universe = base._open_array(
        sources.context.pack["masks"]["pit_universe_has_bar"], dtype=np.bool_
    )
    status_st = base._open_array(sources.context.pack["masks"]["is_st"], dtype=np.bool_)
    status_suspended = base._open_array(
        sources.context.pack["masks"]["is_suspended"], dtype=np.bool_
    )
    f2_count = len(f2_names)
    market_offset = f2_count
    industry_offset = f2_count + len(market_names)
    for position, block in enumerate(
        date_blocks[start_position:], start=start_position
    ):
        date_idx = int(block.date_idx)
        history = slice(date_idx - max(live.FEATURE_WINDOWS), date_idx + 1)
        raw = np.asarray(sources.daily_raw[history], dtype=np.float32)
        market_values = live._market_features(
            raw=raw,
            universe=np.asarray(pit_universe[date_idx], dtype=bool),
            index_membership={
                name: index_membership[name][date_idx] for name in index_membership
            },
            is_suspended=np.asarray(status_suspended[date_idx], dtype=bool),
            is_st=np.asarray(status_st[date_idx], dtype=bool),
        )
        selected_symbols = sources.context.row_index.iloc[block.start : block.stop][
            "symbol_idx"
        ].to_numpy(dtype=np.int32)
        industry_values = live._industry_features(
            raw=raw,
            universe=np.asarray(pit_universe[date_idx], dtype=bool),
            industry=industry_codes[date_idx],
            industry_source_age=industry_age[date_idx],
            selected_symbols=selected_symbols,
        )
        values = np.full(
            (block.stop - block.start, len(override_names)),
            np.nan,
            dtype=np.float32,
        )
        values[:, :f2_count] = np.asarray(
            repaired_rank.values[block.start : block.stop], dtype=np.float32
        )
        for index, name in enumerate(market_names):
            values[:, market_offset + index] = np.float32(market_values[name])
        for index, name in enumerate(industry_names):
            values[:, industry_offset + index] = industry_values[name]
        repaired[block.start : block.stop] = values
        completed = position + 1
        if completed % 25 == 0 or completed == len(date_blocks):
            repaired.flush()
            _write_json(
                progress_path,
                {"fingerprint": fingerprint, "completed_date_count": completed},
            )
        if completed % 250 == 0:
            base._emit(
                "rebuildable_core_progress",
                completed_date_count=completed,
                total_date_count=len(date_blocks),
            )
    repaired.flush()
    del repaired, industry_codes, industry_age, index_membership
    gc.collect()
    os.replace(partial_path, values_path)
    progress_path.unlink(missing_ok=True)
    values = np.memmap(
        values_path,
        dtype=np.float32,
        mode="r",
        shape=(len(sources.context.row_index), len(override_names)),
    )
    result = {
        "schema": "seq100_full_market_rebuildable_core_features/1",
        "status": "completed",
        "completed_at": _now(),
        "fingerprint": fingerprint,
        "row_count": len(sources.context.row_index),
        "feature_count": len(override_names),
        "feature_names": list(override_names),
        "model_columns": override_columns.tolist(),
        "formal_minimum_date": date_blocks[0].trade_date,
        "formal_maximum_date": date_blocks[-1].trade_date,
        "causality": {
            "industry_and_index_source": "active QDP date-specific snapshots",
            "underlying_price_source": "causal daily pack",
            "future_return_or_fill_read": False,
            "forbidden_2026_row_count": 0,
        },
        "file": _file_record(
            values_path,
            dtype="float32",
            shape=[len(sources.context.row_index), len(override_names)],
        ),
        "sources": {
            "repaired_cross_sectional": _file_record(
                output_root / "repaired_cross_sectional_features/manifest.json"
            ),
        },
    }
    _write_json(manifest_path, result)
    return StaticFeatureOverride(
        values=values,
        columns=override_columns,
        feature_names=override_names,
        manifest=result,
    )


def _active_source_boundaries() -> dict[str, Any]:
    active_path = (
        base.WORKSPACE_ROOT / "quant_data_platform/data/qdp_v2/active/active.json"
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    qdp_root = active_path.parents[1]
    boundaries: dict[str, Any] = {}
    for domain, dataset_id in dict(active["datasets"]).items():
        metadata_path = qdp_root / "datasets" / domain / dataset_id / "dataset.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        boundaries[str(domain)] = {
            "dataset_id": str(dataset_id),
            "end_date": metadata.get("end_date"),
            "metadata_sha256": base._sha256(metadata_path),
        }
    return {
        "active_as_of_date": str(active["active_as_of_date"]),
        "active_sha256": base._sha256(active_path),
        "datasets": boundaries,
    }


def _live_outage_feature_contract(
    sources: SequenceSources,
) -> tuple[tuple[str, ...], list[dict[str, Any]], np.ndarray]:
    families = ("announcements", "traditional_moneyflow")
    records, columns = _masked_feature_contract(
        sources, families=families, expected_count=45
    )
    return families, records, columns


def _masked_feature_contract(
    sources: SequenceSources,
    *,
    families: Sequence[str],
    expected_count: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    family_set = frozenset(str(value) for value in families)
    records = [
        record
        for record in sources.context.model_manifest["features"]
        if str(record["analytic_family"]) in family_set
    ]
    columns = np.asarray(
        sorted(int(record["column_index"]) for record in records), dtype=np.int32
    )
    if len(columns) != int(expected_count) or len(np.unique(columns)) != int(
        expected_count
    ):
        raise SequenceChallengerError("availability_mask_feature_count_changed")
    return records, columns


def _source_exact_core_feature_contract(
    sources: SequenceSources,
) -> tuple[tuple[str, ...], list[dict[str, Any]], np.ndarray]:
    """Mask fields whose historical source semantics cannot be rebuilt exactly.

    The retained live subset is deliberately mechanical: direct daily path
    fields, all-market (not index-specific) state, size/status fields, and the
    exact same-day five-minute summaries.
    """

    retained_families = {
        "daily_price_volume_technical",
        "size_liquidity_and_status",
        "same_day_5m",
    }
    records = list(sources.context.model_manifest["features"])
    retained = {
        str(record["feature_name"])
        for record in records
        if str(record["analytic_family"]) in retained_families
        or (
            str(record["analytic_family"]) == "market_state"
            and str(record["feature_name"]).startswith("market_all__")
        )
    }
    masked_records = [
        record for record in records if str(record["feature_name"]) not in retained
    ]
    columns = np.asarray(
        sorted(int(record["column_index"]) for record in masked_records),
        dtype=np.int32,
    )
    if len(retained) != 161 or len(columns) != 396 or len(np.unique(columns)) != 396:
        raise SequenceChallengerError("source_exact_core_feature_count_changed")
    labels = (
        "non_market_path_families",
        "daily_cross_sectional_source_semantics",
        "industry_source_semantics",
        "index_specific_market_state",
    )
    return labels, masked_records, columns


def _repaired_rank_core_feature_contract(
    sources: SequenceSources,
) -> tuple[tuple[str, ...], list[dict[str, Any]], np.ndarray]:
    """Keep repaired current-universe percentiles but mask larger source drift."""

    retained_families = {
        "daily_price_volume_technical",
        "daily_cross_sectional_technical",
        "size_liquidity_and_status",
        "same_day_5m",
    }
    records = list(sources.context.model_manifest["features"])
    retained = {
        str(record["feature_name"])
        for record in records
        if str(record["analytic_family"]) in retained_families
        or (
            str(record["analytic_family"]) == "market_state"
            and str(record["feature_name"]).startswith("market_all__")
        )
    }
    masked_records = [
        record for record in records if str(record["feature_name"]) not in retained
    ]
    columns = np.asarray(
        sorted(int(record["column_index"]) for record in masked_records),
        dtype=np.int32,
    )
    if len(retained) != 266 or len(columns) != 291 or len(np.unique(columns)) != 291:
        raise SequenceChallengerError("repaired_rank_core_feature_count_changed")
    labels = (
        "non_market_path_families",
        "industry_source_semantics",
        "index_specific_market_state",
    )
    return labels, masked_records, columns


def evaluate_source_availability_stress(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    lookback: int = DEFAULT_LOOKBACK,
    date_batch_size: int = DEFAULT_DATE_BATCH_SIZE,
    starting_cash: float = 1_000_000.0,
    availability_profile: str = "current_source_outage",
) -> dict[str, Any]:
    """Mask currently unavailable 2026 feature families and rerun frozen OOF heads."""

    if not torch.cuda.is_available():
        raise SequenceChallengerError("cuda_is_required_for_availability_stress")
    study = base.load_study(study_path)
    sources = _load_sources(study, output_root=output_root)
    static_feature_override: StaticFeatureOverride | None = None
    if availability_profile == "current_source_outage":
        family_names, masked_records, masked_columns = _live_outage_feature_contract(
            sources
        )
        output_name = "source_availability_stress"
    elif availability_profile == "rebuildable_market_path_core":
        retained = frozenset(
            {
                "daily_price_volume_technical",
                "daily_cross_sectional_technical",
                "market_state",
                "industry_context",
                "size_liquidity_and_status",
                "same_day_5m",
            }
        )
        all_families = {
            str(record["analytic_family"])
            for record in sources.context.model_manifest["features"]
        }
        family_names = tuple(sorted(all_families.difference(retained)))
        masked_records, masked_columns = _masked_feature_contract(
            sources, families=family_names, expected_count=243
        )
        output_name = "core_source_availability_stress"
    elif availability_profile == "source_exact_market_path_core":
        family_names, masked_records, masked_columns = (
            _source_exact_core_feature_contract(sources)
        )
        output_name = "exact_core_source_availability_stress"
    elif availability_profile == "repaired_rank_market_path_core":
        family_names, masked_records, masked_columns = (
            _repaired_rank_core_feature_contract(sources)
        )
        static_feature_override = prepare_repaired_cross_sectional_features(
            sources, output_root=output_root
        )
        output_name = "corrected_rank_core_source_availability_stress"
    elif availability_profile == "corrected_rank_rebuildable_market_path_core":
        retained = frozenset(
            {
                "daily_price_volume_technical",
                "daily_cross_sectional_technical",
                "market_state",
                "industry_context",
                "size_liquidity_and_status",
                "same_day_5m",
            }
        )
        all_families = {
            str(record["analytic_family"])
            for record in sources.context.model_manifest["features"]
        }
        family_names = tuple(sorted(all_families.difference(retained)))
        masked_records, masked_columns = _masked_feature_contract(
            sources, families=family_names, expected_count=243
        )
        static_feature_override = prepare_repaired_cross_sectional_features(
            sources, output_root=output_root
        )
        output_name = "corrected_rank_full_core_source_availability_stress"
    elif availability_profile == "rebuildable_current_core":
        retained = frozenset(
            {
                "daily_price_volume_technical",
                "daily_cross_sectional_technical",
                "market_state",
                "industry_context",
                "size_liquidity_and_status",
                "same_day_5m",
            }
        )
        all_families = {
            str(record["analytic_family"])
            for record in sources.context.model_manifest["features"]
        }
        family_names = tuple(sorted(all_families.difference(retained)))
        masked_records, masked_columns = _masked_feature_contract(
            sources, families=family_names, expected_count=243
        )
        from daily_research.path_policy import (
            seq100_quality_liquidity_data_prep as prep,
        )

        _, qdp_paths = prep._qdp_snapshot(base.WORKSPACE_ROOT)
        static_feature_override = prepare_rebuildable_core_features(
            sources, output_root=output_root, qdp_paths=qdp_paths
        )
        output_name = "rebuildable_current_core_source_availability_stress"
    else:
        raise SequenceChallengerError(
            f"unknown_availability_profile:{availability_profile}"
        )
    target_columns = list(sources.target_manifest["target_columns"])
    sources.base_column = target_columns.index("exact_net_return_d3_base")
    sources.stress_column = target_columns.index("exact_net_return_d3_stress")
    path_columns = list(sources.path_manifest["target_columns"])
    sources.gross_column = path_columns.index("legal_exit_return_d3")
    horizons = list(sources.path_manifest["files"]["legal_fill_days"]["horizons"])
    sources.horizon_column = horizons.index(3)
    folds = base.build_forward_folds(
        date_idx=sources.context.row_index["date_idx"].to_numpy(dtype=np.int32),
        trade_date=sources.context.row_index["trade_date"].astype(str).to_numpy(),
        validation_start_date=str(study["validation"]["validation_start_date"]),
        validation_end_date=str(study["validation"]["validation_end_date"]),
        fold_count=int(study["validation"]["forward_fold_count"]),
        purge_days=int(study["validation"]["common_purge_trading_days"]),
    )
    tree_tasks = [
        base._oof_task_result(
            output_root,
            fold=int(fold["fold"]),
            target="exact_net_return_d5_rank",
            profile=str(study["lightgbm"]["primary_profile"]),
            feature_variant=base.DEFAULT_FEATURE_VARIANT,
            training_mode="causal_nested",
        )
        for fold in folds
    ]
    sequence_tasks = [
        _completed_sequence_task(output_root, fold=int(fold["fold"]), lookback=lookback)
        for fold in folds
    ]
    source_boundaries = _active_source_boundaries()
    candidate_path = (
        base.WORKSPACE_ROOT
        / "daily_research/studies/seq100_full_market_dual_gate_d3_candidate_v1.json"
    )
    fingerprint = _stable_hash(
        {
            "schema": AVAILABILITY_SCHEMA,
            "implementation_sha256": base._sha256(Path(__file__)),
            "candidate_sha256": base._sha256(candidate_path),
            "input_fingerprint": sources.context.model_manifest["input_fingerprint"],
            "masked_families": family_names,
            "availability_profile": availability_profile,
            "masked_features": [
                str(record["feature_name"]) for record in masked_records
            ],
            "static_feature_override_fingerprint": (
                static_feature_override.manifest["fingerprint"]
                if static_feature_override is not None
                else None
            ),
            "tree_tasks": {str(path): base._sha256(path) for _, path in tree_tasks},
            "sequence_tasks": {
                str(path): base._sha256(path) for _, path in sequence_tasks
            },
            "active_source_boundaries": source_boundaries,
            "stock_score": "equal_tree_sequence_percentile_rank",
            "market_gate": "unchanged_tree_and_neural_consensus",
            "exit": "d3_legal_exit",
            "top_k": 10,
            "starting_cash": float(starting_cash),
        }
    )
    output_dir = output_root / output_name
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            current.get("status") == "completed"
            and current.get("fingerprint") == fingerprint
        ):
            return current
    resource = base.resource_plan(
        reserve_gib=float(study["resources"]["reserve_system_gib"]),
        maximum_threads=int(study["resources"]["maximum_cpu_threads"]),
        histogram_pool_cap_mb=int(study["resources"]["histogram_pool_cap_mb"]),
        sequence_batch_cap=int(study["resources"]["sequence_batch_cap"]),
    )
    device = torch.device("cuda")
    daily_parts: list[pd.DataFrame] = []
    selection_parts: list[pd.DataFrame] = []
    neural_market_parts: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    for fold, (tree_task, tree_path), (sequence_task, sequence_path) in zip(
        folds, tree_tasks, sequence_tasks, strict=True
    ):
        fold_number = int(fold["fold"])
        rows = base._fold_rows(sources.context.row_index, fold, "validation")
        tree_prediction = _masked_tree_prediction(
            model_path=Path(tree_task["files"]["model"]["path"]),
            best_iteration=int(tree_task["best_iteration"]),
            matrix=sources.matrix,
            rows=rows,
            masked_columns=masked_columns,
            batch_size=int(resource.sequence_batch_size),
            static_feature_override=static_feature_override,
        )
        checkpoint_path = Path(sequence_task["files"]["model"]["path"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        normalization = Normalization.from_payload(checkpoint["normalization"])
        model = PayoffSequenceModel().to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        assembler = BatchAssembler(
            sources,
            lookback=lookback,
            normalization=normalization,
            device=device,
            require_targets=False,
            masked_static_columns=masked_columns,
            static_feature_override=static_feature_override,
        )
        validation_dates = sorted(
            date_idx
            for date_idx in sources.blocks
            if int(fold["validation_start_date_idx"])
            <= date_idx
            <= int(fold["validation_end_date_idx"])
        )
        sequence_prediction = _predict_dates(
            model,
            assembler,
            validation_dates,
            date_batch_size=date_batch_size,
        )
        if not np.array_equal(sequence_prediction.rows, rows):
            raise SequenceChallengerError(
                "availability_sequence_prediction_alignment_failed"
            )
        frame = sources.context.row_index.iloc[rows][
            ["candidate_id", "date_idx"]
        ].copy()
        frame["tree_prediction"] = tree_prediction
        frame["sequence_prediction"] = sequence_prediction.stock_score
        frame["tree_rank"] = frame.groupby("date_idx", sort=False)[
            "tree_prediction"
        ].rank(pct=True)
        frame["sequence_rank"] = frame.groupby("date_idx", sort=False)[
            "sequence_prediction"
        ].rank(pct=True)
        ensemble_score = (
            frame["tree_rank"].to_numpy(dtype=np.float32)
            + frame["sequence_rank"].to_numpy(dtype=np.float32)
        ) / np.float32(2.0)
        combined_prediction = PredictionOutput(
            rows=rows,
            stock_score=ensemble_score,
            market_return=sequence_prediction.market_return,
            market_probability=sequence_prediction.market_probability,
        )
        daily, _, selections, payoff = _build_evaluation_frames(
            sources,
            combined_prediction,
            fold=fold_number,
            unfilled_as_cash=True,
        )
        daily_parts.append(daily)
        selection_parts.append(selections)
        evaluated_dates = sorted(sequence_prediction.market_return)
        neural_market_parts.append(
            pd.DataFrame(
                {
                    "date_idx": evaluated_dates,
                    "predicted_return": [
                        sequence_prediction.market_return[value]
                        for value in evaluated_dates
                    ],
                    "positive_probability": [
                        sequence_prediction.market_probability[value]
                        for value in evaluated_dates
                    ],
                }
            )
        )
        fold_records.append(
            {
                "fold": fold_number,
                "row_count": len(rows),
                "payoff": payoff,
                "tree_task_sha256": base._sha256(tree_path),
                "sequence_task_sha256": base._sha256(sequence_path),
            }
        )
        del model, assembler, checkpoint, tree_prediction, sequence_prediction, frame
        gc.collect()
        torch.cuda.empty_cache()
        base._trim_working_set()
        base._emit(
            "source_availability_stress_fold_completed",
            fold=fold_number,
            rank_ic=payoff["daily_rank_ic_mean"],
        )
    daily = pd.concat(daily_parts, ignore_index=True)
    selections = pd.concat(selection_parts, ignore_index=True)
    neural_market = (
        pd.concat(neural_market_parts, ignore_index=True)
        .drop_duplicates("date_idx", keep="last")
        .set_index("date_idx")
    )
    tree_market, tree_market_manifest = _formal_market_predictions(output_root)
    common_dates = sorted(
        set(tree_market.index.astype(int)).intersection(neural_market.index.astype(int))
    )
    dual_gate = (
        tree_market.loc[common_dates, "ridge_prediction"].gt(0.0)
        & tree_market.loc[common_dates, "positive_probability"].gt(0.5)
        & neural_market.loc[common_dates, "predicted_return"].gt(0.0)
        & neural_market.loc[common_dates, "positive_probability"].gt(0.5)
    )
    dual_gate.index = np.asarray(common_dates, dtype=np.int32)
    active_mask = selections["date_idx"].map(dual_gate).eq(True).to_numpy(dtype=bool)
    active = selections.loc[active_mask].copy()
    rows = active["model_row_position"].to_numpy(dtype=np.int64)
    known = ~active["market_entry_filled"].to_numpy(dtype=bool) | np.asarray(
        sources.path_valid[rows, sources.gross_column], dtype=bool
    )
    date_known = pd.Series(known, index=active.index).groupby(active["date_idx"]).all()
    active = active.loc[active["date_idx"].map(date_known).fillna(False)].copy()
    rows = active["model_row_position"].to_numpy(dtype=np.int64)
    active["legal_gross_return"] = np.asarray(
        sources.path_values[rows, sources.gross_column], dtype=np.float32
    )
    active["fill_day"] = np.asarray(
        sources.fill_days[rows, sources.horizon_column], dtype=np.int16
    )
    active["variant"] = "dual_market_equal_rank_source_outage"
    active["exit_policy"] = "planned_close"
    active["gate"] = "tree_and_neural_consensus"
    active["top_k"] = 10
    active["cost_scenario"] = "base"
    active["take_profit_hit"] = False
    spec = {
        "variant": "dual_market_equal_rank_source_outage",
        "exit_policy": "planned_close",
        "gate": "tree_and_neural_consensus",
        "top_k": 10,
        "cost_scenario": "stress",
        "slippage_multiplier": 2.0,
        "cohort_equity_fraction": 1.0 / 3.0,
        "planned_fill_day": 3,
    }
    daily_raw = base._open_array(
        sources.context.pack["feature_channels"]["daily_raw"], dtype=np.float32
    )
    raw_open = base._open_array(
        sources.context.pack["execution_arrays"]["entry_open_raw"], dtype=np.float32
    )
    market_entry_filled = base._open_array(
        sources.context.pack["masks"]["entry_filled"], dtype=np.bool_
    )
    account, equity, trades = base._simulate_account_spec(
        spec=spec,
        selections=active,
        context=sources.context,
        daily_raw=daily_raw,
        raw_open=raw_open,
        entry_filled=market_entry_filled,
        costs=base.parse_execution_costs(sources.context.pack),
        starting_cash=float(starting_cash),
    )
    account["annualized_net_return"] = float(
        (account["ending_equity"] / float(starting_cash))
        ** (242.0 / max(len(equity), 1))
        - 1.0
    )
    account["daily_return_hac"] = base._newey_west_interval(
        equity["daily_net_return"].to_numpy(dtype=np.float64), lag=20
    )
    total_pnl = float(trades["pnl"].sum()) if len(trades) else 0.0
    account["largest_10_trade_pnl_share"] = (
        float(trades.nlargest(10, "pnl")["pnl"].sum() / total_pnl)
        if total_pnl > 0.0
        else math.nan
    )
    baseline_manifest_path = output_root / "dual_market_horizon_challenge/manifest.json"
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    baseline = next(
        item
        for item in baseline_manifest["summaries"]
        if item["variant_name"] == "h3__top10__uncapped__all_folds"
    )
    original_selections = pd.read_parquet(
        output_root / "sequence_ensemble_evaluation/top10_selections.parquet",
        columns=["date_idx", "candidate_id"],
    )
    original_keys = set(
        zip(
            original_selections["date_idx"].astype(int),
            original_selections["candidate_id"].astype(int),
            strict=True,
        )
    )
    active_keys = set(
        zip(
            active["date_idx"].astype(int),
            active["candidate_id"].astype(int),
            strict=True,
        )
    )
    selection_overlap = len(original_keys & active_keys) / max(len(active_keys), 1)
    gate = {
        "requires_positive_daily_hac_lower": True,
        "requires_six_positive_years": True,
        "requires_drawdown_above_minus_20pct": True,
        "passed": bool(
            account["daily_return_hac"]["lower"] is not None
            and float(account["daily_return_hac"]["lower"]) > 0.0
            and int(account["positive_year_count"]) == 6
            and float(account["maximum_drawdown"]) > -0.20
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / "daily_metrics.parquet"
    selections_path = output_dir / "top10_selections.parquet"
    equity_path = output_dir / "equity.parquet"
    trades_path = output_dir / "trades.parquet"
    base._write_parquet(daily, daily_path)
    base._write_parquet(active, selections_path)
    base._write_parquet(equity, equity_path)
    base._write_parquet(trades, trades_path)
    result = {
        "schema": AVAILABILITY_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": base.STUDY_ID,
        "fingerprint": fingerprint,
        "availability_profile": availability_profile,
        "masked_families": list(family_names),
        "masked_feature_count": len(masked_columns),
        "masked_features": [str(record["feature_name"]) for record in masked_records],
        "repaired_cross_sectional_features": (
            {
                "enabled": True,
                "feature_count": len(static_feature_override.feature_names),
                "fingerprint": static_feature_override.manifest["fingerprint"],
                "source": _file_record(
                    output_root / "repaired_cross_sectional_features/manifest.json"
                ),
            }
            if static_feature_override is not None
            else {"enabled": False}
        ),
        "source_boundaries": source_boundaries,
        "fold_records": fold_records,
        "account": account,
        "baseline_account": {
            key: baseline[key]
            for key in (
                "ending_equity",
                "total_net_return",
                "annualized_net_return",
                "maximum_drawdown",
                "positive_year_count",
            )
        },
        "selection_overlap_fraction": float(selection_overlap),
        "availability_gate": gate,
        "decision_boundary": {
            "mask_was_fixed_from_current_source_metadata": True,
            "threshold_or_family_grid_searched": False,
            "technical_indicators_are_recomputable_from_current_market_data": True,
            "financial_values_may_carry_forward_with_causal_age": True,
            "historical_result_is_adaptive_robustness_evidence": True,
            "market_gate_and_sequence_market_branch_keep_all_54_market_fields": True,
            "stable_profit_claim_allowed": False,
            "forbidden_2026_outcome_read_count": 0,
        },
        "files": {
            "daily_metrics": _file_record(daily_path, row_count=len(daily)),
            "top10_selections": _file_record(selections_path, row_count=len(active)),
            "equity": _file_record(equity_path, row_count=len(equity)),
            "trades": _file_record(trades_path, row_count=len(trades)),
        },
        "sources": {
            "candidate": _file_record(candidate_path),
            "baseline": _file_record(baseline_manifest_path),
            "tree_market": _file_record(tree_market_manifest),
        },
    }
    _write_json(manifest_path, result)
    base._emit(
        "source_availability_stress_completed",
        gate=gate["passed"],
        total_return=account["total_net_return"],
        maximum_drawdown=account["maximum_drawdown"],
        overlap=selection_overlap,
    )
    return result


def replay_rebuildable_core_robustness(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    starting_cash: float = 1_000_000.0,
) -> dict[str, Any]:
    """Stress the rebuilt-source candidate without searching a new policy."""

    source_path = (
        output_root
        / "rebuildable_current_core_source_availability_stress/manifest.json"
    )
    if not source_path.is_file():
        raise SequenceChallengerError("rebuildable_current_core_stress_missing")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != "completed" or not bool(
        source.get("availability_gate", {}).get("passed")
    ):
        raise SequenceChallengerError("rebuildable_current_core_stress_not_passed")
    selection_path = Path(source["files"]["top10_selections"]["path"])
    selections = pd.read_parquet(selection_path)
    required = {
        "selection_rank",
        "fold",
        "date_idx",
        "candidate_id",
        "model_row_position",
        "market_entry_filled",
        "symbol",
        "variant",
    }
    if not required.issubset(selections.columns):
        raise SequenceChallengerError("rebuildable_selection_contract_changed")
    variants = [
        {
            "name": "primary_h3_top10_2x",
            "horizon": 3,
            "top_k": 10,
            "slippage_multiplier": 2.0,
            "minimum_fold": 1,
        },
        {
            "name": "exit_h2_top10_2x",
            "horizon": 2,
            "top_k": 10,
            "slippage_multiplier": 2.0,
            "minimum_fold": 1,
        },
        {
            "name": "exit_h5_top10_2x",
            "horizon": 5,
            "top_k": 10,
            "slippage_multiplier": 2.0,
            "minimum_fold": 1,
        },
        {
            "name": "concentration_h3_top1_2x",
            "horizon": 3,
            "top_k": 1,
            "slippage_multiplier": 2.0,
            "minimum_fold": 1,
        },
        {
            "name": "concentration_h3_top3_2x",
            "horizon": 3,
            "top_k": 3,
            "slippage_multiplier": 2.0,
            "minimum_fold": 1,
        },
        {
            "name": "cost_h3_top10_3x",
            "horizon": 3,
            "top_k": 10,
            "slippage_multiplier": 3.0,
            "minimum_fold": 1,
        },
        {
            "name": "tail_cap10_h3_top10_2x",
            "horizon": 3,
            "top_k": 10,
            "slippage_multiplier": 2.0,
            "minimum_fold": 1,
            "maximum_credited_gross_return": 0.10,
        },
        {
            "name": "tail_cap5_h3_top10_2x",
            "horizon": 3,
            "top_k": 10,
            "slippage_multiplier": 2.0,
            "minimum_fold": 1,
            "maximum_credited_gross_return": 0.05,
        },
        {
            "name": "confirmation_h3_top10_2x_folds2to5",
            "horizon": 3,
            "top_k": 10,
            "slippage_multiplier": 2.0,
            "minimum_fold": 2,
        },
    ]
    fingerprint = _stable_hash(
        {
            "schema": REBUILDABLE_ROBUSTNESS_SCHEMA,
            "source_sha256": base._sha256(source_path),
            "selection_sha256": base._sha256(selection_path),
            "study_sha256": base._sha256(study_path),
            "path_target_manifest_sha256": base._sha256(
                output_root / "targets" / "manifest.json"
            ),
            "starting_cash": float(starting_cash),
            "variants": variants,
            "no_variant_selection": True,
        }
    )
    output_dir = output_root / "rebuildable_core_account_robustness"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            current.get("status") == "completed"
            and current.get("fingerprint") == fingerprint
        ):
            return current
    study = base.load_study(study_path)
    sources = _load_sources(study, output_root=output_root)
    context = sources.context
    daily_raw = sources.daily_raw
    raw_open = base._open_array(
        context.pack["execution_arrays"]["entry_open_raw"], dtype=np.float32
    )
    entry_filled = sources.entry_filled
    costs = base.parse_execution_costs(context.pack)
    target_columns = list(sources.path_manifest["target_columns"])
    horizons = list(sources.path_manifest["files"]["legal_fill_days"]["horizons"])
    source_variants = selections["variant"].drop_duplicates().astype(str).tolist()
    if len(source_variants) != 1:
        raise SequenceChallengerError(
            f"rebuildable_selection_variant_changed:{source_variants}"
        )
    source_variant = source_variants[0]
    summaries: list[dict[str, Any]] = []
    task_files: dict[str, Any] = {}
    for variant in variants:
        horizon = int(variant["horizon"])
        top_k = int(variant["top_k"])
        current = selections.loc[
            (selections["selection_rank"] <= top_k)
            & (selections["fold"] >= int(variant["minimum_fold"]))
        ].copy()
        gross_column = target_columns.index(f"legal_exit_return_d{horizon}")
        horizon_column = horizons.index(horizon)
        rows = current["model_row_position"].to_numpy(dtype=np.int64)
        known = ~current["market_entry_filled"].to_numpy(dtype=bool) | np.asarray(
            sources.path_valid[rows, gross_column], dtype=bool
        )
        date_known = (
            pd.Series(known, index=current.index).groupby(current["date_idx"]).all()
        )
        current = current.loc[
            current["date_idx"].map(date_known).fillna(False)
        ].copy()
        rows = current["model_row_position"].to_numpy(dtype=np.int64)
        current["legal_gross_return"] = np.asarray(
            sources.path_values[rows, gross_column], dtype=np.float32
        )
        current["fill_day"] = np.asarray(
            sources.fill_days[rows, horizon_column], dtype=np.int16
        )
        # Reuse the source selector's identity.  The account simulator applies
        # the requested stress costs at replay time, but it intentionally
        # matches selections on the source run's ``variant`` and ``base``
        # selection cost label.  Replacing either label here makes the schedule
        # empty even though the parquet contains valid selections.
        current["variant"] = source_variant
        current["exit_policy"] = "planned_close"
        current["gate"] = "tree_and_neural_consensus"
        current["top_k"] = top_k
        current["cost_scenario"] = "base"
        current["take_profit_hit"] = False
        spec: dict[str, Any] = {
            "variant": source_variant,
            "exit_policy": "planned_close",
            "gate": "tree_and_neural_consensus",
            "top_k": top_k,
            "cost_scenario": "stress",
            "slippage_multiplier": float(variant["slippage_multiplier"]),
            "cohort_equity_fraction": 1.0 / float(horizon),
            "planned_fill_day": horizon,
        }
        if "maximum_credited_gross_return" in variant:
            spec["maximum_credited_gross_return"] = float(
                variant["maximum_credited_gross_return"]
            )
        result, equity, trades = base._simulate_account_spec(
            spec=spec,
            selections=current,
            context=context,
            daily_raw=daily_raw,
            raw_open=raw_open,
            entry_filled=entry_filled,
            costs=costs,
            starting_cash=float(starting_cash),
        )
        result["variant_name"] = str(variant["name"])
        result["minimum_fold"] = int(variant["minimum_fold"])
        result["annualized_net_return"] = float(
            (result["ending_equity"] / float(starting_cash))
            ** (242.0 / max(len(equity), 1))
            - 1.0
        )
        result["daily_return_hac"] = base._newey_west_interval(
            equity["daily_net_return"].to_numpy(dtype=np.float64), lag=20
        )
        result["mean_invested_fraction"] = float(
            np.mean(
                1.0
                - equity["cash"].to_numpy(dtype=np.float64)
                / equity["equity"].to_numpy(dtype=np.float64)
            )
        )
        total_pnl = float(trades["pnl"].sum()) if len(trades) else 0.0
        result["largest_10_trade_pnl_share"] = (
            float(trades.nlargest(10, "pnl")["pnl"].sum() / total_pnl)
            if total_pnl > 0.0
            else math.nan
        )
        task_dir = output_dir / "tasks" / str(variant["name"])
        equity_path = task_dir / "equity.parquet"
        trades_path = task_dir / "trades.parquet"
        task_path = task_dir / "task_result.json"
        base._write_parquet(equity, equity_path)
        base._write_parquet(trades, trades_path)
        payload = {
            "schema": REBUILDABLE_ROBUSTNESS_SCHEMA,
            "status": "completed",
            "completed_at": _now(),
            "fingerprint": fingerprint,
            **result,
            "files": {
                "equity": _file_record(equity_path, row_count=len(equity)),
                "trades": _file_record(trades_path, row_count=len(trades)),
            },
        }
        _write_json(task_path, payload)
        task_files[str(variant["name"])] = _file_record(task_path)
        summaries.append(result)
    indexed = {str(item["variant_name"]): item for item in summaries}
    required_positive = (
        "primary_h3_top10_2x",
        "exit_h2_top10_2x",
        "exit_h5_top10_2x",
        "concentration_h3_top3_2x",
        "cost_h3_top10_3x",
        "tail_cap10_h3_top10_2x",
        "confirmation_h3_top10_2x_folds2to5",
    )
    primary = indexed["primary_h3_top10_2x"]
    robustness_gate = {
        "required_positive_variants": list(required_positive),
        "cap5_is_diagnostic_not_a_required_trading_rule": True,
        "requires_primary_positive_hac_lower": True,
        "requires_primary_six_positive_years": True,
        "requires_primary_drawdown_above_minus_20pct": True,
        "passed": bool(
            all(
                float(indexed[name]["total_net_return"]) > 0.0
                for name in required_positive
            )
            and primary["daily_return_hac"]["lower"] is not None
            and float(primary["daily_return_hac"]["lower"]) > 0.0
            and int(primary["positive_year_count"]) == 6
            and float(primary["maximum_drawdown"]) > -0.20
        ),
    }
    summary_frame = pd.DataFrame(
        [
            {
                "variant": item["variant_name"],
                "horizon": item["spec"]["planned_fill_day"],
                "top_k": item["spec"]["top_k"],
                "slippage_multiplier": item["spec"]["slippage_multiplier"],
                "maximum_credited_gross_return": item["spec"].get(
                    "maximum_credited_gross_return"
                ),
                "minimum_fold": item["minimum_fold"],
                "total_net_return": item["total_net_return"],
                "annualized_net_return": item["annualized_net_return"],
                "maximum_drawdown": item["maximum_drawdown"],
                "positive_year_count": item["positive_year_count"],
                "worst_year_return": item["worst_year_return"],
                "daily_hac_lower": item["daily_return_hac"]["lower"],
                "largest_10_trade_pnl_share": item["largest_10_trade_pnl_share"],
                "trade_count": item["trade_count"],
            }
            for item in summaries
        ]
    )
    summary_path = output_dir / "summary.parquet"
    base._write_parquet(summary_frame, summary_path)
    result = {
        "schema": REBUILDABLE_ROBUSTNESS_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": base.STUDY_ID,
        "fingerprint": fingerprint,
        "variant_count": len(variants),
        "summaries": summaries,
        "robustness_gate": robustness_gate,
        "decision_boundary": {
            "candidate_and_market_gate_frozen_before_this_test": True,
            "variants_are_diagnostics_not_a_new_policy_search": True,
            "historical_result_is_adaptive_development_evidence": True,
            "stable_profit_claim_allowed": False,
            "forbidden_2026_outcome_read_count": 0,
        },
        "files": {
            "summary": _file_record(summary_path, row_count=len(summary_frame)),
            "tasks": task_files,
        },
        "sources": {
            "rebuildable_current_core_stress": _file_record(source_path),
            "selections": _file_record(selection_path),
        },
    }
    _write_json(manifest_path, result)
    return result


def freeze_final_policy_bundle(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    lookback: int = DEFAULT_LOOKBACK,
    date_batch_size: int = DEFAULT_DATE_BATCH_SIZE,
) -> dict[str, Any]:
    """Refit the frozen candidate on all causally labeled history through 2025."""

    if not torch.cuda.is_available():
        raise SequenceChallengerError("cuda_is_required_for_final_bundle")
    study = base.load_study(study_path)
    sources = _load_sources(study, output_root=output_root)
    candidate_path = (
        base.WORKSPACE_ROOT
        / "daily_research/studies/seq100_full_market_dual_gate_d3_candidate_v1.json"
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if (
        candidate.get("status") != "frozen_adaptive_historical_candidate"
        or candidate["frozen_policy"]["tree_stock_head"]
        != "exact_net_return_d5_rank__strong_127__causal_nested"
        or candidate["frozen_policy"]["exit"]
        != "D3 close request, otherwise first legally sellable open"
    ):
        raise SequenceChallengerError("frozen_candidate_contract_changed")
    availability_path = output_root / "source_availability_stress/manifest.json"
    availability = json.loads(availability_path.read_text(encoding="utf-8"))
    if not bool(availability["availability_gate"]["passed"]):
        raise SequenceChallengerError("source_availability_gate_failed")
    family_names, masked_records, masked_columns = _live_outage_feature_contract(
        sources
    )
    folds = base.build_forward_folds(
        date_idx=sources.context.row_index["date_idx"].to_numpy(dtype=np.int32),
        trade_date=sources.context.row_index["trade_date"].astype(str).to_numpy(),
        validation_start_date=str(study["validation"]["validation_start_date"]),
        validation_end_date=str(study["validation"]["validation_end_date"]),
        fold_count=int(study["validation"]["forward_fold_count"]),
        purge_days=int(study["validation"]["common_purge_trading_days"]),
    )
    tree_tasks = [
        base._oof_task_result(
            output_root,
            fold=int(fold["fold"]),
            target="exact_net_return_d5_rank",
            profile=str(study["lightgbm"]["primary_profile"]),
            feature_variant=base.DEFAULT_FEATURE_VARIANT,
            training_mode="causal_nested",
        )
        for fold in folds
    ]
    sequence_tasks = [
        _completed_sequence_task(output_root, fold=int(fold["fold"]), lookback=lookback)
        for fold in folds
    ]
    tree_iterations = [int(task[0]["best_iteration"]) for task in tree_tasks]
    sequence_epochs = [int(task[0]["best_epoch"]) for task in sequence_tasks]
    final_tree_iterations = int(frozen_median(tree_iterations))
    final_sequence_epochs = int(frozen_median(sequence_epochs))
    market_manifest_path = output_root / "market_regime_evaluation/manifest.json"
    market_manifest = json.loads(market_manifest_path.read_text(encoding="utf-8"))
    ridge_alpha = float(
        frozen_median(
            [float(record["ridge_alpha"]) for record in market_manifest["fit_records"]]
        )
    )
    logistic_c = float(
        frozen_median(
            [float(record["logistic_c"]) for record in market_manifest["fit_records"]]
        )
    )
    fingerprint = _stable_hash(
        {
            "schema": FINAL_BUNDLE_SCHEMA,
            "implementation_sha256": base._sha256(Path(__file__)),
            "study_sha256": base._sha256(study_path),
            "candidate_sha256": base._sha256(candidate_path),
            "availability_sha256": base._sha256(availability_path),
            "input_fingerprint": sources.context.model_manifest["input_fingerprint"],
            "target_fingerprint": sources.target_manifest["fingerprint"],
            "tree_tasks": {str(path): base._sha256(path) for _, path in tree_tasks},
            "sequence_tasks": {
                str(path): base._sha256(path) for _, path in sequence_tasks
            },
            "tree_iterations": tree_iterations,
            "final_tree_iterations": final_tree_iterations,
            "sequence_epochs": sequence_epochs,
            "final_sequence_epochs": final_sequence_epochs,
            "ridge_alpha": ridge_alpha,
            "logistic_c": logistic_c,
            "lookback": int(lookback),
            "date_batch_size": int(date_batch_size),
            "masked_live_features": [
                str(record["feature_name"]) for record in masked_records
            ],
        }
    )
    bundle_root = output_root / "forward_policy_bundle_v1"
    manifest_path = bundle_root / "manifest.json"
    if manifest_path.is_file():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            current.get("status") == "completed"
            and current.get("fingerprint") == fingerprint
        ):
            return current
    bundle_root.mkdir(parents=True, exist_ok=True)
    resource = base.resource_plan(
        reserve_gib=float(study["resources"]["reserve_system_gib"]),
        maximum_threads=int(study["resources"]["maximum_cpu_threads"]),
        histogram_pool_cap_mb=int(study["resources"]["histogram_pool_cap_mb"]),
        sequence_batch_cap=int(study["resources"]["sequence_batch_cap"]),
    )
    all_rows = np.arange(len(sources.context.row_index), dtype=np.int64)
    all_dates = sources.context.row_index["date_idx"].to_numpy(dtype=np.int32)
    cache_root = bundle_root / "dataset_cache"
    cache_path = cache_root / "all_history.bin"
    cache_fingerprint = base._stable_hash(
        {
            "schema": "seq100_full_market_final_dataset_cache/1",
            "input_fingerprint": sources.context.model_manifest["input_fingerprint"],
            "target_fingerprint": sources.target_manifest["fingerprint"],
            "row_count": len(all_rows),
            "feature_names": sources.feature_names,
            "max_bin": int(study["lightgbm"]["max_bin"]),
        }
    )
    cache_record = base.build_dataset_cache(
        matrix=sources.matrix,
        rows=all_rows,
        feature_names=sources.feature_names,
        binary_path=cache_path,
        meta_path=cache_root / "all_history.json",
        fingerprint=cache_fingerprint,
        max_bin=int(study["lightgbm"]["max_bin"]),
        seed=int(study["validation"]["seed"]),
        resource=resource,
    )
    raw_target = np.asarray(
        sources.exact_values[:, sources.base_column], dtype=np.float32
    )
    target_valid = np.asarray(
        sources.exact_valid[:, sources.base_column], dtype=bool
    ) & np.isfinite(raw_target)
    labels = base._date_relevance_labels(all_dates, raw_target, target_valid)
    labels[~target_valid] = 0.0
    weights = base._equal_date_weights(all_dates, target_valid)
    binary_params = {
        "max_bin": int(study["lightgbm"]["max_bin"]),
        "data_random_seed": int(study["validation"]["seed"]),
        "feature_pre_filter": False,
        "num_threads": int(resource.cpu_threads),
        "verbosity": -1,
    }
    train_set = lgb.Dataset(
        cache_record["file"]["path"], params=binary_params, free_raw_data=True
    ).construct()
    train_set.set_label(labels)
    train_set.set_weight(weights)
    train_set.set_group(base._date_group_sizes(all_dates))
    parameters = base._model_parameters(
        study,
        kind="ranking",
        resource=resource,
        profile=str(study["lightgbm"]["primary_profile"]),
    )
    base._emit(
        "final_tree_refit_started",
        row_count=len(all_rows),
        valid_count=int(target_valid.sum()),
        iterations=final_tree_iterations,
        resource_plan=asdict(resource),
    )
    with base.ResourceMonitor() as tree_monitor:
        booster = lgb.train(
            parameters,
            train_set,
            num_boost_round=final_tree_iterations,
            callbacks=[lgb.log_evaluation(period=25)],
        )
        tree_metrics = tree_monitor.metrics()
    tree_path = bundle_root / "stock_rank_tree.txt"
    tree_partial = tree_path.with_suffix(".txt.partial")
    booster.save_model(str(tree_partial), num_iteration=final_tree_iterations)
    os.replace(tree_partial, tree_path)
    importance = pd.DataFrame(
        {
            "feature_name": sources.feature_names,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain", "split"], ascending=False)
    importance_path = bundle_root / "stock_rank_tree_importance.parquet"
    base._write_parquet(importance, importance_path)
    del booster, train_set, labels, weights, raw_target
    gc.collect()
    base._trim_working_set()

    training_dates = sorted(
        int(date_idx)
        for date_idx in sources.blocks
        if int(date_idx) <= int(sources.context.cutoff_idx)
    )
    device = torch.device("cuda")
    sequence_seed = int(study["validation"]["seed"]) + 90_000 + int(lookback)
    base._emit(
        "final_sequence_refit_started",
        date_count=len(training_dates),
        epochs=final_sequence_epochs,
        lookback=lookback,
    )
    with base.ResourceMonitor() as sequence_monitor:
        model, normalization, sequence_history = _refit_model(
            sources,
            train_dates=training_dates,
            lookback=lookback,
            device=device,
            epochs=final_sequence_epochs,
            date_batch_size=date_batch_size,
            seed=sequence_seed,
        )
        sequence_metrics = sequence_monitor.metrics()
    sequence_path = bundle_root / "stock_market_sequence.pt"
    sequence_partial = sequence_path.with_suffix(".pt.partial")
    torch.save(
        {
            "schema": FINAL_BUNDLE_SCHEMA,
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "normalization": normalization.payload(),
            "lookback": int(lookback),
            "epochs": final_sequence_epochs,
            "feature_names": sources.feature_names,
            "sequence_feature_names": sources.pack_feature_names,
            "market_feature_names": [
                sources.feature_names[int(index)] for index in sources.market_columns
            ],
        },
        sequence_partial,
    )
    os.replace(sequence_partial, sequence_path)
    history_path = bundle_root / "sequence_training_history.json"
    _write_json(history_path, {"outer_refit": sequence_history})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    base._trim_working_set()

    row_boundaries = np.flatnonzero(np.r_[True, all_dates[1:] != all_dates[:-1], True])
    first_rows = row_boundaries[:-1]
    last_rows = row_boundaries[1:] - 1
    unique_dates = all_dates[first_rows]
    market_matrix = np.asarray(
        sources.matrix[np.ix_(first_rows, sources.market_columns)], dtype=np.float64
    )
    market_matrix_last = np.asarray(
        sources.matrix[np.ix_(last_rows, sources.market_columns)], dtype=np.float64
    )
    if not np.allclose(
        market_matrix, market_matrix_last, rtol=0.0, atol=0.0, equal_nan=True
    ):
        raise SequenceChallengerError("market_features_are_not_date_invariant")
    market_target = np.full(len(unique_dates), np.nan, dtype=np.float64)
    for position, (left, right) in enumerate(pairwise(row_boundaries)):
        valid = np.asarray(
            sources.exact_valid[left:right, sources.base_column], dtype=bool
        ) & np.asarray(
            sources.exact_valid[left:right, sources.stress_column], dtype=bool
        )
        if valid.any():
            market_target[position] = float(
                np.asarray(
                    sources.exact_values[left:right, sources.base_column],
                    dtype=np.float64,
                )[valid].mean()
            )
    market_valid = np.isfinite(market_target)
    ridge_model = base.make_pipeline(
        base.SimpleImputer(strategy="median", add_indicator=True),
        base.RobustScaler(),
        base.Ridge(alpha=ridge_alpha),
    )
    logistic_model = base.make_pipeline(
        base.SimpleImputer(strategy="median", add_indicator=True),
        base.RobustScaler(),
        base.LogisticRegression(C=logistic_c, max_iter=2000),
    )
    ridge_model.fit(market_matrix[market_valid], market_target[market_valid])
    logistic_model.fit(
        market_matrix[market_valid],
        (market_target[market_valid] > 0.0).astype(np.int8),
    )
    market_path = bundle_root / "market_gate_models.pkl"
    market_partial = market_path.with_suffix(".pkl.partial")
    with market_partial.open("wb") as stream:
        pickle.dump(
            {
                "schema": FINAL_BUNDLE_SCHEMA,
                "ridge": ridge_model,
                "logistic": logistic_model,
                "ridge_alpha": ridge_alpha,
                "logistic_c": logistic_c,
                "market_feature_names": [
                    sources.feature_names[int(index)]
                    for index in sources.market_columns
                ],
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(market_partial, market_path)

    maximum_training_signal_date = str(
        sources.context.row_index.loc[target_valid, "trade_date"].astype(str).max()
    )
    result = {
        "schema": FINAL_BUNDLE_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "fingerprint": fingerprint,
        "study_id": base.STUDY_ID,
        "candidate_study_id": candidate["study_id"],
        "training_boundary": {
            "minimum_signal_date": str(
                sources.context.row_index["trade_date"].astype(str).min()
            ),
            "maximum_input_date": str(
                sources.context.row_index["trade_date"].astype(str).max()
            ),
            "maximum_labeled_training_signal_date": maximum_training_signal_date,
            "maximum_consumed_outcome_date": base.MAXIMUM_OUTCOME_DATE,
            "forbidden_2026_outcome_read_count": 0,
        },
        "stock_tree": {
            "target": "exact_net_return_d5_rank",
            "oof_selected_iterations": tree_iterations,
            "aggregation": "deterministic_middle_order_statistic",
            "final_iterations": final_tree_iterations,
            "training_row_count": len(all_rows),
            "valid_label_count": int(target_valid.sum()),
            "feature_count": len(sources.feature_names),
            "resource_metrics": tree_metrics,
        },
        "stock_and_market_sequence": {
            "lookback": int(lookback),
            "oof_selected_epochs": sequence_epochs,
            "aggregation": "deterministic_middle_order_statistic",
            "final_epochs": final_sequence_epochs,
            "training_date_count": len(training_dates),
            "resource_metrics": sequence_metrics,
        },
        "linear_market_gate": {
            "ridge_alpha": ridge_alpha,
            "logistic_c": logistic_c,
            "training_date_count": int(market_valid.sum()),
        },
        "live_inference_contract": {
            "stock_score": "equal percentile rank of tree and sequence stock heads",
            "market_gate": "ridge_above_zero_and_logistic_above_0p5_and_neural_return_above_zero_and_neural_probability_above_0p5",
            "top_k": 10,
            "entry": "next legal open",
            "exit": "D3 close request then first legally sellable open",
            "unfilled_or_unaffordable_order": "cash_without_rank_substitution",
            "cohort_equity_fraction": 1.0 / 3.0,
            "source_outage_policy": "mask_named_features_as_nan",
            "masked_families": list(family_names),
            "masked_feature_count": len(masked_columns),
            "masked_features": [
                str(record["feature_name"]) for record in masked_records
            ],
        },
        "decision_boundary": {
            "all_2012_2025_outcomes_are_consumed": True,
            "final_hyperparameters_are_aggregated_from_existing_oof_choices": True,
            "no_new_threshold_or_horizon_search": True,
            "no_2026_outcome_read": True,
            "stable_profit_claim_allowed": False,
            "real_order_authority": False,
        },
        "resource_plan": asdict(resource),
        "files": {
            "tree_model": _file_record(tree_path),
            "tree_importance": _file_record(importance_path, row_count=len(importance)),
            "sequence_model": _file_record(sequence_path),
            "sequence_history": _file_record(history_path),
            "market_models": _file_record(market_path),
            "dataset_cache": _file_record(cache_root / "all_history.json"),
        },
        "sources": {
            "candidate": _file_record(candidate_path),
            "availability_stress": _file_record(availability_path),
            "model_inputs": _file_record(sources.context.model_manifest_path),
            "exact_net_targets": _file_record(
                output_root / "exact_net_targets/manifest.json"
            ),
            "market_oof": _file_record(market_manifest_path),
            "tree_oof_tasks": [_file_record(path) for _, path in tree_tasks],
            "sequence_oof_tasks": [_file_record(path) for _, path in sequence_tasks],
        },
    }
    _write_json(manifest_path, result)
    base._emit(
        "final_policy_bundle_completed",
        tree_iterations=final_tree_iterations,
        sequence_epochs=final_sequence_epochs,
        maximum_training_signal_date=maximum_training_signal_date,
        masked_feature_count=len(masked_columns),
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--maximum-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--date-batch-size", type=int, default=DEFAULT_DATE_BATCH_SIZE)
    parser.add_argument("--evaluate-ensemble", action="store_true")
    parser.add_argument("--replay-accounts", action="store_true")
    parser.add_argument("--robustness", action="store_true")
    parser.add_argument("--dual-horizons", action="store_true")
    parser.add_argument("--availability-stress", action="store_true")
    parser.add_argument("--core-availability-stress", action="store_true")
    parser.add_argument("--exact-core-availability-stress", action="store_true")
    parser.add_argument("--repaired-rank-core-availability-stress", action="store_true")
    parser.add_argument(
        "--corrected-rank-full-core-availability-stress", action="store_true"
    )
    parser.add_argument(
        "--rebuildable-current-core-availability-stress", action="store_true"
    )
    parser.add_argument("--rebuildable-core-robustness", action="store_true")
    parser.add_argument("--freeze-final-bundle", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.rebuildable_core_robustness:
        result = replay_rebuildable_core_robustness(
            study_path=args.study,
            output_root=args.output_root,
        )
    elif args.freeze_final_bundle:
        result = freeze_final_policy_bundle(
            study_path=args.study,
            output_root=args.output_root,
            lookback=args.lookback,
            date_batch_size=args.date_batch_size,
        )
    elif args.rebuildable_current_core_availability_stress:
        result = evaluate_source_availability_stress(
            study_path=args.study,
            output_root=args.output_root,
            lookback=args.lookback,
            date_batch_size=args.date_batch_size,
            availability_profile="rebuildable_current_core",
        )
    elif args.corrected_rank_full_core_availability_stress:
        result = evaluate_source_availability_stress(
            study_path=args.study,
            output_root=args.output_root,
            lookback=args.lookback,
            date_batch_size=args.date_batch_size,
            availability_profile="corrected_rank_rebuildable_market_path_core",
        )
    elif args.repaired_rank_core_availability_stress:
        result = evaluate_source_availability_stress(
            study_path=args.study,
            output_root=args.output_root,
            lookback=args.lookback,
            date_batch_size=args.date_batch_size,
            availability_profile="repaired_rank_market_path_core",
        )
    elif args.exact_core_availability_stress:
        result = evaluate_source_availability_stress(
            study_path=args.study,
            output_root=args.output_root,
            lookback=args.lookback,
            date_batch_size=args.date_batch_size,
            availability_profile="source_exact_market_path_core",
        )
    elif args.core_availability_stress:
        result = evaluate_source_availability_stress(
            study_path=args.study,
            output_root=args.output_root,
            lookback=args.lookback,
            date_batch_size=args.date_batch_size,
            availability_profile="rebuildable_market_path_core",
        )
    elif args.availability_stress:
        result = evaluate_source_availability_stress(
            study_path=args.study,
            output_root=args.output_root,
            lookback=args.lookback,
            date_batch_size=args.date_batch_size,
        )
    elif args.dual_horizons:
        result = replay_dual_gate_horizons(
            study_path=args.study,
            output_root=args.output_root,
            lookback=args.lookback,
        )
    elif args.robustness:
        result = replay_account_robustness(
            study_path=args.study,
            output_root=args.output_root,
            lookback=args.lookback,
        )
    elif args.replay_accounts:
        result = replay_ensemble_accounts(
            study_path=args.study,
            output_root=args.output_root,
            lookback=args.lookback,
        )
    elif args.evaluate_ensemble:
        result = evaluate_ensemble(
            study_path=args.study,
            output_root=args.output_root,
            lookback=args.lookback,
        )
    else:
        result = train_fold(
            study_path=args.study,
            output_root=args.output_root,
            fold_number=args.fold,
            lookback=args.lookback,
            maximum_epochs=args.maximum_epochs,
            patience=args.patience,
            date_batch_size=args.date_batch_size,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
