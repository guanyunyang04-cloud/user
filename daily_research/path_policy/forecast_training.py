from __future__ import annotations

import contextlib
from datetime import datetime, timedelta
import inspect
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from daily_research.continuous_policy.runtime import write_json
from daily_research.path_policy.forecast_dataset import (
    ForecastMemmapDataset,
    ForecastSequenceDataset,
    ForecastShardedMemmapDataset,
    ForecastTrainingPackDataset,
)
from daily_research.path_policy.labels import PATH20_CUMULATIVE_HORIZONS, PATH20_HORIZON
from daily_research.path_policy.models import (
    DEFAULT_GROUP_MIXER_CHUNK_SIZE,
    DLinearPath20Forecaster,
    DateSlateAlphaFusionV1Forecaster,
    ExpertFusionPath20Forecaster,
    GRUPath20Forecaster,
    HybridStructuredAlphaV2Forecaster,
    HybridMultiScaleRecencyAwarePath20Forecaster,
    LinearPath20Forecaster,
    PatchTransformerPath20Forecaster,
    Path20ForecasterMLP,
    RegimeRoutedMultiExpertHorizonForecaster,
    SectorSlotMixerPath20Forecaster,
    StaticContextPath20Forecaster,
    StockMixerPath20Forecaster,
    normalize_path20_cumulative_horizons,
    pairwise_rank_loss,
    path20_decision_aux_dim,
    path20_forecast_aux_dim,
    pinball_loss,
)


FORECAST_MODEL_FAMILIES = (
    "linear_last_day",
    "dlinear_sequence",
    "mlp_last_day",
    "gru_sequence",
    "patch_transformer",
    "gru_sequence_static_context",
    "patch_transformer_static_context",
    "stock_mixer_sequence",
    "sector_slot_mixer_sequence",
    "hybrid_expert_fusion_static_context",
    "hybrid_multiscale_recency_aware_v1",
    "hybrid_structured_alpha_v2",
    "date_slate_alpha_fusion_v1",
    "regime_routed_multi_expert_horizon_v1",
)
FORECAST_CROSS_SECTIONAL_MODEL_FAMILIES = (
    "stock_mixer_sequence",
    "sector_slot_mixer_sequence",
    "date_slate_alpha_fusion_v1",
    "regime_routed_multi_expert_horizon_v1",
)
FORECAST_MEMMAP_DATASET_TYPES = (ForecastMemmapDataset, ForecastShardedMemmapDataset, ForecastTrainingPackDataset)
FORECAST_OUTPUT_PROFILES = ("forecast_path_v1", "decision_utility_v1", "forecast_incremental_path_v2")
FORECAST_SELECTION_PROFILES = ("multiscale", "trend20", "short_burst", "decision_utility", "validation_loss")
FORECAST_LOSS_PROFILES = (
    "default",
    "rank_aux",
    "multitask_v1",
    "forecast_path_v1_baseline",
    "decision_utility_v1",
    "decision_utility_v1_baseline",
    "decision_utility_path_aux_v1",
    "decision_utility_hit_risk_aux_v1",
    "decision_utility_rank_aux_v1",
    "score_monthly_robust_v1",
    "horizon_entropy_regularized_v1",
    "risk_drawdown_reweighted_v1",
    "horizon_target_normalized_v1",
    "horizon_head_soft_constraint_v1",
    "target_norm_head_constraint_v1",
    "horizon_30d_soft_penalty_v1",
    "topn_excess_rank_v1",
    "decision_score_topk_alignment_v1",
    "score_to_weight_proxy_v1",
    "bad_month_aware_v1",
    "personal_alpha_scorer_hybrid_v1",
    "personal_time_efficient_topk_v1",
    "hybrid_alpha_score_v1",
    "hybrid_alpha_score_v2",
    "date_grouped_alpha_score_v1",
)
FORECAST_RANKING_BASELINES = ("none", "lightgbm", "xgboost")
FORECAST_RISK_AUX_NAMES = ("downside_floor", "worst_1d", "upside")
FORECAST_RANK_LOSS_WEIGHTS = {1: 0.0025, 3: 0.0050, 5: 0.0075, 10: 0.0075, 20: 0.0100}
FORECAST_PROFILE_HORIZON_WEIGHTS: dict[str, dict[int, float]] = {
    "multiscale": {1: 0.05, 3: 0.15, 5: 0.20, 10: 0.25, 20: 0.25},
    "trend20": {10: 0.35, 20: 0.65},
    "short_burst": {1: 0.05, 3: 0.35, 5: 0.30, 10: 0.15, 20: 0.05},
}
_FORECAST_DECISION_LOSS_PROFILES = {
    "decision_utility_v1",
    "decision_utility_v1_baseline",
    "decision_utility_path_aux_v1",
    "decision_utility_hit_risk_aux_v1",
    "decision_utility_rank_aux_v1",
    "score_monthly_robust_v1",
    "horizon_entropy_regularized_v1",
    "risk_drawdown_reweighted_v1",
    "horizon_target_normalized_v1",
    "horizon_head_soft_constraint_v1",
    "target_norm_head_constraint_v1",
    "horizon_30d_soft_penalty_v1",
    "topn_excess_rank_v1",
    "decision_score_topk_alignment_v1",
    "score_to_weight_proxy_v1",
    "bad_month_aware_v1",
    "personal_alpha_scorer_hybrid_v1",
    "personal_time_efficient_topk_v1",
}
_FORECAST_PREDICTION_FIRST_LOSS_PROFILES = {
    "default",
    "rank_aux",
    "multitask_v1",
    "forecast_path_v1_baseline",
    "hybrid_alpha_score_v1",
    "hybrid_alpha_score_v2",
    "date_grouped_alpha_score_v1",
}
_FORECAST_LOSS_WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "default": {
        "path_daily": 1.00,
        "quantile": 0.60,
        "path_aux": 0.25,
        "risk_aux": 0.05,
        "rank_aux": 1.00,
        "risk_rank_aux": 0.005,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.0,
        "decision_utility": 0.0,
        "hit_aux": 0.0,
        "horizon_classification": 0.0,
        "decision_rank_aux": 0.0,
    },
    "rank_aux": {
        "path_daily": 1.00,
        "quantile": 0.60,
        "path_aux": 0.25,
        "risk_aux": 0.05,
        "rank_aux": 1.75,
        "risk_rank_aux": 0.005,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.0,
        "decision_utility": 0.0,
        "hit_aux": 0.0,
        "horizon_classification": 0.0,
        "decision_rank_aux": 0.0,
    },
    "multitask_v1": {
        "path_daily": 1.00,
        "quantile": 0.60,
        "path_aux": 0.25,
        "risk_aux": 0.05,
        "rank_aux": 2.25,
        "risk_rank_aux": 0.005,
        "direction_aux": 0.025,
        "downside_rank_aux": 0.010,
        "decision_utility": 0.0,
        "hit_aux": 0.0,
        "horizon_classification": 0.0,
        "decision_rank_aux": 0.0,
    },
    "forecast_path_v1_baseline": {
        "path_daily": 1.00,
        "quantile": 0.60,
        "path_aux": 0.25,
        "risk_aux": 0.05,
        "rank_aux": 1.00,
        "risk_rank_aux": 0.005,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.0,
        "decision_utility": 0.0,
        "hit_aux": 0.0,
        "horizon_classification": 0.0,
        "decision_rank_aux": 0.0,
    },
    "decision_utility_v1": {
        "path_daily": 0.55,
        "quantile": 0.20,
        "path_aux": 0.15,
        "risk_aux": 0.05,
        "rank_aux": 0.75,
        "risk_rank_aux": 0.005,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.0,
        "decision_utility": 1.00,
        "hit_aux": 0.25,
        "horizon_classification": 0.25,
        "decision_rank_aux": 0.25,
    },
    "decision_utility_v1_baseline": {
        "path_daily": 0.55,
        "quantile": 0.20,
        "path_aux": 0.15,
        "risk_aux": 0.05,
        "rank_aux": 0.75,
        "risk_rank_aux": 0.005,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.0,
        "decision_utility": 1.00,
        "hit_aux": 0.25,
        "horizon_classification": 0.25,
        "decision_rank_aux": 0.25,
    },
    "decision_utility_path_aux_v1": {
        "path_daily": 0.65,
        "quantile": 0.25,
        "path_aux": 0.30,
        "risk_aux": 0.05,
        "rank_aux": 0.75,
        "risk_rank_aux": 0.005,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.0,
        "decision_utility": 1.00,
        "hit_aux": 0.20,
        "horizon_classification": 0.20,
        "decision_rank_aux": 0.20,
    },
    "decision_utility_hit_risk_aux_v1": {
        "path_daily": 0.50,
        "quantile": 0.20,
        "path_aux": 0.15,
        "risk_aux": 0.30,
        "rank_aux": 0.75,
        "risk_rank_aux": 0.010,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.010,
        "decision_utility": 1.00,
        "hit_aux": 0.35,
        "horizon_classification": 0.20,
        "decision_rank_aux": 0.20,
    },
    "decision_utility_rank_aux_v1": {
        "path_daily": 0.50,
        "quantile": 0.20,
        "path_aux": 0.15,
        "risk_aux": 0.05,
        "rank_aux": 0.95,
        "risk_rank_aux": 0.005,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.0,
        "decision_utility": 1.00,
        "hit_aux": 0.20,
        "horizon_classification": 0.20,
        "decision_rank_aux": 0.45,
    },
    "score_monthly_robust_v1": {
        "path_daily": 0.50,
        "quantile": 0.20,
        "path_aux": 0.20,
        "risk_aux": 0.05,
        "rank_aux": 1.05,
        "risk_rank_aux": 0.005,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.0,
        "decision_utility": 1.00,
        "hit_aux": 0.20,
        "horizon_classification": 0.15,
        "decision_rank_aux": 0.60,
        "horizon_entropy": 0.0,
    },
    "horizon_entropy_regularized_v1": {
        "path_daily": 0.60,
        "quantile": 0.25,
        "path_aux": 0.30,
        "risk_aux": 0.05,
        "rank_aux": 0.75,
        "risk_rank_aux": 0.005,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.0,
        "decision_utility": 1.00,
        "hit_aux": 0.20,
        "horizon_classification": 0.10,
        "decision_rank_aux": 0.25,
        "horizon_entropy": 0.05,
    },
    "risk_drawdown_reweighted_v1": {
        "path_daily": 0.50,
        "quantile": 0.20,
        "path_aux": 0.20,
        "risk_aux": 0.25,
        "rank_aux": 0.75,
        "risk_rank_aux": 0.015,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.025,
        "decision_utility": 1.00,
        "hit_aux": 0.25,
        "horizon_classification": 0.15,
        "decision_rank_aux": 0.25,
        "horizon_entropy": 0.0,
    },
    "horizon_target_normalized_v1": {
        "path_daily": 0.50,
        "quantile": 0.20,
        "path_aux": 0.20,
        "risk_aux": 0.05,
        "rank_aux": 0.95,
        "risk_rank_aux": 0.005,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.0,
        "decision_utility": 1.00,
        "hit_aux": 0.20,
        "horizon_classification": 0.12,
        "decision_rank_aux": 0.50,
        "horizon_entropy": 0.0,
        "horizon_target_normalization": 1.0,
        "horizon_head_soft_constraint": 0.0,
    },
    "horizon_head_soft_constraint_v1": {
        "path_daily": 0.50,
        "quantile": 0.20,
        "path_aux": 0.20,
        "risk_aux": 0.05,
        "rank_aux": 1.00,
        "risk_rank_aux": 0.005,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.0,
        "decision_utility": 1.00,
        "hit_aux": 0.20,
        "horizon_classification": 0.10,
        "decision_rank_aux": 0.55,
        "horizon_entropy": 0.0,
        "horizon_target_normalization": 0.0,
        "horizon_head_soft_constraint": 0.05,
    },
    "target_norm_head_constraint_v1": {
        "path_daily": 0.50,
        "quantile": 0.20,
        "path_aux": 0.20,
        "risk_aux": 0.05,
        "rank_aux": 0.95,
        "risk_rank_aux": 0.005,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.0,
        "decision_utility": 1.00,
        "hit_aux": 0.20,
        "horizon_classification": 0.10,
        "decision_rank_aux": 0.55,
        "horizon_entropy": 0.0,
        "horizon_target_normalization": 1.0,
        "horizon_head_soft_constraint": 0.05,
    },
    "horizon_30d_soft_penalty_v1": {
        "path_daily": 0.60,
        "quantile": 0.25,
        "path_aux": 0.30,
        "risk_aux": 0.05,
        "rank_aux": 0.75,
        "risk_rank_aux": 0.005,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.0,
        "decision_utility": 1.00,
        "hit_aux": 0.20,
        "horizon_classification": 0.10,
        "decision_rank_aux": 0.20,
        "horizon_entropy": 0.05,
        "horizon_target_normalization": 0.0,
        "horizon_head_soft_constraint": 0.03,
        "calibrated_decision_rank_aux": 0.30,
        "router_entropy_floor": 0.005,
        "expert_diversity": 0.002,
        "bad_state_calibration": 0.010,
    },
    "topn_excess_rank_v1": {
        "path_daily": 0.45,
        "quantile": 0.15,
        "path_aux": 0.15,
        "risk_aux": 0.04,
        "rank_aux": 1.20,
        "risk_rank_aux": 0.005,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.0,
        "decision_utility": 0.85,
        "hit_aux": 0.20,
        "horizon_classification": 0.08,
        "decision_rank_aux": 0.35,
        "horizon_entropy": 0.02,
        "topn_excess_rank": 0.45,
        "score_to_weight_proxy": 0.0,
        "bad_month_aware": 0.0,
    },
    "decision_score_topk_alignment_v1": {
        "path_daily": 0.42,
        "quantile": 0.14,
        "path_aux": 0.14,
        "risk_aux": 0.05,
        "rank_aux": 1.00,
        "risk_rank_aux": 0.005,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.010,
        "decision_utility": 0.85,
        "hit_aux": 0.18,
        "horizon_classification": 0.08,
        "decision_rank_aux": 0.50,
        "horizon_entropy": 0.02,
        "topn_excess_rank": 0.25,
        "decision_score_topk_alignment": 0.35,
        "score_to_weight_proxy": 0.0,
        "bad_month_aware": 0.0,
    },
    "score_to_weight_proxy_v1": {
        "path_daily": 0.40,
        "quantile": 0.12,
        "path_aux": 0.12,
        "risk_aux": 0.04,
        "rank_aux": 0.90,
        "risk_rank_aux": 0.005,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.0,
        "decision_utility": 0.80,
        "hit_aux": 0.15,
        "horizon_classification": 0.08,
        "decision_rank_aux": 0.25,
        "horizon_entropy": 0.02,
        "topn_excess_rank": 0.25,
        "score_to_weight_proxy": 0.55,
        "bad_month_aware": 0.0,
    },
    "bad_month_aware_v1": {
        "path_daily": 0.45,
        "quantile": 0.15,
        "path_aux": 0.12,
        "risk_aux": 0.18,
        "rank_aux": 0.95,
        "risk_rank_aux": 0.015,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.035,
        "decision_utility": 0.80,
        "hit_aux": 0.18,
        "horizon_classification": 0.08,
        "decision_rank_aux": 0.25,
        "horizon_entropy": 0.02,
        "topn_excess_rank": 0.20,
        "score_to_weight_proxy": 0.25,
        "bad_month_aware": 0.45,
    },
    "personal_alpha_scorer_hybrid_v1": {
        "path_daily": 0.42,
        "quantile": 0.14,
        "path_aux": 0.14,
        "risk_aux": 0.08,
        "rank_aux": 1.05,
        "risk_rank_aux": 0.010,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.025,
        "decision_utility": 0.85,
        "hit_aux": 0.18,
        "horizon_classification": 0.08,
        "decision_rank_aux": 0.42,
        "horizon_entropy": 0.02,
        "topn_excess_rank": 0.30,
        "score_to_weight_proxy": 0.20,
        "bad_month_aware": 0.18,
    },
    "personal_time_efficient_topk_v1": {
        "path_daily": 0.18,
        "quantile": 0.06,
        "path_aux": 0.08,
        "risk_aux": 0.05,
        "rank_aux": 0.30,
        "risk_rank_aux": 0.003,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.010,
        "decision_utility": 2.00,
        "hit_aux": 0.25,
        "horizon_classification": 0.20,
        "decision_rank_aux": 1.25,
        "horizon_entropy": 0.0,
        "time_eff_topk_alignment": 0.35,
        "score_to_weight_proxy": 0.0,
        "bad_month_aware": 0.0,
    },
    "hybrid_alpha_score_v1": {
        "path_daily": 0.70,
        "quantile": 0.25,
        "path_aux": 0.35,
        "risk_aux": 0.08,
        "rank_aux": 1.35,
        "risk_rank_aux": 0.010,
        "direction_aux": 0.020,
        "downside_rank_aux": 0.020,
        "alpha_efficiency_rank_aux": 0.35,
        "decision_utility": 0.0,
        "hit_aux": 0.0,
        "horizon_classification": 0.0,
        "decision_rank_aux": 0.0,
        "horizon_entropy": 0.0,
        "time_eff_topk_alignment": 0.0,
        "score_to_weight_proxy": 0.0,
        "bad_month_aware": 0.0,
    },
    "hybrid_alpha_score_v2": {
        "path_daily": 0.58,
        "quantile": 0.18,
        "path_aux": 0.30,
        "risk_aux": 0.06,
        "rank_aux": 1.55,
        "risk_rank_aux": 0.006,
        "direction_aux": 0.010,
        "downside_rank_aux": 0.012,
        "alpha_efficiency_rank_aux": 0.85,
        "decision_utility": 0.0,
        "hit_aux": 0.0,
        "horizon_classification": 0.0,
        "decision_rank_aux": 0.0,
        "horizon_entropy": 0.0,
        "time_eff_topk_alignment": 0.0,
        "score_to_weight_proxy": 0.0,
        "bad_month_aware": 0.0,
    },
    "date_grouped_alpha_score_v1": {
        "path_daily": 0.60,
        "quantile": 0.18,
        "path_aux": 0.25,
        "risk_aux": 0.06,
        "rank_aux": 0.0,
        "risk_rank_aux": 0.0,
        "direction_aux": 0.0,
        "downside_rank_aux": 0.010,
        "upside_rank_aux": 0.006,
        "date_grouped_rank": 1.20,
        "unit_time_alpha_rank": 0.70,
        "decision_utility": 0.0,
        "hit_aux": 0.0,
        "horizon_classification": 0.0,
        "decision_rank_aux": 0.0,
        "horizon_entropy": 0.0,
        "time_eff_topk_alignment": 0.0,
        "score_to_weight_proxy": 0.0,
        "bad_month_aware": 0.0,
    },
}

_FORECAST_TARGET_NORMALIZED_LOSS_PROFILES = {
    "horizon_target_normalized_v1",
    "target_norm_head_constraint_v1",
}
_FORECAST_HEAD_CONSTRAINT_LOSS_PROFILES = {
    "horizon_head_soft_constraint_v1",
    "target_norm_head_constraint_v1",
    "horizon_30d_soft_penalty_v1",
}
_HORIZON_HEAD_CONSTRAINT_MAX_30D_PROBABILITY = 0.75
_HORIZON_HEAD_CONSTRAINT_LONG_HORIZONS = (15, 20, 30)
_HORIZON_HEAD_CONSTRAINT_MIN_LONG_PROBABILITY = 0.40
_FORECAST_UTILITY_30D_SOFT_PENALTY_PROFILES = {"horizon_30d_soft_penalty_v1"}
_HORIZON_30D_SOFT_PENALTY = 0.005
_HIGH_RETURN_PROXY_LOSS_PROFILES = {
    "topn_excess_rank_v1",
    "decision_score_topk_alignment_v1",
    "score_to_weight_proxy_v1",
    "bad_month_aware_v1",
    "personal_alpha_scorer_hybrid_v1",
}
_TIME_EFFICIENT_TOPK_LOSS_PROFILES = {"personal_time_efficient_topk_v1"}
_HYBRID_ALPHA_SCORE_LOSS_PROFILES = {"hybrid_alpha_score_v1", "hybrid_alpha_score_v2", "date_grouped_alpha_score_v1"}
_FORECAST_PROFILE_RANK_LOSS_WEIGHTS: dict[str, dict[int, float]] = {
    "hybrid_alpha_score_v2": {1: 0.0100, 3: 0.0100, 5: 0.0100, 10: 0.0075, 20: 0.0050},
    "date_grouped_alpha_score_v1": {1: 0.0100, 3: 0.0100, 5: 0.0100, 10: 0.0075, 20: 0.0050},
}
_TIME_EFFICIENT_WORST_DAY_PENALTY = 0.10


class _EagerTorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: ForecastSequenceDataset, indices: np.ndarray, *, target_scale: float) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.target_scale = float(target_scale)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        row_idx = int(self.indices[int(item)])
        y_risk = np.stack(
            [
                np.asarray(self.dataset.y_drawdown_by_horizon[row_idx], dtype=np.float32),
                np.asarray(self.dataset.y_worst_by_horizon[row_idx], dtype=np.float32),
                np.asarray(self.dataset.y_upside_by_horizon[row_idx], dtype=np.float32),
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
        items: tuple[torch.Tensor, ...] = (
            torch.as_tensor(self.dataset.x[row_idx], dtype=torch.float32),
            torch.as_tensor(self.dataset.y_daily_excess[row_idx] * self.target_scale, dtype=torch.float32),
            torch.as_tensor(self.dataset.y_cum_excess[row_idx] * self.target_scale, dtype=torch.float32),
            torch.as_tensor(y_risk * self.target_scale, dtype=torch.float32),
            torch.as_tensor(row_idx, dtype=torch.long),
        )
        if self.dataset.static_context_ids is not None:
            items = (*items, torch.as_tensor(self.dataset.static_context_ids[row_idx], dtype=torch.long))
        return items


class _ForecastDatasetView:
    def __init__(
        self,
        dataset: ForecastSequenceDataset | ForecastMemmapDataset,
        *,
        static_context_fields_override: tuple[str, ...] | list[str] | str | None = None,
    ) -> None:
        self.dataset = dataset
        self.manifest = dict(dataset.manifest)
        self.normalization_manifest = dict(dataset.normalization_manifest)
        self.feature_columns = list(dataset.feature_columns)
        self.dataset_mode = str(self.manifest.get("dataset_mode", "eager"))
        self.source_static_context_schema = dict(self.manifest.get("static_context_schema", {}) or {"enabled": False})
        self.static_context_schema = self._effective_static_context_schema(
            self.source_static_context_schema,
            static_context_fields_override=static_context_fields_override,
        )
        self.static_context_vocab_sizes = dict(self.static_context_schema.get("vocab_sizes", {}) or {})
        self.static_context_source_fields = tuple(
            str(item) for item in self.source_static_context_schema.get("fields", []) or []
        )
        self.static_context_effective_fields = tuple(
            str(item) for item in self.static_context_schema.get("fields", []) or []
        )
        self.static_context_field_indices = tuple(
            self.static_context_source_fields.index(field)
            for field in self.static_context_effective_fields
            if field in self.static_context_source_fields
        )
        self.horizon = int(self.manifest.get("horizon", PATH20_HORIZON) or PATH20_HORIZON)
        self.cumulative_horizons = normalize_path20_cumulative_horizons(
            self.manifest.get("cumulative_horizons", PATH20_CUMULATIVE_HORIZONS),
            horizon=self.horizon,
        )
        self.symbol_vocab_fingerprint = str(self.manifest.get("symbol_vocab_fingerprint", "") or "")
        self.industry_vocab_fingerprint = str(self.manifest.get("industry_vocab_fingerprint", "") or "")
        self.board_vocab_fingerprint = str(self.manifest.get("board_vocab_fingerprint", "") or "")
        if isinstance(dataset, FORECAST_MEMMAP_DATASET_TYPES):
            self.row_count = int(dataset.row_count)
            self.input_dim = int(dataset.input_dim)
            self.lookback_days = int(dataset.lookback_days)
        else:
            self.row_count = int(dataset.x.shape[0])
            self.input_dim = int(dataset.x.shape[-1]) if dataset.x.ndim == 3 else 0
            self.lookback_days = int(dataset.x.shape[1]) if dataset.x.ndim == 3 else int(self.manifest.get("lookback_days", 0))

    @staticmethod
    def _normalize_static_context_override(
        fields: tuple[str, ...] | list[str] | str | None,
    ) -> tuple[str, ...] | None:
        if fields is None:
            return None
        if isinstance(fields, str):
            parsed = tuple(str(item).strip() for item in fields.split(",") if str(item).strip())
        else:
            parsed = tuple(str(item).strip() for item in fields if str(item).strip())
        return parsed or None

    @classmethod
    def _effective_static_context_schema(
        cls,
        schema: dict[str, Any],
        *,
        static_context_fields_override: tuple[str, ...] | list[str] | str | None,
    ) -> dict[str, Any]:
        source_schema = dict(schema or {"enabled": False})
        if not bool(source_schema.get("enabled", False)):
            if cls._normalize_static_context_override(static_context_fields_override):
                raise ValueError("static_context_fields_override requires a dataset with static context enabled.")
            return source_schema
        override_fields = cls._normalize_static_context_override(static_context_fields_override)
        if override_fields is None:
            effective_schema = dict(source_schema)
            effective_schema.setdefault("training_fields_override", [])
            effective_schema.setdefault("source_fields", list(source_schema.get("fields", []) or []))
            return effective_schema
        source_fields = tuple(str(item) for item in source_schema.get("fields", []) or [])
        missing = [field for field in override_fields if field not in source_fields]
        if missing:
            raise ValueError(
                "static_context_fields_override must be a subset of manifest static fields; "
                f"missing: {', '.join(missing)}"
            )
        vocab_sizes = dict(source_schema.get("vocab_sizes", {}) or {})
        embedding_defaults = dict(source_schema.get("embedding_defaults", {}) or {})
        effective_schema = dict(source_schema)
        effective_schema["fields"] = list(override_fields)
        effective_schema["id_columns"] = [f"{field}_id" for field in override_fields]
        effective_schema["vocab_sizes"] = {field: vocab_sizes[field] for field in override_fields if field in vocab_sizes}
        effective_schema["embedding_defaults"] = {
            key: value
            for key, value in embedding_defaults.items()
            if key == "dropout" or key in set(override_fields)
        }
        effective_schema["source_fields"] = list(source_fields)
        effective_schema["training_fields_override"] = list(override_fields)
        return effective_schema

    def role_indices(self, role: str) -> np.ndarray:
        if isinstance(self.dataset, FORECAST_MEMMAP_DATASET_TYPES):
            return self.dataset.role_indices(role)
        return np.flatnonzero(self.dataset.role == role)

    def date_count_for_indices(self, indices: np.ndarray) -> int:
        row_indices = np.asarray(indices, dtype=np.int64)
        if row_indices.size == 0:
            return 0
        if isinstance(self.dataset, FORECAST_MEMMAP_DATASET_TYPES):
            return int(pd.to_datetime(self.dataset.sample_index.iloc[row_indices]["date"]).nunique())
        return int(pd.Series(pd.to_datetime(self.dataset.date[row_indices])).nunique())

    def date_values_for_indices(self, indices: np.ndarray) -> np.ndarray:
        row_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if row_indices.size == 0:
            return np.asarray([], dtype=object)
        if isinstance(self.dataset, FORECAST_MEMMAP_DATASET_TYPES):
            return pd.to_datetime(self.dataset.sample_index.iloc[row_indices]["date"]).to_numpy(dtype=object)
        return pd.to_datetime(self.dataset.date[row_indices]).to_numpy(dtype=object)

    def torch_dataset(self, indices: np.ndarray, *, target_scale: float) -> torch.utils.data.Dataset:
        if isinstance(self.dataset, FORECAST_MEMMAP_DATASET_TYPES):
            return self.dataset.torch_dataset(indices, target_scale=target_scale)
        return _EagerTorchDataset(self.dataset, indices, target_scale=target_scale)

    def batch_torch_dataset(self, indices: np.ndarray, *, batch_size: int, target_scale: float) -> torch.utils.data.Dataset:
        if isinstance(self.dataset, FORECAST_MEMMAP_DATASET_TYPES):
            return self.dataset.batch_torch_dataset(indices, batch_size=batch_size, target_scale=target_scale)
        return self.torch_dataset(indices, target_scale=target_scale)

    def date_batch_torch_dataset(self, indices: np.ndarray, *, target_scale: float) -> torch.utils.data.Dataset:
        if not isinstance(self.dataset, FORECAST_MEMMAP_DATASET_TYPES):
            raise ValueError("date-level forecast batches require a memmap dataset.")
        return self.dataset.date_batch_torch_dataset(indices, target_scale=target_scale)

    def date_slate_torch_dataset(
        self,
        indices: np.ndarray,
        *,
        dates_per_batch: int,
        stocks_per_date: int,
        target_scale: float,
        shuffle_stocks: bool,
        seed: int,
    ) -> torch.utils.data.Dataset:
        if not isinstance(self.dataset, ForecastTrainingPackDataset):
            raise ValueError("date-slate forecast batches require a QDP training pack dataset.")
        return self.dataset.date_slate_torch_dataset(
            indices,
            dates_per_batch=dates_per_batch,
            stocks_per_date=stocks_per_date,
            target_scale=target_scale,
            shuffle_stocks=shuffle_stocks,
            seed=seed,
        )

    @property
    def supports_static_context(self) -> bool:
        return bool(self.static_context_schema.get("enabled", False))

    def filter_static_context_ids(self, static_context_ids: torch.Tensor | None) -> torch.Tensor | None:
        if static_context_ids is None:
            return None
        if not self.static_context_field_indices:
            return static_context_ids
        if tuple(range(len(self.static_context_source_fields))) == self.static_context_field_indices:
            return static_context_ids
        indices = torch.as_tensor(self.static_context_field_indices, dtype=torch.long, device=static_context_ids.device)
        return static_context_ids.index_select(dim=-1, index=indices)


def _apply_train_date_stride(
    dataset_view: _ForecastDatasetView,
    train_indices: np.ndarray,
    *,
    train_date_stride: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    row_indices = np.asarray(train_indices, dtype=np.int64).reshape(-1)
    stride = int(train_date_stride)
    if stride <= 0:
        raise ValueError("train_date_stride must be positive.")
    source_dates = pd.Series(pd.to_datetime(dataset_view.date_values_for_indices(row_indices))).dt.normalize()
    unique_dates = sorted(pd.Timestamp(item) for item in source_dates.dropna().unique())
    if stride <= 1 or not unique_dates:
        summary = {
            "enabled": False,
            "mode": "all_train_dates",
            "train_date_stride": int(stride),
            "source_train_rows": int(row_indices.size),
            "kept_train_rows": int(row_indices.size),
            "dropped_train_rows": 0,
            "source_train_dates": int(len(unique_dates)),
            "kept_train_dates": int(len(unique_dates)),
            "dropped_train_dates": 0,
            "first_kept_train_date": pd.Timestamp(unique_dates[0]).strftime("%Y-%m-%d") if unique_dates else "",
            "last_kept_train_date": pd.Timestamp(unique_dates[-1]).strftime("%Y-%m-%d") if unique_dates else "",
            "validation_test_full": True,
        }
        return row_indices, summary

    kept_dates = unique_dates[::stride]
    kept_date_set = {pd.Timestamp(item) for item in kept_dates}
    keep_mask = source_dates.map(lambda item: pd.Timestamp(item) in kept_date_set if pd.notna(item) else False).to_numpy(dtype=bool)
    filtered = row_indices[keep_mask]
    summary = {
        "enabled": True,
        "mode": "train_date_stride",
        "train_date_stride": int(stride),
        "source_train_rows": int(row_indices.size),
        "kept_train_rows": int(filtered.size),
        "dropped_train_rows": int(row_indices.size - filtered.size),
        "source_train_dates": int(len(unique_dates)),
        "kept_train_dates": int(len(kept_dates)),
        "dropped_train_dates": int(len(unique_dates) - len(kept_dates)),
        "first_kept_train_date": pd.Timestamp(kept_dates[0]).strftime("%Y-%m-%d") if kept_dates else "",
        "last_kept_train_date": pd.Timestamp(kept_dates[-1]).strftime("%Y-%m-%d") if kept_dates else "",
        "validation_test_full": True,
    }
    return filtered, summary


class LinearLastDayPath20Forecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int = PATH20_HORIZON,
        output_profile: str = "forecast_path_v1",
        cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.output_profile = str(output_profile or "forecast_path_v1").strip().lower()
        self.cumulative_horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=int(horizon))
        self.base = LinearPath20Forecaster(
            input_dim=int(input_dim),
            horizon=int(horizon),
            output_profile=self.output_profile,
            cumulative_horizons=self.cumulative_horizons,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.base(x)


class MLPLastDayPath20Forecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 192,
        dropout: float = 0.15,
        horizon: int = PATH20_HORIZON,
        output_profile: str = "forecast_path_v1",
        cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.output_profile = str(output_profile or "forecast_path_v1").strip().lower()
        self.cumulative_horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=int(horizon))
        self.base = Path20ForecasterMLP(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            horizon=int(horizon),
            output_profile=self.output_profile,
            cumulative_horizons=self.cumulative_horizons,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.base(x)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def make_forecast_model(
    family: str,
    *,
    input_dim: int,
    hidden_dim: int = 192,
    horizon: int = PATH20_HORIZON,
    dropout: float = 0.15,
    gru_layers: int = 2,
    transformer_layers: int = 4,
    transformer_heads: int = 6,
    patch_sizes: tuple[int, ...] | list[int] = (4, 20),
    cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    static_context_vocab_sizes: dict[str, int] | None = None,
    static_context_embedding_dims: dict[str, int] | None = None,
    static_context_fields: tuple[str, ...] | list[str] | None = None,
    static_context_dropout: float = 0.20,
    intraday_feature_indices: tuple[int, ...] | list[int] | None = None,
    feature_group_indices: dict[str, tuple[int, ...] | list[int]] | None = None,
    group_mixer_chunk_size: int = DEFAULT_GROUP_MIXER_CHUNK_SIZE,
    slot_count: int = 8,
    output_profile: str = "forecast_path_v1",
) -> nn.Module:
    family = str(family).strip()
    output_profile = str(output_profile or "forecast_path_v1").strip().lower()
    if output_profile not in FORECAST_OUTPUT_PROFILES:
        raise ValueError(f"Unsupported forecast output profile: {output_profile}")
    resolved_horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=int(horizon))
    if family == "linear_last_day":
        return LinearLastDayPath20Forecaster(input_dim=input_dim, horizon=horizon, output_profile=output_profile, cumulative_horizons=resolved_horizons)
    if family == "dlinear_sequence":
        return DLinearPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            output_profile=output_profile,
            cumulative_horizons=resolved_horizons,
        )
    if family == "mlp_last_day":
        return MLPLastDayPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            horizon=horizon,
            output_profile=output_profile,
            cumulative_horizons=resolved_horizons,
        )
    if family == "gru_sequence":
        return GRUPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            horizon=horizon,
            num_layers=gru_layers,
            output_profile=output_profile,
            cumulative_horizons=resolved_horizons,
        )
    if family == "patch_transformer":
        requested_heads = max(int(transformer_heads), 1)
        num_heads = requested_heads if int(hidden_dim) % requested_heads == 0 else 1
        return PatchTransformerPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            patch_sizes=tuple(int(item) for item in patch_sizes if int(item) > 0),
            num_layers=transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            output_profile=output_profile,
            cumulative_horizons=resolved_horizons,
        )
    if family == "gru_sequence_static_context":
        temporal = GRUPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            horizon=horizon,
            num_layers=gru_layers,
            cumulative_horizons=resolved_horizons,
        )
        return StaticContextPath20Forecaster(
            temporal_encoder=temporal,
            temporal_dim=int(hidden_dim) * 2,
            hidden_dim=hidden_dim,
            horizon=horizon,
            vocab_sizes=static_context_vocab_sizes,
            embedding_dims=static_context_embedding_dims,
            static_fields=static_context_fields,
            static_dropout=static_context_dropout,
            dropout=dropout,
            output_profile=output_profile,
            cumulative_horizons=resolved_horizons,
        )
    if family == "patch_transformer_static_context":
        requested_heads = max(int(transformer_heads), 1)
        num_heads = requested_heads if int(hidden_dim) % requested_heads == 0 else 1
        temporal = PatchTransformerPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            patch_sizes=tuple(int(item) for item in patch_sizes if int(item) > 0),
            num_layers=transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            cumulative_horizons=resolved_horizons,
        )
        return StaticContextPath20Forecaster(
            temporal_encoder=temporal,
            temporal_dim=int(hidden_dim),
            hidden_dim=hidden_dim,
            horizon=horizon,
            vocab_sizes=static_context_vocab_sizes,
            embedding_dims=static_context_embedding_dims,
            static_fields=static_context_fields,
            static_dropout=static_context_dropout,
            dropout=dropout,
            output_profile=output_profile,
            cumulative_horizons=resolved_horizons,
        )
    if family == "stock_mixer_sequence":
        return StockMixerPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            dropout=dropout,
            output_profile=output_profile,
            cumulative_horizons=resolved_horizons,
        )
    if family == "sector_slot_mixer_sequence":
        return SectorSlotMixerPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            dropout=dropout,
            slot_count=slot_count,
            output_profile=output_profile,
            cumulative_horizons=resolved_horizons,
        )
    if family == "hybrid_expert_fusion_static_context":
        return ExpertFusionPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            dropout=dropout,
            gru_layers=gru_layers,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            patch_sizes=tuple(int(item) for item in patch_sizes if int(item) > 0),
            static_context_vocab_sizes=static_context_vocab_sizes,
            static_context_embedding_dims=static_context_embedding_dims,
            static_context_fields=static_context_fields,
            static_context_dropout=static_context_dropout,
            output_profile=output_profile,
            cumulative_horizons=resolved_horizons,
        )
    if family == "hybrid_multiscale_recency_aware_v1":
        return HybridMultiScaleRecencyAwarePath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            dropout=dropout,
            gru_layers=gru_layers,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            patch_sizes=tuple(int(item) for item in patch_sizes if int(item) > 0),
            static_context_vocab_sizes=static_context_vocab_sizes,
            static_context_embedding_dims=static_context_embedding_dims,
            static_context_fields=static_context_fields,
            static_context_dropout=static_context_dropout,
            intraday_feature_indices=tuple(int(item) for item in (intraday_feature_indices or ()) if int(item) >= 0),
            output_profile=output_profile,
            cumulative_horizons=resolved_horizons,
        )
    if family == "hybrid_structured_alpha_v2":
        static_fields = tuple(str(item) for item in (static_context_fields or ("exchange", "industry")) if str(item))
        if any(item == "symbol" for item in static_fields):
            raise ValueError("hybrid_structured_alpha_v2 requires symbol-free static context; use exchange,industry.")
        if output_profile != "forecast_path_v1":
            raise ValueError("hybrid_structured_alpha_v2 only supports forecast_path_v1 output_profile.")
        return HybridStructuredAlphaV2Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            dropout=dropout,
            gru_layers=gru_layers,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            patch_sizes=tuple(int(item) for item in patch_sizes if int(item) > 0),
            static_context_vocab_sizes=static_context_vocab_sizes,
            static_context_embedding_dims=static_context_embedding_dims,
            static_context_fields=static_fields,
            static_context_dropout=static_context_dropout,
            feature_group_indices=feature_group_indices,
            group_mixer_chunk_size=int(group_mixer_chunk_size),
            output_profile=output_profile,
            cumulative_horizons=resolved_horizons,
        )
    if family == "date_slate_alpha_fusion_v1":
        static_fields = tuple(str(item) for item in (static_context_fields or ("exchange", "industry")) if str(item))
        if any(item == "symbol" for item in static_fields):
            raise ValueError("date_slate_alpha_fusion_v1 requires symbol-free static context; use exchange,industry.")
        missing_required = [field for field in ("exchange", "industry") if field not in static_fields]
        if missing_required:
            raise ValueError(f"date_slate_alpha_fusion_v1 requires exchange,industry static context; missing: {missing_required}")
        if output_profile != "forecast_incremental_path_v2":
            raise ValueError("date_slate_alpha_fusion_v1 only supports forecast_incremental_path_v2 output_profile.")
        return DateSlateAlphaFusionV1Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            dropout=dropout,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            patch_sizes=tuple(int(item) for item in patch_sizes if int(item) > 0),
            static_context_vocab_sizes=static_context_vocab_sizes,
            static_context_embedding_dims=static_context_embedding_dims,
            static_context_fields=static_fields,
            static_context_dropout=static_context_dropout,
            feature_group_indices=feature_group_indices,
            group_mixer_chunk_size=int(group_mixer_chunk_size),
            output_profile=output_profile,
            cumulative_horizons=resolved_horizons,
        )
    if family == "regime_routed_multi_expert_horizon_v1":
        return RegimeRoutedMultiExpertHorizonForecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            dropout=dropout,
            gru_layers=gru_layers,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            patch_sizes=tuple(int(item) for item in patch_sizes if int(item) > 0),
            static_context_vocab_sizes=static_context_vocab_sizes,
            static_context_embedding_dims=static_context_embedding_dims,
            static_context_fields=static_context_fields,
            static_context_dropout=static_context_dropout,
            output_profile=output_profile,
            cumulative_horizons=resolved_horizons,
        )
    raise ValueError(f"Unsupported forecast model family: {family}")


def _write_frame(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path.resolve())


def _now_iso_seconds() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _normalize_forecast_loss_profile(loss_profile: str | None) -> str:
    profile = str(loss_profile or "default").strip().lower()
    if profile not in FORECAST_LOSS_PROFILES:
        raise ValueError(f"Unsupported forecast loss profile: {profile}")
    return profile


def _rank_loss_weights_for_profile(loss_profile: str | None) -> dict[int, float]:
    profile = _normalize_forecast_loss_profile(loss_profile)
    return dict(_FORECAST_PROFILE_RANK_LOSS_WEIGHTS.get(profile, FORECAST_RANK_LOSS_WEIGHTS))


def _decision_score_calibration_contract(loss_profile: str | None) -> dict[str, Any]:
    profile = _normalize_forecast_loss_profile(loss_profile)
    enabled = profile in _FORECAST_UTILITY_30D_SOFT_PENALTY_PROFILES
    return {
        "enabled": bool(enabled),
        "method": "max_horizon_utility_soft_penalty" if enabled else "none",
        "penalized_horizon": 30 if enabled else None,
        "utility_penalty": float(_HORIZON_30D_SOFT_PENALTY) if enabled else 0.0,
        "applies_to": ["pred_decision_score", "trade_utility_score", "pred_best_horizon"] if enabled else [],
        "source_evidence": "mh_v2_horizon_concentration_repair_anchor_20260603_01/penalty_30d_0p005"
        if enabled
        else "",
    }


def _time_efficient_topk_contract(loss_profile: str) -> dict[str, Any]:
    profile = _normalize_forecast_loss_profile(loss_profile)
    enabled = profile in _TIME_EFFICIENT_TOPK_LOSS_PROFILES
    if not enabled:
        return {"enabled": False, "method": "none"}
    return {
        "enabled": True,
        "method": "max_net_utility_per_horizon_day",
        "future_score": "max_h((future_cumulative_excess_return_h - cost - path_risk_penalty_h) / h)",
        "horizon_normalization": "divide_by_horizon_days",
        "cost_semantics": "fixed round-trip cost is subtracted before horizon normalization",
        "risk_penalty": {
            "drawdown_penalty_source": "forecast_decision_drawdown_penalty",
            "worst_day_penalty": float(_TIME_EFFICIENT_WORST_DAY_PENALTY),
            "horizon_risk_scale": "sqrt(h / forecast_horizon)",
        },
        "prediction_slots": {
            "decision_aux_first_block": "pred_time_eff_utility_by_horizon_scaled",
            "decision_aux_second_block": "pred_time_eff_hit_logits_by_horizon",
            "decision_aux_third_block": "pred_time_eff_best_horizon_logits",
        },
        "checkpoint_selection": "lowest_validation_loss_for_this_exact_loss_profile",
        "uses_active_execution_artifact": False,
        "not_a_backtest": True,
        "shadow_only": True,
        "promotion_allowed": False,
    }


def _high_return_proxy_contract(loss_profile: str) -> dict[str, Any]:
    profile = _normalize_forecast_loss_profile(loss_profile)
    enabled = profile in _HIGH_RETURN_PROXY_LOSS_PROFILES
    if not enabled:
        return {"enabled": False, "method": "none", "proxy_only": True}
    methods = {
        "topn_excess_rank_v1": "batch_top_quintile_excess_rank_surrogate",
        "decision_score_topk_alignment_v1": "batch_small_topk_decision_score_alignment_surrogate",
        "score_to_weight_proxy_v1": "batch_soft_topn_score_to_weight_surrogate",
        "bad_month_aware_v1": "batch_downside_tail_reweighted_score_surrogate",
        "personal_alpha_scorer_hybrid_v1": "hybrid_batch_top_tail_soft_weight_downside_surrogate",
    }
    descriptions = {
        "topn_excess_rank_v1": "Batch-level top 20% future 20d excess-return proxy plus pairwise rank alignment.",
        "decision_score_topk_alignment_v1": (
            "Batch-level small topK future-return proxy for decision_score alignment; "
            "approximates personal top1/top3/top5 but is not date-cross-sectional topK."
        ),
        "score_to_weight_proxy_v1": "Batch-level soft score-to-weight proxy; not a real daily portfolio.",
        "bad_month_aware_v1": "Sample-level downside-tail proxy despite legacy name; not a monthly aggregation.",
        "personal_alpha_scorer_hybrid_v1": (
            "Hybrid alpha-scorer proxy combining batch top-tail, soft score concentration, "
            "and downside-tail penalties; not date-cross-sectional topK or a backtest."
        ),
    }
    return {
        "enabled": True,
        "method": methods[profile],
        "description": descriptions.get(profile, ""),
        "proxy_only": True,
        "not_a_backtest": True,
        "uses_active_execution_artifact": False,
        "target": "decision_score_scaled_and_future_cum_excess_return_20d",
        "intended_use": "single-seed high-return scout objective alignment",
    }


def forecast_loss_profile_contract(
    loss_profile: str | None,
    *,
    cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    forecast_horizon: int = PATH20_HORIZON,
) -> dict[str, Any]:
    profile = _normalize_forecast_loss_profile(loss_profile)
    horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=int(forecast_horizon))
    weights = dict(_FORECAST_LOSS_WEIGHT_PRESETS[profile])
    if profile in _TIME_EFFICIENT_TOPK_LOSS_PROFILES:
        auxiliary_objectives = [
            key
            for key, value in weights.items()
            if float(value) > 0.0 and key not in {"decision_utility", "quantile"}
        ]
        return _json_ready(
            {
                "schema_version": 1,
                "status": "active",
                "loss_profile": profile,
                "profile_family": "time_efficient_topk_aux",
                "required_output_profile": "decision_utility_v1",
                "primary_objective": "personal_time_efficient_topk",
                "auxiliary_objectives": auxiliary_objectives,
                "loss_component_weights": weights,
                "forecast_horizon": int(forecast_horizon),
                "cumulative_horizons": [int(item) for item in horizons],
                "target_normalization": "per_horizon_utility_divided_by_day",
                "horizon_head_constraint": {
                    "enabled": False,
                    "max_30d_probability": _HORIZON_HEAD_CONSTRAINT_MAX_30D_PROBABILITY,
                    "min_long_horizon_probability": _HORIZON_HEAD_CONSTRAINT_MIN_LONG_PROBABILITY,
                    "long_horizons": list(_HORIZON_HEAD_CONSTRAINT_LONG_HORIZONS),
                },
                "decision_score_calibration": _decision_score_calibration_contract(profile),
                "time_efficient_topk_objective": _time_efficient_topk_contract(profile),
                "high_return_proxy_objective": _high_return_proxy_contract(profile),
                "shadow_only": True,
                "promotion_allowed": False,
                "active_execution_strategy_expected_diff": "none",
            }
        )
    if profile in _HYBRID_ALPHA_SCORE_LOSS_PROFILES:
        auxiliary_objectives = [
            key
            for key, value in weights.items()
            if float(value) > 0.0 and key not in {"path_daily", "quantile"}
        ]
        if profile == "date_grouped_alpha_score_v1":
            return _json_ready(
                {
                    "schema_version": 1,
                    "status": "active",
                    "loss_profile": profile,
                    "profile_family": "date_grouped_alpha_score_forecast",
                    "required_output_profile": "forecast_incremental_path_v2",
                    "primary_objective": "same_date_prediction_first_alpha_score",
                    "auxiliary_objectives": auxiliary_objectives,
                    "loss_component_weights": weights,
                    "forecast_horizon": int(forecast_horizon),
                    "cumulative_horizons": [int(item) for item in horizons],
                    "target_normalization": "market_fact_prediction_with_same_date_rank_and_unit_time_rank",
                    "alpha_score_objective": {
                        "enabled": True,
                        "method": "daily_increment_path_plus_derived_cumulative_path_plus_same_date_rank",
                        "model_role": "stable_alpha_score_generator",
                        "execution_cost_in_loss": False,
                        "execution_risk_penalty_in_loss": False,
                        "batch_topk_alignment_in_loss": False,
                        "rank_loss_scope": "within_same_prediction_date_only",
                        "unit_time_efficiency_in_loss": True,
                        "rank_loss_horizon_weights": _rank_loss_weights_for_profile(profile),
                        "direct_train_target": "market_fact_alpha_score_not_execution_policy",
                        "required_static_context": "exchange,industry",
                        "symbol_static_context_allowed": False,
                    },
                    "shadow_only": True,
                    "promotion_allowed": False,
                    "active_execution_strategy_expected_diff": "none",
                }
            )
        return _json_ready(
            {
                "schema_version": 1,
                "status": "active",
                "loss_profile": profile,
                "profile_family": "hybrid_alpha_score_forecast",
                "required_output_profile": "forecast_path_v1",
                "primary_objective": "hybrid_alpha_score",
                "auxiliary_objectives": auxiliary_objectives,
                "loss_component_weights": weights,
                "forecast_horizon": int(forecast_horizon),
                "cumulative_horizons": [int(item) for item in horizons],
                "target_normalization": "market_fact_prediction_with_per_day_alpha_rank_aux",
                "alpha_score_objective": {
                    "enabled": True,
                    "method": (
                        "forecast_path_plus_multi_horizon_rank_plus_per_day_alpha_efficiency_rank"
                        if profile == "hybrid_alpha_score_v1"
                        else "forecast_path_plus_short_mid_horizon_rank_plus_unit_time_alpha_efficiency_rank"
                    ),
                    "model_role": "stable_alpha_score_generator",
                    "execution_cost_in_loss": False,
                    "execution_risk_penalty_in_loss": False,
                    "batch_topk_alignment_in_loss": False,
                    "unit_time_efficiency_in_loss": bool(float(weights.get("alpha_efficiency_rank_aux", 0.0) or 0.0) > 0.0),
                    "rank_loss_horizon_weights": _rank_loss_weights_for_profile(profile),
                    "rank_horizon_bias": "legacy_20d_tilt" if profile == "hybrid_alpha_score_v1" else "short_mid_horizon_unit_time_tilt",
                    "direct_train_target": "market_fact_alpha_score_not_execution_policy",
                    "preferred_static_context": "exchange,industry",
                    "intended_external_scorer": "personal_topk_v1_or_successor_handles_cost_risk_horizon_and_trade_constraints",
                },
                "horizon_head_constraint": {
                    "enabled": False,
                    "max_30d_probability": _HORIZON_HEAD_CONSTRAINT_MAX_30D_PROBABILITY,
                    "min_long_horizon_probability": _HORIZON_HEAD_CONSTRAINT_MIN_LONG_PROBABILITY,
                    "long_horizons": list(_HORIZON_HEAD_CONSTRAINT_LONG_HORIZONS),
                },
                "decision_score_calibration": _decision_score_calibration_contract(profile),
                "time_efficient_topk_objective": _time_efficient_topk_contract(profile),
                "high_return_proxy_objective": _high_return_proxy_contract(profile),
                "shadow_only": True,
                "promotion_allowed": False,
                "active_execution_strategy_expected_diff": "none",
            }
        )
    return _json_ready(
        {
            "schema_version": 1,
            "status": "active",
            "loss_profile": profile,
            "profile_family": "decision_utility_aux" if profile in _FORECAST_DECISION_LOSS_PROFILES else "path_forecast",
            "required_output_profile": "decision_utility_v1"
            if profile in _FORECAST_DECISION_LOSS_PROFILES
            else "forecast_path_v1",
            "primary_objective": "decision_utility" if profile in _FORECAST_DECISION_LOSS_PROFILES else "path_forecast",
            "auxiliary_objectives": [
                key
                for key, value in weights.items()
                if float(value) > 0.0
                and key
                not in {
                    "decision_utility" if profile in _FORECAST_DECISION_LOSS_PROFILES else "path_daily",
                    "quantile",
                }
            ],
            "loss_component_weights": weights,
            "forecast_horizon": int(forecast_horizon),
            "cumulative_horizons": [int(item) for item in horizons],
            "target_normalization": "per_horizon_utility_zscore"
            if profile in _FORECAST_TARGET_NORMALIZED_LOSS_PROFILES
            else "none",
            "horizon_head_constraint": {
                "enabled": profile in _FORECAST_HEAD_CONSTRAINT_LOSS_PROFILES,
                "max_30d_probability": _HORIZON_HEAD_CONSTRAINT_MAX_30D_PROBABILITY,
                "min_long_horizon_probability": _HORIZON_HEAD_CONSTRAINT_MIN_LONG_PROBABILITY,
                "long_horizons": list(_HORIZON_HEAD_CONSTRAINT_LONG_HORIZONS),
            },
            "decision_score_calibration": _decision_score_calibration_contract(profile),
            "time_efficient_topk_objective": _time_efficient_topk_contract(profile),
            "high_return_proxy_objective": _high_return_proxy_contract(profile),
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
    )


def _forecast_resume_contract(
    *,
    model_family: str,
    seed: int,
    feature_columns: list[str],
    feature_profile: str,
    lookback_days: int,
    horizon: int,
    cumulative_horizons: tuple[int, ...] | list[int] | str | None,
    target_scale: float,
    model_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    selection_profile: str,
    loss_profile: str = "default",
    output_profile: str = "forecast_path_v1",
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
    static_context_schema: dict[str, Any] | None = None,
    sampling_config: dict[str, Any] | None = None,
    symbol_vocab_fingerprint: str = "",
    industry_vocab_fingerprint: str = "",
    board_vocab_fingerprint: str = "",
) -> dict[str, Any]:
    return {
        "model_family": str(model_family),
        "seed": int(seed),
        "feature_columns": list(feature_columns),
        "feature_profile": str(feature_profile),
        "lookback_days": int(lookback_days),
        "horizon": int(horizon),
        "forecast_horizon": int(horizon),
        "cumulative_horizons": [int(item) for item in normalize_path20_cumulative_horizons(cumulative_horizons, horizon=int(horizon))],
        "target_scale": float(target_scale),
        "model_config": _json_ready(model_config),
        "optimizer_config": _json_ready(optimizer_config),
        "selection_profile": str(selection_profile),
        "loss_profile": str(loss_profile or "default"),
        "output_profile": str(output_profile or "forecast_path_v1"),
        "loss_profile_contract": forecast_loss_profile_contract(
            loss_profile,
            cumulative_horizons=cumulative_horizons,
            forecast_horizon=int(horizon),
        ),
        "decision_cost_bps": float(decision_cost_bps),
        "decision_hit_threshold_bps": float(decision_hit_threshold_bps),
        "decision_drawdown_penalty": float(decision_drawdown_penalty),
        "static_context_schema": _json_ready(static_context_schema or {"enabled": False}),
        "sampling_config": _json_ready(sampling_config or {}),
        "symbol_vocab_fingerprint": str(symbol_vocab_fingerprint or ""),
        "industry_vocab_fingerprint": str(industry_vocab_fingerprint or ""),
        "board_vocab_fingerprint": str(board_vocab_fingerprint or ""),
    }


def _forecast_model_state_dict(model: nn.Module) -> dict[str, Any]:
    return {
        str(key): value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for key, value in model.state_dict().items()
    }


def _forecast_checkpoint_payload(
    *,
    checkpoint_kind: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    model_family: str,
    seed: int,
    epoch: int,
    best_epoch: int,
    best_score: float,
    best_validation_loss: float,
    best_validation_metrics: dict[str, Any],
    best_train_loss: float,
    last_train_loss: float,
    patience_used: int,
    feature_columns: list[str],
    feature_profile: str,
    feature_manifest: dict[str, Any],
    normalization_manifest: dict[str, Any],
    target_scale: float,
    lookback_days: int,
    horizon: int,
    cumulative_horizons: tuple[int, ...] | list[int] | str | None,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    selection_profile: str,
    learning_rows: list[dict[str, Any]],
    loss_profile: str = "default",
    output_profile: str = "forecast_path_v1",
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
    static_context_schema: dict[str, Any] | None = None,
    symbol_vocab_fingerprint: str = "",
    industry_vocab_fingerprint: str = "",
    board_vocab_fingerprint: str = "",
    best_checkpoint_pt: Path | None = None,
    best_checkpoint_payload: dict[str, Any] | None = None,
    rng_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checkpoint_kind": str(checkpoint_kind),
        "checkpoint_created_at": _now_iso_seconds(),
        "model_family": str(model_family),
        "seed": int(seed),
        "epoch": int(epoch),
        "state_dict": _forecast_model_state_dict(model),
        "feature_columns": list(feature_columns),
        "feature_profile": str(feature_profile),
        "feature_manifest": _json_ready(feature_manifest),
        "normalization": _json_ready(normalization_manifest),
        "target_scale": float(target_scale),
        "lookback_days": int(lookback_days),
        "horizon": int(horizon),
        "forecast_horizon": int(horizon),
        "cumulative_horizons": [int(item) for item in normalize_path20_cumulative_horizons(cumulative_horizons, horizon=int(horizon))],
        "model_config": _json_ready(model_config),
        "training_config": _json_ready(training_config),
        "optimizer_config": _json_ready(optimizer_config),
        "resume_contract": _forecast_resume_contract(
            model_family=model_family,
            seed=int(seed),
            feature_columns=list(feature_columns),
            feature_profile=feature_profile,
            lookback_days=int(lookback_days),
            horizon=int(horizon),
            cumulative_horizons=cumulative_horizons,
            target_scale=float(target_scale),
            model_config=model_config,
            optimizer_config=optimizer_config,
            selection_profile=selection_profile,
            loss_profile=loss_profile,
            output_profile=output_profile,
            decision_cost_bps=decision_cost_bps,
            decision_hit_threshold_bps=decision_hit_threshold_bps,
            decision_drawdown_penalty=decision_drawdown_penalty,
            static_context_schema=static_context_schema,
            sampling_config=dict(training_config.get("sampling_config", {}) or {}),
            symbol_vocab_fingerprint=symbol_vocab_fingerprint,
            industry_vocab_fingerprint=industry_vocab_fingerprint,
            board_vocab_fingerprint=board_vocab_fingerprint,
        ),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_metrics": _json_ready(best_validation_metrics),
        "best_train_loss": float(best_train_loss),
        "last_train_loss": float(last_train_loss),
        "patience_used": int(patience_used),
        "learning_rows": _json_ready(learning_rows),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if best_checkpoint_pt is not None:
        payload["best_checkpoint_pt"] = str(best_checkpoint_pt.resolve())
    if best_checkpoint_payload is not None:
        payload["best_checkpoint_payload"] = best_checkpoint_payload
    if rng_state is not None:
        payload["rng_state"] = rng_state
    return payload


def _save_forecast_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)
    return str(path.resolve())


def _load_forecast_resume_checkpoint(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"forecast resume checkpoint does not exist: {resolved}")
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("forecast resume checkpoint payload must be a dictionary.")
    if payload.get("checkpoint_kind") != "last":
        raise ValueError("forecast resume checkpoint must be a last checkpoint.")
    for key in ("state_dict", "optimizer_state_dict", "scaler_state_dict", "epoch", "resume_contract"):
        if key not in payload:
            raise ValueError(f"forecast resume checkpoint is missing {key}.")
    return payload


def _validate_forecast_resume_checkpoint(
    payload: dict[str, Any],
    *,
    model_family: str,
    seed: int,
    dataset_view: "_ForecastDatasetView",
    target_scale: float,
    model_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    selection_profile: str,
    loss_profile: str = "default",
    output_profile: str = "forecast_path_v1",
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
    sampling_config: dict[str, Any] | None = None,
) -> None:
    expected = _forecast_resume_contract(
        model_family=model_family,
        seed=int(seed),
        feature_columns=list(dataset_view.feature_columns),
        feature_profile=str(dataset_view.manifest.get("feature_profile", "")),
        lookback_days=int(dataset_view.lookback_days),
        horizon=int(dataset_view.horizon),
        cumulative_horizons=dataset_view.cumulative_horizons,
        target_scale=float(target_scale),
        model_config=model_config,
        optimizer_config=optimizer_config,
        selection_profile=selection_profile,
        loss_profile=loss_profile,
        output_profile=output_profile,
        decision_cost_bps=decision_cost_bps,
        decision_hit_threshold_bps=decision_hit_threshold_bps,
        decision_drawdown_penalty=decision_drawdown_penalty,
        static_context_schema=dataset_view.static_context_schema,
        sampling_config=sampling_config,
        symbol_vocab_fingerprint=dataset_view.symbol_vocab_fingerprint,
        industry_vocab_fingerprint=dataset_view.industry_vocab_fingerprint,
        board_vocab_fingerprint=dataset_view.board_vocab_fingerprint,
    )
    actual = dict(payload.get("resume_contract", {}) or {})
    if not actual:
        raise ValueError("forecast resume checkpoint is missing resume_contract.")
    for key, expected_value in expected.items():
        if key == "sampling_config" and key not in actual and (
            expected_value == {} or not bool(dict(expected_value or {}).get("enabled", False))
        ):
            continue
        if key == "model_config":
            actual_model_config = dict(actual.get(key, {}) or {})
            expected_model_config = dict(expected_value or {})
            if (
                "group_mixer_chunk_size" not in actual_model_config
                and int(expected_model_config.get("group_mixer_chunk_size", DEFAULT_GROUP_MIXER_CHUNK_SIZE))
                == DEFAULT_GROUP_MIXER_CHUNK_SIZE
            ):
                actual_model_config["group_mixer_chunk_size"] = DEFAULT_GROUP_MIXER_CHUNK_SIZE
            if actual_model_config == expected_model_config:
                continue
        if actual.get(key) != expected_value:
            raise ValueError(
                f"forecast resume checkpoint {key} mismatch: "
                f"expected {expected_value!r}, got {actual.get(key)!r}"
            )


def _forecast_rng_state(generator: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "data_loader_generator_state": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_forecast_rng_state(payload: dict[str, Any], generator: torch.Generator) -> None:
    state = payload.get("rng_state")
    if not isinstance(state, dict):
        return
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"])
    if "numpy_rng_state" in state:
        np.random.set_state(state["numpy_rng_state"])
    if "data_loader_generator_state" in state:
        generator.set_state(state["data_loader_generator_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])


def _write_learning_curve_incremental(path: Path, learning_rows: list[dict[str, Any]]) -> str:
    return _write_frame(path, pd.DataFrame(_json_ready(learning_rows)))


def _write_forecast_progress(
    path: Path,
    *,
    status: str,
    model_family: str,
    seed: int,
    current_epoch: int,
    max_epochs: int,
    min_epochs: int,
    patience_limit: int,
    patience_used: int,
    best_epoch: int,
    best_score: float,
    best_validation_loss: float,
    best_validation_metrics: dict[str, Any],
    last_train_loss: float,
    epoch_seconds: float,
    run_started_at: str,
    elapsed_seconds: float,
    learning_rows: list[dict[str, Any]],
    last_checkpoint_pt: Path | None,
    best_checkpoint_pt: Path | None,
    phase: str = "epoch_end",
    current_step: int = 0,
    total_steps: int = 0,
    samples_processed_epoch: int = 0,
    samples_per_second: float = 0.0,
    epoch_eta_seconds: float = 0.0,
    throughput_meta: dict[str, Any] | None = None,
) -> str:
    current_rows = [
        row
        for row in learning_rows
        if str(row.get("model_family")) == str(model_family) and int(row.get("seed", -1)) == int(seed)
    ]
    epoch_durations = [float(row.get("epoch_seconds", 0.0) or 0.0) for row in current_rows if row.get("epoch_seconds") is not None]
    avg_epoch_seconds = float(np.mean(epoch_durations)) if epoch_durations else float(epoch_seconds)
    remaining_epochs = max(int(max_epochs) - int(current_epoch), 0) if str(status) == "running" else 0
    estimated_remaining_seconds = float(avg_epoch_seconds * remaining_epochs)
    eta_at = (
        datetime.now().astimezone() + timedelta(seconds=estimated_remaining_seconds)
    ).isoformat(timespec="seconds")
    payload = {
        "status": str(status),
        "updated_at": _now_iso_seconds(),
        "run_started_at": str(run_started_at),
        "model_family": str(model_family),
        "seed": int(seed),
        "current_epoch": int(current_epoch),
        "max_epochs": int(max_epochs),
        "min_epochs": int(min_epochs),
        "patience_limit": int(patience_limit),
        "patience_used": int(patience_used),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_metrics": _json_ready(best_validation_metrics),
        "last_train_loss": float(last_train_loss),
        "epoch_seconds": float(epoch_seconds),
        "avg_epoch_seconds": float(avg_epoch_seconds),
        "elapsed_seconds": float(elapsed_seconds),
        "estimated_remaining_seconds": float(estimated_remaining_seconds),
        "eta_at": eta_at,
        "last_checkpoint_pt": str(last_checkpoint_pt.resolve()) if last_checkpoint_pt is not None else "",
        "best_checkpoint_pt": str(best_checkpoint_pt.resolve()) if best_checkpoint_pt is not None else "",
        "phase": str(phase),
        "current_step": int(current_step),
        "total_steps": int(total_steps),
        "samples_processed_epoch": int(samples_processed_epoch),
        "samples_per_second": float(samples_per_second),
        "epoch_eta_seconds": float(epoch_eta_seconds),
        "throughput": _json_ready(throughput_meta or {}),
    }
    write_json(path, _json_ready(payload))
    return str(path.resolve())


def _data_loader_kwargs(*, pin_memory: bool, dataloader_num_workers: int, prefetch_factor: int) -> dict[str, Any]:
    workers = max(int(dataloader_num_workers), 0)
    kwargs: dict[str, Any] = {"num_workers": workers, "pin_memory": bool(pin_memory)}
    if workers > 0:
        kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)
    return kwargs


def _tensor_finite_summary(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach()
    finite = torch.isfinite(tensor)
    finite_count = int(finite.sum().detach().cpu())
    total = int(tensor.numel())
    if finite_count > 0:
        finite_values = tensor[finite].to(dtype=torch.float32)
        min_value = float(finite_values.min().detach().cpu())
        max_value = float(finite_values.max().detach().cpu())
        absmax = float(finite_values.abs().max().detach().cpu())
        mean_value = float(finite_values.mean().detach().cpu())
    else:
        min_value = max_value = absmax = mean_value = 0.0
    return {
        "shape": [int(item) for item in tensor.shape],
        "dtype": str(tensor.dtype),
        "finite_count": finite_count,
        "total_count": total,
        "nonfinite_count": int(total - finite_count),
        "finite_ratio": float(finite_count / max(total, 1)),
        "min": min_value,
        "max": max_value,
        "absmax": absmax,
        "mean": mean_value,
    }


def _first_nonfinite_tensor(name: str, value: Any) -> tuple[str, torch.Tensor] | None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all().detach().cpu()):
            return name, value
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            found = _first_nonfinite_tensor(f"{name}.{key}", item)
            if found is not None:
                return found
    return None


def _write_forecast_bad_batch_dump(
    dump_dir: Path,
    *,
    reason: str,
    model_family: str,
    seed: int,
    epoch: int,
    step: int,
    role: str,
    row_ids: torch.Tensor | None,
    date_group_ids: torch.Tensor | None,
    tensors: dict[str, Any],
    scaler: torch.amp.GradScaler | None = None,
) -> Path:
    dump_dir.mkdir(parents=True, exist_ok=True)
    path = dump_dir / f"bad_batch_{model_family}_seed{int(seed)}_e{int(epoch)}_s{int(step)}_{str(reason).replace(' ', '_')}.json"
    row_id_values: list[int] = []
    if isinstance(row_ids, torch.Tensor):
        row_id_values = [int(item) for item in row_ids.detach().reshape(-1).cpu().tolist()[:256] if int(item) >= 0]
    date_group_values: list[int] = []
    if isinstance(date_group_ids, torch.Tensor):
        date_group_values = [int(item) for item in date_group_ids.detach().reshape(-1).cpu().tolist()[:256]]
    tensor_summaries: dict[str, Any] = {}
    for key, value in tensors.items():
        if isinstance(value, torch.Tensor):
            tensor_summaries[key] = _tensor_finite_summary(value)
        elif isinstance(value, dict):
            tensor_summaries[key] = {
                str(child_key): _tensor_finite_summary(child_value)
                for child_key, child_value in value.items()
                if isinstance(child_value, torch.Tensor)
            }
    payload = {
        "status": "nonfinite_detected",
        "reason": str(reason),
        "created_at": _now_iso_seconds(),
        "model_family": str(model_family),
        "seed": int(seed),
        "epoch": int(epoch),
        "step": int(step),
        "role": str(role),
        "row_ids_prefix": row_id_values,
        "date_group_ids_prefix": date_group_values,
        "tensor_summaries": tensor_summaries,
        "amp_scale": float(scaler.get_scale()) if scaler is not None else None,
    }
    write_json(path, _json_ready(payload))
    return path


def _forecast_finite_guard(
    *,
    enabled: bool,
    dump_dir: Path,
    reason_prefix: str,
    model_family: str,
    seed: int,
    epoch: int,
    step: int,
    role: str,
    row_ids: torch.Tensor | None,
    date_group_ids: torch.Tensor | None,
    tensors: dict[str, Any],
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    if not bool(enabled):
        return
    for key, value in tensors.items():
        found = _first_nonfinite_tensor(key, value)
        if found is None:
            continue
        found_name, _ = found
        dump_path = _write_forecast_bad_batch_dump(
            dump_dir,
            reason=f"{reason_prefix}_{found_name}",
            model_family=model_family,
            seed=int(seed),
            epoch=int(epoch),
            step=int(step),
            role=role,
            row_ids=row_ids,
            date_group_ids=date_group_ids,
            tensors=tensors,
            scaler=scaler,
        )
        raise FloatingPointError(f"forecast_nonfinite_detected: {reason_prefix}.{found_name}; dump={dump_path}")


def _forecast_grad_param_finite_guard(
    *,
    enabled: bool,
    dump_dir: Path,
    model: nn.Module,
    model_family: str,
    seed: int,
    epoch: int,
    step: int,
    role: str,
    row_ids: torch.Tensor | None,
    date_group_ids: torch.Tensor | None,
    scaler: torch.amp.GradScaler | None = None,
    check_params: bool = False,
) -> None:
    if not bool(enabled):
        return
    tensors: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            tensors[f"grad.{name}"] = param.grad
        if bool(check_params):
            tensors[f"param.{name}"] = param
    for key, tensor in tensors.items():
        if bool(torch.isfinite(tensor).all().detach().cpu()):
            continue
        dump_path = _write_forecast_bad_batch_dump(
            dump_dir,
            reason=f"nonfinite_{key}",
            model_family=model_family,
            seed=int(seed),
            epoch=int(epoch),
            step=int(step),
            role=role,
            row_ids=row_ids,
            date_group_ids=date_group_ids,
            tensors={key: tensor},
            scaler=scaler,
        )
        raise FloatingPointError(f"forecast_nonfinite_detected: {key}; dump={dump_path}")


def _forecast_loss(
    prediction: dict[str, torch.Tensor],
    y_daily_scaled: torch.Tensor,
    y_cum_scaled: torch.Tensor,
    y_risk_scaled: torch.Tensor,
    *,
    loss_profile: str = "default",
    target_scale: float = 100.0,
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
    cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
) -> torch.Tensor:
    profile = _normalize_forecast_loss_profile(loss_profile)
    weights = dict(_FORECAST_LOSS_WEIGHT_PRESETS[profile])
    daily_weights = torch.ones((y_daily_scaled.shape[1],), device=y_daily_scaled.device, dtype=y_daily_scaled.dtype)
    daily_weights[:3] = 1.15
    daily_weights[3:5] = 1.05
    daily_loss = F.huber_loss(prediction["mu"], y_daily_scaled, reduction="none")
    loss = float(weights["path_daily"]) * (daily_loss * daily_weights.reshape(1, -1)).mean()
    quantile_weight = float(weights["quantile"]) / 3.0
    loss = loss + quantile_weight * pinball_loss(prediction["q10"], y_daily_scaled, 0.10)
    loss = loss + quantile_weight * pinball_loss(prediction["q50"], y_daily_scaled, 0.50)
    loss = loss + quantile_weight * pinball_loss(prediction["q90"], y_daily_scaled, 0.90)
    horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=int(y_daily_scaled.shape[1]))
    cum_count = len(horizons)
    loss = loss + float(weights["path_aux"]) * F.huber_loss(prediction["aux"][:, :cum_count], y_cum_scaled)
    risk_start = cum_count
    risk_width = cum_count * 3
    risk_pred = prediction["aux"][:, risk_start : risk_start + risk_width].reshape(-1, cum_count, 3)
    if y_risk_scaled.ndim == 2:
        y_risk_scaled = y_risk_scaled.reshape(y_risk_scaled.shape[0], 1, y_risk_scaled.shape[1]).expand(-1, cum_count, -1)
    loss = loss + float(weights["risk_aux"]) * F.huber_loss(risk_pred, y_risk_scaled)
    rank_loss_weights = _rank_loss_weights_for_profile(profile)
    for pos, horizon in enumerate(horizons):
        weight = float(rank_loss_weights.get(int(horizon), 0.0)) * float(weights["rank_aux"])
        if weight <= 0.0:
            continue
        score = prediction["mu"][:, : int(horizon)].sum(dim=1)
        target = y_cum_scaled[:, pos]
        loss = loss + weight * pairwise_rank_loss(score, target)
    efficiency_rank_weight = float(weights.get("alpha_efficiency_rank_aux", 0.0) or 0.0)
    if efficiency_rank_weight > 0.0:
        for pos, horizon in enumerate(horizons):
            weight = float(rank_loss_weights.get(int(horizon), 0.0)) * efficiency_rank_weight
            if weight <= 0.0:
                continue
            horizon_scale = float(max(int(horizon), 1))
            score = prediction["mu"][:, : int(horizon)].sum(dim=1) / horizon_scale
            target = y_cum_scaled[:, pos] / horizon_scale
            loss = loss + weight * pairwise_rank_loss(score, target)
    loss = loss + float(weights["risk_rank_aux"]) * pairwise_rank_loss(risk_pred[:, -1, 2], y_risk_scaled[:, -1, 2])
    if float(weights["direction_aux"]) > 0.0:
        direction_target = (y_cum_scaled[:, -1] > 0).to(dtype=prediction["aux"].dtype)
        direction_logit = prediction["mu"].sum(dim=1)
        loss = loss + float(weights["direction_aux"]) * F.binary_cross_entropy_with_logits(direction_logit, direction_target)
    if float(weights["downside_rank_aux"]) > 0.0:
        downside_target = y_risk_scaled[:, -1, 0]
        downside_score = risk_pred[:, -1, 0]
        loss = loss + float(weights["downside_rank_aux"]) * pairwise_rank_loss(-downside_score, -downside_target)
    if profile in _FORECAST_DECISION_LOSS_PROFILES:
        if "decision_aux" not in prediction:
            raise ValueError(f"{profile} loss requires decision_aux outputs.")
        decision_aux = prediction["decision_aux"]
        expected_decision_width = path20_decision_aux_dim(horizons, horizon=int(y_daily_scaled.shape[1]))
        if int(decision_aux.shape[1]) != expected_decision_width:
            raise ValueError(f"decision_aux must have width {expected_decision_width}.")
        if profile in _TIME_EFFICIENT_TOPK_LOSS_PROFILES:
            targets = _time_efficient_utility_targets(
                y_cum_scaled,
                y_risk_scaled,
                target_scale=float(target_scale),
                cost_bps=float(decision_cost_bps),
                hit_threshold_bps=float(decision_hit_threshold_bps),
                drawdown_penalty=float(decision_drawdown_penalty),
                worst_day_penalty=float(_TIME_EFFICIENT_WORST_DAY_PENALTY),
                cumulative_horizons=horizons,
                max_horizon=int(y_daily_scaled.shape[1]),
            )
        else:
            targets = _decision_utility_targets(
                y_cum_scaled,
                y_risk_scaled,
                target_scale=float(target_scale),
                cost_bps=float(decision_cost_bps),
                hit_threshold_bps=float(decision_hit_threshold_bps),
                drawdown_penalty=float(decision_drawdown_penalty),
                cumulative_horizons=horizons,
                max_horizon=int(y_daily_scaled.shape[1]),
                normalize_per_horizon=profile in _FORECAST_TARGET_NORMALIZED_LOSS_PROFILES,
            )
        utility_pred = decision_aux[:, :cum_count]
        hit_logits = decision_aux[:, cum_count : cum_count * 2]
        horizon_logits = decision_aux[:, cum_count * 2 : cum_count * 3]
        utility_target = targets["utility_scaled"].to(dtype=utility_pred.dtype)
        hit_target = targets["hit_label"].to(dtype=hit_logits.dtype)
        best_horizon_index = targets["best_horizon_index"].to(device=horizon_logits.device, dtype=torch.long)
        pred_decision_score = utility_pred.max(dim=1).values
        future_decision_score = targets["decision_score_scaled"].to(dtype=pred_decision_score.dtype)
        loss = loss + float(weights["decision_utility"]) * F.huber_loss(utility_pred, utility_target)
        loss = loss + float(weights["hit_aux"]) * F.binary_cross_entropy_with_logits(hit_logits, hit_target)
        loss = loss + float(weights["horizon_classification"]) * F.cross_entropy(horizon_logits, best_horizon_index)
        loss = loss + float(weights["decision_rank_aux"]) * pairwise_rank_loss(pred_decision_score, future_decision_score)
        if profile in _TIME_EFFICIENT_TOPK_LOSS_PROFILES:
            loss = loss + _time_efficient_topk_alignment_loss(
                pred_score=pred_decision_score,
                future_score=future_decision_score,
                weight=float(weights.get("time_eff_topk_alignment", 0.0)),
            )
        calibrated_rank_weight = float(weights.get("calibrated_decision_rank_aux", 0.0))
        if calibrated_rank_weight > 0.0 and profile in _FORECAST_UTILITY_30D_SOFT_PENALTY_PROFILES:
            calibrated_utility = utility_pred.clone()
            horizon_values = torch.as_tensor(horizons, dtype=torch.long, device=utility_pred.device)
            penalty_mask = horizon_values == int(max(horizons))
            if bool(penalty_mask.any()):
                calibrated_utility[:, penalty_mask] = calibrated_utility[:, penalty_mask] - (
                    float(_HORIZON_30D_SOFT_PENALTY) * float(target_scale)
                )
            calibrated_decision_score = calibrated_utility.max(dim=1).values
            loss = loss + calibrated_rank_weight * pairwise_rank_loss(calibrated_decision_score, future_decision_score)
        constraint_weight = float(weights.get("horizon_head_soft_constraint", 0.0))
        if constraint_weight > 0.0:
            probabilities = F.softmax(horizon_logits, dim=1)
            horizon_values = torch.as_tensor(horizons, dtype=torch.long, device=horizon_logits.device)
            thirty_mask = horizon_values == int(max(horizons))
            long_mask = torch.zeros_like(horizon_values, dtype=torch.bool)
            for item in _HORIZON_HEAD_CONSTRAINT_LONG_HORIZONS:
                long_mask = long_mask | (horizon_values == int(item))
            thirty_probability = probabilities[:, thirty_mask].sum(dim=1) if bool(thirty_mask.any()) else probabilities.new_zeros(probabilities.shape[0])
            long_probability = probabilities[:, long_mask].sum(dim=1) if bool(long_mask.any()) else probabilities.new_zeros(probabilities.shape[0])
            over_thirty = torch.clamp(thirty_probability - _HORIZON_HEAD_CONSTRAINT_MAX_30D_PROBABILITY, min=0.0)
            under_long = torch.clamp(_HORIZON_HEAD_CONSTRAINT_MIN_LONG_PROBABILITY - long_probability, min=0.0)
            loss = loss + constraint_weight * (over_thirty.square().mean() + 0.25 * under_long.square().mean())
        entropy_weight = float(weights.get("horizon_entropy", 0.0))
        if entropy_weight > 0.0:
            probabilities = F.softmax(horizon_logits, dim=1)
            entropy = -(probabilities * torch.log(torch.clamp(probabilities, min=1.0e-8))).sum(dim=1)
            max_entropy = torch.log(torch.as_tensor(float(cum_count), device=entropy.device, dtype=entropy.dtype))
            loss = loss + entropy_weight * torch.clamp(max_entropy - entropy, min=0.0).mean()
        bad_state_weight = float(weights.get("bad_state_calibration", 0.0))
        if bad_state_weight > 0.0 and "bad_state_intensity" in prediction:
            probabilities = F.softmax(horizon_logits, dim=1)
            horizon_values = torch.as_tensor(horizons, dtype=torch.long, device=horizon_logits.device)
            thirty_mask = horizon_values == int(max(horizons))
            if bool(thirty_mask.any()):
                thirty_probability = probabilities[:, thirty_mask].sum(dim=1)
                bad_state = prediction["bad_state_intensity"].to(device=thirty_probability.device, dtype=thirty_probability.dtype)
                loss = loss + bad_state_weight * (bad_state * torch.clamp(thirty_probability - 0.50, min=0.0).square()).mean()
        if profile in _HIGH_RETURN_PROXY_LOSS_PROFILES:
            loss = loss + _high_return_proxy_loss(
                pred_decision_score=pred_decision_score,
                y_cum_scaled=y_cum_scaled,
                y_risk_scaled=y_risk_scaled,
                weights=weights,
                target_scale=float(target_scale),
                cumulative_horizons=horizons,
            )
    router_entropy_weight = float(weights.get("router_entropy_floor", 0.0))
    if router_entropy_weight > 0.0 and "router_entropy" in prediction and "router_weights" in prediction:
        router_entropy = prediction["router_entropy"]
        expert_count = int(prediction["router_weights"].shape[-1])
        max_entropy = torch.log(torch.as_tensor(float(max(expert_count, 1)), device=router_entropy.device, dtype=router_entropy.dtype))
        entropy_floor = max_entropy * 0.60
        loss = loss + router_entropy_weight * torch.clamp(entropy_floor - router_entropy, min=0.0).square().mean()
    diversity_weight = float(weights.get("expert_diversity", 0.0))
    if diversity_weight > 0.0 and "expert_token_diversity" in prediction:
        diversity = prediction["expert_token_diversity"]
        loss = loss + diversity_weight * torch.clamp(0.05 - diversity, min=0.0).square().mean()
    return loss


def _date_grouped_rank_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    date_group_ids: torch.Tensor,
    *,
    min_group_size: int = 8,
    max_pairs_per_date: int = 4096,
) -> torch.Tensor:
    if date_group_ids is None:
        return score.new_tensor(0.0)
    groups = torch.unique(date_group_ids.detach())
    losses: list[torch.Tensor] = []
    for group_id in groups.tolist():
        mask = date_group_ids == int(group_id)
        if int(mask.sum().detach().cpu()) < int(min_group_size):
            continue
        losses.append(pairwise_rank_loss(score[mask], target[mask], max_pairs=int(max_pairs_per_date)))
    if not losses:
        return score.new_tensor(0.0)
    return torch.stack(losses).mean()


def _date_grouped_forecast_loss(
    prediction: dict[str, torch.Tensor],
    y_daily_scaled: torch.Tensor,
    y_cum_scaled: torch.Tensor,
    y_risk_scaled: torch.Tensor,
    date_group_ids: torch.Tensor,
    *,
    loss_profile: str,
    target_scale: float = 100.0,
    cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    rank_min_group_size: int = 8,
    rank_max_pairs_per_date: int = 4096,
) -> torch.Tensor:
    del target_scale
    profile = _normalize_forecast_loss_profile(loss_profile)
    weights = dict(_FORECAST_LOSS_WEIGHT_PRESETS[profile])
    horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=int(y_daily_scaled.shape[1]))
    cum_count = len(horizons)
    daily_weights = torch.ones((y_daily_scaled.shape[1],), device=y_daily_scaled.device, dtype=y_daily_scaled.dtype)
    daily_weights[:3] = 1.15
    daily_weights[3:5] = 1.05
    daily_loss = F.huber_loss(prediction["mu"], y_daily_scaled, reduction="none")
    loss = float(weights["path_daily"]) * (daily_loss * daily_weights.reshape(1, -1)).mean()
    quantile_weight = float(weights["quantile"]) / 3.0
    loss = loss + quantile_weight * pinball_loss(prediction["q10"], y_daily_scaled, 0.10)
    loss = loss + quantile_weight * pinball_loss(prediction["q50"], y_daily_scaled, 0.50)
    loss = loss + quantile_weight * pinball_loss(prediction["q90"], y_daily_scaled, 0.90)
    loss = loss + float(weights["path_aux"]) * F.huber_loss(prediction["aux"][:, :cum_count], y_cum_scaled)
    risk_pred = prediction["aux"][:, cum_count : cum_count + cum_count * 3].reshape(-1, cum_count, 3)
    if y_risk_scaled.ndim == 2:
        y_risk_scaled = y_risk_scaled.reshape(y_risk_scaled.shape[0], 1, y_risk_scaled.shape[1]).expand(-1, cum_count, -1)
    loss = loss + float(weights["risk_aux"]) * F.huber_loss(risk_pred, y_risk_scaled)
    rank_loss_weights = _rank_loss_weights_for_profile(profile)
    rank_weight = float(weights.get("date_grouped_rank", 0.0) or 0.0)
    efficiency_rank_weight = float(weights.get("unit_time_alpha_rank", 0.0) or 0.0)
    for pos, horizon in enumerate(horizons):
        horizon_weight = float(rank_loss_weights.get(int(horizon), 0.0))
        if horizon_weight <= 0.0:
            continue
        score = prediction["mu"][:, : int(horizon)].sum(dim=1)
        target = y_cum_scaled[:, pos]
        if rank_weight > 0.0:
            loss = loss + rank_weight * horizon_weight * _date_grouped_rank_loss(
                score,
                target,
                date_group_ids,
                min_group_size=int(rank_min_group_size),
                max_pairs_per_date=int(rank_max_pairs_per_date),
            )
        if efficiency_rank_weight > 0.0:
            horizon_scale = float(max(int(horizon), 1))
            loss = loss + efficiency_rank_weight * horizon_weight * _date_grouped_rank_loss(
                score / horizon_scale,
                target / horizon_scale,
                date_group_ids,
                min_group_size=int(rank_min_group_size),
                max_pairs_per_date=int(rank_max_pairs_per_date),
            )
    downside_weight = float(weights.get("downside_rank_aux", 0.0) or 0.0)
    if downside_weight > 0.0:
        loss = loss + downside_weight * _date_grouped_rank_loss(
            -risk_pred[:, -1, 0],
            -y_risk_scaled[:, -1, 0],
            date_group_ids,
            min_group_size=int(rank_min_group_size),
            max_pairs_per_date=int(rank_max_pairs_per_date),
        )
    upside_weight = float(weights.get("upside_rank_aux", 0.0) or 0.0)
    if upside_weight > 0.0:
        loss = loss + upside_weight * _date_grouped_rank_loss(
            risk_pred[:, -1, 2],
            y_risk_scaled[:, -1, 2],
            date_group_ids,
            min_group_size=int(rank_min_group_size),
            max_pairs_per_date=int(rank_max_pairs_per_date),
        )
    router_entropy_weight = float(weights.get("router_entropy_floor", 0.0))
    if router_entropy_weight > 0.0 and "router_entropy" in prediction and "router_weights" in prediction:
        router_entropy = prediction["router_entropy"]
        expert_count = int(prediction["router_weights"].shape[-1])
        max_entropy = torch.log(torch.as_tensor(float(max(expert_count, 1)), device=router_entropy.device, dtype=router_entropy.dtype))
        entropy_floor = max_entropy * 0.60
        loss = loss + router_entropy_weight * torch.clamp(entropy_floor - router_entropy, min=0.0).square().mean()
    diversity_weight = float(weights.get("expert_diversity", 0.0))
    if diversity_weight > 0.0 and "expert_token_diversity" in prediction:
        diversity = prediction["expert_token_diversity"]
        loss = loss + diversity_weight * torch.clamp(0.05 - diversity, min=0.0).square().mean()
    return loss


def _autocast_context(device: torch.device, amp_enabled: bool) -> Any:
    if device.type == "cuda" and bool(amp_enabled):
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    value = str(device or "auto").strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("forecast training requested CUDA, but torch.cuda.is_available() is False.")
    if value not in {"cpu", "cuda"}:
        raise ValueError("--forecast-device must be auto, cpu, or cuda.")
    return torch.device(value)


def _unpack_forecast_batch(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if len(batch) == 6:
        batch_x, batch_y_daily, batch_y_cum, batch_y_risk, row_idx, static_context_ids = batch
        return batch_x, batch_y_daily, batch_y_cum, batch_y_risk, row_idx, static_context_ids
    if len(batch) == 5:
        batch_x, batch_y_daily, batch_y_cum, batch_y_risk, row_idx = batch
        return batch_x, batch_y_daily, batch_y_cum, batch_y_risk, row_idx, None
    raise ValueError(f"forecast batch must contain 5 or 6 tensors, got {len(batch)}.")


def _decision_utility_targets(
    y_cum_scaled: torch.Tensor,
    y_risk_scaled: torch.Tensor,
    *,
    target_scale: float,
    cost_bps: float,
    hit_threshold_bps: float,
    drawdown_penalty: float,
    cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    max_horizon: int | None = None,
    normalize_per_horizon: bool = False,
) -> dict[str, torch.Tensor]:
    scale = max(float(target_scale), 1.0e-8)
    y_cum = y_cum_scaled / scale
    resolved_horizons = normalize_path20_cumulative_horizons(
        cumulative_horizons,
        horizon=int(max_horizon or max(PATH20_HORIZON, y_cum_scaled.shape[1])),
    )
    risk = y_risk_scaled / scale
    if risk.ndim == 2:
        risk = risk.reshape(risk.shape[0], 1, risk.shape[1]).expand(-1, len(resolved_horizons), -1)
    max_drawdown = risk[:, :, 0]
    horizons = torch.as_tensor(resolved_horizons, dtype=y_cum.dtype, device=y_cum.device).reshape(1, -1)
    horizon_scale = torch.sqrt(horizons / float(max(int(max_horizon or int(max(resolved_horizons))), 1)))
    downside = torch.clamp(-max_drawdown, min=0.0)
    utility = y_cum - float(cost_bps) / 10000.0 - float(drawdown_penalty) * downside * horizon_scale
    if normalize_per_horizon:
        mean = utility.mean(dim=0, keepdim=True)
        std = torch.clamp(utility.std(dim=0, keepdim=True, unbiased=False), min=1.0e-6)
        selection_utility = (utility - mean) / std
    else:
        selection_utility = utility
    hit_label = utility > (float(hit_threshold_bps) / 10000.0)
    best_horizon_index = torch.argmax(selection_utility, dim=1)
    decision_score = selection_utility.max(dim=1).values if normalize_per_horizon else utility.max(dim=1).values
    return {
        "utility": utility,
        "selection_utility": selection_utility,
        "utility_scaled": selection_utility * scale if normalize_per_horizon else utility * scale,
        "hit_label": hit_label,
        "best_horizon_index": best_horizon_index,
        "decision_score": decision_score,
        "decision_score_scaled": decision_score * scale,
    }


def _decision_utility_targets_np(
    y_cum: np.ndarray,
    max_drawdown: np.ndarray,
    *,
    cost_bps: float,
    hit_threshold_bps: float,
    drawdown_penalty: float,
    cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    max_horizon: int | None = None,
) -> dict[str, np.ndarray]:
    y_cum_arr = np.asarray(y_cum, dtype=np.float64)
    resolved_horizons = normalize_path20_cumulative_horizons(
        cumulative_horizons,
        horizon=int(max_horizon or max(PATH20_HORIZON, y_cum_arr.shape[1])),
    )
    max_dd = np.asarray(max_drawdown, dtype=np.float64)
    if max_dd.ndim == 1:
        max_dd = max_dd.reshape(-1, 1)
    if max_dd.ndim == 2 and max_dd.shape[1] != len(resolved_horizons):
        max_dd = np.repeat(max_dd[:, :1], len(resolved_horizons), axis=1)
    elif max_dd.ndim == 3:
        max_dd = max_dd[:, :, 0]
    horizons = np.asarray(resolved_horizons, dtype=np.float64).reshape(1, -1)
    utility = (
        y_cum_arr
        - float(cost_bps) / 10000.0
        - float(drawdown_penalty) * np.maximum(0.0, -max_dd) * np.sqrt(horizons / float(max(int(max_horizon or int(max(resolved_horizons))), 1)))
    )
    return {
        "utility": utility,
        "hit_label": utility > (float(hit_threshold_bps) / 10000.0),
        "best_horizon_index": np.argmax(utility, axis=1),
        "decision_score": np.max(utility, axis=1),
    }


def _time_efficient_utility_targets(
    y_cum_scaled: torch.Tensor,
    y_risk_scaled: torch.Tensor,
    *,
    target_scale: float,
    cost_bps: float,
    hit_threshold_bps: float,
    drawdown_penalty: float,
    worst_day_penalty: float = _TIME_EFFICIENT_WORST_DAY_PENALTY,
    cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    max_horizon: int | None = None,
) -> dict[str, torch.Tensor]:
    scale = max(float(target_scale), 1.0e-8)
    y_cum = y_cum_scaled / scale
    resolved_horizons = normalize_path20_cumulative_horizons(
        cumulative_horizons,
        horizon=int(max_horizon or max(PATH20_HORIZON, y_cum_scaled.shape[1])),
    )
    risk = y_risk_scaled / scale
    if risk.ndim == 2:
        risk = risk.reshape(risk.shape[0], 1, risk.shape[1]).expand(-1, len(resolved_horizons), -1)
    max_drawdown = risk[:, :, 0]
    worst_day = risk[:, :, 1] if risk.shape[-1] > 1 else max_drawdown
    horizon_values = torch.as_tensor(resolved_horizons, dtype=y_cum.dtype, device=y_cum.device).reshape(1, -1)
    max_h = float(max(int(max_horizon or int(max(resolved_horizons))), 1))
    horizon_risk_scale = torch.sqrt(horizon_values / max_h)
    downside = torch.clamp(-max_drawdown, min=0.0)
    worst_downside = torch.clamp(-worst_day, min=0.0)
    net_utility = (
        y_cum
        - float(cost_bps) / 10000.0
        - float(drawdown_penalty) * downside * horizon_risk_scale
        - float(worst_day_penalty) * worst_downside * horizon_risk_scale
    )
    efficient_utility = net_utility / horizon_values
    hit_threshold = (float(hit_threshold_bps) / 10000.0) / horizon_values
    hit_label = efficient_utility > hit_threshold
    best_horizon_index = torch.argmax(efficient_utility, dim=1)
    decision_score = efficient_utility.max(dim=1).values
    return {
        "net_utility": net_utility,
        "efficient_utility": efficient_utility,
        "utility": efficient_utility,
        "utility_scaled": efficient_utility * scale,
        "hit_label": hit_label,
        "hit_threshold": hit_threshold,
        "best_horizon_index": best_horizon_index,
        "decision_score": decision_score,
        "decision_score_scaled": decision_score * scale,
    }


def _time_efficient_utility_targets_np(
    y_cum: np.ndarray,
    risk_by_horizon: np.ndarray,
    *,
    cost_bps: float,
    hit_threshold_bps: float,
    drawdown_penalty: float,
    worst_day_penalty: float = _TIME_EFFICIENT_WORST_DAY_PENALTY,
    cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    max_horizon: int | None = None,
) -> dict[str, np.ndarray]:
    y_cum_arr = np.asarray(y_cum, dtype=np.float64)
    resolved_horizons = normalize_path20_cumulative_horizons(
        cumulative_horizons,
        horizon=int(max_horizon or max(PATH20_HORIZON, y_cum_arr.shape[1])),
    )
    risk = np.asarray(risk_by_horizon, dtype=np.float64)
    if risk.ndim == 1:
        risk = risk.reshape(-1, 1, 1)
    if risk.ndim == 2:
        if risk.shape[1] == len(resolved_horizons):
            risk = np.stack([risk, risk, np.zeros_like(risk)], axis=-1)
        else:
            risk = np.repeat(risk[:, :1, None], len(resolved_horizons), axis=1)
    if risk.shape[1] != len(resolved_horizons):
        risk = np.repeat(risk[:, :1, :], len(resolved_horizons), axis=1)
    max_drawdown = risk[:, :, 0]
    worst_day = risk[:, :, 1] if risk.shape[-1] > 1 else max_drawdown
    horizon_values = np.asarray(resolved_horizons, dtype=np.float64).reshape(1, -1)
    horizon_risk_scale = np.sqrt(horizon_values / float(max(int(max_horizon or int(max(resolved_horizons))), 1)))
    net_utility = (
        y_cum_arr
        - float(cost_bps) / 10000.0
        - float(drawdown_penalty) * np.maximum(0.0, -max_drawdown) * horizon_risk_scale
        - float(worst_day_penalty) * np.maximum(0.0, -worst_day) * horizon_risk_scale
    )
    efficient_utility = net_utility / horizon_values
    hit_threshold = (float(hit_threshold_bps) / 10000.0) / horizon_values
    return {
        "net_utility": net_utility,
        "efficient_utility": efficient_utility,
        "utility": efficient_utility,
        "hit_label": efficient_utility > hit_threshold,
        "hit_threshold": hit_threshold,
        "best_horizon_index": np.argmax(efficient_utility, axis=1),
        "decision_score": np.max(efficient_utility, axis=1),
    }


def _standardize_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.numel() <= 1:
        return value * 0.0
    mean = value.mean()
    std = torch.clamp(value.std(unbiased=False), min=1.0e-6)
    return (value - mean) / std


def _high_return_proxy_loss(
    *,
    pred_decision_score: torch.Tensor,
    y_cum_scaled: torch.Tensor,
    y_risk_scaled: torch.Tensor,
    weights: dict[str, float],
    target_scale: float,
    cumulative_horizons: tuple[int, ...],
) -> torch.Tensor:
    if pred_decision_score.numel() < 2:
        return pred_decision_score.new_tensor(0.0)
    horizon_values = tuple(int(item) for item in cumulative_horizons)
    target_pos = horizon_values.index(20) if 20 in horizon_values else len(horizon_values) - 1
    scale = max(float(target_scale), 1.0e-8)
    score = _standardize_tensor(pred_decision_score)
    target_return = y_cum_scaled[:, target_pos] / scale
    target_rank = _standardize_tensor(target_return.detach())
    total = pred_decision_score.new_tensor(0.0)

    topn_weight = float(weights.get("topn_excess_rank", 0.0) or 0.0)
    if topn_weight > 0.0:
        k = max(int(round(float(pred_decision_score.numel()) * 0.20)), 1)
        threshold = torch.topk(target_return.detach(), k=k).values.min()
        top_label = (target_return.detach() >= threshold).to(dtype=score.dtype)
        topn_loss = F.binary_cross_entropy_with_logits(score, top_label)
        total = total + topn_weight * (topn_loss + 0.10 * pairwise_rank_loss(score, target_rank))

    topk_alignment_weight = float(weights.get("decision_score_topk_alignment", 0.0) or 0.0)
    if topk_alignment_weight > 0.0:
        components: list[torch.Tensor] = []
        for requested_k in (1, 3, 5):
            if pred_decision_score.numel() < int(requested_k):
                continue
            threshold = torch.topk(target_return.detach(), k=int(requested_k)).values.min()
            top_label = (target_return.detach() >= threshold).to(dtype=score.dtype)
            components.append(F.binary_cross_entropy_with_logits(score, top_label))
        if components:
            topk_loss = torch.stack(components).mean()
            total = total + topk_alignment_weight * (topk_loss + 0.15 * pairwise_rank_loss(score, target_rank))

    proxy_weight = float(weights.get("score_to_weight_proxy", 0.0) or 0.0)
    if proxy_weight > 0.0:
        score_weights = F.softmax(score / 0.50, dim=0)
        target_weights = F.softmax(target_rank / 0.50, dim=0).detach()
        alignment = F.kl_div(torch.log(torch.clamp(score_weights, min=1.0e-8)), target_weights, reduction="sum")
        expected_return = torch.sum(score_weights * target_return.detach())
        concentration = torch.sum(score_weights.square())
        total = total + proxy_weight * (alignment - expected_return + 0.02 * concentration)

    bad_weight = float(weights.get("bad_month_aware", 0.0) or 0.0)
    if bad_weight > 0.0:
        risk = y_risk_scaled / scale
        if risk.ndim == 2:
            risk = risk.reshape(risk.shape[0], 1, risk.shape[1]).expand(-1, len(horizon_values), -1)
        local_drawdown = torch.clamp(-risk[:, target_pos, 0], min=0.0)
        local_worst = torch.clamp(-risk[:, target_pos, 1], min=0.0)
        negative_return = torch.clamp(-target_return, min=0.0)
        bad_intensity = (negative_return + 0.50 * local_drawdown + 0.25 * local_worst).detach()
        bad_intensity = _standardize_tensor(bad_intensity)
        bad_loss = (torch.sigmoid(score) * torch.clamp(bad_intensity, min=0.0)).mean()
        total = total + bad_weight * (bad_loss + 0.10 * pairwise_rank_loss(-score, bad_intensity))

    return total


def _time_efficient_topk_alignment_loss(
    *,
    pred_score: torch.Tensor,
    future_score: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    if float(weight) <= 0.0 or pred_score.numel() < 2:
        return pred_score.new_tensor(0.0)
    score = _standardize_tensor(pred_score)
    target = future_score.detach()
    components: list[torch.Tensor] = []
    for requested_k in (1, 3, 5):
        k = min(int(requested_k), int(target.numel()))
        if k <= 0:
            continue
        threshold = torch.topk(target, k=k).values.min()
        top_label = (target >= threshold).to(dtype=score.dtype)
        components.append(F.binary_cross_entropy_with_logits(score, top_label))
    if not components:
        return pred_score.new_tensor(0.0)
    target_rank = _standardize_tensor(target)
    return float(weight) * (torch.stack(components).mean() + 0.15 * pairwise_rank_loss(score, target_rank))


def _collate_forecast_date_batches(batch: list[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    max_stocks = max(int(item[0].shape[0]) for item in batch) if batch else 0
    if max_stocks <= 0:
        raise ValueError("date-level forecast batch is empty.")
    x_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    y_daily_rows: list[torch.Tensor] = []
    y_cum_rows: list[torch.Tensor] = []
    y_risk_rows: list[torch.Tensor] = []
    row_index_rows: list[torch.Tensor] = []
    static_rows: list[torch.Tensor] = []
    has_static = len(batch[0]) == 7
    for item in batch:
        x, mask, y_daily, y_cum, y_risk, row_indices, *maybe_static = item
        pad = max_stocks - int(x.shape[0])
        x_rows.append(F.pad(x, (0, 0, 0, 0, 0, pad)))
        mask_rows.append(F.pad(mask.to(dtype=torch.bool), (0, pad), value=False))
        y_daily_rows.append(F.pad(y_daily, (0, 0, 0, pad)))
        y_cum_rows.append(F.pad(y_cum, (0, 0, 0, pad)))
        y_risk_rows.append(F.pad(y_risk, (0, 0, 0, 0, 0, pad)))
        row_index_rows.append(F.pad(row_indices.to(dtype=torch.long), (0, pad), value=-1))
        if has_static:
            static_rows.append(F.pad(maybe_static[0].to(dtype=torch.long), (0, 0, 0, pad), value=0))
    static_context = torch.stack(static_rows, dim=0) if has_static else None
    return (
        torch.stack(x_rows, dim=0),
        torch.stack(mask_rows, dim=0),
        torch.stack(y_daily_rows, dim=0),
        torch.stack(y_cum_rows, dim=0),
        torch.stack(y_risk_rows, dim=0),
        torch.stack(row_index_rows, dim=0),
        static_context,
    )


def _forecast_model_accepts_static_context(model: nn.Module) -> bool:
    try:
        return "static_context_ids" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _forecast_model_forward(
    model: nn.Module,
    batch_x: torch.Tensor,
    static_context_ids: torch.Tensor | None = None,
    stock_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    accepts_static = _forecast_model_accepts_static_context(model)
    accepts_stock_mask = _forecast_model_accepts_stock_mask(model)
    if static_context_ids is not None and stock_mask is not None and accepts_static and accepts_stock_mask:
        return model(batch_x, static_context_ids=static_context_ids, stock_mask=stock_mask)
    if static_context_ids is not None and accepts_static:
        return model(batch_x, static_context_ids=static_context_ids)
    if stock_mask is not None and accepts_stock_mask:
        return model(batch_x, stock_mask=stock_mask)
    return model(batch_x)


def _forecast_model_accepts_stock_mask(model: nn.Module) -> bool:
    try:
        return "stock_mask" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _predict_all(
    model: nn.Module,
    x: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device | None = None,
    amp_enabled: bool = False,
) -> dict[str, np.ndarray]:
    model.eval()
    resolved_device = device or next(model.parameters()).device
    chunks: dict[str, list[np.ndarray]] = {"mu": [], "q10": [], "q50": [], "q90": [], "aux": []}
    with torch.no_grad():
        for start in range(0, x.shape[0], max(int(batch_size), 1)):
            batch = x[start : start + max(int(batch_size), 1)].to(resolved_device, non_blocking=resolved_device.type == "cuda")
            with _autocast_context(resolved_device, amp_enabled):
                pred = _forecast_model_forward(model, batch)
            if "decision_aux" in pred and "decision_aux" not in chunks:
                chunks["decision_aux"] = []
            for key in chunks:
                if key in pred:
                    chunks[key].append(pred[key].detach().cpu().numpy())
    horizon = int(getattr(model, "horizon", PATH20_HORIZON) or PATH20_HORIZON)
    horizons = normalize_path20_cumulative_horizons(getattr(model, "cumulative_horizons", None), horizon=horizon)
    empty_shapes = {
        "mu": (0, horizon),
        "q10": (0, horizon),
        "q50": (0, horizon),
        "q90": (0, horizon),
        "aux": (0, path20_forecast_aux_dim(horizons, horizon=horizon)),
        "decision_aux": (0, path20_decision_aux_dim(horizons, horizon=horizon)),
    }
    return {key: np.concatenate(values, axis=0) if values else np.empty(empty_shapes[key]) for key, values in chunks.items()}


def _predict_indices(
    model: nn.Module,
    dataset_view_or_x: _ForecastDatasetView | torch.Tensor,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    target_scale: float = 100.0,
) -> dict[str, np.ndarray]:
    if len(indices) == 0:
        horizon = int(getattr(model, "horizon", PATH20_HORIZON) or PATH20_HORIZON)
        horizons = normalize_path20_cumulative_horizons(getattr(model, "cumulative_horizons", None), horizon=horizon)
        return {
            "mu": np.empty((0, horizon)),
            "q10": np.empty((0, horizon)),
            "q50": np.empty((0, horizon)),
            "q90": np.empty((0, horizon)),
            "aux": np.empty((0, path20_forecast_aux_dim(horizons, horizon=horizon))),
        }
    if isinstance(dataset_view_or_x, _ForecastDatasetView):
        horizon = int(dataset_view_or_x.horizon)
        horizons = dataset_view_or_x.cumulative_horizons
        aux_dim = path20_forecast_aux_dim(horizons, horizon=horizon)
        decision_aux_dim = path20_decision_aux_dim(horizons, horizon=horizon)
        model.eval()
        if _forecast_model_accepts_stock_mask(model) and dataset_view_or_x.dataset_mode == "memmap":
            ordered_indices = np.asarray(indices, dtype=np.int64)
            index_pos = {int(row_idx): pos for pos, row_idx in enumerate(ordered_indices.tolist())}
            outputs = {
                "mu": np.empty((len(ordered_indices), horizon), dtype=np.float32),
                "q10": np.empty((len(ordered_indices), horizon), dtype=np.float32),
                "q50": np.empty((len(ordered_indices), horizon), dtype=np.float32),
                "q90": np.empty((len(ordered_indices), horizon), dtype=np.float32),
                "aux": np.empty((len(ordered_indices), aux_dim), dtype=np.float32),
            }
            if str(getattr(model, "output_profile", "") or "") == "decision_utility_v1":
                outputs["decision_aux"] = np.empty((len(ordered_indices), decision_aux_dim), dtype=np.float32)
            loader = DataLoader(
                dataset_view_or_x.date_batch_torch_dataset(ordered_indices, target_scale=target_scale),
                batch_size=1,
                shuffle=False,
                pin_memory=device.type == "cuda",
                collate_fn=_collate_forecast_date_batches,
            )
            with torch.no_grad():
                for batch_x_cpu, stock_mask_cpu, _, _, _, row_indices_cpu, static_context_ids_cpu in loader:
                    batch_x = batch_x_cpu.to(device, non_blocking=device.type == "cuda")
                    stock_mask = stock_mask_cpu.to(device, non_blocking=device.type == "cuda")
                    static_context_ids = (
                        static_context_ids_cpu.to(device, non_blocking=device.type == "cuda")
                        if static_context_ids_cpu is not None
                        else None
                    )
                    static_context_ids = dataset_view_or_x.filter_static_context_ids(static_context_ids)
                    with _autocast_context(device, amp_enabled):
                        pred = _forecast_model_forward(model, batch_x, static_context_ids, stock_mask=stock_mask)
                    if "decision_aux" in pred and "decision_aux" not in outputs:
                        outputs["decision_aux"] = np.empty((len(ordered_indices), decision_aux_dim), dtype=np.float32)
                    valid = stock_mask.reshape(-1).detach().cpu().numpy().astype(bool)
                    row_ids = row_indices_cpu.reshape(-1).detach().cpu().numpy().astype(int)[valid]
                    for key in outputs:
                        values = pred[key][torch.as_tensor(valid, dtype=torch.bool, device=pred[key].device)].detach().cpu().numpy()
                        for row_id, value in zip(row_ids.tolist(), values, strict=False):
                            outputs[key][index_pos[int(row_id)]] = value
            return outputs
        if isinstance(dataset_view_or_x.dataset, (ForecastShardedMemmapDataset, ForecastTrainingPackDataset)):
            original_indices = np.asarray(indices, dtype=np.int64)
            ordered_indices = dataset_view_or_x.dataset.cache_friendly_indices(original_indices)
            index_pos = {int(row_idx): pos for pos, row_idx in enumerate(original_indices.tolist())}
            outputs = {
                "mu": np.empty((len(original_indices), horizon), dtype=np.float32),
                "q10": np.empty((len(original_indices), horizon), dtype=np.float32),
                "q50": np.empty((len(original_indices), horizon), dtype=np.float32),
                "q90": np.empty((len(original_indices), horizon), dtype=np.float32),
                "aux": np.empty((len(original_indices), aux_dim), dtype=np.float32),
            }
            if str(getattr(model, "output_profile", "") or "") == "decision_utility_v1":
                outputs["decision_aux"] = np.empty((len(original_indices), decision_aux_dim), dtype=np.float32)
            loader = DataLoader(
                dataset_view_or_x.batch_torch_dataset(
                    ordered_indices,
                    batch_size=max(int(batch_size), 1),
                    target_scale=target_scale,
                ),
                batch_size=None,
                shuffle=False,
                pin_memory=device.type == "cuda",
            )
            with torch.no_grad():
                for raw_batch in loader:
                    batch_x, _, _, _, row_indices_cpu, static_context_ids_cpu = _unpack_forecast_batch(raw_batch)
                    batch = batch_x.to(device, non_blocking=device.type == "cuda")
                    static_context_ids = (
                        static_context_ids_cpu.to(device, non_blocking=device.type == "cuda")
                        if static_context_ids_cpu is not None
                        else None
                    )
                    static_context_ids = dataset_view_or_x.filter_static_context_ids(static_context_ids)
                    with _autocast_context(device, amp_enabled):
                        pred = _forecast_model_forward(model, batch, static_context_ids)
                    if "decision_aux" in pred and "decision_aux" not in outputs:
                        outputs["decision_aux"] = np.empty((len(original_indices), decision_aux_dim), dtype=np.float32)
                    row_ids = row_indices_cpu.reshape(-1).detach().cpu().numpy().astype(int)
                    for key in outputs:
                        if key not in pred:
                            continue
                        values = pred[key].detach().cpu().numpy()
                        for row_id, value in zip(row_ids.tolist(), values, strict=False):
                            outputs[key][index_pos[int(row_id)]] = value
            return outputs
        chunks: dict[str, list[np.ndarray]] = {"mu": [], "q10": [], "q50": [], "q90": [], "aux": []}
        if isinstance(dataset_view_or_x.dataset, (ForecastShardedMemmapDataset, ForecastTrainingPackDataset)):
            loader = DataLoader(
                dataset_view_or_x.batch_torch_dataset(
                    indices,
                    batch_size=max(int(batch_size), 1),
                    target_scale=target_scale,
                ),
                batch_size=None,
                shuffle=False,
                pin_memory=device.type == "cuda",
            )
        else:
            loader = DataLoader(
                dataset_view_or_x.torch_dataset(indices, target_scale=target_scale),
                batch_size=max(int(batch_size), 1),
                shuffle=False,
                pin_memory=device.type == "cuda",
            )
        with torch.no_grad():
            for raw_batch in loader:
                batch_x, _, _, _, _, static_context_ids_cpu = _unpack_forecast_batch(raw_batch)
                batch = batch_x.to(device, non_blocking=device.type == "cuda")
                static_context_ids = (
                    static_context_ids_cpu.to(device, non_blocking=device.type == "cuda")
                    if static_context_ids_cpu is not None
                    else None
                )
                static_context_ids = dataset_view_or_x.filter_static_context_ids(static_context_ids)
                with _autocast_context(device, amp_enabled):
                    pred = _forecast_model_forward(model, batch, static_context_ids)
                if "decision_aux" in pred and "decision_aux" not in chunks:
                    chunks["decision_aux"] = []
                for key in chunks:
                    if key in pred:
                        chunks[key].append(pred[key].detach().cpu().numpy())
        empty_shapes = {
            "mu": (0, horizon),
            "q10": (0, horizon),
            "q50": (0, horizon),
            "q90": (0, horizon),
            "aux": (0, aux_dim),
            "decision_aux": (0, decision_aux_dim),
        }
        return {key: np.concatenate(values, axis=0) if values else np.empty(empty_shapes[key]) for key, values in chunks.items()}
    x = dataset_view_or_x
    return _predict_all(
        model,
        x[torch.tensor(indices, dtype=torch.long)],
        batch_size=batch_size,
        device=device,
        amp_enabled=amp_enabled,
    )


def _rank_ic_by_date(frame: pd.DataFrame, score_column: str, target_column: str) -> float:
    values: list[float] = []
    for _, group in frame.groupby("date", sort=True):
        if len(group) < 2:
            continue
        score = pd.to_numeric(group[score_column], errors="coerce").astype("float64")
        target = pd.to_numeric(group[target_column], errors="coerce").astype("float64")
        valid = score.notna() & target.notna()
        if int(valid.sum()) < 2:
            continue
        corr = score.loc[valid].corr(target.loc[valid], method="spearman")
        if pd.notna(corr):
            values.append(float(corr))
    return float(np.mean(values)) if values else 0.0


def _top_bottom_spread_by_date(frame: pd.DataFrame, score_column: str, target_column: str, frac: float = 0.20) -> float:
    values: list[float] = []
    for _, group in frame.groupby("date", sort=True):
        work = group[[score_column, target_column]].copy()
        work[score_column] = pd.to_numeric(work[score_column], errors="coerce").astype("float64")
        work[target_column] = pd.to_numeric(work[target_column], errors="coerce").astype("float64")
        work = work.dropna()
        if len(work) < 2:
            continue
        k = max(int(len(work) * float(frac)), 1)
        top = work.nlargest(k, score_column)[target_column].mean()
        bottom = work.nsmallest(k, score_column)[target_column].mean()
        if pd.notna(top) and pd.notna(bottom):
            values.append(float(top - bottom))
    return float(np.mean(values)) if values else 0.0


def _calibrate_pred_decision_utility_np(
    pred_utility: np.ndarray,
    *,
    horizons: tuple[int, ...],
    loss_profile: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    profile = _normalize_forecast_loss_profile(loss_profile)
    calibrated = np.asarray(pred_utility, dtype=np.float64).copy()
    contract = _decision_score_calibration_contract(profile)
    if not bool(contract.get("enabled", False)):
        return calibrated, contract
    horizon_values = np.asarray(horizons, dtype=int)
    penalty_horizon = int(contract.get("penalized_horizon") or max(horizons))
    mask = horizon_values == penalty_horizon
    if mask.any():
        calibrated[:, mask] -= float(contract.get("utility_penalty", 0.0) or 0.0)
    return calibrated, contract


def _add_decision_utility_columns(
    columns: dict[str, Any],
    *,
    predictions: dict[str, np.ndarray],
    y_cum: np.ndarray,
    drawdown_by_horizon: np.ndarray,
    worst_by_horizon: np.ndarray | None = None,
    target_scale: float,
    decision_cost_bps: float,
    decision_hit_threshold_bps: float,
    decision_drawdown_penalty: float,
    cumulative_horizons: tuple[int, ...] | list[int] | str | None,
    max_horizon: int,
    loss_profile: str | None = None,
) -> None:
    if "decision_aux" not in predictions:
        return
    decision_aux = np.asarray(predictions["decision_aux"], dtype=np.float64)
    if decision_aux.size == 0:
        return
    profile = _normalize_forecast_loss_profile(loss_profile)
    horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=int(max_horizon))
    cum_count = len(horizons)
    raw_pred_utility = decision_aux[:, :cum_count] / max(float(target_scale), 1.0e-8)
    hit_logits = decision_aux[:, cum_count : cum_count * 2]
    horizon_logits = decision_aux[:, cum_count * 2 : cum_count * 3]
    pred_hit_prob = 1.0 / (1.0 + np.exp(-np.clip(hit_logits, -60.0, 60.0)))
    horizon_values = np.asarray(horizons, dtype=int)
    if profile in _TIME_EFFICIENT_TOPK_LOSS_PROFILES:
        risk_parts = [
            np.asarray(drawdown_by_horizon, dtype=np.float64),
            np.asarray(worst_by_horizon if worst_by_horizon is not None else drawdown_by_horizon, dtype=np.float64),
            np.zeros_like(np.asarray(drawdown_by_horizon, dtype=np.float64)),
        ]
        future = _time_efficient_utility_targets_np(
            y_cum,
            np.stack(risk_parts, axis=-1),
            cost_bps=float(decision_cost_bps),
            hit_threshold_bps=float(decision_hit_threshold_bps),
            drawdown_penalty=float(decision_drawdown_penalty),
            worst_day_penalty=float(_TIME_EFFICIENT_WORST_DAY_PENALTY),
            cumulative_horizons=horizons,
            max_horizon=max_horizon,
        )
        pred_best_idx = np.argmax(raw_pred_utility, axis=1)
        pred_score = np.max(raw_pred_utility, axis=1)
        for pos, horizon in enumerate(horizons):
            columns[f"pred_time_eff_utility_{horizon}d"] = raw_pred_utility[:, pos]
            columns[f"future_time_eff_utility_{horizon}d"] = future["efficient_utility"][:, pos]
            columns[f"pred_time_eff_hit_prob_{horizon}d"] = pred_hit_prob[:, pos]
            columns[f"future_time_eff_hit_label_{horizon}d"] = future["hit_label"][:, pos].astype(int)
            columns[f"future_time_eff_net_utility_{horizon}d"] = future["net_utility"][:, pos]
            columns[f"pred_decision_utility_{horizon}d"] = raw_pred_utility[:, pos]
            columns[f"future_decision_utility_{horizon}d"] = future["efficient_utility"][:, pos]
            columns[f"pred_hit_prob_{horizon}d"] = pred_hit_prob[:, pos]
            columns[f"future_hit_label_{horizon}d"] = future["hit_label"][:, pos].astype(int)
        columns["pred_time_eff_best_horizon"] = horizon_values[pred_best_idx]
        columns["future_time_eff_best_horizon"] = horizon_values[future["best_horizon_index"].astype(int)]
        columns["pred_time_eff_score"] = pred_score
        columns["future_time_eff_score"] = future["decision_score"]
        columns["pred_best_horizon"] = columns["pred_time_eff_best_horizon"]
        columns["future_best_horizon"] = columns["future_time_eff_best_horizon"]
        columns["pred_decision_score"] = pred_score
        columns["trade_utility_score"] = pred_score
        columns["future_decision_score"] = future["decision_score"]
        columns["pred_best_horizon_source"] = "time_eff_utility_argmax"
        columns["decision_score_calibration_method"] = "time_efficient_per_day"
        columns["decision_score_calibration_penalty"] = 0.0
        return

    calibrated_pred_utility, calibration_contract = _calibrate_pred_decision_utility_np(
        raw_pred_utility,
        horizons=horizons,
        loss_profile=profile,
    )
    future = _decision_utility_targets_np(
        y_cum,
        drawdown_by_horizon,
        cost_bps=float(decision_cost_bps),
        hit_threshold_bps=float(decision_hit_threshold_bps),
        drawdown_penalty=float(decision_drawdown_penalty),
        cumulative_horizons=horizons,
        max_horizon=max_horizon,
    )
    if bool(calibration_contract.get("enabled", False)):
        pred_best_idx = np.argmax(calibrated_pred_utility, axis=1)
        pred_best_horizon_source = "calibrated_utility_argmax"
    else:
        pred_best_idx = np.argmax(horizon_logits, axis=1)
        pred_best_horizon_source = "horizon_head_logits"
    pred_score = np.max(calibrated_pred_utility, axis=1)
    for pos, horizon in enumerate(horizons):
        columns[f"pred_decision_utility_{horizon}d"] = raw_pred_utility[:, pos]
        if bool(calibration_contract.get("enabled", False)):
            columns[f"calibrated_pred_decision_utility_{horizon}d"] = calibrated_pred_utility[:, pos]
        columns[f"future_decision_utility_{horizon}d"] = future["utility"][:, pos]
        columns[f"pred_hit_prob_{horizon}d"] = pred_hit_prob[:, pos]
        columns[f"future_hit_label_{horizon}d"] = future["hit_label"][:, pos].astype(int)
    columns["pred_best_horizon"] = horizon_values[pred_best_idx]
    columns["future_best_horizon"] = horizon_values[future["best_horizon_index"].astype(int)]
    columns["pred_decision_score"] = pred_score
    columns["trade_utility_score"] = pred_score
    columns["future_decision_score"] = future["decision_score"]
    columns["pred_best_horizon_source"] = pred_best_horizon_source
    columns["decision_score_calibration_method"] = str(calibration_contract.get("method", "none"))
    columns["decision_score_calibration_penalty"] = float(calibration_contract.get("utility_penalty", 0.0) or 0.0)


def _prediction_frame(
    dataset: ForecastSequenceDataset,
    *,
    role: str,
    predictions: dict[str, np.ndarray],
    family: str,
    target_scale: float,
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
    loss_profile: str | None = None,
) -> pd.DataFrame:
    mask = dataset.role == role
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return pd.DataFrame()
    scale = float(target_scale)
    mu = predictions["mu"][idx] / scale
    q10 = predictions["q10"][idx] / scale
    q50 = predictions["q50"][idx] / scale
    q90 = predictions["q90"][idx] / scale
    aux = predictions["aux"][idx] / scale
    y_daily = dataset.y_daily_excess[idx]
    y_cum = dataset.y_cum_excess[idx]
    horizon = int(dataset.manifest.get("horizon", y_daily.shape[1]) or y_daily.shape[1])
    horizons = normalize_path20_cumulative_horizons(dataset.manifest.get("cumulative_horizons", PATH20_CUMULATIVE_HORIZONS), horizon=horizon)
    columns: dict[str, Any] = {
        "date": [pd.Timestamp(item).strftime("%Y-%m-%d") for item in dataset.date[idx].tolist()],
        "stock": dataset.stock[idx].astype(str),
        "role": dataset.role[idx].astype(str),
        "model_family": str(family),
        "future_rank_20d": dataset.y_rank_20d[idx],
        "future_path_max_drawdown_20d": dataset.y_max_drawdown_20d[idx],
        "future_path_worst_1d_20d": dataset.y_worst_1d_20d[idx],
        "future_path_upside_capture_20d": dataset.y_upside_20d[idx],
    }
    for pos, item_horizon in enumerate(horizons):
        columns[f"future_rank_{item_horizon}d"] = dataset.y_rank_by_horizon[idx, pos]
    for pos, item_horizon in enumerate(horizons):
        columns[f"future_cum_excess_return_{item_horizon}d"] = y_cum[:, pos]
        columns[f"future_path_max_drawdown_{item_horizon}d"] = dataset.y_drawdown_by_horizon[idx, pos]
        columns[f"future_path_worst_1d_{item_horizon}d"] = dataset.y_worst_by_horizon[idx, pos]
        columns[f"future_path_upside_capture_{item_horizon}d"] = dataset.y_upside_by_horizon[idx, pos]
        columns[f"pred_cum_mu_{item_horizon}d"] = mu[:, :item_horizon].sum(axis=1)
        columns[f"pred_aux_cum_{item_horizon}d"] = aux[:, pos]
    _add_decision_utility_columns(
        columns,
        predictions=predictions,
        y_cum=y_cum,
        drawdown_by_horizon=dataset.y_drawdown_by_horizon[idx],
        worst_by_horizon=dataset.y_worst_by_horizon[idx],
        target_scale=target_scale,
        decision_cost_bps=decision_cost_bps,
        decision_hit_threshold_bps=decision_hit_threshold_bps,
        decision_drawdown_penalty=decision_drawdown_penalty,
        cumulative_horizons=horizons,
        max_horizon=horizon,
        loss_profile=loss_profile,
    )
    risk_start = len(horizons)
    risk_aux = aux[:, risk_start : risk_start + len(horizons) * 3].reshape(-1, len(horizons), 3)
    for pos, item_horizon in enumerate(horizons):
        columns[f"pred_aux_downside_floor_{item_horizon}d"] = risk_aux[:, pos, 0]
        columns[f"pred_aux_worst_1d_{item_horizon}d"] = risk_aux[:, pos, 1]
        columns[f"pred_aux_upside_{item_horizon}d"] = risk_aux[:, pos, 2]
    for step in range(1, horizon + 1):
        offset = step - 1
        columns[f"future_excess_return_{step}d"] = y_daily[:, offset]
        columns[f"target_excess_{step}d"] = y_daily[:, offset]
        columns[f"pred_mu_{step}d"] = mu[:, offset]
        columns[f"pred_q10_{step}d"] = q10[:, offset]
        columns[f"pred_q50_{step}d"] = q50[:, offset]
        columns[f"pred_q90_{step}d"] = q90[:, offset]
    return pd.DataFrame(columns)


def _prediction_frame_for_indices(
    dataset: ForecastSequenceDataset,
    *,
    indices: np.ndarray,
    predictions: dict[str, np.ndarray],
    family: str,
    target_scale: float,
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
    loss_profile: str | None = None,
) -> pd.DataFrame:
    if len(indices) == 0:
        return pd.DataFrame()
    scale = float(target_scale)
    idx = np.asarray(indices, dtype=int)
    mu = predictions["mu"] / scale
    q10 = predictions["q10"] / scale
    q50 = predictions["q50"] / scale
    q90 = predictions["q90"] / scale
    aux = predictions["aux"] / scale
    y_daily = dataset.y_daily_excess[idx]
    y_cum = dataset.y_cum_excess[idx]
    horizon = int(dataset.manifest.get("horizon", y_daily.shape[1]) or y_daily.shape[1])
    horizons = normalize_path20_cumulative_horizons(dataset.manifest.get("cumulative_horizons", PATH20_CUMULATIVE_HORIZONS), horizon=horizon)
    columns: dict[str, Any] = {
        "date": [pd.Timestamp(item).strftime("%Y-%m-%d") for item in dataset.date[idx].tolist()],
        "stock": dataset.stock[idx].astype(str),
        "role": dataset.role[idx].astype(str),
        "model_family": str(family),
        "future_rank_20d": dataset.y_rank_20d[idx],
        "future_path_max_drawdown_20d": dataset.y_max_drawdown_20d[idx],
        "future_path_worst_1d_20d": dataset.y_worst_1d_20d[idx],
        "future_path_upside_capture_20d": dataset.y_upside_20d[idx],
    }
    for pos, item_horizon in enumerate(horizons):
        columns[f"future_rank_{item_horizon}d"] = dataset.y_rank_by_horizon[idx, pos]
    for pos, item_horizon in enumerate(horizons):
        columns[f"future_cum_excess_return_{item_horizon}d"] = y_cum[:, pos]
        columns[f"future_path_max_drawdown_{item_horizon}d"] = dataset.y_drawdown_by_horizon[idx, pos]
        columns[f"future_path_worst_1d_{item_horizon}d"] = dataset.y_worst_by_horizon[idx, pos]
        columns[f"future_path_upside_capture_{item_horizon}d"] = dataset.y_upside_by_horizon[idx, pos]
        columns[f"pred_cum_mu_{item_horizon}d"] = mu[:, :item_horizon].sum(axis=1)
        columns[f"pred_aux_cum_{item_horizon}d"] = aux[:, pos]
    _add_decision_utility_columns(
        columns,
        predictions=predictions,
        y_cum=y_cum,
        drawdown_by_horizon=dataset.y_drawdown_by_horizon[idx],
        worst_by_horizon=dataset.y_worst_by_horizon[idx],
        target_scale=target_scale,
        decision_cost_bps=decision_cost_bps,
        decision_hit_threshold_bps=decision_hit_threshold_bps,
        decision_drawdown_penalty=decision_drawdown_penalty,
        cumulative_horizons=horizons,
        max_horizon=horizon,
        loss_profile=loss_profile,
    )
    risk_start = len(horizons)
    risk_aux = aux[:, risk_start : risk_start + len(horizons) * 3].reshape(-1, len(horizons), 3)
    for pos, item_horizon in enumerate(horizons):
        columns[f"pred_aux_downside_floor_{item_horizon}d"] = risk_aux[:, pos, 0]
        columns[f"pred_aux_worst_1d_{item_horizon}d"] = risk_aux[:, pos, 1]
        columns[f"pred_aux_upside_{item_horizon}d"] = risk_aux[:, pos, 2]
    for step in range(1, horizon + 1):
        offset = step - 1
        columns[f"future_excess_return_{step}d"] = y_daily[:, offset]
        columns[f"target_excess_{step}d"] = y_daily[:, offset]
        columns[f"pred_mu_{step}d"] = mu[:, offset]
        columns[f"pred_q10_{step}d"] = q10[:, offset]
        columns[f"pred_q50_{step}d"] = q50[:, offset]
        columns[f"pred_q90_{step}d"] = q90[:, offset]
    return pd.DataFrame(columns)


def _prediction_frame_for_dataset_indices(
    dataset: ForecastSequenceDataset | ForecastMemmapDataset,
    *,
    indices: np.ndarray,
    predictions: dict[str, np.ndarray],
    family: str,
    target_scale: float,
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
    loss_profile: str | None = None,
) -> pd.DataFrame:
    if isinstance(dataset, FORECAST_MEMMAP_DATASET_TYPES):
        if len(indices) == 0:
            return pd.DataFrame()
        scale = float(target_scale)
        idx = np.asarray(indices, dtype=int)
        rows = dataset.sample_index.iloc[idx].reset_index(drop=True)
        mu = predictions["mu"] / scale
        q10 = predictions["q10"] / scale
        q50 = predictions["q50"] / scale
        q90 = predictions["q90"] / scale
        aux = predictions["aux"] / scale
        y_daily = dataset.y_daily_excess[idx]
        y_cum = dataset.y_cum_excess[idx]
        horizon = int(dataset.manifest.get("horizon", y_daily.shape[1]) or y_daily.shape[1])
        horizons = dataset.cumulative_horizons
        columns: dict[str, Any] = {
            "date": rows["date"].astype(str).tolist(),
            "stock": rows["stock"].astype(str).to_numpy(),
            "role": rows["role"].astype(str).to_numpy(),
            "model_family": str(family),
            "stock_seen_in_train": rows.get("stock_seen_in_train", pd.Series(False, index=rows.index)).astype(bool).to_numpy(),
            "history_bucket": rows.get("history_bucket", pd.Series("", index=rows.index)).astype(str).to_numpy(),
            "history_valid_ratio": pd.to_numeric(rows.get("history_valid_ratio", pd.Series(np.nan, index=rows.index)), errors="coerce").to_numpy(),
            "future_rank_20d": dataset.y_rank_20d[idx],
            "future_path_max_drawdown_20d": dataset.y_max_drawdown_20d[idx],
            "future_path_worst_1d_20d": dataset.y_worst_1d_20d[idx],
            "future_path_upside_capture_20d": dataset.y_upside_20d[idx],
        }
        for pos, item_horizon in enumerate(horizons):
            columns[f"future_rank_{item_horizon}d"] = dataset.y_rank_by_horizon[idx, pos]
        for pos, item_horizon in enumerate(horizons):
            columns[f"future_cum_excess_return_{item_horizon}d"] = y_cum[:, pos]
            columns[f"future_path_max_drawdown_{item_horizon}d"] = dataset.y_drawdown_by_horizon[idx, pos]
            columns[f"future_path_worst_1d_{item_horizon}d"] = dataset.y_worst_by_horizon[idx, pos]
            columns[f"future_path_upside_capture_{item_horizon}d"] = dataset.y_upside_by_horizon[idx, pos]
            columns[f"pred_cum_mu_{item_horizon}d"] = mu[:, :item_horizon].sum(axis=1)
            columns[f"pred_aux_cum_{item_horizon}d"] = aux[:, pos]
        _add_decision_utility_columns(
            columns,
            predictions=predictions,
            y_cum=y_cum,
            drawdown_by_horizon=dataset.y_drawdown_by_horizon[idx],
            worst_by_horizon=dataset.y_worst_by_horizon[idx],
            target_scale=target_scale,
            decision_cost_bps=decision_cost_bps,
            decision_hit_threshold_bps=decision_hit_threshold_bps,
            decision_drawdown_penalty=decision_drawdown_penalty,
            cumulative_horizons=horizons,
            max_horizon=horizon,
            loss_profile=loss_profile,
        )
        risk_start = len(horizons)
        risk_aux = aux[:, risk_start : risk_start + len(horizons) * 3].reshape(-1, len(horizons), 3)
        for pos, item_horizon in enumerate(horizons):
            columns[f"pred_aux_downside_floor_{item_horizon}d"] = risk_aux[:, pos, 0]
            columns[f"pred_aux_worst_1d_{item_horizon}d"] = risk_aux[:, pos, 1]
            columns[f"pred_aux_upside_{item_horizon}d"] = risk_aux[:, pos, 2]
        for step in range(1, horizon + 1):
            offset = step - 1
            columns[f"future_excess_return_{step}d"] = y_daily[:, offset]
            columns[f"target_excess_{step}d"] = y_daily[:, offset]
            columns[f"pred_mu_{step}d"] = mu[:, offset]
            columns[f"pred_q10_{step}d"] = q10[:, offset]
            columns[f"pred_q50_{step}d"] = q50[:, offset]
            columns[f"pred_q90_{step}d"] = q90[:, offset]
        return pd.DataFrame(columns)
    return _prediction_frame_for_indices(
        dataset,
        indices=indices,
        predictions=predictions,
        family=family,
        target_scale=target_scale,
        decision_cost_bps=decision_cost_bps,
        decision_hit_threshold_bps=decision_hit_threshold_bps,
        decision_drawdown_penalty=decision_drawdown_penalty,
        loss_profile=loss_profile,
    )


def forecast_prediction_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"status": "insufficient_or_incomplete", "reason": "empty_predictions"}
    daily_steps = tuple(
        sorted(
            int(match.group(1))
            for column in frame.columns
            for match in [re.match(r"target_excess_(\d+)d$", str(column))]
            if match is not None
            and f"pred_q10_{int(match.group(1))}d" in frame.columns
            and f"pred_q90_{int(match.group(1))}d" in frame.columns
        )
    )
    horizons = tuple(
        sorted(
            int(match.group(1))
            for column in frame.columns
            for match in [re.match(r"pred_cum_mu_(\d+)d$", str(column))]
            if match is not None and f"future_cum_excess_return_{int(match.group(1))}d" in frame.columns
        )
    )
    if not horizons:
        horizons = PATH20_CUMULATIVE_HORIZONS
    q10_coverages: list[float] = []
    q90_coverages: list[float] = []
    for step in daily_steps:
        target = pd.to_numeric(frame[f"target_excess_{step}d"], errors="coerce")
        q10 = pd.to_numeric(frame[f"pred_q10_{step}d"], errors="coerce")
        q90 = pd.to_numeric(frame[f"pred_q90_{step}d"], errors="coerce")
        valid_q10 = target.notna() & q10.notna()
        valid_q90 = target.notna() & q90.notna()
        if bool(valid_q10.any()):
            q10_coverages.append(float((target.loc[valid_q10] >= q10.loc[valid_q10]).mean()))
        if bool(valid_q90.any()):
            q90_coverages.append(float((target.loc[valid_q90] <= q90.loc[valid_q90]).mean()))
    metrics: dict[str, Any] = {
        "status": "completed",
        "row_count": int(len(frame)),
        "date_count": int(frame["date"].nunique()),
        "q10_coverage_mean": float(np.mean(q10_coverages)) if q10_coverages else 0.0,
        "q90_coverage_mean": float(np.mean(q90_coverages)) if q90_coverages else 0.0,
        "forecast_horizon": int(max(daily_steps)) if daily_steps else 0,
        "cumulative_horizons": [int(item) for item in horizons],
    }
    for horizon in horizons:
        pred = pd.to_numeric(frame[f"pred_cum_mu_{horizon}d"], errors="coerce")
        target = pd.to_numeric(frame[f"future_cum_excess_return_{horizon}d"], errors="coerce")
        valid = pred.notna() & target.notna()
        metrics[f"rank_ic_{horizon}d"] = _rank_ic_by_date(
            frame,
            f"pred_cum_mu_{horizon}d",
            f"future_cum_excess_return_{horizon}d",
        )
        metrics[f"top_bottom_spread_{horizon}d"] = _top_bottom_spread_by_date(
            frame,
            f"pred_cum_mu_{horizon}d",
            f"future_cum_excess_return_{horizon}d",
        )
        metrics[f"direction_accuracy_{horizon}d"] = (
            float((np.sign(pred.loc[valid]) == np.sign(target.loc[valid])).mean()) if bool(valid.any()) else 0.0
        )
    if {"pred_aux_upside_20d", "future_path_upside_capture_20d"}.issubset(frame.columns):
        metrics["rank_ic_upside_20d"] = _rank_ic_by_date(
            frame,
            "pred_aux_upside_20d",
            "future_path_upside_capture_20d",
        )
        metrics["top_bottom_spread_upside_20d"] = _top_bottom_spread_by_date(
            frame,
            "pred_aux_upside_20d",
            "future_path_upside_capture_20d",
        )
    else:
        metrics["rank_ic_upside_20d"] = 0.0
        metrics["top_bottom_spread_upside_20d"] = 0.0
    if {"pred_decision_score", "future_decision_score", "pred_best_horizon", "future_best_horizon"}.issubset(frame.columns):
        metrics["decision_score_rank_ic"] = _rank_ic_by_date(frame, "pred_decision_score", "future_decision_score")
        metrics["decision_score_top_bottom_spread"] = _top_bottom_spread_by_date(
            frame,
            "pred_decision_score",
            "future_decision_score",
        )
        hit_cols = [f"future_hit_label_{int(horizon)}d" for horizon in horizons]
        if set(hit_cols).issubset(frame.columns):
            hit_any = frame[hit_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
            lifts: list[float] = []
            for _, group in frame.assign(_future_decision_hit_any=hit_any).groupby("date", sort=True):
                work = group[["pred_decision_score", "_future_decision_hit_any"]].copy()
                work["pred_decision_score"] = pd.to_numeric(work["pred_decision_score"], errors="coerce")
                work["_future_decision_hit_any"] = pd.to_numeric(work["_future_decision_hit_any"], errors="coerce")
                work = work.dropna()
                if len(work) < 2:
                    continue
                k = max(int(len(work) * 0.20), 1)
                top_hit = float(work.nlargest(k, "pred_decision_score")["_future_decision_hit_any"].mean())
                all_hit = float(work["_future_decision_hit_any"].mean())
                lifts.append(top_hit - all_hit)
            metrics["decision_hit_lift_top20_mean"] = float(np.mean(lifts)) if lifts else 0.0
        else:
            metrics["decision_hit_lift_top20_mean"] = 0.0
        pred_horizon = pd.to_numeric(frame["pred_best_horizon"], errors="coerce")
        future_horizon = pd.to_numeric(frame["future_best_horizon"], errors="coerce")
        valid_horizon = pred_horizon.notna() & future_horizon.notna()
        metrics["decision_best_horizon_accuracy"] = (
            float((pred_horizon.loc[valid_horizon] == future_horizon.loc[valid_horizon]).mean())
            if bool(valid_horizon.any())
            else 0.0
        )
        decision_passed = (
            float(metrics.get("decision_score_rank_ic", 0.0) or 0.0) > 0.0
            and float(metrics.get("decision_score_top_bottom_spread", 0.0) or 0.0) > 0.0
            and float(metrics.get("decision_hit_lift_top20_mean", 0.0) or 0.0) > 0.0
        )
        metrics["decision_utility_profile_status"] = "passed" if decision_passed else "failed"
    else:
        metrics["decision_score_rank_ic"] = 0.0
        metrics["decision_score_top_bottom_spread"] = 0.0
        metrics["decision_hit_lift_top20_mean"] = 0.0
        metrics["decision_best_horizon_accuracy"] = 0.0
        metrics["decision_utility_profile_status"] = "not_available"
    if {"pred_time_eff_score", "future_time_eff_score", "pred_time_eff_best_horizon", "future_time_eff_best_horizon"}.issubset(frame.columns):
        metrics["time_eff_score_rank_ic"] = _rank_ic_by_date(frame, "pred_time_eff_score", "future_time_eff_score")
        metrics["time_eff_score_top_bottom_spread"] = _top_bottom_spread_by_date(
            frame,
            "pred_time_eff_score",
            "future_time_eff_score",
        )
        pred_horizon = pd.to_numeric(frame["pred_time_eff_best_horizon"], errors="coerce")
        future_horizon = pd.to_numeric(frame["future_time_eff_best_horizon"], errors="coerce")
        valid_horizon = pred_horizon.notna() & future_horizon.notna()
        metrics["time_eff_best_horizon_accuracy"] = (
            float((pred_horizon.loc[valid_horizon] == future_horizon.loc[valid_horizon]).mean())
            if bool(valid_horizon.any())
            else 0.0
        )
        hit_cols = [f"future_time_eff_hit_label_{int(horizon)}d" for horizon in horizons]
        if set(hit_cols).issubset(frame.columns):
            hit_any = frame[hit_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
            lifts = []
            for _, group in frame.assign(_future_time_eff_hit_any=hit_any).groupby("date", sort=True):
                work = group[["pred_time_eff_score", "_future_time_eff_hit_any"]].copy()
                work["pred_time_eff_score"] = pd.to_numeric(work["pred_time_eff_score"], errors="coerce")
                work["_future_time_eff_hit_any"] = pd.to_numeric(work["_future_time_eff_hit_any"], errors="coerce")
                work = work.dropna()
                if len(work) < 2:
                    continue
                k = max(int(len(work) * 0.20), 1)
                top_hit = float(work.nlargest(k, "pred_time_eff_score")["_future_time_eff_hit_any"].mean())
                all_hit = float(work["_future_time_eff_hit_any"].mean())
                lifts.append(top_hit - all_hit)
            metrics["time_eff_hit_lift_top20_mean"] = float(np.mean(lifts)) if lifts else 0.0
        else:
            metrics["time_eff_hit_lift_top20_mean"] = 0.0
        metrics["time_eff_profile_status"] = "completed"
    else:
        metrics["time_eff_score_rank_ic"] = 0.0
        metrics["time_eff_score_top_bottom_spread"] = 0.0
        metrics["time_eff_best_horizon_accuracy"] = 0.0
        metrics["time_eff_hit_lift_top20_mean"] = 0.0
        metrics["time_eff_profile_status"] = "not_available"
    metrics["selected_signal_profile"] = forecast_signal_profile(metrics)
    return metrics


def stratified_forecast_prediction_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    out: dict[str, Any] = {}
    if "stock_seen_in_train" in frame.columns:
        for value, group in frame.groupby("stock_seen_in_train", dropna=False):
            key = f"seen_in_train={str(bool(value)).lower()}" if pd.notna(value) else "seen_in_train=unknown"
            out[key] = forecast_prediction_metrics(group)
    if "history_bucket" in frame.columns:
        for value, group in frame.groupby("history_bucket", dropna=False):
            key = f"history_bucket={str(value)}"
            out[key] = forecast_prediction_metrics(group)
    return out


def _coverage_pass(metrics: dict[str, Any], coverage_range: tuple[float, float]) -> bool:
    q10 = float(metrics.get("q10_coverage_mean", 0.0) or 0.0)
    q90 = float(metrics.get("q90_coverage_mean", 0.0) or 0.0)
    low, high = coverage_range
    return bool(low <= q10 <= high and low <= q90 <= high)


def _horizon_gate(metrics: dict[str, Any], horizon: int) -> bool:
    return bool(
        float(metrics.get(f"rank_ic_{int(horizon)}d", 0.0) or 0.0) > 0.0
        and float(metrics.get(f"top_bottom_spread_{int(horizon)}d", 0.0) or 0.0) > 0.0
    )


def _short_burst_gate(metrics: dict[str, Any]) -> bool:
    gates = [
        _horizon_gate(metrics, 3),
        _horizon_gate(metrics, 5),
        bool(
            float(metrics.get("rank_ic_upside_20d", 0.0) or 0.0) > 0.0
            and float(metrics.get("top_bottom_spread_upside_20d", 0.0) or 0.0) > 0.0
        ),
    ]
    return sum(1 for item in gates if item) >= 2


def forecast_signal_profile(metrics: dict[str, Any]) -> str:
    if metrics.get("status") != "completed":
        return "failed"
    trend = _horizon_gate(metrics, 20)
    short_burst = _short_burst_gate(metrics)
    if trend and short_burst:
        return "multiscale"
    if trend:
        return "trend_20d"
    if short_burst:
        return "short_burst"
    return "failed"


def _profile_pass(metrics: dict[str, Any], selection_profile: str) -> bool:
    profile = str(selection_profile or "multiscale").strip().lower()
    if profile == "decision_utility":
        return bool(
            float(metrics.get("decision_score_rank_ic", 0.0) or 0.0) > 0.0
            and float(metrics.get("decision_score_top_bottom_spread", 0.0) or 0.0) > 0.0
            and float(metrics.get("decision_hit_lift_top20_mean", 0.0) or 0.0) > 0.0
        )
    if profile == "validation_loss":
        return bool(metrics.get("status") == "completed")
    if profile == "trend20":
        return _horizon_gate(metrics, 20)
    if profile == "short_burst":
        return _short_burst_gate(metrics)
    return _horizon_gate(metrics, 20) or _short_burst_gate(metrics)


def _profile_score(metrics: dict[str, Any], validation_loss: float, coverage_range: tuple[float, float], selection_profile: str) -> float:
    if metrics.get("status") != "completed":
        return -float(validation_loss)
    profile = str(selection_profile or "multiscale").strip().lower()
    if profile == "validation_loss":
        return float(-validation_loss)
    if profile == "decision_utility":
        gate_bonus = 1_000.0 if _profile_pass(metrics, profile) and _coverage_pass(metrics, coverage_range) else 0.0
        return float(
            gate_bonus
            + float(metrics.get("decision_score_rank_ic", 0.0) or 0.0) * 10.0
            + float(metrics.get("decision_score_top_bottom_spread", 0.0) or 0.0)
            + float(metrics.get("decision_hit_lift_top20_mean", 0.0) or 0.0)
            - max(float(validation_loss), 0.0) * 1.0e-3
        )
    weights = FORECAST_PROFILE_HORIZON_WEIGHTS.get(profile, FORECAST_PROFILE_HORIZON_WEIGHTS["multiscale"])
    rank_score = sum(
        float(weight) * float(metrics.get(f"rank_ic_{int(horizon)}d", 0.0) or 0.0)
        for horizon, weight in weights.items()
    )
    spread_score = sum(
        float(weight) * float(metrics.get(f"top_bottom_spread_{int(horizon)}d", 0.0) or 0.0)
        for horizon, weight in weights.items()
    )
    upside_weight = 0.10 if profile in {"multiscale", "short_burst"} else 0.03
    rank_score += upside_weight * float(metrics.get("rank_ic_upside_20d", 0.0) or 0.0)
    spread_score += upside_weight * float(metrics.get("top_bottom_spread_upside_20d", 0.0) or 0.0)
    gate_bonus = 1_000.0 if _profile_pass(metrics, profile) and _coverage_pass(metrics, coverage_range) else 0.0
    return float(gate_bonus + rank_score * 10.0 + spread_score - max(float(validation_loss), 0.0) * 1.0e-3)


def forecast_evidence_verdict(
    *,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None = None,
    coverage_range: tuple[float, float] = (0.65, 0.95),
    selection_profile: str = "multiscale",
) -> str:
    if validation_metrics.get("status") != "completed":
        return "insufficient_or_incomplete"
    profile = str(selection_profile or "multiscale").strip().lower()
    if profile == "validation_loss":
        return "forecast_promising"
    validation_promising = _profile_pass(validation_metrics, selection_profile) and _coverage_pass(validation_metrics, coverage_range)
    if not validation_promising:
        return "forecast_failed"
    if not test_metrics or test_metrics.get("status") != "completed":
        return "forecast_promising"
    if _profile_pass(test_metrics, selection_profile):
        return "forecast_test_confirmed"
    return "forecast_promising"


def _ranking_baseline_feature_frame(
    dataset: ForecastSequenceDataset | ForecastMemmapDataset,
    indices: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    idx = np.asarray(indices, dtype=int)
    if len(idx) == 0:
        return pd.DataFrame(), np.empty((0,), dtype=np.float32)
    if isinstance(dataset, FORECAST_MEMMAP_DATASET_TYPES):
        store = dataset.open_feature_store()
        rows: list[np.ndarray] = []
        for row_idx in idx:
            window = dataset.input_window(int(row_idx), store=store)
            rows.append(
                np.concatenate(
                    [
                        window[-1],
                        np.nanmean(window, axis=0),
                        np.nanstd(window, axis=0),
                    ]
                ).astype(np.float32)
            )
        meta = dataset.sample_index.iloc[idx].reset_index(drop=True)
        x = np.vstack(rows).astype(np.float32) if rows else np.empty((0, 0), dtype=np.float32)
        y = np.asarray(dataset.y_rank_20d[idx], dtype=np.float32)
    else:
        x = np.concatenate(
            [
                dataset.x[idx, -1, :],
                np.nanmean(dataset.x[idx], axis=1),
                np.nanstd(dataset.x[idx], axis=1),
            ],
            axis=1,
        ).astype(np.float32)
        y = np.asarray(dataset.y_rank_20d[idx], dtype=np.float32)
        meta = pd.DataFrame(
            {
                "date": [pd.Timestamp(item).strftime("%Y-%m-%d") for item in dataset.date[idx].tolist()],
                "stock": dataset.stock[idx].astype(str),
                "role": dataset.role[idx].astype(str),
            }
        )
    columns = [f"ranker_feature_{pos}" for pos in range(x.shape[1])]
    frame = pd.DataFrame(x, columns=columns)
    for column in ("date", "stock", "role"):
        frame[column] = meta[column].astype(str).to_numpy() if column in meta.columns else ""
    return frame, y


def ranking_relevance_labels(
    feature_frame: pd.DataFrame,
    target: pd.Series | np.ndarray,
    *,
    relevance_levels: int = 5,
) -> np.ndarray:
    levels = max(int(relevance_levels), 2)
    work = pd.DataFrame(
        {
            "date": feature_frame["date"].astype(str).to_numpy() if "date" in feature_frame.columns else "",
            "target": pd.to_numeric(pd.Series(target), errors="coerce").to_numpy(dtype=float),
        }
    )
    labels = np.zeros(len(work), dtype=np.int32)
    for _, group in work.dropna(subset=["target"]).groupby("date", sort=False):
        if len(group) <= 1:
            labels[group.index.to_numpy(dtype=int)] = levels - 1
            continue
        order = group["target"].rank(method="first").to_numpy(dtype=float) - 1.0
        group_labels = np.rint(order * float(levels - 1) / float(len(group) - 1)).astype(np.int32)
        labels[group.index.to_numpy(dtype=int)] = np.clip(group_labels, 0, levels - 1)
    return labels.astype(np.int32)


def _ranking_prediction_frame(
    dataset: ForecastSequenceDataset | ForecastMemmapDataset,
    *,
    indices: np.ndarray,
    scores: np.ndarray,
    family: str,
) -> pd.DataFrame:
    idx = np.asarray(indices, dtype=int)
    horizon = int(dataset.manifest.get("horizon", PATH20_HORIZON) or PATH20_HORIZON)
    horizons = (
        dataset.cumulative_horizons
        if isinstance(dataset, FORECAST_MEMMAP_DATASET_TYPES)
        else normalize_path20_cumulative_horizons(dataset.manifest.get("cumulative_horizons", PATH20_CUMULATIVE_HORIZONS), horizon=horizon)
    )
    predictions = {
        "mu": np.repeat(np.asarray(scores, dtype=np.float32).reshape(-1, 1), horizon, axis=1),
        "q10": np.repeat(np.asarray(scores, dtype=np.float32).reshape(-1, 1), horizon, axis=1),
        "q50": np.repeat(np.asarray(scores, dtype=np.float32).reshape(-1, 1), horizon, axis=1),
        "q90": np.repeat(np.asarray(scores, dtype=np.float32).reshape(-1, 1), horizon, axis=1),
        "aux": np.zeros((len(idx), path20_forecast_aux_dim(horizons, horizon=horizon)), dtype=np.float32),
    }
    return _prediction_frame_for_dataset_indices(
        dataset,
        indices=idx,
        predictions=predictions,
        family=family,
        target_scale=1.0,
    )


def run_forecast_ranking_baseline(
    dataset: ForecastSequenceDataset | ForecastMemmapDataset,
    *,
    study_root: Path,
    baseline: str = "none",
) -> dict[str, Any]:
    baseline = str(baseline or "none").strip().lower()
    if baseline == "none":
        return {"status": "skipped", "baseline": "none"}
    if baseline not in FORECAST_RANKING_BASELINES:
        return {
            "status": "dependency_missing",
            "baseline": baseline,
            "reason": "unsupported_ranking_baseline",
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
    study_root.mkdir(parents=True, exist_ok=True)
    view = _ForecastDatasetView(dataset)
    train_indices = view.role_indices("train")
    validation_indices = view.role_indices("validation")
    test_indices = view.role_indices("test")
    if len(train_indices) < 2 or len(validation_indices) < 1:
        summary = {
            "status": "insufficient_or_incomplete",
            "baseline": baseline,
            "reason": "insufficient_role_samples",
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
        write_json(study_root / f"forecast_ranking_baseline_{baseline}.json", _json_ready(summary))
        return summary
    try:
        if baseline == "lightgbm":
            from lightgbm import LGBMRanker  # type: ignore

            ranker: Any = LGBMRanker(n_estimators=80, learning_rate=0.05, random_state=7)
        else:
            from xgboost import XGBRanker  # type: ignore

            ranker = XGBRanker(n_estimators=80, learning_rate=0.05, random_state=7, objective="rank:pairwise")
    except Exception as exc:
        summary = {
            "status": "dependency_missing",
            "baseline": baseline,
            "reason": f"{baseline}_import_failed",
            "error": str(exc),
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
        write_json(study_root / f"forecast_ranking_baseline_{baseline}.json", _json_ready(summary))
        return summary
    train_x, train_y = _ranking_baseline_feature_frame(dataset, train_indices)
    validation_x, _ = _ranking_baseline_feature_frame(dataset, validation_indices)
    test_x, _ = _ranking_baseline_feature_frame(dataset, test_indices)
    feature_columns = [column for column in train_x.columns if column.startswith("ranker_feature_")]
    train_groups = train_x.groupby("date", sort=True).size().to_numpy(dtype=int)
    try:
        train_relevance = ranking_relevance_labels(train_x, train_y)
        ranker.fit(train_x[feature_columns].to_numpy(dtype=np.float32), train_relevance, group=train_groups)
        validation_scores = np.asarray(ranker.predict(validation_x[feature_columns].to_numpy(dtype=np.float32)), dtype=np.float32)
        test_scores = (
            np.asarray(ranker.predict(test_x[feature_columns].to_numpy(dtype=np.float32)), dtype=np.float32)
            if len(test_indices)
            else np.empty((0,), dtype=np.float32)
        )
    except Exception as exc:
        summary = {
            "status": "failed",
            "baseline": baseline,
            "reason": "ranker_fit_or_predict_failed",
            "error": str(exc),
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
        write_json(study_root / f"forecast_ranking_baseline_{baseline}.json", _json_ready(summary))
        return summary
    validation_frame = _ranking_prediction_frame(dataset, indices=validation_indices, scores=validation_scores, family=f"{baseline}_ranker")
    test_frame = _ranking_prediction_frame(dataset, indices=test_indices, scores=test_scores, family=f"{baseline}_ranker")
    validation_csv = _write_frame(study_root / f"forecast_ranking_baseline_{baseline}_validation.csv", validation_frame)
    test_csv = _write_frame(study_root / f"forecast_ranking_baseline_{baseline}_test.csv", test_frame)
    validation_metrics = forecast_prediction_metrics(validation_frame)
    test_metrics = forecast_prediction_metrics(test_frame) if not test_frame.empty else {"status": "insufficient_or_incomplete"}
    summary = {
        "status": "completed",
        "baseline": baseline,
        "feature_count": int(len(feature_columns)),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "test_rows": int(len(test_indices)),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_csv": validation_csv,
        "test_csv": test_csv,
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    write_json(study_root / f"forecast_ranking_baseline_{baseline}.json", _json_ready(summary))
    return _json_ready(summary)


def _write_slot_diagnostics(
    path: Path,
    *,
    model: nn.Module,
    dataset: ForecastSequenceDataset | ForecastMemmapDataset,
    indices: np.ndarray,
    seed: int,
) -> str:
    slot_tensor = getattr(model, "slots", None)
    slot_count = int(slot_tensor.shape[0]) if isinstance(slot_tensor, torch.Tensor) else 0
    rows = pd.DataFrame()
    if isinstance(dataset, FORECAST_MEMMAP_DATASET_TYPES) and len(indices):
        rows = dataset.sample_index.iloc[np.asarray(indices, dtype=int)].copy()
    industry_available = "industry_id" in rows.columns and pd.to_numeric(rows["industry_id"], errors="coerce").fillna(0).ne(0).any()
    board_available = "board_id" in rows.columns and pd.to_numeric(rows["board_id"], errors="coerce").fillna(0).ne(0).any()
    payload = {
        "status": "completed",
        "slot_semantics": "dynamic_theme_factor_not_static_board",
        "seed": int(seed),
        "slot_count": int(slot_count),
        "row_count": int(len(rows)),
        "date_count": int(rows["date"].nunique()) if "date" in rows.columns else 0,
        "industry_available": bool(industry_available),
        "board_available": bool(board_available),
        "industry_id_top_counts": {
            str(key): int(value)
            for key, value in (
                rows["industry_id"].value_counts().head(10).items() if "industry_id" in rows.columns else []
            )
        },
        "board_id_top_counts": {
            str(key): int(value)
            for key, value in (
                rows["board_id"].value_counts().head(10).items() if "board_id" in rows.columns else []
            )
        },
    }
    write_json(path, _json_ready(payload))
    return str(path.resolve())


def _normalize_seeds(seeds: tuple[int, ...] | list[int] | str | None, *, default_seed: int) -> tuple[int, ...]:
    if seeds is None:
        return (int(default_seed),)
    if isinstance(seeds, str):
        parsed = tuple(int(item.strip()) for item in seeds.split(",") if item.strip())
        return parsed or (int(default_seed),)
    parsed = tuple(int(item) for item in seeds)
    return parsed or (int(default_seed),)


def _static_context_model_options(dataset_view: _ForecastDatasetView) -> dict[str, Any]:
    return {
        "static_context_vocab_sizes": dict(dataset_view.static_context_vocab_sizes),
        "static_context_embedding_dims": dict(
            dataset_view.static_context_schema.get("embedding_defaults", {}) or {}
        ),
        "static_context_fields": tuple(str(item) for item in dataset_view.static_context_schema.get("fields", []) or []),
        "static_context_dropout": float(
            dict(dataset_view.static_context_schema.get("embedding_defaults", {}) or {}).get("dropout", 0.20)
        ),
        "slot_count": 8,
    }


def _intraday_feature_indices(feature_columns: list[str] | tuple[str, ...]) -> tuple[int, ...]:
    intraday_prefixes = ("intraday_", "cs_rank_intraday_", "cs_z_intraday_")
    return tuple(
        int(idx)
        for idx, column in enumerate(feature_columns)
        if str(column).startswith(intraday_prefixes)
    )


def _structured_alpha_v2_feature_groups(feature_columns: list[str] | tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    groups: dict[str, list[int]] = {
        "daily_price_volume": [],
        "cross_section": [],
        "market_regime": [],
        "industry_peer": [],
        "valuation_liquidity": [],
        "event_quality": [],
        "intraday": [],
    }

    def _add(group: str, idx: int) -> None:
        groups.setdefault(group, []).append(int(idx))

    for idx, raw_column in enumerate(feature_columns):
        column = str(raw_column)
        lower = column.lower()
        if lower.startswith(("intraday_", "cs_rank_intraday_", "cs_z_intraday_")):
            _add("intraday", idx)
        elif "limit_up_down_pressure" in lower:
            _add("market_regime", idx)
        elif (
            lower.startswith(("adjust_", "index_", "history_valid_", "core_ohlcv_valid_"))
            or lower in {"in_pool", "is_low_history_like"}
            or "adjust_" in lower
            or "suspend" in lower
            or "st_flag" in lower
            or "limit_up" in lower
            or "limit_down" in lower
        ):
            _add("event_quality", idx)
        elif (
            lower.startswith(("valuation_", "turn_", "adv_"))
            or lower in {"turn", "turnover", "amount", "volume", "liquidity"}
            or "amount" in lower
            or "volume" in lower
            or "liquidity" in lower
            or "peer_adv_bucket_id" in lower
            or "peer_price_bucket_id" in lower
            or "peer_vol20_bucket_id" in lower
            or lower.startswith(("cs_rank_turn", "cs_z_turn", "industry_rank_turn", "industry_z_turn"))
        ):
            _add("valuation_liquidity", idx)
        elif lower.startswith(("market_", "benchmark_", "cross_section_")) or "limit_up_down_pressure" in lower:
            _add("market_regime", idx)
        elif (
            lower.startswith(("industry_", "peer_", "relative_to_peer_"))
            or "_minus_industry" in lower
            or "industry_rank_" in lower
            or "industry_z_" in lower
        ):
            _add("industry_peer", idx)
        elif lower.startswith(("cs_rank_", "cs_z_")):
            _add("cross_section", idx)
        else:
            _add("daily_price_volume", idx)
    return {key: tuple(value) for key, value in groups.items()}


def _structured_alpha_v2_feature_groups_for_dataset(
    dataset_view: _ForecastDatasetView,
) -> tuple[dict[str, tuple[int, ...]], str]:
    pack_meta = dict(dataset_view.manifest.get("structured_alpha_v2_pack", {}) or {})
    raw_groups = dict(pack_meta.get("feature_group_indices", {}) or {})
    expected_groups = (
        "daily_price_volume",
        "cross_section",
        "market_regime",
        "industry_peer",
        "valuation_liquidity",
        "event_quality",
        "intraday",
    )
    if raw_groups:
        input_dim = int(dataset_view.input_dim)
        groups: dict[str, tuple[int, ...]] = {}
        seen: set[int] = set()
        for group in expected_groups:
            values = tuple(
                int(item)
                for item in list(raw_groups.get(group, []) or [])
            )
            invalid = [idx for idx in values if idx < 0 or idx >= input_dim]
            if invalid:
                raise ValueError(f"structured_alpha_v2_pack_feature_group_index_out_of_range: {group} {invalid[:5]}")
            duplicates = [idx for idx in values if idx in seen]
            if duplicates:
                raise ValueError(f"structured_alpha_v2_pack_feature_group_index_duplicate: {group} {duplicates[:5]}")
            seen.update(values)
            groups[group] = values
        missing = [idx for idx in range(input_dim) if idx not in seen]
        if missing:
            raise ValueError(f"structured_alpha_v2_pack_feature_group_indices_incomplete: missing {missing[:5]}")
        return groups, "manifest_structured_alpha_v2_pack"
    return _structured_alpha_v2_feature_groups(dataset_view.feature_columns), "column_name_inference"


def _model_config_for_training(
    *,
    family: str,
    hidden_dim: int,
    dropout: float,
    gru_layers: int,
    transformer_layers: int,
    transformer_heads: int,
    patch_sizes: tuple[int, ...] | list[int],
    dataset_view: _ForecastDatasetView,
    static_model_options: dict[str, Any],
    output_profile: str = "forecast_path_v1",
) -> dict[str, Any]:
    config = {
        "model_family": str(family),
        "output_profile": str(output_profile or "forecast_path_v1"),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "gru_layers": int(gru_layers),
        "transformer_layers": int(transformer_layers),
        "transformer_heads": int(transformer_heads),
        "patch_sizes": [int(item) for item in patch_sizes],
        "static_context_vocab_sizes": dict(static_model_options.get("static_context_vocab_sizes", {}) or {}),
        "static_context_embedding_dims": dict(static_model_options.get("static_context_embedding_dims", {}) or {}),
        "static_context_fields": list(static_model_options.get("static_context_fields", ()) or ()),
        "static_context_dropout": float(static_model_options.get("static_context_dropout", 0.20)),
        "slot_count": int(static_model_options.get("slot_count", 8)),
        "static_context_schema": dict(dataset_view.static_context_schema),
        "symbol_vocab_fingerprint": str(dataset_view.symbol_vocab_fingerprint),
        "industry_vocab_fingerprint": str(dataset_view.industry_vocab_fingerprint),
        "board_vocab_fingerprint": str(dataset_view.board_vocab_fingerprint),
    }
    if str(family) == "hybrid_expert_fusion_static_context":
        config.update(
            {
                "fusion_version": "hybrid_expert_fusion_static_context_v1",
                "expert_families": ["gru_sequence_static_context", "patch_transformer_static_context", "dlinear_sequence"],
                "router_temperature": 1.0,
                "router_entropy_loss_weight": 0.0,
                "expert_load_balance_loss_weight": 0.0,
            }
        )
    if str(family) == "hybrid_multiscale_recency_aware_v1":
        intraday_indices = _intraday_feature_indices(dataset_view.feature_columns)
        config.update(
            {
                "fusion_version": "hybrid_multiscale_recency_aware_v1",
                "expert_families": [
                    "gru_main_daily_context",
                    "patch_transformer_recency_bias",
                    "multi_half_life_ewma_trend",
                    "intraday_short_half_life_bottleneck",
                ],
                "router_temperature": 1.0,
                "recency_half_lives": [5.0, 10.0, 20.0, 60.0, 120.0],
                "patch_recency_halflife": 8.0,
                "intraday_recency_halflife": 5.0,
                "intraday_bottleneck_dim": 32,
                "intraday_feature_indices": [int(item) for item in intraday_indices],
                "intraday_feature_count": int(len(intraday_indices)),
                "intraday_feature_columns": [dataset_view.feature_columns[int(item)] for item in intraday_indices],
                "main_feature_count": int(len(dataset_view.feature_columns) - len(intraday_indices)),
            }
        )
    if str(family) == "hybrid_structured_alpha_v2":
        fields = tuple(str(item) for item in static_model_options.get("static_context_fields", ()) or ())
        if any(item == "symbol" for item in fields):
            raise ValueError("hybrid_structured_alpha_v2 model_config requires symbol-free static context; use exchange,industry.")
        feature_groups, feature_group_source = _structured_alpha_v2_feature_groups_for_dataset(dataset_view)
        config.update(
            {
                "fusion_version": "hybrid_structured_alpha_v2",
                "architecture_contract": {
                    "prediction_first": True,
                    "execution_utility_in_model": False,
                    "intraday_role": "residual_short_horizon_correction",
                    "context_role": "static_and_regime_film_conditioning",
                    "symbol_static_context_allowed": False,
                    "default_hidden_dim": 192,
                },
                "feature_group_indices": {
                    group: [int(item) for item in indices]
                    for group, indices in feature_groups.items()
                },
                "feature_group_counts": {
                    group: int(len(indices))
                    for group, indices in feature_groups.items()
                },
                "feature_group_columns": {
                    group: [dataset_view.feature_columns[int(item)] for item in indices]
                    for group, indices in feature_groups.items()
                },
                "feature_group_source": feature_group_source,
                "expert_families": [
                    "gru_continuity_on_group_mixed_sequence",
                    "recency_biased_multiscale_patch_transformer",
                    "multi_half_life_ewma_trend",
                    "local_dilated_tcn",
                ],
                "fusion_layers": 2,
                "group_mixer_layers": 1,
                "group_mixer_chunk_size": DEFAULT_GROUP_MIXER_CHUNK_SIZE,
                "router_temperature": 1.0,
                "recency_half_lives": [3.0, 5.0, 10.0, 20.0, 60.0, 120.0],
                "patch_recency_halflife": 6.0,
                "intraday_recency_halflife": 3.0,
                "intraday_residual_mask": "decays_by_horizon; strongest for daily/short cumulative outputs",
                "required_static_context_fields": ["exchange", "industry"],
                "recommended_loss_profiles": ["hybrid_alpha_score_v2", "hybrid_alpha_score_v1", "forecast_path_v1_baseline"],
                "promotion_allowed": False,
            }
        )
    if str(family) == "date_slate_alpha_fusion_v1":
        fields = tuple(str(item) for item in static_model_options.get("static_context_fields", ()) or ())
        if any(item == "symbol" for item in fields):
            raise ValueError("date_slate_alpha_fusion_v1 model_config requires symbol-free static context; use exchange,industry.")
        feature_groups, feature_group_source = _structured_alpha_v2_feature_groups_for_dataset(dataset_view)
        config.update(
            {
                "fusion_version": "date_slate_alpha_fusion_v1",
                "architecture_contract": {
                    "prediction_first": True,
                    "execution_utility_in_model": False,
                    "date_slate_rank_semantics": True,
                    "daily_increment_outputs": True,
                    "cumulative_outputs_derived_from_daily_mu": True,
                    "intraday_role": "context_gated_daily_increment_residual",
                    "symbol_static_context_allowed": False,
                    "default_hidden_dim": 128,
                },
                "feature_group_indices": {
                    group: [int(item) for item in indices]
                    for group, indices in feature_groups.items()
                },
                "feature_group_counts": {
                    group: int(len(indices))
                    for group, indices in feature_groups.items()
                },
                "feature_group_columns": {
                    group: [dataset_view.feature_columns[int(item)] for item in indices]
                    for group, indices in feature_groups.items()
                },
                "feature_group_source": feature_group_source,
                "expert_families": [
                    "local_dilated_tcn",
                    "recency_biased_patch_transformer",
                    "multi_half_life_ewma_trend",
                ],
                "fusion_layers": 2,
                "group_mixer_layers": 1,
                "group_mixer_chunk_size": DEFAULT_GROUP_MIXER_CHUNK_SIZE,
                "router_use": "residual_gate_and_pooling_bias_not_pre_fusion_hard_suppression",
                "required_static_context_fields": ["exchange", "industry"],
                "required_output_profile": "forecast_incremental_path_v2",
                "recommended_loss_profiles": ["date_grouped_alpha_score_v1"],
                "promotion_allowed": False,
            }
        )
    if str(family) == "regime_routed_multi_expert_horizon_v1":
        config.update(
            {
                "fusion_version": "regime_routed_multi_expert_horizon_v1",
                "expert_families": [
                    "gru_path_continuity",
                    "patch_transformer_multiscale_events",
                    "dlinear_low_frequency_trend",
                    "stock_mixer_cross_section",
                    "local_state_volatility_reversal",
                ],
                "router_inputs": [
                    "expert_pooled_tokens",
                    "static_context_embedding",
                    "local_volatility",
                    "recent_runup",
                    "drawdown_state",
                    "horizon_confidence_summary",
                ],
                "router_temperature": 1.0,
                "router_entropy_floor_weight": 0.005,
                "expert_diversity_weight": 0.002,
                "bad_state_calibration_weight": 0.010,
                "cross_section_batching_required_for_stock_mixer": True,
                "research_only": True,
                "promotion_allowed": False,
            }
        )
    return config


def _validation_score(
    metrics: dict[str, Any],
    validation_loss: float,
    coverage_range: tuple[float, float],
    selection_profile: str,
) -> float:
    return _profile_score(metrics, validation_loss, coverage_range, selection_profile)


def _evaluate_loss(
    model: nn.Module,
    dataset_view_or_x: _ForecastDatasetView | torch.Tensor,
    y_daily: torch.Tensor | None,
    y_cum: torch.Tensor | None,
    y_risk: torch.Tensor | None,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    target_scale: float = 100.0,
    loss_profile: str = "default",
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
    date_slate_dates_per_batch: int = 1,
    date_slate_stocks_per_date: int = 512,
    rank_min_group_size: int = 8,
    rank_max_pairs_per_date: int = 4096,
) -> float:
    if len(indices) == 0:
        return float("inf")
    model.eval()
    values: list[float] = []
    counts: list[int] = []
    with torch.no_grad():
        if isinstance(dataset_view_or_x, _ForecastDatasetView):
            if (
                _normalize_forecast_loss_profile(loss_profile) == "date_grouped_alpha_score_v1"
                and isinstance(dataset_view_or_x.dataset, ForecastTrainingPackDataset)
            ):
                loader = DataLoader(
                    dataset_view_or_x.date_slate_torch_dataset(
                        indices,
                        dates_per_batch=max(int(date_slate_dates_per_batch), 1),
                        stocks_per_date=max(int(date_slate_stocks_per_date), 1),
                        target_scale=target_scale,
                        shuffle_stocks=False,
                        seed=0,
                    ),
                    batch_size=None,
                    shuffle=False,
                    pin_memory=device.type == "cuda",
                )
                for raw_batch in loader:
                    if len(raw_batch) == 7:
                        batch_x_cpu, batch_y_daily_cpu, batch_y_cum_cpu, batch_y_risk_cpu, _, date_group_ids_cpu, static_context_ids_cpu = raw_batch
                    elif len(raw_batch) == 6:
                        batch_x_cpu, batch_y_daily_cpu, batch_y_cum_cpu, batch_y_risk_cpu, _, date_group_ids_cpu = raw_batch
                        static_context_ids_cpu = None
                    else:
                        raise ValueError(f"date-slate eval batch must contain 6 or 7 tensors, got {len(raw_batch)}.")
                    batch_x = batch_x_cpu.to(device, non_blocking=device.type == "cuda")
                    batch_y_daily = batch_y_daily_cpu.to(device, non_blocking=device.type == "cuda")
                    batch_y_cum = batch_y_cum_cpu.to(device, non_blocking=device.type == "cuda")
                    batch_y_risk = batch_y_risk_cpu.to(device, non_blocking=device.type == "cuda")
                    date_group_ids = date_group_ids_cpu.to(device, non_blocking=device.type == "cuda")
                    static_context_ids = (
                        static_context_ids_cpu.to(device, non_blocking=device.type == "cuda")
                        if static_context_ids_cpu is not None
                        else None
                    )
                    static_context_ids = dataset_view_or_x.filter_static_context_ids(static_context_ids)
                    with _autocast_context(device, amp_enabled):
                        pred = _forecast_model_forward(model, batch_x, static_context_ids)
                        loss = _date_grouped_forecast_loss(
                            pred,
                            batch_y_daily,
                            batch_y_cum,
                            batch_y_risk,
                            date_group_ids,
                            loss_profile=loss_profile,
                            target_scale=target_scale,
                            cumulative_horizons=dataset_view_or_x.cumulative_horizons,
                            rank_min_group_size=int(rank_min_group_size),
                            rank_max_pairs_per_date=int(rank_max_pairs_per_date),
                        )
                    values.append(float(loss.detach().cpu()))
                    counts.append(int(batch_x.shape[0]))
                total = sum(counts)
                return float(np.average(values, weights=counts)) if total > 0 else float("inf")
            if _forecast_model_accepts_stock_mask(model) and dataset_view_or_x.dataset_mode == "memmap":
                loader = DataLoader(
                    dataset_view_or_x.date_batch_torch_dataset(indices, target_scale=target_scale),
                    batch_size=1,
                    shuffle=False,
                    pin_memory=device.type == "cuda",
                    collate_fn=_collate_forecast_date_batches,
                )
                for batch_x_cpu, stock_mask_cpu, batch_y_daily_cpu, batch_y_cum_cpu, batch_y_risk_cpu, _, static_context_ids_cpu in loader:
                    batch_x = batch_x_cpu.to(device, non_blocking=device.type == "cuda")
                    stock_mask = stock_mask_cpu.to(device, non_blocking=device.type == "cuda")
                    batch_y_daily = batch_y_daily_cpu.to(device, non_blocking=device.type == "cuda")
                    batch_y_cum = batch_y_cum_cpu.to(device, non_blocking=device.type == "cuda")
                    batch_y_risk = batch_y_risk_cpu.to(device, non_blocking=device.type == "cuda")
                    static_context_ids = (
                        static_context_ids_cpu.to(device, non_blocking=device.type == "cuda")
                        if static_context_ids_cpu is not None
                        else None
                    )
                    static_context_ids = dataset_view_or_x.filter_static_context_ids(static_context_ids)
                    flat_mask = stock_mask.reshape(-1)
                    with _autocast_context(device, amp_enabled):
                        pred = _forecast_model_forward(model, batch_x, static_context_ids, stock_mask=stock_mask)
                        loss = _forecast_loss(
                            {key: value[flat_mask] for key, value in pred.items()},
                            batch_y_daily.reshape(-1, batch_y_daily.shape[-1])[flat_mask],
                            batch_y_cum.reshape(-1, batch_y_cum.shape[-1])[flat_mask],
                            batch_y_risk.reshape(-1, batch_y_risk.shape[-2], batch_y_risk.shape[-1])[flat_mask],
                            loss_profile=loss_profile,
                            target_scale=target_scale,
                            decision_cost_bps=decision_cost_bps,
                            decision_hit_threshold_bps=decision_hit_threshold_bps,
                            decision_drawdown_penalty=decision_drawdown_penalty,
                            cumulative_horizons=dataset_view_or_x.cumulative_horizons,
                        )
                    values.append(float(loss.detach().cpu()))
                    counts.append(int(flat_mask.sum().detach().cpu()))
                total = sum(counts)
                return float(np.average(values, weights=counts)) if total > 0 else float("inf")
            if isinstance(dataset_view_or_x.dataset, (ForecastShardedMemmapDataset, ForecastTrainingPackDataset)):
                eval_indices = dataset_view_or_x.dataset.cache_friendly_indices(indices)
                loader = DataLoader(
                    dataset_view_or_x.batch_torch_dataset(
                        eval_indices,
                        batch_size=max(int(batch_size), 1),
                        target_scale=target_scale,
                    ),
                    batch_size=None,
                    shuffle=False,
                    pin_memory=device.type == "cuda",
                )
            else:
                loader = DataLoader(
                    dataset_view_or_x.torch_dataset(indices, target_scale=target_scale),
                    batch_size=max(int(batch_size), 1),
                    shuffle=False,
                    pin_memory=device.type == "cuda",
                )
            for raw_batch in loader:
                batch_x, batch_y_daily, batch_y_cum, batch_y_risk, _, static_context_ids_cpu = _unpack_forecast_batch(raw_batch)
                batch_x = batch_x.to(device, non_blocking=device.type == "cuda")
                batch_y_daily = batch_y_daily.to(device, non_blocking=device.type == "cuda")
                batch_y_cum = batch_y_cum.to(device, non_blocking=device.type == "cuda")
                batch_y_risk = batch_y_risk.to(device, non_blocking=device.type == "cuda")
                static_context_ids = (
                    static_context_ids_cpu.to(device, non_blocking=device.type == "cuda")
                    if static_context_ids_cpu is not None
                    else None
                )
                static_context_ids = dataset_view_or_x.filter_static_context_ids(static_context_ids)
                with _autocast_context(device, amp_enabled):
                    pred = _forecast_model_forward(model, batch_x, static_context_ids)
                    loss = _forecast_loss(
                        pred,
                        batch_y_daily,
                        batch_y_cum,
                        batch_y_risk,
                        loss_profile=loss_profile,
                        target_scale=target_scale,
                        decision_cost_bps=decision_cost_bps,
                        decision_hit_threshold_bps=decision_hit_threshold_bps,
                        decision_drawdown_penalty=decision_drawdown_penalty,
                        cumulative_horizons=dataset_view_or_x.cumulative_horizons,
                    )
                values.append(float(loss.detach().cpu()))
                counts.append(int(batch_x.shape[0]))
            total = sum(counts)
            return float(np.average(values, weights=counts)) if total > 0 else float("inf")
        x = dataset_view_or_x
        assert y_daily is not None and y_cum is not None and y_risk is not None
        for start in range(0, len(indices), max(int(batch_size), 1)):
            batch_idx = torch.tensor(indices[start : start + max(int(batch_size), 1)], dtype=torch.long)
            batch_x = x[batch_idx].to(device, non_blocking=device.type == "cuda")
            batch_y_daily = y_daily[batch_idx].to(device, non_blocking=device.type == "cuda")
            batch_y_cum = y_cum[batch_idx].to(device, non_blocking=device.type == "cuda")
            batch_y_risk = y_risk[batch_idx].to(device, non_blocking=device.type == "cuda")
            with _autocast_context(device, amp_enabled):
                pred = _forecast_model_forward(model, batch_x)
                loss = _forecast_loss(
                    pred,
                    batch_y_daily,
                    batch_y_cum,
                    batch_y_risk,
                    loss_profile=loss_profile,
                    target_scale=target_scale,
                    decision_cost_bps=decision_cost_bps,
                    decision_hit_threshold_bps=decision_hit_threshold_bps,
                    decision_drawdown_penalty=decision_drawdown_penalty,
                    cumulative_horizons=getattr(model, "cumulative_horizons", None),
                )
            values.append(float(loss.detach().cpu()))
            counts.append(int(batch_idx.numel()))
    total = sum(counts)
    return float(np.average(values, weights=counts)) if total > 0 else float("inf")


def _family_summary(seed_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    validation_metrics = [dict(item.get("validation_metrics", {})) for item in seed_summaries.values()]
    completed_metrics = [metrics for metrics in validation_metrics if metrics.get("status") == "completed"]
    q10_values = [
        float(metrics.get("q10_coverage_mean", 0.0) or 0.0)
        for metrics in completed_metrics
    ]
    q90_values = [
        float(metrics.get("q90_coverage_mean", 0.0) or 0.0)
        for metrics in completed_metrics
    ]
    multiscale_scores = [
        float(item.get("validation_multiscale_score", 0.0) or 0.0)
        for item in seed_summaries.values()
        if dict(item.get("validation_metrics", {})).get("status") == "completed"
    ]
    trend_scores = [
        float(item.get("validation_trend20_score", 0.0) or 0.0)
        for item in seed_summaries.values()
        if dict(item.get("validation_metrics", {})).get("status") == "completed"
    ]
    short_scores = [
        float(item.get("validation_short_burst_score", 0.0) or 0.0)
        for item in seed_summaries.values()
        if dict(item.get("validation_metrics", {})).get("status") == "completed"
    ]
    decision_scores = [
        float(item.get("validation_decision_utility_score", 0.0) or 0.0)
        for item in seed_summaries.values()
        if dict(item.get("validation_metrics", {})).get("status") == "completed"
    ]
    validation_loss_scores = [
        float(item.get("validation_loss_score", 0.0) or 0.0)
        for item in seed_summaries.values()
        if dict(item.get("validation_metrics", {})).get("status") == "completed"
    ]
    count = max(len(seed_summaries), 1)
    summary: dict[str, Any] = {
        "seed_count": int(len(seed_summaries)),
        "validation_q10_coverage_mean": float(np.mean(q10_values)) if q10_values else 0.0,
        "validation_q90_coverage_mean": float(np.mean(q90_values)) if q90_values else 0.0,
        "validation_multiscale_score_mean": float(np.mean(multiscale_scores)) if multiscale_scores else 0.0,
        "validation_multiscale_score_std": float(np.std(multiscale_scores, ddof=0)) if multiscale_scores else 0.0,
        "validation_trend20_score_mean": float(np.mean(trend_scores)) if trend_scores else 0.0,
        "validation_short_burst_score_mean": float(np.mean(short_scores)) if short_scores else 0.0,
        "validation_decision_utility_score_mean": float(np.mean(decision_scores)) if decision_scores else 0.0,
        "validation_loss_score_mean": float(np.mean(validation_loss_scores)) if validation_loss_scores else 0.0,
    }
    profile_counts: dict[str, int] = {"trend_20d": 0, "short_burst": 0, "multiscale": 0, "failed": 0}
    for metrics in completed_metrics:
        profile_counts[forecast_signal_profile(metrics)] = profile_counts.get(forecast_signal_profile(metrics), 0) + 1
    summary["validation_signal_profile_counts"] = profile_counts
    observed_horizons = tuple(
        sorted(
            int(match.group(1))
            for metrics in completed_metrics
            for key in metrics
            for match in [re.match(r"rank_ic_(\d+)d$", str(key))]
            if match is not None
        )
    )
    horizons = tuple(dict.fromkeys(observed_horizons)) or PATH20_CUMULATIVE_HORIZONS
    for horizon in horizons:
        rank_values = [
            float(metrics.get(f"rank_ic_{int(horizon)}d", 0.0) or 0.0)
            for metrics in completed_metrics
        ]
        spread_values = [
            float(metrics.get(f"top_bottom_spread_{int(horizon)}d", 0.0) or 0.0)
            for metrics in completed_metrics
        ]
        summary[f"validation_rank_ic_{int(horizon)}d_mean"] = float(np.mean(rank_values)) if rank_values else 0.0
        summary[f"validation_rank_ic_{int(horizon)}d_std"] = float(np.std(rank_values, ddof=0)) if rank_values else 0.0
        summary[f"validation_top_bottom_spread_{int(horizon)}d_mean"] = (
            float(np.mean(spread_values)) if spread_values else 0.0
        )
        summary[f"validation_top_bottom_spread_{int(horizon)}d_std"] = (
            float(np.std(spread_values, ddof=0)) if spread_values else 0.0
        )
        summary[f"validation_rank_ic_{int(horizon)}d_positive_seed_rate"] = (
            float(sum(1 for value in rank_values if value > 0.0) / count)
        )
        summary[f"validation_top_bottom_spread_{int(horizon)}d_positive_seed_rate"] = (
            float(sum(1 for value in spread_values if value > 0.0) / count)
        )
    summary["validation_rank_ic_positive_seed_rate"] = summary.get("validation_rank_ic_20d_positive_seed_rate", 0.0)
    summary["validation_top_bottom_spread_positive_seed_rate"] = summary.get(
        "validation_top_bottom_spread_20d_positive_seed_rate",
        0.0,
    )
    return summary


def _selection_score_key(selection_profile: str) -> str:
    profile = str(selection_profile or "multiscale").strip().lower()
    if profile == "validation_loss":
        return "validation_loss_score"
    if profile == "decision_utility":
        return "validation_decision_utility_score"
    if profile == "trend20":
        return "validation_trend20_score"
    if profile == "short_burst":
        return "validation_short_burst_score"
    return "validation_multiscale_score"


def _select_family_seed(model_summaries: dict[str, dict[str, Any]], *, selection_profile: str) -> tuple[str, int]:
    score_key = _selection_score_key(selection_profile)
    normalized_profile = str(selection_profile or "multiscale").strip().lower()

    def family_score(item: tuple[str, dict[str, Any]]) -> tuple[int, float, float]:
        _, summary = item
        metrics = dict(summary.get("family_summary", {}))
        score = float(metrics.get(f"{score_key}_mean", 0.0) or 0.0)
        rank = float(metrics.get("validation_rank_ic_20d_mean", 0.0) or 0.0)
        spread = float(metrics.get("validation_top_bottom_spread_20d_mean", 0.0) or 0.0)
        q10 = float(metrics.get("validation_q10_coverage_mean", 0.0) or 0.0)
        q90 = float(metrics.get("validation_q90_coverage_mean", 0.0) or 0.0)
        if normalized_profile == "validation_loss":
            passed = 1 if np.isfinite(score) else 0
        else:
            passed = 1 if score > 1_000.0 and 0.65 <= q10 <= 0.95 and 0.65 <= q90 <= 0.95 else 0
        return (passed, score, rank + spread)

    if not model_summaries:
        return "", 0
    selected_family = max(model_summaries.items(), key=family_score)[0]
    seed_summaries = dict(model_summaries[selected_family].get("seed_summaries", {}))
    if not seed_summaries:
        return selected_family, 0

    def seed_score(item: tuple[str, dict[str, Any]]) -> tuple[int, float, float]:
        _, summary = item
        validation = dict(summary.get("validation_metrics", {}))
        verdict = forecast_evidence_verdict(validation_metrics=validation, test_metrics=None, selection_profile=selection_profile)
        passed = 1 if (verdict == "forecast_promising" or normalized_profile == "validation_loss") else 0
        return (
            passed,
            float(summary.get(score_key, 0.0) or 0.0),
            float(validation.get("rank_ic_20d", 0.0) or 0.0),
        )

    selected_seed_text = max(seed_summaries.items(), key=seed_score)[0]
    return selected_family, int(selected_seed_text)


def train_forecast_models(
    dataset: ForecastSequenceDataset | ForecastMemmapDataset | ForecastShardedMemmapDataset | ForecastTrainingPackDataset,
    *,
    study_root: Path,
    model_families: tuple[str, ...] | list[str] = FORECAST_MODEL_FAMILIES,
    epochs: int = 2,
    min_epochs: int = 1,
    early_stop_patience: int = 12,
    early_stop_min_delta: float = 1.0e-4,
    batch_size: int = 512,
    lr: float = 3.0e-4,
    hidden_dim: int = 192,
    dropout: float = 0.15,
    gru_layers: int = 2,
    transformer_layers: int = 4,
    transformer_heads: int = 6,
    patch_sizes: tuple[int, ...] | list[int] = (4, 20),
    device: str | torch.device = "auto",
    amp: bool = True,
    seeds: tuple[int, ...] | list[int] | str | None = None,
    grad_clip: float = 1.0,
    grad_accum_steps: int = 1,
    weight_decay: float = 1.0e-4,
    write_all_predictions: bool = False,
    selection_profile: str = "multiscale",
    target_scale: float = 100.0,
    seed: int = 7,
    dataloader_num_workers: int = 0,
    prefetch_factor: int = 2,
    resume_from: str | Path | None = None,
    save_last_checkpoint: bool = True,
    checkpoint_every_n_epochs: int = 0,
    per_epoch_prediction_metrics: bool = True,
    progress_json_name: str = "forecast_progress.json",
    output_profile: str = "forecast_path_v1",
    loss_profile: str = "default",
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
    static_context_fields_override: tuple[str, ...] | list[str] | str | None = None,
    train_date_stride: int = 1,
    ranking_baseline: str = "none",
    slot_diagnostics: bool = False,
    finite_guard: bool = False,
    bad_batch_dump_dir: str | Path | None = None,
    date_slate_dates_per_batch: int = 1,
    date_slate_stocks_per_date: int = 512,
    rank_min_group_size: int = 8,
    rank_max_pairs_per_date: int = 4096,
) -> dict[str, Any]:
    study_root.mkdir(parents=True, exist_ok=True)
    resolved_device = _resolve_device(device)
    amp_enabled = bool(amp) and resolved_device.type == "cuda"
    seed_values = _normalize_seeds(seeds, default_seed=int(seed))
    families = tuple(str(item).strip() for item in model_families if str(item).strip())
    invalid = sorted(set(families) - set(FORECAST_MODEL_FAMILIES))
    if invalid:
        raise ValueError(f"Unsupported forecast model families: {', '.join(invalid)}")
    resume_path = Path(resume_from) if resume_from is not None and str(resume_from).strip() else None
    resume_payload: dict[str, Any] | None = None
    if resume_path is not None:
        if len(families) != 1 or len(seed_values) != 1:
            raise ValueError("forecast resume requires exactly one model family and one seed.")
        resume_payload = _load_forecast_resume_checkpoint(resume_path)
    selection_profile = str(selection_profile or "multiscale").strip().lower()
    if selection_profile not in FORECAST_SELECTION_PROFILES:
        raise ValueError(f"Unsupported forecast selection profile: {selection_profile}")
    if not bool(per_epoch_prediction_metrics) and selection_profile != "validation_loss":
        raise ValueError("per_epoch_prediction_metrics=False is only supported with selection_profile=validation_loss.")
    output_profile = str(output_profile or "forecast_path_v1").strip().lower()
    if output_profile not in FORECAST_OUTPUT_PROFILES:
        raise ValueError(f"Unsupported forecast output profile: {output_profile}")
    loss_profile = _normalize_forecast_loss_profile(loss_profile)
    if loss_profile in _FORECAST_DECISION_LOSS_PROFILES and output_profile != "decision_utility_v1":
        raise ValueError(f"{loss_profile} loss requires output_profile=decision_utility_v1.")
    if selection_profile == "decision_utility" and output_profile != "decision_utility_v1":
        raise ValueError("decision_utility selection requires output_profile=decision_utility_v1.")
    if loss_profile == "date_grouped_alpha_score_v1" and output_profile != "forecast_incremental_path_v2":
        raise ValueError("date_grouped_alpha_score_v1 requires output_profile=forecast_incremental_path_v2.")
    train_date_stride = int(train_date_stride)
    if train_date_stride <= 0:
        raise ValueError("train_date_stride must be positive.")
    date_slate_dates_per_batch = max(int(date_slate_dates_per_batch), 1)
    date_slate_stocks_per_date = max(int(date_slate_stocks_per_date), 1)
    rank_min_group_size = max(int(rank_min_group_size), 2)
    rank_max_pairs_per_date = max(int(rank_max_pairs_per_date), 1)
    bad_batch_dump_root = Path(bad_batch_dump_dir) if bad_batch_dump_dir is not None and str(bad_batch_dump_dir).strip() else study_root / "bad_batches"
    dataset_view = _ForecastDatasetView(dataset, static_context_fields_override=static_context_fields_override)
    decision_config = {
        "cost_bps": float(decision_cost_bps),
        "hit_threshold_bps": float(decision_hit_threshold_bps),
        "drawdown_penalty": float(decision_drawdown_penalty),
        "horizons": [int(item) for item in dataset_view.cumulative_horizons],
    }
    loss_profile_contract = forecast_loss_profile_contract(
        loss_profile,
        cumulative_horizons=dataset_view.cumulative_horizons,
        forecast_horizon=int(dataset_view.horizon),
    )
    ranking_baseline = str(ranking_baseline or "none").strip().lower()
    if ranking_baseline not in FORECAST_RANKING_BASELINES:
        ranking_baseline_summary = {
            "status": "dependency_missing",
            "baseline": ranking_baseline,
            "reason": "unsupported_ranking_baseline",
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
    else:
        ranking_baseline_summary = {"status": "skipped", "baseline": "none"} if ranking_baseline == "none" else None
    if ranking_baseline_summary is None:
        ranking_baseline_summary = run_forecast_ranking_baseline(dataset, study_root=study_root, baseline=ranking_baseline)
    cross_section_requested = any(family in FORECAST_CROSS_SECTIONAL_MODEL_FAMILIES for family in families)
    if cross_section_requested and dataset_view.dataset_mode != "memmap":
        raise ValueError("stock_mixer_sequence and sector_slot_mixer_sequence require forecast memmap date-level batching.")
    feature_profile = str(dataset_view.manifest.get("feature_profile", ""))
    feature_manifest = dict(dataset_view.manifest.get("feature_manifest", {}))
    if dataset_view.row_count == 0:
        summary = {
            "status": "insufficient_or_incomplete",
            "reason": "empty_forecast_dataset",
            "models": {},
            "family_summary": {},
            "dataset_mode": dataset_view.dataset_mode,
            "feature_profile": feature_profile,
            "feature_manifest": feature_manifest,
            "selected_seed": 0,
            "selected_model_family": "",
            "selected_signal_profile": "failed",
            "validation_multiscale_score": 0.0,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
        write_json(study_root / "forecast_training_summary.json", _json_ready(summary))
        return summary

    x: torch.Tensor | None = None
    y_daily: torch.Tensor | None = None
    y_cum: torch.Tensor | None = None
    y_risk: torch.Tensor | None = None
    if dataset_view.dataset_mode != "memmap":
        eager_dataset = dataset
        assert isinstance(eager_dataset, ForecastSequenceDataset)
        x = torch.as_tensor(eager_dataset.x, dtype=torch.float32)
        y_daily = torch.as_tensor(eager_dataset.y_daily_excess * float(target_scale), dtype=torch.float32)
        y_cum = torch.as_tensor(eager_dataset.y_cum_excess * float(target_scale), dtype=torch.float32)
        y_risk_np = np.stack(
            [
                eager_dataset.y_drawdown_by_horizon,
                eager_dataset.y_worst_by_horizon,
                eager_dataset.y_upside_by_horizon,
            ],
            axis=-1,
        )
        y_risk = torch.as_tensor(y_risk_np * float(target_scale), dtype=torch.float32)
    source_train_indices = dataset_view.role_indices("train")
    validation_indices = dataset_view.role_indices("validation")
    test_indices = dataset_view.role_indices("test")
    train_indices, sampling_config = _apply_train_date_stride(
        dataset_view,
        source_train_indices,
        train_date_stride=int(train_date_stride),
    )
    intraday_indices = _intraday_feature_indices(dataset_view.feature_columns)
    structured_alpha_v2_feature_groups, structured_alpha_v2_feature_group_source = _structured_alpha_v2_feature_groups_for_dataset(dataset_view)
    if len(train_indices) < 2 or len(validation_indices) < 1:
        summary = {
            "status": "insufficient_or_incomplete",
            "reason": "insufficient_role_samples",
            "train_rows": int(len(train_indices)),
            "source_train_rows": int(len(source_train_indices)),
            "validation_rows": int(len(validation_indices)),
            "test_rows": int(len(test_indices)),
            "sampling_config": dict(sampling_config),
            "models": {},
            "family_summary": {},
            "dataset_mode": dataset_view.dataset_mode,
            "feature_profile": feature_profile,
            "feature_manifest": feature_manifest,
            "selected_seed": 0,
            "selected_model_family": "",
            "selected_signal_profile": "failed",
            "validation_multiscale_score": 0.0,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        }
        write_json(study_root / "forecast_training_summary.json", _json_ready(summary))
        return summary

    model_summaries: dict[str, dict[str, Any]] = {}
    learning_rows: list[dict[str, Any]] = []
    if resume_payload is not None and isinstance(resume_payload.get("learning_rows"), list):
        learning_rows = [dict(row) for row in resume_payload.get("learning_rows", []) if isinstance(row, dict)]
    run_started_at = _now_iso_seconds()
    run_started_monotonic = time.monotonic()
    progress_path = study_root / str(progress_json_name or "forecast_progress.json")
    batch_size = max(int(batch_size), 1)
    max_epochs = max(int(epochs), 1)
    min_epochs = max(int(min_epochs), 1)
    patience_limit = max(int(early_stop_patience), 1)
    checkpoint_interval = max(int(checkpoint_every_n_epochs), 0)
    per_epoch_prediction_metrics_enabled = bool(per_epoch_prediction_metrics)
    accum_steps = max(int(grad_accum_steps), 1)
    pin_memory = resolved_device.type == "cuda"
    loader_kwargs = _data_loader_kwargs(
        pin_memory=pin_memory,
        dataloader_num_workers=int(dataloader_num_workers),
        prefetch_factor=int(prefetch_factor),
    )
    static_model_options = _static_context_model_options(dataset_view)
    single_candidate_run = len(families) == 1 and len(seed_values) == 1
    selected_output_cache: dict[str, Any] = {}

    for family in families:
        seed_summaries: dict[str, dict[str, Any]] = {}
        for current_seed in seed_values:
            torch.manual_seed(int(current_seed))
            np.random.seed(int(current_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(current_seed))
            model = make_forecast_model(
                family,
                input_dim=int(dataset_view.input_dim),
                hidden_dim=int(hidden_dim),
                horizon=int(dataset_view.horizon),
                dropout=float(dropout),
                gru_layers=int(gru_layers),
                transformer_layers=int(transformer_layers),
                transformer_heads=int(transformer_heads),
                patch_sizes=tuple(int(item) for item in patch_sizes),
                cumulative_horizons=dataset_view.cumulative_horizons,
                output_profile=output_profile,
                intraday_feature_indices=intraday_indices,
                feature_group_indices=structured_alpha_v2_feature_groups,
                **static_model_options,
            ).to(resolved_device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
            generator = torch.Generator()
            generator.manual_seed(int(current_seed))
            cross_sectional_batching = bool(
                family in FORECAST_CROSS_SECTIONAL_MODEL_FAMILIES and dataset_view.dataset_mode == "memmap"
            )
            date_slate_batching = bool(family == "date_slate_alpha_fusion_v1")
            if date_slate_batching:
                if not isinstance(dataset_view.dataset, ForecastTrainingPackDataset):
                    raise ValueError("date_slate_alpha_fusion_v1 requires a QDP date-slate/training pack dataset.")
                if not dataset_view.dataset.has_date_major_feature_panel:
                    raise ValueError("date_slate_alpha_fusion_v1 requires a date-major feature panel; build qdp_date_slate_training_pack_v1 first.")
            cache_friendly_train_order = bool(
                isinstance(dataset_view.dataset, (ForecastShardedMemmapDataset, ForecastTrainingPackDataset)) and not cross_sectional_batching
            )
            train_loader_indices = (
                dataset_view.dataset.cache_friendly_indices(train_indices)
                if cache_friendly_train_order and not date_slate_batching
                else train_indices
            )
            if date_slate_batching:
                train_loader = DataLoader(
                    dataset_view.date_slate_torch_dataset(
                        train_loader_indices,
                        dates_per_batch=int(date_slate_dates_per_batch),
                        stocks_per_date=int(date_slate_stocks_per_date),
                        target_scale=target_scale,
                        shuffle_stocks=True,
                        seed=int(current_seed),
                    ),
                    batch_size=None,
                    shuffle=True,
                    generator=generator,
                    **loader_kwargs,
                )
            elif cache_friendly_train_order:
                train_loader = DataLoader(
                    dataset_view.batch_torch_dataset(
                        train_loader_indices,
                        batch_size=batch_size,
                        target_scale=target_scale,
                    ),
                    batch_size=None,
                    shuffle=False,
                    **loader_kwargs,
                )
            else:
                train_loader = DataLoader(
                    dataset_view.date_batch_torch_dataset(train_loader_indices, target_scale=target_scale)
                    if cross_sectional_batching
                    else dataset_view.torch_dataset(train_loader_indices, target_scale=target_scale),
                    batch_size=1 if cross_sectional_batching else batch_size,
                    shuffle=True,
                    generator=generator,
                    collate_fn=_collate_forecast_date_batches if cross_sectional_batching else None,
                    **loader_kwargs,
                )
            best_checkpoint_path = study_root / f"forecast_model_{family}_seed{int(current_seed)}_best.pt"
            last_checkpoint_path = study_root / f"forecast_model_{family}_seed{int(current_seed)}_last.pt"
            best_score = -float("inf")
            best_epoch = 0
            best_validation_loss = float("inf")
            best_validation_metrics: dict[str, Any] = {"status": "not_run"}
            best_train_loss = float("inf")
            stopped_reason = "max_epochs_reached"
            patience_used = 0
            last_train_loss = float("inf")
            resume_from_checkpoint_pt = ""
            resume_start_epoch = 1
            best_checkpoint_payload: dict[str, Any] | None = None
            model_config = _model_config_for_training(
                family=family,
                hidden_dim=int(hidden_dim),
                dropout=float(dropout),
                gru_layers=int(gru_layers),
                transformer_layers=int(transformer_layers),
                transformer_heads=int(transformer_heads),
                patch_sizes=tuple(int(item) for item in patch_sizes),
                dataset_view=dataset_view,
                static_model_options=static_model_options,
                output_profile=output_profile,
            )
            optimizer_config = {
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "grad_clip": float(grad_clip),
                "grad_accum_steps": int(accum_steps),
                "loss_profile": str(loss_profile),
            }
            training_config = {
                "epochs": int(max_epochs),
                "min_epochs": int(min_epochs),
                "early_stop_patience": int(patience_limit),
                "early_stop_min_delta": float(early_stop_min_delta),
                "batch_size": int(batch_size),
                "effective_batch_size": int(date_slate_dates_per_batch * date_slate_stocks_per_date if date_slate_batching else (1 if cross_sectional_batching else batch_size)),
                "cross_section_batching_enabled": bool(cross_sectional_batching),
                "date_slate_batching_enabled": bool(date_slate_batching),
                "date_slate_dates_per_batch": int(date_slate_dates_per_batch),
                "date_slate_stocks_per_date": int(date_slate_stocks_per_date),
                "rank_min_group_size": int(rank_min_group_size),
                "rank_max_pairs_per_date": int(rank_max_pairs_per_date),
                "finite_guard_enabled": bool(finite_guard),
                "bad_batch_dump_dir": str(bad_batch_dump_root.resolve()) if bool(finite_guard) else "",
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "grad_clip": float(grad_clip),
                "grad_accum_steps": int(accum_steps),
                "hidden_dim": int(hidden_dim),
                "dropout": float(dropout),
                "gru_layers": int(gru_layers),
                "transformer_layers": int(transformer_layers),
                "transformer_heads": int(transformer_heads),
                "patch_sizes": [int(item) for item in patch_sizes],
                "feature_profile": feature_profile,
                "feature_count": int(dataset_view.input_dim),
                "structured_alpha_v2_feature_group_source": str(structured_alpha_v2_feature_group_source),
                "forecast_horizon": int(dataset_view.horizon),
                "cumulative_horizons": [int(item) for item in dataset_view.cumulative_horizons],
                "selection_profile": selection_profile,
                "output_profile": str(output_profile),
                "loss_profile": str(loss_profile),
                "loss_profile_contract": dict(loss_profile_contract),
                "loss_component_weights": dict(loss_profile_contract["loss_component_weights"]),
                "decision_utility": dict(decision_config),
                "ranking_baseline": str(ranking_baseline),
                "slot_diagnostics": bool(slot_diagnostics),
                "per_epoch_prediction_metrics": bool(per_epoch_prediction_metrics_enabled),
                "sampling_config": dict(sampling_config),
                "static_context_schema": dict(dataset_view.static_context_schema),
                "static_context_source_schema": dict(dataset_view.source_static_context_schema),
                "static_context_source_fields": [str(item) for item in dataset_view.static_context_source_fields],
                "static_context_training_fields": [str(item) for item in dataset_view.static_context_effective_fields],
                "static_context_training_field_indices": [int(item) for item in dataset_view.static_context_field_indices],
                "symbol_vocab_fingerprint": str(dataset_view.symbol_vocab_fingerprint),
                "industry_vocab_fingerprint": str(dataset_view.industry_vocab_fingerprint),
                "board_vocab_fingerprint": str(dataset_view.board_vocab_fingerprint),
                "cross_section_batching_enabled": bool(cross_sectional_batching),
            }
            if resume_payload is not None:
                _validate_forecast_resume_checkpoint(
                    resume_payload,
                    model_family=family,
                    seed=int(current_seed),
                    dataset_view=dataset_view,
                    target_scale=float(target_scale),
                    model_config=model_config,
                    optimizer_config=optimizer_config,
                    selection_profile=selection_profile,
                    loss_profile=loss_profile,
                    output_profile=output_profile,
                    decision_cost_bps=decision_cost_bps,
                    decision_hit_threshold_bps=decision_hit_threshold_bps,
                    decision_drawdown_penalty=decision_drawdown_penalty,
                    sampling_config=dict(sampling_config),
                )
                model.load_state_dict(resume_payload["state_dict"])
                optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
                scaler.load_state_dict(resume_payload["scaler_state_dict"])
                _restore_forecast_rng_state(resume_payload, generator)
                resume_from_checkpoint_pt = str(Path(resume_from).resolve()) if resume_from is not None else ""
                resume_checkpoint_epoch = int(resume_payload.get("epoch", 0))
                resume_start_epoch = resume_checkpoint_epoch + 1
                if resume_start_epoch > max_epochs:
                    if resume_checkpoint_epoch == max_epochs:
                        stopped_reason = "resume_evaluation_only"
                    else:
                        raise ValueError("--forecast-resume-from epoch must be lower than or equal to --forecast-epochs.")
                best_score = float(resume_payload.get("best_score", -float("inf")) or -float("inf"))
                best_epoch = int(resume_payload.get("best_epoch", 0) or 0)
                best_validation_loss = float(resume_payload.get("best_validation_loss", float("inf")) or float("inf"))
                best_validation_metrics = dict(resume_payload.get("best_validation_metrics", {}) or {})
                best_train_loss = float(resume_payload.get("best_train_loss", float("inf")) or float("inf"))
                last_train_loss = float(resume_payload.get("last_train_loss", float("inf")) or float("inf"))
                patience_used = int(resume_payload.get("patience_used", 0) or 0)
                if isinstance(resume_payload.get("best_checkpoint_payload"), dict):
                    best_checkpoint_payload = dict(resume_payload["best_checkpoint_payload"])
                    _save_forecast_checkpoint_atomic(best_checkpoint_path, best_checkpoint_payload)
                else:
                    prior_best_path = Path(str(resume_payload.get("best_checkpoint_pt", "") or ""))
                    if prior_best_path.exists():
                        prior_best_payload = torch.load(prior_best_path, map_location="cpu", weights_only=False)
                        if isinstance(prior_best_payload, dict):
                            best_checkpoint_payload = prior_best_payload
                            _save_forecast_checkpoint_atomic(best_checkpoint_path, best_checkpoint_payload)
            for epoch in range(resume_start_epoch, max_epochs + 1):
                epoch_started_monotonic = time.monotonic()
                last_progress_monotonic = epoch_started_monotonic
                samples_processed_epoch = 0
                total_train_steps = int(len(train_loader))
                model.train()
                epoch_losses: list[float] = []
                epoch_counts: list[int] = []
                optimizer.zero_grad(set_to_none=True)
                for step, raw_batch in enumerate(train_loader, start=1):
                    row_ids_for_guard: torch.Tensor | None = None
                    date_group_ids_for_guard: torch.Tensor | None = None
                    if date_slate_batching:
                        if len(raw_batch) == 7:
                            batch_x_cpu, batch_y_daily_cpu, batch_y_cum_cpu, batch_y_risk_cpu, row_ids_cpu, date_group_ids_cpu, static_context_ids_cpu = raw_batch
                        elif len(raw_batch) == 6:
                            batch_x_cpu, batch_y_daily_cpu, batch_y_cum_cpu, batch_y_risk_cpu, row_ids_cpu, date_group_ids_cpu = raw_batch
                            static_context_ids_cpu = None
                        else:
                            raise ValueError(f"date-slate forecast batch must contain 6 or 7 tensors, got {len(raw_batch)}.")
                        batch_x = batch_x_cpu.to(resolved_device, non_blocking=pin_memory)
                        batch_y_daily = batch_y_daily_cpu.to(resolved_device, non_blocking=pin_memory)
                        batch_y_cum = batch_y_cum_cpu.to(resolved_device, non_blocking=pin_memory)
                        batch_y_risk = batch_y_risk_cpu.to(resolved_device, non_blocking=pin_memory)
                        row_ids_for_guard = row_ids_cpu
                        date_group_ids = date_group_ids_cpu.to(resolved_device, non_blocking=pin_memory)
                        date_group_ids_for_guard = date_group_ids
                        static_context_ids = (
                            static_context_ids_cpu.to(resolved_device, non_blocking=pin_memory)
                            if static_context_ids_cpu is not None
                            else None
                        )
                        static_context_ids = dataset_view.filter_static_context_ids(static_context_ids)
                        _forecast_finite_guard(
                            enabled=bool(finite_guard),
                            dump_dir=bad_batch_dump_root,
                            reason_prefix="train_input",
                            model_family=family,
                            seed=int(current_seed),
                            epoch=int(epoch),
                            step=int(step),
                            role="train",
                            row_ids=row_ids_for_guard,
                            date_group_ids=date_group_ids_for_guard,
                            tensors={"x": batch_x, "y_daily": batch_y_daily, "y_cum": batch_y_cum, "y_risk": batch_y_risk},
                            scaler=scaler,
                        )
                        with _autocast_context(resolved_device, amp_enabled):
                            pred = _forecast_model_forward(model, batch_x, static_context_ids)
                            loss = _date_grouped_forecast_loss(
                                pred,
                                batch_y_daily,
                                batch_y_cum,
                                batch_y_risk,
                                date_group_ids,
                                loss_profile=loss_profile,
                                target_scale=target_scale,
                                cumulative_horizons=dataset_view.cumulative_horizons,
                                rank_min_group_size=int(rank_min_group_size),
                                rank_max_pairs_per_date=int(rank_max_pairs_per_date),
                            )
                        _forecast_finite_guard(
                            enabled=bool(finite_guard),
                            dump_dir=bad_batch_dump_root,
                            reason_prefix="train_forward",
                            model_family=family,
                            seed=int(current_seed),
                            epoch=int(epoch),
                            step=int(step),
                            role="train",
                            row_ids=row_ids_for_guard,
                            date_group_ids=date_group_ids_for_guard,
                            tensors={"prediction": pred, "loss": loss},
                            scaler=scaler,
                        )
                        batch_count = int(batch_x.shape[0])
                    elif cross_sectional_batching:
                        batch_x_cpu, stock_mask_cpu, batch_y_daily_cpu, batch_y_cum_cpu, batch_y_risk_cpu, _, static_context_ids_cpu = raw_batch
                        batch_x = batch_x_cpu.to(resolved_device, non_blocking=pin_memory)
                        stock_mask = stock_mask_cpu.to(resolved_device, non_blocking=pin_memory)
                        batch_y_daily = batch_y_daily_cpu.to(resolved_device, non_blocking=pin_memory)
                        batch_y_cum = batch_y_cum_cpu.to(resolved_device, non_blocking=pin_memory)
                        batch_y_risk = batch_y_risk_cpu.to(resolved_device, non_blocking=pin_memory)
                        static_context_ids = (
                            static_context_ids_cpu.to(resolved_device, non_blocking=pin_memory)
                            if static_context_ids_cpu is not None
                            else None
                        )
                        static_context_ids = dataset_view.filter_static_context_ids(static_context_ids)
                        _forecast_finite_guard(
                            enabled=bool(finite_guard),
                            dump_dir=bad_batch_dump_root,
                            reason_prefix="train_input",
                            model_family=family,
                            seed=int(current_seed),
                            epoch=int(epoch),
                            step=int(step),
                            role="train",
                            row_ids=None,
                            date_group_ids=None,
                            tensors={"x": batch_x, "y_daily": batch_y_daily, "y_cum": batch_y_cum, "y_risk": batch_y_risk},
                            scaler=scaler,
                        )
                        flat_mask = stock_mask.reshape(-1)
                        with _autocast_context(resolved_device, amp_enabled):
                            pred = _forecast_model_forward(model, batch_x, static_context_ids, stock_mask=stock_mask)
                            loss = _forecast_loss(
                                {key: value[flat_mask] for key, value in pred.items()},
                                batch_y_daily.reshape(-1, batch_y_daily.shape[-1])[flat_mask],
                                batch_y_cum.reshape(-1, batch_y_cum.shape[-1])[flat_mask],
                                batch_y_risk.reshape(-1, batch_y_risk.shape[-2], batch_y_risk.shape[-1])[flat_mask],
                                loss_profile=loss_profile,
                                target_scale=target_scale,
                                decision_cost_bps=decision_cost_bps,
                                decision_hit_threshold_bps=decision_hit_threshold_bps,
                                decision_drawdown_penalty=decision_drawdown_penalty,
                                cumulative_horizons=dataset_view.cumulative_horizons,
                            )
                        _forecast_finite_guard(
                            enabled=bool(finite_guard),
                            dump_dir=bad_batch_dump_root,
                            reason_prefix="train_forward",
                            model_family=family,
                            seed=int(current_seed),
                            epoch=int(epoch),
                            step=int(step),
                            role="train",
                            row_ids=None,
                            date_group_ids=None,
                            tensors={"prediction": pred, "loss": loss},
                            scaler=scaler,
                        )
                        batch_count = int(flat_mask.sum().detach().cpu())
                    else:
                        batch_x_cpu, batch_y_daily_cpu, batch_y_cum_cpu, batch_y_risk_cpu, row_ids_cpu, static_context_ids_cpu = _unpack_forecast_batch(raw_batch)
                        batch_x = batch_x_cpu.to(resolved_device, non_blocking=pin_memory)
                        batch_y_daily = batch_y_daily_cpu.to(resolved_device, non_blocking=pin_memory)
                        batch_y_cum = batch_y_cum_cpu.to(resolved_device, non_blocking=pin_memory)
                        batch_y_risk = batch_y_risk_cpu.to(resolved_device, non_blocking=pin_memory)
                        row_ids_for_guard = row_ids_cpu
                        static_context_ids = (
                            static_context_ids_cpu.to(resolved_device, non_blocking=pin_memory)
                            if static_context_ids_cpu is not None
                            else None
                        )
                        static_context_ids = dataset_view.filter_static_context_ids(static_context_ids)
                        _forecast_finite_guard(
                            enabled=bool(finite_guard),
                            dump_dir=bad_batch_dump_root,
                            reason_prefix="train_input",
                            model_family=family,
                            seed=int(current_seed),
                            epoch=int(epoch),
                            step=int(step),
                            role="train",
                            row_ids=row_ids_for_guard,
                            date_group_ids=None,
                            tensors={"x": batch_x, "y_daily": batch_y_daily, "y_cum": batch_y_cum, "y_risk": batch_y_risk},
                            scaler=scaler,
                        )
                        with _autocast_context(resolved_device, amp_enabled):
                            pred = _forecast_model_forward(model, batch_x, static_context_ids)
                            loss = _forecast_loss(
                                pred,
                                batch_y_daily,
                                batch_y_cum,
                                batch_y_risk,
                                loss_profile=loss_profile,
                                target_scale=target_scale,
                                decision_cost_bps=decision_cost_bps,
                                decision_hit_threshold_bps=decision_hit_threshold_bps,
                                decision_drawdown_penalty=decision_drawdown_penalty,
                                cumulative_horizons=dataset_view.cumulative_horizons,
                            )
                        _forecast_finite_guard(
                            enabled=bool(finite_guard),
                            dump_dir=bad_batch_dump_root,
                            reason_prefix="train_forward",
                            model_family=family,
                            seed=int(current_seed),
                            epoch=int(epoch),
                            step=int(step),
                            role="train",
                            row_ids=row_ids_for_guard,
                            date_group_ids=None,
                            tensors={"prediction": pred, "loss": loss},
                            scaler=scaler,
                        )
                        batch_count = int(batch_x.shape[0])
                    epoch_losses.append(float(loss.detach().cpu()))
                    epoch_counts.append(batch_count)
                    samples_processed_epoch += int(batch_count)
                    scaler.scale(loss / float(accum_steps)).backward()
                    if step % accum_steps == 0 or step == len(train_loader):
                        scaler.unscale_(optimizer)
                        _forecast_grad_param_finite_guard(
                            enabled=bool(finite_guard),
                            dump_dir=bad_batch_dump_root,
                            model=model,
                            model_family=family,
                            seed=int(current_seed),
                            epoch=int(epoch),
                            step=int(step),
                            role="train",
                            row_ids=row_ids_for_guard,
                            date_group_ids=date_group_ids_for_guard,
                            scaler=scaler,
                            check_params=False,
                        )
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                        scaler.step(optimizer)
                        scaler.update()
                        _forecast_grad_param_finite_guard(
                            enabled=bool(finite_guard),
                            dump_dir=bad_batch_dump_root,
                            model=model,
                            model_family=family,
                            seed=int(current_seed),
                            epoch=int(epoch),
                            step=int(step),
                            role="train",
                            row_ids=row_ids_for_guard,
                            date_group_ids=date_group_ids_for_guard,
                            scaler=scaler,
                            check_params=True,
                        )
                        optimizer.zero_grad(set_to_none=True)
                    now_monotonic = time.monotonic()
                    if now_monotonic - last_progress_monotonic >= 60.0 or step == len(train_loader):
                        epoch_elapsed = max(float(now_monotonic - epoch_started_monotonic), 1.0e-6)
                        samples_per_second = float(samples_processed_epoch) / epoch_elapsed
                        remaining_steps = max(int(total_train_steps) - int(step), 0)
                        seconds_per_step = epoch_elapsed / max(int(step), 1)
                        _write_forecast_progress(
                            progress_path,
                            status="running",
                            model_family=family,
                            seed=int(current_seed),
                            current_epoch=int(epoch),
                            max_epochs=int(max_epochs),
                            min_epochs=int(min_epochs),
                            patience_limit=int(patience_limit),
                            patience_used=int(patience_used),
                            best_epoch=int(best_epoch),
                            best_score=float(best_score),
                            best_validation_loss=float(best_validation_loss),
                            best_validation_metrics=best_validation_metrics,
                            last_train_loss=float(np.average(epoch_losses, weights=epoch_counts)) if epoch_losses and epoch_counts else float("inf"),
                            epoch_seconds=float(epoch_elapsed),
                            run_started_at=run_started_at,
                            elapsed_seconds=float(now_monotonic - run_started_monotonic),
                            learning_rows=learning_rows,
                            last_checkpoint_pt=last_checkpoint_path if bool(save_last_checkpoint) else None,
                            best_checkpoint_pt=best_checkpoint_path,
                            phase="train_epoch",
                            current_step=int(step),
                            total_steps=int(total_train_steps),
                            samples_processed_epoch=int(samples_processed_epoch),
                            samples_per_second=float(samples_per_second),
                            epoch_eta_seconds=float(remaining_steps * seconds_per_step),
                            throughput_meta={
                                "batch_size": int(batch_size),
                                "batch_count": int(batch_count),
                                "dataset_mode": str(dataset_view.dataset_mode),
                                "dataset_type": type(dataset_view.dataset).__name__,
                                "cache_friendly_train_order": bool(cache_friendly_train_order),
                                "date_slate_batching_enabled": bool(date_slate_batching),
                                "date_slate_dates_per_batch": int(date_slate_dates_per_batch),
                                "date_slate_stocks_per_date": int(date_slate_stocks_per_date),
                            },
                        )
                        last_progress_monotonic = now_monotonic
                last_train_loss = (
                    float(np.average(epoch_losses, weights=epoch_counts)) if epoch_losses and epoch_counts else float("inf")
                )
                validation_started_monotonic = time.monotonic()
                _write_forecast_progress(
                    progress_path,
                    status="running",
                    model_family=family,
                    seed=int(current_seed),
                    current_epoch=int(epoch),
                    max_epochs=int(max_epochs),
                    min_epochs=int(min_epochs),
                    patience_limit=int(patience_limit),
                    patience_used=int(patience_used),
                    best_epoch=int(best_epoch),
                    best_score=float(best_score),
                    best_validation_loss=float(best_validation_loss),
                    best_validation_metrics=best_validation_metrics,
                    last_train_loss=float(last_train_loss),
                    epoch_seconds=float(time.monotonic() - epoch_started_monotonic),
                    run_started_at=run_started_at,
                    elapsed_seconds=float(time.monotonic() - run_started_monotonic),
                    learning_rows=learning_rows,
                    last_checkpoint_pt=last_checkpoint_path if bool(save_last_checkpoint) else None,
                    best_checkpoint_pt=best_checkpoint_path,
                    phase="validation_loss",
                    current_step=int(total_train_steps),
                    total_steps=int(total_train_steps),
                    samples_processed_epoch=int(samples_processed_epoch),
                    samples_per_second=float(samples_processed_epoch / max(time.monotonic() - epoch_started_monotonic, 1.0e-6)),
                    epoch_eta_seconds=0.0,
                    throughput_meta={
                        "train_epoch_seconds_so_far": float(time.monotonic() - epoch_started_monotonic),
                        "validation_sample_count": int(len(validation_indices)),
                    },
                )
                validation_loss = _evaluate_loss(
                    model,
                    dataset_view if dataset_view.dataset_mode == "memmap" else x,
                    y_daily,
                    y_cum,
                    y_risk,
                    validation_indices,
                    batch_size=batch_size,
                    device=resolved_device,
                    amp_enabled=amp_enabled,
                    target_scale=target_scale,
                    loss_profile=loss_profile,
                    decision_cost_bps=decision_cost_bps,
                    decision_hit_threshold_bps=decision_hit_threshold_bps,
                    decision_drawdown_penalty=decision_drawdown_penalty,
                    date_slate_dates_per_batch=int(date_slate_dates_per_batch),
                    date_slate_stocks_per_date=int(date_slate_stocks_per_date),
                    rank_min_group_size=int(rank_min_group_size),
                    rank_max_pairs_per_date=int(rank_max_pairs_per_date),
                )
                if per_epoch_prediction_metrics_enabled:
                    validation_predictions = _predict_indices(
                        model,
                        dataset_view if dataset_view.dataset_mode == "memmap" else x,
                        validation_indices,
                        batch_size=batch_size,
                        device=resolved_device,
                        amp_enabled=amp_enabled,
                        target_scale=target_scale,
                    )
                    validation_frame = _prediction_frame_for_dataset_indices(
                        dataset,
                        indices=validation_indices,
                        predictions=validation_predictions,
                        family=family,
                        target_scale=target_scale,
                        decision_cost_bps=decision_cost_bps,
                        decision_hit_threshold_bps=decision_hit_threshold_bps,
                        decision_drawdown_penalty=decision_drawdown_penalty,
                        loss_profile=loss_profile,
                    )
                    validation_metrics = forecast_prediction_metrics(validation_frame)
                    validation_metrics_mode = "full_prediction_metrics"
                    del validation_predictions, validation_frame
                else:
                    validation_metrics = {
                        "status": "validation_loss_only",
                        "row_count": int(len(validation_indices)),
                        "date_count": int(dataset_view.date_count_for_indices(validation_indices)),
                        "forecast_horizon": int(dataset_view.horizon),
                        "cumulative_horizons": [int(item) for item in dataset_view.cumulative_horizons],
                        "selection_profile": str(selection_profile),
                        "loss_profile": str(loss_profile),
                        "reason": "per_epoch_prediction_metrics_disabled",
                    }
                    validation_metrics_mode = "validation_loss_only"
                score = _validation_score(validation_metrics, validation_loss, (0.65, 0.95), selection_profile)
                multiscale_score = _profile_score(validation_metrics, validation_loss, (0.65, 0.95), "multiscale")
                improved = score > best_score + float(early_stop_min_delta)
                if improved:
                    best_score = score
                    best_epoch = int(epoch)
                    best_validation_loss = float(validation_loss)
                    best_validation_metrics = dict(validation_metrics)
                    best_train_loss = float(last_train_loss)
                    patience_used = 0
                    best_checkpoint_payload = _forecast_checkpoint_payload(
                        checkpoint_kind="best",
                        model=model,
                        optimizer=None,
                        scaler=None,
                        model_family=family,
                        seed=int(current_seed),
                        epoch=int(epoch),
                        best_epoch=int(best_epoch),
                        best_score=float(best_score),
                        best_validation_loss=float(best_validation_loss),
                        best_validation_metrics=best_validation_metrics,
                        best_train_loss=float(best_train_loss),
                        last_train_loss=float(last_train_loss),
                        patience_used=int(patience_used),
                        feature_columns=list(dataset_view.feature_columns),
                        feature_profile=feature_profile,
                        feature_manifest=feature_manifest,
                        normalization_manifest=dataset_view.normalization_manifest,
                        target_scale=float(target_scale),
                        lookback_days=int(dataset_view.lookback_days),
                        horizon=int(dataset_view.horizon),
                        cumulative_horizons=dataset_view.cumulative_horizons,
                        model_config=model_config,
                        training_config=training_config,
                        optimizer_config=optimizer_config,
                        selection_profile=selection_profile,
                        loss_profile=loss_profile,
                        output_profile=output_profile,
                        decision_cost_bps=decision_cost_bps,
                        decision_hit_threshold_bps=decision_hit_threshold_bps,
                        decision_drawdown_penalty=decision_drawdown_penalty,
                        learning_rows=learning_rows,
                        static_context_schema=dataset_view.static_context_schema,
                        symbol_vocab_fingerprint=dataset_view.symbol_vocab_fingerprint,
                        industry_vocab_fingerprint=dataset_view.industry_vocab_fingerprint,
                        board_vocab_fingerprint=dataset_view.board_vocab_fingerprint,
                    )
                    _save_forecast_checkpoint_atomic(best_checkpoint_path, best_checkpoint_payload)
                else:
                    patience_used += 1
                epoch_seconds = float(time.monotonic() - epoch_started_monotonic)
                learning_rows.append(
                    {
                        "model_family": family,
                        "seed": int(current_seed),
                        "epoch": int(epoch),
                        "train_loss": float(last_train_loss),
                        "validation_loss": float(validation_loss),
                        "validation_loss_score": float(_profile_score(validation_metrics, validation_loss, (0.65, 0.95), "validation_loss")),
                        "validation_multiscale_score": float(multiscale_score),
                        "validation_selection_score": float(score),
                        "selected_signal_profile": forecast_signal_profile(validation_metrics),
                        "rank_ic_20d": float(validation_metrics.get("rank_ic_20d", 0.0) or 0.0),
                        "top_bottom_spread_20d": float(validation_metrics.get("top_bottom_spread_20d", 0.0) or 0.0),
                        "rank_ic_5d": float(validation_metrics.get("rank_ic_5d", 0.0) or 0.0),
                        "top_bottom_spread_5d": float(validation_metrics.get("top_bottom_spread_5d", 0.0) or 0.0),
                        "rank_ic_3d": float(validation_metrics.get("rank_ic_3d", 0.0) or 0.0),
                        "top_bottom_spread_3d": float(validation_metrics.get("top_bottom_spread_3d", 0.0) or 0.0),
                        "rank_ic_upside_20d": float(validation_metrics.get("rank_ic_upside_20d", 0.0) or 0.0),
                        "top_bottom_spread_upside_20d": float(
                            validation_metrics.get("top_bottom_spread_upside_20d", 0.0) or 0.0
                        ),
                        "decision_score_rank_ic": float(validation_metrics.get("decision_score_rank_ic", 0.0) or 0.0),
                        "decision_score_top_bottom_spread": float(
                            validation_metrics.get("decision_score_top_bottom_spread", 0.0) or 0.0
                        ),
                        "decision_hit_lift_top20_mean": float(
                            validation_metrics.get("decision_hit_lift_top20_mean", 0.0) or 0.0
                        ),
                        "time_eff_score_rank_ic": float(validation_metrics.get("time_eff_score_rank_ic", 0.0) or 0.0),
                        "time_eff_score_top_bottom_spread": float(
                            validation_metrics.get("time_eff_score_top_bottom_spread", 0.0) or 0.0
                        ),
                        "time_eff_hit_lift_top20_mean": float(
                            validation_metrics.get("time_eff_hit_lift_top20_mean", 0.0) or 0.0
                        ),
                        "q10_coverage_mean": float(validation_metrics.get("q10_coverage_mean", 0.0) or 0.0),
                        "q90_coverage_mean": float(validation_metrics.get("q90_coverage_mean", 0.0) or 0.0),
                        "direction_accuracy_20d": float(validation_metrics.get("direction_accuracy_20d", 0.0) or 0.0),
                        "per_epoch_prediction_metrics": str(validation_metrics_mode),
                        "is_best": bool(improved),
                        "patience_used": int(patience_used),
                        "epoch_seconds": float(epoch_seconds),
                        "train_sample_count": int(samples_processed_epoch),
                        "train_samples_per_second": float(samples_processed_epoch / max(epoch_seconds, 1.0e-6)),
                        "dataset_type": type(dataset_view.dataset).__name__,
                    }
                )
                _write_learning_curve_incremental(study_root / "forecast_learning_curve.csv", learning_rows)
                if bool(save_last_checkpoint):
                    last_payload = _forecast_checkpoint_payload(
                        checkpoint_kind="last",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        model_family=family,
                        seed=int(current_seed),
                        epoch=int(epoch),
                        best_epoch=int(best_epoch),
                        best_score=float(best_score),
                        best_validation_loss=float(best_validation_loss),
                        best_validation_metrics=best_validation_metrics,
                        best_train_loss=float(best_train_loss),
                        last_train_loss=float(last_train_loss),
                        patience_used=int(patience_used),
                        feature_columns=list(dataset_view.feature_columns),
                        feature_profile=feature_profile,
                        feature_manifest=feature_manifest,
                        normalization_manifest=dataset_view.normalization_manifest,
                        target_scale=float(target_scale),
                        lookback_days=int(dataset_view.lookback_days),
                        horizon=int(dataset_view.horizon),
                        cumulative_horizons=dataset_view.cumulative_horizons,
                        model_config=model_config,
                        training_config=training_config,
                        optimizer_config=optimizer_config,
                        selection_profile=selection_profile,
                        loss_profile=loss_profile,
                        output_profile=output_profile,
                        decision_cost_bps=decision_cost_bps,
                        decision_hit_threshold_bps=decision_hit_threshold_bps,
                        decision_drawdown_penalty=decision_drawdown_penalty,
                        learning_rows=learning_rows,
                        static_context_schema=dataset_view.static_context_schema,
                        symbol_vocab_fingerprint=dataset_view.symbol_vocab_fingerprint,
                        industry_vocab_fingerprint=dataset_view.industry_vocab_fingerprint,
                        board_vocab_fingerprint=dataset_view.board_vocab_fingerprint,
                        best_checkpoint_pt=best_checkpoint_path,
                        best_checkpoint_payload=best_checkpoint_payload,
                        rng_state=_forecast_rng_state(generator),
                    )
                    _save_forecast_checkpoint_atomic(last_checkpoint_path, last_payload)
                if checkpoint_interval > 0 and int(epoch) % checkpoint_interval == 0:
                    epoch_checkpoint_path = study_root / f"forecast_model_{family}_seed{int(current_seed)}_epoch{int(epoch)}.pt"
                    epoch_payload = _forecast_checkpoint_payload(
                        checkpoint_kind="epoch",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        model_family=family,
                        seed=int(current_seed),
                        epoch=int(epoch),
                        best_epoch=int(best_epoch),
                        best_score=float(best_score),
                        best_validation_loss=float(best_validation_loss),
                        best_validation_metrics=best_validation_metrics,
                        best_train_loss=float(best_train_loss),
                        last_train_loss=float(last_train_loss),
                        patience_used=int(patience_used),
                        feature_columns=list(dataset_view.feature_columns),
                        feature_profile=feature_profile,
                        feature_manifest=feature_manifest,
                        normalization_manifest=dataset_view.normalization_manifest,
                        target_scale=float(target_scale),
                        lookback_days=int(dataset_view.lookback_days),
                        horizon=int(dataset_view.horizon),
                        cumulative_horizons=dataset_view.cumulative_horizons,
                        model_config=model_config,
                        training_config=training_config,
                        optimizer_config=optimizer_config,
                        selection_profile=selection_profile,
                        loss_profile=loss_profile,
                        output_profile=output_profile,
                        decision_cost_bps=decision_cost_bps,
                        decision_hit_threshold_bps=decision_hit_threshold_bps,
                        decision_drawdown_penalty=decision_drawdown_penalty,
                        learning_rows=learning_rows,
                        static_context_schema=dataset_view.static_context_schema,
                        symbol_vocab_fingerprint=dataset_view.symbol_vocab_fingerprint,
                        industry_vocab_fingerprint=dataset_view.industry_vocab_fingerprint,
                        board_vocab_fingerprint=dataset_view.board_vocab_fingerprint,
                        best_checkpoint_pt=best_checkpoint_path,
                        best_checkpoint_payload=best_checkpoint_payload,
                        rng_state=_forecast_rng_state(generator),
                    )
                    _save_forecast_checkpoint_atomic(epoch_checkpoint_path, epoch_payload)
                _write_forecast_progress(
                    progress_path,
                    status="running",
                    model_family=family,
                    seed=int(current_seed),
                    current_epoch=int(epoch),
                    max_epochs=int(max_epochs),
                    min_epochs=int(min_epochs),
                    patience_limit=int(patience_limit),
                    patience_used=int(patience_used),
                    best_epoch=int(best_epoch),
                    best_score=float(best_score),
                    best_validation_loss=float(best_validation_loss),
                    best_validation_metrics=best_validation_metrics,
                    last_train_loss=float(last_train_loss),
                    epoch_seconds=float(epoch_seconds),
                    run_started_at=run_started_at,
                    elapsed_seconds=float(time.monotonic() - run_started_monotonic),
                    learning_rows=learning_rows,
                    last_checkpoint_pt=last_checkpoint_path if bool(save_last_checkpoint) else None,
                    best_checkpoint_pt=best_checkpoint_path,
                    phase="epoch_end",
                    current_step=int(total_train_steps),
                    total_steps=int(total_train_steps),
                    samples_processed_epoch=int(samples_processed_epoch),
                    samples_per_second=float(samples_processed_epoch / max(epoch_seconds, 1.0e-6)),
                    epoch_eta_seconds=0.0,
                    throughput_meta={
                        "batch_size": int(batch_size),
                        "dataset_mode": str(dataset_view.dataset_mode),
                        "dataset_type": type(dataset_view.dataset).__name__,
                        "cache_friendly_train_order": bool(cache_friendly_train_order),
                        "validation_loss": float(validation_loss),
                        "validation_phase_started_after_seconds": float(validation_started_monotonic - epoch_started_monotonic),
                    },
                )
                if epoch >= min_epochs and patience_used >= patience_limit:
                    stopped_reason = "early_stopping_patience_exhausted"
                    break
            if not best_checkpoint_path.exists():
                best_checkpoint_payload = _forecast_checkpoint_payload(
                    checkpoint_kind="best",
                    model=model,
                    optimizer=None,
                    scaler=None,
                    model_family=family,
                    seed=int(current_seed),
                    epoch=int(max_epochs),
                    best_epoch=int(best_epoch or max_epochs),
                    best_score=float(best_score),
                    best_validation_loss=float(best_validation_loss if np.isfinite(best_validation_loss) else last_train_loss),
                    best_validation_metrics=best_validation_metrics,
                    best_train_loss=float(best_train_loss if np.isfinite(best_train_loss) else last_train_loss),
                    last_train_loss=float(last_train_loss),
                    patience_used=int(patience_used),
                    feature_columns=list(dataset_view.feature_columns),
                    feature_profile=feature_profile,
                    feature_manifest=feature_manifest,
                    normalization_manifest=dataset_view.normalization_manifest,
                    target_scale=float(target_scale),
                    lookback_days=int(dataset_view.lookback_days),
                    horizon=int(dataset_view.horizon),
                    cumulative_horizons=dataset_view.cumulative_horizons,
                    model_config=model_config,
                    training_config=training_config,
                    optimizer_config=optimizer_config,
                    selection_profile=selection_profile,
                    loss_profile=loss_profile,
                    output_profile=output_profile,
                    decision_cost_bps=decision_cost_bps,
                    decision_hit_threshold_bps=decision_hit_threshold_bps,
                    decision_drawdown_penalty=decision_drawdown_penalty,
                    learning_rows=learning_rows,
                    static_context_schema=dataset_view.static_context_schema,
                    symbol_vocab_fingerprint=dataset_view.symbol_vocab_fingerprint,
                    industry_vocab_fingerprint=dataset_view.industry_vocab_fingerprint,
                    board_vocab_fingerprint=dataset_view.board_vocab_fingerprint,
                )
                _save_forecast_checkpoint_atomic(best_checkpoint_path, best_checkpoint_payload)
            checkpoint = torch.load(best_checkpoint_path, map_location=resolved_device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            validation_predictions = _predict_indices(
                model,
                dataset_view if dataset_view.dataset_mode == "memmap" else x,
                validation_indices,
                batch_size=batch_size,
                device=resolved_device,
                amp_enabled=amp_enabled,
                target_scale=target_scale,
            )
            validation_frame = _prediction_frame_for_dataset_indices(
                dataset,
                indices=validation_indices,
                predictions=validation_predictions,
                family=family,
                target_scale=target_scale,
                decision_cost_bps=decision_cost_bps,
                decision_hit_threshold_bps=decision_hit_threshold_bps,
                decision_drawdown_penalty=decision_drawdown_penalty,
                loss_profile=loss_profile,
            )
            final_validation_metrics = forecast_prediction_metrics(validation_frame)
            if bool(write_all_predictions):
                _write_frame(
                    study_root / f"forecast_predictions_validation_{family}_seed{int(current_seed)}.csv",
                    validation_frame,
                )
            if single_candidate_run:
                selected_output_cache["model_family"] = str(family)
                selected_output_cache["seed"] = int(current_seed)
                selected_output_cache["validation_csv"] = _write_frame(
                    study_root / "forecast_predictions_validation.csv",
                    validation_frame,
                )
                selected_output_cache["validation_stratified_metrics"] = stratified_forecast_prediction_metrics(validation_frame)
            del validation_predictions, validation_frame

            test_predictions = _predict_indices(
                model,
                dataset_view if dataset_view.dataset_mode == "memmap" else x,
                test_indices,
                batch_size=batch_size,
                device=resolved_device,
                amp_enabled=amp_enabled,
                target_scale=target_scale,
            )
            test_frame = _prediction_frame_for_dataset_indices(
                dataset,
                indices=test_indices,
                predictions=test_predictions,
                family=family,
                target_scale=target_scale,
                decision_cost_bps=decision_cost_bps,
                decision_hit_threshold_bps=decision_hit_threshold_bps,
                decision_drawdown_penalty=decision_drawdown_penalty,
                loss_profile=loss_profile,
            )
            final_test_metrics = forecast_prediction_metrics(test_frame)
            if bool(write_all_predictions):
                _write_frame(
                    study_root / f"forecast_predictions_test_{family}_seed{int(current_seed)}.csv",
                    test_frame,
                )
            if single_candidate_run:
                selected_output_cache["test_csv"] = _write_frame(
                    study_root / "forecast_predictions_test.csv",
                    test_frame,
                )
                selected_output_cache["test_stratified_metrics"] = stratified_forecast_prediction_metrics(test_frame)
            del test_predictions, test_frame
            slot_diagnostics_path = ""
            if bool(slot_diagnostics) and family == "sector_slot_mixer_sequence":
                slot_diagnostics_path = _write_slot_diagnostics(
                    study_root / f"forecast_slot_diagnostics_{family}_seed{int(current_seed)}.json",
                    model=model,
                    dataset=dataset,
                    indices=validation_indices,
                    seed=int(current_seed),
                )
            final_multiscale_score = _profile_score(final_validation_metrics, best_validation_loss, (0.65, 0.95), "multiscale")
            final_trend20_score = _profile_score(final_validation_metrics, best_validation_loss, (0.65, 0.95), "trend20")
            final_short_burst_score = _profile_score(
                final_validation_metrics,
                best_validation_loss,
                (0.65, 0.95),
                "short_burst",
            )
            final_decision_utility_score = _profile_score(
                final_validation_metrics,
                best_validation_loss,
                (0.65, 0.95),
                "decision_utility",
            )
            final_validation_loss_score = _profile_score(
                final_validation_metrics,
                best_validation_loss,
                (0.65, 0.95),
                "validation_loss",
            )
            selection_score_by_profile = {
                "multiscale": final_multiscale_score,
                "trend20": final_trend20_score,
                "short_burst": final_short_burst_score,
                "decision_utility": final_decision_utility_score,
                "validation_loss": final_validation_loss_score,
            }
            seed_summaries[str(int(current_seed))] = {
                "status": "completed",
                "model_family": family,
                "seed": int(current_seed),
                "epochs_ran": int(
                    max(row["epoch"] for row in learning_rows if row["model_family"] == family and row["seed"] == int(current_seed))
                ),
                "best_epoch": int(best_epoch),
                "stopped_reason": stopped_reason,
                "final_train_loss": float(last_train_loss),
                "best_train_loss": float(best_train_loss),
                "best_validation_loss": float(best_validation_loss),
                "best_score": float(best_score),
                "best_validation_metrics": best_validation_metrics,
                "validation_metrics": final_validation_metrics,
                "test_metrics": final_test_metrics,
                "validation_signal_profile": forecast_signal_profile(final_validation_metrics),
                "validation_multiscale_score": float(final_multiscale_score),
                "validation_trend20_score": float(final_trend20_score),
                "validation_short_burst_score": float(final_short_burst_score),
                "validation_decision_utility_score": float(final_decision_utility_score),
                "validation_loss_score": float(final_validation_loss_score),
                "validation_time_eff_score": float(final_validation_metrics.get("time_eff_score_rank_ic", 0.0) or 0.0),
                "validation_selection_score": float(selection_score_by_profile[selection_profile]),
                "checkpoint_pt": str(best_checkpoint_path.resolve()),
                "last_checkpoint_pt": str(last_checkpoint_path.resolve()) if last_checkpoint_path.exists() else "",
                "resume_from_checkpoint_pt": resume_from_checkpoint_pt,
                "resume_start_epoch": int(resume_start_epoch),
                "slot_diagnostics_path": slot_diagnostics_path,
            }
        model_summaries[family] = {
            "status": "completed",
            "model_family": family,
            "epochs": int(max_epochs),
            "min_epochs": int(min_epochs),
            "early_stop_patience": int(patience_limit),
            "batch_size": int(batch_size),
            "effective_batch_size": int(
                date_slate_dates_per_batch * date_slate_stocks_per_date
                if family == "date_slate_alpha_fusion_v1"
                else (1 if family in FORECAST_CROSS_SECTIONAL_MODEL_FAMILIES else batch_size)
            ),
            "date_slate_batching_enabled": bool(family == "date_slate_alpha_fusion_v1"),
            "date_slate_dates_per_batch": int(date_slate_dates_per_batch),
            "date_slate_stocks_per_date": int(date_slate_stocks_per_date),
            "rank_min_group_size": int(rank_min_group_size),
            "rank_max_pairs_per_date": int(rank_max_pairs_per_date),
            "finite_guard_enabled": bool(finite_guard),
            "lr": float(lr),
            "hidden_dim": int(hidden_dim),
            "feature_count": int(dataset_view.input_dim),
            "selection_profile": selection_profile,
            "output_profile": str(output_profile),
            "loss_profile": str(loss_profile),
            "loss_profile_contract": dict(loss_profile_contract),
            "loss_component_weights": dict(loss_profile_contract["loss_component_weights"]),
            "decision_utility": dict(decision_config),
            "slot_diagnostics": bool(slot_diagnostics),
            "per_epoch_prediction_metrics": bool(per_epoch_prediction_metrics_enabled),
            "train_rows": int(len(train_indices)),
            "source_train_rows": int(len(source_train_indices)),
            "validation_rows": int(len(validation_indices)),
            "test_rows": int(len(test_indices)),
            "sampling_config": dict(sampling_config),
            "static_context_schema": dict(dataset_view.static_context_schema),
            "symbol_vocab_fingerprint": str(dataset_view.symbol_vocab_fingerprint),
            "industry_vocab_fingerprint": str(dataset_view.industry_vocab_fingerprint),
            "board_vocab_fingerprint": str(dataset_view.board_vocab_fingerprint),
            "cross_section_batching_enabled": bool(family in FORECAST_CROSS_SECTIONAL_MODEL_FAMILIES),
            "seed_summaries": seed_summaries,
            "family_summary": _family_summary(seed_summaries),
        }

    learning_curve_path = study_root / "forecast_learning_curve.csv"
    _write_frame(learning_curve_path, pd.DataFrame(learning_rows))
    selected_family, selected_seed = _select_family_seed(model_summaries, selection_profile=selection_profile)
    selected_seed_summary = dict(
        model_summaries.get(selected_family, {}).get("seed_summaries", {}).get(str(int(selected_seed)), {})
    )
    selected_checkpoint_path = str(selected_seed_summary.get("checkpoint_pt", "") or "")
    validation_csv = ""
    test_csv = ""
    validation_stratified_metrics: dict[str, Any] = {}
    test_stratified_metrics: dict[str, Any] = {}
    cache_matches_selected = (
        bool(selected_output_cache)
        and str(selected_output_cache.get("model_family", "")) == str(selected_family)
        and int(selected_output_cache.get("seed", -1)) == int(selected_seed)
        and str(selected_output_cache.get("validation_csv", "")).strip()
        and str(selected_output_cache.get("test_csv", "")).strip()
    )
    if cache_matches_selected:
        validation_csv = str(selected_output_cache.get("validation_csv", ""))
        test_csv = str(selected_output_cache.get("test_csv", ""))
        validation_stratified_metrics = dict(selected_output_cache.get("validation_stratified_metrics", {}) or {})
        test_stratified_metrics = dict(selected_output_cache.get("test_stratified_metrics", {}) or {})
    elif selected_family and selected_seed and selected_checkpoint_path:
        selected_model = make_forecast_model(
            selected_family,
            input_dim=int(dataset_view.input_dim),
            hidden_dim=int(hidden_dim),
            horizon=int(dataset_view.horizon),
            dropout=float(dropout),
            gru_layers=int(gru_layers),
            transformer_layers=int(transformer_layers),
            transformer_heads=int(transformer_heads),
            patch_sizes=tuple(int(item) for item in patch_sizes),
            cumulative_horizons=dataset_view.cumulative_horizons,
            output_profile=output_profile,
            intraday_feature_indices=intraday_indices,
            feature_group_indices=structured_alpha_v2_feature_groups,
            **static_model_options,
        ).to(resolved_device)
        checkpoint = torch.load(selected_checkpoint_path, map_location=resolved_device, weights_only=False)
        selected_model.load_state_dict(checkpoint["state_dict"])
        validation_predictions_np = _predict_indices(
            selected_model,
            dataset_view if dataset_view.dataset_mode == "memmap" else x,
            validation_indices,
            batch_size=batch_size,
            device=resolved_device,
            amp_enabled=amp_enabled,
            target_scale=target_scale,
        )
        selected_predictions_validation = _prediction_frame_for_dataset_indices(
            dataset,
            indices=validation_indices,
            predictions=validation_predictions_np,
            family=selected_family,
            target_scale=target_scale,
            decision_cost_bps=decision_cost_bps,
            decision_hit_threshold_bps=decision_hit_threshold_bps,
            decision_drawdown_penalty=decision_drawdown_penalty,
            loss_profile=loss_profile,
        )
        validation_csv = _write_frame(study_root / "forecast_predictions_validation.csv", selected_predictions_validation)
        validation_stratified_metrics = stratified_forecast_prediction_metrics(selected_predictions_validation)
        del validation_predictions_np, selected_predictions_validation

        test_predictions_np = _predict_indices(
            selected_model,
            dataset_view if dataset_view.dataset_mode == "memmap" else x,
            test_indices,
            batch_size=batch_size,
            device=resolved_device,
            amp_enabled=amp_enabled,
            target_scale=target_scale,
        )
        selected_predictions_test = _prediction_frame_for_dataset_indices(
            dataset,
            indices=test_indices,
            predictions=test_predictions_np,
            family=selected_family,
            target_scale=target_scale,
            decision_cost_bps=decision_cost_bps,
            decision_hit_threshold_bps=decision_hit_threshold_bps,
            decision_drawdown_penalty=decision_drawdown_penalty,
            loss_profile=loss_profile,
        )
        test_csv = _write_frame(study_root / "forecast_predictions_test.csv", selected_predictions_test)
        test_stratified_metrics = stratified_forecast_prediction_metrics(selected_predictions_test)
        del test_predictions_np, selected_predictions_test
    else:
        validation_csv = _write_frame(study_root / "forecast_predictions_validation.csv", pd.DataFrame())
        test_csv = _write_frame(study_root / "forecast_predictions_test.csv", pd.DataFrame())
    validation_metrics = dict(selected_seed_summary.get("validation_metrics", {}))
    test_metrics = dict(selected_seed_summary.get("test_metrics", {}))
    verdict = forecast_evidence_verdict(
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        selection_profile=selection_profile,
    )
    selected_signal_profile = forecast_signal_profile(validation_metrics)
    if selection_profile == "validation_loss":
        selection_rule = "lowest_validation_loss_for_training_loss_profile_then_seed_score"
    else:
        selection_rule = f"{selection_profile}_validation_score_after_profile_and_coverage_gates_then_seed_score"
    summary = {
        "status": "completed",
        "stage": "forecast_train",
        "target_scale": float(target_scale),
        "device": str(resolved_device),
        "amp_enabled": bool(amp_enabled),
        "seeds": [int(item) for item in seed_values],
        "feature_profile": feature_profile,
        "feature_manifest": feature_manifest,
        "training_config": {
            "epochs": int(max_epochs),
            "min_epochs": int(min_epochs),
            "early_stop_patience": int(patience_limit),
            "early_stop_min_delta": float(early_stop_min_delta),
            "batch_size": int(batch_size),
            "effective_batch_size": int(1 if selected_family in FORECAST_CROSS_SECTIONAL_MODEL_FAMILIES else batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "grad_clip": float(grad_clip),
            "grad_accum_steps": int(accum_steps),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "gru_layers": int(gru_layers),
            "transformer_layers": int(transformer_layers),
            "transformer_heads": int(transformer_heads),
            "patch_sizes": [int(item) for item in patch_sizes],
            "feature_profile": feature_profile,
            "feature_count": int(dataset_view.input_dim),
            "structured_alpha_v2_feature_group_source": str(structured_alpha_v2_feature_group_source),
            "forecast_horizon": int(dataset_view.horizon),
            "cumulative_horizons": [int(item) for item in dataset_view.cumulative_horizons],
            "selection_profile": selection_profile,
            "output_profile": str(output_profile),
            "loss_profile": str(loss_profile),
            "loss_profile_contract": dict(loss_profile_contract),
            "loss_component_weights": dict(loss_profile_contract["loss_component_weights"]),
            "decision_utility": dict(decision_config),
            "ranking_baseline": str(ranking_baseline),
            "slot_diagnostics": bool(slot_diagnostics),
            "per_epoch_prediction_metrics": bool(per_epoch_prediction_metrics_enabled),
            "save_last_checkpoint": bool(save_last_checkpoint),
            "checkpoint_every_n_epochs": int(checkpoint_interval),
            "resume_from_checkpoint_pt": str(Path(resume_from).resolve()) if resume_from is not None and str(resume_from).strip() else "",
            "sampling_config": dict(sampling_config),
            "static_context_schema": dict(dataset_view.static_context_schema),
            "static_context_source_schema": dict(dataset_view.source_static_context_schema),
            "static_context_source_fields": [str(item) for item in dataset_view.static_context_source_fields],
            "static_context_training_fields": [str(item) for item in dataset_view.static_context_effective_fields],
            "static_context_training_field_indices": [int(item) for item in dataset_view.static_context_field_indices],
            "symbol_vocab_fingerprint": str(dataset_view.symbol_vocab_fingerprint),
            "industry_vocab_fingerprint": str(dataset_view.industry_vocab_fingerprint),
            "board_vocab_fingerprint": str(dataset_view.board_vocab_fingerprint),
            "cross_section_batching_enabled": bool(
                selected_family in FORECAST_CROSS_SECTIONAL_MODEL_FAMILIES if selected_family else False
            ),
        },
        "models": model_summaries,
        "ranking_baseline_summary": ranking_baseline_summary,
        "family_summary": {family: dict(summary.get("family_summary", {})) for family, summary in model_summaries.items()},
        "selected_model_family": selected_family,
        "selected_seed": int(selected_seed),
        "selected_checkpoint_pt": selected_checkpoint_path,
        "selection_rule": selection_rule,
        "selected_signal_profile": selected_signal_profile,
        "validation_multiscale_score": float(selected_seed_summary.get("validation_multiscale_score", 0.0) or 0.0),
        "validation_decision_utility_score": float(
            selected_seed_summary.get("validation_decision_utility_score", 0.0) or 0.0
        ),
        "validation_loss_score": float(selected_seed_summary.get("validation_loss_score", 0.0) or 0.0),
        "validation_selection_score": float(selected_seed_summary.get("validation_selection_score", 0.0) or 0.0),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "loss_profile_contract": dict(loss_profile_contract),
        "loss_component_weights": dict(loss_profile_contract["loss_component_weights"]),
        "test_interpretable": verdict in {"forecast_test_confirmed", "forecast_promising"}
        and validation_metrics.get("status") == "completed"
        and (selection_profile == "validation_loss" or _profile_pass(validation_metrics, selection_profile)),
        "evidence_verdict": verdict,
        "forecast_predictions_validation_csv": validation_csv,
        "forecast_predictions_test_csv": test_csv,
        "forecast_learning_curve_csv": str(learning_curve_path.resolve()),
        "progress_json": str(progress_path.resolve()),
        "resume_from_checkpoint_pt": str(Path(resume_from).resolve()) if resume_from is not None and str(resume_from).strip() else "",
        "dataset_mode": dataset_view.dataset_mode,
        "validation_stratified_metrics": validation_stratified_metrics,
        "test_stratified_metrics": test_stratified_metrics,
        "shadow_only": True,
        "promotion_allowed": False,
        "active_execution_strategy_expected_diff": "none",
    }
    selected_rows = [
        row
        for row in learning_rows
        if str(row.get("model_family")) == str(selected_family) and int(row.get("seed", -1)) == int(selected_seed)
    ]
    if selected_rows:
        latest_selected_row = max(selected_rows, key=lambda row: int(row.get("epoch", 0) or 0))
        selected_last_path_text = str(selected_seed_summary.get("last_checkpoint_pt", "") or "")
        _write_forecast_progress(
            progress_path,
            status="completed",
            model_family=str(selected_family),
            seed=int(selected_seed),
            current_epoch=int(selected_seed_summary.get("epochs_ran", latest_selected_row.get("epoch", 0)) or 0),
            max_epochs=int(max_epochs),
            min_epochs=int(min_epochs),
            patience_limit=int(patience_limit),
            patience_used=int(latest_selected_row.get("patience_used", 0) or 0),
            best_epoch=int(selected_seed_summary.get("best_epoch", 0) or 0),
            best_score=float(selected_seed_summary.get("best_score", 0.0) or 0.0),
            best_validation_loss=float(selected_seed_summary.get("best_validation_loss", 0.0) or 0.0),
            best_validation_metrics=dict(selected_seed_summary.get("best_validation_metrics", {}) or {}),
            last_train_loss=float(selected_seed_summary.get("final_train_loss", 0.0) or 0.0),
            epoch_seconds=float(latest_selected_row.get("epoch_seconds", 0.0) or 0.0),
            run_started_at=run_started_at,
            elapsed_seconds=float(time.monotonic() - run_started_monotonic),
            learning_rows=learning_rows,
            last_checkpoint_pt=Path(selected_last_path_text) if selected_last_path_text else None,
            best_checkpoint_pt=Path(selected_checkpoint_path) if selected_checkpoint_path else None,
            phase="completed",
            samples_processed_epoch=int(latest_selected_row.get("train_sample_count", 0) or 0),
            samples_per_second=float(latest_selected_row.get("train_samples_per_second", 0.0) or 0.0),
            throughput_meta={
                "batch_size": int(batch_size),
                "dataset_mode": str(dataset_view.dataset_mode),
                "dataset_type": type(dataset_view.dataset).__name__,
                "selected_model_family": str(selected_family),
                "final_epoch_seconds": float(latest_selected_row.get("epoch_seconds", 0.0) or 0.0),
                "total_elapsed_seconds": float(time.monotonic() - run_started_monotonic),
            },
        )
    summary = _json_ready(summary)
    write_json(study_root / "forecast_training_summary.json", summary)
    return summary
