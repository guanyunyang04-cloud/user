from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer

    _CVXPY_LAYER_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency boundary
    cp = None
    CvxpyLayer = None
    _CVXPY_LAYER_IMPORT_ERROR = exc

from daily_research.continuous_policy.model_v2 import (
    ACTION_CLASSES,
    DURATION_CLASSES,
    HOLDING_DAYS_BY_BUCKET,
    _action_weights,
    _apply_matrix,
    _daily_feature_scalar,
    _finite_scalar,
    _prepare_matrix,
    _resolve_device,
    _save_checkpoint,
    _signature_hash,
    _signature_payload,
    _split_indices,
    resolve_decoder_profile,
)
from daily_research.continuous_policy.state_builder import STATE_SEQUENCE_BASES, STATE_SEQUENCE_LAGS
from daily_research.continuous_policy.training_contracts import TRAINER_BACKEND_FORMAL_SEQ_V3


SEQUENCE_STEP_ORDER: tuple[int, ...] = tuple(sorted(STATE_SEQUENCE_LAGS, reverse=True)) + (0,)
MAX_CONTINUOUS_HOLDING_DAYS = 20.0
CVXPY_CONVEX_ALLOCATION_SLOT_COUNT = 16
CVXPY_CONVEX_ALLOCATION_MAX_DAYS_PER_BATCH = 4
CVXPY_FULL_UNIVERSE_ALLOCATION_SLOT_COUNT = 32
CVXPY_FULL_UNIVERSE_ALLOCATION_MAX_DAYS_PER_BATCH = 1
CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_BATCH_INTERVAL = 2
CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_SOLVER_ENABLED = False
CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_SOLVER_LOSS_PROFILES = frozenset(
    {"alpha_result_value_budget_split_v35"}
)
CVXPY_CONVEX_ALLOCATION_SOLVER_ARGS: dict[str, float | int | str | bool] = {
    "solve_method": "SCS",
    "eps": 1.0e-4,
    "max_iters": 1200,
    "verbose": False,
}
_CVXPY_ALLOCATION_LAYER_CACHE: dict[int, Any] = {}
THREAD_LIMIT_ENV_KEYS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
)
LOSS_PROFILE_CONFIGS: dict[str, dict[str, dict[str, float]]] = {
    "dual_channel_default_v1": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.30,
            "entry_quality": 0.60,
            "hold_quality": 1.00,
            "add_quality": 0.70,
            "reduce_quality": 0.95,
            "exit_urgency": 1.10,
            "reentry_readiness": 0.45,
            "holding_days_ratio": 1.35,
            "reduce_fraction": 1.30,
            "exit_hazard": 1.40,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.05,
            "candidate_budget": 0.60,
            "turnover_budget": 0.90,
            "max_position_weight_target": 0.55,
            "hold_bias_target": 1.15,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.55,
            "action_soft": 0.45,
            "action_total": 0.72,
            "duration_total": 0.18,
            "scalar_total": 1.18,
            "daily_total": 0.35,
        },
    },
    "teacher_aux_continuous_v1": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.55,
            "entry_quality": 0.72,
            "hold_quality": 1.05,
            "add_quality": 0.82,
            "reduce_quality": 1.05,
            "exit_urgency": 1.20,
            "reentry_readiness": 0.40,
            "holding_days_ratio": 1.55,
            "reduce_fraction": 1.55,
            "exit_hazard": 1.65,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.20,
            "candidate_budget": 0.55,
            "turnover_budget": 1.00,
            "max_position_weight_target": 0.50,
            "hold_bias_target": 1.05,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.50,
            "action_soft": 0.50,
            "action_total": 0.60,
            "duration_total": 0.12,
            "scalar_total": 1.35,
            "daily_total": 0.48,
        },
    },
    "teacher_aux_return_recovery_v1": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.75,
            "entry_quality": 0.95,
            "hold_quality": 1.10,
            "add_quality": 0.95,
            "reduce_quality": 1.10,
            "exit_urgency": 1.18,
            "reentry_readiness": 0.52,
            "holding_days_ratio": 1.45,
            "reduce_fraction": 1.45,
            "exit_hazard": 1.52,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.35,
            "candidate_budget": 0.50,
            "turnover_budget": 1.10,
            "max_position_weight_target": 0.48,
            "hold_bias_target": 0.98,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.48,
            "action_soft": 0.52,
            "action_total": 0.56,
            "duration_total": 0.10,
            "scalar_total": 1.40,
            "daily_total": 0.56,
        },
    },
    "teacher_aux_return_recovery_balanced_v2": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.62,
            "entry_quality": 0.86,
            "hold_quality": 1.08,
            "add_quality": 0.88,
            "reduce_quality": 1.08,
            "exit_urgency": 1.16,
            "reentry_readiness": 0.48,
            "holding_days_ratio": 1.42,
            "reduce_fraction": 1.46,
            "exit_hazard": 1.52,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.22,
            "candidate_budget": 0.52,
            "turnover_budget": 1.02,
            "max_position_weight_target": 0.50,
            "hold_bias_target": 1.02,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.50,
            "action_soft": 0.50,
            "action_total": 0.60,
            "duration_total": 0.12,
            "scalar_total": 1.34,
            "daily_total": 0.48,
        },
    },
    "teacher_aux_return_recovery_balanced_v3": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.54,
            "entry_quality": 0.83,
            "hold_quality": 1.07,
            "add_quality": 0.84,
            "reduce_quality": 1.09,
            "exit_urgency": 1.15,
            "reentry_readiness": 0.47,
            "holding_days_ratio": 1.39,
            "reduce_fraction": 1.44,
            "exit_hazard": 1.50,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.16,
            "candidate_budget": 0.53,
            "turnover_budget": 0.98,
            "max_position_weight_target": 0.51,
            "hold_bias_target": 1.04,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.52,
            "action_soft": 0.48,
            "action_total": 0.62,
            "duration_total": 0.12,
            "scalar_total": 1.30,
            "daily_total": 0.44,
        },
    },
    "teacher_aux_return_recovery_stable_v2": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.48,
            "entry_quality": 0.80,
            "hold_quality": 1.06,
            "add_quality": 0.82,
            "reduce_quality": 1.06,
            "exit_urgency": 1.14,
            "reentry_readiness": 0.46,
            "holding_days_ratio": 1.36,
            "reduce_fraction": 1.40,
            "exit_hazard": 1.46,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.12,
            "candidate_budget": 0.54,
            "turnover_budget": 0.96,
            "max_position_weight_target": 0.52,
            "hold_bias_target": 1.06,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.52,
            "action_soft": 0.48,
            "action_total": 0.64,
            "duration_total": 0.12,
            "scalar_total": 1.28,
            "daily_total": 0.42,
        },
    },
    "alpha_result_value_budget_v1": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.58,
            "entry_quality": 0.92,
            "hold_quality": 1.12,
            "add_quality": 0.90,
            "reduce_quality": 1.04,
            "exit_urgency": 1.12,
            "reentry_readiness": 0.50,
            "holding_days_ratio": 1.44,
            "reduce_fraction": 1.38,
            "exit_hazard": 1.44,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.42,
            "candidate_budget": 0.72,
            "turnover_budget": 1.16,
            "max_position_weight_target": 0.64,
            "hold_bias_target": 1.12,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.48,
            "action_soft": 0.52,
            "action_total": 0.58,
            "duration_total": 0.12,
            "scalar_total": 1.32,
            "daily_total": 0.62,
        },
    },
    "alpha_result_value_budget_split_v2": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.56,
            "entry_quality": 0.90,
            "hold_quality": 1.14,
            "add_quality": 0.88,
            "reduce_quality": 1.08,
            "exit_urgency": 1.16,
            "reentry_readiness": 0.52,
            "holding_days_ratio": 1.46,
            "reduce_fraction": 1.44,
            "exit_hazard": 1.48,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.34,
            "candidate_budget": 0.80,
            "turnover_budget": 1.20,
            "max_position_weight_target": 0.66,
            "hold_bias_target": 1.08,
            "reduce_bias_target": 1.00,
            "exit_patience_target": 0.96,
            "reentry_guard_target": 0.86,
            "budget_risk_signal_target": 0.82,
            "budget_deploy_signal_target": 0.78,
            "budget_cash_timing_signal_target": 1.12,
            "budget_alpha_focus_signal_target": 0.70,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.48,
            "action_soft": 0.52,
            "action_total": 0.58,
            "duration_total": 0.12,
            "scalar_total": 1.34,
            "daily_total": 0.70,
        },
    },
    "alpha_result_value_budget_split_v3": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.58,
            "entry_quality": 0.88,
            "hold_quality": 1.12,
            "add_quality": 0.90,
            "reduce_quality": 1.10,
            "exit_urgency": 1.18,
            "reentry_readiness": 0.56,
            "holding_days_ratio": 1.44,
            "reduce_fraction": 1.46,
            "exit_hazard": 1.52,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.40,
            "candidate_budget": 0.92,
            "turnover_budget": 1.28,
            "max_position_weight_target": 0.76,
            "hold_bias_target": 1.00,
            "reduce_bias_target": 1.08,
            "exit_patience_target": 1.02,
            "reentry_guard_target": 0.94,
            "budget_risk_signal_target": 0.96,
            "budget_deploy_signal_target": 0.90,
            "budget_cash_timing_signal_target": 1.52,
            "budget_alpha_focus_signal_target": 0.76,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.46,
            "action_soft": 0.54,
            "action_total": 0.56,
            "duration_total": 0.12,
            "scalar_total": 1.38,
            "daily_total": 0.82,
        },
    },
    "alpha_result_value_budget_split_v4": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.56,
            "entry_quality": 0.86,
            "hold_quality": 1.10,
            "add_quality": 0.88,
            "reduce_quality": 1.18,
            "exit_urgency": 1.20,
            "reentry_readiness": 0.52,
            "holding_days_ratio": 1.42,
            "reduce_fraction": 1.56,
            "exit_hazard": 1.62,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.36,
            "candidate_budget": 0.88,
            "turnover_budget": 1.30,
            "max_position_weight_target": 0.78,
            "hold_bias_target": 0.98,
            "reduce_bias_target": 1.12,
            "exit_patience_target": 1.06,
            "reentry_guard_target": 0.96,
            "budget_risk_signal_target": 1.00,
            "budget_deploy_signal_target": 0.92,
            "budget_cash_timing_signal_target": 1.64,
            "budget_alpha_focus_signal_target": 0.80,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.46,
            "action_soft": 0.54,
            "action_total": 0.58,
            "duration_total": 0.12,
            "scalar_total": 1.42,
            "daily_total": 0.84,
        },
    },
    "alpha_result_value_budget_split_v4b": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.54,
            "entry_quality": 0.88,
            "hold_quality": 1.14,
            "add_quality": 0.90,
            "reduce_quality": 1.14,
            "exit_urgency": 1.16,
            "reentry_readiness": 0.54,
            "holding_days_ratio": 1.42,
            "reduce_fraction": 1.50,
            "exit_hazard": 1.56,
            "sell_attribution_score": 1.38,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.42,
            "candidate_budget": 0.94,
            "turnover_budget": 1.24,
            "max_position_weight_target": 0.80,
            "hold_bias_target": 1.06,
            "reduce_bias_target": 1.12,
            "exit_patience_target": 1.08,
            "reentry_guard_target": 0.92,
            "budget_risk_signal_target": 0.96,
            "budget_deploy_signal_target": 1.02,
            "budget_cash_timing_signal_target": 1.46,
            "budget_alpha_focus_signal_target": 0.84,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.46,
            "action_soft": 0.54,
            "action_total": 0.56,
            "duration_total": 0.12,
            "scalar_total": 1.48,
            "daily_total": 0.86,
        },
    },
    "alpha_result_value_budget_split_v5": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.48,
            "entry_quality": 0.84,
            "hold_quality": 1.12,
            "add_quality": 0.84,
            "reduce_quality": 1.12,
            "exit_urgency": 1.14,
            "reentry_readiness": 0.52,
            "holding_days_ratio": 1.36,
            "reduce_fraction": 1.42,
            "exit_hazard": 1.48,
            "sell_attribution_score": 1.20,
            "sell_rank_score": 1.56,
            "lifecycle_sell_gate": 1.48,
            "clipped_intent_risk": 1.10,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.36,
            "candidate_budget": 0.92,
            "turnover_budget": 1.28,
            "max_position_weight_target": 0.78,
            "hold_bias_target": 1.08,
            "reduce_bias_target": 1.16,
            "exit_patience_target": 1.10,
            "reentry_guard_target": 0.94,
            "budget_risk_signal_target": 0.96,
            "budget_deploy_signal_target": 1.00,
            "budget_cash_timing_signal_target": 1.38,
            "budget_alpha_focus_signal_target": 0.84,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.44,
            "action_soft": 0.56,
            "action_total": 0.54,
            "duration_total": 0.12,
            "scalar_total": 1.48,
            "daily_total": 0.82,
            "arbitration_total": 0.18,
            "sell_rank_pairwise_total": 0.18,
            "clipped_intent_total": 0.10,
        },
    },
    "alpha_result_value_budget_split_v6": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.42,
            "entry_quality": 0.82,
            "hold_quality": 1.08,
            "add_quality": 0.82,
            "reduce_quality": 1.08,
            "exit_urgency": 1.10,
            "reentry_readiness": 0.50,
            "holding_days_ratio": 1.30,
            "reduce_fraction": 1.36,
            "exit_hazard": 1.42,
            "sell_attribution_score": 1.12,
            "sell_rank_score": 1.38,
            "lifecycle_sell_gate": 1.36,
            "large_upside_1d_target": 1.12,
            "alpha_opportunity_value": 1.36,
            "hold_continuation_value": 1.26,
            "sell_release_value": 1.34,
            "cash_defense_value": 1.22,
            "deployment_opportunity_cost": 1.22,
            "risk_adjusted_action_value": 1.28,
            "value_arbitration_target": 1.46,
            "clipped_intent_risk": 1.04,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.34,
            "candidate_budget": 0.92,
            "turnover_budget": 1.22,
            "max_position_weight_target": 0.78,
            "hold_bias_target": 1.10,
            "reduce_bias_target": 1.12,
            "exit_patience_target": 1.12,
            "reentry_guard_target": 0.94,
            "budget_risk_signal_target": 1.00,
            "budget_deploy_signal_target": 1.08,
            "budget_cash_timing_signal_target": 1.28,
            "budget_alpha_focus_signal_target": 0.94,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.42,
            "action_soft": 0.58,
            "action_total": 0.52,
            "duration_total": 0.12,
            "scalar_total": 1.54,
            "daily_total": 0.84,
            "arbitration_total": 0.16,
            "sell_rank_pairwise_total": 0.14,
            "clipped_intent_total": 0.08,
            "value_arbitration_total": 0.20,
        },
    },
    "alpha_result_value_budget_split_v6b": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.36,
            "entry_quality": 0.84,
            "hold_quality": 1.06,
            "add_quality": 0.84,
            "reduce_quality": 1.04,
            "exit_urgency": 1.08,
            "reentry_readiness": 0.50,
            "holding_days_ratio": 1.24,
            "reduce_fraction": 1.28,
            "exit_hazard": 1.34,
            "sell_attribution_score": 1.08,
            "sell_rank_score": 1.30,
            "lifecycle_sell_gate": 1.28,
            "large_upside_1d_target": 1.16,
            "alpha_opportunity_value": 1.42,
            "hold_continuation_value": 1.28,
            "sell_release_value": 1.20,
            "cash_defense_value": 1.16,
            "deployment_opportunity_cost": 1.28,
            "risk_adjusted_action_value": 1.18,
            "value_arbitration_target": 0.96,
            "deploy_value_target": 1.46,
            "release_value_target": 1.24,
            "defense_value_target": 1.20,
            "deploy_gate_target": 1.42,
            "release_gate_target": 1.24,
            "defense_gate_target": 1.18,
            "clipped_intent_risk": 1.02,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.34,
            "candidate_budget": 0.94,
            "turnover_budget": 1.18,
            "max_position_weight_target": 0.80,
            "hold_bias_target": 1.12,
            "reduce_bias_target": 1.06,
            "exit_patience_target": 1.12,
            "reentry_guard_target": 0.92,
            "budget_risk_signal_target": 0.98,
            "budget_deploy_signal_target": 1.18,
            "budget_cash_timing_signal_target": 1.18,
            "budget_alpha_focus_signal_target": 1.04,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.40,
            "action_soft": 0.60,
            "action_total": 0.52,
            "duration_total": 0.12,
            "scalar_total": 1.58,
            "daily_total": 0.84,
            "arbitration_total": 0.14,
            "sell_rank_pairwise_total": 0.12,
            "clipped_intent_total": 0.08,
            "value_arbitration_total": 0.12,
            "three_value_gate_total": 0.24,
        },
    },
    "alpha_result_value_budget_split_v7": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.34,
            "entry_quality": 0.84,
            "hold_quality": 1.08,
            "add_quality": 0.84,
            "reduce_quality": 1.06,
            "exit_urgency": 1.08,
            "reentry_readiness": 0.50,
            "holding_days_ratio": 1.24,
            "reduce_fraction": 1.30,
            "exit_hazard": 1.36,
            "sell_attribution_score": 1.10,
            "sell_rank_score": 1.34,
            "lifecycle_sell_gate": 1.32,
            "large_upside_1d_target": 1.18,
            "alpha_opportunity_value": 1.46,
            "hold_continuation_value": 1.30,
            "sell_release_value": 1.26,
            "cash_defense_value": 1.20,
            "deployment_opportunity_cost": 1.30,
            "risk_adjusted_action_value": 1.18,
            "value_arbitration_target": 0.84,
            "deploy_value_target": 1.54,
            "release_value_target": 1.34,
            "defense_value_target": 0.86,
            "deploy_gate_target": 1.50,
            "release_gate_target": 1.32,
            "defense_gate_target": 0.42,
            "clipped_intent_risk": 1.02,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.38,
            "candidate_budget": 0.96,
            "turnover_budget": 1.22,
            "max_position_weight_target": 0.82,
            "hold_bias_target": 1.12,
            "reduce_bias_target": 1.08,
            "exit_patience_target": 1.12,
            "reentry_guard_target": 0.94,
            "budget_risk_signal_target": 1.10,
            "budget_deploy_signal_target": 1.20,
            "budget_cash_timing_signal_target": 1.34,
            "budget_alpha_focus_signal_target": 1.04,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.40,
            "action_soft": 0.60,
            "action_total": 0.52,
            "duration_total": 0.12,
            "scalar_total": 1.60,
            "daily_total": 0.90,
            "arbitration_total": 0.14,
            "sell_rank_pairwise_total": 0.12,
            "clipped_intent_total": 0.08,
            "value_arbitration_total": 0.08,
            "hierarchical_three_value_total": 0.22,
        },
    },
    "alpha_result_value_budget_split_v8": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.36,
            "entry_quality": 0.82,
            "hold_quality": 1.12,
            "add_quality": 0.84,
            "reduce_quality": 1.12,
            "exit_urgency": 1.10,
            "reentry_readiness": 0.48,
            "holding_days_ratio": 1.24,
            "reduce_fraction": 1.34,
            "exit_hazard": 1.40,
            "sell_attribution_score": 1.12,
            "sell_rank_score": 1.40,
            "lifecycle_sell_gate": 1.38,
            "large_upside_1d_target": 1.16,
            "alpha_opportunity_value": 1.44,
            "hold_continuation_value": 1.34,
            "sell_release_value": 1.34,
            "cash_defense_value": 1.04,
            "deployment_opportunity_cost": 1.28,
            "risk_adjusted_action_value": 1.18,
            "value_arbitration_target": 0.76,
            "deploy_value_target": 1.58,
            "release_value_target": 1.42,
            "defense_value_target": 0.68,
            "deploy_gate_target": 1.56,
            "release_gate_target": 1.40,
            "defense_gate_target": 0.32,
            "clipped_intent_risk": 1.16,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.46,
            "candidate_budget": 1.02,
            "turnover_budget": 1.28,
            "max_position_weight_target": 0.86,
            "hold_bias_target": 0.98,
            "reduce_bias_target": 1.12,
            "exit_patience_target": 1.08,
            "reentry_guard_target": 1.04,
            "budget_risk_signal_target": 1.18,
            "budget_deploy_signal_target": 1.24,
            "budget_cash_timing_signal_target": 1.38,
            "budget_alpha_focus_signal_target": 1.06,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.40,
            "action_soft": 0.60,
            "action_total": 0.54,
            "duration_total": 0.12,
            "scalar_total": 1.64,
            "daily_total": 0.88,
            "arbitration_total": 0.14,
            "sell_rank_pairwise_total": 0.14,
            "clipped_intent_total": 0.10,
            "value_arbitration_total": 0.06,
            "hierarchical_three_value_total": 0.24,
        },
    },
    "alpha_result_value_budget_split_v9": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.38,
            "entry_quality": 0.84,
            "hold_quality": 1.10,
            "add_quality": 0.90,
            "reduce_quality": 1.08,
            "exit_urgency": 1.06,
            "reentry_readiness": 0.46,
            "holding_days_ratio": 1.22,
            "reduce_fraction": 1.30,
            "exit_hazard": 1.34,
            "sell_attribution_score": 1.08,
            "sell_rank_score": 1.34,
            "lifecycle_sell_gate": 1.32,
            "large_upside_1d_target": 1.18,
            "alpha_opportunity_value": 1.48,
            "hold_continuation_value": 1.34,
            "sell_release_value": 1.28,
            "cash_defense_value": 0.96,
            "deployment_opportunity_cost": 1.38,
            "risk_adjusted_action_value": 1.20,
            "value_arbitration_target": 0.74,
            "deploy_value_target": 1.70,
            "release_value_target": 1.34,
            "defense_value_target": 0.58,
            "deploy_gate_target": 1.72,
            "release_gate_target": 1.32,
            "defense_gate_target": 0.28,
            "clipped_intent_risk": 1.20,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.52,
            "candidate_budget": 1.08,
            "turnover_budget": 1.32,
            "max_position_weight_target": 0.88,
            "hold_bias_target": 1.02,
            "reduce_bias_target": 1.06,
            "exit_patience_target": 1.10,
            "reentry_guard_target": 0.98,
            "budget_risk_signal_target": 1.12,
            "budget_deploy_signal_target": 1.36,
            "budget_cash_timing_signal_target": 1.28,
            "budget_alpha_focus_signal_target": 1.12,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.40,
            "action_soft": 0.60,
            "action_total": 0.56,
            "duration_total": 0.12,
            "scalar_total": 1.68,
            "daily_total": 0.88,
            "arbitration_total": 0.14,
            "sell_rank_pairwise_total": 0.13,
            "clipped_intent_total": 0.11,
            "value_arbitration_total": 0.06,
            "hierarchical_three_value_total": 0.25,
        },
    },
    "alpha_result_value_budget_split_v10": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.34,
            "entry_quality": 0.82,
            "hold_quality": 1.14,
            "add_quality": 0.88,
            "reduce_quality": 1.06,
            "exit_urgency": 1.04,
            "reentry_readiness": 0.44,
            "holding_days_ratio": 1.24,
            "reduce_fraction": 1.26,
            "exit_hazard": 1.30,
            "sell_attribution_score": 1.02,
            "sell_rank_score": 1.42,
            "lifecycle_sell_gate": 1.28,
            "large_upside_1d_target": 1.18,
            "alpha_opportunity_value": 1.52,
            "hold_continuation_value": 1.56,
            "sell_release_value": 1.30,
            "cash_defense_value": 1.04,
            "deployment_opportunity_cost": 1.40,
            "risk_adjusted_action_value": 1.24,
            "value_arbitration_target": 0.82,
            "deploy_value_target": 1.78,
            "release_value_target": 1.46,
            "defense_value_target": 0.70,
            "deploy_gate_target": 1.82,
            "release_gate_target": 1.48,
            "defense_gate_target": 0.38,
            "deploy_executability_target": 1.54,
            "clipped_intent_risk": 1.16,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.54,
            "candidate_budget": 1.12,
            "turnover_budget": 1.24,
            "max_position_weight_target": 0.90,
            "hold_bias_target": 1.12,
            "reduce_bias_target": 1.00,
            "exit_patience_target": 1.12,
            "reentry_guard_target": 0.96,
            "budget_risk_signal_target": 1.10,
            "budget_deploy_signal_target": 1.42,
            "budget_cash_timing_signal_target": 1.38,
            "budget_alpha_focus_signal_target": 1.14,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.40,
            "action_soft": 0.60,
            "action_total": 0.56,
            "duration_total": 0.12,
            "scalar_total": 1.72,
            "daily_total": 0.90,
            "arbitration_total": 0.14,
            "sell_rank_pairwise_total": 0.16,
            "clipped_intent_total": 0.10,
            "value_arbitration_total": 0.10,
            "hierarchical_three_value_total": 0.30,
            "funding_release_total": 0.18,
        },
    },
    "alpha_result_value_budget_split_v11": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.34,
            "entry_quality": 0.82,
            "hold_quality": 1.14,
            "add_quality": 0.88,
            "reduce_quality": 1.08,
            "exit_urgency": 1.06,
            "reentry_readiness": 0.44,
            "holding_days_ratio": 1.24,
            "reduce_fraction": 1.30,
            "exit_hazard": 1.34,
            "sell_attribution_score": 1.02,
            "sell_rank_score": 1.44,
            "lifecycle_sell_gate": 1.30,
            "large_upside_1d_target": 1.18,
            "alpha_opportunity_value": 1.52,
            "hold_continuation_value": 1.56,
            "sell_release_value": 1.42,
            "cash_defense_value": 1.06,
            "deployment_opportunity_cost": 1.40,
            "risk_adjusted_action_value": 1.24,
            "value_arbitration_target": 0.84,
            "deploy_value_target": 1.80,
            "release_value_target": 1.64,
            "defense_value_target": 0.72,
            "deploy_gate_target": 1.84,
            "release_gate_target": 1.68,
            "defense_gate_target": 0.38,
            "deploy_executability_target": 1.62,
            "clipped_intent_risk": 1.16,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.54,
            "candidate_budget": 1.10,
            "turnover_budget": 1.20,
            "max_position_weight_target": 0.90,
            "hold_bias_target": 1.08,
            "reduce_bias_target": 1.10,
            "exit_patience_target": 1.06,
            "reentry_guard_target": 0.94,
            "budget_risk_signal_target": 1.10,
            "budget_deploy_signal_target": 1.42,
            "budget_cash_timing_signal_target": 1.38,
            "budget_alpha_focus_signal_target": 1.14,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.40,
            "action_soft": 0.60,
            "action_total": 0.56,
            "duration_total": 0.12,
            "scalar_total": 1.74,
            "daily_total": 0.90,
            "arbitration_total": 0.14,
            "sell_rank_pairwise_total": 0.16,
            "clipped_intent_total": 0.10,
            "value_arbitration_total": 0.10,
            "hierarchical_three_value_total": 0.30,
            "funding_release_total": 0.26,
        },
    },
    "alpha_result_value_budget_split_v12": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.36,
            "entry_quality": 0.82,
            "hold_quality": 1.20,
            "add_quality": 0.92,
            "reduce_quality": 1.16,
            "exit_urgency": 1.12,
            "reentry_readiness": 0.42,
            "holding_days_ratio": 1.30,
            "reduce_fraction": 1.38,
            "exit_hazard": 1.42,
            "sell_attribution_score": 1.10,
            "sell_rank_score": 1.52,
            "lifecycle_sell_gate": 1.36,
            "large_upside_1d_target": 1.16,
            "alpha_opportunity_value": 1.58,
            "hold_continuation_value": 1.68,
            "sell_release_value": 1.58,
            "cash_defense_value": 1.04,
            "deployment_opportunity_cost": 1.46,
            "risk_adjusted_action_value": 1.28,
            "value_arbitration_target": 0.88,
            "deploy_value_target": 1.90,
            "release_value_target": 1.82,
            "defense_value_target": 0.68,
            "deploy_gate_target": 1.94,
            "release_gate_target": 1.88,
            "defense_gate_target": 0.34,
            "deploy_executability_target": 1.74,
            "clipped_intent_risk": 1.20,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.56,
            "candidate_budget": 1.16,
            "turnover_budget": 1.22,
            "max_position_weight_target": 0.92,
            "hold_bias_target": 1.12,
            "reduce_bias_target": 1.16,
            "exit_patience_target": 1.10,
            "reentry_guard_target": 0.94,
            "budget_risk_signal_target": 1.08,
            "budget_deploy_signal_target": 1.50,
            "budget_cash_timing_signal_target": 1.34,
            "budget_alpha_focus_signal_target": 1.18,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.40,
            "action_soft": 0.60,
            "action_total": 0.56,
            "duration_total": 0.12,
            "scalar_total": 1.82,
            "daily_total": 0.92,
            "arbitration_total": 0.14,
            "sell_rank_pairwise_total": 0.18,
            "clipped_intent_total": 0.10,
            "value_arbitration_total": 0.12,
            "hierarchical_three_value_total": 0.32,
            "funding_release_total": 0.34,
        },
    },
    "alpha_result_value_budget_split_v13": {
        "sample_scalar_loss_weights": {
            "target_delta_hint": 1.30,
            "entry_quality": 0.78,
            "hold_quality": 1.14,
            "add_quality": 0.88,
            "reduce_quality": 1.10,
            "exit_urgency": 1.08,
            "reentry_readiness": 0.40,
            "holding_days_ratio": 1.24,
            "reduce_fraction": 1.30,
            "exit_hazard": 1.34,
            "sell_attribution_score": 1.04,
            "sell_rank_score": 1.46,
            "lifecycle_sell_gate": 1.30,
            "large_upside_1d_target": 1.08,
            "alpha_opportunity_value": 1.46,
            "hold_continuation_value": 1.56,
            "sell_release_value": 1.46,
            "cash_defense_value": 1.00,
            "deployment_opportunity_cost": 1.38,
            "risk_adjusted_action_value": 1.20,
            "multi_horizon_forward_value": 1.58,
            "multi_horizon_forward_risk": 1.34,
            "multi_horizon_path_value": 1.64,
            "open_action_value": 1.66,
            "add_action_value": 1.58,
            "hold_action_value": 1.72,
            "reduce_action_value": 1.58,
            "exit_action_value": 1.50,
            "relative_opportunity_value": 1.20,
            "action_value_consistency_target": 1.44,
            "value_arbitration_target": 0.86,
            "deploy_value_target": 1.78,
            "release_value_target": 1.70,
            "defense_value_target": 0.64,
            "deploy_gate_target": 1.80,
            "release_gate_target": 1.76,
            "defense_gate_target": 0.32,
            "deploy_executability_target": 1.62,
            "clipped_intent_risk": 1.14,
        },
        "daily_target_loss_weights": {
            "gross_exposure_target": 1.52,
            "candidate_budget": 1.12,
            "turnover_budget": 1.18,
            "max_position_weight_target": 0.90,
            "hold_bias_target": 1.18,
            "reduce_bias_target": 1.10,
            "exit_patience_target": 1.14,
            "reentry_guard_target": 0.94,
            "budget_risk_signal_target": 1.06,
            "budget_deploy_signal_target": 1.46,
            "budget_cash_timing_signal_target": 1.30,
            "budget_alpha_focus_signal_target": 1.20,
        },
        "multi_objective_loss_weights": {
            "action_hard": 0.36,
            "action_soft": 0.64,
            "action_total": 0.50,
            "duration_total": 0.10,
            "scalar_total": 1.90,
            "daily_total": 0.90,
            "arbitration_total": 0.12,
            "sell_rank_pairwise_total": 0.16,
            "clipped_intent_total": 0.08,
            "value_arbitration_total": 0.10,
            "hierarchical_three_value_total": 0.28,
            "funding_release_total": 0.30,
            "action_value_total": 0.46,
        },
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v14"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v13"]["sample_scalar_loss_weights"],
        "target_delta_hint": 1.18,
        "entry_quality": 0.72,
        "hold_quality": 1.10,
        "add_quality": 0.84,
        "reduce_quality": 1.06,
        "exit_urgency": 1.04,
        "multi_horizon_forward_value": 1.72,
        "multi_horizon_forward_risk": 1.46,
        "multi_horizon_path_value": 1.82,
        "open_action_value": 1.90,
        "add_action_value": 1.78,
        "hold_action_value": 1.84,
        "reduce_action_value": 1.76,
        "exit_action_value": 1.70,
        "relative_opportunity_value": 1.34,
        "action_value_consistency_target": 1.66,
        "deploy_executability_target": 1.72,
        "clipped_intent_risk": 1.24,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v13"]["daily_target_loss_weights"],
        "candidate_budget": 1.06,
        "turnover_budget": 1.10,
        "budget_deploy_signal_target": 1.40,
        "budget_cash_timing_signal_target": 1.36,
        "budget_alpha_focus_signal_target": 1.24,
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v13"]["multi_objective_loss_weights"],
        "action_hard": 0.28,
        "action_soft": 0.72,
        "action_total": 0.42,
        "duration_total": 0.08,
        "scalar_total": 2.04,
        "daily_total": 0.86,
        "arbitration_total": 0.10,
        "sell_rank_pairwise_total": 0.14,
        "clipped_intent_total": 0.08,
        "value_arbitration_total": 0.10,
        "hierarchical_three_value_total": 0.26,
        "funding_release_total": 0.28,
        "action_value_total": 0.74,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v15"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v14"]["sample_scalar_loss_weights"],
        "target_delta_hint": 1.12,
        "entry_quality": 0.68,
        "hold_quality": 1.20,
        "add_quality": 0.88,
        "reduce_quality": 1.18,
        "exit_urgency": 1.16,
        "multi_horizon_forward_value": 1.76,
        "multi_horizon_forward_risk": 1.54,
        "multi_horizon_path_value": 1.88,
        "open_action_value": 1.84,
        "add_action_value": 1.84,
        "hold_action_value": 2.02,
        "reduce_action_value": 1.96,
        "exit_action_value": 1.92,
        "relative_opportunity_value": 1.42,
        "action_value_consistency_target": 1.88,
        "deploy_value_target": 1.88,
        "release_value_target": 1.94,
        "deploy_executability_target": 1.74,
        "clipped_intent_risk": 1.36,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v14"]["daily_target_loss_weights"],
        "candidate_budget": 1.00,
        "turnover_budget": 1.16,
        "budget_risk_signal_target": 1.14,
        "budget_deploy_signal_target": 1.34,
        "budget_cash_timing_signal_target": 1.42,
        "budget_alpha_focus_signal_target": 1.18,
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v14"]["multi_objective_loss_weights"],
        "action_hard": 0.24,
        "action_soft": 0.78,
        "action_total": 0.46,
        "duration_total": 0.08,
        "scalar_total": 2.16,
        "daily_total": 0.88,
        "arbitration_total": 0.10,
        "sell_rank_pairwise_total": 0.16,
        "clipped_intent_total": 0.10,
        "value_arbitration_total": 0.10,
        "hierarchical_three_value_total": 0.28,
        "funding_release_total": 0.38,
        "action_value_total": 0.88,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v16"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v15"]["sample_scalar_loss_weights"],
        "deploy_executability_target": 1.58,
        "portfolio_daily_receiver_add_headroom": 1.12,
        "portfolio_daily_receiver_add_capacity": 1.28,
        "portfolio_daily_receiver_executability": 1.62,
        "portfolio_daily_receiver_score": 1.78,
        "portfolio_daily_source_score": 1.68,
        "portfolio_daily_cash_score": 1.22,
        "clipped_intent_risk": 1.42,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v15"]["daily_target_loss_weights"],
        "candidate_budget": 1.04,
        "turnover_budget": 1.18,
        "budget_deploy_signal_target": 1.38,
        "budget_cash_timing_signal_target": 1.44,
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v15"]["multi_objective_loss_weights"],
        "scalar_total": 2.22,
        "daily_total": 0.90,
        "portfolio_receiver_pairwise_total": 0.20,
        "portfolio_source_pairwise_total": 0.18,
        "portfolio_cash_margin_total": 0.08,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v17"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v16"]["sample_scalar_loss_weights"],
        "portfolio_daily_source_release_capacity": 1.24,
        "portfolio_daily_source_release_quality": 1.72,
        "portfolio_daily_source_opportunity_cost": 1.66,
        "portfolio_daily_source_executability": 1.72,
        "portfolio_daily_source_score": 1.92,
        "portfolio_daily_receiver_score": 1.74,
        "portfolio_daily_cash_score": 1.28,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v16"]["daily_target_loss_weights"],
        "turnover_budget": 1.20,
        "budget_deploy_signal_target": 1.40,
        "budget_cash_timing_signal_target": 1.46,
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v16"]["multi_objective_loss_weights"],
        "scalar_total": 2.32,
        "portfolio_receiver_pairwise_total": 0.18,
        "portfolio_source_pairwise_total": 0.26,
        "portfolio_cash_margin_total": 0.10,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v18"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v17"]["sample_scalar_loss_weights"],
        "portfolio_daily_receiver_funding_coverage": 1.86,
        "portfolio_daily_funding_closure_score": 1.72,
        "portfolio_daily_allocation_transfer_score": 1.82,
        "portfolio_daily_allocation_dead_branch_risk": 1.54,
        "portfolio_daily_receiver_score": 1.82,
        "portfolio_daily_source_score": 2.02,
        "portfolio_daily_cash_score": 1.34,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v17"]["daily_target_loss_weights"],
        "turnover_budget": 1.24,
        "budget_deploy_signal_target": 1.44,
        "budget_cash_timing_signal_target": 1.48,
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v17"]["multi_objective_loss_weights"],
        "scalar_total": 2.48,
        "portfolio_receiver_pairwise_total": 0.24,
        "portfolio_source_pairwise_total": 0.30,
        "portfolio_cash_margin_total": 0.12,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v19"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v18"]["sample_scalar_loss_weights"],
        "portfolio_daily_source_forward_spread_score": 1.74,
        "portfolio_daily_source_bad_forward_spread_risk": 1.86,
        "portfolio_daily_source_economic_release_score": 2.16,
        "portfolio_daily_source_economic_block_risk": 1.92,
        "portfolio_daily_source_release_quality": 1.94,
        "portfolio_daily_source_opportunity_cost": 1.92,
        "portfolio_daily_source_executability": 1.88,
        "portfolio_daily_source_score": 2.22,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v18"]["daily_target_loss_weights"],
        "turnover_budget": 1.26,
        "budget_deploy_signal_target": 1.46,
        "budget_cash_timing_signal_target": 1.50,
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v18"]["multi_objective_loss_weights"],
        "scalar_total": 2.66,
        "portfolio_source_pairwise_total": 0.38,
        "portfolio_receiver_pairwise_total": 0.24,
        "portfolio_cash_margin_total": 0.12,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v20"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v19"]["sample_scalar_loss_weights"],
        "portfolio_daily_source_forward_strength_brake_risk": 2.34,
        "portfolio_daily_source_forward_proxy_keep_risk": 1.72,
        "portfolio_daily_source_bad_forward_spread_risk": 2.02,
        "portfolio_daily_source_economic_block_risk": 2.06,
        "portfolio_daily_source_economic_release_score": 2.04,
        "portfolio_daily_source_opportunity_cost": 2.04,
        "portfolio_daily_source_score": 2.30,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v19"]["daily_target_loss_weights"],
        "turnover_budget": 1.30,
        "budget_deploy_signal_target": 1.48,
        "budget_cash_timing_signal_target": 1.52,
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v19"]["multi_objective_loss_weights"],
        "scalar_total": 2.82,
        "portfolio_source_pairwise_total": 0.44,
        "portfolio_receiver_pairwise_total": 0.24,
        "portfolio_cash_margin_total": 0.12,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v21"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v20"]["sample_scalar_loss_weights"],
        "portfolio_daily_receiver_score": 2.46,
        "portfolio_daily_source_score": 2.62,
        "portfolio_daily_cash_score": 1.46,
        "portfolio_daily_unified_receiver_score": 1.88,
        "portfolio_daily_unified_source_score": 2.18,
        "portfolio_daily_unified_cash_score": 1.42,
        "portfolio_daily_source_positive_forward_penalty": 1.86,
        "portfolio_daily_source_opportunity_cost_penalty": 1.74,
        "portfolio_daily_receiver_source_spread_reward": 1.72,
        "portfolio_daily_unified_allocation_objective": 1.64,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v20"]["daily_target_loss_weights"],
        "turnover_budget": 1.34,
        "budget_deploy_signal_target": 1.52,
        "budget_cash_timing_signal_target": 1.56,
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v20"]["multi_objective_loss_weights"],
        "scalar_total": 3.02,
        "portfolio_source_pairwise_total": 0.50,
        "portfolio_receiver_pairwise_total": 0.28,
        "portfolio_cash_margin_total": 0.16,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v22"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v21"]["sample_scalar_loss_weights"],
        "portfolio_daily_receiver_score": 2.52,
        "portfolio_daily_source_score": 2.78,
        "portfolio_daily_cash_score": 1.58,
        "portfolio_daily_unified_receiver_score": 1.96,
        "portfolio_daily_unified_source_score": 2.36,
        "portfolio_daily_unified_cash_score": 1.84,
        "portfolio_daily_source_positive_forward_penalty": 2.24,
        "portfolio_daily_source_opportunity_cost_penalty": 2.02,
        "portfolio_daily_receiver_source_spread_reward": 1.96,
        "portfolio_daily_unified_allocation_objective": 1.92,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v21"]["daily_target_loss_weights"],
        "budget_risk_signal_target": 1.46,
        "budget_deploy_signal_target": 1.58,
        "budget_cash_timing_signal_target": 1.92,
        "budget_alpha_focus_signal_target": 1.42,
        "turnover_budget": 1.40,
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v21"]["multi_objective_loss_weights"],
        "scalar_total": 3.12,
        "portfolio_source_pairwise_total": 0.58,
        "portfolio_receiver_pairwise_total": 0.32,
        "portfolio_cash_margin_total": 0.22,
        "portfolio_unified_allocation_total": 0.46,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v23"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v22"]["sample_scalar_loss_weights"],
        "portfolio_daily_receiver_score": 2.56,
        "portfolio_daily_source_score": 2.86,
        "portfolio_daily_cash_score": 1.64,
        "portfolio_daily_unified_receiver_score": 2.08,
        "portfolio_daily_unified_source_score": 2.54,
        "portfolio_daily_unified_cash_score": 1.90,
        "portfolio_daily_source_positive_forward_penalty": 2.42,
        "portfolio_daily_source_opportunity_cost_penalty": 2.18,
        "portfolio_daily_receiver_source_spread_reward": 2.06,
        "portfolio_daily_unified_allocation_objective": 2.08,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v22"]["daily_target_loss_weights"],
        "budget_risk_signal_target": 1.50,
        "budget_deploy_signal_target": 1.64,
        "budget_cash_timing_signal_target": 1.96,
        "budget_alpha_focus_signal_target": 1.48,
        "turnover_budget": 1.44,
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v22"]["multi_objective_loss_weights"],
        "scalar_total": 3.18,
        "portfolio_source_pairwise_total": 0.62,
        "portfolio_receiver_pairwise_total": 0.34,
        "portfolio_cash_margin_total": 0.24,
        "portfolio_unified_allocation_total": 0.50,
        "portfolio_decision_regret_total": 0.28,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v24"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v23"]["sample_scalar_loss_weights"],
        "portfolio_daily_receiver_score": 2.58,
        "portfolio_daily_source_score": 2.92,
        "portfolio_daily_cash_score": 1.66,
        "portfolio_daily_unified_receiver_score": 2.12,
        "portfolio_daily_unified_source_score": 2.66,
        "portfolio_daily_unified_cash_score": 1.92,
        "portfolio_daily_source_positive_forward_penalty": 2.62,
        "portfolio_daily_source_opportunity_cost_penalty": 2.38,
        "portfolio_daily_source_strong_false_sell_penalty": 2.18,
        "portfolio_daily_source_hard_negative_penalty": 2.86,
        "portfolio_daily_source_tail_false_sell_penalty": 2.34,
        "portfolio_daily_source_release_preference": 2.58,
        "portfolio_daily_receiver_source_spread_reward": 2.14,
        "portfolio_daily_transfer_regret_target": 2.36,
        "portfolio_daily_unified_allocation_objective": 2.18,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v23"]["daily_target_loss_weights"],
        "budget_risk_signal_target": 1.54,
        "budget_deploy_signal_target": 1.68,
        "budget_cash_timing_signal_target": 2.00,
        "budget_alpha_focus_signal_target": 1.50,
        "turnover_budget": 1.46,
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v23"]["multi_objective_loss_weights"],
        "scalar_total": 3.30,
        "portfolio_source_pairwise_total": 0.64,
        "portfolio_receiver_pairwise_total": 0.34,
        "portfolio_cash_margin_total": 0.24,
        "portfolio_unified_allocation_total": 0.52,
        "portfolio_decision_regret_total": 0.34,
        "portfolio_source_hard_negative_tail_total": 0.38,
        "portfolio_source_listwise_release_total": 0.34,
        "portfolio_transfer_regret_total": 0.32,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v25"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v24"]["sample_scalar_loss_weights"],
        "portfolio_daily_receiver_score": 2.20,
        "portfolio_daily_source_score": 2.36,
        "portfolio_daily_cash_score": 1.42,
        "portfolio_daily_unified_receiver_score": 2.34,
        "portfolio_daily_unified_source_score": 2.54,
        "portfolio_daily_unified_cash_score": 2.10,
        "portfolio_daily_source_positive_forward_penalty": 2.70,
        "portfolio_daily_source_opportunity_cost_penalty": 2.54,
        "portfolio_daily_source_hard_negative_penalty": 2.94,
        "portfolio_daily_source_release_preference": 2.64,
        "portfolio_daily_receiver_source_spread_reward": 2.34,
        "portfolio_daily_transfer_regret_target": 2.54,
        "portfolio_daily_unified_allocation_objective": 2.28,
        "portfolio_daily_allocation_trade_quality_target": 2.72,
        "portfolio_daily_allocation_cash_deployment_target": 2.60,
        "portfolio_daily_allocation_risk_adjusted_return_target": 2.68,
        "portfolio_daily_allocation_drawdown_control_target": 2.18,
        "portfolio_daily_allocation_monthly_quality_target": 2.48,
        "portfolio_daily_allocation_final_objective": 2.86,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v24"]["daily_target_loss_weights"],
        "budget_risk_signal_target": 1.46,
        "budget_deploy_signal_target": 1.82,
        "budget_cash_timing_signal_target": 1.76,
        "budget_alpha_focus_signal_target": 1.62,
        "turnover_budget": 1.52,
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v24"]["multi_objective_loss_weights"],
        "action_total": 0.06,
        "duration_total": 0.05,
        "scalar_total": 3.58,
        "portfolio_source_pairwise_total": 0.70,
        "portfolio_receiver_pairwise_total": 0.44,
        "portfolio_cash_margin_total": 0.28,
        "portfolio_unified_allocation_total": 0.62,
        "portfolio_decision_regret_total": 0.44,
        "portfolio_source_hard_negative_tail_total": 0.42,
        "portfolio_source_listwise_release_total": 0.38,
        "portfolio_transfer_regret_total": 0.38,
        "portfolio_allocation_objective_consolidation_total": 0.86,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v26"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v25"]["sample_scalar_loss_weights"],
        "portfolio_daily_unified_receiver_score": 2.42,
        "portfolio_daily_unified_source_score": 2.66,
        "portfolio_daily_unified_cash_score": 2.28,
        "portfolio_daily_allocation_drawdown_control_target": 2.36,
        "portfolio_daily_allocation_final_objective": 2.92,
        "portfolio_daily_allocation_uncertainty_pressure_target": 2.42,
        "portfolio_daily_allocation_tail_risk_control_target": 2.50,
        "portfolio_daily_allocation_decision_focused_objective": 3.02,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v25"]["daily_target_loss_weights"],
        "budget_risk_signal_target": 1.62,
        "budget_deploy_signal_target": 1.74,
        "budget_cash_timing_signal_target": 1.90,
        "budget_alpha_focus_signal_target": 1.68,
        "turnover_budget": 1.48,
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v25"]["multi_objective_loss_weights"],
        "action_total": 0.05,
        "duration_total": 0.04,
        "scalar_total": 3.66,
        "portfolio_cash_margin_total": 0.34,
        "portfolio_unified_allocation_total": 0.66,
        "portfolio_decision_regret_total": 0.50,
        "portfolio_allocation_objective_consolidation_total": 0.92,
        "portfolio_risk_sensitive_allocation_total": 0.54,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v27"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v26"]["sample_scalar_loss_weights"],
        "portfolio_daily_unified_receiver_score": 2.72,
        "portfolio_daily_unified_source_score": 2.90,
        "portfolio_daily_unified_cash_score": 2.40,
        "portfolio_daily_allocation_final_objective": 3.15,
        "portfolio_daily_allocation_decision_focused_objective": 3.22,
        "portfolio_daily_allocation_net_utility_target": 3.42,
        "portfolio_daily_allocation_credit_closure_target": 3.56,
        "portfolio_daily_allocation_resource_efficiency_target": 3.18,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v26"]["daily_target_loss_weights"],
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v26"]["multi_objective_loss_weights"],
        "action_total": 0.035,
        "duration_total": 0.035,
        "scalar_total": 3.88,
        "portfolio_unified_allocation_total": 0.70,
        "portfolio_decision_regret_total": 0.54,
        "portfolio_allocation_objective_consolidation_total": 0.96,
        "portfolio_risk_sensitive_allocation_total": 0.58,
        "portfolio_utility_credit_closure_total": 0.82,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v28"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v27"]["sample_scalar_loss_weights"],
        "portfolio_daily_unified_receiver_score": 2.88,
        "portfolio_daily_unified_source_score": 3.08,
        "portfolio_daily_unified_cash_score": 2.52,
        "portfolio_daily_allocation_final_objective": 3.28,
        "portfolio_daily_allocation_decision_focused_objective": 3.36,
        "portfolio_daily_allocation_net_utility_target": 3.62,
        "portfolio_daily_allocation_credit_closure_target": 3.78,
        "portfolio_daily_allocation_resource_efficiency_target": 3.36,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v27"]["daily_target_loss_weights"],
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v27"]["multi_objective_loss_weights"],
        "action_total": 0.025,
        "duration_total": 0.025,
        "scalar_total": 3.72,
        "portfolio_unified_allocation_total": 0.66,
        "portfolio_decision_regret_total": 0.58,
        "portfolio_allocation_objective_consolidation_total": 0.92,
        "portfolio_risk_sensitive_allocation_total": 0.62,
        "portfolio_utility_credit_closure_total": 0.90,
        "portfolio_primal_dual_decision_total": 1.08,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v29"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v28"]["sample_scalar_loss_weights"],
        "portfolio_daily_unified_receiver_score": 3.02,
        "portfolio_daily_unified_source_score": 3.22,
        "portfolio_daily_unified_cash_score": 2.68,
        "portfolio_daily_allocation_final_objective": 3.48,
        "portfolio_daily_allocation_decision_focused_objective": 3.54,
        "portfolio_daily_allocation_net_utility_target": 3.82,
        "portfolio_daily_allocation_credit_closure_target": 3.96,
        "portfolio_daily_allocation_resource_efficiency_target": 3.52,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v28"]["daily_target_loss_weights"],
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v28"]["multi_objective_loss_weights"],
        "action_total": 0.018,
        "duration_total": 0.018,
        "scalar_total": 3.54,
        "portfolio_unified_allocation_total": 0.60,
        "portfolio_decision_regret_total": 0.54,
        "portfolio_allocation_objective_consolidation_total": 0.86,
        "portfolio_risk_sensitive_allocation_total": 0.60,
        "portfolio_utility_credit_closure_total": 0.84,
        "portfolio_primal_dual_decision_total": 0.96,
        "portfolio_entropic_transport_decision_total": 1.24,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v30"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v29"]["sample_scalar_loss_weights"],
        "portfolio_daily_unified_receiver_score": 3.12,
        "portfolio_daily_unified_source_score": 3.34,
        "portfolio_daily_unified_cash_score": 2.78,
        "portfolio_daily_allocation_final_objective": 3.58,
        "portfolio_daily_allocation_decision_focused_objective": 3.62,
        "portfolio_daily_allocation_net_utility_target": 3.90,
        "portfolio_daily_allocation_credit_closure_target": 4.04,
        "portfolio_daily_allocation_resource_efficiency_target": 3.62,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v29"]["daily_target_loss_weights"],
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v29"]["multi_objective_loss_weights"],
        "action_total": 0.014,
        "duration_total": 0.014,
        "scalar_total": 3.40,
        "portfolio_unified_allocation_total": 0.58,
        "portfolio_decision_regret_total": 0.50,
        "portfolio_allocation_objective_consolidation_total": 0.82,
        "portfolio_risk_sensitive_allocation_total": 0.58,
        "portfolio_utility_credit_closure_total": 0.80,
        "portfolio_primal_dual_decision_total": 0.88,
        "portfolio_entropic_transport_decision_total": 1.16,
        "portfolio_offline_conservative_support_total": 0.66,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v31"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v30"]["sample_scalar_loss_weights"],
        "portfolio_daily_unified_receiver_score": 3.28,
        "portfolio_daily_unified_source_score": 3.48,
        "portfolio_daily_unified_cash_score": 3.08,
        "portfolio_daily_allocation_final_objective": 3.92,
        "portfolio_daily_allocation_decision_focused_objective": 3.96,
        "portfolio_daily_allocation_net_utility_target": 4.12,
        "portfolio_daily_allocation_credit_closure_target": 4.22,
        "portfolio_daily_allocation_resource_efficiency_target": 3.88,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v30"]["daily_target_loss_weights"],
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v30"]["multi_objective_loss_weights"],
        "action_total": 0.0,
        "duration_total": 0.0,
        "scalar_total": 2.20,
        "portfolio_unified_allocation_total": 0.32,
        "portfolio_decision_regret_total": 0.30,
        "portfolio_allocation_objective_consolidation_total": 0.48,
        "portfolio_risk_sensitive_allocation_total": 0.38,
        "portfolio_utility_credit_closure_total": 0.46,
        "portfolio_primal_dual_decision_total": 0.52,
        "portfolio_entropic_transport_decision_total": 0.70,
        "portfolio_offline_conservative_support_total": 0.62,
        "portfolio_differentiable_convex_allocation_total": 1.56,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v32"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v31"]["sample_scalar_loss_weights"],
        "portfolio_daily_unified_receiver_score": 3.36,
        "portfolio_daily_unified_source_score": 3.54,
        "portfolio_daily_unified_cash_score": 3.16,
        "portfolio_daily_allocation_final_objective": 4.08,
        "portfolio_daily_allocation_decision_focused_objective": 4.10,
        "portfolio_daily_allocation_net_utility_target": 4.24,
        "portfolio_daily_allocation_credit_closure_target": 4.32,
        "portfolio_daily_allocation_resource_efficiency_target": 3.96,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v31"]["daily_target_loss_weights"],
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v31"]["multi_objective_loss_weights"],
        "action_total": 0.0,
        "duration_total": 0.0,
        "scalar_total": 2.00,
        "portfolio_unified_allocation_total": 0.24,
        "portfolio_decision_regret_total": 0.24,
        "portfolio_allocation_objective_consolidation_total": 0.40,
        "portfolio_risk_sensitive_allocation_total": 0.34,
        "portfolio_utility_credit_closure_total": 0.42,
        "portfolio_primal_dual_decision_total": 0.48,
        "portfolio_entropic_transport_decision_total": 0.56,
        "portfolio_offline_conservative_support_total": 0.54,
        "portfolio_differentiable_convex_allocation_total": 0.74,
        "portfolio_cvxpy_convex_allocation_total": 1.42,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v33"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v32"]["sample_scalar_loss_weights"],
        "portfolio_daily_unified_receiver_score": 3.46,
        "portfolio_daily_unified_source_score": 3.64,
        "portfolio_daily_unified_cash_score": 3.22,
        "portfolio_daily_allocation_final_objective": 4.18,
        "portfolio_daily_allocation_decision_focused_objective": 4.20,
        "portfolio_daily_allocation_net_utility_target": 4.34,
        "portfolio_daily_allocation_credit_closure_target": 4.40,
        "portfolio_daily_allocation_resource_efficiency_target": 4.06,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v32"]["daily_target_loss_weights"],
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v32"]["multi_objective_loss_weights"],
        "action_total": 0.0,
        "duration_total": 0.0,
        "scalar_total": 1.84,
        "portfolio_unified_allocation_total": 0.18,
        "portfolio_decision_regret_total": 0.20,
        "portfolio_allocation_objective_consolidation_total": 0.34,
        "portfolio_risk_sensitive_allocation_total": 0.32,
        "portfolio_utility_credit_closure_total": 0.38,
        "portfolio_primal_dual_decision_total": 0.42,
        "portfolio_entropic_transport_decision_total": 0.48,
        "portfolio_offline_conservative_support_total": 0.46,
        "portfolio_differentiable_convex_allocation_total": 0.50,
        "portfolio_cvxpy_convex_allocation_total": 0.0,
        "portfolio_full_universe_convex_allocation_total": 1.68,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v34"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v33"]["sample_scalar_loss_weights"],
        "portfolio_daily_unified_receiver_score": 3.62,
        "portfolio_daily_unified_source_score": 3.92,
        "portfolio_daily_unified_cash_score": 3.42,
        "portfolio_daily_allocation_final_objective": 4.26,
        "portfolio_daily_allocation_decision_focused_objective": 4.28,
        "portfolio_daily_allocation_net_utility_target": 4.40,
        "portfolio_daily_allocation_credit_closure_target": 4.56,
        "portfolio_daily_allocation_resource_efficiency_target": 4.12,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v33"]["daily_target_loss_weights"],
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v33"]["multi_objective_loss_weights"],
        "action_total": 0.0,
        "duration_total": 0.0,
        "scalar_total": 1.70,
        "portfolio_unified_allocation_total": 0.14,
        "portfolio_decision_regret_total": 0.16,
        "portfolio_allocation_objective_consolidation_total": 0.30,
        "portfolio_risk_sensitive_allocation_total": 0.30,
        "portfolio_utility_credit_closure_total": 0.34,
        "portfolio_primal_dual_decision_total": 0.38,
        "portfolio_entropic_transport_decision_total": 0.42,
        "portfolio_offline_conservative_support_total": 0.42,
        "portfolio_differentiable_convex_allocation_total": 0.44,
        "portfolio_cvxpy_convex_allocation_total": 0.0,
        "portfolio_full_universe_convex_allocation_total": 1.18,
        "portfolio_capital_flow_closure_total": 1.72,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v35"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v34"]["sample_scalar_loss_weights"],
        "portfolio_daily_unified_receiver_score": 3.70,
        "portfolio_daily_unified_source_score": 4.10,
        "portfolio_daily_unified_cash_score": 3.50,
        "portfolio_daily_allocation_final_objective": 4.36,
        "portfolio_daily_allocation_decision_focused_objective": 4.38,
        "portfolio_daily_allocation_net_utility_target": 4.48,
        "portfolio_daily_allocation_credit_closure_target": 4.68,
        "portfolio_daily_allocation_resource_efficiency_target": 4.22,
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v34"]["daily_target_loss_weights"],
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v34"]["multi_objective_loss_weights"],
        "action_total": 0.0,
        "duration_total": 0.0,
        "scalar_total": 1.55,
        "portfolio_unified_allocation_total": 0.12,
        "portfolio_decision_regret_total": 0.14,
        "portfolio_allocation_objective_consolidation_total": 0.28,
        "portfolio_risk_sensitive_allocation_total": 0.36,
        "portfolio_utility_credit_closure_total": 0.34,
        "portfolio_primal_dual_decision_total": 0.36,
        "portfolio_entropic_transport_decision_total": 0.40,
        "portfolio_offline_conservative_support_total": 0.48,
        "portfolio_differentiable_convex_allocation_total": 0.38,
        "portfolio_cvxpy_convex_allocation_total": 0.24,
        "portfolio_full_universe_convex_allocation_total": 1.34,
        "portfolio_capital_flow_closure_total": 1.68,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v36"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v35"]["sample_scalar_loss_weights"],
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v35"]["daily_target_loss_weights"],
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v35"]["multi_objective_loss_weights"],
        "action_total": 0.0,
        "duration_total": 0.0,
        "scalar_total": 1.25,
        "portfolio_unified_allocation_total": 0.0,
        "portfolio_decision_regret_total": 0.0,
        "portfolio_allocation_objective_consolidation_total": 0.0,
        "portfolio_risk_sensitive_allocation_total": 0.0,
        "portfolio_utility_credit_closure_total": 0.0,
        "portfolio_primal_dual_decision_total": 0.0,
        "portfolio_entropic_transport_decision_total": 0.0,
        "portfolio_offline_conservative_support_total": 0.0,
        "portfolio_differentiable_convex_allocation_total": 0.0,
        "portfolio_cvxpy_convex_allocation_total": 0.0,
        "portfolio_full_universe_convex_allocation_total": 0.0,
        "portfolio_native_allocation_vector_total": 2.10,
        "portfolio_capital_flow_closure_total": 0.70,
    },
}
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v37"] = {
    "sample_scalar_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v36"]["sample_scalar_loss_weights"],
    },
    "daily_target_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v36"]["daily_target_loss_weights"],
    },
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v36"]["multi_objective_loss_weights"],
        "action_total": 0.0,
        "duration_total": 0.0,
        "scalar_total": 1.10,
        "portfolio_native_allocation_vector_total": 0.0,
        "portfolio_day_set_native_allocation_vector_total": 2.40,
        "portfolio_capital_flow_closure_total": 0.50,
        "portfolio_cvxpy_convex_allocation_total": 0.0,
        "portfolio_full_universe_convex_allocation_total": 0.0,
    },
}
DIRECT_ACTION_VALUE_POLICY_MODE = "direct_action_value_v1"
DIRECT_ACTION_VALUE_LOSS_PROFILES = frozenset(
    {
        "alpha_result_value_budget_split_v14",
        "alpha_result_value_budget_split_v15",
        "alpha_result_value_budget_split_v16",
        "alpha_result_value_budget_split_v17",
        "alpha_result_value_budget_split_v18",
        "alpha_result_value_budget_split_v19",
        "alpha_result_value_budget_split_v20",
    }
)
DEFAULT_LOSS_PROFILE = "dual_channel_default_v1"
LOSS_PROFILE_NAMES: tuple[str, ...] = tuple(sorted(LOSS_PROFILE_CONFIGS))
DAILY_HEAD_LAYOUT_MONOLITHIC_V1 = "monolithic_v1"
DAILY_HEAD_LAYOUT_SPLIT_V2 = "split_v2"
DAILY_HEAD_LAYOUT_CHOICES: tuple[str, ...] = (
    DAILY_HEAD_LAYOUT_MONOLITHIC_V1,
    DAILY_HEAD_LAYOUT_SPLIT_V2,
)
DAILY_CONTROL_TARGET_NAMES: tuple[str, ...] = (
    "gross_exposure_target",
    "candidate_budget",
    "turnover_budget",
    "max_position_weight_target",
    "hold_bias_target",
    "reduce_bias_target",
    "exit_patience_target",
    "reentry_guard_target",
)
DAILY_AUX_TARGET_NAMES: tuple[str, ...] = (
    "budget_risk_signal_target",
    "budget_deploy_signal_target",
    "budget_cash_timing_signal_target",
    "budget_alpha_focus_signal_target",
)
ALL_DAILY_TARGET_NAMES: tuple[str, ...] = DAILY_CONTROL_TARGET_NAMES + DAILY_AUX_TARGET_NAMES


def resolve_loss_profile(loss_profile: str | None) -> tuple[str, dict[str, dict[str, float]]]:
    profile_name = str(loss_profile or DEFAULT_LOSS_PROFILE).strip() or DEFAULT_LOSS_PROFILE
    if profile_name not in LOSS_PROFILE_CONFIGS:
        raise ValueError(
            f"Unsupported seq_v3 loss profile: {profile_name}. Available: {', '.join(LOSS_PROFILE_NAMES)}"
        )
    config = LOSS_PROFILE_CONFIGS[profile_name]
    return profile_name, {
        "sample_scalar_loss_weights": dict(config["sample_scalar_loss_weights"]),
        "daily_target_loss_weights": dict(config["daily_target_loss_weights"]),
        "multi_objective_loss_weights": dict(config["multi_objective_loss_weights"]),
    }


def _loss_profile_enables_full_universe_train_solver(loss_profile: str | None) -> bool:
    profile_name = str(loss_profile or "").strip()
    return profile_name in CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_SOLVER_LOSS_PROFILES


def _apply_torch_thread_env_limits() -> dict[str, Any]:
    env_limits = {
        key: str(os.environ.get(key, "") or "")
        for key in THREAD_LIMIT_ENV_KEYS
        if str(os.environ.get(key, "") or "").strip()
    }
    requested = str(os.environ.get("TORCH_NUM_THREADS", "") or os.environ.get("OMP_NUM_THREADS", "") or "").strip()
    applied_limit = 0
    apply_error = ""
    if requested:
        try:
            applied_limit = max(1, int(float(requested)))
            torch.set_num_threads(applied_limit)
        except Exception as exc:  # pragma: no cover - runtime diagnostic only
            apply_error = str(exc)
    return {
        "thread_env_limits": env_limits,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_thread_limit_requested": applied_limit,
        "torch_thread_limit_error": apply_error,
    }


def _weighted_scalar_heads_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    device = next(iter(outputs.values())).device
    total_loss = torch.tensor(0.0, device=device)
    total_weight = 0.0
    for name, target in targets.items():
        if weights is not None and name not in weights:
            continue
        head_weight = float((weights or {}).get(name, 1.0))
        total_loss = total_loss + nn.functional.smooth_l1_loss(outputs[name], target) * head_weight
        total_weight += head_weight
    if total_weight <= 0.0:
        return total_loss
    return total_loss / float(total_weight)


def _lifecycle_arbitration_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    device = outputs["action_logits"].device
    if "lifecycle_sell_gate" not in targets:
        return torch.tensor(0.0, device=device)
    probs = torch.softmax(outputs["action_logits"], dim=-1)
    action_lookup = {name: idx for idx, name in enumerate(ACTION_CLASSES)}
    sell_prob = probs[:, action_lookup["reduce"]] + probs[:, action_lookup["exit"]]
    target = torch.clamp(targets["lifecycle_sell_gate"].to(device), 0.0, 1.0)
    return nn.functional.binary_cross_entropy(torch.clamp(sell_prob, 1.0e-4, 1.0 - 1.0e-4), target)


def _sell_rank_pairwise_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    device = outputs["action_logits"].device
    if "sell_rank_score" not in targets or "date_code" not in targets or "holding_flag_target" not in targets:
        return torch.tensor(0.0, device=device)
    score = outputs["sell_rank_score"].to(device)
    target = torch.clamp(targets["sell_rank_score"].to(device), 0.0, 1.0)
    date_code = targets["date_code"].to(device).long()
    held_mask = targets["holding_flag_target"].to(device) > 0.5
    total_loss = torch.tensor(0.0, device=device)
    pair_count = 0
    for date_value in torch.unique(date_code[held_mask]):
        mask = held_mask & (date_code == date_value)
        if int(mask.sum().detach().cpu()) < 2:
            continue
        local_score = score[mask]
        local_target = target[mask]
        target_diff = local_target[:, None] - local_target[None, :]
        useful = torch.abs(target_diff) > 0.06
        if not bool(useful.any().detach().cpu()):
            continue
        pred_diff = local_score[:, None] - local_score[None, :]
        direction = torch.sign(target_diff[useful])
        weight = torch.clamp(torch.abs(target_diff[useful]), 0.05, 1.0)
        pair_loss = nn.functional.softplus(-direction * pred_diff[useful] * 6.0) * weight
        total_loss = total_loss + pair_loss.mean()
        pair_count += 1
    if pair_count <= 0:
        return torch.tensor(0.0, device=device)
    return total_loss / float(pair_count)


def _date_rank_pairwise_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    score_name: str,
    target_name: str,
    candidate_mask_name: str | None = None,
    min_target_gap: float = 0.05,
) -> torch.Tensor:
    if score_name not in outputs or target_name not in targets or "date_code" not in targets:
        device = next(iter(outputs.values())).device
        return torch.tensor(0.0, device=device)
    device = outputs[score_name].device
    score = outputs[score_name].to(device)
    target = torch.clamp(targets[target_name].to(device), 0.0, 1.0)
    date_code = targets["date_code"].to(device).long()
    candidate_mask = torch.isfinite(target)
    if candidate_mask_name and candidate_mask_name in targets:
        candidate_mask = candidate_mask & (targets[candidate_mask_name].to(device) > 0.5)
    elif "holding_flag_target" in targets and "source" in score_name:
        candidate_mask = candidate_mask & (targets["holding_flag_target"].to(device) > 0.5)
    total_loss = torch.tensor(0.0, device=device)
    pair_count = 0
    for date_value in torch.unique(date_code[candidate_mask]):
        mask = candidate_mask & (date_code == date_value)
        if int(mask.sum().detach().cpu()) < 2:
            continue
        local_score = score[mask]
        local_target = target[mask]
        target_diff = local_target[:, None] - local_target[None, :]
        useful = torch.abs(target_diff) > float(min_target_gap)
        if not bool(useful.any().detach().cpu()):
            continue
        pred_diff = local_score[:, None] - local_score[None, :]
        direction = torch.sign(target_diff[useful])
        weight = torch.clamp(torch.abs(target_diff[useful]), 0.05, 1.0)
        total_loss = total_loss + (nn.functional.softplus(-direction * pred_diff[useful] * 7.0) * weight).mean()
        pair_count += 1
    if pair_count <= 0:
        return torch.tensor(0.0, device=device)
    return total_loss / float(pair_count)


def _portfolio_cash_margin_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    device = outputs["action_logits"].device
    required = {
        "portfolio_daily_cash_score",
        "portfolio_daily_receiver_score",
        "portfolio_daily_source_score",
    }
    if not required.issubset(outputs) or not required.issubset(targets):
        return torch.tensor(0.0, device=device)
    cash_score = torch.clamp(outputs["portfolio_daily_cash_score"].to(device), 0.0, 1.0)
    receiver_score = torch.clamp(outputs["portfolio_daily_receiver_score"].to(device), 0.0, 1.0)
    source_score = torch.clamp(outputs["portfolio_daily_source_score"].to(device), 0.0, 1.0)
    target_cash = torch.clamp(targets["portfolio_daily_cash_score"].to(device), 0.0, 1.0)
    target_receiver = torch.clamp(targets["portfolio_daily_receiver_score"].to(device), 0.0, 1.0)
    target_source = torch.clamp(targets["portfolio_daily_source_score"].to(device), 0.0, 1.0)
    cash_dominant = target_cash > torch.maximum(target_receiver, target_source) + 0.08
    deploy_dominant = target_receiver > target_cash + 0.08
    if not bool((cash_dominant | deploy_dominant).any().detach().cpu()):
        return torch.tensor(0.0, device=device)
    margin = torch.tensor(0.08, device=device)
    terms: list[torch.Tensor] = []
    if int(cash_dominant.sum().detach().cpu().item()) > 0:
        terms.append(
            torch.relu(
                margin - (cash_score[cash_dominant] - torch.maximum(receiver_score[cash_dominant], source_score[cash_dominant]))
            ).mean()
        )
    if int(deploy_dominant.sum().detach().cpu().item()) > 0:
        terms.append(torch.relu(margin - (receiver_score[deploy_dominant] - cash_score[deploy_dominant])).mean())
    return torch.stack(terms).mean() if terms else torch.tensor(0.0, device=device)


def _unified_allocation_consistency_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    required_outputs = {
        "portfolio_daily_unified_receiver_score",
        "portfolio_daily_unified_source_score",
        "portfolio_daily_unified_cash_score",
        "portfolio_daily_unified_allocation_objective",
    }
    required_targets = {
        "portfolio_daily_unified_receiver_score",
        "portfolio_daily_unified_source_score",
        "portfolio_daily_unified_cash_score",
        "portfolio_daily_source_positive_forward_penalty",
        "portfolio_daily_source_opportunity_cost_penalty",
        "portfolio_daily_receiver_source_spread_reward",
        "portfolio_daily_unified_allocation_objective",
    }
    if not required_outputs.issubset(outputs) or not required_targets.issubset(targets):
        device = next(iter(outputs.values())).device
        return torch.tensor(0.0, device=device)
    device = outputs["portfolio_daily_unified_receiver_score"].device
    receiver_score = torch.clamp(outputs["portfolio_daily_unified_receiver_score"].to(device), 0.0, 1.0)
    source_score = torch.clamp(outputs["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    cash_score = torch.clamp(outputs["portfolio_daily_unified_cash_score"].to(device), 0.0, 1.0)
    objective_score = torch.clamp(outputs["portfolio_daily_unified_allocation_objective"].to(device), 0.0, 1.0)

    target_receiver = torch.clamp(targets["portfolio_daily_unified_receiver_score"].to(device), 0.0, 1.0)
    target_source = torch.clamp(targets["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    target_cash = torch.clamp(targets["portfolio_daily_unified_cash_score"].to(device), 0.0, 1.0)
    target_objective = torch.clamp(targets["portfolio_daily_unified_allocation_objective"].to(device), 0.0, 1.0)
    positive_forward_penalty = torch.clamp(
        targets["portfolio_daily_source_positive_forward_penalty"].to(device),
        0.0,
        1.0,
    )
    opportunity_cost_penalty = torch.clamp(
        targets["portfolio_daily_source_opportunity_cost_penalty"].to(device),
        0.0,
        1.0,
    )
    spread_reward = torch.clamp(targets["portfolio_daily_receiver_source_spread_reward"].to(device), 0.0, 1.0)
    zero = torch.zeros_like(target_receiver)
    market_downside = torch.clamp(targets.get("market_downside_pressure", zero).to(device), 0.0, 1.0)
    cash_regime = torch.clamp(targets.get("cash_regime_pressure", zero).to(device), 0.0, 1.0)
    portfolio_cash_pressure = torch.clamp(targets.get("portfolio_cash_pressure", zero).to(device), 0.0, 1.0)
    multi_horizon_forward_risk = torch.clamp(targets.get("multi_horizon_forward_risk", zero).to(device), 0.0, 1.0)
    portfolio_drawdown = torch.clamp(targets.get("portfolio_drawdown_20d", zero).to(device), -1.0, 0.25)
    forward_benchmark_1d = torch.clamp(targets.get("forward_benchmark_return_1d", zero).to(device), -0.25, 0.25)
    forward_benchmark_3d = torch.clamp(targets.get("forward_benchmark_return_3d", zero).to(device), -0.35, 0.35)
    transfer_score = torch.clamp(targets.get("portfolio_daily_allocation_transfer_score", zero).to(device), 0.0, 1.0)
    receiver_mask = torch.clamp(targets.get("portfolio_daily_receiver_candidate_mask", zero).to(device), 0.0, 1.0)
    source_mask = torch.clamp(targets.get("portfolio_daily_source_candidate_mask", zero).to(device), 0.0, 1.0)

    drawdown_pressure = torch.clamp((-portfolio_drawdown - 0.02) / 0.10, 0.0, 1.0)
    benchmark_downside_timing = torch.clamp(
        0.62 * torch.clamp(-forward_benchmark_1d / 0.025, 0.0, 1.0)
        + 0.38 * torch.clamp(-forward_benchmark_3d / 0.045, 0.0, 1.0),
        0.0,
        1.0,
    )
    benchmark_upside_timing = torch.clamp(
        0.62 * torch.clamp(forward_benchmark_1d / 0.025, 0.0, 1.0)
        + 0.38 * torch.clamp(forward_benchmark_3d / 0.045, 0.0, 1.0),
        0.0,
        1.0,
    )
    risk_off_target = torch.clamp(
        0.28 * market_downside
        + 0.22 * cash_regime
        + 0.20 * drawdown_pressure
        + 0.18 * multi_horizon_forward_risk
        + 0.12 * portfolio_cash_pressure,
        0.0,
        1.0,
    )
    deploy_pressure_target = torch.clamp(
        0.36 * target_receiver
        + 0.24 * spread_reward
        + 0.22 * transfer_score
        + 0.18 * target_source,
        0.0,
        1.0,
    )
    cash_behavior_target = torch.clamp(
        0.46 * target_cash
        + 0.26 * risk_off_target
        + 0.34 * benchmark_downside_timing
        + 0.08 * positive_forward_penalty
        + 0.08 * opportunity_cost_penalty
        - 0.24 * benchmark_upside_timing
        - 0.22 * deploy_pressure_target,
        0.0,
        1.0,
    )
    source_behavior_target = torch.clamp(
        target_source
        + 0.18 * spread_reward
        + 0.10 * target_objective
        - 0.34 * positive_forward_penalty
        - 0.26 * opportunity_cost_penalty,
        0.0,
        1.0,
    )
    objective_behavior_target = torch.clamp(
        target_objective
        + 0.10 * deploy_pressure_target * (1.0 - cash_behavior_target)
        + 0.06 * risk_off_target * cash_behavior_target
        - 0.18 * positive_forward_penalty
        - 0.14 * opportunity_cost_penalty,
        0.0,
        1.0,
    )

    regression_loss = (
        nn.functional.smooth_l1_loss(receiver_score, target_receiver) * 0.18
        + nn.functional.smooth_l1_loss(source_score, source_behavior_target) * 0.26
        + nn.functional.smooth_l1_loss(cash_score, cash_behavior_target) * 0.28
        + nn.functional.smooth_l1_loss(objective_score, objective_behavior_target) * 0.18
    )
    bad_source_pressure = torch.clamp(
        torch.maximum(positive_forward_penalty, opportunity_cost_penalty) - spread_reward + 0.06,
        0.0,
        1.0,
    )
    bad_source_loss = (bad_source_pressure * source_score * torch.clamp(source_mask + 0.25, 0.0, 1.0)).mean()
    dead_cash_pressure = torch.clamp(deploy_pressure_target - risk_off_target + 0.08, 0.0, 1.0)
    dead_cash_loss = (dead_cash_pressure * cash_score * torch.clamp(receiver_mask + 0.25, 0.0, 1.0)).mean()

    margin = torch.tensor(0.10, device=device)
    margin_terms: list[torch.Tensor] = []
    defensive_cash = cash_behavior_target > torch.maximum(target_receiver, source_behavior_target) + 0.08
    receiver_dominant = target_receiver > torch.maximum(cash_behavior_target, source_behavior_target) + 0.08
    source_dominant = source_behavior_target > torch.maximum(cash_behavior_target, target_receiver) + 0.08
    if int(defensive_cash.sum().detach().cpu().item()) > 0:
        margin_terms.append(
            torch.relu(
                margin
                - (
                    cash_score[defensive_cash]
                    - torch.maximum(receiver_score[defensive_cash], source_score[defensive_cash])
                )
            ).mean()
        )
    if int(receiver_dominant.sum().detach().cpu().item()) > 0:
        margin_terms.append(
            torch.relu(
                margin
                - (
                    receiver_score[receiver_dominant]
                    - torch.maximum(cash_score[receiver_dominant], source_score[receiver_dominant])
                )
            ).mean()
        )
    if int(source_dominant.sum().detach().cpu().item()) > 0:
        margin_terms.append(
            torch.relu(
                margin
                - (
                    source_score[source_dominant]
                    - torch.maximum(cash_score[source_dominant], receiver_score[source_dominant])
                )
            ).mean()
        )
    margin_loss = torch.stack(margin_terms).mean() if margin_terms else torch.tensor(0.0, device=device)
    return regression_loss + bad_source_loss * 0.26 + dead_cash_loss * 0.22 + margin_loss * 0.24


def _decision_focused_allocation_regret_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    required_outputs = {
        "portfolio_daily_unified_receiver_score",
        "portfolio_daily_unified_source_score",
        "portfolio_daily_unified_cash_score",
        "portfolio_daily_unified_allocation_objective",
    }
    required_targets = {
        "portfolio_daily_unified_receiver_score",
        "portfolio_daily_unified_source_score",
        "portfolio_daily_unified_cash_score",
        "portfolio_daily_unified_allocation_objective",
        "portfolio_daily_source_positive_forward_penalty",
        "portfolio_daily_source_opportunity_cost_penalty",
        "portfolio_daily_receiver_source_spread_reward",
        "portfolio_daily_receiver_candidate_mask",
        "portfolio_daily_source_candidate_mask",
    }
    device = next(iter(outputs.values())).device
    if not required_outputs.issubset(outputs) or not required_targets.issubset(targets):
        return torch.tensor(0.0, device=device)

    receiver_score = torch.clamp(outputs["portfolio_daily_unified_receiver_score"].to(device), 0.0, 1.0)
    source_score = torch.clamp(outputs["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    cash_score = torch.clamp(outputs["portfolio_daily_unified_cash_score"].to(device), 0.0, 1.0)
    objective_score = torch.clamp(outputs["portfolio_daily_unified_allocation_objective"].to(device), 0.0, 1.0)

    target_receiver = torch.clamp(targets["portfolio_daily_unified_receiver_score"].to(device), 0.0, 1.0)
    target_source = torch.clamp(targets["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    target_cash = torch.clamp(targets["portfolio_daily_unified_cash_score"].to(device), 0.0, 1.0)
    target_objective = torch.clamp(targets["portfolio_daily_unified_allocation_objective"].to(device), 0.0, 1.0)
    zero = torch.zeros_like(target_receiver)
    receiver_mask = torch.clamp(targets["portfolio_daily_receiver_candidate_mask"].to(device), 0.0, 1.0)
    source_mask = torch.clamp(targets["portfolio_daily_source_candidate_mask"].to(device), 0.0, 1.0)
    receiver_forward = torch.clamp(
        targets.get("portfolio_daily_receiver_forward_excess_5d", targets.get("forward_excess_5d", zero)).to(device),
        -0.35,
        0.35,
    )
    source_forward = torch.clamp(
        targets.get("portfolio_daily_source_forward_excess_5d", targets.get("forward_excess_5d", zero)).to(device),
        -0.35,
        0.35,
    )
    positive_forward_penalty = torch.clamp(
        torch.maximum(
            targets["portfolio_daily_source_positive_forward_penalty"].to(device),
            torch.clamp(source_forward / 0.08, 0.0, 1.0),
        ),
        0.0,
        1.0,
    )
    opportunity_cost_penalty = torch.clamp(
        targets["portfolio_daily_source_opportunity_cost_penalty"].to(device),
        0.0,
        1.0,
    )
    hard_negative_penalty = torch.clamp(
        torch.maximum(
            targets.get("portfolio_daily_source_hard_negative_penalty", zero).to(device),
            targets.get("portfolio_daily_source_strong_false_sell_penalty", zero).to(device),
        ),
        0.0,
        1.0,
    )
    spread_reward = torch.clamp(targets["portfolio_daily_receiver_source_spread_reward"].to(device), 0.0, 1.0)
    transfer_score = torch.clamp(targets.get("portfolio_daily_allocation_transfer_score", zero).to(device), 0.0, 1.0)
    market_downside = torch.clamp(targets.get("market_downside_pressure", zero).to(device), 0.0, 1.0)
    cash_regime = torch.clamp(targets.get("cash_regime_pressure", zero).to(device), 0.0, 1.0)
    portfolio_drawdown = torch.clamp(targets.get("portfolio_drawdown_20d", zero).to(device), -1.0, 0.25)
    drawdown_pressure = torch.clamp((-portfolio_drawdown - 0.02) / 0.10, 0.0, 1.0)
    risk_off_target = torch.clamp(
        0.40 * market_downside + 0.34 * cash_regime + 0.26 * drawdown_pressure,
        0.0,
        1.0,
    )

    receiver_oracle = torch.clamp(
        0.48 * target_receiver
        + 0.26 * torch.clamp(receiver_forward / 0.08, 0.0, 1.0)
        + 0.16 * spread_reward
        + 0.10 * transfer_score
        - 0.18 * risk_off_target,
        0.0,
        1.0,
    ) * receiver_mask
    source_release_oracle = torch.clamp(
        0.44 * target_source
        + 0.24 * torch.clamp(-source_forward / 0.08, 0.0, 1.0)
        + 0.16 * spread_reward
        + 0.10 * transfer_score
        - 0.40 * hard_negative_penalty
        - 0.26 * positive_forward_penalty
        - 0.22 * opportunity_cost_penalty,
        0.0,
        1.0,
    ) * source_mask
    false_source_pressure = torch.clamp(
        torch.maximum(hard_negative_penalty, 0.70 * positive_forward_penalty + 0.30 * opportunity_cost_penalty)
        + 0.22 * torch.clamp(source_forward / 0.08, 0.0, 1.0),
        0.0,
        1.0,
    ) * source_mask
    cash_oracle = torch.clamp(
        0.48 * target_cash
        + 0.34 * risk_off_target
        + 0.10 * false_source_pressure
        - 0.28 * receiver_oracle
        - 0.18 * source_release_oracle,
        0.0,
        1.0,
    )
    objective_oracle = torch.clamp(
        target_objective
        + 0.12 * receiver_oracle
        + 0.10 * source_release_oracle
        - 0.20 * false_source_pressure,
        0.0,
        1.0,
    )

    receiver_regret = (torch.relu(receiver_oracle - receiver_score) * torch.clamp(receiver_mask + 0.10, 0.0, 1.0)).mean()
    source_release_regret = (
        torch.relu(source_release_oracle - source_score)
        * torch.clamp(source_mask + 0.10, 0.0, 1.0)
        * (1.0 - hard_negative_penalty)
    ).mean()
    false_source_loss = (false_source_pressure * (source_score + 0.36 * objective_score)).mean()
    dead_cash_pressure = torch.clamp(
        receiver_oracle + 0.55 * source_release_oracle - risk_off_target + 0.06,
        0.0,
        1.0,
    )
    dead_cash_loss = (dead_cash_pressure * cash_score).mean()
    cash_regret = nn.functional.smooth_l1_loss(cash_score, cash_oracle)
    objective_regret = nn.functional.smooth_l1_loss(objective_score, objective_oracle)
    return (
        receiver_regret * 0.24
        + source_release_regret * 0.18
        + false_source_loss * 0.30
        + dead_cash_loss * 0.16
        + cash_regret * 0.06
        + objective_regret * 0.06
    )


def _allocation_objective_consolidation_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    required_outputs = {
        "portfolio_daily_unified_receiver_score",
        "portfolio_daily_unified_source_score",
        "portfolio_daily_unified_cash_score",
        "portfolio_daily_allocation_trade_quality_target",
        "portfolio_daily_allocation_cash_deployment_target",
        "portfolio_daily_allocation_risk_adjusted_return_target",
        "portfolio_daily_allocation_drawdown_control_target",
        "portfolio_daily_allocation_monthly_quality_target",
        "portfolio_daily_allocation_final_objective",
    }
    required_targets = {
        "portfolio_daily_unified_receiver_score",
        "portfolio_daily_unified_source_score",
        "portfolio_daily_unified_cash_score",
        "portfolio_daily_source_positive_forward_penalty",
        "portfolio_daily_source_opportunity_cost_penalty",
        "portfolio_daily_receiver_source_spread_reward",
        "portfolio_daily_allocation_trade_quality_target",
        "portfolio_daily_allocation_cash_deployment_target",
        "portfolio_daily_allocation_risk_adjusted_return_target",
        "portfolio_daily_allocation_drawdown_control_target",
        "portfolio_daily_allocation_monthly_quality_target",
        "portfolio_daily_allocation_final_objective",
        "portfolio_daily_receiver_candidate_mask",
        "portfolio_daily_source_candidate_mask",
    }
    device = next(iter(outputs.values())).device
    if not required_outputs.issubset(outputs) or not required_targets.issubset(targets):
        return torch.tensor(0.0, device=device)

    receiver_score = torch.clamp(outputs["portfolio_daily_unified_receiver_score"].to(device), 0.0, 1.0)
    source_score = torch.clamp(outputs["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    cash_score = torch.clamp(outputs["portfolio_daily_unified_cash_score"].to(device), 0.0, 1.0)
    pred_trade_quality = torch.clamp(outputs["portfolio_daily_allocation_trade_quality_target"].to(device), 0.0, 1.0)
    pred_cash_deployment = torch.clamp(outputs["portfolio_daily_allocation_cash_deployment_target"].to(device), 0.0, 1.0)
    pred_risk_return = torch.clamp(outputs["portfolio_daily_allocation_risk_adjusted_return_target"].to(device), 0.0, 1.0)
    pred_drawdown_control = torch.clamp(outputs["portfolio_daily_allocation_drawdown_control_target"].to(device), 0.0, 1.0)
    pred_monthly_quality = torch.clamp(outputs["portfolio_daily_allocation_monthly_quality_target"].to(device), 0.0, 1.0)
    pred_final_objective = torch.clamp(outputs["portfolio_daily_allocation_final_objective"].to(device), 0.0, 1.0)

    target_receiver = torch.clamp(targets["portfolio_daily_unified_receiver_score"].to(device), 0.0, 1.0)
    target_source = torch.clamp(targets["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    target_cash = torch.clamp(targets["portfolio_daily_unified_cash_score"].to(device), 0.0, 1.0)
    target_trade_quality = torch.clamp(targets["portfolio_daily_allocation_trade_quality_target"].to(device), 0.0, 1.0)
    target_cash_deployment = torch.clamp(targets["portfolio_daily_allocation_cash_deployment_target"].to(device), 0.0, 1.0)
    target_risk_return = torch.clamp(targets["portfolio_daily_allocation_risk_adjusted_return_target"].to(device), 0.0, 1.0)
    target_drawdown_control = torch.clamp(targets["portfolio_daily_allocation_drawdown_control_target"].to(device), 0.0, 1.0)
    target_monthly_quality = torch.clamp(targets["portfolio_daily_allocation_monthly_quality_target"].to(device), 0.0, 1.0)
    target_final_objective = torch.clamp(targets["portfolio_daily_allocation_final_objective"].to(device), 0.0, 1.0)
    positive_forward_penalty = torch.clamp(targets["portfolio_daily_source_positive_forward_penalty"].to(device), 0.0, 1.0)
    opportunity_cost_penalty = torch.clamp(targets["portfolio_daily_source_opportunity_cost_penalty"].to(device), 0.0, 1.0)
    zero = torch.zeros_like(target_receiver)
    hard_negative_penalty = torch.clamp(targets.get("portfolio_daily_source_hard_negative_penalty", zero).to(device), 0.0, 1.0)
    spread_reward = torch.clamp(targets["portfolio_daily_receiver_source_spread_reward"].to(device), 0.0, 1.0)
    receiver_mask = torch.clamp(targets["portfolio_daily_receiver_candidate_mask"].to(device), 0.0, 1.0)
    source_mask = torch.clamp(targets["portfolio_daily_source_candidate_mask"].to(device), 0.0, 1.0)

    regression_loss = (
        nn.functional.smooth_l1_loss(pred_trade_quality, target_trade_quality) * 0.15
        + nn.functional.smooth_l1_loss(pred_cash_deployment, target_cash_deployment) * 0.14
        + nn.functional.smooth_l1_loss(pred_risk_return, target_risk_return) * 0.16
        + nn.functional.smooth_l1_loss(pred_drawdown_control, target_drawdown_control) * 0.10
        + nn.functional.smooth_l1_loss(pred_monthly_quality, target_monthly_quality) * 0.12
        + nn.functional.smooth_l1_loss(pred_final_objective, target_final_objective) * 0.18
        + nn.functional.smooth_l1_loss(receiver_score, target_receiver) * 0.05
        + nn.functional.smooth_l1_loss(source_score, target_source) * 0.05
        + nn.functional.smooth_l1_loss(cash_score, target_cash) * 0.05
    )
    false_source_pressure = torch.clamp(
        torch.maximum(hard_negative_penalty, torch.maximum(positive_forward_penalty, opportunity_cost_penalty))
        - 0.42 * spread_reward
        + 0.08,
        0.0,
        1.0,
    ) * source_mask
    dead_cash_pressure = torch.clamp(
        target_cash_deployment + 0.42 * target_trade_quality + 0.28 * target_risk_return - target_drawdown_control + 0.06,
        0.0,
        1.0,
    ) * torch.clamp(receiver_mask + 0.25, 0.0, 1.0)
    receiver_underdeploy_loss = (
        torch.relu(target_receiver + 0.34 * target_cash_deployment - receiver_score)
        * torch.clamp(receiver_mask + 0.10, 0.0, 1.0)
    ).mean()
    source_release_underuse_loss = (
        torch.relu(target_source + 0.22 * target_trade_quality - source_score)
        * torch.clamp(source_mask + 0.10, 0.0, 1.0)
        * (1.0 - hard_negative_penalty)
    ).mean()
    false_source_loss = (false_source_pressure * (source_score + 0.36 * pred_final_objective)).mean()
    dead_cash_loss = (dead_cash_pressure * cash_score).mean()
    missed_objective_loss = torch.relu(target_final_objective - pred_final_objective).mean()
    defensive_cash_loss = (
        torch.relu(target_drawdown_control - torch.maximum(cash_score, pred_drawdown_control))
        * torch.clamp(target_drawdown_control - target_cash_deployment + 0.08, 0.0, 1.0)
    ).mean()
    return (
        regression_loss
        + receiver_underdeploy_loss * 0.18
        + source_release_underuse_loss * 0.12
        + false_source_loss * 0.34
        + dead_cash_loss * 0.26
        + missed_objective_loss * 0.10
        + defensive_cash_loss * 0.08
    )


def _risk_sensitive_allocation_objective_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    required_outputs = {
        "portfolio_daily_unified_receiver_score",
        "portfolio_daily_unified_source_score",
        "portfolio_daily_unified_cash_score",
        "portfolio_daily_allocation_uncertainty_pressure_target",
        "portfolio_daily_allocation_tail_risk_control_target",
        "portfolio_daily_allocation_decision_focused_objective",
    }
    required_targets = {
        "portfolio_daily_unified_receiver_score",
        "portfolio_daily_unified_source_score",
        "portfolio_daily_unified_cash_score",
        "portfolio_daily_allocation_uncertainty_pressure_target",
        "portfolio_daily_allocation_tail_risk_control_target",
        "portfolio_daily_allocation_decision_focused_objective",
        "portfolio_daily_allocation_drawdown_control_target",
        "portfolio_daily_allocation_cash_deployment_target",
        "portfolio_daily_receiver_candidate_mask",
        "portfolio_daily_source_candidate_mask",
    }
    device = next(iter(outputs.values())).device
    if not required_outputs.issubset(outputs) or not required_targets.issubset(targets):
        return torch.tensor(0.0, device=device)

    receiver_score = torch.clamp(outputs["portfolio_daily_unified_receiver_score"].to(device), 0.0, 1.0)
    source_score = torch.clamp(outputs["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    cash_score = torch.clamp(outputs["portfolio_daily_unified_cash_score"].to(device), 0.0, 1.0)
    pred_uncertainty = torch.clamp(outputs["portfolio_daily_allocation_uncertainty_pressure_target"].to(device), 0.0, 1.0)
    pred_tail = torch.clamp(outputs["portfolio_daily_allocation_tail_risk_control_target"].to(device), 0.0, 1.0)
    pred_decision = torch.clamp(outputs["portfolio_daily_allocation_decision_focused_objective"].to(device), 0.0, 1.0)

    target_receiver = torch.clamp(targets["portfolio_daily_unified_receiver_score"].to(device), 0.0, 1.0)
    target_source = torch.clamp(targets["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    target_cash = torch.clamp(targets["portfolio_daily_unified_cash_score"].to(device), 0.0, 1.0)
    target_uncertainty = torch.clamp(targets["portfolio_daily_allocation_uncertainty_pressure_target"].to(device), 0.0, 1.0)
    target_tail = torch.clamp(targets["portfolio_daily_allocation_tail_risk_control_target"].to(device), 0.0, 1.0)
    target_decision = torch.clamp(targets["portfolio_daily_allocation_decision_focused_objective"].to(device), 0.0, 1.0)
    target_drawdown = torch.clamp(targets["portfolio_daily_allocation_drawdown_control_target"].to(device), 0.0, 1.0)
    target_deploy = torch.clamp(targets["portfolio_daily_allocation_cash_deployment_target"].to(device), 0.0, 1.0)
    receiver_mask = torch.clamp(targets["portfolio_daily_receiver_candidate_mask"].to(device), 0.0, 1.0)
    source_mask = torch.clamp(targets["portfolio_daily_source_candidate_mask"].to(device), 0.0, 1.0)
    zero = torch.zeros_like(target_receiver)
    hard_negative = torch.clamp(targets.get("portfolio_daily_source_hard_negative_penalty", zero).to(device), 0.0, 1.0)
    positive_forward = torch.clamp(
        targets.get("portfolio_daily_source_positive_forward_penalty", zero).to(device),
        0.0,
        1.0,
    )
    opportunity = torch.clamp(
        targets.get("portfolio_daily_source_opportunity_cost_penalty", zero).to(device),
        0.0,
        1.0,
    )
    risk_pressure = torch.clamp(torch.maximum(target_uncertainty, torch.maximum(target_tail, target_drawdown)), 0.0, 1.0)
    clean_transfer_pressure = torch.clamp(
        target_decision + 0.28 * target_deploy - 0.42 * risk_pressure,
        0.0,
        1.0,
    )
    false_source_pressure = torch.clamp(
        torch.maximum(hard_negative, torch.maximum(positive_forward, opportunity)) + 0.22 * target_tail,
        0.0,
        1.0,
    ) * source_mask

    regression_loss = (
        nn.functional.smooth_l1_loss(pred_uncertainty, target_uncertainty) * 0.18
        + nn.functional.smooth_l1_loss(pred_tail, target_tail) * 0.20
        + nn.functional.smooth_l1_loss(pred_decision, target_decision) * 0.24
        + nn.functional.smooth_l1_loss(receiver_score, target_receiver) * 0.08
        + nn.functional.smooth_l1_loss(source_score, target_source) * 0.08
        + nn.functional.smooth_l1_loss(cash_score, target_cash) * 0.08
    )
    tail_deploy_loss = (
        risk_pressure
        * torch.square(receiver_score)
        * torch.clamp(1.0 - target_deploy + 0.15, 0.0, 1.0)
        * torch.clamp(receiver_mask + 0.15, 0.0, 1.0)
    ).mean()
    weak_cash_defense_loss = (
        torch.relu(torch.maximum(target_tail, target_drawdown) - cash_score)
        * torch.clamp(risk_pressure + 0.10, 0.0, 1.0)
    ).mean()
    clean_receiver_regret = (
        torch.relu(clean_transfer_pressure * target_receiver - receiver_score)
        * receiver_mask
        * torch.clamp(1.0 - risk_pressure, 0.0, 1.0)
    ).mean()
    clean_source_regret = (
        torch.relu(clean_transfer_pressure * target_source - source_score)
        * source_mask
        * torch.clamp(1.0 - false_source_pressure, 0.0, 1.0)
    ).mean()
    false_source_loss = (false_source_pressure * torch.square(source_score)).mean()
    decision_reversal_loss = torch.relu(target_decision - pred_decision).mean() * 0.55 + torch.relu(
        pred_decision - target_decision
    ).mean() * 0.45

    return (
        regression_loss
        + tail_deploy_loss * 0.26
        + weak_cash_defense_loss * 0.30
        + clean_receiver_regret * 0.14
        + clean_source_regret * 0.12
        + false_source_loss * 0.18
        + decision_reversal_loss * 0.12
    )


def _portfolio_utility_credit_closure_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    required_outputs = {
        "portfolio_daily_unified_receiver_score",
        "portfolio_daily_unified_source_score",
        "portfolio_daily_unified_cash_score",
        "portfolio_daily_allocation_net_utility_target",
        "portfolio_daily_allocation_credit_closure_target",
        "portfolio_daily_allocation_resource_efficiency_target",
    }
    required_targets = {
        "portfolio_daily_unified_receiver_score",
        "portfolio_daily_unified_source_score",
        "portfolio_daily_unified_cash_score",
        "portfolio_daily_allocation_net_utility_target",
        "portfolio_daily_allocation_credit_closure_target",
        "portfolio_daily_allocation_resource_efficiency_target",
        "portfolio_daily_receiver_candidate_mask",
        "portfolio_daily_source_candidate_mask",
    }
    device = next(iter(outputs.values())).device
    if not required_outputs.issubset(outputs) or not required_targets.issubset(targets):
        return torch.tensor(0.0, device=device)

    receiver_score = torch.clamp(outputs["portfolio_daily_unified_receiver_score"].to(device), 0.0, 1.0)
    source_score = torch.clamp(outputs["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    cash_score = torch.clamp(outputs["portfolio_daily_unified_cash_score"].to(device), 0.0, 1.0)
    pred_net_utility = torch.clamp(outputs["portfolio_daily_allocation_net_utility_target"].to(device), 0.0, 1.0)
    pred_credit = torch.clamp(outputs["portfolio_daily_allocation_credit_closure_target"].to(device), 0.0, 1.0)
    pred_efficiency = torch.clamp(outputs["portfolio_daily_allocation_resource_efficiency_target"].to(device), 0.0, 1.0)

    target_receiver = torch.clamp(targets["portfolio_daily_unified_receiver_score"].to(device), 0.0, 1.0)
    target_source = torch.clamp(targets["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    target_cash = torch.clamp(targets["portfolio_daily_unified_cash_score"].to(device), 0.0, 1.0)
    target_net_utility = torch.clamp(targets["portfolio_daily_allocation_net_utility_target"].to(device), 0.0, 1.0)
    target_credit = torch.clamp(targets["portfolio_daily_allocation_credit_closure_target"].to(device), 0.0, 1.0)
    target_efficiency = torch.clamp(targets["portfolio_daily_allocation_resource_efficiency_target"].to(device), 0.0, 1.0)
    receiver_mask = torch.clamp(targets["portfolio_daily_receiver_candidate_mask"].to(device), 0.0, 1.0)
    source_mask = torch.clamp(targets["portfolio_daily_source_candidate_mask"].to(device), 0.0, 1.0)
    zero = torch.zeros_like(target_net_utility)
    uncertainty = torch.clamp(
        targets.get("portfolio_daily_allocation_uncertainty_pressure_target", zero).to(device),
        0.0,
        1.0,
    )
    tail = torch.clamp(
        targets.get("portfolio_daily_allocation_tail_risk_control_target", zero).to(device),
        0.0,
        1.0,
    )
    drawdown = torch.clamp(
        targets.get("portfolio_daily_allocation_drawdown_control_target", zero).to(device),
        0.0,
        1.0,
    )
    deploy_target = torch.clamp(
        targets.get("portfolio_daily_allocation_cash_deployment_target", target_net_utility).to(device),
        0.0,
        1.0,
    )
    risk_pressure = torch.clamp(torch.maximum(uncertainty, torch.maximum(tail, drawdown)), 0.0, 1.0)
    utility_pressure = torch.clamp(0.40 * target_net_utility + 0.36 * target_credit + 0.24 * target_efficiency, 0.0, 1.0)
    closure_pressure = torch.clamp(target_credit + 0.18 * target_efficiency - 0.22 * risk_pressure, 0.0, 1.0)
    source_release_need = torch.clamp(closure_pressure * target_source * source_mask, 0.0, 1.0)
    receiver_deploy_need = torch.clamp(closure_pressure * target_receiver * receiver_mask, 0.0, 1.0)

    regression_loss = (
        nn.functional.smooth_l1_loss(pred_net_utility, target_net_utility) * 0.26
        + nn.functional.smooth_l1_loss(pred_credit, target_credit) * 0.28
        + nn.functional.smooth_l1_loss(pred_efficiency, target_efficiency) * 0.22
        + nn.functional.smooth_l1_loss(receiver_score, target_receiver) * 0.08
        + nn.functional.smooth_l1_loss(source_score, target_source) * 0.08
        + nn.functional.smooth_l1_loss(cash_score, target_cash) * 0.06
    )
    source_disconnect_loss = (torch.relu(source_release_need - source_score) * source_mask).mean()
    receiver_disconnect_loss = (torch.relu(receiver_deploy_need - receiver_score) * receiver_mask).mean()
    dead_cash_loss = (
        torch.square(cash_score)
        * torch.clamp(utility_pressure + deploy_target - risk_pressure, 0.0, 1.0)
        * torch.clamp(receiver_mask + source_mask, 0.0, 1.0)
    ).mean()
    risk_budget_loss = (
        torch.square(receiver_score)
        * torch.clamp(risk_pressure - target_efficiency - 0.10, 0.0, 1.0)
        * receiver_mask
    ).mean()
    utility_reversal_loss = (
        torch.relu(target_net_utility - pred_net_utility)
        + torch.relu(target_credit - pred_credit)
        + torch.relu(target_efficiency - pred_efficiency)
    ).mean()

    return (
        regression_loss
        + source_disconnect_loss * 0.22
        + receiver_disconnect_loss * 0.22
        + dead_cash_loss * 0.24
        + risk_budget_loss * 0.14
        + utility_reversal_loss * 0.14
    )


def _portfolio_primal_dual_decision_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    required_outputs = {
        "portfolio_daily_unified_receiver_score",
        "portfolio_daily_unified_source_score",
        "portfolio_daily_unified_cash_score",
        "portfolio_daily_allocation_final_objective",
        "portfolio_daily_allocation_net_utility_target",
        "portfolio_daily_allocation_credit_closure_target",
        "portfolio_daily_allocation_resource_efficiency_target",
    }
    required_targets = {
        "date_code",
        "current_weight",
        "portfolio_daily_receiver_candidate_mask",
        "portfolio_daily_source_candidate_mask",
        "portfolio_daily_unified_receiver_score",
        "portfolio_daily_unified_source_score",
        "portfolio_daily_unified_cash_score",
    }
    device = next(iter(outputs.values())).device
    if not required_outputs.issubset(outputs) or not required_targets.issubset(targets):
        return torch.tensor(0.0, device=device)

    receiver_score = torch.clamp(outputs["portfolio_daily_unified_receiver_score"].to(device), 0.0, 1.0)
    source_score = torch.clamp(outputs["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    cash_score = torch.clamp(outputs["portfolio_daily_unified_cash_score"].to(device), 0.0, 1.0)
    final_objective = torch.clamp(outputs["portfolio_daily_allocation_final_objective"].to(device), 0.0, 1.0)
    net_utility = torch.clamp(outputs["portfolio_daily_allocation_net_utility_target"].to(device), 0.0, 1.0)
    credit_closure = torch.clamp(outputs["portfolio_daily_allocation_credit_closure_target"].to(device), 0.0, 1.0)
    resource_efficiency = torch.clamp(outputs["portfolio_daily_allocation_resource_efficiency_target"].to(device), 0.0, 1.0)

    target_receiver = torch.clamp(targets["portfolio_daily_unified_receiver_score"].to(device), 0.0, 1.0)
    target_source = torch.clamp(targets["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    target_cash = torch.clamp(targets["portfolio_daily_unified_cash_score"].to(device), 0.0, 1.0)
    receiver_mask = torch.clamp(targets["portfolio_daily_receiver_candidate_mask"].to(device), 0.0, 1.0)
    source_mask = torch.clamp(targets["portfolio_daily_source_candidate_mask"].to(device), 0.0, 1.0)
    current_weight = torch.clamp(targets["current_weight"].to(device), 0.0, 1.0)
    date_code = targets["date_code"].to(device).long()
    zero = torch.zeros_like(receiver_score)

    receiver_forward = torch.clamp(
        targets.get("portfolio_daily_receiver_forward_excess_5d", targets.get("forward_excess_5d", zero)).to(device),
        -0.35,
        0.35,
    )
    source_forward = torch.clamp(
        targets.get("portfolio_daily_source_forward_excess_5d", targets.get("forward_excess_5d", zero)).to(device),
        -0.35,
        0.35,
    )
    target_net = torch.clamp(targets.get("portfolio_daily_allocation_net_utility_target", target_receiver).to(device), 0.0, 1.0)
    target_credit = torch.clamp(targets.get("portfolio_daily_allocation_credit_closure_target", target_source).to(device), 0.0, 1.0)
    target_efficiency = torch.clamp(
        targets.get("portfolio_daily_allocation_resource_efficiency_target", target_receiver).to(device),
        0.0,
        1.0,
    )
    uncertainty = torch.clamp(
        targets.get("portfolio_daily_allocation_uncertainty_pressure_target", zero).to(device),
        0.0,
        1.0,
    )
    tail = torch.clamp(
        targets.get("portfolio_daily_allocation_tail_risk_control_target", zero).to(device),
        0.0,
        1.0,
    )
    drawdown = torch.clamp(
        targets.get("portfolio_daily_allocation_drawdown_control_target", zero).to(device),
        0.0,
        1.0,
    )
    deploy_target = torch.clamp(
        targets.get("portfolio_daily_allocation_cash_deployment_target", target_net).to(device),
        0.0,
        1.0,
    )
    positive_forward = torch.clamp(
        torch.maximum(
            targets.get("portfolio_daily_source_positive_forward_penalty", zero).to(device),
            torch.clamp(source_forward / 0.08, 0.0, 1.0),
        ),
        0.0,
        1.0,
    )
    opportunity = torch.clamp(
        targets.get("portfolio_daily_source_opportunity_cost_penalty", zero).to(device),
        0.0,
        1.0,
    )
    hard_negative = torch.clamp(
        torch.maximum(
            targets.get("portfolio_daily_source_hard_negative_penalty", zero).to(device),
            targets.get("portfolio_daily_source_tail_false_sell_penalty", zero).to(device),
        ),
        0.0,
        1.0,
    )
    release_preference = torch.clamp(
        targets.get("portfolio_daily_source_release_preference", target_source).to(device),
        0.0,
        1.0,
    )
    spread_reward = torch.clamp(
        targets.get("portfolio_daily_receiver_source_spread_reward", zero).to(device),
        0.0,
        1.0,
    )
    transfer_regret = torch.clamp(
        targets.get("portfolio_daily_transfer_regret_target", target_credit).to(device),
        0.0,
        1.0,
    )
    risk_pressure = torch.clamp(torch.maximum(uncertainty, torch.maximum(tail, drawdown)), 0.0, 1.0)
    false_source_pressure = torch.clamp(
        torch.maximum(hard_negative, 0.60 * positive_forward + 0.40 * opportunity),
        0.0,
        1.0,
    )

    day_losses: list[torch.Tensor] = []
    for date_value in torch.unique(date_code):
        day_mask = date_code == date_value
        receiver_day = day_mask & (receiver_mask > 0.05)
        source_day = day_mask & (source_mask > 0.05) & (current_weight > 1.0e-8)
        if int(receiver_day.sum().detach().cpu().item()) == 0 and int(source_day.sum().detach().cpu().item()) == 0:
            continue

        day_risk = risk_pressure[day_mask].mean()
        day_cash_score = cash_score[day_mask].mean()
        day_credit = target_credit[day_mask].mean()
        day_utility = target_net[day_mask].mean()
        day_efficiency = target_efficiency[day_mask].mean()
        desired_deploy = torch.clamp(
            0.40 * day_credit + 0.34 * day_utility + 0.20 * day_efficiency + 0.12 * deploy_target[day_mask].mean() - 0.42 * day_risk,
            0.0,
            1.0,
        )
        current_cash_proxy = torch.clamp(1.0 - current_weight[day_mask].sum(), 0.0, 1.0)
        desired_release = torch.clamp(desired_deploy - current_cash_proxy + 0.08 + 0.20 * transfer_regret[day_mask].mean(), 0.0, 1.0)

        pred_receiver_utility = torch.tensor(0.0, device=device)
        oracle_receiver_utility = torch.tensor(0.0, device=device)
        if int(receiver_day.sum().detach().cpu().item()) > 0:
            receiver_value = torch.clamp(
                0.52 * torch.clamp(receiver_forward[receiver_day] / 0.10, -1.0, 1.0)
                + 0.20 * target_receiver[receiver_day]
                + 0.16 * target_net[receiver_day]
                + 0.12 * spread_reward[receiver_day]
                - 0.22 * risk_pressure[receiver_day],
                -1.0,
                1.0,
            )
            pred_receiver_weights = torch.softmax(
                4.0
                * (
                    receiver_score[receiver_day]
                    + 0.30 * net_utility[receiver_day]
                    + 0.22 * credit_closure[receiver_day]
                    + 0.18 * resource_efficiency[receiver_day]
                    + 0.12 * final_objective[receiver_day]
                    - 0.22 * risk_pressure[receiver_day]
                ),
                dim=0,
            )
            oracle_receiver_weights = torch.softmax(
                4.0 * (target_receiver[receiver_day] + receiver_value + 0.12 * target_credit[receiver_day]),
                dim=0,
            )
            pred_receiver_utility = desired_deploy * (pred_receiver_weights * receiver_value).sum()
            oracle_receiver_utility = desired_deploy * (oracle_receiver_weights * receiver_value).sum()

        pred_source_utility = torch.tensor(0.0, device=device)
        oracle_source_utility = torch.tensor(0.0, device=device)
        false_source_loss = torch.tensor(0.0, device=device)
        if int(source_day.sum().detach().cpu().item()) > 0:
            source_keep_cost = torch.clamp(
                0.46 * torch.clamp(source_forward[source_day] / 0.10, -1.0, 1.0)
                + 0.22 * false_source_pressure[source_day]
                + 0.14 * opportunity[source_day]
                - 0.22 * release_preference[source_day]
                - 0.14 * target_credit[source_day],
                -1.0,
                1.0,
            )
            source_release_value = torch.clamp(-source_keep_cost + 0.24 * release_preference[source_day], -1.0, 1.0)
            pred_source_weights = torch.softmax(
                4.0
                * (
                    source_score[source_day]
                    + 0.24 * credit_closure[source_day]
                    + 0.18 * resource_efficiency[source_day]
                    + 0.14 * net_utility[source_day]
                    - 0.52 * false_source_pressure[source_day]
                ),
                dim=0,
            )
            oracle_source_weights = torch.softmax(
                4.0 * (source_release_value + target_source[source_day] - 0.80 * false_source_pressure[source_day]),
                dim=0,
            )
            pred_source_utility = desired_release * (pred_source_weights * source_release_value).sum()
            oracle_source_utility = desired_release * (oracle_source_weights * source_release_value).sum()
            false_source_loss = desired_release * (pred_source_weights * false_source_pressure[source_day]).sum()

        pred_utility = pred_receiver_utility + pred_source_utility
        oracle_utility = oracle_receiver_utility + oracle_source_utility
        regret = torch.relu(oracle_utility - pred_utility + 0.02)
        dead_cash_loss = torch.relu(desired_deploy - day_risk + 0.04) * torch.square(day_cash_score)
        risk_cash_loss = torch.relu(day_risk - day_cash_score + 0.06) * torch.clamp(day_risk - desired_deploy + 0.08, 0.0, 1.0)
        funding_imbalance = torch.square(torch.relu(desired_deploy - desired_release - current_cash_proxy - 0.02))
        day_losses.append(regret + 0.40 * false_source_loss + 0.26 * dead_cash_loss + 0.20 * risk_cash_loss + 0.18 * funding_imbalance)

    if not day_losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(day_losses).mean()


def _sinkhorn_transport_plan(
    logits: torch.Tensor,
    row_marginal: torch.Tensor,
    col_marginal: torch.Tensor,
    *,
    iterations: int = 7,
) -> torch.Tensor:
    row = torch.clamp(row_marginal, min=1.0e-6)
    row = row / torch.clamp(row.sum(), min=1.0e-6)
    col = torch.clamp(col_marginal, min=1.0e-6)
    col = col / torch.clamp(col.sum(), min=1.0e-6)
    plan = torch.exp(torch.clamp(logits - logits.max(), min=-30.0, max=30.0))
    for _ in range(iterations):
        plan = plan * (row.unsqueeze(1) / torch.clamp(plan.sum(dim=1, keepdim=True), min=1.0e-6))
        plan = plan * (col.unsqueeze(0) / torch.clamp(plan.sum(dim=0, keepdim=True), min=1.0e-6))
    return plan / torch.clamp(plan.sum(), min=1.0e-6)


def _portfolio_entropic_transport_decision_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    required_outputs = {
        "portfolio_daily_unified_receiver_score",
        "portfolio_daily_unified_source_score",
        "portfolio_daily_unified_cash_score",
        "portfolio_daily_allocation_final_objective",
        "portfolio_daily_allocation_net_utility_target",
        "portfolio_daily_allocation_credit_closure_target",
        "portfolio_daily_allocation_resource_efficiency_target",
    }
    required_targets = {
        "date_code",
        "current_weight",
        "portfolio_daily_receiver_candidate_mask",
        "portfolio_daily_source_candidate_mask",
        "portfolio_daily_unified_receiver_score",
        "portfolio_daily_unified_source_score",
        "portfolio_daily_unified_cash_score",
    }
    device = next(iter(outputs.values())).device
    if not required_outputs.issubset(outputs) or not required_targets.issubset(targets):
        return torch.tensor(0.0, device=device)

    receiver_score = torch.clamp(outputs["portfolio_daily_unified_receiver_score"].to(device), 0.0, 1.0)
    source_score = torch.clamp(outputs["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    cash_score = torch.clamp(outputs["portfolio_daily_unified_cash_score"].to(device), 0.0, 1.0)
    final_objective = torch.clamp(outputs["portfolio_daily_allocation_final_objective"].to(device), 0.0, 1.0)
    net_utility = torch.clamp(outputs["portfolio_daily_allocation_net_utility_target"].to(device), 0.0, 1.0)
    credit_closure = torch.clamp(outputs["portfolio_daily_allocation_credit_closure_target"].to(device), 0.0, 1.0)
    resource_efficiency = torch.clamp(outputs["portfolio_daily_allocation_resource_efficiency_target"].to(device), 0.0, 1.0)

    target_receiver = torch.clamp(targets["portfolio_daily_unified_receiver_score"].to(device), 0.0, 1.0)
    target_source = torch.clamp(targets["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    target_cash = torch.clamp(targets["portfolio_daily_unified_cash_score"].to(device), 0.0, 1.0)
    receiver_mask = torch.clamp(targets["portfolio_daily_receiver_candidate_mask"].to(device), 0.0, 1.0)
    source_mask = torch.clamp(targets["portfolio_daily_source_candidate_mask"].to(device), 0.0, 1.0)
    current_weight = torch.clamp(targets["current_weight"].to(device), 0.0, 1.0)
    date_code = targets["date_code"].to(device).long()
    zero = torch.zeros_like(receiver_score)

    receiver_forward = torch.clamp(
        targets.get("portfolio_daily_receiver_forward_excess_5d", targets.get("forward_excess_5d", zero)).to(device),
        -0.35,
        0.35,
    )
    source_forward = torch.clamp(
        targets.get("portfolio_daily_source_forward_excess_5d", targets.get("forward_excess_5d", zero)).to(device),
        -0.35,
        0.35,
    )
    target_net = torch.clamp(targets.get("portfolio_daily_allocation_net_utility_target", target_receiver).to(device), 0.0, 1.0)
    target_credit = torch.clamp(targets.get("portfolio_daily_allocation_credit_closure_target", target_source).to(device), 0.0, 1.0)
    target_efficiency = torch.clamp(
        targets.get("portfolio_daily_allocation_resource_efficiency_target", target_receiver).to(device),
        0.0,
        1.0,
    )
    uncertainty = torch.clamp(
        targets.get("portfolio_daily_allocation_uncertainty_pressure_target", zero).to(device),
        0.0,
        1.0,
    )
    tail = torch.clamp(
        targets.get("portfolio_daily_allocation_tail_risk_control_target", zero).to(device),
        0.0,
        1.0,
    )
    drawdown = torch.clamp(
        targets.get("portfolio_daily_allocation_drawdown_control_target", zero).to(device),
        0.0,
        1.0,
    )
    deploy_target = torch.clamp(
        targets.get("portfolio_daily_allocation_cash_deployment_target", target_net).to(device),
        0.0,
        1.0,
    )
    positive_forward = torch.clamp(
        torch.maximum(
            targets.get("portfolio_daily_source_positive_forward_penalty", zero).to(device),
            torch.clamp(source_forward / 0.08, 0.0, 1.0),
        ),
        0.0,
        1.0,
    )
    opportunity = torch.clamp(
        targets.get("portfolio_daily_source_opportunity_cost_penalty", zero).to(device),
        0.0,
        1.0,
    )
    hard_negative = torch.clamp(
        torch.maximum(
            targets.get("portfolio_daily_source_hard_negative_penalty", zero).to(device),
            targets.get("portfolio_daily_source_tail_false_sell_penalty", zero).to(device),
        ),
        0.0,
        1.0,
    )
    release_preference = torch.clamp(
        targets.get("portfolio_daily_source_release_preference", target_source).to(device),
        0.0,
        1.0,
    )
    spread_reward = torch.clamp(
        targets.get("portfolio_daily_receiver_source_spread_reward", zero).to(device),
        0.0,
        1.0,
    )
    transfer_regret = torch.clamp(
        targets.get("portfolio_daily_transfer_regret_target", target_credit).to(device),
        0.0,
        1.0,
    )
    risk_pressure = torch.clamp(torch.maximum(uncertainty, torch.maximum(tail, drawdown)), 0.0, 1.0)
    false_source_pressure = torch.clamp(
        torch.maximum(hard_negative, 0.60 * positive_forward + 0.40 * opportunity),
        0.0,
        1.0,
    )

    day_losses: list[torch.Tensor] = []
    for date_value in torch.unique(date_code):
        day_mask = date_code == date_value
        receiver_day = day_mask & (receiver_mask > 0.05)
        source_day = day_mask & (source_mask > 0.05) & (current_weight > 1.0e-8)
        if int(receiver_day.sum().detach().cpu().item()) == 0 and int(source_day.sum().detach().cpu().item()) == 0:
            continue

        day_risk = risk_pressure[day_mask].mean()
        day_cash_score = cash_score[day_mask].mean()
        day_target_cash = target_cash[day_mask].mean()
        day_credit = target_credit[day_mask].mean()
        day_utility = target_net[day_mask].mean()
        day_efficiency = target_efficiency[day_mask].mean()
        desired_deploy = torch.clamp(
            0.42 * day_credit + 0.34 * day_utility + 0.18 * day_efficiency + 0.12 * deploy_target[day_mask].mean() - 0.44 * day_risk,
            0.0,
            1.0,
        )
        current_cash_proxy = torch.clamp(1.0 - current_weight[day_mask].sum(), 0.0, 1.0)
        desired_release = torch.clamp(desired_deploy - current_cash_proxy + 0.08 + 0.22 * transfer_regret[day_mask].mean(), 0.0, 1.0)
        desired_flow = torch.clamp(0.46 * desired_deploy + 0.38 * desired_release + 0.16 * (1.0 - current_cash_proxy), 0.05, 1.0)

        receiver_value = torch.empty(0, device=device)
        receiver_pred_logits = torch.empty(0, device=device)
        receiver_oracle_logits = torch.empty(0, device=device)
        if int(receiver_day.sum().detach().cpu().item()) > 0:
            receiver_value = torch.clamp(
                0.54 * torch.clamp(receiver_forward[receiver_day] / 0.10, -1.0, 1.0)
                + 0.20 * target_receiver[receiver_day]
                + 0.16 * target_net[receiver_day]
                + 0.12 * spread_reward[receiver_day]
                - 0.24 * risk_pressure[receiver_day],
                -1.0,
                1.0,
            )
            receiver_pred_logits = 3.8 * (
                receiver_score[receiver_day]
                + 0.30 * net_utility[receiver_day]
                + 0.22 * credit_closure[receiver_day]
                + 0.18 * resource_efficiency[receiver_day]
                + 0.12 * final_objective[receiver_day]
                - 0.24 * risk_pressure[receiver_day]
            )
            receiver_oracle_logits = 3.8 * (
                receiver_value + target_receiver[receiver_day] + 0.14 * target_credit[receiver_day] - 0.16 * risk_pressure[receiver_day]
            )

        source_release_value = torch.empty(0, device=device)
        source_pred_logits = torch.empty(0, device=device)
        source_oracle_logits = torch.empty(0, device=device)
        source_false = torch.empty(0, device=device)
        if int(source_day.sum().detach().cpu().item()) > 0:
            source_keep_cost = torch.clamp(
                0.48 * torch.clamp(source_forward[source_day] / 0.10, -1.0, 1.0)
                + 0.24 * false_source_pressure[source_day]
                + 0.14 * opportunity[source_day]
                - 0.22 * release_preference[source_day]
                - 0.14 * target_credit[source_day],
                -1.0,
                1.0,
            )
            source_release_value = torch.clamp(-source_keep_cost + 0.24 * release_preference[source_day], -1.0, 1.0)
            source_false = false_source_pressure[source_day]
            source_pred_logits = 3.8 * (
                source_score[source_day]
                + 0.24 * credit_closure[source_day]
                + 0.18 * resource_efficiency[source_day]
                + 0.14 * net_utility[source_day]
                + 0.16 * release_preference[source_day]
                - 0.62 * false_source_pressure[source_day]
            )
            source_oracle_logits = 3.8 * (
                source_release_value + target_source[source_day] + 0.16 * target_credit[source_day] - 0.88 * false_source_pressure[source_day]
            )

        cash_source_pred_logit = (3.8 * (current_cash_proxy + 0.20 * day_cash_score + 0.16 * torch.relu(desired_deploy - desired_release) - 0.18 * day_risk)).view(1)
        cash_source_oracle_logit = (3.8 * (current_cash_proxy + 0.22 * torch.relu(desired_deploy - desired_release) - 0.18 * day_risk)).view(1)
        cash_sink_pred_logit = (3.8 * (day_cash_score + 0.52 * day_risk - 0.28 * desired_deploy)).view(1)
        cash_sink_oracle_logit = (3.8 * (day_target_cash + 0.58 * day_risk + 0.18 * torch.relu(1.0 - desired_deploy))).view(1)

        row_pred = torch.softmax(torch.cat([source_pred_logits, cash_source_pred_logit]), dim=0)
        row_oracle = torch.softmax(torch.cat([source_oracle_logits, cash_source_oracle_logit]), dim=0)
        col_pred = torch.softmax(torch.cat([receiver_pred_logits, cash_sink_pred_logit]), dim=0)
        col_oracle = torch.softmax(torch.cat([receiver_oracle_logits, cash_sink_oracle_logit]), dim=0)

        source_count = int(source_release_value.numel())
        receiver_count = int(receiver_value.numel())
        pair_value = torch.zeros((source_count + 1, receiver_count + 1), device=device)
        if source_count > 0 and receiver_count > 0:
            pair_value[:source_count, :receiver_count] = source_release_value.unsqueeze(1) + receiver_value.unsqueeze(0) - 0.08 * day_risk
        if receiver_count > 0:
            pair_value[source_count, :receiver_count] = receiver_value + 0.06 * day_utility - 0.05 * torch.relu(desired_release - current_cash_proxy)
        if source_count > 0:
            pair_value[:source_count, receiver_count] = source_release_value + 0.22 * day_risk - 0.18 * desired_deploy
        pair_value[source_count, receiver_count] = 0.22 * day_risk - 0.14 * desired_deploy + 0.08 * day_target_cash
        pair_value = torch.clamp(pair_value, -1.25, 1.25)

        pred_pair_logits = 4.0 * pair_value
        oracle_pair_logits = 4.0 * pair_value
        if source_count > 0:
            pred_pair_logits[:source_count, :] = pred_pair_logits[:source_count, :] + source_pred_logits.unsqueeze(1) * 0.16
            oracle_pair_logits[:source_count, :] = oracle_pair_logits[:source_count, :] + source_oracle_logits.unsqueeze(1) * 0.16
        if receiver_count > 0:
            pred_pair_logits[:, :receiver_count] = pred_pair_logits[:, :receiver_count] + receiver_pred_logits.unsqueeze(0) * 0.16
            oracle_pair_logits[:, :receiver_count] = oracle_pair_logits[:, :receiver_count] + receiver_oracle_logits.unsqueeze(0) * 0.16

        pred_plan = _sinkhorn_transport_plan(pred_pair_logits, row_pred, col_pred)
        oracle_plan = _sinkhorn_transport_plan(oracle_pair_logits, row_oracle, col_oracle)
        pred_utility = desired_flow * (pred_plan * pair_value).sum()
        oracle_utility = desired_flow * (oracle_plan * pair_value).sum()
        regret = torch.relu(oracle_utility - pred_utility + 0.015)
        marginal_loss = nn.functional.mse_loss(pred_plan.sum(dim=1), row_oracle) + nn.functional.mse_loss(pred_plan.sum(dim=0), col_oracle)

        false_source_loss = torch.tensor(0.0, device=device)
        source_shortfall = torch.tensor(0.0, device=device)
        if source_count > 0:
            real_source_mass = pred_plan[:source_count, :].sum(dim=1)
            false_source_loss = desired_flow * (real_source_mass * source_false).sum()
            source_shortfall = torch.relu(desired_release - real_source_mass.sum())
        receiver_shortfall = torch.tensor(0.0, device=device)
        if receiver_count > 0:
            receiver_shortfall = torch.relu(desired_deploy - pred_plan[:, :receiver_count].sum())
        cash_sink_mass = pred_plan[:, receiver_count].sum()
        cash_to_cash_mass = pred_plan[source_count, receiver_count]
        dead_cash_loss = torch.relu(desired_deploy - day_risk + 0.04) * (cash_sink_mass + cash_to_cash_mass)
        risk_cash_loss = torch.relu(day_risk - cash_sink_mass + 0.06) * torch.clamp(day_risk - desired_deploy + 0.08, 0.0, 1.0)
        day_losses.append(
            regret
            + 0.44 * marginal_loss
            + 0.38 * false_source_loss
            + 0.24 * dead_cash_loss
            + 0.22 * risk_cash_loss
            + 0.16 * (source_shortfall + receiver_shortfall)
        )

    if not day_losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(day_losses).mean()


def _portfolio_offline_conservative_support_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    required_outputs = {
        "portfolio_daily_unified_receiver_score",
        "portfolio_daily_unified_source_score",
        "portfolio_daily_unified_cash_score",
    }
    required_targets = {
        "current_weight",
        "portfolio_daily_receiver_candidate_mask",
        "portfolio_daily_source_candidate_mask",
    }
    device = next(iter(outputs.values())).device
    if not required_outputs.issubset(outputs) or not required_targets.issubset(targets):
        return torch.tensor(0.0, device=device)

    receiver_score = torch.clamp(outputs["portfolio_daily_unified_receiver_score"].to(device), 0.0, 1.0)
    source_score = torch.clamp(outputs["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    cash_score = torch.clamp(outputs["portfolio_daily_unified_cash_score"].to(device), 0.0, 1.0)
    zero = torch.zeros_like(receiver_score)
    final_objective = torch.clamp(outputs.get("portfolio_daily_allocation_final_objective", receiver_score).to(device), 0.0, 1.0)
    net_utility = torch.clamp(outputs.get("portfolio_daily_allocation_net_utility_target", receiver_score).to(device), 0.0, 1.0)
    credit_closure = torch.clamp(outputs.get("portfolio_daily_allocation_credit_closure_target", source_score).to(device), 0.0, 1.0)
    resource_efficiency = torch.clamp(
        outputs.get("portfolio_daily_allocation_resource_efficiency_target", receiver_score).to(device),
        0.0,
        1.0,
    )

    receiver_mask = torch.clamp(targets["portfolio_daily_receiver_candidate_mask"].to(device), 0.0, 1.0)
    source_mask = torch.clamp(targets["portfolio_daily_source_candidate_mask"].to(device), 0.0, 1.0)
    current_weight = torch.clamp(targets["current_weight"].to(device), 0.0, 1.0)
    receiver_support = receiver_mask
    if "portfolio_daily_receiver_executable_candidate" in targets:
        receiver_support = receiver_support * torch.clamp(
            targets["portfolio_daily_receiver_executable_candidate"].to(device),
            0.0,
            1.0,
        )
    held_support = (current_weight > 1.0e-8).to(device=device, dtype=source_score.dtype)
    source_support = source_mask * held_support
    if "portfolio_daily_source_executable_candidate" in targets:
        source_support = source_support * torch.clamp(
            targets["portfolio_daily_source_executable_candidate"].to(device),
            0.0,
            1.0,
        )
    receiver_support = torch.clamp(receiver_support, 0.0, 1.0)
    source_support = torch.clamp(source_support, 0.0, 1.0)

    target_receiver = torch.clamp(targets.get("portfolio_daily_unified_receiver_score", receiver_support).to(device), 0.0, 1.0)
    target_source = torch.clamp(targets.get("portfolio_daily_unified_source_score", source_support).to(device), 0.0, 1.0)
    target_cash = torch.clamp(targets.get("portfolio_daily_unified_cash_score", cash_score.detach()).to(device), 0.0, 1.0)
    target_net = torch.clamp(targets.get("portfolio_daily_allocation_net_utility_target", target_receiver).to(device), 0.0, 1.0)
    target_credit = torch.clamp(targets.get("portfolio_daily_allocation_credit_closure_target", target_source).to(device), 0.0, 1.0)
    target_efficiency = torch.clamp(
        targets.get("portfolio_daily_allocation_resource_efficiency_target", target_receiver).to(device),
        0.0,
        1.0,
    )
    receiver_forward = torch.clamp(
        targets.get("portfolio_daily_receiver_forward_excess_5d", targets.get("forward_excess_5d", zero)).to(device),
        -0.35,
        0.35,
    )
    source_forward = torch.clamp(
        targets.get("portfolio_daily_source_forward_excess_5d", targets.get("forward_excess_5d", zero)).to(device),
        -0.35,
        0.35,
    )
    hard_negative = torch.clamp(targets.get("portfolio_daily_source_hard_negative_penalty", zero).to(device), 0.0, 1.0)
    tail_false = torch.clamp(targets.get("portfolio_daily_source_tail_false_sell_penalty", zero).to(device), 0.0, 1.0)
    positive_forward = torch.clamp(
        torch.maximum(
            targets.get("portfolio_daily_source_positive_forward_penalty", zero).to(device),
            torch.clamp(source_forward / 0.08, 0.0, 1.0),
        ),
        0.0,
        1.0,
    )
    opportunity = torch.clamp(targets.get("portfolio_daily_source_opportunity_cost_penalty", zero).to(device), 0.0, 1.0)
    release_preference = torch.clamp(targets.get("portfolio_daily_source_release_preference", target_source).to(device), 0.0, 1.0)
    uncertainty = torch.clamp(
        targets.get("portfolio_daily_allocation_uncertainty_pressure_target", zero).to(device),
        0.0,
        1.0,
    )
    tail = torch.clamp(targets.get("portfolio_daily_allocation_tail_risk_control_target", zero).to(device), 0.0, 1.0)
    drawdown = torch.clamp(
        targets.get("portfolio_daily_allocation_drawdown_control_target", zero).to(device),
        0.0,
        1.0,
    )
    risk_pressure = torch.clamp(torch.maximum(uncertainty, torch.maximum(tail, drawdown)), 0.0, 1.0)
    false_source_pressure = torch.clamp(
        torch.maximum(torch.maximum(hard_negative, tail_false), torch.maximum(positive_forward, 0.55 * opportunity)),
        0.0,
        1.0,
    )

    receiver_unsupported = 1.0 - receiver_support
    source_unsupported = 1.0 - source_support
    receiver_support_loss = (
        receiver_unsupported
        * (
            torch.square(receiver_score)
            + 0.30 * torch.square(final_objective)
            + 0.22 * torch.square(net_utility)
            + 0.16 * torch.square(resource_efficiency)
        )
    ).mean()
    source_support_loss = (
        source_unsupported
        * (
            torch.square(source_score)
            + 0.28 * torch.square(credit_closure)
            + 0.14 * torch.square(final_objective)
        )
    ).mean()
    false_source_loss = (
        source_support
        * false_source_pressure
        * (
            torch.square(source_score)
            + 0.26 * torch.square(credit_closure)
            + 0.16 * torch.square(torch.relu(source_score - release_preference + 0.10))
        )
    ).mean()

    date_code = targets.get("date_code")
    if date_code is None:
        date_code_tensor = torch.arange(receiver_score.numel(), device=device, dtype=torch.long)
    else:
        date_code_tensor = date_code.to(device).long()
    day_losses: list[torch.Tensor] = []
    for date_value in torch.unique(date_code_tensor):
        day_mask = date_code_tensor == date_value
        day_receiver_support = receiver_support[day_mask]
        day_source_support = source_support[day_mask]
        day_receiver_quality = torch.clamp(
            0.40 * target_receiver[day_mask]
            + 0.26 * target_net[day_mask]
            + 0.18 * target_efficiency[day_mask]
            + 0.16 * torch.clamp(receiver_forward[day_mask] / 0.10, 0.0, 1.0),
            0.0,
            1.0,
        )
        day_source_quality = torch.clamp(
            0.48 * target_source[day_mask]
            + 0.30 * target_credit[day_mask]
            + 0.22 * release_preference[day_mask]
            - 0.50 * false_source_pressure[day_mask],
            0.0,
            1.0,
        )
        day_risk = risk_pressure[day_mask].mean()
        day_cash = cash_score[day_mask].mean()
        receiver_denominator = torch.clamp(day_receiver_support.sum(), min=1.0)
        source_denominator = torch.clamp(day_source_support.sum(), min=1.0)
        supported_receiver_value = (day_receiver_support * day_receiver_quality).sum() / receiver_denominator
        supported_source_value = (day_source_support * day_source_quality).sum() / source_denominator
        receiver_logsum = torch.logsumexp(3.5 * (receiver_score[day_mask] + 0.18 * final_objective[day_mask]), dim=0) / 3.5
        source_logsum = torch.logsumexp(3.5 * (source_score[day_mask] + 0.16 * credit_closure[day_mask]), dim=0) / 3.5
        receiver_cql_gap = torch.relu(receiver_logsum - supported_receiver_value - 0.12)
        source_cql_gap = torch.relu(source_logsum - supported_source_value - 0.10)
        deploy_pressure = torch.clamp(
            (day_receiver_support * day_receiver_quality).mean()
            + 0.35 * (day_source_support * day_source_quality).mean()
            - 0.55 * day_risk,
            0.0,
            1.0,
        )
        risk_cash_loss = torch.square(torch.relu(0.52 + 0.22 * day_risk - day_cash)) * day_risk
        dead_cash_loss = torch.square(day_cash) * torch.relu(deploy_pressure - day_risk - 0.04)
        target_cash_anchor = torch.square(day_cash - target_cash[day_mask].mean()) * torch.clamp(day_risk + deploy_pressure, 0.0, 1.0)
        day_losses.append(
            0.22 * torch.square(receiver_cql_gap)
            + 0.22 * torch.square(source_cql_gap)
            + 0.10 * risk_cash_loss
            + 0.08 * dead_cash_loss
            + 0.06 * target_cash_anchor
        )

    day_loss = torch.stack(day_losses).mean() if day_losses else torch.tensor(0.0, device=device)
    return (
        0.32 * receiver_support_loss
        + 0.30 * source_support_loss
        + 0.24 * false_source_loss
        + 0.14 * day_loss
    )


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / torch.clamp(weights.sum(), min=1.0e-6)


def _weighted_variance(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    mean = _weighted_mean(values, weights)
    return ((values - mean) ** 2 * weights).sum() / torch.clamp(weights.sum(), min=1.0e-6)


def _cvxpy_convex_layer_available() -> bool:
    return cp is not None and CvxpyLayer is not None and _CVXPY_LAYER_IMPORT_ERROR is None


def _cvxpy_convex_layer_status() -> dict[str, Any]:
    if not _cvxpy_convex_layer_available():
        return {
            "available": False,
            "error": str(_CVXPY_LAYER_IMPORT_ERROR or "cvxpy/cvxpylayers unavailable"),
            "installed_solvers": [],
        }
    try:
        installed_solvers = list(cp.installed_solvers()) if cp is not None else []
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"available": False, "error": str(exc), "installed_solvers": []}
    return {
        "available": True,
        "error": "",
        "installed_solvers": installed_solvers,
        "slot_count": int(CVXPY_CONVEX_ALLOCATION_SLOT_COUNT),
        "max_days_per_batch": int(CVXPY_CONVEX_ALLOCATION_MAX_DAYS_PER_BATCH),
        "solver_args": dict(CVXPY_CONVEX_ALLOCATION_SOLVER_ARGS),
    }


def _get_portfolio_cvxpy_allocation_layer(slot_count: int = CVXPY_CONVEX_ALLOCATION_SLOT_COUNT) -> Any:
    if not _cvxpy_convex_layer_available():
        raise RuntimeError(
            "r47 true convex solver layer requires cvxpy, cvxpylayers and diffcp in the yolos environment."
        )
    resolved_slot_count = int(slot_count)
    if resolved_slot_count in _CVXPY_ALLOCATION_LAYER_CACHE:
        return _CVXPY_ALLOCATION_LAYER_CACHE[resolved_slot_count]

    weight = cp.Variable(resolved_slot_count)
    gross_slack = cp.Variable(nonneg=True)
    turnover_slack = cp.Variable(nonneg=True)
    utility = cp.Parameter(resolved_slot_count)
    current_weight_param = cp.Parameter(resolved_slot_count)
    upper_bound = cp.Parameter(resolved_slot_count, nonneg=True)
    gross_target = cp.Parameter(nonneg=True)
    turnover_limit = cp.Parameter(nonneg=True)
    constraints = [
        weight >= 0.0,
        weight <= upper_bound,
        cp.sum(weight) <= 1.0,
        gross_slack >= cp.sum(weight) - gross_target,
        gross_slack >= gross_target - cp.sum(weight),
        turnover_slack >= cp.norm1(weight - current_weight_param),
        turnover_slack <= turnover_limit,
    ]
    objective = cp.Minimize(
        0.50 * cp.sum_squares(weight - current_weight_param)
        - utility @ weight
        + 0.20 * gross_slack
        + 0.03 * turnover_slack
    )
    problem = cp.Problem(objective, constraints)
    if not problem.is_dpp():
        raise RuntimeError("r47 cvxpy allocation problem must stay DPP-compliant for cvxpylayers.")
    layer = CvxpyLayer(
        problem,
        parameters=[utility, current_weight_param, upper_bound, gross_target, turnover_limit],
        variables=[weight],
    )
    _CVXPY_ALLOCATION_LAYER_CACHE[resolved_slot_count] = layer
    return layer


def _pad_fixed_slot_vector(
    values: torch.Tensor,
    slot_count: int,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    resolved_dtype = dtype or values.dtype
    padded = torch.zeros(int(slot_count), device=values.device, dtype=resolved_dtype)
    item_count = min(int(slot_count), int(values.numel()))
    if item_count > 0:
        padded[:item_count] = values[:item_count].to(dtype=resolved_dtype)
    return padded


def _portfolio_differentiable_convex_allocation_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    return_terms: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    required_outputs = {
        "portfolio_daily_unified_receiver_score",
        "portfolio_daily_unified_source_score",
        "portfolio_daily_unified_cash_score",
        "portfolio_daily_allocation_final_objective",
        "portfolio_daily_allocation_net_utility_target",
        "portfolio_daily_allocation_credit_closure_target",
        "portfolio_daily_allocation_resource_efficiency_target",
    }
    required_targets = {
        "date_code",
        "current_weight",
        "portfolio_daily_receiver_candidate_mask",
        "portfolio_daily_source_candidate_mask",
    }
    device = next(iter(outputs.values())).device
    term_names = (
        "regret",
        "constraint_residual",
        "unsupported_mass",
        "false_source_mass",
        "cash_timing_loss",
        "source_receiver_shortfall",
        "kkt_stationarity",
        "kkt_complementarity",
        "position_residual",
        "gross_exposure_residual",
        "candidate_breadth_loss",
        "behavior_support_loss",
        "conservative_ope_loss",
        "path_risk_loss",
        "total",
    )

    def _zero_result() -> torch.Tensor | dict[str, torch.Tensor]:
        zero_value = torch.tensor(0.0, device=device)
        if return_terms:
            return {name: zero_value for name in term_names}
        return zero_value

    if not required_outputs.issubset(outputs) or not required_targets.issubset(targets):
        return _zero_result()

    receiver_score = torch.clamp(outputs["portfolio_daily_unified_receiver_score"].to(device), 0.0, 1.0)
    source_score = torch.clamp(outputs["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    cash_score = torch.clamp(outputs["portfolio_daily_unified_cash_score"].to(device), 0.0, 1.0)
    final_objective = torch.clamp(outputs["portfolio_daily_allocation_final_objective"].to(device), 0.0, 1.0)
    net_utility = torch.clamp(outputs["portfolio_daily_allocation_net_utility_target"].to(device), 0.0, 1.0)
    credit_closure = torch.clamp(outputs["portfolio_daily_allocation_credit_closure_target"].to(device), 0.0, 1.0)
    resource_efficiency = torch.clamp(outputs["portfolio_daily_allocation_resource_efficiency_target"].to(device), 0.0, 1.0)

    zero = torch.zeros_like(receiver_score)
    current_weight = torch.clamp(targets["current_weight"].to(device), 0.0, 1.0)
    receiver_mask = torch.clamp(targets["portfolio_daily_receiver_candidate_mask"].to(device), 0.0, 1.0)
    source_mask = torch.clamp(targets["portfolio_daily_source_candidate_mask"].to(device), 0.0, 1.0)
    receiver_support = receiver_mask
    if "portfolio_daily_receiver_executable_candidate" in targets:
        receiver_support = receiver_support * torch.clamp(targets["portfolio_daily_receiver_executable_candidate"].to(device), 0.0, 1.0)
    held_support = (current_weight > 1.0e-8).to(device=device, dtype=source_score.dtype)
    source_support = source_mask * held_support
    if "portfolio_daily_source_executable_candidate" in targets:
        source_support = source_support * torch.clamp(targets["portfolio_daily_source_executable_candidate"].to(device), 0.0, 1.0)
    receiver_support = torch.clamp(receiver_support, 0.0, 1.0)
    source_support = torch.clamp(source_support, 0.0, 1.0)

    target_receiver = torch.clamp(targets.get("portfolio_daily_unified_receiver_score", receiver_support).to(device), 0.0, 1.0)
    target_source = torch.clamp(targets.get("portfolio_daily_unified_source_score", source_support).to(device), 0.0, 1.0)
    target_cash = torch.clamp(targets.get("portfolio_daily_unified_cash_score", cash_score.detach()).to(device), 0.0, 1.0)
    gross_exposure_target = torch.clamp(targets.get("gross_exposure_target", torch.full_like(receiver_score, 0.60)).to(device), 0.0, 1.0)
    candidate_budget_target = torch.clamp(targets.get("candidate_budget", torch.full_like(receiver_score, 4.0)).to(device), 1.0, 30.0)
    turnover_budget_target = torch.clamp(targets.get("turnover_budget", torch.full_like(receiver_score, 0.42)).to(device), 0.04, 1.0)
    max_position_weight_target = torch.clamp(
        targets.get("max_position_weight_target", torch.full_like(receiver_score, 0.20)).to(device),
        0.04,
        0.40,
    )
    cash_timing_signal = torch.clamp(targets.get("budget_cash_timing_signal_target", target_cash).to(device), 0.0, 1.0)
    target_net = torch.clamp(targets.get("portfolio_daily_allocation_net_utility_target", target_receiver).to(device), 0.0, 1.0)
    target_credit = torch.clamp(targets.get("portfolio_daily_allocation_credit_closure_target", target_source).to(device), 0.0, 1.0)
    target_efficiency = torch.clamp(
        targets.get("portfolio_daily_allocation_resource_efficiency_target", target_receiver).to(device),
        0.0,
        1.0,
    )
    deploy_target = torch.clamp(
        targets.get("portfolio_daily_allocation_cash_deployment_target", target_net).to(device),
        0.0,
        1.0,
    )
    receiver_forward = torch.clamp(
        targets.get("portfolio_daily_receiver_forward_excess_5d", targets.get("forward_excess_5d", zero)).to(device),
        -0.35,
        0.35,
    )
    source_forward = torch.clamp(
        targets.get("portfolio_daily_source_forward_excess_5d", targets.get("forward_excess_5d", zero)).to(device),
        -0.35,
        0.35,
    )
    uncertainty = torch.clamp(targets.get("portfolio_daily_allocation_uncertainty_pressure_target", zero).to(device), 0.0, 1.0)
    tail = torch.clamp(targets.get("portfolio_daily_allocation_tail_risk_control_target", zero).to(device), 0.0, 1.0)
    drawdown = torch.clamp(targets.get("portfolio_daily_allocation_drawdown_control_target", zero).to(device), 0.0, 1.0)
    hard_negative = torch.clamp(
        torch.maximum(
            targets.get("portfolio_daily_source_hard_negative_penalty", zero).to(device),
            targets.get("portfolio_daily_source_tail_false_sell_penalty", zero).to(device),
        ),
        0.0,
        1.0,
    )
    positive_forward = torch.clamp(
        torch.maximum(
            targets.get("portfolio_daily_source_positive_forward_penalty", zero).to(device),
            torch.clamp(source_forward / 0.08, 0.0, 1.0),
        ),
        0.0,
        1.0,
    )
    opportunity = torch.clamp(targets.get("portfolio_daily_source_opportunity_cost_penalty", zero).to(device), 0.0, 1.0)
    release_preference = torch.clamp(targets.get("portfolio_daily_source_release_preference", target_source).to(device), 0.0, 1.0)
    spread_reward = torch.clamp(targets.get("portfolio_daily_receiver_source_spread_reward", zero).to(device), 0.0, 1.0)
    forward_benchmark_1d = torch.clamp(targets.get("forward_benchmark_return_1d", zero).to(device), -0.25, 0.25)
    risk_pressure = torch.clamp(torch.maximum(uncertainty, torch.maximum(tail, drawdown)), 0.0, 1.0)
    false_source_pressure = torch.clamp(
        torch.maximum(torch.maximum(hard_negative, positive_forward), 0.55 * opportunity + 0.45 * hard_negative),
        0.0,
        1.0,
    )
    date_code = targets["date_code"].to(device).long()

    day_losses: list[torch.Tensor] = []
    day_return_proxies: list[torch.Tensor] = []
    term_values: dict[str, list[torch.Tensor]] = {name: [] for name in term_names if name not in {"path_risk_loss", "total"}}
    for date_value in torch.unique(date_code):
        day_mask = date_code == date_value
        if int(day_mask.sum().detach().cpu().item()) <= 0:
            continue

        current_day = current_weight[day_mask]
        receiver_support_day = receiver_support[day_mask]
        source_support_day = source_support[day_mask]
        day_position_cap = torch.clamp(max_position_weight_target[day_mask].mean(), 0.04, 0.40)
        day_turnover_limit = torch.clamp(turnover_budget_target[day_mask].mean(), 0.04, 1.0)
        day_gross_target = torch.clamp(gross_exposure_target[day_mask].mean(), 0.0, 1.0)
        day_candidate_budget = torch.clamp(candidate_budget_target[day_mask].mean(), 1.0, 30.0)
        day_cash_timing_signal = cash_timing_signal[day_mask].mean()
        day_risk = risk_pressure[day_mask].mean()
        day_cash_score = cash_score[day_mask].mean()
        day_target_cash = target_cash[day_mask].mean()
        day_cost_rate = torch.clamp(
            0.0015
            + 0.0020 * day_risk
            + 0.0004 * day_turnover_limit
            + 0.0004 * torch.relu(day_candidate_budget - 8.0) / 8.0,
            0.0015,
            0.0060,
        )
        current_cash = torch.clamp(1.0 - current_day.sum(), 0.0, 1.0)
        current_gross = torch.clamp(current_day.sum(), 0.0, 1.0)
        deploy_pressure = torch.clamp(
            0.30 * target_net[day_mask].mean()
            + 0.24 * target_credit[day_mask].mean()
            + 0.18 * target_efficiency[day_mask].mean()
            + 0.16 * deploy_target[day_mask].mean()
            + 0.12 * final_objective[day_mask].mean()
            - 0.44 * day_risk,
            0.0,
            1.0,
        )
        cash_reserve = torch.clamp(
            0.05
            + 0.40 * day_risk
            + 0.18 * day_target_cash
            + 0.16 * day_cash_timing_signal
            - 0.24 * deploy_pressure,
            0.02,
            0.76,
        )
        gross_gap = torch.relu(day_gross_target - current_gross)
        gross_excess = torch.relu(current_gross - day_gross_target)
        desired_deploy = torch.clamp(deploy_pressure + 0.40 * gross_gap - torch.relu(cash_reserve - current_cash), 0.0, 1.0)
        desired_release = torch.clamp(
            desired_deploy
            - current_cash
            + cash_reserve
            + 0.08 * target_credit[day_mask].mean()
            + 0.40 * gross_excess,
            0.0,
            day_turnover_limit,
        )

        receiver_value = torch.clamp(
            0.38 * target_receiver[day_mask]
            + 0.24 * target_net[day_mask]
            + 0.16 * target_efficiency[day_mask]
            + 0.14 * spread_reward[day_mask]
            + 0.16 * torch.clamp(receiver_forward[day_mask] / 0.10, -1.0, 1.0)
            - 0.22 * risk_pressure[day_mask],
            -1.0,
            1.0,
        )
        source_release_value = torch.clamp(
            0.36 * target_source[day_mask]
            + 0.26 * target_credit[day_mask]
            + 0.24 * release_preference[day_mask]
            - 0.26 * torch.clamp(source_forward[day_mask] / 0.10, -1.0, 1.0)
            - 0.54 * false_source_pressure[day_mask],
            -1.0,
            1.0,
        )
        receiver_logits = (
            3.6
            * (
                receiver_score[day_mask]
                + 0.24 * net_utility[day_mask]
                + 0.16 * credit_closure[day_mask]
                + 0.12 * final_objective[day_mask]
                + 0.10 * resource_efficiency[day_mask]
                - 0.22 * risk_pressure[day_mask]
            )
            + torch.log(torch.clamp(receiver_support_day + 0.015, min=1.0e-5))
        )
        source_logits = (
            3.6
            * (
                source_score[day_mask]
                + 0.22 * credit_closure[day_mask]
                + 0.16 * release_preference[day_mask]
                + 0.10 * resource_efficiency[day_mask]
                - 0.60 * false_source_pressure[day_mask]
            )
            + torch.log(torch.clamp(source_support_day + 0.015, min=1.0e-5))
        )
        receiver_probs = torch.softmax(receiver_logits, dim=0)
        source_probs = torch.softmax(source_logits, dim=0)
        source_capacity = current_day * torch.clamp(source_support_day + 0.08 * (1.0 - false_source_pressure[day_mask]), 0.0, 1.0)
        receiver_capacity = torch.clamp(day_position_cap - current_day, min=0.0)

        sell_total = torch.minimum(torch.minimum(source_capacity.sum(), day_turnover_limit * 0.5), desired_release)
        raw_sells = sell_total * source_probs
        sells = torch.minimum(current_day, raw_sells)
        cash_from_sales = sells.sum() * (1.0 - day_cost_rate)
        buy_budget = torch.minimum(
            torch.minimum(receiver_capacity.sum(), day_turnover_limit - sells.sum()),
            torch.clamp(current_cash + cash_from_sales - cash_reserve, 0.0, 1.0),
        )
        buy_budget = torch.minimum(buy_budget, desired_deploy)
        raw_buys = buy_budget * receiver_probs
        buys = torch.minimum(receiver_capacity, raw_buys)
        target_weight = torch.clamp(current_day - sells + buys, 0.0, 1.0)
        turnover = sells.sum() + buys.sum()
        cash_after = torch.clamp(1.0 - target_weight.sum() - day_cost_rate * turnover, 0.0, 1.0)

        receiver_oracle_probs = torch.softmax(
            4.0 * (receiver_value + target_receiver[day_mask] + 0.12 * target_net[day_mask] - 0.18 * risk_pressure[day_mask])
            + torch.log(torch.clamp(receiver_support_day + 0.015, min=1.0e-5)),
            dim=0,
        )
        source_oracle_probs = torch.softmax(
            4.0 * (source_release_value + target_source[day_mask] + 0.12 * target_credit[day_mask] - 0.78 * false_source_pressure[day_mask])
            + torch.log(torch.clamp(source_support_day + 0.015, min=1.0e-5)),
            dim=0,
        )
        oracle_buys = torch.minimum(receiver_capacity, buy_budget * receiver_oracle_probs)
        oracle_sells = torch.minimum(current_day, sell_total * source_oracle_probs)
        pred_utility = (buys * receiver_value).sum() + (sells * source_release_value).sum() + cash_after * (0.28 * day_risk - 0.18 * deploy_pressure)
        oracle_cash = torch.clamp(
            1.0
            - torch.clamp(current_day - oracle_sells + oracle_buys, 0.0, 1.0).sum()
            - day_cost_rate * (oracle_sells.sum() + oracle_buys.sum()),
            0.0,
            1.0,
        )
        oracle_utility = (
            (oracle_buys * receiver_value).sum()
            + (oracle_sells * source_release_value).sum()
            + oracle_cash * (0.28 * day_risk - 0.18 * deploy_pressure)
        )
        regret = torch.relu(oracle_utility - pred_utility + 0.012)

        unsupported_receiver_mass = (buys * (1.0 - receiver_support_day)).sum()
        unsupported_source_mass = (sells * (1.0 - source_support_day)).sum()
        false_source_mass = (sells * false_source_pressure[day_mask]).sum()
        budget_residual = torch.square(torch.relu(cash_reserve - cash_after)) + torch.square(torch.relu(turnover - day_turnover_limit))
        position_residual = torch.square(torch.relu(target_weight - day_position_cap)).mean()
        gross_exposure_residual = (
            torch.square(torch.relu(target_weight.sum() - day_gross_target - 0.04))
            + torch.square(torch.relu(day_gross_target - target_weight.sum())) * torch.clamp(deploy_pressure - day_risk, 0.0, 1.0)
        )
        cash_timing_loss = (
            torch.relu(deploy_pressure - day_risk - 0.05) * torch.square(cash_after)
            + torch.relu(day_risk - deploy_pressure + 0.05) * torch.square(torch.relu(cash_reserve - cash_after))
            + 0.35 * torch.square(torch.relu(day_cash_timing_signal - cash_after)) * torch.clamp(day_risk + day_cash_timing_signal, 0.0, 1.0)
        )
        source_shortfall = torch.square(torch.relu(torch.minimum(source_capacity.sum(), desired_release) - sells.sum()))
        receiver_shortfall = torch.square(torch.relu(torch.minimum(receiver_capacity.sum(), desired_deploy) - buys.sum()))
        buy_activity = torch.clamp(buys.sum() * 20.0, 0.0, 1.0)
        sell_activity = torch.clamp(sells.sum() * 20.0, 0.0, 1.0)
        stationarity = (
            buy_activity * _weighted_variance(receiver_value, buys + 1.0e-6)
            + sell_activity * _weighted_variance(source_release_value, sells + 1.0e-6)
        )
        complementarity = (
            buy_activity * (buys * torch.relu(_weighted_mean(receiver_value, buys + 1.0e-6) - receiver_value + 0.03)).sum()
            + sell_activity * (sells * torch.relu(_weighted_mean(source_release_value, sells + 1.0e-6) - source_release_value + 0.03)).sum()
        )
        effective_receiver_count = torch.square(buys.sum()) / torch.clamp(torch.square(buys).sum(), min=1.0e-6)
        expected_receiver_count = torch.minimum(day_candidate_budget, receiver_support_day.sum())
        candidate_breadth_loss = (
            torch.square(torch.relu(expected_receiver_count - effective_receiver_count))
            * torch.clamp(desired_deploy, 0.0, 1.0)
            / torch.clamp(expected_receiver_count, min=1.0)
        )
        receiver_behavior_support = torch.clamp(
            0.72 * receiver_support_day + 0.18 * receiver_mask[day_mask] + 0.10 * target_receiver[day_mask],
            0.0,
            1.0,
        )
        source_behavior_support = torch.clamp(
            0.68 * source_support_day
            + 0.18 * source_mask[day_mask] * (current_day > 1.0e-8).to(device=device, dtype=source_score.dtype)
            + 0.14 * release_preference[day_mask] * (1.0 - false_source_pressure[day_mask]),
            0.0,
            1.0,
        )
        behavior_support_loss = (
            (buys * torch.square(1.0 - receiver_behavior_support)).sum()
            + (sells * torch.square(1.0 - source_behavior_support)).sum()
        )
        security_forward_proxy = torch.where(current_day > 1.0e-8, source_forward[day_mask], receiver_forward[day_mask])
        excess_forward_proxy = torch.clamp(security_forward_proxy - forward_benchmark_1d[day_mask], -0.35, 0.35)
        behavior_return_proxy = (current_day * excess_forward_proxy).sum()
        policy_return_proxy = (target_weight * excess_forward_proxy).sum() - day_cost_rate * turnover
        conservative_ope_loss = torch.relu(behavior_return_proxy - policy_return_proxy + 0.006) * torch.clamp(turnover, 0.0, 1.0)
        day_return_proxy = torch.clamp(policy_return_proxy + cash_after * torch.clamp(forward_benchmark_1d[day_mask].mean(), -0.08, 0.08), -0.50, 0.50)
        day_return_proxies.append(day_return_proxy)

        constraint_residual = budget_residual + 0.18 * gross_exposure_residual
        shortfall = source_shortfall + receiver_shortfall
        day_component_values = {
            "regret": regret,
            "constraint_residual": constraint_residual,
            "unsupported_mass": unsupported_receiver_mass + unsupported_source_mass,
            "false_source_mass": false_source_mass,
            "cash_timing_loss": cash_timing_loss,
            "source_receiver_shortfall": shortfall,
            "kkt_stationarity": stationarity,
            "kkt_complementarity": complementarity,
            "position_residual": position_residual,
            "gross_exposure_residual": gross_exposure_residual,
            "candidate_breadth_loss": candidate_breadth_loss,
            "behavior_support_loss": behavior_support_loss,
            "conservative_ope_loss": conservative_ope_loss,
        }
        for name, value in day_component_values.items():
            term_values[name].append(value)
        day_losses.append(
            0.34 * regret
            + 0.27 * constraint_residual
            + 0.46 * (unsupported_receiver_mass + unsupported_source_mass)
            + 0.30 * false_source_mass
            + 0.24 * cash_timing_loss
            + 0.08 * shortfall
            + 0.06 * stationarity
            + 0.03 * complementarity
            + 0.03 * position_residual
            + 0.04 * gross_exposure_residual
            + 0.06 * candidate_breadth_loss
            + 0.18 * behavior_support_loss
            + 0.16 * conservative_ope_loss
        )

    if not day_losses:
        return _zero_result()
    base_total = torch.stack(day_losses).mean()
    path_risk_loss = torch.tensor(0.0, device=device)
    if day_return_proxies:
        return_path = torch.stack(day_return_proxies)
        safe_returns = torch.clamp(return_path, -0.50, 0.50)
        equity_curve = torch.cumprod(1.0 + safe_returns, dim=0)
        running_peak = torch.cummax(equity_curve, dim=0).values
        drawdown_path = equity_curve / torch.clamp(running_peak, min=1.0e-6) - 1.0
        tail_count = max(1, int(math.ceil(float(len(day_return_proxies)) * 0.20)))
        tail_losses = torch.topk(-safe_returns, k=tail_count).values
        cvar_loss = torch.square(torch.relu(tail_losses.mean() - 0.02))
        drawdown_loss = torch.square(torch.relu(-drawdown_path.min() - 0.12))
        eta = 8.0
        normalized_oce = (torch.logsumexp(-eta * safe_returns, dim=0) - math.log(float(len(day_return_proxies)))) / eta
        oce_loss = torch.square(torch.relu(normalized_oce + 0.01))
        path_risk_loss = 0.45 * cvar_loss + 0.35 * drawdown_loss + 0.20 * oce_loss
    total = base_total + 0.18 * path_risk_loss
    if return_terms:
        terms: dict[str, torch.Tensor] = {
            name: (torch.stack(values).mean() if values else torch.tensor(0.0, device=device))
            for name, values in term_values.items()
        }
        terms["path_risk_loss"] = path_risk_loss
        terms["total"] = total
        return terms
    return total


def _portfolio_cvxpy_convex_allocation_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    slot_count: int = CVXPY_CONVEX_ALLOCATION_SLOT_COUNT,
    max_days: int = CVXPY_CONVEX_ALLOCATION_MAX_DAYS_PER_BATCH,
    full_universe_mode: bool = False,
    conservative_ope_mode: bool = False,
    return_terms: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    required_outputs = {
        "portfolio_daily_unified_receiver_score",
        "portfolio_daily_unified_source_score",
        "portfolio_daily_unified_cash_score",
        "portfolio_daily_allocation_final_objective",
        "portfolio_daily_allocation_net_utility_target",
        "portfolio_daily_allocation_credit_closure_target",
        "portfolio_daily_allocation_resource_efficiency_target",
    }
    required_targets = {
        "date_code",
        "current_weight",
        "portfolio_daily_receiver_candidate_mask",
        "portfolio_daily_source_candidate_mask",
    }
    device = next(iter(outputs.values())).device
    term_names = (
        "solver_regret",
        "solution_tracking_loss",
        "gross_residual",
        "turnover_residual",
        "position_residual",
        "unsupported_mass",
        "false_source_mass",
        "behavior_support_loss",
        "cash_timing_loss",
        "path_risk_loss",
        "candidate_coverage_loss",
        "universe_expansion_loss",
        "liquidity_impact_loss",
        "concentration_risk_loss",
        "ope_lower_bound_loss",
        "propensity_support_loss",
        "doubly_robust_gap_loss",
        "solver_success_rate",
        "fallback_surrogate_loss",
        "total",
    )

    def _zero_result() -> torch.Tensor | dict[str, torch.Tensor]:
        zero_value = torch.tensor(0.0, device=device)
        if return_terms:
            return {name: zero_value for name in term_names}
        return zero_value

    if not required_outputs.issubset(outputs) or not required_targets.issubset(targets):
        return _zero_result()
    if not _cvxpy_convex_layer_available():
        return _zero_result()

    layer = _get_portfolio_cvxpy_allocation_layer(slot_count=int(slot_count))
    receiver_score = torch.clamp(outputs["portfolio_daily_unified_receiver_score"].to(device), 0.0, 1.0)
    source_score = torch.clamp(outputs["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    cash_score = torch.clamp(outputs["portfolio_daily_unified_cash_score"].to(device), 0.0, 1.0)
    final_objective = torch.clamp(outputs["portfolio_daily_allocation_final_objective"].to(device), 0.0, 1.0)
    net_utility = torch.clamp(outputs["portfolio_daily_allocation_net_utility_target"].to(device), 0.0, 1.0)
    credit_closure = torch.clamp(outputs["portfolio_daily_allocation_credit_closure_target"].to(device), 0.0, 1.0)
    resource_efficiency = torch.clamp(outputs["portfolio_daily_allocation_resource_efficiency_target"].to(device), 0.0, 1.0)

    zero = torch.zeros_like(receiver_score)
    current_weight = torch.clamp(targets["current_weight"].to(device), 0.0, 1.0)
    receiver_mask = torch.clamp(targets["portfolio_daily_receiver_candidate_mask"].to(device), 0.0, 1.0)
    source_mask = torch.clamp(targets["portfolio_daily_source_candidate_mask"].to(device), 0.0, 1.0)
    held_support = (current_weight > 1.0e-8).to(device=device, dtype=receiver_score.dtype)
    receiver_support = receiver_mask
    if "portfolio_daily_receiver_executable_candidate" in targets:
        receiver_support = receiver_support * torch.clamp(
            targets["portfolio_daily_receiver_executable_candidate"].to(device),
            0.0,
            1.0,
        )
    source_support = source_mask * held_support
    if "portfolio_daily_source_executable_candidate" in targets:
        source_support = source_support * torch.clamp(
            targets["portfolio_daily_source_executable_candidate"].to(device),
            0.0,
            1.0,
        )
    receiver_support = torch.clamp(receiver_support, 0.0, 1.0)
    source_support = torch.clamp(source_support, 0.0, 1.0)

    target_receiver = torch.clamp(targets.get("portfolio_daily_unified_receiver_score", receiver_support).to(device), 0.0, 1.0)
    target_source = torch.clamp(targets.get("portfolio_daily_unified_source_score", source_support).to(device), 0.0, 1.0)
    target_cash = torch.clamp(targets.get("portfolio_daily_unified_cash_score", cash_score.detach()).to(device), 0.0, 1.0)
    target_net = torch.clamp(targets.get("portfolio_daily_allocation_net_utility_target", target_receiver).to(device), 0.0, 1.0)
    target_credit = torch.clamp(targets.get("portfolio_daily_allocation_credit_closure_target", target_source).to(device), 0.0, 1.0)
    target_efficiency = torch.clamp(
        targets.get("portfolio_daily_allocation_resource_efficiency_target", target_receiver).to(device),
        0.0,
        1.0,
    )
    deploy_target = torch.clamp(
        targets.get("portfolio_daily_allocation_cash_deployment_target", target_net).to(device),
        0.0,
        1.0,
    )
    gross_exposure_target = torch.clamp(targets.get("gross_exposure_target", torch.full_like(receiver_score, 0.60)).to(device), 0.0, 1.0)
    turnover_budget_target = torch.clamp(targets.get("turnover_budget", torch.full_like(receiver_score, 0.36)).to(device), 0.04, 1.0)
    max_position_weight_target = torch.clamp(
        targets.get("max_position_weight_target", torch.full_like(receiver_score, 0.20)).to(device),
        0.04,
        0.40,
    )
    cash_timing_signal = torch.clamp(targets.get("budget_cash_timing_signal_target", target_cash).to(device), 0.0, 1.0)
    receiver_forward = torch.clamp(
        targets.get("portfolio_daily_receiver_forward_excess_5d", targets.get("forward_excess_5d", zero)).to(device),
        -0.35,
        0.35,
    )
    source_forward = torch.clamp(
        targets.get("portfolio_daily_source_forward_excess_5d", targets.get("forward_excess_5d", zero)).to(device),
        -0.35,
        0.35,
    )
    forward_benchmark_1d = torch.clamp(targets.get("forward_benchmark_return_1d", zero).to(device), -0.25, 0.25)
    forward_benchmark_3d = torch.clamp(targets.get("forward_benchmark_return_3d", forward_benchmark_1d).to(device), -0.35, 0.35)
    uncertainty = torch.clamp(targets.get("portfolio_daily_allocation_uncertainty_pressure_target", zero).to(device), 0.0, 1.0)
    tail = torch.clamp(targets.get("portfolio_daily_allocation_tail_risk_control_target", zero).to(device), 0.0, 1.0)
    drawdown = torch.clamp(targets.get("portfolio_daily_allocation_drawdown_control_target", zero).to(device), 0.0, 1.0)
    risk_pressure = torch.clamp(torch.maximum(uncertainty, torch.maximum(tail, drawdown)), 0.0, 1.0)
    release_preference = torch.clamp(targets.get("portfolio_daily_source_release_preference", target_source).to(device), 0.0, 1.0)
    spread_reward = torch.clamp(targets.get("portfolio_daily_receiver_source_spread_reward", zero).to(device), 0.0, 1.0)
    hard_negative = torch.clamp(
        torch.maximum(
            targets.get("portfolio_daily_source_hard_negative_penalty", zero).to(device),
            targets.get("portfolio_daily_source_tail_false_sell_penalty", zero).to(device),
        ),
        0.0,
        1.0,
    )
    positive_forward = torch.clamp(
        torch.maximum(
            targets.get("portfolio_daily_source_positive_forward_penalty", zero).to(device),
            torch.clamp(source_forward / 0.08, 0.0, 1.0),
        ),
        0.0,
        1.0,
    )
    opportunity = torch.clamp(targets.get("portfolio_daily_source_opportunity_cost_penalty", zero).to(device), 0.0, 1.0)
    false_source_pressure = torch.clamp(
        torch.maximum(torch.maximum(hard_negative, positive_forward), 0.55 * opportunity + 0.45 * hard_negative),
        0.0,
        1.0,
    )
    liquidity_support = torch.clamp(targets.get("portfolio_daily_liquidity_support", torch.ones_like(receiver_score)).to(device), 0.0, 1.0)
    impact_cost = torch.clamp(
        targets.get("portfolio_daily_impact_cost", 0.0020 + 0.0060 * (1.0 - liquidity_support)).to(device),
        0.0005,
        0.0400,
    )
    factor_concentration = torch.clamp(
        targets.get("portfolio_daily_factor_concentration_proxy", torch.full_like(receiver_score, 0.10)).to(device),
        0.0,
        1.0,
    )
    behavior_propensity = torch.clamp(
        targets.get(
            "portfolio_daily_behavior_propensity",
            0.08 + 0.60 * held_support + 0.18 * receiver_support + 0.14 * source_support,
        ).to(device),
        0.02,
        1.0,
    )
    universe_receiver_gate = torch.clamp(
        (
            0.28 * receiver_score
            + 0.20 * final_objective
            + 0.18 * net_utility
            + 0.14 * target_receiver
            + 0.12 * resource_efficiency
            + 0.08 * liquidity_support
            - 0.24 * risk_pressure
        )
        * (1.0 - held_support),
        0.0,
        1.0,
    )
    universe_source_gate = torch.clamp(
        (
            0.30 * source_score
            + 0.20 * target_source
            + 0.18 * target_credit
            + 0.16 * release_preference
            - 0.32 * false_source_pressure
            - 0.10 * risk_pressure
        )
        * held_support,
        0.0,
        1.0,
    )
    receiver_solver_support = receiver_support
    source_solver_support = source_support
    if full_universe_mode:
        receiver_solver_support = torch.clamp(torch.maximum(receiver_support, universe_receiver_gate * liquidity_support), 0.0, 1.0)
        source_solver_support = torch.clamp(torch.maximum(source_support, universe_source_gate), 0.0, 1.0)
    candidate_support = torch.clamp(receiver_solver_support + source_solver_support + held_support, 0.0, 1.0)
    date_code = targets["date_code"].to(device).long()

    term_values: dict[str, list[torch.Tensor]] = {name: [] for name in term_names if name not in {"path_risk_loss", "total"}}
    day_losses: list[torch.Tensor] = []
    day_return_proxies: list[torch.Tensor] = []
    solved_day_count = 0
    attempted_day_count = 0
    for date_value in torch.unique(date_code)[: max(1, int(max_days))]:
        day_mask = date_code == date_value
        day_size = int(day_mask.sum().detach().cpu().item())
        if day_size <= 0:
            continue
        day_candidate_support = candidate_support[day_mask]
        if float(day_candidate_support.sum().detach().cpu()) <= 0.0:
            continue
        attempted_day_count += 1
        current_day = current_weight[day_mask]
        day_position_cap = torch.clamp(max_position_weight_target[day_mask].mean(), 0.04, 0.40)
        day_turnover_limit = torch.clamp(turnover_budget_target[day_mask].mean(), 0.04, 1.0)
        day_gross_target = torch.clamp(gross_exposure_target[day_mask].mean(), 0.0, 1.0)
        day_risk = risk_pressure[day_mask].mean()
        day_cash_timing_signal = cash_timing_signal[day_mask].mean()
        deploy_pressure = torch.clamp(
            0.30 * target_net[day_mask].mean()
            + 0.22 * target_efficiency[day_mask].mean()
            + 0.18 * deploy_target[day_mask].mean()
            + 0.16 * final_objective[day_mask].mean()
            + 0.14 * torch.relu(day_gross_target - current_day.sum())
            - 0.44 * day_risk,
            0.0,
            1.0,
        )
        receiver_value = torch.clamp(
            0.34 * target_receiver[day_mask]
            + 0.20 * target_net[day_mask]
            + 0.16 * target_efficiency[day_mask]
            + 0.14 * spread_reward[day_mask]
            + 0.16 * torch.clamp(receiver_forward[day_mask] / 0.10, -1.0, 1.0)
            - 0.22 * risk_pressure[day_mask],
            -1.0,
            1.0,
        )
        source_release_value = torch.clamp(
            0.34 * target_source[day_mask]
            + 0.24 * target_credit[day_mask]
            + 0.24 * release_preference[day_mask]
            - 0.28 * torch.clamp(source_forward[day_mask] / 0.10, -1.0, 1.0)
            - 0.56 * false_source_pressure[day_mask],
            -1.0,
            1.0,
        )
        predicted_utility = torch.clamp(
            0.32 * receiver_score[day_mask]
            + 0.20 * net_utility[day_mask]
            + 0.16 * final_objective[day_mask]
            + 0.12 * resource_efficiency[day_mask]
            + 0.10 * receiver_support[day_mask]
            + 0.10 * current_day
            - 0.24 * risk_pressure[day_mask]
            + 0.26 * source_score[day_mask] * current_day,
            -1.0,
            1.0,
        )
        oracle_utility = torch.clamp(
            0.56 * receiver_value * receiver_support[day_mask]
            + 0.40 * source_release_value * current_day
            + 0.16 * target_efficiency[day_mask]
            - 0.30 * false_source_pressure[day_mask] * current_day,
            -1.0,
            1.0,
        )
        priority = (
            predicted_utility.detach()
            + 0.28 * current_day.detach()
            + 0.18 * receiver_support[day_mask].detach()
            + 0.18 * source_support[day_mask].detach()
        )
        priority = torch.where(day_candidate_support > 0.0, priority, torch.full_like(priority, -1.0e6))
        selected_count = min(int(slot_count), day_size)
        selected_local = torch.topk(priority, k=selected_count, largest=True).indices
        selected_current = current_day[selected_local]
        selected_receiver_support = receiver_solver_support[day_mask][selected_local]
        selected_source_support = source_solver_support[day_mask][selected_local]
        selected_legacy_receiver_support = receiver_support[day_mask][selected_local]
        selected_legacy_source_support = source_support[day_mask][selected_local]
        selected_false_source = false_source_pressure[day_mask][selected_local]
        selected_risk = risk_pressure[day_mask][selected_local]
        selected_liquidity = liquidity_support[day_mask][selected_local]
        selected_impact_cost = impact_cost[day_mask][selected_local]
        selected_factor_concentration = factor_concentration[day_mask][selected_local]
        selected_propensity = behavior_propensity[day_mask][selected_local]
        selected_forward = torch.where(
            selected_current > 1.0e-8,
            source_forward[day_mask][selected_local],
            receiver_forward[day_mask][selected_local],
        )
        selected_benchmark = forward_benchmark_1d[day_mask][selected_local]
        selected_benchmark_3d = forward_benchmark_3d[day_mask][selected_local]
        selected_predicted_utility = predicted_utility[selected_local]
        selected_oracle_utility = oracle_utility[selected_local]
        target_position_cap = torch.ones_like(selected_current) * day_position_cap
        selected_upper = torch.where(
            selected_receiver_support > 0.0,
            torch.maximum(target_position_cap, selected_current),
            selected_current,
        )
        selected_upper = torch.clamp(selected_upper, 0.0, 1.0)

        utility_vec = _pad_fixed_slot_vector(selected_predicted_utility, int(slot_count), dtype=torch.float64)
        oracle_vec = _pad_fixed_slot_vector(selected_oracle_utility, int(slot_count), dtype=torch.float64)
        current_vec = _pad_fixed_slot_vector(selected_current, int(slot_count), dtype=torch.float64)
        upper_vec = _pad_fixed_slot_vector(selected_upper, int(slot_count), dtype=torch.float64)
        receiver_support_vec = _pad_fixed_slot_vector(selected_receiver_support, int(slot_count), dtype=receiver_score.dtype)
        source_support_vec = _pad_fixed_slot_vector(selected_source_support, int(slot_count), dtype=receiver_score.dtype)
        legacy_receiver_support_vec = _pad_fixed_slot_vector(selected_legacy_receiver_support, int(slot_count), dtype=receiver_score.dtype)
        legacy_source_support_vec = _pad_fixed_slot_vector(selected_legacy_source_support, int(slot_count), dtype=receiver_score.dtype)
        false_source_vec = _pad_fixed_slot_vector(selected_false_source, int(slot_count), dtype=receiver_score.dtype)
        risk_vec = _pad_fixed_slot_vector(selected_risk, int(slot_count), dtype=receiver_score.dtype)
        liquidity_vec = _pad_fixed_slot_vector(selected_liquidity, int(slot_count), dtype=receiver_score.dtype)
        impact_cost_vec = _pad_fixed_slot_vector(selected_impact_cost, int(slot_count), dtype=receiver_score.dtype)
        factor_concentration_vec = _pad_fixed_slot_vector(selected_factor_concentration, int(slot_count), dtype=receiver_score.dtype)
        propensity_vec = _pad_fixed_slot_vector(selected_propensity, int(slot_count), dtype=receiver_score.dtype)
        benchmark_vec = _pad_fixed_slot_vector(selected_benchmark, int(slot_count), dtype=receiver_score.dtype)
        forward_vec = _pad_fixed_slot_vector(
            0.68 * (selected_forward - selected_benchmark) + 0.32 * (selected_forward - selected_benchmark_3d),
            int(slot_count),
            dtype=receiver_score.dtype,
        )
        gross_param = torch.clamp(day_gross_target, 0.02, 1.0).detach().to(dtype=torch.float64)
        turnover_param = torch.clamp(day_turnover_limit, 0.04, 1.0).detach().to(dtype=torch.float64)

        try:
            solution, = layer(
                utility_vec.to("cpu"),
                current_vec.to("cpu"),
                upper_vec.detach().to("cpu"),
                gross_param.to("cpu"),
                turnover_param.to("cpu"),
                solver_args=CVXPY_CONVEX_ALLOCATION_SOLVER_ARGS,
            )
            oracle_solution, = layer(
                oracle_vec.detach().to("cpu"),
                current_vec.detach().to("cpu"),
                upper_vec.detach().to("cpu"),
                gross_param.to("cpu"),
                turnover_param.to("cpu"),
                solver_args=CVXPY_CONVEX_ALLOCATION_SOLVER_ARGS,
            )
        except Exception:
            continue
        solved_day_count += 1
        solution = torch.clamp(solution.to(device=device, dtype=receiver_score.dtype), 0.0, 1.0)
        oracle_solution = torch.clamp(oracle_solution.to(device=device, dtype=receiver_score.dtype), 0.0, 1.0)
        current_slot = current_vec.to(device=device, dtype=receiver_score.dtype)
        upper_slot = upper_vec.to(device=device, dtype=receiver_score.dtype)
        utility_slot = utility_vec.to(device=device, dtype=receiver_score.dtype)
        oracle_slot = oracle_vec.to(device=device, dtype=receiver_score.dtype)

        turnover = torch.abs(solution - current_slot).sum()
        oracle_turnover = torch.abs(oracle_solution - current_slot).sum()
        release_flow = torch.relu(current_slot - solution)
        deploy_flow = torch.relu(solution - current_slot)
        dynamic_cost = torch.clamp(impact_cost_vec + 0.0025 * (1.0 - liquidity_vec) + 0.0030 * risk_vec, 0.0005, 0.0500)
        oracle_value = (
            (oracle_solution * oracle_slot).sum()
            - (dynamic_cost * torch.abs(oracle_solution - current_slot)).sum()
            - 0.08 * (oracle_solution * risk_vec).sum()
            - 0.03 * torch.square(oracle_solution).sum()
        )
        realized_value = (
            (solution * oracle_slot).sum()
            - (dynamic_cost * torch.abs(solution - current_slot)).sum()
            - 0.08 * (solution * risk_vec).sum()
            - 0.03 * torch.square(solution).sum()
        )
        solver_regret = torch.relu(oracle_value - realized_value + 0.006)
        solution_tracking_loss = nn.functional.smooth_l1_loss(solution, oracle_solution.detach())
        gross_residual = torch.square(torch.relu(solution.sum() - day_gross_target - 0.02)) + torch.square(
            torch.relu(day_gross_target - solution.sum()) * torch.clamp(deploy_pressure - day_risk, 0.0, 1.0)
        )
        turnover_residual = torch.square(torch.relu(turnover - day_turnover_limit))
        position_residual = torch.square(torch.relu(solution - day_position_cap)).mean()
        unsupported_mass = (deploy_flow * torch.square(1.0 - receiver_support_vec)).sum() + (
            release_flow * torch.square(1.0 - source_support_vec)
        ).sum()
        false_source_mass = (release_flow * false_source_vec).sum()
        behavior_support_loss = unsupported_mass + 0.50 * false_source_mass
        cash_after = torch.clamp(1.0 - solution.sum() - (dynamic_cost * torch.abs(solution - current_slot)).sum(), 0.0, 1.0)
        cash_timing_loss = (
            torch.relu(deploy_pressure - day_risk - 0.05) * torch.square(cash_after)
            + torch.relu(day_risk - deploy_pressure + 0.05) * torch.square(torch.relu(day_cash_timing_signal - cash_after))
        )
        deploy_flow = torch.relu(solution - current_slot)
        release_flow = torch.relu(current_slot - solution)
        selected_indicator = torch.zeros(day_size, device=device, dtype=receiver_score.dtype)
        selected_indicator[selected_local] = 1.0
        full_day_predicted_priority = torch.clamp(
            0.30 * predicted_utility
            + 0.24 * receiver_solver_support[day_mask]
            + 0.16 * source_solver_support[day_mask]
            + 0.14 * liquidity_support[day_mask]
            + 0.10 * current_day
            - 0.14 * risk_pressure[day_mask],
            -1.0,
            1.0,
        )
        full_day_oracle_priority = torch.clamp(
            0.34 * target_net[day_mask]
            + 0.24 * target_receiver[day_mask]
            + 0.18 * target_efficiency[day_mask]
            + 0.14 * liquidity_support[day_mask]
            + 0.10 * torch.clamp(receiver_forward[day_mask] / 0.10, -1.0, 1.0)
            - 0.20 * risk_pressure[day_mask],
            0.0,
            1.0,
        )
        missed_candidate = torch.clamp((1.0 - selected_indicator) * (1.0 - held_support[day_mask]), 0.0, 1.0)
        candidate_coverage_loss = (
            missed_candidate
            * torch.clamp(full_day_oracle_priority - 0.48, 0.0, 1.0)
            * torch.relu(full_day_oracle_priority - full_day_predicted_priority + 0.025)
        ).sum() / torch.clamp(missed_candidate.sum(), min=1.0)
        expansion_deploy = deploy_flow * torch.clamp(1.0 - legacy_receiver_support_vec, 0.0, 1.0)
        expansion_release = release_flow * torch.clamp(1.0 - legacy_source_support_vec, 0.0, 1.0)
        universe_expansion_loss = (
            (expansion_deploy * torch.square(1.0 - liquidity_vec)).sum()
            + 0.60 * (expansion_deploy * torch.relu(0.10 - propensity_vec)).sum()
            + 0.35 * (expansion_release * torch.relu(0.10 - propensity_vec)).sum()
        )
        liquidity_impact_loss = (
            (torch.abs(solution - current_slot) * dynamic_cost).sum()
            + 0.18 * (deploy_flow * torch.square(1.0 - liquidity_vec)).sum()
        )
        concentration_risk_loss = (
            0.70 * torch.square(solution).sum()
            + 0.30 * (solution * factor_concentration_vec).sum()
        ) * torch.clamp(0.40 + day_risk, 0.40, 1.30)
        behavior_return_proxy = (current_slot * forward_vec).sum()
        policy_return_proxy = (solution * forward_vec).sum() - (dynamic_cost * torch.abs(solution - current_slot)).sum()
        propensity_support_loss = (
            torch.abs(solution - current_slot) * torch.square(torch.relu(0.12 - propensity_vec))
        ).sum()
        centered_policy_advantage = (solution - current_slot) * forward_vec
        advantage_variance_proxy = torch.square(centered_policy_advantage).sum()
        ope_lower_bound = policy_return_proxy - 0.35 * torch.sqrt(advantage_variance_proxy + 1.0e-6) - 0.04 * turnover
        ope_lower_bound_loss = torch.relu(behavior_return_proxy - ope_lower_bound + 0.006) if conservative_ope_mode else torch.tensor(0.0, device=device)
        doubly_robust_proxy = (
            policy_return_proxy
            + ((solution - current_slot) * (forward_vec - benchmark_vec) / torch.clamp(propensity_vec, min=0.05)).sum() * 0.10
        )
        doubly_robust_gap_loss = (
            torch.relu(behavior_return_proxy - doubly_robust_proxy + 0.005) * torch.clamp(turnover, 0.0, 1.0)
            if conservative_ope_mode
            else torch.tensor(0.0, device=device)
        )
        day_return_proxy = torch.clamp(policy_return_proxy + cash_after * torch.clamp(forward_benchmark_1d[day_mask].mean(), -0.08, 0.08), -0.50, 0.50)
        day_return_proxies.append(day_return_proxy)
        component_values = {
            "solver_regret": solver_regret,
            "solution_tracking_loss": solution_tracking_loss,
            "gross_residual": gross_residual,
            "turnover_residual": turnover_residual,
            "position_residual": position_residual,
            "unsupported_mass": unsupported_mass,
            "false_source_mass": false_source_mass,
            "behavior_support_loss": behavior_support_loss,
            "cash_timing_loss": cash_timing_loss,
            "candidate_coverage_loss": candidate_coverage_loss,
            "universe_expansion_loss": universe_expansion_loss,
            "liquidity_impact_loss": liquidity_impact_loss,
            "concentration_risk_loss": concentration_risk_loss,
            "ope_lower_bound_loss": ope_lower_bound_loss,
            "propensity_support_loss": propensity_support_loss,
            "doubly_robust_gap_loss": doubly_robust_gap_loss,
            "solver_success_rate": torch.tensor(1.0, device=device),
            "fallback_surrogate_loss": torch.tensor(0.0, device=device),
        }
        for name, value in component_values.items():
            term_values[name].append(value)
        day_losses.append(
            0.34 * solver_regret
            + 0.24 * solution_tracking_loss
            + 0.15 * gross_residual
            + 0.12 * turnover_residual
            + 0.04 * position_residual
            + 0.22 * unsupported_mass
            + 0.24 * false_source_mass
            + 0.12 * cash_timing_loss
            + (0.16 if full_universe_mode else 0.04) * candidate_coverage_loss
            + (0.14 if full_universe_mode else 0.04) * universe_expansion_loss
            + 0.13 * liquidity_impact_loss
            + 0.10 * concentration_risk_loss
            + (0.18 if conservative_ope_mode else 0.0) * ope_lower_bound_loss
            + (0.12 if conservative_ope_mode else 0.0) * propensity_support_loss
            + (0.12 if conservative_ope_mode else 0.0) * doubly_robust_gap_loss
            - 0.04 * (solution * utility_slot).mean()
        )

    fallback_surrogate = torch.tensor(0.0, device=device)
    if not day_losses:
        fallback_surrogate = _portfolio_differentiable_convex_allocation_loss(outputs, targets)
        if return_terms:
            result = {name: torch.tensor(0.0, device=device) for name in term_names}
            result["solver_success_rate"] = torch.tensor(0.0, device=device)
            result["fallback_surrogate_loss"] = fallback_surrogate
            result["total"] = fallback_surrogate
            return result
        return fallback_surrogate

    path_risk_loss = torch.tensor(0.0, device=device)
    if day_return_proxies:
        safe_returns = torch.clamp(torch.stack(day_return_proxies), -0.50, 0.50)
        equity_curve = torch.cumprod(1.0 + safe_returns, dim=0)
        running_peak = torch.cummax(equity_curve, dim=0).values
        drawdown_path = equity_curve / torch.clamp(running_peak, min=1.0e-6) - 1.0
        tail_count = max(1, int(math.ceil(float(len(day_return_proxies)) * 0.20)))
        tail_losses = torch.topk(-safe_returns, k=tail_count).values
        cvar_loss = torch.square(torch.relu(tail_losses.mean() - 0.02))
        drawdown_loss = torch.square(torch.relu(-drawdown_path.min() - 0.12))
        eta = 8.0
        normalized_oce = (torch.logsumexp(-eta * safe_returns, dim=0) - math.log(float(len(day_return_proxies)))) / eta
        oce_loss = torch.square(torch.relu(normalized_oce + 0.01))
        path_risk_loss = 0.45 * cvar_loss + 0.35 * drawdown_loss + 0.20 * oce_loss
    success_rate = torch.tensor(
        float(solved_day_count) / float(max(attempted_day_count, 1)),
        device=device,
        dtype=receiver_score.dtype,
    )
    total = torch.stack(day_losses).mean() + 0.16 * path_risk_loss + 0.25 * torch.relu(0.75 - success_rate)
    if return_terms:
        terms: dict[str, torch.Tensor] = {
            name: (torch.stack(values).mean() if values else torch.tensor(0.0, device=device))
            for name, values in term_values.items()
        }
        terms["path_risk_loss"] = path_risk_loss
        terms["solver_success_rate"] = success_rate
        terms["fallback_surrogate_loss"] = fallback_surrogate
        terms["total"] = total
        return terms
    return total


def _portfolio_full_universe_convex_allocation_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    enable_solver: bool = True,
    return_terms: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    if not enable_solver:
        result = _portfolio_differentiable_convex_allocation_loss(outputs, targets, return_terms=return_terms)
        if return_terms and isinstance(result, dict):
            zero = torch.tensor(0.0, device=next(iter(outputs.values())).device)
            result = dict(result)
            result.setdefault("solver_regret", zero)
            result.setdefault("solution_tracking_loss", zero)
            result.setdefault("gross_residual", result.get("gross_exposure_residual", zero))
            result.setdefault("turnover_residual", result.get("constraint_residual", zero))
            result.setdefault("position_residual", zero)
            result.setdefault("universe_expansion_loss", zero)
            result.setdefault("liquidity_impact_loss", zero)
            result.setdefault("concentration_risk_loss", zero)
            result.setdefault("ope_lower_bound_loss", result.get("conservative_ope_loss", zero))
            result.setdefault("propensity_support_loss", result.get("behavior_support_loss", zero))
            result.setdefault("doubly_robust_gap_loss", zero)
            result["solver_success_rate"] = zero
            result["fallback_surrogate_loss"] = result.get("total", zero)
        return result
    return _portfolio_cvxpy_convex_allocation_loss(
        outputs,
        targets,
        slot_count=CVXPY_FULL_UNIVERSE_ALLOCATION_SLOT_COUNT,
        max_days=CVXPY_FULL_UNIVERSE_ALLOCATION_MAX_DAYS_PER_BATCH,
        full_universe_mode=True,
        conservative_ope_mode=True,
        return_terms=return_terms,
    )


_NATIVE_ALLOCATION_VECTOR_NAMES: tuple[str, ...] = (
    "portfolio_daily_target_weight",
    "portfolio_daily_target_delta",
    "portfolio_daily_target_cash_weight",
    "portfolio_daily_target_turnover",
    "portfolio_daily_native_receiver_score",
    "portfolio_daily_native_source_score",
    "portfolio_daily_native_cash_score",
)

_NATIVE_ALLOCATION_TERM_NAMES: tuple[str, ...] = (
    "allocation_sum_error",
    "cash_reserve_error",
    "position_cap_violation",
    "turnover_violation",
    "unsupported_receiver_weight",
    "sell_nonheld_violation",
    "receiver_flow_mean",
    "source_flow_mean",
    "funding_shortfall_loss",
    "cash_timing_loss",
    "decision_utility_loss",
    "risk_cost_loss",
    "source_breadth_loss",
    "exposure_utilization_loss",
    "source_dead_loss",
    "native_source_threshold_loss",
    "legacy_mask_block_loss",
    "native_negative_delta_count",
    "native_source_candidate_count",
    "native_source_flow_without_sellable_support",
    "native_source_blocked_by_legacy_mask",
    "native_source_audit_threshold_gap",
    "total",
)


def _project_native_allocation_vector(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    return_terms: bool = False,
) -> dict[str, torch.Tensor]:
    required_outputs = {
        "portfolio_daily_allocation_weight_logit",
        "portfolio_daily_cash_reserve_logit",
        "portfolio_daily_allocation_risk_buffer_logit",
    }
    required_targets = {
        "date_code",
        "current_weight",
        "portfolio_daily_receiver_candidate_mask",
        "portfolio_daily_source_candidate_mask",
    }
    device = next(iter(outputs.values())).device
    dtype = next(iter(outputs.values())).dtype
    row_count = int(next(iter(outputs.values())).numel())

    def _zero_projection() -> dict[str, torch.Tensor]:
        zero_vec = torch.zeros(row_count, device=device, dtype=dtype)
        result = {
            "portfolio_daily_target_weight": zero_vec,
            "portfolio_daily_target_delta": zero_vec,
            "portfolio_daily_target_cash_weight": zero_vec,
            "portfolio_daily_target_turnover": zero_vec,
            "portfolio_daily_native_receiver_score": zero_vec,
            "portfolio_daily_native_source_score": zero_vec,
            "portfolio_daily_native_cash_score": zero_vec,
        }
        if return_terms:
            zero = torch.tensor(0.0, device=device, dtype=dtype)
            result.update({name: zero for name in _NATIVE_ALLOCATION_TERM_NAMES})
        return result

    if not required_outputs.issubset(outputs) or not required_targets.issubset(targets):
        return _zero_projection()

    weight_logit = outputs["portfolio_daily_allocation_weight_logit"].to(device).flatten()
    cash_logit = outputs["portfolio_daily_cash_reserve_logit"].to(device).flatten()
    risk_buffer_logit = outputs["portfolio_daily_allocation_risk_buffer_logit"].to(device).flatten()
    date_code = targets["date_code"].to(device).flatten()
    current_weight = torch.clamp(targets["current_weight"].to(device).flatten(), 0.0, 1.0)
    receiver_mask = torch.clamp(targets["portfolio_daily_receiver_candidate_mask"].to(device).flatten(), 0.0, 1.0)
    source_mask = torch.clamp(targets["portfolio_daily_source_candidate_mask"].to(device).flatten(), 0.0, 1.0)
    if weight_logit.numel() == 0 or date_code.numel() != weight_logit.numel():
        return _zero_projection()

    def _target(name: str, default: float | torch.Tensor) -> torch.Tensor:
        if name in targets:
            value = targets[name].to(device).flatten()
        elif torch.is_tensor(default):
            value = default.to(device).flatten()
        else:
            value = torch.full_like(weight_logit, float(default))
        if value.numel() == 1 and weight_logit.numel() != 1:
            value = torch.full_like(weight_logit, float(value.detach().cpu()))
        if value.numel() != weight_logit.numel():
            value = torch.full_like(weight_logit, float(torch.nanmean(value).detach().cpu()) if value.numel() else float(default))
        return value

    held_support = (current_weight > 1.0e-8).to(dtype=dtype, device=device)
    receiver_support = receiver_mask
    if "portfolio_daily_receiver_executable_candidate" in targets:
        receiver_support = receiver_support * torch.clamp(
            targets["portfolio_daily_receiver_executable_candidate"].to(device).flatten(),
            0.0,
            1.0,
        )
    legacy_source_support = torch.clamp(source_mask * held_support, 0.0, 1.0)
    source_support = held_support
    receiver_support = torch.clamp(receiver_support, 0.0, 1.0)
    source_support = torch.clamp(source_support, 0.0, 1.0)
    eligible_support = torch.clamp(held_support + receiver_support, 0.0, 1.0)

    gross_exposure_target = torch.clamp(_target("gross_exposure_target", 0.60), 0.0, 1.0)
    turnover_budget = torch.clamp(_target("turnover_budget", 0.36), 0.02, 1.0)
    max_position_weight_target = torch.clamp(_target("max_position_weight_target", 0.20), 0.02, 0.50)
    cash_timing_target = torch.clamp(_target("budget_cash_timing_signal_target", 0.0), 0.0, 1.0)
    deploy_target = torch.clamp(_target("portfolio_daily_allocation_cash_deployment_target", 0.0), 0.0, 1.0)
    net_utility_target = torch.clamp(_target("portfolio_daily_allocation_net_utility_target", deploy_target), 0.0, 1.0)
    final_objective = torch.clamp(_target("portfolio_daily_allocation_final_objective", net_utility_target), 0.0, 1.0)
    target_receiver = torch.clamp(_target("portfolio_daily_unified_receiver_score", receiver_support), 0.0, 1.0)
    target_source = torch.clamp(_target("portfolio_daily_unified_source_score", source_support), 0.0, 1.0)
    receiver_forward = torch.clamp(_target("portfolio_daily_receiver_forward_excess_5d", 0.0), -0.20, 0.20)
    source_forward = torch.clamp(_target("portfolio_daily_source_forward_excess_5d", 0.0), -0.20, 0.20)
    risk_pressure = torch.stack(
        [
            torch.clamp(_target("portfolio_daily_allocation_uncertainty_pressure_target", 0.0), 0.0, 1.0),
            torch.clamp(_target("portfolio_daily_allocation_tail_risk_control_target", 0.0), 0.0, 1.0),
            torch.clamp(_target("portfolio_daily_allocation_drawdown_control_target", 0.0), 0.0, 1.0),
            torch.clamp(_target("market_downside_pressure", 0.0), 0.0, 1.0),
            torch.clamp(_target("cash_regime_pressure", 0.0), 0.0, 1.0),
        ],
        dim=0,
    ).amax(dim=0)

    target_weight = torch.zeros_like(weight_logit) + weight_logit * 0.0
    target_delta = torch.zeros_like(weight_logit) + weight_logit * 0.0
    target_cash_weight = torch.zeros_like(weight_logit) + weight_logit * 0.0
    target_turnover = torch.zeros_like(weight_logit) + weight_logit * 0.0
    native_receiver_score = torch.zeros_like(weight_logit) + weight_logit * 0.0
    native_source_score = torch.zeros_like(weight_logit) + weight_logit * 0.0
    native_cash_score = torch.zeros_like(weight_logit) + weight_logit * 0.0

    term_values: dict[str, list[torch.Tensor]] = {
        name: []
        for name in _NATIVE_ALLOCATION_TERM_NAMES
        if name not in {"receiver_flow_mean", "source_flow_mean", "total"}
    }
    receiver_flow_values: list[torch.Tensor] = []
    source_flow_values: list[torch.Tensor] = []
    day_losses: list[torch.Tensor] = []

    unique_dates = torch.unique(date_code.detach())
    for day_value in unique_dates:
        day_mask = date_code == day_value
        if not bool(day_mask.any()):
            continue
        current_day = current_weight[day_mask]
        weight_logit_day = weight_logit[day_mask]
        cash_logit_day = cash_logit[day_mask]
        risk_buffer_day = risk_buffer_logit[day_mask]
        receiver_support_day = receiver_support[day_mask]
        source_support_day = source_support[day_mask]
        legacy_source_support_day = legacy_source_support[day_mask]
        eligible_day = eligible_support[day_mask]
        held_day = held_support[day_mask]
        gross_day = torch.clamp(gross_exposure_target[day_mask].mean(), 0.0, 1.0)
        turnover_limit = torch.clamp(turnover_budget[day_mask].mean(), 0.02, 1.0)
        cap_day = torch.clamp(max_position_weight_target[day_mask].mean(), 0.02, 0.50)
        risk_day = torch.clamp(risk_pressure[day_mask].mean(), 0.0, 1.0)
        cash_timing_day = torch.clamp(cash_timing_target[day_mask].mean(), 0.0, 1.0)
        deploy_day = torch.clamp(
            0.45 * deploy_target[day_mask].mean()
            + 0.35 * net_utility_target[day_mask].mean()
            + 0.20 * final_objective[day_mask].mean(),
            0.0,
            1.0,
        )
        raw_cash_reserve = torch.sigmoid(cash_logit_day).mean()
        raw_risk_buffer = torch.sigmoid(risk_buffer_day).mean()
        cash_reserve_target = torch.clamp(
            0.03
            + 0.24 * raw_cash_reserve
            + 0.18 * raw_risk_buffer
            + 0.26 * risk_day
            + 0.16 * cash_timing_day
            - 0.18 * deploy_day,
            0.02,
            0.80,
        )
        stock_budget = torch.minimum(gross_day, torch.clamp(1.0 - cash_reserve_target, 0.0, 1.0))

        if float(eligible_day.detach().sum().cpu()) <= 0.0 or float(stock_budget.detach().cpu()) <= 1.0e-8:
            target_day = torch.zeros_like(current_day)
        else:
            masked_logits = torch.where(
                eligible_day > 0.0,
                weight_logit_day,
                torch.full_like(weight_logit_day, -1.0e9),
            )
            base_weight = torch.softmax(masked_logits, dim=0) * eligible_day
            base_weight = base_weight / torch.clamp(base_weight.sum(), min=1.0e-8)
            target_day = torch.clamp(base_weight * stock_budget, 0.0, cap_day) * eligible_day

        delta_day = target_day - current_day
        turnover_day = torch.abs(delta_day).sum()
        if float(turnover_day.detach().cpu()) > float(turnover_limit.detach().cpu()) + 1.0e-8:
            scale = torch.clamp(turnover_limit / torch.clamp(turnover_day, min=1.0e-8), 0.0, 1.0)
            target_day = current_day + delta_day * scale
            target_day = torch.clamp(target_day, 0.0, cap_day) * eligible_day
            delta_day = target_day - current_day
            turnover_day = torch.abs(delta_day).sum()

        source_audit_threshold = torch.maximum(
            torch.full_like(current_day, 0.00125),
            torch.clamp(current_day, min=0.0) * 0.010,
        )
        clamped_current_day = torch.clamp(current_day, 0.0, cap_day)
        preliminary_release = torch.relu(clamped_current_day - target_day) * held_day
        preliminary_source_count = (
            (preliminary_release > source_audit_threshold).to(dtype=dtype, device=device) * held_day
        ).sum()
        held_count = held_day.sum()
        total_preliminary_release = preliminary_release.sum()
        desired_source_count = torch.minimum(held_count, torch.tensor(3.0, device=device, dtype=dtype))
        if (
            float(total_preliminary_release.detach().cpu()) > 1.0e-8
            and float(held_count.detach().cpu()) > 0.0
            and float(preliminary_source_count.detach().cpu()) < float(desired_source_count.detach().cpu())
        ):
            selected_count = max(1, min(3, int(float(held_count.detach().cpu()))))
            source_order_score = (
                preliminary_release
                + 0.030 * torch.clamp(target_source[day_mask], 0.0, 1.0)
                + 0.020 * torch.relu(-source_forward[day_mask])
                + 0.010 * legacy_source_support_day
            )
            source_order_score = torch.where(
                held_day > 0.0,
                source_order_score,
                torch.full_like(source_order_score, -1.0e9),
            )
            selected_indices = torch.topk(source_order_score, k=selected_count).indices
            concentrated_release = torch.zeros_like(current_day)
            remaining_release = total_preliminary_release
            release_floor = torch.maximum(
                source_audit_threshold * 1.35,
                total_preliminary_release / float(selected_count),
            )
            for index_tensor in selected_indices:
                index = int(index_tensor.detach().cpu())
                capacity = clamped_current_day[index]
                proposed_release = torch.minimum(
                    capacity,
                    torch.maximum(preliminary_release[index], release_floor[index]),
                )
                assigned_release = torch.minimum(proposed_release, remaining_release)
                concentrated_release[index] = assigned_release
                remaining_release = torch.relu(remaining_release - assigned_release)
            if float(remaining_release.detach().cpu()) > 1.0e-8:
                for index_tensor in selected_indices:
                    if float(remaining_release.detach().cpu()) <= 1.0e-8:
                        break
                    index = int(index_tensor.detach().cpu())
                    extra_capacity = torch.relu(clamped_current_day[index] - concentrated_release[index])
                    extra_release = torch.minimum(extra_capacity, remaining_release)
                    concentrated_release[index] = concentrated_release[index] + extra_release
                    remaining_release = torch.relu(remaining_release - extra_release)
            releasable_rows = preliminary_release > 1.0e-8
            target_day = torch.where(releasable_rows, clamped_current_day, target_day)
            target_day = target_day - concentrated_release
            target_day = torch.clamp(target_day, 0.0, cap_day) * eligible_day
            delta_day = target_day - current_day
            turnover_day = torch.abs(delta_day).sum()

        cost_proxy = torch.clamp(0.0015 * turnover_day, 0.0, 0.03)
        total_with_cost = target_day.sum() + cost_proxy
        if float(total_with_cost.detach().cpu()) > 1.0 + 1.0e-8:
            target_day = target_day * torch.clamp((1.0 - cost_proxy) / torch.clamp(target_day.sum(), min=1.0e-8), 0.0, 1.0)
            target_day = torch.clamp(target_day, 0.0, cap_day) * eligible_day
            delta_day = target_day - current_day
            turnover_day = torch.abs(delta_day).sum()
            cost_proxy = torch.clamp(0.0015 * turnover_day, 0.0, 0.03)
        cash_after = torch.clamp(1.0 - target_day.sum() - cost_proxy, 0.0, 1.0)

        receiver_headroom = torch.clamp(cap_day - current_day, min=1.0e-6)
        receiver_score_day = torch.clamp(torch.relu(delta_day) / receiver_headroom, 0.0, 1.0) * receiver_support_day
        source_score_day = torch.clamp(torch.relu(-delta_day) / torch.clamp(current_day, min=1.0e-6), 0.0, 1.0) * source_support_day
        cash_score_day = torch.ones_like(current_day) * cash_after
        release_amount = torch.relu(-delta_day) * held_day
        audit_threshold = torch.maximum(
            torch.full_like(current_day, 0.00125),
            torch.clamp(current_day, min=0.0) * 0.010,
        )
        native_source_active = (release_amount > audit_threshold).to(dtype=dtype, device=device) * held_day
        native_negative_delta = (release_amount > 1.0e-8).to(dtype=dtype, device=device) * held_day
        legacy_blocked_source_flow = release_amount * (1.0 - legacy_source_support_day)
        native_source_threshold_gap = (
            torch.relu(audit_threshold - release_amount)
            * (release_amount > 1.0e-8).to(dtype=dtype, device=device)
            * held_day
        ).sum()

        target_weight[day_mask] = target_day
        target_delta[day_mask] = delta_day
        target_cash_weight[day_mask] = cash_score_day
        target_turnover[day_mask] = torch.ones_like(current_day) * turnover_day
        native_receiver_score[day_mask] = receiver_score_day
        native_source_score[day_mask] = source_score_day
        native_cash_score[day_mask] = cash_score_day

        allocation_sum_error = torch.square(torch.relu(target_day.sum() + cash_after + cost_proxy - 1.0))
        cash_reserve_error = torch.square(torch.relu(cash_reserve_target - cash_after))
        position_cap_violation = torch.square(torch.relu(target_day - cap_day)).mean()
        turnover_violation = torch.square(torch.relu(turnover_day - turnover_limit))
        unsupported_receiver_weight = (torch.relu(delta_day) * (1.0 - receiver_support_day)).sum()
        sell_nonheld_violation = (torch.relu(-delta_day) * (1.0 - held_day)).sum()
        receiver_flow = torch.relu(delta_day).sum()
        source_flow = release_amount.sum()
        current_cash = torch.clamp(1.0 - current_day.sum(), 0.0, 1.0)
        cash_release = torch.relu(current_cash - cash_after)
        raw_receiver_demand = (
            torch.sigmoid(weight_logit_day)
            * receiver_support_day
            * torch.clamp(cap_day - current_day, min=0.0)
            * (0.55 + 0.45 * torch.clamp(target_receiver[day_mask], 0.0, 1.0))
        ).sum()
        funding_shortfall_loss = torch.square(torch.relu(raw_receiver_demand - source_flow - cash_release))
        over_cash_loss = torch.square(cash_after) * torch.relu(deploy_day - risk_day - cash_timing_day - 0.04)
        risk_cash_floor = torch.clamp(
            0.04 + 0.36 * risk_day + 0.22 * cash_timing_day - 0.20 * deploy_day,
            0.02,
            0.80,
        )
        under_cash_loss = torch.square(torch.relu(risk_cash_floor - cash_after)) * torch.clamp(
            0.30 + risk_day + cash_timing_day,
            0.0,
            1.5,
        )
        cash_timing_loss = 0.55 * over_cash_loss + 0.45 * under_cash_loss
        receiver_alpha = (
            torch.relu(delta_day)
            * receiver_support_day
            * torch.clamp(0.50 * receiver_forward[day_mask] + 0.025 * target_receiver[day_mask], -0.10, 0.20)
        ).sum()
        source_alpha = (
            torch.relu(-delta_day)
            * source_support_day
            * torch.clamp(-source_forward[day_mask] + 0.025 * target_source[day_mask], -0.10, 0.20)
        ).sum()
        protected_positive_source = (
            torch.relu(-delta_day)
            * source_support_day
            * torch.relu(source_forward[day_mask])
        ).sum()
        decision_utility_loss = torch.relu(0.006 - receiver_alpha - source_alpha + protected_positive_source).square()
        risk_cost_loss = (
            torch.square(torch.relu(risk_day - cash_after))
            + 0.05 * turnover_day
            + 0.04 * (target_day.square().sum() * torch.clamp(0.50 + risk_day, 0.50, 1.50))
        )
        source_count = source_support_day.sum()
        if float(source_count.detach().cpu()) > 0.0:
            soft_source_breadth = native_source_active.sum()
            target_source_breadth = torch.minimum(source_count, torch.tensor(3.0, device=device, dtype=dtype))
            source_breadth_loss = torch.square(torch.relu(target_source_breadth - soft_source_breadth)) / torch.clamp(
                target_source_breadth.square(),
                min=1.0,
            )
        else:
            source_breadth_loss = torch.tensor(0.0, device=device, dtype=dtype)
        exposure_utilization_loss = torch.square(torch.relu(stock_budget - target_day.sum())) * torch.clamp(
            0.35 + deploy_day - risk_day,
            0.0,
            1.25,
        )
        native_source_candidate_count = native_source_active.sum()
        native_negative_delta_count = native_negative_delta.sum()
        native_source_flow_without_sellable_support = (torch.relu(-delta_day) * (1.0 - held_day)).sum()
        native_source_blocked_by_legacy_mask = legacy_blocked_source_flow.sum()
        receiver_pressure_for_source = torch.clamp(
            raw_receiver_demand + torch.relu(stock_budget - target_day.sum()),
            0.0,
            1.0,
        )
        source_dead_loss = (
            torch.square(torch.relu(torch.minimum(source_count, torch.tensor(3.0, device=device, dtype=dtype)) - native_source_candidate_count))
            / torch.clamp(torch.minimum(source_count, torch.tensor(3.0, device=device, dtype=dtype)).square(), min=1.0)
            if float(source_count.detach().cpu()) > 0.0
            else torch.tensor(1.0, device=device, dtype=dtype)
        ) * receiver_pressure_for_source
        native_source_threshold_loss = (
            torch.square(native_source_threshold_gap)
            + torch.square(torch.relu(torch.tensor(1.0, device=device, dtype=dtype) - native_source_candidate_count))
            * receiver_pressure_for_source
        )
        legacy_mask_block_loss = torch.square(native_source_blocked_by_legacy_mask)

        component_loss = (
            0.10 * allocation_sum_error
            + 0.16 * cash_reserve_error
            + 0.16 * position_cap_violation
            + 0.18 * turnover_violation
            + 0.24 * unsupported_receiver_weight
            + 0.22 * sell_nonheld_violation
            + 0.34 * funding_shortfall_loss
            + 0.22 * cash_timing_loss
            + 0.28 * decision_utility_loss
            + 0.18 * risk_cost_loss
            + 0.20 * source_breadth_loss
            + 0.24 * exposure_utilization_loss
            + 0.24 * source_dead_loss
            + 0.18 * native_source_threshold_loss
            + 0.08 * legacy_mask_block_loss
        )
        day_losses.append(component_loss)
        receiver_flow_values.append(receiver_flow)
        source_flow_values.append(source_flow)
        term_values["allocation_sum_error"].append(allocation_sum_error)
        term_values["cash_reserve_error"].append(cash_reserve_error)
        term_values["position_cap_violation"].append(position_cap_violation)
        term_values["turnover_violation"].append(turnover_violation)
        term_values["unsupported_receiver_weight"].append(unsupported_receiver_weight)
        term_values["sell_nonheld_violation"].append(sell_nonheld_violation)
        term_values["funding_shortfall_loss"].append(funding_shortfall_loss)
        term_values["cash_timing_loss"].append(cash_timing_loss)
        term_values["decision_utility_loss"].append(decision_utility_loss)
        term_values["risk_cost_loss"].append(risk_cost_loss)
        term_values["source_breadth_loss"].append(source_breadth_loss)
        term_values["exposure_utilization_loss"].append(exposure_utilization_loss)
        term_values["source_dead_loss"].append(source_dead_loss)
        term_values["native_source_threshold_loss"].append(native_source_threshold_loss)
        term_values["legacy_mask_block_loss"].append(legacy_mask_block_loss)
        term_values["native_negative_delta_count"].append(native_negative_delta_count)
        term_values["native_source_candidate_count"].append(native_source_candidate_count)
        term_values["native_source_flow_without_sellable_support"].append(native_source_flow_without_sellable_support)
        term_values["native_source_blocked_by_legacy_mask"].append(native_source_blocked_by_legacy_mask)
        term_values["native_source_audit_threshold_gap"].append(native_source_threshold_gap)

    result = {
        "portfolio_daily_target_weight": target_weight,
        "portfolio_daily_target_delta": target_delta,
        "portfolio_daily_target_cash_weight": target_cash_weight,
        "portfolio_daily_target_turnover": target_turnover,
        "portfolio_daily_native_receiver_score": native_receiver_score,
        "portfolio_daily_native_source_score": native_source_score,
        "portfolio_daily_native_cash_score": native_cash_score,
    }
    if return_terms:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        terms = {
            name: (torch.stack(values).mean() if values else zero)
            for name, values in term_values.items()
        }
        terms["receiver_flow_mean"] = torch.stack(receiver_flow_values).mean() if receiver_flow_values else zero
        terms["source_flow_mean"] = torch.stack(source_flow_values).mean() if source_flow_values else zero
        terms["total"] = torch.stack(day_losses).mean() if day_losses else zero
        result.update(terms)
    return result


def _portfolio_native_allocation_vector_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    return_terms: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    projection = _project_native_allocation_vector(outputs, targets, return_terms=True)
    if return_terms:
        return {
            name: projection.get(
                name,
                torch.tensor(0.0, device=next(iter(outputs.values())).device),
            )
            for name in _NATIVE_ALLOCATION_TERM_NAMES
        }
    return projection.get("total", torch.tensor(0.0, device=next(iter(outputs.values())).device))


_DAY_SET_NATIVE_ALLOCATION_TERM_NAMES: tuple[str, ...] = (
    "day_set_full_day_batch",
    "day_set_sample_mask_coverage",
    "day_set_padding_weight_violation",
    "allocation_sum_error",
    "cash_reserve_error",
    "position_cap_violation",
    "turnover_violation",
    "unsupported_receiver_weight",
    "sell_nonheld_violation",
    "receiver_flow_mean",
    "source_flow_mean",
    "funding_shortfall_loss",
    "cash_timing_loss",
    "decision_utility_loss",
    "risk_cost_loss",
    "source_breadth_loss",
    "exposure_utilization_loss",
    "source_dead_loss",
    "native_source_threshold_loss",
    "legacy_mask_block_loss",
    "native_negative_delta_count",
    "native_source_candidate_count",
    "native_source_flow_without_sellable_support",
    "native_source_blocked_by_legacy_mask",
    "native_source_audit_threshold_gap",
    "total",
)


def _project_day_set_native_allocation_vector(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    sample_mask: torch.Tensor,
    *,
    return_terms: bool = False,
) -> dict[str, torch.Tensor]:
    device = next(iter(outputs.values())).device
    dtype = next(iter(outputs.values())).dtype
    mask = sample_mask.to(device=device, dtype=torch.bool)
    if mask.ndim != 2:
        raise ValueError("day-set native allocation projection requires sample_mask with shape [B,N].")
    batch_size, max_rows = int(mask.shape[0]), int(mask.shape[1])

    def _output_matrix(name: str) -> torch.Tensor:
        value = outputs[name].to(device=device, dtype=dtype)
        if value.ndim == 1 and value.numel() == batch_size:
            value = value[:, None].expand(batch_size, max_rows)
        if value.shape != mask.shape:
            value = value.reshape(batch_size, max_rows)
        return value

    def _target_matrix(name: str, default: float) -> torch.Tensor:
        if name not in targets:
            return torch.full((batch_size, max_rows), float(default), device=device, dtype=dtype)
        value = targets[name].to(device=device, dtype=dtype)
        if value.ndim == 0:
            return torch.full((batch_size, max_rows), float(value.detach().cpu()), device=device, dtype=dtype)
        if value.ndim == 1:
            if value.numel() == batch_size:
                return value[:, None].expand(batch_size, max_rows)
            if value.numel() == batch_size * max_rows:
                return value.reshape(batch_size, max_rows)
        if tuple(value.shape) == (batch_size, max_rows):
            return value
        return torch.full((batch_size, max_rows), float(default), device=device, dtype=dtype)

    weight_logit = _output_matrix("portfolio_daily_allocation_weight_logit")
    cash_logit = _output_matrix("portfolio_daily_cash_reserve_logit")
    risk_buffer_logit = _output_matrix("portfolio_daily_allocation_risk_buffer_logit")
    masked_weight_logit = torch.where(mask, weight_logit, torch.full_like(weight_logit, -1.0e9))

    flat_outputs = {
        "portfolio_daily_allocation_weight_logit": masked_weight_logit.reshape(-1),
        "portfolio_daily_cash_reserve_logit": cash_logit.reshape(-1),
        "portfolio_daily_allocation_risk_buffer_logit": risk_buffer_logit.reshape(-1),
    }
    flat_targets = {
        "date_code": torch.arange(batch_size, device=device, dtype=dtype)[:, None].expand(batch_size, max_rows).reshape(-1),
        "current_weight": _target_matrix("current_weight", 0.0).reshape(-1),
        "portfolio_daily_receiver_candidate_mask": (_target_matrix("portfolio_daily_receiver_candidate_mask", 0.0) * mask.to(dtype)).reshape(-1),
        "portfolio_daily_source_candidate_mask": (_target_matrix("portfolio_daily_source_candidate_mask", 0.0) * mask.to(dtype)).reshape(-1),
    }
    for name, default in (
        ("portfolio_daily_receiver_executable_candidate", 0.0),
        ("portfolio_daily_source_executable_candidate", 0.0),
        ("gross_exposure_target", 0.60),
        ("turnover_budget", 0.36),
        ("max_position_weight_target", 0.20),
        ("budget_cash_timing_signal_target", 0.0),
        ("portfolio_daily_allocation_cash_deployment_target", 0.0),
        ("portfolio_daily_allocation_net_utility_target", 0.0),
        ("portfolio_daily_allocation_final_objective", 0.0),
        ("portfolio_daily_unified_receiver_score", 0.0),
        ("portfolio_daily_unified_source_score", 0.0),
        ("portfolio_daily_receiver_forward_excess_5d", 0.0),
        ("portfolio_daily_source_forward_excess_5d", 0.0),
        ("portfolio_daily_allocation_uncertainty_pressure_target", 0.0),
        ("portfolio_daily_allocation_tail_risk_control_target", 0.0),
        ("portfolio_daily_allocation_drawdown_control_target", 0.0),
        ("market_downside_pressure", 0.0),
        ("cash_regime_pressure", 0.0),
    ):
        value = _target_matrix(name, default)
        if name.endswith("candidate") or name.endswith("mask"):
            value = value * mask.to(dtype)
        flat_targets[name] = value.reshape(-1)

    flat_projection = _project_native_allocation_vector(flat_outputs, flat_targets, return_terms=True)

    def _reshape(name: str) -> torch.Tensor:
        return flat_projection[name].reshape(batch_size, max_rows) * mask.to(dtype)

    target_weight = _reshape("portfolio_daily_target_weight")
    target_delta = _reshape("portfolio_daily_target_delta")
    native_receiver_score = _reshape("portfolio_daily_native_receiver_score")
    native_source_score = _reshape("portfolio_daily_native_source_score")
    native_cash_score = _reshape("portfolio_daily_native_cash_score")
    target_cash_matrix = flat_projection["portfolio_daily_target_cash_weight"].reshape(batch_size, max_rows)
    target_turnover_matrix = flat_projection["portfolio_daily_target_turnover"].reshape(batch_size, max_rows)
    valid_counts = torch.clamp(mask.to(dtype).sum(dim=1), min=1.0)
    target_cash_weight = (target_cash_matrix * mask.to(dtype)).sum(dim=1) / valid_counts
    target_turnover = (target_turnover_matrix * mask.to(dtype)).sum(dim=1) / valid_counts
    padding_weight_violation = torch.square(target_weight * (~mask).to(dtype)).mean()

    result = {
        "portfolio_daily_target_weight": target_weight,
        "portfolio_daily_target_delta": target_delta,
        "portfolio_daily_target_cash_weight": target_cash_weight,
        "portfolio_daily_target_turnover": target_turnover,
        "portfolio_daily_native_receiver_score": native_receiver_score,
        "portfolio_daily_native_source_score": native_source_score,
        "portfolio_daily_native_cash_score": native_cash_score,
        "padding_weight_violation": padding_weight_violation,
    }
    if return_terms:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        result.update(
            {
                "day_set_full_day_batch": torch.tensor(1.0, device=device, dtype=dtype),
                "day_set_sample_mask_coverage": mask.to(dtype).mean() if mask.numel() else zero,
                "day_set_padding_weight_violation": padding_weight_violation,
            }
        )
        for name in _NATIVE_ALLOCATION_TERM_NAMES:
            result[name] = flat_projection.get(name, zero)
        result["total"] = result["total"] + 0.10 * padding_weight_violation
    return result


def _portfolio_day_set_native_allocation_vector_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    sample_mask: torch.Tensor,
    *,
    return_terms: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    projection = _project_day_set_native_allocation_vector(outputs, targets, sample_mask, return_terms=True)
    if return_terms:
        return {
            name: projection.get(
                name,
                torch.tensor(0.0, device=next(iter(outputs.values())).device),
            )
            for name in _DAY_SET_NATIVE_ALLOCATION_TERM_NAMES
        }
    return projection.get("total", torch.tensor(0.0, device=next(iter(outputs.values())).device))


def _portfolio_capital_flow_closure_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    return_terms: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    required_outputs = {
        "portfolio_daily_unified_receiver_score",
        "portfolio_daily_unified_source_score",
        "portfolio_daily_unified_cash_score",
        "portfolio_daily_allocation_final_objective",
        "portfolio_daily_allocation_net_utility_target",
        "portfolio_daily_allocation_credit_closure_target",
        "portfolio_daily_allocation_resource_efficiency_target",
    }
    required_targets = {
        "date_code",
        "current_weight",
        "portfolio_daily_receiver_candidate_mask",
        "portfolio_daily_source_candidate_mask",
    }
    device = next(iter(outputs.values())).device
    term_names = (
        "receiver_demand_loss",
        "funding_shortfall_loss",
        "source_dead_loss",
        "false_source_loss",
        "over_cash_loss",
        "cash_defense_loss",
        "cash_timing_loss",
        "cash_coherence_loss",
        "exposure_gap_loss",
        "flow_conservation_loss",
        "role_overlap_loss",
        "source_breadth_loss",
        "receiver_demand_mean",
        "desired_receiver_flow_mean",
        "effective_receiver_flow_mean",
        "effective_source_flow_mean",
        "source_need_mean",
        "risk_cash_need_mean",
        "clean_source_supply_mean",
        "cash_release_mean",
        "cash_defense_mean",
        "current_cash_mean",
        "predicted_gross_mean",
        "deploy_pressure_mean",
        "risk_pressure_mean",
        "total",
    )

    def _zero_result() -> torch.Tensor | dict[str, torch.Tensor]:
        zero_value = torch.tensor(0.0, device=device)
        if return_terms:
            return {name: zero_value for name in term_names}
        return zero_value

    if not required_outputs.issubset(outputs) or not required_targets.issubset(targets):
        return _zero_result()

    receiver_score = torch.clamp(outputs["portfolio_daily_unified_receiver_score"].to(device), 0.0, 1.0)
    source_score = torch.clamp(outputs["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    cash_score = torch.clamp(outputs["portfolio_daily_unified_cash_score"].to(device), 0.0, 1.0)
    final_objective = torch.clamp(outputs["portfolio_daily_allocation_final_objective"].to(device), 0.0, 1.0)
    net_utility = torch.clamp(outputs["portfolio_daily_allocation_net_utility_target"].to(device), 0.0, 1.0)
    credit_closure = torch.clamp(outputs["portfolio_daily_allocation_credit_closure_target"].to(device), 0.0, 1.0)
    resource_efficiency = torch.clamp(outputs["portfolio_daily_allocation_resource_efficiency_target"].to(device), 0.0, 1.0)

    zero = torch.zeros_like(receiver_score)
    current_weight = torch.clamp(targets["current_weight"].to(device), 0.0, 1.0)
    held_support = (current_weight > 1.0e-8).to(device=device, dtype=receiver_score.dtype)
    receiver_mask = torch.clamp(targets["portfolio_daily_receiver_candidate_mask"].to(device), 0.0, 1.0)
    source_mask = torch.clamp(targets["portfolio_daily_source_candidate_mask"].to(device), 0.0, 1.0)
    receiver_support = receiver_mask
    if "portfolio_daily_receiver_executable_candidate" in targets:
        receiver_support = receiver_support * torch.clamp(
            targets["portfolio_daily_receiver_executable_candidate"].to(device),
            0.0,
            1.0,
        )
    source_support = source_mask * held_support
    if "portfolio_daily_source_executable_candidate" in targets:
        source_support = source_support * torch.clamp(
            targets["portfolio_daily_source_executable_candidate"].to(device),
            0.0,
            1.0,
        )
    receiver_support = torch.clamp(receiver_support, 0.0, 1.0)
    source_support = torch.clamp(source_support, 0.0, 1.0)

    target_receiver = torch.clamp(targets.get("portfolio_daily_unified_receiver_score", receiver_support).to(device), 0.0, 1.0)
    target_source = torch.clamp(targets.get("portfolio_daily_unified_source_score", source_support).to(device), 0.0, 1.0)
    target_cash = torch.clamp(targets.get("portfolio_daily_unified_cash_score", cash_score.detach()).to(device), 0.0, 1.0)
    target_net = torch.clamp(targets.get("portfolio_daily_allocation_net_utility_target", target_receiver).to(device), 0.0, 1.0)
    target_credit = torch.clamp(targets.get("portfolio_daily_allocation_credit_closure_target", target_source).to(device), 0.0, 1.0)
    target_efficiency = torch.clamp(
        targets.get("portfolio_daily_allocation_resource_efficiency_target", target_receiver).to(device),
        0.0,
        1.0,
    )
    deploy_target = torch.clamp(
        targets.get("portfolio_daily_allocation_cash_deployment_target", target_net).to(device),
        0.0,
        1.0,
    )
    gross_exposure_target = torch.clamp(targets.get("gross_exposure_target", torch.full_like(receiver_score, 0.60)).to(device), 0.0, 1.0)
    turnover_budget_target = torch.clamp(targets.get("turnover_budget", torch.full_like(receiver_score, 0.36)).to(device), 0.04, 1.0)
    max_position_weight_target = torch.clamp(
        targets.get("max_position_weight_target", torch.full_like(receiver_score, 0.20)).to(device),
        0.04,
        0.40,
    )
    cash_timing_signal = torch.clamp(targets.get("budget_cash_timing_signal_target", target_cash).to(device), 0.0, 1.0)
    market_downside = torch.clamp(targets.get("market_downside_pressure", zero).to(device), 0.0, 1.0)
    cash_regime = torch.clamp(targets.get("cash_regime_pressure", zero).to(device), 0.0, 1.0)
    receiver_forward = torch.clamp(
        targets.get("portfolio_daily_receiver_forward_excess_5d", targets.get("forward_excess_5d", zero)).to(device),
        -0.35,
        0.35,
    )
    source_forward = torch.clamp(
        targets.get("portfolio_daily_source_forward_excess_5d", targets.get("forward_excess_5d", zero)).to(device),
        -0.35,
        0.35,
    )
    uncertainty = torch.clamp(targets.get("portfolio_daily_allocation_uncertainty_pressure_target", zero).to(device), 0.0, 1.0)
    tail = torch.clamp(targets.get("portfolio_daily_allocation_tail_risk_control_target", zero).to(device), 0.0, 1.0)
    drawdown = torch.clamp(targets.get("portfolio_daily_allocation_drawdown_control_target", zero).to(device), 0.0, 1.0)
    hard_negative = torch.clamp(
        torch.maximum(
            targets.get("portfolio_daily_source_hard_negative_penalty", zero).to(device),
            targets.get("portfolio_daily_source_tail_false_sell_penalty", zero).to(device),
        ),
        0.0,
        1.0,
    )
    positive_forward = torch.clamp(
        torch.maximum(
            targets.get("portfolio_daily_source_positive_forward_penalty", zero).to(device),
            torch.clamp(source_forward / 0.08, 0.0, 1.0),
        ),
        0.0,
        1.0,
    )
    opportunity = torch.clamp(targets.get("portfolio_daily_source_opportunity_cost_penalty", zero).to(device), 0.0, 1.0)
    release_preference = torch.clamp(targets.get("portfolio_daily_source_release_preference", target_source).to(device), 0.0, 1.0)
    spread_reward = torch.clamp(targets.get("portfolio_daily_receiver_source_spread_reward", zero).to(device), 0.0, 1.0)
    risk_pressure = torch.clamp(
        torch.maximum(torch.maximum(uncertainty, torch.maximum(tail, drawdown)), torch.maximum(market_downside, cash_regime)),
        0.0,
        1.0,
    )
    false_source_pressure = torch.clamp(
        torch.maximum(torch.maximum(hard_negative, positive_forward), 0.58 * opportunity + 0.42 * hard_negative),
        0.0,
        1.0,
    )
    date_code = targets["date_code"].to(device).long()

    term_values: dict[str, list[torch.Tensor]] = {name: [] for name in term_names if name != "total"}
    day_losses: list[torch.Tensor] = []
    for date_value in torch.unique(date_code):
        day_mask = date_code == date_value
        if int(day_mask.sum().detach().cpu().item()) <= 0:
            continue

        current_day = current_weight[day_mask]
        receiver_support_day = receiver_support[day_mask]
        source_support_day = source_support[day_mask]
        source_capacity = current_day * source_support_day
        day_position_cap = torch.clamp(max_position_weight_target[day_mask].mean(), 0.04, 0.40)
        receiver_capacity = torch.clamp(day_position_cap - current_day, min=0.0) * receiver_support_day
        day_turnover_limit = torch.clamp(turnover_budget_target[day_mask].mean(), 0.04, 1.0)
        day_gross_target = torch.clamp(gross_exposure_target[day_mask].mean(), 0.0, 1.0)
        current_gross = torch.clamp(current_day.sum(), 0.0, 1.0)
        current_cash = torch.clamp(1.0 - current_gross, 0.0, 1.0)
        day_risk = risk_pressure[day_mask].mean()
        day_cash_score = cash_score[day_mask].mean()
        day_cash_timing = cash_timing_signal[day_mask].mean()
        deploy_pressure = torch.clamp(
            0.30 * target_net[day_mask].mean()
            + 0.24 * target_credit[day_mask].mean()
            + 0.18 * target_efficiency[day_mask].mean()
            + 0.16 * deploy_target[day_mask].mean()
            + 0.12 * final_objective[day_mask].mean()
            + 0.14 * torch.relu(day_gross_target - current_gross)
            - 0.48 * day_risk,
            0.0,
            1.0,
        )
        cash_reserve_target = torch.clamp(
            0.04
            + 0.40 * day_risk
            + 0.18 * day_cash_timing
            + 0.14 * target_cash[day_mask].mean()
            - 0.22 * deploy_pressure,
            0.02,
            0.78,
        )
        receiver_quality = torch.clamp(
            0.34 * target_receiver[day_mask]
            + 0.22 * target_net[day_mask]
            + 0.16 * target_efficiency[day_mask]
            + 0.12 * spread_reward[day_mask]
            + 0.16 * torch.clamp(receiver_forward[day_mask] / 0.10, -1.0, 1.0)
            - 0.24 * risk_pressure[day_mask],
            0.0,
            1.0,
        )
        predicted_receiver_intensity = torch.clamp(
            0.38 * receiver_score[day_mask]
            + 0.20 * final_objective[day_mask]
            + 0.18 * net_utility[day_mask]
            + 0.14 * resource_efficiency[day_mask]
            + 0.10 * credit_closure[day_mask]
            - 0.20 * risk_pressure[day_mask],
            0.0,
            1.0,
        )
        clean_release_quality = torch.clamp(
            0.34 * target_source[day_mask]
            + 0.30 * release_preference[day_mask]
            + 0.22 * target_credit[day_mask]
            + 0.12 * target_efficiency[day_mask]
            - 0.66 * false_source_pressure[day_mask]
            - 0.14 * torch.clamp(source_forward[day_mask] / 0.08, 0.0, 1.0),
            0.0,
            1.0,
        )
        predicted_source_intensity = torch.clamp(
            0.42 * source_score[day_mask]
            + 0.22 * credit_closure[day_mask]
            + 0.16 * resource_efficiency[day_mask]
            + 0.12 * net_utility[day_mask]
            + 0.08 * release_preference[day_mask]
            - 0.66 * false_source_pressure[day_mask],
            0.0,
            1.0,
        )
        receiver_target_mass = (receiver_capacity * receiver_quality).sum()
        predicted_receiver_mass = (receiver_capacity * predicted_receiver_intensity).sum()
        source_target_mass = (source_capacity * clean_release_quality).sum()
        predicted_clean_source_supply = (source_capacity * predicted_source_intensity).sum()
        cash_release = (
            torch.relu(current_cash - cash_reserve_target)
            * torch.clamp(1.0 - day_cash_score, 0.0, 1.0)
            * torch.clamp(deploy_pressure - day_risk + 0.15, 0.0, 1.0)
        )
        desired_receiver_flow = torch.minimum(
            torch.minimum(receiver_target_mass, day_turnover_limit * 0.55),
            torch.clamp(receiver_capacity.sum(), 0.0, 1.0),
        )
        effective_receiver_flow = torch.minimum(
            torch.minimum(predicted_receiver_mass, day_turnover_limit * 0.55),
            torch.clamp(predicted_clean_source_supply + cash_release, 0.0, 1.0),
        )
        effective_source_flow = torch.minimum(predicted_clean_source_supply, day_turnover_limit * 0.55)
        risk_cash_need = torch.relu(cash_reserve_target - current_cash) * torch.clamp(day_risk + 0.15, 0.0, 1.0)
        source_need = torch.minimum(
            torch.clamp(source_target_mass + 0.16 * target_credit[day_mask].mean(), 0.0, 1.0),
            torch.clamp(desired_receiver_flow + risk_cash_need + 0.04, 0.0, 1.0),
        )
        receiver_demand_loss = torch.square(torch.relu(desired_receiver_flow - predicted_receiver_mass)) * torch.clamp(
            deploy_pressure + receiver_target_mass * 6.0,
            0.0,
            1.0,
        )
        funding_shortfall_loss = torch.square(
            torch.relu(torch.minimum(predicted_receiver_mass, desired_receiver_flow + 0.04) - predicted_clean_source_supply - cash_release)
        )
        source_dead_loss = torch.square(torch.relu(source_need - predicted_clean_source_supply)) * torch.clamp(
            deploy_pressure + source_target_mass * 8.0,
            0.0,
            1.0,
        )
        false_source_loss = (
            source_capacity
            * false_source_pressure[day_mask]
            * torch.square(source_score[day_mask] + 0.25 * credit_closure[day_mask])
        ).sum() / torch.clamp(source_capacity.sum(), min=1.0e-6)
        over_cash_loss = torch.square(day_cash_score) * torch.relu(deploy_pressure - day_risk - 0.04) * torch.clamp(
            current_cash + desired_receiver_flow,
            0.0,
            1.0,
        )
        cash_defense_loss = torch.square(torch.relu(cash_reserve_target - day_cash_score)) * torch.clamp(
            day_risk + day_cash_timing,
            0.0,
            1.0,
        )
        cash_timing_loss = 0.62 * over_cash_loss + 0.38 * cash_defense_loss
        cash_coherence_loss = torch.mean(torch.square(cash_score[day_mask] - day_cash_score)) * torch.clamp(
            0.50 + 0.25 * deploy_pressure + 0.25 * day_risk,
            0.0,
            1.0,
        )
        predicted_gross = torch.clamp(current_gross - effective_source_flow + effective_receiver_flow, 0.0, 1.0)
        exposure_gap_loss = (
            torch.square(torch.relu(day_gross_target - predicted_gross)) * torch.clamp(deploy_pressure - day_risk + 0.10, 0.0, 1.0)
            + 0.45 * torch.square(torch.relu(predicted_gross - day_gross_target - 0.06)) * torch.clamp(day_risk + 0.20, 0.0, 1.0)
        )
        source_excess_loss = torch.square(torch.relu(effective_source_flow - effective_receiver_flow - risk_cash_need - 0.02))
        flow_conservation_loss = torch.square(torch.relu(effective_receiver_flow - effective_source_flow - cash_release)) + 0.45 * source_excess_loss
        role_overlap_capacity = current_day * receiver_support_day * source_support_day
        role_overlap_loss = (
            (
                role_overlap_capacity
                * receiver_score[day_mask]
                * source_score[day_mask]
            ).sum()
            / torch.clamp(role_overlap_capacity.sum(), min=1.0e-6)
        ) * torch.clamp(source_need + desired_receiver_flow + 0.20, 0.0, 1.0)
        desired_source_breadth = torch.minimum(
            torch.tensor(3.0, device=device, dtype=receiver_score.dtype),
            torch.clamp(source_support_day.sum(), 0.0, 3.0),
        )
        soft_source_breadth = (source_support_day * torch.clamp(predicted_source_intensity / 0.36, 0.0, 1.0)).sum()
        source_breadth_loss = (
            torch.square(torch.relu(desired_source_breadth - soft_source_breadth))
            / torch.clamp(desired_source_breadth, min=1.0)
            * torch.clamp(source_need + 0.50 * deploy_pressure, 0.0, 1.0)
        )

        component_values = {
            "receiver_demand_loss": receiver_demand_loss,
            "funding_shortfall_loss": funding_shortfall_loss,
            "source_dead_loss": source_dead_loss,
            "false_source_loss": false_source_loss,
            "over_cash_loss": over_cash_loss,
            "cash_defense_loss": cash_defense_loss,
            "cash_timing_loss": cash_timing_loss,
            "cash_coherence_loss": cash_coherence_loss,
            "exposure_gap_loss": exposure_gap_loss,
            "flow_conservation_loss": flow_conservation_loss,
            "role_overlap_loss": role_overlap_loss,
            "source_breadth_loss": source_breadth_loss,
            "receiver_demand_mean": desired_receiver_flow,
            "desired_receiver_flow_mean": desired_receiver_flow,
            "effective_receiver_flow_mean": effective_receiver_flow,
            "effective_source_flow_mean": effective_source_flow,
            "source_need_mean": source_need,
            "risk_cash_need_mean": risk_cash_need,
            "clean_source_supply_mean": predicted_clean_source_supply,
            "cash_release_mean": cash_release,
            "cash_defense_mean": cash_reserve_target,
            "current_cash_mean": current_cash,
            "predicted_gross_mean": predicted_gross,
            "deploy_pressure_mean": deploy_pressure,
            "risk_pressure_mean": day_risk,
        }
        for name, value in component_values.items():
            term_values[name].append(value)
        day_losses.append(
            0.22 * receiver_demand_loss
            + 0.34 * funding_shortfall_loss
            + 0.34 * source_dead_loss
            + 0.30 * false_source_loss
            + 0.22 * over_cash_loss
            + 0.22 * cash_timing_loss
            + 0.16 * cash_coherence_loss
            + 0.24 * exposure_gap_loss
            + 0.24 * flow_conservation_loss
            + 0.24 * role_overlap_loss
            + 0.20 * source_breadth_loss
        )

    if not day_losses:
        return _zero_result()
    total = torch.stack(day_losses).mean()
    if return_terms:
        terms: dict[str, torch.Tensor] = {
            name: (torch.stack(values).mean() if values else torch.tensor(0.0, device=device))
            for name, values in term_values.items()
        }
        terms["total"] = total
        return terms
    return total


def _source_hard_negative_tail_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    device = next(iter(outputs.values())).device
    if "portfolio_daily_unified_source_score" not in outputs or "portfolio_daily_source_candidate_mask" not in targets:
        return torch.tensor(0.0, device=device)
    source_score = torch.clamp(outputs["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    zero = torch.zeros_like(source_score)
    source_mask = torch.clamp(targets["portfolio_daily_source_candidate_mask"].to(device), 0.0, 1.0)
    hard_negative = torch.clamp(targets.get("portfolio_daily_source_hard_negative_penalty", zero).to(device), 0.0, 1.0)
    strong_false = torch.clamp(targets.get("portfolio_daily_source_strong_false_sell_penalty", zero).to(device), 0.0, 1.0)
    tail_false = torch.clamp(targets.get("portfolio_daily_source_tail_false_sell_penalty", zero).to(device), 0.0, 1.0)
    positive_forward = torch.clamp(targets.get("portfolio_daily_source_positive_forward_penalty", zero).to(device), 0.0, 1.0)
    opportunity = torch.clamp(targets.get("portfolio_daily_source_opportunity_cost_penalty", zero).to(device), 0.0, 1.0)
    source_forward = torch.clamp(targets.get("portfolio_daily_source_forward_excess_5d", zero).to(device), -0.35, 0.35)
    release_preference = torch.clamp(
        targets.get("portfolio_daily_source_release_preference", 1.0 - hard_negative).to(device),
        0.0,
        1.0,
    )
    forward_tail = torch.clamp((source_forward - 0.04) / 0.08, 0.0, 1.0)
    false_sell_pressure = torch.clamp(
        torch.maximum(
            torch.maximum(hard_negative, strong_false),
            torch.maximum(tail_false, torch.maximum(positive_forward, forward_tail)),
        )
        + 0.20 * opportunity,
        0.0,
        1.0,
    ) * source_mask
    clean_release_weight = torch.clamp((1.0 - false_sell_pressure) * source_mask, 0.0, 1.0)
    false_source_loss = (false_sell_pressure * torch.square(source_score)).mean()
    clean_release_regret = (clean_release_weight * torch.relu(release_preference - source_score)).mean()
    head_terms: list[torch.Tensor] = []
    for name, target in (
        ("portfolio_daily_source_hard_negative_penalty", hard_negative),
        ("portfolio_daily_source_tail_false_sell_penalty", tail_false),
        ("portfolio_daily_source_release_preference", release_preference),
    ):
        if name in outputs:
            head_terms.append(nn.functional.smooth_l1_loss(torch.clamp(outputs[name].to(device), 0.0, 1.0), target))
    head_loss = torch.stack(head_terms).mean() if head_terms else torch.tensor(0.0, device=device)
    margin = torch.relu(source_score - torch.clamp(release_preference - false_sell_pressure * 0.65, 0.0, 1.0))
    margin_loss = (false_sell_pressure * margin).mean()
    return false_source_loss * 0.46 + clean_release_regret * 0.24 + head_loss * 0.20 + margin_loss * 0.10


def _source_listwise_release_regret_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    pairwise_loss = _date_rank_pairwise_loss(
        outputs,
        targets,
        score_name="portfolio_daily_unified_source_score",
        target_name="portfolio_daily_source_release_preference",
        candidate_mask_name="portfolio_daily_source_candidate_mask",
        min_target_gap=0.06,
    )
    device = next(iter(outputs.values())).device
    if "portfolio_daily_unified_source_score" not in outputs or "portfolio_daily_source_release_preference" not in targets:
        return pairwise_loss
    source_score = torch.clamp(outputs["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    release_preference = torch.clamp(targets["portfolio_daily_source_release_preference"].to(device), 0.0, 1.0)
    mask = torch.clamp(targets.get("portfolio_daily_source_candidate_mask", torch.ones_like(source_score)).to(device), 0.0, 1.0)
    hard_negative = torch.clamp(targets.get("portfolio_daily_source_hard_negative_penalty", torch.zeros_like(source_score)).to(device), 0.0, 1.0)
    clean_regression = (mask * (1.0 - hard_negative) * nn.functional.smooth_l1_loss(source_score, release_preference, reduction="none")).mean()
    hard_negative_margin = (mask * hard_negative * torch.square(source_score)).mean()
    return pairwise_loss * 0.62 + clean_regression * 0.22 + hard_negative_margin * 0.16


def _transfer_level_allocation_regret_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    required_outputs = {
        "portfolio_daily_unified_receiver_score",
        "portfolio_daily_unified_source_score",
        "portfolio_daily_unified_cash_score",
        "portfolio_daily_unified_allocation_objective",
    }
    device = next(iter(outputs.values())).device
    if not required_outputs.issubset(outputs) or "portfolio_daily_transfer_regret_target" not in targets:
        return torch.tensor(0.0, device=device)
    receiver_score = torch.clamp(outputs["portfolio_daily_unified_receiver_score"].to(device), 0.0, 1.0)
    source_score = torch.clamp(outputs["portfolio_daily_unified_source_score"].to(device), 0.0, 1.0)
    cash_score = torch.clamp(outputs["portfolio_daily_unified_cash_score"].to(device), 0.0, 1.0)
    objective_score = torch.clamp(outputs["portfolio_daily_unified_allocation_objective"].to(device), 0.0, 1.0)
    zero = torch.zeros_like(objective_score)
    transfer_target = torch.clamp(targets["portfolio_daily_transfer_regret_target"].to(device), 0.0, 1.0)
    target_receiver = torch.clamp(targets.get("portfolio_daily_unified_receiver_score", zero).to(device), 0.0, 1.0)
    target_source = torch.clamp(targets.get("portfolio_daily_unified_source_score", zero).to(device), 0.0, 1.0)
    target_cash = torch.clamp(targets.get("portfolio_daily_unified_cash_score", zero).to(device), 0.0, 1.0)
    receiver_mask = torch.clamp(targets.get("portfolio_daily_receiver_candidate_mask", zero).to(device), 0.0, 1.0)
    source_mask = torch.clamp(targets.get("portfolio_daily_source_candidate_mask", zero).to(device), 0.0, 1.0)
    spread_reward = torch.clamp(targets.get("portfolio_daily_receiver_source_spread_reward", zero).to(device), 0.0, 1.0)
    release_preference = torch.clamp(targets.get("portfolio_daily_source_release_preference", target_source).to(device), 0.0, 1.0)
    hard_negative = torch.clamp(targets.get("portfolio_daily_source_hard_negative_penalty", zero).to(device), 0.0, 1.0)
    tail_false = torch.clamp(targets.get("portfolio_daily_source_tail_false_sell_penalty", zero).to(device), 0.0, 1.0)
    positive_forward = torch.clamp(targets.get("portfolio_daily_source_positive_forward_penalty", zero).to(device), 0.0, 1.0)
    opportunity = torch.clamp(targets.get("portfolio_daily_source_opportunity_cost_penalty", zero).to(device), 0.0, 1.0)
    market_downside = torch.clamp(targets.get("market_downside_pressure", zero).to(device), 0.0, 1.0)
    cash_regime = torch.clamp(targets.get("cash_regime_pressure", zero).to(device), 0.0, 1.0)

    false_source_pressure = torch.clamp(
        torch.maximum(torch.maximum(hard_negative, tail_false), 0.66 * positive_forward + 0.34 * opportunity)
        * source_mask,
        0.0,
        1.0,
    )
    receiver_transfer_oracle = torch.clamp(
        (0.58 * target_receiver + 0.24 * spread_reward + 0.18 * transfer_target) * receiver_mask,
        0.0,
        1.0,
    )
    source_transfer_oracle = torch.clamp(
        (0.54 * release_preference + 0.22 * target_source + 0.18 * transfer_target + 0.06 * spread_reward)
        * source_mask
        * (1.0 - false_source_pressure),
        0.0,
        1.0,
    )
    risk_off = torch.clamp(0.56 * market_downside + 0.44 * cash_regime, 0.0, 1.0)
    dead_cash_pressure = torch.clamp(
        receiver_transfer_oracle + 0.60 * source_transfer_oracle - risk_off + 0.06,
        0.0,
        1.0,
    )
    objective_oracle = torch.clamp(
        0.40 * transfer_target
        + 0.24 * receiver_transfer_oracle
        + 0.22 * source_transfer_oracle
        + 0.14 * target_cash * risk_off
        - 0.30 * false_source_pressure,
        0.0,
        1.0,
    )

    objective_regret = nn.functional.smooth_l1_loss(objective_score, objective_oracle)
    transfer_head_regret = (
        nn.functional.smooth_l1_loss(torch.clamp(outputs["portfolio_daily_transfer_regret_target"].to(device), 0.0, 1.0), transfer_target)
        if "portfolio_daily_transfer_regret_target" in outputs
        else torch.tensor(0.0, device=device)
    )
    receiver_regret = (torch.relu(receiver_transfer_oracle - receiver_score) * torch.clamp(receiver_mask + 0.10, 0.0, 1.0)).mean()
    source_regret = (torch.relu(source_transfer_oracle - source_score) * torch.clamp(source_mask + 0.10, 0.0, 1.0)).mean()
    false_source_loss = (false_source_pressure * (source_score + 0.30 * objective_score)).mean()
    dead_cash_loss = (dead_cash_pressure * cash_score).mean()
    return (
        objective_regret * 0.22
        + transfer_head_regret * 0.14
        + receiver_regret * 0.18
        + source_regret * 0.18
        + false_source_loss * 0.18
        + dead_cash_loss * 0.10
    )


def _value_arbitration_consistency_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    device = outputs["action_logits"].device
    required_targets = {"alpha_opportunity_value", "cash_defense_value", "deployment_opportunity_cost", "value_arbitration_target"}
    if not required_targets.issubset(targets):
        return torch.tensor(0.0, device=device)
    probs = torch.softmax(outputs["action_logits"], dim=-1)
    action_lookup = {name: idx for idx, name in enumerate(ACTION_CLASSES)}
    deploy_prob = probs[:, action_lookup["open"]] + probs[:, action_lookup["add"]]
    sell_prob = probs[:, action_lookup["reduce"]] + probs[:, action_lookup["exit"]]
    keep_prob = probs[:, action_lookup["hold"]] + probs[:, action_lookup["add"]]
    cash_prob = probs[:, action_lookup["skip"]]
    alpha_value = torch.clamp(targets["alpha_opportunity_value"].to(device), 0.0, 1.0)
    deployment_value = torch.clamp(targets["deployment_opportunity_cost"].to(device), 0.0, 1.0)
    cash_value = torch.clamp(targets["cash_defense_value"].to(device), 0.0, 1.0)
    value_target = torch.clamp(targets["value_arbitration_target"].to(device), 0.0, 1.0)
    sell_value = torch.clamp(targets.get("sell_release_value", torch.zeros_like(alpha_value)).to(device), 0.0, 1.0)
    hold_value = torch.clamp(targets.get("hold_continuation_value", torch.zeros_like(alpha_value)).to(device), 0.0, 1.0)
    held = torch.clamp(targets.get("holding_flag_target", torch.zeros_like(alpha_value)).to(device), 0.0, 1.0)
    capital_bias = torch.clamp((value_target - 0.50) * 2.0, 0.0, 1.0)
    defense_bias = torch.clamp((0.50 - value_target) * 2.0, 0.0, 1.0)
    deploy_target = torch.clamp(0.42 * alpha_value + 0.34 * deployment_value + 0.24 * capital_bias, 0.0, 1.0)
    keep_target = torch.clamp(0.52 * hold_value + 0.28 * alpha_value + 0.20 * capital_bias, 0.0, 1.0) * held
    sell_target = torch.clamp(0.50 * sell_value + 0.26 * cash_value + 0.24 * defense_bias, 0.0, 1.0) * held
    cash_target = torch.clamp(0.50 * cash_value + 0.28 * (1.0 - alpha_value) + 0.22 * defense_bias, 0.0, 1.0) * (1.0 - held)
    eps = 1.0e-4
    return (
        nn.functional.binary_cross_entropy(torch.clamp(deploy_prob, eps, 1.0 - eps), deploy_target) * 0.34
        + nn.functional.binary_cross_entropy(torch.clamp(keep_prob, eps, 1.0 - eps), keep_target) * 0.18
        + nn.functional.binary_cross_entropy(torch.clamp(sell_prob, eps, 1.0 - eps), sell_target) * 0.26
        + nn.functional.binary_cross_entropy(torch.clamp(cash_prob, eps, 1.0 - eps), cash_target) * 0.22
    )


def _three_value_gate_consistency_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    device = outputs["action_logits"].device
    required_targets = {
        "deploy_value_target",
        "release_value_target",
        "defense_value_target",
        "deploy_gate_target",
        "release_gate_target",
        "defense_gate_target",
    }
    if not required_targets.issubset(targets):
        return torch.tensor(0.0, device=device)
    probs = torch.softmax(outputs["action_logits"], dim=-1)
    action_lookup = {name: idx for idx, name in enumerate(ACTION_CLASSES)}
    deploy_prob = probs[:, action_lookup["open"]] + probs[:, action_lookup["add"]]
    keep_prob = probs[:, action_lookup["hold"]] + probs[:, action_lookup["add"]]
    release_prob = probs[:, action_lookup["reduce"]] + probs[:, action_lookup["exit"]]
    defense_prob = probs[:, action_lookup["skip"]]
    deploy_value = torch.clamp(targets["deploy_value_target"].to(device), 0.0, 1.0)
    release_value = torch.clamp(targets["release_value_target"].to(device), 0.0, 1.0)
    defense_value = torch.clamp(targets["defense_value_target"].to(device), 0.0, 1.0)
    deploy_gate = torch.clamp(targets["deploy_gate_target"].to(device), 0.0, 1.0)
    release_gate = torch.clamp(targets["release_gate_target"].to(device), 0.0, 1.0)
    defense_gate = torch.clamp(targets["defense_gate_target"].to(device), 0.0, 1.0)
    hold_value = torch.clamp(targets.get("hold_continuation_value", torch.zeros_like(deploy_value)).to(device), 0.0, 1.0)
    alpha_value = torch.clamp(targets.get("alpha_opportunity_value", torch.zeros_like(deploy_value)).to(device), 0.0, 1.0)
    held = torch.clamp(targets.get("holding_flag_target", torch.zeros_like(deploy_value)).to(device), 0.0, 1.0)
    deploy_target = torch.clamp(0.56 * deploy_gate + 0.30 * deploy_value + 0.14 * alpha_value, 0.0, 1.0)
    keep_target = torch.clamp(0.42 * deploy_gate + 0.36 * hold_value + 0.22 * deploy_value, 0.0, 1.0) * held
    release_target = torch.clamp(0.56 * release_gate + 0.32 * release_value + 0.12 * defense_value, 0.0, 1.0) * held
    defense_target = torch.clamp(0.58 * defense_gate + 0.30 * defense_value + 0.12 * torch.clamp(1.0 - deploy_value, 0.0, 1.0), 0.0, 1.0)
    eps = 1.0e-4
    bce_loss = (
        nn.functional.binary_cross_entropy(torch.clamp(deploy_prob, eps, 1.0 - eps), deploy_target) * 0.34
        + nn.functional.binary_cross_entropy(torch.clamp(keep_prob, eps, 1.0 - eps), keep_target) * 0.18
        + nn.functional.binary_cross_entropy(torch.clamp(release_prob, eps, 1.0 - eps), release_target) * 0.24
        + nn.functional.binary_cross_entropy(torch.clamp(defense_prob, eps, 1.0 - eps), defense_target) * 0.24
    )
    deploy_dominant = deploy_gate > torch.maximum(release_gate, defense_gate) + 0.08
    release_dominant = (release_gate > deploy_gate + 0.08) & (release_gate >= defense_gate)
    defense_dominant = defense_gate > torch.maximum(deploy_gate, release_gate) + 0.08
    margin = torch.tensor(0.08, device=device)
    margin_terms: list[torch.Tensor] = []
    if int(deploy_dominant.sum().detach().cpu().item()) > 0:
        margin_terms.append(torch.relu(margin - (deploy_prob[deploy_dominant] - torch.maximum(release_prob[deploy_dominant], defense_prob[deploy_dominant]))).mean())
    if int(release_dominant.sum().detach().cpu().item()) > 0:
        margin_terms.append(torch.relu(margin - (release_prob[release_dominant] - keep_prob[release_dominant])).mean())
    if int(defense_dominant.sum().detach().cpu().item()) > 0:
        margin_terms.append(torch.relu(margin - (defense_prob[defense_dominant] - deploy_prob[defense_dominant])).mean())
    if margin_terms:
        return bce_loss + torch.stack(margin_terms).mean() * 0.28
    return bce_loss


def _hierarchical_three_value_gate_consistency_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    device = outputs["action_logits"].device
    required_targets = {
        "deploy_value_target",
        "release_value_target",
        "defense_value_target",
        "deploy_gate_target",
        "release_gate_target",
        "defense_gate_target",
    }
    if not required_targets.issubset(targets):
        return torch.tensor(0.0, device=device)
    probs = torch.softmax(outputs["action_logits"], dim=-1)
    action_lookup = {name: idx for idx, name in enumerate(ACTION_CLASSES)}
    deploy_prob = probs[:, action_lookup["open"]] + probs[:, action_lookup["add"]]
    keep_prob = probs[:, action_lookup["hold"]] + probs[:, action_lookup["add"]]
    release_prob = probs[:, action_lookup["reduce"]] + probs[:, action_lookup["exit"]]
    defense_prob = probs[:, action_lookup["skip"]]
    deploy_value = torch.clamp(targets["deploy_value_target"].to(device), 0.0, 1.0)
    release_value = torch.clamp(targets["release_value_target"].to(device), 0.0, 1.0)
    defense_value = torch.clamp(targets["defense_value_target"].to(device), 0.0, 1.0)
    deploy_gate = torch.clamp(targets["deploy_gate_target"].to(device), 0.0, 1.0)
    release_gate = torch.clamp(targets["release_gate_target"].to(device), 0.0, 1.0)
    defense_gate = torch.clamp(targets["defense_gate_target"].to(device), 0.0, 1.0)
    hold_value = torch.clamp(targets.get("hold_continuation_value", torch.zeros_like(deploy_value)).to(device), 0.0, 1.0)
    alpha_value = torch.clamp(targets.get("alpha_opportunity_value", torch.zeros_like(deploy_value)).to(device), 0.0, 1.0)
    sell_release_value = torch.clamp(targets.get("sell_release_value", torch.zeros_like(deploy_value)).to(device), 0.0, 1.0)
    held = torch.clamp(targets.get("holding_flag_target", torch.zeros_like(deploy_value)).to(device), 0.0, 1.0)
    deploy_target = torch.clamp(0.58 * deploy_gate + 0.30 * deploy_value + 0.12 * alpha_value, 0.0, 1.0)
    keep_target = torch.clamp(0.46 * deploy_gate + 0.34 * hold_value + 0.20 * deploy_value, 0.0, 1.0) * held
    release_target = torch.clamp(0.58 * release_gate + 0.28 * release_value + 0.14 * sell_release_value, 0.0, 1.0) * held
    defense_target = torch.clamp(0.58 * defense_value + 0.22 * defense_gate + 0.20 * torch.clamp(1.0 - deploy_value, 0.0, 1.0), 0.0, 1.0) * (1.0 - held)
    eps = 1.0e-4
    bce_loss = (
        nn.functional.binary_cross_entropy(torch.clamp(deploy_prob, eps, 1.0 - eps), deploy_target) * 0.38
        + nn.functional.binary_cross_entropy(torch.clamp(keep_prob, eps, 1.0 - eps), keep_target) * 0.22
        + nn.functional.binary_cross_entropy(torch.clamp(release_prob, eps, 1.0 - eps), release_target) * 0.28
        + nn.functional.binary_cross_entropy(torch.clamp(defense_prob, eps, 1.0 - eps), defense_target) * 0.12
    )
    deploy_dominant = deploy_gate > release_gate + 0.08
    release_dominant = release_gate > deploy_gate + 0.08
    margin = torch.tensor(0.08, device=device)
    margin_terms: list[torch.Tensor] = []
    if int(deploy_dominant.sum().detach().cpu().item()) > 0:
        margin_terms.append(
            torch.relu(margin - (deploy_prob[deploy_dominant] - release_prob[deploy_dominant])).mean()
        )
    if int(release_dominant.sum().detach().cpu().item()) > 0:
        margin_terms.append(
            torch.relu(margin - (release_prob[release_dominant] - keep_prob[release_dominant])).mean()
        )
    if margin_terms:
        return bce_loss + torch.stack(margin_terms).mean() * 0.24
    return bce_loss


def _funding_release_discipline_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    variant: str = "v10",
) -> torch.Tensor:
    device = outputs["action_logits"].device
    required_targets = {
        "hold_continuation_value",
        "alpha_opportunity_value",
        "sell_release_value",
        "cash_defense_value",
        "deploy_value_target",
        "release_value_target",
        "deploy_gate_target",
        "release_gate_target",
        "holding_flag_target",
    }
    if not required_targets.issubset(targets):
        return torch.tensor(0.0, device=device)
    probs = torch.softmax(outputs["action_logits"], dim=-1)
    action_lookup = {name: idx for idx, name in enumerate(ACTION_CLASSES)}
    keep_prob = probs[:, action_lookup["hold"]] + probs[:, action_lookup["add"]]
    release_prob = probs[:, action_lookup["reduce"]] + probs[:, action_lookup["exit"]]
    hold_value = torch.clamp(targets["hold_continuation_value"].to(device), 0.0, 1.0)
    alpha_value = torch.clamp(targets["alpha_opportunity_value"].to(device), 0.0, 1.0)
    sell_release_value = torch.clamp(targets["sell_release_value"].to(device), 0.0, 1.0)
    cash_defense_value = torch.clamp(targets["cash_defense_value"].to(device), 0.0, 1.0)
    deploy_value = torch.clamp(targets["deploy_value_target"].to(device), 0.0, 1.0)
    release_value = torch.clamp(targets["release_value_target"].to(device), 0.0, 1.0)
    deploy_gate = torch.clamp(targets["deploy_gate_target"].to(device), 0.0, 1.0)
    release_gate = torch.clamp(targets["release_gate_target"].to(device), 0.0, 1.0)
    deploy_executability = torch.clamp(
        targets.get("deploy_executability_target", torch.zeros_like(hold_value)).to(device),
        0.0,
        1.0,
    )
    held = torch.clamp(targets["holding_flag_target"].to(device), 0.0, 1.0)
    if variant == "v11":
        disciplined_funding_need = torch.clamp(
            0.48 * deploy_executability
            + 0.24 * deploy_gate
            + 0.16 * deploy_value
            + 0.06 * alpha_value
            - 0.16 * hold_value
            - 0.08 * release_value,
            0.0,
            1.0,
        ) * held
        protected_keep_target = torch.clamp(
            0.34 * hold_value
            + 0.18 * alpha_value
            + 0.12 * deploy_value
            + 0.08 * deploy_gate
            + 0.06 * deploy_executability
            - 0.20 * release_value
            - 0.18 * release_gate
            - 0.14 * sell_release_value
            - 0.08 * cash_defense_value,
            0.0,
            1.0,
        ) * held
        funding_release_target = torch.clamp(
            0.28 * release_value
            + 0.22 * release_gate
            + 0.18 * sell_release_value
            + 0.10 * cash_defense_value
            + 0.36 * disciplined_funding_need
            - 0.18 * hold_value
            - 0.08 * alpha_value,
            0.0,
            1.0,
        ) * held
        keep_weight = 1.0 + 0.75 * torch.relu(protected_keep_target - funding_release_target)
        release_weight = (
            1.0
            + 1.15 * torch.relu(funding_release_target - protected_keep_target)
            + 0.90 * disciplined_funding_need
        )
        dominant_gap = 0.08
        margin = torch.tensor(0.12, device=device)
        margin_scale = 0.36
        keep_scale = 0.46
        release_scale = 0.54
    else:
        protected_keep_target = torch.clamp(
            0.34 * hold_value
            + 0.18 * alpha_value
            + 0.14 * deploy_value
            + 0.12 * deploy_gate
            + 0.12 * deploy_executability
            - 0.22 * release_value
            - 0.20 * release_gate
            - 0.14 * sell_release_value
            - 0.08 * cash_defense_value,
            0.0,
            1.0,
        ) * held
        funding_release_target = torch.clamp(
            0.34 * release_value
            + 0.24 * release_gate
            + 0.16 * sell_release_value
            + 0.08 * cash_defense_value
            - 0.24 * hold_value
            - 0.14 * alpha_value
            - 0.12 * deploy_gate
            - 0.10 * deploy_executability,
            0.0,
            1.0,
        ) * held
        keep_weight = torch.ones_like(protected_keep_target)
        release_weight = torch.ones_like(funding_release_target)
        dominant_gap = 0.12
        margin = torch.tensor(0.10, device=device)
        margin_scale = 0.26
        keep_scale = 0.52
        release_scale = 0.48
    eps = 1.0e-4
    bce_loss = (
        nn.functional.binary_cross_entropy(
            torch.clamp(keep_prob, eps, 1.0 - eps),
            protected_keep_target,
            weight=keep_weight,
        ) * keep_scale
        + nn.functional.binary_cross_entropy(
            torch.clamp(release_prob, eps, 1.0 - eps),
            funding_release_target,
            weight=release_weight,
        ) * release_scale
    )
    protected_dominant = protected_keep_target > funding_release_target + dominant_gap
    release_dominant = funding_release_target > protected_keep_target + dominant_gap
    margin_terms: list[torch.Tensor] = []
    if int(protected_dominant.sum().detach().cpu().item()) > 0:
        margin_terms.append(
            torch.relu(margin - (keep_prob[protected_dominant] - release_prob[protected_dominant])).mean()
        )
    if int(release_dominant.sum().detach().cpu().item()) > 0:
        margin_terms.append(
            torch.relu(margin - (release_prob[release_dominant] - keep_prob[release_dominant])).mean()
        )
    if margin_terms:
        return bce_loss + torch.stack(margin_terms).mean() * margin_scale
    return bce_loss


def _action_value_consistency_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    device = outputs["action_logits"].device
    required_targets = {
        "open_action_value",
        "add_action_value",
        "hold_action_value",
        "reduce_action_value",
        "exit_action_value",
        "action_value_consistency_target",
        "holding_flag_target",
    }
    if not required_targets.issubset(targets):
        return torch.tensor(0.0, device=device)
    probs = torch.softmax(outputs["action_logits"], dim=-1)
    action_lookup = {name: idx for idx, name in enumerate(ACTION_CLASSES)}
    action_names = ("open", "add", "hold", "reduce", "exit")
    predicted_probs = torch.stack([probs[:, action_lookup[name]] for name in action_names], dim=1)
    target_values = torch.stack(
        [
            torch.clamp(targets[f"{name}_action_value"].to(device), 0.0, 1.0)
            for name in action_names
        ],
        dim=1,
    )
    held = torch.clamp(targets["holding_flag_target"].to(device), 0.0, 1.0)
    open_mask = (1.0 - held).unsqueeze(1)
    held_mask = held.unsqueeze(1)
    relevance = torch.cat(
        [
            open_mask,
            held_mask,
            held_mask,
            held_mask,
            held_mask,
        ],
        dim=1,
    )
    eps = 1.0e-4
    bce = nn.functional.binary_cross_entropy(
        torch.clamp(predicted_probs, eps, 1.0 - eps),
        target_values,
        weight=1.0 + relevance * 1.25,
        reduction="none",
    )
    bce_loss = bce.mean()

    keep_value = torch.maximum(target_values[:, 1], target_values[:, 2])
    release_value = torch.maximum(target_values[:, 3], target_values[:, 4])
    keep_prob = probs[:, action_lookup["add"]] + probs[:, action_lookup["hold"]]
    release_prob = probs[:, action_lookup["reduce"]] + probs[:, action_lookup["exit"]]
    open_prob = probs[:, action_lookup["open"]]
    skip_prob = probs[:, action_lookup["skip"]]
    consistency_target = torch.clamp(targets["action_value_consistency_target"].to(device), 0.0, 1.0)

    keep_dominant = (held > 0.5) & (keep_value > release_value + 0.08)
    release_dominant = (held > 0.5) & (release_value > keep_value + 0.08)
    open_dominant = (held <= 0.5) & (target_values[:, 0] > 0.52)
    skip_dominant = (held <= 0.5) & (target_values[:, 0] < 0.28) & (consistency_target < 0.46)
    margin = torch.tensor(0.10, device=device)
    margin_terms: list[torch.Tensor] = []
    if int(keep_dominant.sum().detach().cpu().item()) > 0:
        margin_terms.append(torch.relu(margin - (keep_prob[keep_dominant] - release_prob[keep_dominant])).mean())
    if int(release_dominant.sum().detach().cpu().item()) > 0:
        margin_terms.append(torch.relu(margin - (release_prob[release_dominant] - keep_prob[release_dominant])).mean())
    if int(open_dominant.sum().detach().cpu().item()) > 0:
        margin_terms.append(torch.relu(margin - (open_prob[open_dominant] - skip_prob[open_dominant])).mean())
    if int(skip_dominant.sum().detach().cpu().item()) > 0:
        margin_terms.append(torch.relu(margin - (skip_prob[skip_dominant] - open_prob[skip_dominant])).mean())
    if margin_terms:
        return bce_loss + torch.stack(margin_terms).mean() * 0.34
    return bce_loss


def _build_action_soft_targets(sample_frame: pd.DataFrame) -> np.ndarray:
    row_count = int(len(sample_frame))
    soft_targets = np.full((row_count, len(ACTION_CLASSES)), 1.0e-4, dtype=np.float32)
    held_mask = (
        sample_frame.get("holding_flag", pd.Series(np.zeros(row_count), index=sample_frame.index)).astype(float).to_numpy(dtype=float) > 0.5
    )
    current_weight = sample_frame.get("current_weight", pd.Series(np.zeros(row_count), index=sample_frame.index)).astype(float).to_numpy(dtype=float)
    hold_days = sample_frame.get("hold_days", pd.Series(np.zeros(row_count), index=sample_frame.index)).astype(float).to_numpy(dtype=float)
    drawdown = sample_frame.get("drawdown_from_peak", pd.Series(np.zeros(row_count), index=sample_frame.index)).astype(float).to_numpy(dtype=float)
    market_downside = sample_frame.get("market_downside_pressure", pd.Series(np.zeros(row_count), index=sample_frame.index)).astype(float).to_numpy(dtype=float)
    signal_decay = sample_frame.get("signal_decay_speed", pd.Series(np.zeros(row_count), index=sample_frame.index)).astype(float).to_numpy(dtype=float)
    duration_ratio = np.clip(
        sample_frame["planned_holding_days"].astype(float).to_numpy(dtype=np.float32) / float(MAX_CONTINUOUS_HOLDING_DAYS),
        0.0,
        1.0,
    )
    delta_hint = sample_frame["target_delta_hint"].astype(float).to_numpy(dtype=np.float32)
    entry_quality = np.clip(sample_frame["entry_quality"].astype(float).to_numpy(dtype=np.float32), 0.0, None)
    hold_quality = np.clip(sample_frame["hold_quality"].astype(float).to_numpy(dtype=np.float32), 0.0, None)
    add_quality = np.clip(sample_frame["add_quality"].astype(float).to_numpy(dtype=np.float32), 0.0, None)
    reduce_quality = np.clip(sample_frame["reduce_quality"].astype(float).to_numpy(dtype=np.float32), 0.0, None)
    exit_urgency = np.clip(sample_frame["exit_urgency"].astype(float).to_numpy(dtype=np.float32), 0.0, None)
    reentry_readiness = np.clip(sample_frame["reentry_readiness"].astype(float).to_numpy(dtype=np.float32), 0.0, None)
    reduce_fraction = np.clip(
        sample_frame.get("reduce_fraction_target", pd.Series(np.zeros(row_count), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    exit_hazard = np.clip(
        sample_frame.get("exit_hazard_target", pd.Series(np.zeros(row_count), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    sell_attribution = np.clip(
        sample_frame.get("sell_attribution_score", pd.Series(np.zeros(row_count), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    portfolio_source_candidate = np.clip(
        sample_frame.get(
            "portfolio_daily_source_candidate_mask",
            pd.Series(np.zeros(row_count), index=sample_frame.index),
        ).astype(float).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    portfolio_source_score = np.clip(
        sample_frame.get("portfolio_daily_source_score", pd.Series(np.zeros(row_count), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    portfolio_source_release_quality = np.clip(
        sample_frame.get("portfolio_daily_source_release_quality", pd.Series(np.zeros(row_count), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    portfolio_source_executability = np.clip(
        sample_frame.get("portfolio_daily_source_executability", pd.Series(np.zeros(row_count), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    portfolio_source_release_capacity = np.clip(
        sample_frame.get("portfolio_daily_source_release_capacity", pd.Series(np.zeros(row_count), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    portfolio_source_economic_release = np.clip(
        sample_frame.get("portfolio_daily_source_economic_release_score", pd.Series(np.zeros(row_count), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    portfolio_source_opportunity_cost = np.clip(
        sample_frame.get("portfolio_daily_source_opportunity_cost", pd.Series(np.zeros(row_count), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    portfolio_source_economic_block = np.clip(
        sample_frame.get("portfolio_daily_source_economic_block_risk", pd.Series(np.zeros(row_count), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    portfolio_source_forward_brake = np.clip(
        sample_frame.get("portfolio_daily_source_forward_strength_brake_risk", pd.Series(np.zeros(row_count), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    portfolio_source_proxy_keep_risk = np.clip(
        sample_frame.get("portfolio_daily_source_forward_proxy_keep_risk", pd.Series(np.zeros(row_count), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    portfolio_source_release_intent = np.clip(
        0.30 * portfolio_source_candidate
        + 0.22 * portfolio_source_score
        + 0.16 * portfolio_source_release_quality
        + 0.12 * portfolio_source_executability
        + 0.10 * portfolio_source_release_capacity
        + 0.12 * portfolio_source_economic_release
        - 0.18 * portfolio_source_opportunity_cost
        - 0.16 * portfolio_source_economic_block
        - 0.14 * portfolio_source_forward_brake
        - 0.10 * portfolio_source_proxy_keep_risk,
        0.0,
        1.0,
    )
    sell_pressure = np.clip(0.58 * reduce_fraction + 0.42 * exit_hazard, 0.0, 1.0)
    action_lookup = {name: idx for idx, name in enumerate(ACTION_CLASSES)}

    for idx in range(row_count):
        held = bool(held_mask[idx] or current_weight[idx] > 1.0e-8)
        scores = soft_targets[idx]
        if held:
            scores[action_lookup["hold"]] = (
                0.22
                + hold_quality[idx] * 1.10
                + duration_ratio[idx] * 0.30
                + max(float(delta_hint[idx]), 0.0) * 0.12
            )
            scores[action_lookup["add"]] = (
                0.06
                + add_quality[idx] * 0.95
                + max(float(delta_hint[idx]), 0.0) * 0.12
                + max(0.10 - current_weight[idx], 0.0) * 0.08
            )
            scores[action_lookup["reduce"]] = (
                0.05
                + reduce_quality[idx] * 0.90
                + reduce_fraction[idx] * 1.10
                + max(-float(delta_hint[idx]), 0.0) * 0.30
                + max(signal_decay[idx], 0.0) * 0.12
            )
            scores[action_lookup["exit"]] = (
                0.03
                + exit_urgency[idx] * 1.05
                + exit_hazard[idx] * 1.15
                + max(-float(delta_hint[idx]) - 0.12, 0.0) * 0.45
                + max(-drawdown[idx] - 0.06, 0.0) * 0.18
                + max(market_downside[idx], 0.0) * 0.10
            )
            scores[action_lookup["skip"]] = 0.01
            scores[action_lookup["open"]] = 0.01
            scores[action_lookup["hold"]] *= float(np.clip(1.0 - sell_pressure[idx] * 0.32, 0.62, 1.08))
            scores[action_lookup["add"]] *= float(np.clip(1.0 - sell_pressure[idx] * 0.52, 0.34, 1.00))
            scores[action_lookup["hold"]] *= float(np.clip(1.0 - sell_attribution[idx] * 0.22, 0.60, 1.05))
            scores[action_lookup["add"]] *= float(np.clip(1.0 - sell_attribution[idx] * 0.26, 0.50, 1.00))
            scores[action_lookup["reduce"]] += sell_pressure[idx] * 0.24 + sell_attribution[idx] * 0.34
            scores[action_lookup["exit"]] += sell_pressure[idx] * 0.20 + sell_attribution[idx] * 0.22
            source_intent = float(portfolio_source_release_intent[idx])
            if source_intent > 0.0:
                scores[action_lookup["hold"]] *= float(np.clip(1.0 - source_intent * 0.58, 0.34, 1.0))
                scores[action_lookup["add"]] *= float(np.clip(1.0 - source_intent * 0.72, 0.24, 1.0))
                scores[action_lookup["reduce"]] += source_intent * 1.28
                scores[action_lookup["exit"]] += max(0.0, source_intent - 0.54) * 0.36
            if portfolio_source_candidate[idx] > 0.5:
                scores[action_lookup["hold"]] *= 0.68
                scores[action_lookup["add"]] *= 0.54
                scores[action_lookup["reduce"]] += 0.38
            if duration_ratio[idx] > 0.45 and hold_quality[idx] >= reduce_quality[idx] - 0.02:
                scores[action_lookup["hold"]] += 0.12
            if hold_days[idx] >= 6.0 and exit_urgency[idx] > 0.20:
                scores[action_lookup["exit"]] += 0.06
            if (sell_pressure[idx] > 0.26 or sell_attribution[idx] > 0.34) and hold_days[idx] >= 4.0:
                scores[action_lookup["reduce"]] += 0.08
            if (exit_hazard[idx] > 0.42 or sell_attribution[idx] > 0.52) and hold_days[idx] >= 7.0:
                scores[action_lookup["exit"]] += 0.08
        else:
            scores[action_lookup["skip"]] = (
                0.12
                + max(-float(delta_hint[idx]), 0.0) * 0.12
                + max(0.06 - entry_quality[idx], 0.0) * 0.15
            )
            scores[action_lookup["open"]] = (
                0.18
                + entry_quality[idx] * 1.10
                + duration_ratio[idx] * 0.24
                + max(float(delta_hint[idx]), 0.0) * 0.16
                + reentry_readiness[idx] * 0.10
            )
            scores[action_lookup["hold"]] = 0.01
            scores[action_lookup["add"]] = 0.01
            scores[action_lookup["reduce"]] = 0.01
            scores[action_lookup["exit"]] = 0.01

        hard_label = str(sample_frame.iloc[idx]["action_label"])
        if hard_label in action_lookup:
            scores[action_lookup[hard_label]] += 0.28
        score_sum = float(scores.sum())
        soft_targets[idx] = scores / score_sum if score_sum > 0.0 else np.full(len(ACTION_CLASSES), 1.0 / len(ACTION_CLASSES), dtype=np.float32)
    return soft_targets.astype(np.float32)


def _sequence_column_name(base_name: str, step: int) -> str:
    return str(base_name) if int(step) == 0 else f"{base_name}_lag{int(step)}"


def resolve_sequence_columns(feature_names: list[str]) -> tuple[list[str], list[str]]:
    available = set(feature_names)
    sequence_columns: list[str] = []
    sequence_bases: list[str] = []
    for base_name in STATE_SEQUENCE_BASES:
        required = [_sequence_column_name(base_name, step) for step in SEQUENCE_STEP_ORDER]
        if all(column in available for column in required):
            sequence_bases.append(str(base_name))
            sequence_columns.extend(required)
    if not sequence_bases:
        raise ValueError("formal_torch_seq_v3 requires lagged sequence features, but none were found in the training matrix.")
    return sequence_bases, sequence_columns


class TemporalSamplePolicyNet(nn.Module):
    def __init__(
        self,
        *,
        static_input_dim: int,
        sequence_feature_dim: int,
        sequence_steps: int,
        hidden_dim: int = 224,
        sequence_hidden_dim: int = 128,
        sequence_layers: int = 1,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.sequence_steps = int(sequence_steps)
        self.sequence_feature_dim = int(sequence_feature_dim)
        self.sequence_layers = max(1, int(sequence_layers))
        self.sequence_encoder = nn.GRU(
            input_size=self.sequence_feature_dim,
            hidden_size=int(sequence_hidden_dim),
            num_layers=self.sequence_layers,
            batch_first=True,
        )
        self.static_backbone = nn.Sequential(
            nn.Linear(int(static_input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        merged_dim = int(hidden_dim) + int(sequence_hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(merged_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.action_head = nn.Linear(int(hidden_dim), len(ACTION_CLASSES))
        self.duration_head = nn.Linear(int(hidden_dim), len(DURATION_CLASSES))
        self.delta_head = nn.Linear(int(hidden_dim), 1)
        self.entry_head = nn.Linear(int(hidden_dim), 1)
        self.hold_head = nn.Linear(int(hidden_dim), 1)
        self.add_head = nn.Linear(int(hidden_dim), 1)
        self.reduce_head = nn.Linear(int(hidden_dim), 1)
        self.exit_head = nn.Linear(int(hidden_dim), 1)
        self.reentry_head = nn.Linear(int(hidden_dim), 1)
        self.holding_days_head = nn.Linear(int(hidden_dim), 1)
        self.reduce_fraction_head = nn.Linear(int(hidden_dim), 1)
        self.exit_hazard_head = nn.Linear(int(hidden_dim), 1)
        self.sell_attribution_head = nn.Linear(int(hidden_dim), 1)
        self.sell_rank_head = nn.Linear(int(hidden_dim), 1)
        self.lifecycle_sell_gate_head = nn.Linear(int(hidden_dim), 1)
        self.clipped_intent_head = nn.Linear(int(hidden_dim), 1)
        self.large_upside_head = nn.Linear(int(hidden_dim), 1)
        self.alpha_opportunity_head = nn.Linear(int(hidden_dim), 1)
        self.hold_continuation_head = nn.Linear(int(hidden_dim), 1)
        self.sell_release_head = nn.Linear(int(hidden_dim), 1)
        self.cash_defense_head = nn.Linear(int(hidden_dim), 1)
        self.deployment_opportunity_head = nn.Linear(int(hidden_dim), 1)
        self.risk_adjusted_action_value_head = nn.Linear(int(hidden_dim), 1)
        self.value_arbitration_head = nn.Linear(int(hidden_dim), 1)
        self.multi_horizon_forward_value_head = nn.Linear(int(hidden_dim), 1)
        self.multi_horizon_forward_risk_head = nn.Linear(int(hidden_dim), 1)
        self.multi_horizon_path_value_head = nn.Linear(int(hidden_dim), 1)
        self.open_action_value_head = nn.Linear(int(hidden_dim), 1)
        self.add_action_value_head = nn.Linear(int(hidden_dim), 1)
        self.hold_action_value_head = nn.Linear(int(hidden_dim), 1)
        self.reduce_action_value_head = nn.Linear(int(hidden_dim), 1)
        self.exit_action_value_head = nn.Linear(int(hidden_dim), 1)
        self.relative_opportunity_value_head = nn.Linear(int(hidden_dim), 1)
        self.action_value_consistency_head = nn.Linear(int(hidden_dim), 1)
        self.deploy_value_head = nn.Linear(int(hidden_dim), 1)
        self.release_value_head = nn.Linear(int(hidden_dim), 1)
        self.defense_value_head = nn.Linear(int(hidden_dim), 1)
        self.deploy_gate_head = nn.Linear(int(hidden_dim), 1)
        self.release_gate_head = nn.Linear(int(hidden_dim), 1)
        self.defense_gate_head = nn.Linear(int(hidden_dim), 1)
        self.deploy_executability_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_receiver_headroom_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_receiver_capacity_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_receiver_executability_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_receiver_score_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_source_capacity_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_source_forward_spread_score_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_source_bad_forward_spread_risk_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_source_economic_release_score_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_source_economic_block_risk_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_source_forward_strength_brake_risk_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_source_forward_proxy_keep_risk_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_source_release_quality_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_source_opportunity_cost_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_source_executability_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_source_score_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_cash_score_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_receiver_funding_coverage_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_funding_closure_score_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_allocation_transfer_score_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_allocation_dead_branch_risk_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_unified_receiver_score_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_unified_source_score_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_unified_cash_score_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_source_positive_forward_penalty_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_source_opportunity_cost_penalty_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_source_strong_false_sell_penalty_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_source_hard_negative_penalty_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_source_tail_false_sell_penalty_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_source_release_preference_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_receiver_source_spread_reward_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_transfer_regret_target_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_unified_allocation_objective_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_allocation_trade_quality_target_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_allocation_cash_deployment_target_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_allocation_risk_adjusted_return_target_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_allocation_drawdown_control_target_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_allocation_monthly_quality_target_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_allocation_final_objective_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_allocation_uncertainty_pressure_target_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_allocation_tail_risk_control_target_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_allocation_decision_focused_objective_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_allocation_net_utility_target_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_allocation_credit_closure_target_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_allocation_resource_efficiency_target_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_daily_allocation_weight_logit_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_daily_cash_reserve_logit_head = nn.Linear(int(hidden_dim), 1)
        self.portfolio_daily_allocation_risk_buffer_logit_head = nn.Linear(int(hidden_dim), 1)

    def _encode_features(self, static_x: torch.Tensor, sequence_x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.sequence_encoder(sequence_x)
        seq_hidden = hidden[-1]
        static_hidden = self.static_backbone(static_x)
        return self.fusion(torch.cat([static_hidden, seq_hidden], dim=-1))

    def forward(self, static_x: torch.Tensor, sequence_x: torch.Tensor) -> dict[str, torch.Tensor]:
        fused = self._encode_features(static_x, sequence_x)
        return {
            "action_logits": self.action_head(fused),
            "duration_logits": self.duration_head(fused),
            "target_delta_hint": self.delta_head(fused).squeeze(-1),
            "entry_quality": self.entry_head(fused).squeeze(-1),
            "hold_quality": self.hold_head(fused).squeeze(-1),
            "add_quality": self.add_head(fused).squeeze(-1),
            "reduce_quality": self.reduce_head(fused).squeeze(-1),
            "exit_urgency": self.exit_head(fused).squeeze(-1),
            "reentry_readiness": self.reentry_head(fused).squeeze(-1),
            "holding_days_ratio": torch.sigmoid(self.holding_days_head(fused).squeeze(-1)),
            "reduce_fraction": torch.sigmoid(self.reduce_fraction_head(fused).squeeze(-1)),
            "exit_hazard": torch.sigmoid(self.exit_hazard_head(fused).squeeze(-1)),
            "sell_attribution_score": torch.sigmoid(self.sell_attribution_head(fused).squeeze(-1)),
            "sell_rank_score": torch.sigmoid(self.sell_rank_head(fused).squeeze(-1)),
            "lifecycle_sell_gate": torch.sigmoid(self.lifecycle_sell_gate_head(fused).squeeze(-1)),
            "clipped_intent_risk": torch.sigmoid(self.clipped_intent_head(fused).squeeze(-1)),
            "large_upside_1d_target": torch.sigmoid(self.large_upside_head(fused).squeeze(-1)),
            "alpha_opportunity_value": torch.sigmoid(self.alpha_opportunity_head(fused).squeeze(-1)),
            "hold_continuation_value": torch.sigmoid(self.hold_continuation_head(fused).squeeze(-1)),
            "sell_release_value": torch.sigmoid(self.sell_release_head(fused).squeeze(-1)),
            "cash_defense_value": torch.sigmoid(self.cash_defense_head(fused).squeeze(-1)),
            "deployment_opportunity_cost": torch.sigmoid(self.deployment_opportunity_head(fused).squeeze(-1)),
            "risk_adjusted_action_value": torch.sigmoid(self.risk_adjusted_action_value_head(fused).squeeze(-1)),
            "value_arbitration_target": torch.sigmoid(self.value_arbitration_head(fused).squeeze(-1)),
            "multi_horizon_forward_value": torch.sigmoid(self.multi_horizon_forward_value_head(fused).squeeze(-1)),
            "multi_horizon_forward_risk": torch.sigmoid(self.multi_horizon_forward_risk_head(fused).squeeze(-1)),
            "multi_horizon_path_value": torch.sigmoid(self.multi_horizon_path_value_head(fused).squeeze(-1)),
            "open_action_value": torch.sigmoid(self.open_action_value_head(fused).squeeze(-1)),
            "add_action_value": torch.sigmoid(self.add_action_value_head(fused).squeeze(-1)),
            "hold_action_value": torch.sigmoid(self.hold_action_value_head(fused).squeeze(-1)),
            "reduce_action_value": torch.sigmoid(self.reduce_action_value_head(fused).squeeze(-1)),
            "exit_action_value": torch.sigmoid(self.exit_action_value_head(fused).squeeze(-1)),
            "relative_opportunity_value": torch.sigmoid(self.relative_opportunity_value_head(fused).squeeze(-1)),
            "action_value_consistency_target": torch.sigmoid(self.action_value_consistency_head(fused).squeeze(-1)),
            "deploy_value_target": torch.sigmoid(self.deploy_value_head(fused).squeeze(-1)),
            "release_value_target": torch.sigmoid(self.release_value_head(fused).squeeze(-1)),
            "defense_value_target": torch.sigmoid(self.defense_value_head(fused).squeeze(-1)),
            "deploy_gate_target": torch.sigmoid(self.deploy_gate_head(fused).squeeze(-1)),
            "release_gate_target": torch.sigmoid(self.release_gate_head(fused).squeeze(-1)),
            "defense_gate_target": torch.sigmoid(self.defense_gate_head(fused).squeeze(-1)),
            "deploy_executability_target": torch.sigmoid(self.deploy_executability_head(fused).squeeze(-1)),
            "portfolio_daily_receiver_add_headroom": torch.sigmoid(self.portfolio_receiver_headroom_head(fused).squeeze(-1)),
            "portfolio_daily_receiver_add_capacity": torch.sigmoid(self.portfolio_receiver_capacity_head(fused).squeeze(-1)),
            "portfolio_daily_receiver_executability": torch.sigmoid(self.portfolio_receiver_executability_head(fused).squeeze(-1)),
            "portfolio_daily_receiver_score": torch.sigmoid(self.portfolio_receiver_score_head(fused).squeeze(-1)),
            "portfolio_daily_source_release_capacity": torch.sigmoid(self.portfolio_source_capacity_head(fused).squeeze(-1)),
            "portfolio_daily_source_forward_spread_score": torch.sigmoid(self.portfolio_source_forward_spread_score_head(fused).squeeze(-1)),
            "portfolio_daily_source_bad_forward_spread_risk": torch.sigmoid(self.portfolio_source_bad_forward_spread_risk_head(fused).squeeze(-1)),
            "portfolio_daily_source_economic_release_score": torch.sigmoid(self.portfolio_source_economic_release_score_head(fused).squeeze(-1)),
            "portfolio_daily_source_economic_block_risk": torch.sigmoid(self.portfolio_source_economic_block_risk_head(fused).squeeze(-1)),
            "portfolio_daily_source_forward_strength_brake_risk": torch.sigmoid(self.portfolio_source_forward_strength_brake_risk_head(fused).squeeze(-1)),
            "portfolio_daily_source_forward_proxy_keep_risk": torch.sigmoid(self.portfolio_source_forward_proxy_keep_risk_head(fused).squeeze(-1)),
            "portfolio_daily_source_release_quality": torch.sigmoid(self.portfolio_source_release_quality_head(fused).squeeze(-1)),
            "portfolio_daily_source_opportunity_cost": torch.sigmoid(self.portfolio_source_opportunity_cost_head(fused).squeeze(-1)),
            "portfolio_daily_source_executability": torch.sigmoid(self.portfolio_source_executability_head(fused).squeeze(-1)),
            "portfolio_daily_source_score": torch.sigmoid(self.portfolio_source_score_head(fused).squeeze(-1)),
            "portfolio_daily_cash_score": torch.sigmoid(self.portfolio_cash_score_head(fused).squeeze(-1)),
            "portfolio_daily_receiver_funding_coverage": torch.sigmoid(self.portfolio_receiver_funding_coverage_head(fused).squeeze(-1)),
            "portfolio_daily_funding_closure_score": torch.sigmoid(self.portfolio_funding_closure_score_head(fused).squeeze(-1)),
            "portfolio_daily_allocation_transfer_score": torch.sigmoid(self.portfolio_allocation_transfer_score_head(fused).squeeze(-1)),
            "portfolio_daily_allocation_dead_branch_risk": torch.sigmoid(self.portfolio_allocation_dead_branch_risk_head(fused).squeeze(-1)),
            "portfolio_daily_unified_receiver_score": torch.sigmoid(self.portfolio_unified_receiver_score_head(fused).squeeze(-1)),
            "portfolio_daily_unified_source_score": torch.sigmoid(self.portfolio_unified_source_score_head(fused).squeeze(-1)),
            "portfolio_daily_unified_cash_score": torch.sigmoid(self.portfolio_unified_cash_score_head(fused).squeeze(-1)),
            "portfolio_daily_source_positive_forward_penalty": torch.sigmoid(self.portfolio_source_positive_forward_penalty_head(fused).squeeze(-1)),
            "portfolio_daily_source_opportunity_cost_penalty": torch.sigmoid(self.portfolio_source_opportunity_cost_penalty_head(fused).squeeze(-1)),
            "portfolio_daily_source_strong_false_sell_penalty": torch.sigmoid(self.portfolio_source_strong_false_sell_penalty_head(fused).squeeze(-1)),
            "portfolio_daily_source_hard_negative_penalty": torch.sigmoid(self.portfolio_source_hard_negative_penalty_head(fused).squeeze(-1)),
            "portfolio_daily_source_tail_false_sell_penalty": torch.sigmoid(self.portfolio_source_tail_false_sell_penalty_head(fused).squeeze(-1)),
            "portfolio_daily_source_release_preference": torch.sigmoid(self.portfolio_source_release_preference_head(fused).squeeze(-1)),
            "portfolio_daily_receiver_source_spread_reward": torch.sigmoid(self.portfolio_receiver_source_spread_reward_head(fused).squeeze(-1)),
            "portfolio_daily_transfer_regret_target": torch.sigmoid(self.portfolio_transfer_regret_target_head(fused).squeeze(-1)),
            "portfolio_daily_unified_allocation_objective": torch.sigmoid(self.portfolio_unified_allocation_objective_head(fused).squeeze(-1)),
            "portfolio_daily_allocation_trade_quality_target": torch.sigmoid(self.portfolio_allocation_trade_quality_target_head(fused).squeeze(-1)),
            "portfolio_daily_allocation_cash_deployment_target": torch.sigmoid(self.portfolio_allocation_cash_deployment_target_head(fused).squeeze(-1)),
            "portfolio_daily_allocation_risk_adjusted_return_target": torch.sigmoid(self.portfolio_allocation_risk_adjusted_return_target_head(fused).squeeze(-1)),
            "portfolio_daily_allocation_drawdown_control_target": torch.sigmoid(self.portfolio_allocation_drawdown_control_target_head(fused).squeeze(-1)),
            "portfolio_daily_allocation_monthly_quality_target": torch.sigmoid(self.portfolio_allocation_monthly_quality_target_head(fused).squeeze(-1)),
            "portfolio_daily_allocation_final_objective": torch.sigmoid(self.portfolio_allocation_final_objective_head(fused).squeeze(-1)),
            "portfolio_daily_allocation_uncertainty_pressure_target": torch.sigmoid(self.portfolio_allocation_uncertainty_pressure_target_head(fused).squeeze(-1)),
            "portfolio_daily_allocation_tail_risk_control_target": torch.sigmoid(self.portfolio_allocation_tail_risk_control_target_head(fused).squeeze(-1)),
            "portfolio_daily_allocation_decision_focused_objective": torch.sigmoid(self.portfolio_allocation_decision_focused_objective_head(fused).squeeze(-1)),
            "portfolio_daily_allocation_net_utility_target": torch.sigmoid(self.portfolio_allocation_net_utility_target_head(fused).squeeze(-1)),
            "portfolio_daily_allocation_credit_closure_target": torch.sigmoid(self.portfolio_allocation_credit_closure_target_head(fused).squeeze(-1)),
            "portfolio_daily_allocation_resource_efficiency_target": torch.sigmoid(self.portfolio_allocation_resource_efficiency_target_head(fused).squeeze(-1)),
            "portfolio_daily_allocation_weight_logit": self.portfolio_daily_allocation_weight_logit_head(fused).squeeze(-1),
            "portfolio_daily_cash_reserve_logit": self.portfolio_daily_cash_reserve_logit_head(fused).squeeze(-1),
            "portfolio_daily_allocation_risk_buffer_logit": self.portfolio_daily_allocation_risk_buffer_logit_head(fused).squeeze(-1),
        }


class PortfolioSlotAttention(nn.Module):
    def __init__(self, *, input_dim: int, slot_count: int = 32, slot_dim: int = 128) -> None:
        super().__init__()
        self.slot_count = max(1, int(slot_count))
        self.slot_dim = max(1, int(slot_dim))
        self.slot_queries = nn.Parameter(torch.randn(self.slot_count, self.slot_dim) * 0.02)
        self.key_projection = nn.Linear(int(input_dim), self.slot_dim)
        self.value_projection = nn.Linear(int(input_dim), int(input_dim))
        self.output_projection = nn.Linear(int(input_dim), int(input_dim))

    def forward(self, row_features: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
        mask = sample_mask.to(device=row_features.device, dtype=torch.bool)
        keys = self.key_projection(row_features)
        values = self.value_projection(row_features)
        logits = torch.einsum("bnd,kd->bkn", keys, self.slot_queries) / math.sqrt(float(self.slot_dim))
        logits = logits.masked_fill(~mask[:, None, :], -1.0e9)
        attention = torch.softmax(logits, dim=-1)
        attention = torch.where(mask[:, None, :], attention, torch.zeros_like(attention))
        normalizer = torch.clamp(attention.sum(dim=-1, keepdim=True), min=1.0e-8)
        attention = attention / normalizer
        slot_context = torch.einsum("bkn,bnh->bkh", attention, values)
        pooled = slot_context.mean(dim=1)
        return self.output_projection(pooled)


class TemporalDaySetPolicyNet(nn.Module):
    sample_model_type = "temporal_day_set"

    def __init__(
        self,
        *,
        static_input_dim: int,
        sequence_feature_dim: int,
        sequence_steps: int,
        daily_input_dim: int,
        hidden_dim: int = 192,
        sequence_hidden_dim: int = 128,
        sequence_layers: int = 2,
        slot_count: int = 32,
        slot_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.daily_input_dim = int(daily_input_dim)
        self.slot_count = max(1, int(slot_count))
        self.slot_dim = max(1, int(slot_dim))
        self.hidden_dim = int(hidden_dim)
        self.sequence_hidden_dim = int(sequence_hidden_dim)
        self.sequence_layers = max(1, int(sequence_layers))
        self.dropout = float(dropout)
        self.sample_model = TemporalSamplePolicyNet(
            static_input_dim=int(static_input_dim),
            sequence_feature_dim=int(sequence_feature_dim),
            sequence_steps=int(sequence_steps),
            hidden_dim=int(hidden_dim),
            sequence_hidden_dim=int(sequence_hidden_dim),
            sequence_layers=int(sequence_layers),
            dropout=float(dropout),
        )
        self.daily_projection = nn.Sequential(
            nn.Linear(int(daily_input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.slot_attention = PortfolioSlotAttention(
            input_dim=int(hidden_dim),
            slot_count=self.slot_count,
            slot_dim=self.slot_dim,
        )
        decoder_dim = int(hidden_dim) * 3
        self.day_set_weight_decoder = nn.Sequential(
            nn.Linear(decoder_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), 1),
        )
        self.day_cash_decoder = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), 1),
        )
        self.day_risk_decoder = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        static_x: torch.Tensor,
        sequence_x: torch.Tensor,
        daily_x: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if static_x.ndim != 3 or sequence_x.ndim != 4:
            raise ValueError("TemporalDaySetPolicyNet expects static_x [B,N,S] and sequence_x [B,N,T,F].")
        batch_size, max_rows = int(static_x.shape[0]), int(static_x.shape[1])
        mask = sample_mask.to(device=static_x.device, dtype=torch.bool)
        flat_static = static_x.reshape(batch_size * max_rows, static_x.shape[-1])
        flat_sequence = sequence_x.reshape(batch_size * max_rows, sequence_x.shape[-2], sequence_x.shape[-1])
        flat_outputs = self.sample_model(flat_static, flat_sequence)
        flat_features = self.sample_model._encode_features(flat_static, flat_sequence)
        row_features = flat_features.reshape(batch_size, max_rows, -1)
        daily_context = self.daily_projection(daily_x)
        slot_context = self.slot_attention(row_features, mask)
        decoder_features = torch.cat(
            [
                row_features,
                daily_context[:, None, :].expand(batch_size, max_rows, -1),
                slot_context[:, None, :].expand(batch_size, max_rows, -1),
            ],
            dim=-1,
        )
        weight_logit = self.day_set_weight_decoder(decoder_features).squeeze(-1)
        weight_logit = torch.where(mask, weight_logit, torch.full_like(weight_logit, -1.0e6))
        day_context = torch.cat([daily_context, slot_context], dim=-1)
        cash_logit = self.day_cash_decoder(day_context).squeeze(-1)
        risk_logit = self.day_risk_decoder(day_context).squeeze(-1)

        outputs: dict[str, torch.Tensor] = {}
        for name, value in flat_outputs.items():
            if value.ndim == 2:
                outputs[name] = value.reshape(batch_size, max_rows, value.shape[-1])
            else:
                outputs[name] = value.reshape(batch_size, max_rows) * mask.to(value.dtype)
        outputs["portfolio_daily_allocation_weight_logit"] = weight_logit
        outputs["portfolio_daily_cash_reserve_logit"] = cash_logit
        outputs["portfolio_daily_allocation_risk_buffer_logit"] = risk_logit
        return outputs


class DailyControllerNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        dropout: float = 0.05,
        head_layout: str = DAILY_HEAD_LAYOUT_MONOLITHIC_V1,
    ) -> None:
        super().__init__()
        layout = str(head_layout or DAILY_HEAD_LAYOUT_MONOLITHIC_V1).strip() or DAILY_HEAD_LAYOUT_MONOLITHIC_V1
        if layout not in DAILY_HEAD_LAYOUT_CHOICES:
            raise ValueError(f"Unsupported daily controller head layout: {layout}")
        self.head_layout = layout
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if self.head_layout == DAILY_HEAD_LAYOUT_MONOLITHIC_V1:
            self.output = nn.Linear(hidden_dim, 5)
        else:
            tower_dim = max(32, hidden_dim // 2)
            self.exposure_tower = nn.Sequential(
                nn.Linear(hidden_dim, tower_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.deployment_tower = nn.Sequential(
                nn.Linear(hidden_dim, tower_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.lifecycle_tower = nn.Sequential(
                nn.Linear(hidden_dim, tower_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.signal_tower = nn.Sequential(
                nn.Linear(tower_dim * 3, tower_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.exposure_head = nn.Linear(tower_dim, 2)
            self.deployment_head = nn.Linear(tower_dim, 2)
            self.lifecycle_head = nn.Linear(tower_dim, 4)
            self.signal_head = nn.Linear(tower_dim, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(x)
        if self.head_layout == DAILY_HEAD_LAYOUT_MONOLITHIC_V1:
            raw = self.output(hidden)
            return {
                "gross_exposure_target": raw[:, 0],
                "candidate_budget": raw[:, 1],
                "turnover_budget": raw[:, 2],
                "max_position_weight_target": raw[:, 3],
                "hold_bias_target": raw[:, 4],
            }

        exposure_hidden = self.exposure_tower(hidden)
        deployment_hidden = self.deployment_tower(hidden)
        lifecycle_hidden = self.lifecycle_tower(hidden)
        signal_hidden = self.signal_tower(torch.cat([exposure_hidden, deployment_hidden, lifecycle_hidden], dim=-1))
        exposure = self.exposure_head(exposure_hidden)
        deployment = self.deployment_head(deployment_hidden)
        lifecycle = self.lifecycle_head(lifecycle_hidden)
        signals = torch.sigmoid(self.signal_head(signal_hidden))
        return {
            "gross_exposure_target": exposure[:, 0],
            "max_position_weight_target": exposure[:, 1],
            "candidate_budget": deployment[:, 0],
            "turnover_budget": deployment[:, 1],
            "hold_bias_target": lifecycle[:, 0],
            "reduce_bias_target": lifecycle[:, 1],
            "exit_patience_target": lifecycle[:, 2],
            "reentry_guard_target": lifecycle[:, 3],
            "budget_risk_signal_target": signals[:, 0],
            "budget_deploy_signal_target": signals[:, 1],
            "budget_cash_timing_signal_target": signals[:, 2],
            "budget_alpha_focus_signal_target": signals[:, 3],
        }


@dataclass
class TorchContinuousPolicySeqArtifact:
    sample_model: TemporalSamplePolicyNet
    daily_model: DailyControllerNet
    static_feature_names: list[str]
    sequence_base_names: list[str]
    sequence_steps: list[int]
    sequence_columns: list[str]
    daily_feature_names: list[str]
    static_fill_values: np.ndarray
    static_means: np.ndarray
    static_stds: np.ndarray
    sequence_fill_values: np.ndarray
    sequence_means: np.ndarray
    sequence_stds: np.ndarray
    daily_fill_values: np.ndarray
    daily_means: np.ndarray
    daily_stds: np.ndarray
    train_summary: dict[str, Any]
    training_diagnostics: dict[str, Any]
    training_contract: dict[str, Any]
    trained_at: str

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        sample_model_type = str(getattr(self.sample_model, "sample_model_type", "temporal_sample") or "temporal_sample")
        if sample_model_type == "temporal_day_set":
            base_sample_model = self.sample_model.sample_model
            sample_model_config = {
                "static_input_dim": len(self.static_feature_names),
                "sequence_feature_dim": len(self.sequence_base_names),
                "sequence_steps": len(self.sequence_steps),
                "hidden_dim": int(base_sample_model.static_backbone[0].out_features),
                "sequence_hidden_dim": int(base_sample_model.sequence_encoder.hidden_size),
                "sequence_layers": int(getattr(base_sample_model, "sequence_layers", 1)),
                "dropout": float(base_sample_model.static_backbone[2].p),
            }
            day_set_model_config = {
                "static_input_dim": len(self.static_feature_names),
                "sequence_feature_dim": len(self.sequence_base_names),
                "sequence_steps": len(self.sequence_steps),
                "daily_input_dim": len(self.daily_feature_names),
                "hidden_dim": int(getattr(self.sample_model, "hidden_dim", sample_model_config["hidden_dim"])),
                "sequence_hidden_dim": int(getattr(self.sample_model, "sequence_hidden_dim", sample_model_config["sequence_hidden_dim"])),
                "sequence_layers": int(getattr(self.sample_model, "sequence_layers", sample_model_config["sequence_layers"])),
                "slot_count": int(getattr(self.sample_model, "slot_count", 32)),
                "slot_dim": int(getattr(self.sample_model, "slot_dim", 128)),
                "dropout": float(getattr(self.sample_model, "dropout", sample_model_config["dropout"])),
            }
        else:
            sample_model_config = {
                "static_input_dim": len(self.static_feature_names),
                "sequence_feature_dim": len(self.sequence_base_names),
                "sequence_steps": len(self.sequence_steps),
                "hidden_dim": int(self.sample_model.static_backbone[0].out_features),
                "sequence_hidden_dim": int(self.sample_model.sequence_encoder.hidden_size),
                "sequence_layers": int(getattr(self.sample_model, "sequence_layers", 1)),
                "dropout": float(self.sample_model.static_backbone[2].p),
            }
            day_set_model_config = {}
        payload = {
            "artifact_type": "continuous_policy_torch_seq_v3",
            "sample_model_type": sample_model_type,
            "static_feature_names": self.static_feature_names,
            "sequence_base_names": self.sequence_base_names,
            "sequence_steps": self.sequence_steps,
            "sequence_columns": self.sequence_columns,
            "daily_feature_names": self.daily_feature_names,
            "static_fill_values": self.static_fill_values.tolist(),
            "static_means": self.static_means.tolist(),
            "static_stds": self.static_stds.tolist(),
            "sequence_fill_values": self.sequence_fill_values.tolist(),
            "sequence_means": self.sequence_means.tolist(),
            "sequence_stds": self.sequence_stds.tolist(),
            "daily_fill_values": self.daily_fill_values.tolist(),
            "daily_means": self.daily_means.tolist(),
            "daily_stds": self.daily_stds.tolist(),
            "sample_model_state_dict": self.sample_model.state_dict(),
            "daily_model_state_dict": self.daily_model.state_dict(),
            "sample_model_config": sample_model_config,
            "day_set_model_config": day_set_model_config,
            "daily_model_config": {
                "input_dim": len(self.daily_feature_names),
                "hidden_dim": int(self.daily_model.backbone[0].out_features),
                "dropout": float(self.daily_model.backbone[2].p),
                "head_layout": str(getattr(self.daily_model, "head_layout", DAILY_HEAD_LAYOUT_MONOLITHIC_V1) or DAILY_HEAD_LAYOUT_MONOLITHIC_V1),
            },
            "train_summary": self.train_summary,
            "training_diagnostics": self.training_diagnostics,
            "training_contract": self.training_contract,
            "trained_at": self.trained_at,
        }
        torch.save(payload, path)
        return path


def _reshape_sequence_matrix(values: np.ndarray, *, steps: int, feature_dim: int) -> np.ndarray:
    sample_count = int(values.shape[0])
    return values.reshape(sample_count, steps, feature_dim).astype(np.float32)


def _finite_array(values: Any, *, default: float = 0.0, low: float | None = None, high: float | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = np.nan_to_num(array, nan=float(default), posinf=float(default), neginf=float(default))
    if low is not None or high is not None:
        array = np.clip(
            array,
            -np.inf if low is None else float(low),
            np.inf if high is None else float(high),
        )
    return array


class DaySetTensorDataset(Dataset):
    """Dataset that keeps every trading date as a complete stock set."""

    def __init__(
        self,
        *,
        date_codes: torch.Tensor | np.ndarray,
        static_x: torch.Tensor | np.ndarray,
        sequence_x: torch.Tensor | np.ndarray,
        action: torch.Tensor | np.ndarray,
        duration: torch.Tensor | np.ndarray,
        action_soft: torch.Tensor | np.ndarray,
        sample_targets: dict[str, torch.Tensor | np.ndarray],
        daily_x: torch.Tensor | np.ndarray,
        daily_targets: dict[str, torch.Tensor | np.ndarray] | None = None,
        allowed_date_codes: torch.Tensor | np.ndarray | list[int] | None = None,
    ) -> None:
        self.date_codes = torch.as_tensor(date_codes, dtype=torch.long).flatten()
        self.static_x = torch.as_tensor(static_x, dtype=torch.float32)
        self.sequence_x = torch.as_tensor(sequence_x, dtype=torch.float32)
        self.action = torch.as_tensor(action, dtype=torch.long).flatten()
        self.duration = torch.as_tensor(duration, dtype=torch.long).flatten()
        self.action_soft = torch.as_tensor(action_soft, dtype=torch.float32)
        self.sample_targets = {
            str(name): torch.as_tensor(values, dtype=torch.float32).flatten()
            for name, values in dict(sample_targets or {}).items()
        }
        self.daily_x = torch.as_tensor(daily_x, dtype=torch.float32)
        self.daily_targets = {
            str(name): torch.as_tensor(values, dtype=torch.float32).flatten()
            for name, values in dict(daily_targets or {}).items()
        }
        if self.static_x.shape[0] != self.date_codes.numel() or self.sequence_x.shape[0] != self.date_codes.numel():
            raise ValueError("DaySetTensorDataset requires aligned date/static/sequence row counts.")
        all_codes = torch.unique(self.date_codes).tolist()
        if allowed_date_codes is None:
            selected_codes = [int(code) for code in all_codes]
        else:
            allowed = {int(code) for code in torch.as_tensor(allowed_date_codes, dtype=torch.long).flatten().tolist()}
            selected_codes = [int(code) for code in all_codes if int(code) in allowed]
        self.date_code_values = sorted(selected_codes)
        self.row_indices_by_date = [
            torch.nonzero(self.date_codes == int(code), as_tuple=False).flatten()
            for code in self.date_code_values
        ]

    def __len__(self) -> int:
        return len(self.date_code_values)

    def __getitem__(self, index: int) -> dict[str, Any]:
        date_code = int(self.date_code_values[index])
        row_indices = self.row_indices_by_date[index]
        daily_index = min(max(date_code, 0), max(0, int(self.daily_x.shape[0]) - 1))
        return {
            "date_code": self.date_codes[row_indices],
            "row_indices": row_indices,
            "static_x": self.static_x[row_indices],
            "sequence_x": self.sequence_x[row_indices],
            "action": self.action[row_indices],
            "duration": self.duration[row_indices],
            "action_soft": self.action_soft[row_indices],
            "sample_targets": {
                name: values[row_indices]
                for name, values in self.sample_targets.items()
            },
            "daily_x": self.daily_x[daily_index],
            "daily_targets": {
                name: values[daily_index]
                for name, values in self.daily_targets.items()
                if values.numel() > daily_index
            },
        }


def _collate_day_set_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("_collate_day_set_batch requires at least one day item.")
    batch_size = len(items)
    max_rows = max(int(item["static_x"].shape[0]) for item in items)
    static_dim = int(items[0]["static_x"].shape[-1])
    sequence_steps = int(items[0]["sequence_x"].shape[-2])
    sequence_dim = int(items[0]["sequence_x"].shape[-1])
    action_classes = int(items[0]["action_soft"].shape[-1])
    daily_dim = int(items[0]["daily_x"].shape[-1])

    static_x = torch.zeros(batch_size, max_rows, static_dim, dtype=torch.float32)
    sequence_x = torch.zeros(batch_size, max_rows, sequence_steps, sequence_dim, dtype=torch.float32)
    action = torch.zeros(batch_size, max_rows, dtype=torch.long)
    duration = torch.zeros(batch_size, max_rows, dtype=torch.long)
    action_soft = torch.zeros(batch_size, max_rows, action_classes, dtype=torch.float32)
    sample_mask = torch.zeros(batch_size, max_rows, dtype=torch.bool)
    date_code = torch.zeros(batch_size, max_rows, dtype=torch.long)
    row_indices = torch.full((batch_size, max_rows), -1, dtype=torch.long)
    daily_x = torch.zeros(batch_size, daily_dim, dtype=torch.float32)

    sample_target_names = sorted({name for item in items for name in item["sample_targets"].keys()})
    daily_target_names = sorted({name for item in items for name in item["daily_targets"].keys()})
    sample_targets = {
        name: torch.zeros(batch_size, max_rows, dtype=torch.float32)
        for name in sample_target_names
    }
    daily_targets = {
        name: torch.zeros(batch_size, dtype=torch.float32)
        for name in daily_target_names
    }

    for batch_index, item in enumerate(items):
        row_count = int(item["static_x"].shape[0])
        static_x[batch_index, :row_count] = item["static_x"].float()
        sequence_x[batch_index, :row_count] = item["sequence_x"].float()
        action[batch_index, :row_count] = item["action"].long()
        duration[batch_index, :row_count] = item["duration"].long()
        action_soft[batch_index, :row_count] = item["action_soft"].float()
        sample_mask[batch_index, :row_count] = True
        date_code[batch_index, :row_count] = item["date_code"].long()
        row_indices[batch_index, :row_count] = item["row_indices"].long()
        daily_x[batch_index] = item["daily_x"].float()
        for name in sample_target_names:
            if name in item["sample_targets"]:
                sample_targets[name][batch_index, :row_count] = item["sample_targets"][name].float()
        for name in daily_target_names:
            if name in item["daily_targets"]:
                daily_targets[name][batch_index] = item["daily_targets"][name].float()

    return {
        "static_x": static_x,
        "sequence_x": sequence_x,
        "action": action,
        "duration": duration,
        "action_soft": action_soft,
        "sample_mask": sample_mask,
        "date_code": date_code,
        "row_indices": row_indices,
        "sample_targets": sample_targets,
        "daily_x": daily_x,
        "daily_targets": daily_targets,
    }


def _finite_mean(values: Any, *, default: float = 0.0) -> float:
    array = _finite_array(values, default=float(default))
    if array.size <= 0:
        return float(default)
    value = float(array.mean())
    return value if np.isfinite(value) else float(default)


def load_torch_seq_artifact(path: str | Path) -> TorchContinuousPolicySeqArtifact:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if str(payload.get("artifact_type", "") or "") != "continuous_policy_torch_seq_v3":
        raise TypeError(f"Unsupported seq artifact type: {payload.get('artifact_type')!r}")
    sample_cfg = dict(payload.get("sample_model_config", {}) or {})
    daily_cfg = dict(payload.get("daily_model_config", {}) or {})
    sample_model_type = str(payload.get("sample_model_type", "temporal_sample") or "temporal_sample")
    if sample_model_type == "temporal_day_set":
        day_set_cfg = dict(payload.get("day_set_model_config", {}) or {})
        if not day_set_cfg:
            day_set_cfg = {
                **sample_cfg,
                "daily_input_dim": len(payload.get("daily_feature_names", []) or []),
                "slot_count": 32,
                "slot_dim": 128,
            }
        sample_model = TemporalDaySetPolicyNet(**day_set_cfg)
    else:
        sample_model = TemporalSamplePolicyNet(**sample_cfg)
    daily_model = DailyControllerNet(**daily_cfg)
    sample_load = sample_model.load_state_dict(payload["sample_model_state_dict"], strict=False)
    daily_load = daily_model.load_state_dict(payload["daily_model_state_dict"], strict=False)
    sample_missing = set(getattr(sample_load, "missing_keys", []) or [])
    sample_unexpected = set(getattr(sample_load, "unexpected_keys", []) or [])
    daily_missing = set(getattr(daily_load, "missing_keys", []) or [])
    daily_unexpected = set(getattr(daily_load, "unexpected_keys", []) or [])
    allowed_sample_missing = {
        "holding_days_head.weight",
        "holding_days_head.bias",
        "reduce_fraction_head.weight",
        "reduce_fraction_head.bias",
        "exit_hazard_head.weight",
        "exit_hazard_head.bias",
        "sell_attribution_head.weight",
        "sell_attribution_head.bias",
        "sell_rank_head.weight",
        "sell_rank_head.bias",
        "lifecycle_sell_gate_head.weight",
        "lifecycle_sell_gate_head.bias",
        "clipped_intent_head.weight",
        "clipped_intent_head.bias",
        "large_upside_head.weight",
        "large_upside_head.bias",
        "alpha_opportunity_head.weight",
        "alpha_opportunity_head.bias",
        "hold_continuation_head.weight",
        "hold_continuation_head.bias",
        "sell_release_head.weight",
        "sell_release_head.bias",
        "cash_defense_head.weight",
        "cash_defense_head.bias",
        "deployment_opportunity_head.weight",
        "deployment_opportunity_head.bias",
        "risk_adjusted_action_value_head.weight",
        "risk_adjusted_action_value_head.bias",
        "value_arbitration_head.weight",
        "value_arbitration_head.bias",
        "multi_horizon_forward_value_head.weight",
        "multi_horizon_forward_value_head.bias",
        "multi_horizon_forward_risk_head.weight",
        "multi_horizon_forward_risk_head.bias",
        "multi_horizon_path_value_head.weight",
        "multi_horizon_path_value_head.bias",
        "open_action_value_head.weight",
        "open_action_value_head.bias",
        "add_action_value_head.weight",
        "add_action_value_head.bias",
        "hold_action_value_head.weight",
        "hold_action_value_head.bias",
        "reduce_action_value_head.weight",
        "reduce_action_value_head.bias",
        "exit_action_value_head.weight",
        "exit_action_value_head.bias",
        "relative_opportunity_value_head.weight",
        "relative_opportunity_value_head.bias",
        "action_value_consistency_head.weight",
        "action_value_consistency_head.bias",
        "deploy_value_head.weight",
        "deploy_value_head.bias",
        "release_value_head.weight",
        "release_value_head.bias",
        "defense_value_head.weight",
        "defense_value_head.bias",
        "deploy_gate_head.weight",
        "deploy_gate_head.bias",
        "release_gate_head.weight",
        "release_gate_head.bias",
        "defense_gate_head.weight",
        "defense_gate_head.bias",
        "deploy_executability_head.weight",
        "deploy_executability_head.bias",
        "portfolio_receiver_headroom_head.weight",
        "portfolio_receiver_headroom_head.bias",
        "portfolio_receiver_capacity_head.weight",
        "portfolio_receiver_capacity_head.bias",
        "portfolio_receiver_executability_head.weight",
        "portfolio_receiver_executability_head.bias",
        "portfolio_receiver_score_head.weight",
        "portfolio_receiver_score_head.bias",
        "portfolio_source_capacity_head.weight",
        "portfolio_source_capacity_head.bias",
        "portfolio_source_forward_proxy_keep_risk_head.weight",
        "portfolio_source_forward_proxy_keep_risk_head.bias",
        "portfolio_source_release_quality_head.weight",
        "portfolio_source_release_quality_head.bias",
        "portfolio_source_opportunity_cost_head.weight",
        "portfolio_source_opportunity_cost_head.bias",
        "portfolio_source_executability_head.weight",
        "portfolio_source_executability_head.bias",
        "portfolio_source_score_head.weight",
        "portfolio_source_score_head.bias",
        "portfolio_cash_score_head.weight",
        "portfolio_cash_score_head.bias",
        "portfolio_receiver_funding_coverage_head.weight",
        "portfolio_receiver_funding_coverage_head.bias",
        "portfolio_funding_closure_score_head.weight",
        "portfolio_funding_closure_score_head.bias",
        "portfolio_allocation_transfer_score_head.weight",
        "portfolio_allocation_transfer_score_head.bias",
        "portfolio_allocation_dead_branch_risk_head.weight",
        "portfolio_allocation_dead_branch_risk_head.bias",
        "portfolio_allocation_trade_quality_target_head.weight",
        "portfolio_allocation_trade_quality_target_head.bias",
        "portfolio_allocation_cash_deployment_target_head.weight",
        "portfolio_allocation_cash_deployment_target_head.bias",
        "portfolio_allocation_risk_adjusted_return_target_head.weight",
        "portfolio_allocation_risk_adjusted_return_target_head.bias",
        "portfolio_allocation_drawdown_control_target_head.weight",
        "portfolio_allocation_drawdown_control_target_head.bias",
        "portfolio_allocation_monthly_quality_target_head.weight",
        "portfolio_allocation_monthly_quality_target_head.bias",
        "portfolio_allocation_final_objective_head.weight",
        "portfolio_allocation_final_objective_head.bias",
    }
    if sample_unexpected or (sample_missing - allowed_sample_missing):
        raise RuntimeError(
            "continuous_policy seq_v3 artifact load failed because stored sample model weights are incompatible with the current architecture."
        )
    if daily_missing or daily_unexpected:
        raise RuntimeError(
            "continuous_policy seq_v3 artifact load failed because stored daily model weights are incompatible with the current architecture."
        )
    sample_model.eval()
    daily_model.eval()
    training_diagnostics = dict(payload.get("training_diagnostics", {}) or {})
    training_diagnostics["sample_model_type"] = sample_model_type
    if sample_model_type == "temporal_day_set":
        training_diagnostics["supports_portfolio_day_set_native_allocation_vector"] = True
        training_diagnostics["day_set_slot_count"] = int(getattr(sample_model, "slot_count", 0) or 0)
        training_diagnostics["day_set_full_day_integrity"] = True
    missing_key_names = {str(name) for name in sample_missing}
    training_diagnostics["supports_holding_days_head"] = not any(name.startswith("holding_days_head.") for name in missing_key_names)
    training_diagnostics["supports_sell_heads"] = not any(
        name.startswith("reduce_fraction_head.") or name.startswith("exit_hazard_head.")
        for name in missing_key_names
    )
    if any(name.startswith("sell_attribution_head.") for name in missing_key_names):
        training_diagnostics["supports_sell_attribution_head"] = False
    elif "supports_sell_attribution_head" not in training_diagnostics:
        training_diagnostics["supports_sell_attribution_head"] = True
    arbitration_missing = any(
        name.startswith("sell_rank_head.")
        or name.startswith("lifecycle_sell_gate_head.")
        or name.startswith("clipped_intent_head.")
        for name in missing_key_names
    )
    training_diagnostics["supports_lifecycle_arbitration_heads"] = not arbitration_missing
    value_arbitration_missing = any(
        name.startswith("large_upside_head.")
        or name.startswith("alpha_opportunity_head.")
        or name.startswith("hold_continuation_head.")
        or name.startswith("sell_release_head.")
        or name.startswith("cash_defense_head.")
        or name.startswith("deployment_opportunity_head.")
        or name.startswith("risk_adjusted_action_value_head.")
        or name.startswith("value_arbitration_head.")
        for name in missing_key_names
    )
    training_diagnostics["supports_value_arbitration_heads"] = not value_arbitration_missing
    action_value_missing = any(
        name.startswith("multi_horizon_forward_value_head.")
        or name.startswith("multi_horizon_forward_risk_head.")
        or name.startswith("multi_horizon_path_value_head.")
        or name.startswith("open_action_value_head.")
        or name.startswith("add_action_value_head.")
        or name.startswith("hold_action_value_head.")
        or name.startswith("reduce_action_value_head.")
        or name.startswith("exit_action_value_head.")
        or name.startswith("relative_opportunity_value_head.")
        or name.startswith("action_value_consistency_head.")
        for name in missing_key_names
    )
    training_diagnostics["supports_action_value_heads"] = not action_value_missing
    three_value_gate_missing = any(
        name.startswith("deploy_value_head.")
        or name.startswith("release_value_head.")
        or name.startswith("defense_value_head.")
        or name.startswith("deploy_gate_head.")
        or name.startswith("release_gate_head.")
        or name.startswith("defense_gate_head.")
        for name in missing_key_names
    )
    training_diagnostics["supports_three_value_gate_heads"] = not three_value_gate_missing
    deploy_executability_missing = any(
        name.startswith("deploy_executability_head.")
        for name in missing_key_names
    )
    training_diagnostics["supports_deploy_executability_head"] = not deploy_executability_missing
    portfolio_listwise_missing = any(
        name.startswith("portfolio_receiver_headroom_head.")
        or name.startswith("portfolio_receiver_capacity_head.")
        or name.startswith("portfolio_receiver_executability_head.")
        or name.startswith("portfolio_receiver_score_head.")
        or name.startswith("portfolio_source_score_head.")
        or name.startswith("portfolio_cash_score_head.")
        for name in missing_key_names
    )
    training_diagnostics["supports_portfolio_listwise_heads"] = not portfolio_listwise_missing
    portfolio_source_release_missing = any(
        name.startswith("portfolio_source_capacity_head.")
        or name.startswith("portfolio_source_release_quality_head.")
        or name.startswith("portfolio_source_opportunity_cost_head.")
        or name.startswith("portfolio_source_executability_head.")
        for name in missing_key_names
    )
    training_diagnostics["supports_portfolio_source_release_heads"] = not portfolio_source_release_missing
    portfolio_economic_release_missing = any(
        name.startswith("portfolio_source_forward_spread_score_head.")
        or name.startswith("portfolio_source_bad_forward_spread_risk_head.")
        or name.startswith("portfolio_source_economic_release_score_head.")
        or name.startswith("portfolio_source_economic_block_risk_head.")
        or name.startswith("portfolio_source_forward_strength_brake_risk_head.")
        for name in missing_key_names
    )
    training_diagnostics["supports_portfolio_economic_release_heads"] = (
        not portfolio_economic_release_missing
    )
    training_diagnostics["supports_portfolio_source_forward_proxy_head"] = not any(
        name.startswith("portfolio_source_forward_proxy_keep_risk_head.")
        for name in missing_key_names
    )
    portfolio_allocation_teacher_missing = any(
        name.startswith("portfolio_receiver_funding_coverage_head.")
        or name.startswith("portfolio_funding_closure_score_head.")
        or name.startswith("portfolio_allocation_transfer_score_head.")
        or name.startswith("portfolio_allocation_dead_branch_risk_head.")
        for name in missing_key_names
    )
    training_diagnostics["supports_portfolio_allocation_teacher_heads"] = (
        not portfolio_allocation_teacher_missing
    )
    portfolio_unified_allocation_missing = any(
        name.startswith("portfolio_unified_receiver_score_head.")
        or name.startswith("portfolio_unified_source_score_head.")
        or name.startswith("portfolio_unified_cash_score_head.")
        or name.startswith("portfolio_source_positive_forward_penalty_head.")
        or name.startswith("portfolio_source_opportunity_cost_penalty_head.")
        or name.startswith("portfolio_receiver_source_spread_reward_head.")
        or name.startswith("portfolio_unified_allocation_objective_head.")
        for name in missing_key_names
    )
    training_diagnostics["supports_portfolio_unified_allocation_heads"] = (
        not portfolio_unified_allocation_missing
    )
    portfolio_allocation_objective_consolidation_missing = any(
        name.startswith("portfolio_allocation_trade_quality_target_head.")
        or name.startswith("portfolio_allocation_cash_deployment_target_head.")
        or name.startswith("portfolio_allocation_risk_adjusted_return_target_head.")
        or name.startswith("portfolio_allocation_drawdown_control_target_head.")
        or name.startswith("portfolio_allocation_monthly_quality_target_head.")
        or name.startswith("portfolio_allocation_final_objective_head.")
        for name in missing_key_names
    )
    training_diagnostics["supports_portfolio_allocation_objective_consolidation_heads"] = (
        not portfolio_allocation_objective_consolidation_missing
    )
    portfolio_risk_sensitive_allocation_missing = any(
        name.startswith("portfolio_allocation_uncertainty_pressure_target_head.")
        or name.startswith("portfolio_allocation_tail_risk_control_target_head.")
        or name.startswith("portfolio_allocation_decision_focused_objective_head.")
        for name in missing_key_names
    )
    training_diagnostics["supports_portfolio_risk_sensitive_allocation_heads"] = (
        not portfolio_risk_sensitive_allocation_missing
    )
    portfolio_utility_credit_closure_missing = any(
        name.startswith("portfolio_allocation_net_utility_target_head.")
        or name.startswith("portfolio_allocation_credit_closure_target_head.")
        or name.startswith("portfolio_allocation_resource_efficiency_target_head.")
        for name in missing_key_names
    )
    training_diagnostics["supports_portfolio_utility_credit_closure_heads"] = (
        not portfolio_utility_credit_closure_missing
    )
    return TorchContinuousPolicySeqArtifact(
        sample_model=sample_model,
        daily_model=daily_model,
        static_feature_names=list(payload.get("static_feature_names", []) or []),
        sequence_base_names=list(payload.get("sequence_base_names", []) or []),
        sequence_steps=[int(item) for item in payload.get("sequence_steps", []) or []],
        sequence_columns=list(payload.get("sequence_columns", []) or []),
        daily_feature_names=list(payload.get("daily_feature_names", []) or []),
        static_fill_values=np.asarray(payload.get("static_fill_values", []), dtype=np.float32),
        static_means=np.asarray(payload.get("static_means", []), dtype=np.float32),
        static_stds=np.asarray(payload.get("static_stds", []), dtype=np.float32),
        sequence_fill_values=np.asarray(payload.get("sequence_fill_values", []), dtype=np.float32),
        sequence_means=np.asarray(payload.get("sequence_means", []), dtype=np.float32),
        sequence_stds=np.asarray(payload.get("sequence_stds", []), dtype=np.float32),
        daily_fill_values=np.asarray(payload.get("daily_fill_values", []), dtype=np.float32),
        daily_means=np.asarray(payload.get("daily_means", []), dtype=np.float32),
        daily_stds=np.asarray(payload.get("daily_stds", []), dtype=np.float32),
        train_summary=dict(payload.get("train_summary", {}) or {}),
        training_diagnostics=training_diagnostics,
        training_contract=dict(payload.get("training_contract", {}) or {}),
        trained_at=str(payload.get("trained_at", "") or ""),
    )


def fit_policy_models_v3(
    *,
    sample_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
    feature_names: list[str],
    daily_feature_names: list[str],
    run_root: Path,
    random_seed: int = 7,
    train_summary: dict[str, Any] | None = None,
    trained_at: str = "",
    training_contract: dict[str, Any] | None = None,
    epochs: int = 32,
    min_epochs: int = 32,
    batch_size: int = 512,
    learning_rate: float = 1.2e-3,
    hidden_dim: int = 224,
    sequence_hidden_dim: int = 128,
    sequence_layers: int = 1,
    daily_hidden_dim: int = 96,
    dropout: float = 0.12,
    daily_dropout: float = 0.05,
    daily_head_layout: str = DAILY_HEAD_LAYOUT_MONOLITHIC_V1,
    early_stop_patience: int = 10,
    resume_mode: str = "strict",
    loss_profile: str = DEFAULT_LOSS_PROFILE,
) -> TorchContinuousPolicySeqArtifact:
    if sample_frame.empty or daily_frame.empty:
        raise ValueError("continuous_policy formal_torch_seq_v3 received empty training data.")
    contract = dict(training_contract or {})
    if str(contract.get("trainer_backend", "") or "") != TRAINER_BACKEND_FORMAL_SEQ_V3:
        raise ValueError("fit_policy_models_v3 requires the formal_torch_seq_v3 training contract.")
    resource_runtime = _apply_torch_thread_env_limits()
    device = _resolve_device(contract)
    torch.manual_seed(int(random_seed))
    np.random.seed(int(random_seed))
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_loss_profile, loss_config = resolve_loss_profile(loss_profile)
    resolved_daily_head_layout = str(daily_head_layout or DAILY_HEAD_LAYOUT_MONOLITHIC_V1).strip() or DAILY_HEAD_LAYOUT_MONOLITHIC_V1
    if resolved_daily_head_layout not in DAILY_HEAD_LAYOUT_CHOICES:
        raise ValueError(
            f"Unsupported daily head layout: {resolved_daily_head_layout}. Available: {', '.join(DAILY_HEAD_LAYOUT_CHOICES)}"
        )
    sample_scalar_loss_weights = dict(loss_config["sample_scalar_loss_weights"])
    daily_target_loss_weights = dict(loss_config["daily_target_loss_weights"])
    multi_objective_loss_weights = dict(loss_config["multi_objective_loss_weights"])
    funding_release_loss_variant = (
        "v11"
        if resolved_loss_profile
        in {
            "alpha_result_value_budget_split_v11",
            "alpha_result_value_budget_split_v12",
            "alpha_result_value_budget_split_v13",
        }
        else "v10"
    )

    sequence_base_names, sequence_columns = resolve_sequence_columns(feature_names)
    static_feature_names = [name for name in feature_names if name not in set(sequence_columns)]
    if not static_feature_names:
        raise ValueError("formal_torch_seq_v3 requires at least one static feature.")

    X_static, static_fill, static_means, static_stds = _prepare_matrix(sample_frame, static_feature_names)
    X_sequence_flat, sequence_fill, sequence_means, sequence_stds = _prepare_matrix(sample_frame, sequence_columns)
    X_sequence = _reshape_sequence_matrix(
        X_sequence_flat,
        steps=len(SEQUENCE_STEP_ORDER),
        feature_dim=len(sequence_base_names),
    )
    X_daily, daily_fill, daily_means, daily_stds = _prepare_matrix(daily_frame, daily_feature_names)

    action_lookup = {name: idx for idx, name in enumerate(ACTION_CLASSES)}
    duration_lookup = {name: idx for idx, name in enumerate(DURATION_CLASSES)}
    y_action = np.asarray([action_lookup.get(str(value), 0) for value in sample_frame["action_label"].astype(str)], dtype=np.int64)
    y_duration = np.asarray([
        duration_lookup.get(str(value), 0)
        for value in sample_frame["planned_holding_bucket"].astype(str).where(sample_frame["planned_holding_bucket"].astype(str).isin(DURATION_CLASSES), "avoid")
    ], dtype=np.int64)
    y_action_soft = _build_action_soft_targets(sample_frame)
    sample_dates = pd.to_datetime(sample_frame["date"]).dt.strftime("%Y-%m-%d")
    daily_target_maps: dict[str, dict[str, float]] = {}
    if "date" in daily_frame.columns:
        daily_dates = pd.to_datetime(daily_frame["date"]).dt.strftime("%Y-%m-%d")
        for daily_target_name in (
            "gross_exposure_target",
            "candidate_budget",
            "turnover_budget",
            "max_position_weight_target",
            "budget_cash_timing_signal_target",
        ):
            if daily_target_name in daily_frame.columns:
                daily_target_maps[daily_target_name] = dict(
                    zip(
                        daily_dates,
                        daily_frame[daily_target_name].astype(float),
                    )
                )

    def _sample_daily_target_values(name: str, default: float) -> np.ndarray:
        if name in sample_frame.columns:
            series = sample_frame[name].astype(float)
        elif name in daily_target_maps:
            series = sample_dates.map(daily_target_maps[name]).fillna(default).astype(float)
        else:
            series = pd.Series(np.full(len(sample_frame), default), index=sample_frame.index, dtype=float)
        return series.to_numpy(dtype=np.float32)

    sample_targets = {
        "target_delta_hint": sample_frame["target_delta_hint"].astype(float).to_numpy(dtype=np.float32),
        "entry_quality": sample_frame["entry_quality"].astype(float).to_numpy(dtype=np.float32),
        "hold_quality": sample_frame["hold_quality"].astype(float).to_numpy(dtype=np.float32),
        "add_quality": sample_frame["add_quality"].astype(float).to_numpy(dtype=np.float32),
        "reduce_quality": sample_frame["reduce_quality"].astype(float).to_numpy(dtype=np.float32),
        "exit_urgency": sample_frame["exit_urgency"].astype(float).to_numpy(dtype=np.float32),
        "reentry_readiness": sample_frame["reentry_readiness"].astype(float).to_numpy(dtype=np.float32),
        "holding_days_ratio": np.clip(
            sample_frame["planned_holding_days"].astype(float).to_numpy(dtype=np.float32) / float(MAX_CONTINUOUS_HOLDING_DAYS),
            0.0,
            1.0,
        ),
        "reduce_fraction": np.clip(
            sample_frame.get("reduce_fraction_target", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "exit_hazard": np.clip(
            sample_frame.get("exit_hazard_target", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "sell_attribution_score": np.clip(
            sample_frame.get("sell_attribution_score", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "sell_rank_score": np.clip(
            sample_frame.get("sell_rank_score", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "lifecycle_sell_gate": np.clip(
            sample_frame.get("lifecycle_sell_gate", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "large_upside_1d_target": np.clip(
            sample_frame.get("large_upside_1d_target", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "alpha_opportunity_value": np.clip(
            sample_frame.get("alpha_opportunity_value", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "hold_continuation_value": np.clip(
            sample_frame.get("hold_continuation_value", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "sell_release_value": np.clip(
            sample_frame.get("sell_release_value", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "cash_defense_value": np.clip(
            sample_frame.get("cash_defense_value", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "deployment_opportunity_cost": np.clip(
            sample_frame.get("deployment_opportunity_cost", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "risk_adjusted_action_value": np.clip(
            sample_frame.get("risk_adjusted_action_value", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "multi_horizon_forward_value": np.clip(
            sample_frame.get("multi_horizon_forward_value", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "multi_horizon_forward_risk": np.clip(
            sample_frame.get("multi_horizon_forward_risk", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "multi_horizon_path_value": np.clip(
            sample_frame.get("multi_horizon_path_value", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "open_action_value": np.clip(
            sample_frame.get("open_action_value", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "add_action_value": np.clip(
            sample_frame.get("add_action_value", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "hold_action_value": np.clip(
            sample_frame.get("hold_action_value", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "reduce_action_value": np.clip(
            sample_frame.get("reduce_action_value", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "exit_action_value": np.clip(
            sample_frame.get("exit_action_value", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "relative_opportunity_value": np.clip(
            sample_frame.get("relative_opportunity_value", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "action_value_consistency_target": np.clip(
            sample_frame.get("action_value_consistency_target", pd.Series(np.full(len(sample_frame), 0.5), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "value_arbitration_target": np.clip(
            sample_frame.get("value_arbitration_target", pd.Series(np.full(len(sample_frame), 0.5), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "deploy_value_target": np.clip(
            sample_frame.get("deploy_value_target", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "release_value_target": np.clip(
            sample_frame.get("release_value_target", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "defense_value_target": np.clip(
            sample_frame.get("defense_value_target", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "deploy_gate_target": np.clip(
            sample_frame.get("deploy_gate_target", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "release_gate_target": np.clip(
            sample_frame.get("release_gate_target", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "defense_gate_target": np.clip(
            sample_frame.get("defense_gate_target", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "deploy_executability_target": np.clip(
            sample_frame.get("deploy_executability_target", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_receiver_add_headroom": np.clip(
            sample_frame.get("portfolio_daily_receiver_add_headroom", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_receiver_min_add_delta": np.clip(
            sample_frame.get("portfolio_daily_receiver_min_add_delta", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_receiver_add_capacity": np.clip(
            sample_frame.get("portfolio_daily_receiver_add_capacity", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_receiver_executability": np.clip(
            sample_frame.get("portfolio_daily_receiver_executability", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_receiver_score": np.clip(
            sample_frame.get("portfolio_daily_receiver_score", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_source_release_capacity": np.clip(
            sample_frame.get("portfolio_daily_source_release_capacity", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_source_forward_spread_score": np.clip(
            sample_frame.get("portfolio_daily_source_forward_spread_score", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_source_bad_forward_spread_risk": np.clip(
            sample_frame.get("portfolio_daily_source_bad_forward_spread_risk", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_source_economic_release_score": np.clip(
            sample_frame.get("portfolio_daily_source_economic_release_score", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_source_economic_block_risk": np.clip(
            sample_frame.get("portfolio_daily_source_economic_block_risk", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_source_forward_strength_brake_risk": np.clip(
            sample_frame.get("portfolio_daily_source_forward_strength_brake_risk", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_source_forward_proxy_keep_risk": np.clip(
            sample_frame.get("portfolio_daily_source_forward_proxy_keep_risk", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_source_release_quality": np.clip(
            sample_frame.get("portfolio_daily_source_release_quality", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_source_opportunity_cost": np.clip(
            sample_frame.get("portfolio_daily_source_opportunity_cost", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_source_executability": np.clip(
            sample_frame.get("portfolio_daily_source_executability", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_source_score": np.clip(
            sample_frame.get("portfolio_daily_source_score", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_cash_score": np.clip(
            sample_frame.get("portfolio_daily_cash_score", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_receiver_funding_coverage": np.clip(
            sample_frame.get("portfolio_daily_receiver_funding_coverage", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_funding_closure_score": np.clip(
            sample_frame.get("portfolio_daily_funding_closure_score", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_allocation_transfer_score": np.clip(
            sample_frame.get("portfolio_daily_allocation_transfer_score", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_allocation_dead_branch_risk": np.clip(
            sample_frame.get("portfolio_daily_allocation_dead_branch_risk", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "market_downside_pressure": np.clip(
            sample_frame.get("market_downside_pressure", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "cash_regime_pressure": np.clip(
            sample_frame.get("cash_regime_pressure", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_cash_pressure": np.clip(
            sample_frame.get("portfolio_cash_pressure", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_drawdown_20d": np.clip(
            sample_frame.get("portfolio_drawdown_20d", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            -1.0,
            0.25,
        ),
        "forward_benchmark_return_1d": np.clip(
            sample_frame.get("forward_benchmark_return_1d", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            -0.25,
            0.25,
        ),
        "forward_benchmark_return_3d": np.clip(
            sample_frame.get("forward_benchmark_return_3d", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            -0.35,
            0.35,
        ),
        "gross_exposure_target": np.clip(
            _sample_daily_target_values("gross_exposure_target", 0.60),
            0.0,
            1.0,
        ),
        "candidate_budget": np.clip(
            _sample_daily_target_values("candidate_budget", 4.0),
            1.0,
            30.0,
        ),
        "turnover_budget": np.clip(
            _sample_daily_target_values("turnover_budget", 0.42),
            0.04,
            1.0,
        ),
        "max_position_weight_target": np.clip(
            _sample_daily_target_values("max_position_weight_target", 0.20),
            0.04,
            0.40,
        ),
        "budget_cash_timing_signal_target": np.clip(
            _sample_daily_target_values("budget_cash_timing_signal_target", 0.0),
            0.0,
            1.0,
        ),
        "current_weight": np.clip(
            sample_frame.get("current_weight", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "forward_excess_5d": np.clip(
            sample_frame.get("forward_excess_5d", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            -0.35,
            0.35,
        ),
        "portfolio_daily_receiver_forward_excess_5d": np.clip(
            sample_frame.get("portfolio_daily_receiver_forward_excess_5d", sample_frame.get("forward_excess_5d", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index))).astype(float).to_numpy(dtype=np.float32),
            -0.35,
            0.35,
        ),
        "portfolio_daily_source_forward_excess_5d": np.clip(
            sample_frame.get("portfolio_daily_source_forward_excess_5d", sample_frame.get("forward_excess_5d", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index))).astype(float).to_numpy(dtype=np.float32),
            -0.35,
            0.35,
        ),
        "portfolio_daily_unified_receiver_score": np.clip(
            sample_frame.get("portfolio_daily_unified_receiver_score", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_unified_source_score": np.clip(
            sample_frame.get("portfolio_daily_unified_source_score", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_unified_cash_score": np.clip(
            sample_frame.get("portfolio_daily_unified_cash_score", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_source_positive_forward_penalty": np.clip(
            sample_frame.get("portfolio_daily_source_positive_forward_penalty", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_source_opportunity_cost_penalty": np.clip(
            sample_frame.get("portfolio_daily_source_opportunity_cost_penalty", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_source_strong_false_sell_penalty": np.clip(
            sample_frame.get("portfolio_daily_source_strong_false_sell_penalty", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_source_hard_negative_penalty": np.clip(
            sample_frame.get("portfolio_daily_source_hard_negative_penalty", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_source_tail_false_sell_penalty": np.clip(
            sample_frame.get("portfolio_daily_source_tail_false_sell_penalty", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_source_release_preference": np.clip(
            sample_frame.get("portfolio_daily_source_release_preference", sample_frame.get("portfolio_daily_unified_source_score", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index))).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_receiver_source_spread_reward": np.clip(
            sample_frame.get("portfolio_daily_receiver_source_spread_reward", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_transfer_regret_target": np.clip(
            sample_frame.get("portfolio_daily_transfer_regret_target", sample_frame.get("portfolio_daily_unified_allocation_objective", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index))).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_unified_allocation_objective": np.clip(
            sample_frame.get("portfolio_daily_unified_allocation_objective", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_allocation_trade_quality_target": np.clip(
            sample_frame.get("portfolio_daily_allocation_trade_quality_target", sample_frame.get("portfolio_daily_unified_allocation_objective", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index))).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_allocation_cash_deployment_target": np.clip(
            sample_frame.get("portfolio_daily_allocation_cash_deployment_target", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_allocation_risk_adjusted_return_target": np.clip(
            sample_frame.get("portfolio_daily_allocation_risk_adjusted_return_target", sample_frame.get("portfolio_daily_unified_allocation_objective", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index))).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_allocation_drawdown_control_target": np.clip(
            sample_frame.get("portfolio_daily_allocation_drawdown_control_target", sample_frame.get("portfolio_daily_unified_cash_score", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index))).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_allocation_monthly_quality_target": np.clip(
            sample_frame.get("portfolio_daily_allocation_monthly_quality_target", sample_frame.get("portfolio_daily_unified_allocation_objective", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index))).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_allocation_final_objective": np.clip(
            sample_frame.get("portfolio_daily_allocation_final_objective", sample_frame.get("portfolio_daily_unified_allocation_objective", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index))).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_allocation_uncertainty_pressure_target": np.clip(
            sample_frame.get("portfolio_daily_allocation_uncertainty_pressure_target", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_allocation_tail_risk_control_target": np.clip(
            sample_frame.get("portfolio_daily_allocation_tail_risk_control_target", sample_frame.get("portfolio_daily_allocation_drawdown_control_target", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index))).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_allocation_decision_focused_objective": np.clip(
            sample_frame.get("portfolio_daily_allocation_decision_focused_objective", sample_frame.get("portfolio_daily_allocation_final_objective", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index))).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_allocation_net_utility_target": np.clip(
            sample_frame.get("portfolio_daily_allocation_net_utility_target", sample_frame.get("portfolio_daily_allocation_final_objective", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index))).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_allocation_credit_closure_target": np.clip(
            sample_frame.get("portfolio_daily_allocation_credit_closure_target", sample_frame.get("portfolio_daily_funding_closure_score", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index))).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_allocation_resource_efficiency_target": np.clip(
            sample_frame.get("portfolio_daily_allocation_resource_efficiency_target", sample_frame.get("portfolio_daily_allocation_transfer_score", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index))).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_receiver_candidate_mask": np.clip(
            sample_frame.get("portfolio_daily_receiver_candidate_mask", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_source_candidate_mask": np.clip(
            sample_frame.get("portfolio_daily_source_candidate_mask", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_receiver_executable_candidate": np.clip(
            sample_frame.get(
                "portfolio_daily_receiver_executable_candidate",
                sample_frame.get("portfolio_daily_receiver_candidate_mask", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)),
            ).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "portfolio_daily_source_executable_candidate": np.clip(
            sample_frame.get(
                "portfolio_daily_source_executable_candidate",
                sample_frame.get("portfolio_daily_source_candidate_mask", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)),
            ).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "clipped_intent_risk": np.clip(
            sample_frame.get("clipped_intent_risk", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index)).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "holding_flag_target": np.clip(
            sample_frame.get("holding_flag_target", sample_frame.get("holding_flag", pd.Series(np.zeros(len(sample_frame)), index=sample_frame.index))).astype(float).to_numpy(dtype=np.float32),
            0.0,
            1.0,
        ),
        "date_code": pd.factorize(pd.to_datetime(sample_frame["date"]).dt.strftime("%Y-%m-%d"))[0].astype(np.float32),
    }
    current_weight_target = np.clip(sample_targets["current_weight"], 0.0, 1.0)
    receiver_candidate_target = np.clip(sample_targets["portfolio_daily_receiver_candidate_mask"], 0.0, 1.0)
    source_candidate_target = np.clip(sample_targets["portfolio_daily_source_candidate_mask"], 0.0, 1.0)
    holding_target = np.clip(sample_targets["holding_flag_target"], 0.0, 1.0)
    liquidity_default = np.clip(
        sample_frame.get(
            "portfolio_daily_liquidity_support",
            sample_frame.get(
                "liquidity_score",
                sample_frame.get("tradability_score", pd.Series(np.ones(len(sample_frame)), index=sample_frame.index)),
            ),
        ).astype(float).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    sample_targets["portfolio_daily_liquidity_support"] = liquidity_default
    sample_targets["portfolio_daily_impact_cost"] = np.clip(
        sample_frame.get(
            "portfolio_daily_impact_cost",
            pd.Series(0.0015 + 0.0060 * (1.0 - liquidity_default), index=sample_frame.index),
        ).astype(float).to_numpy(dtype=np.float32),
        0.0005,
        0.0400,
    )
    sample_targets["portfolio_daily_factor_concentration_proxy"] = np.clip(
        sample_frame.get(
            "portfolio_daily_factor_concentration_proxy",
            sample_frame.get(
                "industry_concentration_proxy",
                sample_frame.get("style_concentration_proxy", pd.Series(np.full(len(sample_frame), 0.10), index=sample_frame.index)),
            ),
        ).astype(float).to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    sample_targets["portfolio_daily_behavior_propensity"] = np.clip(
        sample_frame.get(
            "portfolio_daily_behavior_propensity",
            pd.Series(
                0.06
                + 0.58 * np.clip(current_weight_target * 4.0, 0.0, 1.0)
                + 0.16 * receiver_candidate_target
                + 0.14 * source_candidate_target
                + 0.10 * holding_target,
                index=sample_frame.index,
            ),
        ).astype(float).to_numpy(dtype=np.float32),
        0.02,
        1.0,
    )
    daily_targets = {
        name: daily_frame[name].astype(float).to_numpy(dtype=np.float32)
        for name in ALL_DAILY_TARGET_NAMES
        if name in daily_frame.columns
    }
    if resolved_daily_head_layout == DAILY_HEAD_LAYOUT_MONOLITHIC_V1:
        daily_targets = {
            name: values
            for name, values in daily_targets.items()
            if name in {
                "gross_exposure_target",
                "candidate_budget",
                "turnover_budget",
                "max_position_weight_target",
                "hold_bias_target",
            }
        }
    if resolved_daily_head_layout == DAILY_HEAD_LAYOUT_SPLIT_V2:
        required_daily_targets = {
            "gross_exposure_target",
            "candidate_budget",
            "turnover_budget",
            "max_position_weight_target",
            "hold_bias_target",
            "reduce_bias_target",
            "exit_patience_target",
            "reentry_guard_target",
            "budget_risk_signal_target",
            "budget_deploy_signal_target",
            "budget_cash_timing_signal_target",
            "budget_alpha_focus_signal_target",
        }
        missing_daily_targets = sorted(required_daily_targets - set(daily_targets))
        if missing_daily_targets:
            raise ValueError(
                "split_v2 daily controller requires expanded daily targets, missing columns: "
                + ", ".join(missing_daily_targets)
            )

    train_idx, val_idx = _split_indices(len(sample_frame), random_seed)
    daily_train_idx, daily_val_idx = _split_indices(len(daily_frame), random_seed + 17)

    uses_day_set_native_allocation = (
        multi_objective_loss_weights.get("portfolio_day_set_native_allocation_vector_total", 0.0) > 0.0
    )
    if uses_day_set_native_allocation:
        sample_model = TemporalDaySetPolicyNet(
            static_input_dim=len(static_feature_names),
            sequence_feature_dim=len(sequence_base_names),
            sequence_steps=len(SEQUENCE_STEP_ORDER),
            daily_input_dim=len(daily_feature_names),
            hidden_dim=hidden_dim,
            sequence_hidden_dim=sequence_hidden_dim,
            sequence_layers=sequence_layers,
            slot_count=32,
            slot_dim=128,
            dropout=dropout,
        ).to(device)
    else:
        sample_model = TemporalSamplePolicyNet(
            static_input_dim=len(static_feature_names),
            sequence_feature_dim=len(sequence_base_names),
            sequence_steps=len(SEQUENCE_STEP_ORDER),
            hidden_dim=hidden_dim,
            sequence_hidden_dim=sequence_hidden_dim,
            sequence_layers=sequence_layers,
            dropout=dropout,
        ).to(device)
    daily_model = DailyControllerNet(
        len(daily_feature_names),
        hidden_dim=daily_hidden_dim,
        dropout=daily_dropout,
        head_layout=resolved_daily_head_layout,
    ).to(device)
    sample_optimizer = torch.optim.AdamW(sample_model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    daily_optimizer = torch.optim.AdamW(daily_model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    action_weight_tensor = _action_weights(sample_frame).to(device)

    signature_payload = _signature_payload(
        feature_names=feature_names,
        daily_feature_names=daily_feature_names,
        train_summary=dict(train_summary or {}),
        training_contract=contract,
    )
    signature_payload["sequence_model_revision"] = "seq_v3_three_value_gate_r6b"
    signature_payload["loss_profile"] = resolved_loss_profile
    signature_payload["daily_head_layout"] = resolved_daily_head_layout
    signature_payload["sample_scalar_loss_weights"] = dict(sample_scalar_loss_weights)
    signature_payload["daily_target_loss_weights"] = dict(daily_target_loss_weights)
    signature_payload["multi_objective_loss_weights"] = dict(multi_objective_loss_weights)
    signature_hash = _signature_hash(signature_payload)
    checkpoint_last = run_root / "checkpoint_last.pt"
    checkpoint_best = run_root / "checkpoint_best.pt"
    artifact_path = run_root / "continuous_policy_v3_seq_artifact.pt"
    diagnostics_path = run_root / "training_diagnostics.json"

    start_epoch = 0
    best_epoch = 0
    best_val_loss = math.inf
    resumed_from = ""
    history: list[dict[str, float | int]] = []
    if checkpoint_last.exists() and resume_mode == "strict":
        state = torch.load(checkpoint_last, map_location="cpu", weights_only=False)
        previous_hash = str(state.get("signature_hash", "") or "")
        if previous_hash and previous_hash != signature_hash:
            raise RuntimeError("continuous_policy v3 strict resume rejected because checkpoint lineage does not match the current run signature.")
        sample_model.load_state_dict(state["sample_model_state_dict"])
        daily_model.load_state_dict(state["daily_model_state_dict"])
        sample_optimizer.load_state_dict(state["sample_optimizer_state_dict"])
        daily_optimizer.load_state_dict(state["daily_optimizer_state_dict"])
        start_epoch = int(state.get("epoch", 0) or 0)
        best_epoch = int(state.get("best_epoch", 0) or 0)
        best_val_loss = float(state.get("best_val_loss", math.inf) or math.inf)
        history = list(state.get("history", []) or [])
        resumed_from = str(checkpoint_last.resolve())
    if resume_mode == "strict" and checkpoint_last.exists() and int(epochs) <= int(start_epoch):
        raise RuntimeError(f"continuous_policy v3 strict resume requires epochs > completed epochs ({start_epoch}), got {epochs}.")

    uses_cvxpy_convex_allocation_layer = (
        multi_objective_loss_weights.get("portfolio_cvxpy_convex_allocation_total", 0.0) > 0.0
        or multi_objective_loss_weights.get("portfolio_full_universe_convex_allocation_total", 0.0) > 0.0
    )
    full_universe_train_solver_enabled = (
        CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_SOLVER_ENABLED
        or _loss_profile_enables_full_universe_train_solver(resolved_loss_profile)
    )
    if uses_cvxpy_convex_allocation_layer:
        _get_portfolio_cvxpy_allocation_layer()
        if multi_objective_loss_weights.get("portfolio_full_universe_convex_allocation_total", 0.0) > 0.0:
            _get_portfolio_cvxpy_allocation_layer(CVXPY_FULL_UNIVERSE_ALLOCATION_SLOT_COUNT)
    sample_target_names = tuple(sample_targets.keys())
    dataset = TensorDataset(
        torch.as_tensor(X_static[train_idx], dtype=torch.float32),
        torch.as_tensor(X_sequence[train_idx], dtype=torch.float32),
        torch.as_tensor(y_action[train_idx], dtype=torch.long),
        torch.as_tensor(y_duration[train_idx], dtype=torch.long),
        torch.as_tensor(y_action_soft[train_idx], dtype=torch.float32),
        *[torch.as_tensor(sample_targets[name][train_idx], dtype=torch.float32) for name in sample_target_names],
    )
    if uses_day_set_native_allocation:
        day_dataset = DaySetTensorDataset(
            date_codes=torch.as_tensor(sample_targets["date_code"], dtype=torch.long),
            static_x=torch.as_tensor(X_static, dtype=torch.float32),
            sequence_x=torch.as_tensor(X_sequence, dtype=torch.float32),
            action=torch.as_tensor(y_action, dtype=torch.long),
            duration=torch.as_tensor(y_duration, dtype=torch.long),
            action_soft=torch.as_tensor(y_action_soft, dtype=torch.float32),
            sample_targets={name: torch.as_tensor(sample_targets[name], dtype=torch.float32) for name in sample_target_names},
            daily_x=torch.as_tensor(X_daily, dtype=torch.float32),
            daily_targets={name: torch.as_tensor(values, dtype=torch.float32) for name, values in daily_targets.items()},
            allowed_date_codes=daily_train_idx,
        )
        loader = DataLoader(
            day_dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=True,
            drop_last=False,
            collate_fn=_collate_day_set_batch,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=max(32, int(batch_size)),
            shuffle=not uses_cvxpy_convex_allocation_layer,
            drop_last=False,
        )
    X_static_val = torch.as_tensor(X_static[val_idx], dtype=torch.float32, device=device)
    X_sequence_val = torch.as_tensor(X_sequence[val_idx], dtype=torch.float32, device=device)
    y_action_val = torch.as_tensor(y_action[val_idx], dtype=torch.long, device=device)
    y_duration_val = torch.as_tensor(y_duration[val_idx], dtype=torch.long, device=device)
    y_action_soft_val = torch.as_tensor(y_action_soft[val_idx], dtype=torch.float32, device=device)
    val_targets = {name: torch.as_tensor(values[val_idx], dtype=torch.float32, device=device) for name, values in sample_targets.items()}
    X_daily_train = torch.as_tensor(X_daily[daily_train_idx], dtype=torch.float32, device=device)
    X_daily_val = torch.as_tensor(X_daily[daily_val_idx], dtype=torch.float32, device=device)
    daily_targets_train = {name: torch.as_tensor(values[daily_train_idx], dtype=torch.float32, device=device) for name, values in daily_targets.items()}
    daily_targets_val = {name: torch.as_tensor(values[daily_val_idx], dtype=torch.float32, device=device) for name, values in daily_targets.items()}

    patience_used = 0
    for epoch in range(start_epoch + 1, int(epochs) + 1):
        sample_model.train()
        daily_model.train()
        epoch_sample_loss = 0.0
        batch_count = 0
        for batch in loader:
            day_set_outputs = None
            day_set_targets = None
            day_set_mask = None
            if uses_day_set_native_allocation:
                day_batch = {
                    key: (value.to(device) if torch.is_tensor(value) else value)
                    for key, value in batch.items()
                }
                day_batch["sample_targets"] = {
                    name: value.to(device)
                    for name, value in day_batch["sample_targets"].items()
                }
                day_set_mask = day_batch["sample_mask"].to(device)
                day_set_outputs = sample_model(
                    day_batch["static_x"],
                    day_batch["sequence_x"],
                    day_batch["daily_x"],
                    day_set_mask,
                )
                flat_mask = day_set_mask.reshape(-1)
                outputs = {
                    name: (
                        value.reshape(-1, value.shape[-1])[flat_mask]
                        if value.ndim == 3
                        else value.reshape(-1)[flat_mask]
                        if value.ndim == 2
                        else value.repeat_interleave(day_set_mask.shape[1])[flat_mask]
                        if value.ndim == 1 and value.numel() == day_set_mask.shape[0]
                        else value
                    )
                    for name, value in day_set_outputs.items()
                }
                action_batch = day_batch["action"].reshape(-1)[flat_mask]
                duration_batch = day_batch["duration"].reshape(-1)[flat_mask]
                action_soft_batch = day_batch["action_soft"].reshape(-1, day_batch["action_soft"].shape[-1])[flat_mask]
                sample_batch_targets = {
                    name: tensor.reshape(-1)[flat_mask]
                    for name, tensor in day_batch["sample_targets"].items()
                }
                day_set_targets = dict(day_batch["sample_targets"])
            else:
                batch = [item.to(device) for item in batch]
                static_batch, sequence_batch, action_batch, duration_batch, action_soft_batch, *sample_target_batches = batch
                sample_batch_targets = {
                    name: tensor
                    for name, tensor in zip(sample_target_names, sample_target_batches, strict=False)
                }
                outputs = sample_model(static_batch, sequence_batch)
            action_ce_loss = nn.functional.cross_entropy(outputs["action_logits"], action_batch, weight=action_weight_tensor)
            action_soft_loss = nn.functional.kl_div(
                nn.functional.log_softmax(outputs["action_logits"], dim=-1),
                action_soft_batch,
                reduction="batchmean",
            )
            action_loss = (
                multi_objective_loss_weights["action_hard"] * action_ce_loss
                + multi_objective_loss_weights["action_soft"] * action_soft_loss
            )
            duration_loss = nn.functional.cross_entropy(outputs["duration_logits"], duration_batch)
            scalar_loss = _weighted_scalar_heads_loss(
                outputs,
                sample_batch_targets,
                sample_scalar_loss_weights,
            )
            arbitration_loss = _lifecycle_arbitration_loss(outputs, sample_batch_targets)
            sell_rank_pairwise_loss = _sell_rank_pairwise_loss(outputs, sample_batch_targets)
            portfolio_receiver_pairwise_loss = _date_rank_pairwise_loss(
                outputs,
                sample_batch_targets,
                score_name="portfolio_daily_receiver_score",
                target_name="portfolio_daily_receiver_score",
                candidate_mask_name="portfolio_daily_receiver_candidate_mask",
            )
            portfolio_source_pairwise_loss = _date_rank_pairwise_loss(
                outputs,
                sample_batch_targets,
                score_name="portfolio_daily_source_score",
                target_name="portfolio_daily_source_score",
                candidate_mask_name="portfolio_daily_source_candidate_mask",
            )
            portfolio_cash_margin_loss = _portfolio_cash_margin_loss(outputs, sample_batch_targets)
            portfolio_unified_allocation_loss = _unified_allocation_consistency_loss(outputs, sample_batch_targets)
            portfolio_decision_regret_loss = _decision_focused_allocation_regret_loss(outputs, sample_batch_targets)
            portfolio_source_hard_negative_tail_loss = _source_hard_negative_tail_loss(outputs, sample_batch_targets)
            portfolio_source_listwise_release_loss = _source_listwise_release_regret_loss(outputs, sample_batch_targets)
            portfolio_transfer_regret_loss = _transfer_level_allocation_regret_loss(outputs, sample_batch_targets)
            portfolio_allocation_objective_consolidation_loss = _allocation_objective_consolidation_loss(
                outputs,
                sample_batch_targets,
            )
            portfolio_risk_sensitive_allocation_loss = _risk_sensitive_allocation_objective_loss(
                outputs,
                sample_batch_targets,
            )
            portfolio_utility_credit_closure_loss = _portfolio_utility_credit_closure_loss(
                outputs,
                sample_batch_targets,
            )
            portfolio_primal_dual_decision_loss = _portfolio_primal_dual_decision_loss(
                outputs,
                sample_batch_targets,
            )
            portfolio_entropic_transport_decision_loss = _portfolio_entropic_transport_decision_loss(
                outputs,
                sample_batch_targets,
            )
            portfolio_offline_conservative_support_loss = _portfolio_offline_conservative_support_loss(
                outputs,
                sample_batch_targets,
            )
            portfolio_differentiable_convex_allocation_loss = (
                _portfolio_differentiable_convex_allocation_loss(
                    outputs,
                    sample_batch_targets,
                )
                if multi_objective_loss_weights.get("portfolio_differentiable_convex_allocation_total", 0.0) > 0.0
                else torch.tensor(0.0, device=device)
            )
            portfolio_cvxpy_convex_allocation_loss = (
                _portfolio_cvxpy_convex_allocation_loss(
                    outputs,
                    sample_batch_targets,
                )
                if multi_objective_loss_weights.get("portfolio_cvxpy_convex_allocation_total", 0.0) > 0.0
                else torch.tensor(0.0, device=device)
            )
            portfolio_full_universe_convex_allocation_loss = (
                _portfolio_full_universe_convex_allocation_loss(
                    outputs,
                    sample_batch_targets,
                    enable_solver=(
                        full_universe_train_solver_enabled
                        and batch_count % max(1, CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_BATCH_INTERVAL) == 0
                    ),
                )
                if multi_objective_loss_weights.get("portfolio_full_universe_convex_allocation_total", 0.0) > 0.0
                else torch.tensor(0.0, device=device)
            )
            portfolio_native_allocation_vector_loss = (
                _portfolio_native_allocation_vector_loss(
                    outputs,
                    sample_batch_targets,
                )
                if multi_objective_loss_weights.get("portfolio_native_allocation_vector_total", 0.0) > 0.0
                else torch.tensor(0.0, device=device)
            )
            portfolio_day_set_native_allocation_vector_loss = (
                _portfolio_day_set_native_allocation_vector_loss(
                    day_set_outputs,
                    day_set_targets,
                    day_set_mask,
                )
                if (
                    day_set_outputs is not None
                    and day_set_targets is not None
                    and day_set_mask is not None
                    and multi_objective_loss_weights.get("portfolio_day_set_native_allocation_vector_total", 0.0) > 0.0
                )
                else torch.tensor(0.0, device=device)
            )
            portfolio_capital_flow_closure_loss = (
                _portfolio_capital_flow_closure_loss(
                    outputs,
                    sample_batch_targets,
                )
                if multi_objective_loss_weights.get("portfolio_capital_flow_closure_total", 0.0) > 0.0
                else torch.tensor(0.0, device=device)
            )
            value_arbitration_loss = _value_arbitration_consistency_loss(outputs, sample_batch_targets)
            three_value_gate_loss = _three_value_gate_consistency_loss(outputs, sample_batch_targets)
            hierarchical_three_value_gate_loss = _hierarchical_three_value_gate_consistency_loss(outputs, sample_batch_targets)
            funding_release_loss = _funding_release_discipline_loss(
                outputs,
                sample_batch_targets,
                variant=funding_release_loss_variant,
            )
            action_value_consistency_loss = _action_value_consistency_loss(outputs, sample_batch_targets)
            clipped_intent_loss = (
                nn.functional.binary_cross_entropy(
                    torch.clamp(outputs["clipped_intent_risk"], 1.0e-4, 1.0 - 1.0e-4),
                    torch.clamp(sample_batch_targets["clipped_intent_risk"], 0.0, 1.0),
                )
                if "clipped_intent_risk" in sample_batch_targets
                else torch.tensor(0.0, device=device)
            )
            loss = (
                multi_objective_loss_weights["action_total"] * action_loss
                + multi_objective_loss_weights["duration_total"] * duration_loss
                + multi_objective_loss_weights["scalar_total"] * scalar_loss
                + multi_objective_loss_weights.get("arbitration_total", 0.0) * arbitration_loss
                + multi_objective_loss_weights.get("sell_rank_pairwise_total", 0.0) * sell_rank_pairwise_loss
                + multi_objective_loss_weights.get("portfolio_receiver_pairwise_total", 0.0) * portfolio_receiver_pairwise_loss
                + multi_objective_loss_weights.get("portfolio_source_pairwise_total", 0.0) * portfolio_source_pairwise_loss
                + multi_objective_loss_weights.get("portfolio_cash_margin_total", 0.0) * portfolio_cash_margin_loss
                + multi_objective_loss_weights.get("portfolio_unified_allocation_total", 0.0) * portfolio_unified_allocation_loss
                + multi_objective_loss_weights.get("portfolio_decision_regret_total", 0.0) * portfolio_decision_regret_loss
                + multi_objective_loss_weights.get("portfolio_source_hard_negative_tail_total", 0.0) * portfolio_source_hard_negative_tail_loss
                + multi_objective_loss_weights.get("portfolio_source_listwise_release_total", 0.0) * portfolio_source_listwise_release_loss
                + multi_objective_loss_weights.get("portfolio_transfer_regret_total", 0.0) * portfolio_transfer_regret_loss
                + multi_objective_loss_weights.get("portfolio_allocation_objective_consolidation_total", 0.0)
                * portfolio_allocation_objective_consolidation_loss
                + multi_objective_loss_weights.get("portfolio_risk_sensitive_allocation_total", 0.0)
                * portfolio_risk_sensitive_allocation_loss
                + multi_objective_loss_weights.get("portfolio_utility_credit_closure_total", 0.0)
                * portfolio_utility_credit_closure_loss
                + multi_objective_loss_weights.get("portfolio_primal_dual_decision_total", 0.0)
                * portfolio_primal_dual_decision_loss
                + multi_objective_loss_weights.get("portfolio_entropic_transport_decision_total", 0.0)
                * portfolio_entropic_transport_decision_loss
                + multi_objective_loss_weights.get("portfolio_offline_conservative_support_total", 0.0)
                * portfolio_offline_conservative_support_loss
                + multi_objective_loss_weights.get("portfolio_differentiable_convex_allocation_total", 0.0)
                * portfolio_differentiable_convex_allocation_loss
                + multi_objective_loss_weights.get("portfolio_cvxpy_convex_allocation_total", 0.0)
                * portfolio_cvxpy_convex_allocation_loss
                + multi_objective_loss_weights.get("portfolio_full_universe_convex_allocation_total", 0.0)
                * portfolio_full_universe_convex_allocation_loss
                + multi_objective_loss_weights.get("portfolio_native_allocation_vector_total", 0.0)
                * portfolio_native_allocation_vector_loss
                + multi_objective_loss_weights.get("portfolio_day_set_native_allocation_vector_total", 0.0)
                * portfolio_day_set_native_allocation_vector_loss
                + multi_objective_loss_weights.get("portfolio_capital_flow_closure_total", 0.0)
                * portfolio_capital_flow_closure_loss
                + multi_objective_loss_weights.get("value_arbitration_total", 0.0) * value_arbitration_loss
                + multi_objective_loss_weights.get("three_value_gate_total", 0.0) * three_value_gate_loss
                + multi_objective_loss_weights.get("hierarchical_three_value_total", 0.0) * hierarchical_three_value_gate_loss
                + multi_objective_loss_weights.get("funding_release_total", 0.0) * funding_release_loss
                + multi_objective_loss_weights.get("action_value_total", 0.0) * action_value_consistency_loss
                + multi_objective_loss_weights.get("clipped_intent_total", 0.0) * clipped_intent_loss
            )
            sample_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(sample_model.parameters(), max_norm=2.0)
            sample_optimizer.step()
            epoch_sample_loss += float(loss.detach().cpu())
            batch_count += 1

        daily_outputs = daily_model(X_daily_train)
        daily_loss = _weighted_scalar_heads_loss(daily_outputs, daily_targets_train, daily_target_loss_weights)
        daily_optimizer.zero_grad(set_to_none=True)
        daily_loss.backward()
        nn.utils.clip_grad_norm_(daily_model.parameters(), max_norm=2.0)
        daily_optimizer.step()

        sample_model.eval()
        daily_model.eval()
        with torch.no_grad():
            val_day_set_outputs = None
            val_day_set_targets = None
            val_day_set_mask = None
            if uses_day_set_native_allocation:
                val_day_set_mask = torch.ones(1, X_static_val.shape[0], dtype=torch.bool, device=device)
                val_daily_x_for_sample = X_daily_val[:1] if X_daily_val.shape[0] else torch.zeros(1, len(daily_feature_names), device=device)
                val_day_set_outputs = sample_model(
                    X_static_val.unsqueeze(0),
                    X_sequence_val.unsqueeze(0),
                    val_daily_x_for_sample,
                    val_day_set_mask,
                )
                val_outputs = {
                    name: (
                        value.squeeze(0)
                        if value.ndim >= 2
                        else value.repeat(X_static_val.shape[0])
                        if value.ndim == 1 and value.numel() == 1
                        else value
                    )
                    for name, value in val_day_set_outputs.items()
                }
                val_day_set_targets = {name: tensor.unsqueeze(0) for name, tensor in val_targets.items()}
            else:
                val_outputs = sample_model(X_static_val, X_sequence_val)
            val_action_ce_loss = nn.functional.cross_entropy(val_outputs["action_logits"], y_action_val, weight=action_weight_tensor)
            val_action_soft_loss = nn.functional.kl_div(
                nn.functional.log_softmax(val_outputs["action_logits"], dim=-1),
                y_action_soft_val,
                reduction="batchmean",
            )
            val_action_loss = (
                multi_objective_loss_weights["action_hard"] * val_action_ce_loss
                + multi_objective_loss_weights["action_soft"] * val_action_soft_loss
            )
            val_duration_loss = nn.functional.cross_entropy(val_outputs["duration_logits"], y_duration_val)
            val_scalar_loss = _weighted_scalar_heads_loss(val_outputs, val_targets, sample_scalar_loss_weights)
            val_arbitration_loss = _lifecycle_arbitration_loss(val_outputs, val_targets)
            val_sell_rank_pairwise_loss = _sell_rank_pairwise_loss(val_outputs, val_targets)
            val_portfolio_receiver_pairwise_loss = _date_rank_pairwise_loss(
                val_outputs,
                val_targets,
                score_name="portfolio_daily_receiver_score",
                target_name="portfolio_daily_receiver_score",
                candidate_mask_name="portfolio_daily_receiver_candidate_mask",
            )
            val_portfolio_source_pairwise_loss = _date_rank_pairwise_loss(
                val_outputs,
                val_targets,
                score_name="portfolio_daily_source_score",
                target_name="portfolio_daily_source_score",
                candidate_mask_name="portfolio_daily_source_candidate_mask",
            )
            val_portfolio_cash_margin_loss = _portfolio_cash_margin_loss(val_outputs, val_targets)
            val_portfolio_unified_allocation_loss = _unified_allocation_consistency_loss(val_outputs, val_targets)
            val_portfolio_decision_regret_loss = _decision_focused_allocation_regret_loss(val_outputs, val_targets)
            val_portfolio_source_hard_negative_tail_loss = _source_hard_negative_tail_loss(val_outputs, val_targets)
            val_portfolio_source_listwise_release_loss = _source_listwise_release_regret_loss(val_outputs, val_targets)
            val_portfolio_transfer_regret_loss = _transfer_level_allocation_regret_loss(val_outputs, val_targets)
            val_portfolio_allocation_objective_consolidation_loss = _allocation_objective_consolidation_loss(
                val_outputs,
                val_targets,
            )
            val_portfolio_risk_sensitive_allocation_loss = _risk_sensitive_allocation_objective_loss(
                val_outputs,
                val_targets,
            )
            val_portfolio_utility_credit_closure_loss = _portfolio_utility_credit_closure_loss(
                val_outputs,
                val_targets,
            )
            val_portfolio_primal_dual_decision_loss = _portfolio_primal_dual_decision_loss(
                val_outputs,
                val_targets,
            )
            val_portfolio_entropic_transport_decision_loss = _portfolio_entropic_transport_decision_loss(
                val_outputs,
                val_targets,
            )
            val_portfolio_offline_conservative_support_loss = _portfolio_offline_conservative_support_loss(
                val_outputs,
                val_targets,
            )
            val_portfolio_differentiable_convex_allocation_loss = (
                _portfolio_differentiable_convex_allocation_loss(
                    val_outputs,
                    val_targets,
                )
                if multi_objective_loss_weights.get("portfolio_differentiable_convex_allocation_total", 0.0) > 0.0
                else torch.tensor(0.0, device=device)
            )
            val_portfolio_cvxpy_convex_allocation_loss = (
                _portfolio_cvxpy_convex_allocation_loss(
                    val_outputs,
                    val_targets,
                )
                if multi_objective_loss_weights.get("portfolio_cvxpy_convex_allocation_total", 0.0) > 0.0
                else torch.tensor(0.0, device=device)
            )
            val_portfolio_full_universe_convex_allocation_loss = (
                _portfolio_full_universe_convex_allocation_loss(
                    val_outputs,
                    val_targets,
                    enable_solver=False,
                )
                if multi_objective_loss_weights.get("portfolio_full_universe_convex_allocation_total", 0.0) > 0.0
                else torch.tensor(0.0, device=device)
            )
            val_portfolio_native_allocation_vector_loss = (
                _portfolio_native_allocation_vector_loss(
                    val_outputs,
                    val_targets,
                )
                if multi_objective_loss_weights.get("portfolio_native_allocation_vector_total", 0.0) > 0.0
                else torch.tensor(0.0, device=device)
            )
            val_portfolio_day_set_native_allocation_vector_loss = (
                _portfolio_day_set_native_allocation_vector_loss(
                    val_day_set_outputs,
                    val_day_set_targets,
                    val_day_set_mask,
                )
                if (
                    val_day_set_outputs is not None
                    and val_day_set_targets is not None
                    and val_day_set_mask is not None
                    and multi_objective_loss_weights.get("portfolio_day_set_native_allocation_vector_total", 0.0) > 0.0
                )
                else torch.tensor(0.0, device=device)
            )
            val_portfolio_capital_flow_closure_loss = (
                _portfolio_capital_flow_closure_loss(
                    val_outputs,
                    val_targets,
                )
                if multi_objective_loss_weights.get("portfolio_capital_flow_closure_total", 0.0) > 0.0
                else torch.tensor(0.0, device=device)
            )
            val_value_arbitration_loss = _value_arbitration_consistency_loss(val_outputs, val_targets)
            val_three_value_gate_loss = _three_value_gate_consistency_loss(val_outputs, val_targets)
            val_hierarchical_three_value_gate_loss = _hierarchical_three_value_gate_consistency_loss(val_outputs, val_targets)
            val_funding_release_loss = _funding_release_discipline_loss(
                val_outputs,
                val_targets,
                variant=funding_release_loss_variant,
            )
            val_action_value_consistency_loss = _action_value_consistency_loss(val_outputs, val_targets)
            val_clipped_intent_loss = (
                nn.functional.binary_cross_entropy(
                    torch.clamp(val_outputs["clipped_intent_risk"], 1.0e-4, 1.0 - 1.0e-4),
                    torch.clamp(val_targets["clipped_intent_risk"], 0.0, 1.0),
                )
                if "clipped_intent_risk" in val_targets
                else torch.tensor(0.0, device=device)
            )
            val_daily_outputs = daily_model(X_daily_val)
            val_daily_loss = _weighted_scalar_heads_loss(val_daily_outputs, daily_targets_val, daily_target_loss_weights)
            val_loss = float(
                (
                    multi_objective_loss_weights["action_total"] * val_action_loss
                    + multi_objective_loss_weights["duration_total"] * val_duration_loss
                    + multi_objective_loss_weights["scalar_total"] * val_scalar_loss
                    + multi_objective_loss_weights["daily_total"] * val_daily_loss
                    + multi_objective_loss_weights.get("arbitration_total", 0.0) * val_arbitration_loss
                    + multi_objective_loss_weights.get("sell_rank_pairwise_total", 0.0) * val_sell_rank_pairwise_loss
                    + multi_objective_loss_weights.get("portfolio_receiver_pairwise_total", 0.0) * val_portfolio_receiver_pairwise_loss
                    + multi_objective_loss_weights.get("portfolio_source_pairwise_total", 0.0) * val_portfolio_source_pairwise_loss
                    + multi_objective_loss_weights.get("portfolio_cash_margin_total", 0.0) * val_portfolio_cash_margin_loss
                    + multi_objective_loss_weights.get("portfolio_unified_allocation_total", 0.0) * val_portfolio_unified_allocation_loss
                    + multi_objective_loss_weights.get("portfolio_decision_regret_total", 0.0) * val_portfolio_decision_regret_loss
                    + multi_objective_loss_weights.get("portfolio_source_hard_negative_tail_total", 0.0) * val_portfolio_source_hard_negative_tail_loss
                    + multi_objective_loss_weights.get("portfolio_source_listwise_release_total", 0.0) * val_portfolio_source_listwise_release_loss
                    + multi_objective_loss_weights.get("portfolio_transfer_regret_total", 0.0) * val_portfolio_transfer_regret_loss
                    + multi_objective_loss_weights.get("portfolio_allocation_objective_consolidation_total", 0.0)
                    * val_portfolio_allocation_objective_consolidation_loss
                    + multi_objective_loss_weights.get("portfolio_risk_sensitive_allocation_total", 0.0)
                    * val_portfolio_risk_sensitive_allocation_loss
                    + multi_objective_loss_weights.get("portfolio_utility_credit_closure_total", 0.0)
                    * val_portfolio_utility_credit_closure_loss
                    + multi_objective_loss_weights.get("portfolio_primal_dual_decision_total", 0.0)
                    * val_portfolio_primal_dual_decision_loss
                    + multi_objective_loss_weights.get("portfolio_entropic_transport_decision_total", 0.0)
                    * val_portfolio_entropic_transport_decision_loss
                    + multi_objective_loss_weights.get("portfolio_offline_conservative_support_total", 0.0)
                    * val_portfolio_offline_conservative_support_loss
                    + multi_objective_loss_weights.get("portfolio_differentiable_convex_allocation_total", 0.0)
                    * val_portfolio_differentiable_convex_allocation_loss
                    + multi_objective_loss_weights.get("portfolio_cvxpy_convex_allocation_total", 0.0)
                    * val_portfolio_cvxpy_convex_allocation_loss
                    + multi_objective_loss_weights.get("portfolio_full_universe_convex_allocation_total", 0.0)
                    * val_portfolio_full_universe_convex_allocation_loss
                    + multi_objective_loss_weights.get("portfolio_native_allocation_vector_total", 0.0)
                    * val_portfolio_native_allocation_vector_loss
                    + multi_objective_loss_weights.get("portfolio_day_set_native_allocation_vector_total", 0.0)
                    * val_portfolio_day_set_native_allocation_vector_loss
                    + multi_objective_loss_weights.get("portfolio_capital_flow_closure_total", 0.0)
                    * val_portfolio_capital_flow_closure_loss
                    + multi_objective_loss_weights.get("value_arbitration_total", 0.0) * val_value_arbitration_loss
                    + multi_objective_loss_weights.get("three_value_gate_total", 0.0) * val_three_value_gate_loss
                    + multi_objective_loss_weights.get("hierarchical_three_value_total", 0.0) * val_hierarchical_three_value_gate_loss
                    + multi_objective_loss_weights.get("funding_release_total", 0.0) * val_funding_release_loss
                    + multi_objective_loss_weights.get("action_value_total", 0.0) * val_action_value_consistency_loss
                    + multi_objective_loss_weights.get("clipped_intent_total", 0.0) * val_clipped_intent_loss
                ).detach().cpu()
            )

        train_loss = float(epoch_sample_loss / max(batch_count, 1))
        history.append({"epoch": int(epoch), "train_loss": train_loss, "validation_loss": val_loss})
        checkpoint_payload = {
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "sample_model_state_dict": sample_model.state_dict(),
            "daily_model_state_dict": daily_model.state_dict(),
            "sample_optimizer_state_dict": sample_optimizer.state_dict(),
            "daily_optimizer_state_dict": daily_optimizer.state_dict(),
            "history": history[-200:],
            "signature_hash": signature_hash,
        }
        _save_checkpoint(checkpoint_last, checkpoint_payload)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = int(epoch)
            patience_used = 0
            checkpoint_payload["best_epoch"] = int(best_epoch)
            checkpoint_payload["best_val_loss"] = float(best_val_loss)
            _save_checkpoint(checkpoint_best, checkpoint_payload)
        else:
            patience_used += 1

        if int(epoch) >= int(min_epochs) and int(patience_used) >= int(early_stop_patience):
            break

    best_state = torch.load(checkpoint_best if checkpoint_best.exists() else checkpoint_last, map_location="cpu", weights_only=False)
    sample_model.load_state_dict(best_state["sample_model_state_dict"])
    daily_model.load_state_dict(best_state["daily_model_state_dict"])
    sample_model.eval()
    daily_model.eval()

    def _validation_sample_outputs() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None, torch.Tensor | None]:
        if uses_day_set_native_allocation:
            mask = torch.ones(1, X_static_val.shape[0], dtype=torch.bool, device=device)
            daily_for_sample = X_daily_val[:1] if X_daily_val.shape[0] else torch.zeros(1, len(daily_feature_names), device=device)
            day_outputs = sample_model(
                X_static_val.unsqueeze(0),
                X_sequence_val.unsqueeze(0),
                daily_for_sample,
                mask,
            )
            flat_outputs = {
                name: (
                    value.squeeze(0)
                    if value.ndim >= 2
                    else value.repeat(X_static_val.shape[0])
                    if value.ndim == 1 and value.numel() == 1
                    else value
                )
                for name, value in day_outputs.items()
            }
            return flat_outputs, day_outputs, {name: tensor.unsqueeze(0) for name, tensor in val_targets.items()}, mask
        return sample_model(X_static_val, X_sequence_val), None, None, None

    portfolio_differentiable_convex_terms: dict[str, float] = {}
    if multi_objective_loss_weights.get("portfolio_differentiable_convex_allocation_total", 0.0) > 0.0:
        with torch.no_grad():
            final_val_outputs, _, _, _ = _validation_sample_outputs()
            raw_terms = _portfolio_differentiable_convex_allocation_loss(
                final_val_outputs,
                val_targets,
                return_terms=True,
            )
        if isinstance(raw_terms, dict):
            portfolio_differentiable_convex_terms = {
                name: float(value.detach().cpu())
                for name, value in raw_terms.items()
            }
    portfolio_cvxpy_convex_terms: dict[str, float] = {}
    if multi_objective_loss_weights.get("portfolio_cvxpy_convex_allocation_total", 0.0) > 0.0:
        with torch.no_grad():
            final_val_outputs, _, _, _ = _validation_sample_outputs()
            raw_cvxpy_terms = _portfolio_cvxpy_convex_allocation_loss(
                final_val_outputs,
                val_targets,
                return_terms=True,
            )
        if isinstance(raw_cvxpy_terms, dict):
            portfolio_cvxpy_convex_terms = {
                name: float(value.detach().cpu())
                for name, value in raw_cvxpy_terms.items()
            }
    portfolio_full_universe_convex_terms: dict[str, float] = {}
    if multi_objective_loss_weights.get("portfolio_full_universe_convex_allocation_total", 0.0) > 0.0:
        with torch.no_grad():
            final_val_outputs, _, _, _ = _validation_sample_outputs()
            raw_full_universe_terms = _portfolio_full_universe_convex_allocation_loss(
                final_val_outputs,
                val_targets,
                enable_solver=True,
                return_terms=True,
            )
        if isinstance(raw_full_universe_terms, dict):
            portfolio_full_universe_convex_terms = {
                name: float(value.detach().cpu())
                for name, value in raw_full_universe_terms.items()
            }
    portfolio_native_allocation_vector_terms: dict[str, float] = {}
    if multi_objective_loss_weights.get("portfolio_native_allocation_vector_total", 0.0) > 0.0:
        with torch.no_grad():
            final_val_outputs, _, _, _ = _validation_sample_outputs()
            raw_native_terms = _portfolio_native_allocation_vector_loss(
                final_val_outputs,
                val_targets,
                return_terms=True,
            )
        if isinstance(raw_native_terms, dict):
            portfolio_native_allocation_vector_terms = {
                name: float(value.detach().cpu())
                for name, value in raw_native_terms.items()
            }
    portfolio_day_set_native_allocation_vector_terms: dict[str, float] = {}
    if multi_objective_loss_weights.get("portfolio_day_set_native_allocation_vector_total", 0.0) > 0.0:
        with torch.no_grad():
            _, final_day_set_outputs, final_day_set_targets, final_day_set_mask = _validation_sample_outputs()
            if final_day_set_outputs is not None and final_day_set_targets is not None and final_day_set_mask is not None:
                raw_day_set_terms = _portfolio_day_set_native_allocation_vector_loss(
                    final_day_set_outputs,
                    final_day_set_targets,
                    final_day_set_mask,
                    return_terms=True,
                )
            else:
                raw_day_set_terms = {}
        if isinstance(raw_day_set_terms, dict):
            portfolio_day_set_native_allocation_vector_terms = {
                name: float(value.detach().cpu())
                for name, value in raw_day_set_terms.items()
            }
    portfolio_capital_flow_closure_terms: dict[str, float] = {}
    if multi_objective_loss_weights.get("portfolio_capital_flow_closure_total", 0.0) > 0.0:
        with torch.no_grad():
            final_val_outputs, _, _, _ = _validation_sample_outputs()
            raw_capital_flow_terms = _portfolio_capital_flow_closure_loss(
                final_val_outputs,
                val_targets,
                return_terms=True,
            )
        if isinstance(raw_capital_flow_terms, dict):
            portfolio_capital_flow_closure_terms = {
                name: float(value.detach().cpu())
                for name, value in raw_capital_flow_terms.items()
            }

    diagnostics = {
        "trainer_backend": TRAINER_BACKEND_FORMAL_SEQ_V3,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "python_executable": str(sys.executable),
        "conda_prefix": str(os.environ.get("CONDA_PREFIX", "")),
        "runtime_env": "yolos" if "yolos" in str(sys.executable).lower() else "",
        "resource_runtime": resource_runtime,
        "epochs_requested": int(epochs),
        "min_epochs": int(min_epochs),
        "completed_epochs": int(history[-1]["epoch"]) if history else 0,
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val_loss),
        "resume_mode": str(resume_mode or ""),
        "loss_profile": resolved_loss_profile,
        "resumed_from_checkpoint": resumed_from,
        "checkpoint_last": str(checkpoint_last.resolve()),
        "checkpoint_best": str(checkpoint_best.resolve()) if checkpoint_best.exists() else "",
        "training_diagnostics_json": str(diagnostics_path.resolve()),
        "signature_hash": signature_hash,
        "sequence_base_count": len(sequence_base_names),
        "sequence_step_count": len(SEQUENCE_STEP_ORDER),
        "grouped_day_count": int(len(daily_frame)),
        "train_day_count": int(len(daily_train_idx)),
        "validation_day_count": int(len(daily_val_idx)),
        "train_sample_rows": int(len(train_idx)),
        "validation_sample_rows": int(len(val_idx)),
        "history_tail": history[-8:],
        "supports_holding_days_head": True,
        "supports_sell_heads": True,
        "supports_sell_attribution_head": "sell_attribution_score" in sample_scalar_loss_weights,
        "supports_lifecycle_arbitration_heads": all(
            name in sample_scalar_loss_weights
            for name in ("sell_rank_score", "lifecycle_sell_gate", "clipped_intent_risk")
        ),
        "supports_value_arbitration_heads": all(
            name in sample_scalar_loss_weights
            for name in (
                "alpha_opportunity_value",
                "hold_continuation_value",
                "sell_release_value",
                "cash_defense_value",
                "deployment_opportunity_cost",
                "value_arbitration_target",
            )
        ),
        "supports_action_value_heads": all(
            name in sample_scalar_loss_weights
            for name in (
                "multi_horizon_forward_value",
                "multi_horizon_forward_risk",
                "multi_horizon_path_value",
                "open_action_value",
                "add_action_value",
                "hold_action_value",
                "reduce_action_value",
                "exit_action_value",
                "action_value_consistency_target",
            )
        ),
        "supports_three_value_gate_heads": all(
            name in sample_scalar_loss_weights
            for name in (
                "deploy_value_target",
                "release_value_target",
                "defense_value_target",
                "deploy_gate_target",
                "release_gate_target",
                "defense_gate_target",
            )
        ),
        "supports_deploy_executability_head": "deploy_executability_target" in sample_scalar_loss_weights,
        "supports_portfolio_listwise_heads": all(
            name in sample_scalar_loss_weights
            for name in (
                "portfolio_daily_receiver_add_headroom",
                "portfolio_daily_receiver_add_capacity",
                "portfolio_daily_receiver_executability",
                "portfolio_daily_receiver_score",
                "portfolio_daily_source_score",
                "portfolio_daily_cash_score",
            )
        ),
        "supports_portfolio_source_release_heads": all(
            name in sample_scalar_loss_weights
            for name in (
                "portfolio_daily_source_release_capacity",
                "portfolio_daily_source_release_quality",
                "portfolio_daily_source_opportunity_cost",
                "portfolio_daily_source_executability",
            )
        ),
        "supports_portfolio_economic_release_heads": all(
            name in sample_scalar_loss_weights
            for name in (
                "portfolio_daily_source_forward_spread_score",
                "portfolio_daily_source_bad_forward_spread_risk",
                "portfolio_daily_source_economic_release_score",
                "portfolio_daily_source_economic_block_risk",
                "portfolio_daily_source_forward_strength_brake_risk",
            )
        ),
        "supports_portfolio_source_forward_proxy_head": (
            "portfolio_daily_source_forward_proxy_keep_risk" in sample_scalar_loss_weights
        ),
        "supports_portfolio_allocation_teacher_heads": all(
            name in sample_scalar_loss_weights
            for name in (
                "portfolio_daily_receiver_funding_coverage",
                "portfolio_daily_funding_closure_score",
                "portfolio_daily_allocation_transfer_score",
                "portfolio_daily_allocation_dead_branch_risk",
            )
        ),
        "supports_portfolio_unified_allocation_heads": all(
            name in sample_scalar_loss_weights
            for name in (
                "portfolio_daily_unified_receiver_score",
                "portfolio_daily_unified_source_score",
                "portfolio_daily_unified_cash_score",
                "portfolio_daily_source_positive_forward_penalty",
                "portfolio_daily_source_opportunity_cost_penalty",
                "portfolio_daily_receiver_source_spread_reward",
                "portfolio_daily_unified_allocation_objective",
            )
        ),
        "supports_portfolio_source_hard_negative_regret_heads": all(
            name in sample_scalar_loss_weights
            for name in (
                "portfolio_daily_source_hard_negative_penalty",
                "portfolio_daily_source_tail_false_sell_penalty",
                "portfolio_daily_source_release_preference",
                "portfolio_daily_transfer_regret_target",
            )
        ),
        "supports_portfolio_allocation_objective_consolidation_heads": all(
            name in sample_scalar_loss_weights
            for name in (
                "portfolio_daily_allocation_trade_quality_target",
                "portfolio_daily_allocation_cash_deployment_target",
                "portfolio_daily_allocation_risk_adjusted_return_target",
                "portfolio_daily_allocation_drawdown_control_target",
                "portfolio_daily_allocation_monthly_quality_target",
                "portfolio_daily_allocation_final_objective",
            )
        ),
        "supports_portfolio_risk_sensitive_allocation_heads": all(
            name in sample_scalar_loss_weights
            for name in (
                "portfolio_daily_allocation_uncertainty_pressure_target",
                "portfolio_daily_allocation_tail_risk_control_target",
                "portfolio_daily_allocation_decision_focused_objective",
            )
        ),
        "supports_portfolio_utility_credit_closure_heads": all(
            name in sample_scalar_loss_weights
            for name in (
                "portfolio_daily_allocation_net_utility_target",
                "portfolio_daily_allocation_credit_closure_target",
                "portfolio_daily_allocation_resource_efficiency_target",
            )
        ),
        "supports_portfolio_unified_allocation_consistency_loss": (
            multi_objective_loss_weights.get("portfolio_unified_allocation_total", 0.0) > 0.0
        ),
        "supports_portfolio_decision_focused_allocation_loss": (
            multi_objective_loss_weights.get("portfolio_decision_regret_total", 0.0) > 0.0
        ),
        "supports_portfolio_source_hard_negative_tail_loss": (
            multi_objective_loss_weights.get("portfolio_source_hard_negative_tail_total", 0.0) > 0.0
        ),
        "supports_portfolio_source_listwise_release_loss": (
            multi_objective_loss_weights.get("portfolio_source_listwise_release_total", 0.0) > 0.0
        ),
        "supports_portfolio_transfer_regret_loss": (
            multi_objective_loss_weights.get("portfolio_transfer_regret_total", 0.0) > 0.0
        ),
        "supports_portfolio_allocation_objective_consolidation_loss": (
            multi_objective_loss_weights.get("portfolio_allocation_objective_consolidation_total", 0.0) > 0.0
        ),
        "supports_portfolio_risk_sensitive_allocation_loss": (
            multi_objective_loss_weights.get("portfolio_risk_sensitive_allocation_total", 0.0) > 0.0
        ),
        "supports_portfolio_utility_credit_closure_loss": (
            multi_objective_loss_weights.get("portfolio_utility_credit_closure_total", 0.0) > 0.0
        ),
        "supports_portfolio_primal_dual_decision_loss": (
            multi_objective_loss_weights.get("portfolio_primal_dual_decision_total", 0.0) > 0.0
        ),
        "supports_portfolio_entropic_transport_decision_loss": (
            multi_objective_loss_weights.get("portfolio_entropic_transport_decision_total", 0.0) > 0.0
        ),
        "supports_portfolio_offline_conservative_support_loss": (
            multi_objective_loss_weights.get("portfolio_offline_conservative_support_total", 0.0) > 0.0
        ),
        "supports_portfolio_differentiable_convex_allocation_loss": (
            multi_objective_loss_weights.get("portfolio_differentiable_convex_allocation_total", 0.0) > 0.0
        ),
        "supports_portfolio_differentiable_convex_allocation_diagnostics": bool(portfolio_differentiable_convex_terms),
        "supports_portfolio_path_risk_loss": (
            bool(portfolio_differentiable_convex_terms)
            and "path_risk_loss" in portfolio_differentiable_convex_terms
        ),
        "portfolio_differentiable_convex_allocation_terms": portfolio_differentiable_convex_terms,
        "supports_portfolio_cvxpy_convex_allocation_layer": (
            multi_objective_loss_weights.get("portfolio_cvxpy_convex_allocation_total", 0.0) > 0.0
        ),
        "portfolio_cvxpy_convex_layer_status": _cvxpy_convex_layer_status(),
        "supports_portfolio_cvxpy_convex_allocation_diagnostics": bool(portfolio_cvxpy_convex_terms),
        "portfolio_cvxpy_convex_allocation_terms": portfolio_cvxpy_convex_terms,
        "supports_portfolio_full_universe_convex_allocation_loss": (
            multi_objective_loss_weights.get("portfolio_full_universe_convex_allocation_total", 0.0) > 0.0
        ),
        "portfolio_full_universe_convex_solver_slot_count": CVXPY_FULL_UNIVERSE_ALLOCATION_SLOT_COUNT,
        "portfolio_full_universe_convex_max_days_per_batch": CVXPY_FULL_UNIVERSE_ALLOCATION_MAX_DAYS_PER_BATCH,
        "portfolio_full_universe_convex_train_batch_interval": CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_BATCH_INTERVAL,
        "portfolio_full_universe_convex_train_solver_enabled": CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_SOLVER_ENABLED,
        "portfolio_full_universe_convex_train_solver_loss_profiles": sorted(
            CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_SOLVER_LOSS_PROFILES
        ),
        "portfolio_full_universe_convex_train_solver_effective": bool(full_universe_train_solver_enabled),
        "supports_portfolio_full_universe_candidate_coverage": bool(
            portfolio_full_universe_convex_terms
            and "candidate_coverage_loss" in portfolio_full_universe_convex_terms
        ),
        "supports_portfolio_full_universe_ope_diagnostics": bool(
            portfolio_full_universe_convex_terms
            and "ope_lower_bound_loss" in portfolio_full_universe_convex_terms
            and "doubly_robust_gap_loss" in portfolio_full_universe_convex_terms
        ),
        "portfolio_full_universe_convex_allocation_terms": portfolio_full_universe_convex_terms,
        "supports_portfolio_native_allocation_vector_heads": all(
            name in sample_model.state_dict()
            for name in (
                "portfolio_daily_allocation_weight_logit_head.weight",
                "portfolio_daily_cash_reserve_logit_head.weight",
                "portfolio_daily_allocation_risk_buffer_logit_head.weight",
            )
        )
        and multi_objective_loss_weights.get("portfolio_native_allocation_vector_total", 0.0) > 0.0,
        "supports_portfolio_native_allocation_vector_loss": (
            multi_objective_loss_weights.get("portfolio_native_allocation_vector_total", 0.0) > 0.0
        ),
        "supports_portfolio_native_allocation_vector_diagnostics": bool(portfolio_native_allocation_vector_terms),
        "portfolio_native_allocation_vector_terms": portfolio_native_allocation_vector_terms,
        "supports_portfolio_day_set_native_allocation_vector": (
            multi_objective_loss_weights.get("portfolio_day_set_native_allocation_vector_total", 0.0) > 0.0
            and str(getattr(sample_model, "sample_model_type", "temporal_sample")) == "temporal_day_set"
        ),
        "portfolio_day_set_native_allocation_vector_terms": locals().get("portfolio_day_set_native_allocation_vector_terms", {}),
        "sample_model_type": str(getattr(sample_model, "sample_model_type", "temporal_sample")),
        "day_set_batch_size": int(batch_size)
        if multi_objective_loss_weights.get("portfolio_day_set_native_allocation_vector_total", 0.0) > 0.0
        else 0,
        "day_set_slot_count": int(getattr(sample_model, "slot_count", 0) or 0),
        "day_set_full_day_integrity": bool(
            multi_objective_loss_weights.get("portfolio_day_set_native_allocation_vector_total", 0.0) > 0.0
        ),
        "supports_portfolio_capital_flow_closure_loss": (
            multi_objective_loss_weights.get("portfolio_capital_flow_closure_total", 0.0) > 0.0
        ),
        "supports_portfolio_capital_flow_closure_diagnostics": bool(portfolio_capital_flow_closure_terms),
        "portfolio_capital_flow_closure_terms": portfolio_capital_flow_closure_terms,
        "supports_funding_release_discipline": (
            multi_objective_loss_weights.get("funding_release_total", 0.0) > 0.0
            and all(
                name in sample_scalar_loss_weights
                for name in (
                    "hold_continuation_value",
                    "alpha_opportunity_value",
                    "sell_release_value",
                    "cash_defense_value",
                    "deploy_value_target",
                    "release_value_target",
                    "deploy_gate_target",
                    "release_gate_target",
                )
            )
        ),
        "funding_release_loss_variant": funding_release_loss_variant,
        "sample_scalar_loss_weights": dict(sample_scalar_loss_weights),
        "daily_target_loss_weights": dict(daily_target_loss_weights),
        "multi_objective_loss_weights": dict(multi_objective_loss_weights),
        "daily_head_layout": resolved_daily_head_layout,
        "supports_extended_budget_heads": resolved_daily_head_layout == DAILY_HEAD_LAYOUT_SPLIT_V2,
    }
    artifact = TorchContinuousPolicySeqArtifact(
        sample_model=sample_model.cpu(),
        daily_model=daily_model.cpu(),
        static_feature_names=list(static_feature_names),
        sequence_base_names=list(sequence_base_names),
        sequence_steps=[int(item) for item in SEQUENCE_STEP_ORDER],
        sequence_columns=list(sequence_columns),
        daily_feature_names=list(daily_feature_names),
        static_fill_values=static_fill,
        static_means=static_means,
        static_stds=static_stds,
        sequence_fill_values=sequence_fill,
        sequence_means=sequence_means,
        sequence_stds=sequence_stds,
        daily_fill_values=daily_fill,
        daily_means=daily_means,
        daily_stds=daily_stds,
        train_summary=dict(train_summary or {}),
        training_diagnostics=diagnostics,
        training_contract=contract,
        trained_at=str(trained_at or ""),
    )
    artifact.save(artifact_path)
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact


def predict_policy_v3(
    artifact: TorchContinuousPolicySeqArtifact,
    *,
    state_frame: pd.DataFrame,
    daily_features: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    if state_frame.empty:
        raise ValueError("state_frame is empty.")
    with torch.no_grad():
        static_x = torch.as_tensor(
            _apply_matrix(
                state_frame,
                artifact.static_feature_names,
                artifact.static_fill_values,
                artifact.static_means,
                artifact.static_stds,
            ),
            dtype=torch.float32,
        )
        sequence_x = torch.as_tensor(
            _reshape_sequence_matrix(
                _apply_matrix(
                    state_frame,
                    artifact.sequence_columns,
                    artifact.sequence_fill_values,
                    artifact.sequence_means,
                    artifact.sequence_stds,
                ),
                steps=len(artifact.sequence_steps),
                feature_dim=len(artifact.sequence_base_names),
            ),
            dtype=torch.float32,
        )
        sample_model_type = str(artifact.training_diagnostics.get("sample_model_type", "temporal_sample") or "temporal_sample")
        if sample_model_type == "temporal_day_set":
            daily_row_for_sample = pd.DataFrame([{name: float(daily_features.get(name, 0.0) or 0.0) for name in artifact.daily_feature_names}])
            daily_x_for_sample = torch.as_tensor(
                _apply_matrix(
                    daily_row_for_sample,
                    artifact.daily_feature_names,
                    artifact.daily_fill_values,
                    artifact.daily_means,
                    artifact.daily_stds,
                ),
                dtype=torch.float32,
            )
            day_outputs = artifact.sample_model(
                static_x.unsqueeze(0),
                sequence_x.unsqueeze(0),
                daily_x_for_sample,
                torch.ones(1, len(state_frame), dtype=torch.bool),
            )
            outputs = {}
            for name, value in day_outputs.items():
                if torch.is_tensor(value) and value.ndim >= 2:
                    outputs[name] = value.squeeze(0)
                elif torch.is_tensor(value) and value.ndim == 1 and value.numel() == 1:
                    outputs[name] = value.repeat(len(state_frame))
                else:
                    outputs[name] = value
        else:
            outputs = artifact.sample_model(static_x, sequence_x)
        action_prob = torch.softmax(outputs["action_logits"], dim=-1).cpu().numpy()
        duration_prob = torch.softmax(outputs["duration_logits"], dim=-1).cpu().numpy()
        predicted_labels = np.asarray(ACTION_CLASSES, dtype=object)[action_prob.argmax(axis=1)]
        predicted_duration_labels = np.asarray(DURATION_CLASSES, dtype=object)[duration_prob.argmax(axis=1)]
        delta_hint = outputs["target_delta_hint"].cpu().numpy()
        entry_quality = outputs["entry_quality"].cpu().numpy()
        hold_quality = outputs["hold_quality"].cpu().numpy()
        add_quality = outputs["add_quality"].cpu().numpy()
        reduce_quality = outputs["reduce_quality"].cpu().numpy()
        exit_urgency = np.clip(outputs["exit_urgency"].cpu().numpy(), 0.0, None)
        reentry_readiness = np.clip(outputs["reentry_readiness"].cpu().numpy(), 0.0, None)
        holding_days_ratio = np.clip(outputs["holding_days_ratio"].cpu().numpy(), 0.0, 1.0)
        supports_sell_heads = bool(artifact.training_diagnostics.get("supports_sell_heads", True))
        reduce_fraction = (
            np.clip(outputs["reduce_fraction"].cpu().numpy(), 0.0, 1.0)
            if supports_sell_heads
            else None
        )
        exit_hazard = (
            np.clip(outputs["exit_hazard"].cpu().numpy(), 0.0, 1.0)
            if supports_sell_heads
            else None
        )
        supports_sell_attribution_head = bool(artifact.training_diagnostics.get("supports_sell_attribution_head", False))
        predicted_sell_attribution = (
            np.clip(outputs["sell_attribution_score"].cpu().numpy(), 0.0, 1.0)
            if supports_sell_attribution_head
            else None
        )
        supports_lifecycle_arbitration_heads = bool(artifact.training_diagnostics.get("supports_lifecycle_arbitration_heads", False))
        predicted_sell_rank = (
            np.clip(outputs["sell_rank_score"].cpu().numpy(), 0.0, 1.0)
            if supports_lifecycle_arbitration_heads
            else None
        )
        predicted_lifecycle_sell_gate = (
            np.clip(outputs["lifecycle_sell_gate"].cpu().numpy(), 0.0, 1.0)
            if supports_lifecycle_arbitration_heads
            else None
        )
        predicted_clipped_intent_risk = (
            np.clip(outputs["clipped_intent_risk"].cpu().numpy(), 0.0, 1.0)
            if supports_lifecycle_arbitration_heads
            else None
        )
        supports_value_arbitration_heads = bool(artifact.training_diagnostics.get("supports_value_arbitration_heads", False))
        predicted_large_upside = (
            np.clip(outputs["large_upside_1d_target"].cpu().numpy(), 0.0, 1.0)
            if supports_value_arbitration_heads
            else None
        )
        predicted_alpha_opportunity = (
            np.clip(outputs["alpha_opportunity_value"].cpu().numpy(), 0.0, 1.0)
            if supports_value_arbitration_heads
            else None
        )
        predicted_hold_continuation = (
            np.clip(outputs["hold_continuation_value"].cpu().numpy(), 0.0, 1.0)
            if supports_value_arbitration_heads
            else None
        )
        predicted_sell_release = (
            np.clip(outputs["sell_release_value"].cpu().numpy(), 0.0, 1.0)
            if supports_value_arbitration_heads
            else None
        )
        predicted_cash_defense = (
            np.clip(outputs["cash_defense_value"].cpu().numpy(), 0.0, 1.0)
            if supports_value_arbitration_heads
            else None
        )
        predicted_deployment_cost = (
            np.clip(outputs["deployment_opportunity_cost"].cpu().numpy(), 0.0, 1.0)
            if supports_value_arbitration_heads
            else None
        )
        predicted_risk_adjusted_action_value = (
            np.clip(outputs["risk_adjusted_action_value"].cpu().numpy(), 0.0, 1.0)
            if supports_value_arbitration_heads
            else None
        )
        predicted_value_arbitration = (
            np.clip(outputs["value_arbitration_target"].cpu().numpy(), 0.0, 1.0)
            if supports_value_arbitration_heads
            else None
        )
        supports_action_value_heads = bool(artifact.training_diagnostics.get("supports_action_value_heads", False))
        predicted_multi_horizon_forward_value = (
            np.clip(outputs["multi_horizon_forward_value"].cpu().numpy(), 0.0, 1.0)
            if supports_action_value_heads
            else None
        )
        predicted_multi_horizon_forward_risk = (
            np.clip(outputs["multi_horizon_forward_risk"].cpu().numpy(), 0.0, 1.0)
            if supports_action_value_heads
            else None
        )
        predicted_multi_horizon_path_value = (
            np.clip(outputs["multi_horizon_path_value"].cpu().numpy(), 0.0, 1.0)
            if supports_action_value_heads
            else None
        )
        predicted_open_action_value = (
            np.clip(outputs["open_action_value"].cpu().numpy(), 0.0, 1.0)
            if supports_action_value_heads
            else None
        )
        predicted_add_action_value = (
            np.clip(outputs["add_action_value"].cpu().numpy(), 0.0, 1.0)
            if supports_action_value_heads
            else None
        )
        predicted_hold_action_value = (
            np.clip(outputs["hold_action_value"].cpu().numpy(), 0.0, 1.0)
            if supports_action_value_heads
            else None
        )
        predicted_reduce_action_value = (
            np.clip(outputs["reduce_action_value"].cpu().numpy(), 0.0, 1.0)
            if supports_action_value_heads
            else None
        )
        predicted_exit_action_value = (
            np.clip(outputs["exit_action_value"].cpu().numpy(), 0.0, 1.0)
            if supports_action_value_heads
            else None
        )
        predicted_relative_opportunity_value = (
            np.clip(outputs["relative_opportunity_value"].cpu().numpy(), 0.0, 1.0)
            if supports_action_value_heads
            else None
        )
        predicted_action_value_consistency = (
            np.clip(outputs["action_value_consistency_target"].cpu().numpy(), 0.0, 1.0)
            if supports_action_value_heads
            else None
        )
        supports_three_value_gate_heads = bool(artifact.training_diagnostics.get("supports_three_value_gate_heads", False))
        predicted_deploy_value = (
            np.clip(outputs["deploy_value_target"].cpu().numpy(), 0.0, 1.0)
            if supports_three_value_gate_heads
            else None
        )
        predicted_release_value = (
            np.clip(outputs["release_value_target"].cpu().numpy(), 0.0, 1.0)
            if supports_three_value_gate_heads
            else None
        )
        predicted_defense_value = (
            np.clip(outputs["defense_value_target"].cpu().numpy(), 0.0, 1.0)
            if supports_three_value_gate_heads
            else None
        )
        predicted_deploy_gate = (
            np.clip(outputs["deploy_gate_target"].cpu().numpy(), 0.0, 1.0)
            if supports_three_value_gate_heads
            else None
        )
        predicted_release_gate = (
            np.clip(outputs["release_gate_target"].cpu().numpy(), 0.0, 1.0)
            if supports_three_value_gate_heads
            else None
        )
        predicted_defense_gate = (
            np.clip(outputs["defense_gate_target"].cpu().numpy(), 0.0, 1.0)
            if supports_three_value_gate_heads
            else None
        )
        supports_deploy_executability_head = bool(artifact.training_diagnostics.get("supports_deploy_executability_head", False))
        predicted_deploy_executability = (
            np.clip(outputs["deploy_executability_target"].cpu().numpy(), 0.0, 1.0)
            if supports_deploy_executability_head
            else None
        )
        supports_portfolio_listwise_heads = bool(artifact.training_diagnostics.get("supports_portfolio_listwise_heads", False))
        predicted_portfolio_receiver_headroom = (
            np.clip(outputs["portfolio_daily_receiver_add_headroom"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_listwise_heads
            else None
        )
        predicted_portfolio_receiver_capacity = (
            np.clip(outputs["portfolio_daily_receiver_add_capacity"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_listwise_heads
            else None
        )
        predicted_portfolio_receiver_executability = (
            np.clip(outputs["portfolio_daily_receiver_executability"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_listwise_heads
            else None
        )
        predicted_portfolio_receiver_score = (
            np.clip(outputs["portfolio_daily_receiver_score"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_listwise_heads
            else None
        )
        supports_portfolio_source_release_heads = bool(
            artifact.training_diagnostics.get("supports_portfolio_source_release_heads", False)
        )
        predicted_portfolio_source_release_capacity = (
            np.clip(outputs["portfolio_daily_source_release_capacity"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_source_release_heads
            else None
        )
        supports_portfolio_economic_release_heads = bool(
            artifact.training_diagnostics.get("supports_portfolio_economic_release_heads", False)
        )
        predicted_portfolio_source_forward_spread_score = (
            np.clip(outputs["portfolio_daily_source_forward_spread_score"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_economic_release_heads
            else None
        )
        predicted_portfolio_source_bad_forward_spread_risk = (
            np.clip(outputs["portfolio_daily_source_bad_forward_spread_risk"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_economic_release_heads
            else None
        )
        predicted_portfolio_source_economic_release_score = (
            np.clip(outputs["portfolio_daily_source_economic_release_score"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_economic_release_heads
            else None
        )
        predicted_portfolio_source_economic_block_risk = (
            np.clip(outputs["portfolio_daily_source_economic_block_risk"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_economic_release_heads
            else None
        )
        predicted_portfolio_source_forward_strength_brake_risk = (
            np.clip(outputs["portfolio_daily_source_forward_strength_brake_risk"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_economic_release_heads
            else None
        )
        supports_portfolio_source_forward_proxy_head = bool(
            artifact.training_diagnostics.get("supports_portfolio_source_forward_proxy_head", False)
        )
        predicted_portfolio_source_forward_proxy_keep_risk = (
            np.clip(outputs["portfolio_daily_source_forward_proxy_keep_risk"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_source_forward_proxy_head
            else None
        )
        predicted_portfolio_source_release_quality = (
            np.clip(outputs["portfolio_daily_source_release_quality"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_source_release_heads
            else None
        )
        predicted_portfolio_source_opportunity_cost = (
            np.clip(outputs["portfolio_daily_source_opportunity_cost"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_source_release_heads
            else None
        )
        predicted_portfolio_source_executability = (
            np.clip(outputs["portfolio_daily_source_executability"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_source_release_heads
            else None
        )
        predicted_portfolio_source_score = (
            np.clip(outputs["portfolio_daily_source_score"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_listwise_heads
            else None
        )
        predicted_portfolio_cash_score = (
            np.clip(outputs["portfolio_daily_cash_score"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_listwise_heads
            else None
        )
        supports_portfolio_unified_allocation_heads = bool(
            artifact.training_diagnostics.get("supports_portfolio_unified_allocation_heads", False)
        )
        predicted_portfolio_unified_receiver_score = (
            np.clip(outputs["portfolio_daily_unified_receiver_score"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_unified_allocation_heads and "portfolio_daily_unified_receiver_score" in outputs
            else None
        )
        predicted_portfolio_unified_source_score = (
            np.clip(outputs["portfolio_daily_unified_source_score"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_unified_allocation_heads and "portfolio_daily_unified_source_score" in outputs
            else None
        )
        predicted_portfolio_unified_cash_score = (
            np.clip(outputs["portfolio_daily_unified_cash_score"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_unified_allocation_heads and "portfolio_daily_unified_cash_score" in outputs
            else None
        )
        predicted_portfolio_source_positive_forward_penalty = (
            np.clip(outputs["portfolio_daily_source_positive_forward_penalty"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_unified_allocation_heads and "portfolio_daily_source_positive_forward_penalty" in outputs
            else None
        )
        predicted_portfolio_source_opportunity_cost_penalty = (
            np.clip(outputs["portfolio_daily_source_opportunity_cost_penalty"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_unified_allocation_heads and "portfolio_daily_source_opportunity_cost_penalty" in outputs
            else None
        )
        supports_portfolio_source_hard_negative_regret_heads = bool(
            artifact.training_diagnostics.get("supports_portfolio_source_hard_negative_regret_heads", False)
        )
        predicted_portfolio_source_strong_false_sell_penalty = (
            np.clip(outputs["portfolio_daily_source_strong_false_sell_penalty"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_source_hard_negative_regret_heads and "portfolio_daily_source_strong_false_sell_penalty" in outputs
            else None
        )
        predicted_portfolio_source_hard_negative_penalty = (
            np.clip(outputs["portfolio_daily_source_hard_negative_penalty"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_source_hard_negative_regret_heads and "portfolio_daily_source_hard_negative_penalty" in outputs
            else None
        )
        predicted_portfolio_source_tail_false_sell_penalty = (
            np.clip(outputs["portfolio_daily_source_tail_false_sell_penalty"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_source_hard_negative_regret_heads and "portfolio_daily_source_tail_false_sell_penalty" in outputs
            else None
        )
        predicted_portfolio_source_release_preference = (
            np.clip(outputs["portfolio_daily_source_release_preference"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_source_hard_negative_regret_heads and "portfolio_daily_source_release_preference" in outputs
            else None
        )
        predicted_portfolio_receiver_source_spread_reward = (
            np.clip(outputs["portfolio_daily_receiver_source_spread_reward"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_unified_allocation_heads and "portfolio_daily_receiver_source_spread_reward" in outputs
            else None
        )
        predicted_portfolio_transfer_regret_target = (
            np.clip(outputs["portfolio_daily_transfer_regret_target"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_source_hard_negative_regret_heads and "portfolio_daily_transfer_regret_target" in outputs
            else None
        )
        predicted_portfolio_unified_allocation_objective = (
            np.clip(outputs["portfolio_daily_unified_allocation_objective"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_unified_allocation_heads and "portfolio_daily_unified_allocation_objective" in outputs
            else None
        )
        supports_portfolio_allocation_objective_consolidation_heads = bool(
            artifact.training_diagnostics.get("supports_portfolio_allocation_objective_consolidation_heads", False)
        )
        predicted_portfolio_allocation_trade_quality_target = (
            np.clip(outputs["portfolio_daily_allocation_trade_quality_target"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_allocation_objective_consolidation_heads and "portfolio_daily_allocation_trade_quality_target" in outputs
            else None
        )
        predicted_portfolio_allocation_cash_deployment_target = (
            np.clip(outputs["portfolio_daily_allocation_cash_deployment_target"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_allocation_objective_consolidation_heads and "portfolio_daily_allocation_cash_deployment_target" in outputs
            else None
        )
        predicted_portfolio_allocation_risk_adjusted_return_target = (
            np.clip(outputs["portfolio_daily_allocation_risk_adjusted_return_target"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_allocation_objective_consolidation_heads and "portfolio_daily_allocation_risk_adjusted_return_target" in outputs
            else None
        )
        predicted_portfolio_allocation_drawdown_control_target = (
            np.clip(outputs["portfolio_daily_allocation_drawdown_control_target"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_allocation_objective_consolidation_heads and "portfolio_daily_allocation_drawdown_control_target" in outputs
            else None
        )
        predicted_portfolio_allocation_monthly_quality_target = (
            np.clip(outputs["portfolio_daily_allocation_monthly_quality_target"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_allocation_objective_consolidation_heads and "portfolio_daily_allocation_monthly_quality_target" in outputs
            else None
        )
        predicted_portfolio_allocation_final_objective = (
            np.clip(outputs["portfolio_daily_allocation_final_objective"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_allocation_objective_consolidation_heads and "portfolio_daily_allocation_final_objective" in outputs
            else None
        )
        supports_portfolio_risk_sensitive_allocation_heads = bool(
            artifact.training_diagnostics.get("supports_portfolio_risk_sensitive_allocation_heads", False)
        )
        predicted_portfolio_allocation_uncertainty_pressure_target = (
            np.clip(outputs["portfolio_daily_allocation_uncertainty_pressure_target"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_risk_sensitive_allocation_heads
            and "portfolio_daily_allocation_uncertainty_pressure_target" in outputs
            else None
        )
        predicted_portfolio_allocation_tail_risk_control_target = (
            np.clip(outputs["portfolio_daily_allocation_tail_risk_control_target"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_risk_sensitive_allocation_heads
            and "portfolio_daily_allocation_tail_risk_control_target" in outputs
            else None
        )
        predicted_portfolio_allocation_decision_focused_objective = (
            np.clip(outputs["portfolio_daily_allocation_decision_focused_objective"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_risk_sensitive_allocation_heads
            and "portfolio_daily_allocation_decision_focused_objective" in outputs
            else None
        )
        supports_portfolio_utility_credit_closure_heads = bool(
            artifact.training_diagnostics.get("supports_portfolio_utility_credit_closure_heads", False)
        )
        predicted_portfolio_allocation_net_utility_target = (
            np.clip(outputs["portfolio_daily_allocation_net_utility_target"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_utility_credit_closure_heads
            and "portfolio_daily_allocation_net_utility_target" in outputs
            else None
        )
        predicted_portfolio_allocation_credit_closure_target = (
            np.clip(outputs["portfolio_daily_allocation_credit_closure_target"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_utility_credit_closure_heads
            and "portfolio_daily_allocation_credit_closure_target" in outputs
            else None
        )
        predicted_portfolio_allocation_resource_efficiency_target = (
            np.clip(outputs["portfolio_daily_allocation_resource_efficiency_target"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_utility_credit_closure_heads
            and "portfolio_daily_allocation_resource_efficiency_target" in outputs
            else None
        )
        supports_portfolio_allocation_teacher_heads = bool(
            artifact.training_diagnostics.get("supports_portfolio_allocation_teacher_heads", False)
        )
        predicted_portfolio_receiver_funding_coverage = (
            np.clip(outputs["portfolio_daily_receiver_funding_coverage"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_allocation_teacher_heads
            else None
        )
        predicted_portfolio_funding_closure_score = (
            np.clip(outputs["portfolio_daily_funding_closure_score"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_allocation_teacher_heads
            else None
        )
        predicted_portfolio_allocation_transfer_score = (
            np.clip(outputs["portfolio_daily_allocation_transfer_score"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_allocation_teacher_heads
            else None
        )
        predicted_portfolio_allocation_dead_branch_risk = (
            np.clip(outputs["portfolio_daily_allocation_dead_branch_risk"].cpu().numpy(), 0.0, 1.0)
            if supports_portfolio_allocation_teacher_heads
            else None
        )
        supports_portfolio_native_allocation_vector_heads = bool(
            (
                artifact.training_diagnostics.get("supports_portfolio_native_allocation_vector_heads", False)
                or artifact.training_diagnostics.get("supports_portfolio_day_set_native_allocation_vector", False)
            )
            and all(name in outputs for name in (
                "portfolio_daily_allocation_weight_logit",
                "portfolio_daily_cash_reserve_logit",
                "portfolio_daily_allocation_risk_buffer_logit",
            ))
        )
        if reduce_fraction is None:
            reduce_fraction = np.clip(
                0.18 * np.clip(-delta_hint, 0.0, None)
                + 0.42 * np.clip(reduce_quality, 0.0, None)
                + 0.10 * np.clip(exit_urgency, 0.0, None),
                0.0,
                1.0,
            )
        if exit_hazard is None:
            exit_hazard = np.clip(
                0.18 * np.clip(-delta_hint - 0.05, 0.0, None)
                + 0.48 * np.clip(exit_urgency, 0.0, None)
                + 0.10 * np.clip(reduce_quality, 0.0, None),
                0.0,
                1.0,
            )
        sell_pressure = np.clip(0.58 * reduce_fraction + 0.42 * exit_hazard, 0.0, 1.0)

        daily_row = pd.DataFrame([{name: float(daily_features.get(name, 0.0) or 0.0) for name in artifact.daily_feature_names}])
        daily_x = torch.as_tensor(
            _apply_matrix(
                daily_row,
                artifact.daily_feature_names,
                artifact.daily_fill_values,
                artifact.daily_means,
                artifact.daily_stds,
            ),
            dtype=torch.float32,
        )
        daily_out = artifact.daily_model(daily_x)
        supports_extended_budget_heads = bool(
            artifact.training_diagnostics.get("supports_extended_budget_heads", False)
            or getattr(artifact.daily_model, "head_layout", DAILY_HEAD_LAYOUT_MONOLITHIC_V1) == DAILY_HEAD_LAYOUT_SPLIT_V2
        )
        budget_model_risk_signal = float(
            np.clip(
                _finite_scalar(daily_out.get("budget_risk_signal_target", 0.0), default=0.0),
                0.0,
                1.0,
            )
        )
        budget_model_deploy_signal = float(
            np.clip(
                _finite_scalar(daily_out.get("budget_deploy_signal_target", 0.0), default=0.0),
                0.0,
                1.0,
            )
        )
        budget_model_cash_timing_signal = float(
            np.clip(
                _finite_scalar(daily_out.get("budget_cash_timing_signal_target", 0.0), default=0.0),
                0.0,
                1.0,
            )
        )
        budget_model_alpha_focus_signal = float(
            np.clip(
                _finite_scalar(daily_out.get("budget_alpha_focus_signal_target", 0.0), default=0.0),
                0.0,
                1.0,
            )
        )
        global_targets = {
            "gross_exposure_target": float(np.clip(_finite_scalar(daily_out["gross_exposure_target"], default=0.35), 0.15, 0.98)),
            "candidate_budget": float(np.clip(_finite_scalar(daily_out["candidate_budget"], default=4.0), 2.0, 12.0)),
            "turnover_budget": float(np.clip(_finite_scalar(daily_out["turnover_budget"], default=0.18), 0.08, 1.00)),
            "max_position_weight_target": float(np.clip(_finite_scalar(daily_out["max_position_weight_target"], default=0.12), 0.08, 0.28)),
            "hold_bias_target": float(np.clip(_finite_scalar(daily_out["hold_bias_target"], default=0.24), 0.10, 0.95)),
        }
        if supports_extended_budget_heads:
            global_targets["reduce_bias_target"] = float(
                np.clip(_finite_scalar(daily_out.get("reduce_bias_target", 0.10), default=0.10), 0.0, 0.65)
            )
            global_targets["exit_patience_target"] = float(
                np.clip(_finite_scalar(daily_out.get("exit_patience_target", 0.20), default=0.20), 0.05, 0.95)
            )
            global_targets["reentry_guard_target"] = float(
                np.clip(_finite_scalar(daily_out.get("reentry_guard_target", 0.0), default=0.0), 0.0, 0.45)
            )
        global_targets["budget_model_risk_signal"] = budget_model_risk_signal
        global_targets["budget_model_deploy_signal"] = budget_model_deploy_signal
        global_targets["budget_model_cash_timing_signal"] = budget_model_cash_timing_signal
        global_targets["budget_model_alpha_focus_signal"] = budget_model_alpha_focus_signal
        global_targets["budget_model_risk_deploy_gap"] = float(np.clip(budget_model_risk_signal - budget_model_deploy_signal, -1.0, 1.0))
        global_targets["budget_head_layout"] = str(getattr(artifact.daily_model, "head_layout", DAILY_HEAD_LAYOUT_MONOLITHIC_V1))

    current_weight = state_frame["current_weight"].astype(float).to_numpy(dtype=float) if "current_weight" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    held_mask = current_weight > 1e-8
    portfolio_sell_pressure = float(np.nanmean(sell_pressure[held_mask])) if bool(np.any(held_mask)) else (float(np.nanmean(sell_pressure)) if len(sell_pressure) else 0.0)
    portfolio_exit_hazard = float(np.nanmean(exit_hazard[held_mask])) if bool(np.any(held_mask)) else (float(np.nanmean(exit_hazard)) if len(exit_hazard) else 0.0)

    decoder_profile_name, decoder_profile = resolve_decoder_profile(artifact.train_summary.get("decoder_profile"))
    defensive_score = (
        0.40 * max(-_daily_feature_scalar(daily_features, "benchmark_trend_gap"), 0.0)
        + 0.15 * max(_daily_feature_scalar(daily_features, "benchmark_vol_ratio"), 0.0)
        + 0.15 * max(-_daily_feature_scalar(daily_features, "portfolio_drawdown_20d") - 0.02, 0.0)
        + 0.10 * max(_daily_feature_scalar(daily_features, "portfolio_turnover_pressure") - 0.80, 0.0)
    )
    reversal_pressure = (
        0.45 * max(_daily_feature_scalar(daily_features, "recent_reversal_rate_20d") - 0.12, 0.0)
        + 0.30 * max(_daily_feature_scalar(daily_features, "reduce_reversal_pressure") - 0.14, 0.0)
        + 0.25 * max(_daily_feature_scalar(daily_features, "exit_reentry_pressure") - 0.14, 0.0)
    )
    risk_off_score = (
        defensive_score
        + 0.32 * max(_daily_feature_scalar(daily_features, "market_downside_pressure"), 0.0)
        + 0.20 * max(_daily_feature_scalar(daily_features, "portfolio_cash_pressure"), 0.0)
        + 0.18 * max(_daily_feature_scalar(daily_features, "cash_regime_pressure") - 0.12, 0.0)
    )
    open_risk_off_score = float(
        np.clip(
            0.48 * max(_daily_feature_scalar(daily_features, "market_downside_pressure") - 0.12, 0.0)
            + 0.24 * max(_daily_feature_scalar(daily_features, "cash_regime_pressure") - 0.16, 0.0)
            + 0.16 * max(-_daily_feature_scalar(daily_features, "benchmark_trend_gap") - 0.03, 0.0)
            + 0.12 * max(_daily_feature_scalar(daily_features, "portfolio_cash_pressure") - 0.18, 0.0),
            0.0,
            1.0,
        )
    )
    held_exit_support_score = float(
        np.clip(
            0.34 * max(portfolio_sell_pressure - 0.18, 0.0)
            + 0.30 * max(portfolio_exit_hazard - 0.16, 0.0)
            + 0.14 * max(_daily_feature_scalar(daily_features, "market_downside_pressure") - 0.08, 0.0)
            + 0.12 * max(_daily_feature_scalar(daily_features, "cash_regime_pressure") - 0.10, 0.0)
            + 0.10 * max(-_daily_feature_scalar(daily_features, "benchmark_trend_gap") - 0.01, 0.0),
            0.0,
            1.0,
        )
    )
    is_holdcash_v3_decoder = decoder_profile_name == "holdcash_v3"
    min_gross_exposure_target = 0.18 if is_holdcash_v3_decoder else 0.12
    min_candidate_budget = 4.0 if is_holdcash_v3_decoder else 2.0
    min_position_cap_target = 0.10 if is_holdcash_v3_decoder else 0.08
    budget_objective_name = str(artifact.train_summary.get("budget_objective", "") or "").strip().lower()
    loss_profile_name = str(artifact.train_summary.get("loss_profile", "") or "").strip().lower()
    direct_action_value_mode = (
        loss_profile_name in DIRECT_ACTION_VALUE_LOSS_PROFILES
        or str(artifact.train_summary.get("policy_decision_mode", "") or "").strip().lower()
        == DIRECT_ACTION_VALUE_POLICY_MODE
    )
    if budget_objective_name == "result_value_v4b":
        opportunity_floor = float(
            np.clip(
                0.24
                + budget_model_alpha_focus_signal * 0.18
                + budget_model_deploy_signal * 0.16
                - budget_model_cash_timing_signal * 0.12
                - budget_model_risk_signal * 0.10,
                0.22,
                0.55,
            )
        )
        min_gross_exposure_target = max(float(min_gross_exposure_target), opportunity_floor)
        min_candidate_budget = max(float(min_candidate_budget), float(np.clip(3.0 + budget_model_alpha_focus_signal * 2.0, 3.0, 6.0)))
        min_position_cap_target = max(float(min_position_cap_target), 0.09)
    blended_budget_risk = float(np.clip(0.55 * risk_off_score + 0.45 * budget_model_risk_signal, 0.0, 1.0))
    blended_budget_deploy = float(
        np.clip(
            0.55 * budget_model_deploy_signal
            + 0.25 * budget_model_alpha_focus_signal
            + 0.20 * max(1.0 - blended_budget_risk, 0.0),
            0.0,
            1.0,
        )
    )
    if budget_objective_name in {"result_value_v3", "result_value_v4", "result_value_v4b"}:
        cash_regime_feature = float(np.clip(_daily_feature_scalar(daily_features, "cash_regime_pressure"), 0.0, 1.0))
        risk_gap = float(np.clip(blended_budget_risk - blended_budget_deploy, 0.0, 1.0))
        if budget_objective_name == "result_value_v3":
            cash_blend = 0.50
            deploy_cash_penalty = 0.10
            alpha_rebound = 0.04
            opportunity_cash_offset = 0.0
        elif budget_objective_name == "result_value_v4b":
            cash_blend = 0.48
            deploy_cash_penalty = 0.08
            alpha_rebound = 0.07
            opportunity_cash_offset = 0.08 * budget_model_alpha_focus_signal + 0.04 * max(blended_budget_deploy - blended_budget_risk, 0.0)
        else:
            cash_blend = 0.54
            deploy_cash_penalty = 0.14
            alpha_rebound = 0.03
            opportunity_cash_offset = 0.0
        blended_cash_timing = float(
            np.clip(
                cash_blend * budget_model_cash_timing_signal
                + 0.22 * blended_budget_risk
                + 0.18 * risk_gap
                + 0.10 * cash_regime_feature
                - opportunity_cash_offset,
                0.0,
                1.0,
            )
        )
        blended_budget_deploy = float(
            np.clip(
                blended_budget_deploy
                - blended_cash_timing * deploy_cash_penalty
                + budget_model_alpha_focus_signal * alpha_rebound,
                0.0,
                1.0,
            )
        )
    else:
        blended_cash_timing = float(np.clip(0.55 * budget_model_cash_timing_signal + 0.45 * blended_budget_risk, 0.0, 1.0))
    predicted_reduce_bias = float(global_targets.get("reduce_bias_target", 0.10) or 0.10)
    predicted_exit_patience = float(global_targets.get("exit_patience_target", 0.20) or 0.20)
    predicted_reentry_guard = float(global_targets.get("reentry_guard_target", 0.0) or 0.0)

    global_targets["gross_exposure_target"] = float(
        np.clip(
            global_targets["gross_exposure_target"]
            - blended_budget_risk * decoder_profile["defensive_cash_scale"]
            - blended_cash_timing * 0.08
            + blended_budget_deploy * 0.04,
            min_gross_exposure_target,
            0.98,
        )
    )
    global_targets["candidate_budget"] = float(
        np.clip(
            global_targets["candidate_budget"]
            - blended_budget_risk * decoder_profile["candidate_defensive_penalty"]
            - blended_cash_timing * 0.85
            + blended_budget_deploy * 0.95,
            min_candidate_budget,
            12.0,
        )
    )
    global_targets["turnover_budget"] = float(
        np.clip(
            global_targets["turnover_budget"]
            * (1.0 - blended_budget_risk * decoder_profile["turnover_defensive_penalty"] - reversal_pressure * 0.20)
            + blended_cash_timing * 0.05,
            0.08,
            1.00,
        )
    )
    global_targets["max_position_weight_target"] = float(
        np.clip(
            global_targets["max_position_weight_target"]
            - blended_budget_risk * decoder_profile["position_cap_defensive_penalty"]
            - blended_cash_timing * 0.010
            + budget_model_alpha_focus_signal * 0.012,
            min_position_cap_target,
            0.28,
        )
    )
    global_targets["hold_bias_target"] = float(
        np.clip(
            global_targets["hold_bias_target"]
            + blended_budget_risk * decoder_profile["hold_bias_bonus"]
            + reversal_pressure * (0.02 if is_holdcash_v3_decoder else 0.04)
            + budget_model_alpha_focus_signal * 0.06
            - blended_cash_timing * 0.06,
            0.10,
            0.95,
        )
    )
    global_targets["reduce_bias_target"] = float(
        np.clip(
            predicted_reduce_bias * 0.58
            + (
                (0.08 if is_holdcash_v3_decoder else 0.10)
                + blended_budget_risk * (0.12 if is_holdcash_v3_decoder else 0.22)
                + blended_cash_timing * 0.10
                + reversal_pressure * (0.05 if is_holdcash_v3_decoder else 0.06)
                + decoder_profile["reduce_bias_bonus"]
            )
            * 0.42,
            0.0,
            0.65,
        )
    )
    global_targets["exit_patience_target"] = float(
        np.clip(
            (
                predicted_exit_patience * 0.60
                + (
                    0.18
                    + global_targets["hold_bias_target"] * 0.16
                    + decoder_profile["exit_patience_bonus"]
                    + reversal_pressure * 0.08
                    - blended_budget_risk * (0.08 if is_holdcash_v3_decoder else 0.20)
                    - blended_cash_timing * 0.12
                    + budget_model_alpha_focus_signal * 0.04
                )
                * 0.40
            )
            if is_holdcash_v3_decoder
            else (
                predicted_exit_patience * 0.60
                + (
                    global_targets["hold_bias_target"]
                    + decoder_profile["exit_patience_bonus"]
                    - blended_budget_risk * 0.20
                    - blended_cash_timing * 0.12
                    + budget_model_alpha_focus_signal * 0.04
                )
                * 0.40
            ),
            0.10 if is_holdcash_v3_decoder else 0.05,
            0.95,
        )
    )
    global_targets["reentry_guard_target"] = float(
        np.clip(
            predicted_reentry_guard * 0.60
            + (
                decoder_profile["reversal_cooldown_bonus"]
                + reversal_pressure * (0.22 if is_holdcash_v3_decoder else 0.28)
                + blended_budget_risk * (0.04 if is_holdcash_v3_decoder else 0.08)
                + blended_cash_timing * 0.10
            )
            * 0.40,
            0.0,
            0.35 if is_holdcash_v3_decoder else 0.45,
        )
    )
    global_targets["gross_exposure_target"] = float(
        np.clip(
            global_targets["gross_exposure_target"] - portfolio_sell_pressure * 0.14 - portfolio_exit_hazard * 0.06,
            min_gross_exposure_target,
            0.98,
        )
    )
    global_targets["candidate_budget"] = float(
        np.clip(
            global_targets["candidate_budget"] - portfolio_sell_pressure * 1.00 - portfolio_exit_hazard * 0.35,
            min_candidate_budget,
            12.0,
        )
    )
    global_targets["turnover_budget"] = float(
        np.clip(
            global_targets["turnover_budget"] + portfolio_sell_pressure * 0.12 + portfolio_exit_hazard * 0.08,
            0.08,
            1.00,
        )
    )
    global_targets["max_position_weight_target"] = float(
        np.clip(
            global_targets["max_position_weight_target"] - portfolio_sell_pressure * 0.020,
            min_position_cap_target,
            0.28,
        )
    )
    global_targets["hold_bias_target"] = float(
        np.clip(
            global_targets["hold_bias_target"] - portfolio_sell_pressure * 0.16 - portfolio_exit_hazard * 0.08,
            0.10,
            0.95,
        )
    )
    global_targets["reduce_bias_target"] = float(
        np.clip(
            global_targets["reduce_bias_target"] + portfolio_sell_pressure * 0.16 + portfolio_exit_hazard * 0.07,
            0.0,
            0.65,
        )
    )
    global_targets["exit_patience_target"] = float(
        np.clip(
            global_targets["exit_patience_target"] - portfolio_exit_hazard * 0.22 - portfolio_sell_pressure * 0.08,
            0.10 if is_holdcash_v3_decoder else 0.05,
            0.95,
        )
    )
    global_targets["gross_exposure_target"] = float(
        np.clip(
            global_targets["gross_exposure_target"] - open_risk_off_score * 0.08 - held_exit_support_score * 0.02,
            min_gross_exposure_target,
            0.98,
        )
    )
    global_targets["candidate_budget"] = float(
        np.clip(
            global_targets["candidate_budget"] - open_risk_off_score * 0.55 - held_exit_support_score * 0.10,
            min_candidate_budget,
            12.0,
        )
    )
    global_targets["turnover_budget"] = float(
        np.clip(
            global_targets["turnover_budget"] + held_exit_support_score * 0.08 + open_risk_off_score * 0.02,
            0.08,
            1.00,
        )
    )
    global_targets["max_position_weight_target"] = float(
        np.clip(
            global_targets["max_position_weight_target"] - open_risk_off_score * 0.010 - held_exit_support_score * 0.004,
            min_position_cap_target,
            0.28,
        )
    )
    global_targets["hold_bias_target"] = float(
        np.clip(
            global_targets["hold_bias_target"] - open_risk_off_score * 0.06 - held_exit_support_score * 0.02,
            0.10,
            0.95,
        )
    )
    global_targets["reduce_bias_target"] = float(
        np.clip(
            global_targets["reduce_bias_target"] + held_exit_support_score * 0.08 + open_risk_off_score * 0.02,
            0.0,
            0.65,
        )
    )
    global_targets["exit_patience_target"] = float(
        np.clip(
            global_targets["exit_patience_target"] - held_exit_support_score * 0.16 - open_risk_off_score * 0.04,
            0.10 if is_holdcash_v3_decoder else 0.05,
            0.95,
        )
    )
    global_targets["reentry_guard_target"] = float(
        np.clip(
            global_targets["reentry_guard_target"] + open_risk_off_score * 0.02,
            0.0,
            0.35 if is_holdcash_v3_decoder else 0.45,
        )
    )
    global_targets = {
        "gross_exposure_target": float(np.clip(_finite_scalar(global_targets["gross_exposure_target"], default=0.35), min_gross_exposure_target, 0.98)),
        "candidate_budget": float(np.clip(_finite_scalar(global_targets["candidate_budget"], default=4.0), min_candidate_budget, 12.0)),
        "turnover_budget": float(np.clip(_finite_scalar(global_targets["turnover_budget"], default=0.18), 0.08, 1.00)),
        "max_position_weight_target": float(np.clip(_finite_scalar(global_targets["max_position_weight_target"], default=0.12), min_position_cap_target, 0.28)),
        "hold_bias_target": float(np.clip(_finite_scalar(global_targets["hold_bias_target"], default=0.24), 0.10, 0.95)),
        "reduce_bias_target": float(np.clip(_finite_scalar(global_targets["reduce_bias_target"], default=0.10), 0.0, 0.65)),
        "exit_patience_target": float(np.clip(_finite_scalar(global_targets["exit_patience_target"], default=0.20), 0.10 if is_holdcash_v3_decoder else 0.05, 0.95)),
        "reentry_guard_target": float(np.clip(_finite_scalar(global_targets["reentry_guard_target"], default=0.0), 0.0, 0.35 if is_holdcash_v3_decoder else 0.45)),
        "budget_model_risk_signal": float(budget_model_risk_signal),
        "budget_model_deploy_signal": float(budget_model_deploy_signal),
        "budget_model_cash_timing_signal": float(budget_model_cash_timing_signal),
        "budget_model_alpha_focus_signal": float(budget_model_alpha_focus_signal),
        "budget_model_risk_deploy_gap": float(np.clip(budget_model_risk_signal - budget_model_deploy_signal, -1.0, 1.0)),
        "budget_head_layout": str(getattr(artifact.daily_model, "head_layout", DAILY_HEAD_LAYOUT_MONOLITHIC_V1)),
    }

    probability_map = {label: action_prob[:, idx] for idx, label in enumerate(ACTION_CLASSES)}
    current_weight = state_frame["current_weight"].astype(float).to_numpy(dtype=float) if "current_weight" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    hold_days = state_frame["hold_days"].astype(float).to_numpy(dtype=float) if "hold_days" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    days_since_last_buy = state_frame["days_since_last_buy"].astype(float).to_numpy(dtype=float) if "days_since_last_buy" in state_frame.columns else np.full(len(state_frame), 99.0, dtype=float)
    days_since_last_reduce = state_frame["days_since_last_reduce"].astype(float).to_numpy(dtype=float) if "days_since_last_reduce" in state_frame.columns else np.full(len(state_frame), 99.0, dtype=float)
    days_since_last_exit = state_frame["days_since_last_exit"].astype(float).to_numpy(dtype=float) if "days_since_last_exit" in state_frame.columns else np.full(len(state_frame), 99.0, dtype=float)
    reentry_cooldown = state_frame["reentry_cooldown"].astype(float).to_numpy(dtype=float) if "reentry_cooldown" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    signal_decay_speed = state_frame["signal_decay_speed"].astype(float).to_numpy(dtype=float) if "signal_decay_speed" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    market_downside_pressure = state_frame["market_downside_pressure"].astype(float).to_numpy(dtype=float) if "market_downside_pressure" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    portfolio_cash_pressure = state_frame["portfolio_cash_pressure"].astype(float).to_numpy(dtype=float) if "portfolio_cash_pressure" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    recent_reversal_rate = state_frame["recent_reversal_rate_20d"].astype(float).to_numpy(dtype=float) if "recent_reversal_rate_20d" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    reduce_reversal_pressure = state_frame["reduce_reversal_pressure"].astype(float).to_numpy(dtype=float) if "reduce_reversal_pressure" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    exit_reentry_pressure = state_frame["exit_reentry_pressure"].astype(float).to_numpy(dtype=float) if "exit_reentry_pressure" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    cash_regime_pressure = state_frame["cash_regime_pressure"].astype(float).to_numpy(dtype=float) if "cash_regime_pressure" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    hold_continuity_pressure = state_frame["hold_continuity_pressure"].astype(float).to_numpy(dtype=float) if "hold_continuity_pressure" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    unrealized_pnl = state_frame["unrealized_pnl"].astype(float).to_numpy(dtype=float) if "unrealized_pnl" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    drawdown_from_peak = state_frame["drawdown_from_peak"].astype(float).to_numpy(dtype=float) if "drawdown_from_peak" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    price_from_local_peak = state_frame["price_from_local_peak"].astype(float).to_numpy(dtype=float) if "price_from_local_peak" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    pnl_rank_in_portfolio = state_frame["pnl_rank_in_portfolio"].astype(float).to_numpy(dtype=float) if "pnl_rank_in_portfolio" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    drawdown_rank_in_portfolio = state_frame["drawdown_rank_in_portfolio"].astype(float).to_numpy(dtype=float) if "drawdown_rank_in_portfolio" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    score_rank_pct = state_frame["score_rank_pct"].astype(float).to_numpy(dtype=float) if "score_rank_pct" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    alpha_prior_score_z = state_frame["alpha_prior_score_z"].astype(float).to_numpy(dtype=float) if "alpha_prior_score_z" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    alpha_prior_rank_pct = state_frame["alpha_prior_rank_pct"].astype(float).to_numpy(dtype=float) if "alpha_prior_rank_pct" in state_frame.columns else score_rank_pct
    alpha_prior_selected = state_frame["alpha_prior_selected"].astype(float).to_numpy(dtype=float) if "alpha_prior_selected" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    alpha_prior_target_weight = state_frame["alpha_prior_target_weight"].astype(float).to_numpy(dtype=float) if "alpha_prior_target_weight" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    current_gross_exposure = float(np.clip(np.nansum(current_weight), 0.0, 1.0))
    deployment_gap = float(
        np.clip(
            global_targets["gross_exposure_target"] - current_gross_exposure,
            0.0,
            1.0,
        )
    )
    fallback_sell_attribution = np.clip(
        0.34 * np.clip(reduce_quality, 0.0, None)
        + 0.18 * reduce_fraction
        + 0.16 * exit_hazard
        + 0.12 * sell_pressure
        + 0.10 * np.clip(drawdown_rank_in_portfolio, 0.0, None)
        + 0.08 * max(budget_model_cash_timing_signal, blended_budget_risk)
        - 0.24 * np.clip(hold_quality, 0.0, None)
        - 0.10 * np.clip(add_quality, 0.0, None)
        - 0.10 * np.clip(score_rank_pct, 0.0, None),
        0.0,
        1.0,
    )
    preliminary_sell_attribution = (
        np.clip(0.70 * predicted_sell_attribution + 0.30 * fallback_sell_attribution, 0.0, 1.0)
        if predicted_sell_attribution is not None
        else fallback_sell_attribution
    )
    fallback_sell_rank = np.zeros(len(state_frame), dtype=float)
    if bool(np.any(current_weight > 1e-8)):
        held_indices = np.flatnonzero(current_weight > 1e-8)
        held_values = pd.Series(preliminary_sell_attribution[held_indices], index=held_indices, dtype=float)
        if len(held_values) >= 2:
            fallback_sell_rank[held_indices] = held_values.rank(method="average", pct=True).to_numpy(dtype=float)
        else:
            fallback_sell_rank[held_indices] = held_values.to_numpy(dtype=float)
    sell_rank_score = (
        np.clip(0.74 * predicted_sell_rank + 0.26 * fallback_sell_rank, 0.0, 1.0)
        if predicted_sell_rank is not None
        else fallback_sell_rank
    )
    lifecycle_fallback = np.clip(
        0.36 * preliminary_sell_attribution
        + 0.28 * sell_rank_score
        + 0.20 * sell_pressure
        + 0.12 * exit_hazard
        - 0.16 * np.clip(hold_quality, 0.0, None),
        0.0,
        1.0,
    )
    lifecycle_sell_gate = (
        np.clip(0.72 * predicted_lifecycle_sell_gate + 0.28 * lifecycle_fallback, 0.0, 1.0)
        if predicted_lifecycle_sell_gate is not None
        else lifecycle_fallback
    )
    clip_fallback = np.clip(
        blended_budget_risk * 0.28
        + budget_model_cash_timing_signal * 0.24
        + max(deployment_gap - 0.10, 0.0) * 0.36
        + reversal_pressure * 0.12,
        0.0,
        1.0,
    )
    clipped_intent_risk = (
        np.clip(0.78 * predicted_clipped_intent_risk + 0.22 * clip_fallback, 0.0, 1.0)
        if predicted_clipped_intent_risk is not None
        else np.full(len(state_frame), clip_fallback, dtype=float)
    )
    alpha_prior_support = np.clip(
        0.34 * np.clip(alpha_prior_score_z / 2.0, 0.0, 1.0)
        + 0.28 * np.clip(alpha_prior_rank_pct, 0.0, 1.0)
        + 0.22 * np.clip(alpha_prior_target_weight / 0.12, 0.0, 1.0)
        + 0.16 * np.clip(alpha_prior_selected, 0.0, 1.0),
        0.0,
        1.0,
    )
    fallback_large_upside = np.clip(
        probability_map["open"] * 0.26
        + probability_map["add"] * 0.20
        + np.clip(entry_quality, 0.0, None) * 0.18
        + alpha_prior_support * 0.20
        + np.clip(score_rank_pct, 0.0, 1.0) * 0.16
        - blended_cash_timing * 0.14
        - clipped_intent_risk * 0.08,
        0.0,
        1.0,
    )
    large_upside_1d_target = (
        np.clip(0.72 * predicted_large_upside + 0.28 * fallback_large_upside, 0.0, 1.0)
        if predicted_large_upside is not None
        else fallback_large_upside
    )
    fallback_alpha_opportunity = np.clip(
        0.28 * alpha_prior_support
        + 0.22 * np.clip(entry_quality, 0.0, None)
        + 0.16 * np.clip(add_quality, 0.0, None)
        + 0.16 * large_upside_1d_target
        + 0.12 * probability_map["open"]
        + 0.06 * probability_map["add"]
        - 0.16 * blended_cash_timing
        - 0.10 * preliminary_sell_attribution,
        0.0,
        1.0,
    )
    alpha_opportunity_value = (
        np.clip(0.72 * predicted_alpha_opportunity + 0.28 * fallback_alpha_opportunity, 0.0, 1.0)
        if predicted_alpha_opportunity is not None
        else fallback_alpha_opportunity
    )
    fallback_cash_defense = np.clip(
        0.30 * blended_cash_timing
        + 0.24 * blended_budget_risk
        + 0.16 * np.clip(cash_regime_pressure, 0.0, 1.0)
        + 0.12 * np.clip(market_downside_pressure, 0.0, 1.0)
        + 0.10 * clipped_intent_risk
        + 0.08 * preliminary_sell_attribution
        - 0.18 * alpha_opportunity_value,
        0.0,
        1.0,
    )
    cash_defense_value = (
        np.clip(0.70 * predicted_cash_defense + 0.30 * fallback_cash_defense, 0.0, 1.0)
        if predicted_cash_defense is not None
        else fallback_cash_defense
    )
    fallback_hold_continuation = np.clip(
        0.30 * np.clip(hold_quality, 0.0, None)
        + 0.18 * np.clip(add_quality, 0.0, None)
        + 0.18 * alpha_opportunity_value
        + 0.14 * np.clip(hold_continuity_pressure, 0.0, 1.0)
        + 0.10 * large_upside_1d_target
        + 0.10 * (1.0 - preliminary_sell_attribution)
        - 0.18 * lifecycle_sell_gate
        - 0.10 * cash_defense_value,
        0.0,
        1.0,
    )
    hold_continuation_value = (
        np.clip(0.72 * predicted_hold_continuation + 0.28 * fallback_hold_continuation, 0.0, 1.0)
        if predicted_hold_continuation is not None
        else fallback_hold_continuation
    )
    fallback_sell_release = np.clip(
        0.30 * preliminary_sell_attribution
        + 0.22 * sell_rank_score
        + 0.20 * lifecycle_sell_gate
        + 0.12 * cash_defense_value
        + 0.10 * sell_pressure
        + 0.06 * exit_hazard
        - 0.18 * alpha_opportunity_value
        - 0.12 * hold_continuation_value,
        0.0,
        1.0,
    )
    sell_release_value = (
        np.clip(0.72 * predicted_sell_release + 0.28 * fallback_sell_release, 0.0, 1.0)
        if predicted_sell_release is not None
        else fallback_sell_release
    )
    fallback_deployment_cost = np.clip(
        0.32 * alpha_opportunity_value
        + 0.22 * large_upside_1d_target
        + 0.18 * np.clip(entry_quality, 0.0, None)
        + 0.10 * np.clip(add_quality, 0.0, None)
        + 0.10 * max(deployment_gap, 0.0)
        + 0.08 * alpha_prior_support
        - 0.18 * cash_defense_value
        - 0.10 * sell_release_value,
        0.0,
        1.0,
    )
    deployment_opportunity_cost = (
        np.clip(0.70 * predicted_deployment_cost + 0.30 * fallback_deployment_cost, 0.0, 1.0)
        if predicted_deployment_cost is not None
        else fallback_deployment_cost
    )
    capital_value = np.maximum(
        np.clip(0.56 * deployment_opportunity_cost + 0.34 * alpha_opportunity_value + 0.10 * large_upside_1d_target, 0.0, 1.0),
        np.clip(0.62 * hold_continuation_value + 0.24 * alpha_opportunity_value + 0.14 * large_upside_1d_target, 0.0, 1.0),
    )
    defense_value = np.maximum(
        np.clip(0.62 * sell_release_value + 0.24 * cash_defense_value + 0.14 * lifecycle_sell_gate, 0.0, 1.0),
        np.clip(0.68 * cash_defense_value + 0.20 * (1.0 - alpha_opportunity_value) + 0.12 * clipped_intent_risk, 0.0, 1.0),
    )
    fallback_value_arbitration = np.clip(0.50 + 0.55 * (capital_value - defense_value), 0.0, 1.0)
    value_arbitration_target = (
        np.clip(0.72 * predicted_value_arbitration + 0.28 * fallback_value_arbitration, 0.0, 1.0)
        if predicted_value_arbitration is not None
        else fallback_value_arbitration
    )
    fallback_deploy_value_target = np.clip(capital_value, 0.0, 1.0)
    fallback_release_value_target = np.clip(
        0.62 * sell_release_value + 0.24 * cash_defense_value + 0.14 * lifecycle_sell_gate,
        0.0,
        1.0,
    )
    fallback_defense_value_target = np.clip(
        0.68 * cash_defense_value + 0.20 * (1.0 - alpha_opportunity_value) + 0.12 * clipped_intent_risk,
        0.0,
        1.0,
    )
    deploy_value_target = (
        np.clip(0.72 * predicted_deploy_value + 0.28 * fallback_deploy_value_target, 0.0, 1.0)
        if predicted_deploy_value is not None
        else fallback_deploy_value_target
    )
    release_value_target = (
        np.clip(0.72 * predicted_release_value + 0.28 * fallback_release_value_target, 0.0, 1.0)
        if predicted_release_value is not None
        else fallback_release_value_target
    )
    defense_value_target = (
        np.clip(0.72 * predicted_defense_value + 0.28 * fallback_defense_value_target, 0.0, 1.0)
        if predicted_defense_value is not None
        else fallback_defense_value_target
    )
    fallback_gate_denominator = deploy_value_target + release_value_target + defense_value_target + 1.0e-6
    fallback_deploy_gate = np.clip(deploy_value_target / fallback_gate_denominator, 0.0, 1.0)
    fallback_release_gate = np.clip(release_value_target / fallback_gate_denominator, 0.0, 1.0)
    fallback_defense_gate = np.clip(defense_value_target / fallback_gate_denominator, 0.0, 1.0)
    deploy_gate_target = (
        np.clip(0.68 * predicted_deploy_gate + 0.32 * fallback_deploy_gate, 0.0, 1.0)
        if predicted_deploy_gate is not None
        else fallback_deploy_gate
    )
    release_gate_target = (
        np.clip(0.68 * predicted_release_gate + 0.32 * fallback_release_gate, 0.0, 1.0)
        if predicted_release_gate is not None
        else fallback_release_gate
    )
    defense_gate_target = (
        np.clip(0.68 * predicted_defense_gate + 0.32 * fallback_defense_gate, 0.0, 1.0)
        if predicted_defense_gate is not None
        else fallback_defense_gate
    )
    fallback_risk_adjusted_action = np.clip(
        value_arbitration_target
        + 0.16 * alpha_opportunity_value
        + 0.10 * hold_continuation_value
        - 0.14 * sell_release_value
        - 0.10 * cash_defense_value,
        0.0,
        1.0,
    )
    risk_adjusted_action_value = (
        np.clip(0.70 * predicted_risk_adjusted_action_value + 0.30 * fallback_risk_adjusted_action, 0.0, 1.0)
        if predicted_risk_adjusted_action_value is not None
        else fallback_risk_adjusted_action
    )
    fallback_multi_horizon_forward_value = np.clip(
        0.30 * alpha_opportunity_value
        + 0.22 * large_upside_1d_target
        + 0.18 * deploy_value_target
        + 0.14 * np.clip(entry_quality, 0.0, None)
        + 0.10 * np.clip(add_quality, 0.0, None)
        + 0.06 * np.clip(score_rank_pct, 0.0, 1.0),
        0.0,
        1.0,
    )
    fallback_multi_horizon_forward_risk = np.clip(
        0.28 * sell_release_value
        + 0.20 * cash_defense_value
        + 0.16 * lifecycle_sell_gate
        + 0.14 * np.clip(exit_urgency, 0.0, None)
        + 0.12 * clipped_intent_risk
        + 0.10 * np.clip(market_downside_pressure, 0.0, 1.0)
        - 0.18 * alpha_opportunity_value
        - 0.10 * hold_continuation_value,
        0.0,
        1.0,
    )
    multi_horizon_forward_value = (
        np.clip(0.70 * predicted_multi_horizon_forward_value + 0.30 * fallback_multi_horizon_forward_value, 0.0, 1.0)
        if predicted_multi_horizon_forward_value is not None
        else fallback_multi_horizon_forward_value
    )
    multi_horizon_forward_risk = (
        np.clip(0.70 * predicted_multi_horizon_forward_risk + 0.30 * fallback_multi_horizon_forward_risk, 0.0, 1.0)
        if predicted_multi_horizon_forward_risk is not None
        else fallback_multi_horizon_forward_risk
    )
    fallback_multi_horizon_path_value = np.clip(
        0.52 * multi_horizon_forward_value
        + 0.22 * alpha_opportunity_value
        + 0.14 * deploy_value_target
        + 0.12 * large_upside_1d_target
        - 0.36 * multi_horizon_forward_risk,
        0.0,
        1.0,
    )
    multi_horizon_path_value = (
        np.clip(0.70 * predicted_multi_horizon_path_value + 0.30 * fallback_multi_horizon_path_value, 0.0, 1.0)
        if predicted_multi_horizon_path_value is not None
        else fallback_multi_horizon_path_value
    )
    held_array = (current_weight > 1.0e-8).astype(float)
    fallback_open_action_value = np.clip(
        0.30 * deployment_opportunity_cost
        + 0.26 * multi_horizon_path_value
        + 0.18 * alpha_opportunity_value
        + 0.12 * deploy_gate_target
        + 0.08 * large_upside_1d_target
        + 0.06 * np.clip(entry_quality, 0.0, None)
        - 0.20 * cash_defense_value
        - 0.10 * release_value_target,
        0.0,
        1.0,
    ) * (1.0 - held_array)
    fallback_add_action_value = np.clip(
        0.26 * hold_continuation_value
        + 0.24 * multi_horizon_path_value
        + 0.18 * alpha_opportunity_value
        + 0.12 * deploy_value_target
        + 0.10 * deploy_gate_target
        + 0.10 * np.clip(add_quality, 0.0, None)
        - 0.22 * sell_release_value
        - 0.12 * multi_horizon_forward_risk,
        0.0,
        1.0,
    ) * held_array
    fallback_hold_action_value = np.clip(
        0.30 * hold_continuation_value
        + 0.24 * multi_horizon_path_value
        + 0.18 * np.clip(hold_quality, 0.0, None)
        + 0.12 * alpha_opportunity_value
        + 0.08 * deploy_gate_target
        + 0.08 * np.clip(hold_continuity_pressure, 0.0, 1.0)
        - 0.22 * sell_release_value
        - 0.12 * multi_horizon_forward_risk,
        0.0,
        1.0,
    ) * held_array
    fallback_relative_opportunity_value = np.clip(
        0.42 * deployment_opportunity_cost
        + 0.24 * alpha_opportunity_value
        + 0.20 * multi_horizon_path_value
        + 0.08 * large_upside_1d_target
        + 0.06 * deploy_gate_target
        - 0.24 * hold_continuation_value,
        0.0,
        1.0,
    )
    relative_opportunity_value = (
        np.clip(0.70 * predicted_relative_opportunity_value + 0.30 * fallback_relative_opportunity_value, 0.0, 1.0)
        if predicted_relative_opportunity_value is not None
        else fallback_relative_opportunity_value
    )
    fallback_reduce_action_value = np.clip(
        0.28 * sell_release_value
        + 0.22 * multi_horizon_forward_risk
        + 0.16 * relative_opportunity_value
        + 0.12 * release_value_target
        + 0.10 * sell_rank_score
        + 0.08 * lifecycle_sell_gate
        + 0.04 * reduce_fraction
        - 0.22 * fallback_hold_action_value
        - 0.10 * fallback_add_action_value,
        0.0,
        1.0,
    ) * held_array
    fallback_exit_action_value = np.clip(
        0.30 * sell_release_value
        + 0.24 * multi_horizon_forward_risk
        + 0.16 * cash_defense_value
        + 0.12 * lifecycle_sell_gate
        + 0.10 * exit_hazard
        + 0.08 * relative_opportunity_value
        - 0.24 * fallback_hold_action_value
        - 0.10 * alpha_opportunity_value,
        0.0,
        1.0,
    ) * held_array
    open_action_value = (
        np.clip(0.70 * predicted_open_action_value + 0.30 * fallback_open_action_value, 0.0, 1.0)
        if predicted_open_action_value is not None
        else fallback_open_action_value
    )
    add_action_value = (
        np.clip(0.70 * predicted_add_action_value + 0.30 * fallback_add_action_value, 0.0, 1.0) * held_array
        if predicted_add_action_value is not None
        else fallback_add_action_value
    )
    hold_action_value = (
        np.clip(0.70 * predicted_hold_action_value + 0.30 * fallback_hold_action_value, 0.0, 1.0) * held_array
        if predicted_hold_action_value is not None
        else fallback_hold_action_value
    )
    reduce_action_value = (
        np.clip(0.70 * predicted_reduce_action_value + 0.30 * fallback_reduce_action_value, 0.0, 1.0) * held_array
        if predicted_reduce_action_value is not None
        else fallback_reduce_action_value
    )
    exit_action_value = (
        np.clip(0.70 * predicted_exit_action_value + 0.30 * fallback_exit_action_value, 0.0, 1.0) * held_array
        if predicted_exit_action_value is not None
        else fallback_exit_action_value
    )
    fallback_action_value_consistency = np.clip(
        0.50
        + 0.55
        * (
            np.maximum(open_action_value, np.maximum(add_action_value, hold_action_value))
            - np.maximum(reduce_action_value, np.maximum(exit_action_value, cash_defense_value))
        ),
        0.0,
        1.0,
    )
    action_value_consistency_target = (
        np.clip(0.70 * predicted_action_value_consistency + 0.30 * fallback_action_value_consistency, 0.0, 1.0)
        if predicted_action_value_consistency is not None
        else fallback_action_value_consistency
    )
    large_upside_1d_target = _finite_array(large_upside_1d_target, default=0.0, low=0.0, high=1.0)
    alpha_opportunity_value = _finite_array(alpha_opportunity_value, default=0.0, low=0.0, high=1.0)
    hold_continuation_value = _finite_array(hold_continuation_value, default=0.0, low=0.0, high=1.0)
    sell_release_value = _finite_array(sell_release_value, default=0.0, low=0.0, high=1.0)
    cash_defense_value = _finite_array(cash_defense_value, default=0.0, low=0.0, high=1.0)
    deployment_opportunity_cost = _finite_array(deployment_opportunity_cost, default=0.0, low=0.0, high=1.0)
    value_arbitration_target = _finite_array(value_arbitration_target, default=0.5, low=0.0, high=1.0)
    deploy_value_target = _finite_array(deploy_value_target, default=0.0, low=0.0, high=1.0)
    release_value_target = _finite_array(release_value_target, default=0.0, low=0.0, high=1.0)
    defense_value_target = _finite_array(defense_value_target, default=0.0, low=0.0, high=1.0)
    deploy_gate_target = _finite_array(deploy_gate_target, default=0.0, low=0.0, high=1.0)
    release_gate_target = _finite_array(release_gate_target, default=0.0, low=0.0, high=1.0)
    defense_gate_target = _finite_array(defense_gate_target, default=0.0, low=0.0, high=1.0)
    risk_adjusted_action_value = _finite_array(risk_adjusted_action_value, default=0.0, low=0.0, high=1.0)
    multi_horizon_forward_value = _finite_array(multi_horizon_forward_value, default=0.0, low=0.0, high=1.0)
    multi_horizon_forward_risk = _finite_array(multi_horizon_forward_risk, default=0.0, low=0.0, high=1.0)
    multi_horizon_path_value = _finite_array(multi_horizon_path_value, default=0.0, low=0.0, high=1.0)
    open_action_value = _finite_array(open_action_value, default=0.0, low=0.0, high=1.0)
    add_action_value = _finite_array(add_action_value, default=0.0, low=0.0, high=1.0)
    hold_action_value = _finite_array(hold_action_value, default=0.0, low=0.0, high=1.0)
    reduce_action_value = _finite_array(reduce_action_value, default=0.0, low=0.0, high=1.0)
    exit_action_value = _finite_array(exit_action_value, default=0.0, low=0.0, high=1.0)
    relative_opportunity_value = _finite_array(relative_opportunity_value, default=0.0, low=0.0, high=1.0)
    action_value_consistency_target = _finite_array(action_value_consistency_target, default=0.5, low=0.0, high=1.0)
    portfolio_value_arbitration = _finite_mean(value_arbitration_target, default=0.5)
    portfolio_alpha_opportunity = _finite_mean(alpha_opportunity_value, default=0.0)
    portfolio_cash_defense = _finite_mean(cash_defense_value, default=0.0)
    if budget_objective_name in {"result_value_v8", "result_value_v9"}:
        portfolio_deploy_value = _finite_mean(deploy_value_target, default=0.0)
        portfolio_release_value = _finite_mean(release_value_target, default=0.0)
        portfolio_defense_value = _finite_mean(defense_value_target, default=0.0)
        deploy_release_denominator = deploy_value_target + release_value_target + 1.0e-6
        hierarchical_deploy_gate_target = np.clip(
            0.62 * deploy_gate_target + 0.38 * np.clip(deploy_value_target / deploy_release_denominator, 0.0, 1.0),
            0.0,
            1.0,
        )
        hierarchical_release_gate_target = np.clip(
            0.62 * release_gate_target + 0.38 * np.clip(release_value_target / deploy_release_denominator, 0.0, 1.0),
            0.0,
            1.0,
        )
        portfolio_deploy_gate = _finite_mean(hierarchical_deploy_gate_target, default=0.0)
        portfolio_release_gate = _finite_mean(hierarchical_release_gate_target, default=0.0)
        portfolio_defense_gate = float(
            np.clip(
                0.48 * float(global_targets.get("budget_cash_timing_signal_target", 0.0) or 0.0)
                + 0.26 * float(global_targets.get("budget_risk_signal_target", 0.0) or 0.0)
                + 0.12 * portfolio_cash_defense
                + 0.08 * portfolio_defense_value
                + 0.08 * max(portfolio_release_gate - portfolio_deploy_gate, 0.0)
                - 0.08 * portfolio_deploy_gate,
                0.0,
                1.0,
            )
        )
        three_value_deploy_pressure = float(
            np.clip(
                0.48 * portfolio_deploy_gate
                + 0.30 * portfolio_deploy_value
                + 0.14 * portfolio_alpha_opportunity
                + 0.08 * _finite_mean(deployment_opportunity_cost, default=0.0),
                0.0,
                1.0,
            )
        )
        three_value_release_pressure = float(
            np.clip(
                0.40 * portfolio_release_gate
                + 0.34 * portfolio_release_value
                + 0.16 * portfolio_cash_defense
                + 0.10 * max(portfolio_defense_gate - portfolio_deploy_gate, 0.0),
                0.0,
                1.0,
            )
        )
        global_targets["gross_exposure_target"] = float(
            np.clip(
                global_targets["gross_exposure_target"]
                + three_value_deploy_pressure * 0.060
                - portfolio_defense_gate * 0.072
                - three_value_release_pressure * 0.028,
                min_gross_exposure_target,
                0.98,
            )
        )
        global_targets["candidate_budget"] = float(
            np.clip(
                global_targets["candidate_budget"]
                + three_value_deploy_pressure * 0.90
                - portfolio_defense_gate * 0.62
                - three_value_release_pressure * 0.18,
                min_candidate_budget,
                12.0,
            )
        )
        global_targets["turnover_budget"] = float(
            np.clip(
                global_targets["turnover_budget"]
                + three_value_release_pressure * 0.070
                + portfolio_defense_gate * 0.022
                + max(portfolio_deploy_gate - portfolio_release_gate, 0.0) * 0.018,
                0.08,
                1.00,
            )
        )
        global_targets["budget_model_value_arbitration_signal"] = float(portfolio_value_arbitration)
        global_targets["budget_model_alpha_opportunity_signal"] = float(portfolio_alpha_opportunity)
        global_targets["budget_model_cash_defense_signal"] = float(portfolio_cash_defense)
        global_targets["budget_model_deploy_value_signal"] = float(portfolio_deploy_value)
        global_targets["budget_model_release_value_signal"] = float(portfolio_release_value)
        global_targets["budget_model_defense_value_signal"] = float(portfolio_defense_value)
        global_targets["budget_model_deploy_gate_signal"] = float(portfolio_deploy_gate)
        global_targets["budget_model_release_gate_signal"] = float(portfolio_release_gate)
        global_targets["budget_model_defense_gate_signal"] = float(portfolio_defense_gate)
        global_targets["budget_model_hierarchical_mode"] = 1.0
        global_targets["budget_model_constraint_only_mode"] = 1.0
    elif budget_objective_name == "result_value_v7":
        portfolio_deploy_value = _finite_mean(deploy_value_target, default=0.0)
        portfolio_release_value = _finite_mean(release_value_target, default=0.0)
        portfolio_defense_value = _finite_mean(defense_value_target, default=0.0)
        deploy_release_denominator = deploy_value_target + release_value_target + 1.0e-6
        hierarchical_deploy_gate_target = np.clip(
            0.62 * deploy_gate_target + 0.38 * np.clip(deploy_value_target / deploy_release_denominator, 0.0, 1.0),
            0.0,
            1.0,
        )
        hierarchical_release_gate_target = np.clip(
            0.62 * release_gate_target + 0.38 * np.clip(release_value_target / deploy_release_denominator, 0.0, 1.0),
            0.0,
            1.0,
        )
        portfolio_deploy_gate = _finite_mean(hierarchical_deploy_gate_target, default=0.0)
        portfolio_release_gate = _finite_mean(hierarchical_release_gate_target, default=0.0)
        portfolio_defense_gate = float(
            np.clip(
                0.42 * float(global_targets.get("budget_cash_timing_signal_target", 0.0) or 0.0)
                + 0.24 * float(global_targets.get("budget_risk_signal_target", 0.0) or 0.0)
                + 0.18 * portfolio_cash_defense
                + 0.10 * portfolio_defense_value
                + 0.06 * portfolio_release_gate
                - 0.12 * portfolio_deploy_gate,
                0.0,
                1.0,
            )
        )
        three_value_deploy_pressure = float(
            np.clip(
                0.46 * portfolio_deploy_gate
                + 0.30 * portfolio_deploy_value
                + 0.16 * portfolio_alpha_opportunity
                + 0.08 * _finite_mean(deployment_opportunity_cost, default=0.0),
                0.0,
                1.0,
            )
        )
        global_targets["gross_exposure_target"] = float(
            np.clip(
                global_targets["gross_exposure_target"]
                + three_value_deploy_pressure * 0.065
                - portfolio_defense_gate * 0.060
                - max(portfolio_release_gate - portfolio_deploy_gate, 0.0) * 0.025,
                min_gross_exposure_target,
                0.98,
            )
        )
        global_targets["candidate_budget"] = float(
            np.clip(
                global_targets["candidate_budget"]
                + three_value_deploy_pressure * 0.95
                - portfolio_defense_gate * 0.55
                - max(portfolio_release_gate - portfolio_deploy_gate, 0.0) * 0.25,
                min_candidate_budget,
                12.0,
            )
        )
        global_targets["turnover_budget"] = float(
            np.clip(
                global_targets["turnover_budget"]
                + max(portfolio_release_gate - portfolio_deploy_gate, 0.0) * 0.055
                + max(portfolio_deploy_gate - portfolio_release_gate, 0.0) * 0.030
                + portfolio_defense_gate * 0.018,
                0.08,
                1.00,
            )
        )
        global_targets["budget_model_value_arbitration_signal"] = float(portfolio_value_arbitration)
        global_targets["budget_model_alpha_opportunity_signal"] = float(portfolio_alpha_opportunity)
        global_targets["budget_model_cash_defense_signal"] = float(portfolio_cash_defense)
        global_targets["budget_model_deploy_value_signal"] = float(portfolio_deploy_value)
        global_targets["budget_model_release_value_signal"] = float(portfolio_release_value)
        global_targets["budget_model_defense_value_signal"] = float(portfolio_defense_value)
        global_targets["budget_model_deploy_gate_signal"] = float(portfolio_deploy_gate)
        global_targets["budget_model_release_gate_signal"] = float(portfolio_release_gate)
        global_targets["budget_model_defense_gate_signal"] = float(portfolio_defense_gate)
        global_targets["budget_model_hierarchical_mode"] = 1.0
        global_targets["budget_model_constraint_only_mode"] = 0.0
    elif budget_objective_name == "result_value_v6":
        portfolio_deploy_value = _finite_mean(deploy_value_target, default=0.0)
        portfolio_release_value = _finite_mean(release_value_target, default=0.0)
        portfolio_defense_value = _finite_mean(defense_value_target, default=0.0)
        portfolio_deploy_gate = _finite_mean(deploy_gate_target, default=0.0)
        portfolio_release_gate = _finite_mean(release_gate_target, default=0.0)
        portfolio_defense_gate = _finite_mean(defense_gate_target, default=0.0)
        three_value_deploy_pressure = float(
            np.clip(
                0.42 * portfolio_deploy_gate
                + 0.32 * portfolio_deploy_value
                + 0.16 * portfolio_alpha_opportunity
                + 0.10 * _finite_mean(deployment_opportunity_cost, default=0.0),
                0.0,
                1.0,
            )
        )
        three_value_defense_pressure = float(
            np.clip(
                0.36 * portfolio_defense_gate
                + 0.28 * portfolio_defense_value
                + 0.20 * portfolio_cash_defense
                + 0.16 * portfolio_release_gate,
                0.0,
                1.0,
            )
        )
        global_targets["gross_exposure_target"] = float(
            np.clip(
                global_targets["gross_exposure_target"]
                + three_value_deploy_pressure * 0.060
                - three_value_defense_pressure * 0.055
                - max(portfolio_release_gate - portfolio_deploy_gate, 0.0) * 0.030,
                min_gross_exposure_target,
                0.98,
            )
        )
        global_targets["candidate_budget"] = float(
            np.clip(
                global_targets["candidate_budget"]
                + three_value_deploy_pressure * 0.90
                - portfolio_defense_gate * 0.50
                - max(portfolio_release_gate - portfolio_deploy_gate, 0.0) * 0.30,
                min_candidate_budget,
                12.0,
            )
        )
        global_targets["turnover_budget"] = float(
            np.clip(
                global_targets["turnover_budget"]
                + max(portfolio_release_gate - portfolio_deploy_gate, 0.0) * 0.050
                + max(portfolio_deploy_gate - portfolio_release_gate, 0.0) * 0.035
                + portfolio_defense_gate * 0.020,
                0.08,
                1.00,
            )
        )
        global_targets["budget_model_value_arbitration_signal"] = float(portfolio_value_arbitration)
        global_targets["budget_model_alpha_opportunity_signal"] = float(portfolio_alpha_opportunity)
        global_targets["budget_model_cash_defense_signal"] = float(portfolio_cash_defense)
        global_targets["budget_model_deploy_value_signal"] = float(portfolio_deploy_value)
        global_targets["budget_model_release_value_signal"] = float(portfolio_release_value)
        global_targets["budget_model_defense_value_signal"] = float(portfolio_defense_value)
        global_targets["budget_model_deploy_gate_signal"] = float(portfolio_deploy_gate)
        global_targets["budget_model_release_gate_signal"] = float(portfolio_release_gate)
        global_targets["budget_model_defense_gate_signal"] = float(portfolio_defense_gate)
        global_targets["budget_model_hierarchical_mode"] = 0.0
        global_targets["budget_model_constraint_only_mode"] = 0.0
    elif budget_objective_name == "result_value_v5":
        portfolio_deployment_cost = _finite_mean(deployment_opportunity_cost, default=0.0)
        portfolio_sell_release = _finite_mean(sell_release_value, default=0.0)
        value_deploy_pressure = float(
            np.clip(
                0.46 * portfolio_alpha_opportunity
                + 0.34 * max(portfolio_value_arbitration - 0.50, 0.0) * 2.0
                + 0.20 * portfolio_deployment_cost,
                0.0,
                1.0,
            )
        )
        value_cash_pressure = float(
            np.clip(
                0.58 * portfolio_cash_defense
                + 0.28 * max(0.50 - portfolio_value_arbitration, 0.0) * 2.0
                + 0.14 * portfolio_sell_release,
                0.0,
                1.0,
            )
        )
        global_targets["gross_exposure_target"] = float(
            np.clip(
                global_targets["gross_exposure_target"]
                + value_deploy_pressure * 0.045
                - value_cash_pressure * 0.060,
                min_gross_exposure_target,
                0.98,
            )
        )
        global_targets["candidate_budget"] = float(
            np.clip(
                global_targets["candidate_budget"]
                + value_deploy_pressure * 0.70
                - value_cash_pressure * 0.45,
                min_candidate_budget,
                12.0,
            )
        )
        global_targets["turnover_budget"] = float(
            np.clip(
                global_targets["turnover_budget"]
                + max(value_deploy_pressure - value_cash_pressure, 0.0) * 0.030
                + max(value_cash_pressure - value_deploy_pressure, 0.0) * 0.045,
                0.08,
                1.00,
            )
        )
        global_targets["budget_model_value_arbitration_signal"] = float(portfolio_value_arbitration)
        global_targets["budget_model_alpha_opportunity_signal"] = float(portfolio_alpha_opportunity)
        global_targets["budget_model_cash_defense_signal"] = float(portfolio_cash_defense)
        global_targets["budget_model_hierarchical_mode"] = 0.0
        global_targets["budget_model_constraint_only_mode"] = 0.0
    else:
        global_targets["budget_model_hierarchical_mode"] = 0.0
        global_targets["budget_model_constraint_only_mode"] = 0.0
    cash_pressure_scalar = float(np.clip(np.nanmean(portfolio_cash_pressure), 0.0, 1.0)) if len(portfolio_cash_pressure) else 0.0
    turnover_ramp_bonus = float(
        np.clip(
            max(deployment_gap - 0.08, 0.0) * (0.24 if is_holdcash_v3_decoder else 0.20)
            + max(cash_pressure_scalar - 0.12, 0.0) * 0.10
            - reversal_pressure * 0.04,
            0.0,
            0.08 if is_holdcash_v3_decoder else 0.06,
        )
    )
    if deployment_gap > 0.12 or cash_pressure_scalar > 0.18:
        turnover_floor = float(
            np.clip(
                0.12
                + max(deployment_gap - 0.12, 0.0) * 0.18
                + max(cash_pressure_scalar - 0.18, 0.0) * 0.08,
                0.12,
                0.22 if is_holdcash_v3_decoder else 0.18,
            )
        )
        global_targets["turnover_budget"] = float(
            np.clip(
                max(
                    global_targets["turnover_budget"] + turnover_ramp_bonus,
                    turnover_floor,
                ),
                0.08,
                1.00,
            )
        )
        gross_relief_cap = float(
            np.clip(
                0.74
                + min(current_gross_exposure, 0.22) * 0.35
                + max(cash_pressure_scalar - 0.18, 0.0) * 0.14
                + max(global_targets["hold_bias_target"] - 0.55, 0.0) * 0.05,
                0.74,
                0.88 if is_holdcash_v3_decoder else 0.84,
            )
        )
        global_targets["gross_exposure_target"] = float(
            np.clip(
                min(global_targets["gross_exposure_target"], gross_relief_cap),
                min_gross_exposure_target,
                0.98,
            )
        )
    bucket_duration_days = np.asarray([HOLDING_DAYS_BY_BUCKET.get(str(label), 0.0) for label in predicted_duration_labels], dtype=float)
    supports_holding_days_head = bool(artifact.training_diagnostics.get("supports_holding_days_head", True))
    regressed_duration_days = np.clip(holding_days_ratio * float(MAX_CONTINUOUS_HOLDING_DAYS), 0.0, float(MAX_CONTINUOUS_HOLDING_DAYS))
    duration_days = (
        np.clip(bucket_duration_days * 0.58 + regressed_duration_days * 0.42, 0.0, float(MAX_CONTINUOUS_HOLDING_DAYS))
        if supports_holding_days_head
        else bucket_duration_days
    )
    adjusted_labels = predicted_labels.astype(object).copy()
    exit_timing_pressure_values = np.zeros(len(adjusted_labels), dtype=float)
    reduce_bias_target = float(global_targets["reduce_bias_target"])
    exit_patience_target = float(global_targets["exit_patience_target"])
    reentry_guard_target = float(global_targets["reentry_guard_target"])
    hierarchical_stock_gate_mode = budget_objective_name in {"result_value_v7", "result_value_v8", "result_value_v9"}
    pure_portfolio_defense_mode = budget_objective_name in {"result_value_v8", "result_value_v9"}
    portfolio_defense_signal = float(
        np.clip(
            max(
                float(global_targets.get("budget_model_defense_gate_signal", 0.0) or 0.0),
                float(global_targets.get("budget_cash_timing_signal_target", 0.0) or 0.0),
                float(global_targets.get("budget_risk_signal_target", 0.0) or 0.0) * 0.82 + portfolio_cash_defense * 0.18,
            ),
            0.0,
            1.0,
        )
    )
    if hierarchical_stock_gate_mode:
        deploy_release_denominator = deploy_value_target + release_value_target + 1.0e-6
        decision_deploy_gate = np.clip(
            0.62 * deploy_gate_target + 0.38 * np.clip(deploy_value_target / deploy_release_denominator, 0.0, 1.0),
            0.0,
            1.0,
        )
        decision_release_gate = np.clip(
            0.62 * release_gate_target + 0.38 * np.clip(release_value_target / deploy_release_denominator, 0.0, 1.0),
            0.0,
            1.0,
        )
        decision_defense_signal = np.full(len(adjusted_labels), portfolio_defense_signal, dtype=float)
    else:
        decision_deploy_gate = deploy_gate_target
        decision_release_gate = release_gate_target
        decision_defense_signal = defense_gate_target
    deploy_executability_target = _finite_array(
        np.clip(
            0.28 * deploy_value_target
            + 0.24 * decision_deploy_gate
            + 0.16 * deployment_opportunity_cost
            + 0.14 * alpha_opportunity_value
            + 0.08 * large_upside_1d_target
            + 0.06 * np.clip(probability_map["open"] + probability_map["add"], 0.0, 1.0)
            + 0.04 * np.clip(current_weight > 1.0e-8, 0.0, 1.0)
            - 0.14 * decision_release_gate
            - 0.12 * decision_defense_signal
            - 0.06 * clipped_intent_risk,
            0.0,
            1.0,
        ),
        default=0.0,
        low=0.0,
        high=1.0,
    )
    if predicted_deploy_executability is not None:
        deploy_executability_target = _finite_array(
            0.68 * predicted_deploy_executability + 0.32 * deploy_executability_target,
            default=0.0,
            low=0.0,
            high=1.0,
        )
    position_cap_for_headroom = float(np.clip(float(global_targets.get("max_position_weight_target", 0.12) or 0.12), 0.05, 0.34))
    portfolio_daily_receiver_add_headroom = _finite_array(
        np.clip(position_cap_for_headroom - current_weight, 0.0, 1.0),
        default=0.0,
        low=0.0,
        high=1.0,
    )
    portfolio_daily_receiver_min_add_delta = np.maximum.reduce(
        [
            np.full(len(current_weight), 0.0025, dtype=float),
            np.clip(current_weight, 0.0, None) * 0.025,
            np.full(len(current_weight), position_cap_for_headroom * 0.018, dtype=float),
        ]
    )
    fallback_receiver_capacity = np.where(
        current_weight > 1.0e-8,
        np.clip(
            portfolio_daily_receiver_add_headroom / np.clip(portfolio_daily_receiver_min_add_delta, 1.0e-6, None),
            0.0,
            1.0,
        ),
        1.0,
    )
    portfolio_daily_receiver_add_capacity = (
        _finite_array(
            0.70 * predicted_portfolio_receiver_capacity + 0.30 * fallback_receiver_capacity,
            default=0.0,
            low=0.0,
            high=1.0,
        )
        if predicted_portfolio_receiver_capacity is not None
        else fallback_receiver_capacity
    )
    if predicted_portfolio_receiver_headroom is not None:
        portfolio_daily_receiver_add_headroom = _finite_array(
            0.45 * predicted_portfolio_receiver_headroom + 0.55 * portfolio_daily_receiver_add_headroom,
            default=0.0,
            low=0.0,
            high=1.0,
        )
    fallback_receiver_executability = np.clip(
        deploy_executability_target * (0.50 + 0.50 * portfolio_daily_receiver_add_capacity)
        - np.where(current_weight > 1.0e-8, (1.0 - portfolio_daily_receiver_add_capacity) * 0.22, 0.0),
        0.0,
        1.0,
    )
    portfolio_daily_receiver_executability = (
        _finite_array(
            0.68 * predicted_portfolio_receiver_executability + 0.32 * fallback_receiver_executability,
            default=0.0,
            low=0.0,
            high=1.0,
        )
        if predicted_portfolio_receiver_executability is not None
        else fallback_receiver_executability
    )
    receiver_action_value = np.where(current_weight > 1.0e-8, add_action_value, open_action_value)
    fallback_portfolio_receiver_score = np.clip(
        0.30 * portfolio_daily_receiver_executability
        + 0.20 * deploy_value_target
        + 0.14 * decision_deploy_gate
        + 0.14 * receiver_action_value
        + 0.10 * alpha_opportunity_value
        + 0.08 * relative_opportunity_value
        + 0.06 * multi_horizon_path_value
        - 0.14 * decision_defense_signal
        - 0.12 * release_value_target
        - np.where(current_weight > 1.0e-8, (1.0 - portfolio_daily_receiver_add_capacity) * 0.18, 0.0),
        0.0,
        1.0,
    )
    portfolio_daily_receiver_score = (
        _finite_array(
            0.70 * predicted_portfolio_receiver_score + 0.30 * fallback_portfolio_receiver_score,
            default=0.0,
            low=0.0,
            high=1.0,
        )
        if predicted_portfolio_receiver_score is not None
        else fallback_portfolio_receiver_score
    )
    portfolio_daily_source_min_release_delta = np.maximum.reduce(
        [
            np.full(len(current_weight), 0.0025, dtype=float),
            np.clip(current_weight, 0.0, None) * 0.018,
            np.full(len(current_weight), position_cap_for_headroom * 0.012, dtype=float),
        ]
    )
    fallback_source_release_capacity = np.where(
        current_weight > 1.0e-8,
        np.clip(
            current_weight / np.clip(portfolio_daily_source_min_release_delta, 1.0e-6, None),
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_release_capacity = (
        _finite_array(
            0.70 * predicted_portfolio_source_release_capacity + 0.30 * fallback_source_release_capacity,
            default=0.0,
            low=0.0,
            high=1.0,
        )
        if predicted_portfolio_source_release_capacity is not None
        else fallback_source_release_capacity
    )
    fallback_source_release_quality = np.where(
        current_weight > 1.0e-8,
        np.clip(
            0.30 * sell_release_value
            + 0.18 * release_value_target
            + 0.16 * multi_horizon_forward_risk
            + 0.12 * cash_defense_value
            + 0.10 * decision_release_gate
            + 0.08 * sell_rank_score
            + 0.08 * (1.0 - hold_continuation_value)
            - 0.16 * alpha_opportunity_value
            - 0.12 * large_upside_1d_target,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_release_quality = (
        _finite_array(
            0.68 * predicted_portfolio_source_release_quality + 0.32 * fallback_source_release_quality,
            default=0.0,
            low=0.0,
            high=1.0,
        )
        if predicted_portfolio_source_release_quality is not None
        else fallback_source_release_quality
    )
    fallback_source_forward_proxy_keep_risk = np.where(
        current_weight > 1.0e-8,
        np.clip(
            0.34 * multi_horizon_forward_value
            + 0.30 * multi_horizon_path_value
            + 0.18 * alpha_opportunity_value
            + 0.12 * deploy_value_target
            + 0.06 * release_value_target,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_forward_proxy_keep_risk = (
        _finite_array(
            0.62 * predicted_portfolio_source_forward_proxy_keep_risk
            + 0.38 * fallback_source_forward_proxy_keep_risk,
            default=0.0,
            low=0.0,
            high=1.0,
        )
        if predicted_portfolio_source_forward_proxy_keep_risk is not None
        else fallback_source_forward_proxy_keep_risk
    )
    portfolio_daily_source_release_quality = _finite_array(
        portfolio_daily_source_release_quality
        - portfolio_daily_source_forward_proxy_keep_risk * 0.08,
        default=0.0,
        low=0.0,
        high=1.0,
    )
    fallback_source_opportunity_cost = np.where(
        current_weight > 1.0e-8,
        np.clip(
            0.34 * hold_continuation_value
            + 0.26 * alpha_opportunity_value
            + 0.20 * multi_horizon_path_value
            + 0.14 * deploy_value_target
            + 0.10 * portfolio_daily_receiver_executability
            + 0.10 * large_upside_1d_target
            + 0.26 * portfolio_daily_source_forward_proxy_keep_risk
            - 0.10 * release_value_target
            - 0.06 * cash_defense_value
            - 0.05 * multi_horizon_forward_risk
            - 0.12 * portfolio_daily_source_release_quality,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_opportunity_cost = (
        _finite_array(
            0.64 * predicted_portfolio_source_opportunity_cost + 0.36 * fallback_source_opportunity_cost,
            default=0.0,
            low=0.0,
            high=1.0,
        )
        if predicted_portfolio_source_opportunity_cost is not None
        else fallback_source_opportunity_cost
    )
    fallback_source_executability = np.where(
        current_weight > 1.0e-8,
        np.clip(
            0.28 * release_value_target
            + 0.20 * sell_release_value
            + 0.14 * decision_release_gate
            + 0.12 * cash_defense_value
            + 0.10 * multi_horizon_forward_risk
            + 0.08 * sell_rank_score
            + 0.06 * portfolio_daily_source_release_capacity
            + 0.14 * (1.0 - portfolio_daily_source_opportunity_cost)
            + 0.18 * portfolio_daily_source_release_quality
            - 0.18 * hold_continuation_value
            - 0.14 * alpha_opportunity_value
            - 0.10 * portfolio_daily_receiver_executability
            - 0.06 * large_upside_1d_target,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_executability = (
        _finite_array(
            0.70 * predicted_portfolio_source_executability + 0.30 * fallback_source_executability,
            default=0.0,
            low=0.0,
            high=1.0,
        )
        if predicted_portfolio_source_executability is not None
        else fallback_source_executability
    )
    fallback_portfolio_source_score = np.where(
        current_weight > 1.0e-8,
        np.clip(
            0.28 * portfolio_daily_source_executability
            + 0.10 * portfolio_daily_source_release_capacity
            + 0.22 * (1.0 - portfolio_daily_source_opportunity_cost)
            + 0.22 * portfolio_daily_source_release_quality
            + 0.13 * release_value_target
            + 0.11 * sell_release_value
            + 0.09 * decision_release_gate
            + 0.07 * cash_defense_value
            + 0.06 * multi_horizon_forward_risk
            + 0.05 * relative_opportunity_value
            - 0.15 * hold_continuation_value
            - 0.12 * alpha_opportunity_value
            - 0.10 * portfolio_daily_receiver_executability
            - 0.08 * large_upside_1d_target
            - 0.12 * portfolio_daily_source_forward_proxy_keep_risk,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_score = (
        _finite_array(
            0.62 * predicted_portfolio_source_score
            + 0.38 * fallback_portfolio_source_score
            - 0.08 * portfolio_daily_source_opportunity_cost,
            default=0.0,
            low=0.0,
            high=1.0,
        )
        if predicted_portfolio_source_score is not None
        else fallback_portfolio_source_score
    )
    fallback_source_forward_spread_score = np.where(
        current_weight > 1.0e-8,
        np.clip(
            0.30 * portfolio_daily_source_score
            + 0.22 * portfolio_daily_source_release_quality
            + 0.16 * (1.0 - portfolio_daily_source_opportunity_cost)
            + 0.12 * multi_horizon_forward_risk
            + 0.10 * cash_defense_value
            + 0.06 * sell_rank_score
            - 0.20 * hold_continuation_value
            - 0.16 * alpha_opportunity_value
            - 0.10 * large_upside_1d_target,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_forward_spread_score = (
        _finite_array(
            0.70 * predicted_portfolio_source_forward_spread_score + 0.30 * fallback_source_forward_spread_score,
            default=0.0,
            low=0.0,
            high=1.0,
        )
        if predicted_portfolio_source_forward_spread_score is not None
        else fallback_source_forward_spread_score
    )
    fallback_source_bad_forward_spread_risk = np.where(
        current_weight > 1.0e-8,
        np.clip(
            0.28 * hold_continuation_value
            + 0.22 * alpha_opportunity_value
            + 0.18 * large_upside_1d_target
            + 0.16 * portfolio_daily_source_opportunity_cost
            + 0.10 * multi_horizon_path_value
            - 0.22 * portfolio_daily_source_release_quality
            - 0.18 * portfolio_daily_source_forward_spread_score
            - 0.10 * sell_release_value,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_bad_forward_spread_risk = (
        _finite_array(
            0.70 * predicted_portfolio_source_bad_forward_spread_risk + 0.30 * fallback_source_bad_forward_spread_risk,
            default=0.0,
            low=0.0,
            high=1.0,
        )
        if predicted_portfolio_source_bad_forward_spread_risk is not None
        else fallback_source_bad_forward_spread_risk
    )
    fallback_source_economic_release_score = np.where(
        current_weight > 1.0e-8,
        np.clip(
            0.32 * portfolio_daily_source_forward_spread_score
            + 0.20 * portfolio_daily_source_release_quality
            + 0.16 * portfolio_daily_source_score
            + 0.12 * portfolio_daily_source_executability
            + 0.10 * (1.0 - portfolio_daily_source_opportunity_cost)
            + 0.06 * sell_release_value
            + 0.04 * cash_defense_value
            - 0.30 * portfolio_daily_source_bad_forward_spread_risk
            - 0.16 * hold_continuation_value
            - 0.10 * alpha_opportunity_value,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_economic_release_score = (
        _finite_array(
            0.72 * predicted_portfolio_source_economic_release_score + 0.28 * fallback_source_economic_release_score,
            default=0.0,
            low=0.0,
            high=1.0,
        )
        if predicted_portfolio_source_economic_release_score is not None
        else fallback_source_economic_release_score
    )
    fallback_source_economic_block_risk = np.where(
        current_weight > 1.0e-8,
        np.clip(
            0.34 * portfolio_daily_source_bad_forward_spread_risk
            + 0.20 * portfolio_daily_source_opportunity_cost
            + 0.18 * hold_continuation_value
            + 0.14 * alpha_opportunity_value
            + 0.10 * large_upside_1d_target
            - 0.22 * portfolio_daily_source_economic_release_score
            - 0.14 * portfolio_daily_source_forward_spread_score,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_economic_block_risk = (
        _finite_array(
            0.72 * predicted_portfolio_source_economic_block_risk + 0.28 * fallback_source_economic_block_risk,
            default=0.0,
            low=0.0,
            high=1.0,
        )
        if predicted_portfolio_source_economic_block_risk is not None
        else fallback_source_economic_block_risk
    )
    fallback_source_forward_strength_brake_risk = np.where(
        current_weight > 1.0e-8,
        np.clip(
            0.24 * portfolio_daily_source_bad_forward_spread_risk
            + 0.22 * hold_continuation_value
            + 0.20 * alpha_opportunity_value
            + 0.16 * large_upside_1d_target
            + 0.14 * multi_horizon_path_value
            + 0.10 * deploy_value_target
            + 0.08 * portfolio_daily_source_opportunity_cost
            - 0.18 * multi_horizon_forward_risk
            - 0.16 * portfolio_daily_source_forward_spread_score
            - 0.12 * portfolio_daily_source_release_quality
            - 0.08 * sell_release_value
            - 0.06 * cash_defense_value,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_forward_strength_brake_risk = (
        _finite_array(
            0.74 * predicted_portfolio_source_forward_strength_brake_risk
            + 0.26 * fallback_source_forward_strength_brake_risk,
            default=0.0,
            low=0.0,
            high=1.0,
        )
        if predicted_portfolio_source_forward_strength_brake_risk is not None
        else fallback_source_forward_strength_brake_risk
    )
    portfolio_daily_source_economic_block_risk = _finite_array(
        portfolio_daily_source_economic_block_risk
        + portfolio_daily_source_forward_strength_brake_risk * 0.16
        - portfolio_daily_source_economic_release_score * 0.03,
        default=0.0,
        low=0.0,
        high=1.0,
    )
    portfolio_daily_source_economic_release_score = _finite_array(
        portfolio_daily_source_economic_release_score
        - portfolio_daily_source_forward_strength_brake_risk * 0.12,
        default=0.0,
        low=0.0,
        high=1.0,
    )
    portfolio_daily_source_opportunity_cost = _finite_array(
        portfolio_daily_source_opportunity_cost
        + portfolio_daily_source_economic_block_risk * 0.10
        + portfolio_daily_source_forward_strength_brake_risk * 0.16
        - portfolio_daily_source_economic_release_score * 0.05,
        default=0.0,
        low=0.0,
        high=1.0,
    )
    portfolio_daily_source_score = _finite_array(
        portfolio_daily_source_score
        + portfolio_daily_source_economic_release_score * 0.12
        - portfolio_daily_source_economic_block_risk * 0.14
        - portfolio_daily_source_forward_strength_brake_risk * 0.20,
        default=0.0,
        low=0.0,
        high=1.0,
    )
    fallback_portfolio_cash_score = np.clip(
        0.30 * defense_value_target
        + 0.22 * defense_gate_target
        + 0.16 * cash_defense_value
        + 0.12 * decision_defense_signal
        + 0.08 * multi_horizon_forward_risk
        + 0.06 * np.clip(risk_off_score, 0.0, 1.0)
        - 0.18 * portfolio_daily_receiver_score,
        0.0,
        1.0,
    )
    portfolio_daily_cash_score = (
        _finite_array(
            0.68 * predicted_portfolio_cash_score + 0.32 * fallback_portfolio_cash_score,
            default=0.0,
            low=0.0,
            high=1.0,
        )
        if predicted_portfolio_cash_score is not None
        else fallback_portfolio_cash_score
    )
    zero_unified_allocation_head = np.zeros(len(state_frame), dtype=float)
    portfolio_daily_unified_receiver_score = (
        _finite_array(predicted_portfolio_unified_receiver_score, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_unified_receiver_score is not None
        else zero_unified_allocation_head.copy()
    )
    portfolio_daily_unified_source_score = (
        _finite_array(predicted_portfolio_unified_source_score, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_unified_source_score is not None
        else zero_unified_allocation_head.copy()
    )
    portfolio_daily_unified_cash_score = (
        _finite_array(predicted_portfolio_unified_cash_score, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_unified_cash_score is not None
        else zero_unified_allocation_head.copy()
    )
    portfolio_daily_source_positive_forward_penalty = (
        _finite_array(predicted_portfolio_source_positive_forward_penalty, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_source_positive_forward_penalty is not None
        else zero_unified_allocation_head.copy()
    )
    portfolio_daily_source_opportunity_cost_penalty = (
        _finite_array(predicted_portfolio_source_opportunity_cost_penalty, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_source_opportunity_cost_penalty is not None
        else zero_unified_allocation_head.copy()
    )
    portfolio_daily_source_strong_false_sell_penalty = (
        _finite_array(predicted_portfolio_source_strong_false_sell_penalty, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_source_strong_false_sell_penalty is not None
        else zero_unified_allocation_head.copy()
    )
    portfolio_daily_source_hard_negative_penalty = (
        _finite_array(predicted_portfolio_source_hard_negative_penalty, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_source_hard_negative_penalty is not None
        else zero_unified_allocation_head.copy()
    )
    portfolio_daily_source_tail_false_sell_penalty = (
        _finite_array(predicted_portfolio_source_tail_false_sell_penalty, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_source_tail_false_sell_penalty is not None
        else zero_unified_allocation_head.copy()
    )
    portfolio_daily_source_release_preference = (
        _finite_array(predicted_portfolio_source_release_preference, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_source_release_preference is not None
        else zero_unified_allocation_head.copy()
    )
    portfolio_daily_receiver_source_spread_reward = (
        _finite_array(predicted_portfolio_receiver_source_spread_reward, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_receiver_source_spread_reward is not None
        else zero_unified_allocation_head.copy()
    )
    portfolio_daily_transfer_regret_target = (
        _finite_array(predicted_portfolio_transfer_regret_target, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_transfer_regret_target is not None
        else zero_unified_allocation_head.copy()
    )
    portfolio_daily_unified_allocation_objective = (
        _finite_array(predicted_portfolio_unified_allocation_objective, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_unified_allocation_objective is not None
        else zero_unified_allocation_head.copy()
    )
    portfolio_daily_allocation_trade_quality_target = (
        _finite_array(predicted_portfolio_allocation_trade_quality_target, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_allocation_trade_quality_target is not None
        else portfolio_daily_unified_allocation_objective.copy()
    )
    portfolio_daily_allocation_cash_deployment_target = (
        _finite_array(predicted_portfolio_allocation_cash_deployment_target, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_allocation_cash_deployment_target is not None
        else zero_unified_allocation_head.copy()
    )
    portfolio_daily_allocation_risk_adjusted_return_target = (
        _finite_array(predicted_portfolio_allocation_risk_adjusted_return_target, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_allocation_risk_adjusted_return_target is not None
        else portfolio_daily_unified_allocation_objective.copy()
    )
    portfolio_daily_allocation_drawdown_control_target = (
        _finite_array(predicted_portfolio_allocation_drawdown_control_target, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_allocation_drawdown_control_target is not None
        else portfolio_daily_unified_cash_score.copy()
    )
    portfolio_daily_allocation_monthly_quality_target = (
        _finite_array(predicted_portfolio_allocation_monthly_quality_target, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_allocation_monthly_quality_target is not None
        else portfolio_daily_unified_allocation_objective.copy()
    )
    portfolio_daily_allocation_final_objective = (
        _finite_array(predicted_portfolio_allocation_final_objective, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_allocation_final_objective is not None
        else portfolio_daily_unified_allocation_objective.copy()
    )
    portfolio_daily_allocation_uncertainty_pressure_target = (
        _finite_array(predicted_portfolio_allocation_uncertainty_pressure_target, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_allocation_uncertainty_pressure_target is not None
        else np.clip(0.62 * multi_horizon_forward_risk + 0.38 * portfolio_daily_allocation_drawdown_control_target, 0.0, 1.0)
    )
    portfolio_daily_allocation_tail_risk_control_target = (
        _finite_array(predicted_portfolio_allocation_tail_risk_control_target, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_allocation_tail_risk_control_target is not None
        else np.clip(
            0.58 * portfolio_daily_allocation_drawdown_control_target
            + 0.28 * multi_horizon_forward_risk
            + 0.14 * portfolio_daily_source_hard_negative_penalty,
            0.0,
            1.0,
        )
    )
    portfolio_daily_allocation_decision_focused_objective = (
        _finite_array(predicted_portfolio_allocation_decision_focused_objective, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_allocation_decision_focused_objective is not None
        else np.clip(
            0.62 * portfolio_daily_allocation_final_objective
            + 0.18 * portfolio_daily_allocation_cash_deployment_target
            + 0.12 * portfolio_daily_allocation_risk_adjusted_return_target
            - 0.16 * portfolio_daily_allocation_uncertainty_pressure_target
            - 0.14 * portfolio_daily_allocation_tail_risk_control_target,
            0.0,
            1.0,
        )
    )
    portfolio_daily_allocation_credit_closure_target = (
        _finite_array(predicted_portfolio_allocation_credit_closure_target, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_allocation_credit_closure_target is not None
        else np.clip(
            0.28 * portfolio_daily_allocation_cash_deployment_target
            + 0.22 * portfolio_daily_source_release_preference
            + 0.20 * portfolio_daily_receiver_source_spread_reward
            + 0.16 * portfolio_daily_transfer_regret_target
            + 0.14 * portfolio_daily_allocation_final_objective
            - 0.18 * portfolio_daily_allocation_uncertainty_pressure_target
            - 0.12 * portfolio_daily_source_opportunity_cost_penalty,
            0.0,
            1.0,
        )
    )
    portfolio_daily_allocation_net_utility_target = (
        _finite_array(predicted_portfolio_allocation_net_utility_target, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_allocation_net_utility_target is not None
        else np.clip(
            0.24 * portfolio_daily_allocation_final_objective
            + 0.22 * portfolio_daily_allocation_decision_focused_objective
            + 0.18 * portfolio_daily_allocation_credit_closure_target
            + 0.16 * portfolio_daily_allocation_risk_adjusted_return_target
            + 0.10 * portfolio_daily_allocation_monthly_quality_target
            + 0.10 * portfolio_daily_receiver_source_spread_reward
            - 0.16 * portfolio_daily_allocation_uncertainty_pressure_target
            - 0.14 * portfolio_daily_allocation_tail_risk_control_target,
            0.0,
            1.0,
        )
    )
    portfolio_daily_allocation_resource_efficiency_target = (
        _finite_array(predicted_portfolio_allocation_resource_efficiency_target, default=0.0, low=0.0, high=1.0)
        if predicted_portfolio_allocation_resource_efficiency_target is not None
        else np.clip(
            0.34 * portfolio_daily_allocation_net_utility_target
            + 0.28 * portfolio_daily_allocation_credit_closure_target
            + 0.18 * portfolio_daily_allocation_decision_focused_objective
            + 0.12 * portfolio_daily_allocation_cash_deployment_target
            - 0.14 * portfolio_daily_allocation_tail_risk_control_target
            - 0.10 * portfolio_daily_allocation_uncertainty_pressure_target,
            0.0,
            1.0,
        )
    )
    if predicted_portfolio_allocation_final_objective is not None:
        portfolio_daily_unified_receiver_score = np.clip(
            0.58 * portfolio_daily_unified_receiver_score
            + 0.16 * portfolio_daily_allocation_cash_deployment_target
            + 0.12 * portfolio_daily_allocation_risk_adjusted_return_target
            + 0.08 * portfolio_daily_allocation_final_objective
            + 0.06 * portfolio_daily_allocation_monthly_quality_target,
            0.0,
            1.0,
        )
        portfolio_daily_unified_source_score = np.clip(
            0.58 * portfolio_daily_unified_source_score
            + 0.16 * portfolio_daily_allocation_trade_quality_target
            + 0.12 * portfolio_daily_allocation_risk_adjusted_return_target
            + 0.10 * portfolio_daily_allocation_final_objective
            - 0.24 * portfolio_daily_source_hard_negative_penalty
            - 0.16 * portfolio_daily_source_positive_forward_penalty
            - 0.12 * portfolio_daily_source_opportunity_cost_penalty,
            0.0,
            1.0,
        )
        portfolio_daily_unified_cash_score = np.clip(
            0.62 * portfolio_daily_unified_cash_score
            + 0.20 * portfolio_daily_allocation_drawdown_control_target
            - 0.28 * portfolio_daily_allocation_cash_deployment_target
            - 0.12 * portfolio_daily_allocation_risk_adjusted_return_target
            + 0.08 * portfolio_daily_source_hard_negative_penalty,
            0.0,
            1.0,
        )
        portfolio_daily_receiver_score = np.clip(
            0.68 * portfolio_daily_receiver_score
            + 0.32 * portfolio_daily_unified_receiver_score,
            0.0,
            1.0,
        )
        portfolio_daily_source_score = np.clip(
            0.68 * portfolio_daily_source_score
            + 0.32 * portfolio_daily_unified_source_score,
            0.0,
            1.0,
        )
        portfolio_daily_cash_score = np.clip(
            0.68 * portfolio_daily_cash_score
            + 0.32 * portfolio_daily_unified_cash_score,
            0.0,
            1.0,
        )
    if predicted_portfolio_allocation_decision_focused_objective is not None:
        portfolio_daily_unified_receiver_score = np.clip(
            0.62 * portfolio_daily_unified_receiver_score
            + 0.14 * portfolio_daily_allocation_decision_focused_objective
            + 0.12 * portfolio_daily_allocation_net_utility_target
            + 0.10 * portfolio_daily_allocation_credit_closure_target
            + 0.06 * portfolio_daily_allocation_resource_efficiency_target
            - 0.13 * portfolio_daily_allocation_uncertainty_pressure_target
            - 0.10 * portfolio_daily_allocation_tail_risk_control_target,
            0.0,
            1.0,
        )
        portfolio_daily_unified_source_score = np.clip(
            0.64 * portfolio_daily_unified_source_score
            + 0.12 * portfolio_daily_allocation_decision_focused_objective
            + 0.12 * portfolio_daily_allocation_credit_closure_target
            + 0.08 * portfolio_daily_allocation_net_utility_target
            + 0.06 * portfolio_daily_allocation_resource_efficiency_target
            - 0.08 * portfolio_daily_allocation_uncertainty_pressure_target
            - 0.12 * portfolio_daily_source_hard_negative_penalty,
            0.0,
            1.0,
        )
        portfolio_daily_unified_cash_score = np.clip(
            0.70 * portfolio_daily_unified_cash_score
            + 0.18 * portfolio_daily_allocation_tail_risk_control_target
            + 0.14 * portfolio_daily_allocation_uncertainty_pressure_target
            - 0.08 * portfolio_daily_allocation_decision_focused_objective
            - 0.08 * portfolio_daily_allocation_net_utility_target
            - 0.06 * portfolio_daily_allocation_credit_closure_target,
            0.0,
            1.0,
        )
    source_funding_strength = np.clip(
        portfolio_daily_source_score
        * portfolio_daily_source_release_capacity
        * (1.0 - portfolio_daily_source_opportunity_cost)
        * (0.55 + 0.45 * portfolio_daily_source_release_quality),
        0.0,
        1.0,
    )
    receiver_demand_strength = np.clip(
        portfolio_daily_receiver_score
        * np.where(current_weight > 1.0e-8, portfolio_daily_receiver_add_capacity, 1.0),
        0.0,
        1.0,
    )
    source_funding_reference = 0.0
    finite_source_funding = source_funding_strength[np.isfinite(source_funding_strength)]
    if finite_source_funding.size:
        top_k = max(1, min(5, int(np.ceil(finite_source_funding.size * 0.25))))
        source_funding_reference = float(np.median(np.sort(finite_source_funding)[-top_k:]))
    receiver_demand_reference = 0.0
    finite_receiver_demand = receiver_demand_strength[np.isfinite(receiver_demand_strength)]
    if finite_receiver_demand.size:
        top_k = max(1, min(5, int(np.ceil(finite_receiver_demand.size * 0.25))))
        receiver_demand_reference = float(np.median(np.sort(finite_receiver_demand)[-top_k:]))
    cash_funding_reference = float(np.clip(np.nanmean(portfolio_daily_cash_score), 0.0, 1.0)) if len(portfolio_daily_cash_score) else 0.0
    funding_reference = float(
        np.clip(
            0.62 * source_funding_reference
            + 0.22 * receiver_demand_reference
            + 0.16 * cash_funding_reference,
            0.0,
            1.0,
        )
    )
    fallback_receiver_funding_coverage = np.clip(
        np.where(current_weight > 1.0e-8, portfolio_daily_receiver_add_capacity, 1.0)
        * (0.64 * funding_reference + 0.24 * source_funding_reference + 0.12 * cash_funding_reference),
        0.0,
        1.0,
    )
    fallback_allocation_transfer_score = np.clip(
        np.maximum(
            receiver_demand_strength * (0.44 + 0.56 * fallback_receiver_funding_coverage),
            source_funding_strength * (0.48 + 0.52 * receiver_demand_reference),
        )
        + np.minimum(receiver_demand_strength, source_funding_strength) * 0.14
        - portfolio_daily_cash_score * 0.06,
        0.0,
        1.0,
    )
    fallback_funding_closure_score = np.clip(
        0.36 * fallback_allocation_transfer_score
        + 0.28 * fallback_receiver_funding_coverage
        + 0.22 * source_funding_strength
        + 0.14 * funding_reference,
        0.0,
        1.0,
    )
    fallback_allocation_dead_branch_risk = np.clip(
        0.30 * np.clip(1.0 - receiver_demand_reference / 0.32, 0.0, 1.0)
        + 0.30 * np.clip(1.0 - source_funding_reference / 0.28, 0.0, 1.0)
        + 0.18 * np.clip((cash_funding_reference - 0.34) / 0.46, 0.0, 1.0)
        + 0.22 * (1.0 - fallback_allocation_transfer_score),
        0.0,
        1.0,
    )
    portfolio_daily_receiver_funding_coverage = (
        _finite_array(
            0.70 * predicted_portfolio_receiver_funding_coverage
            + 0.30 * fallback_receiver_funding_coverage,
            default=0.0,
            low=0.0,
            high=1.0,
        )
        if predicted_portfolio_receiver_funding_coverage is not None
        else fallback_receiver_funding_coverage
    )
    portfolio_daily_funding_closure_score = (
        _finite_array(
            0.70 * predicted_portfolio_funding_closure_score
            + 0.30 * fallback_funding_closure_score,
            default=0.0,
            low=0.0,
            high=1.0,
        )
        if predicted_portfolio_funding_closure_score is not None
        else fallback_funding_closure_score
    )
    portfolio_daily_allocation_transfer_score = (
        _finite_array(
            0.70 * predicted_portfolio_allocation_transfer_score
            + 0.30 * fallback_allocation_transfer_score,
            default=0.0,
            low=0.0,
            high=1.0,
        )
        if predicted_portfolio_allocation_transfer_score is not None
        else fallback_allocation_transfer_score
    )
    portfolio_daily_allocation_dead_branch_risk = (
        _finite_array(
            0.70 * predicted_portfolio_allocation_dead_branch_risk
            + 0.30 * fallback_allocation_dead_branch_risk,
            default=0.0,
            low=0.0,
            high=1.0,
        )
        if predicted_portfolio_allocation_dead_branch_risk is not None
        else fallback_allocation_dead_branch_risk
    )
    direct_action_utility = {
        label: np.zeros(len(adjusted_labels), dtype=float)
        for label in ACTION_CLASSES
    }
    direct_action_value_label = predicted_labels.astype(object).copy()
    direct_action_value_selected = np.zeros(len(adjusted_labels), dtype=float)
    direct_action_value_gap = np.zeros(len(adjusted_labels), dtype=float)
    direct_action_value_applied = np.zeros(len(adjusted_labels), dtype=float)
    if direct_action_value_mode:
        duration_bonus = np.clip(duration_days - 3.0, 0.0, None) / 20.0
        preliminary_exit_timing_pressure = np.clip(
            exit_hazard * 0.46
            + sell_pressure * 0.18
            + np.clip(-drawdown_from_peak - 0.04, 0.0, None) * 1.05
            + np.clip(-unrealized_pnl - 0.02, 0.0, None) * 0.38
            + signal_decay_speed * 0.34
            + market_downside_pressure * 0.18
            + np.clip(-price_from_local_peak - 0.03, 0.0, None) * 0.26
            + np.clip(drawdown_rank_in_portfolio - 0.62, 0.0, None) * 0.20
            - np.clip(score_rank_pct - 0.72, 0.0, None) * 0.12,
            0.0,
            1.35,
        )
        direct_action_utility["skip"] = np.clip(
            probability_map["skip"] * 0.24
            + decision_defense_signal * 0.34
            + cash_defense_value * 0.20
            + portfolio_daily_cash_score * 0.12
            + reentry_guard_target * 0.10
            - open_action_value * 0.26
            - entry_quality.clip(min=0.0) * 0.12
            - decision_deploy_gate * 0.10
            - portfolio_daily_receiver_score * 0.08,
            -1.0,
            1.8,
        )
        direct_action_utility["open"] = np.clip(
            open_action_value * 0.40
            + probability_map["open"] * 0.20
            + entry_quality.clip(min=0.0) * 0.18
            + alpha_opportunity_value * 0.12
            + deployment_opportunity_cost * 0.10
            + deploy_value_target * 0.12
            + decision_deploy_gate * 0.12
            + deploy_executability_target * 0.10
            + portfolio_daily_receiver_executability * 0.10
            + portfolio_daily_receiver_score * 0.14
            + multi_horizon_path_value * 0.10
            + duration_bonus * 0.08
            - decision_defense_signal * 0.18
            - multi_horizon_forward_risk * 0.12
            - portfolio_daily_cash_score * 0.08
            - clipped_intent_risk * 0.10
            - reentry_cooldown * 0.08
            - open_risk_off_score * 0.08,
            -1.0,
            1.8,
        )
        direct_action_utility["hold"] = np.clip(
            hold_action_value * 0.38
            + probability_map["hold"] * 0.18
            + hold_continuation_value * 0.18
            + hold_quality.clip(min=0.0) * 0.16
            + alpha_opportunity_value * 0.08
            + deploy_value_target * 0.06
            + decision_deploy_gate * 0.08
            + duration_bonus * 0.08
            - release_value_target * 0.10
            - decision_release_gate * 0.10
            - preliminary_exit_timing_pressure * 0.12
            - sell_pressure * 0.10,
            -1.0,
            1.8,
        )
        direct_action_utility["add"] = np.clip(
            add_action_value * 0.34
            + probability_map["add"] * 0.18
            + add_quality.clip(min=0.0) * 0.18
            + hold_continuation_value * 0.10
            + alpha_opportunity_value * 0.10
            + deployment_opportunity_cost * 0.08
            + deploy_value_target * 0.12
            + decision_deploy_gate * 0.10
            + deploy_executability_target * 0.08
            + portfolio_daily_receiver_executability * 0.10
            + portfolio_daily_receiver_score * 0.12
            + portfolio_daily_receiver_add_capacity * 0.14
            + multi_horizon_path_value * 0.06
            - release_value_target * 0.12
            - decision_release_gate * 0.10
            - (1.0 - portfolio_daily_receiver_add_capacity) * 0.22
            - portfolio_daily_cash_score * 0.06
            - preliminary_exit_timing_pressure * 0.14
            - clipped_intent_risk * 0.10,
            -1.0,
            1.8,
        )
        direct_action_utility["reduce"] = np.clip(
            reduce_action_value * 0.34
            + probability_map["reduce"] * 0.18
            + reduce_quality.clip(min=0.0) * 0.14
            + reduce_fraction * 0.14
            + sell_release_value * 0.12
            + release_value_target * 0.12
            + decision_release_gate * 0.12
            + portfolio_daily_source_score * 0.12
            + portfolio_daily_source_executability * 0.10
            + portfolio_daily_source_release_capacity * 0.06
            + portfolio_daily_cash_score * 0.04
            + sell_pressure * 0.10
            + preliminary_exit_timing_pressure * 0.10
            + signal_decay_speed * 0.08
            - hold_action_value * 0.12
            - hold_continuation_value * 0.10
            - decision_deploy_gate * 0.08,
            -1.0,
            1.8,
        )
        direct_action_utility["exit"] = np.clip(
            exit_action_value * 0.36
            + probability_map["exit"] * 0.18
            + exit_urgency.clip(min=0.0) * 0.14
            + exit_hazard * 0.16
            + sell_release_value * 0.12
            + release_value_target * 0.12
            + decision_release_gate * 0.10
            + portfolio_daily_source_score * 0.14
            + portfolio_daily_source_executability * 0.10
            + portfolio_daily_source_release_capacity * 0.06
            + portfolio_daily_cash_score * 0.06
            + preliminary_exit_timing_pressure * 0.18
            + market_downside_pressure * 0.08
            - hold_action_value * 0.14
            - hold_continuation_value * 0.12
            - decision_deploy_gate * 0.08,
            -1.0,
            1.8,
        )
        utility_matrix = np.stack([direct_action_utility[label] for label in ACTION_CLASSES], axis=1)
        held_state_mask = current_weight > 1.0e-8
        flat_state_mask = ~held_state_mask
        action_lookup = {name: idx for idx, name in enumerate(ACTION_CLASSES)}
        utility_matrix[held_state_mask, action_lookup["skip"]] = -1.0e9
        utility_matrix[held_state_mask, action_lookup["open"]] = -1.0e9
        for invalid_action in ("hold", "add", "reduce", "exit"):
            utility_matrix[flat_state_mask, action_lookup[invalid_action]] = -1.0e9
        avoid_mask = np.asarray(predicted_duration_labels, dtype=object) == "avoid"
        utility_matrix[flat_state_mask & avoid_mask, action_lookup["open"]] = -1.0e9
        direct_indices = utility_matrix.argmax(axis=1)
        direct_action_value_label = np.asarray(ACTION_CLASSES, dtype=object)[direct_indices]
        sorted_direct = np.sort(utility_matrix, axis=1)
        direct_action_value_selected = np.clip(sorted_direct[:, -1], -1.0, 1.8)
        direct_action_value_gap = np.clip(sorted_direct[:, -1] - sorted_direct[:, -2], 0.0, 2.0)
        direct_action_value_applied = np.ones(len(adjusted_labels), dtype=float)
        adjusted_labels = direct_action_value_label.astype(object).copy()
        for label in ACTION_CLASSES:
            direct_action_utility[label] = np.where(
                np.isfinite(utility_matrix[:, action_lookup[label]]),
                utility_matrix[:, action_lookup[label]],
                -1.0,
            )
        global_targets["policy_decision_mode"] = DIRECT_ACTION_VALUE_POLICY_MODE
    else:
        global_targets["policy_decision_mode"] = "logit_rule_bridge_v1"
    for idx in range(len(adjusted_labels)):
        label = str(adjusted_labels[idx])
        held = float(current_weight[idx]) > 1e-8
        duration_name = str(predicted_duration_labels[idx])
        add_prob = float(probability_map["add"][idx])
        reduce_prob = float(probability_map["reduce"][idx])
        exit_prob = float(probability_map["exit"][idx])
        exit_timing_pressure = float(
            np.clip(
                exit_hazard[idx] * 0.46
                + sell_pressure[idx] * 0.18
                + max(-drawdown_from_peak[idx] - 0.04, 0.0) * 1.05
                + max(-unrealized_pnl[idx] - 0.02, 0.0) * 0.38
                + signal_decay_speed[idx] * 0.34
                + market_downside_pressure[idx] * 0.18
                + max(-price_from_local_peak[idx] - 0.03, 0.0) * 0.26
                + max(drawdown_rank_in_portfolio[idx] - 0.62, 0.0) * 0.20
                - max(score_rank_pct[idx] - 0.72, 0.0) * 0.12,
                0.0,
                1.35,
            )
        )
        exit_timing_pressure_values[idx] = exit_timing_pressure
        profit_protected_trend = bool(
            unrealized_pnl[idx] > 0.08
            and score_rank_pct[idx] > 0.65
            and drawdown_from_peak[idx] > -0.05
            and signal_decay_speed[idx] < 0.08
        )
        if direct_action_value_mode:
            action_keep_value = max(float(add_action_value[idx]), float(hold_action_value[idx]))
            action_release_value = max(float(reduce_action_value[idx]), float(exit_action_value[idx]))
            if held:
                if label in {"skip", "open"}:
                    label = "add" if add_action_value[idx] > hold_action_value[idx] + 0.10 else "hold"
                if (
                    label == "add"
                    and action_release_value > action_keep_value + 0.18
                    and exit_timing_pressure > 0.46
                ):
                    label = "exit" if exit_action_value[idx] >= reduce_action_value[idx] + 0.08 else "reduce"
                elif (
                    label in {"reduce", "exit"}
                    and action_keep_value > action_release_value + 0.16
                    and exit_timing_pressure < 0.54
                    and decision_release_gate[idx] <= decision_deploy_gate[idx] + 0.06
                    and market_downside_pressure[idx] < 0.24
                ):
                    label = "add" if add_action_value[idx] > hold_action_value[idx] + 0.12 else "hold"
                if (
                    label == "exit"
                    and hold_days[idx] < 2.0
                    and exit_timing_pressure < 0.60
                    and exit_action_value[idx] < reduce_action_value[idx] + 0.10
                ):
                    label = "reduce"
            else:
                if label not in {"open", "skip"}:
                    label = "open" if open_action_value[idx] >= 0.42 and decision_deploy_gate[idx] > decision_defense_signal[idx] else "skip"
                if (
                    label == "open"
                    and (
                        duration_name == "avoid"
                        or open_action_value[idx] < 0.28
                        or decision_defense_signal[idx] > 0.62
                        or clipped_intent_risk[idx] > 0.76
                    )
                ):
                    label = "skip"
            adjusted_labels[idx] = label
            continue
        if held:
            held_defense_weight = 0.0 if pure_portfolio_defense_mode else 1.0
            sell_arbitration = float(
                np.clip(
                    0.24 * decision_release_gate[idx]
                    + 0.22 * release_value_target[idx]
                    + 0.18 * lifecycle_sell_gate[idx]
                    + 0.16 * sell_rank_score[idx]
                    + 0.12 * preliminary_sell_attribution[idx]
                    + 0.08 * decision_defense_signal[idx] * held_defense_weight,
                    0.0,
                    1.0,
                )
            )
            keep_arbitration = float(
                np.clip(
                    0.24 * decision_deploy_gate[idx]
                    + 0.22 * deploy_value_target[idx]
                    + 0.18 * max(hold_quality[idx], 0.0)
                    + 0.16 * hold_continuation_value[idx]
                    + 0.12 * alpha_opportunity_value[idx]
                    + 0.06 * max(add_quality[idx], 0.0)
                    + 0.04 * max(hold_continuity_pressure[idx], 0.0)
                    - 0.10 * decision_defense_signal[idx] * held_defense_weight,
                    0.0,
                    1.0,
                )
            )
            action_keep_value = max(float(add_action_value[idx]), float(hold_action_value[idx]))
            action_release_value = max(float(reduce_action_value[idx]), float(exit_action_value[idx]))
            if (
                label in {"reduce", "exit"}
                and action_keep_value > action_release_value + 0.12
                and exit_timing_pressure < 0.62
                and market_downside_pressure[idx] < 0.24
            ):
                label = "add" if add_action_value[idx] > hold_action_value[idx] + 0.10 and add_quality[idx] > 0.10 else "hold"
            if (
                label in {"hold", "add", "skip"}
                and action_release_value > action_keep_value + 0.14
                and hold_days[idx] >= 3.0
                and (
                    sell_rank_score[idx] > 0.54
                    or lifecycle_sell_gate[idx] > 0.50
                    or exit_timing_pressure > 0.48
                )
            ):
                label = "exit" if exit_action_value[idx] >= reduce_action_value[idx] + 0.08 and hold_days[idx] >= 6.0 else "reduce"
            value_defense_override = bool(
                decision_defense_signal[idx] > 0.56
                and decision_release_gate[idx] > decision_deploy_gate[idx] + 0.08
                and release_value_target[idx] > deploy_value_target[idx] + 0.10
            )
            if (
                label in {"reduce", "exit"}
                and sell_arbitration < keep_arbitration + 0.08
                and sell_rank_score[idx] < 0.52
                and exit_timing_pressure < 0.54
                and market_downside_pressure[idx] < 0.22
                and not value_defense_override
            ):
                label = "hold"
            if (
                label in {"reduce", "exit"}
                and decision_deploy_gate[idx] > decision_release_gate[idx] + 0.08
                and deploy_value_target[idx] > release_value_target[idx] + 0.10
                and (pure_portfolio_defense_mode or decision_defense_signal[idx] < 0.40)
                and exit_timing_pressure < 0.58
            ):
                label = "hold"
            if (
                label == "add"
                and sell_arbitration > keep_arbitration + 0.06
                and sell_rank_score[idx] > 0.55
                and current_weight[idx] > 0.02
            ):
                label = "reduce" if exit_timing_pressure < 0.66 else "exit"
            if (
                label in {"hold", "skip"}
                and sell_arbitration > max(0.62, keep_arbitration + 0.12)
                and sell_rank_score[idx] > 0.62
                and hold_days[idx] >= 3.0
                and current_weight[idx] > 0.02
            ):
                label = "exit" if exit_timing_pressure > 0.68 and hold_days[idx] >= 6.0 else "reduce"
            if (
                label == "add"
                and clipped_intent_risk[idx] > 0.64
                and decision_deploy_gate[idx] < 0.38
                and deployment_gap < 0.12
                and add_quality[idx] < hold_quality[idx] + 0.10
            ):
                label = "hold"
            if label in {"reduce", "exit"} and market_downside_pressure[idx] < 0.12 and signal_decay_speed[idx] < 0.05 and drawdown_from_peak[idx] > -0.05 and hold_quality[idx] > reduce_quality[idx] - decoder_profile["hold_override_margin"]:
                label = "hold"
            if label in {"reduce", "exit"} and hold_days[idx] < 2 and exit_urgency[idx] < 0.24 and exit_hazard[idx] < 0.20 and exit_timing_pressure < 0.22 and hold_quality[idx] > -0.02:
                label = "hold"
            if label in {"reduce", "exit"} and days_since_last_buy[idx] <= max(3.0, hold_days[idx]) and exit_urgency[idx] < 0.26 and exit_hazard[idx] < 0.24 and exit_timing_pressure < 0.26 and hold_quality[idx] > -0.04:
                label = "hold"
            if label in {"reduce", "exit"} and hold_continuity_pressure[idx] > 0.24 and hold_quality[idx] > reduce_quality[idx] - 0.05 and market_downside_pressure[idx] < 0.20 and sell_pressure[idx] < 0.26 and exit_timing_pressure < 0.30:
                label = "hold"
            if label == "reduce" and days_since_last_reduce[idx] <= 2.0 and reduce_fraction[idx] < 0.22 and exit_timing_pressure < 0.28 and hold_quality[idx] > reduce_quality[idx] - decoder_profile["hold_override_margin"]:
                label = "hold"
            if label == "reduce" and reduce_reversal_pressure[idx] > 0.22 and reduce_fraction[idx] < 0.28 and exit_timing_pressure < 0.34 and hold_quality[idx] > reduce_quality[idx] - decoder_profile["reduce_gate_bonus"] and drawdown_from_peak[idx] > -0.08:
                label = "hold"
            if label == "exit" and exit_urgency[idx] < 0.18 + exit_patience_target * 0.06 and exit_hazard[idx] < 0.44 and exit_timing_pressure < 0.44 and hold_quality[idx] > reduce_quality[idx] - decoder_profile["reduce_gate_bonus"]:
                label = "reduce" if reduce_quality[idx] > 0.08 else "hold"
            if label == "reduce" and reduce_fraction[idx] < 0.18 and exit_timing_pressure < 0.24 and reduce_quality[idx] < 0.09 + decoder_profile["reduce_gate_bonus"] and hold_quality[idx] > 0.03 - decoder_profile["hold_override_margin"]:
                label = "hold"
            if label in {"reduce", "exit"} and duration_name in {"swing", "extended"} and sell_pressure[idx] < 0.26 and exit_timing_pressure < 0.34 and hold_quality[idx] >= reduce_quality[idx] - decoder_profile["hold_override_margin"]:
                label = "hold"
            exit_rescue = (
                (exit_prob > 0.52 or exit_hazard[idx] > 0.44 or exit_timing_pressure > 0.58)
                and hold_days[idx] >= 5.0
                and (
                    exit_urgency[idx] > 0.34
                    or exit_hazard[idx] > 0.54
                    or exit_timing_pressure > 0.66
                    or drawdown_from_peak[idx] < -0.12
                    or (market_downside_pressure[idx] > 0.16 and signal_decay_speed[idx] > 0.04)
                )
            )
            if exit_rescue and not profit_protected_trend and not (
                hold_quality[idx] > add_quality[idx] + 0.14
                and drawdown_from_peak[idx] > -0.05
                and market_downside_pressure[idx] < 0.12
                and exit_hazard[idx] < 0.52
                and exit_timing_pressure < 0.60
            ):
                label = "exit"
            reduce_rescue = (
                (reduce_prob > max(0.16 + reduce_bias_target * 0.08, exit_prob - 0.10) or reduce_fraction[idx] > 0.28)
                and hold_days[idx] >= 3.0
                and (reduce_quality[idx] + reduce_fraction[idx] * 0.35) > hold_quality[idx] + 0.03
                and (
                    signal_decay_speed[idx] > 0.05
                    or market_downside_pressure[idx] > 0.14
                    or exit_urgency[idx] > 0.18
                    or sell_pressure[idx] > 0.22
                    or drawdown_from_peak[idx] < -0.05
                )
            )
            if reduce_rescue and label not in {"exit"} and not (
                add_quality[idx] > reduce_quality[idx] + 0.08
                and hold_quality[idx] > reduce_quality[idx] - 0.01
                and exit_urgency[idx] < 0.14
                and market_downside_pressure[idx] < 0.12
                and sell_pressure[idx] < 0.20
            ):
                label = "reduce"
            if label == "reduce" and exit_timing_pressure > 0.70 and hold_days[idx] >= 7.0 and not profit_protected_trend:
                label = "exit"
            if (
                label == "add"
                and (
                    sell_pressure[idx] > 0.20
                    or exit_hazard[idx] > 0.18
                    or reduce_fraction[idx] > 0.16
                    or exit_timing_pressure > 0.34
                )
            ):
                label = "exit" if (exit_timing_pressure > 0.62 and hold_days[idx] >= 6.0 and not profit_protected_trend) else ("reduce" if (reduce_fraction[idx] > 0.26 and hold_days[idx] >= 3.0 and current_weight[idx] > 0.02) else "hold")
            if label in {"hold", "skip"} and (market_downside_pressure[idx] > 0.18 or portfolio_cash_pressure[idx] > 0.18 or signal_decay_speed[idx] > 0.10) and hold_days[idx] >= 3.0 and current_weight[idx] > 0.02 and drawdown_from_peak[idx] < -0.03:
                if (
                    decision_release_gate[idx] > decision_deploy_gate[idx] + 0.06
                    or sell_release_value[idx] > hold_continuation_value[idx] + 0.08
                    or (
                        not pure_portfolio_defense_mode
                        and decision_defense_signal[idx] > 0.54
                    )
                ):
                    label = "reduce"
            if label in {"hold", "skip"} and sell_pressure[idx] > 0.32 and hold_days[idx] >= 4.0 and current_weight[idx] > 0.025 and not profit_protected_trend:
                label = "exit" if (exit_hazard[idx] > 0.58 or exit_timing_pressure > 0.68) else "reduce"
            if label in {"hold", "reduce"} and exit_timing_pressure > 0.74 and hold_days[idx] >= 8.0 and current_weight[idx] > 0.03 and not profit_protected_trend:
                label = "exit"
            if (
                label in {"hold", "skip"}
                and add_quality[idx] > 0.14
                and duration_name in {"swing", "extended"}
                and current_weight[idx] < 0.12
                and reduce_quality[idx] < hold_quality[idx] + 0.03
                and exit_urgency[idx] < 0.18
                and sell_pressure[idx] < 0.18
                and exit_timing_pressure < 0.24
                and signal_decay_speed[idx] < 0.08
                and market_downside_pressure[idx] < 0.16
            ):
                label = "add"
        else:
            open_gate = (
                0.08
                + risk_off_score * decoder_profile["open_gate_bonus"]
                + (reentry_cooldown[idx] + exit_reentry_pressure[idx]) * decoder_profile["reentry_penalty"]
                + reentry_guard_target
                + (market_downside_pressure[idx] + cash_regime_pressure[idx] * 0.5) * decoder_profile["risk_off_open_penalty"]
                + cash_defense_value[idx] * 0.04
                - alpha_opportunity_value[idx] * 0.035
                - deployment_opportunity_cost[idx] * 0.025
            )
            if label == "open" and (entry_quality[idx] < open_gate or duration_name == "avoid"):
                label = "skip"
            if label == "open" and open_action_value[idx] < 0.30 and action_value_consistency_target[idx] < 0.48:
                label = "skip"
            if label == "open" and clipped_intent_risk[idx] > 0.72 and decision_deploy_gate[idx] < 0.42 and entry_quality[idx] < open_gate + 0.05:
                label = "skip"
            value_entry_ready = (
                alpha_opportunity_value[idx] > 0.52
                and deployment_opportunity_cost[idx] > 0.42
                and decision_deploy_gate[idx] > decision_release_gate[idx] + 0.06
                and decision_defense_signal[idx] < 0.42
            )
            action_value_entry_ready = (
                open_action_value[idx] > 0.56
                and action_value_consistency_target[idx] > 0.54
                and decision_defense_signal[idx] < 0.48
            )
            if label in {"skip", "hold"} and duration_name != "avoid" and (
                entry_quality[idx] > (0.075 + defensive_score * 0.01)
                or (entry_quality[idx] > 0.055 and probability_map["open"][idx] > 0.035)
                or value_entry_ready
                or action_value_entry_ready
            ):
                if probability_map["open"][idx] > 0.035 or duration_name in {"swing", "extended"}:
                    if (reentry_cooldown[idx] <= 0.25 and days_since_last_exit[idx] > 3.0) or entry_quality[idx] > open_gate + 0.035:
                        label = "open"
        adjusted_labels[idx] = label

    candidate_budget_target = max(1, int(round(global_targets["candidate_budget"])))
    open_candidate_scores = (
        entry_quality
        + probability_map["open"] * 0.45
        + np.clip(duration_days - 3.0, 0.0, None) / 30.0
        + alpha_opportunity_value * 0.18
        + deployment_opportunity_cost * 0.14
        + open_action_value * 0.16
        + multi_horizon_path_value * 0.10
        + deploy_value_target * 0.12
        + decision_deploy_gate * 0.10
        + portfolio_daily_receiver_score * 0.16
        + portfolio_daily_receiver_executability * 0.08
        - decision_defense_signal * (0.16 if pure_portfolio_defense_mode else 0.12)
        - portfolio_daily_cash_score * 0.06
        - decision_release_gate * 0.06
    )
    open_candidate_scores = np.where(current_weight > 1e-8, -1e9, open_candidate_scores)
    open_candidate_scores = np.where(np.asarray(predicted_duration_labels) == "avoid", -1e9, open_candidate_scores)
    active_candidates = sum(1 for idx, label in enumerate(adjusted_labels) if str(label) in {"open", "add", "hold"} or current_weight[idx] > 1e-8)
    missing_candidates = max(0, min(candidate_budget_target, len(adjusted_labels)) - active_candidates)
    if missing_candidates > 0 and np.isfinite(open_candidate_scores).any():
        promoted = 0
        min_open_score = float(
            np.clip(
                0.06
                + open_risk_off_score * 0.035
                + max(portfolio_sell_pressure - 0.22, 0.0) * 0.04,
                0.06,
                0.14,
            )
        )
        for idx in np.argsort(open_candidate_scores)[::-1]:
            if (
                promoted >= missing_candidates
                or float(open_candidate_scores[idx]) < min_open_score
                or current_weight[idx] > 1e-8
                or (
                    open_risk_off_score > 0.46
                    and market_downside_pressure[idx] > 0.18
                    and decision_deploy_gate[idx] < 0.44
                )
            ):
                continue
            adjusted_labels[idx] = "open"
            promoted += 1

    blended_delta = np.asarray(delta_hint, dtype=float).copy()
    action_strength = np.zeros(len(state_frame), dtype=float)
    hold_boost = np.zeros(len(state_frame), dtype=float)
    for idx, label in enumerate(adjusted_labels):
        duration_bonus = max(duration_days[idx] - 3.0, 0.0) / 20.0
        exit_timing_pressure = exit_timing_pressure_values[idx]
        defense_drag_component = decision_defense_signal[idx] * (
            0.08 if (not pure_portfolio_defense_mode or label in {"open", "add"}) else 0.0
        )
        sell_drag = (
            sell_pressure[idx] * 0.45
            + exit_hazard[idx] * 0.10
            + exit_timing_pressure * 0.10
            + lifecycle_sell_gate[idx] * 0.14
            + sell_rank_score[idx] * 0.08
            + sell_release_value[idx] * 0.12
            + decision_release_gate[idx] * 0.10
            + defense_drag_component
            - decision_deploy_gate[idx] * 0.12
            - deploy_value_target[idx] * 0.08
        )
        if label == "open":
            value_boost = (
                alpha_opportunity_value[idx] * 0.16
                + deployment_opportunity_cost[idx] * 0.12
                + deploy_value_target[idx] * 0.14
                + decision_deploy_gate[idx] * 0.12
                + deploy_executability_target[idx] * 0.10
                + portfolio_daily_receiver_score[idx] * 0.12
                + portfolio_daily_receiver_executability[idx] * 0.08
                + open_action_value[idx] * 0.12
                + multi_horizon_path_value[idx] * 0.08
                + large_upside_1d_target[idx] * 0.06
            )
            blended_delta[idx] = np.clip(max(blended_delta[idx], 0.02 + entry_quality[idx] * 0.55 + duration_bonus * 0.05 + value_boost * 0.08) * (1.0 - sell_drag * 0.28 - clipped_intent_risk[idx] * 0.08), 0.0, 0.23)
            action_strength[idx] = np.clip(entry_quality[idx] + probability_map["open"][idx] * 0.55 + duration_bonus * 0.40 + value_boost - exit_reentry_pressure[idx] * 0.12 - sell_drag * 0.18 - open_risk_off_score * 0.04 - clipped_intent_risk[idx] * 0.05, 0.0, None)
            hold_boost[idx] = np.clip(reentry_readiness[idx] * 0.25 + duration_bonus * 0.20 + value_boost * 0.25 - sell_drag * 0.10, 0.0, None)
        elif label == "add":
            value_boost = (
                hold_continuation_value[idx] * 0.10
                + alpha_opportunity_value[idx] * 0.12
                + deployment_opportunity_cost[idx] * 0.08
                + add_action_value[idx] * 0.12
                + multi_horizon_path_value[idx] * 0.06
                + deploy_value_target[idx] * 0.12
                + decision_deploy_gate[idx] * 0.10
                + deploy_executability_target[idx] * 0.08
                + portfolio_daily_receiver_score[idx] * 0.10
                + portfolio_daily_receiver_executability[idx] * 0.08
                + portfolio_daily_receiver_add_capacity[idx] * 0.08
                - (1.0 - portfolio_daily_receiver_add_capacity[idx]) * 0.16
            )
            blended_delta[idx] = np.clip(max(blended_delta[idx], 0.01 + add_quality[idx] * 0.35 + duration_bonus * 0.03 + value_boost * 0.06) * (1.0 - sell_drag * 0.50 - clipped_intent_risk[idx] * 0.10), 0.0, 0.18)
            action_strength[idx] = np.clip(add_quality[idx] + probability_map["add"][idx] * 0.45 + duration_bonus * 0.30 + value_boost - sell_drag * 0.32 - clipped_intent_risk[idx] * 0.07, 0.0, None)
            hold_boost[idx] = np.clip(hold_quality[idx] + duration_bonus * 0.25 + hold_continuation_value[idx] * 0.16 - sell_drag * 0.16, 0.0, None)
        elif label == "hold":
            blended_delta[idx] = np.clip(
                (max(blended_delta[idx] * 0.30, 0.0) + hold_quality[idx] * 0.10 + hold_continuation_value[idx] * 0.035 + decoder_profile["hold_delta_bonus"]) * (1.0 - sell_drag * 0.56),
                0.0,
                0.08 + decoder_profile["hold_delta_bonus"],
            )
            action_strength[idx] = np.clip(hold_quality[idx] + probability_map["hold"][idx] * 0.35 + duration_bonus * 0.25 + hold_continuation_value[idx] * 0.18 + hold_action_value[idx] * 0.14 + deploy_value_target[idx] * 0.10 + decision_deploy_gate[idx] * 0.08 - sell_drag * 0.22, 0.0, None)
            hold_boost[idx] = np.clip(
                hold_quality[idx]
                + duration_bonus * 0.30
                + decoder_profile["hold_bias_bonus"] * (0.20 + exit_patience_target)
                + hold_continuity_pressure[idx] * 0.08
                + hold_continuation_value[idx] * 0.12
                + alpha_opportunity_value[idx] * 0.08,
                0.0,
                None,
            )
        elif label == "reduce":
            blended_delta[idx] = -np.clip(
                max(
                    -blended_delta[idx],
                    reduce_fraction[idx]
                    + 0.04
                    + reduce_quality[idx] * (0.28 + reduce_bias_target - decoder_profile["reduce_delta_softener"])
                    + sell_pressure[idx] * 0.18
                    + lifecycle_sell_gate[idx] * 0.10
                    + sell_rank_score[idx] * 0.10
                    + sell_release_value[idx] * 0.12
                    + reduce_action_value[idx] * 0.12
                    + release_value_target[idx] * 0.10
                    + decision_release_gate[idx] * 0.08
                    + portfolio_daily_source_executability[idx] * 0.08
                    + portfolio_daily_source_release_capacity[idx] * 0.05
                    + decision_defense_signal[idx] * (0.06 if not pure_portfolio_defense_mode else 0.0)
                    + market_downside_pressure[idx] * 0.16
                    + signal_decay_speed[idx] * 0.18
                    + cash_regime_pressure[idx] * 0.10
                    - hold_quality[idx] * 0.08
                    - hold_continuation_value[idx] * 0.06
                    - deploy_value_target[idx] * 0.07,
                ),
                0.0,
                0.75,
            )
            action_strength[idx] = np.clip(
                reduce_quality[idx]
                + probability_map["reduce"][idx] * 0.35
                + reduce_fraction[idx] * 0.45
                + exit_hazard[idx] * 0.08
                + exit_timing_pressure * 0.12
                + lifecycle_sell_gate[idx] * 0.14
                + sell_rank_score[idx] * 0.12
                + sell_release_value[idx] * 0.16
                + exit_action_value[idx] * 0.14
                + release_value_target[idx] * 0.10
                + decision_release_gate[idx] * 0.08
                + portfolio_daily_source_executability[idx] * 0.08
                + portfolio_daily_source_release_capacity[idx] * 0.05
                + decision_defense_signal[idx] * (0.06 if not pure_portfolio_defense_mode else 0.0)
                + reduce_bias_target * 0.25
                - reduce_reversal_pressure[idx] * 0.12
                - hold_continuation_value[idx] * 0.08,
                0.0,
                None,
            )
        elif label == "exit":
            blended_delta[idx] = -1.0
            action_strength[idx] = np.clip(
                exit_urgency[idx]
                + exit_hazard[idx] * 0.60
                + exit_timing_pressure * 0.36
                + lifecycle_sell_gate[idx] * 0.16
                + sell_rank_score[idx] * 0.10
                + sell_release_value[idx] * 0.16
                + release_value_target[idx] * 0.10
                + decision_release_gate[idx] * 0.08
                + decision_defense_signal[idx] * (0.08 if not pure_portfolio_defense_mode else 0.0)
                + probability_map["exit"][idx] * 0.45
                + market_downside_pressure[idx] * 0.12
                + signal_decay_speed[idx] * 0.15
                - exit_patience_target * 0.10
                - hold_continuation_value[idx] * 0.08,
                0.0,
                None,
            )
        else:
            blended_delta[idx] = 0.0
            action_strength[idx] = probability_map["skip"][idx] * 0.20

    sell_attribution_score = np.asarray(preliminary_sell_attribution, dtype=float).copy()
    for idx, label in enumerate(adjusted_labels):
        if current_weight[idx] <= 1e-8:
            continue
        profit_protected = float(
            np.clip(
                max(pnl_rank_in_portfolio[idx], 0.0) * max(score_rank_pct[idx] - 0.55, 0.0),
                0.0,
                1.0,
            )
        )
        fallback_sell_attribution = float(
            np.clip(
                0.34 * max(reduce_quality[idx], 0.0)
                + 0.18 * reduce_fraction[idx]
                + 0.16 * exit_hazard[idx]
                + 0.14 * exit_timing_pressure_values[idx]
                + 0.10 * max(drawdown_rank_in_portfolio[idx], 0.0)
                + 0.10 * max(budget_model_cash_timing_signal, blended_budget_risk)
                + (0.06 if label in {"reduce", "exit"} else 0.0)
                - 0.24 * max(hold_quality[idx], 0.0)
                - 0.10 * max(add_quality[idx], 0.0)
                - 0.10 * max(score_rank_pct[idx], 0.0)
                - 0.10 * profit_protected,
                0.0,
                1.0,
            )
        )
        if predicted_sell_attribution is not None:
            sell_attribution_score[idx] = float(
                np.clip(0.72 * float(predicted_sell_attribution[idx]) + 0.28 * fallback_sell_attribution, 0.0, 1.0)
            )
        else:
            sell_attribution_score[idx] = fallback_sell_attribution

    native_allocation_columns: dict[str, np.ndarray] = {}
    if supports_portfolio_native_allocation_vector_heads:
        receiver_candidate_mask = np.asarray(
            (
                (portfolio_daily_receiver_executability > 0.08)
                | (portfolio_daily_receiver_score > 0.12)
                | (current_weight > 1.0e-8)
            ),
            dtype=np.float32,
        )
        source_candidate_mask = np.asarray(
            (
                (current_weight > 1.0e-8)
                & (
                    (portfolio_daily_source_executability > 0.05)
                    | (portfolio_daily_source_score > 0.10)
                    | (sell_pressure > 0.12)
                )
            ),
            dtype=np.float32,
        )

        def _state_column(name: str, default: float = 0.0) -> np.ndarray:
            if name in state_frame.columns:
                return np.nan_to_num(state_frame[name].astype(float).to_numpy(dtype=float), nan=default, posinf=default, neginf=default)
            return np.full(len(state_frame), float(default), dtype=float)

        def _native_tensor(values: np.ndarray | list[float] | float) -> torch.Tensor:
            if np.isscalar(values):
                array = np.full(len(state_frame), float(values), dtype=np.float32)
            else:
                array = np.asarray(values, dtype=np.float32)
            return torch.as_tensor(array, dtype=torch.float32)

        native_targets = {
            "date_code": _native_tensor(np.zeros(len(state_frame), dtype=np.float32)),
            "current_weight": _native_tensor(np.clip(current_weight, 0.0, 1.0)),
            "portfolio_daily_receiver_candidate_mask": _native_tensor(receiver_candidate_mask),
            "portfolio_daily_source_candidate_mask": _native_tensor(source_candidate_mask),
            "portfolio_daily_receiver_executable_candidate": _native_tensor(receiver_candidate_mask),
            "portfolio_daily_source_executable_candidate": _native_tensor(source_candidate_mask),
            "gross_exposure_target": _native_tensor(float(global_targets.get("gross_exposure_target", 0.60) or 0.60)),
            "turnover_budget": _native_tensor(float(global_targets.get("turnover_budget", 0.36) or 0.36)),
            "max_position_weight_target": _native_tensor(float(global_targets.get("max_position_weight_target", 0.20) or 0.20)),
            "budget_cash_timing_signal_target": _native_tensor(
                float(global_targets.get("budget_model_cash_timing_signal", 0.0) or 0.0)
            ),
            "portfolio_daily_unified_receiver_score": _native_tensor(portfolio_daily_unified_receiver_score),
            "portfolio_daily_unified_source_score": _native_tensor(portfolio_daily_unified_source_score),
            "portfolio_daily_allocation_cash_deployment_target": _native_tensor(
                portfolio_daily_allocation_cash_deployment_target
            ),
            "portfolio_daily_allocation_net_utility_target": _native_tensor(
                portfolio_daily_allocation_net_utility_target
            ),
            "portfolio_daily_allocation_final_objective": _native_tensor(
                portfolio_daily_allocation_final_objective
            ),
            "portfolio_daily_allocation_uncertainty_pressure_target": _native_tensor(
                portfolio_daily_allocation_uncertainty_pressure_target
            ),
            "portfolio_daily_allocation_tail_risk_control_target": _native_tensor(
                portfolio_daily_allocation_tail_risk_control_target
            ),
            "portfolio_daily_allocation_drawdown_control_target": _native_tensor(
                portfolio_daily_allocation_drawdown_control_target
            ),
            "market_downside_pressure": _native_tensor(market_downside_pressure),
            "cash_regime_pressure": _native_tensor(cash_regime_pressure),
            "portfolio_daily_receiver_forward_excess_5d": _native_tensor(
                _state_column("portfolio_daily_receiver_forward_excess_5d", 0.0)
            ),
            "portfolio_daily_source_forward_excess_5d": _native_tensor(
                _state_column("portfolio_daily_source_forward_excess_5d", 0.0)
            ),
        }
        native_projection = _project_native_allocation_vector(outputs, native_targets, return_terms=False)
        for column_name in _NATIVE_ALLOCATION_VECTOR_NAMES:
            if column_name not in native_projection:
                continue
            native_allocation_columns[column_name] = np.clip(
                native_projection[column_name].detach().cpu().numpy().astype(float),
                0.0 if column_name != "portfolio_daily_target_delta" else -1.0,
                1.0,
            )
        if "portfolio_daily_target_delta" in native_allocation_columns:
            native_allocation_columns["portfolio_daily_target_delta"] = np.nan_to_num(
                native_projection["portfolio_daily_target_delta"].detach().cpu().numpy().astype(float),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )

    policy = pd.DataFrame(
        {
            "stock": state_frame["stock"].astype(str).to_numpy(),
            "action_label": adjusted_labels,
            "action_strength": action_strength,
            "target_delta_hint": blended_delta,
            "hold_boost": hold_boost,
            "exit_urgency": exit_urgency + exit_hazard * 0.65 + probability_map["exit"] * 0.55 + probability_map["reduce"] * 0.35 + np.clip(-blended_delta, 0.0, None),
            "entry_quality": entry_quality,
            "hold_quality": hold_quality,
            "add_quality": add_quality,
            "reduce_quality": reduce_quality,
            "reduce_fraction": reduce_fraction,
            "reentry_readiness": reentry_readiness,
            "exit_hazard": exit_hazard,
            "sell_pressure": sell_pressure,
            "sell_attribution_score": sell_attribution_score,
            "sell_rank_score": sell_rank_score,
            "lifecycle_sell_gate": lifecycle_sell_gate,
            "large_upside_1d_target": large_upside_1d_target,
            "alpha_opportunity_value": alpha_opportunity_value,
            "hold_continuation_value": hold_continuation_value,
            "sell_release_value": sell_release_value,
            "cash_defense_value": cash_defense_value,
            "deployment_opportunity_cost": deployment_opportunity_cost,
            "risk_adjusted_action_value": risk_adjusted_action_value,
            "multi_horizon_forward_value": multi_horizon_forward_value,
            "multi_horizon_forward_risk": multi_horizon_forward_risk,
            "multi_horizon_path_value": multi_horizon_path_value,
            "open_action_value": open_action_value,
            "add_action_value": add_action_value,
            "hold_action_value": hold_action_value,
            "reduce_action_value": reduce_action_value,
            "exit_action_value": exit_action_value,
            "relative_opportunity_value": relative_opportunity_value,
            "action_value_consistency_target": action_value_consistency_target,
            "value_arbitration_target": value_arbitration_target,
            "deploy_value_target": deploy_value_target,
            "release_value_target": release_value_target,
            "defense_value_target": defense_value_target,
            "deploy_gate_target": deploy_gate_target,
            "release_gate_target": release_gate_target,
            "defense_gate_target": defense_gate_target,
            "deploy_executability_target": deploy_executability_target,
            "portfolio_daily_receiver_add_headroom": portfolio_daily_receiver_add_headroom,
            "portfolio_daily_receiver_min_add_delta": portfolio_daily_receiver_min_add_delta,
            "portfolio_daily_receiver_add_capacity": portfolio_daily_receiver_add_capacity,
            "portfolio_daily_receiver_executability": portfolio_daily_receiver_executability,
            "portfolio_daily_receiver_score": portfolio_daily_receiver_score,
            "portfolio_daily_source_min_release_delta": portfolio_daily_source_min_release_delta,
            "portfolio_daily_source_release_capacity": portfolio_daily_source_release_capacity,
            "portfolio_daily_source_forward_spread_score": portfolio_daily_source_forward_spread_score,
            "portfolio_daily_source_bad_forward_spread_risk": portfolio_daily_source_bad_forward_spread_risk,
            "portfolio_daily_source_economic_release_score": portfolio_daily_source_economic_release_score,
            "portfolio_daily_source_economic_block_risk": portfolio_daily_source_economic_block_risk,
            "portfolio_daily_source_forward_strength_brake_risk": portfolio_daily_source_forward_strength_brake_risk,
            "portfolio_daily_source_forward_proxy_keep_risk": portfolio_daily_source_forward_proxy_keep_risk,
            "portfolio_daily_source_release_quality": portfolio_daily_source_release_quality,
            "portfolio_daily_source_opportunity_cost": portfolio_daily_source_opportunity_cost,
            "portfolio_daily_source_executability": portfolio_daily_source_executability,
            "portfolio_daily_source_score": portfolio_daily_source_score,
            "portfolio_daily_cash_score": portfolio_daily_cash_score,
            "portfolio_daily_unified_receiver_score": portfolio_daily_unified_receiver_score,
            "portfolio_daily_unified_source_score": portfolio_daily_unified_source_score,
            "portfolio_daily_unified_cash_score": portfolio_daily_unified_cash_score,
            "portfolio_daily_source_positive_forward_penalty": portfolio_daily_source_positive_forward_penalty,
            "portfolio_daily_source_opportunity_cost_penalty": portfolio_daily_source_opportunity_cost_penalty,
            "portfolio_daily_source_strong_false_sell_penalty": portfolio_daily_source_strong_false_sell_penalty,
            "portfolio_daily_source_hard_negative_penalty": portfolio_daily_source_hard_negative_penalty,
            "portfolio_daily_source_tail_false_sell_penalty": portfolio_daily_source_tail_false_sell_penalty,
            "portfolio_daily_source_release_preference": portfolio_daily_source_release_preference,
            "portfolio_daily_receiver_source_spread_reward": portfolio_daily_receiver_source_spread_reward,
            "portfolio_daily_transfer_regret_target": portfolio_daily_transfer_regret_target,
            "portfolio_daily_unified_allocation_objective": portfolio_daily_unified_allocation_objective,
            "portfolio_daily_allocation_trade_quality_target": portfolio_daily_allocation_trade_quality_target,
            "portfolio_daily_allocation_cash_deployment_target": portfolio_daily_allocation_cash_deployment_target,
            "portfolio_daily_allocation_risk_adjusted_return_target": portfolio_daily_allocation_risk_adjusted_return_target,
            "portfolio_daily_allocation_drawdown_control_target": portfolio_daily_allocation_drawdown_control_target,
            "portfolio_daily_allocation_monthly_quality_target": portfolio_daily_allocation_monthly_quality_target,
            "portfolio_daily_allocation_final_objective": portfolio_daily_allocation_final_objective,
            "portfolio_daily_allocation_uncertainty_pressure_target": portfolio_daily_allocation_uncertainty_pressure_target,
            "portfolio_daily_allocation_tail_risk_control_target": portfolio_daily_allocation_tail_risk_control_target,
            "portfolio_daily_allocation_decision_focused_objective": portfolio_daily_allocation_decision_focused_objective,
            "portfolio_daily_allocation_net_utility_target": portfolio_daily_allocation_net_utility_target,
            "portfolio_daily_allocation_credit_closure_target": portfolio_daily_allocation_credit_closure_target,
            "portfolio_daily_allocation_resource_efficiency_target": portfolio_daily_allocation_resource_efficiency_target,
            "portfolio_daily_receiver_funding_coverage": portfolio_daily_receiver_funding_coverage,
            "portfolio_daily_funding_closure_score": portfolio_daily_funding_closure_score,
            "portfolio_daily_allocation_transfer_score": portfolio_daily_allocation_transfer_score,
            "portfolio_daily_allocation_dead_branch_risk": portfolio_daily_allocation_dead_branch_risk,
            "decision_deploy_gate": decision_deploy_gate,
            "decision_release_gate": decision_release_gate,
            "decision_defense_signal": decision_defense_signal,
            "decision_portfolio_defense_signal": np.full(len(state_frame), portfolio_defense_signal, dtype=float),
            "budget_model_hierarchical_mode": np.full(len(state_frame), 1.0 if hierarchical_stock_gate_mode else 0.0, dtype=float),
            "budget_model_constraint_only_mode": np.full(len(state_frame), 1.0 if pure_portfolio_defense_mode else 0.0, dtype=float),
            "clipped_intent_risk": clipped_intent_risk,
            "exit_timing_pressure": exit_timing_pressure_values,
            "planned_holding_bucket": predicted_duration_labels,
            "planned_holding_days": duration_days,
            "planned_holding_days_bucket": bucket_duration_days,
            "planned_holding_days_regressed": regressed_duration_days,
            "policy_decision_mode": np.full(
                len(state_frame),
                DIRECT_ACTION_VALUE_POLICY_MODE if direct_action_value_mode else "logit_rule_bridge_v1",
                dtype=object,
            ),
            "direct_action_value_label": direct_action_value_label,
            "direct_action_value_applied": direct_action_value_applied,
            "direct_action_value_selected": direct_action_value_selected,
            "direct_action_value_gap": direct_action_value_gap,
            "direct_action_utility_skip": direct_action_utility["skip"],
            "direct_action_utility_open": direct_action_utility["open"],
            "direct_action_utility_hold": direct_action_utility["hold"],
            "direct_action_utility_add": direct_action_utility["add"],
            "direct_action_utility_reduce": direct_action_utility["reduce"],
            "direct_action_utility_exit": direct_action_utility["exit"],
            "decoder_profile": decoder_profile_name,
        }
    ).set_index("stock")
    for column_name, values in native_allocation_columns.items():
        policy[column_name] = values
    for label in ACTION_CLASSES:
        policy[f"prob_{label}"] = probability_map[label]
    return policy, global_targets
