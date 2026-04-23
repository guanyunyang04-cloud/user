from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
}
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
    eps = 1.0e-4
    bce_loss = (
        nn.functional.binary_cross_entropy(torch.clamp(keep_prob, eps, 1.0 - eps), protected_keep_target) * 0.52
        + nn.functional.binary_cross_entropy(torch.clamp(release_prob, eps, 1.0 - eps), funding_release_target) * 0.48
    )
    protected_dominant = protected_keep_target > funding_release_target + 0.12
    release_dominant = funding_release_target > protected_keep_target + 0.12
    margin = torch.tensor(0.10, device=device)
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
        return bce_loss + torch.stack(margin_terms).mean() * 0.26
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
        self.deploy_value_head = nn.Linear(int(hidden_dim), 1)
        self.release_value_head = nn.Linear(int(hidden_dim), 1)
        self.defense_value_head = nn.Linear(int(hidden_dim), 1)
        self.deploy_gate_head = nn.Linear(int(hidden_dim), 1)
        self.release_gate_head = nn.Linear(int(hidden_dim), 1)
        self.defense_gate_head = nn.Linear(int(hidden_dim), 1)

    def forward(self, static_x: torch.Tensor, sequence_x: torch.Tensor) -> dict[str, torch.Tensor]:
        _, hidden = self.sequence_encoder(sequence_x)
        seq_hidden = hidden[-1]
        static_hidden = self.static_backbone(static_x)
        fused = self.fusion(torch.cat([static_hidden, seq_hidden], dim=-1))
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
            "deploy_value_target": torch.sigmoid(self.deploy_value_head(fused).squeeze(-1)),
            "release_value_target": torch.sigmoid(self.release_value_head(fused).squeeze(-1)),
            "defense_value_target": torch.sigmoid(self.defense_value_head(fused).squeeze(-1)),
            "deploy_gate_target": torch.sigmoid(self.deploy_gate_head(fused).squeeze(-1)),
            "release_gate_target": torch.sigmoid(self.release_gate_head(fused).squeeze(-1)),
            "defense_gate_target": torch.sigmoid(self.defense_gate_head(fused).squeeze(-1)),
        }


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
        payload = {
            "artifact_type": "continuous_policy_torch_seq_v3",
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
            "sample_model_config": {
                "static_input_dim": len(self.static_feature_names),
                "sequence_feature_dim": len(self.sequence_base_names),
                "sequence_steps": len(self.sequence_steps),
                "hidden_dim": int(self.sample_model.static_backbone[0].out_features),
                "sequence_hidden_dim": int(self.sample_model.sequence_encoder.hidden_size),
                "sequence_layers": int(getattr(self.sample_model, "sequence_layers", 1)),
                "dropout": float(self.sample_model.static_backbone[2].p),
            },
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

    sample_target_names = tuple(sample_targets.keys())
    dataset = TensorDataset(
        torch.as_tensor(X_static[train_idx], dtype=torch.float32),
        torch.as_tensor(X_sequence[train_idx], dtype=torch.float32),
        torch.as_tensor(y_action[train_idx], dtype=torch.long),
        torch.as_tensor(y_duration[train_idx], dtype=torch.long),
        torch.as_tensor(y_action_soft[train_idx], dtype=torch.float32),
        *[torch.as_tensor(sample_targets[name][train_idx], dtype=torch.float32) for name in sample_target_names],
    )
    loader = DataLoader(dataset, batch_size=max(32, int(batch_size)), shuffle=True, drop_last=False)
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
            value_arbitration_loss = _value_arbitration_consistency_loss(outputs, sample_batch_targets)
            three_value_gate_loss = _three_value_gate_consistency_loss(outputs, sample_batch_targets)
            hierarchical_three_value_gate_loss = _hierarchical_three_value_gate_consistency_loss(outputs, sample_batch_targets)
            funding_release_loss = _funding_release_discipline_loss(outputs, sample_batch_targets)
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
                + multi_objective_loss_weights.get("value_arbitration_total", 0.0) * value_arbitration_loss
                + multi_objective_loss_weights.get("three_value_gate_total", 0.0) * three_value_gate_loss
                + multi_objective_loss_weights.get("hierarchical_three_value_total", 0.0) * hierarchical_three_value_gate_loss
                + multi_objective_loss_weights.get("funding_release_total", 0.0) * funding_release_loss
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
            val_value_arbitration_loss = _value_arbitration_consistency_loss(val_outputs, val_targets)
            val_three_value_gate_loss = _three_value_gate_consistency_loss(val_outputs, val_targets)
            val_hierarchical_three_value_gate_loss = _hierarchical_three_value_gate_consistency_loss(val_outputs, val_targets)
            val_funding_release_loss = _funding_release_discipline_loss(val_outputs, val_targets)
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
                    + multi_objective_loss_weights.get("value_arbitration_total", 0.0) * val_value_arbitration_loss
                    + multi_objective_loss_weights.get("three_value_gate_total", 0.0) * val_three_value_gate_loss
                    + multi_objective_loss_weights.get("hierarchical_three_value_total", 0.0) * val_hierarchical_three_value_gate_loss
                    + multi_objective_loss_weights.get("funding_release_total", 0.0) * val_funding_release_loss
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

    diagnostics = {
        "trainer_backend": TRAINER_BACKEND_FORMAL_SEQ_V3,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
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
            if label == "open" and clipped_intent_risk[idx] > 0.72 and decision_deploy_gate[idx] < 0.42 and entry_quality[idx] < open_gate + 0.05:
                label = "skip"
            value_entry_ready = (
                alpha_opportunity_value[idx] > 0.52
                and deployment_opportunity_cost[idx] > 0.42
                and decision_deploy_gate[idx] > decision_release_gate[idx] + 0.06
                and decision_defense_signal[idx] < 0.42
            )
            if label in {"skip", "hold"} and duration_name != "avoid" and (
                entry_quality[idx] > (0.075 + defensive_score * 0.01)
                or (entry_quality[idx] > 0.055 and probability_map["open"][idx] > 0.035)
                or value_entry_ready
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
        + deploy_value_target * 0.12
        + decision_deploy_gate * 0.10
        - decision_defense_signal * (0.16 if pure_portfolio_defense_mode else 0.12)
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
                + deploy_value_target[idx] * 0.12
                + decision_deploy_gate[idx] * 0.10
                + deploy_executability_target[idx] * 0.08
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
            action_strength[idx] = np.clip(hold_quality[idx] + probability_map["hold"][idx] * 0.35 + duration_bonus * 0.25 + hold_continuation_value[idx] * 0.18 + deploy_value_target[idx] * 0.10 + decision_deploy_gate[idx] * 0.08 - sell_drag * 0.22, 0.0, None)
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
                    + release_value_target[idx] * 0.10
                    + decision_release_gate[idx] * 0.08
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
                + release_value_target[idx] * 0.10
                + decision_release_gate[idx] * 0.08
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
            "value_arbitration_target": value_arbitration_target,
            "deploy_value_target": deploy_value_target,
            "release_value_target": release_value_target,
            "defense_value_target": defense_value_target,
            "deploy_gate_target": deploy_gate_target,
            "release_gate_target": release_gate_target,
            "defense_gate_target": defense_gate_target,
            "deploy_executability_target": deploy_executability_target,
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
            "decoder_profile": decoder_profile_name,
        }
    ).set_index("stock")
    for label in ACTION_CLASSES:
        policy[f"prob_{label}"] = probability_map[label]
    return policy, global_targets
