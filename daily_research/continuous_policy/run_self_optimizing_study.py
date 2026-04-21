from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.baseline.data_provider import get_latest_completed_trading_date
from daily_research.continuous_policy.model_seq_v3 import (
    DAILY_HEAD_LAYOUT_CHOICES,
    DAILY_HEAD_LAYOUT_MONOLITHIC_V1,
    DEFAULT_LOSS_PROFILE,
)
from daily_research.continuous_policy.pipeline_utils import (
    BUDGET_OBJECTIVE_CHOICES,
    DEFAULT_BUDGET_OBJECTIVE,
)
from daily_research.continuous_policy.portfolio_simulator import (
    BUDGET_CALIBRATION_CHOICES,
    BUDGET_SEMANTICS_CHOICES,
    DEFAULT_BUDGET_CALIBRATION,
    DEFAULT_BUDGET_SEMANTICS,
    DEFAULT_EXECUTION_SEMANTICS,
    EXECUTION_SEMANTICS_CHOICES,
)
from daily_research.continuous_policy.run_continuous_policy_protocol import (
    PROMOTION_THRESHOLDS,
    main as protocol_main,
)
from daily_research.continuous_policy.runtime import (
    LATEST_BEHAVIOR_AUDIT_SUMMARY_PATH,
    LATEST_CONCLUSION_LEDGER_PATH,
    LATEST_EVALUATION_SUMMARY_PATH,
    LATEST_EXPORT_SUMMARY_PATH,
    LATEST_PROTOCOL_SUMMARY_PATH,
    LATEST_TRAIN_SUMMARY_PATH,
    PROTOCOLS_ROOT,
    RUNTIME_STATE_PATH,
    STUDIES_ROOT,
    ensure_layout,
    now_iso,
    read_json,
    timestamp_tag,
    update_latest_summary,
    write_json,
)
from daily_research.continuous_policy.training_contracts import (
    TRAINER_BACKENDS,
    TRAINER_BACKEND_FORMAL_SEQ_V3,
)


LATEST_STATE_PATHS: dict[str, Path] = {
    "latest_train_summary": LATEST_TRAIN_SUMMARY_PATH,
    "latest_evaluation_summary": LATEST_EVALUATION_SUMMARY_PATH,
    "latest_export_summary": LATEST_EXPORT_SUMMARY_PATH,
    "latest_protocol_summary": LATEST_PROTOCOL_SUMMARY_PATH,
    "latest_behavior_audit_summary": LATEST_BEHAVIOR_AUDIT_SUMMARY_PATH,
    "latest_conclusion_ledger": LATEST_CONCLUSION_LEDGER_PATH,
    "runtime_state": RUNTIME_STATE_PATH,
}


SEARCH_PROFILES: dict[str, dict[str, list[Any]]] = {
    "focused_seq_v1": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["holdcash_v3", "reduceexit_v4", "budget_v3", "reduceexit_cash_v4"],
        "loss_profile": [DEFAULT_LOSS_PROFILE],
        "learning_rate": [1.0e-3, 1.2e-3, 1.5e-3],
        "hidden_dim": [224, 256],
        "sequence_layers": [1, 2],
        "daily_hidden_dim": [96, 128],
        "dropout": [0.10, 0.12],
        "daily_dropout": [0.05, 0.08],
        "batch_size": [384, 512],
    },
    "seq2_return_recovery_v1": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": [
            DEFAULT_LOSS_PROFILE,
            "teacher_aux_continuous_v1",
            "teacher_aux_return_recovery_v1",
        ],
        "learning_rate": [1.2e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.12],
        "daily_dropout": [0.08],
        "batch_size": [512],
    },
    "seq2_return_recovery_v2": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": [
            DEFAULT_LOSS_PROFILE,
            "teacher_aux_return_recovery_v1",
            "teacher_aux_return_recovery_balanced_v2",
            "teacher_aux_return_recovery_stable_v2",
        ],
        "learning_rate": [1.2e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.12],
        "daily_dropout": [0.08],
        "batch_size": [512],
    },
    "seq2_return_recovery_v3": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": [
            "teacher_aux_return_recovery_v1",
            "teacher_aux_return_recovery_balanced_v2",
            "teacher_aux_return_recovery_balanced_v3",
            "teacher_aux_return_recovery_stable_v2",
        ],
        "learning_rate": [1.2e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.12],
        "daily_dropout": [0.08],
        "batch_size": [512],
    },
    "budget_layer_ablation_v1": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["teacher_aux_return_recovery_balanced_v2"],
        "budget_semantics": ["legacy_total_candidate", "action_budget_split_v1"],
        "budget_calibration": ["none", "cash_exit_guard_v1"],
        "learning_rate": [1.2e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.12],
        "daily_dropout": [0.08],
        "batch_size": [512],
    },
    "alpha_result_value_budget_r1": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_v1"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_exit_guard_v1"],
        "budget_objective": [DEFAULT_BUDGET_OBJECTIVE, "result_value_v1"],
        "alpha_prior_source": ["none", "active_execution_strategy"],
        "learning_rate": [1.2e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.12],
        "daily_dropout": [0.08],
        "batch_size": [512],
    },
    "split_heads_cash_timing_r1": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v2"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_exit_guard_v1"],
        "budget_objective": [DEFAULT_BUDGET_OBJECTIVE, "result_value_v2"],
        "alpha_prior_source": ["none", "active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [1.2e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.12],
        "daily_dropout": [0.08],
        "batch_size": [512],
    },
    "split_heads_cash_timing_r2": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v3"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_exit_guard_v1", "cash_translation_guard_v2"],
        "budget_objective": ["result_value_v2", "result_value_v3"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [1.2e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.12],
        "daily_dropout": [0.08],
        "batch_size": [512],
    },
    "split_heads_cash_timing_r3": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v4"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_translation_guard_v2", "cash_translation_sell_guard_v3"],
        "budget_objective": ["result_value_v2", "result_value_v4"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [1.2e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.12],
        "daily_dropout": [0.08],
        "batch_size": [512],
    },
    "split_heads_cash_timing_r4": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v4b"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_translation_guard_v2", "cash_translation_sell_guard_v3"],
        "budget_objective": ["result_value_v2", "result_value_v4b"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [1.2e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.12],
        "daily_dropout": [0.08],
        "batch_size": [512],
    },
    "split_heads_lifecycle_arbitration_r5": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v5"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_translation_guard_v2", "cash_translation_sell_guard_v3"],
        "budget_objective": ["result_value_v2", "result_value_v4b"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [1.2e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.12],
        "daily_dropout": [0.08],
        "batch_size": [512],
    },
    "split_heads_value_arbitration_r6": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v6"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_translation_guard_v2", "cash_translation_sell_guard_v3"],
        "budget_objective": ["result_value_v4b", "result_value_v5"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [1.2e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.12],
        "daily_dropout": [0.08],
        "batch_size": [512],
    },
    "split_heads_three_value_gate_r6b": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v6b"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_translation_guard_v2", "cash_translation_sell_guard_v3"],
        "budget_objective": ["result_value_v5", "result_value_v6"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [1.2e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.12],
        "daily_dropout": [0.08],
        "batch_size": [512],
    },
    "split_heads_hierarchical_arbitration_r7": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v6b", "alpha_result_value_budget_split_v7"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_translation_guard_v2", "cash_translation_sell_guard_v3"],
        "budget_objective": ["result_value_v7"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [1.2e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.12],
        "daily_dropout": [0.08],
        "batch_size": [512],
    },
    "split_heads_constraint_arbitration_r8": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v7", "alpha_result_value_budget_split_v8"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_translation_guard_v2", "cash_constraint_guard_v4"],
        "budget_objective": ["result_value_v8"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [1.2e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.12],
        "daily_dropout": [0.08],
        "batch_size": [512],
    },
    "split_heads_intent_preserving_translation_r9": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v7", "alpha_result_value_budget_split_v8"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_guard_v4", "cash_constraint_intent_guard_v5"],
        "budget_objective": ["result_value_v8"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [1.2e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.12],
        "daily_dropout": [0.08],
        "batch_size": [512],
    },
}


SEARCH_PROFILE_BASE_TRIALS: dict[str, dict[str, Any]] = {
    "focused_seq_v1": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "holdcash_v3",
        "loss_profile": DEFAULT_LOSS_PROFILE,
        "learning_rate": 1.5e-3,
        "hidden_dim": 224,
        "sequence_layers": 1,
        "daily_hidden_dim": 96,
        "dropout": 0.10,
        "daily_dropout": 0.05,
        "batch_size": 512,
    },
    "seq2_return_recovery_v1": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": DEFAULT_LOSS_PROFILE,
        "learning_rate": 1.2e-3,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.12,
        "daily_dropout": 0.08,
        "batch_size": 512,
    },
    "seq2_return_recovery_v2": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "teacher_aux_return_recovery_v1",
        "learning_rate": 1.2e-3,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.12,
        "daily_dropout": 0.08,
        "batch_size": 512,
    },
    "seq2_return_recovery_v3": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "teacher_aux_return_recovery_balanced_v2",
        "budget_semantics": "legacy_total_candidate",
        "budget_calibration": "none",
        "learning_rate": 1.2e-3,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.12,
        "daily_dropout": 0.08,
        "batch_size": 512,
    },
    "budget_layer_ablation_v1": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "teacher_aux_return_recovery_balanced_v2",
        "budget_semantics": "legacy_total_candidate",
        "budget_calibration": "none",
        "learning_rate": 1.2e-3,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.12,
        "daily_dropout": 0.08,
        "batch_size": 512,
    },
    "alpha_result_value_budget_r1": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_v1",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_exit_guard_v1",
        "budget_objective": DEFAULT_BUDGET_OBJECTIVE,
        "alpha_prior_source": "none",
        "learning_rate": 1.2e-3,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.12,
        "daily_dropout": 0.08,
        "batch_size": 512,
    },
    "split_heads_cash_timing_r1": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v2",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_exit_guard_v1",
        "budget_objective": DEFAULT_BUDGET_OBJECTIVE,
        "alpha_prior_source": "none",
        "daily_head_layout": "split_v2",
        "learning_rate": 1.2e-3,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.12,
        "daily_dropout": 0.08,
        "batch_size": 512,
    },
    "split_heads_cash_timing_r2": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v3",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_exit_guard_v1",
        "budget_objective": "result_value_v2",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 1.2e-3,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.12,
        "daily_dropout": 0.08,
        "batch_size": 512,
    },
    "split_heads_cash_timing_r3": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v4",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_translation_guard_v2",
        "budget_objective": "result_value_v2",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 1.2e-3,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.12,
        "daily_dropout": 0.08,
        "batch_size": 512,
    },
    "split_heads_cash_timing_r4": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v4b",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_translation_guard_v2",
        "budget_objective": "result_value_v2",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 1.2e-3,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.12,
        "daily_dropout": 0.08,
        "batch_size": 512,
    },
    "split_heads_lifecycle_arbitration_r5": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v5",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_translation_guard_v2",
        "budget_objective": "result_value_v2",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 1.2e-3,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.12,
        "daily_dropout": 0.08,
        "batch_size": 512,
    },
    "split_heads_value_arbitration_r6": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v6",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_translation_guard_v2",
        "budget_objective": "result_value_v5",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 1.2e-3,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.12,
        "daily_dropout": 0.08,
        "batch_size": 512,
    },
    "split_heads_three_value_gate_r6b": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v6b",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_translation_guard_v2",
        "budget_objective": "result_value_v6",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 1.2e-3,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.12,
        "daily_dropout": 0.08,
        "batch_size": 512,
    },
    "split_heads_hierarchical_arbitration_r7": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v7",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_translation_guard_v2",
        "budget_objective": "result_value_v7",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 1.2e-3,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.12,
        "daily_dropout": 0.08,
        "batch_size": 512,
    },
    "split_heads_constraint_arbitration_r8": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v8",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_guard_v4",
        "budget_objective": "result_value_v8",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 1.2e-3,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.12,
        "daily_dropout": 0.08,
        "batch_size": 512,
    },
    "split_heads_intent_preserving_translation_r9": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v7",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_intent_guard_v5",
        "budget_objective": "result_value_v8",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 1.2e-3,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.12,
        "daily_dropout": 0.08,
        "batch_size": 512,
    },
}


SEARCH_PROFILE_DEFAULT_OBJECTIVES: dict[str, str] = {
    "focused_seq_v1": "promotion_balanced_v2",
    "seq2_return_recovery_v1": "return_recovery_v2",
    "seq2_return_recovery_v2": "return_recovery_v2",
    "seq2_return_recovery_v3": "return_recovery_v2",
    "budget_layer_ablation_v1": "return_recovery_v2",
    "alpha_result_value_budget_r1": "return_recovery_v2",
    "split_heads_cash_timing_r1": "return_recovery_v2",
    "split_heads_cash_timing_r2": "return_recovery_v2",
    "split_heads_cash_timing_r3": "return_recovery_v2",
    "split_heads_cash_timing_r4": "return_recovery_v2",
    "split_heads_lifecycle_arbitration_r5": "return_recovery_v2",
    "split_heads_value_arbitration_r6": "return_recovery_v2",
    "split_heads_three_value_gate_r6b": "return_recovery_v2",
    "split_heads_hierarchical_arbitration_r7": "return_recovery_v2",
    "split_heads_constraint_arbitration_r8": "return_recovery_v2",
}


def _normalize_date_text(value: str) -> str:
    return pd.Timestamp(str(value)).strftime("%Y%m%d")


def _bounded(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    clipped = min(max(float(value), low), high)
    return (clipped - low) / float(high - low)


def _exposure_penalty(avg_gross_exposure: float) -> float:
    exposure = float(avg_gross_exposure)
    if 0.30 <= exposure <= 0.70:
        return 0.0
    if exposure < 0.30:
        return min((0.30 - exposure) * 2.2, 0.60)
    return min((exposure - 0.70) * 1.5, 0.60)


def _score_protocol_summary(
    protocol_summary: dict[str, Any],
    *,
    objective_profile: str = "promotion_balanced_v2",
) -> dict[str, Any]:
    evaluation = dict(protocol_summary.get("evaluation", {}) or {})
    metrics = dict(evaluation.get("continuous_policy_metrics", {}) or {})
    continuity = dict(evaluation.get("continuity_metrics", {}) or {})
    promotion_gate = dict(protocol_summary.get("promotion_gate", {}) or {})
    gate_checks = dict(promotion_gate.get("checks", {}) or {})
    training_evidence = dict(protocol_summary.get("training_evidence", {}) or {})
    behavior_audit = dict(protocol_summary.get("latest_behavior_audit", {}) or {})
    semantic_conflicts = dict(behavior_audit.get("semantic_conflicts", {}) or {})
    shadow = dict(protocol_summary.get("shadow", {}) or {})
    shadow_continuity = dict(shadow.get("continuity_metrics", {}) or {})

    annual_return = float(metrics.get("annual_return", 0.0) or 0.0)
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
    avg_gross_exposure = float(metrics.get("avg_gross_exposure", 0.0) or 0.0)
    open_win = float(continuity.get("open_win_rate_5d", 0.0) or 0.0)
    reduce_success = float(continuity.get("reduce_success_rate_5d", 0.0) or 0.0)
    exit_timeliness = float(continuity.get("exit_timeliness_rate_5d", 0.0) or 0.0)
    cash_timing = float(continuity.get("cash_timing_quality_1d", 0.0) or 0.0)
    hold_share = float(continuity.get("hold_share", 0.0) or 0.0)
    trend_capture = float(continuity.get("trend_capture_rate_10d", 0.0) or 0.0)
    reversal = float(continuity.get("immediate_reversal_rate_3d", 0.0) or 0.0)
    shadow_reversal = float(shadow_continuity.get("immediate_reversal_rate_3d", 0.0) or 0.0)
    training_evidence_ok = str(training_evidence.get("status", "") or "") == "sufficient"
    semantic_conflict_rate = float(semantic_conflicts.get("semantic_conflict_rate", 0.0) or 0.0)
    order_translation_conflict_rate = float(
        semantic_conflicts.get(
            "order_translation_conflict_rate",
            semantic_conflicts.get("weight_change_conflict_rate", semantic_conflict_rate),
        )
        or 0.0
    )

    threshold_gap_penalty = (
        max(0.0, PROMOTION_THRESHOLDS["open_win_rate_5d"] - open_win) * 1.00
        + max(0.0, PROMOTION_THRESHOLDS["reduce_success_rate_5d"] - reduce_success) * 1.10
        + max(0.0, PROMOTION_THRESHOLDS["exit_timeliness_rate_5d"] - exit_timeliness) * 1.20
        + max(0.0, PROMOTION_THRESHOLDS["cash_timing_quality_1d"] - cash_timing) * 1.25
    )
    reversal_excess_penalty = (
        max(0.0, reversal - 0.25) * 3.0
        + max(0.0, shadow_reversal - 0.18) * 2.5
    )
    negative_return_penalty = max(0.0, -annual_return)
    negative_sharpe_penalty = max(0.0, -sharpe)
    drawdown_excess_penalty = max(0.0, abs(min(max_drawdown, 0.0)) - 0.08)
    cash_floor_penalty = max(0.0, -0.08 - cash_timing)
    trend_floor_penalty = max(0.0, 0.30 - trend_capture)
    active_alignment_bonus = 0.0
    if gate_checks.get("annual_return_vs_active") is True:
        active_alignment_bonus += 0.20
    elif "annual_return_vs_active" in gate_checks:
        active_alignment_bonus -= 0.20
    if gate_checks.get("sharpe_vs_active") is True:
        active_alignment_bonus += 0.20
    elif "sharpe_vs_active" in gate_checks:
        active_alignment_bonus -= 0.20

    passed_checks = sum(1 for value in gate_checks.values() if bool(value))
    total_checks = max(len(gate_checks), 1)
    gate_pass_ratio = passed_checks / float(total_checks)

    if str(objective_profile or "promotion_balanced_v2") == "return_recovery_v2":
        performance_breakdown = {
            "annual_return": annual_return * 3.10,
            "sharpe": sharpe * 0.42,
            "open_win_rate_5d": open_win * 1.10,
            "reduce_success_rate_5d": reduce_success * 1.20,
            "exit_timeliness_rate_5d": exit_timeliness * 1.25,
            "cash_timing_quality_1d": _bounded(cash_timing, -0.35, 0.12) * 0.90,
            "hold_share": hold_share * 0.28,
            "trend_capture_rate_10d": trend_capture * 1.10,
            "active_alignment_bonus": active_alignment_bonus,
            "gate_pass_ratio": gate_pass_ratio * 0.48,
            "drawdown_penalty": -abs(min(max_drawdown, 0.0)) * 3.70,
            "exposure_penalty": -_exposure_penalty(avg_gross_exposure),
            "negative_return_penalty": -negative_return_penalty * 3.80,
            "negative_sharpe_penalty": -negative_sharpe_penalty * 0.72,
            "cash_floor_penalty": -cash_floor_penalty * 2.00,
            "drawdown_excess_penalty": -drawdown_excess_penalty * 5.20,
            "trend_floor_penalty": -trend_floor_penalty * 0.90,
            "semantic_conflict_penalty": -semantic_conflict_rate * 1.60,
            "order_translation_conflict_penalty": -order_translation_conflict_rate * 0.55,
        }
        stability_breakdown = {
            "gate_pass_ratio": gate_pass_ratio * 0.78,
            "reduce_success_rate_5d": reduce_success * 0.58,
            "exit_timeliness_rate_5d": exit_timeliness * 0.62,
            "cash_timing_quality_1d": _bounded(cash_timing, -0.35, 0.12) * 0.56,
            "hold_share": hold_share * 0.16,
            "reversal_penalty": -reversal * 1.20,
            "shadow_reversal_penalty": -shadow_reversal * 1.20,
            "reversal_excess_penalty": -reversal_excess_penalty,
            "drawdown_penalty": -abs(min(max_drawdown, 0.0)) * 5.90,
            "threshold_gap_penalty": -threshold_gap_penalty * 1.35,
            "training_evidence_bonus": 0.32 if training_evidence_ok else -0.40,
            "negative_return_penalty": -negative_return_penalty * 2.60,
            "negative_sharpe_penalty": -negative_sharpe_penalty * 0.60,
            "cash_floor_penalty": -cash_floor_penalty * 2.25,
            "drawdown_excess_penalty": -drawdown_excess_penalty * 4.80,
            "trend_floor_penalty": -trend_floor_penalty * 0.65,
            "semantic_conflict_penalty": -semantic_conflict_rate * 2.10,
            "order_translation_conflict_penalty": -order_translation_conflict_rate * 0.70,
        }
    else:
        performance_breakdown = {
            "annual_return": annual_return * 2.40,
            "sharpe": sharpe * 0.28,
            "open_win_rate_5d": open_win * 1.20,
            "reduce_success_rate_5d": reduce_success * 1.35,
            "exit_timeliness_rate_5d": exit_timeliness * 1.50,
            "cash_timing_quality_1d": _bounded(cash_timing, -0.35, 0.12) * 1.05,
            "hold_share": hold_share * 0.45,
            "trend_capture_rate_10d": trend_capture * 1.00,
            "active_alignment_bonus": active_alignment_bonus,
            "gate_pass_ratio": gate_pass_ratio * 0.40,
            "drawdown_penalty": -abs(min(max_drawdown, 0.0)) * 3.20,
            "exposure_penalty": -_exposure_penalty(avg_gross_exposure),
            "semantic_conflict_penalty": -semantic_conflict_rate * 1.45,
            "order_translation_conflict_penalty": -order_translation_conflict_rate * 0.45,
        }
        stability_breakdown = {
            "gate_pass_ratio": gate_pass_ratio * 0.65,
            "reduce_success_rate_5d": reduce_success * 0.55,
            "exit_timeliness_rate_5d": exit_timeliness * 0.60,
            "cash_timing_quality_1d": _bounded(cash_timing, -0.35, 0.12) * 0.45,
            "hold_share": hold_share * 0.20,
            "reversal_penalty": -reversal * 1.30,
            "shadow_reversal_penalty": -shadow_reversal * 1.25,
            "reversal_excess_penalty": -reversal_excess_penalty,
            "drawdown_penalty": -abs(min(max_drawdown, 0.0)) * 5.20,
            "threshold_gap_penalty": -threshold_gap_penalty,
            "training_evidence_bonus": 0.35 if training_evidence_ok else -0.35,
            "semantic_conflict_penalty": -semantic_conflict_rate * 1.95,
            "order_translation_conflict_penalty": -order_translation_conflict_rate * 0.60,
        }
    performance_score = round(sum(performance_breakdown.values()), 6)
    stability_score = round(sum(stability_breakdown.values()), 6)
    composite_score = round(performance_score + stability_score, 6)
    score_breakdown = {
        "performance": performance_breakdown,
        "stability": stability_breakdown,
    }
    return {
        "performance_score": performance_score,
        "stability_score": stability_score,
        "composite_score": composite_score,
        "score_breakdown": score_breakdown,
        "gate_pass_ratio": gate_pass_ratio,
        "passed_check_count": passed_checks,
        "total_check_count": total_checks,
        "objective_profile": str(objective_profile or "promotion_balanced_v2"),
        "primary_metrics": {
            "annual_return": annual_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "avg_gross_exposure": avg_gross_exposure,
            "open_win_rate_5d": open_win,
            "reduce_success_rate_5d": reduce_success,
            "exit_timeliness_rate_5d": exit_timeliness,
            "cash_timing_quality_1d": cash_timing,
            "hold_share": hold_share,
            "trend_capture_rate_10d": trend_capture,
            "immediate_reversal_rate_3d": reversal,
            "shadow_reversal_rate_3d": shadow_reversal,
            "semantic_conflict_rate": semantic_conflict_rate,
            "order_translation_conflict_rate": order_translation_conflict_rate,
            "training_evidence_status": str(training_evidence.get("status", "") or ""),
        },
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _build_candidate_pool(base: dict[str, Any], profile_name: str) -> list[dict[str, Any]]:
    search_space = SEARCH_PROFILES[profile_name]
    axes = []
    for key in search_space:
        axes.append([(key, value) for value in search_space[key]])
    candidate_pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for combo in itertools.product(*axes):
        candidate = dict(base)
        for key, value in combo:
            candidate[str(key)] = value
        key = json.dumps(candidate, ensure_ascii=False, sort_keys=True, default=_json_scalar)
        if key in seen:
            continue
        seen.add(key)
        candidate_pool.append(candidate)
    return candidate_pool


def _select_trials(
    *,
    base: dict[str, Any],
    profile_name: str,
    trial_count: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    candidate_pool = _build_candidate_pool(base, profile_name)
    baseline_key = json.dumps(base, ensure_ascii=False, sort_keys=True, default=_json_scalar)
    baseline_trial = dict(base)
    baseline_trial["trial_role"] = "baseline"
    remaining = [
        candidate
        for candidate in candidate_pool
        if json.dumps(candidate, ensure_ascii=False, sort_keys=True, default=_json_scalar) != baseline_key
    ]
    rng = random.Random(int(random_seed))
    rng.shuffle(remaining)
    selected = [baseline_trial]
    for candidate in remaining[: max(int(trial_count) - 1, 0)]:
        selected.append(candidate)
    return selected[: max(int(trial_count), 1)]


def _build_trial_result_from_protocol(
    *,
    trial_id: int,
    trial_tag: str,
    trial_config: dict[str, Any],
    protocol_summary_path: Path,
    objective_profile: str,
) -> "TrialResult":
    protocol_summary = read_json(protocol_summary_path)
    if not protocol_summary:
        raise FileNotFoundError(f"Missing protocol summary: {protocol_summary_path}")
    score_payload = _score_protocol_summary(protocol_summary, objective_profile=objective_profile)
    promotion_gate = dict(protocol_summary.get("promotion_gate", {}) or {})
    return TrialResult(
        trial_id=trial_id,
        trial_tag=trial_tag,
        status="completed",
        phase="screening",
        role="",
        source_trial_tag="",
        trial_config=trial_config,
        protocol_summary_path=str(protocol_summary_path.resolve()),
        performance_score=float(score_payload["performance_score"]),
        stability_score=float(score_payload["stability_score"]),
        composite_score=float(score_payload["composite_score"]),
        score_breakdown=score_payload["score_breakdown"],
        primary_metrics=score_payload["primary_metrics"],
        promotion_status=str(promotion_gate.get("status", "")),
        failed_checks=list(promotion_gate.get("failed_checks", []) or []),
        gate_pass_ratio=float(score_payload["gate_pass_ratio"]),
        passed_check_count=int(score_payload["passed_check_count"]),
        total_check_count=int(score_payload["total_check_count"]),
    )


def _snapshot_latest_state() -> dict[str, str | None]:
    snapshot: dict[str, str | None] = {}
    for label, path in LATEST_STATE_PATHS.items():
        snapshot[label] = path.read_text(encoding="utf-8") if path.exists() else None
    return snapshot


def _restore_latest_state(snapshot: dict[str, str | None]) -> None:
    ensure_layout()
    for label, path in LATEST_STATE_PATHS.items():
        payload = snapshot.get(label)
        if payload is None:
            if path.exists():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")


def _read_seed_study_trials(
    seed_study_tag: str,
    *,
    objective_profile: str,
) -> tuple[dict[str, Any], list["TrialResult"]]:
    study_summary_path = STUDIES_ROOT / str(seed_study_tag) / "study_summary.json"
    study_summary = read_json(study_summary_path)
    if not study_summary:
        raise FileNotFoundError(f"Seed study summary not found: {study_summary_path}")
    seeded_trials: list[TrialResult] = []
    for index, item in enumerate(study_summary.get("trials", []) or [], start=1):
        protocol_summary_path = Path(str(item.get("protocol_summary_json", "") or "")).expanduser()
        if not protocol_summary_path.exists():
            continue
        seeded_result = _build_trial_result_from_protocol(
            trial_id=index,
            trial_tag=str(item.get("trial_tag", f"seed_trial_{index:02d}")),
            trial_config=dict(item.get("trial_config", {}) or {}),
            protocol_summary_path=protocol_summary_path,
            objective_profile=objective_profile,
        )
        seeded_trials.append(seeded_result)
    return study_summary, seeded_trials


def _historical_leaderboard(
    *,
    pool_name: str,
    trainer_backend: str,
    label_preset: str,
    objective_profile: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    leaderboard: list[dict[str, Any]] = []
    for summary_path in sorted(PROTOCOLS_ROOT.glob("*/protocol_summary.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        payload = read_json(summary_path)
        if not payload:
            continue
        if str(payload.get("pool_name", "") or "") != str(pool_name):
            continue
        if str(payload.get("trainer_backend", "") or "") != str(trainer_backend):
            continue
        if str(payload.get("label_preset", "") or "") != str(label_preset):
            continue
        score_payload = _score_protocol_summary(payload, objective_profile=objective_profile)
        leaderboard.append(
            {
                "run_tag": str(payload.get("run_tag", "")),
                "label_preset": str(payload.get("label_preset", "")),
                "decoder_profile": str(payload.get("decoder_profile", "")),
                "loss_profile": str(payload.get("loss_profile", "")),
                "execution_semantics": str(payload.get("execution_semantics", "")),
                "budget_semantics": str(payload.get("budget_semantics", "")),
                "budget_calibration": str(payload.get("budget_calibration", "")),
                "promotion_status": str(dict(payload.get("promotion_gate", {}) or {}).get("status", "")),
                "protocol_summary_json": str(summary_path.resolve()),
                "performance_score": score_payload["performance_score"],
                "stability_score": score_payload["stability_score"],
                "composite_score": score_payload["composite_score"],
                **score_payload["primary_metrics"],
            }
        )
    leaderboard.sort(key=lambda item: float(item.get("composite_score", 0.0)), reverse=True)
    return leaderboard[: max(int(limit), 0)]


def _build_protocol_args(args: argparse.Namespace, trial_config: dict[str, Any], protocol_tag: str) -> list[str]:
    protocol_args = [
        "--pool-name",
        str(args.pool_name),
        "--benchmark",
        str(args.benchmark),
        "--data-source",
        str(args.data_source),
        "--pool-rebalance-days",
        str(args.pool_rebalance_days),
        "--pool-adv-window",
        str(args.pool_adv_window),
        "--max-universe-size",
        str(args.max_universe_size),
        "--train-start-date",
        str(args.train_start_date),
        "--train-end-date",
        str(args.train_end_date),
        "--eval-start-date",
        str(args.eval_start_date),
        "--eval-end-date",
        str(args.eval_end_date),
        "--shadow-start-date",
        str(args.shadow_start_date),
        "--shadow-end-date",
        str(args.shadow_end_date),
        "--transaction-cost-bps",
        str(args.transaction_cost_bps),
        "--slippage-bps",
        str(args.slippage_bps),
        "--sell-tax-bps",
        str(args.sell_tax_bps),
        "--random-seed",
        str(args.random_seed),
        "--skip-multiplier",
        str(args.skip_multiplier),
        "--execution-semantics",
        str(args.execution_semantics),
        "--budget-semantics",
        str(trial_config.get("budget_semantics", args.budget_semantics)),
        "--budget-calibration",
        str(trial_config.get("budget_calibration", args.budget_calibration)),
        "--budget-objective",
        str(trial_config.get("budget_objective", args.budget_objective)),
        "--alpha-prior-source",
        str(trial_config.get("alpha_prior_source", args.alpha_prior_source)),
        "--alpha-prior-score-panel",
        str(trial_config.get("alpha_prior_score_panel", args.alpha_prior_score_panel)),
        "--alpha-prior-target-weight-panel",
        str(trial_config.get("alpha_prior_target_weight_panel", args.alpha_prior_target_weight_panel)),
        "--label-preset",
        str(trial_config["label_preset"]),
        "--trainer-backend",
        str(args.trainer_backend),
        "--decoder-profile",
        str(trial_config["decoder_profile"]),
        "--loss-profile",
        str(trial_config.get("loss_profile", DEFAULT_LOSS_PROFILE)),
        "--epochs",
        str(trial_config["epochs"]),
        "--min-epochs",
        str(trial_config["min_epochs"]),
        "--batch-size",
        str(trial_config["batch_size"]),
        "--learning-rate",
        str(trial_config["learning_rate"]),
        "--hidden-dim",
        str(trial_config["hidden_dim"]),
        "--sequence-layers",
        str(trial_config["sequence_layers"]),
        "--daily-hidden-dim",
        str(trial_config["daily_hidden_dim"]),
        "--daily-head-layout",
        str(trial_config.get("daily_head_layout", args.daily_head_layout)),
        "--dropout",
        str(trial_config["dropout"]),
        "--daily-dropout",
        str(trial_config["daily_dropout"]),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--resume-mode",
        str(args.resume_mode),
        "--tag",
        protocol_tag,
    ]
    if str(args.csv_folder or "").strip():
        protocol_args.extend(["--csv-folder", str(args.csv_folder)])
    if args.refresh_cache:
        protocol_args.append("--refresh-cache")
    return protocol_args


def _build_confirmatory_protocol_args(
    args: argparse.Namespace,
    trial_config: dict[str, Any],
    protocol_tag: str,
) -> list[str]:
    confirmatory_config = dict(trial_config)
    confirmatory_config["epochs"] = int(args.confirmatory_epochs)
    confirmatory_config["min_epochs"] = int(args.confirmatory_min_epochs)
    return _build_protocol_args(args, confirmatory_config, protocol_tag)


@dataclass
class TrialResult:
    trial_id: int
    trial_tag: str
    status: str
    phase: str
    role: str
    source_trial_tag: str
    trial_config: dict[str, Any]
    protocol_summary_path: str
    performance_score: float
    stability_score: float
    composite_score: float
    score_breakdown: dict[str, Any]
    primary_metrics: dict[str, Any]
    promotion_status: str
    failed_checks: list[str]
    gate_pass_ratio: float
    passed_check_count: int
    total_check_count: int
    error: str = ""

    def to_summary(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "trial_tag": self.trial_tag,
            "status": self.status,
            "phase": self.phase,
            "role": self.role,
            "source_trial_tag": self.source_trial_tag,
            "trial_config": self.trial_config,
            "protocol_summary_json": self.protocol_summary_path,
            "performance_score": self.performance_score,
            "stability_score": self.stability_score,
            "composite_score": self.composite_score,
            "score_breakdown": self.score_breakdown,
            "primary_metrics": self.primary_metrics,
            "promotion_status": self.promotion_status,
            "failed_checks": self.failed_checks,
            "gate_pass_ratio": self.gate_pass_ratio,
            "passed_check_count": self.passed_check_count,
            "total_check_count": self.total_check_count,
            "error": self.error,
        }


def _pick_confirmatory_candidates(
    completed_trials: list[TrialResult],
    *,
    base_sequence_layers: int,
    max_candidates: int,
) -> list[tuple[str, TrialResult]]:
    if not completed_trials or max_candidates <= 0:
        return []
    candidates: list[tuple[str, TrialResult]] = []
    performance_champion = max(completed_trials, key=lambda item: float(item.performance_score))
    candidates.append(("performance_champion", performance_champion))

    depth_trials = [
        item
        for item in completed_trials
        if int(item.trial_config.get("sequence_layers", base_sequence_layers) or base_sequence_layers) > int(base_sequence_layers)
    ]
    if depth_trials:
        depth_challenger = max(
            depth_trials,
            key=lambda item: (float(item.gate_pass_ratio), float(item.stability_score), float(item.composite_score)),
        )
        candidates.append(("depth_challenger", depth_challenger))

    stability_champion = max(completed_trials, key=lambda item: float(item.stability_score))
    candidates.append(("stability_champion", stability_champion))

    unique: list[tuple[str, TrialResult]] = []
    seen_tags: set[str] = set()
    for role, trial in candidates:
        if trial.trial_tag in seen_tags:
            continue
        seen_tags.add(trial.trial_tag)
        unique.append((role, trial))
        if len(unique) >= int(max_candidates):
            break
    if len(unique) < int(max_candidates):
        remaining = sorted(completed_trials, key=lambda item: float(item.composite_score), reverse=True)
        for trial in remaining:
            if trial.trial_tag in seen_tags:
                continue
            seen_tags.add(trial.trial_tag)
            unique.append(("composite_runner_up", trial))
            if len(unique) >= int(max_candidates):
                break
    return unique


def build_parser() -> argparse.ArgumentParser:
    latest_completed = _normalize_date_text(get_latest_completed_trading_date())
    parser = argparse.ArgumentParser(
        description="Run a bounded self-optimizing continuous-policy study on top of the formal protocol."
    )
    parser.add_argument("--pool-name", default="learned_all_a")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--data-source", default="tq", choices=("tq", "csv"))
    parser.add_argument("--csv-folder", default="")
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)
    parser.add_argument("--max-universe-size", type=int, default=1200)
    parser.add_argument("--train-start-date", default="20240102")
    parser.add_argument("--train-end-date", default="20251231")
    parser.add_argument("--eval-start-date", default="20260102")
    parser.add_argument("--eval-end-date", default=latest_completed)
    parser.add_argument("--shadow-start-date", default="")
    parser.add_argument("--shadow-end-date", default=latest_completed)
    parser.add_argument("--transaction-cost-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=7.0)
    parser.add_argument("--sell-tax-bps", type=float, default=10.0)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--skip-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--execution-semantics",
        default=DEFAULT_EXECUTION_SEMANTICS,
        choices=EXECUTION_SEMANTICS_CHOICES,
    )
    parser.add_argument(
        "--budget-semantics",
        default=DEFAULT_BUDGET_SEMANTICS,
        choices=BUDGET_SEMANTICS_CHOICES,
    )
    parser.add_argument(
        "--budget-calibration",
        default=DEFAULT_BUDGET_CALIBRATION,
        choices=BUDGET_CALIBRATION_CHOICES,
    )
    parser.add_argument("--budget-objective", default=DEFAULT_BUDGET_OBJECTIVE, choices=BUDGET_OBJECTIVE_CHOICES)
    parser.add_argument("--alpha-prior-source", default="none")
    parser.add_argument("--alpha-prior-score-panel", default="")
    parser.add_argument("--alpha-prior-target-weight-panel", default="")
    parser.add_argument("--trainer-backend", default=TRAINER_BACKEND_FORMAL_SEQ_V3, choices=TRAINER_BACKENDS)
    parser.add_argument("--resume-mode", default="strict", choices=("strict", "fresh"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=32)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--daily-head-layout", default=DAILY_HEAD_LAYOUT_MONOLITHIC_V1, choices=DAILY_HEAD_LAYOUT_CHOICES)
    parser.add_argument("--search-profile", default="focused_seq_v1", choices=tuple(sorted(SEARCH_PROFILES)))
    parser.add_argument("--objective-profile", default="")
    parser.add_argument("--trial-count", type=int, default=3)
    parser.add_argument("--seed-study-tag", default="")
    parser.add_argument("--skip-screening", action="store_true")
    parser.add_argument("--disable-confirmatory", action="store_true")
    parser.add_argument("--confirmatory-max-candidates", type=int, default=2)
    parser.add_argument("--confirmatory-epochs", type=int, default=64)
    parser.add_argument("--confirmatory-min-epochs", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--study-tag", default="")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_layout()

    study_tag = str(args.study_tag or timestamp_tag("self_opt_study"))
    study_root = STUDIES_ROOT / study_tag
    study_root.mkdir(parents=True, exist_ok=True)
    objective_profile = str(
        args.objective_profile
        or SEARCH_PROFILE_DEFAULT_OBJECTIVES.get(args.search_profile, "promotion_balanced_v2")
    )

    base_trial = {
        **dict(SEARCH_PROFILE_BASE_TRIALS[args.search_profile]),
        "epochs": int(args.epochs),
        "min_epochs": int(args.min_epochs),
    }
    selected_trials = _select_trials(
        base=base_trial,
        profile_name=args.search_profile,
        trial_count=args.trial_count,
        random_seed=args.random_seed,
    )
    seed_study_summary: dict[str, Any] = {}
    screening_seed_trials: list[TrialResult] = []
    if str(args.seed_study_tag or "").strip():
        seed_study_summary, screening_seed_trials = _read_seed_study_trials(
            str(args.seed_study_tag).strip(),
            objective_profile=objective_profile,
        )
        if screening_seed_trials:
            selected_trials = [dict(item.trial_config) for item in screening_seed_trials]
    study_plan = {
        "run_tag": study_tag,
        "created_at": now_iso(),
        "search_profile": args.search_profile,
        "objective_profile": objective_profile,
        "seed_study_tag": str(args.seed_study_tag or ""),
        "skip_screening": bool(args.skip_screening),
        "confirmatory_enabled": not bool(args.disable_confirmatory),
        "confirmatory_max_candidates": int(args.confirmatory_max_candidates),
        "confirmatory_epochs": int(args.confirmatory_epochs),
        "confirmatory_min_epochs": int(args.confirmatory_min_epochs),
        "execution_semantics": str(args.execution_semantics),
        "budget_semantics": str(args.budget_semantics),
        "budget_calibration": str(args.budget_calibration),
        "budget_objective": str(args.budget_objective),
        "daily_head_layout": str(args.daily_head_layout),
        "alpha_prior_source": str(args.alpha_prior_source),
        "alpha_prior_score_panel": str(args.alpha_prior_score_panel),
        "alpha_prior_target_weight_panel": str(args.alpha_prior_target_weight_panel),
        "trial_count": len(selected_trials),
        "base_trial": base_trial,
        "selected_trials": selected_trials,
    }
    write_json(study_root / "study_plan.json", study_plan)
    if args.dry_run:
        print(json.dumps(study_plan, ensure_ascii=False, indent=2))
        return 0

    latest_snapshot = _snapshot_latest_state()
    screening_results: list[TrialResult] = []
    confirmatory_results: list[TrialResult] = []
    try:
        if args.skip_screening and screening_seed_trials:
            screening_results = list(screening_seed_trials)
        else:
            for index, trial_config in enumerate(selected_trials, start=1):
                trial_tag = f"{study_tag}__trial_{index:02d}"
                protocol_args = _build_protocol_args(args, trial_config, trial_tag)
                trial_error = ""
                try:
                    exit_code = int(protocol_main(protocol_args))
                    if exit_code != 0:
                        raise RuntimeError(f"protocol exited with code {exit_code}")
                    protocol_summary_path = PROTOCOLS_ROOT / trial_tag / "protocol_summary.json"
                    screening_results.append(
                        _build_trial_result_from_protocol(
                            trial_id=index,
                            trial_tag=trial_tag,
                            trial_config=trial_config,
                            protocol_summary_path=protocol_summary_path,
                            objective_profile=objective_profile,
                        )
                    )
                except Exception as exc:
                    trial_error = str(exc)
                    screening_results.append(
                        TrialResult(
                            trial_id=index,
                            trial_tag=trial_tag,
                            status="failed",
                            phase="screening",
                            role="",
                            source_trial_tag="",
                            trial_config=trial_config,
                            protocol_summary_path=str((PROTOCOLS_ROOT / trial_tag / "protocol_summary.json").resolve()),
                            performance_score=-999.0,
                            stability_score=-999.0,
                            composite_score=-999.0,
                            score_breakdown={},
                            primary_metrics={},
                            promotion_status="failed",
                            failed_checks=["trial_execution_failed"],
                            gate_pass_ratio=0.0,
                            passed_check_count=0,
                            total_check_count=0,
                            error=trial_error,
                        )
                    )

        screening_results.sort(key=lambda item: float(item.composite_score), reverse=True)
        completed_screening = [item for item in screening_results if item.status == "completed"]
        if not args.disable_confirmatory:
            confirmatory_candidates = _pick_confirmatory_candidates(
                completed_screening,
                base_sequence_layers=int(base_trial.get("sequence_layers", 1)),
                max_candidates=int(args.confirmatory_max_candidates),
            )
            for index, (role, source_trial) in enumerate(confirmatory_candidates, start=1):
                confirm_tag = f"{study_tag}__confirm_{index:02d}"
                protocol_args = _build_confirmatory_protocol_args(args, source_trial.trial_config, confirm_tag)
                trial_error = ""
                try:
                    exit_code = int(protocol_main(protocol_args))
                    if exit_code != 0:
                        raise RuntimeError(f"protocol exited with code {exit_code}")
                    protocol_summary_path = PROTOCOLS_ROOT / confirm_tag / "protocol_summary.json"
                    built = _build_trial_result_from_protocol(
                        trial_id=index,
                        trial_tag=confirm_tag,
                        trial_config={
                            **source_trial.trial_config,
                            "epochs": int(args.confirmatory_epochs),
                            "min_epochs": int(args.confirmatory_min_epochs),
                        },
                        protocol_summary_path=protocol_summary_path,
                        objective_profile=objective_profile,
                    )
                    built.phase = "confirmatory"
                    built.role = role
                    built.source_trial_tag = source_trial.trial_tag
                    confirmatory_results.append(built)
                except Exception as exc:
                    trial_error = str(exc)
                    confirmatory_results.append(
                        TrialResult(
                            trial_id=index,
                            trial_tag=confirm_tag,
                            status="failed",
                            phase="confirmatory",
                            role=role,
                            source_trial_tag=source_trial.trial_tag,
                            trial_config={
                                **source_trial.trial_config,
                                "epochs": int(args.confirmatory_epochs),
                                "min_epochs": int(args.confirmatory_min_epochs),
                            },
                            protocol_summary_path=str((PROTOCOLS_ROOT / confirm_tag / "protocol_summary.json").resolve()),
                            performance_score=-999.0,
                            stability_score=-999.0,
                            composite_score=-999.0,
                            score_breakdown={},
                            primary_metrics={},
                            promotion_status="failed",
                            failed_checks=["trial_execution_failed"],
                            gate_pass_ratio=0.0,
                            passed_check_count=0,
                            total_check_count=0,
                            error=trial_error,
                        )
                    )
        confirmatory_results.sort(key=lambda item: float(item.composite_score), reverse=True)
    finally:
        _restore_latest_state(latest_snapshot)

    ranking_rows = [item.to_summary() for item in [*screening_results, *confirmatory_results]]
    pd.DataFrame(
        [
            {
                "trial_id": row["trial_id"],
                "trial_tag": row["trial_tag"],
                "status": row["status"],
                "phase": row["phase"],
                "role": row["role"],
                "source_trial_tag": row["source_trial_tag"],
                "budget_semantics": str(dict(row.get("trial_config", {}) or {}).get("budget_semantics", args.budget_semantics)),
                "budget_calibration": str(dict(row.get("trial_config", {}) or {}).get("budget_calibration", args.budget_calibration)),
                "budget_objective": str(dict(row.get("trial_config", {}) or {}).get("budget_objective", args.budget_objective)),
                "daily_head_layout": str(dict(row.get("trial_config", {}) or {}).get("daily_head_layout", args.daily_head_layout)),
                "alpha_prior_source": str(dict(row.get("trial_config", {}) or {}).get("alpha_prior_source", args.alpha_prior_source)),
                "performance_score": row["performance_score"],
                "stability_score": row["stability_score"],
                "composite_score": row["composite_score"],
                "promotion_status": row["promotion_status"],
                "failed_checks": ",".join(row["failed_checks"]),
                **row["primary_metrics"],
            }
            for row in ranking_rows
        ]
    ).to_csv(study_root / "trial_ranking.csv", index=False, encoding="utf-8-sig")

    completed_screening = [item for item in screening_results if item.status == "completed"]
    completed_confirmatory = [item for item in confirmatory_results if item.status == "completed"]
    screen_performance_champion = (
        max(completed_screening, key=lambda item: float(item.performance_score)).to_summary()
        if completed_screening
        else {}
    )
    screen_stability_champion = (
        max(completed_screening, key=lambda item: float(item.stability_score)).to_summary()
        if completed_screening
        else {}
    )
    screen_depth_trials = [
        item for item in completed_screening if int(item.trial_config.get("sequence_layers", 1) or 1) > int(base_trial.get("sequence_layers", 1))
    ]
    screen_depth_challenger = (
        max(screen_depth_trials, key=lambda item: (float(item.gate_pass_ratio), float(item.stability_score), float(item.composite_score))).to_summary()
        if screen_depth_trials
        else {}
    )
    champion = (
        completed_confirmatory[0].to_summary()
        if completed_confirmatory
        else (completed_screening[0].to_summary() if completed_screening else {})
    )
    historical = _historical_leaderboard(
        pool_name=args.pool_name,
        trainer_backend=args.trainer_backend,
        label_preset=str(base_trial["label_preset"]),
        objective_profile=objective_profile,
        limit=max(args.top_k, 5),
    )
    study_summary = {
        "run_tag": study_tag,
        "study_tag": study_tag,
        "executed_at": now_iso(),
        "objective_profile": objective_profile,
        "search_profile": args.search_profile,
        "pool_name": args.pool_name,
        "benchmark": args.benchmark,
        "trainer_backend": args.trainer_backend,
        "execution_semantics": str(args.execution_semantics),
        "budget_semantics": str(args.budget_semantics),
        "budget_calibration": str(args.budget_calibration),
        "study_plan_json": str((study_root / "study_plan.json").resolve()),
        "trial_ranking_csv": str((study_root / "trial_ranking.csv").resolve()),
        "trial_count": len(selected_trials),
        "completed_trial_count": len(completed_screening),
        "failed_trial_count": len([item for item in screening_results if item.status != "completed"]),
        "screening_trials": [item.to_summary() for item in screening_results],
        "confirmatory_trials": [item.to_summary() for item in confirmatory_results],
        "trials": ranking_rows,
        "screen_performance_champion": screen_performance_champion,
        "screen_stability_champion": screen_stability_champion,
        "screen_depth_challenger": screen_depth_challenger,
        "champion": champion,
        "historical_leaderboard": historical,
        "latest_state_restored": True,
        "seed_study_tag": str(args.seed_study_tag or ""),
        "seed_study_summary": {
            "run_tag": str(seed_study_summary.get("run_tag", "")),
            "study_tag": str(seed_study_summary.get("study_tag", seed_study_summary.get("run_tag", ""))),
            "study_summary_json": str((STUDIES_ROOT / str(args.seed_study_tag) / "study_summary.json").resolve()) if str(args.seed_study_tag or "").strip() else "",
        },
        "confirmatory_enabled": not bool(args.disable_confirmatory),
        "confirmatory_completed_trial_count": len(completed_confirmatory),
        "restored_to_protocol_run_tag": str(read_json(LATEST_PROTOCOL_SUMMARY_PATH).get("run_tag", "")),
    }
    write_json(study_root / "study_summary.json", study_summary)
    update_latest_summary("study", study_summary)
    print(json.dumps(study_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
