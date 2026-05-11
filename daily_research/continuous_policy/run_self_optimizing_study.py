from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.baseline.data_provider import get_latest_completed_trading_date
from daily_research.continuous_policy.model_seq_v3 import (
    CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_SOLVER_ENABLED,
    DAILY_HEAD_LAYOUT_CHOICES,
    DAILY_HEAD_LAYOUT_MONOLITHIC_V1,
    DEFAULT_LOSS_PROFILE,
    _loss_profile_enables_full_universe_train_solver,
    resolve_loss_profile,
)
from daily_research.continuous_policy.pipeline_utils import (
    BUDGET_OBJECTIVE_CHOICES,
    DEFAULT_BUDGET_OBJECTIVE,
)
from daily_research.continuous_policy.portfolio_simulator import (
    BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
    BUDGET_CALIBRATION_CHOICES,
    BUDGET_SEMANTICS_ALLOCATION_LAYER,
    BUDGET_SEMANTICS_CHOICES,
    DEFAULT_BUDGET_CALIBRATION,
    DEFAULT_BUDGET_SEMANTICS,
    DEFAULT_EXECUTION_SEMANTICS,
    EXECUTION_SEMANTICS_CHOICES,
)
from daily_research.continuous_policy.run_continuous_policy_protocol import (
    PROMOTION_THRESHOLDS,
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
    safe_print_json,
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

RUNTIME_CHANNEL_FAILURE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("cuda_unavailable", "requires CUDA, but torch.cuda.is_available() is False"),
    ("cuda_init_unknown_error", "CUDA initialization: CUDA unknown error"),
    ("cuda_no_gpu_available", "No CUDA GPUs are available"),
    ("openmp_duplicate_runtime", "OMP: Error #15"),
    ("openmp_duplicate_runtime", "libiomp5md.dll already initialized"),
)

RESOURCE_PROFILE_CHOICES: tuple[str, ...] = ("auto", "safe", "balanced", "full")
RESOURCE_PRIORITY_CHOICES: tuple[str, ...] = ("auto", "normal", "below_normal", "idle")
THREAD_LIMIT_ENV_KEYS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
)
TRUE_SOLVER_RESOURCE_SEARCH_PROFILES = frozenset(
    {
        "split_heads_portfolio_daily_true_convex_solver_allocation_r47",
        "split_heads_portfolio_daily_integrated_convex_capital_flow_r50",
    }
)


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
    "split_heads_deploy_executability_r10": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v8", "alpha_result_value_budget_split_v9"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_intent_guard_v5", "cash_constraint_deploy_guard_v6"],
        "budget_objective": ["result_value_v8", "result_value_v9"],
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
    "split_heads_sell_source_contract_r11": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v9", "alpha_result_value_budget_split_v10"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_sell_source_guard_v7"],
        "budget_objective": ["result_value_v9", "result_value_v10"],
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
    "split_heads_sell_source_contract_r11b": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v10", "alpha_result_value_budget_split_v11"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_sell_source_guard_v7"],
        "budget_objective": ["result_value_v9", "result_value_v10"],
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
    "split_heads_release_translation_deploy_r12": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v11", "alpha_result_value_budget_split_v12"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_sell_source_guard_v7"],
        "budget_objective": ["result_value_v9", "result_value_v10"],
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
    "split_heads_action_value_unification_r13": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v12", "alpha_result_value_budget_split_v13"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_sell_source_guard_v7"],
        "budget_objective": ["result_value_v9", "result_value_v10"],
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
    "split_heads_direct_action_value_r14": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v13", "alpha_result_value_budget_split_v14"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_sell_source_guard_v7"],
        "budget_objective": ["result_value_v9", "result_value_v10"],
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
    "split_heads_direct_action_translation_r15": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v14", "alpha_result_value_budget_split_v15"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_direct_action_guard_v8"],
        "budget_objective": ["result_value_v9", "result_value_v10"],
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
    "split_heads_direct_action_reallocation_r16": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v14", "alpha_result_value_budget_split_v15"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_direct_action_reallocation_guard_v9"],
        "budget_objective": ["result_value_v9", "result_value_v10"],
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
    "split_heads_direct_action_pair_reallocation_r17": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v14", "alpha_result_value_budget_split_v15"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_direct_action_pair_reallocation_guard_v10"],
        "budget_objective": ["result_value_v9", "result_value_v10"],
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
    "split_heads_direct_action_pair_cost_guard_r18": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v14", "alpha_result_value_budget_split_v15"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_direct_action_pair_cost_guard_v11"],
        "budget_objective": ["result_value_v9", "result_value_v10"],
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
    "split_heads_portfolio_daily_ranking_r19": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v14", "alpha_result_value_budget_split_v15"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_cash_aware_guard_v13"],
        "budget_objective": ["result_value_v9", "result_value_v10"],
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
    "split_heads_portfolio_daily_ranking_stability_r20": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v14", "alpha_result_value_budget_split_v15"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_cash_aware_guard_v13"],
        "budget_objective": ["result_value_v9", "result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [8.0e-4, 1.0e-3, 1.2e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.12, 0.16],
        "daily_dropout": [0.08, 0.10],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_ranking_source_exec_r21": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v14", "alpha_result_value_budget_split_v15"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_source_exec_guard_v14"],
        "budget_objective": ["result_value_v9", "result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [8.0e-4, 1.0e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.16, 0.18],
        "daily_dropout": [0.10, 0.12],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_ranking_receiver_exec_r22": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v15"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"],
        "budget_objective": ["result_value_v9", "result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [8.0e-4, 1.0e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.16, 0.18],
        "daily_dropout": [0.10, 0.12],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_ranking_receiver_exec_stability_r23": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v15"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"],
        "budget_objective": ["result_value_v9"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [5.0e-4, 6.5e-4, 8.0e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.18, 0.20, 0.22],
        "daily_dropout": [0.12, 0.14],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_listwise_allocation_r24": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v16"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"],
        "budget_objective": ["result_value_v9"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [6.5e-4, 8.0e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.18, 0.20],
        "daily_dropout": [0.12, 0.14],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_source_release_listwise_r25": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v17"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"],
        "budget_objective": ["result_value_v9", "result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [6.5e-4, 8.0e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.18, 0.20],
        "daily_dropout": [0.12, 0.14],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_allocation_teacher_r26": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v18"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"],
        "budget_objective": ["result_value_v9"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [5.0e-4, 6.5e-4, 8.0e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.18, 0.20, 0.22],
        "daily_dropout": [0.12, 0.14],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_economic_release_r27": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v19"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [6.5e-4, 8.0e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.18, 0.20],
        "daily_dropout": [0.12, 0.14],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_source_brake_r28": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v20"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [6.5e-4, 8.0e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.18, 0.20],
        "daily_dropout": [0.12, 0.14],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_source_brake_relief_r29": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v20"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [6.5e-4, 8.0e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.18, 0.20],
        "daily_dropout": [0.12, 0.14],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_source_brake_cash_relief_r30": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v20"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [6.5e-4, 8.0e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.18, 0.20],
        "daily_dropout": [0.12, 0.14],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_receiver_semantic_closure_r31": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v20"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [8.0e-4, 1.0e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.16, 0.18],
        "daily_dropout": [0.12, 0.14],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_source_distribution_quality_r32": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v20"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [6.5e-4, 8.0e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.18, 0.20],
        "daily_dropout": [0.14, 0.16],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_source_forward_proxy_r33": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v20"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [6.5e-4, 8.0e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.18, 0.20],
        "daily_dropout": [0.14, 0.16],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_allocation_breadth_r34": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v20"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"],
        "budget_objective": ["result_value_v9", "result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [8.0e-4, 1.0e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.16, 0.18],
        "daily_dropout": [0.12, 0.14],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_unified_allocation_r35": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v21"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [8.0e-4, 1.0e-3],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.16, 0.18],
        "daily_dropout": [0.12, 0.14],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_risk_aware_unified_allocation_r36": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v22"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [6.5e-4, 8.0e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.18, 0.20],
        "daily_dropout": [0.14, 0.16],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_decision_focused_allocation_r37": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v23"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [6.5e-4, 8.0e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.18, 0.20],
        "daily_dropout": [0.14, 0.16],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_source_hard_negative_regret_r38": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v24"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [6.0e-4, 6.5e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.18, 0.20],
        "daily_dropout": [0.14, 0.16],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_allocation_objective_consolidation_r39": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v25"],
        "budget_semantics": ["action_budget_split_v1"],
        "budget_calibration": ["cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [6.0e-4, 6.5e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.18, 0.20],
        "daily_dropout": [0.14, 0.16],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_end_to_end_allocation_layer_r40": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v25"],
        "budget_semantics": [BUDGET_SEMANTICS_ALLOCATION_LAYER],
        "budget_calibration": [BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [6.0e-4, 6.5e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.18, 0.20],
        "daily_dropout": [0.14, 0.16],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_risk_sensitive_allocation_layer_r41": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v26"],
        "budget_semantics": [BUDGET_SEMANTICS_ALLOCATION_LAYER],
        "budget_calibration": [BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [5.5e-4, 6.0e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.20, 0.22],
        "daily_dropout": [0.16],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_utility_credit_allocation_r42": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v27"],
        "budget_semantics": [BUDGET_SEMANTICS_ALLOCATION_LAYER],
        "budget_calibration": [BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [5.2e-4, 5.8e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.20, 0.22],
        "daily_dropout": [0.16],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_primal_dual_decision_allocation_r43": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v28"],
        "budget_semantics": [BUDGET_SEMANTICS_ALLOCATION_LAYER],
        "budget_calibration": [BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [4.8e-4, 5.2e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.22, 0.24],
        "daily_dropout": [0.16],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_entropic_transport_allocation_r44": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v29"],
        "budget_semantics": [BUDGET_SEMANTICS_ALLOCATION_LAYER],
        "budget_calibration": [BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [4.4e-4, 4.8e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.24, 0.26],
        "daily_dropout": [0.16],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_conservative_transport_allocation_r45": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v30"],
        "budget_semantics": [BUDGET_SEMANTICS_ALLOCATION_LAYER],
        "budget_calibration": [BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [4.0e-4, 4.4e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.26, 0.28],
        "daily_dropout": [0.16],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_differentiable_convex_allocation_r46": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v31"],
        "budget_semantics": [BUDGET_SEMANTICS_ALLOCATION_LAYER],
        "budget_calibration": [BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [3.6e-4, 4.0e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.26, 0.30],
        "daily_dropout": [0.16],
        "batch_size": [512],
    },
    "split_heads_portfolio_daily_true_convex_solver_allocation_r47": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v32"],
        "budget_semantics": [BUDGET_SEMANTICS_ALLOCATION_LAYER],
        "budget_calibration": [BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [3.0e-4, 3.4e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.28, 0.32],
        "daily_dropout": [0.16],
        "batch_size": [384],
    },
    "split_heads_portfolio_daily_full_universe_convex_ope_allocation_r48": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v33"],
        "budget_semantics": [BUDGET_SEMANTICS_ALLOCATION_LAYER],
        "budget_calibration": [BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [2.6e-4, 3.0e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.30, 0.34],
        "daily_dropout": [0.16],
        "batch_size": [256],
    },
    "split_heads_portfolio_daily_capital_flow_closure_r49": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v34"],
        "budget_semantics": [BUDGET_SEMANTICS_ALLOCATION_LAYER],
        "budget_calibration": [BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [2.2e-4, 2.6e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.30, 0.34],
        "daily_dropout": [0.16],
        "batch_size": [256],
    },
    "split_heads_portfolio_daily_integrated_convex_capital_flow_r50": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v35"],
        "budget_semantics": [BUDGET_SEMANTICS_ALLOCATION_LAYER],
        "budget_calibration": [BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [2.0e-4, 2.4e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.32, 0.36],
        "daily_dropout": [0.16],
        "batch_size": [192],
    },
    "split_heads_portfolio_daily_native_allocation_vector_r51": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v36"],
        "budget_semantics": [BUDGET_SEMANTICS_ALLOCATION_LAYER],
        "budget_calibration": [BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [2.0e-4, 2.4e-4],
        "hidden_dim": [224],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.30, 0.34],
        "daily_dropout": [0.16],
        "batch_size": [256],
    },
    "split_heads_portfolio_daily_day_set_native_allocation_vector_r52": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v37"],
        "budget_semantics": [BUDGET_SEMANTICS_ALLOCATION_LAYER],
        "budget_calibration": [BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [1.6e-4, 2.0e-4],
        "hidden_dim": [192],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.28, 0.32],
        "daily_dropout": [0.14],
        "batch_size": [1, 2],
    },
    "split_heads_portfolio_daily_day_set_native_target_validity_closure_r52b": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v38"],
        "budget_semantics": [BUDGET_SEMANTICS_ALLOCATION_LAYER],
        "budget_calibration": [BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [1.4e-4, 1.8e-4],
        "hidden_dim": [192],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.28, 0.32],
        "daily_dropout": [0.14],
        "batch_size": [1, 2],
    },
    "split_heads_portfolio_daily_day_set_native_executable_receiver_closure_r52c": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v39"],
        "budget_semantics": [BUDGET_SEMANTICS_ALLOCATION_LAYER],
        "budget_calibration": [BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [1.4e-4, 1.8e-4],
        "hidden_dim": [192],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.28, 0.32],
        "daily_dropout": [0.14],
        "batch_size": [1, 2],
    },
    "split_heads_portfolio_daily_day_set_native_validation_closure_r52d": {
        "label_preset": ["holdcash_v3"],
        "decoder_profile": ["budget_v3"],
        "loss_profile": ["alpha_result_value_budget_split_v40"],
        "budget_semantics": [BUDGET_SEMANTICS_ALLOCATION_LAYER],
        "budget_calibration": [BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER],
        "budget_objective": ["result_value_v10"],
        "alpha_prior_source": ["active_execution_strategy"],
        "daily_head_layout": ["split_v2"],
        "learning_rate": [1.2e-4, 1.6e-4],
        "hidden_dim": [192],
        "sequence_layers": [2],
        "daily_hidden_dim": [128],
        "dropout": [0.28, 0.32],
        "daily_dropout": [0.14],
        "batch_size": [1, 2],
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
    "split_heads_deploy_executability_r10": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v9",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_deploy_guard_v6",
        "budget_objective": "result_value_v9",
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
    "split_heads_sell_source_contract_r11": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v10",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_sell_source_guard_v7",
        "budget_objective": "result_value_v10",
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
    "split_heads_sell_source_contract_r11b": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v11",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_sell_source_guard_v7",
        "budget_objective": "result_value_v9",
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
    "split_heads_release_translation_deploy_r12": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v12",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_sell_source_guard_v7",
        "budget_objective": "result_value_v9",
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
    "split_heads_action_value_unification_r13": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v13",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_sell_source_guard_v7",
        "budget_objective": "result_value_v9",
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
    "split_heads_direct_action_value_r14": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v14",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_sell_source_guard_v7",
        "budget_objective": "result_value_v10",
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
    "split_heads_direct_action_translation_r15": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v15",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_direct_action_guard_v8",
        "budget_objective": "result_value_v9",
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
    "split_heads_direct_action_reallocation_r16": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v15",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_direct_action_reallocation_guard_v9",
        "budget_objective": "result_value_v9",
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
    "split_heads_direct_action_pair_reallocation_r17": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v15",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_direct_action_pair_reallocation_guard_v10",
        "budget_objective": "result_value_v9",
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
    "split_heads_direct_action_pair_cost_guard_r18": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v15",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_direct_action_pair_cost_guard_v11",
        "budget_objective": "result_value_v9",
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
    "split_heads_portfolio_daily_ranking_r19": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v15",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_cash_aware_guard_v13",
        "budget_objective": "result_value_v9",
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
    "split_heads_portfolio_daily_ranking_stability_r20": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v15",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_cash_aware_guard_v13",
        "budget_objective": "result_value_v9",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 1.0e-3,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.16,
        "daily_dropout": 0.10,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_ranking_source_exec_r21": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v15",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_source_exec_guard_v14",
        "budget_objective": "result_value_v9",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 8.0e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.18,
        "daily_dropout": 0.12,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_ranking_receiver_exec_r22": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v15",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15",
        "budget_objective": "result_value_v9",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 8.0e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.18,
        "daily_dropout": 0.12,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_ranking_receiver_exec_stability_r23": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v15",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15",
        "budget_objective": "result_value_v9",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 6.5e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.20,
        "daily_dropout": 0.14,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_listwise_allocation_r24": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v16",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15",
        "budget_objective": "result_value_v9",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 6.5e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.20,
        "daily_dropout": 0.14,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_source_release_listwise_r25": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v17",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15",
        "budget_objective": "result_value_v9",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 6.5e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.20,
        "daily_dropout": 0.14,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_allocation_teacher_r26": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v18",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15",
        "budget_objective": "result_value_v9",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 6.5e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.20,
        "daily_dropout": 0.14,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_economic_release_r27": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v19",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15",
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 6.5e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.20,
        "daily_dropout": 0.14,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_source_brake_r28": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v20",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15",
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 6.5e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.20,
        "daily_dropout": 0.14,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_source_brake_relief_r29": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v20",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15",
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 6.5e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.20,
        "daily_dropout": 0.14,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_source_brake_cash_relief_r30": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v20",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15",
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 6.5e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.20,
        "daily_dropout": 0.14,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_receiver_semantic_closure_r31": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v20",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15",
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 8.0e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.18,
        "daily_dropout": 0.14,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_source_distribution_quality_r32": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v20",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15",
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 6.5e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.20,
        "daily_dropout": 0.16,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_source_forward_proxy_r33": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v20",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15",
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 6.5e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.20,
        "daily_dropout": 0.16,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_allocation_breadth_r34": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v20",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15",
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 8.0e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.18,
        "daily_dropout": 0.14,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_unified_allocation_r35": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v21",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15",
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 8.0e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.18,
        "daily_dropout": 0.14,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_risk_aware_unified_allocation_r36": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v22",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15",
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 6.5e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.20,
        "daily_dropout": 0.16,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_decision_focused_allocation_r37": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v23",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15",
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 6.5e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.20,
        "daily_dropout": 0.16,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_source_hard_negative_regret_r38": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v24",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15",
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 6.0e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.20,
        "daily_dropout": 0.16,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_allocation_objective_consolidation_r39": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v25",
        "budget_semantics": "action_budget_split_v1",
        "budget_calibration": "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15",
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 6.0e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.20,
        "daily_dropout": 0.16,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_end_to_end_allocation_layer_r40": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v25",
        "budget_semantics": BUDGET_SEMANTICS_ALLOCATION_LAYER,
        "budget_calibration": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 6.0e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.20,
        "daily_dropout": 0.16,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_risk_sensitive_allocation_layer_r41": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v26",
        "budget_semantics": BUDGET_SEMANTICS_ALLOCATION_LAYER,
        "budget_calibration": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 5.5e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.22,
        "daily_dropout": 0.16,
        "batch_size": 512,
    },
    "split_heads_portfolio_daily_utility_credit_allocation_r42": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v27",
        "budget_semantics": BUDGET_SEMANTICS_ALLOCATION_LAYER,
        "budget_calibration": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 5.2e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.20,
        "daily_dropout": 0.16,
        "batch_size": 512,
        "epochs": 24,
        "min_epochs": 16,
    },
    "split_heads_portfolio_daily_primal_dual_decision_allocation_r43": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v28",
        "budget_semantics": BUDGET_SEMANTICS_ALLOCATION_LAYER,
        "budget_calibration": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 4.8e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.22,
        "daily_dropout": 0.16,
        "batch_size": 512,
        "epochs": 20,
        "min_epochs": 14,
    },
    "split_heads_portfolio_daily_entropic_transport_allocation_r44": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v29",
        "budget_semantics": BUDGET_SEMANTICS_ALLOCATION_LAYER,
        "budget_calibration": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 4.4e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.24,
        "daily_dropout": 0.16,
        "batch_size": 512,
        "epochs": 18,
        "min_epochs": 12,
    },
    "split_heads_portfolio_daily_conservative_transport_allocation_r45": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v30",
        "budget_semantics": BUDGET_SEMANTICS_ALLOCATION_LAYER,
        "budget_calibration": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 4.0e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.26,
        "daily_dropout": 0.16,
        "batch_size": 512,
        "epochs": 16,
        "min_epochs": 10,
    },
    "split_heads_portfolio_daily_differentiable_convex_allocation_r46": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v31",
        "budget_semantics": BUDGET_SEMANTICS_ALLOCATION_LAYER,
        "budget_calibration": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 3.6e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.28,
        "daily_dropout": 0.16,
        "batch_size": 512,
        "epochs": 14,
        "min_epochs": 9,
    },
    "split_heads_portfolio_daily_true_convex_solver_allocation_r47": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v32",
        "budget_semantics": BUDGET_SEMANTICS_ALLOCATION_LAYER,
        "budget_calibration": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 3.0e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.30,
        "daily_dropout": 0.16,
        "batch_size": 384,
        "epochs": 10,
        "min_epochs": 7,
    },
    "split_heads_portfolio_daily_full_universe_convex_ope_allocation_r48": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v33",
        "budget_semantics": BUDGET_SEMANTICS_ALLOCATION_LAYER,
        "budget_calibration": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 2.6e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.32,
        "daily_dropout": 0.16,
        "batch_size": 256,
        "epochs": 8,
        "min_epochs": 6,
    },
    "split_heads_portfolio_daily_capital_flow_closure_r49": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v34",
        "budget_semantics": BUDGET_SEMANTICS_ALLOCATION_LAYER,
        "budget_calibration": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 2.4e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.32,
        "daily_dropout": 0.16,
        "batch_size": 256,
        "epochs": 8,
        "min_epochs": 6,
    },
    "split_heads_portfolio_daily_integrated_convex_capital_flow_r50": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v35",
        "budget_semantics": BUDGET_SEMANTICS_ALLOCATION_LAYER,
        "budget_calibration": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 2.2e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.34,
        "daily_dropout": 0.16,
        "batch_size": 192,
        "epochs": 6,
        "min_epochs": 5,
    },
    "split_heads_portfolio_daily_native_allocation_vector_r51": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v36",
        "budget_semantics": BUDGET_SEMANTICS_ALLOCATION_LAYER,
        "budget_calibration": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 2.0e-4,
        "hidden_dim": 224,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.30,
        "daily_dropout": 0.16,
        "batch_size": 256,
        "epochs": 8,
        "min_epochs": 6,
    },
    "split_heads_portfolio_daily_day_set_native_allocation_vector_r52": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v37",
        "budget_semantics": BUDGET_SEMANTICS_ALLOCATION_LAYER,
        "budget_calibration": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 1.6e-4,
        "hidden_dim": 192,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.28,
        "daily_dropout": 0.14,
        "batch_size": 1,
        "epochs": 6,
        "min_epochs": 4,
    },
    "split_heads_portfolio_daily_day_set_native_target_validity_closure_r52b": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v38",
        "budget_semantics": BUDGET_SEMANTICS_ALLOCATION_LAYER,
        "budget_calibration": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 1.4e-4,
        "hidden_dim": 192,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.28,
        "daily_dropout": 0.14,
        "batch_size": 1,
        "epochs": 6,
        "min_epochs": 4,
    },
    "split_heads_portfolio_daily_day_set_native_executable_receiver_closure_r52c": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v39",
        "budget_semantics": BUDGET_SEMANTICS_ALLOCATION_LAYER,
        "budget_calibration": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 1.4e-4,
        "hidden_dim": 192,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.28,
        "daily_dropout": 0.14,
        "batch_size": 1,
        "epochs": 6,
        "min_epochs": 4,
    },
    "split_heads_portfolio_daily_day_set_native_validation_closure_r52d": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v40",
        "budget_semantics": BUDGET_SEMANTICS_ALLOCATION_LAYER,
        "budget_calibration": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 1.2e-4,
        "hidden_dim": 192,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.28,
        "daily_dropout": 0.14,
        "batch_size": 1,
        "epochs": 6,
        "min_epochs": 4,
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
    "split_heads_deploy_executability_r10": "deploy_executability_v1",
    "split_heads_sell_source_contract_r11": "sell_source_contract_v1",
    "split_heads_sell_source_contract_r11b": "sell_source_contract_v2",
    "split_heads_release_translation_deploy_r12": "release_translation_deploy_v1",
    "split_heads_action_value_unification_r13": "action_value_unification_v1",
    "split_heads_direct_action_value_r14": "direct_daily_policy_v1",
    "split_heads_direct_action_translation_r15": "direct_action_translation_v1",
    "split_heads_direct_action_reallocation_r16": "direct_action_reallocation_v1",
    "split_heads_direct_action_pair_reallocation_r17": "direct_action_pair_reallocation_v1",
    "split_heads_direct_action_pair_cost_guard_r18": "direct_action_pair_cost_guard_v1",
    "split_heads_portfolio_daily_ranking_r19": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_ranking_stability_r20": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_ranking_source_exec_r21": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_ranking_receiver_exec_r22": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_ranking_receiver_exec_stability_r23": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_listwise_allocation_r24": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_source_release_listwise_r25": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_allocation_teacher_r26": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_economic_release_r27": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_source_brake_r28": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_source_brake_relief_r29": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_source_brake_cash_relief_r30": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_receiver_semantic_closure_r31": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_source_distribution_quality_r32": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_source_forward_proxy_r33": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_allocation_breadth_r34": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_unified_allocation_r35": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_risk_aware_unified_allocation_r36": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_decision_focused_allocation_r37": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_source_hard_negative_regret_r38": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_allocation_objective_consolidation_r39": "portfolio_daily_ranking_v2_gated",
    "split_heads_portfolio_daily_end_to_end_allocation_layer_r40": "end_to_end_allocation_layer_v1",
    "split_heads_portfolio_daily_risk_sensitive_allocation_layer_r41": "end_to_end_allocation_layer_v1",
    "split_heads_portfolio_daily_utility_credit_allocation_r42": "end_to_end_allocation_layer_v1",
    "split_heads_portfolio_daily_primal_dual_decision_allocation_r43": "end_to_end_allocation_layer_v1",
    "split_heads_portfolio_daily_entropic_transport_allocation_r44": "end_to_end_allocation_layer_v1",
    "split_heads_portfolio_daily_conservative_transport_allocation_r45": "end_to_end_allocation_layer_v1",
    "split_heads_portfolio_daily_differentiable_convex_allocation_r46": "end_to_end_allocation_layer_v1",
    "split_heads_portfolio_daily_true_convex_solver_allocation_r47": "end_to_end_allocation_layer_v1",
    "split_heads_portfolio_daily_full_universe_convex_ope_allocation_r48": "end_to_end_allocation_layer_v1",
    "split_heads_portfolio_daily_capital_flow_closure_r49": "end_to_end_allocation_layer_v1",
    "split_heads_portfolio_daily_integrated_convex_capital_flow_r50": "end_to_end_allocation_layer_v1",
    "split_heads_portfolio_daily_native_allocation_vector_r51": "end_to_end_allocation_layer_v1",
    "split_heads_portfolio_daily_day_set_native_allocation_vector_r52": "end_to_end_allocation_layer_v1",
    "split_heads_portfolio_daily_day_set_native_target_validity_closure_r52b": "end_to_end_allocation_layer_v1",
    "split_heads_portfolio_daily_day_set_native_executable_receiver_closure_r52c": "end_to_end_allocation_layer_v1",
    "split_heads_portfolio_daily_day_set_native_validation_closure_r52d": "end_to_end_allocation_layer_v1",
}

PORTFOLIO_DAILY_GATE_OBJECTIVES = {
    "portfolio_daily_ranking_v2_gated",
    "end_to_end_allocation_layer_v1",
}

RESOURCE_GATED_SEARCH_PROFILES: dict[str, dict[str, Any]] = {
    "split_heads_portfolio_daily_utility_credit_allocation_r42": {
        "min_completed_screening": 1,
        "source_count_floor": 1.0,
        "source_sell_rate_floor": 0.20,
        "cash_timing_floor": -0.10,
        "drawdown_floor": -0.20,
        "monthly_return_floor": -0.002,
        "annual_return_floor": 0.04,
        "receiver_unrealized_cap": 0.08,
    },
    "split_heads_portfolio_daily_primal_dual_decision_allocation_r43": {
        "min_completed_screening": 1,
        "source_count_floor": 1.0,
        "source_sell_rate_floor": 0.22,
        "cash_timing_floor": -0.08,
        "drawdown_floor": -0.18,
        "monthly_return_floor": 0.0,
        "annual_return_floor": 0.06,
        "receiver_unrealized_cap": 0.06,
    },
    "split_heads_portfolio_daily_entropic_transport_allocation_r44": {
        "min_completed_screening": 1,
        "source_count_floor": 1.0,
        "source_sell_rate_floor": 0.24,
        "cash_timing_floor": -0.06,
        "drawdown_floor": -0.17,
        "monthly_return_floor": 0.001,
        "annual_return_floor": 0.08,
        "receiver_unrealized_cap": 0.05,
    },
    "split_heads_portfolio_daily_conservative_transport_allocation_r45": {
        "min_completed_screening": 1,
        "source_count_floor": 1.0,
        "source_sell_rate_floor": 0.26,
        "cash_timing_floor": -0.04,
        "drawdown_floor": -0.16,
        "monthly_return_floor": 0.001,
        "annual_return_floor": 0.08,
        "receiver_unrealized_cap": 0.04,
    },
    "split_heads_portfolio_daily_differentiable_convex_allocation_r46": {
        "min_completed_screening": 1,
        "source_count_floor": 2.0,
        "source_sell_rate_floor": 0.30,
        "cash_timing_floor": -0.02,
        "drawdown_floor": -0.15,
        "monthly_return_floor": 0.002,
        "annual_return_floor": 0.10,
        "receiver_unrealized_cap": 0.035,
    },
    "split_heads_portfolio_daily_true_convex_solver_allocation_r47": {
        "min_completed_screening": 1,
        "source_count_floor": 2.0,
        "source_sell_rate_floor": 0.32,
        "cash_timing_floor": -0.015,
        "drawdown_floor": -0.145,
        "monthly_return_floor": 0.002,
        "annual_return_floor": 0.10,
        "receiver_unrealized_cap": 0.032,
    },
    "split_heads_portfolio_daily_full_universe_convex_ope_allocation_r48": {
        "min_completed_screening": 1,
        "source_count_floor": 2.0,
        "source_sell_rate_floor": 0.34,
        "cash_timing_floor": -0.010,
        "drawdown_floor": -0.140,
        "monthly_return_floor": 0.0025,
        "annual_return_floor": 0.12,
        "receiver_unrealized_cap": 0.030,
    },
    "split_heads_portfolio_daily_capital_flow_closure_r49": {
        "min_completed_screening": 1,
        "source_count_floor": 3.0,
        "source_sell_rate_floor": 0.35,
        "cash_timing_floor": 0.0,
        "drawdown_floor": -0.135,
        "monthly_return_floor": 0.003,
        "annual_return_floor": 0.12,
        "receiver_unrealized_cap": 0.025,
        "exposure_utilization_floor": 0.50,
    },
    "split_heads_portfolio_daily_integrated_convex_capital_flow_r50": {
        "min_completed_screening": 1,
        "source_count_floor": 3.0,
        "source_sell_rate_floor": 0.35,
        "cash_timing_floor": 0.0,
        "drawdown_floor": -0.130,
        "monthly_return_floor": 0.003,
        "annual_return_floor": 0.12,
        "receiver_unrealized_cap": 0.025,
        "exposure_utilization_floor": 0.52,
    },
    "split_heads_portfolio_daily_native_allocation_vector_r51": {
        "min_completed_screening": 1,
        "source_count_floor": 3.0,
        "source_sell_rate_floor": 0.35,
        "cash_timing_floor": 0.0,
        "drawdown_floor": -0.135,
        "monthly_return_floor": 0.003,
        "annual_return_floor": 0.12,
        "receiver_unrealized_cap": 0.025,
        "exposure_utilization_floor": 0.50,
    },
    "split_heads_portfolio_daily_day_set_native_allocation_vector_r52": {
        "min_completed_screening": 1,
        "source_count_floor": 3.0,
        "source_sell_rate_floor": 0.35,
        "cash_timing_floor": 0.0,
        "drawdown_floor": -0.135,
        "monthly_return_floor": 0.003,
        "annual_return_floor": 0.12,
        "receiver_unrealized_cap": 0.025,
        "exposure_utilization_floor": 0.50,
    },
    "split_heads_portfolio_daily_day_set_native_target_validity_closure_r52b": {
        "min_completed_screening": 1,
        "source_count_floor": 3.0,
        "source_sell_rate_floor": 0.35,
        "cash_timing_floor": 0.0,
        "drawdown_floor": -0.135,
        "monthly_return_floor": 0.003,
        "annual_return_floor": 0.12,
        "receiver_unrealized_cap": 0.025,
        "exposure_utilization_floor": 0.50,
    },
    "split_heads_portfolio_daily_day_set_native_executable_receiver_closure_r52c": {
        "min_completed_screening": 1,
        "source_count_floor": 3.0,
        "source_sell_rate_floor": 0.35,
        "cash_timing_floor": 0.0,
        "drawdown_floor": -0.135,
        "monthly_return_floor": 0.003,
        "annual_return_floor": 0.12,
        "receiver_unrealized_cap": 0.025,
        "exposure_utilization_floor": 0.50,
    },
    "split_heads_portfolio_daily_day_set_native_validation_closure_r52d": {
        "min_completed_screening": 1,
        "source_count_floor": 3.0,
        "source_sell_rate_floor": 0.35,
        "cash_timing_floor": 0.0,
        "drawdown_floor": -0.135,
        "monthly_return_floor": 0.003,
        "annual_return_floor": 0.12,
        "receiver_unrealized_cap": 0.025,
        "exposure_utilization_floor": 0.50,
    },
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
    train_section = dict(protocol_summary.get("train", {}) or {})
    teacher_summary = dict(
        train_section.get("teacher_summary", {})
        or protocol_summary.get("teacher_summary", {})
        or {}
    )
    unified_allocation_summary = dict(
        teacher_summary.get("unified_allocation_summary_mean", {})
        or protocol_summary.get("unified_allocation_summary_mean", {})
        or {}
    )
    shadow = dict(protocol_summary.get("shadow", {}) or {})
    shadow_continuity = dict(shadow.get("continuity_metrics", {}) or {})

    def _semantic_metric(key: str, default: float = 0.0) -> float:
        value = semantic_conflicts.get(key, None)
        if value is None or value == "":
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _summary_metric(key: str, default: float = 0.0) -> float:
        value = semantic_conflicts.get(key, None)
        if value is None or value == "":
            value = continuity.get(key, None)
        if value is None or value == "":
            value = metrics.get(key, default)
        if value is None or value == "":
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _unified_allocation_metric(key: str, default: float = 0.0) -> float:
        value = unified_allocation_summary.get(key, None)
        if value is None or value == "":
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    annual_return = float(metrics.get("annual_return", 0.0) or 0.0)
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
    monthly_return_mean = float(metrics.get("monthly_return_mean", 0.0) or 0.0)
    monthly_win_rate = float(metrics.get("monthly_win_rate", 0.0) or 0.0)
    monthly_worst_return = float(metrics.get("monthly_worst_return", 0.0) or 0.0)
    monthly_sharpe = float(metrics.get("monthly_sharpe", 0.0) or 0.0)
    monthly_consistency_score = float(metrics.get("monthly_consistency_score", 0.0) or 0.0)
    monthly_max_consecutive_loss_months = float(metrics.get("monthly_max_consecutive_loss_months", 0.0) or 0.0)
    monthly_intramonth_max_drawdown = float(metrics.get("monthly_intramonth_max_drawdown", 0.0) or 0.0)
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
    deploy_intent_realized_rate = float(
        semantic_conflicts.get(
            "deploy_intent_realized_rate",
            continuity.get("deploy_intent_realized_rate", 0.0),
        )
        or 0.0
    )
    open_add_positive_weight_change_rate = float(
        semantic_conflicts.get(
            "open_add_positive_weight_change_rate",
            continuity.get("open_add_positive_weight_change_rate", 0.0),
        )
        or 0.0
    )
    add_to_hold_conflict_share = float(
        semantic_conflicts.get(
            "add_to_hold_conflict_share",
            continuity.get("add_to_hold_conflict_share", 0.0),
        )
        or 0.0
    )
    deploy_intent_hold_conflict_share = float(
        semantic_conflicts.get(
            "deploy_intent_hold_conflict_share",
            continuity.get("deploy_intent_hold_conflict_share", 0.0),
        )
        or 0.0
    )
    deploy_intent_dropped_share = float(semantic_conflicts.get("deploy_intent_dropped_share", 0.0) or 0.0)
    deploy_intent_candidate_budget_drop_share = float(
        semantic_conflicts.get("avg_deploy_intent_candidate_budget_drop_share", deploy_intent_dropped_share) or 0.0
    )
    deploy_executability_alignment = float(continuity.get("deploy_executability_forward_alignment_5d", 0.0) or 0.0)
    deploy_gate_alignment = float(continuity.get("deploy_gate_forward_alignment_5d", 0.0) or 0.0)
    release_gate_alignment = float(continuity.get("release_gate_forward_alignment_5d", 0.0) or 0.0)
    value_arbitration_alignment = float(continuity.get("value_arbitration_forward_alignment_5d", 0.0) or 0.0)
    multi_horizon_path_alignment = float(continuity.get("multi_horizon_path_value_alignment_5d", 0.0) or 0.0)
    open_action_value_alignment = float(continuity.get("open_action_value_forward_alignment_5d", 0.0) or 0.0)
    add_action_value_alignment = float(continuity.get("add_action_value_forward_alignment_5d", 0.0) or 0.0)
    hold_action_value_alignment = float(continuity.get("hold_action_value_forward_alignment_5d", 0.0) or 0.0)
    reduce_action_value_avoidance = float(continuity.get("reduce_action_value_forward_avoidance_5d", 0.0) or 0.0)
    exit_action_value_avoidance = float(continuity.get("exit_action_value_forward_avoidance_5d", 0.0) or 0.0)
    multi_horizon_forward_risk_avoidance = float(
        continuity.get("multi_horizon_forward_risk_avoidance_5d", 0.0) or 0.0
    )
    action_value_consistency_score = float(
        semantic_conflicts.get(
            "action_value_consistency_score",
            continuity.get("action_value_consistency_score", 0.0),
        )
        or 0.0
    )
    action_value_conflict_share = float(
        semantic_conflicts.get(
            "action_value_conflict_share",
            continuity.get("action_value_conflict_share", 0.0),
        )
        or 0.0
    )
    sell_against_keep_value_share = float(
        semantic_conflicts.get(
            "sell_against_keep_value_share",
            continuity.get("sell_against_keep_value_share", 0.0),
        )
        or 0.0
    )
    keep_against_release_value_share = float(
        semantic_conflicts.get(
            "keep_against_release_value_share",
            continuity.get("keep_against_release_value_share", 0.0),
        )
        or 0.0
    )
    open_low_action_value_share = float(
        semantic_conflicts.get(
            "open_low_action_value_share",
            continuity.get("open_low_action_value_share", 0.0),
        )
        or 0.0
    )
    held_keep_release_value_gap = float(
        semantic_conflicts.get(
            "held_keep_release_value_gap",
            continuity.get("held_keep_release_value_gap", 0.0),
        )
        or 0.0
    )
    direct_action_value_mode_share = float(
        semantic_conflicts.get(
            "direct_action_value_mode_share",
            continuity.get("direct_action_value_mode_share", 0.0),
        )
        or 0.0
    )
    direct_action_value_label_match_share = float(
        semantic_conflicts.get(
            "direct_action_value_label_match_share",
            continuity.get("direct_action_value_label_match_share", 0.0),
        )
        or 0.0
    )
    direct_action_value_gap_mean = float(
        semantic_conflicts.get(
            "direct_action_value_gap_mean",
            continuity.get("direct_action_value_gap_mean", 0.0),
        )
        or 0.0
    )
    direct_action_value_low_margin_share = float(
        semantic_conflicts.get(
            "direct_action_value_low_margin_share",
            continuity.get("direct_action_value_low_margin_share", 0.0),
        )
        or 0.0
    )
    direct_action_order_translation_conflict_rate = float(
        semantic_conflicts.get(
            "direct_action_order_translation_conflict_rate",
            continuity.get("direct_action_order_translation_conflict_rate", 0.0),
        )
        or 0.0
    )
    direct_action_intent_preserved_share = float(
        semantic_conflicts.get(
            "direct_action_intent_preserved_share",
            continuity.get("direct_action_intent_preserved_share", 0.0),
        )
        or 0.0
    )
    direct_action_funding_authorized_sell_share = float(
        semantic_conflicts.get(
            "direct_action_funding_authorized_sell_share",
            continuity.get("direct_action_funding_authorized_sell_share", 0.0),
        )
        or 0.0
    )
    direct_action_funding_protected_sell_share = float(
        semantic_conflicts.get(
            "direct_action_funding_protected_sell_share",
            continuity.get("direct_action_funding_protected_sell_share", 0.0),
        )
        or 0.0
    )
    direct_action_release_advantage_mean = float(
        semantic_conflicts.get(
            "direct_action_release_advantage_mean",
            continuity.get("direct_action_release_advantage_mean", 0.0),
        )
        or 0.0
    )
    direct_action_deploy_advantage_mean = float(
        semantic_conflicts.get(
            "direct_action_deploy_advantage_mean",
            continuity.get("direct_action_deploy_advantage_mean", 0.0),
        )
        or 0.0
    )
    direct_action_add_authorized_count = float(
        semantic_conflicts.get(
            "direct_action_add_authorized_count",
            continuity.get("direct_action_add_authorized_count", 0.0),
        )
        or 0.0
    )
    direct_action_add_authorized_realized_rate = float(
        semantic_conflicts.get(
            "direct_action_add_authorized_realized_rate",
            continuity.get("direct_action_add_authorized_realized_rate", 0.0),
        )
        or 0.0
    )
    direct_action_deploy_authorized_count = float(
        semantic_conflicts.get(
            "direct_action_deploy_authorized_count",
            continuity.get("direct_action_deploy_authorized_count", 0.0),
        )
        or 0.0
    )
    direct_action_deploy_authorized_realized_rate = float(
        semantic_conflicts.get(
            "direct_action_deploy_authorized_realized_rate",
            continuity.get("direct_action_deploy_authorized_realized_rate", 0.0),
        )
        or 0.0
    )
    direct_action_core_deploy_target_count = float(
        semantic_conflicts.get(
            "direct_action_core_deploy_target_count",
            continuity.get("direct_action_core_deploy_target_count", 0.0),
        )
        or 0.0
    )
    direct_action_core_deploy_target_realized_rate = float(
        semantic_conflicts.get(
            "direct_action_core_deploy_target_realized_rate",
            continuity.get("direct_action_core_deploy_target_realized_rate", 0.0),
        )
        or 0.0
    )
    direct_action_reallocation_source_count = float(
        semantic_conflicts.get(
            "direct_action_reallocation_source_count",
            continuity.get("direct_action_reallocation_source_count", 0.0),
        )
        or 0.0
    )
    direct_action_pair_reallocation_source_count = float(
        semantic_conflicts.get(
            "direct_action_pair_reallocation_source_count",
            continuity.get("direct_action_pair_reallocation_source_count", 0.0),
        )
        or 0.0
    )
    direct_action_pair_cost_guard_pass_count = float(
        semantic_conflicts.get(
            "direct_action_pair_cost_guard_pass_count",
            continuity.get("direct_action_pair_cost_guard_pass_count", 0.0),
        )
        or 0.0
    )
    direct_action_pair_cost_guard_blocked_count = float(
        semantic_conflicts.get(
            "direct_action_pair_cost_guard_blocked_count",
            continuity.get("direct_action_pair_cost_guard_blocked_count", 0.0),
        )
        or 0.0
    )
    direct_action_pair_cost_guard_pass_rate = float(
        semantic_conflicts.get(
            "direct_action_pair_cost_guard_pass_rate",
            continuity.get("direct_action_pair_cost_guard_pass_rate", 0.0),
        )
        or 0.0
    )
    direct_action_pair_source_spread_mean = float(
        semantic_conflicts.get(
            "direct_action_pair_source_spread_mean",
            continuity.get("direct_action_pair_source_spread_mean", 0.0),
        )
        or 0.0
    )
    direct_action_pair_source_cost_mean = float(
        semantic_conflicts.get(
            "direct_action_pair_source_cost_mean",
            continuity.get("direct_action_pair_source_cost_mean", 0.0),
        )
        or 0.0
    )
    direct_action_pair_source_forward_excess_5d = float(
        semantic_conflicts.get(
            "direct_action_pair_source_forward_excess_5d",
            continuity.get("direct_action_pair_source_forward_excess_5d", 0.0),
        )
        or 0.0
    )
    direct_action_core_target_forward_excess_5d = float(
        semantic_conflicts.get(
            "direct_action_core_target_forward_excess_5d",
            continuity.get("direct_action_core_target_forward_excess_5d", 0.0),
        )
        or 0.0
    )
    direct_action_core_minus_pair_forward_excess_5d = float(
        semantic_conflicts.get(
            "direct_action_core_minus_pair_forward_excess_5d",
            continuity.get("direct_action_core_minus_pair_forward_excess_5d", 0.0),
        )
        or 0.0
    )
    portfolio_daily_receiver_target_count = float(
        semantic_conflicts.get(
            "portfolio_daily_receiver_target_count",
            continuity.get("portfolio_daily_receiver_target_count", 0.0),
        )
        or 0.0
    )
    portfolio_daily_receiver_exec_guard_count = float(
        semantic_conflicts.get(
            "portfolio_daily_receiver_exec_guard_count",
            continuity.get("portfolio_daily_receiver_exec_guard_count", 0.0),
        )
        or 0.0
    )
    portfolio_daily_receiver_realized_deploy_rate = float(
        semantic_conflicts.get(
            "portfolio_daily_receiver_realized_deploy_rate",
            continuity.get("portfolio_daily_receiver_realized_deploy_rate", 0.0),
        )
        or 0.0
    )
    portfolio_daily_receiver_unrealized_deploy_share = float(
        semantic_conflicts.get(
            "portfolio_daily_receiver_unrealized_deploy_share",
            continuity.get("portfolio_daily_receiver_unrealized_deploy_share", 0.0),
        )
        or 0.0
    )
    portfolio_daily_receiver_candidate_count = float(
        semantic_conflicts.get(
            "portfolio_daily_receiver_candidate_count",
            continuity.get("portfolio_daily_receiver_candidate_count", 0.0),
        )
        or 0.0
    )
    portfolio_daily_receiver_semantic_no_headroom_count = float(
        semantic_conflicts.get(
            "portfolio_daily_receiver_semantic_no_headroom_count",
            continuity.get("portfolio_daily_receiver_semantic_no_headroom_count", 0.0),
        )
        or 0.0
    )
    portfolio_daily_receiver_open_breadth_candidate_count = float(
        semantic_conflicts.get(
            "portfolio_daily_receiver_open_breadth_candidate_count",
            continuity.get("portfolio_daily_receiver_open_breadth_candidate_count", 0.0),
        )
        or 0.0
    )
    authorized_add_no_weight_change_share = float(
        semantic_conflicts.get(
            "authorized_add_no_weight_change_share",
            continuity.get("authorized_add_no_weight_change_share", 0.0),
        )
        or 0.0
    )
    deploy_intent_unrealized_share = float(
        semantic_conflicts.get(
            "deploy_intent_unrealized_share",
            continuity.get("deploy_intent_unrealized_share", 0.0),
        )
        or 0.0
    )
    direct_action_authorization_subset_violation_count = float(
        semantic_conflicts.get(
            "direct_action_authorization_subset_violation_count",
            continuity.get("direct_action_authorization_subset_violation_count", 0.0),
        )
        or 0.0
    )
    portfolio_daily_source_target_count = float(
        semantic_conflicts.get(
            "portfolio_daily_source_target_count",
            continuity.get("portfolio_daily_source_target_count", 0.0),
        )
        or 0.0
    )
    portfolio_daily_source_candidate_count = float(
        semantic_conflicts.get(
            "portfolio_daily_source_candidate_count",
            continuity.get("portfolio_daily_source_candidate_count", 0.0),
        )
        or 0.0
    )
    portfolio_daily_source_realized_sell_rate = float(
        semantic_conflicts.get(
            "portfolio_daily_source_realized_sell_rate",
            continuity.get("portfolio_daily_source_realized_sell_rate", 0.0),
        )
        or 0.0
    )
    portfolio_daily_source_target_not_sold_share = float(
        semantic_conflicts.get(
            "portfolio_daily_source_target_not_sold_share",
            continuity.get("portfolio_daily_source_target_not_sold_share", 0.0),
        )
        or 0.0
    )
    portfolio_daily_source_exec_cap_guard_count = float(
        semantic_conflicts.get(
            "portfolio_daily_source_exec_cap_guard_count",
            continuity.get("portfolio_daily_source_exec_cap_guard_count", 0.0),
        )
        or 0.0
    )
    portfolio_daily_effective_capital_transfer_count = float(
        semantic_conflicts.get(
            "portfolio_daily_effective_capital_transfer_count",
            continuity.get("portfolio_daily_effective_capital_transfer_count", 0.0),
        )
        or 0.0
    )
    portfolio_daily_cash_score_mean = float(
        semantic_conflicts.get(
            "portfolio_daily_cash_score_mean",
            continuity.get("portfolio_daily_cash_score_mean", 0.0),
        )
        or 0.0
    )
    portfolio_daily_cash_reserve_rate = float(
        semantic_conflicts.get(
            "portfolio_daily_cash_reserve_rate",
            continuity.get("portfolio_daily_cash_reserve_rate", 0.0),
        )
        or 0.0
    )
    portfolio_daily_source_gap_mean = float(
        semantic_conflicts.get(
            "portfolio_daily_source_gap_mean",
            continuity.get("portfolio_daily_source_gap_mean", 0.0),
        )
        or 0.0
    )
    portfolio_daily_source_forward_spread_score_mean = float(
        semantic_conflicts.get(
            "portfolio_daily_source_forward_spread_score_mean",
            continuity.get(
                "portfolio_daily_source_forward_spread_score_mean",
                metrics.get("avg_portfolio_daily_source_forward_spread_score", 0.0),
            ),
        )
        or 0.0
    )
    portfolio_daily_source_bad_forward_spread_risk_mean = float(
        semantic_conflicts.get(
            "portfolio_daily_source_bad_forward_spread_risk_mean",
            continuity.get(
                "portfolio_daily_source_bad_forward_spread_risk_mean",
                metrics.get("avg_portfolio_daily_source_bad_forward_spread_risk", 0.0),
            ),
        )
        or 0.0
    )
    portfolio_daily_source_economic_release_score_mean = float(
        semantic_conflicts.get(
            "portfolio_daily_source_economic_release_score_mean",
            continuity.get(
                "portfolio_daily_source_economic_release_score_mean",
                metrics.get("avg_portfolio_daily_source_economic_release_score", 0.0),
            ),
        )
        or 0.0
    )
    portfolio_daily_source_economic_block_risk_mean = float(
        semantic_conflicts.get(
            "portfolio_daily_source_economic_block_risk_mean",
            continuity.get(
                "portfolio_daily_source_economic_block_risk_mean",
                metrics.get("avg_portfolio_daily_source_economic_block_risk", 0.0),
            ),
        )
        or 0.0
    )
    portfolio_daily_source_forward_strength_brake_risk_mean = float(
        semantic_conflicts.get(
            "portfolio_daily_source_forward_strength_brake_risk_mean",
            continuity.get(
                "portfolio_daily_source_forward_strength_brake_risk_mean",
                metrics.get("avg_portfolio_daily_source_forward_strength_brake_risk", 0.0),
            ),
        )
        or 0.0
    )
    portfolio_daily_source_forward_proxy_keep_risk_mean = float(
        semantic_conflicts.get(
            "portfolio_daily_source_forward_proxy_keep_risk_mean",
            continuity.get(
                "portfolio_daily_source_forward_proxy_keep_risk_mean",
                metrics.get("avg_portfolio_daily_source_forward_proxy_keep_risk", 0.0),
            ),
        )
        or 0.0
    )
    portfolio_daily_source_release_conviction_mean = float(
        semantic_conflicts.get(
            "portfolio_daily_source_release_conviction_mean",
            continuity.get(
                "portfolio_daily_source_release_conviction_mean",
                metrics.get("avg_portfolio_daily_source_release_conviction", 0.0),
            ),
        )
        or 0.0
    )
    portfolio_daily_source_distribution_clean_pass_mean = float(
        semantic_conflicts.get(
            "portfolio_daily_source_distribution_clean_pass_mean",
            continuity.get(
                "portfolio_daily_source_distribution_clean_pass_mean",
                metrics.get("avg_portfolio_daily_source_distribution_clean_pass", 0.0),
            ),
        )
        or 0.0
    )
    portfolio_daily_source_distribution_clean_blocked_count = float(
        semantic_conflicts.get(
            "portfolio_daily_source_distribution_clean_blocked_count",
            continuity.get(
                "portfolio_daily_source_distribution_clean_blocked_count",
                metrics.get("avg_portfolio_daily_source_distribution_clean_blocked_count", 0.0),
            ),
        )
        or 0.0
    )
    portfolio_daily_source_direct_release_relief_score_mean = float(
        semantic_conflicts.get(
            "portfolio_daily_source_direct_release_relief_score_mean",
            continuity.get(
                "portfolio_daily_source_direct_release_relief_score_mean",
                metrics.get("avg_portfolio_daily_source_direct_release_relief_score", 0.0),
            ),
        )
        or 0.0
    )
    portfolio_daily_receiver_funding_coverage_mean = float(
        semantic_conflicts.get(
            "portfolio_daily_receiver_funding_coverage_mean",
            continuity.get(
                "portfolio_daily_receiver_funding_coverage_mean",
                metrics.get("avg_portfolio_daily_receiver_funding_coverage", 0.0),
            ),
        )
        or 0.0
    )
    portfolio_daily_funding_closure_score_mean = float(
        semantic_conflicts.get(
            "portfolio_daily_funding_closure_score_mean",
            continuity.get(
                "portfolio_daily_funding_closure_score_mean",
                metrics.get("avg_portfolio_daily_funding_closure_score", 0.0),
            ),
        )
        or 0.0
    )
    portfolio_daily_allocation_transfer_score_mean = float(
        semantic_conflicts.get(
            "portfolio_daily_allocation_transfer_score_mean",
            continuity.get(
                "portfolio_daily_allocation_transfer_score_mean",
                metrics.get("avg_portfolio_daily_allocation_transfer_score", 0.0),
            ),
        )
        or 0.0
    )
    portfolio_daily_allocation_dead_branch_risk_mean = float(
        semantic_conflicts.get(
            "portfolio_daily_allocation_dead_branch_risk_mean",
            continuity.get(
                "portfolio_daily_allocation_dead_branch_risk_mean",
                metrics.get("avg_portfolio_daily_allocation_dead_branch_risk", 0.0),
            ),
        )
        or 0.0
    )
    portfolio_daily_unified_allocation_objective = _unified_allocation_metric(
        "portfolio_daily_unified_allocation_objective"
    )
    portfolio_daily_unified_solution_objective = _unified_allocation_metric(
        "portfolio_daily_unified_solution_objective"
    )
    portfolio_daily_unified_receiver_score = _unified_allocation_metric(
        "portfolio_daily_unified_receiver_score"
    )
    portfolio_daily_unified_source_score = _unified_allocation_metric(
        "portfolio_daily_unified_source_score"
    )
    portfolio_daily_unified_cash_score = _unified_allocation_metric(
        "portfolio_daily_unified_cash_score"
    )
    portfolio_daily_source_positive_forward_penalty = _unified_allocation_metric(
        "portfolio_daily_source_positive_forward_penalty"
    )
    portfolio_daily_source_opportunity_cost_penalty = _unified_allocation_metric(
        "portfolio_daily_source_opportunity_cost_penalty"
    )
    portfolio_daily_source_hard_negative_penalty = _unified_allocation_metric(
        "portfolio_daily_source_hard_negative_penalty"
    )
    portfolio_daily_source_tail_false_sell_penalty = _unified_allocation_metric(
        "portfolio_daily_source_tail_false_sell_penalty"
    )
    portfolio_daily_source_release_preference = _unified_allocation_metric(
        "portfolio_daily_source_release_preference"
    )
    portfolio_daily_receiver_source_spread_reward = _unified_allocation_metric(
        "portfolio_daily_receiver_source_spread_reward"
    )
    portfolio_daily_transfer_regret_target = _unified_allocation_metric(
        "portfolio_daily_transfer_regret_target"
    )
    portfolio_daily_source_hard_negative_prevalence = _unified_allocation_metric(
        "portfolio_daily_source_hard_negative_prevalence"
    )
    portfolio_daily_source_hard_negative_selected_pressure = _unified_allocation_metric(
        "portfolio_daily_source_hard_negative_selected_pressure"
    )
    portfolio_daily_unified_constraint_violations = _unified_allocation_metric(
        "portfolio_daily_unified_constraint_violations"
    )
    portfolio_daily_receiver_forward_excess_5d = float(
        semantic_conflicts.get(
            "portfolio_daily_receiver_forward_excess_5d",
            continuity.get("portfolio_daily_receiver_forward_excess_5d", 0.0),
        )
        or 0.0
    )
    portfolio_daily_source_forward_excess_5d = float(
        semantic_conflicts.get(
            "portfolio_daily_source_forward_excess_5d",
            continuity.get("portfolio_daily_source_forward_excess_5d", 0.0),
        )
        or 0.0
    )
    portfolio_daily_source_positive_forward_sell_share = float(
        semantic_conflicts.get(
            "portfolio_daily_source_positive_forward_sell_share",
            continuity.get("portfolio_daily_source_positive_forward_sell_share", 0.0),
        )
        or 0.0
    )
    portfolio_daily_source_strong_positive_forward_sell_count = float(
        semantic_conflicts.get(
            "portfolio_daily_source_strong_positive_forward_sell_count",
            continuity.get("portfolio_daily_source_strong_positive_forward_sell_count", 0.0),
        )
        or 0.0
    )
    portfolio_daily_source_max_forward_excess_5d = float(
        semantic_conflicts.get(
            "portfolio_daily_source_max_forward_excess_5d",
            continuity.get("portfolio_daily_source_max_forward_excess_5d", 0.0),
        )
        or 0.0
    )
    portfolio_daily_source_p75_forward_excess_5d = float(
        semantic_conflicts.get(
            "portfolio_daily_source_p75_forward_excess_5d",
            continuity.get("portfolio_daily_source_p75_forward_excess_5d", 0.0),
        )
        or 0.0
    )
    portfolio_daily_receiver_minus_source_forward_excess_5d = float(
        semantic_conflicts.get(
            "portfolio_daily_receiver_minus_source_forward_excess_5d",
            continuity.get("portfolio_daily_receiver_minus_source_forward_excess_5d", 0.0),
        )
        or 0.0
    )
    avg_gross_exposure_target = float(
        semantic_conflicts.get(
            "avg_gross_exposure_target",
            continuity.get("avg_gross_exposure_target", metrics.get("avg_gross_exposure_target", 0.0)),
        )
        or 0.0
    )
    portfolio_daily_exposure_utilization = float(
        semantic_conflicts.get(
            "portfolio_daily_exposure_utilization",
            continuity.get("portfolio_daily_exposure_utilization", metrics.get("portfolio_daily_exposure_utilization", 0.0)),
        )
        or 0.0
    )
    if portfolio_daily_exposure_utilization <= 0.0 and avg_gross_exposure_target > 0.0:
        portfolio_daily_exposure_utilization = float(
            avg_gross_exposure / max(avg_gross_exposure_target, 1.0e-8)
        )
    sell_selection_quality = float(continuity.get("sell_selection_quality_5d", 0.0) or 0.0)
    budget_origin_sell_share = float(semantic_conflicts.get("budget_origin_sell_share", 0.0) or 0.0)
    high_cash_budget_origin_sell_share = float(semantic_conflicts.get("high_cash_budget_origin_sell_share", 0.0) or 0.0)
    deploy_funding_rebalance_sell_share = float(
        semantic_conflicts.get("deploy_funding_rebalance_sell_share", 0.0) or 0.0
    )
    deploy_funding_rebalance_sell_count = float(
        semantic_conflicts.get("deploy_funding_rebalance_sell_count", 0.0) or 0.0
    )
    deploy_funding_rebalance_forward_excess_5d = float(
        semantic_conflicts.get("deploy_funding_rebalance_forward_excess_5d", 0.0) or 0.0
    )
    deploy_funding_against_protected_hold_share = float(
        semantic_conflicts.get("deploy_funding_against_protected_hold_share", 0.0) or 0.0
    )
    deploy_funding_release_consistent_share = float(
        semantic_conflicts.get("deploy_funding_release_consistent_share", 0.0) or 0.0
    )
    model_release_signal_sell_count = float(semantic_conflicts.get("model_release_signal_sell_count", 0.0) or 0.0)
    model_release_signal_forward_excess_5d = float(
        semantic_conflicts.get("model_release_signal_forward_excess_5d", 0.0) or 0.0
    )
    model_release_against_protected_hold_share = float(
        semantic_conflicts.get("model_release_against_protected_hold_share", 0.0) or 0.0
    )
    model_release_release_consistent_share = float(
        semantic_conflicts.get("model_release_release_consistent_share", 0.0) or 0.0
    )
    sell_source_floor_guard_share = float(semantic_conflicts.get("sell_source_floor_guard_share", 0.0) or 0.0)
    deploy_intent_action_count = _semantic_metric("deploy_intent_action_count", 0.0)
    release_translation_deploy_deploy_score = _semantic_metric(
        "release_translation_deploy_deploy_score",
        _bounded(deploy_intent_realized_rate, 0.35, 0.85) if deploy_intent_action_count >= 3.0 else 0.0,
    )
    release_translation_deploy_release_score = _semantic_metric(
        "release_translation_deploy_release_score",
        (
            _bounded(deploy_funding_release_consistent_share, 0.05, 0.65)
            if deploy_funding_rebalance_sell_count >= 5.0
            else 0.0
        ),
    )
    release_translation_deploy_translation_score = _semantic_metric(
        "release_translation_deploy_translation_score",
        1.0 - _bounded(order_translation_conflict_rate, 0.06, 0.35),
    )
    _funding_forward_score = 1.0 - _bounded(max(0.0, deploy_funding_rebalance_forward_excess_5d), 0.0, 0.04)
    _funding_protected_hold_score = 1.0 - _bounded(deploy_funding_against_protected_hold_share, 0.10, 0.45)
    _funding_sell_share_score = 1.0 - _bounded(max(0.0, deploy_funding_rebalance_sell_share - 0.35), 0.0, 0.45)
    _funding_budget_origin_score = 1.0 - _bounded(budget_origin_sell_share, 0.0, 0.35)
    release_translation_deploy_funding_score = _semantic_metric(
        "release_translation_deploy_funding_score",
        (
            _funding_forward_score * 0.30
            + _funding_protected_hold_score * 0.30
            + _funding_sell_share_score * 0.20
            + _funding_budget_origin_score * 0.20
            if deploy_funding_rebalance_sell_count >= 5.0
            else 0.0
        ),
    )
    release_translation_deploy_model_release_score = _semantic_metric(
        "release_translation_deploy_model_release_score",
        (
            (1.0 - _bounded(max(0.0, model_release_signal_forward_excess_5d), 0.0, 0.04)) * 0.35
            + (1.0 - _bounded(model_release_against_protected_hold_share, 0.10, 0.45)) * 0.30
            + _bounded(model_release_release_consistent_share, 0.05, 0.65) * 0.35
            if model_release_signal_sell_count >= 5.0
            else release_translation_deploy_release_score
        ),
    )
    release_translation_deploy_health_score = _semantic_metric(
        "release_translation_deploy_health_score",
        release_translation_deploy_deploy_score * 0.26
        + release_translation_deploy_release_score * 0.28
        + release_translation_deploy_translation_score * 0.20
        + release_translation_deploy_funding_score * 0.20
        + release_translation_deploy_model_release_score * 0.06,
    )
    release_translation_deploy_failure_mode = str(
        semantic_conflicts.get("release_translation_deploy_failure_mode", "") or ""
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
    monthly_worst_loss = abs(min(monthly_worst_return, 0.0))
    monthly_loss_streak_penalty = max(0.0, monthly_max_consecutive_loss_months - 1.0)
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

    objective_profile_name = str(objective_profile or "promotion_balanced_v2")
    if objective_profile_name == "deploy_executability_v1":
        performance_breakdown = {
            "annual_return": annual_return * 2.55,
            "sharpe": sharpe * 0.30,
            "open_win_rate_5d": open_win * 1.05,
            "deploy_intent_realized_rate": deploy_intent_realized_rate * 1.45,
            "open_add_positive_weight_change_rate": open_add_positive_weight_change_rate * 1.15,
            "deploy_gate_forward_alignment_5d": _bounded(deploy_gate_alignment, -0.25, 0.20) * 0.70,
            "deploy_executability_forward_alignment_5d": _bounded(deploy_executability_alignment, -0.25, 0.20) * 0.78,
            "sell_selection_quality_5d": _bounded(sell_selection_quality, -0.04, 0.05) * 0.38,
            "reduce_success_rate_5d": reduce_success * 0.88,
            "exit_timeliness_rate_5d": exit_timeliness * 0.90,
            "cash_timing_quality_1d": _bounded(cash_timing, -0.35, 0.12) * 0.72,
            "trend_capture_rate_10d": trend_capture * 0.88,
            "active_alignment_bonus": active_alignment_bonus,
            "gate_pass_ratio": gate_pass_ratio * 0.36,
            "drawdown_penalty": -abs(min(max_drawdown, 0.0)) * 3.40,
            "exposure_penalty": -_exposure_penalty(avg_gross_exposure),
            "negative_return_penalty": -negative_return_penalty * 3.20,
            "negative_sharpe_penalty": -negative_sharpe_penalty * 0.58,
            "semantic_conflict_penalty": -semantic_conflict_rate * 1.40,
            "order_translation_conflict_penalty": -order_translation_conflict_rate * 0.80,
            "add_to_hold_conflict_penalty": -add_to_hold_conflict_share * 1.35,
            "deploy_hold_conflict_penalty": -deploy_intent_hold_conflict_share * 1.05,
            "deploy_dropped_penalty": -deploy_intent_dropped_share * 1.15,
            "deploy_candidate_budget_drop_penalty": -deploy_intent_candidate_budget_drop_share * 1.05,
        }
        stability_breakdown = {
            "gate_pass_ratio": gate_pass_ratio * 0.62,
            "deploy_intent_realized_rate": deploy_intent_realized_rate * 1.05,
            "open_add_positive_weight_change_rate": open_add_positive_weight_change_rate * 0.88,
            "deploy_executability_forward_alignment_5d": _bounded(deploy_executability_alignment, -0.25, 0.20) * 0.54,
            "reduce_success_rate_5d": reduce_success * 0.46,
            "exit_timeliness_rate_5d": exit_timeliness * 0.48,
            "cash_timing_quality_1d": _bounded(cash_timing, -0.35, 0.12) * 0.44,
            "hold_share": hold_share * 0.18,
            "reversal_penalty": -reversal * 1.15,
            "shadow_reversal_penalty": -shadow_reversal * 1.10,
            "reversal_excess_penalty": -reversal_excess_penalty,
            "drawdown_penalty": -abs(min(max_drawdown, 0.0)) * 5.10,
            "threshold_gap_penalty": -threshold_gap_penalty * 1.05,
            "training_evidence_bonus": 0.32 if training_evidence_ok else -0.40,
            "semantic_conflict_penalty": -semantic_conflict_rate * 1.95,
            "order_translation_conflict_penalty": -order_translation_conflict_rate * 1.05,
            "add_to_hold_conflict_penalty": -add_to_hold_conflict_share * 1.60,
            "deploy_hold_conflict_penalty": -deploy_intent_hold_conflict_share * 1.20,
            "deploy_dropped_penalty": -deploy_intent_dropped_share * 1.25,
            "deploy_candidate_budget_drop_penalty": -deploy_intent_candidate_budget_drop_share * 1.25,
        }
    elif objective_profile_name == "sell_source_contract_v1":
        performance_breakdown = {
            "annual_return": annual_return * 2.25,
            "sharpe": sharpe * 0.28,
            "deploy_intent_realized_rate": deploy_intent_realized_rate * 1.10,
            "sell_selection_quality_5d": _bounded(sell_selection_quality, -0.06, 0.08) * 0.80,
            "reduce_success_rate_5d": reduce_success * 0.95,
            "exit_timeliness_rate_5d": exit_timeliness * 1.00,
            "release_gate_forward_alignment_5d": _bounded(release_gate_alignment, -0.10, 0.10) * 0.74,
            "value_arbitration_forward_alignment_5d": _bounded(value_arbitration_alignment, -0.10, 0.12) * 0.64,
            "cash_timing_quality_1d": _bounded(cash_timing, -0.35, 0.12) * 0.68,
            "active_alignment_bonus": active_alignment_bonus,
            "gate_pass_ratio": gate_pass_ratio * 0.30,
            "drawdown_penalty": -abs(min(max_drawdown, 0.0)) * 3.35,
            "negative_return_penalty": -negative_return_penalty * 3.20,
            "negative_sharpe_penalty": -negative_sharpe_penalty * 0.55,
            "budget_origin_sell_penalty": -budget_origin_sell_share * 2.80,
            "high_cash_budget_origin_sell_penalty": -high_cash_budget_origin_sell_share * 1.80,
            "deploy_funding_sell_share_penalty": -max(0.0, deploy_funding_rebalance_sell_share - 0.55) * 1.60,
            "deploy_funding_forward_penalty": -max(0.0, deploy_funding_rebalance_forward_excess_5d) * 8.00,
            "sell_source_floor_guard_penalty": -max(0.0, sell_source_floor_guard_share - 0.60) * 1.20,
            "order_translation_conflict_penalty": -order_translation_conflict_rate * 0.70,
            "add_to_hold_conflict_penalty": -add_to_hold_conflict_share * 1.10,
        }
        stability_breakdown = {
            "gate_pass_ratio": gate_pass_ratio * 0.68,
            "reduce_success_rate_5d": reduce_success * 0.48,
            "exit_timeliness_rate_5d": exit_timeliness * 0.52,
            "release_gate_forward_alignment_5d": _bounded(release_gate_alignment, -0.10, 0.10) * 0.56,
            "value_arbitration_forward_alignment_5d": _bounded(value_arbitration_alignment, -0.10, 0.12) * 0.46,
            "cash_timing_quality_1d": _bounded(cash_timing, -0.35, 0.12) * 0.44,
            "hold_share": hold_share * 0.14,
            "training_evidence_bonus": 0.34 if training_evidence_ok else -0.40,
            "reversal_penalty": -reversal * 1.10,
            "shadow_reversal_penalty": -shadow_reversal * 1.08,
            "reversal_excess_penalty": -reversal_excess_penalty,
            "drawdown_penalty": -abs(min(max_drawdown, 0.0)) * 5.10,
            "threshold_gap_penalty": -threshold_gap_penalty * 1.05,
            "budget_origin_sell_penalty": -budget_origin_sell_share * 3.10,
            "high_cash_budget_origin_sell_penalty": -high_cash_budget_origin_sell_share * 1.95,
            "deploy_funding_sell_share_penalty": -max(0.0, deploy_funding_rebalance_sell_share - 0.55) * 1.85,
            "deploy_funding_forward_penalty": -max(0.0, deploy_funding_rebalance_forward_excess_5d) * 8.80,
            "sell_source_floor_guard_penalty": -max(0.0, sell_source_floor_guard_share - 0.58) * 1.35,
            "semantic_conflict_penalty": -semantic_conflict_rate * 1.70,
            "order_translation_conflict_penalty": -order_translation_conflict_rate * 0.88,
            "add_to_hold_conflict_penalty": -add_to_hold_conflict_share * 1.20,
        }
    elif objective_profile_name == "sell_source_contract_v2":
        performance_breakdown = {
            "annual_return": annual_return * 2.20,
            "sharpe": sharpe * 0.28,
            "deploy_intent_realized_rate": deploy_intent_realized_rate * 1.02,
            "sell_selection_quality_5d": _bounded(sell_selection_quality, -0.06, 0.08) * 0.84,
            "reduce_success_rate_5d": reduce_success * 0.98,
            "exit_timeliness_rate_5d": exit_timeliness * 1.02,
            "release_gate_forward_alignment_5d": _bounded(release_gate_alignment, -0.10, 0.10) * 0.78,
            "value_arbitration_forward_alignment_5d": _bounded(value_arbitration_alignment, -0.10, 0.12) * 0.70,
            "cash_timing_quality_1d": _bounded(cash_timing, -0.35, 0.12) * 0.66,
            "active_alignment_bonus": active_alignment_bonus,
            "gate_pass_ratio": gate_pass_ratio * 0.32,
            "drawdown_penalty": -abs(min(max_drawdown, 0.0)) * 3.40,
            "negative_return_penalty": -negative_return_penalty * 3.30,
            "negative_sharpe_penalty": -negative_sharpe_penalty * 0.58,
            "budget_origin_sell_penalty": -budget_origin_sell_share * 2.90,
            "high_cash_budget_origin_sell_penalty": -high_cash_budget_origin_sell_share * 1.90,
            "deploy_funding_sell_share_penalty": -max(0.0, deploy_funding_rebalance_sell_share - 0.45) * 1.90,
            "deploy_funding_forward_penalty": -max(0.0, deploy_funding_rebalance_forward_excess_5d) * 9.50,
            "deploy_funding_against_hold_penalty": -(
                max(0.0, deploy_funding_against_protected_hold_share - 0.25) * 2.40
                if deploy_funding_rebalance_sell_count >= 5.0
                else 0.0
            ),
            "deploy_funding_release_consistency_penalty": -(
                max(0.0, 0.60 - deploy_funding_release_consistent_share) * 1.80
                if deploy_funding_rebalance_sell_count >= 5.0
                else 0.0
            ),
            "model_release_forward_penalty": -(
                max(0.0, model_release_signal_forward_excess_5d) * 6.50
                if model_release_signal_sell_count >= 5.0
                else 0.0
            ),
            "model_release_against_hold_penalty": -(
                max(0.0, model_release_against_protected_hold_share - 0.25) * 2.00
                if model_release_signal_sell_count >= 5.0
                else 0.0
            ),
            "model_release_consistency_penalty": -(
                max(0.0, 0.60 - model_release_release_consistent_share) * 1.40
                if model_release_signal_sell_count >= 5.0
                else 0.0
            ),
            "sell_source_floor_guard_penalty": -max(0.0, sell_source_floor_guard_share - 0.56) * 1.25,
            "order_translation_conflict_penalty": -order_translation_conflict_rate * 0.75,
            "add_to_hold_conflict_penalty": -add_to_hold_conflict_share * 1.10,
        }
        stability_breakdown = {
            "gate_pass_ratio": gate_pass_ratio * 0.72,
            "reduce_success_rate_5d": reduce_success * 0.50,
            "exit_timeliness_rate_5d": exit_timeliness * 0.54,
            "release_gate_forward_alignment_5d": _bounded(release_gate_alignment, -0.10, 0.10) * 0.60,
            "value_arbitration_forward_alignment_5d": _bounded(value_arbitration_alignment, -0.10, 0.12) * 0.52,
            "cash_timing_quality_1d": _bounded(cash_timing, -0.35, 0.12) * 0.46,
            "hold_share": hold_share * 0.14,
            "training_evidence_bonus": 0.34 if training_evidence_ok else -0.40,
            "reversal_penalty": -reversal * 1.10,
            "shadow_reversal_penalty": -shadow_reversal * 1.08,
            "reversal_excess_penalty": -reversal_excess_penalty,
            "drawdown_penalty": -abs(min(max_drawdown, 0.0)) * 5.20,
            "threshold_gap_penalty": -threshold_gap_penalty * 1.08,
            "budget_origin_sell_penalty": -budget_origin_sell_share * 3.20,
            "high_cash_budget_origin_sell_penalty": -high_cash_budget_origin_sell_share * 2.00,
            "deploy_funding_sell_share_penalty": -max(0.0, deploy_funding_rebalance_sell_share - 0.45) * 2.10,
            "deploy_funding_forward_penalty": -max(0.0, deploy_funding_rebalance_forward_excess_5d) * 10.20,
            "deploy_funding_against_hold_penalty": -(
                max(0.0, deploy_funding_against_protected_hold_share - 0.25) * 2.70
                if deploy_funding_rebalance_sell_count >= 5.0
                else 0.0
            ),
            "deploy_funding_release_consistency_penalty": -(
                max(0.0, 0.60 - deploy_funding_release_consistent_share) * 2.10
                if deploy_funding_rebalance_sell_count >= 5.0
                else 0.0
            ),
            "model_release_forward_penalty": -(
                max(0.0, model_release_signal_forward_excess_5d) * 7.10
                if model_release_signal_sell_count >= 5.0
                else 0.0
            ),
            "model_release_against_hold_penalty": -(
                max(0.0, model_release_against_protected_hold_share - 0.25) * 2.20
                if model_release_signal_sell_count >= 5.0
                else 0.0
            ),
            "model_release_consistency_penalty": -(
                max(0.0, 0.60 - model_release_release_consistent_share) * 1.60
                if model_release_signal_sell_count >= 5.0
                else 0.0
            ),
            "sell_source_floor_guard_penalty": -max(0.0, sell_source_floor_guard_share - 0.55) * 1.45,
            "semantic_conflict_penalty": -semantic_conflict_rate * 1.75,
            "order_translation_conflict_penalty": -order_translation_conflict_rate * 0.92,
            "add_to_hold_conflict_penalty": -add_to_hold_conflict_share * 1.25,
        }
    elif objective_profile_name == "release_translation_deploy_v1":
        performance_breakdown = {
            "annual_return": annual_return * 2.08,
            "sharpe": sharpe * 0.26,
            "release_translation_deploy_health_score": release_translation_deploy_health_score * 1.58,
            "release_translation_deploy_deploy_score": release_translation_deploy_deploy_score * 0.86,
            "release_translation_deploy_release_score": release_translation_deploy_release_score * 1.06,
            "release_translation_deploy_translation_score": release_translation_deploy_translation_score * 0.64,
            "release_translation_deploy_funding_score": release_translation_deploy_funding_score * 0.90,
            "release_translation_deploy_model_release_score": release_translation_deploy_model_release_score * 0.40,
            "deploy_intent_realized_rate": deploy_intent_realized_rate * 0.88,
            "deploy_executability_forward_alignment_5d": _bounded(deploy_executability_alignment, -0.25, 0.20) * 0.58,
            "deploy_gate_forward_alignment_5d": _bounded(deploy_gate_alignment, -0.25, 0.20) * 0.54,
            "release_gate_forward_alignment_5d": _bounded(release_gate_alignment, -0.10, 0.10) * 0.76,
            "value_arbitration_forward_alignment_5d": _bounded(value_arbitration_alignment, -0.10, 0.12) * 0.66,
            "sell_selection_quality_5d": _bounded(sell_selection_quality, -0.06, 0.08) * 0.78,
            "reduce_success_rate_5d": reduce_success * 0.80,
            "exit_timeliness_rate_5d": exit_timeliness * 0.82,
            "cash_timing_quality_1d": _bounded(cash_timing, -0.35, 0.12) * 0.54,
            "active_alignment_bonus": active_alignment_bonus,
            "gate_pass_ratio": gate_pass_ratio * 0.30,
            "drawdown_penalty": -abs(min(max_drawdown, 0.0)) * 3.45,
            "negative_return_penalty": -negative_return_penalty * 3.20,
            "negative_sharpe_penalty": -negative_sharpe_penalty * 0.56,
            "budget_origin_sell_penalty": -budget_origin_sell_share * 2.95,
            "high_cash_budget_origin_sell_penalty": -high_cash_budget_origin_sell_share * 1.85,
            "deploy_funding_sell_share_penalty": -max(0.0, deploy_funding_rebalance_sell_share - 0.42) * 2.20,
            "deploy_funding_forward_penalty": -max(0.0, deploy_funding_rebalance_forward_excess_5d) * 10.50,
            "deploy_funding_against_hold_penalty": -(
                max(0.0, deploy_funding_against_protected_hold_share - 0.20) * 3.05
                if deploy_funding_rebalance_sell_count >= 5.0
                else 0.0
            ),
            "deploy_funding_release_consistency_penalty": -(
                max(0.0, 0.58 - deploy_funding_release_consistent_share) * 2.35
                if deploy_funding_rebalance_sell_count >= 5.0
                else 0.0
            ),
            "model_release_forward_penalty": -(
                max(0.0, model_release_signal_forward_excess_5d) * 7.40
                if model_release_signal_sell_count >= 5.0
                else 0.0
            ),
            "model_release_against_hold_penalty": -(
                max(0.0, model_release_against_protected_hold_share - 0.20) * 2.45
                if model_release_signal_sell_count >= 5.0
                else 0.0
            ),
            "model_release_consistency_penalty": -(
                max(0.0, 0.58 - model_release_release_consistent_share) * 1.85
                if model_release_signal_sell_count >= 5.0
                else 0.0
            ),
            "sell_source_floor_guard_penalty": -max(0.0, sell_source_floor_guard_share - 0.54) * 1.35,
            "semantic_conflict_penalty": -semantic_conflict_rate * 1.85,
            "order_translation_conflict_penalty": -order_translation_conflict_rate * 1.18,
            "add_to_hold_conflict_penalty": -add_to_hold_conflict_share * 1.38,
            "deploy_hold_conflict_penalty": -deploy_intent_hold_conflict_share * 1.15,
            "deploy_dropped_penalty": -deploy_intent_dropped_share * 1.18,
            "deploy_candidate_budget_drop_penalty": -deploy_intent_candidate_budget_drop_share * 1.10,
        }
        stability_breakdown = {
            "gate_pass_ratio": gate_pass_ratio * 0.74,
            "release_translation_deploy_health_score": release_translation_deploy_health_score * 1.72,
            "release_translation_deploy_deploy_score": release_translation_deploy_deploy_score * 0.72,
            "release_translation_deploy_release_score": release_translation_deploy_release_score * 1.18,
            "release_translation_deploy_translation_score": release_translation_deploy_translation_score * 0.86,
            "release_translation_deploy_funding_score": release_translation_deploy_funding_score * 1.02,
            "release_translation_deploy_model_release_score": release_translation_deploy_model_release_score * 0.46,
            "deploy_intent_realized_rate": deploy_intent_realized_rate * 0.70,
            "deploy_executability_forward_alignment_5d": _bounded(deploy_executability_alignment, -0.25, 0.20) * 0.48,
            "release_gate_forward_alignment_5d": _bounded(release_gate_alignment, -0.10, 0.10) * 0.62,
            "value_arbitration_forward_alignment_5d": _bounded(value_arbitration_alignment, -0.10, 0.12) * 0.54,
            "reduce_success_rate_5d": reduce_success * 0.46,
            "exit_timeliness_rate_5d": exit_timeliness * 0.48,
            "cash_timing_quality_1d": _bounded(cash_timing, -0.35, 0.12) * 0.42,
            "hold_share": hold_share * 0.14,
            "training_evidence_bonus": 0.34 if training_evidence_ok else -0.42,
            "reversal_penalty": -reversal * 1.10,
            "shadow_reversal_penalty": -shadow_reversal * 1.08,
            "reversal_excess_penalty": -reversal_excess_penalty,
            "drawdown_penalty": -abs(min(max_drawdown, 0.0)) * 5.25,
            "threshold_gap_penalty": -threshold_gap_penalty * 1.10,
            "budget_origin_sell_penalty": -budget_origin_sell_share * 3.30,
            "high_cash_budget_origin_sell_penalty": -high_cash_budget_origin_sell_share * 2.05,
            "deploy_funding_sell_share_penalty": -max(0.0, deploy_funding_rebalance_sell_share - 0.42) * 2.40,
            "deploy_funding_forward_penalty": -max(0.0, deploy_funding_rebalance_forward_excess_5d) * 11.20,
            "deploy_funding_against_hold_penalty": -(
                max(0.0, deploy_funding_against_protected_hold_share - 0.20) * 3.35
                if deploy_funding_rebalance_sell_count >= 5.0
                else 0.0
            ),
            "deploy_funding_release_consistency_penalty": -(
                max(0.0, 0.58 - deploy_funding_release_consistent_share) * 2.60
                if deploy_funding_rebalance_sell_count >= 5.0
                else 0.0
            ),
            "model_release_forward_penalty": -(
                max(0.0, model_release_signal_forward_excess_5d) * 7.80
                if model_release_signal_sell_count >= 5.0
                else 0.0
            ),
            "model_release_against_hold_penalty": -(
                max(0.0, model_release_against_protected_hold_share - 0.20) * 2.70
                if model_release_signal_sell_count >= 5.0
                else 0.0
            ),
            "model_release_consistency_penalty": -(
                max(0.0, 0.58 - model_release_release_consistent_share) * 2.00
                if model_release_signal_sell_count >= 5.0
                else 0.0
            ),
            "sell_source_floor_guard_penalty": -max(0.0, sell_source_floor_guard_share - 0.52) * 1.55,
            "semantic_conflict_penalty": -semantic_conflict_rate * 2.05,
            "order_translation_conflict_penalty": -order_translation_conflict_rate * 1.42,
            "add_to_hold_conflict_penalty": -add_to_hold_conflict_share * 1.50,
            "deploy_hold_conflict_penalty": -deploy_intent_hold_conflict_share * 1.30,
            "deploy_dropped_penalty": -deploy_intent_dropped_share * 1.30,
            "deploy_candidate_budget_drop_penalty": -deploy_intent_candidate_budget_drop_share * 1.22,
        }
    elif objective_profile_name == "action_value_unification_v1":
        action_alignment_score = (
            _bounded(multi_horizon_path_alignment, -0.10, 0.16) * 0.30
            + _bounded(open_action_value_alignment, -0.10, 0.16) * 0.22
            + _bounded(add_action_value_alignment, -0.10, 0.16) * 0.18
            + _bounded(hold_action_value_alignment, -0.10, 0.16) * 0.18
            + _bounded(reduce_action_value_avoidance, -0.10, 0.16) * 0.06
            + _bounded(exit_action_value_avoidance, -0.10, 0.16) * 0.04
            + _bounded(value_arbitration_alignment, -0.10, 0.12) * 0.02
        )
        action_conflict_penalty = (
            action_value_conflict_share * 1.45
            + sell_against_keep_value_share * 1.05
            + keep_against_release_value_share * 0.72
            + open_low_action_value_share * 0.46
        )
        performance_breakdown = {
            "annual_return": annual_return * 1.76,
            "sharpe": sharpe * 0.24,
            "action_value_consistency_score": action_value_consistency_score * 1.72,
            "action_value_alignment_score": action_alignment_score * 1.10,
            "multi_horizon_path_value_alignment_5d": _bounded(multi_horizon_path_alignment, -0.10, 0.16) * 0.82,
            "open_action_value_forward_alignment_5d": _bounded(open_action_value_alignment, -0.10, 0.16) * 0.58,
            "add_action_value_forward_alignment_5d": _bounded(add_action_value_alignment, -0.10, 0.16) * 0.46,
            "hold_action_value_forward_alignment_5d": _bounded(hold_action_value_alignment, -0.10, 0.16) * 0.46,
            "reduce_action_value_forward_avoidance_5d": _bounded(reduce_action_value_avoidance, -0.10, 0.16) * 0.34,
            "exit_action_value_forward_avoidance_5d": _bounded(exit_action_value_avoidance, -0.10, 0.16) * 0.30,
            "multi_horizon_forward_risk_avoidance_5d": _bounded(multi_horizon_forward_risk_avoidance, -0.10, 0.16) * 0.24,
            "release_translation_deploy_health_score": release_translation_deploy_health_score * 0.86,
            "release_translation_deploy_translation_score": release_translation_deploy_translation_score * 0.56,
            "deploy_intent_realized_rate": deploy_intent_realized_rate * 0.70,
            "reduce_success_rate_5d": reduce_success * 0.72,
            "exit_timeliness_rate_5d": exit_timeliness * 0.74,
            "sell_selection_quality_5d": _bounded(sell_selection_quality, -0.06, 0.08) * 0.54,
            "cash_timing_quality_1d": _bounded(cash_timing, -0.35, 0.12) * 0.38,
            "active_alignment_bonus": active_alignment_bonus,
            "gate_pass_ratio": gate_pass_ratio * 0.24,
            "drawdown_penalty": -abs(min(max_drawdown, 0.0)) * 3.65,
            "negative_return_penalty": -negative_return_penalty * 3.00,
            "negative_sharpe_penalty": -negative_sharpe_penalty * 0.52,
            "action_value_conflict_penalty": -action_conflict_penalty,
            "order_translation_conflict_penalty": -order_translation_conflict_rate * 1.28,
            "add_to_hold_conflict_penalty": -add_to_hold_conflict_share * 1.12,
            "deploy_hold_conflict_penalty": -deploy_intent_hold_conflict_share * 1.02,
            "deploy_candidate_budget_drop_penalty": -deploy_intent_candidate_budget_drop_share * 0.82,
            "deploy_funding_release_consistency_penalty": -(
                max(0.0, 0.54 - deploy_funding_release_consistent_share) * 1.70
                if deploy_funding_rebalance_sell_count >= 5.0
                else 0.0
            ),
        }
        stability_breakdown = {
            "gate_pass_ratio": gate_pass_ratio * 0.58,
            "action_value_consistency_score": action_value_consistency_score * 1.92,
            "action_value_alignment_score": action_alignment_score * 0.86,
            "held_keep_release_value_gap": _bounded(held_keep_release_value_gap, -0.20, 0.20) * 0.46,
            "release_translation_deploy_health_score": release_translation_deploy_health_score * 0.94,
            "release_translation_deploy_translation_score": release_translation_deploy_translation_score * 0.70,
            "deploy_intent_realized_rate": deploy_intent_realized_rate * 0.56,
            "reduce_success_rate_5d": reduce_success * 0.44,
            "exit_timeliness_rate_5d": exit_timeliness * 0.46,
            "cash_timing_quality_1d": _bounded(cash_timing, -0.35, 0.12) * 0.34,
            "hold_share": hold_share * 0.16,
            "training_evidence_bonus": 0.34 if training_evidence_ok else -0.42,
            "reversal_penalty": -reversal * 1.05,
            "shadow_reversal_penalty": -shadow_reversal * 1.02,
            "reversal_excess_penalty": -reversal_excess_penalty,
            "drawdown_penalty": -abs(min(max_drawdown, 0.0)) * 5.30,
            "threshold_gap_penalty": -threshold_gap_penalty * 0.92,
            "action_value_conflict_penalty": -action_conflict_penalty * 1.18,
            "semantic_conflict_penalty": -semantic_conflict_rate * 1.64,
            "order_translation_conflict_penalty": -order_translation_conflict_rate * 1.54,
            "deploy_funding_forward_penalty": -max(0.0, deploy_funding_rebalance_forward_excess_5d) * 9.20,
            "deploy_funding_against_hold_penalty": -(
                max(0.0, deploy_funding_against_protected_hold_share - 0.22) * 2.50
                if deploy_funding_rebalance_sell_count >= 5.0
                else 0.0
            ),
        }
    elif objective_profile_name in {
        "direct_action_translation_v1",
        "direct_action_reallocation_v1",
        "direct_action_pair_reallocation_v1",
        "direct_action_pair_cost_guard_v1",
        "portfolio_daily_ranking_v1",
        "portfolio_daily_ranking_v2_gated",
        "end_to_end_allocation_layer_v1",
    }:
        direct_reallocation_objective = objective_profile_name == "direct_action_reallocation_v1"
        portfolio_daily_ranking_objective = objective_profile_name in {
            "portfolio_daily_ranking_v1",
            "portfolio_daily_ranking_v2_gated",
            "end_to_end_allocation_layer_v1",
        }
        direct_pair_cost_guard_objective = objective_profile_name in {
            "direct_action_pair_cost_guard_v1",
            "portfolio_daily_ranking_v1",
            "portfolio_daily_ranking_v2_gated",
        }
        direct_pair_reallocation_objective = objective_profile_name in {
            "direct_action_pair_reallocation_v1",
            "direct_action_pair_cost_guard_v1",
            "portfolio_daily_ranking_v1",
            "portfolio_daily_ranking_v2_gated",
        }
        action_alignment_score = (
            _bounded(multi_horizon_path_alignment, -0.10, 0.16) * 0.24
            + _bounded(open_action_value_alignment, -0.10, 0.16) * 0.16
            + _bounded(add_action_value_alignment, -0.10, 0.16) * 0.14
            + _bounded(hold_action_value_alignment, -0.10, 0.16) * 0.18
            + _bounded(reduce_action_value_avoidance, -0.10, 0.16) * 0.14
            + _bounded(exit_action_value_avoidance, -0.10, 0.16) * 0.14
        )
        direct_translation_penalty = max(
            order_translation_conflict_rate,
            direct_action_order_translation_conflict_rate,
        )
        funding_authorization_gap = (
            max(0.0, 0.62 - direct_action_funding_authorized_sell_share)
            if deploy_funding_rebalance_sell_count >= 5.0
            else 0.0
        )
        direct_funding_protected_penalty = (
            direct_action_funding_protected_sell_share
            if deploy_funding_rebalance_sell_count >= 5.0
            else 0.0
        )
        direct_mode_activation_penalty = max(0.0, 0.94 - direct_action_value_mode_share) * 1.20
        direct_conflict_penalty = (
            action_value_conflict_share * 1.12
            + open_low_action_value_share * 0.82
            + sell_against_keep_value_share * 0.82
            + keep_against_release_value_share * 0.78
            + direct_action_value_low_margin_share * 0.42
        )
        direct_authorized_deploy_gap = (
            max(0.0, 0.70 - direct_action_deploy_authorized_realized_rate)
            if direct_action_deploy_authorized_count >= 5.0
            else 0.0
        )
        direct_authorized_add_gap = (
            max(0.0, 0.62 - direct_action_add_authorized_realized_rate)
            if direct_action_add_authorized_count >= 3.0
            else 0.0
        )
        direct_reallocation_source_sparse_penalty = (
            max(0.0, min(direct_action_deploy_authorized_count * 0.35, 5.0) - direct_action_reallocation_source_count)
            * 0.12
            if direct_action_deploy_authorized_count >= 5.0
            else 0.0
        )
        direct_reallocation_weight = 1.0 if (direct_reallocation_objective or direct_pair_reallocation_objective) else 0.0
        direct_pair_reallocation_weight = 1.0 if direct_pair_reallocation_objective else 0.0
        direct_pair_cost_guard_weight = 1.0 if direct_pair_cost_guard_objective else 0.0
        portfolio_daily_ranking_weight = 1.0 if portfolio_daily_ranking_objective else 0.0
        direct_pair_source_observed = (
            direct_action_pair_reallocation_source_count >= 3.0
            or direct_action_pair_cost_guard_pass_count + direct_action_pair_cost_guard_blocked_count >= 3.0
        )
        portfolio_daily_observed = (
            portfolio_daily_receiver_target_count >= 3.0
            or portfolio_daily_source_target_count >= 3.0
        )
        direct_pair_source_cost_penalty = (
            max(0.0, direct_action_pair_source_cost_mean - 0.740) * 3.80
            if direct_pair_source_observed
            else 0.0
        )
        direct_pair_forward_penalty = (
            max(0.0, direct_action_pair_source_forward_excess_5d) * 9.60
            if direct_action_pair_reallocation_source_count >= 3.0
            else 0.0
        )
        direct_pair_relative_forward_penalty = (
            max(0.0, -direct_action_core_minus_pair_forward_excess_5d) * 12.00
            if direct_action_pair_reallocation_source_count >= 3.0 and direct_action_core_deploy_target_count >= 3.0
            else 0.0
        )
        direct_pair_spread_gap_penalty = (
            max(0.0, 0.006 - direct_action_pair_source_spread_mean) * 18.00
            if direct_action_pair_reallocation_source_count >= 3.0
            else 0.0
        )
        portfolio_daily_relative_forward_penalty = (
            max(0.0, -portfolio_daily_receiver_minus_source_forward_excess_5d) * 14.00
            if portfolio_daily_receiver_target_count >= 3.0 and portfolio_daily_source_target_count >= 3.0
            else 0.0
        )
        portfolio_daily_source_forward_penalty = (
            max(0.0, portfolio_daily_source_forward_excess_5d) * 10.50
            if portfolio_daily_source_target_count >= 3.0
            else 0.0
        )
        portfolio_daily_source_positive_distribution_penalty = (
            max(0.0, portfolio_daily_source_positive_forward_sell_share - 0.45) * 1.90
            if portfolio_daily_source_target_count >= 3.0
            else 0.0
        )
        portfolio_daily_source_strong_false_sell_penalty = (
            (
                min(portfolio_daily_source_strong_positive_forward_sell_count, 4.0) * 0.22
                + max(0.0, portfolio_daily_source_max_forward_excess_5d - 0.080) * 8.00
                + max(0.0, portfolio_daily_source_p75_forward_excess_5d - 0.030) * 6.00
            )
            if portfolio_daily_source_target_count >= 3.0
            else 0.0
        )
        portfolio_daily_source_realization_gap = (
            max(0.0, 0.58 - portfolio_daily_source_realized_sell_rate)
            if portfolio_daily_source_target_count >= 3.0
            else 0.0
        )
        portfolio_daily_source_not_sold_excess = (
            max(0.0, portfolio_daily_source_target_not_sold_share - 0.50)
            if portfolio_daily_source_target_count >= 3.0
            else 0.0
        )
        portfolio_daily_effective_transfer_sparse_penalty = (
            max(
                0.0,
                min(portfolio_daily_receiver_target_count, portfolio_daily_source_target_count) * 0.30
                - portfolio_daily_effective_capital_transfer_count,
            )
            * 0.09
            if portfolio_daily_receiver_target_count >= 3.0 and portfolio_daily_source_target_count >= 3.0
            else 0.0
        )
        portfolio_daily_receiver_activity_gap = (
            max(0.0, 3.0 - portfolio_daily_receiver_target_count)
            if portfolio_daily_ranking_objective
            else 0.0
        )
        portfolio_daily_source_activity_gap = (
            max(0.0, 3.0 - portfolio_daily_source_target_count)
            if portfolio_daily_ranking_objective and portfolio_daily_receiver_target_count >= 3.0
            else 0.0
        )
        portfolio_daily_dead_allocation_branch = (
            1.0
            if (
                portfolio_daily_ranking_objective
                and portfolio_daily_receiver_target_count < 3.0
                and portfolio_daily_source_target_count < 1.0
                and portfolio_daily_cash_reserve_rate >= 0.75
                and annual_return < 0.05
            )
            else 0.0
        )
        portfolio_daily_source_sparse_penalty = (
            max(0.0, min(portfolio_daily_receiver_target_count * 0.65, 6.0) - portfolio_daily_source_target_count)
            * 0.12
            if portfolio_daily_receiver_target_count >= 5.0 and portfolio_daily_cash_score_mean < 0.62
            else 0.0
        )
        portfolio_daily_cash_drag_penalty = (
            max(0.0, portfolio_daily_cash_reserve_rate - 0.26) * 1.25
            if annual_return < 0.12 and portfolio_daily_ranking_objective
            else 0.0
        )
        portfolio_daily_funding_coverage_gap = (
            max(0.0, 0.18 - portfolio_daily_receiver_funding_coverage_mean)
            if portfolio_daily_receiver_target_count >= 3.0
            else 0.0
        )
        portfolio_daily_closure_gap = (
            max(0.0, 0.20 - portfolio_daily_funding_closure_score_mean)
            if portfolio_daily_observed
            else 0.0
        )
        portfolio_daily_transfer_gap = (
            max(0.0, 0.18 - portfolio_daily_allocation_transfer_score_mean)
            if portfolio_daily_observed
            else 0.0
        )
        portfolio_daily_economic_release_gap = (
            max(0.0, 0.16 - portfolio_daily_source_economic_release_score_mean)
            if portfolio_daily_source_target_count >= 3.0
            else 0.0
        )
        portfolio_daily_bad_forward_spread_risk_penalty = (
            max(0.0, portfolio_daily_source_bad_forward_spread_risk_mean - 0.48)
            if portfolio_daily_source_target_count >= 3.0
            else 0.0
        )
        portfolio_daily_economic_block_risk_penalty = (
            max(0.0, portfolio_daily_source_economic_block_risk_mean - 0.56)
            if portfolio_daily_source_target_count >= 3.0
            else 0.0
        )
        portfolio_daily_forward_strength_brake_risk_penalty = (
            max(0.0, portfolio_daily_source_forward_strength_brake_risk_mean - 0.52)
            if portfolio_daily_source_target_count >= 3.0
            else 0.0
        )
        portfolio_daily_forward_proxy_keep_risk_penalty = (
            max(0.0, portfolio_daily_source_forward_proxy_keep_risk_mean - 0.260)
            if portfolio_daily_source_target_count >= 3.0
            else 0.0
        )
        portfolio_daily_source_release_conviction_gap = (
            max(0.0, 0.300 - portfolio_daily_source_release_conviction_mean)
            if portfolio_daily_source_target_count >= 3.0
            else 0.0
        )
        portfolio_daily_dead_branch_risk_penalty = (
            max(0.0, portfolio_daily_allocation_dead_branch_risk_mean - 0.46)
            if portfolio_daily_ranking_objective
            else 0.0
        )
        portfolio_daily_receiver_semantic_bypass_penalty = (
            authorized_add_no_weight_change_share * 2.80
            + min(direct_action_authorization_subset_violation_count, 6.0) * 0.45
            + min(portfolio_daily_receiver_semantic_no_headroom_count, 12.0) * 0.045
            if portfolio_daily_ranking_objective
            else 0.0
        )
        portfolio_daily_unrealized_deploy_penalty = (
            max(0.0, deploy_intent_unrealized_share - 0.15) * 2.20
            if portfolio_daily_ranking_objective and deploy_intent_action_count >= 3.0
            else 0.0
        )
        portfolio_daily_open_breadth_bonus = (
            min(portfolio_daily_receiver_open_breadth_candidate_count, 8.0) * 0.020
            if portfolio_daily_ranking_objective and portfolio_daily_receiver_target_count >= 3.0
            else 0.0
        )
        portfolio_daily_receiver_candidate_breadth = (
            _bounded(portfolio_daily_receiver_candidate_count, 3.0, 9.0)
            if portfolio_daily_ranking_objective
            else 0.0
        )
        source_clean_quality = (
            min(max(portfolio_daily_source_distribution_clean_pass_mean, 0.0), 1.0) * 0.42
            + (1.0 - _bounded(portfolio_daily_source_forward_proxy_keep_risk_mean, 0.16, 0.46)) * 0.24
            + _bounded(portfolio_daily_source_release_conviction_mean, 0.24, 0.48) * 0.24
            + (1.0 - _bounded(portfolio_daily_source_positive_forward_sell_share, 0.20, 0.55)) * 0.10
        )
        portfolio_daily_clean_source_candidate_breadth = (
            _bounded(portfolio_daily_source_candidate_count, 3.0, 9.0)
            * float(min(max(source_clean_quality, 0.0), 1.0))
            if portfolio_daily_ranking_objective
            else 0.0
        )
        portfolio_daily_exposure_utilization_gap = (
            max(0.0, 0.52 - portfolio_daily_exposure_utilization)
            if portfolio_daily_ranking_objective and avg_gross_exposure_target >= 0.42
            else 0.0
        )
        portfolio_daily_joint_economic_quality_gap = (
            max(0.0, 0.006 - monthly_return_mean) * 12.0
            + max(0.0, 0.002 - portfolio_daily_receiver_minus_source_forward_excess_5d) * 10.0
            + (
                max(0.0, 0.82 - portfolio_daily_receiver_realized_deploy_rate) * 0.42
                if portfolio_daily_receiver_target_count >= 3.0
                else 0.0
            )
            + portfolio_daily_exposure_utilization_gap * 0.70
            + max(0.0, abs(min(max_drawdown, 0.0)) - 0.14) * 1.80
            if portfolio_daily_ranking_objective
            else 0.0
        )
        performance_breakdown = {
            "annual_return": annual_return * 1.52,
            "sharpe": sharpe * 0.22,
            "monthly_return_mean_annualized_focus": monthly_return_mean * 12.0 * 0.44,
            "monthly_win_rate_focus": monthly_win_rate * 0.28,
            "monthly_consistency_focus": monthly_consistency_score * 0.28,
            "direct_action_value_mode_share": direct_action_value_mode_share * 0.58,
            "direct_action_intent_preserved_share": direct_action_intent_preserved_share * 0.84,
            "direct_action_value_gap_mean": _bounded(direct_action_value_gap_mean, 0.00, 0.24) * 0.48,
            "direct_action_release_advantage_mean": _bounded(direct_action_release_advantage_mean, -0.08, 0.12) * 0.22,
            "direct_action_deploy_advantage_mean": _bounded(direct_action_deploy_advantage_mean, -0.08, 0.18)
            * 0.20
            * direct_reallocation_weight,
            "direct_action_deploy_authorized_realized_rate": direct_action_deploy_authorized_realized_rate
            * 0.74
            * direct_reallocation_weight,
            "direct_action_core_deploy_target_realized_rate": direct_action_core_deploy_target_realized_rate
            * 0.86
            * direct_pair_reallocation_weight,
            "direct_action_add_authorized_realized_rate": direct_action_add_authorized_realized_rate
            * 0.40
            * direct_reallocation_weight,
            "direct_action_pair_reallocation_source_count": min(direct_action_pair_reallocation_source_count, 12.0)
            * 0.025
            * direct_pair_reallocation_weight,
            "direct_action_pair_cost_guard_pass_rate": direct_action_pair_cost_guard_pass_rate
            * 0.22
            * direct_pair_cost_guard_weight,
            "direct_action_pair_source_spread_mean": _bounded(direct_action_pair_source_spread_mean, -0.010, 0.040)
            * 0.62
            * direct_pair_cost_guard_weight,
            "direct_action_core_minus_pair_forward_excess_5d": _bounded(
                direct_action_core_minus_pair_forward_excess_5d,
                -0.015,
                0.045,
            )
            * 0.88
            * direct_pair_cost_guard_weight,
            "direct_action_pair_source_cost_quality": (
                (1.0 - _bounded(direct_action_pair_source_cost_mean, 0.620, 0.880))
                if direct_pair_source_observed
                else 0.0
            )
            * 0.44
            * direct_pair_cost_guard_weight,
            "portfolio_daily_receiver_minus_source_forward_excess_5d": _bounded(
                portfolio_daily_receiver_minus_source_forward_excess_5d,
                -0.015,
                0.050,
            )
            * 1.10
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_gap_mean": _bounded(portfolio_daily_source_gap_mean, -0.020, 0.080)
            * 0.48
            * portfolio_daily_ranking_weight,
            "portfolio_daily_receiver_forward_excess_5d": _bounded(
                portfolio_daily_receiver_forward_excess_5d,
                -0.020,
                0.050,
            )
            * 0.42
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_realized_sell_rate": portfolio_daily_source_realized_sell_rate
            * 0.34
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_forward_spread_score_mean": _bounded(
                portfolio_daily_source_forward_spread_score_mean,
                0.04,
                0.34,
            )
            * 0.22
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_economic_release_score_mean": _bounded(
                portfolio_daily_source_economic_release_score_mean,
                0.04,
                0.36,
            )
            * 0.30
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_bad_forward_spread_risk_quality": (
                1.0 - _bounded(portfolio_daily_source_bad_forward_spread_risk_mean, 0.30, 0.72)
            )
            * 0.18
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_economic_block_risk_quality": (
                1.0 - _bounded(portfolio_daily_source_economic_block_risk_mean, 0.28, 0.72)
            )
            * 0.20
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_forward_strength_brake_quality": (
                1.0 - _bounded(portfolio_daily_source_forward_strength_brake_risk_mean, 0.24, 0.68)
            )
            * 0.22
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_forward_proxy_keep_risk_quality": (
                1.0 - _bounded(portfolio_daily_source_forward_proxy_keep_risk_mean, 0.16, 0.46)
            )
            * 0.24
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_release_conviction_mean": _bounded(
                portfolio_daily_source_release_conviction_mean,
                0.24,
                0.48,
            )
            * 0.22
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_direct_release_relief_score_mean": _bounded(
                portfolio_daily_source_direct_release_relief_score_mean,
                0.08,
                0.46,
            )
            * 0.12
            * portfolio_daily_ranking_weight,
            "portfolio_daily_effective_capital_transfer_count": min(
                portfolio_daily_effective_capital_transfer_count,
                10.0,
            )
            * 0.020
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_exec_cap_guard_count": min(portfolio_daily_source_exec_cap_guard_count, 10.0)
            * 0.010
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_count": min(portfolio_daily_source_target_count, 14.0)
            * 0.018
            * portfolio_daily_ranking_weight,
            "portfolio_daily_receiver_candidate_breadth": portfolio_daily_receiver_candidate_breadth
            * 0.11
            * portfolio_daily_ranking_weight,
            "portfolio_daily_clean_source_candidate_breadth": portfolio_daily_clean_source_candidate_breadth
            * 0.13
            * portfolio_daily_ranking_weight,
            "portfolio_daily_receiver_exec_guard_count": min(portfolio_daily_receiver_exec_guard_count, 20.0)
            * 0.010
            * portfolio_daily_ranking_weight,
            "portfolio_daily_receiver_realized_deploy_rate": portfolio_daily_receiver_realized_deploy_rate
            * 0.18
            * portfolio_daily_ranking_weight,
            "portfolio_daily_open_breadth_candidate_count": portfolio_daily_open_breadth_bonus
            * portfolio_daily_ranking_weight,
            "portfolio_daily_exposure_utilization": _bounded(
                portfolio_daily_exposure_utilization,
                0.35,
                0.86,
            )
            * 0.18
            * portfolio_daily_ranking_weight,
            "portfolio_daily_receiver_funding_coverage_mean": _bounded(
                portfolio_daily_receiver_funding_coverage_mean,
                0.06,
                0.38,
            )
            * 0.24
            * portfolio_daily_ranking_weight,
            "portfolio_daily_funding_closure_score_mean": _bounded(
                portfolio_daily_funding_closure_score_mean,
                0.08,
                0.42,
            )
            * 0.22
            * portfolio_daily_ranking_weight,
            "portfolio_daily_allocation_transfer_score_mean": _bounded(
                portfolio_daily_allocation_transfer_score_mean,
                0.08,
                0.42,
            )
            * 0.26
            * portfolio_daily_ranking_weight,
            "portfolio_daily_unified_allocation_objective": _bounded(
                portfolio_daily_unified_allocation_objective,
                0.10,
                0.62,
            )
            * 0.34
            * portfolio_daily_ranking_weight,
            "portfolio_daily_unified_solution_objective": _bounded(
                portfolio_daily_unified_solution_objective,
                0.02,
                0.16,
            )
            * 0.18
            * portfolio_daily_ranking_weight,
            "portfolio_daily_unified_receiver_score": _bounded(
                portfolio_daily_unified_receiver_score,
                0.08,
                0.52,
            )
            * 0.12
            * portfolio_daily_ranking_weight,
            "portfolio_daily_unified_source_score": _bounded(
                portfolio_daily_unified_source_score,
                0.04,
                0.42,
            )
            * 0.16
            * portfolio_daily_ranking_weight,
            "portfolio_daily_receiver_source_spread_reward": _bounded(
                portfolio_daily_receiver_source_spread_reward,
                0.08,
                0.58,
            )
            * 0.20
            * portfolio_daily_ranking_weight,
            "portfolio_daily_allocation_dead_branch_risk_quality": (
                1.0 - _bounded(portfolio_daily_allocation_dead_branch_risk_mean, 0.34, 0.78)
            )
            * 0.18
            * portfolio_daily_ranking_weight,
            "action_value_consistency_score": action_value_consistency_score * 1.08,
            "action_value_alignment_score": action_alignment_score * 0.92,
            "release_translation_deploy_health_score": release_translation_deploy_health_score * 0.96,
            "release_translation_deploy_translation_score": release_translation_deploy_translation_score * 0.82,
            "release_translation_deploy_funding_score": release_translation_deploy_funding_score * 0.94,
            "deploy_intent_realized_rate": deploy_intent_realized_rate * 0.44,
            "reduce_success_rate_5d": reduce_success * 0.72,
            "exit_timeliness_rate_5d": exit_timeliness * 0.74,
            "sell_selection_quality_5d": _bounded(sell_selection_quality, -0.06, 0.08) * 0.52,
            "cash_timing_quality_1d": _bounded(cash_timing, -0.35, 0.12) * 0.34,
            "active_alignment_bonus": active_alignment_bonus,
            "gate_pass_ratio": gate_pass_ratio * 0.20,
            "drawdown_penalty": -abs(min(max_drawdown, 0.0)) * 4.05,
            "monthly_worst_return_focus_penalty": -monthly_worst_loss * 1.34,
            "negative_return_penalty": -negative_return_penalty * 3.10,
            "negative_sharpe_penalty": -negative_sharpe_penalty * 0.52,
            "direct_mode_activation_penalty": -direct_mode_activation_penalty,
            "direct_action_conflict_penalty": -direct_conflict_penalty,
            "direct_translation_penalty": -direct_translation_penalty * 1.92,
            "deploy_candidate_budget_drop_penalty": -deploy_intent_candidate_budget_drop_share * 0.90,
            "deploy_funding_sell_share_penalty": -max(0.0, deploy_funding_rebalance_sell_share - 0.30) * 2.45,
            "deploy_funding_authorization_gap_penalty": -funding_authorization_gap * 2.85,
            "direct_funding_protected_sell_penalty": -direct_funding_protected_penalty * 2.60,
            "direct_authorized_deploy_gap_penalty": -direct_authorized_deploy_gap
            * 2.35
            * direct_reallocation_weight,
            "direct_authorized_add_gap_penalty": -direct_authorized_add_gap * 1.25 * direct_reallocation_weight,
            "direct_reallocation_source_sparse_penalty": -direct_reallocation_source_sparse_penalty
            * direct_reallocation_weight,
            "direct_core_deploy_target_gap_penalty": -(
                max(0.0, 0.74 - direct_action_core_deploy_target_realized_rate) * 1.85
                if direct_action_core_deploy_target_count >= 3.0
                else 0.0
            )
            * direct_pair_reallocation_weight,
            "direct_pair_source_cost_penalty": -direct_pair_source_cost_penalty * direct_pair_cost_guard_weight,
            "direct_pair_source_forward_penalty": -direct_pair_forward_penalty * direct_pair_cost_guard_weight,
            "direct_pair_relative_forward_penalty": -direct_pair_relative_forward_penalty * direct_pair_cost_guard_weight,
            "direct_pair_source_spread_gap_penalty": -direct_pair_spread_gap_penalty * direct_pair_cost_guard_weight,
            "portfolio_daily_source_forward_penalty": -portfolio_daily_source_forward_penalty
            * portfolio_daily_ranking_weight,
            "portfolio_daily_relative_forward_penalty": -portfolio_daily_relative_forward_penalty
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_realization_gap_penalty": -portfolio_daily_source_realization_gap
            * 2.20
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_not_sold_penalty": -portfolio_daily_source_not_sold_excess
            * 1.20
            * portfolio_daily_ranking_weight,
            "portfolio_daily_effective_transfer_sparse_penalty": -portfolio_daily_effective_transfer_sparse_penalty
            * portfolio_daily_ranking_weight,
            "portfolio_daily_receiver_activity_gap_penalty": -portfolio_daily_receiver_activity_gap
            * 0.58
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_activity_gap_penalty": -portfolio_daily_source_activity_gap
            * 0.42
            * portfolio_daily_ranking_weight,
            "portfolio_daily_dead_allocation_branch_penalty": -portfolio_daily_dead_allocation_branch
            * 2.40
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_sparse_penalty": -portfolio_daily_source_sparse_penalty
            * portfolio_daily_ranking_weight,
            "portfolio_daily_cash_drag_penalty": -portfolio_daily_cash_drag_penalty
            * portfolio_daily_ranking_weight,
            "portfolio_daily_funding_coverage_gap_penalty": -portfolio_daily_funding_coverage_gap
            * 1.10
            * portfolio_daily_ranking_weight,
            "portfolio_daily_closure_gap_penalty": -portfolio_daily_closure_gap
            * 1.05
            * portfolio_daily_ranking_weight,
            "portfolio_daily_transfer_gap_penalty": -portfolio_daily_transfer_gap
            * 1.15
            * portfolio_daily_ranking_weight,
            "portfolio_daily_economic_release_gap_penalty": -portfolio_daily_economic_release_gap
            * 1.35
            * portfolio_daily_ranking_weight,
            "portfolio_daily_bad_forward_spread_risk_penalty": -portfolio_daily_bad_forward_spread_risk_penalty
            * 1.60
            * portfolio_daily_ranking_weight,
            "portfolio_daily_economic_block_risk_penalty": -portfolio_daily_economic_block_risk_penalty
            * 1.45
            * portfolio_daily_ranking_weight,
            "portfolio_daily_forward_strength_brake_risk_penalty": -portfolio_daily_forward_strength_brake_risk_penalty
            * 1.70
            * portfolio_daily_ranking_weight,
            "portfolio_daily_forward_proxy_keep_risk_penalty": -portfolio_daily_forward_proxy_keep_risk_penalty
            * 1.80
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_release_conviction_gap_penalty": -portfolio_daily_source_release_conviction_gap
            * 1.55
            * portfolio_daily_ranking_weight,
            "portfolio_daily_allocation_dead_branch_risk_penalty": -portfolio_daily_dead_branch_risk_penalty
            * 1.30
            * portfolio_daily_ranking_weight,
            "portfolio_daily_receiver_semantic_bypass_penalty": -portfolio_daily_receiver_semantic_bypass_penalty
            * portfolio_daily_ranking_weight,
            "portfolio_daily_unrealized_deploy_penalty": -portfolio_daily_unrealized_deploy_penalty
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_positive_distribution_penalty": -portfolio_daily_source_positive_distribution_penalty
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_positive_forward_penalty": -portfolio_daily_source_positive_forward_penalty
            * 0.92
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_opportunity_cost_penalty": -portfolio_daily_source_opportunity_cost_penalty
            * 0.74
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_hard_negative_penalty": -portfolio_daily_source_hard_negative_penalty
            * 0.88
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_tail_false_sell_penalty": -portfolio_daily_source_tail_false_sell_penalty
            * 0.72
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_hard_negative_prevalence_penalty": -portfolio_daily_source_hard_negative_prevalence
            * 0.62
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_hard_negative_selected_pressure_penalty": -portfolio_daily_source_hard_negative_selected_pressure
            * 1.04
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_release_preference_reward": portfolio_daily_source_release_preference
            * 0.44
            * portfolio_daily_ranking_weight,
            "portfolio_daily_transfer_regret_target_reward": portfolio_daily_transfer_regret_target
            * 0.36
            * portfolio_daily_ranking_weight,
            "portfolio_daily_unified_constraint_violation_penalty": -portfolio_daily_unified_constraint_violations
            * 1.10
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_strong_false_sell_penalty": -portfolio_daily_source_strong_false_sell_penalty
            * portfolio_daily_ranking_weight,
            "portfolio_daily_exposure_utilization_gap_penalty": -portfolio_daily_exposure_utilization_gap
            * 1.20
            * portfolio_daily_ranking_weight,
            "portfolio_daily_joint_economic_quality_gate": -portfolio_daily_joint_economic_quality_gap
            * 1.12
            * portfolio_daily_ranking_weight,
            "deploy_funding_forward_penalty": -max(0.0, deploy_funding_rebalance_forward_excess_5d) * 10.80,
            "deploy_funding_release_consistency_penalty": -(
                max(0.0, 0.56 - deploy_funding_release_consistent_share) * 2.35
                if deploy_funding_rebalance_sell_count >= 5.0
                else 0.0
            ),
        }
        stability_breakdown = {
            "gate_pass_ratio": gate_pass_ratio * 0.54,
            "monthly_consistency_score": monthly_consistency_score * 0.48,
            "monthly_worst_return_penalty": -monthly_worst_loss * 1.86,
            "monthly_intramonth_drawdown_penalty": -abs(min(monthly_intramonth_max_drawdown, 0.0)) * 1.02,
            "direct_action_value_mode_share": direct_action_value_mode_share * 0.62,
            "direct_action_intent_preserved_share": direct_action_intent_preserved_share * 1.02,
            "direct_action_value_gap_mean": _bounded(direct_action_value_gap_mean, 0.00, 0.24) * 0.42,
            "direct_action_deploy_authorized_realized_rate": direct_action_deploy_authorized_realized_rate
            * 0.92
            * direct_reallocation_weight,
            "direct_action_core_deploy_target_realized_rate": direct_action_core_deploy_target_realized_rate
            * 1.04
            * direct_pair_reallocation_weight,
            "direct_action_add_authorized_realized_rate": direct_action_add_authorized_realized_rate
            * 0.46
            * direct_reallocation_weight,
            "direct_action_pair_cost_guard_pass_rate": direct_action_pair_cost_guard_pass_rate
            * 0.24
            * direct_pair_cost_guard_weight,
            "direct_action_pair_source_spread_mean": _bounded(direct_action_pair_source_spread_mean, -0.010, 0.040)
            * 0.54
            * direct_pair_cost_guard_weight,
            "direct_action_core_minus_pair_forward_excess_5d": _bounded(
                direct_action_core_minus_pair_forward_excess_5d,
                -0.015,
                0.045,
            )
            * 0.78
            * direct_pair_cost_guard_weight,
            "direct_action_pair_source_cost_quality": (
                (1.0 - _bounded(direct_action_pair_source_cost_mean, 0.620, 0.880))
                if direct_pair_source_observed
                else 0.0
            )
            * 0.48
            * direct_pair_cost_guard_weight,
            "portfolio_daily_receiver_minus_source_forward_excess_5d": _bounded(
                portfolio_daily_receiver_minus_source_forward_excess_5d,
                -0.015,
                0.050,
            )
            * 1.02
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_realized_sell_rate": portfolio_daily_source_realized_sell_rate
            * 0.44
            * portfolio_daily_ranking_weight,
            "portfolio_daily_effective_capital_transfer_count": min(
                portfolio_daily_effective_capital_transfer_count,
                10.0,
            )
            * 0.024
            * portfolio_daily_ranking_weight,
            "portfolio_daily_cash_reserve_rate": (1.0 - _bounded(portfolio_daily_cash_reserve_rate, 0.00, 0.42))
            * 0.18
            * portfolio_daily_ranking_weight
            if portfolio_daily_observed
            else 0.0,
            "portfolio_daily_receiver_exec_guard_count": min(portfolio_daily_receiver_exec_guard_count, 20.0)
            * 0.012
            * portfolio_daily_ranking_weight,
            "portfolio_daily_receiver_realized_deploy_rate": portfolio_daily_receiver_realized_deploy_rate
            * 0.22
            * portfolio_daily_ranking_weight,
            "portfolio_daily_open_breadth_candidate_count": portfolio_daily_open_breadth_bonus
            * 1.10
            * portfolio_daily_ranking_weight,
            "portfolio_daily_receiver_candidate_breadth": portfolio_daily_receiver_candidate_breadth
            * 0.13
            * portfolio_daily_ranking_weight,
            "portfolio_daily_clean_source_candidate_breadth": portfolio_daily_clean_source_candidate_breadth
            * 0.15
            * portfolio_daily_ranking_weight,
            "portfolio_daily_exposure_utilization": _bounded(
                portfolio_daily_exposure_utilization,
                0.35,
                0.86,
            )
            * 0.20
            * portfolio_daily_ranking_weight,
            "portfolio_daily_receiver_funding_coverage_mean": _bounded(
                portfolio_daily_receiver_funding_coverage_mean,
                0.06,
                0.38,
            )
            * 0.28
            * portfolio_daily_ranking_weight,
            "portfolio_daily_funding_closure_score_mean": _bounded(
                portfolio_daily_funding_closure_score_mean,
                0.08,
                0.42,
            )
            * 0.26
            * portfolio_daily_ranking_weight,
            "portfolio_daily_allocation_transfer_score_mean": _bounded(
                portfolio_daily_allocation_transfer_score_mean,
                0.08,
                0.42,
            )
            * 0.28
            * portfolio_daily_ranking_weight,
            "action_value_consistency_score": action_value_consistency_score * 1.18,
            "release_translation_deploy_health_score": release_translation_deploy_health_score * 1.02,
            "release_translation_deploy_translation_score": release_translation_deploy_translation_score * 0.96,
            "release_translation_deploy_funding_score": release_translation_deploy_funding_score * 1.04,
            "deploy_intent_realized_rate": deploy_intent_realized_rate * 0.34,
            "reduce_success_rate_5d": reduce_success * 0.46,
            "exit_timeliness_rate_5d": exit_timeliness * 0.48,
            "cash_timing_quality_1d": _bounded(cash_timing, -0.35, 0.12) * 0.34,
            "training_evidence_bonus": 0.34 if training_evidence_ok else -0.42,
            "reversal_penalty": -reversal * 1.04,
            "shadow_reversal_penalty": -shadow_reversal * 1.02,
            "reversal_excess_penalty": -reversal_excess_penalty,
            "drawdown_penalty": -abs(min(max_drawdown, 0.0)) * 5.62,
            "threshold_gap_penalty": -threshold_gap_penalty * 0.94,
            "direct_mode_activation_penalty": -direct_mode_activation_penalty,
            "direct_action_conflict_penalty": -direct_conflict_penalty * 1.20,
            "direct_translation_penalty": -direct_translation_penalty * 2.18,
            "deploy_funding_sell_share_penalty": -max(0.0, deploy_funding_rebalance_sell_share - 0.30) * 2.70,
            "deploy_funding_authorization_gap_penalty": -funding_authorization_gap * 3.05,
            "direct_funding_protected_sell_penalty": -direct_funding_protected_penalty * 2.95,
            "direct_authorized_deploy_gap_penalty": -direct_authorized_deploy_gap
            * 2.85
            * direct_reallocation_weight,
            "direct_authorized_add_gap_penalty": -direct_authorized_add_gap * 1.55 * direct_reallocation_weight,
            "direct_reallocation_source_sparse_penalty": -direct_reallocation_source_sparse_penalty
            * 1.25
            * direct_reallocation_weight,
            "direct_pair_source_cost_penalty": -direct_pair_source_cost_penalty * 1.15 * direct_pair_cost_guard_weight,
            "direct_pair_source_forward_penalty": -direct_pair_forward_penalty * 1.10 * direct_pair_cost_guard_weight,
            "direct_pair_relative_forward_penalty": -direct_pair_relative_forward_penalty
            * 1.15
            * direct_pair_cost_guard_weight,
            "direct_pair_source_spread_gap_penalty": -direct_pair_spread_gap_penalty
            * 1.10
            * direct_pair_cost_guard_weight,
            "portfolio_daily_source_forward_penalty": -portfolio_daily_source_forward_penalty
            * 1.12
            * portfolio_daily_ranking_weight,
            "portfolio_daily_relative_forward_penalty": -portfolio_daily_relative_forward_penalty
            * 1.16
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_realization_gap_penalty": -portfolio_daily_source_realization_gap
            * 2.55
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_not_sold_penalty": -portfolio_daily_source_not_sold_excess
            * 1.45
            * portfolio_daily_ranking_weight,
            "portfolio_daily_effective_transfer_sparse_penalty": -portfolio_daily_effective_transfer_sparse_penalty
            * 1.12
            * portfolio_daily_ranking_weight,
            "portfolio_daily_receiver_activity_gap_penalty": -portfolio_daily_receiver_activity_gap
            * 0.74
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_activity_gap_penalty": -portfolio_daily_source_activity_gap
            * 0.52
            * portfolio_daily_ranking_weight,
            "portfolio_daily_dead_allocation_branch_penalty": -portfolio_daily_dead_allocation_branch
            * 2.85
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_sparse_penalty": -portfolio_daily_source_sparse_penalty
            * 1.18
            * portfolio_daily_ranking_weight,
            "portfolio_daily_funding_coverage_gap_penalty": -portfolio_daily_funding_coverage_gap
            * 1.24
            * portfolio_daily_ranking_weight,
            "portfolio_daily_closure_gap_penalty": -portfolio_daily_closure_gap
            * 1.18
            * portfolio_daily_ranking_weight,
            "portfolio_daily_transfer_gap_penalty": -portfolio_daily_transfer_gap
            * 1.26
            * portfolio_daily_ranking_weight,
            "portfolio_daily_allocation_dead_branch_risk_penalty": -portfolio_daily_dead_branch_risk_penalty
            * 1.42
            * portfolio_daily_ranking_weight,
            "portfolio_daily_receiver_semantic_bypass_penalty": -portfolio_daily_receiver_semantic_bypass_penalty
            * 1.20
            * portfolio_daily_ranking_weight,
            "portfolio_daily_unrealized_deploy_penalty": -portfolio_daily_unrealized_deploy_penalty
            * 1.18
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_positive_distribution_penalty": -portfolio_daily_source_positive_distribution_penalty
            * 1.18
            * portfolio_daily_ranking_weight,
            "portfolio_daily_source_strong_false_sell_penalty": -portfolio_daily_source_strong_false_sell_penalty
            * 1.22
            * portfolio_daily_ranking_weight,
            "portfolio_daily_exposure_utilization_gap_penalty": -portfolio_daily_exposure_utilization_gap
            * 1.35
            * portfolio_daily_ranking_weight,
            "portfolio_daily_joint_economic_quality_gate": -portfolio_daily_joint_economic_quality_gap
            * 0.86
            * portfolio_daily_ranking_weight,
            "deploy_funding_forward_penalty": -max(0.0, deploy_funding_rebalance_forward_excess_5d) * 11.60,
            "deploy_funding_against_hold_penalty": -(
                max(0.0, deploy_funding_against_protected_hold_share - 0.20) * 2.95
                if deploy_funding_rebalance_sell_count >= 5.0
                else 0.0
            ),
        }
        if objective_profile_name in PORTFOLIO_DAILY_GATE_OBJECTIVES:
            v2_observed = portfolio_daily_observed
            v2_negative_annual_return = max(0.0, -annual_return)
            v2_negative_sharpe = max(0.0, -sharpe)
            v2_negative_monthly_mean = max(0.0, -monthly_return_mean)
            v2_drawdown_excess = max(0.0, abs(min(max_drawdown, 0.0)) - 0.18)
            v2_monthly_consistency_gap = max(0.0, 0.45 - monthly_consistency_score)
            v2_source_realization_gap = (
                max(0.0, 0.35 - portfolio_daily_source_realized_sell_rate)
                if portfolio_daily_source_target_count >= 3.0
                else 0.0
            )
            v2_source_not_sold_gap = (
                max(0.0, portfolio_daily_source_target_not_sold_share - 0.65)
                if portfolio_daily_source_target_count >= 3.0
                else 0.0
            )
            v2_order_translation_gap = max(0.0, direct_translation_penalty - 0.24)
            v2_add_to_hold_gap = max(0.0, add_to_hold_conflict_share - 0.35)
            v2_authorized_add_no_weight_gap = max(0.0, authorized_add_no_weight_change_share - 0.02)
            v2_deploy_unrealized_gap = max(0.0, deploy_intent_unrealized_share - 0.18)
            v2_source_positive_distribution_gap = (
                max(0.0, portfolio_daily_source_positive_forward_sell_share - 0.45)
                if portfolio_daily_source_target_count >= 3.0
                else 0.0
            )
            v2_source_strong_false_sell_gap = (
                max(0.0, portfolio_daily_source_max_forward_excess_5d - 0.080)
                + min(portfolio_daily_source_strong_positive_forward_sell_count, 4.0) * 0.04
                if portfolio_daily_source_target_count >= 3.0
                else 0.0
            )
            v2_source_release_conviction_gap = (
                max(0.0, 0.300 - portfolio_daily_source_release_conviction_mean)
                if portfolio_daily_source_target_count >= 3.0
                else 0.0
            )
            v2_exposure_utilization_gap = (
                max(0.0, 0.50 - portfolio_daily_exposure_utilization)
                if avg_gross_exposure_target >= 0.42
                else 0.0
            )
            v2_cash_dead_branch = 1.0 if v2_observed and portfolio_daily_cash_reserve_rate <= 0.0 else 0.0
            v2_receiver_activity_gap = max(0.0, 3.0 - portfolio_daily_receiver_target_count)
            v2_source_activity_gap = (
                max(0.0, 1.0 - portfolio_daily_source_target_count)
                if portfolio_daily_receiver_target_count >= 3.0
                else 0.0
            )
            v2_dead_allocation_branch = portfolio_daily_dead_allocation_branch
            v2_clean_spread_allowed = (
                annual_return > 0.0
                and sharpe > 0.0
                and monthly_return_mean > 0.0
                and max_drawdown >= -0.18
                and monthly_consistency_score >= 0.45
                and portfolio_daily_receiver_target_count >= 3.0
                and portfolio_daily_receiver_unrealized_deploy_share <= 0.02
                and portfolio_daily_source_realized_sell_rate >= 0.35
                and (portfolio_daily_source_target_count < 3.0 or portfolio_daily_source_realized_sell_rate >= 0.35)
                and (portfolio_daily_source_target_count < 3.0 or portfolio_daily_source_target_not_sold_share <= 0.65)
                and authorized_add_no_weight_change_share <= 0.02
                and deploy_intent_unrealized_share <= 0.18
                and direct_action_authorization_subset_violation_count <= 0.0
                and (
                    portfolio_daily_source_target_count < 3.0
                    or portfolio_daily_source_positive_forward_sell_share <= 0.45
                )
                and (
                    portfolio_daily_source_target_count < 3.0
                    or portfolio_daily_source_max_forward_excess_5d <= 0.080
                )
                and (
                    portfolio_daily_source_target_count < 3.0
                    or portfolio_daily_source_release_conviction_mean >= 0.300
                )
                and (
                    avg_gross_exposure_target < 0.42
                    or portfolio_daily_exposure_utilization >= 0.50
                )
                and direct_translation_penalty <= 0.24
                and add_to_hold_conflict_share <= 0.35
                and (not v2_observed or portfolio_daily_cash_reserve_rate > 0.0)
            )
            if not v2_clean_spread_allowed:
                performance_breakdown.update(
                    {
                        "portfolio_daily_v2_negative_annual_return_gate": -v2_negative_annual_return * 24.0,
                        "portfolio_daily_v2_negative_sharpe_gate": -v2_negative_sharpe * 8.0,
                        "portfolio_daily_v2_negative_monthly_mean_gate": -v2_negative_monthly_mean * 60.0,
                        "portfolio_daily_v2_drawdown_excess_gate": -v2_drawdown_excess * 18.0,
                        "portfolio_daily_v2_source_realization_gate": -v2_source_realization_gap * 8.0,
                        "portfolio_daily_v2_source_not_sold_gate": -v2_source_not_sold_gap * 6.0,
                        "portfolio_daily_v2_authorized_add_no_weight_gate": -v2_authorized_add_no_weight_gap * 10.0,
                        "portfolio_daily_v2_deploy_unrealized_gate": -v2_deploy_unrealized_gap * 6.5,
                        "portfolio_daily_v2_source_positive_distribution_gate": -v2_source_positive_distribution_gap * 7.5,
                        "portfolio_daily_v2_source_strong_false_sell_gate": -v2_source_strong_false_sell_gap * 9.0,
                        "portfolio_daily_v2_source_release_conviction_gate": (
                            -v2_source_release_conviction_gap * 7.0
                        ),
                        "portfolio_daily_v2_exposure_utilization_gate": -v2_exposure_utilization_gap * 3.0,
                        "portfolio_daily_v2_receiver_activity_gate": -v2_receiver_activity_gap * 1.35,
                        "portfolio_daily_v2_source_activity_gate": -v2_source_activity_gap * 1.10,
                        "portfolio_daily_v2_dead_allocation_branch_gate": -v2_dead_allocation_branch * 3.20,
                        "portfolio_daily_v2_clean_spread_rebate_removed": -max(
                            0.0,
                            _bounded(portfolio_daily_receiver_minus_source_forward_excess_5d, -0.015, 0.050)
                            * 1.10,
                        ),
                    }
                )
                stability_breakdown.update(
                    {
                        "portfolio_daily_v2_monthly_consistency_gate": -v2_monthly_consistency_gap * 3.2,
                        "portfolio_daily_v2_source_realization_gate": -v2_source_realization_gap * 3.2,
                        "portfolio_daily_v2_source_not_sold_gate": -v2_source_not_sold_gap * 2.4,
                        "portfolio_daily_v2_order_translation_gate": -v2_order_translation_gap * 8.0,
                        "portfolio_daily_v2_add_to_hold_gate": -v2_add_to_hold_gap * 4.2,
                        "portfolio_daily_v2_authorized_add_no_weight_gate": -v2_authorized_add_no_weight_gap * 6.0,
                        "portfolio_daily_v2_deploy_unrealized_gate": -v2_deploy_unrealized_gap * 4.6,
                        "portfolio_daily_v2_source_positive_distribution_gate": -v2_source_positive_distribution_gap * 5.2,
                        "portfolio_daily_v2_source_strong_false_sell_gate": -v2_source_strong_false_sell_gap * 6.4,
                        "portfolio_daily_v2_source_release_conviction_gate": (
                            -v2_source_release_conviction_gap * 4.5
                        ),
                        "portfolio_daily_v2_exposure_utilization_gate": -v2_exposure_utilization_gap * 2.4,
                        "portfolio_daily_v2_cash_dead_branch_gate": -v2_cash_dead_branch * 1.6,
                        "portfolio_daily_v2_receiver_activity_gate": -v2_receiver_activity_gap * 0.85,
                        "portfolio_daily_v2_source_activity_gate": -v2_source_activity_gap * 0.70,
                        "portfolio_daily_v2_dead_allocation_branch_gate": -v2_dead_allocation_branch * 2.20,
                    }
                )
            else:
                performance_breakdown["portfolio_daily_v2_economic_gate_bonus"] = 0.42
                stability_breakdown["portfolio_daily_v2_execution_gate_bonus"] = 0.28
    elif objective_profile_name == "direct_daily_policy_v1":
        action_alignment_score = (
            _bounded(multi_horizon_path_alignment, -0.10, 0.16) * 0.28
            + _bounded(open_action_value_alignment, -0.10, 0.16) * 0.20
            + _bounded(add_action_value_alignment, -0.10, 0.16) * 0.16
            + _bounded(hold_action_value_alignment, -0.10, 0.16) * 0.18
            + _bounded(reduce_action_value_avoidance, -0.10, 0.16) * 0.10
            + _bounded(exit_action_value_avoidance, -0.10, 0.16) * 0.08
        )
        direct_conflict_penalty = (
            action_value_conflict_share * 1.28
            + open_low_action_value_share * 0.92
            + sell_against_keep_value_share * 0.92
            + keep_against_release_value_share * 0.70
            + direct_action_value_low_margin_share * 0.24
        )
        direct_translation_penalty = max(
            order_translation_conflict_rate,
            direct_action_order_translation_conflict_rate,
        )
        direct_mode_activation_penalty = max(0.0, 0.92 - direct_action_value_mode_share) * 1.20
        performance_breakdown = {
            "annual_return": annual_return * 1.64,
            "sharpe": sharpe * 0.22,
            "monthly_return_mean_annualized_focus": monthly_return_mean * 12.0 * 0.44,
            "monthly_win_rate_focus": monthly_win_rate * 0.28,
            "monthly_consistency_focus": monthly_consistency_score * 0.24,
            "action_value_consistency_score": action_value_consistency_score * 1.28,
            "action_value_alignment_score": action_alignment_score * 1.04,
            "direct_action_value_mode_share": direct_action_value_mode_share * 0.72,
            "direct_action_value_label_match_share": direct_action_value_label_match_share * 0.36,
            "direct_action_value_gap_mean": _bounded(direct_action_value_gap_mean, 0.00, 0.24) * 0.42,
            "release_translation_deploy_health_score": release_translation_deploy_health_score * 0.74,
            "deploy_intent_realized_rate": deploy_intent_realized_rate * 0.56,
            "reduce_success_rate_5d": reduce_success * 0.66,
            "exit_timeliness_rate_5d": exit_timeliness * 0.68,
            "sell_selection_quality_5d": _bounded(sell_selection_quality, -0.06, 0.08) * 0.48,
            "trend_capture_rate_10d": trend_capture * 0.52,
            "active_alignment_bonus": active_alignment_bonus,
            "gate_pass_ratio": gate_pass_ratio * 0.18,
            "drawdown_penalty": -abs(min(max_drawdown, 0.0)) * 3.80,
            "monthly_worst_return_focus_penalty": -monthly_worst_loss * 1.20,
            "negative_return_penalty": -negative_return_penalty * 3.00,
            "negative_sharpe_penalty": -negative_sharpe_penalty * 0.50,
            "direct_mode_activation_penalty": -direct_mode_activation_penalty,
            "direct_action_conflict_penalty": -direct_conflict_penalty,
            "order_translation_conflict_penalty": -direct_translation_penalty * 1.42,
            "add_to_hold_conflict_penalty": -add_to_hold_conflict_share * 1.04,
            "deploy_hold_conflict_penalty": -deploy_intent_hold_conflict_share * 0.96,
            "deploy_candidate_budget_drop_penalty": -deploy_intent_candidate_budget_drop_share * 0.82,
        }
        stability_breakdown = {
            "gate_pass_ratio": gate_pass_ratio * 0.52,
            "monthly_consistency_score": monthly_consistency_score * 0.46,
            "monthly_worst_return_penalty": -monthly_worst_loss * 1.72,
            "monthly_intramonth_drawdown_penalty": -abs(min(monthly_intramonth_max_drawdown, 0.0)) * 0.96,
            "action_value_consistency_score": action_value_consistency_score * 1.54,
            "action_value_alignment_score": action_alignment_score * 0.72,
            "direct_action_value_mode_share": direct_action_value_mode_share * 0.86,
            "direct_action_value_label_match_share": direct_action_value_label_match_share * 0.42,
            "direct_action_value_gap_mean": _bounded(direct_action_value_gap_mean, 0.00, 0.24) * 0.36,
            "release_translation_deploy_translation_score": release_translation_deploy_translation_score * 0.72,
            "release_translation_deploy_health_score": release_translation_deploy_health_score * 0.78,
            "deploy_intent_realized_rate": deploy_intent_realized_rate * 0.42,
            "reduce_success_rate_5d": reduce_success * 0.40,
            "exit_timeliness_rate_5d": exit_timeliness * 0.42,
            "cash_timing_quality_1d": _bounded(cash_timing, -0.35, 0.12) * 0.30,
            "hold_share": hold_share * 0.14,
            "training_evidence_bonus": 0.34 if training_evidence_ok else -0.42,
            "reversal_penalty": -reversal * 1.02,
            "shadow_reversal_penalty": -shadow_reversal * 1.00,
            "reversal_excess_penalty": -reversal_excess_penalty,
            "drawdown_penalty": -abs(min(max_drawdown, 0.0)) * 5.36,
            "threshold_gap_penalty": -threshold_gap_penalty * 0.88,
            "direct_mode_activation_penalty": -direct_mode_activation_penalty,
            "direct_action_conflict_penalty": -direct_conflict_penalty * 1.16,
            "semantic_conflict_penalty": -semantic_conflict_rate * 1.58,
            "order_translation_conflict_penalty": -direct_translation_penalty * 1.72,
            "deploy_funding_forward_penalty": -max(0.0, deploy_funding_rebalance_forward_excess_5d) * 8.80,
            "deploy_funding_against_hold_penalty": -(
                max(0.0, deploy_funding_against_protected_hold_share - 0.22) * 2.25
                if deploy_funding_rebalance_sell_count >= 5.0
                else 0.0
            ),
        }
    elif objective_profile_name == "return_recovery_v2":
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
    performance_breakdown.update(
        {
            "monthly_return_mean_annualized": monthly_return_mean * 12.0 * 0.35,
            "monthly_win_rate": monthly_win_rate * 0.18,
            "monthly_sharpe": monthly_sharpe * 0.04,
        }
    )
    stability_breakdown.update(
        {
            "monthly_consistency_score": monthly_consistency_score * 0.32,
            "monthly_worst_return_penalty": -monthly_worst_loss * 1.40,
            "monthly_loss_streak_penalty": -monthly_loss_streak_penalty * 0.10,
            "monthly_intramonth_drawdown_penalty": -abs(min(monthly_intramonth_max_drawdown, 0.0)) * 0.80,
        }
    )
    performance_score = round(sum(performance_breakdown.values()), 6)
    stability_score = round(sum(stability_breakdown.values()), 6)
    composite_score = round(performance_score + stability_score, 6)
    score_breakdown = {
        "performance": performance_breakdown,
        "stability": stability_breakdown,
    }
    native_target_metrics = {
        "allocation_layer_native_target_used": _summary_metric("allocation_layer_native_target_used"),
        "allocation_layer_native_fallback_used": _summary_metric("allocation_layer_native_fallback_used"),
        "allocation_layer_source_target_count": _summary_metric("allocation_layer_source_target_count"),
        "allocation_layer_source_executable_candidate_count": _summary_metric(
            "allocation_layer_source_executable_candidate_count"
        ),
        "native_negative_delta_count": _summary_metric("native_negative_delta_count"),
        "native_receiver_executable_mask_count": _summary_metric("native_receiver_executable_mask_count"),
        "native_positive_delta_count": _summary_metric("native_positive_delta_count"),
        "native_positive_delta_unsupported_share": _summary_metric("native_positive_delta_unsupported_share"),
        "native_receiver_mask_mismatch_count": _summary_metric("native_receiver_mask_mismatch_count"),
        "native_source_target_count": _summary_metric("native_source_target_count"),
        "native_target_valid": _summary_metric("native_target_valid"),
        "native_target_constraint_violations": _summary_metric("native_target_constraint_violations"),
        "native_target_invalid_sum_count": _summary_metric("native_target_invalid_sum_count"),
        "native_target_invalid_turnover_count": _summary_metric("native_target_invalid_turnover_count"),
        "native_target_invalid_cap_count": _summary_metric("native_target_invalid_cap_count"),
        "native_target_invalid_negative_weight_count": _summary_metric("native_target_invalid_negative_weight_count"),
        "native_target_invalid_unsupported_receiver_count": _summary_metric(
            "native_target_invalid_unsupported_receiver_count"
        ),
        "native_target_invalid_sell_nonheld_count": _summary_metric("native_target_invalid_sell_nonheld_count"),
    }
    return {
        "performance_score": performance_score,
        "stability_score": stability_score,
        "composite_score": composite_score,
        "score_breakdown": score_breakdown,
        "gate_pass_ratio": gate_pass_ratio,
        "passed_check_count": passed_checks,
        "total_check_count": total_checks,
        "objective_profile": objective_profile_name,
        "primary_metrics": {
            "annual_return": annual_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "monthly_return_mean": monthly_return_mean,
            "monthly_win_rate": monthly_win_rate,
            "monthly_worst_return": monthly_worst_return,
            "monthly_sharpe": monthly_sharpe,
            "monthly_consistency_score": monthly_consistency_score,
            "monthly_max_consecutive_loss_months": monthly_max_consecutive_loss_months,
            "monthly_intramonth_max_drawdown": monthly_intramonth_max_drawdown,
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
            "deploy_intent_realized_rate": deploy_intent_realized_rate,
            "open_add_positive_weight_change_rate": open_add_positive_weight_change_rate,
            "add_to_hold_conflict_share": add_to_hold_conflict_share,
            "deploy_intent_hold_conflict_share": deploy_intent_hold_conflict_share,
            "deploy_intent_dropped_share": deploy_intent_dropped_share,
            "deploy_intent_candidate_budget_drop_share": deploy_intent_candidate_budget_drop_share,
            "deploy_executability_forward_alignment_5d": deploy_executability_alignment,
            "deploy_gate_forward_alignment_5d": deploy_gate_alignment,
            "release_gate_forward_alignment_5d": release_gate_alignment,
            "value_arbitration_forward_alignment_5d": value_arbitration_alignment,
            "multi_horizon_path_value_alignment_5d": multi_horizon_path_alignment,
            "open_action_value_forward_alignment_5d": open_action_value_alignment,
            "add_action_value_forward_alignment_5d": add_action_value_alignment,
            "hold_action_value_forward_alignment_5d": hold_action_value_alignment,
            "reduce_action_value_forward_avoidance_5d": reduce_action_value_avoidance,
            "exit_action_value_forward_avoidance_5d": exit_action_value_avoidance,
            "multi_horizon_forward_risk_avoidance_5d": multi_horizon_forward_risk_avoidance,
            "action_value_consistency_score": action_value_consistency_score,
            "action_value_conflict_share": action_value_conflict_share,
            "sell_against_keep_value_share": sell_against_keep_value_share,
            "keep_against_release_value_share": keep_against_release_value_share,
            "open_low_action_value_share": open_low_action_value_share,
            "held_keep_release_value_gap": held_keep_release_value_gap,
            "direct_action_value_mode_share": direct_action_value_mode_share,
            "direct_action_value_label_match_share": direct_action_value_label_match_share,
            "direct_action_value_gap_mean": direct_action_value_gap_mean,
            "direct_action_value_low_margin_share": direct_action_value_low_margin_share,
            "direct_action_order_translation_conflict_rate": direct_action_order_translation_conflict_rate,
            "direct_action_intent_preserved_share": direct_action_intent_preserved_share,
            "direct_action_funding_authorized_sell_share": direct_action_funding_authorized_sell_share,
            "direct_action_funding_protected_sell_share": direct_action_funding_protected_sell_share,
            "direct_action_release_advantage_mean": direct_action_release_advantage_mean,
            "direct_action_deploy_advantage_mean": direct_action_deploy_advantage_mean,
            "direct_action_core_deploy_target_count": direct_action_core_deploy_target_count,
            "direct_action_core_deploy_target_realized_rate": direct_action_core_deploy_target_realized_rate,
            "direct_action_add_authorized_count": direct_action_add_authorized_count,
            "direct_action_add_authorized_realized_rate": direct_action_add_authorized_realized_rate,
            "authorized_add_no_weight_change_share": authorized_add_no_weight_change_share,
            "direct_action_deploy_authorized_count": direct_action_deploy_authorized_count,
            "direct_action_deploy_authorized_realized_rate": direct_action_deploy_authorized_realized_rate,
            "direct_action_authorization_subset_violation_count": direct_action_authorization_subset_violation_count,
            "direct_action_reallocation_source_count": direct_action_reallocation_source_count,
            "direct_action_pair_reallocation_source_count": direct_action_pair_reallocation_source_count,
            "direct_action_pair_cost_guard_pass_count": direct_action_pair_cost_guard_pass_count,
            "direct_action_pair_cost_guard_blocked_count": direct_action_pair_cost_guard_blocked_count,
            "direct_action_pair_cost_guard_pass_rate": direct_action_pair_cost_guard_pass_rate,
            "direct_action_pair_source_spread_mean": direct_action_pair_source_spread_mean,
            "direct_action_pair_source_cost_mean": direct_action_pair_source_cost_mean,
            "direct_action_pair_source_forward_excess_5d": direct_action_pair_source_forward_excess_5d,
            "direct_action_core_target_forward_excess_5d": direct_action_core_target_forward_excess_5d,
            "direct_action_core_minus_pair_forward_excess_5d": direct_action_core_minus_pair_forward_excess_5d,
            "portfolio_daily_receiver_candidate_count": portfolio_daily_receiver_candidate_count,
            "portfolio_daily_receiver_target_count": portfolio_daily_receiver_target_count,
            "portfolio_daily_receiver_exec_guard_count": portfolio_daily_receiver_exec_guard_count,
            "portfolio_daily_receiver_semantic_no_headroom_count": portfolio_daily_receiver_semantic_no_headroom_count,
            "portfolio_daily_receiver_open_breadth_candidate_count": portfolio_daily_receiver_open_breadth_candidate_count,
            "portfolio_daily_receiver_realized_deploy_rate": portfolio_daily_receiver_realized_deploy_rate,
            "portfolio_daily_receiver_unrealized_deploy_share": portfolio_daily_receiver_unrealized_deploy_share,
            "portfolio_daily_source_target_count": portfolio_daily_source_target_count,
            "portfolio_daily_source_realized_sell_rate": portfolio_daily_source_realized_sell_rate,
            "portfolio_daily_source_target_not_sold_share": portfolio_daily_source_target_not_sold_share,
            "portfolio_daily_source_exec_cap_guard_count": portfolio_daily_source_exec_cap_guard_count,
            "portfolio_daily_effective_capital_transfer_count": portfolio_daily_effective_capital_transfer_count,
            "portfolio_daily_cash_score_mean": portfolio_daily_cash_score_mean,
            "portfolio_daily_cash_reserve_rate": portfolio_daily_cash_reserve_rate,
            "portfolio_daily_source_gap_mean": portfolio_daily_source_gap_mean,
            "portfolio_daily_source_forward_spread_score_mean": portfolio_daily_source_forward_spread_score_mean,
            "portfolio_daily_source_bad_forward_spread_risk_mean": portfolio_daily_source_bad_forward_spread_risk_mean,
            "portfolio_daily_source_economic_release_score_mean": portfolio_daily_source_economic_release_score_mean,
            "portfolio_daily_source_economic_block_risk_mean": portfolio_daily_source_economic_block_risk_mean,
            "portfolio_daily_source_forward_strength_brake_risk_mean": (
                portfolio_daily_source_forward_strength_brake_risk_mean
            ),
            "portfolio_daily_source_forward_proxy_keep_risk_mean": (
                portfolio_daily_source_forward_proxy_keep_risk_mean
            ),
            "portfolio_daily_source_release_conviction_mean": (
                portfolio_daily_source_release_conviction_mean
            ),
            "portfolio_daily_source_distribution_clean_blocked_count": (
                portfolio_daily_source_distribution_clean_blocked_count
            ),
            "portfolio_daily_source_direct_release_relief_score_mean": (
                portfolio_daily_source_direct_release_relief_score_mean
            ),
            "portfolio_daily_receiver_funding_coverage_mean": portfolio_daily_receiver_funding_coverage_mean,
            "portfolio_daily_funding_closure_score_mean": portfolio_daily_funding_closure_score_mean,
            "portfolio_daily_allocation_transfer_score_mean": portfolio_daily_allocation_transfer_score_mean,
            "portfolio_daily_allocation_dead_branch_risk_mean": portfolio_daily_allocation_dead_branch_risk_mean,
            "portfolio_daily_receiver_forward_excess_5d": portfolio_daily_receiver_forward_excess_5d,
            "portfolio_daily_source_forward_excess_5d": portfolio_daily_source_forward_excess_5d,
            "portfolio_daily_source_positive_forward_sell_share": portfolio_daily_source_positive_forward_sell_share,
            "portfolio_daily_source_strong_positive_forward_sell_count": portfolio_daily_source_strong_positive_forward_sell_count,
            "portfolio_daily_source_max_forward_excess_5d": portfolio_daily_source_max_forward_excess_5d,
            "portfolio_daily_source_p75_forward_excess_5d": portfolio_daily_source_p75_forward_excess_5d,
            "portfolio_daily_source_hard_negative_penalty": portfolio_daily_source_hard_negative_penalty,
            "portfolio_daily_source_tail_false_sell_penalty": portfolio_daily_source_tail_false_sell_penalty,
            "portfolio_daily_source_release_preference": portfolio_daily_source_release_preference,
            "portfolio_daily_transfer_regret_target": portfolio_daily_transfer_regret_target,
            "portfolio_daily_source_hard_negative_prevalence": portfolio_daily_source_hard_negative_prevalence,
            "portfolio_daily_source_hard_negative_selected_pressure": (
                portfolio_daily_source_hard_negative_selected_pressure
            ),
            "portfolio_daily_receiver_minus_source_forward_excess_5d": (
                portfolio_daily_receiver_minus_source_forward_excess_5d
            ),
            "avg_gross_exposure_target": avg_gross_exposure_target,
            "portfolio_daily_exposure_utilization": portfolio_daily_exposure_utilization,
            "deploy_intent_unrealized_share": deploy_intent_unrealized_share,
            "sell_selection_quality_5d": sell_selection_quality,
            "budget_origin_sell_share": budget_origin_sell_share,
            "high_cash_budget_origin_sell_share": high_cash_budget_origin_sell_share,
            "deploy_funding_rebalance_sell_share": deploy_funding_rebalance_sell_share,
            "deploy_funding_rebalance_sell_count": deploy_funding_rebalance_sell_count,
            "deploy_funding_rebalance_forward_excess_5d": deploy_funding_rebalance_forward_excess_5d,
            "deploy_funding_against_protected_hold_share": deploy_funding_against_protected_hold_share,
            "deploy_funding_release_consistent_share": deploy_funding_release_consistent_share,
            "model_release_signal_sell_count": model_release_signal_sell_count,
            "model_release_signal_forward_excess_5d": model_release_signal_forward_excess_5d,
            "model_release_against_protected_hold_share": model_release_against_protected_hold_share,
            "model_release_release_consistent_share": model_release_release_consistent_share,
            "sell_source_floor_guard_share": sell_source_floor_guard_share,
            "release_translation_deploy_health_score": release_translation_deploy_health_score,
            "release_translation_deploy_deploy_score": release_translation_deploy_deploy_score,
            "release_translation_deploy_release_score": release_translation_deploy_release_score,
            "release_translation_deploy_translation_score": release_translation_deploy_translation_score,
            "release_translation_deploy_funding_score": release_translation_deploy_funding_score,
            "release_translation_deploy_model_release_score": release_translation_deploy_model_release_score,
            "release_translation_deploy_failure_mode": release_translation_deploy_failure_mode,
            **native_target_metrics,
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


def _write_study_progress_event(study_root: Path, *, event: str, **payload: Any) -> dict[str, Any]:
    study_root.mkdir(parents=True, exist_ok=True)
    progress_event = {
        "event": str(event),
        "updated_at": now_iso(),
        **payload,
    }
    progress_jsonl = study_root / "study_progress.jsonl"
    with progress_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(progress_event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    write_json(study_root / "study_progress.json", progress_event)
    return progress_event


def _detect_runtime_channel_failure(error_text: str) -> str:
    payload = str(error_text or "").strip()
    if not payload:
        return ""
    for reason, pattern in RUNTIME_CHANNEL_FAILURE_PATTERNS:
        if pattern in payload:
            return reason
    return ""


def _default_resource_profile(search_profile: str) -> str:
    return "safe" if str(search_profile or "") in TRUE_SOLVER_RESOURCE_SEARCH_PROFILES else "balanced"


def _logical_cpu_count() -> int:
    return max(1, int(os.cpu_count() or 1))


def _resolve_thread_limit(resource_profile: str, requested_thread_limit: int | None = None) -> int:
    if requested_thread_limit is not None and int(requested_thread_limit) > 0:
        return max(1, int(requested_thread_limit))
    profile = str(resource_profile or "balanced").strip().lower()
    logical = _logical_cpu_count()
    if profile == "full":
        return 0
    if profile == "safe":
        return max(1, min(4, logical // 2 if logical > 1 else 1))
    return max(1, min(8, max(2, logical // 2)))


def _resolve_resource_priority(resource_profile: str, requested_priority: str = "auto") -> str:
    priority = str(requested_priority or "auto").strip().lower()
    if priority != "auto":
        return priority
    profile = str(resource_profile or "balanced").strip().lower()
    if profile == "safe":
        return "below_normal"
    if profile == "full":
        return "normal"
    return "below_normal"


def _resolve_cpu_affinity_mask(
    resource_profile: str,
    thread_limit: int,
    requested_cpu_count: int | None = None,
) -> int:
    logical = _logical_cpu_count()
    requested = int(requested_cpu_count or 0)
    if requested > 0:
        cpu_count = max(1, min(logical, requested))
    elif str(resource_profile or "").strip().lower() == "full" or int(thread_limit) <= 0:
        return 0
    else:
        cpu_count = max(1, min(logical, int(thread_limit)))
    return (1 << cpu_count) - 1


def _build_resource_limits(
    *,
    search_profile: str,
    resource_profile: str = "auto",
    thread_limit: int | None = None,
    process_priority: str = "auto",
    cpu_affinity_count: int | None = None,
) -> dict[str, Any]:
    resolved_profile = str(resource_profile or "auto").strip().lower()
    if resolved_profile == "auto":
        resolved_profile = _default_resource_profile(search_profile)
    if resolved_profile not in RESOURCE_PROFILE_CHOICES or resolved_profile == "auto":
        raise ValueError(f"Unsupported resource profile: {resource_profile}")
    resolved_thread_limit = _resolve_thread_limit(resolved_profile, thread_limit)
    resolved_priority = _resolve_resource_priority(resolved_profile, process_priority)
    if resolved_priority not in RESOURCE_PRIORITY_CHOICES or resolved_priority == "auto":
        raise ValueError(f"Unsupported process priority: {process_priority}")
    affinity_mask = _resolve_cpu_affinity_mask(resolved_profile, resolved_thread_limit, cpu_affinity_count)
    return {
        "resource_profile": resolved_profile,
        "search_profile": str(search_profile or ""),
        "thread_limit": int(resolved_thread_limit),
        "process_priority": resolved_priority,
        "cpu_affinity_count": int(cpu_affinity_count or 0),
        "cpu_affinity_mask": int(affinity_mask),
        "logical_cpu_count": _logical_cpu_count(),
        "thread_env_keys": list(THREAD_LIMIT_ENV_KEYS),
        "applies_thread_env": bool(resolved_thread_limit > 0),
        "applies_cpu_affinity": bool(affinity_mask > 0),
    }


def _resource_limited_child_env(resource_limits: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    thread_limit = int(resource_limits.get("thread_limit", 0) or 0)
    if thread_limit > 0:
        for key in THREAD_LIMIT_ENV_KEYS:
            env[key] = str(thread_limit)
    env["CONTINUOUS_POLICY_RESOURCE_PROFILE"] = str(resource_limits.get("resource_profile", "") or "")
    env["CONTINUOUS_POLICY_PROCESS_PRIORITY"] = str(resource_limits.get("process_priority", "") or "")
    env["CONTINUOUS_POLICY_CPU_AFFINITY_MASK"] = str(int(resource_limits.get("cpu_affinity_mask", 0) or 0))
    return env


def _apply_windows_process_limits(process: subprocess.Popen[Any], resource_limits: dict[str, Any]) -> dict[str, Any]:
    applied: dict[str, Any] = {"priority_applied": False, "affinity_applied": False, "error": ""}
    if os.name != "nt":
        applied["platform"] = os.name
        return applied
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x0200 | 0x0400 | 0x0100, False, int(process.pid))
        if not handle:
            raise OSError(ctypes.get_last_error(), "OpenProcess failed")
        try:
            priority_map = {
                "normal": 0x00000020,
                "below_normal": 0x00004000,
                "idle": 0x00000040,
            }
            priority = str(resource_limits.get("process_priority", "normal") or "normal").lower()
            if priority in priority_map:
                applied["priority_applied"] = bool(kernel32.SetPriorityClass(handle, priority_map[priority]))
            affinity_mask = int(resource_limits.get("cpu_affinity_mask", 0) or 0)
            if affinity_mask > 0:
                applied["affinity_applied"] = bool(kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(affinity_mask)))
        finally:
            kernel32.CloseHandle(handle)
    except Exception as exc:
        applied["error"] = str(exc)
    return applied


def _run_protocol_with_progress(
    *,
    study_root: Path,
    phase: str,
    role: str,
    trial_id: int,
    trial_tag: str,
    source_trial_tag: str,
    protocol_args: list[str],
    protocol_fn: Any | None = None,
    heartbeat_interval_seconds: float = 300.0,
    progress_context: dict[str, Any] | None = None,
    resource_limits: dict[str, Any] | None = None,
) -> int:
    started_monotonic = time.monotonic()
    stop_event = threading.Event()
    progress_lock = threading.Lock()
    runner_command: list[str] = [
        str(sys.executable),
        "-m",
        "daily_research.continuous_policy.run_continuous_policy_protocol",
        *list(protocol_args),
    ]
    runner_mode = "subprocess" if protocol_fn is None else "callable"
    resolved_resource_limits = dict(resource_limits or {})
    process_limit_result: dict[str, Any] = {}

    def emit(event: str, **payload: Any) -> None:
        progress_payload = {
            "phase": phase,
            "role": role,
            "trial_id": int(trial_id),
            "trial_tag": trial_tag,
            "source_trial_tag": source_trial_tag,
            "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
            "process_id": os.getpid(),
            "protocol_args": list(protocol_args),
            "protocol_runner_mode": runner_mode,
        }
        if runner_mode == "subprocess":
            progress_payload["protocol_runner_command"] = list(runner_command)
        if resolved_resource_limits:
            progress_payload["resource_limits"] = dict(resolved_resource_limits)
        if process_limit_result:
            progress_payload["process_limit_result"] = dict(process_limit_result)
        progress_payload.update(dict(progress_context or {}))
        progress_payload.update(payload)
        with progress_lock:
            _write_study_progress_event(
                study_root,
                event=event,
                **progress_payload,
            )

    def heartbeat_loop() -> None:
        while not stop_event.wait(float(heartbeat_interval_seconds)):
            emit("protocol_heartbeat", status="running")

    emit("protocol_start", status="running")
    heartbeat_thread = threading.Thread(
        target=heartbeat_loop,
        name=f"study-progress-{trial_tag}",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        if protocol_fn is None:
            process = subprocess.Popen(
                runner_command,
                cwd=str(Path(__file__).resolve().parents[2]),
                env=_resource_limited_child_env(resolved_resource_limits),
            )
            if resolved_resource_limits:
                process_limit_result.update(_apply_windows_process_limits(process, resolved_resource_limits))
                emit("protocol_resource_limits_applied", status="running", child_process_id=int(process.pid))
            exit_code = int(process.wait())
        else:
            exit_code = int(protocol_fn(protocol_args))
    except Exception as exc:
        stop_event.set()
        heartbeat_thread.join(timeout=1.0)
        emit("protocol_failed", status="failed", error=str(exc))
        raise

    stop_event.set()
    heartbeat_thread.join(timeout=1.0)
    if exit_code == 0:
        emit("protocol_complete", status="completed", exit_code=exit_code)
    else:
        emit("protocol_failed", status="failed", exit_code=exit_code)
    return exit_code


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


def _portfolio_daily_v2_gate_pass(metrics: dict[str, Any]) -> bool:
    receiver_count = float(metrics.get("portfolio_daily_receiver_target_count", 0.0) or 0.0)
    source_count = float(metrics.get("portfolio_daily_source_target_count", 0.0) or 0.0)
    observed = receiver_count >= 3.0 or source_count >= 3.0
    receiver_observed = receiver_count >= 3.0
    source_observed = source_count >= 3.0
    return (
        observed
        and receiver_observed
        and source_count >= 3.0
        and float(metrics.get("annual_return", 0.0) or 0.0) > 0.0
        and float(metrics.get("sharpe", 0.0) or 0.0) > 0.0
        and float(metrics.get("monthly_return_mean", 0.0) or 0.0) > 0.0
        and float(metrics.get("max_drawdown", 0.0) or 0.0) >= -0.18
        and float(metrics.get("monthly_consistency_score", 0.0) or 0.0) >= 0.45
        and float(metrics.get("portfolio_daily_receiver_unrealized_deploy_share", 1.0) or 0.0) <= 0.02
        and float(metrics.get("portfolio_daily_source_realized_sell_rate", 0.0) or 0.0) >= 0.35
        and ((not source_observed) or float(metrics.get("portfolio_daily_source_realized_sell_rate", 0.0) or 0.0) >= 0.35)
        and ((not source_observed) or float(metrics.get("portfolio_daily_source_target_not_sold_share", 0.0) or 0.0) <= 0.65)
        and ((not source_observed) or float(metrics.get("portfolio_daily_receiver_minus_source_forward_excess_5d", 0.0) or 0.0) >= -0.002)
        and ((not source_observed) or float(metrics.get("portfolio_daily_source_forward_excess_5d", 0.0) or 0.0) <= 0.010)
        and ((not source_observed) or float(metrics.get("portfolio_daily_source_positive_forward_sell_share", 0.0) or 0.0) <= 0.45)
        and ((not source_observed) or float(metrics.get("portfolio_daily_source_max_forward_excess_5d", 0.0) or 0.0) <= 0.080)
        and ((not source_observed) or float(metrics.get("portfolio_daily_source_strong_positive_forward_sell_count", 0.0) or 0.0) <= 0.0)
        and ((not source_observed) or float(metrics.get("portfolio_daily_source_economic_release_score_mean", 0.0) or 0.0) >= 0.10)
        and ((not source_observed) or float(metrics.get("portfolio_daily_source_bad_forward_spread_risk_mean", 0.0) or 0.0) <= 0.60)
        and ((not source_observed) or float(metrics.get("portfolio_daily_source_economic_block_risk_mean", 0.0) or 0.0) <= 0.64)
        and ((not source_observed) or float(metrics.get("portfolio_daily_source_forward_strength_brake_risk_mean", 0.0) or 0.0) <= 0.58)
        and ((not source_observed) or float(metrics.get("portfolio_daily_source_forward_proxy_keep_risk_mean", 0.0) or 0.0) <= 0.280)
        and ((not source_observed) or float(metrics.get("portfolio_daily_source_release_conviction_mean", 0.0) or 0.0) >= 0.300)
        and float(metrics.get("authorized_add_no_weight_change_share", 0.0) or 0.0) <= 0.02
        and float(metrics.get("deploy_intent_unrealized_share", 0.0) or 0.0) <= 0.18
        and float(metrics.get("direct_action_authorization_subset_violation_count", 0.0) or 0.0) <= 0.0
        and (
            float(metrics.get("avg_gross_exposure_target", 0.0) or 0.0) < 0.42
            or float(metrics.get("portfolio_daily_exposure_utilization", 0.0) or 0.0) >= 0.50
        )
        and float(metrics.get("order_translation_conflict_rate", 0.0) or 0.0) <= 0.24
        and float(metrics.get("direct_action_order_translation_conflict_rate", 0.0) or 0.0) <= 0.24
        and float(metrics.get("add_to_hold_conflict_share", 0.0) or 0.0) <= 0.35
        and float(metrics.get("portfolio_daily_cash_reserve_rate", 0.0) or 0.0) > 0.0
    )


def _metric(metrics: dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(name, default) or default)
    except (TypeError, ValueError):
        return float(default)


def _portfolio_daily_v2_confirm_stability(
    confirm_metrics: dict[str, Any],
    source_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_metrics = dict(source_metrics or {})
    source_training_evidence_sufficient = (
        str(source_metrics.get("training_evidence_status", "") or "").strip().lower() == "sufficient"
        if source_metrics
        else True
    )
    confirm_training_evidence_sufficient = (
        str(confirm_metrics.get("training_evidence_status", "") or "").strip().lower() == "sufficient"
    )
    confirm_gate_pass = _portfolio_daily_v2_gate_pass(confirm_metrics)
    source_gate_pass = _portfolio_daily_v2_gate_pass(source_metrics) if source_metrics else False
    annual_return = _metric(confirm_metrics, "annual_return")
    sharpe = _metric(confirm_metrics, "sharpe")
    monthly_return_mean = _metric(confirm_metrics, "monthly_return_mean")
    max_drawdown = _metric(confirm_metrics, "max_drawdown")
    source_count = _metric(confirm_metrics, "portfolio_daily_source_target_count")
    receiver_source_spread = _metric(confirm_metrics, "portfolio_daily_receiver_minus_source_forward_excess_5d")
    source_forward = _metric(confirm_metrics, "portfolio_daily_source_forward_excess_5d")
    source_positive_forward_share = _metric(confirm_metrics, "portfolio_daily_source_positive_forward_sell_share")
    source_strong_positive_count = _metric(confirm_metrics, "portfolio_daily_source_strong_positive_forward_sell_count")
    source_max_forward = _metric(confirm_metrics, "portfolio_daily_source_max_forward_excess_5d")
    authorized_add_no_weight_share = _metric(confirm_metrics, "authorized_add_no_weight_change_share")
    deploy_unrealized_share = _metric(confirm_metrics, "deploy_intent_unrealized_share")
    exposure_utilization = _metric(confirm_metrics, "portfolio_daily_exposure_utilization")
    avg_gross_exposure_target = _metric(confirm_metrics, "avg_gross_exposure_target")
    source_forward_strength_brake_risk = _metric(
        confirm_metrics,
        "portfolio_daily_source_forward_strength_brake_risk_mean",
    )
    source_forward_proxy_keep_risk = _metric(
        confirm_metrics,
        "portfolio_daily_source_forward_proxy_keep_risk_mean",
    )
    source_release_conviction = _metric(
        confirm_metrics,
        "portfolio_daily_source_release_conviction_mean",
    )
    annual_delta = annual_return - _metric(source_metrics, "annual_return")
    sharpe_delta = sharpe - _metric(source_metrics, "sharpe")
    monthly_delta = monthly_return_mean - _metric(source_metrics, "monthly_return_mean")
    drawdown_delta = max_drawdown - _metric(source_metrics, "max_drawdown")
    order_delta = _metric(confirm_metrics, "order_translation_conflict_rate") - _metric(
        source_metrics,
        "order_translation_conflict_rate",
    )
    add_to_hold_delta = _metric(confirm_metrics, "add_to_hold_conflict_share") - _metric(
        source_metrics,
        "add_to_hold_conflict_share",
    )
    stable_checks = {
        "source_training_evidence_sufficient": source_training_evidence_sufficient,
        "confirm_training_evidence_sufficient": confirm_training_evidence_sufficient,
        "confirm_gate_pass": confirm_gate_pass,
        "confirm_annual_return_floor": annual_return >= 0.12,
        "confirm_sharpe_floor": sharpe >= 0.50,
        "confirm_monthly_return_floor": monthly_return_mean >= 0.006,
        "confirm_drawdown_floor": max_drawdown >= -0.18,
        "confirm_source_count_floor": source_count >= 3.0,
        "confirm_receiver_source_forward_spread_floor": receiver_source_spread >= -0.002,
        "confirm_source_forward_not_positive": source_forward <= 0.010,
        "confirm_source_positive_distribution_floor": source_count < 3.0 or source_positive_forward_share <= 0.45,
        "confirm_source_no_strong_false_sell": source_count < 3.0 or source_strong_positive_count <= 0.0,
        "confirm_source_max_forward_cap": source_count < 3.0 or source_max_forward <= 0.080,
        "confirm_authorized_add_no_weight_clean": authorized_add_no_weight_share <= 0.02,
        "confirm_deploy_unrealized_clean": deploy_unrealized_share <= 0.18,
        "confirm_exposure_utilization_floor": avg_gross_exposure_target < 0.42 or exposure_utilization >= 0.50,
        "confirm_source_forward_strength_brake_floor": source_forward_strength_brake_risk <= 0.58,
        "confirm_source_forward_proxy_keep_risk_floor": source_count < 3.0 or source_forward_proxy_keep_risk <= 0.280,
        "confirm_source_release_conviction_floor": source_count < 3.0 or source_release_conviction >= 0.300,
        "annual_return_decay_limit": (not source_metrics) or annual_delta >= -0.35,
        "sharpe_decay_limit": (not source_metrics) or sharpe_delta >= -1.20,
        "monthly_return_decay_limit": (not source_metrics) or monthly_delta >= -0.025,
        "drawdown_decay_limit": (not source_metrics) or drawdown_delta >= -0.07,
        "order_translation_decay_limit": (not source_metrics) or order_delta <= 0.18,
        "add_to_hold_decay_limit": (not source_metrics) or add_to_hold_delta <= 0.22,
    }
    failed = [name for name, passed in stable_checks.items() if not passed]
    return {
        "stable_confirmatory": not failed,
        "failed_stability_checks": failed,
        "source_training_evidence_status": str(source_metrics.get("training_evidence_status", "") or ""),
        "confirm_training_evidence_status": str(confirm_metrics.get("training_evidence_status", "") or ""),
        "source_gate_pass": source_gate_pass,
        "confirm_gate_pass": confirm_gate_pass,
        "annual_return_delta": annual_delta if source_metrics else 0.0,
        "sharpe_delta": sharpe_delta if source_metrics else 0.0,
        "monthly_return_mean_delta": monthly_delta if source_metrics else 0.0,
        "max_drawdown_delta": drawdown_delta if source_metrics else 0.0,
        "order_translation_conflict_delta": order_delta if source_metrics else 0.0,
        "add_to_hold_conflict_delta": add_to_hold_delta if source_metrics else 0.0,
        "confirm_annual_return": annual_return,
        "confirm_sharpe": sharpe,
        "confirm_monthly_return_mean": monthly_return_mean,
        "confirm_max_drawdown": max_drawdown,
        "confirm_source_target_count": source_count,
        "confirm_receiver_minus_source_forward_excess_5d": receiver_source_spread,
        "confirm_source_forward_excess_5d": source_forward,
        "confirm_source_positive_forward_sell_share": source_positive_forward_share,
        "confirm_source_strong_positive_forward_sell_count": source_strong_positive_count,
        "confirm_source_max_forward_excess_5d": source_max_forward,
        "confirm_source_forward_proxy_keep_risk_mean": source_forward_proxy_keep_risk,
        "confirm_source_release_conviction_mean": source_release_conviction,
        "confirm_authorized_add_no_weight_change_share": authorized_add_no_weight_share,
        "confirm_deploy_intent_unrealized_share": deploy_unrealized_share,
        "confirm_portfolio_daily_exposure_utilization": exposure_utilization,
    }


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


def _trial_has_sufficient_training_evidence(item: TrialResult) -> bool:
    return str(item.primary_metrics.get("training_evidence_status", "") or "").strip().lower() == "sufficient"


def _trial_is_portfolio_daily_v2_qualified(item: TrialResult) -> bool:
    return _trial_has_sufficient_training_evidence(item) and _portfolio_daily_v2_gate_pass(item.primary_metrics)


def _resource_gate_after_screening(
    profile_name: str,
    screening_results: list[TrialResult],
    *,
    selected_trial_count: int,
) -> dict[str, Any]:
    config = RESOURCE_GATED_SEARCH_PROFILES.get(str(profile_name), {})
    completed = [item for item in screening_results if item.status == "completed"]
    if not config:
        return {
            "resource_gate_enabled": False,
            "resource_gate_triggered": False,
            "continue_screening": True,
            "failed_resource_checks": [],
            "estimated_saved_screening_trials": 0,
        }
    min_completed = int(config.get("min_completed_screening", 1) or 1)
    if len(completed) < min_completed:
        return {
            "resource_gate_enabled": True,
            "resource_gate_triggered": False,
            "continue_screening": True,
            "completed_screening_count": len(completed),
            "failed_resource_checks": [],
            "estimated_saved_screening_trials": 0,
        }

    latest = completed[-1]
    metrics = dict(latest.primary_metrics or {})
    source_count = float(metrics.get("portfolio_daily_source_target_count", 0.0) or 0.0)
    source_sell_rate = float(metrics.get("portfolio_daily_source_realized_sell_rate", 0.0) or 0.0)
    receiver_count = float(metrics.get("portfolio_daily_receiver_target_count", 0.0) or 0.0)
    receiver_unrealized = float(metrics.get("portfolio_daily_receiver_unrealized_deploy_share", 1.0) or 0.0)
    cash_timing = float(metrics.get("cash_timing_quality_1d", 0.0) or 0.0)
    max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
    monthly_return = float(metrics.get("monthly_return_mean", 0.0) or 0.0)
    annual_return = float(metrics.get("annual_return", 0.0) or 0.0)
    exposure_utilization = float(metrics.get("portfolio_daily_exposure_utilization", 0.0) or 0.0)
    avg_gross_exposure_target = float(metrics.get("avg_gross_exposure_target", 0.0) or 0.0)
    failed: list[str] = []
    if source_count < float(config.get("source_count_floor", 1.0) or 1.0) or source_sell_rate < float(
        config.get("source_sell_rate_floor", 0.20) or 0.20
    ):
        failed.append("source_release_dead")
    if receiver_count < 3.0 or receiver_unrealized > float(config.get("receiver_unrealized_cap", 0.08) or 0.08):
        failed.append("receiver_deploy_not_clean")
    if cash_timing < float(config.get("cash_timing_floor", -0.10) or -0.10):
        failed.append("cash_timing_bad")
    if max_drawdown < float(config.get("drawdown_floor", -0.20) or -0.20):
        failed.append("drawdown_bad")
    if annual_return < float(config.get("annual_return_floor", 0.04) or 0.04) or monthly_return < float(
        config.get("monthly_return_floor", -0.002) or -0.002
    ):
        failed.append("economic_signal_too_weak")
    if "exposure_utilization_floor" in config and avg_gross_exposure_target >= 0.42:
        if exposure_utilization < float(config.get("exposure_utilization_floor", 0.50) or 0.50):
            failed.append("exposure_utilization_low")

    trigger_reasons = set(failed)
    terminal_failure = (
        "source_release_dead" in trigger_reasons
        and (
            "cash_timing_bad" in trigger_reasons
            or "drawdown_bad" in trigger_reasons
            or "exposure_utilization_low" in trigger_reasons
        )
    ) or (
        "receiver_deploy_not_clean" in trigger_reasons
        and (
            "economic_signal_too_weak" in trigger_reasons
            or "drawdown_bad" in trigger_reasons
            or "exposure_utilization_low" in trigger_reasons
        )
    )
    remaining = max(0, int(selected_trial_count) - len(screening_results))
    return {
        "resource_gate_enabled": True,
        "resource_gate_triggered": bool(terminal_failure),
        "continue_screening": not bool(terminal_failure),
        "completed_screening_count": len(completed),
        "evaluated_trial_tag": latest.trial_tag,
        "failed_resource_checks": failed,
        "estimated_saved_screening_trials": remaining if terminal_failure else 0,
        "resource_gate_metrics": {
            "portfolio_daily_source_target_count": source_count,
            "portfolio_daily_source_realized_sell_rate": source_sell_rate,
            "portfolio_daily_receiver_target_count": receiver_count,
            "portfolio_daily_receiver_unrealized_deploy_share": receiver_unrealized,
            "cash_timing_quality_1d": cash_timing,
            "max_drawdown": max_drawdown,
            "monthly_return_mean": monthly_return,
            "annual_return": annual_return,
            "portfolio_daily_exposure_utilization": exposure_utilization,
            "avg_gross_exposure_target": avg_gross_exposure_target,
        },
    }


def _pick_confirmatory_candidates(
    completed_trials: list[TrialResult],
    *,
    base_sequence_layers: int,
    max_candidates: int,
) -> list[tuple[str, TrialResult]]:
    if not completed_trials or max_candidates <= 0:
        return []
    v2_qualified_trials = [item for item in completed_trials if _trial_is_portfolio_daily_v2_qualified(item)]
    evidence_sufficient_trials = [item for item in completed_trials if _trial_has_sufficient_training_evidence(item)]
    if v2_qualified_trials:
        completed_trials = v2_qualified_trials
    elif evidence_sufficient_trials:
        completed_trials = evidence_sufficient_trials
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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=None)
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
    parser.add_argument(
        "--resource-profile",
        default="auto",
        choices=RESOURCE_PROFILE_CHOICES,
        help="Resource guard for subprocess studies. auto uses safe mode for true solver profiles and balanced mode otherwise.",
    )
    parser.add_argument(
        "--thread-limit",
        type=int,
        default=0,
        help="Override BLAS/OpenMP/Torch thread count for protocol subprocesses; 0 lets the resource profile decide.",
    )
    parser.add_argument(
        "--process-priority",
        default="auto",
        choices=RESOURCE_PRIORITY_CHOICES,
        help="Windows process priority for protocol subprocesses.",
    )
    parser.add_argument(
        "--cpu-affinity-count",
        type=int,
        default=0,
        help="Limit protocol subprocesses to the first N logical CPUs; 0 lets the resource profile decide.",
    )
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
    resource_limits = _build_resource_limits(
        search_profile=args.search_profile,
        resource_profile=args.resource_profile,
        thread_limit=args.thread_limit if int(args.thread_limit or 0) > 0 else None,
        process_priority=args.process_priority,
        cpu_affinity_count=args.cpu_affinity_count if int(args.cpu_affinity_count or 0) > 0 else None,
    )

    base_trial = dict(SEARCH_PROFILE_BASE_TRIALS[args.search_profile])
    if args.epochs is not None:
        base_trial["epochs"] = int(args.epochs)
    else:
        base_trial.setdefault("epochs", 40)
    if args.min_epochs is not None:
        base_trial["min_epochs"] = int(args.min_epochs)
    else:
        base_trial.setdefault("min_epochs", 32)
    active_budget_semantics = str(base_trial.get("budget_semantics", args.budget_semantics))
    active_budget_calibration = str(base_trial.get("budget_calibration", args.budget_calibration))
    active_budget_objective = str(base_trial.get("budget_objective", args.budget_objective))
    active_daily_head_layout = str(base_trial.get("daily_head_layout", args.daily_head_layout))
    active_alpha_prior_source = str(base_trial.get("alpha_prior_source", args.alpha_prior_source))
    selected_trials = _select_trials(
        base=base_trial,
        profile_name=args.search_profile,
        trial_count=args.trial_count,
        random_seed=args.random_seed,
    )
    selected_trial_loss_profiles: list[str] = []
    selected_trial_resolved_loss_configs: list[dict[str, Any]] = []
    for index, trial_config in enumerate(selected_trials, start=1):
        resolved_loss_name, resolved_loss_config = resolve_loss_profile(
            str(trial_config.get("loss_profile", DEFAULT_LOSS_PROFILE))
        )
        selected_trial_loss_profiles.append(resolved_loss_name)
        selected_trial_resolved_loss_configs.append(resolved_loss_config)
    native_allocation_vector_support = any(
        float(
            config.get("multi_objective_loss_weights", {}).get(
                "portfolio_native_allocation_vector_total",
                0.0,
            )
        )
        > 0.0
        for config in selected_trial_resolved_loss_configs
    )
    day_set_native_allocation_vector_support = any(
        float(
            config.get("multi_objective_loss_weights", {}).get(
                "portfolio_day_set_native_allocation_vector_total",
                0.0,
            )
        )
        > 0.0
        for config in selected_trial_resolved_loss_configs
    )
    full_universe_train_solver_effective = bool(
        CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_SOLVER_ENABLED
        and any(_loss_profile_enables_full_universe_train_solver(name) for name in selected_trial_loss_profiles)
    )
    native_source_delta_alignment_support = bool(
        native_allocation_vector_support or day_set_native_allocation_vector_support
    )
    native_target_validity_closure_support = any(
        name in {"alpha_result_value_budget_split_v38", "alpha_result_value_budget_split_v39", "alpha_result_value_budget_split_v40"}
        for name in selected_trial_loss_profiles
    )
    native_executable_receiver_closure_support = any(
        name in {"alpha_result_value_budget_split_v39", "alpha_result_value_budget_split_v40"}
        for name in selected_trial_loss_profiles
    )
    native_validation_closure_support = any(
        name == "alpha_result_value_budget_split_v40" for name in selected_trial_loss_profiles
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
            selected_trial_loss_profiles = []
            selected_trial_resolved_loss_configs = []
            for trial_config in selected_trials:
                resolved_loss_name, resolved_loss_config = resolve_loss_profile(
                    str(trial_config.get("loss_profile", DEFAULT_LOSS_PROFILE))
                )
                selected_trial_loss_profiles.append(resolved_loss_name)
                selected_trial_resolved_loss_configs.append(resolved_loss_config)
            native_allocation_vector_support = any(
                float(
                    config.get("multi_objective_loss_weights", {}).get(
                        "portfolio_native_allocation_vector_total",
                        0.0,
                    )
                )
                > 0.0
                for config in selected_trial_resolved_loss_configs
            )
            day_set_native_allocation_vector_support = any(
                float(
                    config.get("multi_objective_loss_weights", {}).get(
                        "portfolio_day_set_native_allocation_vector_total",
                        0.0,
                    )
                )
                > 0.0
                for config in selected_trial_resolved_loss_configs
            )
            full_universe_train_solver_effective = bool(
                CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_SOLVER_ENABLED
                and any(_loss_profile_enables_full_universe_train_solver(name) for name in selected_trial_loss_profiles)
            )
            native_source_delta_alignment_support = bool(
                native_allocation_vector_support or day_set_native_allocation_vector_support
            )
            native_target_validity_closure_support = any(
                name in {"alpha_result_value_budget_split_v38", "alpha_result_value_budget_split_v39", "alpha_result_value_budget_split_v40"}
                for name in selected_trial_loss_profiles
            )
            native_executable_receiver_closure_support = any(
                name in {"alpha_result_value_budget_split_v39", "alpha_result_value_budget_split_v40"}
                for name in selected_trial_loss_profiles
            )
            native_validation_closure_support = any(
                name == "alpha_result_value_budget_split_v40" for name in selected_trial_loss_profiles
            )
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
        "budget_semantics": active_budget_semantics,
        "budget_calibration": active_budget_calibration,
        "budget_objective": active_budget_objective,
        "daily_head_layout": active_daily_head_layout,
        "alpha_prior_source": active_alpha_prior_source,
        "alpha_prior_score_panel": str(args.alpha_prior_score_panel),
        "alpha_prior_target_weight_panel": str(args.alpha_prior_target_weight_panel),
        "trial_count": len(selected_trials),
        "base_trial": base_trial,
        "selected_trials": selected_trials,
        "selected_trial_loss_profiles": selected_trial_loss_profiles,
        "native_allocation_vector_support": bool(native_allocation_vector_support),
        "day_set_native_allocation_vector_support": bool(day_set_native_allocation_vector_support),
        "native_source_delta_alignment_support": bool(native_source_delta_alignment_support),
        "native_target_validity_closure_support": bool(native_target_validity_closure_support),
        "native_executable_receiver_closure_support": bool(native_executable_receiver_closure_support),
        "native_validation_closure_support": bool(native_validation_closure_support),
        "full_universe_train_solver_effective": bool(full_universe_train_solver_effective),
        "resource_limits": resource_limits,
        "resource_gate": RESOURCE_GATED_SEARCH_PROFILES.get(args.search_profile, {}),
        "study_progress_json": str((study_root / "study_progress.json").resolve()),
        "study_progress_jsonl": str((study_root / "study_progress.jsonl").resolve()),
    }
    write_json(study_root / "study_plan.json", study_plan)
    _write_study_progress_event(
        study_root,
        event="study_plan_written",
        status="dry_run" if args.dry_run else "planned",
        study_tag=study_tag,
        search_profile=args.search_profile,
        objective_profile=objective_profile,
        trial_count=len(selected_trials),
        screening_trial_count=len(selected_trials),
        confirmatory_enabled=not bool(args.disable_confirmatory),
        selected_trial_loss_profiles=selected_trial_loss_profiles,
        native_allocation_vector_support=bool(native_allocation_vector_support),
        day_set_native_allocation_vector_support=bool(day_set_native_allocation_vector_support),
        native_source_delta_alignment_support=bool(native_source_delta_alignment_support),
        native_target_validity_closure_support=bool(native_target_validity_closure_support),
        native_executable_receiver_closure_support=bool(native_executable_receiver_closure_support),
        native_validation_closure_support=bool(native_validation_closure_support),
        full_universe_train_solver_effective=bool(full_universe_train_solver_effective),
        resource_gate=RESOURCE_GATED_SEARCH_PROFILES.get(args.search_profile, {}),
        resource_limits=resource_limits,
    )
    if args.dry_run:
        safe_print_json(study_plan)
        return 0

    latest_snapshot = _snapshot_latest_state()
    screening_results: list[TrialResult] = []
    confirmatory_results: list[TrialResult] = []
    resource_gate_summary: dict[str, Any] = {
        "resource_gate_enabled": bool(args.search_profile in RESOURCE_GATED_SEARCH_PROFILES),
        "resource_gate_triggered": False,
        "continue_screening": True,
        "failed_resource_checks": [],
        "estimated_saved_screening_trials": 0,
    }
    try:
        _write_study_progress_event(
            study_root,
            event="study_start",
            status="running",
            study_tag=study_tag,
            screening_trial_count=len(selected_trials),
            confirmatory_enabled=not bool(args.disable_confirmatory),
            native_allocation_vector_support=bool(native_allocation_vector_support),
            day_set_native_allocation_vector_support=bool(day_set_native_allocation_vector_support),
            native_source_delta_alignment_support=bool(native_source_delta_alignment_support),
            native_target_validity_closure_support=bool(native_target_validity_closure_support),
            native_executable_receiver_closure_support=bool(native_executable_receiver_closure_support),
            native_validation_closure_support=bool(native_validation_closure_support),
            full_universe_train_solver_effective=bool(full_universe_train_solver_effective),
            process_id=os.getpid(),
            resource_limits=resource_limits,
        )
        if args.skip_screening and screening_seed_trials:
            screening_results = list(screening_seed_trials)
            _write_study_progress_event(
                study_root,
                event="screening_seed_loaded",
                status="completed",
                study_tag=study_tag,
                loaded_trial_count=len(screening_results),
            )
        else:
            for index, trial_config in enumerate(selected_trials, start=1):
                trial_tag = f"{study_tag}__trial_{index:02d}"
                protocol_args = _build_protocol_args(args, trial_config, trial_tag)
                trial_error = ""
                try:
                    exit_code = int(
                        _run_protocol_with_progress(
                            study_root=study_root,
                            phase="screening",
                            role="",
                            trial_id=index,
                            trial_tag=trial_tag,
                            source_trial_tag="",
                            protocol_args=protocol_args,
                            progress_context={
                                "study_tag": study_tag,
                                "progress_index": index,
                                "progress_total": len(selected_trials),
                                "progress_label": f"screening {index}/{len(selected_trials)}",
                            },
                            resource_limits=resource_limits,
                        )
                    )
                    if exit_code != 0:
                        raise RuntimeError(f"protocol exited with code {exit_code}")
                    protocol_summary_path = PROTOCOLS_ROOT / trial_tag / "protocol_summary.json"
                    built = _build_trial_result_from_protocol(
                        trial_id=index,
                        trial_tag=trial_tag,
                        trial_config=trial_config,
                        protocol_summary_path=protocol_summary_path,
                        objective_profile=objective_profile,
                    )
                    screening_results.append(built)
                    _write_study_progress_event(
                        study_root,
                        event="trial_completed",
                        status="completed",
                        study_tag=study_tag,
                        phase="screening",
                        trial_id=index,
                        trial_tag=trial_tag,
                        progress_index=index,
                        progress_total=len(selected_trials),
                        composite_score=float(built.composite_score),
                        performance_score=float(built.performance_score),
                        stability_score=float(built.stability_score),
                        protocol_summary_json=str(protocol_summary_path.resolve()),
                    )
                except Exception as exc:
                    trial_error = traceback.format_exc().strip() or str(exc)
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
                    _write_study_progress_event(
                        study_root,
                        event="trial_failed",
                        status="failed",
                        study_tag=study_tag,
                        phase="screening",
                        trial_id=index,
                        trial_tag=trial_tag,
                        progress_index=index,
                        progress_total=len(selected_trials),
                        protocol_summary_json=str((PROTOCOLS_ROOT / trial_tag / "protocol_summary.json").resolve()),
                        error=trial_error[-4000:],
                    )
                    runtime_channel_reason = _detect_runtime_channel_failure(trial_error)
                    if runtime_channel_reason:
                        remaining = max(0, int(len(selected_trials)) - int(index))
                        resource_gate_summary = {
                            "resource_gate_enabled": True,
                            "resource_gate_triggered": True,
                            "continue_screening": False,
                            "failed_resource_checks": ["runtime_channel_failure"],
                            "estimated_saved_screening_trials": remaining,
                            "runtime_channel_failure": True,
                            "runtime_channel_failure_reason": runtime_channel_reason,
                        }
                        _write_study_progress_event(
                            study_root,
                            event="runtime_channel_stopped_screening",
                            status="stopped",
                            study_tag=study_tag,
                            phase="screening",
                            trial_id=index,
                            trial_tag=trial_tag,
                            progress_index=index,
                            progress_total=len(selected_trials),
                            **resource_gate_summary,
                        )
                        break
                resource_gate_summary = _resource_gate_after_screening(
                    args.search_profile,
                    screening_results,
                    selected_trial_count=len(selected_trials),
                )
                if bool(resource_gate_summary.get("resource_gate_triggered", False)):
                    _write_study_progress_event(
                        study_root,
                        event="resource_gate_stopped_screening",
                        status="stopped",
                        study_tag=study_tag,
                        phase="screening",
                        progress_index=index,
                        progress_total=len(selected_trials),
                        **resource_gate_summary,
                    )
                    break

        screening_results.sort(key=lambda item: float(item.composite_score), reverse=True)
        completed_screening = [item for item in screening_results if item.status == "completed"]
        if not args.disable_confirmatory and not bool(resource_gate_summary.get("resource_gate_triggered", False)):
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
                    exit_code = int(
                        _run_protocol_with_progress(
                            study_root=study_root,
                            phase="confirmatory",
                            role=role,
                            trial_id=index,
                            trial_tag=confirm_tag,
                            source_trial_tag=source_trial.trial_tag,
                            protocol_args=protocol_args,
                            progress_context={
                                "study_tag": study_tag,
                                "progress_index": index,
                                "progress_total": len(confirmatory_candidates),
                                "progress_label": f"confirmatory {index}/{len(confirmatory_candidates)}",
                            },
                            resource_limits=resource_limits,
                        )
                    )
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
                    _write_study_progress_event(
                        study_root,
                        event="trial_completed",
                        status="completed",
                        study_tag=study_tag,
                        phase="confirmatory",
                        role=role,
                        source_trial_tag=source_trial.trial_tag,
                        trial_id=index,
                        trial_tag=confirm_tag,
                        progress_index=index,
                        progress_total=len(confirmatory_candidates),
                        composite_score=float(built.composite_score),
                        performance_score=float(built.performance_score),
                        stability_score=float(built.stability_score),
                        protocol_summary_json=str(protocol_summary_path.resolve()),
                    )
                except Exception as exc:
                    trial_error = traceback.format_exc().strip() or str(exc)
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
                    _write_study_progress_event(
                        study_root,
                        event="trial_failed",
                        status="failed",
                        study_tag=study_tag,
                        phase="confirmatory",
                        role=role,
                        source_trial_tag=source_trial.trial_tag,
                        trial_id=index,
                        trial_tag=confirm_tag,
                        progress_index=index,
                        progress_total=len(confirmatory_candidates),
                        protocol_summary_json=str((PROTOCOLS_ROOT / confirm_tag / "protocol_summary.json").resolve()),
                        error=trial_error[-4000:],
                    )
                    runtime_channel_reason = _detect_runtime_channel_failure(trial_error)
                    if runtime_channel_reason:
                        remaining = max(0, int(len(confirmatory_candidates)) - int(index))
                        _write_study_progress_event(
                            study_root,
                            event="runtime_channel_stopped_confirmatory",
                            status="stopped",
                            study_tag=study_tag,
                            phase="confirmatory",
                            role=role,
                            source_trial_tag=source_trial.trial_tag,
                            trial_id=index,
                            trial_tag=confirm_tag,
                            progress_index=index,
                            progress_total=len(confirmatory_candidates),
                            runtime_channel_failure=True,
                            runtime_channel_failure_reason=runtime_channel_reason,
                            estimated_saved_confirmatory_trials=remaining,
                        )
                        break
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
    screening_by_tag = {item.trial_tag: item for item in completed_screening}
    confirmatory_stability_checks: list[dict[str, Any]] = []
    stable_confirmatory_tags: set[str] = set()
    for item in completed_confirmatory:
        source_item = screening_by_tag.get(str(item.source_trial_tag or ""))
        stability = _portfolio_daily_v2_confirm_stability(
            item.primary_metrics,
            source_item.primary_metrics if source_item is not None else {},
        )
        stability.update(
            {
                "trial_tag": item.trial_tag,
                "source_trial_tag": str(item.source_trial_tag or ""),
                "role": item.role,
                "source_found": source_item is not None,
            }
        )
        confirmatory_stability_checks.append(stability)
        if bool(stability.get("stable_confirmatory")):
            stable_confirmatory_tags.add(item.trial_tag)
    qualified_confirmatory = list(completed_confirmatory)
    rejected_confirmatory = []
    if objective_profile in PORTFOLIO_DAILY_GATE_OBJECTIVES:
        qualified_confirmatory = [
            item for item in completed_confirmatory if item.trial_tag in stable_confirmatory_tags
        ]
        rejected_confirmatory = [
            {
                **item.to_summary(),
                "portfolio_daily_v2_confirm_stability": next(
                    (
                        stability
                        for stability in confirmatory_stability_checks
                        if stability.get("trial_tag") == item.trial_tag
                    ),
                    {},
                ),
            }
            for item in completed_confirmatory
            if item.trial_tag not in stable_confirmatory_tags
        ]
    screening_fallback_pool = completed_screening
    if objective_profile in PORTFOLIO_DAILY_GATE_OBJECTIVES:
        v2_qualified_screening = [
            item for item in completed_screening if _trial_is_portfolio_daily_v2_qualified(item)
        ]
        evidence_sufficient_screening = [
            item for item in completed_screening if _trial_has_sufficient_training_evidence(item)
        ]
        if v2_qualified_screening:
            screening_fallback_pool = v2_qualified_screening
        elif evidence_sufficient_screening:
            screening_fallback_pool = evidence_sufficient_screening
    champion = (
        qualified_confirmatory[0].to_summary()
        if qualified_confirmatory
        else (screening_fallback_pool[0].to_summary() if screening_fallback_pool else {})
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
        "budget_semantics": active_budget_semantics,
        "budget_calibration": active_budget_calibration,
        "study_plan_json": str((study_root / "study_plan.json").resolve()),
        "study_progress_json": str((study_root / "study_progress.json").resolve()),
        "study_progress_jsonl": str((study_root / "study_progress.jsonl").resolve()),
        "trial_ranking_csv": str((study_root / "trial_ranking.csv").resolve()),
        "trial_count": len(selected_trials),
        "native_allocation_vector_support": bool(native_allocation_vector_support),
        "day_set_native_allocation_vector_support": bool(day_set_native_allocation_vector_support),
        "native_source_delta_alignment_support": bool(native_source_delta_alignment_support),
        "native_target_validity_closure_support": bool(native_target_validity_closure_support),
        "native_executable_receiver_closure_support": bool(native_executable_receiver_closure_support),
        "native_validation_closure_support": bool(native_validation_closure_support),
        "full_universe_train_solver_effective": bool(full_universe_train_solver_effective),
        "resource_limits": resource_limits,
        "completed_trial_count": len(completed_screening),
        "failed_trial_count": len([item for item in screening_results if item.status != "completed"]),
        "screening_trials": [item.to_summary() for item in screening_results],
        "confirmatory_trials": [item.to_summary() for item in confirmatory_results],
        "trials": ranking_rows,
        "screen_performance_champion": screen_performance_champion,
        "screen_stability_champion": screen_stability_champion,
        "screen_depth_challenger": screen_depth_challenger,
        "champion": champion,
        "champion_selection_policy": (
            "portfolio_daily_v2_stable_confirmatory_then_screening_fallback"
            if objective_profile in PORTFOLIO_DAILY_GATE_OBJECTIVES
            else "confirmatory_preferred"
        ),
        "portfolio_daily_v2_confirm_stability_checks": (
            confirmatory_stability_checks if objective_profile in PORTFOLIO_DAILY_GATE_OBJECTIVES else []
        ),
        "portfolio_daily_v2_stable_confirmatory_trials": (
            [item.to_summary() for item in completed_confirmatory if item.trial_tag in stable_confirmatory_tags]
            if objective_profile in PORTFOLIO_DAILY_GATE_OBJECTIVES
            else []
        ),
        "rejected_confirmatory_trials": rejected_confirmatory,
        "resource_gate": resource_gate_summary,
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
    _write_study_progress_event(
        study_root,
        event="study_complete",
        status="completed",
        study_tag=study_tag,
        completed_trial_count=len(completed_screening),
        failed_trial_count=len([item for item in screening_results if item.status != "completed"]),
        confirmatory_completed_trial_count=len(completed_confirmatory),
        native_allocation_vector_support=bool(native_allocation_vector_support),
        day_set_native_allocation_vector_support=bool(day_set_native_allocation_vector_support),
        native_source_delta_alignment_support=bool(native_source_delta_alignment_support),
        native_target_validity_closure_support=bool(native_target_validity_closure_support),
        native_executable_receiver_closure_support=bool(native_executable_receiver_closure_support),
        native_validation_closure_support=bool(native_validation_closure_support),
        full_universe_train_solver_effective=bool(full_universe_train_solver_effective),
        study_summary_json=str((study_root / "study_summary.json").resolve()),
        trial_ranking_csv=str((study_root / "trial_ranking.csv").resolve()),
    )
    safe_print_json(study_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
