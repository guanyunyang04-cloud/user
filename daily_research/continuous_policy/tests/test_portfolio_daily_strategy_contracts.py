import json
import inspect
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
import pandas as pd

import daily_research.continuous_policy.model_seq_v3 as model_seq_v3
from daily_research.continuous_policy.allocation_optimizer import (
    AllocationOptimizerConstraints,
    build_unified_allocation_problem,
    solve_semidifferentiable_allocation,
)
from daily_research.continuous_policy.analyze_behavior_gap import _compute_exposure_utilization_from_turnover
from daily_research.continuous_policy.allocation_teacher import build_allocation_teacher_summary
from daily_research.continuous_policy.model_seq_v3 import (
    DailyControllerNet,
    DaySetTensorDataset,
    LOSS_PROFILE_CONFIGS,
    TemporalDaySetPolicyNet,
    TemporalSamplePolicyNet,
    TorchContinuousPolicySeqArtifact,
    _allocation_objective_consolidation_loss,
    _collate_day_set_batch,
    _cvxpy_convex_layer_available,
    _cvxpy_convex_layer_status,
    _decision_focused_allocation_regret_loss,
    _portfolio_differentiable_convex_allocation_loss,
    _portfolio_capital_flow_closure_loss,
    _portfolio_cvxpy_convex_allocation_loss,
    _portfolio_entropic_transport_decision_loss,
    _portfolio_full_universe_convex_allocation_loss,
    _portfolio_day_set_native_allocation_vector_loss,
    _portfolio_native_allocation_vector_loss,
    _project_day_set_native_allocation_vector,
    _project_native_allocation_vector,
    _loss_profile_enables_full_universe_train_solver,
    _portfolio_offline_conservative_support_loss,
    _portfolio_primal_dual_decision_loss,
    _portfolio_utility_credit_closure_loss,
    _source_hard_negative_tail_loss,
    _source_listwise_release_regret_loss,
    _transfer_level_allocation_regret_loss,
    _unified_allocation_consistency_loss,
    load_torch_seq_artifact,
    predict_policy_v3,
)
from daily_research.continuous_policy.model_v2 import ACTION_CLASSES, DURATION_CLASSES
from daily_research.continuous_policy.portfolio_simulator import (
    BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
    BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC,
    BUDGET_SEMANTICS_ALLOCATION_LAYER,
    BUDGET_SEMANTICS_SPLIT,
    HoldingState,
    PortfolioState,
    normalize_budget_calibration,
    normalize_budget_semantics,
)
from daily_research.continuous_policy.run_self_optimizing_study import (
    SEARCH_PROFILE_BASE_TRIALS,
    SEARCH_PROFILE_DEFAULT_OBJECTIVES,
    SEARCH_PROFILES,
    TRUE_SOLVER_RESOURCE_SEARCH_PROFILES,
    TrialResult,
    _build_resource_limits,
    _pick_confirmatory_candidates,
    _portfolio_daily_v2_confirm_stability,
    _resource_gate_after_screening,
    _resource_limited_child_env,
    _run_protocol_with_progress,
    _score_protocol_summary,
    _write_study_progress_event,
)
from daily_research.continuous_policy.runtime import safe_print_json


def _protocol_summary(metrics: dict | None = None, semantic: dict | None = None) -> dict:
    return {
        "evaluation": {
            "continuous_policy_metrics": {
                "annual_return": 0.18,
                "sharpe": 0.82,
                "monthly_return_mean": 0.012,
                "monthly_win_rate": 0.60,
                "monthly_consistency_score": 0.56,
                "monthly_worst_return": -0.020,
                "monthly_intramonth_max_drawdown": -0.030,
                "max_drawdown": -0.10,
                "avg_gross_exposure": 0.56,
                **(metrics or {}),
            },
            "continuity_metrics": {},
        },
        "promotion_gate": {"checks": {"annual_return": True, "sharpe": True}},
        "training_evidence": {"status": "sufficient"},
        "latest_behavior_audit": {"semantic_conflicts": _healthy_semantic_conflicts() | (semantic or {})},
    }


def _healthy_semantic_conflicts() -> dict:
    return {
        "direct_action_value_mode_share": 1.0,
        "direct_action_intent_preserved_share": 1.0,
        "portfolio_daily_receiver_candidate_count": 4,
        "portfolio_daily_receiver_open_breadth_candidate_count": 3,
        "portfolio_daily_receiver_target_count": 4,
        "portfolio_daily_receiver_realized_deploy_rate": 0.92,
        "portfolio_daily_receiver_unrealized_deploy_share": 0.0,
        "portfolio_daily_source_candidate_count": 4,
        "portfolio_daily_source_target_count": 4,
        "portfolio_daily_source_realized_sell_rate": 0.50,
        "portfolio_daily_source_target_not_sold_share": 0.35,
        "portfolio_daily_receiver_minus_source_forward_excess_5d": 0.012,
        "portfolio_daily_receiver_forward_excess_5d": 0.014,
        "portfolio_daily_source_forward_excess_5d": -0.004,
        "portfolio_daily_source_positive_forward_sell_share": 0.20,
        "portfolio_daily_source_strong_positive_forward_sell_count": 0,
        "portfolio_daily_source_max_forward_excess_5d": 0.030,
        "portfolio_daily_source_p75_forward_excess_5d": 0.000,
        "portfolio_daily_source_economic_release_score_mean": 0.22,
        "portfolio_daily_source_bad_forward_spread_risk_mean": 0.22,
        "portfolio_daily_source_economic_block_risk_mean": 0.24,
        "portfolio_daily_source_forward_strength_brake_risk_mean": 0.20,
        "portfolio_daily_source_forward_proxy_keep_risk_mean": 0.14,
        "portfolio_daily_source_release_conviction_mean": 0.42,
        "portfolio_daily_cash_reserve_rate": 0.12,
        "avg_gross_exposure_target": 0.64,
        "portfolio_daily_exposure_utilization": 0.86,
        "authorized_add_no_weight_change_share": 0.0,
        "deploy_intent_unrealized_share": 0.0,
        "direct_action_authorization_subset_violation_count": 0.0,
        "order_translation_conflict_rate": 0.0,
        "direct_action_order_translation_conflict_rate": 0.0,
        "add_to_hold_conflict_share": 0.0,
    }


def _trial(
    tag: str,
    *,
    evidence_status: str,
    performance_score: float,
    stability_score: float,
    composite_score: float,
) -> TrialResult:
    return TrialResult(
        trial_id=1,
        trial_tag=tag,
        status="completed",
        phase="screening",
        role="",
        source_trial_tag="",
        trial_config={"sequence_layers": 2},
        protocol_summary_path="",
        performance_score=performance_score,
        stability_score=stability_score,
        composite_score=composite_score,
        score_breakdown={},
        primary_metrics={
            "training_evidence_status": evidence_status,
            "annual_return": 0.30,
            "sharpe": 1.00,
            "monthly_return_mean": 0.020,
            "monthly_consistency_score": 0.60,
            "max_drawdown": -0.08,
        },
        promotion_status="shadow_only",
        failed_checks=[],
        gate_pass_ratio=1.0,
        passed_check_count=1,
        total_check_count=1,
    )


def _base_policy(index: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(index=index)
    base_values = {
        "action_label": "skip",
        "action_strength": 0.10,
        "target_delta_hint": 0.0,
        "hold_boost": 0.0,
        "portfolio_daily_unified_receiver_score": 0.0,
        "portfolio_daily_unified_source_score": 0.0,
        "portfolio_daily_unified_cash_score": 0.05,
        "portfolio_daily_source_positive_forward_penalty": 0.0,
        "portfolio_daily_source_opportunity_cost_penalty": 0.0,
        "portfolio_daily_receiver_source_spread_reward": 0.0,
        "portfolio_daily_unified_allocation_objective": 0.0,
        "portfolio_daily_receiver_score": 0.0,
        "portfolio_daily_receiver_executability": 0.0,
        "portfolio_daily_source_score": 0.0,
        "portfolio_daily_source_release_capacity": 1.0,
        "portfolio_daily_source_release_quality": 0.0,
        "portfolio_daily_source_opportunity_cost": 0.20,
        "portfolio_daily_source_executability": 0.0,
        "portfolio_daily_source_forward_proxy_keep_risk": 0.0,
        "portfolio_daily_source_forward_strength_brake_risk": 0.0,
        "portfolio_daily_source_bad_forward_spread_risk": 0.0,
        "portfolio_daily_source_economic_release_score": 0.0,
        "portfolio_daily_source_economic_block_risk": 0.0,
        "deploy_value_target": 0.0,
        "release_value_target": 0.0,
        "deploy_gate_target": 0.0,
        "release_gate_target": 0.0,
        "alpha_opportunity_value": 0.0,
        "deployment_opportunity_cost": 0.0,
        "cash_defense_value": 0.05,
        "exit_timing_pressure": 0.0,
        "multi_horizon_forward_value": 0.0,
        "multi_horizon_path_value": 0.0,
        "multi_horizon_forward_risk": 0.0,
        "hold_continuation_value": 0.20,
        "large_upside_1d_target": 0.0,
    }
    for column, value in base_values.items():
        frame[column] = value
    return frame


class _FakePolicyModel:
    def __call__(self, static_x: torch.Tensor, sequence_x: torch.Tensor) -> dict[str, torch.Tensor]:
        row_count = int(static_x.shape[0])
        ones = torch.ones(row_count, dtype=torch.float32)
        zeros = torch.zeros(row_count, dtype=torch.float32)
        action_logits = torch.zeros((row_count, len(ACTION_CLASSES)), dtype=torch.float32)
        action_logits[:, ACTION_CLASSES.index("skip")] = 4.0
        duration_logits = torch.zeros((row_count, len(DURATION_CLASSES)), dtype=torch.float32)
        return {
            "action_logits": action_logits,
            "duration_logits": duration_logits,
            "target_delta_hint": zeros,
            "entry_quality": zeros,
            "hold_quality": zeros,
            "add_quality": zeros,
            "reduce_quality": zeros,
            "exit_urgency": zeros,
            "reentry_readiness": zeros,
            "holding_days_ratio": zeros,
            "portfolio_daily_receiver_add_headroom": ones * 0.80,
            "portfolio_daily_receiver_add_capacity": ones * 0.80,
            "portfolio_daily_receiver_executability": ones * 0.80,
            "portfolio_daily_receiver_score": ones * 0.20,
            "portfolio_daily_source_score": ones * 0.10,
            "portfolio_daily_cash_score": ones * 0.30,
            "portfolio_daily_unified_receiver_score": torch.linspace(0.62, 0.82, row_count),
            "portfolio_daily_unified_source_score": torch.linspace(0.22, 0.42, row_count),
            "portfolio_daily_unified_cash_score": torch.linspace(0.12, 0.32, row_count),
            "portfolio_daily_source_positive_forward_penalty": torch.linspace(0.02, 0.12, row_count),
            "portfolio_daily_source_opportunity_cost_penalty": torch.linspace(0.04, 0.14, row_count),
            "portfolio_daily_receiver_source_spread_reward": torch.linspace(0.30, 0.50, row_count),
            "portfolio_daily_unified_allocation_objective": torch.linspace(0.55, 0.75, row_count),
            "portfolio_daily_allocation_trade_quality_target": torch.linspace(0.58, 0.78, row_count),
            "portfolio_daily_allocation_cash_deployment_target": torch.linspace(0.62, 0.82, row_count),
            "portfolio_daily_allocation_risk_adjusted_return_target": torch.linspace(0.56, 0.76, row_count),
            "portfolio_daily_allocation_drawdown_control_target": torch.linspace(0.08, 0.18, row_count),
            "portfolio_daily_allocation_monthly_quality_target": torch.linspace(0.54, 0.74, row_count),
            "portfolio_daily_allocation_final_objective": torch.linspace(0.60, 0.80, row_count),
        }


class _FakeDailyModel:
    def __call__(self, daily_x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "gross_exposure_target": torch.tensor([0.80], dtype=torch.float32),
            "candidate_budget": torch.tensor([4.0], dtype=torch.float32),
            "turnover_budget": torch.tensor([0.40], dtype=torch.float32),
            "max_position_weight_target": torch.tensor([0.20], dtype=torch.float32),
            "hold_bias_target": torch.tensor([0.20], dtype=torch.float32),
            "budget_risk_signal_target": torch.tensor([0.10], dtype=torch.float32),
            "budget_deploy_signal_target": torch.tensor([0.80], dtype=torch.float32),
            "budget_cash_timing_signal_target": torch.tensor([0.20], dtype=torch.float32),
            "budget_alpha_focus_signal_target": torch.tensor([0.70], dtype=torch.float32),
        }


class PortfolioDailyStrategyContractsTest(unittest.TestCase):
    def test_v2_scoring_rewards_receiver_and_clean_source_breadth(self) -> None:
        narrow = _score_protocol_summary(
            _protocol_summary(
                semantic={
                    "portfolio_daily_receiver_candidate_count": 3,
                    "portfolio_daily_receiver_open_breadth_candidate_count": 1,
                    "portfolio_daily_source_candidate_count": 3,
                }
            ),
            objective_profile="portfolio_daily_ranking_v2_gated",
        )
        wide = _score_protocol_summary(
            _protocol_summary(
                semantic={
                    "portfolio_daily_receiver_candidate_count": 9,
                    "portfolio_daily_receiver_open_breadth_candidate_count": 7,
                    "portfolio_daily_source_candidate_count": 9,
                }
            ),
            objective_profile="portfolio_daily_ranking_v2_gated",
        )

        narrow_perf = narrow["score_breakdown"]["performance"]
        wide_perf = wide["score_breakdown"]["performance"]
        self.assertIn("portfolio_daily_receiver_candidate_breadth", wide_perf)
        self.assertIn("portfolio_daily_clean_source_candidate_breadth", wide_perf)
        self.assertGreater(
            wide_perf["portfolio_daily_receiver_candidate_breadth"],
            narrow_perf["portfolio_daily_receiver_candidate_breadth"],
        )
        self.assertGreater(
            wide_perf["portfolio_daily_clean_source_candidate_breadth"],
            narrow_perf["portfolio_daily_clean_source_candidate_breadth"],
        )
        self.assertGreater(wide["composite_score"], narrow["composite_score"])

    def test_v2_scoring_penalizes_joint_economic_quality_gap(self) -> None:
        weak = _score_protocol_summary(
            _protocol_summary(
                metrics={
                    "monthly_return_mean": -0.004,
                    "max_drawdown": -0.19,
                },
                semantic={
                    "portfolio_daily_receiver_realized_deploy_rate": 0.20,
                    "portfolio_daily_receiver_minus_source_forward_excess_5d": -0.018,
                    "portfolio_daily_exposure_utilization": 0.32,
                },
            ),
            objective_profile="portfolio_daily_ranking_v2_gated",
        )

        weak_perf = weak["score_breakdown"]["performance"]
        weak_stability = weak["score_breakdown"]["stability"]
        self.assertIn("portfolio_daily_joint_economic_quality_gate", weak_perf)
        self.assertIn("portfolio_daily_joint_economic_quality_gate", weak_stability)
        self.assertLess(weak_perf["portfolio_daily_joint_economic_quality_gate"], -0.20)
        self.assertLess(weak_stability["portfolio_daily_joint_economic_quality_gate"], -0.15)

    def test_r34_profile_exists_for_bounded_breadth_confirmatory(self) -> None:
        tag = "split_heads_portfolio_daily_allocation_breadth_r34"
        self.assertIn(tag, SEARCH_PROFILES)
        self.assertIn(tag, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[tag], "portfolio_daily_ranking_v2_gated")
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[tag]["loss_profile"], "alpha_result_value_budget_split_v20")
        self.assertEqual(
            SEARCH_PROFILE_BASE_TRIALS[tag]["budget_calibration"],
            "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15",
        )

    def test_allocation_teacher_summary_exposes_source_receiver_cash_targets(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "action_label": "open",
                    "portfolio_daily_receiver_score": 0.82,
                    "portfolio_daily_receiver_candidate_mask": 1.0,
                    "portfolio_daily_source_score": 0.0,
                    "portfolio_daily_source_candidate_mask": 0.0,
                    "portfolio_daily_cash_score": 0.12,
                    "portfolio_daily_allocation_transfer_score": 0.58,
                },
                {
                    "action_label": "reduce",
                    "portfolio_daily_receiver_score": 0.10,
                    "portfolio_daily_receiver_candidate_mask": 0.0,
                    "portfolio_daily_source_score": 0.76,
                    "portfolio_daily_source_candidate_mask": 1.0,
                    "portfolio_daily_cash_score": 0.20,
                    "portfolio_daily_allocation_transfer_score": 0.63,
                },
                {
                    "action_label": "hold",
                    "portfolio_daily_receiver_score": 0.34,
                    "portfolio_daily_receiver_candidate_mask": 0.0,
                    "portfolio_daily_source_score": 0.22,
                    "portfolio_daily_source_candidate_mask": 0.0,
                    "portfolio_daily_cash_score": 0.28,
                    "portfolio_daily_allocation_transfer_score": 0.20,
                },
            ]
        )

        summary = build_allocation_teacher_summary(frame)
        self.assertEqual(summary["allocation_receiver_candidate_count"], 1.0)
        self.assertEqual(summary["allocation_source_candidate_count"], 1.0)
        self.assertGreater(summary["allocation_transfer_intensity_target"], 0.40)
        self.assertGreaterEqual(summary["allocation_cash_reserve_target"], 0.0)
        self.assertLessEqual(summary["allocation_cash_reserve_target"], 1.0)

    def test_unified_allocation_problem_penalizes_positive_forward_and_opportunity_cost_source(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "stock": "strong_source",
                    "current_weight": 0.14,
                    "portfolio_daily_source_score": 0.88,
                    "portfolio_daily_source_candidate_mask": 1.0,
                    "portfolio_daily_source_executability": 1.0,
                    "portfolio_daily_source_forward_excess_5d": 0.12,
                    "portfolio_daily_source_receiver_forward_spread": -0.08,
                    "portfolio_daily_source_opportunity_cost": 0.86,
                },
                {
                    "stock": "clean_source",
                    "current_weight": 0.12,
                    "portfolio_daily_source_score": 0.58,
                    "portfolio_daily_source_candidate_mask": 1.0,
                    "portfolio_daily_source_executability": 1.0,
                    "portfolio_daily_source_forward_excess_5d": -0.04,
                    "portfolio_daily_source_receiver_forward_spread": 0.05,
                    "portfolio_daily_source_opportunity_cost": 0.06,
                },
            ]
        )

        problem = build_unified_allocation_problem(frame)
        by_stock = problem.set_index("stock")

        self.assertGreater(
            by_stock.loc["strong_source", "portfolio_daily_source_positive_forward_penalty"],
            by_stock.loc["clean_source", "portfolio_daily_source_positive_forward_penalty"],
        )
        self.assertGreater(
            by_stock.loc["strong_source", "portfolio_daily_source_opportunity_cost_penalty"],
            by_stock.loc["clean_source", "portfolio_daily_source_opportunity_cost_penalty"],
        )
        self.assertGreater(
            by_stock.loc["clean_source", "portfolio_daily_receiver_source_spread_reward"],
            by_stock.loc["strong_source", "portfolio_daily_receiver_source_spread_reward"],
        )
        self.assertGreater(
            by_stock.loc["clean_source", "portfolio_daily_unified_source_score"],
            by_stock.loc["strong_source", "portfolio_daily_unified_source_score"],
        )

    def test_unified_allocation_problem_applies_hard_negative_penalty_to_strong_false_source(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "stock": "false_source",
                    "current_weight": 0.15,
                    "portfolio_daily_source_score": 0.94,
                    "portfolio_daily_source_candidate_mask": 1.0,
                    "portfolio_daily_source_executability": 1.0,
                    "portfolio_daily_source_release_capacity": 1.0,
                    "portfolio_daily_source_release_quality": 0.86,
                    "portfolio_daily_source_economic_release_score": 0.74,
                    "portfolio_daily_source_forward_excess_5d": 0.16,
                    "portfolio_daily_source_receiver_forward_spread": 0.06,
                    "portfolio_daily_source_opportunity_cost": 0.82,
                    "portfolio_daily_source_forward_strength_brake_risk": 0.70,
                    "portfolio_daily_source_forward_proxy_keep_risk": 0.72,
                },
                {
                    "stock": "clean_source",
                    "current_weight": 0.14,
                    "portfolio_daily_source_score": 0.58,
                    "portfolio_daily_source_candidate_mask": 1.0,
                    "portfolio_daily_source_executability": 1.0,
                    "portfolio_daily_source_release_capacity": 1.0,
                    "portfolio_daily_source_release_quality": 0.62,
                    "portfolio_daily_source_economic_release_score": 0.46,
                    "portfolio_daily_source_forward_excess_5d": -0.05,
                    "portfolio_daily_source_receiver_forward_spread": 0.04,
                    "portfolio_daily_source_opportunity_cost": 0.04,
                },
            ]
        )

        problem = build_unified_allocation_problem(frame).set_index("stock")

        self.assertGreater(
            problem.loc["false_source", "portfolio_daily_source_hard_negative_penalty"],
            problem.loc["clean_source", "portfolio_daily_source_hard_negative_penalty"] + 0.50,
        )
        self.assertGreater(
            problem.loc["false_source", "portfolio_daily_source_strong_false_sell_penalty"],
            0.80,
        )
        self.assertGreater(
            problem.loc["clean_source", "portfolio_daily_unified_source_score"],
            problem.loc["false_source", "portfolio_daily_unified_source_score"] + 0.20,
        )
        self.assertEqual(problem.loc["false_source", "portfolio_daily_unified_source_candidate"], 0.0)

    def test_unified_allocation_problem_exposes_source_release_and_transfer_regret_targets(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "stock": "false_source",
                    "current_weight": 0.15,
                    "portfolio_daily_source_score": 0.94,
                    "portfolio_daily_source_candidate_mask": 1.0,
                    "portfolio_daily_source_executability": 1.0,
                    "portfolio_daily_source_release_capacity": 1.0,
                    "portfolio_daily_source_release_quality": 0.86,
                    "portfolio_daily_source_economic_release_score": 0.74,
                    "portfolio_daily_source_forward_excess_5d": 0.15,
                    "portfolio_daily_source_receiver_forward_spread": -0.08,
                    "portfolio_daily_source_opportunity_cost": 0.82,
                    "portfolio_daily_source_forward_strength_brake_risk": 0.70,
                    "portfolio_daily_source_forward_proxy_keep_risk": 0.72,
                },
                {
                    "stock": "clean_source",
                    "current_weight": 0.14,
                    "portfolio_daily_source_score": 0.58,
                    "portfolio_daily_source_candidate_mask": 1.0,
                    "portfolio_daily_source_executability": 1.0,
                    "portfolio_daily_source_release_capacity": 1.0,
                    "portfolio_daily_source_release_quality": 0.62,
                    "portfolio_daily_source_economic_release_score": 0.46,
                    "portfolio_daily_source_forward_excess_5d": -0.05,
                    "portfolio_daily_source_receiver_forward_spread": 0.07,
                    "portfolio_daily_source_opportunity_cost": 0.04,
                },
            ]
        )

        problem = build_unified_allocation_problem(frame).set_index("stock")

        for column in (
            "portfolio_daily_source_tail_false_sell_penalty",
            "portfolio_daily_source_release_preference",
            "portfolio_daily_transfer_regret_target",
        ):
            self.assertIn(column, problem.columns)
        self.assertGreater(
            problem.loc["clean_source", "portfolio_daily_source_release_preference"],
            problem.loc["false_source", "portfolio_daily_source_release_preference"] + 0.35,
        )
        self.assertGreater(
            problem.loc["clean_source", "portfolio_daily_transfer_regret_target"],
            problem.loc["false_source", "portfolio_daily_transfer_regret_target"] + 0.30,
        )

    def test_unified_allocation_problem_uses_predicted_source_penalties_without_future_labels(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "stock": "predicted_false_source",
                    "current_weight": 0.15,
                    "portfolio_daily_source_score": 0.92,
                    "portfolio_daily_source_candidate_mask": 1.0,
                    "portfolio_daily_source_executability": 1.0,
                    "portfolio_daily_source_release_capacity": 1.0,
                    "portfolio_daily_source_release_quality": 0.82,
                    "portfolio_daily_source_economic_release_score": 0.70,
                    "portfolio_daily_source_positive_forward_penalty": 0.94,
                    "portfolio_daily_source_opportunity_cost_penalty": 0.86,
                    "portfolio_daily_source_forward_strength_brake_risk": 0.66,
                    "portfolio_daily_source_forward_proxy_keep_risk": 0.72,
                },
                {
                    "stock": "predicted_clean_source",
                    "current_weight": 0.14,
                    "portfolio_daily_source_score": 0.62,
                    "portfolio_daily_source_candidate_mask": 1.0,
                    "portfolio_daily_source_executability": 1.0,
                    "portfolio_daily_source_release_capacity": 1.0,
                    "portfolio_daily_source_release_quality": 0.62,
                    "portfolio_daily_source_economic_release_score": 0.50,
                    "portfolio_daily_source_positive_forward_penalty": 0.04,
                    "portfolio_daily_source_opportunity_cost_penalty": 0.04,
                },
            ]
        )

        problem = build_unified_allocation_problem(frame).set_index("stock")

        self.assertGreater(
            problem.loc["predicted_false_source", "portfolio_daily_source_hard_negative_penalty"],
            0.72,
        )
        self.assertEqual(problem.loc["predicted_false_source", "portfolio_daily_unified_source_candidate"], 0.0)
        self.assertGreater(
            problem.loc["predicted_clean_source", "portfolio_daily_unified_source_score"],
            problem.loc["predicted_false_source", "portfolio_daily_unified_source_score"] + 0.20,
        )

    def test_unified_allocation_problem_separates_defensive_cash_from_dead_cash(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "stock": "risk_off_receiver",
                    "current_weight": 0.0,
                    "portfolio_daily_receiver_score": 0.40,
                    "portfolio_daily_receiver_candidate_mask": 1.0,
                    "portfolio_daily_receiver_executability": 0.70,
                    "portfolio_daily_cash_score": 0.45,
                    "market_downside_pressure": 0.80,
                    "cash_regime_pressure": 0.70,
                    "portfolio_drawdown_20d": -0.12,
                    "multi_horizon_forward_risk": 0.65,
                    "portfolio_daily_allocation_transfer_score": 0.16,
                },
                {
                    "stock": "deploy_receiver",
                    "current_weight": 0.0,
                    "portfolio_daily_receiver_score": 0.86,
                    "portfolio_daily_receiver_candidate_mask": 1.0,
                    "portfolio_daily_receiver_executability": 0.90,
                    "portfolio_daily_cash_score": 0.45,
                    "market_downside_pressure": 0.02,
                    "cash_regime_pressure": 0.02,
                    "portfolio_drawdown_20d": -0.01,
                    "multi_horizon_forward_risk": 0.05,
                    "portfolio_daily_receiver_source_spread_reward": 0.62,
                    "portfolio_daily_allocation_transfer_score": 0.70,
                },
            ]
        )

        problem = build_unified_allocation_problem(frame).set_index("stock")

        self.assertGreater(
            problem.loc["risk_off_receiver", "portfolio_daily_unified_cash_score"],
            problem.loc["deploy_receiver", "portfolio_daily_unified_cash_score"] + 0.12,
        )
        self.assertGreater(
            problem.loc["deploy_receiver", "portfolio_daily_unified_allocation_objective"],
            problem.loc["risk_off_receiver", "portfolio_daily_unified_allocation_objective"],
        )

    def test_unified_allocation_problem_uses_forward_benchmark_for_cash_timing_target(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "stock": "downside_day",
                    "current_weight": 0.0,
                    "portfolio_daily_receiver_score": 0.54,
                    "portfolio_daily_receiver_candidate_mask": 1.0,
                    "portfolio_daily_receiver_executability": 0.74,
                    "portfolio_daily_cash_score": 0.26,
                    "market_downside_pressure": 0.18,
                    "cash_regime_pressure": 0.12,
                    "portfolio_drawdown_20d": -0.02,
                    "multi_horizon_forward_risk": 0.20,
                    "forward_benchmark_return_1d": -0.028,
                    "forward_benchmark_return_3d": -0.036,
                    "portfolio_daily_allocation_transfer_score": 0.42,
                },
                {
                    "stock": "upside_day",
                    "current_weight": 0.0,
                    "portfolio_daily_receiver_score": 0.54,
                    "portfolio_daily_receiver_candidate_mask": 1.0,
                    "portfolio_daily_receiver_executability": 0.74,
                    "portfolio_daily_cash_score": 0.26,
                    "market_downside_pressure": 0.18,
                    "cash_regime_pressure": 0.12,
                    "portfolio_drawdown_20d": -0.02,
                    "multi_horizon_forward_risk": 0.20,
                    "forward_benchmark_return_1d": 0.028,
                    "forward_benchmark_return_3d": 0.036,
                    "portfolio_daily_allocation_transfer_score": 0.42,
                },
            ]
        )

        problem = build_unified_allocation_problem(frame).set_index("stock")

        self.assertGreater(
            problem.loc["downside_day", "portfolio_daily_unified_cash_score"],
            problem.loc["upside_day", "portfolio_daily_unified_cash_score"] + 0.10,
        )

    def test_unified_allocation_consistency_loss_penalizes_bad_cash_and_source_distribution(self) -> None:
        targets = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.15, 0.88, 0.20], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.12, 0.08, 0.78], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.tensor([0.86, 0.10, 0.18], dtype=torch.float32),
            "portfolio_daily_source_positive_forward_penalty": torch.tensor([0.02, 0.02, 0.06], dtype=torch.float32),
            "portfolio_daily_source_opportunity_cost_penalty": torch.tensor([0.04, 0.03, 0.05], dtype=torch.float32),
            "portfolio_daily_receiver_source_spread_reward": torch.tensor([0.06, 0.64, 0.58], dtype=torch.float32),
            "portfolio_daily_unified_allocation_objective": torch.tensor([0.18, 0.78, 0.70], dtype=torch.float32),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
            "portfolio_daily_source_candidate_mask": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
            "holding_flag_target": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
            "market_downside_pressure": torch.tensor([0.80, 0.02, 0.02], dtype=torch.float32),
            "cash_regime_pressure": torch.tensor([0.70, 0.02, 0.02], dtype=torch.float32),
            "multi_horizon_forward_risk": torch.tensor([0.68, 0.08, 0.08], dtype=torch.float32),
            "portfolio_daily_allocation_transfer_score": torch.tensor([0.10, 0.74, 0.66], dtype=torch.float32),
        }
        good_outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.10, 0.86, 0.18], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.08, 0.05, 0.76], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.tensor([0.82, 0.12, 0.16], dtype=torch.float32),
            "portfolio_daily_unified_allocation_objective": torch.tensor([0.18, 0.76, 0.68], dtype=torch.float32),
        }
        bad_outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.72, 0.20, 0.70], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.66, 0.70, 0.18], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.tensor([0.12, 0.84, 0.76], dtype=torch.float32),
            "portfolio_daily_unified_allocation_objective": torch.tensor([0.74, 0.20, 0.22], dtype=torch.float32),
        }

        good_loss = _unified_allocation_consistency_loss(good_outputs, targets)
        bad_loss = _unified_allocation_consistency_loss(bad_outputs, targets)

        self.assertLess(float(good_loss.detach().cpu()), float(bad_loss.detach().cpu()) * 0.55)

    def test_decision_focused_allocation_regret_loss_penalizes_false_source_and_dead_cash(self) -> None:
        targets = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.86, 0.08, 0.18], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.00, 0.72], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.tensor([0.12, 0.18, 0.14], dtype=torch.float32),
            "portfolio_daily_unified_allocation_objective": torch.tensor([0.82, 0.04, 0.68], dtype=torch.float32),
            "portfolio_daily_source_positive_forward_penalty": torch.tensor([0.00, 0.96, 0.02], dtype=torch.float32),
            "portfolio_daily_source_opportunity_cost_penalty": torch.tensor([0.00, 0.90, 0.04], dtype=torch.float32),
            "portfolio_daily_source_hard_negative_penalty": torch.tensor([0.00, 0.94, 0.02], dtype=torch.float32),
            "portfolio_daily_source_strong_false_sell_penalty": torch.tensor([0.00, 0.90, 0.00], dtype=torch.float32),
            "portfolio_daily_receiver_source_spread_reward": torch.tensor([0.74, 0.12, 0.62], dtype=torch.float32),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_source_candidate_mask": torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32),
            "portfolio_daily_receiver_forward_excess_5d": torch.tensor([0.09, 0.00, 0.00], dtype=torch.float32),
            "portfolio_daily_source_forward_excess_5d": torch.tensor([0.00, 0.13, -0.07], dtype=torch.float32),
            "portfolio_daily_allocation_transfer_score": torch.tensor([0.76, 0.10, 0.64], dtype=torch.float32),
            "market_downside_pressure": torch.tensor([0.02, 0.02, 0.04], dtype=torch.float32),
            "cash_regime_pressure": torch.tensor([0.02, 0.02, 0.04], dtype=torch.float32),
        }
        good_outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.84, 0.04, 0.18], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.02, 0.70], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.tensor([0.10, 0.16, 0.12], dtype=torch.float32),
            "portfolio_daily_unified_allocation_objective": torch.tensor([0.80, 0.04, 0.66], dtype=torch.float32),
        }
        bad_outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.12, 0.12, 0.12], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.08, 0.82, 0.16], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.tensor([0.74, 0.70, 0.68], dtype=torch.float32),
            "portfolio_daily_unified_allocation_objective": torch.tensor([0.18, 0.70, 0.18], dtype=torch.float32),
        }

        good_loss = _decision_focused_allocation_regret_loss(good_outputs, targets)
        bad_loss = _decision_focused_allocation_regret_loss(bad_outputs, targets)

        self.assertGreater(float(bad_loss.detach().cpu()), float(good_loss.detach().cpu()) * 2.0)

    def test_source_hard_negative_tail_loss_reweights_false_source_pressure(self) -> None:
        targets = {
            "portfolio_daily_source_candidate_mask": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
            "portfolio_daily_source_hard_negative_penalty": torch.tensor([0.94, 0.02, 0.00], dtype=torch.float32),
            "portfolio_daily_source_strong_false_sell_penalty": torch.tensor([0.90, 0.00, 0.00], dtype=torch.float32),
            "portfolio_daily_source_positive_forward_penalty": torch.tensor([0.96, 0.02, 0.00], dtype=torch.float32),
            "portfolio_daily_source_opportunity_cost_penalty": torch.tensor([0.90, 0.04, 0.00], dtype=torch.float32),
            "portfolio_daily_source_forward_excess_5d": torch.tensor([0.13, -0.07, 0.00], dtype=torch.float32),
            "portfolio_daily_source_tail_false_sell_penalty": torch.tensor([0.84, 0.00, 0.00], dtype=torch.float32),
            "portfolio_daily_source_release_preference": torch.tensor([0.02, 0.84, 0.00], dtype=torch.float32),
        }
        good_outputs = {
            "portfolio_daily_unified_source_score": torch.tensor([0.02, 0.78, 0.06], dtype=torch.float32),
            "portfolio_daily_source_hard_negative_penalty": torch.tensor([0.92, 0.04, 0.00], dtype=torch.float32),
            "portfolio_daily_source_release_preference": torch.tensor([0.04, 0.80, 0.00], dtype=torch.float32),
        }
        bad_outputs = {
            "portfolio_daily_unified_source_score": torch.tensor([0.86, 0.12, 0.06], dtype=torch.float32),
            "portfolio_daily_source_hard_negative_penalty": torch.tensor([0.18, 0.20, 0.00], dtype=torch.float32),
            "portfolio_daily_source_release_preference": torch.tensor([0.82, 0.10, 0.00], dtype=torch.float32),
        }

        good_loss = _source_hard_negative_tail_loss(good_outputs, targets)
        bad_loss = _source_hard_negative_tail_loss(bad_outputs, targets)

        self.assertLess(float(good_loss.detach().cpu()), float(bad_loss.detach().cpu()) * 0.45)

    def test_source_listwise_release_regret_prefers_clean_source_within_day(self) -> None:
        targets = {
            "date_code": torch.tensor([1.0, 1.0, 1.0, 2.0], dtype=torch.float32),
            "portfolio_daily_source_candidate_mask": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
            "portfolio_daily_source_release_preference": torch.tensor([0.02, 0.86, 0.72, 0.55], dtype=torch.float32),
        }
        good_outputs = {
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.82, 0.70, 0.50], dtype=torch.float32),
        }
        bad_outputs = {
            "portfolio_daily_unified_source_score": torch.tensor([0.90, 0.12, 0.18, 0.50], dtype=torch.float32),
        }

        good_loss = _source_listwise_release_regret_loss(good_outputs, targets)
        bad_loss = _source_listwise_release_regret_loss(bad_outputs, targets)

        self.assertLess(float(good_loss.detach().cpu()), float(bad_loss.detach().cpu()) * 0.35)

    def test_transfer_level_allocation_regret_penalizes_negative_spread_and_dead_cash(self) -> None:
        targets = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.86, 0.08, 0.18], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.00, 0.72], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.tensor([0.12, 0.18, 0.14], dtype=torch.float32),
            "portfolio_daily_unified_allocation_objective": torch.tensor([0.82, 0.02, 0.70], dtype=torch.float32),
            "portfolio_daily_transfer_regret_target": torch.tensor([0.84, 0.00, 0.76], dtype=torch.float32),
            "portfolio_daily_source_release_preference": torch.tensor([0.00, 0.02, 0.84], dtype=torch.float32),
            "portfolio_daily_source_hard_negative_penalty": torch.tensor([0.00, 0.94, 0.02], dtype=torch.float32),
            "portfolio_daily_source_tail_false_sell_penalty": torch.tensor([0.00, 0.86, 0.00], dtype=torch.float32),
            "portfolio_daily_source_positive_forward_penalty": torch.tensor([0.00, 0.96, 0.02], dtype=torch.float32),
            "portfolio_daily_source_opportunity_cost_penalty": torch.tensor([0.00, 0.90, 0.04], dtype=torch.float32),
            "portfolio_daily_receiver_source_spread_reward": torch.tensor([0.74, 0.02, 0.68], dtype=torch.float32),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_source_candidate_mask": torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32),
            "market_downside_pressure": torch.tensor([0.02, 0.02, 0.04], dtype=torch.float32),
            "cash_regime_pressure": torch.tensor([0.02, 0.02, 0.04], dtype=torch.float32),
        }
        good_outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.82, 0.04, 0.18], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.02, 0.76], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.tensor([0.10, 0.14, 0.12], dtype=torch.float32),
            "portfolio_daily_unified_allocation_objective": torch.tensor([0.82, 0.04, 0.72], dtype=torch.float32),
            "portfolio_daily_transfer_regret_target": torch.tensor([0.82, 0.02, 0.72], dtype=torch.float32),
        }
        bad_outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.12, 0.18, 0.20], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.08, 0.84, 0.16], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.tensor([0.78, 0.72, 0.70], dtype=torch.float32),
            "portfolio_daily_unified_allocation_objective": torch.tensor([0.14, 0.82, 0.18], dtype=torch.float32),
            "portfolio_daily_transfer_regret_target": torch.tensor([0.12, 0.82, 0.20], dtype=torch.float32),
        }

        good_loss = _transfer_level_allocation_regret_loss(good_outputs, targets)
        bad_loss = _transfer_level_allocation_regret_loss(bad_outputs, targets)

        self.assertLess(float(good_loss.detach().cpu()), float(bad_loss.detach().cpu()) * 0.45)

    def test_semidifferentiable_allocation_solver_respects_cash_turnover_and_position_cap(self) -> None:
        problem = build_unified_allocation_problem(
            pd.DataFrame(
                [
                    {
                        "stock": "bad_source",
                        "current_weight": 0.18,
                        "portfolio_daily_source_score": 0.90,
                        "portfolio_daily_source_candidate_mask": 1.0,
                        "portfolio_daily_source_executability": 1.0,
                        "portfolio_daily_source_forward_excess_5d": 0.13,
                        "portfolio_daily_source_receiver_forward_spread": -0.09,
                        "portfolio_daily_source_opportunity_cost": 0.82,
                    },
                    {
                        "stock": "clean_source",
                        "current_weight": 0.14,
                        "portfolio_daily_source_score": 0.62,
                        "portfolio_daily_source_candidate_mask": 1.0,
                        "portfolio_daily_source_executability": 1.0,
                        "portfolio_daily_source_forward_excess_5d": -0.05,
                        "portfolio_daily_source_receiver_forward_spread": 0.08,
                        "portfolio_daily_source_opportunity_cost": 0.04,
                    },
                    {
                        "stock": "new_receiver",
                        "current_weight": 0.00,
                        "portfolio_daily_receiver_score": 0.86,
                        "portfolio_daily_receiver_candidate_mask": 1.0,
                        "portfolio_daily_receiver_executability": 1.0,
                        "portfolio_daily_receiver_add_headroom": 0.18,
                    },
                    {
                        "stock": "held_receiver",
                        "current_weight": 0.06,
                        "portfolio_daily_receiver_score": 0.70,
                        "portfolio_daily_receiver_candidate_mask": 1.0,
                        "portfolio_daily_receiver_executability": 1.0,
                        "portfolio_daily_receiver_add_headroom": 0.10,
                    },
                ]
            )
        )
        constraints = AllocationOptimizerConstraints(
            cash_reserve_target=0.48,
            turnover_limit=0.18,
            max_position_weight=0.20,
            transaction_cost_bps=3.0,
            slippage_bps=7.0,
            sell_tax_bps=10.0,
        )

        solution = solve_semidifferentiable_allocation(problem, constraints=constraints)
        target = solution.target_weight

        self.assertLessEqual(float(target.max()), constraints.max_position_weight + 1.0e-9)
        self.assertGreaterEqual(solution.cash_after, constraints.cash_reserve_target - 1.0e-9)
        self.assertLessEqual(solution.expected_turnover, constraints.turnover_limit + 1.0e-9)
        self.assertLessEqual(solution.buy_turnover, solution.sell_turnover + solution.available_cash_to_deploy + 1.0e-9)
        self.assertLess(target.loc["clean_source"], float(problem.set_index("stock").loc["clean_source", "current_weight"]))
        self.assertGreaterEqual(target.loc["bad_source"], target.loc["clean_source"])
        self.assertTrue(solution.diagnostics["constraint_violations"] == 0.0)

    def test_r35_unified_allocation_profile_and_loss_targets_are_registered(self) -> None:
        tag = "split_heads_portfolio_daily_unified_allocation_r35"
        self.assertIn(tag, SEARCH_PROFILES)
        self.assertIn(tag, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[tag], "portfolio_daily_ranking_v2_gated")
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[tag]["loss_profile"], "alpha_result_value_budget_split_v21")

        weights = LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v21"]["sample_scalar_loss_weights"]
        for column in (
            "portfolio_daily_unified_receiver_score",
            "portfolio_daily_unified_source_score",
            "portfolio_daily_unified_cash_score",
            "portfolio_daily_source_positive_forward_penalty",
            "portfolio_daily_source_opportunity_cost_penalty",
            "portfolio_daily_receiver_source_spread_reward",
            "portfolio_daily_unified_allocation_objective",
        ):
            self.assertIn(column, weights)

    def test_r36_risk_aware_unified_allocation_profile_is_registered(self) -> None:
        tag = "split_heads_portfolio_daily_risk_aware_unified_allocation_r36"
        self.assertIn(tag, SEARCH_PROFILES)
        self.assertIn(tag, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[tag], "portfolio_daily_ranking_v2_gated")
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[tag]["loss_profile"], "alpha_result_value_budget_split_v22")

        weights = LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v22"]
        self.assertGreater(weights["sample_scalar_loss_weights"]["portfolio_daily_unified_cash_score"], 1.60)
        self.assertGreater(weights["multi_objective_loss_weights"]["portfolio_unified_allocation_total"], 0.0)

    def test_r37_decision_focused_allocation_profile_is_registered(self) -> None:
        tag = "split_heads_portfolio_daily_decision_focused_allocation_r37"
        self.assertIn(tag, SEARCH_PROFILES)
        self.assertIn(tag, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[tag], "portfolio_daily_ranking_v2_gated")
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[tag]["loss_profile"], "alpha_result_value_budget_split_v23")

        weights = LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v23"]
        self.assertGreater(weights["multi_objective_loss_weights"]["portfolio_decision_regret_total"], 0.0)
        self.assertGreater(
            weights["multi_objective_loss_weights"]["portfolio_decision_regret_total"],
            weights["multi_objective_loss_weights"]["portfolio_unified_allocation_total"] * 0.40,
        )

    def test_r38_source_hard_negative_regret_profile_is_registered(self) -> None:
        tag = "split_heads_portfolio_daily_source_hard_negative_regret_r38"
        self.assertIn(tag, SEARCH_PROFILES)
        self.assertIn(tag, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[tag], "portfolio_daily_ranking_v2_gated")
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[tag]["loss_profile"], "alpha_result_value_budget_split_v24")

        weights = LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v24"]
        for column in (
            "portfolio_daily_source_hard_negative_penalty",
            "portfolio_daily_source_tail_false_sell_penalty",
            "portfolio_daily_source_release_preference",
            "portfolio_daily_transfer_regret_target",
        ):
            self.assertIn(column, weights["sample_scalar_loss_weights"])
        multi = weights["multi_objective_loss_weights"]
        self.assertGreater(multi["portfolio_source_hard_negative_tail_total"], 0.0)
        self.assertGreater(multi["portfolio_source_listwise_release_total"], 0.0)
        self.assertGreater(multi["portfolio_transfer_regret_total"], 0.0)
        self.assertGreater(
            multi["portfolio_source_hard_negative_tail_total"],
            multi["portfolio_unified_allocation_total"] * 0.40,
        )

    def test_exposure_utilization_uses_cash_weight_when_gross_exposure_is_empty(self) -> None:
        turnover_frame = pd.DataFrame(
            {
                "cash_weight": [0.20, 0.30],
                "gross_exposure": [0.0, 0.0],
                "gross_exposure_target": [0.80, 0.90],
            }
        )

        avg_target, utilization = _compute_exposure_utilization_from_turnover(turnover_frame)

        self.assertAlmostEqual(avg_target, 0.85, places=6)
        self.assertAlmostEqual(utilization, 0.75 / 0.85, places=6)

    def test_predict_policy_exports_unified_allocation_heads_to_policy_frame(self) -> None:
        state_frame = pd.DataFrame(
            {
                "stock": ["new_a", "new_b"],
                "current_weight": [0.0, 0.0],
                "static_feature": [0.0, 0.0],
                "seq_feature": [0.0, 0.0],
            }
        )
        artifact = SimpleNamespace(
            sample_model=_FakePolicyModel(),
            daily_model=_FakeDailyModel(),
            static_feature_names=["static_feature"],
            sequence_base_names=["seq_feature"],
            sequence_steps=[0],
            sequence_columns=["seq_feature"],
            daily_feature_names=["daily_feature"],
            static_fill_values=pd.Series([0.0]).to_numpy(dtype=float),
            static_means=pd.Series([0.0]).to_numpy(dtype=float),
            static_stds=pd.Series([1.0]).to_numpy(dtype=float),
            sequence_fill_values=pd.Series([0.0]).to_numpy(dtype=float),
            sequence_means=pd.Series([0.0]).to_numpy(dtype=float),
            sequence_stds=pd.Series([1.0]).to_numpy(dtype=float),
            daily_fill_values=pd.Series([0.0]).to_numpy(dtype=float),
            daily_means=pd.Series([0.0]).to_numpy(dtype=float),
            daily_stds=pd.Series([1.0]).to_numpy(dtype=float),
            train_summary={},
            training_diagnostics={
                "supports_sell_heads": False,
                "supports_portfolio_listwise_heads": True,
                "supports_portfolio_unified_allocation_heads": True,
                "supports_portfolio_allocation_objective_consolidation_heads": True,
            },
        )

        policy, _ = predict_policy_v3(artifact, state_frame=state_frame, daily_features={"daily_feature": 0.0})

        for column in (
            "portfolio_daily_unified_receiver_score",
            "portfolio_daily_unified_source_score",
            "portfolio_daily_unified_cash_score",
            "portfolio_daily_source_positive_forward_penalty",
            "portfolio_daily_source_opportunity_cost_penalty",
            "portfolio_daily_receiver_source_spread_reward",
            "portfolio_daily_unified_allocation_objective",
            "portfolio_daily_allocation_trade_quality_target",
            "portfolio_daily_allocation_cash_deployment_target",
            "portfolio_daily_allocation_risk_adjusted_return_target",
            "portfolio_daily_allocation_drawdown_control_target",
            "portfolio_daily_allocation_monthly_quality_target",
            "portfolio_daily_allocation_final_objective",
        ):
            self.assertIn(column, policy.columns)
            if column == "portfolio_daily_unified_cash_score":
                self.assertGreaterEqual(float(policy[column].max()), 0.0)
            else:
                self.assertGreater(float(policy[column].max()), 0.0)
        self.assertLess(float(policy["portfolio_daily_cash_score"].max()), 0.30)

    def test_v2_scoring_rewards_unified_allocation_surface_quality(self) -> None:
        weak = _score_protocol_summary(
            _protocol_summary(
                semantic={
                    "portfolio_daily_receiver_minus_source_forward_excess_5d": -0.06,
                    "portfolio_daily_source_positive_forward_sell_share": 0.72,
                }
            )
            | {
                "train": {
                    "teacher_summary": {
                        "unified_allocation_summary_mean": {
                            "portfolio_daily_unified_allocation_objective": 0.18,
                            "portfolio_daily_source_positive_forward_penalty": 0.68,
                            "portfolio_daily_source_opportunity_cost_penalty": 0.62,
                            "portfolio_daily_receiver_source_spread_reward": 0.10,
                        }
                    }
                }
            },
            objective_profile="portfolio_daily_ranking_v2_gated",
        )
        strong = _score_protocol_summary(
            _protocol_summary(
                semantic={
                    "portfolio_daily_receiver_minus_source_forward_excess_5d": 0.04,
                    "portfolio_daily_source_positive_forward_sell_share": 0.10,
                }
            )
            | {
                "train": {
                    "teacher_summary": {
                        "unified_allocation_summary_mean": {
                            "portfolio_daily_unified_allocation_objective": 0.74,
                            "portfolio_daily_source_positive_forward_penalty": 0.08,
                            "portfolio_daily_source_opportunity_cost_penalty": 0.12,
                            "portfolio_daily_receiver_source_spread_reward": 0.62,
                        }
                    }
                }
            },
            objective_profile="portfolio_daily_ranking_v2_gated",
        )

        self.assertIn("portfolio_daily_unified_allocation_objective", strong["score_breakdown"]["performance"])
        self.assertIn("portfolio_daily_source_positive_forward_penalty", strong["score_breakdown"]["performance"])
        self.assertGreater(strong["composite_score"], weak["composite_score"])

    def test_unified_receiver_head_can_open_without_direct_action_label(self) -> None:
        prices = pd.Series({"new_a": 10.0, "new_b": 20.0, "weak": 30.0})
        policy = _base_policy(list(prices.index))
        policy.loc["new_a", [
            "portfolio_daily_unified_receiver_score",
            "portfolio_daily_receiver_score",
            "portfolio_daily_receiver_executability",
            "deploy_value_target",
            "deploy_gate_target",
            "alpha_opportunity_value",
            "deployment_opportunity_cost",
            "portfolio_daily_unified_allocation_objective",
        ]] = [0.92, 0.92, 0.90, 0.90, 0.90, 0.80, 0.80, 0.72]
        policy.loc["new_b", [
            "portfolio_daily_unified_receiver_score",
            "portfolio_daily_receiver_score",
            "portfolio_daily_receiver_executability",
            "deploy_value_target",
            "deploy_gate_target",
            "alpha_opportunity_value",
            "deployment_opportunity_cost",
        ]] = [0.84, 0.84, 0.82, 0.82, 0.82, 0.70, 0.70]
        state = PortfolioState(cash_weight=1.0, max_positions=4, max_position_weight=0.20, turnover_limit=0.60)

        result = state.step(
            date="2026-01-02",
            prices=prices,
            policy_frame=policy,
            budget_semantics=BUDGET_SEMANTICS_SPLIT,
            budget_calibration=BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC,
        )

        self.assertGreater(result.diagnostics["portfolio_daily_unified_receiver_candidate_count"], 0)
        self.assertGreater(result.diagnostics["portfolio_daily_unified_allocation_objective_mean"], 0.0)
        self.assertIn("portfolio_daily_unified_allocation_objective", result.actions[0])
        self.assertGreater(result.diagnostics["portfolio_daily_receiver_target_count"], 0)
        self.assertGreater(result.diagnostics["direct_action_open_signal_count"], 0)
        self.assertGreater(float(result.weights.sum()), 0.0)

    def test_unified_source_head_can_release_clean_held_without_reduce_label(self) -> None:
        prices = pd.Series({"source": 10.0, "receiver": 20.0, "weak": 30.0})
        policy = _base_policy(list(prices.index))
        policy.loc["receiver", [
            "portfolio_daily_unified_receiver_score",
            "portfolio_daily_receiver_score",
            "portfolio_daily_receiver_executability",
            "deploy_value_target",
            "deploy_gate_target",
            "alpha_opportunity_value",
            "deployment_opportunity_cost",
        ]] = [0.90, 0.90, 0.88, 0.88, 0.88, 0.78, 0.78]
        policy.loc["source", [
            "portfolio_daily_unified_source_score",
            "portfolio_daily_source_score",
            "portfolio_daily_source_release_quality",
            "portfolio_daily_source_executability",
            "portfolio_daily_source_economic_release_score",
            "portfolio_daily_source_opportunity_cost",
            "portfolio_daily_receiver_source_spread_reward",
        ]] = [0.76, 0.70, 0.42, 0.55, 0.36, 0.08, 0.30]
        state = PortfolioState(
            cash_weight=0.80,
            max_positions=4,
            max_position_weight=0.20,
            turnover_limit=0.60,
            holdings={"source": HoldingState(weight=0.20, entry_price=10.0, peak_price=10.0, hold_days=10)},
        )

        result = state.step(
            date="2026-01-02",
            prices=prices,
            policy_frame=policy,
            budget_semantics=BUDGET_SEMANTICS_SPLIT,
            budget_calibration=BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC,
        )

        self.assertGreater(result.diagnostics["portfolio_daily_unified_receiver_candidate_count"], 0)
        self.assertGreater(result.diagnostics["portfolio_daily_unified_source_candidate_count"], 0)
        self.assertGreater(result.diagnostics["portfolio_daily_source_target_count"], 0)
        self.assertLess(float(result.weights["source"]), 0.20)

    def test_unified_source_head_cannot_bypass_source_distribution_gate(self) -> None:
        prices = pd.Series({"source": 10.0, "receiver": 20.0, "weak": 30.0})
        policy = _base_policy(list(prices.index))
        policy.loc["receiver", [
            "portfolio_daily_unified_receiver_score",
            "portfolio_daily_receiver_score",
            "portfolio_daily_receiver_executability",
            "deploy_value_target",
            "deploy_gate_target",
            "alpha_opportunity_value",
            "deployment_opportunity_cost",
        ]] = [0.92, 0.92, 0.90, 0.90, 0.90, 0.80, 0.80]
        policy.loc["source", [
            "portfolio_daily_unified_source_score",
            "portfolio_daily_source_score",
            "portfolio_daily_source_release_quality",
            "portfolio_daily_source_executability",
            "portfolio_daily_source_economic_release_score",
            "portfolio_daily_source_opportunity_cost",
            "portfolio_daily_receiver_source_spread_reward",
            "portfolio_daily_source_positive_forward_penalty",
            "portfolio_daily_source_opportunity_cost_penalty",
            "portfolio_daily_source_forward_strength_brake_risk",
            "portfolio_daily_source_forward_proxy_keep_risk",
            "portfolio_daily_source_bad_forward_spread_risk",
            "portfolio_daily_source_economic_block_risk",
        ]] = [0.88, 0.90, 0.48, 0.80, 0.50, 0.60, 0.82, 0.34, 0.50, 0.50, 0.54, 0.60, 0.76]
        state = PortfolioState(
            cash_weight=0.80,
            max_positions=4,
            max_position_weight=0.20,
            turnover_limit=0.60,
            holdings={"source": HoldingState(weight=0.20, entry_price=10.0, peak_price=10.0, hold_days=10)},
        )

        result = state.step(
            date="2026-01-02",
            prices=prices,
            policy_frame=policy,
            budget_semantics=BUDGET_SEMANTICS_SPLIT,
            budget_calibration=BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC,
        )

        self.assertEqual(result.diagnostics["portfolio_daily_unified_source_candidate_count"], 0)
        self.assertEqual(result.diagnostics["portfolio_daily_source_target_count"], 0)
        self.assertEqual(float(result.weights["source"]), 0.20)

    def test_r39_allocation_objective_consolidation_adds_economic_targets(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2026-01-02", "2026-01-02"],
                "stock": ["receiver", "source"],
                "action_label": ["open", "reduce"],
                "current_weight": [0.0, 0.20],
                "portfolio_daily_receiver_score": [0.88, 0.10],
                "portfolio_daily_source_score": [0.05, 0.74],
                "portfolio_daily_cash_score": [0.72, 0.72],
                "portfolio_daily_receiver_candidate_mask": [1.0, 0.0],
                "portfolio_daily_source_candidate_mask": [0.0, 1.0],
                "portfolio_daily_receiver_executability": [0.92, 0.40],
                "portfolio_daily_receiver_add_headroom": [0.18, 0.04],
                "portfolio_daily_source_executability": [0.25, 0.86],
                "portfolio_daily_source_release_capacity": [0.0, 0.20],
                "portfolio_daily_source_release_quality": [0.0, 0.76],
                "portfolio_daily_source_economic_release_score": [0.0, 0.72],
                "portfolio_daily_source_opportunity_cost": [0.02, 0.04],
                "portfolio_daily_source_forward_excess_5d": [0.0, -0.030],
                "portfolio_daily_source_receiver_forward_spread": [0.060, 0.060],
                "portfolio_daily_receiver_forward_excess_5d": [0.040, 0.040],
                "portfolio_daily_allocation_transfer_score": [0.78, 0.78],
                "portfolio_daily_allocation_dead_branch_risk": [0.05, 0.05],
                "market_downside_pressure": [0.05, 0.05],
                "cash_regime_pressure": [0.04, 0.04],
                "portfolio_cash_pressure": [0.82, 0.82],
                "multi_horizon_forward_risk": [0.05, 0.05],
                "cash_defense_value": [0.05, 0.05],
                "portfolio_drawdown_20d": [-0.010, -0.010],
                "forward_benchmark_return_1d": [0.012, 0.012],
                "forward_benchmark_return_3d": [0.026, 0.026],
            }
        )

        result = build_unified_allocation_problem(frame)

        required = {
            "portfolio_daily_allocation_trade_quality_target",
            "portfolio_daily_allocation_cash_deployment_target",
            "portfolio_daily_allocation_risk_adjusted_return_target",
            "portfolio_daily_allocation_drawdown_control_target",
            "portfolio_daily_allocation_monthly_quality_target",
            "portfolio_daily_allocation_final_objective",
        }
        self.assertTrue(required.issubset(result.columns))
        self.assertGreater(float(result["portfolio_daily_allocation_cash_deployment_target"].mean()), 0.35)
        self.assertLess(float(result["portfolio_daily_unified_cash_score"].mean()), 0.70)
        self.assertGreater(float(result["portfolio_daily_allocation_final_objective"].mean()), 0.40)

    def test_r39_loss_profile_makes_allocation_objective_primary_not_action_head(self) -> None:
        profile_name = "alpha_result_value_budget_split_v25"

        self.assertIn(profile_name, LOSS_PROFILE_CONFIGS)
        config = LOSS_PROFILE_CONFIGS[profile_name]
        sample_weights = config["sample_scalar_loss_weights"]
        multi_weights = config["multi_objective_loss_weights"]

        for name in (
            "portfolio_daily_allocation_trade_quality_target",
            "portfolio_daily_allocation_cash_deployment_target",
            "portfolio_daily_allocation_risk_adjusted_return_target",
            "portfolio_daily_allocation_drawdown_control_target",
            "portfolio_daily_allocation_monthly_quality_target",
            "portfolio_daily_allocation_final_objective",
        ):
            self.assertIn(name, sample_weights)
            self.assertGreater(sample_weights[name], 1.0)
        self.assertLessEqual(multi_weights["action_total"], 0.08)
        self.assertGreater(multi_weights["portfolio_allocation_objective_consolidation_total"], 0.70)
        self.assertGreater(
            multi_weights["portfolio_allocation_objective_consolidation_total"],
            multi_weights["action_total"] * 8.0,
        )

    def test_r39_allocation_consolidation_loss_penalizes_dead_cash_and_bad_source(self) -> None:
        targets = {
            "portfolio_daily_allocation_trade_quality_target": torch.tensor([0.78, 0.20]),
            "portfolio_daily_allocation_cash_deployment_target": torch.tensor([0.82, 0.12]),
            "portfolio_daily_allocation_risk_adjusted_return_target": torch.tensor([0.76, 0.18]),
            "portfolio_daily_allocation_drawdown_control_target": torch.tensor([0.08, 0.72]),
            "portfolio_daily_allocation_monthly_quality_target": torch.tensor([0.74, 0.20]),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.80, 0.16]),
            "portfolio_daily_unified_receiver_score": torch.tensor([0.82, 0.12]),
            "portfolio_daily_unified_source_score": torch.tensor([0.74, 0.10]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.16, 0.70]),
            "portfolio_daily_source_positive_forward_penalty": torch.tensor([0.04, 0.84]),
            "portfolio_daily_source_opportunity_cost_penalty": torch.tensor([0.05, 0.76]),
            "portfolio_daily_source_hard_negative_penalty": torch.tensor([0.02, 0.80]),
            "portfolio_daily_receiver_source_spread_reward": torch.tensor([0.80, 0.04]),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([1.0, 0.0]),
            "portfolio_daily_source_candidate_mask": torch.tensor([1.0, 1.0]),
        }
        good_outputs = {
            "portfolio_daily_allocation_trade_quality_target": torch.tensor([0.74, 0.16]),
            "portfolio_daily_allocation_cash_deployment_target": torch.tensor([0.78, 0.16]),
            "portfolio_daily_allocation_risk_adjusted_return_target": torch.tensor([0.72, 0.16]),
            "portfolio_daily_allocation_drawdown_control_target": torch.tensor([0.10, 0.68]),
            "portfolio_daily_allocation_monthly_quality_target": torch.tensor([0.70, 0.18]),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.76, 0.14]),
            "portfolio_daily_unified_receiver_score": torch.tensor([0.78, 0.10]),
            "portfolio_daily_unified_source_score": torch.tensor([0.70, 0.06]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.12, 0.72]),
        }
        bad_outputs = {
            **good_outputs,
            "portfolio_daily_unified_receiver_score": torch.tensor([0.18, 0.10]),
            "portfolio_daily_unified_source_score": torch.tensor([0.20, 0.72]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.88, 0.30]),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.20, 0.62]),
        }

        good_loss = _allocation_objective_consolidation_loss(good_outputs, targets)
        bad_loss = _allocation_objective_consolidation_loss(bad_outputs, targets)

        self.assertGreater(float(bad_loss), float(good_loss) + 0.15)

    def test_r39_search_profile_is_architecture_consolidation(self) -> None:
        profile = "split_heads_portfolio_daily_allocation_objective_consolidation_r39"

        self.assertIn(profile, SEARCH_PROFILES)
        self.assertIn(profile, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["loss_profile"], "alpha_result_value_budget_split_v25")
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[profile], "portfolio_daily_ranking_v2_gated")
        self.assertEqual(
            SEARCH_PROFILE_BASE_TRIALS[profile]["budget_calibration"],
            BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC,
        )
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["budget_semantics"], BUDGET_SEMANTICS_SPLIT)

    def test_r40_end_to_end_allocation_layer_profile_exits_action_budget_path(self) -> None:
        profile = "split_heads_portfolio_daily_end_to_end_allocation_layer_r40"

        self.assertIn(profile, SEARCH_PROFILES)
        self.assertIn(profile, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["loss_profile"], "alpha_result_value_budget_split_v25")
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[profile], "end_to_end_allocation_layer_v1")
        self.assertEqual(
            SEARCH_PROFILE_BASE_TRIALS[profile]["budget_calibration"],
            BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        )
        self.assertEqual(
            SEARCH_PROFILE_BASE_TRIALS[profile]["budget_semantics"],
            BUDGET_SEMANTICS_ALLOCATION_LAYER,
        )
        self.assertNotEqual(
            SEARCH_PROFILE_BASE_TRIALS[profile]["budget_calibration"],
            BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC,
        )
        self.assertNotEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["budget_semantics"], BUDGET_SEMANTICS_SPLIT)

    def test_r41_unified_allocation_problem_exposes_uncertainty_aware_decision_objective(self) -> None:
        low_risk_frame = pd.DataFrame(
            {
                "stock": ["receiver", "source"],
                "action_label": ["open", "reduce"],
                "current_weight": [0.0, 0.20],
                "portfolio_daily_receiver_score": [0.90, 0.05],
                "portfolio_daily_source_score": [0.05, 0.78],
                "portfolio_daily_cash_score": [0.20, 0.20],
                "portfolio_daily_receiver_candidate_mask": [1.0, 0.0],
                "portfolio_daily_source_candidate_mask": [0.0, 1.0],
                "portfolio_daily_receiver_executability": [0.95, 0.20],
                "portfolio_daily_receiver_add_headroom": [0.18, 0.0],
                "portfolio_daily_source_executability": [0.0, 0.92],
                "portfolio_daily_source_release_capacity": [0.0, 0.20],
                "portfolio_daily_source_release_quality": [0.0, 0.84],
                "portfolio_daily_source_economic_release_score": [0.0, 0.80],
                "portfolio_daily_source_opportunity_cost": [0.0, 0.03],
                "portfolio_daily_source_forward_excess_5d": [0.0, -0.035],
                "portfolio_daily_source_receiver_forward_spread": [0.070, 0.070],
                "portfolio_daily_receiver_forward_excess_5d": [0.045, 0.045],
                "portfolio_daily_allocation_transfer_score": [0.84, 0.84],
                "portfolio_daily_allocation_dead_branch_risk": [0.02, 0.02],
                "market_downside_pressure": [0.03, 0.03],
                "cash_regime_pressure": [0.02, 0.02],
                "portfolio_cash_pressure": [0.70, 0.70],
                "multi_horizon_forward_risk": [0.04, 0.04],
                "cash_defense_value": [0.03, 0.03],
                "portfolio_drawdown_20d": [-0.005, -0.005],
                "forward_benchmark_return_1d": [0.014, 0.014],
                "forward_benchmark_return_3d": [0.030, 0.030],
            }
        )
        high_risk_frame = low_risk_frame.assign(
            market_downside_pressure=[0.92, 0.92],
            cash_regime_pressure=[0.88, 0.88],
            multi_horizon_forward_risk=[0.90, 0.90],
            cash_defense_value=[0.86, 0.86],
            portfolio_drawdown_20d=[-0.18, -0.18],
            forward_benchmark_return_1d=[-0.045, -0.045],
            forward_benchmark_return_3d=[-0.080, -0.080],
        )

        low_risk = build_unified_allocation_problem(low_risk_frame)
        high_risk = build_unified_allocation_problem(high_risk_frame)

        required = {
            "portfolio_daily_allocation_uncertainty_pressure_target",
            "portfolio_daily_allocation_tail_risk_control_target",
            "portfolio_daily_allocation_decision_focused_objective",
        }
        self.assertTrue(required.issubset(low_risk.columns))
        self.assertGreater(
            float(high_risk["portfolio_daily_allocation_uncertainty_pressure_target"].mean()),
            float(low_risk["portfolio_daily_allocation_uncertainty_pressure_target"].mean()) + 0.35,
        )
        self.assertGreater(
            float(high_risk["portfolio_daily_allocation_tail_risk_control_target"].mean()),
            float(low_risk["portfolio_daily_allocation_tail_risk_control_target"].mean()) + 0.35,
        )
        self.assertGreater(
            float(low_risk["portfolio_daily_allocation_decision_focused_objective"].mean()),
            float(high_risk["portfolio_daily_allocation_decision_focused_objective"].mean()) + 0.20,
        )

    def test_r41_solver_brakes_receiver_deploy_under_tail_risk_without_killing_clean_source_release(self) -> None:
        frame = pd.DataFrame(
            {
                "stock": ["RECV", "SRC"],
                "action_label": ["open", "reduce"],
                "current_weight": [0.0, 0.20],
                "portfolio_daily_receiver_score": [0.92, 0.0],
                "portfolio_daily_source_score": [0.0, 0.86],
                "portfolio_daily_cash_score": [0.12, 0.12],
                "portfolio_daily_receiver_candidate_mask": [1.0, 0.0],
                "portfolio_daily_source_candidate_mask": [0.0, 1.0],
                "portfolio_daily_receiver_executability": [0.96, 0.0],
                "portfolio_daily_receiver_add_headroom": [0.20, 0.0],
                "portfolio_daily_source_executability": [0.0, 0.94],
                "portfolio_daily_source_release_capacity": [0.0, 0.20],
                "portfolio_daily_source_release_quality": [0.0, 0.86],
                "portfolio_daily_source_economic_release_score": [0.0, 0.82],
                "portfolio_daily_source_opportunity_cost": [0.0, 0.02],
                "portfolio_daily_source_forward_excess_5d": [0.0, -0.040],
                "portfolio_daily_source_receiver_forward_spread": [0.080, 0.080],
                "portfolio_daily_receiver_forward_excess_5d": [0.050, 0.050],
                "portfolio_daily_allocation_transfer_score": [0.88, 0.88],
                "portfolio_daily_allocation_dead_branch_risk": [0.02, 0.02],
                "market_downside_pressure": [0.03, 0.03],
                "cash_regime_pressure": [0.02, 0.02],
                "portfolio_cash_pressure": [0.65, 0.65],
                "multi_horizon_forward_risk": [0.03, 0.03],
                "cash_defense_value": [0.02, 0.02],
                "portfolio_drawdown_20d": [-0.005, -0.005],
                "forward_benchmark_return_1d": [0.016, 0.016],
                "forward_benchmark_return_3d": [0.034, 0.034],
            }
        )
        stressed = frame.assign(
            market_downside_pressure=[0.94, 0.94],
            cash_regime_pressure=[0.90, 0.90],
            multi_horizon_forward_risk=[0.92, 0.92],
            cash_defense_value=[0.88, 0.88],
            portfolio_drawdown_20d=[-0.20, -0.20],
            forward_benchmark_return_1d=[-0.050, -0.050],
            forward_benchmark_return_3d=[-0.090, -0.090],
        )

        constraints = AllocationOptimizerConstraints(
            cash_reserve_target=0.05,
            turnover_limit=0.40,
            max_position_weight=0.25,
            min_trade_weight=0.0,
        )
        normal_solution = solve_semidifferentiable_allocation(frame, constraints=constraints)
        stressed_solution = solve_semidifferentiable_allocation(stressed, constraints=constraints)

        self.assertLess(normal_solution.target_weight["SRC"], 0.20)
        self.assertGreater(normal_solution.target_weight["RECV"], 0.05)
        self.assertLess(stressed_solution.buy_turnover, normal_solution.buy_turnover * 0.55)
        self.assertGreaterEqual(stressed_solution.cash_after, normal_solution.cash_after + 0.05)

    def test_r41_loss_profile_is_risk_sensitive_allocation_layer_not_r40_reuse(self) -> None:
        profile = "split_heads_portfolio_daily_risk_sensitive_allocation_layer_r41"
        loss_profile = "alpha_result_value_budget_split_v26"

        self.assertIn(profile, SEARCH_PROFILES)
        self.assertIn(profile, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["loss_profile"], loss_profile)
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[profile], "end_to_end_allocation_layer_v1")
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["budget_calibration"], BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["budget_semantics"], BUDGET_SEMANTICS_ALLOCATION_LAYER)
        self.assertNotEqual(
            SEARCH_PROFILE_BASE_TRIALS[profile]["loss_profile"],
            SEARCH_PROFILE_BASE_TRIALS["split_heads_portfolio_daily_end_to_end_allocation_layer_r40"]["loss_profile"],
        )

        config = LOSS_PROFILE_CONFIGS[loss_profile]
        sample_weights = config["sample_scalar_loss_weights"]
        multi_weights = config["multi_objective_loss_weights"]
        for name in (
            "portfolio_daily_allocation_uncertainty_pressure_target",
            "portfolio_daily_allocation_tail_risk_control_target",
            "portfolio_daily_allocation_decision_focused_objective",
        ):
            self.assertIn(name, sample_weights)
            self.assertGreater(
                sample_weights[name],
                LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v25"]["sample_scalar_loss_weights"][
                    "portfolio_daily_allocation_final_objective"
                ]
                * 0.75,
            )
        self.assertLessEqual(multi_weights["action_total"], 0.05)
        self.assertGreater(multi_weights["portfolio_risk_sensitive_allocation_total"], 0.40)

    def test_r41_risk_sensitive_loss_penalizes_tail_deploy_and_weak_cash_defense(self) -> None:
        loss_fn = getattr(model_seq_v3, "_risk_sensitive_allocation_objective_loss", None)
        self.assertIsNotNone(loss_fn)
        targets = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.84, 0.12]),
            "portfolio_daily_unified_source_score": torch.tensor([0.72, 0.08]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.12, 0.82]),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.78, 0.10]),
            "portfolio_daily_allocation_decision_focused_objective": torch.tensor([0.82, 0.08]),
            "portfolio_daily_allocation_uncertainty_pressure_target": torch.tensor([0.08, 0.88]),
            "portfolio_daily_allocation_tail_risk_control_target": torch.tensor([0.06, 0.90]),
            "portfolio_daily_allocation_drawdown_control_target": torch.tensor([0.08, 0.84]),
            "portfolio_daily_allocation_cash_deployment_target": torch.tensor([0.78, 0.05]),
            "portfolio_daily_source_hard_negative_penalty": torch.tensor([0.04, 0.82]),
            "portfolio_daily_source_positive_forward_penalty": torch.tensor([0.04, 0.78]),
            "portfolio_daily_source_opportunity_cost_penalty": torch.tensor([0.05, 0.76]),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([1.0, 0.0]),
            "portfolio_daily_source_candidate_mask": torch.tensor([1.0, 1.0]),
        }
        good_outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.80, 0.06]),
            "portfolio_daily_unified_source_score": torch.tensor([0.68, 0.04]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.14, 0.86]),
            "portfolio_daily_allocation_decision_focused_objective": torch.tensor([0.78, 0.06]),
            "portfolio_daily_allocation_uncertainty_pressure_target": torch.tensor([0.10, 0.84]),
            "portfolio_daily_allocation_tail_risk_control_target": torch.tensor([0.08, 0.86]),
        }
        bad_outputs = {
            **good_outputs,
            "portfolio_daily_unified_receiver_score": torch.tensor([0.28, 0.70]),
            "portfolio_daily_unified_source_score": torch.tensor([0.22, 0.72]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.76, 0.16]),
            "portfolio_daily_allocation_decision_focused_objective": torch.tensor([0.22, 0.68]),
            "portfolio_daily_allocation_uncertainty_pressure_target": torch.tensor([0.16, 0.24]),
            "portfolio_daily_allocation_tail_risk_control_target": torch.tensor([0.12, 0.20]),
        }

        good_loss = loss_fn(good_outputs, targets)
        bad_loss = loss_fn(bad_outputs, targets)

        self.assertGreater(float(bad_loss), float(good_loss) + 0.20)

    def test_r42_allocation_problem_exposes_utility_credit_closure_targets(self) -> None:
        strong_frame = pd.DataFrame(
            {
                "stock": ["RECV", "SRC"],
                "action_label": ["open", "reduce"],
                "current_weight": [0.0, 0.20],
                "portfolio_daily_receiver_score": [0.92, 0.04],
                "portfolio_daily_source_score": [0.04, 0.88],
                "portfolio_daily_cash_score": [0.12, 0.12],
                "portfolio_daily_receiver_candidate_mask": [1.0, 0.0],
                "portfolio_daily_source_candidate_mask": [0.0, 1.0],
                "portfolio_daily_receiver_executability": [0.96, 0.0],
                "portfolio_daily_receiver_add_headroom": [0.20, 0.0],
                "portfolio_daily_source_executability": [0.0, 0.94],
                "portfolio_daily_source_release_capacity": [0.0, 0.20],
                "portfolio_daily_source_release_quality": [0.0, 0.90],
                "portfolio_daily_source_economic_release_score": [0.0, 0.88],
                "portfolio_daily_source_opportunity_cost": [0.0, 0.02],
                "portfolio_daily_source_forward_excess_5d": [0.0, -0.045],
                "portfolio_daily_source_receiver_forward_spread": [0.090, 0.090],
                "portfolio_daily_receiver_forward_excess_5d": [0.055, 0.055],
                "portfolio_daily_allocation_transfer_score": [0.90, 0.90],
                "portfolio_daily_allocation_dead_branch_risk": [0.02, 0.02],
                "portfolio_daily_receiver_realized_deploy_proxy": [0.90, 0.0],
                "portfolio_daily_source_realized_release_proxy": [0.0, 0.88],
                "market_downside_pressure": [0.04, 0.04],
                "cash_regime_pressure": [0.04, 0.04],
                "portfolio_cash_pressure": [0.72, 0.72],
                "multi_horizon_forward_risk": [0.05, 0.05],
                "cash_defense_value": [0.04, 0.04],
                "portfolio_drawdown_20d": [-0.006, -0.006],
                "forward_benchmark_return_1d": [0.018, 0.018],
                "forward_benchmark_return_3d": [0.038, 0.038],
            }
        )
        weak_frame = strong_frame.assign(
            portfolio_daily_source_score=[0.04, 0.16],
            portfolio_daily_source_release_quality=[0.0, 0.18],
            portfolio_daily_source_economic_release_score=[0.0, 0.14],
            portfolio_daily_source_opportunity_cost=[0.0, 0.72],
            portfolio_daily_source_forward_excess_5d=[0.0, 0.090],
            portfolio_daily_source_receiver_forward_spread=[-0.040, -0.040],
            portfolio_daily_allocation_transfer_score=[0.18, 0.18],
            portfolio_daily_allocation_dead_branch_risk=[0.78, 0.78],
            market_downside_pressure=[0.70, 0.70],
            cash_regime_pressure=[0.72, 0.72],
            multi_horizon_forward_risk=[0.74, 0.74],
            portfolio_drawdown_20d=[-0.16, -0.16],
            forward_benchmark_return_1d=[-0.030, -0.030],
            forward_benchmark_return_3d=[-0.060, -0.060],
        )

        strong = build_unified_allocation_problem(strong_frame)
        weak = build_unified_allocation_problem(weak_frame)

        required = {
            "portfolio_daily_allocation_net_utility_target",
            "portfolio_daily_allocation_credit_closure_target",
            "portfolio_daily_allocation_resource_efficiency_target",
        }
        self.assertTrue(required.issubset(strong.columns))
        self.assertGreater(
            float(strong["portfolio_daily_allocation_net_utility_target"].mean()),
            float(weak["portfolio_daily_allocation_net_utility_target"].mean()) + 0.35,
        )
        self.assertGreater(
            float(strong["portfolio_daily_allocation_credit_closure_target"].mean()),
            float(weak["portfolio_daily_allocation_credit_closure_target"].mean()) + 0.35,
        )
        self.assertGreater(
            float(strong["portfolio_daily_allocation_resource_efficiency_target"].mean()),
            float(weak["portfolio_daily_allocation_resource_efficiency_target"].mean()) + 0.30,
        )

    def test_r42_solver_uses_utility_to_deploy_when_credit_closure_is_strong(self) -> None:
        base = pd.DataFrame(
            {
                "stock": ["RECV", "SRC"],
                "action_label": ["open", "reduce"],
                "current_weight": [0.0, 0.20],
                "portfolio_daily_receiver_score": [0.92, 0.0],
                "portfolio_daily_source_score": [0.0, 0.88],
                "portfolio_daily_cash_score": [0.20, 0.20],
                "portfolio_daily_receiver_candidate_mask": [1.0, 0.0],
                "portfolio_daily_source_candidate_mask": [0.0, 1.0],
                "portfolio_daily_receiver_executability": [0.96, 0.0],
                "portfolio_daily_receiver_add_headroom": [0.20, 0.0],
                "portfolio_daily_source_executability": [0.0, 0.94],
                "portfolio_daily_source_release_capacity": [0.0, 0.20],
                "portfolio_daily_source_release_quality": [0.0, 0.90],
                "portfolio_daily_source_economic_release_score": [0.0, 0.88],
                "portfolio_daily_source_opportunity_cost": [0.0, 0.02],
                "portfolio_daily_source_forward_excess_5d": [0.0, -0.050],
                "portfolio_daily_source_receiver_forward_spread": [0.100, 0.100],
                "portfolio_daily_receiver_forward_excess_5d": [0.060, 0.060],
                "portfolio_daily_allocation_transfer_score": [0.92, 0.92],
                "portfolio_daily_allocation_dead_branch_risk": [0.02, 0.02],
                "market_downside_pressure": [0.42, 0.42],
                "cash_regime_pressure": [0.36, 0.36],
                "portfolio_cash_pressure": [0.70, 0.70],
                "multi_horizon_forward_risk": [0.40, 0.40],
                "cash_defense_value": [0.32, 0.32],
                "portfolio_drawdown_20d": [-0.055, -0.055],
                "forward_benchmark_return_1d": [0.006, 0.006],
                "forward_benchmark_return_3d": [0.012, 0.012],
            }
        )
        broken = base.assign(
            portfolio_daily_source_score=[0.0, 0.12],
            portfolio_daily_source_release_quality=[0.0, 0.16],
            portfolio_daily_source_economic_release_score=[0.0, 0.12],
            portfolio_daily_source_opportunity_cost=[0.0, 0.82],
            portfolio_daily_source_forward_excess_5d=[0.0, 0.11],
            portfolio_daily_source_receiver_forward_spread=[-0.050, -0.050],
            portfolio_daily_allocation_transfer_score=[0.16, 0.16],
            portfolio_daily_allocation_dead_branch_risk=[0.82, 0.82],
        )

        constraints = AllocationOptimizerConstraints(
            cash_reserve_target=0.05,
            turnover_limit=0.40,
            max_position_weight=0.25,
            min_trade_weight=0.0,
        )
        strong_solution = solve_semidifferentiable_allocation(base, constraints=constraints)
        broken_solution = solve_semidifferentiable_allocation(broken, constraints=constraints)

        self.assertLess(strong_solution.target_weight["SRC"], 0.20)
        self.assertGreater(strong_solution.target_weight["RECV"], broken_solution.target_weight["RECV"] + 0.05)
        self.assertGreater(
            strong_solution.diagnostics["portfolio_daily_allocation_credit_closure_mean"],
            broken_solution.diagnostics["portfolio_daily_allocation_credit_closure_mean"] + 0.20,
        )

    def test_r42_loss_profile_and_resource_gate_are_not_long_r41_reuse(self) -> None:
        profile = "split_heads_portfolio_daily_utility_credit_allocation_r42"
        loss_profile = "alpha_result_value_budget_split_v27"

        self.assertIn(profile, SEARCH_PROFILES)
        self.assertIn(profile, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["loss_profile"], loss_profile)
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[profile], "end_to_end_allocation_layer_v1")
        self.assertLessEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["epochs"], 24)
        self.assertLessEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["min_epochs"], 16)

        config = LOSS_PROFILE_CONFIGS[loss_profile]
        sample_weights = config["sample_scalar_loss_weights"]
        multi_weights = config["multi_objective_loss_weights"]
        for name in (
            "portfolio_daily_allocation_net_utility_target",
            "portfolio_daily_allocation_credit_closure_target",
            "portfolio_daily_allocation_resource_efficiency_target",
        ):
            self.assertIn(name, sample_weights)
            self.assertGreater(sample_weights[name], 2.5)
        self.assertGreater(multi_weights["portfolio_utility_credit_closure_total"], 0.70)
        self.assertGreater(
            multi_weights["portfolio_utility_credit_closure_total"],
            multi_weights["action_total"] * 16.0,
        )

        bad_trial = TrialResult(
            trial_id=1,
            trial_tag="bad_trial",
            status="completed",
            phase="screening",
            role="",
            source_trial_tag="",
            trial_config=SEARCH_PROFILE_BASE_TRIALS[profile],
            protocol_summary_path="",
            performance_score=-1.0,
            stability_score=-1.0,
            composite_score=-1.0,
            score_breakdown={},
            primary_metrics={
                "training_evidence_status": "sufficient",
                "annual_return": 0.02,
                "monthly_return_mean": -0.004,
                "max_drawdown": -0.21,
                "cash_timing_quality_1d": -0.14,
                "portfolio_daily_receiver_target_count": 6,
                "portfolio_daily_receiver_unrealized_deploy_share": 0.0,
                "portfolio_daily_source_target_count": 0,
                "portfolio_daily_source_realized_sell_rate": 0.0,
            },
            promotion_status="shadow_only",
            failed_checks=["cash_timing_quality_1d", "max_drawdown"],
            gate_pass_ratio=0.25,
            passed_check_count=3,
            total_check_count=12,
        )
        gate = _resource_gate_after_screening(profile, [bad_trial], selected_trial_count=3)
        self.assertFalse(gate["continue_screening"])
        self.assertTrue(gate["resource_gate_triggered"])
        self.assertIn("source_release_dead", gate["failed_resource_checks"])
        self.assertEqual(gate["estimated_saved_screening_trials"], 2)

    def test_r42_utility_credit_loss_penalizes_disconnected_allocation(self) -> None:
        targets = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.86, 0.08]),
            "portfolio_daily_unified_source_score": torch.tensor([0.10, 0.84]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.16, 0.20]),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.88, 0.82]),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.86, 0.84]),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.82, 0.78]),
            "portfolio_daily_allocation_uncertainty_pressure_target": torch.tensor([0.18, 0.16]),
            "portfolio_daily_allocation_tail_risk_control_target": torch.tensor([0.16, 0.14]),
            "portfolio_daily_allocation_cash_deployment_target": torch.tensor([0.78, 0.74]),
            "portfolio_daily_allocation_drawdown_control_target": torch.tensor([0.14, 0.12]),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([1.0, 0.0]),
            "portfolio_daily_source_candidate_mask": torch.tensor([0.0, 1.0]),
        }
        good_outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.82, 0.06]),
            "portfolio_daily_unified_source_score": torch.tensor([0.06, 0.80]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.16, 0.20]),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.84, 0.78]),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.82, 0.80]),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.78, 0.74]),
        }
        bad_outputs = {
            **good_outputs,
            "portfolio_daily_unified_receiver_score": torch.tensor([0.12, 0.70]),
            "portfolio_daily_unified_source_score": torch.tensor([0.68, 0.12]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.78, 0.74]),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.24, 0.20]),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.22, 0.18]),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.18, 0.16]),
        }

        good_loss = _portfolio_utility_credit_closure_loss(good_outputs, targets)
        bad_loss = _portfolio_utility_credit_closure_loss(bad_outputs, targets)

        self.assertGreater(float(bad_loss), float(good_loss) + 0.25)

    def test_r42_risk_and_utility_targets_are_wired_into_fit_targets(self) -> None:
        fit_source = inspect.getsource(model_seq_v3.fit_policy_models_v3)
        for name in (
            "portfolio_daily_allocation_uncertainty_pressure_target",
            "portfolio_daily_allocation_tail_risk_control_target",
            "portfolio_daily_allocation_decision_focused_objective",
            "portfolio_daily_allocation_net_utility_target",
            "portfolio_daily_allocation_credit_closure_target",
            "portfolio_daily_allocation_resource_efficiency_target",
        ):
            self.assertIn(f'sample_frame.get("{name}"', fit_source)

    def test_r43_primal_dual_profile_is_decision_first_not_r42_relabel(self) -> None:
        profile = "split_heads_portfolio_daily_primal_dual_decision_allocation_r43"
        loss_profile = "alpha_result_value_budget_split_v28"

        self.assertIn(profile, SEARCH_PROFILES)
        self.assertIn(profile, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["loss_profile"], loss_profile)
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[profile], "end_to_end_allocation_layer_v1")
        self.assertLessEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["epochs"], 20)
        self.assertLessEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["min_epochs"], 14)

        multi_weights = LOSS_PROFILE_CONFIGS[loss_profile]["multi_objective_loss_weights"]
        self.assertGreater(multi_weights["portfolio_primal_dual_decision_total"], 1.0)
        self.assertGreater(
            multi_weights["portfolio_primal_dual_decision_total"],
            multi_weights["action_total"] * 35.0,
        )
        self.assertGreater(
            multi_weights["portfolio_primal_dual_decision_total"],
            multi_weights["portfolio_utility_credit_closure_total"],
        )

        bad_trial = TrialResult(
            trial_id=1,
            trial_tag="bad_trial",
            status="completed",
            phase="screening",
            role="",
            source_trial_tag="",
            trial_config=SEARCH_PROFILE_BASE_TRIALS[profile],
            protocol_summary_path="",
            performance_score=-1.0,
            stability_score=-1.0,
            composite_score=-1.0,
            score_breakdown={},
            primary_metrics={
                "training_evidence_status": "sufficient",
                "annual_return": 0.04,
                "monthly_return_mean": -0.001,
                "max_drawdown": -0.19,
                "cash_timing_quality_1d": -0.10,
                "portfolio_daily_receiver_target_count": 5,
                "portfolio_daily_receiver_unrealized_deploy_share": 0.07,
                "portfolio_daily_source_target_count": 0,
                "portfolio_daily_source_realized_sell_rate": 0.0,
            },
            promotion_status="shadow_only",
            failed_checks=["cash_timing_quality_1d", "max_drawdown"],
            gate_pass_ratio=0.25,
            passed_check_count=3,
            total_check_count=12,
        )
        gate = _resource_gate_after_screening(profile, [bad_trial], selected_trial_count=3)
        self.assertTrue(gate["resource_gate_triggered"])
        self.assertFalse(gate["continue_screening"])
        self.assertIn("source_release_dead", gate["failed_resource_checks"])

    def test_r43_primal_dual_decision_loss_penalizes_tail_false_source_and_dead_cash(self) -> None:
        targets = {
            "date_code": torch.tensor([0, 0, 1, 1]),
            "current_weight": torch.tensor([0.0, 0.18, 0.20, 0.0]),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([1.0, 0.0, 0.0, 1.0]),
            "portfolio_daily_source_candidate_mask": torch.tensor([0.0, 1.0, 1.0, 0.0]),
            "portfolio_daily_unified_receiver_score": torch.tensor([0.88, 0.05, 0.05, 0.72]),
            "portfolio_daily_unified_source_score": torch.tensor([0.05, 0.82, 0.08, 0.05]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.10, 0.12, 0.72, 0.34]),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.88, 0.84, 0.22, 0.40]),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.86, 0.82, 0.18, 0.38]),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.82, 0.80, 0.16, 0.36]),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.86, 0.78, 0.14, 0.44]),
            "portfolio_daily_allocation_uncertainty_pressure_target": torch.tensor([0.08, 0.08, 0.72, 0.70]),
            "portfolio_daily_allocation_tail_risk_control_target": torch.tensor([0.06, 0.06, 0.78, 0.74]),
            "portfolio_daily_allocation_drawdown_control_target": torch.tensor([0.08, 0.08, 0.76, 0.72]),
            "portfolio_daily_allocation_cash_deployment_target": torch.tensor([0.82, 0.80, 0.10, 0.18]),
            "portfolio_daily_receiver_forward_excess_5d": torch.tensor([0.090, 0.0, 0.0, 0.035]),
            "portfolio_daily_source_forward_excess_5d": torch.tensor([0.0, -0.080, 0.120, 0.0]),
            "portfolio_daily_source_hard_negative_penalty": torch.tensor([0.0, 0.04, 0.92, 0.0]),
            "portfolio_daily_source_tail_false_sell_penalty": torch.tensor([0.0, 0.02, 0.90, 0.0]),
            "portfolio_daily_source_positive_forward_penalty": torch.tensor([0.0, 0.02, 0.88, 0.0]),
            "portfolio_daily_source_opportunity_cost_penalty": torch.tensor([0.0, 0.04, 0.86, 0.0]),
            "portfolio_daily_source_release_preference": torch.tensor([0.0, 0.82, 0.05, 0.0]),
            "portfolio_daily_receiver_source_spread_reward": torch.tensor([0.86, 0.86, 0.04, 0.20]),
            "portfolio_daily_transfer_regret_target": torch.tensor([0.82, 0.82, 0.12, 0.18]),
        }
        good_outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.86, 0.04, 0.03, 0.18]),
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.82, 0.03, 0.04]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.10, 0.10, 0.78, 0.76]),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.84, 0.76, 0.16, 0.34]),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.84, 0.80, 0.20, 0.36]),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.82, 0.78, 0.16, 0.34]),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.78, 0.76, 0.16, 0.32]),
        }
        bad_outputs = {
            **good_outputs,
            "portfolio_daily_unified_receiver_score": torch.tensor([0.18, 0.04, 0.04, 0.70]),
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.12, 0.88, 0.04]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.82, 0.80, 0.12, 0.10]),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.20, 0.18, 0.78, 0.72]),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.22, 0.18, 0.78, 0.72]),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.20, 0.16, 0.76, 0.70]),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.18, 0.14, 0.74, 0.68]),
        }

        good_loss = _portfolio_primal_dual_decision_loss(good_outputs, targets)
        bad_loss = _portfolio_primal_dual_decision_loss(bad_outputs, targets)

        self.assertGreater(float(bad_loss), float(good_loss) + 0.10)

    def test_r44_entropic_transport_profile_is_transport_first_not_r43_relabel(self) -> None:
        profile = "split_heads_portfolio_daily_entropic_transport_allocation_r44"
        loss_profile = "alpha_result_value_budget_split_v29"

        self.assertIn(profile, SEARCH_PROFILES)
        self.assertIn(profile, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["loss_profile"], loss_profile)
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[profile], "end_to_end_allocation_layer_v1")
        self.assertLessEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["epochs"], 18)
        self.assertLessEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["min_epochs"], 12)

        multi_weights = LOSS_PROFILE_CONFIGS[loss_profile]["multi_objective_loss_weights"]
        self.assertGreater(multi_weights["portfolio_entropic_transport_decision_total"], 1.2)
        self.assertGreater(
            multi_weights["portfolio_entropic_transport_decision_total"],
            multi_weights["portfolio_primal_dual_decision_total"],
        )
        self.assertGreater(
            multi_weights["portfolio_entropic_transport_decision_total"],
            multi_weights["action_total"] * 60.0,
        )

        bad_trial = TrialResult(
            trial_id=1,
            trial_tag="bad_trial",
            status="completed",
            phase="screening",
            role="",
            source_trial_tag="",
            trial_config=SEARCH_PROFILE_BASE_TRIALS[profile],
            protocol_summary_path="",
            performance_score=-1.0,
            stability_score=-1.0,
            composite_score=-1.0,
            score_breakdown={},
            primary_metrics={
                "training_evidence_status": "sufficient",
                "annual_return": 0.06,
                "monthly_return_mean": 0.000,
                "max_drawdown": -0.18,
                "cash_timing_quality_1d": -0.08,
                "portfolio_daily_receiver_target_count": 4,
                "portfolio_daily_receiver_unrealized_deploy_share": 0.06,
                "portfolio_daily_source_target_count": 0,
                "portfolio_daily_source_realized_sell_rate": 0.0,
            },
            promotion_status="shadow_only",
            failed_checks=["cash_timing_quality_1d", "max_drawdown"],
            gate_pass_ratio=0.25,
            passed_check_count=3,
            total_check_count=12,
        )
        gate = _resource_gate_after_screening(profile, [bad_trial], selected_trial_count=3)
        self.assertTrue(gate["resource_gate_triggered"])
        self.assertFalse(gate["continue_screening"])
        self.assertIn("source_release_dead", gate["failed_resource_checks"])

    def test_r44_entropic_transport_loss_penalizes_unfunded_and_false_source_flow(self) -> None:
        targets = {
            "date_code": torch.tensor([0, 0, 0, 1, 1]),
            "current_weight": torch.tensor([0.0, 0.16, 0.18, 0.20, 0.0]),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0]),
            "portfolio_daily_source_candidate_mask": torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0]),
            "portfolio_daily_unified_receiver_score": torch.tensor([0.90, 0.04, 0.05, 0.04, 0.28]),
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.84, 0.12, 0.08, 0.04]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.08, 0.08, 0.08, 0.76, 0.72]),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.90, 0.86, 0.20, 0.18, 0.30]),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.88, 0.84, 0.18, 0.16, 0.28]),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.84, 0.82, 0.16, 0.14, 0.26]),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.88, 0.82, 0.16, 0.14, 0.30]),
            "portfolio_daily_allocation_uncertainty_pressure_target": torch.tensor([0.06, 0.06, 0.06, 0.78, 0.78]),
            "portfolio_daily_allocation_tail_risk_control_target": torch.tensor([0.04, 0.04, 0.04, 0.82, 0.82]),
            "portfolio_daily_allocation_drawdown_control_target": torch.tensor([0.06, 0.06, 0.06, 0.80, 0.80]),
            "portfolio_daily_allocation_cash_deployment_target": torch.tensor([0.86, 0.84, 0.82, 0.12, 0.12]),
            "portfolio_daily_receiver_forward_excess_5d": torch.tensor([0.10, 0.0, 0.0, 0.0, 0.02]),
            "portfolio_daily_source_forward_excess_5d": torch.tensor([0.0, -0.09, 0.13, 0.12, 0.0]),
            "portfolio_daily_source_hard_negative_penalty": torch.tensor([0.0, 0.04, 0.88, 0.92, 0.0]),
            "portfolio_daily_source_tail_false_sell_penalty": torch.tensor([0.0, 0.02, 0.86, 0.90, 0.0]),
            "portfolio_daily_source_positive_forward_penalty": torch.tensor([0.0, 0.02, 0.88, 0.92, 0.0]),
            "portfolio_daily_source_opportunity_cost_penalty": torch.tensor([0.0, 0.04, 0.84, 0.88, 0.0]),
            "portfolio_daily_source_release_preference": torch.tensor([0.0, 0.84, 0.06, 0.05, 0.0]),
            "portfolio_daily_receiver_source_spread_reward": torch.tensor([0.88, 0.88, 0.10, 0.04, 0.20]),
            "portfolio_daily_transfer_regret_target": torch.tensor([0.84, 0.84, 0.18, 0.12, 0.16]),
        }
        good_outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.88, 0.04, 0.04, 0.04, 0.12]),
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.84, 0.04, 0.04, 0.04]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.08, 0.08, 0.08, 0.82, 0.80]),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.86, 0.80, 0.16, 0.18, 0.24]),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.86, 0.82, 0.18, 0.20, 0.26]),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.84, 0.80, 0.16, 0.18, 0.24]),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.80, 0.78, 0.14, 0.16, 0.22]),
        }
        bad_outputs = {
            **good_outputs,
            "portfolio_daily_unified_receiver_score": torch.tensor([0.12, 0.04, 0.04, 0.04, 0.82]),
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.10, 0.88, 0.92, 0.04]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.86, 0.84, 0.82, 0.10, 0.08]),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.18, 0.16, 0.82, 0.86, 0.76]),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.20, 0.18, 0.82, 0.86, 0.78]),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.18, 0.16, 0.80, 0.84, 0.76]),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.16, 0.14, 0.78, 0.82, 0.74]),
        }

        good_loss = _portfolio_entropic_transport_decision_loss(good_outputs, targets)
        bad_loss = _portfolio_entropic_transport_decision_loss(bad_outputs, targets)

        self.assertGreater(float(bad_loss), float(good_loss) + 0.08)

    def test_r45_conservative_transport_profile_adds_offline_support_gate(self) -> None:
        profile = "split_heads_portfolio_daily_conservative_transport_allocation_r45"
        loss_profile = "alpha_result_value_budget_split_v30"

        self.assertIn(profile, SEARCH_PROFILES)
        self.assertIn(profile, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["loss_profile"], loss_profile)
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[profile], "end_to_end_allocation_layer_v1")
        self.assertLessEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["epochs"], 16)
        self.assertLessEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["min_epochs"], 10)

        multi_weights = LOSS_PROFILE_CONFIGS[loss_profile]["multi_objective_loss_weights"]
        self.assertGreater(multi_weights["portfolio_offline_conservative_support_total"], 0.60)
        self.assertGreater(multi_weights["portfolio_entropic_transport_decision_total"], 1.10)
        self.assertGreater(
            multi_weights["portfolio_entropic_transport_decision_total"],
            multi_weights["portfolio_primal_dual_decision_total"],
        )
        self.assertGreater(
            multi_weights["portfolio_offline_conservative_support_total"],
            multi_weights["action_total"] * 40.0,
        )

        bad_trial = TrialResult(
            trial_id=1,
            trial_tag="bad_trial",
            status="completed",
            phase="screening",
            role="",
            source_trial_tag="",
            trial_config=SEARCH_PROFILE_BASE_TRIALS[profile],
            protocol_summary_path="",
            performance_score=-1.0,
            stability_score=-1.0,
            composite_score=-1.0,
            score_breakdown={},
            primary_metrics={
                "training_evidence_status": "sufficient",
                "annual_return": 0.079,
                "monthly_return_mean": 0.0008,
                "max_drawdown": -0.165,
                "cash_timing_quality_1d": -0.041,
                "portfolio_daily_receiver_target_count": 4,
                "portfolio_daily_receiver_unrealized_deploy_share": 0.041,
                "portfolio_daily_source_target_count": 1,
                "portfolio_daily_source_realized_sell_rate": 0.25,
            },
            promotion_status="shadow_only",
            failed_checks=["cash_timing_quality_1d", "max_drawdown"],
            gate_pass_ratio=0.25,
            passed_check_count=3,
            total_check_count=12,
        )
        gate = _resource_gate_after_screening(profile, [bad_trial], selected_trial_count=3)
        self.assertTrue(gate["resource_gate_triggered"])
        self.assertFalse(gate["continue_screening"])
        self.assertIn("cash_timing_bad", gate["failed_resource_checks"])

    def test_r45_offline_conservative_support_loss_penalizes_ood_overconfidence(self) -> None:
        targets = {
            "date_code": torch.tensor([0, 0, 0, 1, 1, 1]),
            "current_weight": torch.tensor([0.0, 0.18, 0.20, 0.0, 0.16, 0.0]),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "portfolio_daily_source_candidate_mask": torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0]),
            "portfolio_daily_receiver_executable_candidate": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "portfolio_daily_source_executable_candidate": torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
            "portfolio_daily_unified_receiver_score": torch.tensor([0.88, 0.04, 0.05, 0.16, 0.04, 0.04]),
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.82, 0.06, 0.04, 0.74, 0.04]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.08, 0.08, 0.08, 0.82, 0.82, 0.82]),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.86, 0.82, 0.22, 0.20, 0.18, 0.14]),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.84, 0.80, 0.20, 0.18, 0.16, 0.12]),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.82, 0.78, 0.18, 0.16, 0.14, 0.10]),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.84, 0.78, 0.18, 0.16, 0.14, 0.10]),
            "portfolio_daily_allocation_uncertainty_pressure_target": torch.tensor([0.06, 0.06, 0.06, 0.84, 0.84, 0.84]),
            "portfolio_daily_allocation_tail_risk_control_target": torch.tensor([0.04, 0.04, 0.04, 0.88, 0.88, 0.88]),
            "portfolio_daily_allocation_drawdown_control_target": torch.tensor([0.06, 0.06, 0.06, 0.86, 0.86, 0.86]),
            "portfolio_daily_receiver_forward_excess_5d": torch.tensor([0.09, 0.0, 0.0, 0.03, 0.0, 0.0]),
            "portfolio_daily_source_forward_excess_5d": torch.tensor([0.0, -0.08, 0.13, 0.0, 0.11, 0.0]),
            "portfolio_daily_source_hard_negative_penalty": torch.tensor([0.0, 0.02, 0.90, 0.0, 0.86, 0.0]),
            "portfolio_daily_source_tail_false_sell_penalty": torch.tensor([0.0, 0.02, 0.88, 0.0, 0.84, 0.0]),
            "portfolio_daily_source_positive_forward_penalty": torch.tensor([0.0, 0.02, 0.88, 0.0, 0.82, 0.0]),
            "portfolio_daily_source_opportunity_cost_penalty": torch.tensor([0.0, 0.02, 0.82, 0.0, 0.78, 0.0]),
            "portfolio_daily_source_release_preference": torch.tensor([0.0, 0.82, 0.04, 0.0, 0.06, 0.0]),
        }
        good_outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.86, 0.04, 0.04, 0.06, 0.04, 0.04]),
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.82, 0.04, 0.04, 0.04, 0.04]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.08, 0.08, 0.08, 0.86, 0.86, 0.86]),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.82, 0.78, 0.16, 0.14, 0.12, 0.10]),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.84, 0.80, 0.18, 0.16, 0.14, 0.12]),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.82, 0.78, 0.16, 0.14, 0.12, 0.10]),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.80, 0.76, 0.14, 0.12, 0.10, 0.08]),
        }
        bad_outputs = {
            **good_outputs,
            "portfolio_daily_unified_receiver_score": torch.tensor([0.10, 0.08, 0.80, 0.82, 0.08, 0.76]),
            "portfolio_daily_unified_source_score": torch.tensor([0.78, 0.10, 0.88, 0.74, 0.86, 0.72]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.84, 0.82, 0.80, 0.08, 0.08, 0.08]),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.18, 0.14, 0.86, 0.84, 0.80, 0.76]),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.20, 0.16, 0.86, 0.84, 0.80, 0.76]),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.18, 0.14, 0.84, 0.82, 0.80, 0.76]),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.16, 0.12, 0.82, 0.80, 0.76, 0.72]),
        }

        good_loss = _portfolio_offline_conservative_support_loss(good_outputs, targets)
        bad_loss = _portfolio_offline_conservative_support_loss(bad_outputs, targets)

        self.assertGreater(float(bad_loss), float(good_loss) + 0.12)

    def test_r46_differentiable_convex_profile_is_solver_first_not_action_reuse(self) -> None:
        profile = "split_heads_portfolio_daily_differentiable_convex_allocation_r46"
        loss_profile = "alpha_result_value_budget_split_v31"

        self.assertIn(profile, SEARCH_PROFILES)
        self.assertIn(profile, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["loss_profile"], loss_profile)
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[profile], "end_to_end_allocation_layer_v1")
        self.assertLessEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["epochs"], 14)
        self.assertLessEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["min_epochs"], 9)

        multi_weights = LOSS_PROFILE_CONFIGS[loss_profile]["multi_objective_loss_weights"]
        self.assertEqual(multi_weights["action_total"], 0.0)
        self.assertEqual(multi_weights["duration_total"], 0.0)
        self.assertGreater(multi_weights["portfolio_differentiable_convex_allocation_total"], 1.50)
        self.assertGreater(
            multi_weights["portfolio_differentiable_convex_allocation_total"],
            multi_weights["portfolio_entropic_transport_decision_total"],
        )
        self.assertGreater(
            multi_weights["portfolio_differentiable_convex_allocation_total"],
            multi_weights["portfolio_offline_conservative_support_total"] * 2.0,
        )

        bad_trial = TrialResult(
            trial_id=1,
            trial_tag="bad_trial",
            status="completed",
            phase="screening",
            role="",
            source_trial_tag="",
            trial_config=SEARCH_PROFILE_BASE_TRIALS[profile],
            protocol_summary_path="",
            performance_score=-1.0,
            stability_score=-1.0,
            composite_score=-1.0,
            score_breakdown={},
            primary_metrics={
                "training_evidence_status": "sufficient",
                "annual_return": 0.095,
                "monthly_return_mean": 0.0015,
                "max_drawdown": -0.151,
                "cash_timing_quality_1d": -0.021,
                "portfolio_daily_receiver_target_count": 4,
                "portfolio_daily_receiver_unrealized_deploy_share": 0.036,
                "portfolio_daily_source_target_count": 1,
                "portfolio_daily_source_realized_sell_rate": 0.29,
            },
            promotion_status="shadow_only",
            failed_checks=["cash_timing_quality_1d", "max_drawdown"],
            gate_pass_ratio=0.25,
            passed_check_count=3,
            total_check_count=12,
        )
        gate = _resource_gate_after_screening(profile, [bad_trial], selected_trial_count=3)
        self.assertTrue(gate["resource_gate_triggered"])
        self.assertFalse(gate["continue_screening"])
        self.assertIn("source_release_dead", gate["failed_resource_checks"])
        self.assertIn("cash_timing_bad", gate["failed_resource_checks"])

    def test_r46_differentiable_convex_loss_penalizes_kkt_and_support_violations(self) -> None:
        targets = {
            "date_code": torch.tensor([0, 0, 0, 1, 1, 1]),
            "current_weight": torch.tensor([0.0, 0.18, 0.20, 0.0, 0.16, 0.0]),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0]),
            "portfolio_daily_source_candidate_mask": torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0]),
            "portfolio_daily_receiver_executable_candidate": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            "portfolio_daily_source_executable_candidate": torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
            "portfolio_daily_unified_receiver_score": torch.tensor([0.88, 0.04, 0.05, 0.14, 0.04, 0.72]),
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.82, 0.06, 0.04, 0.74, 0.04]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.08, 0.08, 0.08, 0.78, 0.78, 0.78]),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.84, 0.78, 0.18, 0.16, 0.72, 0.76]),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.86, 0.82, 0.20, 0.18, 0.74, 0.78]),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.84, 0.80, 0.18, 0.16, 0.70, 0.74]),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.82, 0.78, 0.16, 0.14, 0.68, 0.72]),
            "portfolio_daily_allocation_cash_deployment_target": torch.tensor([0.80, 0.74, 0.20, 0.12, 0.70, 0.76]),
            "portfolio_daily_allocation_uncertainty_pressure_target": torch.tensor([0.06, 0.06, 0.06, 0.82, 0.82, 0.82]),
            "portfolio_daily_allocation_tail_risk_control_target": torch.tensor([0.04, 0.04, 0.04, 0.88, 0.88, 0.88]),
            "portfolio_daily_allocation_drawdown_control_target": torch.tensor([0.06, 0.06, 0.06, 0.86, 0.86, 0.86]),
            "portfolio_daily_receiver_forward_excess_5d": torch.tensor([0.09, 0.0, 0.0, 0.02, 0.0, 0.08]),
            "portfolio_daily_source_forward_excess_5d": torch.tensor([0.0, -0.08, 0.13, 0.0, -0.06, 0.0]),
            "portfolio_daily_source_hard_negative_penalty": torch.tensor([0.0, 0.02, 0.90, 0.0, 0.04, 0.0]),
            "portfolio_daily_source_tail_false_sell_penalty": torch.tensor([0.0, 0.02, 0.88, 0.0, 0.04, 0.0]),
            "portfolio_daily_source_positive_forward_penalty": torch.tensor([0.0, 0.02, 0.88, 0.0, 0.02, 0.0]),
            "portfolio_daily_source_opportunity_cost_penalty": torch.tensor([0.0, 0.02, 0.82, 0.0, 0.04, 0.0]),
            "portfolio_daily_source_release_preference": torch.tensor([0.0, 0.82, 0.04, 0.0, 0.74, 0.0]),
            "portfolio_daily_receiver_source_spread_reward": torch.tensor([0.72, 0.04, 0.02, 0.08, 0.12, 0.76]),
            "gross_exposure_target": torch.tensor([0.62, 0.62, 0.62, 0.42, 0.42, 0.42]),
            "candidate_budget": torch.tensor([4.0, 4.0, 4.0, 3.0, 3.0, 3.0]),
            "turnover_budget": torch.tensor([0.42, 0.42, 0.42, 0.28, 0.28, 0.28]),
            "max_position_weight_target": torch.tensor([0.22, 0.22, 0.22, 0.18, 0.18, 0.18]),
            "budget_cash_timing_signal_target": torch.tensor([0.08, 0.08, 0.08, 0.84, 0.84, 0.84]),
            "forward_benchmark_return_1d": torch.tensor([0.01, 0.01, 0.01, -0.02, -0.02, -0.02]),
        }
        good_outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.88, 0.04, 0.04, 0.06, 0.08, 0.74]),
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.84, 0.04, 0.04, 0.76, 0.04]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.08, 0.08, 0.08, 0.86, 0.86, 0.86]),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.84, 0.78, 0.16, 0.14, 0.72, 0.76]),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.86, 0.80, 0.18, 0.16, 0.70, 0.78]),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.84, 0.78, 0.16, 0.14, 0.70, 0.74]),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.82, 0.76, 0.14, 0.12, 0.68, 0.72]),
        }
        bad_outputs = {
            **good_outputs,
            "portfolio_daily_unified_receiver_score": torch.tensor([0.08, 0.10, 0.86, 0.82, 0.14, 0.10]),
            "portfolio_daily_unified_source_score": torch.tensor([0.78, 0.12, 0.90, 0.74, 0.10, 0.72]),
            "portfolio_daily_unified_cash_score": torch.tensor([0.88, 0.84, 0.80, 0.08, 0.08, 0.08]),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.18, 0.14, 0.88, 0.84, 0.18, 0.12]),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.20, 0.16, 0.86, 0.84, 0.18, 0.14]),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.18, 0.14, 0.84, 0.82, 0.18, 0.14]),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.16, 0.12, 0.82, 0.80, 0.16, 0.12]),
        }

        good_loss = _portfolio_differentiable_convex_allocation_loss(good_outputs, targets)
        bad_loss = _portfolio_differentiable_convex_allocation_loss(bad_outputs, targets)
        good_terms = _portfolio_differentiable_convex_allocation_loss(good_outputs, targets, return_terms=True)

        self.assertGreater(float(bad_loss), float(good_loss) + 0.05)
        self.assertIsInstance(good_terms, dict)
        for term_name in (
            "regret",
            "constraint_residual",
            "behavior_support_loss",
            "conservative_ope_loss",
            "path_risk_loss",
            "total",
        ):
            self.assertIn(term_name, good_terms)
            self.assertGreaterEqual(float(good_terms[term_name]), 0.0)
        self.assertAlmostEqual(float(good_terms["total"]), float(good_loss), places=6)

        tight_targets = {
            **targets,
            "gross_exposure_target": torch.tensor([0.08, 0.08, 0.08, 0.08, 0.08, 0.08]),
            "turnover_budget": torch.tensor([0.04, 0.04, 0.04, 0.04, 0.04, 0.04]),
            "max_position_weight_target": torch.tensor([0.08, 0.08, 0.08, 0.08, 0.08, 0.08]),
        }
        loose_targets = {
            **targets,
            "gross_exposure_target": torch.tensor([0.76, 0.76, 0.76, 0.76, 0.76, 0.76]),
            "turnover_budget": torch.tensor([0.62, 0.62, 0.62, 0.62, 0.62, 0.62]),
            "max_position_weight_target": torch.tensor([0.28, 0.28, 0.28, 0.28, 0.28, 0.28]),
        }
        tight_terms = _portfolio_differentiable_convex_allocation_loss(good_outputs, tight_targets, return_terms=True)
        loose_terms = _portfolio_differentiable_convex_allocation_loss(good_outputs, loose_targets, return_terms=True)
        self.assertIsInstance(tight_terms, dict)
        self.assertIsInstance(loose_terms, dict)
        self.assertGreater(float(tight_terms["position_residual"]), float(loose_terms["position_residual"]) + 0.002)
        self.assertGreater(float(tight_terms["constraint_residual"]), float(loose_terms["constraint_residual"]) + 0.001)

    def test_r47_true_convex_solver_profile_uses_cvxpy_layer_not_surrogate_only(self) -> None:
        profile = "split_heads_portfolio_daily_true_convex_solver_allocation_r47"
        loss_profile = "alpha_result_value_budget_split_v32"

        self.assertIn(profile, SEARCH_PROFILES)
        self.assertIn(profile, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["loss_profile"], loss_profile)
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[profile], "end_to_end_allocation_layer_v1")
        self.assertLessEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["epochs"], 10)
        self.assertLessEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["min_epochs"], 7)

        multi_weights = LOSS_PROFILE_CONFIGS[loss_profile]["multi_objective_loss_weights"]
        self.assertEqual(multi_weights["action_total"], 0.0)
        self.assertEqual(multi_weights["duration_total"], 0.0)
        self.assertGreater(multi_weights["portfolio_cvxpy_convex_allocation_total"], 1.30)
        self.assertGreater(
            multi_weights["portfolio_cvxpy_convex_allocation_total"],
            multi_weights["portfolio_differentiable_convex_allocation_total"],
        )

        status = _cvxpy_convex_layer_status()
        self.assertTrue(status["available"], status.get("error", ""))
        self.assertTrue({"SCS", "DIFFCP"}.intersection(set(status["installed_solvers"])))

    def test_r47_true_cvxpy_convex_layer_solves_and_backpropagates(self) -> None:
        if not _cvxpy_convex_layer_available():
            self.skipTest("cvxpy/cvxpylayers not installed in this environment")

        targets = {
            "date_code": torch.tensor([0, 0, 0, 0], dtype=torch.float32),
            "current_weight": torch.tensor([0.00, 0.16, 0.18, 0.00], dtype=torch.float32),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float32),
            "portfolio_daily_source_candidate_mask": torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32),
            "portfolio_daily_receiver_executable_candidate": torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float32),
            "portfolio_daily_source_executable_candidate": torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_unified_receiver_score": torch.tensor([0.86, 0.05, 0.04, 0.74], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.82, 0.04, 0.04], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.tensor([0.10, 0.10, 0.10, 0.10], dtype=torch.float32),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.86, 0.80, 0.14, 0.78], dtype=torch.float32),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.88, 0.82, 0.12, 0.80], dtype=torch.float32),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.84, 0.80, 0.10, 0.76], dtype=torch.float32),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.82, 0.78, 0.12, 0.74], dtype=torch.float32),
            "portfolio_daily_allocation_cash_deployment_target": torch.tensor([0.80, 0.74, 0.10, 0.72], dtype=torch.float32),
            "portfolio_daily_receiver_forward_excess_5d": torch.tensor([0.09, 0.00, 0.00, 0.07], dtype=torch.float32),
            "portfolio_daily_source_forward_excess_5d": torch.tensor([0.00, -0.07, 0.12, 0.00], dtype=torch.float32),
            "portfolio_daily_source_hard_negative_penalty": torch.tensor([0.0, 0.02, 0.88, 0.0], dtype=torch.float32),
            "portfolio_daily_source_tail_false_sell_penalty": torch.tensor([0.0, 0.02, 0.86, 0.0], dtype=torch.float32),
            "portfolio_daily_source_positive_forward_penalty": torch.tensor([0.0, 0.02, 0.86, 0.0], dtype=torch.float32),
            "portfolio_daily_source_opportunity_cost_penalty": torch.tensor([0.0, 0.02, 0.82, 0.0], dtype=torch.float32),
            "portfolio_daily_source_release_preference": torch.tensor([0.0, 0.82, 0.04, 0.0], dtype=torch.float32),
            "portfolio_daily_receiver_source_spread_reward": torch.tensor([0.74, 0.04, 0.02, 0.70], dtype=torch.float32),
            "gross_exposure_target": torch.tensor([0.54, 0.54, 0.54, 0.54], dtype=torch.float32),
            "turnover_budget": torch.tensor([0.36, 0.36, 0.36, 0.36], dtype=torch.float32),
            "max_position_weight_target": torch.tensor([0.22, 0.22, 0.22, 0.22], dtype=torch.float32),
            "budget_cash_timing_signal_target": torch.tensor([0.08, 0.08, 0.08, 0.08], dtype=torch.float32),
            "forward_benchmark_return_1d": torch.tensor([0.01, 0.01, 0.01, 0.01], dtype=torch.float32),
        }
        outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.84, 0.05, 0.06, 0.72], requires_grad=True),
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.80, 0.08, 0.05], requires_grad=True),
            "portfolio_daily_unified_cash_score": torch.tensor([0.12, 0.12, 0.12, 0.12], requires_grad=True),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.84, 0.78, 0.14, 0.76], requires_grad=True),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.86, 0.80, 0.12, 0.78], requires_grad=True),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.82, 0.78, 0.12, 0.74], requires_grad=True),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.80, 0.76, 0.10, 0.72], requires_grad=True),
        }

        terms = _portfolio_cvxpy_convex_allocation_loss(outputs, targets, slot_count=4, max_days=1, return_terms=True)
        self.assertIsInstance(terms, dict)
        self.assertGreater(float(terms["solver_success_rate"]), 0.99)
        self.assertGreaterEqual(float(terms["total"]), 0.0)
        loss = terms["total"]
        loss.backward()
        grad_norm = sum(
            float(value.grad.abs().sum())
            for value in outputs.values()
            if value.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

    def test_r47_cvxpy_layer_keeps_current_weight_feasible_when_held_receiver_exceeds_cap(self) -> None:
        if not _cvxpy_convex_layer_available():
            self.skipTest("cvxpy/cvxpylayers not installed in this environment")

        outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.90, 0.62, 0.04, 0.40], requires_grad=True),
            "portfolio_daily_unified_source_score": torch.tensor([0.10, 0.04, 0.80, 0.05], requires_grad=True),
            "portfolio_daily_unified_cash_score": torch.tensor([0.20, 0.20, 0.20, 0.20], requires_grad=True),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.88, 0.60, 0.08, 0.40], requires_grad=True),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.86, 0.58, 0.06, 0.38], requires_grad=True),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.70, 0.56, 0.74, 0.34], requires_grad=True),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.76, 0.54, 0.08, 0.32], requires_grad=True),
        }
        targets = {
            "date_code": torch.zeros(4, dtype=torch.float32),
            "current_weight": torch.tensor([0.32, 0.00, 0.10, 0.00], dtype=torch.float32),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_source_candidate_mask": torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32),
            "portfolio_daily_receiver_executable_candidate": torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_source_executable_candidate": torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32),
            "portfolio_daily_unified_receiver_score": torch.tensor([0.88, 0.62, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.0, 0.0, 0.72, 0.0], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.full((4,), 0.20, dtype=torch.float32),
            "gross_exposure_target": torch.full((4,), 0.42, dtype=torch.float32),
            "turnover_budget": torch.full((4,), 0.04, dtype=torch.float32),
            "max_position_weight_target": torch.full((4,), 0.20, dtype=torch.float32),
            "budget_cash_timing_signal_target": torch.full((4,), 0.20, dtype=torch.float32),
        }

        terms = _portfolio_cvxpy_convex_allocation_loss(outputs, targets, slot_count=4, max_days=1, return_terms=True)
        self.assertGreater(float(terms["solver_success_rate"]), 0.99)
        self.assertEqual(float(terms["fallback_surrogate_loss"]), 0.0)

    def test_r48_full_universe_profile_extends_r47_with_ope_and_larger_slot_bank(self) -> None:
        profile = "split_heads_portfolio_daily_full_universe_convex_ope_allocation_r48"
        loss_profile = "alpha_result_value_budget_split_v33"

        self.assertIn(profile, SEARCH_PROFILES)
        self.assertIn(profile, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["loss_profile"], loss_profile)
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[profile], "end_to_end_allocation_layer_v1")
        self.assertLessEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["epochs"], 8)
        self.assertLessEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["batch_size"], 256)

        multi_weights = LOSS_PROFILE_CONFIGS[loss_profile]["multi_objective_loss_weights"]
        self.assertEqual(multi_weights["action_total"], 0.0)
        self.assertEqual(multi_weights["duration_total"], 0.0)
        self.assertEqual(multi_weights["portfolio_cvxpy_convex_allocation_total"], 0.0)
        self.assertGreater(multi_weights["portfolio_full_universe_convex_allocation_total"], 1.60)
        self.assertGreater(
            model_seq_v3.CVXPY_FULL_UNIVERSE_ALLOCATION_SLOT_COUNT,
            model_seq_v3.CVXPY_CONVEX_ALLOCATION_SLOT_COUNT,
        )
        self.assertEqual(model_seq_v3.CVXPY_FULL_UNIVERSE_ALLOCATION_SLOT_COUNT, 32)
        self.assertEqual(model_seq_v3.CVXPY_FULL_UNIVERSE_ALLOCATION_MAX_DAYS_PER_BATCH, 1)
        self.assertEqual(model_seq_v3.CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_BATCH_INTERVAL, 2)
        self.assertFalse(model_seq_v3.CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_SOLVER_ENABLED)
        self.assertIn(loss_profile, model_seq_v3.LOSS_PROFILE_CONFIGS)

    def test_r48_full_universe_candidate_coverage_penalizes_old_mask_blind_spots(self) -> None:
        if not _cvxpy_convex_layer_available():
            self.skipTest("cvxpy/cvxpylayers not installed in this environment")

        n = 20
        high_oracle = torch.zeros(n, dtype=torch.float32)
        high_oracle[-4:] = 0.95
        outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.10] * 16 + [0.08] * 4, requires_grad=True),
            "portfolio_daily_unified_source_score": torch.full((n,), 0.10, requires_grad=True),
            "portfolio_daily_unified_cash_score": torch.full((n,), 0.18, requires_grad=True),
            "portfolio_daily_allocation_final_objective": torch.full((n,), 0.10, requires_grad=True),
            "portfolio_daily_allocation_net_utility_target": torch.full((n,), 0.10, requires_grad=True),
            "portfolio_daily_allocation_credit_closure_target": torch.full((n,), 0.10, requires_grad=True),
            "portfolio_daily_allocation_resource_efficiency_target": torch.full((n,), 0.10, requires_grad=True),
        }
        targets = {
            "date_code": torch.zeros(n, dtype=torch.float32),
            "current_weight": torch.zeros(n, dtype=torch.float32),
            "portfolio_daily_receiver_candidate_mask": torch.cat([torch.ones(2), torch.zeros(n - 2)]).float(),
            "portfolio_daily_source_candidate_mask": torch.zeros(n, dtype=torch.float32),
            "portfolio_daily_receiver_executable_candidate": torch.cat([torch.ones(2), torch.zeros(n - 2)]).float(),
            "portfolio_daily_source_executable_candidate": torch.zeros(n, dtype=torch.float32),
            "portfolio_daily_unified_receiver_score": high_oracle,
            "portfolio_daily_unified_source_score": torch.zeros(n, dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.full((n,), 0.10, dtype=torch.float32),
            "portfolio_daily_allocation_net_utility_target": high_oracle,
            "portfolio_daily_allocation_resource_efficiency_target": high_oracle,
            "portfolio_daily_allocation_credit_closure_target": high_oracle,
            "portfolio_daily_receiver_forward_excess_5d": torch.cat([torch.zeros(16), torch.full((4,), 0.12)]).float(),
            "portfolio_daily_liquidity_support": torch.ones(n, dtype=torch.float32),
            "portfolio_daily_behavior_propensity": torch.full((n,), 0.18, dtype=torch.float32),
            "gross_exposure_target": torch.full((n,), 0.55, dtype=torch.float32),
            "turnover_budget": torch.full((n,), 0.30, dtype=torch.float32),
            "max_position_weight_target": torch.full((n,), 0.16, dtype=torch.float32),
        }

        terms = _portfolio_cvxpy_convex_allocation_loss(
            outputs,
            targets,
            slot_count=4,
            max_days=1,
            full_universe_mode=True,
            conservative_ope_mode=True,
            return_terms=True,
        )
        self.assertGreater(float(terms["solver_success_rate"]), 0.99)
        self.assertGreater(float(terms["candidate_coverage_loss"]), 0.01)
        loss = terms["total"]
        loss.backward()
        self.assertGreater(float(outputs["portfolio_daily_unified_receiver_score"].grad.abs().sum()), 0.0)

    def test_r48_full_universe_ope_and_realistic_cost_terms_respond_to_bad_support(self) -> None:
        if not _cvxpy_convex_layer_available():
            self.skipTest("cvxpy/cvxpylayers not installed in this environment")

        outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.92, 0.88, 0.12, 0.10, 0.40, 0.35], requires_grad=True),
            "portfolio_daily_unified_source_score": torch.tensor([0.08, 0.08, 0.82, 0.78, 0.20, 0.18], requires_grad=True),
            "portfolio_daily_unified_cash_score": torch.full((6,), 0.16, requires_grad=True),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.90, 0.84, 0.18, 0.16, 0.42, 0.38], requires_grad=True),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.90, 0.84, 0.18, 0.16, 0.42, 0.38], requires_grad=True),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.80, 0.76, 0.70, 0.66, 0.32, 0.28], requires_grad=True),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.86, 0.82, 0.14, 0.12, 0.34, 0.30], requires_grad=True),
        }
        base_targets = {
            "date_code": torch.zeros(6, dtype=torch.float32),
            "current_weight": torch.tensor([0.00, 0.00, 0.18, 0.14, 0.00, 0.00], dtype=torch.float32),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_source_candidate_mask": torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_unified_receiver_score": torch.tensor([0.90, 0.86, 0.0, 0.0, 0.40, 0.35], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.0, 0.0, 0.72, 0.68, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.full((6,), 0.14, dtype=torch.float32),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.90, 0.86, 0.10, 0.10, 0.40, 0.35], dtype=torch.float32),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.80, 0.76, 0.70, 0.68, 0.30, 0.28], dtype=torch.float32),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.84, 0.80, 0.10, 0.10, 0.35, 0.30], dtype=torch.float32),
            "portfolio_daily_receiver_forward_excess_5d": torch.tensor([0.10, 0.08, 0.0, 0.0, 0.04, 0.03], dtype=torch.float32),
            "portfolio_daily_source_forward_excess_5d": torch.tensor([0.0, 0.0, -0.08, -0.06, 0.0, 0.0], dtype=torch.float32),
            "gross_exposure_target": torch.full((6,), 0.48, dtype=torch.float32),
            "turnover_budget": torch.full((6,), 0.36, dtype=torch.float32),
            "max_position_weight_target": torch.full((6,), 0.20, dtype=torch.float32),
            "forward_benchmark_return_1d": torch.full((6,), 0.005, dtype=torch.float32),
            "forward_benchmark_return_3d": torch.full((6,), 0.006, dtype=torch.float32),
        }
        good_targets = {
            **base_targets,
            "portfolio_daily_liquidity_support": torch.full((6,), 0.92, dtype=torch.float32),
            "portfolio_daily_impact_cost": torch.full((6,), 0.001, dtype=torch.float32),
            "portfolio_daily_behavior_propensity": torch.full((6,), 0.55, dtype=torch.float32),
            "portfolio_daily_factor_concentration_proxy": torch.full((6,), 0.05, dtype=torch.float32),
        }
        bad_targets = {
            **base_targets,
            "portfolio_daily_liquidity_support": torch.full((6,), 0.20, dtype=torch.float32),
            "portfolio_daily_impact_cost": torch.full((6,), 0.030, dtype=torch.float32),
            "portfolio_daily_behavior_propensity": torch.full((6,), 0.03, dtype=torch.float32),
            "portfolio_daily_factor_concentration_proxy": torch.full((6,), 0.60, dtype=torch.float32),
        }

        good_terms = _portfolio_full_universe_convex_allocation_loss(outputs, good_targets, return_terms=True)
        bad_terms = _portfolio_full_universe_convex_allocation_loss(outputs, bad_targets, return_terms=True)
        self.assertGreater(float(bad_terms["liquidity_impact_loss"]), float(good_terms["liquidity_impact_loss"]) + 0.001)
        self.assertGreater(float(bad_terms["propensity_support_loss"]), float(good_terms["propensity_support_loss"]) + 0.0001)
        self.assertGreater(float(bad_terms["concentration_risk_loss"]), float(good_terms["concentration_risk_loss"]) + 0.001)

    def test_r49_capital_flow_closure_profile_targets_source_receiver_cash_residual(self) -> None:
        profile = "split_heads_portfolio_daily_capital_flow_closure_r49"
        loss_profile = "alpha_result_value_budget_split_v34"

        self.assertIn(profile, SEARCH_PROFILES)
        self.assertIn(profile, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["loss_profile"], loss_profile)
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[profile], "end_to_end_allocation_layer_v1")
        self.assertLessEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["epochs"], 8)
        self.assertLessEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["batch_size"], 256)

        multi_weights = LOSS_PROFILE_CONFIGS[loss_profile]["multi_objective_loss_weights"]
        self.assertEqual(multi_weights["action_total"], 0.0)
        self.assertEqual(multi_weights["duration_total"], 0.0)
        self.assertGreater(multi_weights["portfolio_capital_flow_closure_total"], 1.60)
        self.assertGreater(
            multi_weights["portfolio_capital_flow_closure_total"],
            multi_weights["portfolio_full_universe_convex_allocation_total"],
        )

        bad_trial = TrialResult(
            trial_id=1,
            trial_tag="bad_trial",
            status="completed",
            phase="screening",
            role="",
            source_trial_tag="",
            trial_config=SEARCH_PROFILE_BASE_TRIALS[profile],
            protocol_summary_path="",
            performance_score=-1.0,
            stability_score=-1.0,
            composite_score=-1.0,
            score_breakdown={},
            primary_metrics={
                "training_evidence_status": "sufficient",
                "annual_return": 0.14,
                "monthly_return_mean": 0.004,
                "max_drawdown": -0.08,
                "cash_timing_quality_1d": -0.001,
                "avg_gross_exposure_target": 0.60,
                "portfolio_daily_exposure_utilization": 0.34,
                "portfolio_daily_receiver_target_count": 20,
                "portfolio_daily_receiver_unrealized_deploy_share": 0.0,
                "portfolio_daily_source_target_count": 0,
                "portfolio_daily_source_realized_sell_rate": 0.0,
            },
            promotion_status="shadow_only",
            failed_checks=["confirm_source_count_floor", "confirm_exposure_utilization_floor"],
            gate_pass_ratio=0.25,
            passed_check_count=3,
            total_check_count=12,
        )
        gate = _resource_gate_after_screening(profile, [bad_trial], selected_trial_count=3)
        self.assertTrue(gate["resource_gate_triggered"])
        self.assertFalse(gate["continue_screening"])
        self.assertIn("source_release_dead", gate["failed_resource_checks"])
        self.assertIn("exposure_utilization_low", gate["failed_resource_checks"])

    def test_r49_capital_flow_loss_penalizes_receiver_deploy_without_source_or_cash_release(self) -> None:
        targets = {
            "date_code": torch.zeros(6, dtype=torch.float32),
            "current_weight": torch.tensor([0.0, 0.0, 0.16, 0.14, 0.12, 0.0], dtype=torch.float32),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
            "portfolio_daily_source_candidate_mask": torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0], dtype=torch.float32),
            "portfolio_daily_receiver_executable_candidate": torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
            "portfolio_daily_source_executable_candidate": torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0], dtype=torch.float32),
            "portfolio_daily_unified_receiver_score": torch.tensor([0.90, 0.86, 0.0, 0.0, 0.0, 0.78], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.0, 0.0, 0.82, 0.78, 0.72, 0.0], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.full((6,), 0.10, dtype=torch.float32),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.90, 0.86, 0.74, 0.70, 0.66, 0.78], dtype=torch.float32),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.92, 0.88, 0.76, 0.72, 0.68, 0.80], dtype=torch.float32),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.86, 0.82, 0.82, 0.78, 0.72, 0.78], dtype=torch.float32),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.84, 0.80, 0.78, 0.74, 0.70, 0.76], dtype=torch.float32),
            "portfolio_daily_allocation_cash_deployment_target": torch.full((6,), 0.82, dtype=torch.float32),
            "portfolio_daily_receiver_forward_excess_5d": torch.tensor([0.10, 0.08, 0.0, 0.0, 0.0, 0.07], dtype=torch.float32),
            "portfolio_daily_source_forward_excess_5d": torch.tensor([0.0, 0.0, -0.08, -0.06, -0.04, 0.0], dtype=torch.float32),
            "portfolio_daily_source_release_preference": torch.tensor([0.0, 0.0, 0.84, 0.80, 0.76, 0.0], dtype=torch.float32),
            "portfolio_daily_receiver_source_spread_reward": torch.tensor([0.80, 0.76, 0.10, 0.10, 0.10, 0.70], dtype=torch.float32),
            "gross_exposure_target": torch.full((6,), 0.68, dtype=torch.float32),
            "turnover_budget": torch.full((6,), 0.42, dtype=torch.float32),
            "max_position_weight_target": torch.full((6,), 0.22, dtype=torch.float32),
            "budget_cash_timing_signal_target": torch.full((6,), 0.06, dtype=torch.float32),
        }
        good_outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.88, 0.84, 0.04, 0.04, 0.04, 0.76], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.04, 0.82, 0.78, 0.72, 0.04], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.full((6,), 0.10, dtype=torch.float32),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.88, 0.84, 0.74, 0.70, 0.66, 0.76], dtype=torch.float32),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.90, 0.86, 0.76, 0.72, 0.68, 0.78], dtype=torch.float32),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.84, 0.80, 0.82, 0.78, 0.72, 0.76], dtype=torch.float32),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.82, 0.78, 0.78, 0.74, 0.70, 0.74], dtype=torch.float32),
        }
        bad_outputs = {
            **good_outputs,
            "portfolio_daily_unified_source_score": torch.full((6,), 0.03, dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.full((6,), 0.92, dtype=torch.float32),
            "portfolio_daily_allocation_credit_closure_target": torch.full((6,), 0.12, dtype=torch.float32),
        }

        good_terms = _portfolio_capital_flow_closure_loss(good_outputs, targets, return_terms=True)
        bad_terms = _portfolio_capital_flow_closure_loss(bad_outputs, targets, return_terms=True)
        self.assertIsInstance(good_terms, dict)
        self.assertIsInstance(bad_terms, dict)
        self.assertGreater(float(bad_terms["total"]), float(good_terms["total"]) + 0.04)
        self.assertGreater(float(bad_terms["source_dead_loss"]), float(good_terms["source_dead_loss"]) + 0.002)
        self.assertGreater(float(bad_terms["funding_shortfall_loss"]), float(good_terms["funding_shortfall_loss"]) + 0.002)
        self.assertGreater(float(bad_terms["over_cash_loss"]), float(good_terms["over_cash_loss"]) + 0.05)
        for term_name in (
            "receiver_demand_mean",
            "clean_source_supply_mean",
            "cash_release_mean",
            "cash_defense_mean",
            "flow_conservation_loss",
            "total",
        ):
            self.assertIn(term_name, good_terms)
            self.assertGreaterEqual(float(good_terms[term_name]), 0.0)

    def test_r49_capital_flow_loss_keeps_false_source_protection(self) -> None:
        targets = {
            "date_code": torch.zeros(4, dtype=torch.float32),
            "current_weight": torch.tensor([0.0, 0.0, 0.18, 0.16], dtype=torch.float32),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_source_candidate_mask": torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
            "portfolio_daily_unified_receiver_score": torch.tensor([0.82, 0.78, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.0, 0.0, 0.80, 0.76], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.full((4,), 0.12, dtype=torch.float32),
            "portfolio_daily_allocation_final_objective": torch.full((4,), 0.78, dtype=torch.float32),
            "portfolio_daily_allocation_net_utility_target": torch.full((4,), 0.80, dtype=torch.float32),
            "portfolio_daily_allocation_credit_closure_target": torch.full((4,), 0.78, dtype=torch.float32),
            "portfolio_daily_allocation_resource_efficiency_target": torch.full((4,), 0.76, dtype=torch.float32),
            "portfolio_daily_allocation_cash_deployment_target": torch.full((4,), 0.78, dtype=torch.float32),
            "portfolio_daily_source_forward_excess_5d": torch.tensor([0.0, 0.0, 0.12, 0.10], dtype=torch.float32),
            "portfolio_daily_source_positive_forward_penalty": torch.tensor([0.0, 0.0, 0.88, 0.82], dtype=torch.float32),
            "portfolio_daily_source_hard_negative_penalty": torch.tensor([0.0, 0.0, 0.80, 0.76], dtype=torch.float32),
            "portfolio_daily_source_tail_false_sell_penalty": torch.tensor([0.0, 0.0, 0.82, 0.78], dtype=torch.float32),
            "portfolio_daily_source_opportunity_cost_penalty": torch.tensor([0.0, 0.0, 0.76, 0.72], dtype=torch.float32),
            "portfolio_daily_source_release_preference": torch.tensor([0.0, 0.0, 0.06, 0.06], dtype=torch.float32),
            "gross_exposure_target": torch.full((4,), 0.55, dtype=torch.float32),
            "turnover_budget": torch.full((4,), 0.34, dtype=torch.float32),
            "max_position_weight_target": torch.full((4,), 0.22, dtype=torch.float32),
        }
        protected_outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.80, 0.76, 0.04, 0.04], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.04, 0.04, 0.04], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.full((4,), 0.14, dtype=torch.float32),
            "portfolio_daily_allocation_final_objective": torch.full((4,), 0.76, dtype=torch.float32),
            "portfolio_daily_allocation_net_utility_target": torch.full((4,), 0.78, dtype=torch.float32),
            "portfolio_daily_allocation_credit_closure_target": torch.full((4,), 0.74, dtype=torch.float32),
            "portfolio_daily_allocation_resource_efficiency_target": torch.full((4,), 0.74, dtype=torch.float32),
        }
        false_source_outputs = {
            **protected_outputs,
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.04, 0.90, 0.86], dtype=torch.float32),
            "portfolio_daily_allocation_credit_closure_target": torch.full((4,), 0.86, dtype=torch.float32),
        }

        protected_terms = _portfolio_capital_flow_closure_loss(protected_outputs, targets, return_terms=True)
        false_terms = _portfolio_capital_flow_closure_loss(false_source_outputs, targets, return_terms=True)
        self.assertGreater(float(false_terms["false_source_loss"]), float(protected_terms["false_source_loss"]) + 0.10)
        self.assertGreater(float(false_terms["total"]), float(protected_terms["total"]) + 0.01)

    def test_r49_capital_flow_terms_separate_cash_defense_from_dead_cash(self) -> None:
        targets = {
            "date_code": torch.zeros(5, dtype=torch.float32),
            "current_weight": torch.tensor([0.18, 0.16, 0.14, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
            "portfolio_daily_source_candidate_mask": torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_receiver_executable_candidate": torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
            "portfolio_daily_source_executable_candidate": torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_unified_receiver_score": torch.tensor([0.0, 0.0, 0.0, 0.28, 0.24], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.34, 0.32, 0.30, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.full((5,), 0.88, dtype=torch.float32),
            "portfolio_daily_allocation_final_objective": torch.full((5,), 0.24, dtype=torch.float32),
            "portfolio_daily_allocation_net_utility_target": torch.full((5,), 0.22, dtype=torch.float32),
            "portfolio_daily_allocation_credit_closure_target": torch.full((5,), 0.32, dtype=torch.float32),
            "portfolio_daily_allocation_resource_efficiency_target": torch.full((5,), 0.30, dtype=torch.float32),
            "portfolio_daily_allocation_cash_deployment_target": torch.full((5,), 0.10, dtype=torch.float32),
            "portfolio_daily_allocation_uncertainty_pressure_target": torch.full((5,), 0.92, dtype=torch.float32),
            "portfolio_daily_allocation_tail_risk_control_target": torch.full((5,), 0.90, dtype=torch.float32),
            "portfolio_daily_allocation_drawdown_control_target": torch.full((5,), 0.88, dtype=torch.float32),
            "market_downside_pressure": torch.full((5,), 0.90, dtype=torch.float32),
            "cash_regime_pressure": torch.full((5,), 0.84, dtype=torch.float32),
            "budget_cash_timing_signal_target": torch.full((5,), 0.90, dtype=torch.float32),
            "gross_exposure_target": torch.full((5,), 0.46, dtype=torch.float32),
            "turnover_budget": torch.full((5,), 0.24, dtype=torch.float32),
            "max_position_weight_target": torch.full((5,), 0.22, dtype=torch.float32),
        }
        under_defended_outputs = {
            "portfolio_daily_unified_receiver_score": torch.full((5,), 0.08, dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.full((5,), 0.24, dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.full((5,), 0.08, dtype=torch.float32),
            "portfolio_daily_allocation_final_objective": torch.full((5,), 0.20, dtype=torch.float32),
            "portfolio_daily_allocation_net_utility_target": torch.full((5,), 0.20, dtype=torch.float32),
            "portfolio_daily_allocation_credit_closure_target": torch.full((5,), 0.30, dtype=torch.float32),
            "portfolio_daily_allocation_resource_efficiency_target": torch.full((5,), 0.28, dtype=torch.float32),
        }
        defended_outputs = {
            **under_defended_outputs,
            "portfolio_daily_unified_cash_score": torch.full((5,), 0.88, dtype=torch.float32),
        }

        under_terms = _portfolio_capital_flow_closure_loss(under_defended_outputs, targets, return_terms=True)
        defended_terms = _portfolio_capital_flow_closure_loss(defended_outputs, targets, return_terms=True)
        self.assertGreater(float(under_terms["cash_defense_loss"]), float(defended_terms["cash_defense_loss"]) + 0.20)
        self.assertLess(float(defended_terms["over_cash_loss"]), 0.01)
        for term_name in (
            "cash_defense_loss",
            "cash_coherence_loss",
            "desired_receiver_flow_mean",
            "effective_receiver_flow_mean",
            "effective_source_flow_mean",
            "source_need_mean",
            "risk_cash_need_mean",
            "current_cash_mean",
            "predicted_gross_mean",
            "deploy_pressure_mean",
            "risk_pressure_mean",
            "role_overlap_loss",
            "source_breadth_loss",
        ):
            self.assertIn(term_name, defended_terms)

    def test_r49_capital_flow_loss_requires_daily_cash_coherence(self) -> None:
        targets = {
            "date_code": torch.zeros(4, dtype=torch.float32),
            "current_weight": torch.tensor([0.0, 0.0, 0.14, 0.12], dtype=torch.float32),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_source_candidate_mask": torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
            "portfolio_daily_unified_receiver_score": torch.tensor([0.82, 0.78, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.0, 0.0, 0.76, 0.72], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.full((4,), 0.44, dtype=torch.float32),
            "portfolio_daily_allocation_final_objective": torch.full((4,), 0.72, dtype=torch.float32),
            "portfolio_daily_allocation_net_utility_target": torch.full((4,), 0.76, dtype=torch.float32),
            "portfolio_daily_allocation_credit_closure_target": torch.full((4,), 0.74, dtype=torch.float32),
            "portfolio_daily_allocation_resource_efficiency_target": torch.full((4,), 0.72, dtype=torch.float32),
            "portfolio_daily_allocation_cash_deployment_target": torch.full((4,), 0.68, dtype=torch.float32),
            "gross_exposure_target": torch.full((4,), 0.60, dtype=torch.float32),
            "turnover_budget": torch.full((4,), 0.36, dtype=torch.float32),
            "max_position_weight_target": torch.full((4,), 0.22, dtype=torch.float32),
            "budget_cash_timing_signal_target": torch.full((4,), 0.20, dtype=torch.float32),
        }
        coherent_outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.80, 0.76, 0.04, 0.04], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.04, 0.04, 0.74, 0.70], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.full((4,), 0.45, dtype=torch.float32),
            "portfolio_daily_allocation_final_objective": torch.full((4,), 0.72, dtype=torch.float32),
            "portfolio_daily_allocation_net_utility_target": torch.full((4,), 0.76, dtype=torch.float32),
            "portfolio_daily_allocation_credit_closure_target": torch.full((4,), 0.74, dtype=torch.float32),
            "portfolio_daily_allocation_resource_efficiency_target": torch.full((4,), 0.72, dtype=torch.float32),
        }
        noisy_outputs = {
            **coherent_outputs,
            "portfolio_daily_unified_cash_score": torch.tensor([0.0, 0.90, 0.0, 0.90], dtype=torch.float32),
        }

        coherent_terms = _portfolio_capital_flow_closure_loss(coherent_outputs, targets, return_terms=True)
        noisy_terms = _portfolio_capital_flow_closure_loss(noisy_outputs, targets, return_terms=True)
        self.assertGreater(float(noisy_terms["cash_coherence_loss"]), float(coherent_terms["cash_coherence_loss"]) + 0.10)
        self.assertGreater(float(noisy_terms["total"]), float(coherent_terms["total"]) + 0.01)

    def test_r49_capital_flow_loss_blocks_role_overlap_and_narrow_source(self) -> None:
        targets = {
            "date_code": torch.zeros(5, dtype=torch.float32),
            "current_weight": torch.tensor([0.15, 0.14, 0.12, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
            "portfolio_daily_source_candidate_mask": torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_receiver_executable_candidate": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
            "portfolio_daily_source_executable_candidate": torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_unified_receiver_score": torch.tensor([0.12, 0.12, 0.12, 0.88, 0.84], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.82, 0.78, 0.74, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.full((5,), 0.12, dtype=torch.float32),
            "portfolio_daily_allocation_final_objective": torch.full((5,), 0.82, dtype=torch.float32),
            "portfolio_daily_allocation_net_utility_target": torch.full((5,), 0.82, dtype=torch.float32),
            "portfolio_daily_allocation_credit_closure_target": torch.full((5,), 0.82, dtype=torch.float32),
            "portfolio_daily_allocation_resource_efficiency_target": torch.full((5,), 0.80, dtype=torch.float32),
            "portfolio_daily_allocation_cash_deployment_target": torch.full((5,), 0.78, dtype=torch.float32),
            "portfolio_daily_source_release_preference": torch.tensor([0.86, 0.82, 0.78, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_receiver_source_spread_reward": torch.tensor([0.10, 0.10, 0.10, 0.86, 0.82], dtype=torch.float32),
            "gross_exposure_target": torch.full((5,), 0.68, dtype=torch.float32),
            "turnover_budget": torch.full((5,), 0.42, dtype=torch.float32),
            "max_position_weight_target": torch.full((5,), 0.22, dtype=torch.float32),
            "budget_cash_timing_signal_target": torch.full((5,), 0.08, dtype=torch.float32),
        }
        clean_outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.06, 0.06, 0.06, 0.88, 0.84], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.82, 0.78, 0.74, 0.04, 0.04], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.full((5,), 0.12, dtype=torch.float32),
            "portfolio_daily_allocation_final_objective": torch.full((5,), 0.82, dtype=torch.float32),
            "portfolio_daily_allocation_net_utility_target": torch.full((5,), 0.82, dtype=torch.float32),
            "portfolio_daily_allocation_credit_closure_target": torch.full((5,), 0.82, dtype=torch.float32),
            "portfolio_daily_allocation_resource_efficiency_target": torch.full((5,), 0.80, dtype=torch.float32),
        }
        overlapped_narrow_outputs = {
            **clean_outputs,
            "portfolio_daily_unified_receiver_score": torch.tensor([0.90, 0.86, 0.82, 0.40, 0.38], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.94, 0.02, 0.02, 0.02, 0.02], dtype=torch.float32),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.88, 0.04, 0.04, 0.20, 0.20], dtype=torch.float32),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.90, 0.03, 0.03, 0.18, 0.18], dtype=torch.float32),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.84, 0.03, 0.03, 0.18, 0.18], dtype=torch.float32),
        }

        clean_terms = _portfolio_capital_flow_closure_loss(clean_outputs, targets, return_terms=True)
        bad_terms = _portfolio_capital_flow_closure_loss(overlapped_narrow_outputs, targets, return_terms=True)
        self.assertGreater(float(bad_terms["role_overlap_loss"]), float(clean_terms["role_overlap_loss"]) + 0.10)
        self.assertGreater(float(bad_terms["source_breadth_loss"]), float(clean_terms["source_breadth_loss"]) + 0.20)
        self.assertGreater(float(bad_terms["total"]), float(clean_terms["total"]) + 0.05)

    def test_r50_integrated_convex_capital_flow_profile_enables_real_solver_training(self) -> None:
        profile = "split_heads_portfolio_daily_integrated_convex_capital_flow_r50"
        loss_profile = "alpha_result_value_budget_split_v35"

        self.assertIn(profile, SEARCH_PROFILES)
        self.assertIn(profile, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["loss_profile"], loss_profile)
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[profile], "end_to_end_allocation_layer_v1")
        self.assertLessEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["epochs"], 6)
        self.assertLessEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["batch_size"], 192)

        resolved_name, resolved_config = model_seq_v3.resolve_loss_profile(loss_profile)
        self.assertEqual(resolved_name, loss_profile)
        self.assertIn("daily_target_loss_weights", resolved_config)
        self.assertGreater(len(resolved_config["daily_target_loss_weights"]), 0)
        self.assertIn("budget_cash_timing_signal_target", resolved_config["daily_target_loss_weights"])

        multi_weights = LOSS_PROFILE_CONFIGS[loss_profile]["multi_objective_loss_weights"]
        self.assertEqual(multi_weights["action_total"], 0.0)
        self.assertEqual(multi_weights["duration_total"], 0.0)
        self.assertGreater(multi_weights["portfolio_cvxpy_convex_allocation_total"], 0.0)
        self.assertGreater(multi_weights["portfolio_full_universe_convex_allocation_total"], 1.30)
        self.assertGreater(multi_weights["portfolio_capital_flow_closure_total"], 1.60)
        self.assertGreater(multi_weights["portfolio_capital_flow_closure_total"], multi_weights["portfolio_full_universe_convex_allocation_total"])
        self.assertFalse(_loss_profile_enables_full_universe_train_solver("alpha_result_value_budget_split_v34"))
        self.assertEqual(model_seq_v3.CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_BATCH_INTERVAL, 2)
        self.assertTrue(_loss_profile_enables_full_universe_train_solver(loss_profile))
        self.assertFalse(model_seq_v3.CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_SOLVER_ENABLED)
        self.assertIn(loss_profile, model_seq_v3.CVXPY_FULL_UNIVERSE_ALLOCATION_TRAIN_SOLVER_LOSS_PROFILES)
        self.assertIn(loss_profile, model_seq_v3.LOSS_PROFILE_CONFIGS)

        bad_trial = TrialResult(
            trial_id=1,
            trial_tag="bad_r50_trial",
            status="completed",
            phase="screening",
            role="",
            source_trial_tag="",
            trial_config=SEARCH_PROFILE_BASE_TRIALS[profile],
            protocol_summary_path="",
            performance_score=-1.0,
            stability_score=-1.0,
            composite_score=-1.0,
            score_breakdown={},
            primary_metrics={
                "training_evidence_status": "sufficient",
                "annual_return": 0.16,
                "monthly_return_mean": 0.004,
                "max_drawdown": -0.08,
                "cash_timing_quality_1d": -0.001,
                "avg_gross_exposure_target": 0.62,
                "portfolio_daily_exposure_utilization": 0.36,
                "portfolio_daily_receiver_target_count": 18,
                "portfolio_daily_receiver_unrealized_deploy_share": 0.0,
                "portfolio_daily_source_target_count": 0,
                "portfolio_daily_source_realized_sell_rate": 0.0,
            },
            promotion_status="shadow_only",
            failed_checks=["confirm_source_count_floor", "confirm_exposure_utilization_floor"],
            gate_pass_ratio=0.25,
            passed_check_count=3,
            total_check_count=12,
        )
        gate = _resource_gate_after_screening(profile, [bad_trial], selected_trial_count=3)
        self.assertTrue(gate["resource_gate_triggered"])
        self.assertFalse(gate["continue_screening"])
        self.assertIn("source_release_dead", gate["failed_resource_checks"])
        self.assertIn("exposure_utilization_low", gate["failed_resource_checks"])

    def test_r50_full_universe_fallback_terms_do_not_report_fake_solver_success(self) -> None:
        outputs = {
            "portfolio_daily_unified_receiver_score": torch.tensor([0.82, 0.78, 0.12, 0.10], requires_grad=True),
            "portfolio_daily_unified_source_score": torch.tensor([0.06, 0.06, 0.72, 0.68], requires_grad=True),
            "portfolio_daily_unified_cash_score": torch.full((4,), 0.16, requires_grad=True),
            "portfolio_daily_allocation_final_objective": torch.tensor([0.80, 0.76, 0.22, 0.20], requires_grad=True),
            "portfolio_daily_allocation_net_utility_target": torch.tensor([0.82, 0.78, 0.20, 0.18], requires_grad=True),
            "portfolio_daily_allocation_credit_closure_target": torch.tensor([0.76, 0.72, 0.70, 0.66], requires_grad=True),
            "portfolio_daily_allocation_resource_efficiency_target": torch.tensor([0.78, 0.74, 0.18, 0.16], requires_grad=True),
        }
        targets = {
            "date_code": torch.zeros(4, dtype=torch.float32),
            "current_weight": torch.tensor([0.0, 0.0, 0.18, 0.14], dtype=torch.float32),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_source_candidate_mask": torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
            "portfolio_daily_receiver_executable_candidate": torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_source_executable_candidate": torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
            "portfolio_daily_unified_receiver_score": torch.tensor([0.84, 0.80, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.0, 0.0, 0.74, 0.70], dtype=torch.float32),
            "portfolio_daily_unified_cash_score": torch.full((4,), 0.14, dtype=torch.float32),
            "gross_exposure_target": torch.full((4,), 0.55, dtype=torch.float32),
            "turnover_budget": torch.full((4,), 0.34, dtype=torch.float32),
            "max_position_weight_target": torch.full((4,), 0.20, dtype=torch.float32),
        }

        terms = _portfolio_full_universe_convex_allocation_loss(
            outputs,
            targets,
            enable_solver=False,
            return_terms=True,
        )
        self.assertEqual(float(terms["solver_success_rate"]), 0.0)
        self.assertGreater(float(terms["fallback_surrogate_loss"]), 0.0)
        self.assertGreater(float(terms["total"]), 0.0)

    def test_r51_native_allocation_vector_profile_uses_torch_only_loss(self) -> None:
        profile = "split_heads_portfolio_daily_native_allocation_vector_r51"
        loss_profile = "alpha_result_value_budget_split_v36"

        self.assertIn(profile, SEARCH_PROFILES)
        self.assertIn(profile, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["loss_profile"], loss_profile)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["epochs"], 8)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["min_epochs"], 6)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["batch_size"], 256)
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[profile], "end_to_end_allocation_layer_v1")

        resolved_name, resolved_config = model_seq_v3.resolve_loss_profile(loss_profile)
        self.assertEqual(resolved_name, loss_profile)
        multi_weights = resolved_config["multi_objective_loss_weights"]
        self.assertEqual(multi_weights["action_total"], 0.0)
        self.assertEqual(multi_weights["duration_total"], 0.0)
        self.assertEqual(multi_weights["scalar_total"], 1.25)
        self.assertGreater(multi_weights["portfolio_native_allocation_vector_total"], 2.0)
        self.assertGreater(multi_weights["portfolio_capital_flow_closure_total"], 0.0)
        self.assertEqual(multi_weights["portfolio_cvxpy_convex_allocation_total"], 0.0)
        self.assertEqual(multi_weights["portfolio_full_universe_convex_allocation_total"], 0.0)
        self.assertFalse(_loss_profile_enables_full_universe_train_solver(loss_profile))

        limits = _build_resource_limits(search_profile=profile, resource_profile="auto")
        self.assertEqual(limits["resource_profile"], "balanced")
        self.assertGreaterEqual(int(limits["thread_limit"]), 1)

    def test_r51_native_projection_enforces_allocation_constraints_and_derived_roles(self) -> None:
        outputs = {
            "portfolio_daily_allocation_weight_logit": torch.tensor([2.2, 1.8, 3.6, 8.0, 2.0], dtype=torch.float32),
            "portfolio_daily_cash_reserve_logit": torch.full((5,), -2.0, dtype=torch.float32),
            "portfolio_daily_allocation_risk_buffer_logit": torch.full((5,), -1.0, dtype=torch.float32),
        }
        targets = {
            "date_code": torch.zeros(5, dtype=torch.float32),
            "current_weight": torch.tensor([0.16, 0.14, 0.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float32),
            "portfolio_daily_source_candidate_mask": torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_receiver_executable_candidate": torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float32),
            "portfolio_daily_source_executable_candidate": torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            "gross_exposure_target": torch.full((5,), 0.54, dtype=torch.float32),
            "turnover_budget": torch.full((5,), 0.18, dtype=torch.float32),
            "max_position_weight_target": torch.full((5,), 0.20, dtype=torch.float32),
            "budget_cash_timing_signal_target": torch.full((5,), 0.02, dtype=torch.float32),
            "portfolio_daily_allocation_cash_deployment_target": torch.full((5,), 0.75, dtype=torch.float32),
        }

        projection = _project_native_allocation_vector(outputs, targets, return_terms=True)
        target_weight = projection["portfolio_daily_target_weight"]
        target_delta = projection["portfolio_daily_target_delta"]
        cash_weight = projection["portfolio_daily_target_cash_weight"]
        turnover = projection["portfolio_daily_target_turnover"]
        receiver_score = projection["portfolio_daily_native_receiver_score"]
        source_score = projection["portfolio_daily_native_source_score"]

        self.assertTrue(bool(torch.all(target_weight >= -1.0e-8)))
        self.assertTrue(bool(torch.all(target_weight <= 0.200001)))
        self.assertLessEqual(float(target_weight.sum() + cash_weight[0]), 1.0001)
        self.assertLessEqual(float(turnover[0]), 0.1801)
        self.assertAlmostEqual(float(target_weight[3]), 0.0, places=6)
        self.assertTrue(bool(torch.all(target_delta[targets["current_weight"] <= 1.0e-8] >= -1.0e-8)))
        self.assertLess(float((receiver_score * source_score).max()), 1.0e-8)
        self.assertLess(float(projection["unsupported_receiver_weight"]), 1.0e-7)
        self.assertLess(float(projection["sell_nonheld_violation"]), 1.0e-7)
        self.assertLess(float(torch.max(torch.abs(cash_weight - cash_weight[0]))), 1.0e-7)

    def test_r51_native_allocation_loss_penalizes_unfunded_receiver_and_underdefended_cash(self) -> None:
        base_targets = {
            "date_code": torch.zeros(6, dtype=torch.float32),
            "current_weight": torch.tensor([0.23, 0.22, 0.20, 0.18, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
            "portfolio_daily_source_candidate_mask": torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_receiver_executable_candidate": torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
            "portfolio_daily_source_executable_candidate": torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_unified_receiver_score": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.92, 0.88], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.76, 0.72, 0.68, 0.64, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_allocation_cash_deployment_target": torch.full((6,), 0.82, dtype=torch.float32),
            "portfolio_daily_allocation_net_utility_target": torch.full((6,), 0.80, dtype=torch.float32),
            "portfolio_daily_allocation_final_objective": torch.full((6,), 0.80, dtype=torch.float32),
            "portfolio_daily_receiver_forward_excess_5d": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.10, 0.08], dtype=torch.float32),
            "portfolio_daily_source_forward_excess_5d": torch.tensor([-0.04, -0.03, -0.02, -0.01, 0.0, 0.0], dtype=torch.float32),
            "gross_exposure_target": torch.full((6,), 0.95, dtype=torch.float32),
            "turnover_budget": torch.full((6,), 0.30, dtype=torch.float32),
            "max_position_weight_target": torch.full((6,), 0.24, dtype=torch.float32),
            "budget_cash_timing_signal_target": torch.full((6,), 0.05, dtype=torch.float32),
        }
        high_receiver_outputs = {
            "portfolio_daily_allocation_weight_logit": torch.tensor([4.0, 4.0, 4.0, 4.0, 7.0, 7.0], dtype=torch.float32),
            "portfolio_daily_cash_reserve_logit": torch.full((6,), -4.0, dtype=torch.float32),
            "portfolio_daily_allocation_risk_buffer_logit": torch.full((6,), -4.0, dtype=torch.float32),
        }
        low_receiver_outputs = {
            **high_receiver_outputs,
            "portfolio_daily_allocation_weight_logit": torch.tensor([5.0, 5.0, 5.0, 5.0, -5.0, -5.0], dtype=torch.float32),
        }
        high_risk_targets = {
            **base_targets,
            "portfolio_daily_allocation_uncertainty_pressure_target": torch.full((6,), 0.92, dtype=torch.float32),
            "portfolio_daily_allocation_tail_risk_control_target": torch.full((6,), 0.90, dtype=torch.float32),
            "portfolio_daily_allocation_drawdown_control_target": torch.full((6,), 0.88, dtype=torch.float32),
            "market_downside_pressure": torch.full((6,), 0.90, dtype=torch.float32),
            "cash_regime_pressure": torch.full((6,), 0.86, dtype=torch.float32),
            "budget_cash_timing_signal_target": torch.full((6,), 0.88, dtype=torch.float32),
        }
        low_cash_outputs = {
            "portfolio_daily_allocation_weight_logit": torch.full((6,), 2.0, dtype=torch.float32),
            "portfolio_daily_cash_reserve_logit": torch.full((6,), -6.0, dtype=torch.float32),
            "portfolio_daily_allocation_risk_buffer_logit": torch.full((6,), -6.0, dtype=torch.float32),
        }
        defended_cash_outputs = {
            **low_cash_outputs,
            "portfolio_daily_cash_reserve_logit": torch.full((6,), 6.0, dtype=torch.float32),
            "portfolio_daily_allocation_risk_buffer_logit": torch.full((6,), 6.0, dtype=torch.float32),
        }

        high_receiver_terms = _portfolio_native_allocation_vector_loss(high_receiver_outputs, base_targets, return_terms=True)
        low_receiver_terms = _portfolio_native_allocation_vector_loss(low_receiver_outputs, base_targets, return_terms=True)
        low_cash_terms = _portfolio_native_allocation_vector_loss(low_cash_outputs, high_risk_targets, return_terms=True)
        defended_cash_terms = _portfolio_native_allocation_vector_loss(defended_cash_outputs, high_risk_targets, return_terms=True)

        self.assertGreater(float(high_receiver_terms["funding_shortfall_loss"]), float(low_receiver_terms["funding_shortfall_loss"]) + 0.0001)
        self.assertGreater(float(low_cash_terms["cash_timing_loss"]), float(defended_cash_terms["cash_timing_loss"]) + 0.0001)
        self.assertGreater(float(low_cash_terms["risk_cost_loss"]), float(defended_cash_terms["risk_cost_loss"]) + 0.0001)
        for term_name in (
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
            "total",
        ):
            self.assertIn(term_name, high_receiver_terms)
            self.assertGreaterEqual(float(high_receiver_terms[term_name]), 0.0)

    def test_true_solver_resource_profile_defaults_to_safe_not_full_machine(self) -> None:
        limits = _build_resource_limits(
            search_profile="split_heads_portfolio_daily_integrated_convex_capital_flow_r50",
            resource_profile="auto",
        )

        self.assertEqual(limits["resource_profile"], "safe")
        self.assertGreater(int(limits["thread_limit"]), 0)
        self.assertLessEqual(int(limits["thread_limit"]), 4)
        self.assertEqual(limits["process_priority"], "below_normal")
        self.assertGreater(int(limits["cpu_affinity_mask"]), 0)

        env = _resource_limited_child_env(limits)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
            self.assertEqual(env[key], str(limits["thread_limit"]))
        self.assertEqual(env["CONTINUOUS_POLICY_RESOURCE_PROFILE"], "safe")

    def test_full_resource_profile_is_explicit_and_unlimited(self) -> None:
        limits = _build_resource_limits(
            search_profile="split_heads_portfolio_daily_integrated_convex_capital_flow_r50",
            resource_profile="full",
        )

        self.assertEqual(limits["resource_profile"], "full")
        self.assertEqual(int(limits["thread_limit"]), 0)
        self.assertEqual(int(limits["cpu_affinity_mask"]), 0)
        self.assertEqual(limits["process_priority"], "normal")

    def test_unified_allocation_problem_respects_hard_executable_candidate_masks(self) -> None:
        frame = pd.DataFrame(
            {
                "stock": ["HELD_BAD", "HELD_GOOD", "FLAT_BAD", "FLAT_GOOD"],
                "current_weight": [0.20, 0.20, 0.0, 0.0],
                "action_label": ["reduce", "reduce", "open", "open"],
                "portfolio_daily_receiver_score": [0.00, 0.00, 0.99, 0.58],
                "portfolio_daily_source_score": [0.99, 0.58, 0.00, 0.00],
                "portfolio_daily_cash_score": [0.10, 0.10, 0.10, 0.10],
                "portfolio_daily_receiver_executability": [0.0, 0.0, 1.0, 1.0],
                "portfolio_daily_source_executability": [1.0, 1.0, 0.0, 0.0],
                "portfolio_daily_receiver_add_headroom": [0.0, 0.0, 0.30, 0.30],
                "portfolio_daily_source_release_capacity": [0.20, 0.20, 0.0, 0.0],
                "portfolio_daily_receiver_executable_candidate": [0.0, 0.0, 0.0, 1.0],
                "portfolio_daily_source_executable_candidate": [0.0, 1.0, 0.0, 0.0],
            }
        )

        problem = build_unified_allocation_problem(frame)
        solution = solve_semidifferentiable_allocation(
            frame,
            constraints=AllocationOptimizerConstraints(
                cash_reserve_target=0.05,
                turnover_limit=0.40,
                max_position_weight=0.30,
                min_trade_weight=0.0,
            ),
        )

        self.assertEqual(float(problem.loc[0, "portfolio_daily_unified_source_candidate"]), 0.0)
        self.assertEqual(float(problem.loc[1, "portfolio_daily_unified_source_candidate"]), 1.0)
        self.assertEqual(float(problem.loc[2, "portfolio_daily_unified_receiver_candidate"]), 0.0)
        self.assertEqual(float(problem.loc[3, "portfolio_daily_unified_receiver_candidate"]), 1.0)
        self.assertGreater(solution.target_weight["HELD_BAD"], 0.19)
        self.assertLess(solution.target_weight["HELD_GOOD"], 0.20)
        self.assertEqual(float(solution.target_weight["FLAT_BAD"]), 0.0)
        self.assertGreater(solution.target_weight["FLAT_GOOD"], 0.0)

    def test_end_to_end_allocation_layer_step_uses_optimizer_targets_without_direct_action(self) -> None:
        state = PortfolioState(
            cash_weight=0.80,
            holdings={"HELD": HoldingState(weight=0.20, entry_price=10.0, peak_price=10.0)},
            max_positions=4,
            max_position_weight=0.30,
            turnover_limit=0.40,
        )
        prices = pd.Series({"HELD": 10.0, "RECV_BAD": 10.0, "RECV_GOOD": 10.0})
        policy = pd.DataFrame(
            {
                "action_label": ["skip", "skip", "skip"],
                "action_strength": [0.0, 0.0, 0.0],
                "portfolio_daily_receiver_score": [0.0, 0.99, 0.62],
                "portfolio_daily_source_score": [0.82, 0.0, 0.0],
                "portfolio_daily_cash_score": [0.10, 0.10, 0.10],
                "portfolio_daily_receiver_executability": [0.0, 1.0, 1.0],
                "portfolio_daily_source_executability": [1.0, 0.0, 0.0],
                "portfolio_daily_receiver_add_headroom": [0.0, 0.30, 0.30],
                "portfolio_daily_source_release_capacity": [0.20, 0.0, 0.0],
                "portfolio_daily_receiver_executable_candidate": [0.0, 0.0, 1.0],
                "portfolio_daily_source_executable_candidate": [1.0, 0.0, 0.0],
            },
            index=prices.index,
        )

        result = state.step(
            date="2026-05-02",
            prices=prices,
            policy_frame=policy,
            global_targets={
                "gross_exposure_target": 0.45,
                "turnover_budget": 0.40,
                "max_position_weight_target": 0.30,
                "cash_reserve_target": 0.05,
            },
            budget_semantics="allocation_layer_v1",
            budget_calibration="end_to_end_allocation_layer_v1",
        )

        self.assertEqual(normalize_budget_semantics("allocation_layer"), BUDGET_SEMANTICS_ALLOCATION_LAYER)
        self.assertEqual(
            normalize_budget_calibration("end_to_end_allocation_layer"),
            BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        )
        self.assertGreater(result.weights["RECV_GOOD"], 0.0)
        self.assertEqual(float(result.weights["RECV_BAD"]), 0.0)
        self.assertLess(result.weights["HELD"], 0.20)
        self.assertEqual(result.diagnostics["allocation_layer_primary_mode"], 1.0)
        self.assertEqual(result.diagnostics["allocation_layer_receiver_target_count"], 1)
        self.assertEqual(result.diagnostics["allocation_layer_source_target_count"], 1)
        self.assertEqual(result.diagnostics["direct_action_open_signal_count"], 0)

    def test_r51_native_target_weight_takes_priority_in_allocation_layer(self) -> None:
        state = PortfolioState(
            cash_weight=0.80,
            holdings={"HELD": HoldingState(weight=0.20, entry_price=10.0, peak_price=10.0)},
            max_positions=4,
            max_position_weight=0.30,
            turnover_limit=0.40,
        )
        prices = pd.Series({"HELD": 10.0, "RECV": 10.0, "OTHER": 10.0})
        policy = pd.DataFrame(
            {
                "action_label": ["skip", "skip", "skip"],
                "action_strength": [0.0, 0.0, 0.0],
                "portfolio_daily_receiver_score": [0.0, 0.10, 0.99],
                "portfolio_daily_source_score": [0.10, 0.0, 0.0],
                "portfolio_daily_cash_score": [0.20, 0.20, 0.20],
                "portfolio_daily_receiver_executability": [0.0, 1.0, 1.0],
                "portfolio_daily_source_executability": [1.0, 0.0, 0.0],
                "portfolio_daily_receiver_add_headroom": [0.0, 0.30, 0.30],
                "portfolio_daily_source_release_capacity": [0.20, 0.0, 0.0],
                "portfolio_daily_receiver_executable_candidate": [0.0, 1.0, 0.0],
                "portfolio_daily_source_executable_candidate": [1.0, 0.0, 0.0],
                "portfolio_daily_target_weight": [0.10, 0.18, 0.0],
                "portfolio_daily_target_delta": [-0.10, 0.18, 0.0],
                "portfolio_daily_target_cash_weight": [0.72, 0.72, 0.72],
                "portfolio_daily_target_turnover": [0.28, 0.28, 0.28],
                "portfolio_daily_native_receiver_score": [0.0, 0.60, 0.0],
                "portfolio_daily_native_source_score": [0.50, 0.0, 0.0],
                "portfolio_daily_native_cash_score": [0.72, 0.72, 0.72],
            },
            index=prices.index,
        )

        result = state.step(
            date="2026-05-03",
            prices=prices,
            policy_frame=policy,
            global_targets={
                "gross_exposure_target": 0.28,
                "turnover_budget": 0.40,
                "max_position_weight_target": 0.30,
                "cash_reserve_target": 0.05,
            },
            budget_semantics="allocation_layer_v1",
            budget_calibration="end_to_end_allocation_layer_v1",
        )

        self.assertEqual(result.diagnostics["allocation_layer_native_target_used"], 1.0)
        self.assertAlmostEqual(result.diagnostics["allocation_layer_expected_turnover"], 0.28, places=6)
        self.assertEqual(result.diagnostics["allocation_layer_receiver_target_count"], 1)
        self.assertEqual(result.diagnostics["allocation_layer_source_target_count"], 1)
        self.assertGreater(result.weights["RECV"], 0.0)
        self.assertEqual(float(result.weights["OTHER"]), 0.0)

    def test_r52_day_set_dataset_returns_complete_days(self) -> None:
        date_codes = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        dataset = DaySetTensorDataset(
            date_codes=date_codes,
            static_x=torch.arange(15, dtype=torch.float32).reshape(5, 3),
            sequence_x=torch.arange(30, dtype=torch.float32).reshape(5, 2, 3),
            action=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
            duration=torch.tensor([0, 1, 2, 0, 1], dtype=torch.long),
            action_soft=torch.zeros(5, len(ACTION_CLASSES), dtype=torch.float32),
            sample_targets={"current_weight": torch.tensor([0.1, 0.2, 0.0, 0.3, 0.0])},
            daily_x=torch.arange(8, dtype=torch.float32).reshape(2, 4),
            daily_targets={"gross_exposure_target": torch.tensor([0.5, 0.6])},
        )

        self.assertEqual(len(dataset), 2)
        first_day = dataset[0]
        second_day = dataset[1]
        self.assertEqual(first_day["static_x"].shape[0], 3)
        self.assertEqual(second_day["static_x"].shape[0], 2)
        self.assertTrue(bool(torch.all(first_day["date_code"] == 0)))
        self.assertTrue(bool(torch.all(second_day["date_code"] == 1)))

        batch = _collate_day_set_batch([first_day, second_day])
        self.assertEqual(tuple(batch["static_x"].shape), (2, 3, 3))
        self.assertEqual(tuple(batch["sequence_x"].shape), (2, 3, 2, 3))
        self.assertTrue(torch.equal(batch["sample_mask"], torch.tensor([[True, True, True], [True, True, False]])))
        self.assertEqual(float(batch["sample_targets"]["current_weight"][1, 2]), 0.0)

    def test_r52_day_set_model_outputs_day_level_cash_and_row_weights(self) -> None:
        model = TemporalDaySetPolicyNet(
            static_input_dim=3,
            sequence_feature_dim=2,
            sequence_steps=2,
            daily_input_dim=4,
            hidden_dim=16,
            sequence_hidden_dim=8,
            sequence_layers=1,
            slot_count=4,
            slot_dim=8,
            dropout=0.0,
        )
        outputs = model(
            static_x=torch.randn(2, 3, 3),
            sequence_x=torch.randn(2, 3, 2, 2),
            daily_x=torch.randn(2, 4),
            sample_mask=torch.tensor([[True, True, True], [True, False, False]]),
        )

        self.assertEqual(tuple(outputs["portfolio_daily_allocation_weight_logit"].shape), (2, 3))
        self.assertEqual(tuple(outputs["portfolio_daily_cash_reserve_logit"].shape), (2,))
        self.assertEqual(tuple(outputs["portfolio_daily_allocation_risk_buffer_logit"].shape), (2,))
        self.assertLess(float(outputs["portfolio_daily_allocation_weight_logit"][1, 1]), -1.0e5)
        self.assertEqual(tuple(outputs["action_logits"].shape), (2, 3, len(ACTION_CLASSES)))

    def test_r52_day_set_projection_enforces_full_day_constraints(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sample_mask = torch.tensor([[True, True, True, True, False]], dtype=torch.bool, device=device)
        outputs = {
            "portfolio_daily_allocation_weight_logit": torch.tensor([[2.2, 1.8, 5.0, 8.0, 9.0]], dtype=torch.float32, device=device),
            "portfolio_daily_cash_reserve_logit": torch.tensor([-2.0], dtype=torch.float32, device=device),
            "portfolio_daily_allocation_risk_buffer_logit": torch.tensor([-1.0], dtype=torch.float32, device=device),
        }
        targets = {
            "current_weight": torch.tensor([[0.16, 0.14, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]], dtype=torch.float32, device=device),
            "portfolio_daily_source_candidate_mask": torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device),
            "portfolio_daily_receiver_executable_candidate": torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0]], dtype=torch.float32, device=device),
            "portfolio_daily_source_executable_candidate": torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device),
            "gross_exposure_target": torch.tensor([0.54], dtype=torch.float32, device=device),
            "turnover_budget": torch.tensor([0.18], dtype=torch.float32, device=device),
            "max_position_weight_target": torch.tensor([0.20], dtype=torch.float32, device=device),
            "budget_cash_timing_signal_target": torch.tensor([0.02], dtype=torch.float32, device=device),
            "portfolio_daily_allocation_cash_deployment_target": torch.tensor([[0.75, 0.75, 0.75, 0.75, 0.0]], dtype=torch.float32, device=device),
        }

        projection = _project_day_set_native_allocation_vector(outputs, targets, sample_mask, return_terms=True)
        target_weight = projection["portfolio_daily_target_weight"]
        target_delta = projection["portfolio_daily_target_delta"]
        cash_weight = projection["portfolio_daily_target_cash_weight"]
        receiver_score = projection["portfolio_daily_native_receiver_score"]
        source_score = projection["portfolio_daily_native_source_score"]

        self.assertTrue(bool(torch.all(target_weight >= -1.0e-8)))
        self.assertTrue(bool(torch.all(target_weight <= 0.200001)))
        self.assertEqual(float(target_weight[0, 4]), 0.0)
        self.assertAlmostEqual(float(target_weight[0, 3]), 0.0, places=6)
        self.assertLessEqual(float(target_weight[0].sum() + cash_weight[0]), 1.0001)
        self.assertLessEqual(float(projection["portfolio_daily_target_turnover"][0]), 0.1801)
        self.assertTrue(bool(torch.all(target_delta[targets["current_weight"] <= 1.0e-8] >= -1.0e-8)))
        self.assertLess(float((receiver_score * source_score).max()), 1.0e-8)
        self.assertLess(float(projection["unsupported_receiver_weight"]), 1.0e-7)
        self.assertLess(float(projection["sell_nonheld_violation"]), 1.0e-7)
        self.assertLess(float(projection["padding_weight_violation"]), 1.0e-7)

    def test_r52_loss_profile_uses_day_set_native_loss_without_solver(self) -> None:
        loss_profile = "alpha_result_value_budget_split_v37"
        resolved_name, resolved_config = model_seq_v3.resolve_loss_profile(loss_profile)
        self.assertEqual(resolved_name, loss_profile)
        multi_weights = resolved_config["multi_objective_loss_weights"]
        self.assertEqual(multi_weights["action_total"], 0.0)
        self.assertEqual(multi_weights["duration_total"], 0.0)
        self.assertEqual(multi_weights["scalar_total"], 1.10)
        self.assertGreater(multi_weights["portfolio_day_set_native_allocation_vector_total"], 2.0)
        self.assertEqual(multi_weights["portfolio_native_allocation_vector_total"], 0.0)
        self.assertEqual(multi_weights["portfolio_cvxpy_convex_allocation_total"], 0.0)
        self.assertEqual(multi_weights["portfolio_full_universe_convex_allocation_total"], 0.0)
        self.assertFalse(_loss_profile_enables_full_universe_train_solver(loss_profile))

        terms = _portfolio_day_set_native_allocation_vector_loss(
            {
                "portfolio_daily_allocation_weight_logit": torch.zeros(1, 3),
                "portfolio_daily_cash_reserve_logit": torch.zeros(1),
                "portfolio_daily_allocation_risk_buffer_logit": torch.zeros(1),
            },
            {
                "current_weight": torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32),
                "portfolio_daily_receiver_candidate_mask": torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32),
                "portfolio_daily_source_candidate_mask": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
                "portfolio_daily_receiver_executable_candidate": torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32),
                "portfolio_daily_source_executable_candidate": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
                "gross_exposure_target": torch.tensor([0.45], dtype=torch.float32),
                "turnover_budget": torch.tensor([0.20], dtype=torch.float32),
                "max_position_weight_target": torch.tensor([0.20], dtype=torch.float32),
            },
            torch.tensor([[True, True, False]], dtype=torch.bool),
            return_terms=True,
        )
        for name in (
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
            "total",
        ):
            self.assertIn(name, terms)

    def test_r52_profile_registers_day_set_native_allocation_vector(self) -> None:
        profile = "split_heads_portfolio_daily_day_set_native_allocation_vector_r52"
        self.assertIn(profile, SEARCH_PROFILES)
        self.assertIn(profile, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["loss_profile"], "alpha_result_value_budget_split_v37")
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["epochs"], 6)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["min_epochs"], 4)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["batch_size"], 1)
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[profile], "end_to_end_allocation_layer_v1")
        self.assertNotIn(profile, TRUE_SOLVER_RESOURCE_SEARCH_PROFILES)
        limits = _build_resource_limits(search_profile=profile, resource_profile="auto")
        self.assertEqual(limits["resource_profile"], "balanced")

    def test_r52_artifact_loads_old_and_new_model_types(self) -> None:
        feature_arrays = {
            "static_fill_values": pd.Series([0.0]).to_numpy(dtype=float),
            "static_means": pd.Series([0.0]).to_numpy(dtype=float),
            "static_stds": pd.Series([1.0]).to_numpy(dtype=float),
            "sequence_fill_values": pd.Series([0.0]).to_numpy(dtype=float),
            "sequence_means": pd.Series([0.0]).to_numpy(dtype=float),
            "sequence_stds": pd.Series([1.0]).to_numpy(dtype=float),
            "daily_fill_values": pd.Series([0.0]).to_numpy(dtype=float),
            "daily_means": pd.Series([0.0]).to_numpy(dtype=float),
            "daily_stds": pd.Series([1.0]).to_numpy(dtype=float),
        }
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            old_sample = TemporalSamplePolicyNet(
                static_input_dim=1,
                sequence_feature_dim=1,
                sequence_steps=1,
                hidden_dim=8,
                sequence_hidden_dim=4,
                sequence_layers=1,
                dropout=0.0,
            )
            daily_model = DailyControllerNet(input_dim=1, hidden_dim=8, dropout=0.0)
            old_payload = {
                "artifact_type": "continuous_policy_torch_seq_v3",
                "static_feature_names": ["static_feature"],
                "sequence_base_names": ["seq_feature"],
                "sequence_steps": [0],
                "sequence_columns": ["seq_feature"],
                "daily_feature_names": ["daily_feature"],
                **{name: values.tolist() for name, values in feature_arrays.items()},
                "sample_model_state_dict": old_sample.state_dict(),
                "daily_model_state_dict": daily_model.state_dict(),
                "sample_model_config": {
                    "static_input_dim": 1,
                    "sequence_feature_dim": 1,
                    "sequence_steps": 1,
                    "hidden_dim": 8,
                    "sequence_hidden_dim": 4,
                    "sequence_layers": 1,
                    "dropout": 0.0,
                },
                "daily_model_config": {"input_dim": 1, "hidden_dim": 8, "dropout": 0.0},
                "train_summary": {},
                "training_diagnostics": {},
                "training_contract": {},
                "trained_at": "2026-05-10T00:00:00",
            }
            old_path = temp_path / "old_artifact.pt"
            torch.save(old_payload, old_path)

            loaded_old = load_torch_seq_artifact(old_path)
            self.assertIsInstance(loaded_old.sample_model, TemporalSamplePolicyNet)
            self.assertEqual(loaded_old.training_diagnostics["sample_model_type"], "temporal_sample")

            day_set_sample = TemporalDaySetPolicyNet(
                static_input_dim=1,
                sequence_feature_dim=1,
                sequence_steps=1,
                daily_input_dim=1,
                hidden_dim=8,
                sequence_hidden_dim=4,
                sequence_layers=1,
                slot_count=2,
                slot_dim=4,
                dropout=0.0,
            )
            artifact = TorchContinuousPolicySeqArtifact(
                sample_model=day_set_sample,
                daily_model=daily_model,
                static_feature_names=["static_feature"],
                sequence_base_names=["seq_feature"],
                sequence_steps=[0],
                sequence_columns=["seq_feature"],
                daily_feature_names=["daily_feature"],
                **feature_arrays,
                train_summary={},
                training_diagnostics={"supports_portfolio_day_set_native_allocation_vector": True},
                training_contract={},
                trained_at="2026-05-10T00:00:00",
            )
            new_path = artifact.save(temp_path / "r52_artifact.pt")

            loaded_new = load_torch_seq_artifact(new_path)
            self.assertIsInstance(loaded_new.sample_model, TemporalDaySetPolicyNet)
            self.assertEqual(loaded_new.training_diagnostics["sample_model_type"], "temporal_day_set")
            self.assertTrue(loaded_new.training_diagnostics["supports_portfolio_day_set_native_allocation_vector"])
            self.assertEqual(loaded_new.training_diagnostics["day_set_slot_count"], 2)

    def test_r52_predict_exports_native_target_weight(self) -> None:
        sample_model = TemporalDaySetPolicyNet(
            static_input_dim=1,
            sequence_feature_dim=1,
            sequence_steps=1,
            daily_input_dim=1,
            hidden_dim=16,
            sequence_hidden_dim=8,
            sequence_layers=1,
            slot_count=2,
            slot_dim=8,
            dropout=0.0,
        )
        daily_model = DailyControllerNet(input_dim=1, hidden_dim=8, dropout=0.0)
        sample_model.eval()
        daily_model.eval()
        feature_array = pd.Series([0.0]).to_numpy(dtype=float)
        scale_array = pd.Series([1.0]).to_numpy(dtype=float)
        artifact = TorchContinuousPolicySeqArtifact(
            sample_model=sample_model,
            daily_model=daily_model,
            static_feature_names=["static_feature"],
            sequence_base_names=["seq_feature"],
            sequence_steps=[0],
            sequence_columns=["seq_feature"],
            daily_feature_names=["daily_feature"],
            static_fill_values=feature_array,
            static_means=feature_array,
            static_stds=scale_array,
            sequence_fill_values=feature_array,
            sequence_means=feature_array,
            sequence_stds=scale_array,
            daily_fill_values=feature_array,
            daily_means=feature_array,
            daily_stds=scale_array,
            train_summary={},
            training_diagnostics={
                "sample_model_type": "temporal_day_set",
                "supports_portfolio_day_set_native_allocation_vector": True,
            },
            training_contract={},
            trained_at="2026-05-10T00:00:00",
        )
        state_frame = pd.DataFrame(
            {
                "stock": ["HELD", "RECV", "OTHER"],
                "current_weight": [0.20, 0.0, 0.0],
                "static_feature": [0.2, 0.4, -0.1],
                "seq_feature": [0.1, 0.3, -0.2],
            }
        )

        policy, _ = predict_policy_v3(artifact, state_frame=state_frame, daily_features={"daily_feature": 0.0})

        for column in (
            "portfolio_daily_target_weight",
            "portfolio_daily_target_delta",
            "portfolio_daily_target_cash_weight",
            "portfolio_daily_native_receiver_score",
            "portfolio_daily_native_source_score",
            "portfolio_daily_native_cash_score",
        ):
            self.assertIn(column, policy.columns)
        self.assertAlmostEqual(
            float(policy["portfolio_daily_native_cash_score"].max()),
            float(policy["portfolio_daily_native_cash_score"].min()),
            places=7,
        )

        policy["portfolio_daily_receiver_executable_candidate"] = (
            policy["portfolio_daily_target_delta"].astype(float) > 1.0e-8
        ).astype(float)
        policy["portfolio_daily_source_executable_candidate"] = (
            (policy["portfolio_daily_target_delta"].astype(float) < -1.0e-8)
            & (state_frame.set_index("stock")["current_weight"].reindex(policy.index).fillna(0.0) > 1.0e-8)
        ).astype(float)
        state = PortfolioState(
            cash_weight=0.80,
            holdings={"HELD": HoldingState(weight=0.20, entry_price=10.0, peak_price=10.0)},
            max_positions=4,
            max_position_weight=0.50,
            turnover_limit=1.00,
        )
        result = state.step(
            date="2026-05-10",
            prices=pd.Series({"HELD": 10.0, "RECV": 10.0, "OTHER": 10.0}),
            policy_frame=policy,
            global_targets={
                "gross_exposure_target": 0.70,
                "turnover_budget": 1.00,
                "max_position_weight_target": 0.50,
                "cash_reserve_target": 0.05,
            },
            budget_semantics="allocation_layer_v1",
            budget_calibration="end_to_end_allocation_layer_v1",
        )
        self.assertEqual(result.diagnostics["allocation_layer_native_target_used"], 1.0)

    def test_confirmatory_candidate_selection_requires_sufficient_training_evidence(self) -> None:
        insufficient_winner = _trial(
            "insufficient_winner",
            evidence_status="insufficient",
            performance_score=99.0,
            stability_score=99.0,
            composite_score=99.0,
        )
        sufficient_runner_up = _trial(
            "sufficient_runner_up",
            evidence_status="sufficient",
            performance_score=1.0,
            stability_score=1.0,
            composite_score=1.0,
        )

        selected = _pick_confirmatory_candidates(
            [insufficient_winner, sufficient_runner_up],
            base_sequence_layers=2,
            max_candidates=1,
        )

        self.assertEqual(selected[0][1].trial_tag, "sufficient_runner_up")

    def test_confirmatory_candidate_selection_prefers_v2_gate_qualified_trial(self) -> None:
        evidence_only = _trial(
            "evidence_only",
            evidence_status="sufficient",
            performance_score=99.0,
            stability_score=99.0,
            composite_score=99.0,
        )
        v2_qualified = _trial(
            "v2_qualified",
            evidence_status="sufficient",
            performance_score=1.0,
            stability_score=1.0,
            composite_score=1.0,
        )
        evidence_only.primary_metrics |= {
            "portfolio_daily_receiver_target_count": 4.0,
            "portfolio_daily_source_target_count": 4.0,
            "portfolio_daily_receiver_unrealized_deploy_share": 0.0,
            "portfolio_daily_source_realized_sell_rate": 1.0,
            "portfolio_daily_source_target_not_sold_share": 0.0,
            "portfolio_daily_receiver_minus_source_forward_excess_5d": -0.10,
            "portfolio_daily_source_forward_excess_5d": 0.08,
            "portfolio_daily_source_positive_forward_sell_share": 0.70,
            "portfolio_daily_source_strong_positive_forward_sell_count": 4.0,
        }
        v2_qualified.primary_metrics |= {
            "portfolio_daily_receiver_target_count": 4.0,
            "portfolio_daily_source_target_count": 4.0,
            "portfolio_daily_receiver_unrealized_deploy_share": 0.0,
            "portfolio_daily_source_realized_sell_rate": 1.0,
            "portfolio_daily_source_target_not_sold_share": 0.0,
            "portfolio_daily_receiver_minus_source_forward_excess_5d": 0.02,
            "portfolio_daily_source_forward_excess_5d": -0.01,
            "portfolio_daily_source_positive_forward_sell_share": 0.20,
            "portfolio_daily_source_strong_positive_forward_sell_count": 0.0,
            "portfolio_daily_source_max_forward_excess_5d": 0.03,
            "portfolio_daily_source_economic_release_score_mean": 0.30,
            "portfolio_daily_source_bad_forward_spread_risk_mean": 0.20,
            "portfolio_daily_source_economic_block_risk_mean": 0.20,
            "portfolio_daily_source_forward_strength_brake_risk_mean": 0.20,
            "portfolio_daily_source_forward_proxy_keep_risk_mean": 0.10,
            "portfolio_daily_source_release_conviction_mean": 0.50,
            "authorized_add_no_weight_change_share": 0.0,
            "deploy_intent_unrealized_share": 0.0,
            "direct_action_authorization_subset_violation_count": 0.0,
            "avg_gross_exposure_target": 0.60,
            "portfolio_daily_exposure_utilization": 0.80,
            "order_translation_conflict_rate": 0.0,
            "direct_action_order_translation_conflict_rate": 0.0,
            "add_to_hold_conflict_share": 0.0,
            "portfolio_daily_cash_reserve_rate": 0.20,
        }

        selected = _pick_confirmatory_candidates(
            [evidence_only, v2_qualified],
            base_sequence_layers=2,
            max_candidates=1,
        )

        self.assertEqual(selected[0][1].trial_tag, "v2_qualified")

    def test_v2_confirm_stability_rejects_insufficient_training_evidence(self) -> None:
        healthy_metrics = {
            "training_evidence_status": "sufficient",
            "annual_return": 0.30,
            "sharpe": 1.00,
            "monthly_return_mean": 0.020,
            "max_drawdown": -0.08,
            "portfolio_daily_receiver_target_count": 4.0,
            "portfolio_daily_source_target_count": 4.0,
            "portfolio_daily_receiver_unrealized_deploy_share": 0.0,
            "portfolio_daily_source_realized_sell_rate": 0.50,
            "portfolio_daily_source_target_not_sold_share": 0.0,
            "portfolio_daily_receiver_minus_source_forward_excess_5d": 0.010,
            "portfolio_daily_source_forward_excess_5d": -0.010,
            "portfolio_daily_source_positive_forward_sell_share": 0.20,
            "portfolio_daily_source_strong_positive_forward_sell_count": 0.0,
            "portfolio_daily_source_max_forward_excess_5d": 0.030,
            "portfolio_daily_source_economic_release_score_mean": 0.30,
            "portfolio_daily_source_bad_forward_spread_risk_mean": 0.20,
            "portfolio_daily_source_economic_block_risk_mean": 0.20,
            "portfolio_daily_source_forward_strength_brake_risk_mean": 0.20,
            "portfolio_daily_source_forward_proxy_keep_risk_mean": 0.10,
            "portfolio_daily_source_release_conviction_mean": 0.50,
            "authorized_add_no_weight_change_share": 0.0,
            "deploy_intent_unrealized_share": 0.0,
            "avg_gross_exposure_target": 0.60,
            "portfolio_daily_exposure_utilization": 0.80,
        }
        insufficient_source = dict(healthy_metrics)
        insufficient_source["training_evidence_status"] = "insufficient"

        stability = _portfolio_daily_v2_confirm_stability(healthy_metrics, insufficient_source)

        self.assertFalse(stability["stable_confirmatory"])
        self.assertIn("source_training_evidence_sufficient", stability["failed_stability_checks"])

    def test_safe_print_json_does_not_fail_completed_stage_on_closed_stdout(self) -> None:
        class ClosedStdout:
            def write(self, _text: str) -> None:
                raise OSError(22, "Invalid argument")

            def flush(self) -> None:
                raise OSError(22, "Invalid argument")

        with patch("sys.stdout", ClosedStdout()):
            emitted = safe_print_json({"stage": "completed"})

        self.assertFalse(emitted)

    def test_safe_print_json_does_not_fail_completed_stage_on_value_error_stdout(self) -> None:
        class ClosedStdout:
            def write(self, _text: str) -> None:
                raise ValueError("I/O operation on closed file")

            def flush(self) -> None:
                raise ValueError("I/O operation on closed file")

        with patch("sys.stdout", ClosedStdout()):
            emitted = safe_print_json({"stage": "completed"})

        self.assertFalse(emitted)

    def test_study_progress_event_writes_latest_state_and_jsonl(self) -> None:
        with TemporaryDirectory() as temp_dir:
            study_root = Path(temp_dir)

            _write_study_progress_event(
                study_root,
                event="trial_start",
                trial_tag="study__trial_01",
                phase="screening",
                status="running",
            )

            state = json.loads((study_root / "study_progress.json").read_text(encoding="utf-8"))
            lines = (study_root / "study_progress.jsonl").read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(state["event"], "trial_start")
        self.assertEqual(state["trial_tag"], "study__trial_01")
        self.assertEqual(state["status"], "running")
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["phase"], "screening")

    def test_protocol_progress_heartbeat_updates_while_stage_runs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            study_root = Path(temp_dir)
            calls = {"count": 0}

            def slow_protocol(_argv: list[str]) -> int:
                calls["count"] += 1
                time.sleep(0.05)
                return 0

            exit_code = _run_protocol_with_progress(
                study_root=study_root,
                phase="screening",
                role="",
                trial_id=1,
                trial_tag="study__trial_01",
                source_trial_tag="",
                protocol_args=["--tag", "study__trial_01"],
                protocol_fn=slow_protocol,
                heartbeat_interval_seconds=0.01,
                resource_limits={"resource_profile": "safe", "thread_limit": 2, "process_priority": "below_normal"},
            )

            state = json.loads((study_root / "study_progress.json").read_text(encoding="utf-8"))
            events = [
                json.loads(line)
                for line in (study_root / "study_progress.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls["count"], 1)
        self.assertEqual(state["event"], "protocol_complete")
        self.assertEqual(state["trial_tag"], "study__trial_01")
        self.assertEqual(state["resource_limits"]["thread_limit"], 2)
        self.assertTrue(any(event["event"] == "protocol_heartbeat" for event in events))

    def test_every_registered_search_profile_loss_profile_resolves(self) -> None:
        for profile_name, base_trial in SEARCH_PROFILE_BASE_TRIALS.items():
            with self.subTest(profile_name=profile_name):
                loss_profile = str(base_trial.get("loss_profile", model_seq_v3.DEFAULT_LOSS_PROFILE))
                resolved_name, resolved_config = model_seq_v3.resolve_loss_profile(loss_profile)
                self.assertEqual(resolved_name, loss_profile)
                self.assertIn("sample_scalar_loss_weights", resolved_config)
                self.assertIn("daily_target_loss_weights", resolved_config)
                self.assertIn("multi_objective_loss_weights", resolved_config)


if __name__ == "__main__":
    unittest.main()
