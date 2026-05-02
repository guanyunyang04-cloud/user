import unittest
from types import SimpleNamespace

import torch
import pandas as pd

from daily_research.continuous_policy.allocation_optimizer import (
    AllocationOptimizerConstraints,
    build_unified_allocation_problem,
    solve_semidifferentiable_allocation,
)
from daily_research.continuous_policy.analyze_behavior_gap import _compute_exposure_utilization_from_turnover
from daily_research.continuous_policy.allocation_teacher import build_allocation_teacher_summary
from daily_research.continuous_policy.model_seq_v3 import (
    LOSS_PROFILE_CONFIGS,
    _allocation_objective_consolidation_loss,
    _decision_focused_allocation_regret_loss,
    _source_hard_negative_tail_loss,
    _source_listwise_release_regret_loss,
    _transfer_level_allocation_regret_loss,
    _unified_allocation_consistency_loss,
    predict_policy_v3,
)
from daily_research.continuous_policy.model_v2 import ACTION_CLASSES, DURATION_CLASSES
from daily_research.continuous_policy.portfolio_simulator import (
    BUDGET_CALIBRATION_CASH_CONSTRAINT_PORTFOLIO_DAILY_RANKING_RECEIVER_EXEC,
    BUDGET_SEMANTICS_SPLIT,
    HoldingState,
    PortfolioState,
)
from daily_research.continuous_policy.run_self_optimizing_study import (
    SEARCH_PROFILE_BASE_TRIALS,
    SEARCH_PROFILE_DEFAULT_OBJECTIVES,
    SEARCH_PROFILES,
    TrialResult,
    _pick_confirmatory_candidates,
    _portfolio_daily_v2_confirm_stability,
    _score_protocol_summary,
)


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


if __name__ == "__main__":
    unittest.main()
