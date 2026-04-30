import unittest

import pandas as pd

from daily_research.continuous_policy.allocation_optimizer import (
    AllocationOptimizerConstraints,
    build_unified_allocation_problem,
    solve_semidifferentiable_allocation,
)
from daily_research.continuous_policy.allocation_teacher import build_allocation_teacher_summary
from daily_research.continuous_policy.model_seq_v3 import LOSS_PROFILE_CONFIGS
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
