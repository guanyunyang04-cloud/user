import unittest

import pandas as pd

from daily_research.continuous_policy.allocation_teacher import build_allocation_teacher_summary
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
