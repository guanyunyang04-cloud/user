import unittest

import pandas as pd
import torch

import daily_research.continuous_policy.model_seq_v3 as model_seq_v3
from daily_research.continuous_policy.native_allocation import _portfolio_native_allocation_vector_loss
from daily_research.continuous_policy.run_self_optimizing_study import (
    SEARCH_PROFILE_BASE_TRIALS,
    SEARCH_PROFILE_DEFAULT_OBJECTIVES,
    SEARCH_PROFILES,
    TRUE_SOLVER_RESOURCE_SEARCH_PROFILES,
    TrialResult,
    _resource_gate_after_screening,
    _score_protocol_summary,
)
from daily_research.continuous_policy.tests.portfolio_daily_fixtures import _protocol_summary


class AllocationClosureDiagnosticsTest(unittest.TestCase):
    def test_diagnostic_flags_actual_cash_semantics_mismatch(self) -> None:
        from daily_research.continuous_policy.allocation_closure_diagnostics import (
            summarize_allocation_closure_from_turnover,
        )

        turnover_frame = pd.DataFrame(
            {
                "date": ["2026-04-27", "2026-04-28", "2026-04-29"],
                "cash_weight": [0.72, 0.71, 0.73],
                "gross_exposure_target": [0.82, 0.82, 0.82],
                "gross_exposure": [0.28, 0.29, 0.27],
                "allocation_layer_cash_after": [0.72, 0.71, 0.73],
                "allocation_layer_available_cash_to_deploy": [0.66, 0.65, 0.67],
                "allocation_layer_stock_budget": [0.82, 0.82, 0.82],
                "allocation_layer_target_weight_sum": [0.28, 0.29, 0.27],
                "allocation_layer_receiver_candidate_count": [3, 2, 3],
                "allocation_layer_receiver_target_count": [1, 1, 1],
                "allocation_layer_source_target_count": [1, 0, 1],
                "allocation_layer_objective_value": [0.0, 0.0, 0.0],
                "portfolio_daily_cash_reserve_rate": [0.0, 0.0, 0.0],
            }
        )

        summary = summarize_allocation_closure_from_turnover(turnover_frame)

        self.assertTrue(summary["cash_semantics_mismatch"])
        self.assertGreater(summary["actual_cash_weight_mean"], 0.70)
        self.assertLess(summary["actual_gross_exposure_mean"], 0.30)
        self.assertLess(summary["exposure_utilization"], 0.40)
        self.assertGreater(summary["deployable_idle_cash_mean"], 0.35)
        self.assertGreater(summary["receiver_candidate_without_target_day_share"], 0.0)

    def test_diagnostic_reads_simulator_turnover_export_columns(self) -> None:
        from daily_research.continuous_policy.allocation_closure_diagnostics import (
            summarize_allocation_closure_from_turnover,
        )

        turnover_frame = pd.DataFrame(
            {
                "date": ["2026-04-27", "2026-04-28"],
                "cash_weight": [0.72, 0.72],
                "gross_exposure_target": [0.85, 0.85],
                "allocation_layer_available_cash_to_deploy": [0.67, 0.67],
                "allocation_layer_stock_budget": [0.85, 0.85],
                "allocation_layer_target_weight_sum": [0.28, 0.28],
                "allocation_layer_objective_value": [0.0, 0.0],
                "portfolio_daily_cash_reserve_signal": [0.0, 0.0],
                "portfolio_daily_receiver_candidate_count": [2, 2],
                "portfolio_daily_receiver_target_count": [0, 0],
                "portfolio_daily_source_target_count": [0, 0],
                "allocation_layer_native_fallback_used": [0.0, 0.0],
            }
        )

        summary = summarize_allocation_closure_from_turnover(turnover_frame)

        self.assertTrue(summary["cash_semantics_mismatch"])
        self.assertAlmostEqual(summary["actual_gross_exposure_mean"], 0.28, places=6)
        self.assertGreater(summary["deployable_idle_cash_mean"], 0.55)
        self.assertEqual(summary["receiver_candidate_without_target_day_share"], 1.0)

    def test_diagnostic_reports_r53_cash_funded_handoff_fields(self) -> None:
        from daily_research.continuous_policy.allocation_closure_diagnostics import (
            summarize_allocation_closure_from_turnover,
        )

        turnover_frame = pd.DataFrame(
            {
                "date": ["2026-04-27", "2026-04-28"],
                "cash_weight": [0.72, 0.72],
                "gross_exposure_target": [0.85, 0.85],
                "allocation_layer_available_cash_to_deploy": [0.67, 0.67],
                "allocation_layer_stock_budget": [0.85, 0.85],
                "allocation_layer_target_weight_sum": [0.28, 0.28],
                "allocation_layer_cash_funded_deploy_amount": [0.02, 0.01],
                "allocation_layer_source_funded_deploy_amount": [0.00, 0.00],
                "allocation_layer_unused_receiver_headroom": [0.50, 0.48],
                "allocation_layer_receiver_headroom_utilization": [0.04, 0.02],
                "allocation_layer_source_release_required": [0.0, 0.0],
                "allocation_layer_underdeployment_reason": [
                    "cash_funded_deployment_failed",
                    "cash_funded_deployment_failed",
                ],
                "portfolio_daily_cash_reserve_signal": [0.0, 0.0],
                "portfolio_daily_receiver_candidate_count": [4, 4],
                "portfolio_daily_receiver_target_count": [0, 0],
                "portfolio_daily_source_target_count": [0, 0],
            }
        )

        summary = summarize_allocation_closure_from_turnover(turnover_frame)

        self.assertGreater(summary["target_sum_gap"], 0.50)
        self.assertLess(summary["cash_funded_deploy_amount_mean"], 0.05)
        self.assertGreater(summary["unused_receiver_headroom_mean"], 0.45)
        self.assertLess(summary["receiver_headroom_utilization_mean"], 0.10)
        self.assertFalse(summary["source_release_required"])
        self.assertEqual(summary["underdeployment_reason"], "cash_funded_deployment_failed")

    def test_diagnostic_marks_budget_closed_without_receiver_activity_requirement(self) -> None:
        from daily_research.continuous_policy.allocation_closure_diagnostics import (
            summarize_allocation_closure_from_turnover,
        )

        turnover_frame = pd.DataFrame(
            {
                "date": ["2026-05-12", "2026-05-13"],
                "cash_weight": [0.19, 0.20],
                "gross_exposure_target": [0.75, 0.75],
                "gross_exposure": [0.81, 0.80],
                "allocation_layer_available_cash_to_deploy": [0.004, 0.006],
                "allocation_layer_stock_budget": [0.75, 0.75],
                "allocation_layer_target_weight_sum": [0.81, 0.80],
                "allocation_layer_target_sum_gap": [0.004, 0.006],
                "allocation_layer_cash_funded_deploy_amount": [0.01, 0.01],
                "allocation_layer_unused_receiver_headroom": [280.0, 300.0],
                "allocation_layer_receiver_executable_candidate_count": [4, 4],
                "allocation_layer_receiver_target_count": [0, 0],
                "allocation_layer_source_target_count": [0, 0],
                "allocation_layer_source_release_required": [0.0, 0.0],
                "portfolio_daily_cash_reserve_signal": [0.0, 0.0],
            }
        )

        summary = summarize_allocation_closure_from_turnover(turnover_frame)

        self.assertTrue(summary["budget_closed"])
        self.assertFalse(summary["deployment_required"])
        self.assertFalse(summary["receiver_activity_required"])
        self.assertEqual(summary["receiver_candidate_without_target_day_share"], 1.0)
        self.assertEqual(summary["receiver_candidate_without_target_required_day_share"], 0.0)

    def test_native_allocation_loss_penalizes_deployable_idle_cash(self) -> None:
        row_count = 6
        targets = {
            "date_code": torch.zeros(row_count, dtype=torch.float32),
            "current_weight": torch.tensor([0.10, 0.08, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.float32),
            "portfolio_daily_source_candidate_mask": torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.float32),
            "portfolio_daily_receiver_executable_candidate": torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.float32),
            "portfolio_daily_source_executable_candidate": torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.float32),
            "gross_exposure_target": torch.full((row_count,), 0.80, dtype=torch.float32),
            "turnover_budget": torch.full((row_count,), 1.0, dtype=torch.float32),
            "max_position_weight_target": torch.full((row_count,), 0.30, dtype=torch.float32),
            "budget_cash_timing_signal_target": torch.zeros(row_count, dtype=torch.float32),
            "portfolio_daily_allocation_cash_deployment_target": torch.full((row_count,), 0.95, dtype=torch.float32),
            "portfolio_daily_allocation_net_utility_target": torch.full((row_count,), 0.95, dtype=torch.float32),
            "portfolio_daily_allocation_final_objective": torch.full((row_count,), 0.95, dtype=torch.float32),
            "portfolio_daily_unified_receiver_score": torch.tensor([0, 0, 0.92, 0.90, 0.86, 0.82], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.42, 0.38, 0, 0, 0, 0], dtype=torch.float32),
        }
        high_cash_outputs = {
            "portfolio_daily_allocation_weight_logit": torch.tensor([0.0, 0.0, 3.0, 2.9, 2.8, 2.7], dtype=torch.float32),
            "portfolio_daily_cash_reserve_logit": torch.full((row_count,), 6.0, dtype=torch.float32),
            "portfolio_daily_allocation_risk_buffer_logit": torch.full((row_count,), -4.0, dtype=torch.float32),
        }
        deploy_outputs = {
            **high_cash_outputs,
            "portfolio_daily_cash_reserve_logit": torch.full((row_count,), -6.0, dtype=torch.float32),
        }

        high_cash_terms = _portfolio_native_allocation_vector_loss(high_cash_outputs, targets, return_terms=True)
        deploy_terms = _portfolio_native_allocation_vector_loss(deploy_outputs, targets, return_terms=True)

        self.assertIn("deployable_idle_cash_loss", high_cash_terms)
        self.assertIn("stock_budget_gap_loss", high_cash_terms)
        self.assertGreater(
            float(high_cash_terms["deployable_idle_cash_loss"]),
            float(deploy_terms["deployable_idle_cash_loss"]) + 0.05,
        )
        self.assertGreater(float(high_cash_terms["stock_budget_gap_loss"]), 0.05)

    def test_study_scoring_penalizes_actual_cash_idle_even_when_cash_signal_is_zero(self) -> None:
        base = _score_protocol_summary(
            _protocol_summary(
                metrics={"avg_gross_exposure": 0.58},
                semantic={
                    "portfolio_daily_cash_reserve_rate": 0.0,
                    "portfolio_daily_actual_cash_weight_mean": 0.28,
                    "portfolio_daily_actual_gross_exposure_mean": 0.72,
                    "portfolio_daily_deployable_idle_cash_mean": 0.05,
                    "portfolio_daily_cash_semantics_mismatch": 0.0,
                    "avg_gross_exposure_target": 0.74,
                    "portfolio_daily_exposure_utilization": 0.78,
                },
            ),
            objective_profile="end_to_end_allocation_layer_v1",
        )
        high_idle = _score_protocol_summary(
            _protocol_summary(
                metrics={"avg_gross_exposure": 0.24},
                semantic={
                    "portfolio_daily_cash_reserve_rate": 0.0,
                    "portfolio_daily_actual_cash_weight_mean": 0.72,
                    "portfolio_daily_actual_gross_exposure_mean": 0.28,
                    "portfolio_daily_deployable_idle_cash_mean": 0.44,
                    "portfolio_daily_cash_semantics_mismatch": 1.0,
                    "avg_gross_exposure_target": 0.82,
                    "portfolio_daily_exposure_utilization": 0.34,
                },
            ),
            objective_profile="end_to_end_allocation_layer_v1",
        )

        self.assertIn(
            "portfolio_daily_actual_cash_idle_penalty",
            high_idle["score_breakdown"]["performance"],
        )
        self.assertLess(high_idle["composite_score"], base["composite_score"] - 0.40)

    def test_resource_gate_flags_actual_cash_idle_low_exposure(self) -> None:
        profile = "split_heads_portfolio_daily_deployment_cash_exposure_closure_r52e"
        bad_trial = TrialResult(
            trial_id=1,
            trial_tag="bad_trial",
            status="completed",
            phase="screening",
            role="",
            source_trial_tag="",
            trial_config={"sequence_layers": 2},
            protocol_summary_path="",
            performance_score=0.0,
            stability_score=0.0,
            composite_score=0.0,
            score_breakdown={},
            primary_metrics={
                "training_evidence_status": "sufficient",
                "annual_return": 0.18,
                "monthly_return_mean": 0.010,
                "max_drawdown": -0.08,
                "cash_timing_quality_1d": 0.01,
                "portfolio_daily_receiver_target_count": 5,
                "portfolio_daily_receiver_unrealized_deploy_share": 0.0,
                "portfolio_daily_source_target_count": 4,
                "portfolio_daily_source_realized_sell_rate": 0.55,
                "portfolio_daily_cash_reserve_rate": 0.0,
                "portfolio_daily_actual_cash_weight_mean": 0.72,
                "portfolio_daily_deployable_idle_cash_mean": 0.44,
                "portfolio_daily_cash_semantics_mismatch": 1.0,
                "avg_gross_exposure_target": 0.82,
                "portfolio_daily_exposure_utilization": 0.34,
            },
            promotion_status="shadow_only",
            failed_checks=[],
            gate_pass_ratio=1.0,
            passed_check_count=12,
            total_check_count=12,
        )

        gate = _resource_gate_after_screening(profile, [bad_trial], selected_trial_count=3)

        self.assertTrue(gate["resource_gate_triggered"])
        self.assertIn("actual_cash_idle_high", gate["failed_resource_checks"])
        self.assertIn("cash_semantics_mismatch", gate["failed_resource_checks"])
        self.assertIn("exposure_utilization_low", gate["failed_resource_checks"])

    def test_r52e_profile_registers_deployment_cash_exposure_closure(self) -> None:
        profile = "split_heads_portfolio_daily_deployment_cash_exposure_closure_r52e"
        loss_profile = "alpha_result_value_budget_split_v41"

        self.assertIn(profile, SEARCH_PROFILES)
        self.assertIn(profile, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["loss_profile"], loss_profile)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["epochs"], 6)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["min_epochs"], 4)
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[profile], "end_to_end_allocation_layer_v1")
        self.assertNotIn(profile, TRUE_SOLVER_RESOURCE_SEARCH_PROFILES)

        resolved_name, resolved_config = model_seq_v3.resolve_loss_profile(loss_profile)
        self.assertEqual(resolved_name, loss_profile)
        multi_weights = resolved_config["multi_objective_loss_weights"]
        self.assertEqual(multi_weights["action_total"], 0.0)
        self.assertEqual(multi_weights["duration_total"], 0.0)
        self.assertGreater(multi_weights["portfolio_day_set_native_allocation_vector_total"], 3.80)
        self.assertEqual(multi_weights["portfolio_native_allocation_vector_total"], 0.0)

    def test_r53_profile_registers_cash_funded_allocator_without_solver(self) -> None:
        profile = "split_heads_portfolio_daily_cash_funded_allocation_core_r53"
        loss_profile = "alpha_result_value_budget_split_v43"

        self.assertIn(profile, SEARCH_PROFILES)
        self.assertIn(profile, SEARCH_PROFILE_BASE_TRIALS)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["loss_profile"], loss_profile)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["epochs"], 6)
        self.assertEqual(SEARCH_PROFILE_BASE_TRIALS[profile]["min_epochs"], 4)
        self.assertEqual(SEARCH_PROFILE_DEFAULT_OBJECTIVES[profile], "end_to_end_allocation_layer_v1")
        self.assertNotIn(profile, TRUE_SOLVER_RESOURCE_SEARCH_PROFILES)

        resolved_name, resolved_config = model_seq_v3.resolve_loss_profile(loss_profile)
        self.assertEqual(resolved_name, loss_profile)
        multi_weights = resolved_config["multi_objective_loss_weights"]
        self.assertEqual(multi_weights["action_total"], 0.0)
        self.assertEqual(multi_weights["duration_total"], 0.0)
        self.assertGreater(multi_weights["portfolio_day_set_native_allocation_vector_total"], 4.00)
        self.assertGreaterEqual(multi_weights["portfolio_cash_margin_total"], 0.34)
        self.assertGreaterEqual(multi_weights["portfolio_capital_flow_closure_total"], 0.44)
        self.assertEqual(multi_weights["portfolio_cvxpy_convex_allocation_total"], 0.0)
        self.assertEqual(multi_weights["portfolio_full_universe_convex_allocation_total"], 0.0)

    def test_r53_resource_gate_is_cash_first_and_only_requires_source_when_needed(self) -> None:
        profile = "split_heads_portfolio_daily_cash_funded_allocation_core_r53"
        cash_first_bad_trial = TrialResult(
            trial_id=1,
            trial_tag="r53_bad_cash_first",
            status="completed",
            phase="screening",
            role="",
            source_trial_tag="",
            trial_config={"sequence_layers": 2},
            protocol_summary_path="",
            performance_score=0.0,
            stability_score=0.0,
            composite_score=0.0,
            score_breakdown={},
            primary_metrics={
                "training_evidence_status": "sufficient",
                "annual_return": 0.18,
                "monthly_return_mean": 0.010,
                "max_drawdown": -0.08,
                "cash_timing_quality_1d": 0.01,
                "portfolio_daily_receiver_target_count": 5,
                "portfolio_daily_receiver_unrealized_deploy_share": 0.0,
                "portfolio_daily_source_target_count": 0,
                "portfolio_daily_source_realized_sell_rate": 0.0,
                "portfolio_daily_actual_cash_weight_mean": 0.72,
                "portfolio_daily_deployable_idle_cash_mean": 0.44,
                "avg_gross_exposure_target": 0.82,
                "portfolio_daily_exposure_utilization": 0.34,
                "portfolio_daily_cash_funded_deploy_amount_mean": 0.02,
                "portfolio_daily_unused_receiver_headroom_mean": 0.48,
                "portfolio_daily_target_sum_gap": 0.54,
                "portfolio_daily_source_release_required": 0.0,
            },
            promotion_status="shadow_only",
            failed_checks=[],
            gate_pass_ratio=1.0,
            passed_check_count=12,
            total_check_count=12,
        )

        cash_first_gate = _resource_gate_after_screening(profile, [cash_first_bad_trial], selected_trial_count=3)

        self.assertTrue(cash_first_gate["resource_gate_triggered"])
        self.assertIn("cash_funded_deployment_failed", cash_first_gate["failed_resource_checks"])
        self.assertIn("receiver_headroom_unused", cash_first_gate["failed_resource_checks"])
        self.assertIn("target_sum_underdeployed", cash_first_gate["failed_resource_checks"])
        self.assertNotIn("source_release_dead", cash_first_gate["failed_resource_checks"])
        self.assertNotIn("source_release_dead_when_required", cash_first_gate["failed_resource_checks"])

        source_required_trial = TrialResult(
            **{
                **cash_first_bad_trial.__dict__,
                "trial_tag": "r53_bad_source_required",
                "primary_metrics": {
                    **cash_first_bad_trial.primary_metrics,
                    "portfolio_daily_source_release_required": 1.0,
                },
            }
        )
        source_required_gate = _resource_gate_after_screening(profile, [source_required_trial], selected_trial_count=3)

        self.assertIn("source_release_dead_when_required", source_required_gate["failed_resource_checks"])

    def test_r55_native_loss_penalizes_wrong_cash_timing_and_dead_source_release(self) -> None:
        from daily_research.continuous_policy.native_allocation import _portfolio_native_allocation_vector_loss

        row_count = 4
        targets = {
            "date_code": torch.zeros(row_count, dtype=torch.float32),
            "current_weight": torch.tensor([0.25, 0.20, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([0, 0, 1, 1], dtype=torch.float32),
            "portfolio_daily_source_candidate_mask": torch.tensor([1, 1, 0, 0], dtype=torch.float32),
            "portfolio_daily_receiver_executable_candidate": torch.tensor([0, 0, 1, 1], dtype=torch.float32),
            "portfolio_daily_source_executable_candidate": torch.tensor([1, 1, 0, 0], dtype=torch.float32),
            "gross_exposure_target": torch.full((row_count,), 0.80, dtype=torch.float32),
            "turnover_budget": torch.full((row_count,), 1.0, dtype=torch.float32),
            "max_position_weight_target": torch.full((row_count,), 0.30, dtype=torch.float32),
            "budget_cash_timing_signal_target": torch.full((row_count,), 0.90, dtype=torch.float32),
            "portfolio_daily_allocation_cash_deployment_target": torch.full((row_count,), 0.20, dtype=torch.float32),
            "portfolio_daily_allocation_net_utility_target": torch.full((row_count,), 0.80, dtype=torch.float32),
            "portfolio_daily_allocation_final_objective": torch.full((row_count,), 0.80, dtype=torch.float32),
            "portfolio_daily_unified_receiver_score": torch.tensor([0, 0, 0.55, 0.52], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.90, 0.86, 0, 0], dtype=torch.float32),
        }
        wrong_outputs = {
            "portfolio_daily_allocation_weight_logit": torch.tensor([4.0, 4.0, 4.0, 4.0], dtype=torch.float32),
            "portfolio_daily_cash_reserve_logit": torch.full((row_count,), -5.0, dtype=torch.float32),
            "portfolio_daily_allocation_risk_buffer_logit": torch.full((row_count,), -5.0, dtype=torch.float32),
        }
        defensive_outputs = {
            **wrong_outputs,
            "portfolio_daily_cash_reserve_logit": torch.full((row_count,), 5.0, dtype=torch.float32),
            "portfolio_daily_allocation_weight_logit": torch.tensor([-3.0, -3.0, 0.0, 0.0], dtype=torch.float32),
        }

        wrong_terms = _portfolio_native_allocation_vector_loss(wrong_outputs, targets, return_terms=True)
        defensive_terms = _portfolio_native_allocation_vector_loss(defensive_outputs, targets, return_terms=True)

        self.assertIn("cash_timing_directional_loss", wrong_terms)
        self.assertIn("source_release_intent_loss", wrong_terms)
        self.assertGreater(
            float(wrong_terms["cash_timing_directional_loss"]),
            float(defensive_terms["cash_timing_directional_loss"]),
        )
        self.assertGreater(
            float(wrong_terms["source_release_intent_loss"]),
            float(defensive_terms["source_release_intent_loss"]),
        )


if __name__ == "__main__":
    unittest.main()
