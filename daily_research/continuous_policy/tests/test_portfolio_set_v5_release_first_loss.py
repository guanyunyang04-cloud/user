import unittest

import pandas as pd
import torch

from daily_research.continuous_policy.model_portfolio_set_v5 import (
    PORTFOLIO_SET_V5_DFL_PG_V1_VERSION,
    PORTFOLIO_SET_V5_INTERNAL_VERSION,
    PORTFOLIO_SET_V5_R69_INTERNAL_VERSION,
    PORTFOLIO_SET_V5_R71_INTERNAL_VERSION,
    _portfolio_set_loss,
    build_portfolio_set_v5_targets,
    portfolio_set_v5_decision_diagnostics,
    project_portfolio_set_v5_cashflow_oracle,
    resolve_portfolio_set_v5_loss_profile,
)


class PortfolioSetV5ReleaseFirstLossTest(unittest.TestCase):
    def test_v48_and_alias_resolve_to_portfolio_set_release_first_loss(self) -> None:
        expected_names = {
            "alpha_result_value_budget_split_v48": PORTFOLIO_SET_V5_DFL_PG_V1_VERSION,
            "portfolio_set_release_first_decision_v1": PORTFOLIO_SET_V5_DFL_PG_V1_VERSION,
            PORTFOLIO_SET_V5_DFL_PG_V1_VERSION: PORTFOLIO_SET_V5_DFL_PG_V1_VERSION,
            PORTFOLIO_SET_V5_INTERNAL_VERSION: PORTFOLIO_SET_V5_INTERNAL_VERSION,
            PORTFOLIO_SET_V5_R69_INTERNAL_VERSION: PORTFOLIO_SET_V5_R69_INTERNAL_VERSION,
            PORTFOLIO_SET_V5_R71_INTERNAL_VERSION: PORTFOLIO_SET_V5_R71_INTERNAL_VERSION,
        }
        for name, expected_resolved in expected_names.items():
            resolved, config = resolve_portfolio_set_v5_loss_profile(name)
            weights = config["multi_objective_loss_weights"]

            self.assertEqual(resolved, expected_resolved)
            self.assertEqual(weights["action_total"], 0.0)
            self.assertEqual(weights["duration_total"], 0.0)
            self.assertGreater(weights["target_weight_closure_total"], 0.0)
            self.assertGreater(weights["source_supply_total"], 0.0)
            self.assertGreater(weights["receiver_demand_total"], 0.0)
            self.assertGreater(weights["cash_buffer_total"], 0.0)
            self.assertGreater(weights["target_delta_weight_coherence_total"], 0.0)
            self.assertGreater(weights["release_flow_balance_total"], 0.0)
            self.assertGreater(weights["underdeployment_high_cash_total"], 0.0)
            self.assertGreater(weights["intent_translation_conflict_total"], 0.0)
            self.assertGreater(weights["decision_oracle_total"], 0.0)
            self.assertGreater(weights["pg_dfl_surrogate_total"], 0.0)
            self.assertGreater(weights["constraint_violation_total"], 0.0)
            if expected_resolved == PORTFOLIO_SET_V5_R69_INTERNAL_VERSION:
                self.assertGreater(weights["value_arbitration_total"], 0.0)
                self.assertGreater(weights["cash_timing_value_total"], 0.0)
                self.assertGreater(weights["reversal_guard_total"], 0.0)
            if expected_resolved == PORTFOLIO_SET_V5_R71_INTERNAL_VERSION:
                self.assertGreater(weights["value_arbitration_total"], 0.0)
                self.assertGreater(weights["cash_timing_value_total"], 0.0)
                self.assertGreater(weights["reversal_guard_total"], 0.0)
                self.assertGreater(weights["multistage_regret_total"], 0.0)
                self.assertGreater(weights["goal_programming_quality_total"], 0.0)
                self.assertGreater(weights["crowding_penalty_total"], 0.0)

    def test_legacy_losses_are_rejected_for_v5(self) -> None:
        for legacy in ("alpha_result_value_budget_split_v47", "alpha_result_value_budget_split_v46", "teacher_imitation"):
            with self.assertRaises(ValueError):
                resolve_portfolio_set_v5_loss_profile(legacy)

    def test_torch_projection_oracle_conserves_source_receiver_cash(self) -> None:
        current = torch.tensor([[0.20, 0.00, 0.12]], dtype=torch.float32)
        source_score = torch.tensor([[0.95, 0.00, 0.10]], dtype=torch.float32)
        receiver_score = torch.tensor([[0.00, 0.98, 0.05]], dtype=torch.float32)

        oracle = project_portfolio_set_v5_cashflow_oracle(
            current_weight=current,
            source_score=source_score,
            receiver_score=receiver_score,
            cash_buffer_score=torch.zeros_like(current),
            sample_mask=torch.ones_like(current, dtype=torch.bool),
            turnover_budget=torch.tensor([0.18], dtype=torch.float32),
            risk_budget=torch.tensor([0.10], dtype=torch.float32),
        )

        self.assertGreater(float(oracle["source_supply"][0, 0]), 0.003)
        self.assertGreater(float(oracle["receiver_demand"][0, 1]), 0.003)
        self.assertLessEqual(float(oracle["target_weight"].max()), 0.24)
        self.assertGreaterEqual(float(oracle["target_weight"].min()), 0.0)
        self.assertLess(float(oracle["constraint_violation"][0]), 1.0e-6)
        cash_now = 1.0 - float(current.sum())
        cash_after = float(oracle["cash_buffer"][0])
        source = float(oracle["source_supply"].sum())
        receiver = float(oracle["receiver_demand"].sum())
        self.assertAlmostEqual(cash_after, cash_now + source - receiver, places=6)
        self.assertLessEqual(
            float(oracle["source_supply"].sum() + oracle["receiver_demand"].sum()),
            0.18 + 1.0e-6,
        )

    def test_torch_projection_oracle_seeds_cash_funded_receiver_from_flat_book(self) -> None:
        current = torch.zeros((1, 5), dtype=torch.float32)
        source_score = torch.zeros_like(current)
        receiver_score = torch.tensor([[0.10, 0.95, 0.20, 0.75, 0.05]], dtype=torch.float32)

        oracle = project_portfolio_set_v5_cashflow_oracle(
            current_weight=current,
            source_score=source_score,
            receiver_score=receiver_score,
            cash_buffer_score=torch.zeros_like(current),
            sample_mask=torch.ones_like(current, dtype=torch.bool),
            turnover_budget=torch.tensor([0.12], dtype=torch.float32),
            risk_budget=torch.tensor([0.05], dtype=torch.float32),
        )

        self.assertEqual(int((oracle["source_supply"] > 0.003).sum()), 0)
        self.assertGreaterEqual(int((oracle["receiver_demand"] > 0.003).sum()), 1)
        self.assertGreater(float(oracle["target_delta"][0, 1]), 0.003)
        self.assertLessEqual(float((oracle["target_weight"] - current).abs().sum()), 0.12 + 1.0e-6)

    def test_torch_projection_oracle_keeps_source_receiver_roles_disjoint(self) -> None:
        current = torch.tensor([[0.18, 0.00, 0.10, 0.00, 0.00]], dtype=torch.float32)
        source_score = torch.tensor([[0.95, 0.05, 0.80, 0.00, 0.00]], dtype=torch.float32)
        receiver_score = torch.tensor([[0.99, 0.97, 0.90, 0.88, 0.20]], dtype=torch.float32)

        oracle = project_portfolio_set_v5_cashflow_oracle(
            current_weight=current,
            source_score=source_score,
            receiver_score=receiver_score,
            cash_buffer_score=torch.zeros_like(current),
            sample_mask=torch.ones_like(current, dtype=torch.bool),
            turnover_budget=torch.tensor([0.14], dtype=torch.float32),
            risk_budget=torch.tensor([0.10], dtype=torch.float32),
        )

        source_mask = oracle["source_supply"] > 0.003
        receiver_mask = oracle["receiver_demand"] > 0.003
        self.assertGreaterEqual(int(source_mask.sum()), 1)
        self.assertGreaterEqual(int(receiver_mask.sum()), 1)
        self.assertEqual(int((source_mask & receiver_mask).sum()), 0)
        self.assertLessEqual(float((oracle["target_weight"] - current).abs().sum()), 0.14 + 1.0e-6)

    def test_r69_oracle_rejects_wrong_side_source_when_receiver_is_good(self) -> None:
        current = torch.tensor([[0.18, 0.18, 0.00]], dtype=torch.float32)
        source_score = torch.tensor([[0.90, 0.90, 0.00]], dtype=torch.float32)
        receiver_score = torch.tensor([[0.00, 0.00, 0.95]], dtype=torch.float32)

        oracle = project_portfolio_set_v5_cashflow_oracle(
            current_weight=current,
            source_score=source_score,
            receiver_score=receiver_score,
            cash_buffer_score=torch.zeros_like(current),
            deploy_value=torch.tensor([[0.0, 0.0, 0.95]], dtype=torch.float32),
            release_value=torch.tensor([[0.90, 0.90, 0.0]], dtype=torch.float32),
            defense_value=torch.zeros_like(current),
            cash_timing_value=torch.zeros_like(current),
            source_opportunity_cost=torch.tensor([[0.85, 0.05, 0.0]], dtype=torch.float32),
            receiver_source_spread_value=torch.tensor([[0.0, 0.90, 0.90]], dtype=torch.float32),
            reversal_risk_penalty=torch.zeros_like(current),
            source_wrong_side_sell_penalty=torch.tensor([[0.95, 0.0, 0.0]], dtype=torch.float32),
            sample_mask=torch.ones_like(current, dtype=torch.bool),
            turnover_budget=torch.tensor([0.12], dtype=torch.float32),
            risk_budget=torch.tensor([0.05], dtype=torch.float32),
        )

        self.assertLess(float(oracle["source_supply"][0, 0]), 0.003)
        self.assertGreater(float(oracle["source_supply"][0, 1]), 0.003)
        self.assertGreater(float(oracle["receiver_demand"][0, 2]), 0.003)
        self.assertLess(float(oracle["constraint_violation"][0]), 1.0e-6)

    def test_r69_oracle_prefers_cash_on_risk_off_without_spread(self) -> None:
        current = torch.tensor([[0.16, 0.00, 0.00]], dtype=torch.float32)
        source_score = torch.tensor([[0.80, 0.00, 0.00]], dtype=torch.float32)
        receiver_score = torch.tensor([[0.00, 0.65, 0.60]], dtype=torch.float32)

        oracle = project_portfolio_set_v5_cashflow_oracle(
            current_weight=current,
            source_score=source_score,
            receiver_score=receiver_score,
            cash_buffer_score=torch.ones_like(current) * 0.90,
            deploy_value=torch.tensor([[0.0, 0.15, 0.12]], dtype=torch.float32),
            release_value=torch.tensor([[0.85, 0.0, 0.0]], dtype=torch.float32),
            defense_value=torch.ones_like(current) * 0.95,
            cash_timing_value=torch.ones_like(current) * 0.95,
            source_opportunity_cost=torch.zeros_like(current),
            receiver_source_spread_value=torch.zeros_like(current),
            reversal_risk_penalty=torch.zeros_like(current),
            source_wrong_side_sell_penalty=torch.zeros_like(current),
            sample_mask=torch.ones_like(current, dtype=torch.bool),
            turnover_budget=torch.tensor([0.16], dtype=torch.float32),
            risk_budget=torch.tensor([0.95], dtype=torch.float32),
        )

        self.assertGreater(float(oracle["source_supply"][0, 0]), 0.003)
        self.assertLess(float(oracle["receiver_demand"].sum()), 0.003)
        self.assertGreater(float(oracle["cash_buffer"][0]), float(1.0 - current.sum()))

    def test_r69_oracle_defense_allows_source_even_without_receiver(self) -> None:
        current = torch.tensor([[0.20, 0.10, 0.00]], dtype=torch.float32)
        source_score = torch.tensor([[0.80, 0.20, 0.00]], dtype=torch.float32)
        receiver_score = torch.zeros_like(current)

        oracle = project_portfolio_set_v5_cashflow_oracle(
            current_weight=current,
            source_score=source_score,
            receiver_score=receiver_score,
            cash_buffer_score=torch.ones_like(current) * 0.80,
            release_value=torch.tensor([[0.90, 0.10, 0.0]], dtype=torch.float32),
            defense_value=torch.ones_like(current) * 0.90,
            cash_timing_value=torch.ones_like(current) * 0.80,
            source_opportunity_cost=torch.tensor([[0.05, 0.60, 0.0]], dtype=torch.float32),
            sample_mask=torch.ones_like(current, dtype=torch.bool),
            turnover_budget=torch.tensor([0.10], dtype=torch.float32),
            risk_budget=torch.tensor([0.90], dtype=torch.float32),
        )

        self.assertGreater(float(oracle["source_supply"][0, 0]), 0.003)
        self.assertLess(float(oracle["source_supply"][0, 1]), 0.003)
        self.assertEqual(int((oracle["receiver_demand"] > 0.003).sum()), 0)
        self.assertLess(float(oracle["target_weight"].sum()), float(current.sum()))

    def test_r69_oracle_reversal_penalty_blocks_reduce_trap(self) -> None:
        current = torch.tensor([[0.18, 0.18, 0.00]], dtype=torch.float32)
        base_kwargs = dict(
            current_weight=current,
            source_score=torch.tensor([[0.90, 0.90, 0.00]], dtype=torch.float32),
            receiver_score=torch.tensor([[0.00, 0.00, 0.85]], dtype=torch.float32),
            cash_buffer_score=torch.zeros_like(current),
            deploy_value=torch.tensor([[0.0, 0.0, 0.90]], dtype=torch.float32),
            release_value=torch.tensor([[0.90, 0.90, 0.0]], dtype=torch.float32),
            defense_value=torch.zeros_like(current),
            cash_timing_value=torch.zeros_like(current),
            source_opportunity_cost=torch.zeros_like(current),
            receiver_source_spread_value=torch.tensor([[0.80, 0.80, 0.80]], dtype=torch.float32),
            source_wrong_side_sell_penalty=torch.zeros_like(current),
            sample_mask=torch.ones_like(current, dtype=torch.bool),
            turnover_budget=torch.tensor([0.12], dtype=torch.float32),
            risk_budget=torch.tensor([0.05], dtype=torch.float32),
        )
        guarded = project_portfolio_set_v5_cashflow_oracle(
            **base_kwargs,
            reversal_risk_penalty=torch.tensor([[0.95, 0.00, 0.0]], dtype=torch.float32),
        )
        clean = project_portfolio_set_v5_cashflow_oracle(
            **base_kwargs,
            reversal_risk_penalty=torch.zeros_like(current),
        )

        self.assertLess(float(guarded["source_supply"][0, 0]), float(clean["source_supply"][0, 0]))
        self.assertGreater(float(guarded["source_supply"][0, 1]), 0.003)

    def test_r71_oracle_blocks_source_that_rebounds_after_sell(self) -> None:
        current = torch.tensor([[0.18, 0.18, 0.00]], dtype=torch.float32)
        base_kwargs = dict(
            current_weight=current,
            source_score=torch.tensor([[0.92, 0.92, 0.00]], dtype=torch.float32),
            receiver_score=torch.tensor([[0.00, 0.00, 0.90]], dtype=torch.float32),
            cash_buffer_score=torch.zeros_like(current),
            deploy_value=torch.tensor([[0.0, 0.0, 0.92]], dtype=torch.float32),
            release_value=torch.tensor([[0.92, 0.92, 0.0]], dtype=torch.float32),
            defense_value=torch.zeros_like(current),
            cash_timing_value=torch.zeros_like(current),
            source_opportunity_cost=torch.zeros_like(current),
            receiver_source_spread_value=torch.tensor([[0.80, 0.80, 0.80]], dtype=torch.float32),
            reversal_risk_penalty=torch.zeros_like(current),
            source_wrong_side_sell_penalty=torch.zeros_like(current),
            sample_mask=torch.ones_like(current, dtype=torch.bool),
            turnover_budget=torch.tensor([0.12], dtype=torch.float32),
            risk_budget=torch.tensor([0.05], dtype=torch.float32),
        )
        guarded = project_portfolio_set_v5_cashflow_oracle(
            **base_kwargs,
            source_hold_regret_3d=torch.tensor([[0.95, 0.00, 0.0]], dtype=torch.float32),
            source_hold_regret_5d=torch.tensor([[0.95, 0.00, 0.0]], dtype=torch.float32),
            rotation_spread_regret_5d=torch.zeros_like(current),
            reversal_action_regret_3d=torch.zeros_like(current),
            crowding_penalty=torch.zeros_like(current),
        )
        clean = project_portfolio_set_v5_cashflow_oracle(
            **base_kwargs,
            source_hold_regret_3d=torch.zeros_like(current),
            source_hold_regret_5d=torch.zeros_like(current),
            rotation_spread_regret_5d=torch.zeros_like(current),
            reversal_action_regret_3d=torch.zeros_like(current),
            crowding_penalty=torch.zeros_like(current),
        )

        self.assertLess(float(guarded["source_supply"][0, 0]), float(clean["source_supply"][0, 0]))
        self.assertGreater(float(guarded["source_supply"][0, 1]), 0.003)
        self.assertGreater(float(guarded["receiver_demand"][0, 2]), 0.003)
        self.assertLess(float(guarded["constraint_violation"][0]), 1.0e-6)

    def test_r71_oracle_blocks_receiver_when_source_beats_receiver(self) -> None:
        current = torch.tensor([[0.18, 0.00, 0.00]], dtype=torch.float32)
        base_kwargs = dict(
            current_weight=current,
            source_score=torch.tensor([[0.90, 0.00, 0.00]], dtype=torch.float32),
            receiver_score=torch.tensor([[0.00, 0.90, 0.80]], dtype=torch.float32),
            cash_buffer_score=torch.zeros_like(current),
            deploy_value=torch.tensor([[0.0, 0.90, 0.80]], dtype=torch.float32),
            release_value=torch.tensor([[0.90, 0.0, 0.0]], dtype=torch.float32),
            defense_value=torch.zeros_like(current),
            cash_timing_value=torch.zeros_like(current),
            source_opportunity_cost=torch.zeros_like(current),
            receiver_source_spread_value=torch.tensor([[0.0, 0.80, 0.70]], dtype=torch.float32),
            sample_mask=torch.ones_like(current, dtype=torch.bool),
            turnover_budget=torch.tensor([0.12], dtype=torch.float32),
            risk_budget=torch.tensor([0.05], dtype=torch.float32),
        )
        guarded = project_portfolio_set_v5_cashflow_oracle(
            **base_kwargs,
            receiver_deploy_regret_3d=torch.tensor([[0.0, 0.95, 0.95]], dtype=torch.float32),
            receiver_deploy_regret_5d=torch.tensor([[0.0, 0.95, 0.95]], dtype=torch.float32),
            rotation_spread_regret_5d=torch.tensor([[0.0, 0.95, 0.95]], dtype=torch.float32),
        )
        clean = project_portfolio_set_v5_cashflow_oracle(
            **base_kwargs,
            receiver_deploy_regret_3d=torch.zeros_like(current),
            receiver_deploy_regret_5d=torch.zeros_like(current),
            rotation_spread_regret_5d=torch.zeros_like(current),
        )

        self.assertLess(float(guarded["receiver_demand"].sum()), float(clean["receiver_demand"].sum()))
        self.assertLess(float(guarded["constraint_violation"][0]), 1.0e-6)

    def test_r71_oracle_prefers_cash_on_multistage_risk_off(self) -> None:
        current = torch.tensor([[0.16, 0.00, 0.00]], dtype=torch.float32)
        oracle = project_portfolio_set_v5_cashflow_oracle(
            current_weight=current,
            source_score=torch.tensor([[0.80, 0.00, 0.00]], dtype=torch.float32),
            receiver_score=torch.tensor([[0.00, 0.75, 0.65]], dtype=torch.float32),
            cash_buffer_score=torch.ones_like(current) * 0.80,
            deploy_value=torch.tensor([[0.0, 0.20, 0.18]], dtype=torch.float32),
            release_value=torch.tensor([[0.90, 0.0, 0.0]], dtype=torch.float32),
            defense_value=torch.ones_like(current) * 0.80,
            cash_timing_value=torch.ones_like(current) * 0.80,
            cash_defense_regret_1d=torch.ones_like(current) * 0.95,
            cash_defense_regret_3d=torch.ones_like(current) * 0.95,
            receiver_deploy_regret_3d=torch.ones_like(current) * 0.80,
            receiver_deploy_regret_5d=torch.ones_like(current) * 0.80,
            source_opportunity_cost=torch.zeros_like(current),
            receiver_source_spread_value=torch.zeros_like(current),
            sample_mask=torch.ones_like(current, dtype=torch.bool),
            turnover_budget=torch.tensor([0.16], dtype=torch.float32),
            risk_budget=torch.tensor([0.95], dtype=torch.float32),
        )

        self.assertGreater(float(oracle["source_supply"][0, 0]), 0.003)
        self.assertLess(float(oracle["receiver_demand"].sum()), 0.003)
        self.assertGreater(float(oracle["cash_buffer"][0]), float(1.0 - current.sum()))
        self.assertLess(float(oracle["constraint_violation"][0]), 1.0e-6)

    def test_r71_target_builder_seeds_positive_spread_rotation_receiver(self) -> None:
        stocks = [f"SRC{i}" for i in range(4)] + [f"RCV{i}" for i in range(12)]
        frame = pd.DataFrame(
            {
                "date": ["2026-05-14"] * len(stocks),
                "stock": stocks,
                "current_weight": [0.23, 0.23, 0.22, 0.18] + [0.00] * 12,
                "action_label": ["reduce", "hold", "hold", "hold"] + ["hold"] * 12,
                "portfolio_daily_source_score": [0.90, 0.72, 0.58, 0.46] + [0.00] * 12,
                "portfolio_daily_source_release_quality": [0.90, 0.72, 0.58, 0.46] + [0.00] * 12,
                "release_value_target": [0.90, 0.72, 0.58, 0.46] + [0.00] * 12,
                "portfolio_daily_receiver_score": [0.00] * 4 + [0.02] * 12,
                "deploy_value_target": [0.00] * 4 + [0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76, 0.74],
                "portfolio_daily_receiver_source_spread_reward": [0.00] * 4 + [0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76],
                "portfolio_daily_receiver_forward_excess_5d": [0.00] * 4 + [0.10, 0.095, 0.09, 0.085, 0.08, 0.075, 0.07, 0.065, 0.06, 0.055, 0.05, 0.045],
                "portfolio_daily_source_forward_excess_5d": [-0.03, -0.02, -0.02, -0.01] + [0.00] * 12,
                "forward_excess_1d": [-0.01, -0.01, -0.01, -0.01] + [0.03] * 12,
                "forward_excess_3d": [-0.02, -0.02, -0.01, -0.01] + [0.05] * 12,
                "forward_excess_5d": [-0.03, -0.02, -0.02, -0.01] + [0.08] * 12,
                "cash_defense_value": [0.30] * len(stocks),
                "defense_value_target": [0.30] * len(stocks),
                "portfolio_daily_cash_score": [0.30] * len(stocks),
                "portfolio_daily_source_positive_forward_penalty": [0.00] * len(stocks),
                "portfolio_daily_source_opportunity_cost": [0.00] * len(stocks),
                "reduce_reversal_pressure": [0.00] * len(stocks),
                "portfolio_daily_crowding_penalty": [0.00] * len(stocks),
            }
        )

        base = build_portfolio_set_v5_targets(frame, loss_profile=PORTFOLIO_SET_V5_R69_INTERNAL_VERSION)
        r71 = build_portfolio_set_v5_targets(frame, loss_profile=PORTFOLIO_SET_V5_R71_INTERNAL_VERSION)

        self.assertGreater(float(r71["r71_receiver_coverage_floor"].sum()), 0.003)
        self.assertGreaterEqual(int((r71["receiver_demand"] > 0.003).sum()), int((base["receiver_demand"] > 0.003).sum()))
        self.assertGreater(int((r71["receiver_demand"] > 0.003).sum()), 0)
        self.assertGreater(int((r71["source_supply"] > 0.003).sum()), 0)
        self.assertLess(float(r71["constraint_violation"].mean()), 1.0e-6)
        self.assertGreaterEqual(float(r71["source_supply"].sum()), float(r71["r71_receiver_coverage_floor"].sum()))
        self.assertLessEqual(
            float(r71["source_supply"].sum() + r71["receiver_demand"].sum()),
            float(r71["turnover_budget"].iloc[0]) + 1.0e-6,
        )

    def test_pg_dfl_surrogate_prefers_oracle_aligned_logits(self) -> None:
        weights = resolve_portfolio_set_v5_loss_profile(PORTFOLIO_SET_V5_R69_INTERNAL_VERSION)[1]["multi_objective_loss_weights"]
        current = torch.tensor([[0.20, 0.00]], dtype=torch.float32)
        decision_target = torch.tensor(
            [[[0.08, 0.00, 0.50, 0.12, 0.18, 0.10, 0.00, 0.08], [0.00, 0.08, 0.50, 0.08, 0.18, 0.10, 0.00, 0.08]]],
            dtype=torch.float32,
        )
        target_y = torch.tensor(
            [[[0.12, -0.08, 1.00, 0.00, 0.50, 1.00, 1.00, 0.00], [0.08, 0.08, 0.00, 1.00, 0.50, 0.00, 0.00, 0.00]]],
            dtype=torch.float32,
        )
        batch = {
            "target_y": target_y,
            "decision_target_y": decision_target,
            "sample_mask": torch.ones((1, 2), dtype=torch.bool),
            "current_weight": current,
        }
        aligned = torch.tensor([[[0.0, -0.5, 3.0, -3.0, 0.0, 3.0, 3.0, -3.0], [-1.0, 0.5, -3.0, 3.0, 0.0, -3.0, -3.0, -3.0]]], dtype=torch.float32)
        inverted = torch.tensor([[[0.0, 0.5, -3.0, 3.0, 0.0, -3.0, -3.0, -3.0], [-1.0, -0.5, 3.0, -3.0, 0.0, 3.0, 3.0, -3.0]]], dtype=torch.float32)

        aligned_loss = float(_portfolio_set_loss(aligned, batch, weights))
        inverted_loss = float(_portfolio_set_loss(inverted, batch, weights))
        diagnostics = portfolio_set_v5_decision_diagnostics(aligned, batch, weights)

        self.assertLess(aligned_loss, inverted_loss)
        self.assertGreater(diagnostics["decision_oracle_value_mean"], 0.0)
        self.assertGreater(diagnostics["release_flow_source_target_count"], 0.0)
        self.assertGreater(diagnostics["release_flow_receiver_target_count"], 0.0)

    def test_base_loss_ignores_r69_extension_columns(self) -> None:
        weights = resolve_portfolio_set_v5_loss_profile(PORTFOLIO_SET_V5_INTERNAL_VERSION)[1]["multi_objective_loss_weights"]
        current = torch.tensor([[0.20, 0.00]], dtype=torch.float32)
        target_y = torch.tensor(
            [[[0.12, -0.08, 1.00, 0.00, 0.50, 1.00, 1.00, 0.00], [0.08, 0.08, 0.00, 1.00, 0.50, 0.00, 0.00, 0.00]]],
            dtype=torch.float32,
        )
        base_decision = torch.tensor(
            [[[0.08, 0.00, 0.50, 0.12, 0.18, 0.10, 0.00, 0.08, 0, 0, 0, 0, 0, 0, 0, 0],
              [0.00, 0.08, 0.50, 0.08, 0.18, 0.10, 0.00, 0.08, 0, 0, 0, 0, 0, 0, 0, 0]]],
            dtype=torch.float32,
        )
        perturbed_decision = base_decision.clone()
        perturbed_decision[..., 8:] = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        raw = torch.tensor([[[0.0, -0.5, 3.0, -3.0, 0.0, 3.0, 3.0, -3.0], [-1.0, 0.5, -3.0, 3.0, 0.0, -3.0, -3.0, -3.0]]], dtype=torch.float32)
        common = {
            "target_y": target_y,
            "sample_mask": torch.ones((1, 2), dtype=torch.bool),
            "current_weight": current,
        }

        loss_base = float(_portfolio_set_loss(raw, {**common, "decision_target_y": base_decision}, weights))
        loss_perturbed = float(_portfolio_set_loss(raw, {**common, "decision_target_y": perturbed_decision}, weights))

        self.assertAlmostEqual(loss_base, loss_perturbed, places=7)

    def test_r69_loss_ignores_r71_extension_columns(self) -> None:
        weights = resolve_portfolio_set_v5_loss_profile(PORTFOLIO_SET_V5_R69_INTERNAL_VERSION)[1]["multi_objective_loss_weights"]
        current = torch.tensor([[0.20, 0.00]], dtype=torch.float32)
        target_y = torch.tensor(
            [[[0.12, -0.08, 1.00, 0.00, 0.50, 1.00, 1.00, 0.00], [0.08, 0.08, 0.00, 1.00, 0.50, 0.00, 0.00, 0.00]]],
            dtype=torch.float32,
        )
        base_decision = torch.zeros((1, 2, 25), dtype=torch.float32)
        base_decision[..., :16] = torch.tensor(
            [[[0.08, 0.00, 0.50, 0.12, 0.18, 0.10, 0.00, 0.08, 0.0, 0.8, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0],
              [0.00, 0.08, 0.50, 0.08, 0.18, 0.10, 0.00, 0.08, 0.8, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        perturbed_decision = base_decision.clone()
        perturbed_decision[..., 16:] = 1.0
        raw = torch.tensor([[[0.0, -0.5, 3.0, -3.0, 0.0, 3.0, 3.0, -3.0], [-1.0, 0.5, -3.0, 3.0, 0.0, -3.0, -3.0, -3.0]]], dtype=torch.float32)
        common = {
            "target_y": target_y,
            "sample_mask": torch.ones((1, 2), dtype=torch.bool),
            "current_weight": current,
        }

        loss_base = float(_portfolio_set_loss(raw, {**common, "decision_target_y": base_decision}, weights))
        loss_perturbed = float(_portfolio_set_loss(raw, {**common, "decision_target_y": perturbed_decision}, weights))

        self.assertAlmostEqual(loss_base, loss_perturbed, places=7)

    def test_r71_loss_is_sensitive_to_multistage_regret_columns(self) -> None:
        weights = resolve_portfolio_set_v5_loss_profile(PORTFOLIO_SET_V5_R71_INTERNAL_VERSION)[1]["multi_objective_loss_weights"]
        current = torch.tensor([[0.20, 0.00]], dtype=torch.float32)
        target_y = torch.tensor(
            [[[0.12, -0.08, 1.00, 0.00, 0.50, 1.00, 1.00, 0.00], [0.08, 0.08, 0.00, 1.00, 0.50, 0.00, 0.00, 0.00]]],
            dtype=torch.float32,
        )
        decision = torch.zeros((1, 2, 25), dtype=torch.float32)
        decision[..., :16] = torch.tensor(
            [[[0.08, 0.00, 0.50, 0.12, 0.18, 0.10, 0.00, 0.08, 0.0, 0.8, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0],
              [0.00, 0.08, 0.50, 0.08, 0.18, 0.10, 0.00, 0.08, 0.8, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        high_regret = decision.clone()
        high_regret[..., 16:] = 1.0
        raw = torch.tensor([[[0.0, -0.5, 3.0, -3.0, 0.0, 3.0, 3.0, -3.0], [-1.0, 0.5, -3.0, 3.0, 0.0, -3.0, -3.0, -3.0]]], dtype=torch.float32)
        common = {
            "target_y": target_y,
            "sample_mask": torch.ones((1, 2), dtype=torch.bool),
            "current_weight": current,
        }

        clean_loss = float(_portfolio_set_loss(raw, {**common, "decision_target_y": decision}, weights))
        regret_loss = float(_portfolio_set_loss(raw, {**common, "decision_target_y": high_regret}, weights))

        self.assertGreater(regret_loss, clean_loss)


if __name__ == "__main__":
    unittest.main()
