import pandas as pd
import torch

from daily_research.continuous_policy.model_v2 import ACTION_CLASSES, DURATION_CLASSES
from daily_research.continuous_policy.run_self_optimizing_study import TrialResult


def _protocol_summary(
    metrics: dict | None = None,
    semantic: dict | None = None,
    continuity: dict | None = None,
) -> dict:
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
            "continuity_metrics": dict(continuity or {}),
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
