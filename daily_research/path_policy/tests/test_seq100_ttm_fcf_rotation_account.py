from __future__ import annotations

import pandas as pd

from daily_research.path_policy import seq100_ttm_fcf_rotation_account as rotation


def test_contract_freezes_natural_phase_and_exact_execution() -> None:
    study, _ = rotation.load_study()

    assert study["selection"]["natural_phase"] == 0
    assert study["selection"]["monthly_order_count"] == 12
    assert study["account"]["round_lots_and_minimum_commission_modeled"]
    assert study["account"]["cost_scenario"] == "double_slippage"
    assert study["decision_boundary"]["account_optimization_performed"] is False


def test_rotation_mask_advances_one_rank_residue_per_month() -> None:
    ranks = pd.Series(list(range(1, 9)) * 2)
    months = pd.Series([0] * 8 + [1] * 8)

    keep = rotation._rotation_mask(ranks, months, modulus=4, phase=0)

    assert ranks.loc[keep].tolist() == [1, 5, 2, 6]


def test_gate_requires_return_drawdown_and_enough_positive_years() -> None:
    metric = {
        "liquidated_total_return": 0.2,
        "signal_period_cagr_trading_days": 0.08,
        "signal_period_maximum_drawdown": -0.2,
        "ruined": False,
    }
    annual = [{"net_return": 0.1}, {"net_return": -0.02}, {"net_return": 0.03}]

    passed = rotation._gate(
        metric,
        annual,
        minimum_positive_years=2,
        maximum_drawdown=0.25,
    )
    failed = rotation._gate(
        metric,
        annual,
        minimum_positive_years=3,
        maximum_drawdown=0.25,
    )

    assert passed["passed"] is True
    assert failed["passed"] is False
