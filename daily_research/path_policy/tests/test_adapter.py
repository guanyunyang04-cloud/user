from __future__ import annotations

import pandas as pd

from daily_research.path_policy.adapter import build_path_policy_frame, project_target_weights


def test_projection_respects_long_only_cap_sum_mask_and_turnover() -> None:
    raw = pd.Series({"AAA": 0.50, "BBB": 0.30, "CCC": -0.10, "DDD": 0.20})
    current = pd.Series({"AAA": 0.05, "BBB": 0.05, "CCC": 0.10, "DDD": 0.00})
    tradable = pd.Series({"AAA": True, "BBB": True, "CCC": False, "DDD": True})

    target, diagnostics = project_target_weights(
        raw,
        current_weight=current,
        tradable_mask=tradable,
        max_position_weight=0.20,
        max_gross_exposure=0.50,
        max_positions=3,
        turnover_budget=0.40,
    )

    assert float(target.min()) >= 0.0
    assert float(target.max()) <= 0.20 + 1.0e-9
    assert float(target.sum()) <= 0.50 + 1.0e-9
    assert abs(float(target["CCC"]) - 0.10) < 1.0e-9
    assert float((target - current).abs().sum()) <= 0.40 + 1.0e-9
    assert diagnostics.masked_count >= 1


def test_path_policy_frame_derives_source_receiver_from_target_delta_only() -> None:
    base = pd.DataFrame({"stock": ["AAA", "BBB"], "current_weight": [0.15, 0.05]})
    target = pd.Series({"AAA": 0.05, "BBB": 0.20})
    policy, global_targets = build_path_policy_frame(
        base,
        raw_target_weight=target,
        current_weight=pd.Series({"AAA": 0.15, "BBB": 0.05}),
        tradable_mask=pd.Series({"AAA": True, "BBB": True}),
        max_position_weight=0.20,
        max_gross_exposure=0.50,
        max_positions=2,
        turnover_budget=1.00,
    )
    policy = policy.set_index("stock")

    assert policy.loc["AAA", "portfolio_daily_target_delta_intent"] < 0.0
    assert policy.loc["AAA", "path_policy_source_supply_derived"] > 0.0
    assert policy.loc["AAA", "path_policy_receiver_demand_derived"] == 0.0
    assert policy.loc["BBB", "path_policy_receiver_demand_derived"] > 0.0
    assert policy.loc["BBB", "path_policy_source_supply_derived"] == 0.0
    assert global_targets["decision_core_version"] == "alpha_path20_neural_policy_v1"
    assert global_targets["turnover_budget"] == 1.0
