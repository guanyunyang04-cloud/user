from __future__ import annotations

import math

import numpy as np
import pandas as pd

from daily_research.path_policy import qdp_v2_sequence_path_training as training
from daily_research.path_policy import seq100_structured_global_intraday as study


def test_registered_models_and_dynamic_v3_scope() -> None:
    assert list(study.V2_SPECS) == ["l35v2", "g35v2", "l43v2", "g43v2"]
    assert study.V2_SPECS["g43v2"].input_dim == 43
    assert study.V2_SPECS["g43v2"].rank_training_profile == "global_tail_512"
    global_only = study._v3_specs("g35v2")
    assert set(global_only) == {"g35v3", "g43v3"}
    with_local = study._v3_specs("l43v2")
    assert set(with_local) == {"g35v3", "g43v3", "l43v3"}


def test_account_grid_contains_top2_and_treats_d7_normally() -> None:
    jobs = study._account_jobs("g43v3")
    assert len(jobs) == 2_074
    assert {job.top_k for job in jobs} == {1, 2, 3}
    assert {job.slots for job in jobs if job.top_k == 2} == {2, 4, 6, 12, 24, 48}
    d7 = [job for job in jobs if job.policy_name == "fixed_d7"]
    d8 = [job for job in jobs if job.policy_name == "fixed_d8"]
    assert len(d7) == len(d8)
    assert all(job.policy_kind == "fixed" and job.fixed_day == 7 for job in d7)


def test_capital_speed_v3_prefers_higher_unit_time_growth() -> None:
    path = np.zeros((1, 60, 4), dtype=np.float32)
    path[0, 1, 3] = 0.04
    path[0, 59, 3] = 0.20
    v2 = training._derive_path_summary_numpy(
        path, path_value_semantic=training.PATH_VALUE_SEMANTIC_V2
    )
    v3 = training._derive_path_summary_numpy(
        path,
        path_value_semantic=training.PATH_VALUE_SEMANTIC_CAPITAL_SPEED_V3,
        path_value_growth_multiplier=np.ones((1, 60), dtype=np.float32),
    )
    assert int(v2[0, 8]) == 60
    assert int(v3[0, 8]) == 2
    assert math.isclose(float(v3[0, -1]), math.log1p(0.04) / 2.0, rel_tol=1e-6)


def test_v3_cost_multiplier_uses_stamp_tax_schedule() -> None:
    manifest = {
        "date_values": ["2023-08-25", "2023-08-28", "2023-08-29"],
        "execution_cost_contract": {
            "lot_size": 100,
            "commission_bps": 3.0,
            "minimum_commission_cny": 5.0,
            "transfer_fee_bps": 0.1,
            "slippage_bps": 7.0,
            "stress_slippage_multiplier": 2.0,
            "stamp_tax_schedule": [
                {"effective_date": "1900-01-01", "stamp_tax_bps": 10.0},
                {"effective_date": "2023-08-28", "stamp_tax_bps": 5.0},
            ],
        },
    }
    multiplier = training._v3_growth_multiplier_for_dates(
        manifest, np.asarray([0], dtype=np.int64), 2
    )
    assert multiplier.shape == (1, 2)
    assert multiplier[0, 1] == multiplier[0, 0]
    before = training._v3_growth_multiplier_for_dates(
        {**manifest, "date_values": ["2023-08-24", "2023-08-25"]},
        np.asarray([0], dtype=np.int64), 1,
    )
    assert multiplier[0, 0] > before[0, 0]


def test_unit_time_alpha_is_mean_by_signal_date() -> None:
    value = study.unit_time_alpha(
        [0.02, 0.04], [2.0, 4.0], [0.01, 0.02], [2.0, 4.0]
    )
    expected = np.mean(
        [
            math.log1p(0.02) / 2.0 - math.log1p(0.01) / 2.0,
            math.log1p(0.04) / 4.0 - math.log1p(0.02) / 4.0,
        ]
    ) * 252.0
    assert math.isclose(value, expected, rel_tol=1e-12)


def test_selection_uses_double_slippage_and_fixed_fallback() -> None:
    rows = []
    for model, fixed, own in (("l35v2", 0.20, 0.10), ("g35v2", 0.15, 0.25)):
        for scenario in ("base", "double_slippage"):
            for policy, kind, growth in (
                ("fixed_d9", "fixed", fixed),
                ("model_plan", "model_plan", own),
                ("rolling_path", "rolling", own - 0.01),
            ):
                rows.append(
                    {
                        "model_id": model, "top_k": 1, "slot_count": 1,
                        "policy_name": policy, "policy_kind": kind,
                        "fixed_day": 9 if kind == "fixed" else np.nan,
                        "cost_scenario": scenario,
                        "annualized_log_growth": growth,
                        "signal_period_total_return": growth,
                        "signal_period_cagr_trading_days": growth,
                        "signal_period_maximum_drawdown": -0.1,
                        "worst_calendar_year_return": 0.01,
                        "closed_trade_count": 10,
                        "winning_trade_rate": 0.5,
                        "payoff_ratio": 1.2,
                        "profit_factor": 1.2,
                        "mean_trade_expectancy": 0.01,
                        "mean_occupied_sessions": 5.0,
                        "mean_signal_capital_utilization": 0.8,
                        "skipped_no_slot_signal_count": 0,
                        "net_pnl_per_deployed_capital_session": 0.001,
                    }
                )
    result = study._select_models(pd.DataFrame(rows))
    assert result["winner_model_id"] == "g35v2"
    local = next(row for row in result["frontier"] if row["model_id"] == "l35v2")
    assert local["selected_execution"] == "fixed_exit_fallback"
    assert result["d7_special_weight"] is False
