from __future__ import annotations

import numpy as np

from daily_research.path_policy import seq100_exact_value_growth_path_exit as study


def test_shape_labels_follow_frozen_precedence() -> None:
    labels, primary = study._shape_labels(
        {
            "return_d5": 0.02,
            "return_d20": 0.03,
            "return_d60": 0.01,
            "mfe_d20": 0.10,
            "mae_d10": -0.06,
            "peak_day_d60": 45,
        }
    )
    assert labels["early_spike_fade"]
    assert labels["dip_then_recover"]
    assert labels["persistent_trend"]
    assert primary == "early_spike_fade"


def test_buy_then_sell_t_rule_is_chronological() -> None:
    result = study._t_rule_day(
        [100.0, 98.0, 99.0, 100.0],
        [99.0, 97.0, 99.0, 101.0],
        {
            "direction": "buy_then_sell_old_inventory",
            "trigger_from_day_open": 0.01,
            "paired_move": 0.01,
        },
        cost=0.006,
        tranche=0.20,
    )
    assert result["first_fill_bar"] == 1
    assert result["second_fill_bar"] == 3
    assert result["paired_completed"]
    assert np.isclose(result["net_tranche_return"], 100.0 / 98.0 - 1.006)


def test_sell_then_buy_t_rule_uses_final_close_fallback() -> None:
    result = study._t_rule_day(
        [100.0, 102.0, 103.0, 104.0],
        [101.5, 102.5, 103.5, 105.0],
        {
            "direction": "sell_old_then_buy_back",
            "trigger_from_day_open": 0.01,
            "paired_move": 0.01,
        },
        cost=0.006,
        tranche=0.20,
    )
    assert result["first_fill_bar"] == 1
    assert result["second_fill_bar"] == 3
    assert not result["paired_completed"]
    assert np.isclose(result["net_tranche_return"], 102.0 / 105.0 - 1.006)


def test_bh_qvalues_are_monotone_in_rank() -> None:
    p_values = np.asarray([0.03, 0.001, 0.02, np.nan])
    q_values = study._bh_qvalues(p_values)
    order = np.argsort(p_values[:3])
    assert np.all(np.diff(q_values[:3][order]) >= 0.0)
    assert np.isnan(q_values[3])
