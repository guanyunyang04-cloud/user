from __future__ import annotations

import numpy as np

from daily_research.path_policy import seq100_daily_action_value as action


def test_action_study_has_no_fixed_total_holding_horizon() -> None:
    study = action.load_study()
    assert study["source"]["quality_pool_name"] == "quality_liquidity_pit"
    assert study["counterfactual"]["fixed_total_holding_horizon_used"] is False
    assert study["state"]["threshold_selection_performed"] is False
    assert study["replacement"]["selection_uses_future_fill_state"] is False


def test_age_bins_are_log_duration_bins() -> None:
    assert np.array_equal(
        action.age_bin(np.asarray([0, 1, 2, 3, 4, 7, 8, 15, 16])),
        np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int8),
    )


def test_position_code_uses_only_sign_and_order_coordinates() -> None:
    assert action.position_code(11.0, 10.0, 11.0, 9.0) == 7
    assert action.position_code(9.0, 10.0, 12.0, 9.0) == 4
    assert action.position_code(float("nan"), 10.0, 11.0, 9.0) == -1


def test_replacement_excludes_current_and_concurrent_symbols() -> None:
    chosen = action._choose_replacement(
        date_idx=10,
        current_symbol_idx=2,
        rankings={10: (2, 4, 7)},
        active_symbols={10: {4}},
    )
    assert chosen == (7, 3)


def test_action_stats_shrinks_cluster_estimate_to_prior() -> None:
    stats = action.ActionStats(group_count=2, cluster_count=2)
    stats.update(
        np.asarray([0, 0, 0]),
        np.asarray([0, 0, 1]),
        np.asarray([0.03, 0.01, -0.02]),
    )
    prior = stats.predict(
        np.asarray([0]), np.asarray([0]), smoothing=10.0, use_cluster=False
    )[0]
    cluster = stats.predict(
        np.asarray([0]), np.asarray([0]), smoothing=10.0, use_cluster=True
    )[0]
    assert cluster > prior


def test_hac_mean_returns_zero_width_for_constant_series() -> None:
    result = action._hac_mean(np.zeros(10), lag=3)
    assert result["mean"] == 0.0
    assert result["se"] == 0.0
