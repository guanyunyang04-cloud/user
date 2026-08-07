from __future__ import annotations

import numpy as np

from daily_research.path_policy import seq100_daily_continuation_value as value


def test_study_uses_one_step_legal_counterfactual_without_total_horizon() -> None:
    study = value.load_study()
    assert study["source"]["quality_pool_name"] == "quality_liquidity_pit"
    assert study["counterfactual"]["fixed_total_holding_horizon_used"] is False
    assert study["counterfactual"]["exit_now_action"].startswith("request the next")
    assert study["state"]["threshold_selection_performed"] is False


def test_setup_code_separates_pullback_rebound_and_other() -> None:
    result = value._setup_code(
        np.asarray([True, False, True]),
        np.asarray([-0.01, 0.02, 0.03]),
        np.asarray([0.10, -0.10, 0.10]),
    )
    assert np.array_equal(result, np.asarray([1, 2, 0], dtype=np.int8))


def test_continuation_stats_shrink_cluster_mean_to_group_prior() -> None:
    state = value.ContinuationStats(cluster_count=2)
    state.update(
        group=np.asarray([0, 0, 0]),
        cluster=np.asarray([0, 0, 1]),
        value=np.asarray([0.03, 0.01, -0.02]),
        resolution=np.asarray([1, 2, 3]),
    )
    prior, _, _ = state.prior(np.asarray([0, 0]))
    prediction, _, _ = state.predict(
        np.asarray([0, 0]), np.asarray([0, 1]), smoothing=10.0
    )
    assert prediction[0] > prior[0]
    assert prediction[1] < prior[1]


def test_updates_are_applied_only_after_both_actions_resolve() -> None:
    states = {
        "geometry": value.ContinuationStats(2),
        "sequence": value.ContinuationStats(2),
    }
    updates = {
        "group": np.asarray([0, 0], dtype=np.int8),
        "geometry": np.asarray([0, 1], dtype=np.int16),
        "sequence": np.asarray([1, 0], dtype=np.int16),
        "resolution": np.asarray([10, 11], dtype=np.int32),
        "value": np.asarray([0.01, -0.01]),
    }
    cursor = value._apply_updates_before(states, updates, 0, 10)
    assert cursor == 0
    cursor = value._apply_updates_before(states, updates, cursor, 11)
    assert cursor == 1
    assert states["geometry"].last_resolution == 10


def test_sale_resolution_never_reads_past_explicit_date_cutoff() -> None:
    class Pack:
        date_values = np.asarray(["2025-12-30", "2025-12-31", "2026-01-02"])
        exit_sellable = np.ones((3, 1), dtype=bool)
        exit_close_raw = np.asarray([[10.0], [11.0], [12.0]], dtype=np.float32)

    resolved, price = value._resolve_sale(
        pack=Pack(),
        date_idx=np.asarray([0]),
        symbol_idx=np.asarray([0]),
        minimum_offset=2,
        maximum_offset=2,
        maximum_date_idx=1,
    )
    assert resolved[0] == -1
    assert np.isnan(price[0])
