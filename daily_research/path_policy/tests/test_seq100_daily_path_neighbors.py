from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_daily_path_neighbors as neighbors
from daily_research.path_policy import seq100_daily_path_neighbors_validate as validate


def test_study_contract_is_daily_natural_time_and_quality_pool() -> None:
    study = neighbors.load_study()
    assert study["source"]["quality_pool_name"] == "quality_liquidity_pit"
    assert study["outcomes"]["fixed_holding_horizon"] is False
    assert study["path_state"]["turning_scales"] == [4, 8, 16]
    assert study["boundaries"]["report_generation_performed"] is False
    assert study["evaluation"]["primary_target"] == "extreme_ahead"


def test_panel_query_uses_strict_future_event_and_optional_auxiliary_state() -> None:
    query = neighbors._panel_query(
        year=2019,
        quality_path=Path("quality.parquet"),
        state_path=Path("state.parquet"),
        dense_path=Path("dense.parquet"),
        event_path=Path("events.parquet"),
    )
    assert "q.date_idx < f4.confirmation_date_idx" in query
    assert "q.date_idx >= p4.confirmation_date_idx" in query
    assert "f4.extreme_date_idx > q.date_idx" in query
    assert "FROM read_parquet('quality.parquet') q" in query
    assert "LEFT JOIN read_parquet('state.parquet') aux" in query
    assert "path_close_" not in query
    assert "terminal_log_return_5" not in query


def test_updates_are_applied_only_after_resolution_date() -> None:
    state = neighbors.PrototypeStats(cluster_count=2)
    updates = {
        "scale_index": np.array([0, 0], dtype=np.int8),
        "direction": np.array([1, 1], dtype=np.int8),
        "target": np.array([1.0, 0.0]),
        "resolution": np.array([10.0, 11.0]),
        "geometry": np.array([0, 1], dtype=np.int32),
        "log_confirmation_wait": np.array([1.0, 2.0]),
        "entry_confirmation_log_return": np.array([0.1, -0.1]),
        "log_extreme_wait_ahead": np.array([0.5, np.nan]),
        "directional_extreme_log_return_ahead": np.array([0.2, np.nan]),
    }
    states = {"geometry": state}
    cursor = neighbors._apply_updates_before(states, updates, 0, 10)
    assert cursor == 0
    assert state.global_count.sum() == 0
    cursor = neighbors._apply_updates_before(states, updates, cursor, 11)
    assert cursor == 1
    assert state.global_count.sum() == 1
    assert state.last_applied_resolution == 10


def test_prototype_probability_shrinks_to_causal_prior() -> None:
    state = neighbors.PrototypeStats(cluster_count=2)
    state.update(
        scale_index=np.array([0, 0, 0], dtype=np.int8),
        direction=np.array([1, 1, 1], dtype=np.int8),
        target=np.array([1.0, 1.0, 0.0]),
        cluster=np.array([0, 0, 1], dtype=np.int32),
        resolution=np.array([1, 2, 3]),
        metrics={
            name: np.array([1.0, 2.0, 3.0]) for name in neighbors.CONTINUOUS_METRICS
        },
    )
    scale = np.array([0, 0], dtype=np.int8)
    direction = np.array([1, 1], dtype=np.int8)
    cluster = np.array([0, 1], dtype=np.int32)
    prior = state.prior_probability(scale, direction)
    probability = state.probability(scale, direction, cluster, smoothing=10.0)
    assert np.all((probability > 0) & (probability < 1))
    assert probability[0] > prior[0]
    assert probability[1] < prior[1]


def test_preprocessor_imputes_and_returns_finite_embedding() -> None:
    frame = pd.DataFrame(
        {
            "a": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0],
            "b": [1.0, np.inf, 2.0, 3.0, 4.0, 6.0],
            "c": [2.0, 1.0, 0.0, -1.0, -2.0, -3.0],
        }
    )
    preprocessor, embedding = neighbors._fit_preprocessor(
        frame,
        ("a", "b", "c"),
        components=2,
        lower_quantile=0.0,
        upper_quantile=1.0,
        random_seed=7,
    )
    transformed = preprocessor.transform(frame)
    assert embedding.shape == (6, 2)
    assert transformed.shape == (6, 2)
    assert np.isfinite(transformed).all()


def test_selection_rule_prefers_simpler_eligible_configuration() -> None:
    rows = []
    for cluster_count, smoothing, losses in (
        (64, 100.0, [0.49, 0.51, 0.49]),
        (128, 20.0, [0.49, 0.51, 0.49]),
    ):
        for date_idx, loss in enumerate(losses, start=1):
            rows.append(
                {
                    "feature_set": "sequence",
                    "cluster_count": cluster_count,
                    "smoothing_strength": smoothing,
                    "date_idx": date_idx,
                    "prior_log_loss": 0.55,
                    "model_log_loss": loss,
                }
            )
    diagnostics, selected = neighbors._select_configuration(
        pd.DataFrame(rows), hac_lag=1
    )
    assert selected["sequence"]["cluster_count"] == 64
    assert diagnostics[diagnostics["selected"]]["cluster_count"].item() == 64


def test_blend_selection_can_reject_sequence_increment() -> None:
    rows = []
    for date_idx in range(1, 31):
        for weight, loss in ((0.0, 0.50), (0.25, 0.52), (1.0, 0.58)):
            rows.append(
                {
                    "date_idx": date_idx,
                    "sequence_incremental_weight": weight,
                    "geometry_log_loss": 0.50,
                    "blend_log_loss": loss,
                }
            )
    diagnostics, selected = neighbors._select_blend_weight(
        pd.DataFrame(rows), hac_lag=2
    )
    assert selected["incremental_weight"] == 0.0
    assert (
        diagnostics.loc[diagnostics["selected"], "sequence_incremental_weight"].item()
        == 0.0
    )


def test_evaluation_scopes_separate_strict_and_near_complete_years() -> None:
    scopes = neighbors._evaluation_scopes(neighbors.load_study())
    assert list(scopes) == [
        "strict_complete_2019_2023",
        "near_complete_2019_2024",
        "all_2019_2025_provisional",
    ]
    assert scopes["strict_complete_2019_2023"] == set(range(2019, 2024))
    assert scopes["near_complete_2019_2024"] == set(range(2019, 2025))


def test_daily_rank_relationship_recovers_monotone_phase_signal() -> None:
    frame = pd.DataFrame(
        {
            "signal_year": [2019] * 40,
            "date_idx": np.repeat([1, 2], 20),
            "direction": [1] * 40,
            "feature": np.tile(np.arange(20), 2),
            "target": np.tile(np.r_[np.zeros(10), np.ones(10)], 2),
        }
    )
    result = validate._daily_rank_relationship(
        frame,
        feature="feature",
        target="target",
        continuous_target=False,
    )
    assert len(result) == 2
    assert (result["correlation"] > 0.8).all()
    assert (result["top_minus_bottom"] == 1.0).all()


def test_bh_adjustment_is_monotone_in_sorted_p_values() -> None:
    adjusted = validate._bh_adjust(np.array([0.01, 0.04, 0.03, np.nan]))
    assert np.isnan(adjusted[-1])
    order = np.argsort(np.array([0.01, 0.04, 0.03]))
    assert np.all(np.diff(adjusted[:3][order]) >= 0)
