from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from daily_research.path_policy import seq100_turning_path_quality_pool as quality


def test_contract_uses_daily_point_in_time_pool_at_probe_onset() -> None:
    study = quality.load_study()
    assert study["source"]["quality_pool_name"] == "quality_liquidity_pit"
    assert study["universe_semantics"]["membership_time"] == "probe onset date"
    assert (
        "full valid adjusted-close history"
        in study["universe_semantics"]["path_geometry"]
    )
    assert study["analysis"]["profit_claim_allowed"] is False
    assert study["resources"]["duckdb_adaptive"] is True


def test_quality_query_preserves_missing_signal_state_with_left_joins() -> None:
    query = quality._quality_episode_query(
        Path("episodes.parquet"),
        [Path("pool.parquet")],
        [Path("rules.parquet")],
        [Path("neutral.parquet")],
        ["attention"],
    )
    assert "INNER JOIN pool" in query
    assert query.count("LEFT JOIN") == 2
    assert "legacy_panel_covered" in query
    assert "all_rank_features_complete" in query


def test_contrast_stability_uses_discovery_direction_only() -> None:
    annual = pd.DataFrame(
        {
            "episode_type": ["up_pullback"] * 4,
            "feature": ["attention_rank"] * 4,
            "probe_multiplier": [2] * 4,
            "major_threshold_multiplier": [8] * 4,
            "evaluation_year": [2017, 2018, 2019, 2020],
            "difference_mean": [0.10, 0.20, 0.05, -0.01],
        }
    )
    result = quality._contrast_stability(
        annual, discovery_years=[2017, 2018], evaluation_years=[2019, 2020]
    ).iloc[0]
    assert result["discovery_difference_mean"] == pytest.approx(0.15)
    assert result["evaluation_same_discovery_sign_years"] == 1
    assert result["evaluation_years"] == 2


def test_predictive_analyses_exclude_same_day_resolutions() -> None:
    assert "resolution_date_idx > onset_date_idx" in quality.FUTURE_RESOLUTION_FILTER
    query = quality._path_profile_query(
        Path("episodes.parquet"),
        Path("dense.parquet"),
        quality.load_study(),
    )
    assert quality.FUTURE_RESOLUTION_FILTER in query
