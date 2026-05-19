from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from daily_research.continuous_policy.pipeline_utils import select_feature_columns
from daily_research.continuous_policy.portfolio_simulator import PortfolioState
from daily_research.continuous_policy.state_builder import build_cross_section_state
from daily_research.path_policy.forecast_features import (
    FORECAST_FEATURE_PROFILES,
    audit_forecast_feature_profile,
    build_forecast_feature_panels,
)
from daily_research.path_policy.tests.fixtures import make_prepared_policy_inputs


def test_state_v1_matches_existing_state_feature_selection() -> None:
    prepared = make_prepared_policy_inputs(days=80, stocks=("AAA", "BBB", "CCC"), start_date="2024-01-02")
    dates = [pd.Timestamp(prepared.close.index[-1]).normalize()]
    state_frame = build_cross_section_state(prepared, date=dates[0], portfolio_state=PortfolioState()).copy()
    expected = [column for column in select_feature_columns(state_frame) if column not in {"date", "stock", "in_universe"}][:20]

    panels, feature_columns, manifest = build_forecast_feature_panels(
        prepared,
        dates,
        feature_profile="state_v1",
        max_feature_columns=20,
    )

    assert "state_v1" in FORECAST_FEATURE_PROFILES
    assert feature_columns == expected
    assert list(panels[dates[0]].columns) == expected
    assert manifest["feature_profile"] == "state_v1"
    assert manifest["feature_count_after_cap"] == 20


def test_raw_kline_profile_adds_ohlcv_shape_features_with_expected_values() -> None:
    prepared = make_prepared_policy_inputs(days=80, stocks=("AAA", "BBB", "CCC"), start_date="2024-01-02")
    date = pd.Timestamp(prepared.close.index[25]).normalize()
    stock = "AAA"

    panels, feature_columns, manifest = build_forecast_feature_panels(
        prepared,
        [date],
        feature_profile="raw_kline_v1",
        max_feature_columns=192,
    )
    row = panels[date].loc[stock]
    prev_close = float(prepared.close.shift(1).loc[date, stock])
    open_ = float(prepared.open_.loc[date, stock])
    high = float(prepared.high.loc[date, stock])
    low = float(prepared.low.loc[date, stock])
    close = float(prepared.close.loc[date, stock])

    assert "raw_kline_v1" in FORECAST_FEATURE_PROFILES
    assert "raw_open_gap_1d" in feature_columns
    assert "raw_close_to_open_1d" in feature_columns
    assert "raw_intraday_range_1d" in feature_columns
    assert row["raw_open_gap_1d"] == pytest.approx(open_ / prev_close - 1.0)
    assert row["raw_close_to_open_1d"] == pytest.approx(close / open_ - 1.0)
    assert row["raw_intraday_range_1d"] == pytest.approx(high / low - 1.0)
    assert manifest["raw_kline_feature_count"] > 0


def test_raw_kline_features_do_not_change_when_future_prices_are_mutated() -> None:
    prepared = make_prepared_policy_inputs(days=90, stocks=("AAA", "BBB", "CCC"), start_date="2024-01-02")
    date = pd.Timestamp(prepared.close.index[30]).normalize()
    future_mask = prepared.close.index > date

    original_panels, _, _ = build_forecast_feature_panels(
        prepared,
        [date],
        feature_profile="raw_kline_v1",
        max_feature_columns=192,
    )

    mutated_frames = {}
    for name in ("open_", "high", "low", "close", "volume", "amount"):
        frame = getattr(prepared, name).copy()
        frame.loc[future_mask] = frame.loc[future_mask] * 1000.0 + 123.0
        mutated_frames[name] = frame
    mutated = replace(prepared, **mutated_frames)
    mutated_panels, _, _ = build_forecast_feature_panels(
        mutated,
        [date],
        feature_profile="raw_kline_v1",
        max_feature_columns=192,
    )

    raw_columns = [column for column in original_panels[date].columns if column.startswith("raw_")]
    np.testing.assert_allclose(
        original_panels[date][raw_columns].to_numpy(dtype=float),
        mutated_panels[date][raw_columns].to_numpy(dtype=float),
        rtol=1.0e-9,
        atol=1.0e-9,
    )


def test_raw_kline_context_profile_includes_market_benchmark_and_peer_context() -> None:
    prepared = make_prepared_policy_inputs(days=90, stocks=("AAA", "BBB", "CCC", "DDD", "EEE"), start_date="2024-01-02")
    date = pd.Timestamp(prepared.close.index[-1]).normalize()

    panels, feature_columns, manifest = build_forecast_feature_panels(
        prepared,
        [date],
        feature_profile="raw_kline_context_v1",
        max_feature_columns=256,
    )

    assert "market_positive_share_1d" in feature_columns
    assert "benchmark_ret_20d" in feature_columns
    assert "peer_adv_bucket_ret_5d_mean" in feature_columns
    assert "relative_to_peer_adv_ret_20d" in feature_columns
    assert panels[date]["market_positive_share_1d"].nunique(dropna=False) == 1
    assert manifest["market_context_feature_count"] > 0
    assert manifest["peer_context_feature_count"] > 0


def test_sector_context_profile_uses_prepared_metadata_frames() -> None:
    prepared = make_prepared_policy_inputs(days=90, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2024-01-02")
    industry_map = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "industry": ["tech", "tech", "bank", "bank"],
            "source": ["unit"] * 4,
            "as_of_date": ["2026-05-19"] * 4,
        }
    )
    board_membership = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "board_kind": ["GN", "GN", "FG"],
            "board_name": ["ai", "ai", "dividend"],
            "board_code": ["880001", "880001", "880002"],
            "source": ["unit"] * 3,
            "as_of_date": ["2026-05-19"] * 3,
        }
    )
    prepared = replace(
        prepared,
        metadata_frames={"industry_map": industry_map, "board_membership": board_membership},
        metadata_summary={"sector_board_view": {"dataset_id": "policy_sector_board_view__unit"}},
    )
    date = pd.Timestamp(prepared.close.index[-1]).normalize()

    panels, feature_columns, manifest = build_forecast_feature_panels(
        prepared,
        [date],
        feature_profile="raw_kline_context_sector_v1",
        max_feature_columns=256,
    )

    assert "raw_kline_context_sector_v1" in FORECAST_FEATURE_PROFILES
    assert "industry_ret_20_excess" in feature_columns
    assert "industry_rank_ret_20" in feature_columns
    assert "industry_member_count" in feature_columns
    assert "board_member_count" in feature_columns
    assert manifest["sector_context_feature_count"] == 4
    assert manifest["source_sector_board_view_id"] == "policy_sector_board_view__unit"
    assert panels[date].loc["AAA", "industry_member_count"] == pytest.approx(2.0)
    assert panels[date].loc["DDD", "board_member_count"] == pytest.approx(0.0)


def test_no_alpha_prior_profile_removes_old_alpha_score_dependencies() -> None:
    prepared = make_prepared_policy_inputs(days=90, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2024-01-02")
    date = pd.Timestamp(prepared.close.index[-1]).normalize()

    _, feature_columns, manifest = build_forecast_feature_panels(
        prepared,
        [date],
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        max_feature_columns=256,
    )

    forbidden = ("alpha_prior_", "score_none", "score_v2", "score_blend")
    assert not any(column.startswith("alpha_prior_") for column in feature_columns)
    assert not any(column in forbidden for column in feature_columns)
    assert "raw_open_gap_1d" in feature_columns
    assert "market_positive_share_1d" in feature_columns
    assert manifest["alpha_prior_feature_count"] == 0


def test_feature_profile_audit_reports_group_counts_and_profile() -> None:
    prepared = make_prepared_policy_inputs(days=80, stocks=("AAA", "BBB", "CCC"), start_date="2024-01-02")
    dates = [pd.Timestamp(prepared.close.index[-1]).normalize()]

    audit = audit_forecast_feature_profile(
        prepared,
        dates,
        feature_profile="raw_kline_context_v1",
        max_feature_columns=128,
    )

    assert audit["feature_profile"] == "raw_kline_context_v1"
    assert audit["feature_count_before_cap"] >= audit["feature_count_after_cap"]
    assert audit["feature_count_after_cap"] <= 128
    assert audit["feature_group_counts"]["raw_kline"] > 0
