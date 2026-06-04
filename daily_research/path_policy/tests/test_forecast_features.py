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
    _datewise_group_mean_frame,
    _datewise_group_rank_frame,
    _group_mean_frame,
    _group_member_count_frame,
    _group_rank_frame,
    _group_share_frame,
    _group_z_frame,
    audit_forecast_feature_profile,
    build_forecast_feature_panels,
    build_forecast_feature_store,
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


def test_sector_context_profile_records_source_view_from_raw_cache_meta() -> None:
    prepared = make_prepared_policy_inputs(days=90, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2024-01-02")
    prepared.metadata_frames["industry_map"] = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "industry": ["tech", "tech", "bank", "bank"],
        }
    )
    prepared.metadata_frames["board_membership"] = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "board_kind": ["GN", "GN", "FG"],
            "board_name": ["ai", "ai", "dividend"],
            "board_code": ["880001", "880001", "880002"],
        }
    )
    prepared.raw_cache_meta["sector_board_view"] = {
        "dataset_id": "policy_sector_board_view__raw_cache",
        "view_kind": "latest_static_snapshot",
    }
    date = pd.Timestamp(prepared.close.index[-1]).normalize()

    _, feature_columns, manifest = build_forecast_feature_panels(
        prepared,
        [date],
        feature_profile="raw_kline_context_sector_v1",
        max_feature_columns=256,
    )

    assert "industry_ret_20_excess" in feature_columns
    assert manifest["sector_context_feature_count"] == 4
    assert manifest["source_sector_board_view_id"] == "policy_sector_board_view__raw_cache"


def test_datewise_industry_group_helpers_match_naive_semantics() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    columns = ["AAA", "BBB", "CCC", "DDD"]
    values = pd.DataFrame(
        [
            [1.0, 3.0, 10.0, 14.0],
            [np.nan, 4.0, 8.0, 16.0],
            [5.0, 7.0, np.nan, 20.0],
        ],
        index=dates,
        columns=columns,
    )
    groups = pd.DataFrame(
        [
            ["tech", "tech", "bank", "bank"],
            ["bank", "tech", "tech", ""],
            [np.nan, "tech", "tech", "bank"],
        ],
        index=dates,
        columns=columns,
    )
    condition = values.gt(6.0)

    mean_fast = _group_mean_frame(values, groups)
    rank_fast = _group_rank_frame(values, groups)
    z_fast = _group_z_frame(values, groups)
    count_fast = _group_member_count_frame(groups, index=dates, columns=columns)
    share_fast = _group_share_frame(condition, groups)

    pd.testing.assert_frame_equal(mean_fast, _datewise_group_mean_frame(values, groups))
    pd.testing.assert_frame_equal(rank_fast, _datewise_group_rank_frame(values, groups))
    assert mean_fast.loc[dates[1], "AAA"] != mean_fast.loc[dates[1], "AAA"]
    assert mean_fast.loc[dates[1], "BBB"] == pytest.approx(6.0)
    assert count_fast.loc[dates[0], "AAA"] == pytest.approx(2.0)
    assert count_fast.loc[dates[1], "DDD"] == pytest.approx(0.0)
    assert share_fast.loc[dates[0], "AAA"] == pytest.approx(0.0)
    assert share_fast.loc[dates[0], "CCC"] == pytest.approx(1.0)
    assert z_fast.loc[dates[0], "AAA"] == pytest.approx(-1.0)
    assert z_fast.loc[dates[0], "BBB"] == pytest.approx(1.0)


def test_sector_relative_profile_uses_industry_relative_features_and_logs() -> None:
    prepared = make_prepared_policy_inputs(days=90, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2024-01-02")
    prepared.metadata_frames["industry_map"] = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "industry": ["tech", "tech", "bank", "bank"],
        }
    )
    prepared.metadata_frames["board_membership"] = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "board_kind": ["GN", "GN", "FG"],
            "board_name": ["ai", "ai", "dividend"],
            "board_code": ["880001", "880001", "880002"],
        }
    )
    prepared.raw_cache_meta["sector_board_view"] = {
        "dataset_id": "policy_sector_board_view__raw_cache",
        "view_kind": "latest_static_snapshot",
    }
    date = pd.Timestamp(prepared.close.index[-1]).normalize()

    panels, feature_columns, manifest = build_forecast_feature_panels(
        prepared,
        [date],
        feature_profile="raw_kline_context_sector_relative_v1",
        max_feature_columns=256,
    )

    assert "raw_kline_context_sector_relative_v1" in FORECAST_FEATURE_PROFILES
    assert "industry_ret_5_excess" in feature_columns
    assert "stock_ret_20_minus_industry" in feature_columns
    assert "industry_positive_share_20" in feature_columns
    assert "industry_member_count_log" in feature_columns
    assert "board_member_count_log" in feature_columns
    assert manifest["sector_relative_context_feature_count"] == 10
    assert manifest["source_sector_board_view_id"] == "policy_sector_board_view__raw_cache"
    assert panels[date].loc["AAA", "industry_member_count_log"] == pytest.approx(np.log1p(2.0))
    assert panels[date].loc["DDD", "board_member_count_log"] == pytest.approx(0.0)


def test_regime_profile_adds_market_regime_features_without_sector_metadata() -> None:
    prepared = make_prepared_policy_inputs(days=120, stocks=("AAA", "BBB", "CCC", "DDD", "EEE"), start_date="2024-01-02")
    date = pd.Timestamp(prepared.close.index[-1]).normalize()

    _, feature_columns, manifest = build_forecast_feature_panels(
        prepared,
        [date],
        feature_profile="raw_kline_context_regime_v1",
        max_feature_columns=256,
    )

    assert "raw_kline_context_regime_v1" in FORECAST_FEATURE_PROFILES
    assert "market_ret_20_z" in feature_columns
    assert "market_vol_20_z" in feature_columns
    assert "market_drawdown_20" in feature_columns
    assert "cross_section_ret_dispersion_20" in feature_columns
    assert "benchmark_trend_vol_interaction_20" in feature_columns
    assert manifest["regime_context_feature_count"] == 8
    assert manifest["sector_relative_context_feature_count"] == 0


def test_combined_sector_relative_regime_profile_reports_both_groups() -> None:
    prepared = make_prepared_policy_inputs(days=120, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2024-01-02")
    prepared.metadata_frames["industry_map"] = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "industry": ["tech", "tech", "bank", "bank"],
        }
    )
    prepared.metadata_frames["board_membership"] = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "board_kind": ["GN", "GN", "FG"],
            "board_name": ["ai", "ai", "dividend"],
            "board_code": ["880001", "880001", "880002"],
        }
    )
    prepared.metadata_summary["sector_board_view"] = {"dataset_id": "policy_sector_board_view__unit"}
    date = pd.Timestamp(prepared.close.index[-1]).normalize()

    _, feature_columns, manifest = build_forecast_feature_panels(
        prepared,
        [date],
        feature_profile="raw_kline_context_sector_relative_regime_v1",
        max_feature_columns=256,
    )

    assert "raw_kline_context_sector_relative_regime_v1" in FORECAST_FEATURE_PROFILES
    assert "industry_rank_ret_5" in feature_columns
    assert "market_breadth_20" in feature_columns
    assert manifest["sector_relative_context_feature_count"] == 10
    assert manifest["regime_context_feature_count"] == 8
    assert manifest["source_sector_board_view_id"] == "policy_sector_board_view__unit"


def test_combined_profile_new_features_do_not_change_when_future_prices_are_mutated() -> None:
    prepared = make_prepared_policy_inputs(days=140, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2024-01-02")
    prepared.metadata_frames["industry_map"] = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "industry": ["tech", "tech", "bank", "bank"],
        }
    )
    prepared.metadata_frames["board_membership"] = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "board_kind": ["GN", "GN", "FG"],
            "board_name": ["ai", "ai", "dividend"],
            "board_code": ["880001", "880001", "880002"],
        }
    )
    prepared.raw_cache_meta["sector_board_view"] = {
        "dataset_id": "policy_sector_board_view__raw_cache",
        "view_kind": "latest_static_snapshot",
    }
    date = pd.Timestamp(prepared.close.index[70]).normalize()
    future_mask = prepared.close.index > date

    original_panels, _, _ = build_forecast_feature_panels(
        prepared,
        [date],
        feature_profile="raw_kline_context_sector_relative_regime_v1",
        max_feature_columns=256,
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
        feature_profile="raw_kline_context_sector_relative_regime_v1",
        max_feature_columns=256,
    )

    new_columns = [
        column
        for column in original_panels[date].columns
        if column.startswith("industry_")
        or column.startswith("stock_ret_")
        or column.startswith("market_ret_20_z")
        or column.startswith("market_vol_20_z")
        or column.startswith("market_drawdown_20")
        or column.startswith("market_breadth_20")
        or column.startswith("market_liquidity_z20")
        or column.startswith("cross_section_ret_dispersion_20")
        or column.startswith("limit_up_down_pressure_5")
        or column.startswith("benchmark_trend_vol_interaction_20")
        or column.startswith("board_member_count_log")
    ]
    np.testing.assert_allclose(
        original_panels[date][new_columns].to_numpy(dtype=float),
        mutated_panels[date][new_columns].to_numpy(dtype=float),
        rtol=1.0e-9,
        atol=1.0e-9,
    )


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


def test_feature_profile_audit_reports_group_stats_and_future_leakage_smoke() -> None:
    prepared = make_prepared_policy_inputs(days=140, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2024-01-02")
    prepared.metadata_frames["industry_map"] = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "industry": ["tech", "tech", "bank", "bank"],
        }
    )
    prepared.metadata_frames["board_membership"] = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "board_kind": ["GN", "GN", "FG"],
            "board_name": ["ai", "ai", "dividend"],
            "board_code": ["880001", "880001", "880002"],
        }
    )
    prepared.metadata_summary["sector_board_view"] = {"dataset_id": "policy_sector_board_view__unit"}
    dates = [pd.Timestamp(item).normalize() for item in prepared.close.index[-5:]]

    audit = audit_forecast_feature_profile(
        prepared,
        dates,
        feature_profile="raw_kline_context_sector_relative_regime_v1",
        max_feature_columns=192,
    )

    feature_audit = audit["feature_profile_audit"]
    assert feature_audit["feature_profile"] == "raw_kline_context_sector_relative_regime_v1"
    assert feature_audit["retained_groups"]["raw_kline"] is True
    assert feature_audit["retained_groups"]["sector_relative_context"] is True
    assert feature_audit["retained_groups"]["regime_context"] is True
    assert feature_audit["group_stats"]["sector_relative_context"]["feature_count"] > 0
    assert feature_audit["group_stats"]["regime_context"]["feature_count"] > 0
    assert feature_audit["group_stats"]["sector_relative_context"]["finite_ratio"] > 0.0
    assert feature_audit["future_leakage_smoke"]["passed"] is True


def test_feature_store_manifest_includes_profile_audit_for_training_artifacts(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=90, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2024-01-02")
    prepared.metadata_frames["industry_map"] = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "industry": ["tech", "tech", "bank", "bank"],
        }
    )
    prepared.metadata_frames["board_membership"] = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "board_kind": ["GN", "GN", "FG"],
            "board_name": ["ai", "ai", "dividend"],
            "board_code": ["880001", "880001", "880002"],
        }
    )
    prepared.metadata_summary["sector_board_view"] = {"dataset_id": "policy_sector_board_view__unit"}
    dates = [pd.Timestamp(item).normalize() for item in prepared.close.index[-8:]]

    _, _, manifest, _ = build_forecast_feature_store(
        prepared,
        dates,
        root=tmp_path,
        feature_profile="raw_kline_context_sector_relative_regime_v1",
        max_feature_columns=192,
    )

    feature_audit = manifest["feature_profile_audit"]
    assert feature_audit["feature_profile"] == "raw_kline_context_sector_relative_regime_v1"
    assert feature_audit["retained_groups"]["sector_relative_context"] is True
    assert feature_audit["retained_groups"]["regime_context"] is True
    assert feature_audit["group_stats"]["sector_relative_context"]["feature_count"] > 0
    assert feature_audit["future_leakage_smoke"]["passed"] is True
