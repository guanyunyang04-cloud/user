from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy.forecast_features import (
    FORECAST_FEATURE_PROFILES,
    build_forecast_feature_panels,
    build_forecast_feature_store,
)
from daily_research.path_policy.tests.fixtures import make_prepared_policy_inputs


def test_v2_amount_checked_profile_records_normalization_metadata() -> None:
    prepared = make_prepared_policy_inputs(days=80, stocks=("AAA", "BBB", "CCC"), start_date="2024-01-02")
    prepared = replace(prepared, amount=prepared.amount * 10000.0)
    date = pd.Timestamp(prepared.close.index[-1]).normalize()

    _panels, columns, manifest = build_forecast_feature_panels(
        prepared,
        [date],
        feature_profile="raw_kline_context_v2_tradeable_amount_checked",
        max_feature_columns=256,
    )

    amount_audit = manifest["feature_profile_audit"]["amount_unit"]
    assert "raw_kline_context_v2_tradeable_amount_checked" in FORECAST_FEATURE_PROFILES
    assert amount_audit["amount_unit_policy"] == "divide_by_10000"
    assert amount_audit["amount_unit_factor"] == pytest.approx(0.0001)
    assert "raw_amount_z20" in columns
    assert "cs_rank_amount_20d" in columns
    assert "market_liquidity_z20" in columns


def test_v2_amount_checked_profile_excludes_alpha_prior_inputs() -> None:
    prepared = make_prepared_policy_inputs(days=80, stocks=("AAA", "BBB", "CCC"), start_date="2024-01-02")
    date = pd.Timestamp(prepared.close.index[-1]).normalize()

    _panels, columns, manifest = build_forecast_feature_panels(
        prepared,
        [date],
        feature_profile="raw_kline_context_v2_tradeable_amount_checked",
        max_feature_columns=256,
    )

    assert not any(column.startswith("alpha_prior_") for column in columns)
    assert not any(column.startswith("score") or column.startswith("z_score") for column in columns)
    assert manifest["alpha_prior_feature_count"] == 0


def test_v2_amount_checked_features_use_normalized_amount_values() -> None:
    normal = make_prepared_policy_inputs(days=90, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2024-01-02")
    scaled = replace(normal, amount=normal.amount * 10000.0)
    date = pd.Timestamp(normal.close.index[-1]).normalize()

    normal_panels, columns, _normal_manifest = build_forecast_feature_panels(
        normal,
        [date],
        feature_profile="raw_kline_context_v2_tradeable_amount_checked",
        max_feature_columns=256,
    )
    scaled_panels, _scaled_columns, scaled_manifest = build_forecast_feature_panels(
        scaled,
        [date],
        feature_profile="raw_kline_context_v2_tradeable_amount_checked",
        max_feature_columns=256,
    )
    amount_sensitive = [column for column in ("raw_amount_z20", "raw_amount_ratio_5_20", "cs_rank_amount_20d", "market_liquidity_z20") if column in columns]

    np.testing.assert_allclose(
        normal_panels[date][amount_sensitive].to_numpy(dtype=float),
        scaled_panels[date][amount_sensitive].to_numpy(dtype=float),
        rtol=1.0e-6,
        atol=1.0e-6,
        equal_nan=True,
    )
    assert scaled_manifest["feature_profile_audit"]["amount_unit"]["amount_unit_policy"] == "divide_by_10000"


def test_original_corrected_116_profile_is_not_amount_checked() -> None:
    prepared = make_prepared_policy_inputs(days=80, stocks=("AAA", "BBB", "CCC"), start_date="2024-01-02")
    prepared = replace(prepared, amount=prepared.amount * 10000.0)
    date = pd.Timestamp(prepared.close.index[-1]).normalize()

    _panels, _columns, manifest = build_forecast_feature_panels(
        prepared,
        [date],
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        max_feature_columns=256,
    )

    amount_audit = manifest["feature_profile_audit"]["amount_unit"]
    assert amount_audit["amount_unit_policy"] == "not_checked_for_profile"
    assert amount_audit["amount_unit_factor"] == pytest.approx(1.0)


def test_v2_amount_checked_feature_store_manifest_keeps_amount_audit(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=90, stocks=("AAA", "BBB", "CCC"), start_date="2024-01-02")
    prepared = replace(prepared, amount=prepared.amount * 0.0001)
    dates = [pd.Timestamp(item).normalize() for item in prepared.close.index[-5:]]

    _path, _columns, manifest, _history = build_forecast_feature_store(
        prepared,
        dates,
        root=tmp_path,
        feature_profile="raw_kline_context_v2_tradeable_amount_checked",
        max_feature_columns=256,
    )

    amount_audit = manifest["feature_profile_audit"]["amount_unit"]
    assert amount_audit["amount_unit_policy"] == "multiply_by_10000"
    assert amount_audit["amount_unit_factor"] == pytest.approx(10000.0)
    assert manifest["feature_store_shape"][2] == manifest["feature_count_after_cap"]


def test_v2_local_state_profile_adds_local_state_features_without_alpha_prior() -> None:
    prepared = make_prepared_policy_inputs(days=90, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2024-01-02")
    date = pd.Timestamp(prepared.close.index[-1]).normalize()

    _panels, columns, manifest = build_forecast_feature_panels(
        prepared,
        [date],
        feature_profile="raw_kline_context_v2_tradeable_local_state_v1",
        max_feature_columns=256,
    )

    assert "raw_kline_context_v2_tradeable_local_state_v1" in FORECAST_FEATURE_PROFILES
    assert manifest["local_state_context_feature_count"] > 0
    assert manifest["feature_group_counts"]["local_state_context"] == manifest["local_state_context_feature_count"]
    assert "local_vol_20d" in columns
    assert "cs_rank_local_vol_20d" in columns
    assert "local_high_volatility_x_reversal" in columns
    assert "raw_amount_z20" in columns
    assert manifest["feature_profile_audit"]["amount_unit"]["amount_unit_policy"] == "as_is"
    assert not any(column.startswith("alpha_prior_") for column in columns)
    assert not any(column.startswith("score") or column.startswith("z_score") for column in columns)


def test_v2_amount_checked_profile_does_not_gain_local_state_features() -> None:
    prepared = make_prepared_policy_inputs(days=90, stocks=("AAA", "BBB", "CCC"), start_date="2024-01-02")
    date = pd.Timestamp(prepared.close.index[-1]).normalize()

    _panels, columns, manifest = build_forecast_feature_panels(
        prepared,
        [date],
        feature_profile="raw_kline_context_v2_tradeable_amount_checked",
        max_feature_columns=256,
    )

    assert manifest["local_state_context_feature_count"] == 0
    assert "local_vol_20d" not in columns
    assert "cs_rank_local_vol_20d" not in columns


def test_v2_local_state_profile_store_manifest_retains_group_audit(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=90, stocks=("AAA", "BBB", "CCC"), start_date="2024-01-02")
    dates = [pd.Timestamp(item).normalize() for item in prepared.close.index[-5:]]

    _path, columns, manifest, _history = build_forecast_feature_store(
        prepared,
        dates,
        root=tmp_path,
        feature_profile="raw_kline_context_v2_tradeable_local_state_v1",
        max_feature_columns=256,
    )

    assert "local_high_volatility_x_runup" in columns
    assert manifest["feature_store_shape"][2] == manifest["feature_count_after_cap"]
    assert manifest["local_state_context_feature_count"] > 0
    assert manifest["feature_profile_audit"]["retained_groups"]["local_state_context"] is True
    assert manifest["feature_profile_audit"]["amount_unit"]["amount_unit_policy"] == "as_is"


def test_v2_local_state_features_do_not_change_when_future_close_changes() -> None:
    prepared = make_prepared_policy_inputs(days=90, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2024-01-02")
    date = pd.Timestamp(prepared.close.index[-10]).normalize()
    future_shifted_close = prepared.close.copy()
    future_shifted_close.loc[future_shifted_close.index > date, "AAA"] *= 100.0
    future_shifted = replace(prepared, close=future_shifted_close)

    original_panels, columns, _manifest = build_forecast_feature_panels(
        prepared,
        [date],
        feature_profile="raw_kline_context_v2_tradeable_local_state_v1",
        max_feature_columns=256,
    )
    shifted_panels, _shifted_columns, _shifted_manifest = build_forecast_feature_panels(
        future_shifted,
        [date],
        feature_profile="raw_kline_context_v2_tradeable_local_state_v1",
        max_feature_columns=256,
    )
    local_columns = [column for column in columns if column.startswith("local_") or column.startswith("cs_rank_local_") or column.startswith("cs_z_local_")]

    np.testing.assert_allclose(
        original_panels[date][local_columns].to_numpy(dtype=float),
        shifted_panels[date][local_columns].to_numpy(dtype=float),
        rtol=1.0e-6,
        atol=1.0e-6,
        equal_nan=True,
    )
