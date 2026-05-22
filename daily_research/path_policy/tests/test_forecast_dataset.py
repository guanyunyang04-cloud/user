from __future__ import annotations

import numpy as np

from daily_research.path_policy.forecast_dataset import build_forecast_sequence_dataset
from daily_research.path_policy.labels import PATH20_CUMULATIVE_HORIZONS
from daily_research.path_policy.tests.fixtures import make_prepared_policy_inputs


def test_forecast_sequence_dataset_builds_roles_with_purge_and_train_normalization() -> None:
    prepared = make_prepared_policy_inputs(days=820, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2018-01-02")

    dataset = build_forecast_sequence_dataset(
        prepared,
        train_start_year=2018,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        lookback_days=5,
        horizon=20,
        max_samples_per_role=6,
    )

    assert dataset.x.ndim == 3
    assert dataset.x.shape[1] == 5
    assert dataset.x.shape[2] == len(dataset.feature_columns)
    assert dataset.y_daily_excess.shape == (dataset.x.shape[0], 20)
    assert PATH20_CUMULATIVE_HORIZONS == (1, 3, 5, 10, 20)
    assert dataset.y_cum_excess.shape == (dataset.x.shape[0], 5)
    assert dataset.y_rank_by_horizon.shape == (dataset.x.shape[0], 5)
    np.testing.assert_allclose(dataset.y_cum_excess[:, 0], dataset.y_daily_excess[:, 0], rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        dataset.y_cum_excess[:, 1],
        dataset.y_daily_excess[:, :3].sum(axis=1),
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    assert set(dataset.role.tolist()) == {"train", "validation", "test"}
    assert dataset.normalization_manifest["fit_role"] == "train_only"
    assert dataset.manifest["feature_profile"] == "raw_kline_context_v1"
    assert dataset.manifest["feature_count_after_cap"] <= 192
    assert "raw_open_gap_1d" in dataset.feature_columns
    assert "market_positive_share_1d" in dataset.feature_columns
    assert "benchmark_ret_20d" in dataset.feature_columns

    validation_dates = dataset.dates_by_role["validation"]
    test_dates = dataset.dates_by_role["test"]
    assert validation_dates
    assert test_dates
    assert max(validation_dates).year == 2020
    assert max(test_dates).year == 2021
    assert max(validation_dates) <= prepared.close.loc["2020"].index[-22]


def test_forecast_sequence_dataset_accepts_custom_horizon_grid_with_horizon_specific_risk() -> None:
    prepared = make_prepared_policy_inputs(days=900, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2018-01-02")
    horizons = (1, 2, 3, 5, 8, 10, 15, 20, 30)

    dataset = build_forecast_sequence_dataset(
        prepared,
        train_start_year=2018,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        lookback_days=5,
        horizon=30,
        cumulative_horizons=horizons,
        max_samples_per_role=6,
    )

    assert dataset.manifest["horizon"] == 30
    assert dataset.manifest["forecast_horizon"] == 30
    assert dataset.manifest["cumulative_horizons"] == list(horizons)
    assert dataset.manifest["role_purge_trading_days"] == 31
    assert dataset.y_daily_excess.shape == (dataset.x.shape[0], 30)
    assert dataset.y_cum_excess.shape == (dataset.x.shape[0], len(horizons))
    assert dataset.y_rank_by_horizon.shape == (dataset.x.shape[0], len(horizons))
    assert dataset.y_drawdown_by_horizon.shape == (dataset.x.shape[0], len(horizons))
    assert dataset.y_worst_by_horizon.shape == (dataset.x.shape[0], len(horizons))
    assert dataset.y_upside_by_horizon.shape == (dataset.x.shape[0], len(horizons))
    assert np.isfinite(dataset.y_drawdown_by_horizon).all()
    assert max(dataset.dates_by_role["validation"]) <= prepared.close.loc["2020"].index[-32]


def test_forecast_sequence_dataset_validation_inputs_can_use_prior_history_without_crossing_labels() -> None:
    prepared = make_prepared_policy_inputs(days=700, stocks=("AAA", "BBB", "CCC"), start_date="2019-01-02")

    dataset = build_forecast_sequence_dataset(
        prepared,
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        lookback_days=10,
        horizon=20,
        max_samples_per_role=3,
    )

    validation_rows = np.where(dataset.role == "validation")[0]
    assert validation_rows.size > 0
    first_validation = int(validation_rows[0])
    assert dataset.sequence_start_dates[first_validation].year == 2019
    assert dataset.date[first_validation].year == 2020
    assert dataset.label_end_dates[first_validation].year == 2020


def test_forecast_sequence_dataset_accepts_legacy_state_feature_profile() -> None:
    prepared = make_prepared_policy_inputs(days=700, stocks=("AAA", "BBB", "CCC"), start_date="2019-01-02")

    dataset = build_forecast_sequence_dataset(
        prepared,
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        lookback_days=10,
        horizon=20,
        feature_profile="state_v1",
        max_feature_columns=32,
        max_samples_per_role=3,
    )

    assert dataset.manifest["feature_profile"] == "state_v1"
    assert dataset.x.shape[2] <= 32
    assert "raw_open_gap_1d" not in dataset.feature_columns
