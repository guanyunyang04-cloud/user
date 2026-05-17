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

    validation_dates = dataset.dates_by_role["validation"]
    test_dates = dataset.dates_by_role["test"]
    assert validation_dates
    assert test_dates
    assert max(validation_dates).year == 2020
    assert max(test_dates).year == 2021
    assert max(validation_dates) <= prepared.close.loc["2020"].index[-22]


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
