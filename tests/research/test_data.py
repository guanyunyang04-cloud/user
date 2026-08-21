from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.research.data import (
    build_forward_folds,
    cutoff_audit_fields,
    cutoff_violation_count,
    date_group_sizes,
    date_relevance_labels,
    equal_date_weights,
    load_data,
)


def test_cutoff_audit_is_generic_and_reads_legacy_runs() -> None:
    assert cutoff_violation_count(cutoff_audit_fields()) == 0
    assert "forbidden_2026_read_count" not in cutoff_audit_fields()
    assert cutoff_violation_count({"forbidden_2026_read_count": 0}) == 0
    assert cutoff_violation_count({"cutoff_violation_count": 2}) == 2


def test_forward_folds_default_to_the_latest_supplied_date() -> None:
    dates = np.repeat(np.arange(20, dtype=np.int32), 2)
    labels = np.repeat(
        np.asarray(pd.bdate_range("2024-01-02", periods=20).strftime("%Y-%m-%d")),
        2,
    )
    folds = build_forward_folds(
        date_idx=dates,
        trade_date=labels,
        validation_start_date=str(labels[10]),
        fold_count=2,
        purge_days=2,
    )
    assert folds[-1]["validation_end_date"] == str(labels[-1])


def test_feature_sets_are_exact_prefixes_of_current_183() -> None:
    data = load_data()
    assert data.feature_count(158) == 158
    assert data.feature_count(183) == 183
    assert data.feature_families[:104] == ("daily_price_volume_technical",) * 104
    assert data.feature_families[104:158] == ("market_state",) * 54
    assert data.feature_families[158:] == ("same_day_5m",) * 25


def test_forward_folds_have_fixed_purge_and_contiguous_ranges() -> None:
    dates = np.repeat(np.arange(200, dtype=np.int32), 3)
    labels = np.repeat(
        np.asarray(pd.bdate_range("2019-01-02", periods=200).strftime("%Y-%m-%d")),
        3,
    )
    folds = build_forward_folds(
        date_idx=dates,
        trade_date=labels,
        validation_start_date=str(labels[180]),
        validation_end_date=str(labels[-1]),
        fold_count=5,
        purge_days=10,
    )
    assert len(folds) == 5
    assert all(item["purge_days"] == 10 for item in folds)
    assert folds[0]["training_maximum_date_idx"] < folds[0]["validation_start_date_idx"]
    assert folds[-1]["validation_end_date_idx"] == 199


def test_ranking_helpers_preserve_date_balance_and_invalid_mask() -> None:
    dates = np.repeat(np.arange(4, dtype=np.int32), 15)
    values = np.linspace(-0.2, 0.3, len(dates), dtype=np.float32)
    valid = np.ones(len(dates), dtype=bool)
    valid[[2, 19, 44]] = False
    np.testing.assert_array_equal(date_group_sizes(dates), [15, 15, 15, 15])
    weights = equal_date_weights(dates, valid)
    assert np.all(weights[~valid] == 0.0)
    assert np.isclose(weights[valid].sum(), valid.sum())
    labels = date_relevance_labels(dates, values, valid)
    assert labels.shape == values.shape
    assert np.all(labels[~valid] == 0.0)
    assert all(len(np.unique(labels[valid & (dates == date)])) == 10 for date in range(4))
