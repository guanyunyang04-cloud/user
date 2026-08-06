from __future__ import annotations

import numpy as np
import pandas as pd

from daily_research.path_policy import (
    seq100_quality_liquidity_data_certification as certification,
)


def test_feature_matrix_quality_counts_full_population_and_samples_drift(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(certification, "YEARS", (2012, 2013))
    path = tmp_path / "features.float32.dat"
    values = np.memmap(path, dtype=np.float32, mode="w+", shape=(4, 2))
    values[:] = np.asarray(
        [[1.0, np.nan], [2.0, 0.0], [3.0, np.inf], [4.0, 0.0]],
        dtype=np.float32,
    )
    values.flush()
    row_index = pd.DataFrame(
        {
            "trade_date": [
                "2012-01-04",
                "2012-01-05",
                "2013-01-04",
                "2013-01-05",
            ]
        }
    )

    annual, drift, summary = certification._feature_matrix_quality(
        values=values,
        row_index=row_index,
        feature_names=["a", "b"],
        chunk_rows=2,
        quantile_sample_rows_per_year=2,
    )

    assert len(annual) == 4
    assert len(drift) == 2
    assert summary["positive_infinity_value_count"] == 1
    assert summary["any_infinite_row_count"] == 1
    assert summary["all_missing_row_count"] == 0
    assert summary["all_missing_features"] == []
    assert set(drift["feature_type"]) == {"continuous", "constant"}
    assert "maximum_missing_rate_jump_years" in drift.columns


def test_feature_contract_rejects_future_and_adjusted_fields() -> None:
    names = [f"feature_{index}" for index in range(555)] + [
        "future_return",
        "technical_close_qfq",
    ]
    result = certification._feature_contract_checks(
        {
            "feature_groups": {certification.model.COMPACT_VARIANT: names},
            "training_performed": False,
            "feature_set_selected": False,
        }
    )

    assert result["feature_count_exact"] is True
    assert result["future_or_label_metadata_absent"] is False
    assert result["qfq_hfq_features_absent"] is False


def test_feature_contract_requires_stable_balance_semantics() -> None:
    names = [*certification.STABLE_BALANCE_FEATURES] + [
        f"feature_{index}"
        for index in range(
            certification.EXPECTED_FEATURE_COUNT
            - len(certification.STABLE_BALANCE_FEATURES)
        )
    ]

    result = certification._feature_contract_checks(
        {
            "feature_groups": {certification.model.COMPACT_VARIANT: names},
            "training_performed": False,
            "feature_set_selected": False,
        }
    )

    assert result["feature_count_exact"] is True
    assert result["stable_balance_features_present"] is True
    assert result["unstable_balance_component_features_absent"] is True


def test_drift_separates_event_rates_from_near_constant_scale(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(certification, "YEARS", (2012, 2013))
    row_count = 120
    path = tmp_path / "features.float32.dat"
    values = np.memmap(path, dtype=np.float32, mode="w+", shape=(row_count, 3))
    binary = np.r_[np.zeros(60), np.ones(60)]
    sparse = np.zeros(row_count)
    sparse[0] = 5.0
    sparse[60:66] = 5.0
    near_constant = np.r_[np.full(60, 100.0), np.full(60, 101.0)]
    values[:] = np.column_stack([binary, sparse, near_constant]).astype(np.float32)
    values.flush()
    row_index = pd.DataFrame(
        {
            "trade_date": ["2012-01-04"] * 60 + ["2013-01-04"] * 60,
        }
    )

    _, drift, _ = certification._feature_matrix_quality(
        values=values,
        row_index=row_index,
        feature_names=["binary", "sparse", "near_constant"],
        quantile_sample_rows_per_year=60,
    )
    indexed = drift.set_index("feature")

    assert indexed.loc["binary", "feature_type"] == "binary"
    assert bool(indexed.loc["binary", "event_rate_shift_flag"])
    assert indexed.loc["binary", "maximum_event_rate_jump_years"] == "2012->2013"
    assert indexed.loc["sparse", "feature_type"] == "sparse"
    assert bool(indexed.loc["sparse", "event_rate_shift_flag"])
    assert indexed.loc["near_constant", "feature_type"] == "continuous"
    assert (
        indexed.loc[
            "near_constant", "maximum_adjacent_median_shift_robust_scale"
        ]
        < 2.0
    )
    assert not bool(
        indexed.loc["near_constant", "conditional_distribution_shift_flag"]
    )
