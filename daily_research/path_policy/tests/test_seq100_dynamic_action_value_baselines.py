from __future__ import annotations

import numpy as np

from daily_research.path_policy import seq100_dynamic_action_value_baselines as study


def _experience() -> study.AlignedExperience:
    coordinates = np.asarray(
        [
            [0.2, 0.1, 0.8],
            [0.2, 0.9, 0.2],
            [0.8, 0.2, 0.7],
            [0.8, 0.8, 0.3],
        ],
        dtype=np.float64,
    )
    return study.AlignedExperience(
        cost_scenario="base",
        signal_year=2012,
        as_of_year=2012,
        version_role="signal_year_end",
        coordinate_positions=np.arange(4, dtype=np.int64),
        coordinates=coordinates,
        trade_date=np.asarray(
            ["2012-01-03", "2012-01-03", "2012-01-04", "2012-01-04"]
        ),
        date_idx=np.asarray([1, 1, 2, 2], dtype=np.int32),
        symbol_idx=np.asarray([0, 1, 0, 1], dtype=np.int32),
        target=np.asarray([-1.0, 1.0, 0.0, 2.0], dtype=np.float64),
        selected=np.asarray([False, True, False, True]),
        sessions_after_resolution=np.asarray([10, 11, 12, 13], dtype=np.int32),
        source_rows=4,
        maturity_rows=4,
        missing_coordinate_rows=0,
    )


def test_causal_expanding_percentile_does_not_read_future() -> None:
    original = np.asarray([3.0, 1.0, 2.0, 5.0], dtype=np.float64)
    changed = original.copy()
    changed[-1] = -1000.0
    left = study._causal_expanding_percentile(original)
    right = study._causal_expanding_percentile(changed)
    assert np.allclose(left[:3], right[:3])
    assert np.allclose(left[:3], [0.5, 0.25, 0.5])


def test_same_date_percentile_preserves_ties_and_missing() -> None:
    values = np.asarray([1.0, 1.0, 3.0, np.nan, 2.0])
    dates = np.asarray([1, 1, 1, 2, 2])
    result = study._same_date_percentile(values, dates)
    assert np.allclose(result[:3], [1.0 / 3.0, 1.0 / 3.0, 5.0 / 6.0])
    assert np.isnan(result[3])
    assert result[4] == 0.5


def test_accumulator_add_remove_is_reversible() -> None:
    accumulator = study.ActionValueAccumulator(
        market_indices=np.asarray([0]),
        stock_indices=np.asarray([1, 2]),
        matched_indices=np.asarray([0, 1]),
        univariate_bins=2,
        matched_bins=2,
    )
    experience = _experience()
    audit = accumulator.apply(experience, sign=1)
    assert audit["rows"] == 4
    assert audit["dates"] == 2
    assert accumulator.training_rows == 4
    assert accumulator.global_date_count == 2
    accumulator.apply(experience, sign=-1)
    assert accumulator.training_rows == 0
    assert accumulator.global_date_count == 0
    for values in (
        accumulator.market_count,
        accumulator.market_sum,
        accumulator.stock_weight,
        accumulator.stock_sum,
        accumulator.cell_weight,
        accumulator.cell_sum,
    ):
        assert np.allclose(values, 0.0, atol=1.0e-12)


def test_snapshot_and_prediction_are_finite() -> None:
    accumulator = study.ActionValueAccumulator(
        market_indices=np.asarray([0]),
        stock_indices=np.asarray([1, 2]),
        matched_indices=np.asarray([0, 1]),
        univariate_bins=2,
        matched_bins=2,
    )
    experience = _experience()
    accumulator.apply(experience, sign=1)
    contract = {
        "estimators": {
            "market_univariate_smoothing_date_equivalents": 2.0,
            "stock_univariate_smoothing_date_equivalents": 2.0,
            "matched_cell_smoothing_date_equivalents": 3.0,
        }
    }
    snapshot = study.build_snapshot(
        accumulator=accumulator,
        study=contract,
        coordinate_names=["market", "path", "activity"],
        market_indices=np.asarray([0]),
        stock_indices=np.asarray([1, 2]),
        families={"path": [0], "activity": [1]},
        matched_indices=np.asarray([0, 1]),
        training_cutoff_year=2012,
        maximum_label_as_of_year=2012,
        maximum_signal_year=2012,
    )
    predictions = study.predict_from_snapshot(experience.coordinates, snapshot)
    for name in (
        "predicted_prior",
        "predicted_market",
        "predicted_additive",
        "predicted_matched_history",
        "effect_path",
        "effect_activity",
    ):
        assert np.isfinite(predictions[name]).all()
    assert np.all(predictions["matched_cell_weight"] >= 0.0)


def test_latest_record_selection_respects_cutoff() -> None:
    records = {
        ("base", 2012, 2012): {
            "cost_scenario": "base",
            "signal_year": 2012,
            "as_of_year": 2012,
        },
        ("base", 2012, 2013): {
            "cost_scenario": "base",
            "signal_year": 2012,
            "as_of_year": 2013,
        },
        ("base", 2013, 2013): {
            "cost_scenario": "base",
            "signal_year": 2013,
            "as_of_year": 2013,
        },
    }
    first = study.select_latest_records(
        version_records=records,
        cost_scenario="base",
        cutoff_year=2012,
        maximum_signal_year=2012,
    )
    assert first[2012]["as_of_year"] == 2012
    second = study.select_latest_records(
        version_records=records,
        cost_scenario="base",
        cutoff_year=2013,
        maximum_signal_year=2013,
    )
    assert second[2012]["as_of_year"] == 2013
    assert second[2013]["as_of_year"] == 2013


def test_study_contract_loads() -> None:
    loaded = study.load_study()
    assert loaded["study_id"] == study.STUDY_ID
    assert loaded["target"]["minimum_sessions_after_resolution"] == 10

