from __future__ import annotations

import numpy as np

from daily_research.path_policy import seq100_dynamic_action_distribution as study
from daily_research.path_policy import seq100_dynamic_action_value_baselines as baseline


def _experience() -> baseline.AlignedExperience:
    coordinates = np.asarray(
        [
            [0.2, 0.1, 0.8, 0.2],
            [0.2, 0.9, 0.2, 0.8],
            [0.8, 0.2, 0.7, 0.3],
            [0.8, 0.8, 0.3, 0.7],
        ],
        dtype=np.float64,
    )
    return baseline.AlignedExperience(
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


def test_activity_rank_is_same_date_and_future_date_independent() -> None:
    values = np.asarray(
        [
            [0.1, 0.2],
            [0.7, 0.8],
            [0.3, 0.4],
            [0.5, 0.6],
        ],
        dtype=np.float64,
    )
    dates = np.asarray([1, 1, 2, 2], dtype=np.int32)
    changed = values.copy()
    changed[2:] = [[100.0, 100.0], [-100.0, -100.0]]
    left = study._activity_rank(
        values,
        dates,
        coordinate_indices=np.asarray([0, 1]),
        minimum_finite=2,
    )
    right = study._activity_rank(
        changed,
        dates,
        coordinate_indices=np.asarray([0, 1]),
        minimum_finite=2,
    )
    assert np.allclose(left[:2], right[:2])
    assert np.array_equal(study._gate_mask(left, 0.7), [False, True, False, True])


def test_distribution_accumulator_add_remove_is_reversible() -> None:
    accumulator = study.DistributionAccumulator(
        market_indices=np.asarray([0]),
        stock_indices=np.asarray([1, 2, 3]),
        matched_indices=np.asarray([0, 1]),
        univariate_bins=2,
        matched_bins=2,
    )
    experience = _experience()
    audit = accumulator.apply(experience, sign=1)
    assert audit["rows"] == 4
    assert accumulator.training_rows == 4
    assert np.isclose(accumulator.global_sum[3], 1.0)
    accumulator.apply(experience, sign=-1)
    assert accumulator.training_rows == 0
    assert accumulator.global_date_count == 0
    for values in (
        accumulator.global_sum,
        accumulator.market_count,
        accumulator.market_sum,
        accumulator.stock_weight,
        accumulator.stock_sum,
        accumulator.cell_weight,
        accumulator.cell_sum,
    ):
        assert np.allclose(values, 0.0, atol=1.0e-12)


def test_snapshot_predictions_are_coherent_and_bounded() -> None:
    accumulator = study.DistributionAccumulator(
        market_indices=np.asarray([0]),
        stock_indices=np.asarray([1, 2, 3]),
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
        coordinate_names=["market", "path", "activity", "intraday"],
        market_indices=np.asarray([0]),
        stock_indices=np.asarray([1, 2, 3]),
        families={"path": [0], "activity": [1], "intraday": [2]},
        matched_indices=np.asarray([0, 1]),
        scope="all",
        training_cutoff_year=2012,
        maximum_label_as_of_year=2012,
        maximum_signal_year=2012,
    )
    predictions = study.predict_from_snapshot(experience.coordinates, snapshot)
    for model in study.MODEL_NAMES:
        probability = predictions[f"predicted_{model}_positive_probability"]
        upside = predictions[f"predicted_{model}_upside_component"]
        downside = predictions[f"predicted_{model}_downside_component"]
        expected = predictions[f"predicted_{model}_expected_value"]
        assert np.isfinite(probability).all()
        assert np.all((probability > 0.0) & (probability < 1.0))
        assert np.all(upside >= 0.0)
        assert np.all(downside >= 0.0)
        assert np.allclose(expected, upside - downside)


def test_binary_auc_and_bh_are_well_formed() -> None:
    auc = study._binary_auc(
        np.asarray([False, False, True, True]),
        np.asarray([0.1, 0.2, 0.8, 0.9]),
    )
    assert auc == 1.0
    adjusted = study._benjamini_hochberg(np.asarray([0.01, 0.04, 0.03, np.nan]))
    assert np.allclose(adjusted[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])


def test_study_contract_loads() -> None:
    loaded = study.load_study()
    assert loaded["study_id"] == study.STUDY_ID
    assert loaded["activity_gate"]["primary_training_scope"] == "top20"
    assert len(loaded["coordinates"]["specifications"]) == 66
