from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import seq100_causal_retest_state as study_module


def _synthetic_state() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    close = np.asarray([99.0, 102.0, 101.0, 100.8, 102.5, 101.2, 99.5])
    high = close * np.asarray([1.01, 1.02, 1.01, 1.01, 1.02, 1.01, 1.01])
    low = close * np.asarray([0.99, 0.99, 0.985, 0.99, 0.995, 0.985, 0.99])
    unit = 0.01
    boundary = np.log(100.0)
    upper_distance = np.full(len(close), np.nan)
    upper_distance[1:] = (np.log(close[1:]) - boundary) / unit
    path = {
        "chan_breakout_event": np.asarray([0, 1, 0, 0, 0, 0, 0], dtype=np.int8),
        "chan_breakout_active": np.asarray([0, 1, 1, 1, 1, 1, 0], dtype=np.int8),
        "chan_retest_event": np.asarray([0, 0, 1, 1, 0, 1, 0], dtype=np.int8),
        "chan_reentry_event": np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.int8),
        "causal_scale_unit": np.full(len(close), unit),
        "chan_center_distance_upper_units": upper_distance,
        "chan_center_width_units": np.full(len(close), 2.5),
        "chan_center_age": np.full(len(close), 12, dtype=np.int32),
    }
    values = {
        "adj_high": high,
        "adj_low": low,
        "adj_close": close,
        "amount": np.asarray([80, 100, 75, 70, 110, 65, 90], dtype=float) * 1e6,
        "date_indices": np.arange(100, 107, dtype=np.int64),
    }
    return values, path


def test_load_study_rejects_fixed_horizon() -> None:
    study = study_module.load_study()
    broken = copy.deepcopy(study)
    broken["state"]["fixed_holding_horizon_used"] = True
    path = study_module.WORKSPACE_ROOT / "tmp_invalid_retest_state_study.json"
    try:
        study_module._write_json(path, broken)
        with pytest.raises(ValueError, match="fixed_horizon"):
            study_module.load_study(path)
    finally:
        path.unlink(missing_ok=True)


def test_retest_state_locks_boundary_and_counts_touch_episodes() -> None:
    values, path = _synthetic_state()
    state = study_module.derive_causal_retest_state(
        **values, path_features=path
    )
    assert state["up_breakout_event"][1]
    assert state["sessions_since_up_breakout"][5] == 4
    assert state["market_days_since_up_breakout"][5] == 4
    assert state["up_retest_bar_count"][3] == 2
    assert state["up_retest_episode_count"][3] == 1
    assert state["up_retest_episode_count"][5] == 2
    assert state["up_repeated_retest"][5]
    assert state["center_age_at_breakout"][5] == 12
    assert state["center_width_units_at_breakout"][5] == pytest.approx(2.5)
    expected_strength = (np.log(102.0) - np.log(100.0)) / 0.01
    assert state["breakout_strength_units"][5] == pytest.approx(expected_strength)


def test_reentry_row_is_emitted_then_state_terminates() -> None:
    values, path = _synthetic_state()
    state = study_module.derive_causal_retest_state(
        **values, path_features=path
    )
    assert state["up_reentry_event"][6]
    assert not state["up_breakout_active"][6]
    assert np.isfinite(state["boundary_close_distance_units"][6])

    extended_values = {
        key: np.append(value, value[-1]) for key, value in values.items()
    }
    extended_values["date_indices"][-1] += 1
    extended_path = {
        key: np.append(value, 0 if value.dtype.kind in "biu" else value[-1])
        for key, value in path.items()
    }
    extended_path["chan_center_distance_upper_units"][-1] = np.nan
    result = study_module.derive_causal_retest_state(
        **extended_values, path_features=extended_path
    )
    assert result["sessions_since_up_breakout"][-1] == -1
    assert not np.isfinite(result["boundary_close_distance_units"][-1])


def test_retest_state_is_prefix_invariant() -> None:
    values, path = _synthetic_state()
    full = study_module.derive_causal_retest_state(
        **values, path_features=path
    )
    for cutoff in range(1, len(values["adj_close"]) + 1):
        prefix_values = {key: value[:cutoff] for key, value in values.items()}
        prefix_path = {key: value[:cutoff] for key, value in path.items()}
        prefix = study_module.derive_causal_retest_state(
            **prefix_values, path_features=prefix_path
        )
        for name in study_module.STATE_BOOLEAN_FEATURES:
            assert np.array_equal(prefix[name], full[name][:cutoff])
        for name in study_module.STATE_INTEGER_FEATURES:
            assert np.array_equal(prefix[name], full[name][:cutoff])
        for name in study_module.STATE_FLOAT_FEATURES:
            assert np.allclose(
                prefix[name], full[name][:cutoff], equal_nan=True, atol=0.0, rtol=0.0
            )


def test_empirical_marginal_rank_handles_ties_and_missing() -> None:
    values = np.asarray([[1.0, np.nan], [2.0, 5.0], [2.0, 7.0], [4.0, 9.0]])
    transformer = study_module.EmpiricalMarginalRank()
    transformed = transformer.fit_transform(values)
    assert transformed[0, 0] == pytest.approx(0.125)
    assert transformed[1, 0] == pytest.approx(0.5)
    assert transformed[2, 0] == pytest.approx(0.5)
    assert transformed[0, 1] == pytest.approx(0.5)
    new = transformer.transform(np.asarray([[3.0, 8.0]]))
    assert new[0, 0] == pytest.approx(0.75)
    assert new[0, 1] == pytest.approx(2.0 / 3.0)


def test_date_equal_weights_give_each_date_unit_mass() -> None:
    dates = np.asarray([1, 1, 1, 2, 2, 3])
    weights = study_module._date_equal_weights(dates)
    totals = pd.Series(weights).groupby(pd.Series(dates)).sum()
    assert np.allclose(totals.to_numpy(), 1.0)


def test_scope_masks_require_activity_and_frozen_confirmation() -> None:
    panel = pd.DataFrame(
        {
            "up_retest_event": [True, True, False, False],
            "up_breakout_event": [False, False, True, True],
        }
    )
    names = ["price_amount_correlation_5d_rank", "minute_last_30m_return_rank"]
    coordinates = np.asarray([[0.2, 0.2], [0.2, 0.2], [0.7, 0.2], [0.7, 0.7]])
    masks = study_module._scope_masks(
        panel_frame=panel,
        positions=np.arange(4),
        coordinate_values=coordinates,
        coordinate_names=names,
        activity_rank=np.asarray([0.9, 0.7, 0.9, 0.9]),
        threshold=0.8,
    )
    assert masks["up_retest_event"].tolist() == [True, False, False, False]
    assert masks["up_price_amount_confirmed_breakout"].tolist() == [
        False,
        False,
        True,
        False,
    ]


def test_spline_and_neighbor_models_produce_projected_moments() -> None:
    random = np.random.default_rng(20260808)
    rows = 160
    features = random.normal(size=(rows, 4))
    target = 0.03 * features[:, 0] - 0.02 * np.square(features[:, 1])
    data = study_module.ScopeData(
        scope="up_retest_event",
        feature_names=("a", "b", "c", "d"),
        input_row_idx=np.arange(rows),
        trade_date=np.asarray([f"2012-01-{1 + index // 8:02d}" for index in range(rows)]),
        date_idx=np.repeat(np.arange(20), 8),
        symbol_idx=np.tile(np.arange(8), 20),
        features=features,
        moments=study_module._target_moments(target),
    )
    fitted = study_module.fit_scope_models(
        data, study=study_module.load_study(), threads=1
    )
    assert fitted is not None
    predicted = study_module.predict_scope_models(
        fitted, features[:7], study=study_module.load_study()
    )
    for model in ("causal_rank_spline", "historical_neighbor"):
        assert np.isfinite(predicted[model]["expected_value"]).all()
        assert (predicted[model]["positive_probability"] > 0.0).all()
        assert (predicted[model]["positive_probability"] < 1.0).all()
        assert np.allclose(
            predicted[model]["expected_value"],
            predicted[model]["upside_component"]
            - predicted[model]["downside_component"],
        )
