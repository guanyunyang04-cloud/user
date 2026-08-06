from __future__ import annotations

import math

import numpy as np
import pytest

from daily_research.path_policy import seq100_mathematical_market_model as market


def test_horizon_targets_use_only_future_observations() -> None:
    values = np.arange(1.0, 7.0)
    targets = market.build_horizon_targets(values, [1, 2, 3])
    np.testing.assert_allclose(targets[1][:5], [2, 3, 4, 5, 6])
    np.testing.assert_allclose(targets[2][:4], [5, 7, 9, 11])
    np.testing.assert_allclose(targets[3][:3], [9, 12, 15])
    assert np.isnan(targets[1][-1])
    assert np.isnan(targets[3][-1])


def test_robust_standardizer_handles_constant_columns() -> None:
    x = np.array([[1.0, 4.0], [1.0, 8.0], [1.0, 12.0]])
    location, scale = market._fit_standardizer(x)
    z = market._apply_standardizer(x, location, scale)
    assert np.all(np.isfinite(z))
    assert np.all(z[:, 0] == 0.0)
    assert np.allclose(np.median(z[:, 1]), 0.0)


def test_crps_is_nonnegative_and_small_at_the_predictive_center() -> None:
    values = np.array([0.0, 1.0])
    loc = np.array([0.0, 1.0])
    scale = np.array([1.0, 1.0])
    score = market.crps_from_quantiles(values, loc, scale, math.inf, n=256)
    assert np.all(score >= 0.0)
    assert score[0] == pytest.approx(score[1])


def test_joint_static_forecast_has_positive_definite_covariance() -> None:
    y = np.column_stack([np.arange(30.0), np.arange(30.0) ** 2 / 100.0])
    forecast = market._fit_joint_static_gaussian(y, 4)
    assert forecast.loc.shape == (4, 2)
    assert np.all(np.linalg.eigvalsh(forecast.covariance) > 0.0)


def test_market_panel_is_current_certified_input() -> None:
    panel = market.load_market_panel(include_minute_state=True)
    assert len(panel.dates) == 3400
    assert panel.state.shape == (3400, 54)
    assert panel.long_memory_state is not None
    assert panel.long_memory_state.shape == (3400, 7)
    assert panel.dates[0].year == 2012
    assert panel.dates[-1].year == 2025
    assert np.all(np.isfinite(panel.log_return))


def test_long_memory_state_is_causal() -> None:
    minute = np.ones((80, 5), dtype=np.float64)
    minute[:, 0] = np.linspace(0.01, 0.03, len(minute))
    minute[:, 1] = 0.01
    minute[:, 2] = 0.02
    minute[:, 3] = 0.7
    minute[:, 4] = 0.001
    first = market.build_long_memory_state(minute)
    changed = minute.copy()
    changed[60:] *= 9.0
    second = market.build_long_memory_state(changed)
    np.testing.assert_allclose(first[:60], second[:60], equal_nan=True)


def test_iterated_kernel_joint_covariance_matches_marginals() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(240, 6))
    g = rng.normal(scale=0.01, size=240)
    marginal, joint = market._fit_iterated_var_models(
        x[:-10],
        g[:-10],
        x[-10:],
        g[-10:],
        horizons=market.HORIZONS,
        components=2,
    )
    diagonal = np.diag(joint.covariance)
    for position, horizon in enumerate(market.HORIZONS):
        np.testing.assert_allclose(
            marginal[horizon].scale,
            np.sqrt(diagonal[position]),
            rtol=1e-10,
            atol=1e-12,
        )
    assert np.all(np.linalg.eigvalsh(joint.covariance) > 0.0)


def test_reality_check_is_conservative_for_identical_models() -> None:
    rng = np.random.default_rng(9)
    values = rng.normal(size=(120, 3))
    result = market._reality_check(
        values,
        ("a", "b", "c"),
        block_length=10,
        repetitions=200,
        seed=11,
    )
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["winner"] in {"a", "b", "c"}
