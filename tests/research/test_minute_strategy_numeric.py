from __future__ import annotations

import numpy as np

from quantlab.research.minute_strategy_numeric import (
    NUMERIC_FEATURE_NAMES,
    compute_live_ma_arithmetic_numpy,
    compute_live_ma_arithmetic_torch,
)


def _inputs() -> tuple[np.ndarray, ...]:
    close = np.asarray([10.0, 9.9, 10.2], dtype=np.float32)
    return (
        np.asarray([90.0, 90.0, 90.0], dtype=np.float32),
        close,
        np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
        np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
        np.asarray([10.01, 9.85, 9.9], dtype=np.float32),
        np.asarray([10.03, 9.95, 10.3], dtype=np.float32),
    )


def test_numpy_numeric_kernel_matches_live_ma_formulas() -> None:
    prior_sum, close, period, factor, intersection, low, high = _inputs()

    result = compute_live_ma_arithmetic_numpy(
        prior_sum,
        close,
        period,
        factor,
        intersection,
        low,
        high,
    )

    assert tuple(result) == NUMERIC_FEATURE_NAMES
    expected_live = (prior_sum + close) / period
    np.testing.assert_allclose(result["live_ma_adjusted"], expected_live)
    np.testing.assert_allclose(result["live_ma"], expected_live / factor)
    np.testing.assert_allclose(
        result["close_to_intersection_bps"],
        (close / intersection - 1.0) * 10_000.0,
    )
    np.testing.assert_array_equal(result["touched_now"], [False, False, True])
    np.testing.assert_array_equal(
        result["close_below_intersection"], [False, True, False]
    )


def test_numeric_kernel_broadcasts_scalar_period_and_factor() -> None:
    prior_sum, close, _period, _factor, intersection, low, high = _inputs()

    result = compute_live_ma_arithmetic_numpy(
        prior_sum,
        close,
        10.0,
        1.0,
        intersection,
        low,
        high,
    )

    assert all(value.shape == close.shape for value in result.values())


def test_torch_cpu_numeric_kernel_matches_numpy() -> None:
    import torch

    inputs = _inputs()
    expected = compute_live_ma_arithmetic_numpy(*inputs)
    tensors = [torch.from_numpy(value) for value in inputs]

    actual = compute_live_ma_arithmetic_torch(*tensors)

    for name in NUMERIC_FEATURE_NAMES:
        observed = actual[name].detach().cpu().numpy()
        if expected[name].dtype == np.bool_:
            np.testing.assert_array_equal(observed, expected[name])
        else:
            np.testing.assert_allclose(observed, expected[name], rtol=1e-6, atol=1e-5)
