"""Backend-neutral numeric kernels used by minute-strategy research.

The minute strategy pipeline is intentionally still CPU/data-frame driven.  A
small part of the live-MA feature construction is pure element-wise arithmetic,
though, and can be measured independently before considering a GPU path.  This
module contains that arithmetic twice:

* :func:`compute_live_ma_arithmetic_numpy` is the reference implementation;
* :func:`compute_live_ma_arithmetic_torch` is an optional Torch implementation
  that can run on CPU or CUDA without importing Torch at module import time.

The functions do not perform grouping, rolling windows, Parquet I/O, or order
state transitions.  Those operations remain outside the GPU candidate surface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

NUMERIC_KERNEL_SCHEMA = "quantlab.minute_strategy_numeric_kernel/1"

NUMERIC_FEATURE_NAMES = (
    "live_ma_adjusted",
    "live_ma",
    "close_to_live_ma_bps",
    "close_to_intersection_bps",
    "range_distance_to_intersection_bps",
    "touched_now",
    "close_below_intersection",
)


def _broadcast_numpy_inputs(
    values: Sequence[Any],
    *,
    dtype: np.dtype[Any] | type[Any] = np.float32,
) -> tuple[np.ndarray, ...]:
    arrays = [np.asarray(value, dtype=dtype) for value in values]
    broadcast = np.broadcast_arrays(*arrays)
    return tuple(np.ascontiguousarray(value) for value in broadcast)


def _require_feature_map(result: Mapping[str, Any]) -> None:
    missing = [name for name in NUMERIC_FEATURE_NAMES if name not in result]
    if missing:
        raise ValueError(f"minute_strategy_numeric_features_missing:{','.join(missing)}")


def compute_live_ma_arithmetic_numpy(
    prior_close_sum_adjusted: Any,
    adjusted_close: Any,
    ma_period: Any,
    adjust_factor: Any,
    causal_intersection_adjusted: Any,
    adjusted_low: Any,
    adjusted_high: Any,
    *,
    dtype: np.dtype[Any] | type[Any] = np.float32,
) -> dict[str, np.ndarray]:
    """Compute the element-wise causal live-MA features with NumPy.

    Inputs follow the same definitions as ``minute_ma``.  Broadcasting is
    supported so a scalar adjustment factor or MA period can be used for a
    whole vector.  IEEE NaN/inf behavior is retained intentionally; upstream
    validation decides whether a row is eligible for a strategy signal.
    """

    (
        prior_sum,
        close,
        period,
        factor,
        intersection,
        low,
        high,
    ) = _broadcast_numpy_inputs(
        (
            prior_close_sum_adjusted,
            adjusted_close,
            ma_period,
            adjust_factor,
            causal_intersection_adjusted,
            adjusted_low,
            adjusted_high,
        ),
        dtype=dtype,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        live_ma_adjusted = (prior_sum + close) / period
        live_ma = live_ma_adjusted / factor
        close_to_live_ma_bps = (close / live_ma_adjusted - 1.0) * 10_000.0
        close_to_intersection_bps = (close / intersection - 1.0) * 10_000.0
        range_distance_to_intersection_bps = np.select(
            [low > intersection, high < intersection],
            [
                (low / intersection - 1.0) * 10_000.0,
                (high / intersection - 1.0) * 10_000.0,
            ],
            default=0.0,
        )
    result = {
        "live_ma_adjusted": live_ma_adjusted,
        "live_ma": live_ma,
        "close_to_live_ma_bps": close_to_live_ma_bps,
        "close_to_intersection_bps": close_to_intersection_bps,
        "range_distance_to_intersection_bps": range_distance_to_intersection_bps,
        "touched_now": (low <= intersection) & (high >= intersection),
        "close_below_intersection": close < intersection,
    }
    _require_feature_map(result)
    return result


def compute_live_ma_arithmetic_torch(
    prior_close_sum_adjusted: Any,
    adjusted_close: Any,
    ma_period: Any,
    adjust_factor: Any,
    causal_intersection_adjusted: Any,
    adjusted_low: Any,
    adjusted_high: Any,
    *,
    dtype: Any | None = None,
) -> dict[str, Any]:
    """Compute the same arithmetic with Torch tensors.

    Torch is imported lazily so importing the research CLI does not pull in
    the optional ML stack.  Inputs may be tensors on any one device or array-
    like values; callers should place tensors on CUDA before calling this
    function when measuring the GPU kernel itself.
    """

    import torch

    selected_dtype = dtype or torch.float32
    tensors = [
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=selected_dtype)
        for value in (
            prior_close_sum_adjusted,
            adjusted_close,
            ma_period,
            adjust_factor,
            causal_intersection_adjusted,
            adjusted_low,
            adjusted_high,
        )
    ]
    if any(value.device != tensors[0].device for value in tensors[1:]):
        raise ValueError("minute_strategy_numeric_tensor_devices_must_match")
    tensors = [value.to(dtype=selected_dtype) for value in tensors]
    (
        prior_sum,
        close,
        period,
        factor,
        intersection,
        low,
        high,
    ) = torch.broadcast_tensors(*tensors)
    live_ma_adjusted = (prior_sum + close) / period
    live_ma = live_ma_adjusted / factor
    close_to_live_ma_bps = (close / live_ma_adjusted - 1.0) * 10_000.0
    close_to_intersection_bps = (close / intersection - 1.0) * 10_000.0
    range_distance_to_intersection_bps = torch.where(
        low > intersection,
        (low / intersection - 1.0) * 10_000.0,
        torch.where(
            high < intersection,
            (high / intersection - 1.0) * 10_000.0,
            torch.zeros_like(intersection),
        ),
    )
    result = {
        "live_ma_adjusted": live_ma_adjusted,
        "live_ma": live_ma,
        "close_to_live_ma_bps": close_to_live_ma_bps,
        "close_to_intersection_bps": close_to_intersection_bps,
        "range_distance_to_intersection_bps": range_distance_to_intersection_bps,
        "touched_now": (low <= intersection) & (high >= intersection),
        "close_below_intersection": close < intersection,
    }
    _require_feature_map(result)
    return result


def torch_cuda_available() -> bool:
    """Return whether the optional Torch installation exposes CUDA."""

    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


__all__ = [
    "NUMERIC_FEATURE_NAMES",
    "NUMERIC_KERNEL_SCHEMA",
    "compute_live_ma_arithmetic_numpy",
    "compute_live_ma_arithmetic_torch",
    "torch_cuda_available",
]
