from __future__ import annotations

import numpy as np
import torch

from quantlab.research.data import load_data
from quantlab.research.sequence import (
    LOOKBACK,
    RAW_CHANNEL_COUNT,
    RawSequenceBuilder,
    RawSequenceModel,
    lookback_indices,
)


def test_lookback_indices_are_causal_and_include_signal_day() -> None:
    dates = np.asarray([100, 250], dtype=np.int32)
    grid = lookback_indices(dates)
    assert grid.shape == (2, LOOKBACK)
    np.testing.assert_array_equal(grid[:, -1], dates)
    assert np.all(grid <= dates[:, None])
    assert np.all(np.diff(grid, axis=1) == 1)


def test_raw_sequence_builder_produces_finite_expected_shapes() -> None:
    data = load_data()
    rows = np.asarray(
        [0, len(data.row_index) // 2, len(data.row_index) - 100], dtype=np.int64
    )
    builder = RawSequenceBuilder(
        data,
        market_mean=np.zeros(14, dtype=np.float32),
        market_std=np.ones(14, dtype=np.float32),
    )
    sequence, market = builder.build(rows)
    assert sequence.shape == (3, LOOKBACK, RAW_CHANNEL_COUNT)
    assert market.shape == (3, 14)
    assert np.isfinite(sequence).all()
    assert np.isfinite(market).all()


def test_raw_sequence_model_output_shape() -> None:
    model = RawSequenceModel()
    sequence = torch.zeros(5, LOOKBACK, RAW_CHANNEL_COUNT)
    market = torch.zeros(5, 14)
    assert model(sequence, market).shape == (5,)
