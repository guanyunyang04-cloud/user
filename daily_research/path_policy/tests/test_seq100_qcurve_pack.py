from __future__ import annotations

import numpy as np

from daily_research.path_policy.seq100_qcurve_pack import _ema, _lag_ratio, _ratio, _rolling_vwap


def test_ma_helpers_are_pit_and_dimension_stable() -> None:
    values = np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    ema = _ema(values, 3)
    assert ema.shape == values.shape
    assert ema[:, 0].tolist() == [1.0, 1.5, 2.25, 3.125]
    assert np.isnan(_lag_ratio(ema, 2)[:2]).all()
    assert _ratio(ema, ema).tolist() == [[0.0], [0.0], [0.0], [0.0]]


def test_rolling_vwap_uses_only_current_and_past_rows() -> None:
    amount = np.asarray([[10.0], [40.0], [90.0], [160.0]])
    volume = np.asarray([[1.0], [2.0], [3.0], [4.0]])
    vwap = _rolling_vwap(amount, volume, 2)
    assert vwap[:, 0].tolist() == [10.0, 50.0 / 3.0, 26.0, 250.0 / 7.0]
