from types import SimpleNamespace

import numpy as np

from daily_research.path_policy import seq100_causal_pattern_strategy_probe as probe


def test_pending_fill_uses_signal_row_and_entry_day_price() -> None:
    market = SimpleNamespace(
        entry_filled=np.asarray(
            [
                [False, False],
                [True, False],
                [False, True],
                [False, False],
            ],
            dtype=bool,
        ),
        entry_open_raw=np.asarray(
            [
                [10.0, 20.0],
                [11.0, 21.0],
                [12.0, 22.0],
                [13.0, 23.0],
            ]
        ),
    )
    fill, prices = probe._pending_fill_mask(
        market,
        pending=np.asarray([True, True]),
        signal_abs=np.asarray([1, 2]),
        absolute_day=3,
    )
    assert fill.tolist() == [True, True]
    assert prices.tolist() == [13.0, 23.0]


def test_bh_is_monotone_in_rank() -> None:
    adjusted = probe._bh(np.asarray([0.01, 0.04, 0.20]))
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.20])


def test_hac_constant_series_has_zero_standard_error() -> None:
    mean, standard_error = probe._hac(np.ones(20), lag=5)
    assert mean == 1.0
    assert standard_error == 0.0
