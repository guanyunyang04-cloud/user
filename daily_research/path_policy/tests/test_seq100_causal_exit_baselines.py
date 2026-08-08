from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from daily_research.path_policy import seq100_causal_exit_baselines as baselines
from daily_research.path_policy import seq100_causal_pattern_strategy_probe as probe
from daily_research.path_policy import seq100_exit_stopping_ceiling as ceiling


def _structure(day_count: int, symbol_count: int) -> probe.StructureArrays:
    boolean = lambda: np.zeros((day_count, symbol_count), dtype=bool)
    integer = lambda: np.zeros((day_count, symbol_count), dtype=np.int8)
    return probe.StructureArrays(
        retest=boolean(),
        exhaustion=boolean(),
        nested_reacceleration=boolean(),
        breakout=boolean(),
        reentry=boolean(),
        breakout_active=integer(),
        small_event=integer(),
        medium_event=integer(),
        medium_mode=integer(),
        large_mode=integer(),
        rows=0,
        duplicate_rows=0,
        forbidden_rows=0,
    )


def test_load_study_contract() -> None:
    study = baselines.load_study()
    assert study["source"]["quality_pool_name"] == "quality_liquidity_pit"
    assert set(study["policies"]) == set(baselines.POLICY_NAMES)


def test_entry_records_use_all_causal_signals_and_real_fills() -> None:
    dates = np.asarray(
        [
            "2025-11-25",
            "2025-11-26",
            "2025-11-27",
            "2025-11-28",
            "2025-12-01",
            "2025-12-02",
        ],
        dtype=object,
    )
    market = SimpleNamespace(
        date_values=dates,
        symbol_values=np.asarray(["000001.SZ", "000002.SZ"], dtype=object),
        start_idx=0,
        quality_mask=np.ones((6, 2), dtype=bool),
        entry_filled=np.ones((6, 2), dtype=bool),
        entry_open_raw=np.full((6, 2), 10.0, dtype=np.float32),
    )
    structure = _structure(6, 2)
    structure.retest[0, 0] = True
    structure.nested_reacceleration[1, 1] = True
    structure.breakout[2, 0] = True
    market.entry_filled[2, 0] = False
    entries = baselines._entry_records(
        market=market,
        structure=structure,
        liquidation_date="2025-12-01",
    )
    assert entries[["entry_pattern", "signal_date_idx", "symbol_idx"]].to_dict(
        "records"
    ) == [
        {"entry_pattern": "nested_reacceleration", "signal_date_idx": 1, "symbol_idx": 1},
        {"entry_pattern": "retest", "signal_date_idx": 0, "symbol_idx": 0},
    ]
    assert np.array_equal(
        entries["entry_date_idx"].to_numpy(int),
        entries["signal_date_idx"].to_numpy(int) + 1,
    )


def test_risk_scale_uses_only_history_through_signal() -> None:
    close = np.asarray(
        [[100.0], [101.0], [100.0], [102.0], [101.0], [103.0], [999.0]],
        dtype=np.float64,
    )
    actual = baselines._risk_scale(close, 5, 0)
    expected = float(np.std(np.diff(np.log(close[:6, 0])), ddof=1))
    assert np.isclose(actual, expected)


def test_trigger_indices_are_close_observable_requests() -> None:
    path = {
        "days": np.arange(10, 16, dtype=np.int32),
        "return_close": np.asarray([0.01, 0.04, 0.05, 0.01, 0.00, 0.00]),
        "risk_scale": np.full(6, 0.02),
        "filled_prices": np.asarray([101.0, 104.0, 105.0, 101.0, 100.0, 100.0]),
        "peak_gain": np.asarray([0.01, 0.04, 0.05, 0.05, 0.05, 0.05]),
        "drawdown": np.asarray([0.0, 0.0, 0.0, -0.038, -0.048, -0.048]),
        "giveback": np.asarray([0.0, 0.0, 0.0, 0.8, 1.0, 1.0]),
        "breakout_active": np.ones(6, dtype=np.int8),
        "small_event": np.asarray([0, 0, -1, 0, 0, 0], dtype=np.int8),
        "at_liquidation": np.zeros(6, dtype=bool),
    }
    assert baselines._trigger_index("fixed_h5", path, maximum_sessions=5) == 3
    assert (
        baselines._trigger_index("volatility_1_2", path, maximum_sessions=5)
        == 1
    )
    assert baselines._trigger_index("trailing_3pct", path, maximum_sessions=5) == 3
    assert baselines._trigger_index("giveback_half", path, maximum_sessions=5) == 3
    assert (
        baselines._trigger_index(
            "structure_small_reversal", path, maximum_sessions=5
        )
        == 2
    )


def test_resolve_exit_waits_for_the_next_legal_close() -> None:
    market = SimpleNamespace(
        exit_sellable=np.asarray(
            [[False], [False], [False], [False], [True], [True]], dtype=bool
        ),
        exit_close_raw=np.asarray(
            [[10.0], [10.1], [10.2], [10.3], [10.4], [10.5]], dtype=np.float32
        ),
    )
    path = {"days": np.asarray([2, 3, 4, 5], dtype=np.int32)}
    assert baselines._resolve_exit(
        market=market, path=path, symbol=0, trigger_index=0
    ) == 4


def test_stopping_ceiling_maps_each_request_to_a_later_legal_close() -> None:
    market = SimpleNamespace(
        end_idx=5,
        exit_sellable=np.asarray(
            [[False], [False], [True], [False], [True], [True]], dtype=bool
        ),
        exit_close_raw=np.asarray(
            [[10.0], [10.1], [10.2], [10.3], [10.4], [10.5]], dtype=np.float32
        ),
    )
    requests, exits = ceiling._legal_request_exit_pairs(
        market=market,
        entry_abs=1,
        symbol=0,
        liquidation_abs=5,
        maximum_sessions=3,
    )
    assert np.array_equal(requests, np.asarray([1, 2, 3], dtype=np.int32))
    assert np.array_equal(exits, np.asarray([2, 4, 4], dtype=np.int32))
