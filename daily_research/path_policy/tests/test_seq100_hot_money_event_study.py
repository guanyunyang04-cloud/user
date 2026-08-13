from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "seq100_hot_money_event_study.py"
SPEC = importlib.util.spec_from_file_location("hot_money_event_study", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_hac_se_is_zero_for_constant_series() -> None:
    assert MODULE._hac_se(pd.Series([1.0] * 20)) == 0.0


def test_period_boundaries_are_frozen() -> None:
    assert MODULE._period(2012) == "development_2012_2020"
    assert MODULE._period(2021) == "validation_2021_2022"
    assert MODULE._period(2023) == "oos_2023_2025"
    assert MODULE._period(2026) == "outside"


def test_future_columns_expose_legal_t_plus_one_path() -> None:
    columns = MODULE._future_columns()
    assert "legal_future_high_1" in columns
    assert "ROWS BETWEEN 2 FOLLOWING AND 5 FOLLOWING" in columns
    assert "entry_high_next" in columns


def test_minute_efficiency_includes_first_bar_open_to_close_return() -> None:
    sql = MODULE._minute_query([], Path("events.parquet"), 2025)
    assert "abs(ln(last_close / NULLIF(first_open, 0)))" in sql
    assert "WHEN rn = 1 AND open > 0 AND close > 0 THEN abs(ln(close / open))" in sql


def test_analysis_uses_next_open_and_separates_mfe_from_terminal_return() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ"],
            "trade_date": ["2024-01-02", "2024-01-02"],
            "entry_open_next": [10.0, 20.0],
            "participation_shock": [True, False],
            "positive_participation_shock": [True, False],
            "range_expansion": [False, False],
            "breakout20": [False, False],
            "breakout60": [False, False],
            "breakout_confirmed": [False, False],
            "retest20": [False, False],
            "rebreakout20": [False, False],
            "climax_fade": [False, False],
        }
    )
    for horizon in MODULE.HORIZONS:
        frame[f"exit_close_{horizon}"] = [10.10, 19.80]
        frame[f"future_high_{horizon}"] = [10.50, 21.00]
        frame[f"future_low_{horizon}"] = [9.90, 19.50]
        frame[f"legal_future_high_{horizon}"] = [10.50, 21.00] if horizon >= 3 else [np.nan, np.nan]
        frame[f"legal_future_low_{horizon}"] = [9.90, 19.50] if horizon >= 3 else [np.nan, np.nan]
    frame["entry_high_next"] = [10.20, 20.50]
    frame["entry_low_next"] = [9.90, 19.50]
    result = MODULE._prepare_analysis(frame, 60.0)
    assert np.isclose(result.loc[0, "gross_return_5"], 0.01)
    assert np.isclose(result.loc[0, "mfe_5"], 0.05)
    assert np.isclose(result.loc[0, "net_return_5"], 0.004)
    assert np.isnan(result.loc[0, "gross_return_1"])
    assert np.isnan(result.loc[0, "mfe_1"])
