from __future__ import annotations

import pandas as pd

from quantlab.data.provider_symbols import (
    from_baostock_code,
    is_mootdx_index_symbol,
    to_baostock_code,
)
from quantlab.data.qdp_v2.normalization import (
    normalize_daily_basic,
    normalize_name_intervals,
)
from quantlab.data.qdp_v2.pit_normalization import (
    factor_rows,
    normalize_eastmoney_history,
)


def test_provider_symbol_boundary_is_canonical() -> None:
    assert to_baostock_code("600000.SH") == "sh.600000"
    assert from_baostock_code("sz.000001") == "000001.SZ"
    assert is_mootdx_index_symbol("000300.SH")


def test_daily_basic_normalization_preserves_share_units() -> None:
    frame = pd.DataFrame(
        {
            "ts_code": ["600000.SH"],
            "trade_date": ["20260716"],
            "total_share": [100.0],
            "float_share": [80.0],
            "total_mv": [1000.0],
            "circ_mv": [800.0],
        }
    )
    result = normalize_daily_basic(frame, "2026-07-16").iloc[0]
    assert result["total_share"] == 1_000_000.0
    assert result["float_share"] == 800_000.0


def test_name_intervals_are_pit_filtered() -> None:
    frame = pd.DataFrame(
        {
            "ts_code": ["600000.SH", "600000.SH"],
            "name": ["old", "future"],
            "start_date": ["20200101", "20270101"],
            "end_date": ["20201231", ""],
            "ann_date": ["20200102", "20270102"],
        }
    )
    result = normalize_name_intervals(frame, target_date="2026-12-31")
    assert result["name"].tolist() == ["old"]


def test_pit_history_normalizers_keep_provider_provenance() -> None:
    history = normalize_eastmoney_history(
        pd.DataFrame(
            {
                "日期": ["2020-01-02"],
                "开盘": [10.0],
                "最高": [10.5],
                "最低": [9.8],
                "收盘": [10.2],
                "成交量": [12.0],
                "成交额": [1000.0],
            }
        ),
        symbol="600000.SH",
    )
    factors = factor_rows(
        history[["symbol", "trade_date"]],
        pd.DataFrame(
            {
                "trade_date": ["2020-01-01"],
                "back_adjust_factor": [2.0],
            }
        ),
    )
    assert history.loc[0, "volume"] == 1200.0
    assert factors.loc[0, "adjust_factor"] == 1.0
    assert factors.loc[0, "source"].endswith("+pit_history_restore")
