from __future__ import annotations

import pandas as pd
import pytest

from quantlab.data.domains.contracts.intraday import build_intraday_daily_feature_frame
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


def test_intraday_daily_features_preserve_day_keys_and_previous_close_gap() -> None:
    rows = []
    bar_times = ["09:35:00", "09:40:00", "09:45:00", "09:50:00", "09:55:00", "10:00:00"]
    for trade_date, opens in (
        ("2024-01-02", [10.0, 10.1, 10.2, 10.3, 10.4, 10.5]),
        ("2024-01-03", [11.0, 11.1, 11.2, 11.3, 11.4, 11.5]),
    ):
        for index, opening in enumerate(opens):
            close = opening + 0.05
            rows.append(
                {
                    "symbol": "600000.SH",
                    "trade_date": trade_date,
                    "bar_time": bar_times[index],
                    "open": opening,
                    "high": close + 0.02,
                    "low": opening - 0.02,
                    "close": close,
                    "volume": 100.0 + index,
                    "amount": close * (100.0 + index),
                }
            )

    result = build_intraday_daily_feature_frame(pd.DataFrame(rows)).sort_values("trade_date")

    assert result[["trade_date", "symbol"]].values.tolist() == [
        ["2024-01-02", "600000.SH"],
        ["2024-01-03", "600000.SH"],
    ]
    assert result["bar_count"].tolist() == [6.0, 6.0]
    assert pd.isna(result.iloc[0]["open_gap"])
    assert result.iloc[1]["open_gap"] == pytest.approx(11.0 / 10.55 - 1.0)
    assert result.iloc[0]["first_30m_ret"] == pytest.approx(10.55 / 10.0 - 1.0)
