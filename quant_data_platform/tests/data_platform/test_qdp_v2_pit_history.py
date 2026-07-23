from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_data_platform.qdp_v2.pit_history import (
    PitHistoryContext,
    _factor_rows,
    _historical_names,
    _historical_st_status,
    _normalize_eastmoney_history,
    _normalize_sina_factors,
    _prepare_symbol_parts,
)


def test_factor_rows_are_past_only_and_normalized_to_first_observation() -> None:
    history = pd.DataFrame(
        {
            "symbol": ["000001.SZ"] * 4,
            "trade_date": [
                "2020-01-02",
                "2020-01-03",
                "2020-01-06",
                "2020-01-07",
            ],
        }
    )
    events = pd.DataFrame(
        {
            "trade_date": ["2019-01-01", "2020-01-06"],
            "back_adjust_factor": [2.0, 3.0],
        }
    )

    result = _factor_rows(history, events)

    assert result["adjust_factor"].tolist() == [1.0, 1.0, 1.5, 1.5]
    assert result["factor_source_date"].tolist() == [
        "2019-01-01",
        "2019-01-01",
        "2020-01-06",
        "2020-01-06",
    ]
    assert (
        pd.to_datetime(result["factor_source_date"])
        <= pd.to_datetime(result["trade_date"])
    ).all()


def test_historical_names_use_only_effective_intervals() -> None:
    dates = pd.Series(["2020-01-02", "2020-02-03", "2020-03-02"])
    intervals = pd.DataFrame(
        {
            "name": ["旧名", "新名"],
            "start_date": ["2010-01-01", "2020-02-01"],
            "end_date": ["2020-01-31", ""],
        }
    )

    assert _historical_names(dates, intervals, fallback="未知").tolist() == [
        "旧名",
        "新名",
        "新名",
    ]


def test_sse_st_status_uses_dated_transitions_before_archive() -> None:
    history = pd.DataFrame(
        {
            "trade_date": [
                "2010-02-26",
                "2010-03-01",
                "2011-10-31",
                "2011-11-01",
                "2012-01-04",
            ],
            "isST": ["", "", "", "", "0"],
        }
    )
    names = pd.Series(["公司"] * len(history))

    result = _historical_st_status(history, symbol="600077.SH", names=names)

    assert result.tolist() == [False, True, True, False, False]


def test_eastmoney_history_converts_lots_to_shares() -> None:
    raw = pd.DataFrame(
        {
            "日期": ["2020-01-02"],
            "开盘": [10.0],
            "最高": [10.5],
            "最低": [9.8],
            "收盘": [10.2],
            "成交量": [1234],
            "成交额": [1_250_000.0],
            "换手率": [1.5],
            "涨跌幅": [2.0],
        }
    )

    result = _normalize_eastmoney_history(raw, symbol="600001.SH")

    assert result.loc[0, "volume"] == 123_400.0
    assert result.loc[0, "tradestatus"] == "1"
    assert result.loc[0, "history_source"] == "akshare_eastmoney_unadjusted_history"


def test_sina_factor_provenance_is_explicit() -> None:
    raw = pd.DataFrame(
        {"date": ["1900-01-01", "2020-01-06"], "hfq_factor": [1.0, 1.5]}
    )

    result = _normalize_sina_factors(raw, symbol="600001.SH")

    assert result["factor_provider"].unique().tolist() == ["sina_via_akshare"]
    assert result["source"].unique().tolist() == ["akshare_sina_hfq_factor_event"]


def test_prepare_symbol_parts_keeps_suspension_and_infers_float_shares(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    for directory in (
        "history_parts",
        "factor_parts",
        "share_event_parts",
        "name_interval_parts",
    ):
        (runtime / directory).mkdir(parents=True, exist_ok=True)
    symbol_file = "000033_SZ.parquet"
    history = pd.DataFrame(
        {
            "trade_date": ["2020-01-02", "2020-01-03", "2020-01-06"],
            "symbol": ["000033.SZ"] * 3,
            "provider_code": ["sz.000033"] * 3,
            "open": [10.0, 10.0, 9.0],
            "high": [10.5, 10.0, 9.5],
            "low": [9.5, 10.0, 8.5],
            "close": [10.0, 10.0, 9.0],
            "preclose": [9.8, 10.0, 10.0],
            "volume": [1_000_000.0, np.nan, 900_000.0],
            "amount": [10_000_000.0, np.nan, 8_100_000.0],
            "adjustflag": ["3"] * 3,
            "turn": [1.0, np.nan, 1.0],
            "tradestatus": ["1", "0", "1"],
            "pctChg": [2.0, 0.0, -10.0],
            "peTTM": [10.0, 10.0, 9.0],
            "pbMRQ": [1.0, 1.0, 0.9],
            "psTTM": [1.0, 1.0, 0.9],
            "pcfNcfTTM": [5.0, 5.0, 4.5],
            "isST": ["0", "1", "1"],
        }
    )
    factors = pd.DataFrame(
        {
            "symbol": ["000033.SZ"],
            "trade_date": ["2019-01-01"],
            "fore_adjust_factor": [1.0],
            "back_adjust_factor": [1.0],
            "adjust_factor": [1.0],
        }
    )
    history.to_parquet(runtime / "history_parts" / symbol_file, index=False)
    factors.to_parquet(runtime / "factor_parts" / symbol_file, index=False)
    pd.DataFrame(
        columns=[
            "symbol",
            "variation_date",
            "source_date",
            "total_share",
            "float_share",
            "source",
        ]
    ).to_parquet(runtime / "share_event_parts" / symbol_file, index=False)
    pd.DataFrame(
        columns=[
            "symbol",
            "name",
            "start_date",
            "end_date",
            "announcement_date",
            "change_reason",
            "source",
        ]
    ).to_parquet(runtime / "name_interval_parts" / symbol_file, index=False)
    context = PitHistoryContext(
        workspace=tmp_path,
        root=tmp_path / "qdp_v2",
        runtime=runtime,
        start_date="2020-01-01",
        end_date="2020-12-31",
    )

    _prepare_symbol_parts(
        context,
        row={
            "symbol": "000033.SZ",
            "name": "新都退",
            "list_date": "1994-01-03",
            "delist_date": "",
        },
    )

    daily = pd.read_parquet(
        runtime / "domain_parts" / "market_daily_raw" / symbol_file
    )
    status = pd.read_parquet(
        runtime / "domain_parts" / "security_status" / symbol_file
    )
    shares = pd.read_parquet(
        runtime / "domain_parts" / "share_capital" / symbol_file
    )
    assert daily["trade_date"].tolist() == ["2020-01-02", "2020-01-06"]
    assert bool(status.loc[1, "is_suspended"])
    assert bool(status.loc[1, "is_st"])
    assert shares["float_share"].tolist() == [
        100_000_000.0,
        90_000_000.0,
    ]
    assert shares["float_share_source_date"].tolist() == [
        "2020-01-02",
        "2020-01-06",
    ]

    (runtime / "domain_parts" / "done" / "000033_SZ.json").unlink()
    _prepare_symbol_parts(
        context,
        row={
            "symbol": "000033.SZ",
            "name": "新都退",
            "list_date": "1994-01-03",
            "delist_date": "",
        },
        identity_already_known=True,
    )
    assert pd.read_parquet(
        runtime / "domain_parts" / "security_identity" / symbol_file
    ).empty
    assert pd.read_parquet(
        runtime / "domain_parts" / "symbol_history" / symbol_file
    ).empty
