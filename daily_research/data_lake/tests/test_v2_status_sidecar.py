from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd
import pytest

from daily_research.data_lake import ResearchDataLake
from daily_research.data_lake.build_v2_status_sidecar import main as sidecar_main
from daily_research.data_lake.v2_status_sidecar import build_v2_status_sidecar, build_v2_status_sidecar_frame, is_mainboard_symbol


def _market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trade_date": "2026-01-05", "symbol": "600000.SH", "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.1, "volume": 1000.0, "amount": 10100.0},
            {"trade_date": "2026-01-05", "symbol": "000001.SZ", "open": 11.0, "high": 11.2, "low": 10.8, "close": 11.1, "volume": 1000.0, "amount": 11100.0},
            {"trade_date": "2026-01-05", "symbol": "300001.SZ", "open": 12.0, "high": 12.2, "low": 11.8, "close": 12.1, "volume": 1000.0, "amount": 12100.0},
            {"trade_date": "2026-01-05", "symbol": "688001.SH", "open": 13.0, "high": 13.2, "low": 12.8, "close": 13.1, "volume": 1000.0, "amount": 13100.0},
            {"trade_date": "2026-01-05", "symbol": "000002.SZ", "open": 14.0, "high": 14.2, "low": 13.8, "close": 14.1, "volume": 1000.0, "amount": 14100.0},
            {"trade_date": "2026-01-05", "symbol": "600001.SH", "open": 15.0, "high": 15.2, "low": 14.8, "close": 15.1, "volume": 0.0, "amount": 0.0},
            {"trade_date": "2026-01-05", "symbol": "600002.SH", "open": None, "high": None, "low": None, "close": None, "volume": 1000.0, "amount": 10000.0},
            {"trade_date": "2026-01-05", "symbol": "600003.SH", "open": 16.0, "high": 16.2, "low": 15.8, "close": 16.1, "volume": 1000.0, "amount": 16100.0},
            {"trade_date": "2026-01-05", "symbol": "600004.SH", "open": 17.0, "high": 17.2, "low": 16.8, "close": 17.1, "volume": 1000.0, "amount": 17100.0},
            {"trade_date": "2026-01-05", "symbol": "600005.SH", "open": 18.0, "high": 18.2, "low": 17.8, "close": 18.1, "volume": 1000.0, "amount": 18100.0},
        ]
    )


def _universe_frame() -> pd.DataFrame:
    symbols = ["600000.SH", "000001.SZ", "300001.SZ", "688001.SH", "000002.SZ", "600001.SH", "600002.SH", "600003.SH", "600004.SH", "600005.SH"]
    return pd.DataFrame(
        {
            "trade_date": ["2026-01-05"] * len(symbols),
            "symbol": symbols,
            "name": ["ok", "ok", "cyb", "star", "st", "fallback_susp", "missing", "date_delisted", "not_listed", "status_delisted"],
            "exchange": ["SH", "SZ", "SZ", "SH", "SZ", "SH", "SH", "SH", "SH", "SH"],
            "board": ["main"] * len(symbols),
            "list_status": ["L"] * len(symbols),
            "list_date": ["2000-01-01"] * 9 + ["2000-01-01"],
            "delist_date": ["", "", "", "", "", "", "", "2026-01-01", "", ""],
            "source": ["unit"] * len(symbols),
        }
    )


def _status_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2026-01-05"] * 5,
            "symbol": ["000002.SZ", "600001.SH", "600002.SH", "600003.SH", "600005.SH"],
            "is_st": [True, False, False, False, False],
            "is_suspended": [False, pd.NA, pd.NA, False, False],
            "is_delisted": [False, False, False, True, True],
            "status_reason": ["st", "", "", "delisted", "delisted"],
            "source": ["unit"] * 5,
        }
    )


def test_mainboard_symbol_detection_excludes_chinext_and_star_market() -> None:
    assert is_mainboard_symbol("600000.SH") is True
    assert is_mainboard_symbol("000001.SZ") is True
    assert is_mainboard_symbol("002001.SZ") is True
    assert is_mainboard_symbol("300001.SZ") is False
    assert is_mainboard_symbol("688001.SH") is False


def test_v2_status_sidecar_applies_pit_and_reject_reason_order() -> None:
    frame, summary = build_v2_status_sidecar_frame(
        market=_market_frame(),
        universe_snapshot=_universe_frame(),
        security_status=_status_frame(),
    )
    reasons = frame.set_index("symbol")["reject_reason"].to_dict()

    assert summary["status"] == "ok"
    assert bool(frame.loc[frame["symbol"].eq("600000.SH"), "is_tradeable"].iloc[0]) is True
    assert bool(frame.loc[frame["symbol"].eq("000001.SZ"), "is_tradeable"].iloc[0]) is True
    assert reasons["300001.SZ"] == "not_mainboard"
    assert reasons["688001.SH"] == "not_mainboard"
    assert reasons["000002.SZ"] == "st_on_date"
    assert reasons["600001.SH"] == "suspended_on_date"
    assert reasons["600002.SH"] == "missing_bar"
    assert reasons["600003.SH"] == "not_listed_on_date"
    assert reasons["600004.SH"] == ""
    assert reasons["600005.SH"] == "delisted_on_date"


def test_status_sidecar_parses_status_text_by_field_semantics() -> None:
    market = pd.DataFrame(
        [
            {"trade_date": "2026-01-05", "symbol": "600000.SH", "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.1, "volume": 1000.0, "amount": 10100.0},
            {"trade_date": "2026-01-05", "symbol": "600001.SH", "open": 11.0, "high": 11.2, "low": 10.8, "close": 11.1, "volume": 1000.0, "amount": 11100.0},
            {"trade_date": "2026-01-05", "symbol": "600002.SH", "open": 12.0, "high": 12.2, "low": 11.8, "close": 12.1, "volume": 1000.0, "amount": 12100.0},
        ]
    )
    universe = pd.DataFrame(
        {
            "trade_date": ["2026-01-05"] * 3,
            "symbol": ["600000.SH", "600001.SH", "600002.SH"],
            "name": ["正常", "停牌", "退市"],
            "exchange": ["SH", "SH", "SH"],
            "board": ["main", "main", "main"],
            "list_status": ["L", "L", "L"],
            "list_date": ["2000-01-01"] * 3,
            "delist_date": ["", "", ""],
            "source": ["unit"] * 3,
        }
    )
    status = pd.DataFrame(
        {
            "trade_date": ["2026-01-05"] * 3,
            "symbol": ["600000.SH", "600001.SH", "600002.SH"],
            "is_st": ["否", "否", "否"],
            "is_suspended": ["正常", "停牌", "正常"],
            "is_delisted": ["上市", "上市", "退市"],
            "status_reason": ["", "", ""],
            "source": ["unit"] * 3,
        }
    )

    frame, _summary = build_v2_status_sidecar_frame(
        market=market,
        universe_snapshot=universe,
        security_status=status,
    )
    reasons = frame.set_index("symbol")["reject_reason"].to_dict()

    assert reasons["600000.SH"] == ""
    assert reasons["600001.SH"] == "suspended_on_date"
    assert reasons["600002.SH"] == "delisted_on_date"


def test_build_v2_status_sidecar_saves_file_backed_dataset() -> None:
    dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
    stocks = ["600000.SH", "000001.SZ"]
    close = pd.DataFrame(10.0, index=dates, columns=stocks)
    with TemporaryDirectory() as temp_dir:
        lake = ResearchDataLake(Path(temp_dir))
        record = lake.save_market_data_bundle(
            spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
            market_frames={
                "Open": close - 0.1,
                "High": close + 0.2,
                "Low": close - 0.2,
                "Close": close,
                "Volume": pd.DataFrame(1000.0, index=dates, columns=stocks),
                "Amount": pd.DataFrame(10000.0, index=dates, columns=stocks),
            },
            benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
            membership_frame=pd.DataFrame(True, index=dates, columns=stocks),
            feature_frames={"score_none": close * 0.0},
            source="synthetic",
        )
        universe = lake.save_domain_dataset(
            domain="universe_snapshot",
            frame=pd.DataFrame(
                {
                    "trade_date": ["2026-01-05", "2026-01-05"],
                    "symbol": stocks,
                    "name": ["a", "b"],
                    "exchange": ["SH", "SZ"],
                    "board": ["main", "main"],
                    "list_status": ["L", "L"],
                    "list_date": ["2000-01-01", "2000-01-01"],
                    "delist_date": ["", ""],
                    "source": ["unit", "unit"],
                }
            ),
            spec={"dataset": "data_platform_universe_snapshot", "start_date": "2026-01-05", "end_date": "2026-01-05"},
            source="unit",
        )
        status = lake.save_domain_dataset(
            domain="security_status",
            frame=pd.DataFrame(
                {
                    "trade_date": ["2026-01-05", "2026-01-05"],
                    "symbol": stocks,
                    "is_st": [False, False],
                    "is_suspended": [False, False],
                    "is_delisted": [False, False],
                    "status_reason": ["", ""],
                    "source": ["unit", "unit"],
                }
            ),
            spec={"dataset": "data_platform_security_status", "start_date": "2026-01-05", "end_date": "2026-01-05"},
            source="unit",
        )
        sidecar, frame, summary = build_v2_status_sidecar(
            lake=lake,
            source_market_dataset_id=record.dataset_id,
            start_date="2026-01-05",
            end_date="2026-01-06",
            universe_snapshot_dataset_id=universe.dataset_id,
            security_status_dataset_id=status.dataset_id,
        )

    assert sidecar.dataset_kind == "data_platform_v2_status_sidecar"
    assert summary["source_market_dataset_id"] == record.dataset_id
    assert summary["tradeable_rows"] == len(frame)


def test_build_v2_status_sidecar_cli_blocks_active_artifact_diff() -> None:
    with mock.patch("daily_research.data_lake.build_v2_status_sidecar._active_artifact_has_diff", return_value=True):
        with pytest.raises(ValueError, match="active_artifact_diff_blocker"):
            sidecar_main(["--source-market-dataset-id", "policy_input_bundle__unit"])
