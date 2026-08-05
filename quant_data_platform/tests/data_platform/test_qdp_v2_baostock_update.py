from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from quant_data_platform.domains.contracts import DataDomain
from quant_data_platform.providers import (
    _baostock_all_stock_frame,
    _baostock_bulk_daily_domain_frame,
    _baostock_status_frame_from_all_stock,
)
from quant_data_platform.qdp_v2 import baostock_update, factor_update, update


class _FakeBaostock:
    def fetch_domain(self, request):
        return SimpleNamespace(
            data=pd.DataFrame(
                {
                    "trade_date": ["2026-01-05"],
                    "is_open": ["1"],
                    "exchange": ["SSE"],
                }
            ),
            error_report=[],
        )

    def fetch_stock_basic_snapshot(self, *, trade_date: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": ["000001.SZ", "300001.SZ", "688001.SH", "920001.BJ"],
                "name": ["平安银行", "创业板", "科创板", "北交所"],
                "list_date": ["1991-04-03", "2009-10-30", "2019-07-22", "2025-01-01"],
                "delist_date": ["", "", "", ""],
            }
        )

    def fetch_date_partition_with_all_stock(self, request):
        daily = pd.DataFrame(
            {
                "provider_symbol": ["000001.SZ", "300001.SZ", "688001.SH", "920001.BJ"],
                "trade_date": ["2026-01-05"] * 4,
                "open": [10.0] * 4,
                "high": [11.0] * 4,
                "low": [9.0] * 4,
                "close": [10.5] * 4,
                "volume": [100.0] * 4,
                "amount": [1000.0] * 4,
            }
        )
        all_stock = pd.DataFrame(
            {
                "symbol": ["000001.SZ", "300001.SZ", "688001.SH", "920001.BJ"],
                "name": ["平安银行", "创业板", "科创板", "北交所"],
                "is_suspended": [False] * 4,
            }
        )
        return SimpleNamespace(data=daily, error_report=[]), all_stock


class _FakeAllStockQuery:
    fields = ["code", "code_name", "tradeStatus"]
    error_code = "0"
    error_msg = ""

    def __init__(self, rows):
        self.rows = rows
        self.position = -1

    def next(self):
        self.position += 1
        return self.position < len(self.rows)

    def get_row_data(self):
        return self.rows[self.position]


def test_baostock_core_update_builds_only_current_table_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        baostock_update,
        "_identity_additions",
        lambda stock_basic, workspace: (
            baostock_update._empty_identity(),
            baostock_update._empty_history(),
        ),
    )
    monkeypatch.setattr(
        baostock_update,
        "_only_missing_keys",
        lambda frame, domain, workspace: frame,
    )

    result = baostock_update.run_baostock_core_update(
        start_date="2026-01-05",
        end_date="2026-01-05",
        workspace_root=tmp_path,
        provider=_FakeBaostock(),
        apply=False,
    )

    assert result["status"] == "planned"
    assert result["provider"] == "baostock"
    assert result["missing_rows"]["market_daily_raw"] == 1
    assert result["missing_rows"]["universe_snapshot"] == 1
    assert result["missing_rows"]["security_status"] == 1
    assert result["missing_rows"]["trading_calendar"] == 1


def test_supported_increment_scope_is_mainboard_only() -> None:
    accepted = ("600000.SH", "601001.SH", "603001.SH", "605001.SH", "000001.SZ", "001001.SZ", "002001.SZ", "003001.SZ")
    rejected = ("300001.SZ", "301001.SZ", "688001.SH", "920001.BJ", "900001.SH", "200001.SZ")

    assert all(baostock_update._is_supported_mainboard_symbol(item) for item in accepted)
    assert not any(baostock_update._is_supported_mainboard_symbol(item) for item in rejected)


def test_daily_normalization_keeps_all_ohlcva_columns_float64() -> None:
    raw = pd.DataFrame(
        {
            "provider_symbol": ["600000.SH"],
            "open": ["10"],
            "high": ["11"],
            "low": ["9"],
            "close": ["10"],
            "volume": ["100"],
            "amount": ["1000"],
        }
    )

    normalized = baostock_update._daily_frame(raw, trade_date="2026-01-05")

    assert all(
        str(normalized[column].dtype) == "float64"
        for column in ("open", "high", "low", "close", "volume", "amount")
    )


def test_status_prefers_raw_daily_isst_and_preserves_unknown() -> None:
    all_stock = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
            "name": ["normal", "S ST fallback", "G*ST fallback", ""],
            "is_suspended": [False, pd.NA, False, False],
        }
    )
    stock_basic = pd.DataFrame(
        {
            "symbol": all_stock["symbol"],
            "name": ["normal", "S ST fallback", "G*ST fallback", ""],
            "list_date": ["2000-01-01"] * 4,
            "delist_date": [""] * 4,
        }
    )
    raw_daily = pd.DataFrame(
        {
            "code": ["sz.000001", "sz.000002", "sz.000003", "sz.000004"],
            "isST": ["1", "0", "", ""],
        }
    )

    _, status = baostock_update._universe_and_status_frames(
        all_stock,
        daily_status=raw_daily,
        stock_basic=stock_basic,
        trade_date="2026-01-05",
    )
    status = status.set_index("symbol")

    assert bool(status.loc["000001.SZ", "is_st"])
    assert not bool(status.loc["000002.SZ", "is_st"])
    assert pd.isna(status.loc["000002.SZ", "is_suspended"])
    assert status.loc["000002.SZ", "status_reason"] == "status_unknown"
    assert bool(status.loc["000003.SZ", "is_st"])
    assert pd.isna(status.loc["000004.SZ", "is_st"])
    assert status.loc["000001.SZ", "source"] == "baostock.daily_isST"
    assert status.loc["000003.SZ", "source"] == "baostock.all_stock_name"
    assert status.loc["000004.SZ", "status_reason"] == "status_unknown"


def test_baostock_status_parsers_preserve_missing_and_invalid_trade_status() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2026-01-05"] * 4,
            "code": ["sz.000001", "sz.000002", "sz.000003", "sz.000004"],
            "tradestatus": ["", "invalid", "0", "1"],
            "isST": ["0"] * 4,
        }
    )
    bulk = _baostock_bulk_daily_domain_frame(
        raw, domain=DataDomain.SECURITY_STATUS, query_date="2026-01-05"
    )

    assert str(bulk["is_suspended"].dtype) == "boolean"
    assert bulk["is_suspended"].isna().tolist() == [True, True, False, False]
    assert bool(bulk.loc[2, "is_suspended"])
    assert not bool(bulk.loc[3, "is_suspended"])

    rows = [
        ["sz.000001", "normal one", ""],
        ["sz.000002", "normal two", "invalid"],
        ["sz.000003", "normal three", "0"],
        ["sz.000004", "normal four", "1"],
    ]
    universe = _baostock_all_stock_frame(
        _FakeAllStockQuery(rows), trade_date="2026-01-05"
    )
    status = _baostock_status_frame_from_all_stock(
        _FakeAllStockQuery(rows), trade_date="2026-01-05"
    )

    assert universe["is_suspended"].isna().tolist() == [True, True, False, False]
    assert status["is_suspended"].isna().tolist() == [True, True, False, False]


def test_nullable_bool_parser_does_not_guess_market_state_words() -> None:
    assert baostock_update._to_nullable_bool("yes") is True
    assert baostock_update._to_nullable_bool("no") is False
    assert baostock_update._to_nullable_bool("open") is None
    assert baostock_update._to_nullable_bool("closed") is None


def test_factor_tail_initializes_first_qdp_observation_to_one() -> None:
    missing = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "trade_date": ["2010-01-04"],
            "prior_date": [pd.NA],
            "prior_fore": [float("nan")],
            "prior_back": [float("nan")],
            "prior_adjust": [float("nan")],
        }
    )
    provider = SimpleNamespace(
        fetch_date_partition=lambda request: SimpleNamespace(data=pd.DataFrame())
    )

    rows, metrics = factor_update.build_factor_tail_rows(missing, provider=provider)

    assert rows["adjust_factor"].tolist() == [1.0]
    assert rows["source"].tolist() == ["qdp_first_observation_normalized_to_1"]
    assert metrics["initialized_symbol_count"] == 1


def test_factor_tail_continues_with_cumulative_back_factor_ratio() -> None:
    missing = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "trade_date": ["2026-07-15"],
            "prior_date": ["2026-07-14"],
            "prior_fore": [5.0],
            "prior_back": [5.0],
            "prior_adjust": [5.0],
        }
    )

    class Provider:
        def fetch_date_partition(self, request):
            return SimpleNamespace(
                data=pd.DataFrame(
                    {
                        "provider_symbol": ["600000.SH"],
                        "divid_operate_date": ["2026-07-15"],
                        "adjust_factor": [6.0],
                        "back_adjust_factor": [1.02],
                    }
                )
            )

        def fetch_domain(self, request):
            return SimpleNamespace(
                data=pd.DataFrame(
                    {
                        "symbol": ["600000.SH", "600000.SH"],
                        "trade_date": ["2025-07-15", "2026-07-15"],
                        # Per-event factors are not cumulative and must not be
                        # divided across events.
                        "adjust_factor": [2.0, 6.0],
                        "back_adjust_factor": [1.0, 1.02],
                    }
                ),
                error_report=[],
            )

    rows, metrics = factor_update.build_factor_tail_rows(missing, provider=Provider())

    assert rows["adjust_factor"].tolist() == [5.1]
    assert rows["back_adjust_factor"].tolist() == [5.1]
    assert rows["source"].tolist() == [
        "baostock.back_adjust_factor_ratio+qdp_prior_carry"
    ]
    assert metrics["continued_event_count"] == 1


def test_qdp_update_reports_failed_stage_and_keeps_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        update,
        "plan_update",
        lambda **kwargs: {
            "status": "planned",
            "start_date": "2026-01-05",
            "as_of_date": "2026-01-05",
            "workers": 4,
            "provider_order": ["baostock_core"],
            "coverage_before": {},
        },
    )

    def fail_core(**kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(update, "run_baostock_core_update", fail_core)

    result = update.run_update(
        as_of_date="2026-01-05",
        workspace_root=tmp_path,
    )

    assert result["status"] == "failed"
    assert result["failed_stage"] == "baostock_core"
    assert result["runtime_cleanup"] == "retained_after_failure"
