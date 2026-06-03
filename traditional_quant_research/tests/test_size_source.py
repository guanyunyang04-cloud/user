from __future__ import annotations

from pathlib import Path

import pandas as pd

from traditional_quant_research.dataset_v2 import DAILY_SIZE_COLUMNS, load_pit_daily_size
from traditional_quant_research.size_source import (
    AKSHARE_CNINFO_RECONSTRUCTED_SOURCE,
    PROXY_AMOUNT_SOURCE,
    accepted_daily_size_source_grade,
    daily_size_cache_path,
    daily_size_meta_path,
    daily_size_part_path,
    daily_size_progress_path,
    daily_size_source_grade,
    empty_daily_size_frame,
    fetch_tushare_daily_size_cache,
    load_cached_daily_size,
    normalize_cninfo_share_change_events,
    normalize_project_symbol,
    reconstruct_daily_size_from_share_events,
    standardize_akshare_cninfo_reconstructed_size,
    standardize_daily_size_frame,
    standardize_proxy_amount_daily_size,
    standardize_tushare_daily_basic_size,
    write_daily_size_parquet,
)


def test_normalize_project_symbol_handles_vendor_formats() -> None:
    assert normalize_project_symbol("sh.600000") == "600000.SH"
    assert normalize_project_symbol("SZ.000001") == "000001.SZ"
    assert normalize_project_symbol("600000.SH") == "600000.SH"
    assert normalize_project_symbol("000001.sz") == "000001.SZ"


def test_standardize_tushare_daily_basic_maps_units_and_columns() -> None:
    raw = pd.DataFrame(
        [
            {
                "ts_code": "600000.SH",
                "trade_date": "20260601",
                "total_mv": "3000000.5",
                "circ_mv": 2500000.0,
                "total_share": 2935208.0,
                "float_share": 2935208.0,
                "free_share": 1800000.0,
            },
            {
                "ts_code": "sz.000001",
                "trade_date": "2026-06-01",
                "total_mv": 1000000.0,
                "circ_mv": 900000.0,
                "total_share": 1940592.0,
                "float_share": 1940554.0,
                "free_share": 1200000.0,
            },
        ]
    )

    size = standardize_tushare_daily_basic_size(raw)

    assert list(size.columns) == DAILY_SIZE_COLUMNS
    assert size["code"].tolist() == ["000001.SZ", "600000.SH"]
    assert size["date"].tolist() == [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-01")]
    assert size["total_market_cap"].tolist() == [1000000.0, 3000000.5]
    assert size["float_market_cap"].tolist() == [900000.0, 2500000.0]
    assert size["market_cap_unit"].unique().tolist() == ["10k CNY"]
    assert size["share_unit"].unique().tolist() == ["10k shares"]
    assert size["source"].unique().tolist() == ["tushare.daily_basic"]
    assert size["source_trade_date"].tolist() == ["20260601", "20260601"]


def test_standardize_tushare_daily_basic_empty_has_fixed_schema() -> None:
    size = standardize_tushare_daily_basic_size(pd.DataFrame())

    assert size.empty
    assert list(size.columns) == DAILY_SIZE_COLUMNS
    assert list(empty_daily_size_frame().columns) == DAILY_SIZE_COLUMNS


def test_reconstructed_size_computes_market_cap_from_close_and_shares() -> None:
    raw = pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "code": "600000.SH",
                "close": 10.0,
                "total_share": 100.0,
                "float_share": 80.0,
                "free_share": 70.0,
            }
        ]
    )

    size = standardize_akshare_cninfo_reconstructed_size(raw)

    assert size["total_market_cap"].tolist() == [1000.0]
    assert size["float_market_cap"].tolist() == [800.0]
    assert size["market_cap_unit"].tolist() == ["CNY"]
    assert size["share_unit"].tolist() == ["shares"]
    assert size["source"].tolist() == [AKSHARE_CNINFO_RECONSTRUCTED_SOURCE]
    assert accepted_daily_size_source_grade(size.loc[0, "source"])


def test_proxy_amount_daily_size_is_shaped_but_not_gate_eligible() -> None:
    raw = pd.DataFrame([{"date": "2026-06-01", "code": "600000.SH", "amount": 12345.0}])

    size = standardize_proxy_amount_daily_size(raw)

    assert size["total_market_cap"].tolist() == [12345.0]
    assert size["float_market_cap"].tolist() == [12345.0]
    assert size["source"].tolist() == [PROXY_AMOUNT_SOURCE]
    assert daily_size_source_grade(PROXY_AMOUNT_SOURCE) == "proxy_only"
    assert not accepted_daily_size_source_grade(PROXY_AMOUNT_SOURCE)


def test_cninfo_events_forward_fill_to_reconstruct_daily_size() -> None:
    events_raw = pd.DataFrame(
        [
            {"变动日期": "2026-01-01", "总股本": "1万股", "已流通股份": "8000股"},
        ]
    )
    events = normalize_cninfo_share_change_events(events_raw, code="600000.SH")
    bars = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "close": 10.0},
            {"date": pd.Timestamp("2026-01-03"), "code": "600000.SH", "close": 11.0},
        ]
    )

    size = reconstruct_daily_size_from_share_events(bars, events)

    assert size["total_share"].tolist() == [10000.0, 10000.0]
    assert size["float_share"].tolist() == [8000.0, 8000.0]
    assert size["total_market_cap"].tolist() == [100000.0, 110000.0]


def test_standardize_daily_size_frame_deduplicates_and_numeric_coerces() -> None:
    raw = pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "code": "sh.600000",
                "total_market_cap": "bad",
                "float_market_cap": "900",
                "source_trade_date": "2026-06-01",
            },
            {
                "date": "2026-06-01",
                "code": "600000.SH",
                "total_market_cap": "1000",
                "float_market_cap": "900",
                "source_trade_date": "20260601",
            },
        ]
    )

    size = standardize_daily_size_frame(raw)

    assert len(size) == 1
    assert size.loc[0, "code"] == "600000.SH"
    assert size.loc[0, "total_market_cap"] == 1000.0
    assert size.loc[0, "source_trade_date"] == "20260601"


def test_write_daily_size_parquet_is_loader_compatible(tmp_path: Path) -> None:
    frame = standardize_tushare_daily_basic_size(
        pd.DataFrame(
            [
                {
                    "ts_code": "600000.SH",
                    "trade_date": "20260601",
                    "total_mv": 1000.0,
                    "circ_mv": 800.0,
                    "total_share": 100.0,
                    "float_share": 80.0,
                    "free_share": 70.0,
                }
            ]
        )
    )

    path = write_daily_size_parquet(frame, tmp_path, overwrite=False)
    loaded = load_pit_daily_size(tmp_path)

    assert path == tmp_path / "daily_size.parquet"
    assert loaded["code"].tolist() == ["600000.SH"]
    assert loaded["float_market_cap"].tolist() == [800.0]


class FakeTusharePro:
    def __init__(self, *, fail_dates: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.fail_dates = fail_dates or set()

    def daily_basic(self, *, trade_date: str, fields: str) -> pd.DataFrame:
        self.calls.append(trade_date)
        if trade_date in self.fail_dates:
            raise RuntimeError(f"boom {trade_date}")
        return pd.DataFrame(
            [
                {
                    "ts_code": "600000.SH",
                    "trade_date": trade_date,
                    "total_mv": 1000.0,
                    "circ_mv": 800.0,
                    "total_share": 100.0,
                    "float_share": 80.0,
                    "free_share": 70.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": trade_date,
                    "total_mv": 2000.0,
                    "circ_mv": 1600.0,
                    "total_share": 200.0,
                    "float_share": 160.0,
                    "free_share": 140.0,
                },
            ]
        )


class FakeTushareModule:
    def __init__(self, pro: FakeTusharePro) -> None:
        self.pro = pro
        self.token = ""

    def set_token(self, token: str) -> None:
        self.token = token

    def pro_api(self, token: str | None = None) -> FakeTusharePro:
        self.token = token or self.token
        return self.pro


def test_fetch_tushare_daily_size_cache_success_and_cache_hit(tmp_path: Path) -> None:
    pro = FakeTusharePro()
    module = FakeTushareModule(pro)

    summary = fetch_tushare_daily_size_cache(
        output_root=tmp_path,
        year=2026,
        trade_dates=["20260601"],
        start_date="2026-06-01",
        end_date="2026-06-01",
        symbols=["600000.SH"],
        token="token",
        tushare_module=module,
    )

    assert summary["status"] == "passed"
    assert summary["rows"] == 1
    assert pro.calls == ["20260601"]
    assert daily_size_cache_path(tmp_path, 2026, symbols=["600000.SH"]).exists()
    assert daily_size_meta_path(tmp_path, 2026, symbols=["600000.SH"]).exists()
    assert daily_size_progress_path(tmp_path, 2026, symbols=["600000.SH"]).exists()
    assert daily_size_part_path(tmp_path, 2026, "20260601", symbols=["600000.SH"]).exists()
    assert not daily_size_cache_path(tmp_path, 2026).exists()
    assert load_cached_daily_size(tmp_path, [2026], "2026-06-01", "2026-06-01").empty

    cache_hit = fetch_tushare_daily_size_cache(
        output_root=tmp_path,
        year=2026,
        trade_dates=["20260601"],
        start_date="2026-06-01",
        end_date="2026-06-01",
        symbols=["600000.SH"],
        token="token",
        tushare_module=module,
    )

    assert cache_hit["status"] == "cache_hit"
    assert pro.calls == ["20260601"]


def test_full_daily_size_cache_is_formal_loader_source(tmp_path: Path) -> None:
    pro = FakeTusharePro()
    module = FakeTushareModule(pro)

    summary = fetch_tushare_daily_size_cache(
        output_root=tmp_path,
        year=2026,
        trade_dates=["20260601"],
        start_date="2026-06-01",
        end_date="2026-06-01",
        symbols=None,
        token="token",
        tushare_module=module,
    )

    assert summary["status"] == "passed"
    assert summary["rows"] == 2
    assert daily_size_cache_path(tmp_path, 2026).exists()
    cached = load_cached_daily_size(tmp_path, [2026], "2026-06-01", "2026-06-01")
    assert cached["code"].tolist() == ["000001.SZ", "600000.SH"]


def test_fetch_tushare_daily_size_cache_missing_auth_is_skipped(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("TS_TOKEN", raising=False)

    summary = fetch_tushare_daily_size_cache(
        output_root=tmp_path,
        year=2026,
        trade_dates=["20260601"],
        start_date="2026-06-01",
        end_date="2026-06-01",
        symbols=["600000.SH"],
        token="",
        tushare_module=FakeTushareModule(FakeTusharePro()),
    )

    assert summary["status"] == "skipped"
    assert summary["skip_reason"] == "auth_missing"
    assert summary["failure_count"] == 1


def test_fetch_tushare_daily_size_cache_records_partial_failures(tmp_path: Path) -> None:
    pro = FakeTusharePro(fail_dates={"20260602"})
    module = FakeTushareModule(pro)

    summary = fetch_tushare_daily_size_cache(
        output_root=tmp_path,
        year=2026,
        trade_dates=["20260601", "20260602"],
        start_date="2026-06-01",
        end_date="2026-06-02",
        symbols=["600000.SH"],
        token="token",
        tushare_module=module,
    )

    assert summary["status"] == "partial"
    assert summary["rows"] == 1
    assert summary["failure_count"] == 1
    assert summary["failures"][0]["trade_date"] == "20260602"


def test_sample_size_cache_does_not_satisfy_full_request(tmp_path: Path) -> None:
    sample_pro = FakeTusharePro()
    fetch_tushare_daily_size_cache(
        output_root=tmp_path,
        year=2026,
        trade_dates=["20260601"],
        start_date="2026-06-01",
        end_date="2026-06-01",
        symbols=["600000.SH"],
        token="token",
        tushare_module=FakeTushareModule(sample_pro),
    )
    full_pro = FakeTusharePro()

    summary = fetch_tushare_daily_size_cache(
        output_root=tmp_path,
        year=2026,
        trade_dates=["20260601"],
        start_date="2026-06-01",
        end_date="2026-06-01",
        symbols=None,
        token="token",
        tushare_module=FakeTushareModule(full_pro),
    )

    assert summary["status"] == "passed"
    assert summary["rows"] == 2
    assert full_pro.calls == ["20260601"]
