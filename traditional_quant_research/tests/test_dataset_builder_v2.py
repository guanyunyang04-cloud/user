from __future__ import annotations

import pandas as pd

from traditional_quant_research.dataset_builder_v2 import (
    _baostock_to_std_code,
    _std_to_baostock_code,
    _assert_failure_rate,
    BaostockSourceError,
    PitBuildConfig,
    assemble,
    build_daily_universe,
    discover_daily_stock_lists,
)


def test_baostock_code_conversion_round_trip() -> None:
    assert _baostock_to_std_code("sh.600000") == "600000.SH"
    assert _baostock_to_std_code("sz.000001") == "000001.SZ"
    assert _std_to_baostock_code("600000.SH") == "sh.600000"
    assert _std_to_baostock_code("000001.SZ") == "sz.000001"


def test_build_daily_universe_marks_pit_statuses() -> None:
    stock_lists = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "name_on_date": "浦发银行", "query_all_trade_status": "1"},
            {"date": pd.Timestamp("2026-01-02"), "code": "000001.SZ", "name_on_date": "平安银行", "query_all_trade_status": "1"},
            {"date": pd.Timestamp("2026-01-02"), "code": "000004.SZ", "name_on_date": "*ST国华", "query_all_trade_status": "1"},
            {"date": pd.Timestamp("2026-01-02"), "code": "600001.SH", "name_on_date": "退市样例", "query_all_trade_status": "1"},
        ]
    )
    security_master = pd.DataFrame(
        [
            {"code": "600000.SH", "name": "浦发银行", "ipo_date": "1999-11-10", "out_date": "", "security_type": "1", "status": "1"},
            {"code": "000001.SZ", "name": "平安银行", "ipo_date": "1991-04-03", "out_date": "", "security_type": "1", "status": "1"},
            {"code": "000004.SZ", "name": "*ST国华", "ipo_date": "1990-12-01", "out_date": "", "security_type": "1", "status": "1"},
            {"code": "600001.SH", "name": "退市样例", "ipo_date": "1990-12-01", "out_date": "2025-12-31", "security_type": "1", "status": "0"},
        ]
    )
    daily_bars = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 100.0, "amount": 1000.0, "tradestatus": "1", "isST": "0"},
            {"date": pd.Timestamp("2026-01-02"), "code": "000001.SZ", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 0.0, "amount": 0.0, "tradestatus": "0", "isST": "0"},
            {"date": pd.Timestamp("2026-01-02"), "code": "000004.SZ", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 100.0, "amount": 1000.0, "tradestatus": "1", "isST": "1"},
            {"date": pd.Timestamp("2026-01-02"), "code": "600001.SH", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 100.0, "amount": 1000.0, "tradestatus": "1", "isST": "0"},
        ]
    )

    universe, status = build_daily_universe(stock_lists, security_master, daily_bars)

    by_code = universe.set_index("code")
    assert bool(by_code.loc["600000.SH", "is_tradeable"])
    assert by_code.loc["000001.SZ", "reject_reason"] == "suspended_on_date"
    assert by_code.loc["000004.SZ", "reject_reason"] == "st_on_date"
    assert by_code.loc["600001.SH", "reject_reason"] == "not_listed_on_date"
    assert status.set_index("code").loc["000001.SZ", "is_suspended_like"]


def test_build_daily_universe_handles_missing_bars_and_basic() -> None:
    stock_lists = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "name_on_date": "浦发银行", "query_all_trade_status": "1"},
            {"date": pd.Timestamp("2026-01-02"), "code": "600002.SH", "name_on_date": "缺资料", "query_all_trade_status": "1"},
        ]
    )
    security_master = pd.DataFrame(
        [
            {"code": "600000.SH", "name": "浦发银行", "ipo_date": "1999-11-10", "out_date": "", "security_type": "1", "status": "1"},
            {"code": "600002.SH", "name": "", "ipo_date": "", "out_date": "", "security_type": "", "status": "", "basic_error": "empty_stock_basic"},
        ]
    )
    daily_bars = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-02"), "code": "600002.SH", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 100.0, "amount": 1000.0},
        ]
    )

    universe, _ = build_daily_universe(stock_lists, security_master, daily_bars)

    by_code = universe.set_index("code")
    assert by_code.loc["600000.SH", "reject_reason"] == "missing_bar"
    assert by_code.loc["600002.SH", "reject_reason"] == "not_listed_on_date"


def test_assemble_reads_yearly_cache(tmp_path) -> None:
    root = tmp_path / "v2"
    cache = root / "cache"
    (cache / "daily_stock_lists").mkdir(parents=True)
    (cache / "daily_bars").mkdir(parents=True)
    pd.DataFrame([{"date": pd.Timestamp("2026-01-02"), "is_trading_day": True}]).to_parquet(cache / "trade_dates.parquet", index=False)
    pd.DataFrame(
        [{"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "name_on_date": "浦发银行", "query_all_trade_status": "1"}]
    ).to_parquet(cache / "daily_stock_lists" / "year=2026.parquet", index=False)
    pd.DataFrame(
        [
            {
                "code": "600000.SH",
                "baostock_code": "sh.600000",
                "name": "浦发银行",
                "ipo_date": "1999-11-10",
                "out_date": "",
                "security_type": "1",
                "status": "1",
                "first_seen_date": "2026-01-02",
                "last_seen_date": "2026-01-02",
                "basic_error": "",
            }
        ]
    ).to_parquet(cache / "security_master.parquet", index=False)
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "code": "600000.SH",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.0,
                "volume": 100.0,
                "amount": 1000.0,
                "tradestatus": "1",
                "isST": "0",
                "source": "fixture",
            }
        ]
    ).to_parquet(cache / "daily_bars" / "year=2026.parquet", index=False)

    manifest = assemble(PitBuildConfig(output_root=root, start_date="2026-01-02", end_date="2026-01-02", snapshot_id="fixture"))

    assert manifest["snapshot_id"] == "fixture"
    assert manifest["quality"]["tradeable_rows"] == 1
    assert (root / "fixture" / "daily_universe.parquet").exists()


class FakeStockListSource:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def query_all_stock(self, day: str) -> pd.DataFrame:
        self.calls.append(day)
        return pd.DataFrame([{"code": "sh.600000", "tradeStatus": "1", "code_name": "浦发银行"}])


def test_discover_daily_stock_lists_resumes_completed_dates(tmp_path) -> None:
    root = tmp_path / "v2"
    progress = root / "cache" / "progress"
    progress.mkdir(parents=True)
    (progress / "year=2026.json").write_text(
        '{"year": 2026, "completed_stock_dates": ["2026-01-02"], "completed_bar_codes": [], "failures": []}',
        encoding="utf-8",
    )
    part_dir = root / "cache" / "daily_stock_lists" / "parts" / "year=2026"
    part_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "name_on_date": "浦发银行", "query_all_trade_status": "1"}]
    ).to_parquet(part_dir / "month=01.parquet", index=False)
    source = FakeStockListSource()

    frame, summary = discover_daily_stock_lists(
        source,
        pd.Series(pd.to_datetime(["2026-01-02", "2026-01-05"])),
        config=PitBuildConfig(output_root=root, start_date="2026-01-02", end_date="2026-01-05"),
        year=2026,
        explicit_symbols=None,
    )

    assert source.calls == ["2026-01-05"]
    assert summary["stock_list_cache_miss"] == 1
    assert set(frame["date"].dt.strftime("%Y-%m-%d")) == {"2026-01-02", "2026-01-05"}


def test_failure_rate_threshold() -> None:
    _assert_failure_rate(2026, 20, [{"code": "600000.SH"}])
    try:
        _assert_failure_rate(2026, 20, [{"code": str(index)} for index in range(2)])
    except BaostockSourceError as exc:
        assert "failure rate too high" in str(exc)
    else:
        raise AssertionError("expected BaostockSourceError")
