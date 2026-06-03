from __future__ import annotations

import json

import pandas as pd

import traditional_quant_research.dataset_builder_v2 as builder_v2
from traditional_quant_research.dataset_v2 import DAILY_SIZE_COLUMNS
from traditional_quant_research.dataset_builder_v2 import (
    _baostock_to_std_code,
    _std_to_baostock_code,
    _assert_failure_rate,
    _config_cache_signature,
    _codes_cache_compatible,
    _daily_bars_meta_path,
    _daily_metrics_cache_path,
    _daily_metrics_meta_path,
    _daily_metrics_progress_path,
    _is_cache_compatible,
    _stock_industry_meta_path,
    _stock_list_meta_path,
    BaostockSourceError,
    PitBuildConfig,
    STOCK_INDUSTRY_COLUMNS,
    DAILY_METRICS_COLUMNS,
    assemble,
    build_daily_universe,
    derive_stock_lists_from_basic,
    discover_daily_stock_lists,
    fetch_stock_industry,
    fetch_size,
    expand_stock_industry_observations,
    industry_query_dates,
    load_cached_stock_industry,
    normalize_stock_basic,
    normalize_stock_industry,
    normalize_daily_metrics,
)


def test_baostock_code_conversion_round_trip() -> None:
    assert _baostock_to_std_code("sh.600000") == "600000.SH"
    assert _baostock_to_std_code("sz.000001") == "000001.SZ"
    assert _std_to_baostock_code("600000.SH") == "sh.600000"
    assert _std_to_baostock_code("000001.SZ") == "sz.000001"


def test_full_cache_can_serve_symbol_subset_but_sample_cannot_serve_full() -> None:
    full_config = PitBuildConfig(start_date="2026-01-02", end_date="2026-01-02")
    subset_config = PitBuildConfig(start_date="2026-01-02", end_date="2026-01-02", symbols=("600000.SH",))
    sample_config = PitBuildConfig(start_date="2026-01-02", end_date="2026-01-02", max_symbols=20)

    full_meta = {
        **_config_cache_signature(full_config, year=2026, start_date="2026-01-02", end_date="2026-01-02"),
        "complete": True,
    }
    sample_meta = {
        **_config_cache_signature(sample_config, year=2026, start_date="2026-01-02", end_date="2026-01-02"),
        "complete": True,
    }

    assert _is_cache_compatible(full_meta, subset_config, year=2026, start_date="2026-01-02", end_date="2026-01-02")
    assert not _is_cache_compatible(sample_meta, full_config, year=2026, start_date="2026-01-02", end_date="2026-01-02")


def test_yearly_code_cache_can_serve_code_subset_only() -> None:
    cached_codes = ["000001.SZ", "600000.SH"]

    assert _codes_cache_compatible(cached_codes, ["600000.SH"])
    assert _codes_cache_compatible(cached_codes, ["000001.SZ", "600000.SH"])
    assert not _codes_cache_compatible(["600000.SH"], ["000001.SZ", "600000.SH"])


def test_daily_metrics_sample_cache_uses_isolated_scope(tmp_path) -> None:
    full_config = PitBuildConfig(output_root=tmp_path, start_date="2026-05-25", end_date="2026-06-01")
    sample_config = PitBuildConfig(
        output_root=tmp_path,
        start_date="2026-05-25",
        end_date="2026-06-01",
        symbols=("600000.SH", "000001.SZ"),
    )

    full_path = _daily_metrics_cache_path(tmp_path, 2026, config=full_config, start_date="2026-05-25", end_date="2026-06-01")
    sample_path = _daily_metrics_cache_path(tmp_path, 2026, config=sample_config, start_date="2026-05-25", end_date="2026-06-01")
    sample_meta = _daily_metrics_meta_path(tmp_path, 2026, config=sample_config, start_date="2026-05-25", end_date="2026-06-01")
    sample_progress = _daily_metrics_progress_path(tmp_path, 2026, config=sample_config, start_date="2026-05-25", end_date="2026-06-01")

    assert full_path == tmp_path / "cache" / "daily_metrics" / "year=2026.parquet"
    assert "samples" in sample_path.parts
    assert "sample=" in str(sample_path)
    assert sample_path.name == "year=2026.parquet"
    assert "samples" in sample_meta.parts
    assert "samples" in sample_progress.parts
    assert sample_progress.name == "year=2026.json"


def test_stock_basic_derives_point_in_time_stock_lists() -> None:
    raw = pd.DataFrame(
        [
            {"code": "sh.600000", "code_name": "浦发银行", "ipoDate": "1999-11-10", "outDate": "", "type": "1", "status": "1"},
            {"code": "sz.300750", "code_name": "宁德时代", "ipoDate": "2018-06-11", "outDate": "", "type": "1", "status": "1"},
            {"code": "sh.600001", "code_name": "退市样例", "ipoDate": "1990-01-01", "outDate": "2015-12-31", "type": "1", "status": "0"},
            {"code": "sh.000001", "code_name": "上证指数", "ipoDate": "1991-07-15", "outDate": "", "type": "2", "status": "1"},
        ]
    )
    basic = normalize_stock_basic(raw)

    stock_lists = derive_stock_lists_from_basic(basic, pd.Series(pd.to_datetime(["2016-01-04"])))

    assert stock_lists["code"].tolist() == ["600000.SH"]


def test_normalize_stock_industry_standardizes_baostock_rows() -> None:
    raw = pd.DataFrame(
        [
            {
                "updateDate": "2026-06-01",
                "code": "sh.600000",
                "code_name": "浦发银行",
                "industry": "J66货币金融服务",
                "industryClassification": "证监会行业分类",
            }
        ]
    )

    industry = normalize_stock_industry(raw, "2026-06-01")

    assert industry.columns.tolist() == STOCK_INDUSTRY_COLUMNS
    assert industry.iloc[0]["date"] == pd.Timestamp("2026-06-01")
    assert industry.iloc[0]["code"] == "600000.SH"
    assert industry.iloc[0]["industry"] == "J66货币金融服务"
    assert industry.iloc[0]["industry_classification"] == "证监会行业分类"


def test_normalize_daily_metrics_standardizes_numeric_fields() -> None:
    raw = pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "code": "sh.600000",
                "turn": "0.85",
                "pctChg": "1.2",
                "peTTM": "6.5",
                "pbMRQ": "0.7",
                "psTTM": "2.1",
                "pcfNcfTTM": "4.2",
            }
        ]
    )

    metrics = normalize_daily_metrics(raw)

    assert metrics.columns.tolist() == DAILY_METRICS_COLUMNS
    assert metrics.iloc[0]["date"] == pd.Timestamp("2026-06-01")
    assert metrics.iloc[0]["code"] == "600000.SH"
    assert metrics.iloc[0]["turn"] == 0.85
    assert metrics.iloc[0]["source"] == "baostock"


def test_industry_query_dates_support_month_start_frequency() -> None:
    stock_lists = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-05"), "code": "600000.SH", "name_on_date": "浦发银行", "query_all_trade_status": "1"},
            {"date": pd.Timestamp("2026-01-06"), "code": "600000.SH", "name_on_date": "浦发银行", "query_all_trade_status": "1"},
            {"date": pd.Timestamp("2026-02-02"), "code": "600000.SH", "name_on_date": "浦发银行", "query_all_trade_status": "1"},
            {"date": pd.Timestamp("2026-02-03"), "code": "600000.SH", "name_on_date": "浦发银行", "query_all_trade_status": "1"},
        ]
    )

    daily = industry_query_dates(stock_lists, "daily")
    monthly = industry_query_dates(stock_lists, "month-start")

    assert [date.strftime("%Y-%m-%d") for date in daily] == [
        "2026-01-05",
        "2026-01-06",
        "2026-02-02",
        "2026-02-03",
    ]
    assert [date.strftime("%Y-%m-%d") for date in monthly] == ["2026-01-05", "2026-02-02"]


def test_expand_stock_industry_observations_forward_fills_month_start() -> None:
    stock_lists = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-05"), "code": "600000.SH", "name_on_date": "浦发银行", "query_all_trade_status": "1"},
            {"date": pd.Timestamp("2026-01-06"), "code": "600000.SH", "name_on_date": "浦发银行", "query_all_trade_status": "1"},
            {"date": pd.Timestamp("2026-01-06"), "code": "000001.SZ", "name_on_date": "平安银行", "query_all_trade_status": "1"},
        ]
    )
    observations = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-05"),
                "code": "600000.SH",
                "name_on_date": "浦发银行",
                "industry": "J66货币金融服务",
                "industry_classification": "证监会行业分类",
                "industry_update_date": "2026-01-05",
                "source": "baostock",
            }
        ]
    )

    expanded = expand_stock_industry_observations(observations, stock_lists, frequency="month-start")
    by_key = expanded.set_index(["date", "code"])

    assert by_key.loc[(pd.Timestamp("2026-01-06"), "600000.SH"), "industry"] == "J66货币金融服务"
    assert by_key.loc[(pd.Timestamp("2026-01-06"), "600000.SH"), "source"] == "baostock:month-start-ffill"
    assert by_key.loc[(pd.Timestamp("2026-01-06"), "000001.SZ"), "industry"] == ""


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
    config = PitBuildConfig(output_root=root, start_date="2026-01-02", end_date="2026-01-02", snapshot_id="fixture", include_metrics=True, include_size=True)
    pd.DataFrame([{"date": pd.Timestamp("2026-01-02"), "is_trading_day": True}]).to_parquet(cache / "trade_dates.parquet", index=False)
    pd.DataFrame(
        [{"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "name_on_date": "浦发银行", "query_all_trade_status": "1"}]
    ).to_parquet(cache / "daily_stock_lists" / "year=2026.parquet", index=False)
    _stock_list_meta_path(root, 2026).parent.mkdir(parents=True)
    _stock_list_meta_path(root, 2026).write_text(
        json.dumps(
            {
                **_config_cache_signature(config, year=2026, start_date="2026-01-02", end_date="2026-01-02"),
                "complete": True,
                "row_count": 1,
            }
        ),
        encoding="utf-8",
    )
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
    _daily_bars_meta_path(root, 2026).parent.mkdir(parents=True)
    _daily_bars_meta_path(root, 2026).write_text(
        json.dumps(
            {
                **_config_cache_signature(config, year=2026, start_date="2026-01-02", end_date="2026-01-02"),
                "complete": True,
                "codes": ["600000.SH"],
                "row_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (cache / "daily_metrics").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "code": "600000.SH",
                "turn": 0.85,
                "pctChg": 1.2,
                "peTTM": 6.5,
                "pbMRQ": 0.7,
                "psTTM": 2.1,
                "pcfNcfTTM": 4.2,
                "source": "fixture",
            }
        ]
    ).to_parquet(cache / "daily_metrics" / "year=2026.parquet", index=False)
    _daily_metrics_meta_path(root, 2026).parent.mkdir(parents=True)
    _daily_metrics_meta_path(root, 2026).write_text(
        json.dumps(
            {
                **_config_cache_signature(config, year=2026, start_date="2026-01-02", end_date="2026-01-02"),
                "complete": True,
                "codes": ["600000.SH"],
                "row_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (cache / "stock_industry").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "code": "600000.SH",
                "name_on_date": "浦发银行",
                "industry": "J66货币金融服务",
                "industry_classification": "证监会行业分类",
                "industry_update_date": "2026-01-02",
                "source": "fixture",
            }
        ]
    ).to_parquet(cache / "stock_industry" / "year=2026.parquet", index=False)
    _stock_industry_meta_path(root, 2026).parent.mkdir(parents=True)
    _stock_industry_meta_path(root, 2026).write_text(
        json.dumps(
            {
                **_config_cache_signature(config, year=2026, start_date="2026-01-02", end_date="2026-01-02"),
                "complete": True,
                "row_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (cache / "daily_size").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "code": "600000.SH",
                "total_market_cap": 1000.0,
                "float_market_cap": 800.0,
                "total_share": 100.0,
                "float_share": 80.0,
                "free_share": 70.0,
                "market_cap_unit": "10k CNY",
                "share_unit": "10k shares",
                "source": "fixture",
                "source_trade_date": "20260102",
            }
        ],
        columns=DAILY_SIZE_COLUMNS,
    ).to_parquet(cache / "daily_size" / "year=2026.parquet", index=False)

    manifest = assemble(config)

    assert manifest["snapshot_id"] == "fixture"
    assert manifest["quality"]["tradeable_rows"] == 1
    assert manifest["dataset"]["stock_industry_rows"] == 1
    assert manifest["dataset"]["daily_metrics_rows"] == 1
    assert manifest["dataset"]["daily_size_rows"] == 1
    assert manifest["dataset"]["daily_size_fields"] == DAILY_SIZE_COLUMNS
    assert (root / "fixture" / "daily_universe.parquet").exists()
    assert (root / "fixture" / "daily_metrics.parquet").exists()
    assert (root / "fixture" / "stock_industry.parquet").exists()
    assert (root / "fixture" / "daily_size.parquet").exists()


def test_assemble_ignores_incompatible_sample_cache(tmp_path) -> None:
    root = tmp_path / "v2"
    cache = root / "cache"
    (cache / "daily_stock_lists").mkdir(parents=True)
    (cache / "daily_bars").mkdir(parents=True)
    full_config = PitBuildConfig(output_root=root, start_date="2026-01-02", end_date="2026-01-02")
    sample_config = PitBuildConfig(output_root=root, start_date="2026-01-02", end_date="2026-01-02", max_symbols=20)
    pd.DataFrame([{"date": pd.Timestamp("2026-01-02"), "is_trading_day": True}]).to_parquet(cache / "trade_dates.parquet", index=False)
    pd.DataFrame(
        [{"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "name_on_date": "浦发银行", "query_all_trade_status": "1"}]
    ).to_parquet(cache / "daily_stock_lists" / "year=2026.parquet", index=False)
    _stock_list_meta_path(root, 2026).parent.mkdir(parents=True)
    _stock_list_meta_path(root, 2026).write_text(
        json.dumps(
            {
                **_config_cache_signature(sample_config, year=2026, start_date="2026-01-02", end_date="2026-01-02"),
                "complete": True,
                "row_count": 1,
            }
        ),
        encoding="utf-8",
    )

    try:
        assemble(full_config)
    except BaostockSourceError as exc:
        assert "stock list cache is empty" in str(exc)
    else:
        raise AssertionError("expected incompatible sample cache to be ignored")


class FakeStockListSource:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def query_all_stock(self, day: str) -> pd.DataFrame:
        self.calls.append(day)
        return pd.DataFrame([{"code": "sh.600000", "tradeStatus": "1", "code_name": "浦发银行"}])


class FakeIndustrySource:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def query_stock_industry(self, date: str = "", code: str = "") -> pd.DataFrame:
        self.calls.append(date)
        return pd.DataFrame(
            [
                {
                    "updateDate": date,
                    "code": "sh.600000",
                    "code_name": "浦发银行",
                    "industry": "J66货币金融服务",
                    "industryClassification": "证监会行业分类",
                },
                {
                    "updateDate": date,
                    "code": "sz.300750",
                    "code_name": "宁德时代",
                    "industry": "C39计算机、通信和其他电子设备制造业",
                    "industryClassification": "证监会行业分类",
                },
            ]
        )


def test_discover_daily_stock_lists_resumes_completed_dates(tmp_path) -> None:
    root = tmp_path / "v2"
    progress = root / "cache" / "progress"
    progress.mkdir(parents=True)
    config = PitBuildConfig(output_root=root, start_date="2026-01-02", end_date="2026-01-05", discovery_mode="daily")
    (progress / "year=2026.json").write_text(
        json.dumps(
            {
                "year": 2026,
                "completed_stock_dates": ["2026-01-02"],
                "completed_bar_codes": [],
                "failures": [],
                "signature": _config_cache_signature(config, year=2026, start_date="2026-01-02", end_date="2026-01-05"),
            },
            ensure_ascii=False,
        ),
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
        config=config,
        year=2026,
        explicit_symbols=None,
    )

    assert source.calls == ["2026-01-05"]
    assert summary["stock_list_cache_miss"] == 1
    assert set(frame["date"].dt.strftime("%Y-%m-%d")) == {"2026-01-02", "2026-01-05"}


def test_fetch_stock_industry_filters_and_caches_by_stock_lists(tmp_path) -> None:
    root = tmp_path / "v2"
    config = PitBuildConfig(output_root=root, start_date="2026-01-02", end_date="2026-01-05")
    stock_lists = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "code": "600000.SH",
                "name_on_date": "浦发银行",
                "query_all_trade_status": "1",
            },
            {
                "date": pd.Timestamp("2026-01-05"),
                "code": "600000.SH",
                "name_on_date": "浦发银行",
                "query_all_trade_status": "1",
            },
        ]
    )
    source = FakeIndustrySource()

    industry, failures, summary = fetch_stock_industry(
        source,
        stock_lists,
        config=config,
        year=2026,
        start_date="2026-01-02",
        end_date="2026-01-05",
    )

    assert source.calls == ["2026-01-02", "2026-01-05"]
    assert failures == []
    assert summary["stock_industry_cache_miss"] == 1
    assert set(industry["code"]) == {"600000.SH"}
    assert set(industry["date"].dt.strftime("%Y-%m-%d")) == {"2026-01-02", "2026-01-05"}
    assert _stock_industry_meta_path(root, 2026).exists()

    cached = load_cached_stock_industry(root, [2026], "2026-01-02", "2026-01-05", config=config)
    assert cached["industry"].tolist() == ["J66货币金融服务", "J66货币金融服务"]


def test_fetch_stock_industry_month_start_expands_to_daily_stock_lists(tmp_path) -> None:
    root = tmp_path / "v2"
    config = PitBuildConfig(
        output_root=root,
        start_date="2026-01-05",
        end_date="2026-02-03",
        industry_frequency="month-start",
    )
    stock_lists = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-05"), "code": "600000.SH", "name_on_date": "浦发银行", "query_all_trade_status": "1"},
            {"date": pd.Timestamp("2026-01-06"), "code": "600000.SH", "name_on_date": "浦发银行", "query_all_trade_status": "1"},
            {"date": pd.Timestamp("2026-02-02"), "code": "600000.SH", "name_on_date": "浦发银行", "query_all_trade_status": "1"},
            {"date": pd.Timestamp("2026-02-03"), "code": "600000.SH", "name_on_date": "浦发银行", "query_all_trade_status": "1"},
        ]
    )
    source = FakeIndustrySource()

    industry, failures, summary = fetch_stock_industry(
        source,
        stock_lists,
        config=config,
        year=2026,
        start_date="2026-01-05",
        end_date="2026-02-03",
    )

    assert source.calls == ["2026-01-05", "2026-02-02"]
    assert failures == []
    assert summary["stock_industry_cache_miss"] == 1
    assert industry["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-01-05",
        "2026-01-06",
        "2026-02-02",
        "2026-02-03",
    ]
    assert industry["industry"].notna().all()
    assert industry["source"].str.endswith("month-start-ffill").all()
    meta = json.loads(_stock_industry_meta_path(root, 2026).read_text(encoding="utf-8"))
    assert meta["industry_frequency"] == "month-start"
    assert meta["query_date_count"] == 2
    assert meta["date_count"] == 4


def test_failure_rate_threshold() -> None:
    _assert_failure_rate(2026, 20, [{"code": "600000.SH"}])
    try:
        _assert_failure_rate(2026, 20, [{"code": str(index)} for index in range(2)])
    except BaostockSourceError as exc:
        assert "failure rate too high" in str(exc)
    else:
        raise AssertionError("expected BaostockSourceError")


def test_fetch_size_uses_trade_dates_and_cached_stock_list_symbols(tmp_path, monkeypatch) -> None:
    root = tmp_path / "v2"
    cache = root / "cache"
    (cache / "daily_stock_lists").mkdir(parents=True)
    pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-06-01"), "code": "600000.SH", "name_on_date": "浦发银行", "query_all_trade_status": "1"},
            {"date": pd.Timestamp("2026-06-01"), "code": "000001.SZ", "name_on_date": "平安银行", "query_all_trade_status": "1"},
        ]
    ).to_parquet(cache / "daily_stock_lists" / "year=2026.parquet", index=False)
    calls: list[dict[str, object]] = []

    def fake_fetch_tushare_daily_size_cache(**kwargs):
        calls.append(kwargs)
        return {
            "command": "fetch-size",
            "source": "tushare.daily_basic",
            "status": "passed",
            "year": kwargs["year"],
            "rows": 1,
            "date_count": len(kwargs["trade_dates"]),
            "code_count": len(kwargs["symbols"]),
            "failure_count": 0,
            "failures": [],
            "cache_hit": 0,
            "cache_miss": 1,
        }

    monkeypatch.setattr(builder_v2, "fetch_tushare_daily_size_cache", fake_fetch_tushare_daily_size_cache)
    config = PitBuildConfig(
        output_root=root,
        command="fetch-size",
        year=2026,
        start_date="2026-06-01",
        end_date="2026-06-02",
        symbols=("600000.SH",),
        trade_dates=("20260601", "20260602"),
        size_token="token",
    )

    result = fetch_size(config)

    assert result["status"] == "passed"
    assert calls[0]["trade_dates"] == ["20260601", "20260602"]
    assert calls[0]["symbols"] == ["600000.SH"]
    assert calls[0]["token"] == "token"


def test_fetch_size_proxy_amount_writes_diagnostic_cache(tmp_path) -> None:
    root = tmp_path / "v2"
    cache = root / "cache"
    (cache / "daily_bars").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-06-01"),
                "code": "600000.SH",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "volume": 100.0,
                "amount": 1000.0,
                "tradestatus": "1",
                "isST": "0",
                "source": "fixture",
            }
        ]
    ).to_parquet(cache / "daily_bars" / "year=2026.parquet", index=False)
    config = PitBuildConfig(
        output_root=root,
        command="fetch-size",
        year=2026,
        start_date="2026-06-01",
        end_date="2026-06-01",
        size_source=builder_v2.SIZE_SOURCE_PROXY_AMOUNT,
    )

    result = fetch_size(config)

    assert result["status"] == "passed"
    assert result["source"] == "proxy.amount"
    cached = builder_v2.load_cached_daily_size(root, [2026], "2026-06-01", "2026-06-01")
    assert cached["total_market_cap"].tolist() == [1000.0]
    meta = json.loads((cache / "cache_meta" / "daily_size" / "year=2026.json").read_text(encoding="utf-8"))
    assert meta["source_grade"] == "proxy_only"


def test_fetch_size_akshare_cninfo_reconstructs_from_share_events(tmp_path, monkeypatch) -> None:
    root = tmp_path / "v2"
    cache = root / "cache"
    (cache / "daily_bars").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-06-01"),
                "code": "600000.SH",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "volume": 100.0,
                "amount": 1000.0,
                "tradestatus": "1",
                "isST": "0",
                "source": "fixture",
            }
        ]
    ).to_parquet(cache / "daily_bars" / "year=2026.parquet", index=False)

    class FakeAkshare:
        @staticmethod
        def stock_share_change_cninfo(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
            assert symbol == "600000"
            return pd.DataFrame([{"变动日期": "2026-01-01", "总股本": "1万股", "流通股": "8000股"}])

    original_import_module = builder_v2.importlib.import_module
    monkeypatch.setattr(
        builder_v2.importlib,
        "import_module",
        lambda name: FakeAkshare if name == "akshare" else original_import_module(name),
    )
    config = PitBuildConfig(
        output_root=root,
        command="fetch-size",
        year=2026,
        start_date="2026-06-01",
        end_date="2026-06-01",
        size_source=builder_v2.SIZE_SOURCE_AKSHARE_CNINFO_RECONSTRUCTED,
    )

    result = fetch_size(config)

    assert result["status"] == "passed"
    assert result["source"] == "akshare.cninfo_reconstructed"
    cached = builder_v2.load_cached_daily_size(root, [2026], "2026-06-01", "2026-06-01", symbols=["600000.SH"])
    assert cached["total_market_cap"].tolist() == [100000.0]
    assert cached["float_market_cap"].tolist() == [80000.0]
